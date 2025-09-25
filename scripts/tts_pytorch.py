# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi==0.2.11",
#     "torch",
#     "sphn",
#     "sounddevice",
# ]
# ///
import argparse
import sys

import numpy as np
import queue
import sphn
import time
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel


def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS using the PyTorch implementation"
    )
    parser.add_argument("inp", type=str, help="Input file, use - for stdin.")
    parser.add_argument(
        "out", type=str, help="Output file to generate, use - for playing the audio"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo in which to look for the pretrained models.",
    )
    parser.add_argument(
        "--voice-repo",
        default=DEFAULT_DSM_TTS_VOICE_REPO,
        help="HF repo in which to look for pre-computed voice embeddings.",
    )
    parser.add_argument(
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="The voice to use, relative to the voice repo root. "
        f"See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    args = parser.parse_args()

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=args.device
    )
    sample_rate = tts_model.mimi.sample_rate
    frame_size = sample_rate // 1000 * 60

    if args.inp == "-":
        if sys.stdin.isatty():  # Interactive
            print("Enter text to synthesize (Ctrl+D to end input):")
        text = sys.stdin.read().strip()
    else:
        with open(args.inp, "r", encoding="utf-8") as fobj:
            text = fobj.read().strip()

    # If you want to make a dialog, you can pass more than one turn [text_speaker_1, text_speaker_2, text_2_speaker_1, ...]
    entries = tts_model.prepare_script([text], padding_between=1)
    if args.voice.endswith(".safetensors"):
        voice_path = args.voice
    else:
        voice_path = tts_model.get_voice_path(args.voice)
    # CFG coef goes here because the model was trained with CFG distillation,
    # so it's not _actually_ doing CFG at inference time.
    # Also, if you are generating a dialog, you should have two voices in the list.
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )
    emitted_samples = 0

    if args.out == "-":
        # Stream the audio to the speakers using sounddevice.
        import sounddevice as sd

        pcms = queue.Queue()
        pending_pcm = np.array([], dtype=np.float32)

        def _emit_frames():
            nonlocal pending_pcm, emitted_samples
            while pending_pcm.size >= frame_size:
                pcms.put_nowait(pending_pcm[:frame_size].copy())
                pending_pcm = pending_pcm[frame_size:]
                emitted_samples += frame_size
                print(
                    f"generated {emitted_samples / sample_rate:.2f}s",
                    end="\r",
                    flush=True,
                )

        def _on_frame(frame):
            nonlocal pending_pcm
            if (frame != -1).all():
                pcm = (
                    tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy().astype(np.float32)
                )
                clipped = np.clip(pcm[0, 0], -1.0, 1.0)
                pending_pcm = np.concatenate([pending_pcm, clipped])
                _emit_frames()

        def audio_callback(outdata, _a, _b, _c):
            try:
                pcm_data = pcms.get(block=False)
                frames = min(pcm_data.size, outdata.shape[0])
                outdata[:frames, 0] = pcm_data[:frames]
                if frames < outdata.shape[0]:
                    outdata[frames:, 0] = 0
                if pcm_data.size > outdata.shape[0]:
                    remainder = pcm_data[outdata.shape[0] :]
                    if remainder.size:
                        pcms.put_nowait(remainder)
            except queue.Empty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=sample_rate,
            blocksize=frame_size,
            channels=1,
            callback=audio_callback,
        ):
            with tts_model.mimi.streaming(1):
                tts_model.generate(
                    [entries], [condition_attributes], on_frame=_on_frame
                )
            if pending_pcm.size:
                pcms.put_nowait(pending_pcm.copy())
                emitted_samples += pending_pcm.size
                print(
                    f"generated {emitted_samples / sample_rate:.2f}s",
                    end="\r",
                    flush=True,
                )
            time.sleep(3)
            while True:
                if pcms.qsize() == 0:
                    break
                time.sleep(1)
    else:

        def _on_frame(frame):
            nonlocal emitted_samples
            if (frame != -1).all():
                emitted_samples += frame_size
                print(
                    f"generated {emitted_samples / sample_rate:.2f}s",
                    end="\r",
                    flush=True,
                )

        start_time = time.time()
        result = tts_model.generate(
            [entries], [condition_attributes], on_frame=_on_frame
        )
        print(f"\nTotal time: {time.time() - start_time:.2f}s")
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[tts_model.delay_steps :]:
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                clipped = np.clip(pcm[0, 0], -1, 1).astype(np.float32)
                pcms.append(clipped)
            pcm = np.concatenate(pcms, axis=-1)
        if args.out.lower().endswith(".raw"):
            pcm_i16 = np.clip(pcm, -1.0, 1.0)
            pcm_i16 = (pcm_i16 * np.float32(32767.0)).astype("<i2")
            with open(args.out, "wb") as fobj:
                pcm_i16.tofile(fobj)
            print(f"Saved raw 16-bit PCM audio to {args.out} at {sample_rate}Hz")
        else:
            sphn.write_wav(args.out, pcm, sample_rate)


if __name__ == "__main__":
    main()

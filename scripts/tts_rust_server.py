# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sphn",
#     "websockets",
#     "sounddevice",
#     "tqdm",
# ]
# ///
import argparse
import asyncio
import sys
from urllib.parse import urlencode

import msgpack
import numpy as np
import sounddevice as sd
import sphn
import tqdm
import websockets

# The server currently streams audio at 24kHz. Downsample on the client so that the
# saved or played audio uses the expected 8kHz cadence.
SERVER_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 8000
TARGET_FRAME_SIZE = TARGET_SAMPLE_RATE // 1000 * 80
_DOWNSAMPLE_FACTOR = SERVER_SAMPLE_RATE // TARGET_SAMPLE_RATE
_DOWNSAMPLE_KERNEL = np.full((_DOWNSAMPLE_FACTOR,), 1.0 / _DOWNSAMPLE_FACTOR, dtype=np.float32)


def _downsample_to_target_rate(
    pcm: np.ndarray,
    residual: np.ndarray,
    *,
    flush: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a PCM chunk to the target rate, keeping leftover samples.

    The server emits 24kHz audio. We apply a simple moving-average low-pass filter
    followed by decimation by three so that the resulting stream matches 8kHz.
    Any samples that do not fit an even multiple of the downsampling factor are
    returned as residual and will be prefixed to the next chunk.
    """

    if TARGET_SAMPLE_RATE == SERVER_SAMPLE_RATE:
        return pcm.astype(np.float32), np.array([], dtype=np.float32)

    if _DOWNSAMPLE_FACTOR * TARGET_SAMPLE_RATE != SERVER_SAMPLE_RATE:
        msg = (
            "Expected integer downsample factor between server and target sample rates. "
            f"Got server={SERVER_SAMPLE_RATE}, target={TARGET_SAMPLE_RATE}."
        )
        raise ValueError(msg)

    if residual.size:
        pcm = np.concatenate([residual, pcm])

    if flush and pcm.size:
        pad = (-pcm.size) % _DOWNSAMPLE_FACTOR
        if pad:
            pcm = np.pad(pcm, (0, pad))
    elif pcm.size < _DOWNSAMPLE_FACTOR:
        return np.array([], dtype=np.float32), pcm

    trim = pcm.size - (pcm.size % _DOWNSAMPLE_FACTOR)
    main_chunk = pcm[:trim]
    residual = pcm[trim:]

    # Moving-average low-pass filter before decimation to reduce aliasing.
    filtered = np.convolve(main_chunk, _DOWNSAMPLE_KERNEL, mode="same")
    downsampled = filtered.reshape(-1, _DOWNSAMPLE_FACTOR).mean(axis=1)
    return downsampled.astype(np.float32), residual.astype(np.float32)

TTS_TEXT = "Hello, this is a test of the moshi text to speech system, this should result in some nicely sounding generated voice."
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"
AUTH_TOKEN = "public_token"


async def receive_messages(
    websocket: websockets.ClientConnection,
    output_queue: asyncio.Queue,
):
    with tqdm.tqdm(desc="Receiving audio", unit=" seconds generated") as pbar:
        accumulated_samples = 0
        last_seconds = 0
        residual = np.array([], dtype=np.float32)
        pending_output = np.array([], dtype=np.float32)

        async for message_bytes in websocket:
            msg = msgpack.unpackb(message_bytes)

            if msg["type"] == "Audio":
                pcm = np.array(msg["pcm"], dtype=np.float32)
                pcm, residual = _downsample_to_target_rate(pcm, residual)
                if pcm.size:
                    pending_output = np.concatenate([pending_output, pcm])
                    while pending_output.size >= TARGET_FRAME_SIZE:
                        await output_queue.put(pending_output[:TARGET_FRAME_SIZE])
                        pending_output = pending_output[TARGET_FRAME_SIZE:]

                accumulated_samples += len(msg["pcm"])
                current_seconds = accumulated_samples // SERVER_SAMPLE_RATE
                if current_seconds > last_seconds:
                    pbar.update(current_seconds - last_seconds)
                    last_seconds = current_seconds

        if residual.size:
            pcm, residual = _downsample_to_target_rate(
                np.array([], dtype=np.float32),
                residual,
                flush=True,
            )
            if pcm.size:
                pending_output = np.concatenate([pending_output, pcm])

        if pending_output.size:
            await output_queue.put(pending_output)
    print("End of audio.")
    await output_queue.put(None)  # Signal end of audio


async def output_audio(out: str, output_queue: asyncio.Queue):
    if out == "-":
        should_exit = False

        def audio_callback(outdata, _a, _b, _c):
            nonlocal should_exit

            try:
                pcm_data = output_queue.get_nowait()
                if pcm_data is not None:
                    frames = min(pcm_data.size, outdata.shape[0])
                    outdata[:frames, 0] = pcm_data[:frames]
                    if frames < outdata.shape[0]:
                        outdata[frames:, 0] = 0
                    if pcm_data.size > outdata.shape[0]:
                        remainder = pcm_data[outdata.shape[0] :]
                        if remainder.size:
                            output_queue.put_nowait(remainder)
                else:
                    should_exit = True
                    outdata[:] = 0
            except asyncio.QueueEmpty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=TARGET_SAMPLE_RATE,
            blocksize=TARGET_FRAME_SIZE,
            channels=1,
            callback=audio_callback,
        ):
            while True:
                if should_exit:
                    break
                await asyncio.sleep(1)
    else:
        frames = []
        while True:
            item = await output_queue.get()
            if item is None:
                break
            frames.append(item)

        if frames:
            pcm = np.concatenate(frames, axis=0)
        else:
            pcm = np.array([], dtype=np.float32)
        sphn.write_wav(out, pcm, TARGET_SAMPLE_RATE)
        print(f"Saved audio to {out}")


async def read_lines_from_stdin():
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    while True:
        line = await reader.readline()
        if not line:
            break
        yield line.decode().rstrip()


async def read_lines_from_file(path: str):
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                asyncio.run_coroutine_threadsafe(queue.put(line), loop)
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    await asyncio.to_thread(producer)
    while True:
        line = await queue.get()
        if line is None:
            break
        yield line


async def get_lines(source: str):
    if source == "-":
        async for line in read_lines_from_stdin():
            yield line
    else:
        async for line in read_lines_from_file(source):
            yield line


async def websocket_client():
    parser = argparse.ArgumentParser(description="Use the TTS streaming API")
    parser.add_argument("inp", type=str, help="Input file, use - for stdin.")
    parser.add_argument(
        "out", type=str, help="Output file to generate, use - for playing the audio"
    )
    parser.add_argument(
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="The voice to use, relative to the voice repo root. "
        f"See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://127.0.0.1:8082",
    )
    parser.add_argument("--api-key", default="public_token")
    args = parser.parse_args()

    params = {"voice": args.voice, "format": "PcmMessagePack"}
    uri = f"{args.url}/api/tts_streaming?{urlencode(params)}"
    print(uri)

    if args.inp == "-":
        if sys.stdin.isatty():  # Interactive
            print("Enter text to synthesize (Ctrl+D to end input):")
    headers = {"kyutai-api-key": args.api_key}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        print("connected")

        async def send_loop():
            print("go send")
            async for line in get_lines(args.inp):
                for word in line.split():
                    await websocket.send(msgpack.packb({"type": "Text", "text": word}))
            await websocket.send(msgpack.packb({"type": "Eos"}))

        output_queue = asyncio.Queue()
        receive_task = asyncio.create_task(receive_messages(websocket, output_queue))
        output_audio_task = asyncio.create_task(output_audio(args.out, output_queue))
        send_task = asyncio.create_task(send_loop())
        await asyncio.gather(receive_task, output_audio_task, send_task)


if __name__ == "__main__":
    asyncio.run(websocket_client())

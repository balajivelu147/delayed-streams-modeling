# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
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
import wave
import tqdm
import websockets

# The server currently streams audio at 24kHz. Downsample on the client so that the
# saved or played audio uses the expected 8kHz cadence.
DEFAULT_SERVER_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 8000
TARGET_FRAME_SIZE = int(round(TARGET_SAMPLE_RATE * 0.08))


class StreamingDownsampler:
    """Incrementally convert server PCM into the target sample rate."""

    def __init__(self, target_rate: int, *, assume_source: int | None = None) -> None:
        if target_rate <= 0:
            raise ValueError("Target rate must be positive")
        self._target_rate = target_rate
        self._source_rate = None
        self._factor = 1
        self._residual = np.array([], dtype=np.float32)
        if assume_source is not None:
            self.configure_source_rate(assume_source)

    @property
    def source_rate(self) -> int | None:
        return self._source_rate

    def configure_source_rate(self, sample_rate: int) -> None:
        """(Re)configure the upstream sample rate and reset residuals."""

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self._source_rate == sample_rate:
            return
        self._source_rate = sample_rate
        if sample_rate == self._target_rate:
            self._factor = 1
        else:
            if sample_rate % self._target_rate != 0:
                msg = (
                    "Expected integer downsample factor between server and target sample rates. "
                    f"Got server={sample_rate}, target={self._target_rate}."
                )
                raise ValueError(msg)
            self._factor = sample_rate // self._target_rate
        self._residual = np.array([], dtype=np.float32)

    def process(self, pcm: np.ndarray, *, flush: bool = False) -> np.ndarray:
        """Downsample the provided chunk and retain any leftovers for later."""

        if self._source_rate is None:
            raise RuntimeError("Source rate has not been configured yet")

        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0 and self._residual.size == 0:
            return np.array([], dtype=np.float32)

        # Normalise the server PCM if it arrives as 16-bit integers.
        if pcm.size and np.max(np.abs(pcm)) > 1.0:
            pcm = pcm / np.float32(np.iinfo(np.int16).max)

        if self._residual.size:
            pcm = np.concatenate([self._residual, pcm])

        if self._factor == 1:
            self._residual = np.array([], dtype=np.float32)
            return pcm

        if flush and pcm.size:
            pad = (-pcm.size) % self._factor
            if pad:
                pcm = np.pad(pcm, (0, pad))

        trim = pcm.size - (pcm.size % self._factor)
        if trim == 0:
            if flush and pcm.size:
                trim = pcm.size
            else:
                self._residual = pcm
                return np.array([], dtype=np.float32)

        main_chunk = pcm[:trim]
        self._residual = pcm[trim:] if not flush else np.array([], dtype=np.float32)

        # Simple moving-average decimation to reduce aliasing.
        reshaped = main_chunk.reshape(-1, self._factor)
        downsampled = reshaped.mean(axis=1)
        return downsampled.astype(np.float32)
TTS_TEXT = "Hello, this is a test of the moshi text to speech system, this should result in some nicely sounding generated voice."
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"
AUTH_TOKEN = "public_token"


async def receive_messages(
    websocket: websockets.ClientConnection,
    output_queue: asyncio.Queue,
):
    with tqdm.tqdm(desc="Receiving audio", unit=" seconds generated") as pbar:
        downsampler = StreamingDownsampler(
            TARGET_SAMPLE_RATE, assume_source=DEFAULT_SERVER_SAMPLE_RATE
        )
        accumulated_samples = 0
        last_seconds = 0
        pending_output = np.array([], dtype=np.float32)

        try:
            async for message_bytes in websocket:
                msg = msgpack.unpackb(message_bytes)

                if msg["type"] != "Audio":
                    continue

                sample_rate = int(msg.get("sample_rate", downsampler.source_rate or DEFAULT_SERVER_SAMPLE_RATE))
                downsampler.configure_source_rate(sample_rate)

                raw_pcm = msg["pcm"]
                if isinstance(raw_pcm, (bytes, bytearray, memoryview)):
                    # The streaming API serialises PCM as little-endian
                    # float32 values. Accept a raw buffer here so the client
                    # works regardless of how msgpack deserialised the field.
                    buffer = memoryview(raw_pcm)
                    pcm = np.frombuffer(buffer, dtype="<f4", count=buffer.nbytes // 4)
                else:
                    pcm = np.asarray(raw_pcm, dtype=np.float32)

                pcm = downsampler.process(pcm)
                if pcm.size:
                    if pending_output.size:
                        pending_output = np.concatenate([pending_output, pcm])
                    else:
                        pending_output = pcm
                    while pending_output.size >= TARGET_FRAME_SIZE:
                        await output_queue.put(
                            np.ascontiguousarray(pending_output[:TARGET_FRAME_SIZE])
                        )
                        pending_output = pending_output[TARGET_FRAME_SIZE:]

                accumulated_samples += len(msg["pcm"])
                source_rate = downsampler.source_rate or sample_rate
                current_seconds = accumulated_samples // source_rate
                if current_seconds > last_seconds:
                    pbar.update(current_seconds - last_seconds)
                    last_seconds = current_seconds
        finally:
            flushed = downsampler.process(np.array([], dtype=np.float32), flush=True)
            if flushed.size:
                if pending_output.size:
                    pending_output = np.concatenate([pending_output, flushed])
                else:
                    pending_output = flushed

            if pending_output.size:
                await output_queue.put(np.ascontiguousarray(pending_output))

            print("End of audio.")
            await output_queue.put(None)  # Signal end of audio


async def output_audio(out: str, output_queue: asyncio.Queue):
    if out == "-":
        should_exit = False
        stop_requested = False
        pending_local = np.array([], dtype=np.float32)

        def audio_callback(outdata, _frames, _time, _status):
            nonlocal should_exit, stop_requested, pending_local

            if pending_local.size < outdata.shape[0] and not stop_requested:
                try:
                    next_chunk = output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    next_chunk = None
                else:
                    if next_chunk is None:
                        stop_requested = True
                    elif next_chunk.size:
                        if pending_local.size:
                            pending_local = np.concatenate([pending_local, next_chunk])
                        else:
                            pending_local = next_chunk

            frames = min(pending_local.size, outdata.shape[0])
            if frames:
                outdata[:frames, 0] = pending_local[:frames]
            if frames < outdata.shape[0]:
                outdata[frames:, 0] = 0
            if frames:
                pending_local = pending_local[frames:]

            if stop_requested and pending_local.size == 0:
                should_exit = True

        with sd.OutputStream(
            samplerate=TARGET_SAMPLE_RATE,
            blocksize=TARGET_FRAME_SIZE,
            channels=1,
            callback=audio_callback,
        ):
            print(f"Output stream initialised at {TARGET_SAMPLE_RATE} Hz")
            while not should_exit:
                await asyncio.sleep(0.05)
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

        # Persist the file with an explicit 8 kHz WAV header so that downstream
        # tools correctly read the sample rate regardless of third party
        # library quirks.
        with wave.open(out, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            if pcm.size:
                clipped = np.clip(pcm, -1.0, 1.0)
                int16_pcm = np.round(
                    clipped * float(np.iinfo(np.int16).max)
                ).astype("<i2")
                wav_file.writeframes(int16_pcm.tobytes())
            else:
                wav_file.writeframes(b"")
        with wave.open(out, "rb") as wav_file:
            reported_rate = wav_file.getframerate()
        print(f"Saved audio to {out} (sample rate: {reported_rate} Hz)")


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
            try:
                async for line in get_lines(args.inp):
                    for word in line.split():
                        await websocket.send(
                            msgpack.packb({"type": "Text", "text": word})
                        )
                await websocket.send(msgpack.packb({"type": "Eos"}))
            except websockets.ConnectionClosed:
                # The server closed the connection before we finished streaming.
                # This typically happens after it finishes synthesis. Swallow the
                # error so the receive loop can drain the remaining audio.
                pass

        output_queue = asyncio.Queue()
        receive_task = asyncio.create_task(receive_messages(websocket, output_queue))
        output_audio_task = asyncio.create_task(output_audio(args.out, output_queue))
        send_task = asyncio.create_task(send_loop())
        await asyncio.gather(receive_task, output_audio_task, send_task)


if __name__ == "__main__":
    asyncio.run(websocket_client())

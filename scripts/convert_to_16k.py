# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "julius",
#     "numpy",
#     "sphn",
#     "torch",
# ]
# ///
"""Utility script to resample audio to 16 kHz without altering playback speed."""

import argparse
from pathlib import Path

import julius
import numpy as np
import sphn
import torch

TARGET_SAMPLE_RATE = 16_000


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return audio data as float32 PCM and the original sample rate."""
    audio, sample_rate = sphn.read(str(path))

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, int(sample_rate)


def resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample the provided audio to 16 kHz if needed."""
    if src_rate == TARGET_SAMPLE_RATE:
        return audio

    audio_tensor = torch.from_numpy(audio)
    resampled = julius.resample_frac(audio_tensor, src_rate, TARGET_SAMPLE_RATE)
    return resampled.cpu().numpy().astype(np.float32, copy=False)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.shape[0] == 1:
        return audio[0]
    return audio.mean(axis=0)


def load_resampled_audio(path: Path | str, *, keep_channels: bool = False) -> np.ndarray:
    """Load ``path`` and return float32 PCM audio resampled to 16 kHz."""

    audio, source_rate = load_audio(Path(path))

    if keep_channels:
        if audio.ndim == 1:
            audio = audio[None, :]
        resampled = resample(audio, source_rate)
        if resampled.ndim == 1:
            resampled = resampled[None, :]
        return resampled.astype(np.float32, copy=False)

    mono = ensure_mono(audio)
    resampled = resample(mono, source_rate)
    return resampled.astype(np.float32, copy=False)


def save_audio(path: Path, audio: np.ndarray) -> None:
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sphn.write_wav(str(path), audio, TARGET_SAMPLE_RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source audio file")
    parser.add_argument("output", type=Path, help="Destination WAV file")
    parser.add_argument(
        "--keep-channels",
        action="store_true",
        help="Keep the original channel layout instead of mixing down to mono.",
    )

    args = parser.parse_args()

    resampled = load_resampled_audio(args.input, keep_channels=args.keep_channels)

    save_audio(args.output, resampled)
    print(
        f"Saved {args.output} at {TARGET_SAMPLE_RATE} Hz "
        f"({resampled.shape[-1] / TARGET_SAMPLE_RATE:.2f}s)"
    )

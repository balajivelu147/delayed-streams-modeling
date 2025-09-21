# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "julius",
#     "numpy",
#     "sphn",
# ]
# ///

"""Utility to resample audio to 8kHz float PCM using Julius."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sphn

from scripts import load_resampled_audio

TARGET_SAMPLE_RATE = 8_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio file to resample.")
    parser.add_argument("output", type=Path, help="Path to write the resampled WAV file.")
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Downmix to mono before resampling. The default keeps the original channel count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio = load_resampled_audio(
        str(args.input), target_sample_rate=TARGET_SAMPLE_RATE, mono=args.mono
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(args.output, np.asarray(audio, dtype=np.float32), TARGET_SAMPLE_RATE)


if __name__ == "__main__":
    main()

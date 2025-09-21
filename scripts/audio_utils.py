"""Shared audio utilities for scripts."""

from __future__ import annotations

import julius
import numpy as np
import sphn
import torch


DEFAULT_TARGET_SAMPLE_RATE = 8_000


def load_resampled_audio(
    file_path: str,
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    mono: bool = False,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Load audio and resample it to the target sample rate using Julius.

    The returned array is channels-first with dtype ``float32`` (unless another dtype
    is requested). Values are clipped to ``[-1, 1]`` to avoid downstream clipping
    issues, and the duration is preserved by relying on Julius' fractional
    resampling implementation.

    Args:
        file_path: Path to the audio file to load.
        target_sample_rate: The desired sample rate for the returned audio.
        mono: Whether to downmix multi-channel audio to mono by averaging channels.
        dtype: The numpy dtype of the returned array.

    Returns:
        A numpy array of shape ``(channels, samples)`` containing float PCM data.
    """

    audio, input_sample_rate = sphn.read(file_path)
    audio = np.asarray(audio)

    if audio.ndim == 1:
        audio = audio[None, :]

    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    audio_tensor = torch.from_numpy(audio.astype(np.float32, copy=False))

    if int(input_sample_rate) != int(target_sample_rate):
        audio_tensor = julius.resample_frac(
            audio_tensor, int(input_sample_rate), int(target_sample_rate)
        )

    audio = audio_tensor.numpy()
    audio = np.clip(audio, -1.0, 1.0)

    if dtype is not None and audio.dtype != dtype:
        audio = audio.astype(dtype, copy=False)

    return audio

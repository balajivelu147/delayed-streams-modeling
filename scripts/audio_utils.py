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
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = audio[None, :]

    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True, dtype=np.float32)

    audio = np.ascontiguousarray(audio)
    input_sample_rate = int(input_sample_rate)
    target_sample_rate = int(target_sample_rate)

    audio_tensor = torch.from_numpy(audio)
    expected_length: int | None = None

    if input_sample_rate != target_sample_rate:
        original_length = audio_tensor.shape[-1]
        expected_length = int(round(original_length * target_sample_rate / input_sample_rate))

        resampler = julius.ResampleFrac(input_sample_rate, target_sample_rate)
        resampler = resampler.to(audio_tensor)
        audio_tensor = resampler(audio_tensor, output_length=expected_length)

    audio = audio_tensor.detach().cpu().numpy()
    audio = np.clip(audio, -1.0, 1.0)

    if expected_length is not None:
        current_length = audio.shape[-1]
        if current_length < expected_length:
            pad_width = [(0, 0)] * audio.ndim
            pad_width[-1] = (0, expected_length - current_length)
            audio = np.pad(audio, pad_width, mode="constant")
        elif current_length > expected_length:
            audio = audio[..., :expected_length]

    if dtype is not None and audio.dtype != dtype:
        audio = audio.astype(dtype, copy=False)

    return audio

"""PCM WAV writing, validation, duration, and simple audio coercion."""

from __future__ import annotations

import hashlib
import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WavInfo:
    """Facts verified from a PCM WAV file."""

    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_seconds: float
    file_bytes: int


def calculate_rtf(generation_seconds: float, audio_seconds: float) -> float:
    """Return real-time factor, rejecting missing or non-positive durations."""

    if generation_seconds < 0:
        raise ValueError("generation_seconds must be non-negative")
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    return generation_seconds / audio_seconds


def _validate_rate(sample_rate: int) -> None:
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


def write_pcm_wav(path: str | Path, pcm: bytes, sample_rate: int, *, overwrite: bool = False) -> WavInfo:
    """Write mono, signed 16-bit PCM and validate it immediately."""

    _validate_rate(sample_rate)
    if len(pcm) == 0 or len(pcm) % 2:
        raise ValueError("PCM data must be non-empty and 16-bit aligned")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing WAV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return validate_wav(destination)


def validate_wav(path: str | Path) -> WavInfo:
    """Verify a non-empty mono PCM WAV and return measured facts."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            codec = handle.getcomptype()
            if channels != 1 or width != 2 or codec != "NONE" or rate <= 0 or frames <= 0:
                raise ValueError("WAV must be non-empty mono 16-bit PCM")
            duration = frames / rate
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid WAV: {source}") from exc
    return WavInfo(rate, channels, width, frames, duration, source.stat().st_size)


def pcm_from_audio(value: Any) -> bytes:
    """Convert common tensor/array/list audio results to signed 16-bit PCM."""

    if isinstance(value, bytes):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple, array)):
        raise TypeError("engine output is not bytes or a numeric audio sequence")
    if not value:
        raise ValueError("engine returned empty audio")
    floating_point = any(isinstance(item, float) for item in value)
    samples = array("h")
    for item in value:
        number = float(item)
        if floating_point:
            number *= 32767.0
        samples.append(max(-32768, min(32767, int(round(number)))))
    return samples.tobytes()


def deterministic_pcm(seed: str, frames: int) -> bytes:
    """Create a deterministic low-amplitude fake waveform for routine tests."""

    if frames <= 0:
        raise ValueError("frames must be positive")
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    period = 64 + digest[0]
    amplitude = 400 + digest[1]
    values = array("h", (int(amplitude * math.sin((i % period) * 2 * math.pi / period)) for i in range(period)))
    return (values * (frames // period + 1))[:frames].tobytes()

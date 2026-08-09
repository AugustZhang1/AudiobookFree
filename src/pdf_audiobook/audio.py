"""Safe intermediate PCM WAV storage and validation."""

from __future__ import annotations

import hashlib
import math
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
import stat
import tempfile
import wave
from typing import Any


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_seconds: float
    file_bytes: int


def _reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _regular(path: Path) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"missing WAV: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or _reparse(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError("WAV must be a regular non-reparse file")


def validate_wav(path: str | Path, *, expected_sample_rate: int | None = None) -> WavInfo:
    source = Path(path)
    _regular(source)
    try:
        with wave.open(str(source), "rb") as handle:
            info = WavInfo(handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes(), 0.0, source.stat().st_size)
            if info.channels != 1 or info.sample_width != 2 or handle.getcomptype() != "NONE" or info.sample_rate <= 0 or info.frames <= 0:
                raise ValueError("WAV must be non-empty mono signed 16-bit PCM")
            if expected_sample_rate is not None and info.sample_rate != expected_sample_rate:
                raise ValueError("WAV sample rate does not match the synthesis settings")
            return WavInfo(info.sample_rate, 1, 2, info.frames, info.frames / info.sample_rate, info.file_bytes)
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"invalid WAV: {source}") from exc


def write_pcm_wav(path: str | Path, pcm: bytes, sample_rate: int, *, overwrite: bool = False) -> WavInfo:
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError("PCM must be non-empty signed 16-bit bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _regular(destination)
        if not overwrite:
            raise FileExistsError(destination)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with wave.open(raw, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm)
            raw.flush()
            os.fsync(raw.fileno())
        if destination.exists() or destination.is_symlink():
            _regular(destination)
            if not overwrite:
                raise FileExistsError(destination)
        os.replace(temporary, destination)
        return validate_wav(destination, expected_sample_rate=sample_rate)
    finally:
        temporary.unlink(missing_ok=True)


def calculate_rtf(generation_seconds: float, audio_seconds: float) -> float:
    if generation_seconds < 0 or audio_seconds <= 0:
        raise ValueError("durations must be non-negative and audio duration positive")
    return generation_seconds / audio_seconds


def pcm_from_audio(value: Any) -> bytes:
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
    if not isinstance(value, (list, tuple, array)) or not value:
        raise TypeError("engine output is not non-empty audio")
    floating = any(isinstance(item, float) for item in value)
    samples = array("h")
    for item in value:
        number = float(item) * 32767.0 if floating else float(item)
        samples.append(max(-32768, min(32767, int(round(number)))))
    return samples.tobytes()


def deterministic_pcm(seed: str, frames: int) -> bytes:
    if frames <= 0:
        raise ValueError("frames must be positive")
    digest = hashlib.sha256(seed.encode()).digest()
    period, amplitude = 64 + digest[0], 400 + digest[1]
    values = array("h", (int(amplitude * math.sin(i * 2 * math.pi / period)) for i in range(period)))
    return (values * (frames // period + 1))[:frames].tobytes()


__all__ = ["WavInfo", "calculate_rtf", "deterministic_pcm", "pcm_from_audio", "validate_wav", "write_pcm_wav"]

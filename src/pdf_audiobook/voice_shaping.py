"""Bounded FFmpeg/rubberband boundary for deterministic cast shaping."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import subprocess
from typing import Any

from .voice_settings import canonical_voice_settings

SHAPING_IMPLEMENTATION = "voice-shaping-v3"
PREVIEW_SETTINGS_IMPLEMENTATION = "voice-preview-settings-v1"
SHAPING_TIMEOUT_SECONDS = 20
MAX_SHAPING_ERROR_BYTES = 4096

# Warm and bright use opposing broad spectral tilts: a +3 dB shelf at one
# edge and -2 dB at the other. A conservative -3 dB preamp (10^(-3/20) =
# 0.70794578) leaves bounded headroom for the boosted shelf. These constants
# are part of the DSP fingerprint, so changing them invalidates old audio.
WARM_FILTER = "volume=0.70794578,lowshelf=f=180:g=3:w=0.7,highshelf=f=3600:g=-2:w=0.7"
BRIGHT_FILTER = "volume=0.70794578,lowshelf=f=180:g=-2:w=0.7,highshelf=f=3600:g=3:w=0.7"


class VoiceShapingError(RuntimeError):
    pass


class VoiceShapingUnavailable(VoiceShapingError):
    pass


@dataclass(frozen=True, slots=True)
class ShapingCapability:
    ffmpeg: str | None
    rubberband: bool
    ffmpeg_version: str
    rubberband_version: str
    fingerprint: str
    tone_available: bool = False

    @property
    def available(self) -> bool:
        return self.ffmpeg is not None

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "rubberband": self.rubberband, "pitch_available": self.rubberband, "tone_available": self.tone_available, "ffmpeg_version": self.ffmpeg_version, "rubberband_version": self.rubberband_version, "fingerprint": self.fingerprint}


_capability: ShapingCapability | None = None


def _tool() -> str | None:
    configured = os.environ.get("PDF_AUDIOBOOK_FFMPEG")
    if configured:
        path = shutil.which(configured) or configured
        if os.path.isfile(path):
            return path
    return shutil.which("ffmpeg")


def _probe(argv: list[str]) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(argv, shell=False, check=False, capture_output=True, timeout=SHAPING_TIMEOUT_SECONDS)
        return int(getattr(result, "returncode", 1)), bytes(getattr(result, "stdout", b"") or b""), bytes(getattr(result, "stderr", b"") or b"")
    except (OSError, subprocess.TimeoutExpired):
        return 1, b"", b""


def shaping_capability() -> ShapingCapability:
    global _capability
    if _capability is not None:
        return _capability
    ffmpeg = _tool()
    version = "unavailable"
    rubberband = False
    tone_available = False
    rb_version = "unavailable"
    code = 1
    if ffmpeg:
        code, out, err = _probe([ffmpeg, "-version"])
        if code == 0:
            version = (out or err).decode("utf-8", "replace").splitlines()[0][:256] if (out or err) else "unknown"
        code, out, err = _probe([ffmpeg, "-hide_banner", "-filters"])
        text = (out + err).decode("utf-8", "replace")
        rubberband = code == 0 and "rubberband" in text
        tone_available = code == 0 and "lowshelf" in text and "highshelf" in text
        rb_version = "ffmpeg-rubberband" if rubberband else "unavailable"
    raw = json.dumps({"implementation": SHAPING_IMPLEMENTATION, "filters": [WARM_FILTER, BRIGHT_FILTER], "version": version, "rubberband": rubberband, "tone": tone_available, "rubberband_version": rb_version}, sort_keys=True, separators=(",", ":")).encode()
    _capability = ShapingCapability(ffmpeg if code == 0 else None, rubberband, version, rb_version, hashlib.sha256(raw).hexdigest(), tone_available)
    return _capability


def shaping_fingerprint() -> str:
    return shaping_capability().fingerprint


def reset_shaping_capability_cache() -> None:
    global _capability
    _capability = None


def _filter(settings: dict[str, Any]) -> str:
    filters: list[str] = []
    pitch = settings["pitch_semitones"]
    if pitch:
        ratio = 2 ** (pitch / 12)
        filters.append(f"rubberband=pitch={ratio:.12f}:tempo=1:formant=preserved:pitchq=quality")
    if settings["tone_preset"] == "warm":
        filters.append(WARM_FILTER)
    elif settings["tone_preset"] == "bright":
        filters.append(BRIGHT_FILTER)
    return ",".join(filters)


def shape_pcm(pcm: bytes, sample_rate: int, settings: Any) -> bytes:
    """Shape mono signed-16 PCM, bypassing FFmpeg for the neutral preset."""

    source = settings if isinstance(settings, dict) else {}
    canonical = canonical_voice_settings({key: source[key] for key in ("speed", "pitch_semitones", "tone_preset") if key in source})
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise VoiceShapingError("PCM must be non-empty, aligned signed-16 data")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise VoiceShapingError("sample rate is invalid")
    if canonical["pitch_semitones"] == 0 and canonical["tone_preset"] == "neutral":
        return pcm
    capability = shaping_capability()
    if not capability.available:
        raise VoiceShapingUnavailable("FFmpeg is required for non-neutral voice shaping")
    if canonical["pitch_semitones"] and not capability.rubberband:
        raise VoiceShapingUnavailable("FFmpeg rubberband support is required for pitch shaping")
    if canonical["tone_preset"] != "neutral" and not capability.tone_available:
        raise VoiceShapingUnavailable("FFmpeg shelf filters are required for tone shaping")
    argv = [capability.ffmpeg or "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-af", _filter(canonical), "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    try:
        result = subprocess.run(argv, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, check=False, timeout=SHAPING_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise VoiceShapingError("voice shaping timed out") from exc
    except OSError as exc:
        raise VoiceShapingError("voice shaping failed to start") from exc
    output = bytes(getattr(result, "stdout", b"") or b"")
    if getattr(result, "returncode", 1) != 0:
        detail = bytes(getattr(result, "stderr", b"") or b"")[:MAX_SHAPING_ERROR_BYTES].decode("utf-8", "replace")
        raise VoiceShapingError("voice shaping failed" + (f": {detail}" if detail else ""))
    if not output or len(output) % 2:
        raise VoiceShapingError("voice shaping returned empty or misaligned PCM")
    return output


__all__ = ["BRIGHT_FILTER", "MAX_SHAPING_ERROR_BYTES", "PREVIEW_SETTINGS_IMPLEMENTATION", "SHAPING_IMPLEMENTATION", "ShapingCapability", "VoiceShapingError", "VoiceShapingUnavailable", "WARM_FILTER", "reset_shaping_capability_cache", "shape_pcm", "shaping_capability", "shaping_fingerprint"]

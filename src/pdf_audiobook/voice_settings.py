"""Canonical, user-facing per-cast voice settings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


class VoiceSettingsError(ValueError):
    """A cast voice setting is malformed or outside the supported bounds."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


DEFAULT_VOICE_SETTINGS = {"speed": 1.0, "pitch_semitones": 0, "tone_preset": "neutral"}
VOICE_SETTING_FIELDS = frozenset(DEFAULT_VOICE_SETTINGS)
TONE_PRESETS = frozenset({"neutral", "warm", "bright"})


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    speed: float = 1.0
    pitch_semitones: int = 0
    tone_preset: str = "neutral"

    def __post_init__(self) -> None:
        checked = validate_voice_settings({"speed": self.speed, "pitch_semitones": self.pitch_semitones, "tone_preset": self.tone_preset})
        object.__setattr__(self, "speed", checked["speed"])
        object.__setattr__(self, "pitch_semitones", checked["pitch_semitones"])
        object.__setattr__(self, "tone_preset", checked["tone_preset"])

    def as_dict(self) -> dict[str, Any]:
        return {"speed": self.speed, "pitch_semitones": self.pitch_semitones, "tone_preset": self.tone_preset}


def validate_voice_settings(value: Any, *, allow_legacy: bool = True) -> dict[str, Any]:
    """Return the complete canonical mapping; speed-only legacy maps to neutral."""

    if not isinstance(value, Mapping):
        raise VoiceSettingsError("INVALID_VOICE_SETTINGS", "voice_settings must be an object")
    keys = set(value)
    if allow_legacy and keys == {"speed"}:
        value = {**DEFAULT_VOICE_SETTINGS, "speed": value["speed"]}
    elif keys != VOICE_SETTING_FIELDS:
        raise VoiceSettingsError("INVALID_VOICE_SETTINGS", "voice_settings schema mismatch")
    speed = value.get("speed")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise VoiceSettingsError("INVALID_SPEED", "speed must be numeric")
    try:
        speed_float = float(speed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VoiceSettingsError("INVALID_SPEED", "speed must be finite and between 0.5 and 2.0") from exc
    if not math.isfinite(speed_float) or not 0.5 <= speed_float <= 2.0:
        raise VoiceSettingsError("INVALID_SPEED", "speed must be finite and between 0.5 and 2.0")
    pitch = value.get("pitch_semitones")
    if isinstance(pitch, bool) or type(pitch) is not int or not -3 <= pitch <= 3:
        raise VoiceSettingsError("INVALID_PITCH", "pitch_semitones must be an integer from -3 to 3")
    tone = value.get("tone_preset")
    if not isinstance(tone, str) or tone not in TONE_PRESETS:
        raise VoiceSettingsError("INVALID_TONE", "tone_preset must be neutral, warm, or bright")
    return {"speed": speed_float, "pitch_semitones": pitch, "tone_preset": tone}


def canonical_voice_settings(value: Any, *, allow_legacy: bool = True) -> dict[str, Any]:
    return validate_voice_settings(value, allow_legacy=allow_legacy)


def voice_settings_digest(value: Any) -> str:
    canonical = validate_voice_settings(value)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["DEFAULT_VOICE_SETTINGS", "TONE_PRESETS", "VOICE_SETTING_FIELDS", "VoiceSettings", "VoiceSettingsError", "canonical_voice_settings", "validate_voice_settings", "voice_settings_digest"]

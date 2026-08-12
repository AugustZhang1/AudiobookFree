"""Isolated, short Kokoro voice-preview generation boundary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import tempfile
import hashlib
import json
import time
from typing import Any, Callable

from .audio import validate_wav, write_pcm_wav
from .tts import KOKORO_SAMPLE_RATE, SynthesisSettings, load_voice
from .voice_registry import VoiceRegistryError, get_generation_facts, require_enabled_voice_id
from .voice_settings import canonical_voice_settings
from .voice_shaping import shape_pcm, shaping_fingerprint


PREVIEW_TEXT = "This is a short voice preview for your audiobook."
PREVIEW_TEXT_VERSION = "preview-text-v1"


def preview_cache_key(voice_id: str, settings: Any, generation_facts: Any | None = None, *, shaping_identity: str | None = None) -> str:
    facts = dict(generation_facts) if isinstance(generation_facts, dict) else get_generation_facts(voice_id)
    payload = {"voice": voice_id, "settings": canonical_voice_settings(settings), "generation_facts": facts, "text": PREVIEW_TEXT, "text_version": PREVIEW_TEXT_VERSION, "shaping_fingerprint": shaping_identity or shaping_fingerprint()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def preview_cache_target(root: os.PathLike[str] | str, voice_id: str, settings: Any) -> Path:
    """Return a digest filename for shaped previews; neutral uses legacy targets."""

    canonical = canonical_voice_settings(settings)
    base = Path(root)
    if canonical == {"speed": 1.0, "pitch_semitones": 0, "tone_preset": "neutral"}:
        return base / f"sample-kokoro-{voice_id}.wav"
    if not base.is_dir() or base.is_symlink():
        raise ValueError("preview cache root is unsafe")
    cache = base / ".voice-preview-cache"
    if cache.exists() and (cache.is_symlink() or not cache.is_dir()):
        raise ValueError("preview cache directory is unsafe")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{preview_cache_key(voice_id, canonical)}.wav"


def cleanup_preview_cache(root: os.PathLike[str] | str, *, max_files: int = 64, max_age_seconds: int = 7 * 24 * 3600) -> int:
    """Remove only regular digest WAVs from the contained preview cache."""

    if type(max_files) is not int or max_files < 1 or type(max_age_seconds) is not int or max_age_seconds < 1:
        raise ValueError("preview cleanup bounds are invalid")
    cache = Path(root) / ".voice-preview-cache"
    if not cache.is_dir() or cache.is_symlink():
        return 0
    now = time.time()
    files = [entry for entry in cache.iterdir() if entry.is_file() and not entry.is_symlink() and entry.suffix == ".wav"]
    files.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
    removed = 0
    for entry in files:
        stale = now - entry.stat().st_mtime > max_age_seconds
        over_limit = files.index(entry) >= max_files
        if stale or over_limit:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _regular_target(path: Path) -> None:
    """Reject an output path that could redirect an atomic replacement."""

    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("preview target is unavailable") from exc
    reparse = bool(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0) and getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    if stat.S_ISLNK(info.st_mode) or reparse or not stat.S_ISREG(info.st_mode):
        raise ValueError("preview target is unsafe")


def generate_preview(
    voice_id: object,
    target: os.PathLike[str] | str,
    *,
    settings: SynthesisSettings | dict[str, Any] | None = None,
    voice_loader: Callable[[str, SynthesisSettings], Any] | None = None,
) -> Path:
    """Synthesize and atomically publish one validated 24 kHz preview.

    ``voice_loader`` is injectable so unit tests can use a fake voice without
    importing or starting the Kokoro model.
    """

    voice = require_enabled_voice_id(voice_id)
    destination = Path(target)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("preview target directory is unavailable")
    _regular_target(destination)
    settings_value = settings.as_dict() if isinstance(settings, SynthesisSettings) else (settings if settings is not None else {"speed": 1.0})
    settings = SynthesisSettings(**canonical_voice_settings(settings_value), sample_rate=KOKORO_SAMPLE_RATE)
    loader = voice_loader or load_voice
    loaded = None
    temporary: Path | None = None
    try:
        loaded = loader(voice, settings)
        pcm = shape_pcm(loaded.synthesize(PREVIEW_TEXT), KOKORO_SAMPLE_RATE, settings.as_dict())
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
        os.close(fd)
        temporary = Path(name)
        write_pcm_wav(temporary, pcm, KOKORO_SAMPLE_RATE, overwrite=True)
        validate_wav(temporary, expected_sample_rate=KOKORO_SAMPLE_RATE)
        os.replace(temporary, destination)
        temporary = None
        validate_wav(destination, expected_sample_rate=KOKORO_SAMPLE_RATE)
        return destination
    finally:
        try:
            if loaded is not None:
                loaded.close_voice()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one voice preview")
    parser.add_argument("voice")
    parser.add_argument("target")
    parser.add_argument("settings", nargs="?", default='{"speed":1.0}')
    args = parser.parse_args(argv)
    try:
        generate_preview(args.voice, args.target, settings=json.loads(args.settings))
    except Exception:
        # The web process reports only a bounded error; do not expose paths,
        # model details, or source text through this child process.
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    sys.exit(main())


__all__ = ["PREVIEW_TEXT", "PREVIEW_TEXT_VERSION", "cleanup_preview_cache", "generate_preview", "main", "preview_cache_key", "preview_cache_target"]

"""Isolated, short Kokoro voice-preview generation boundary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Callable

from .audio import validate_wav, write_pcm_wav
from .tts import KOKORO_SAMPLE_RATE, SynthesisSettings, load_voice
from .voice_registry import VoiceRegistryError, require_enabled_voice_id


PREVIEW_TEXT = "This is a short voice preview for your audiobook."


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
    settings = SynthesisSettings(speed=1.0, sample_rate=KOKORO_SAMPLE_RATE)
    loader = voice_loader or load_voice
    loaded = None
    temporary: Path | None = None
    try:
        loaded = loader(voice, settings)
        pcm = loaded.synthesize(PREVIEW_TEXT)
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
    args = parser.parse_args(argv)
    try:
        generate_preview(args.voice, args.target)
    except Exception:
        # The web process reports only a bounded error; do not expose paths,
        # model details, or source text through this child process.
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    sys.exit(main())


__all__ = ["PREVIEW_TEXT", "generate_preview", "main"]

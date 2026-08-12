"""Isolated, offline Chatterbox Nano reference-voice preview generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Callable

from .audio import validate_wav, write_pcm_wav
from .chatterbox_reference import validate_reference_file
from .tts import CHATTERBOX_BUILTIN_VOICE, CHATTERBOX_CHUNK_CAP, CHATTERBOX_REFERENCE_VOICE, CHATTERBOX_SAMPLE_RATE, SynthesisSettings, load_voice


PREVIEW_TEXT = "This is a short reference voice preview for your audiobook."
BUILTIN_PREVIEW_TEXT = "This is a short built-in Chatterbox voice preview for your audiobook."


def _safe_regular(path: Path, *, allow_missing: bool = False) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            return
        raise ValueError("path is missing")
    except OSError as exc:
        raise ValueError("path is unavailable") from exc
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or (flag and getattr(info, "st_file_attributes", 0) & flag) or not stat.S_ISREG(info.st_mode):
        raise ValueError("path is unsafe")


def _safe_parent(path: Path) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("target directory is unavailable") from exc
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or (flag and getattr(info, "st_file_attributes", 0) & flag) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("target directory is unsafe")


def generate_preview(
    reference_wav: str | os.PathLike[str] | Path,
    target: str | os.PathLike[str] | Path,
    *,
    voice_loader: Callable[..., Any] | None = None,
) -> Path:
    """Synthesize one fixed Chatterbox preview and atomically publish its WAV."""

    reference = Path(reference_wav)
    destination = Path(target)
    _safe_regular(reference)
    validate_reference_file(reference)
    _safe_parent(destination.parent)
    _safe_regular(destination, allow_missing=True)
    settings = SynthesisSettings(speed=1.0, sample_rate=CHATTERBOX_SAMPLE_RATE, chunk_cap=CHATTERBOX_CHUNK_CAP)
    loader = voice_loader or load_voice
    loaded = None
    temporary: Path | None = None
    try:
        loaded = loader(CHATTERBOX_REFERENCE_VOICE, settings, engine="chatterbox", reference_wav=reference)
        pcm = loaded.synthesize(PREVIEW_TEXT)
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        write_pcm_wav(temporary, pcm, CHATTERBOX_SAMPLE_RATE, overwrite=True)
        validate_wav(temporary, expected_sample_rate=CHATTERBOX_SAMPLE_RATE)
        os.replace(temporary, destination)
        temporary = None
        validate_wav(destination, expected_sample_rate=CHATTERBOX_SAMPLE_RATE)
        return destination
    finally:
        try:
            if loaded is not None:
                loaded.close_voice()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def generate_builtin_preview(
    target: str | os.PathLike[str] | Path,
    *,
    voice_loader: Callable[..., Any] | None = None,
) -> Path:
    """Synthesize the bundled Nano voice without a reference prompt."""

    destination = Path(target)
    _safe_parent(destination.parent)
    _safe_regular(destination, allow_missing=True)
    settings = SynthesisSettings(speed=1.0, sample_rate=CHATTERBOX_SAMPLE_RATE, chunk_cap=CHATTERBOX_CHUNK_CAP)
    loader = voice_loader or load_voice
    loaded = None
    temporary: Path | None = None
    try:
        # Deliberately omit reference_wav: Nano's bundled voice must not depend
        # on an active conversion or a user-uploaded reference.
        loaded = loader(CHATTERBOX_BUILTIN_VOICE, settings, engine="chatterbox")
        pcm = loaded.synthesize(BUILTIN_PREVIEW_TEXT)
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        write_pcm_wav(temporary, pcm, CHATTERBOX_SAMPLE_RATE, overwrite=True)
        validate_wav(temporary, expected_sample_rate=CHATTERBOX_SAMPLE_RATE)
        os.replace(temporary, destination)
        temporary = None
        validate_wav(destination, expected_sample_rate=CHATTERBOX_SAMPLE_RATE)
        return destination
    finally:
        try:
            if loaded is not None:
                loaded.close_voice()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one offline Chatterbox preview")
    parser.add_argument("--builtin", action="store_true", help="preview the bundled Nano voice")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        if args.builtin:
            if len(args.paths) != 1:
                parser.error("--builtin requires exactly one target path")
            generate_builtin_preview(args.paths[0])
        else:
            if len(args.paths) != 2:
                parser.error("custom preview requires a reference WAV and target path")
            generate_preview(args.paths[0], args.paths[1])
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["BUILTIN_PREVIEW_TEXT", "PREVIEW_TEXT", "generate_builtin_preview", "generate_preview", "main"]

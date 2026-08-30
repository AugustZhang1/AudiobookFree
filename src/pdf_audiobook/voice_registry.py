"""Static, validated registry for the application's approved Kokoro voices."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any


class VoiceRegistryError(ValueError):
    """Bounded error raised for invalid voice or unsafe preview inputs."""


# Keep the tuple representation immutable and independent of caller-owned data.
_FIELDS = (
    "id",
    "display_label",
    "description",
    "language",
    "engine",
    "package",
    "package_version",
    "model",
    "model_revision",
    "model_checksum",
    "voice_version",
    "voice_checksum",
    "sample_rate",
    "enabled",
    "quality_tier",
    "attribution",
    "source_url",
    "preview_filename",
    "preview_sha256",
    "preview_available",
)

_GENERATION_FIELDS = (
    "id",
    "language",
    "engine",
    "package",
    "package_version",
    "model",
    "model_revision",
    "model_checksum",
    "voice_version",
    "voice_checksum",
    "sample_rate",
    "enabled",
)

_SOURCE_URL = "https://huggingface.co/hexgrad/Kokoro-82M"
_PREVIEW_PREFIX = "sample-kokoro-"
_PREVIEW_SUFFIX = ".wav"

_VOICE_RECORDS = (
    (
        "af_heart", "Heart", "Warm and clear American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_heart{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_alloy", "Alloy", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_alloy{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_aoede", "Aoede", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_aoede{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_bella", "Bella", "Bright and measured American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_bella{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_jessica", "Jessica", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_jessica{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_kore", "Kore", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_kore{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_nicole", "Nicole", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_nicole{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_nova", "Nova", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_nova{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_river", "River", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_river{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_sarah", "Sarah", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_sarah{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "af_sky", "Sky", "American English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}af_sky{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_adam", "Adam", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_adam{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_echo", "Echo", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_echo{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_eric", "Eric", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_eric{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_fenrir", "Fenrir", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_fenrir{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_liam", "Liam", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_liam{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_michael", "Michael", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_michael{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_onyx", "Onyx", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_onyx{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_puck", "Puck", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_puck{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "am_santa", "Santa", "American English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}am_santa{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bf_alice", "Alice", "British English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bf_alice{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bf_emma", "Emma", "Natural and steady British English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bf_emma{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bf_isabella", "Isabella", "Thoughtful and calm British English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bf_isabella{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bf_lily", "Lily", "British English female voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bf_lily{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bm_daniel", "Daniel", "British English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bm_daniel{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bm_fable", "Fable", "British English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bm_fable{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bm_george", "George", "British English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bm_george{_PREVIEW_SUFFIX}", None, False,
    ),
    (
        "bm_lewis", "Lewis", "British English male voice", "en", "kokoro", "kokoro", "0.9.4",
        "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", "captured-at-download", "unrecorded", 24000,
        True, "standard", "Kokoro voice, provided by Hexgrad", _SOURCE_URL,
        f"{_PREVIEW_PREFIX}bm_lewis{_PREVIEW_SUFFIX}", None, False,
    ),
)

APPROVED_VOICE_IDS = tuple(record[0] for record in _VOICE_RECORDS)

_VOICE_BY_ID = MappingProxyType({record[0]: record for record in _VOICE_RECORDS})

# Public catalog metadata is derived from the stable Kokoro ID family rather
# than generation records. The voice-plan casting contract may consume exactly
# these values: gender is ``female`` or ``male`` and accent is ``American`` or
# ``British``. These descriptive fields are intentionally excluded from
# ``_GENERATION_FIELDS`` and therefore do not affect generation facts or the
# registry revision/hash.
_PUBLIC_METADATA_BY_FAMILY = MappingProxyType({
    "af": ("female", "American"),
    "am": ("male", "American"),
    "bf": ("female", "British"),
    "bm": ("male", "British"),
})


def _record_projection(record: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(_FIELDS, record))


def _public_record_projection(record: tuple[Any, ...]) -> dict[str, Any]:
    projected = _record_projection(record)
    try:
        gender, accent = _PUBLIC_METADATA_BY_FAMILY[projected["id"][:2]]
    except (KeyError, TypeError) as exc:
        raise VoiceRegistryError("voice metadata is unavailable") from exc
    projected.update(gender=gender, accent=accent)
    return projected


def _canonical_generation_entries() -> bytes:
    entries = []
    for record in _VOICE_RECORDS:
        projected = _record_projection(record)
        entries.append({field: projected[field] for field in _GENERATION_FIELDS})
    return json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


REGISTRY_REVISION = hashlib.sha256(_canonical_generation_entries()).hexdigest()


def registry_revision() -> str:
    """Return the deterministic revision for generation-relevant entries."""

    return REGISTRY_REVISION


def list_public_entries() -> tuple[dict[str, Any], ...]:
    """Return ordered public entries with stable catalog metadata.

    Every entry contains ``gender`` (``female`` or ``male``) and ``accent``
    (``American`` or ``British``), derived from the voice ID's ``af``, ``am``,
    ``bf``, or ``bm`` family. Consumers such as voice-plan casting may use
    these descriptive fields for matching; they are not generation inputs and
    are absent from :func:`get_generation_facts`.
    """

    return tuple(_public_record_projection(record) for record in _VOICE_RECORDS)


def _validated_id(voice_id: object) -> str:
    if not isinstance(voice_id, str) or not voice_id or not voice_id.isascii() or voice_id != voice_id.strip():
        raise VoiceRegistryError("voice id is invalid")
    record = _VOICE_BY_ID.get(voice_id)
    if record is None:
        raise VoiceRegistryError("voice id is not registered")
    if record[13] is not True:
        raise VoiceRegistryError("voice is disabled")
    return voice_id


def require_enabled_voice_id(voice_id: object) -> str:
    """Validate and return an enabled, canonical voice identifier."""

    return _validated_id(voice_id)


def get_generation_facts(voice_id: object) -> dict[str, Any]:
    """Return fresh generation metadata for an enabled voice."""

    record = _record_projection(_VOICE_BY_ID[_validated_id(voice_id)])
    facts = {field: record[field] for field in _GENERATION_FIELDS}
    # ``voice`` is the field name used by tts.EngineMetadata.
    facts["voice"] = facts["id"]
    return facts


def _is_reparse(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _unsafe_link_or_reparse(path: Path) -> bool:
    return path.is_symlink() or _is_reparse(path)


def _validate_preview_filename(filename: object) -> str | None:
    if filename is None:
        return None
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise VoiceRegistryError("preview filename is invalid")
    candidate = Path(filename)
    # Path uses the host's flavour; os.path checks the caller's platform while
    # ntpath catches Windows absolute paths when this module is tested elsewhere.
    import ntpath

    if os.path.isabs(filename) or ntpath.isabs(filename) or candidate.drive:
        raise VoiceRegistryError("preview path is unsafe")
    if any(part == ".." for part in candidate.parts):
        raise VoiceRegistryError("preview path is unsafe")
    return filename


def resolve_preview_target(voice_id: object, preview_root: os.PathLike[str] | str) -> Path | None:
    """Resolve a declared preview target, allowing the final file to be absent.

    ``None`` means the preview root is unavailable.  The returned path is
    always contained by a validated root; an existing target must be a regular
    non-link file.  This is the write-side counterpart to
    :func:`resolve_preview_path`.
    """

    voice = _validated_id(voice_id)
    try:
        root = Path(preview_root)
    except (TypeError, ValueError) as exc:
        raise VoiceRegistryError("preview root is invalid") from exc

    if _unsafe_link_or_reparse(root):
        raise VoiceRegistryError("preview root is unsafe")
    try:
        root_info = root.stat(follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISDIR(root_info.st_mode):
        raise VoiceRegistryError("preview root is unsafe")

    record = _record_projection(_VOICE_BY_ID[voice])
    filename = _validate_preview_filename(record["preview_filename"])
    if filename is None:
        return None

    root = root.resolve()
    candidate = root / filename
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise VoiceRegistryError("preview path is unsafe") from exc

    current = root
    parts = Path(filename).parts
    for part in parts[:-1]:
        current = current / part
        if _unsafe_link_or_reparse(current):
            raise VoiceRegistryError("preview path is unsafe")

    if _unsafe_link_or_reparse(candidate):
        raise VoiceRegistryError("preview path is unsafe")
    if candidate.exists():
        try:
            info = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise VoiceRegistryError("preview path is unsafe") from exc
        if not stat.S_ISREG(info.st_mode):
            raise VoiceRegistryError("preview path is unsafe")
    try:
        candidate.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise VoiceRegistryError("preview path is unsafe") from exc
    return candidate


def resolve_preview_path(voice_id: object, preview_root: os.PathLike[str] | str) -> Path | None:
    """Resolve a declared, existing preview below ``preview_root``.

    ``None`` means the preview is not available.  A malformed, escaping, link,
    reparse-point, or non-regular target raises ``VoiceRegistryError``.  When
    the declared target is absent, an older root-level timestamped preview is
    accepted as a read-only fallback.
    """

    candidate = resolve_preview_target(voice_id, preview_root)
    if candidate is None:
        return None
    if candidate.exists():
        return candidate

    voice = _validated_id(voice_id)
    root = candidate
    # ``resolve_preview_target`` has already validated the root and declared
    # path.  Recover the root from the caller's path for root-level legacy
    # discovery without changing the canonical write target.
    try:
        root = Path(preview_root).resolve()
    except (OSError, RuntimeError) as exc:
        raise VoiceRegistryError("preview root is unsafe") from exc

    suffix = f"-kokoro-{voice}{_PREVIEW_SUFFIX}"
    matches: list[tuple[int, str, Path]] = []
    try:
        entries = root.iterdir()
    except OSError:
        return None
    for entry in entries:
        name = entry.name
        if not name.endswith(suffix) or len(name) == len(suffix):
            continue
        if _unsafe_link_or_reparse(entry):
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        try:
            entry.resolve().relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        matches.append((info.st_mtime_ns, name, entry))
    if not matches:
        return None
    # Keep selection stable when downloads share a timestamp.
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def list_voices() -> tuple[dict[str, Any], ...]:
    """Compatibility spelling for callers asking for the public catalog."""

    return list_public_entries()


def require_enabled_voice(voice_id: object) -> str:
    return require_enabled_voice_id(voice_id)


def resolve_preview(voice_id: object, preview_root: os.PathLike[str] | str) -> Path | None:
    return resolve_preview_path(voice_id, preview_root)


__all__ = [
    "APPROVED_VOICE_IDS",
    "REGISTRY_REVISION",
    "VoiceRegistryError",
    "get_generation_facts",
    "list_public_entries",
    "list_voices",
    "registry_revision",
    "require_enabled_voice",
    "require_enabled_voice_id",
    "resolve_preview",
    "resolve_preview_path",
    "resolve_preview_target",
]

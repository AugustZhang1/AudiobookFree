"""Local conversion workspaces and their recovery manifests.

The workspace deliberately has a small file-based contract.  ``active.json``
identifies the one job that can be resumed and ``work/<conversion-id>`` holds
that job's private files.  Manifest objects are intentionally strict: adding
an unreviewed field or schema version must fail loudly rather than being
silently ignored during recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import uuid
from typing import Any, Literal

from .speaker_analysis import MAX_ARTIFACT_BYTES, SpeakerAnalysisError, validate_speaker_analysis
from .chatterbox_reference import (
    REFERENCE_DESCRIPTOR_FILENAME,
    REFERENCE_WAV_FILENAME,
    ReferenceArtifact,
    build_reference_descriptor,
    copy_reference_file,
    public_reference_status,
    validate_reference_descriptor,
    validate_reference_file,
)
from .engine_catalog import CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT
from .tts import CHATTERBOX_BUILTIN_VOICE, CHATTERBOX_SAMPLE_RATE
from .voice_analysis import MAX_ARTIFACT_BYTES as MAX_VOICE_ANALYSIS_STATUS_BYTES, VoiceAnalysisError, validate_voice_analysis_status
from .voice_plan import VoicePlanError, validate_voice_plan


ACTIVE_SCHEMA_VERSION = 1
JOB_SCHEMA_VERSION = 2
LEGACY_GENERATION_SCHEMA_VERSION = 3
GENERATION_SCHEMA_VERSION = 4
INTERACTIVE_GENERATION_SCHEMA_VERSION = 5
_KNOWN_JOB_SCHEMA_VERSION = 1
COPY_CHUNK_SIZE = 1024 * 1024
_REPLACE_RETRY_ATTEMPTS = 3
_REPLACE_RETRY_DELAY_SECONDS = 0.05
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ACTIVE_FIELDS = {"schema_version", "conversion_id", "updated_at"}
_JOB_FIELDS = {
    "schema_version",
    "conversion_id",
    "original_display_filename",
    "source_pdf_sha256",
    "status",
    "stage",
    "created_at",
    "updated_at",
    "cleaned_text_sha256",
    "chapter_plan_sha256",
    "warnings",
    "error",
}
_JOB_FIELDS_V1 = _JOB_FIELDS - {"chapter_plan_sha256"}
_CHAPTER_PLAN_FIELDS = {
    "schema_version",
    "mode",
    "requested_count",
    "cleaned_text_sha256",
    "chapters",
    "warnings",
}
_GENERATION_FIELDS_V3 = _JOB_FIELDS | {
    "tts",
    "total_chunks",
    "completed_chunks",
    "progress",
    "worker",
    "last_safe_error",
}
_GENERATION_FIELDS = _GENERATION_FIELDS_V3 | {"output"}
_COMPLETED_FIELDS_V3 = {"chapter_index", "global_index", "local_index", "input_hash", "relative_path", "duration_seconds"}
_COMPLETED_FIELDS = _COMPLETED_FIELDS_V3 | {"wav_sha256"}
_INTERACTIVE_GENERATION_FIELDS = _GENERATION_FIELDS | {
    "mode",
    "voice_plan_sha256",
    "voice_plan_revision",
    "speaker_analysis_sha256",
    "cast_voice_ids",
    "voice_registry_revision",
}
_INTERACTIVE_COMPLETED_FIELDS = _COMPLETED_FIELDS | {
    "audio_input_hash",
    "span_id",
    "speaker_id",
    "voice_id",
    "segment_type",
    "source_start",
    "source_end",
}
_OUTPUT_FIELDS = {"filename", "path", "size_bytes", "duration_seconds", "chapter_count", "codec", "sha256"}
_TTS_FIELDS = {"engine", "package_version", "model", "model_revision", "model_checksum", "voice", "voice_version", "voice_checksum", "sample_rate", "settings", "speed", "chunk_cap"}


class WorkspaceError(ValueError):
    """Base error for invalid workspace paths or manifests."""


class ManifestError(WorkspaceError):
    """A manifest is missing, malformed, or has an unsupported schema."""


class UnsafePathError(WorkspaceError):
    """A path would escape the workspace or traverse a link."""


@dataclass(frozen=True)
class StartupInspection:
    """Non-destructive startup inspection result."""

    state: Literal["no_active", "resumable", "invalid"]
    conversion_id: str | None = None
    manifest: dict[str, Any] | None = None
    reason: str | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_conversion_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ManifestError("conversion_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise ManifestError("invalid conversion_id") from None
    canonical = str(parsed)
    if value != canonical:
        raise ManifestError("conversion_id must use canonical UUID form")
    return canonical


def _validate_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ManifestError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validate_regular_manifest_file(path: Path, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ManifestError(f"missing {label}") from exc
    mode = info.st_mode
    if stat.S_ISLNK(mode) or _is_reparse(info) or not stat.S_ISREG(mode):
        raise ManifestError(f"{label} must be a regular file")


def validate_active_manifest(value: Any) -> dict[str, Any]:
    """Validate and return an ``active.json`` object."""

    if not isinstance(value, dict):
        raise ManifestError("active manifest must be an object")
    if set(value) != _ACTIVE_FIELDS:
        raise ManifestError("active manifest schema mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != ACTIVE_SCHEMA_VERSION:
        raise ManifestError("unsupported active manifest schema")
    conversion_id = _validate_conversion_id(value["conversion_id"])
    if not _valid_timestamp(value["updated_at"]):
        raise ManifestError("invalid active manifest timestamp")
    return {**value, "conversion_id": conversion_id}


def _validate_job_manifest_fields(value: Any, fields: set[str], schema_version: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("job manifest must be an object")
    if set(value) != fields:
        raise ManifestError("job manifest schema mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != schema_version:
        raise ManifestError("unsupported job manifest schema")
    conversion_id = _validate_conversion_id(value["conversion_id"])
    filename = value["original_display_filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise ManifestError("invalid original display filename")
    source_hash = _validate_sha(value["source_pdf_sha256"], "source_pdf_sha256")
    if not isinstance(value["status"], str) or not value["status"]:
        raise ManifestError("status must be a non-empty string")
    if not isinstance(value["stage"], str) or not value["stage"]:
        raise ManifestError("stage must be a non-empty string")
    if not _valid_timestamp(value["created_at"]) or not _valid_timestamp(value["updated_at"]):
        raise ManifestError("invalid job manifest timestamp")
    cleaned_hash = value["cleaned_text_sha256"]
    if cleaned_hash is not None:
        cleaned_hash = _validate_sha(cleaned_hash, "cleaned_text_sha256")
    chapter_plan_hash = value.get("chapter_plan_sha256")
    if chapter_plan_hash is not None:
        chapter_plan_hash = _validate_sha(chapter_plan_hash, "chapter_plan_sha256")
    warnings = value["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ManifestError("warnings must be a list of strings")
    error = value["error"]
    if error is not None and (not isinstance(error, str) or not error):
        raise ManifestError("error must be null or a non-empty string")
    return {
        **value,
        "conversion_id": conversion_id,
        "source_pdf_sha256": source_hash,
        "cleaned_text_sha256": cleaned_hash,
        "chapter_plan_sha256": chapter_plan_hash,
    }


def validate_job_manifest(value: Any) -> dict[str, Any]:
    """Validate the strict current ``job.json`` schema."""

    return _validate_job_manifest_fields(value, _JOB_FIELDS, JOB_SCHEMA_VERSION)


def _validate_relative_chunk_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ManifestError("chunk path must be a relative string")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parts[:1] != ("chunks",) or ".." in candidate.parts:
        raise ManifestError("chunk path must be contained under chunks")
    return value


def _validate_generation_common(value: Any, fields: set[str], completed_fields: set[str], schema_version: int) -> dict[str, Any]:
    """Validate the shared strict generation fields."""

    if not isinstance(value, dict) or set(value) != fields:
        raise ManifestError("generation manifest schema mismatch")
    _validate_job_manifest_fields({key: value[key] for key in _JOB_FIELDS}, _JOB_FIELDS, schema_version)
    tts = value["tts"]
    if not isinstance(tts, dict) or set(tts) != _TTS_FIELDS or not tts.get("engine") or not tts.get("voice") or type(tts.get("speed")) not in {int, float}:
        raise ManifestError("tts metadata is invalid")
    if not 0.5 <= float(tts["speed"]) <= 2.0 or type(tts.get("sample_rate")) is not int or tts["sample_rate"] <= 0 or type(tts.get("chunk_cap")) is not int or tts["chunk_cap"] <= 0 or not isinstance(tts.get("settings"), dict):
        raise ManifestError("tts settings are invalid")
    if type(value["total_chunks"]) is not int or value["total_chunks"] < 0:
        raise ManifestError("total_chunks must be a non-negative integer")
    completed = value["completed_chunks"]
    if not isinstance(completed, list):
        raise ManifestError("completed_chunks must be a list")
    seen: set[int] = set()
    previous_global = -1
    for record in completed:
        if not isinstance(record, dict) or set(record) != completed_fields:
            raise ManifestError("completed chunk schema mismatch")
        for key in ("chapter_index", "global_index", "local_index"):
            if type(record[key]) is not int or record[key] < 0:
                raise ManifestError("completed chunk indexes are invalid")
        if record["chapter_index"] < 1:
            raise ManifestError("completed chapter indexes must start at one")
        if record["global_index"] >= value["total_chunks"]:
            raise ManifestError("completed global index exceeds total chunks")
        if record["global_index"] <= previous_global:
            raise ManifestError("completed chunks must be in global order")
        previous_global = record["global_index"]
        if record["global_index"] in seen:
            raise ManifestError("duplicate completed chunk index")
        seen.add(record["global_index"])
        _validate_sha(record["input_hash"], "input_hash")
        _validate_relative_chunk_path(record["relative_path"])
        if type(record["duration_seconds"]) not in {int, float} or not math.isfinite(float(record["duration_seconds"])) or record["duration_seconds"] <= 0:
            raise ManifestError("completed chunk duration is invalid")
        if "wav_sha256" in completed_fields:
            _validate_sha(record["wav_sha256"], "wav_sha256")
    progress = value["progress"]
    if not isinstance(progress, dict) or set(progress) != {"completed", "current", "total"}:
        raise ManifestError("progress schema mismatch")
    if any(type(progress[key]) is not int or progress[key] < 0 or progress[key] > value["total_chunks"] for key in progress):
        raise ManifestError("progress values are invalid")
    if progress["completed"] != len(completed) or progress["total"] != value["total_chunks"]:
        raise ManifestError("progress does not match completed chunks")
    worker = value["worker"]
    if worker is not None and (not isinstance(worker, dict) or set(worker) != {"pid", "started_at", "updated_at"}):
        raise ManifestError("worker state schema mismatch")
    if worker is not None and (type(worker["pid"]) is not int or worker["pid"] <= 0 or not _valid_timestamp(worker["started_at"]) or not _valid_timestamp(worker["updated_at"])):
        raise ManifestError("worker state is invalid")
    if value["last_safe_error"] is not None and (not isinstance(value["last_safe_error"], str) or not value["last_safe_error"]):
        raise ManifestError("last_safe_error is invalid")
    return {**value, "schema_version": schema_version}


def validate_generation_manifest_v3(value: Any) -> dict[str, Any]:
    """Validate the strict legacy v3 recovery manifest."""

    return _validate_generation_common(value, _GENERATION_FIELDS_V3, _COMPLETED_FIELDS_V3, LEGACY_GENERATION_SCHEMA_VERSION)


def _validate_output(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _OUTPUT_FIELDS:
        raise ManifestError("output record schema mismatch")
    for key in ("filename", "path", "codec"):
        if not isinstance(value[key], str) or not value[key] or "\x00" in value[key]:
            raise ManifestError(f"output {key} is invalid")
    if not Path(value["path"]).is_absolute():
        raise ManifestError("output path must be absolute")
    if value["filename"] != Path(value["path"]).name:
        raise ManifestError("output filename does not match path")
    if value["codec"] != "aac":
        raise ManifestError("output codec must be AAC")
    if type(value["size_bytes"]) is not int or value["size_bytes"] <= 0:
        raise ManifestError("output size is invalid")
    if type(value["chapter_count"]) is not int or value["chapter_count"] <= 0:
        raise ManifestError("output chapter count is invalid")
    if type(value["duration_seconds"]) not in {int, float} or not __import__("math").isfinite(float(value["duration_seconds"])) or value["duration_seconds"] <= 0:
        raise ManifestError("output duration is invalid")
    _validate_sha(value["sha256"], "output sha256")
    return value


def validate_generation_manifest(value: Any) -> dict[str, Any]:
    """Validate the strict current v4 recovery manifest."""

    result = _validate_generation_common(value, _GENERATION_FIELDS, _COMPLETED_FIELDS, GENERATION_SCHEMA_VERSION)
    output = _validate_output(value["output"])
    if value["status"] == "completed":
        if value["stage"] == "synthesis_complete":
            if output is not None:
                raise ManifestError("synthesis-complete generation cannot include final output")
        elif value["stage"] == "completed":
            if output is None:
                raise ManifestError("final completed generation must include output facts")
        else:
            raise ManifestError("completed generation has an invalid final stage")
        if value["progress"]["completed"] != value["total_chunks"] or value["progress"]["current"] != value["total_chunks"] or len(value["completed_chunks"]) != value["total_chunks"]:
            raise ManifestError("completed generation progress is incomplete")
    elif output is not None:
        raise ManifestError("output is only valid for a completed generation")
    return result


def _validate_generation_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or any(ord(char) < 32 for char in value):
        raise ManifestError(f"{field} must be a bounded non-empty identifier")
    return value


def validate_interactive_generation_manifest(value: Any) -> dict[str, Any]:
    """Validate the strict schema-v5 Interactive Voices recovery manifest."""

    result = _validate_generation_common(
        value,
        _INTERACTIVE_GENERATION_FIELDS,
        _INTERACTIVE_COMPLETED_FIELDS,
        INTERACTIVE_GENERATION_SCHEMA_VERSION,
    )
    if value["mode"] != "interactive_voices":
        raise ManifestError("interactive generation mode is invalid")
    _validate_sha(value["voice_plan_sha256"], "voice_plan_sha256")
    if type(value["voice_plan_revision"]) is not int or value["voice_plan_revision"] <= 0:
        raise ManifestError("voice_plan_revision must be a positive integer")
    _validate_sha(value["speaker_analysis_sha256"], "speaker_analysis_sha256")
    _validate_sha(value["voice_registry_revision"], "voice_registry_revision")
    cast_voice_ids = value["cast_voice_ids"]
    if not isinstance(cast_voice_ids, list) or not cast_voice_ids:
        raise ManifestError("cast_voice_ids must be a non-empty list")
    seen_voice_ids: set[str] = set()
    for voice_id in cast_voice_ids:
        _validate_generation_identifier(voice_id, "cast_voice_id")
        if voice_id in seen_voice_ids:
            raise ManifestError("cast_voice_ids must be ordered and unique")
        seen_voice_ids.add(voice_id)
    if not isinstance(value["tts"].get("voice"), str) or value["tts"]["voice"] not in seen_voice_ids:
        raise ManifestError("tts voice must occur in cast_voice_ids")

    previous_chapter = -1
    previous_local = -1
    for record in value["completed_chunks"]:
        _validate_sha(record["audio_input_hash"], "audio_input_hash")
        _validate_generation_identifier(record["span_id"], "span_id")
        _validate_generation_identifier(record["speaker_id"], "speaker_id")
        _validate_generation_identifier(record["voice_id"], "voice_id")
        if record["voice_id"] not in seen_voice_ids:
            raise ManifestError("completed chunk voice_id must occur in cast_voice_ids")
        if not isinstance(record["segment_type"], str) or record["segment_type"] not in {"narration", "dialogue", "thought", "unknown"}:
            raise ManifestError("segment_type is invalid")
        if type(record["source_start"]) is not int or type(record["source_end"]) is not int:
            raise ManifestError("completed chunk source offsets are invalid")
        if record["source_start"] < 0 or record["source_end"] <= record["source_start"]:
            raise ManifestError("completed chunk source range must be non-empty and non-negative")
        chapter = record["chapter_index"]
        local = record["local_index"]
        if chapter < previous_chapter or (chapter == previous_chapter and local <= previous_local):
            raise ManifestError("completed chunks must be ordered by chapter and local index")
        previous_chapter = chapter
        previous_local = local

    output = _validate_output(value["output"])
    if value["status"] == "completed":
        if value["stage"] == "synthesis_complete":
            if output is not None:
                raise ManifestError("synthesis-complete generation cannot include final output")
        elif value["stage"] == "completed":
            if output is None:
                raise ManifestError("final completed generation must include output facts")
        else:
            raise ManifestError("completed generation has an invalid final stage")
        if value["progress"]["completed"] != value["total_chunks"] or value["progress"]["current"] != value["total_chunks"] or len(value["completed_chunks"]) != value["total_chunks"]:
            raise ManifestError("completed generation progress is incomplete")
    elif output is not None:
        raise ManifestError("output is only valid for a completed generation")
    return result


def validate_generation_manifest_v5(value: Any) -> dict[str, Any]:
    """Compatibility name for the strict Interactive Voices v5 validator."""

    return validate_interactive_generation_manifest(value)


def _upgrade_generation_manifest(value: dict[str, Any], *, tts: dict[str, Any], total_chunks: int) -> dict[str, Any]:
    upgraded = {**value, "schema_version": GENERATION_SCHEMA_VERSION, "tts": dict(tts), "total_chunks": total_chunks, "completed_chunks": [], "progress": {"completed": 0, "current": 0, "total": total_chunks}, "worker": None, "last_safe_error": value.get("error"), "output": None}
    return validate_generation_manifest(upgraded)


def _upgrade_v3_generation_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Add only v4 recovery state; retain all valid v3 synthesis evidence."""

    completed = []
    for record in value["completed_chunks"]:
        item = dict(record)
        path = Path(value.get("_conversion_path", "")) / item["relative_path"] if value.get("_conversion_path") else None
        try:
            if path is None:
                raise ManifestError("legacy chunk path is unavailable")
            _validate_regular_manifest_file(path, "legacy chunk")
        except ManifestError:
            item["wav_sha256"] = "0" * 64
        else:
            item["wav_sha256"] = _hash_file(path)
        completed.append(item)
    upgraded = {**value, "schema_version": GENERATION_SCHEMA_VERSION, "completed_chunks": completed, "output": None}
    upgraded.pop("_conversion_path", None)
    return validate_generation_manifest(upgraded)


def _validate_job_manifest_v1(value: Any) -> dict[str, Any]:
    """Validate the exact known Phase 2 ``job.json`` schema."""

    return _validate_job_manifest_fields(value, _JOB_FIELDS_V1, _KNOWN_JOB_SCHEMA_VERSION)


def _validate_job_for_read(value: Any) -> dict[str, Any]:
    """Validate a current job or migrate the one known legacy schema."""

    if not isinstance(value, dict):
        raise ManifestError("job manifest must be an object")
    version = value.get("schema_version")
    if type(version) is int and version == INTERACTIVE_GENERATION_SCHEMA_VERSION:
        return validate_interactive_generation_manifest(value)
    if type(version) is int and version == GENERATION_SCHEMA_VERSION:
        return validate_generation_manifest(value)
    if type(version) is int and version == LEGACY_GENERATION_SCHEMA_VERSION:
        return validate_generation_manifest_v3(value)
    if type(version) is int and version == _KNOWN_JOB_SCHEMA_VERSION:
        migrated = _validate_job_manifest_v1(value)
        migrated["schema_version"] = JOB_SCHEMA_VERSION
        migrated["chapter_plan_sha256"] = None
        return migrated
    return validate_job_manifest(value)


def _validate_chapter_plan(value: Any, cleaned_hash: str) -> dict[str, Any]:
    """Validate the persistence-critical shape of a chapter plan."""

    if not isinstance(value, dict) or set(value) != _CHAPTER_PLAN_FIELDS:
        raise ManifestError("chapter plan schema mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ManifestError("unsupported chapter plan schema")
    mode = value["mode"]
    requested_count = value["requested_count"]
    if mode not in {"original", "custom", "whole"}:
        raise ManifestError("invalid chapter plan mode")
    if mode == "custom":
        if type(requested_count) is not int or not 2 <= requested_count <= 50:
            raise ManifestError("custom chapter plan count must be an integer from 2 through 50")
    elif requested_count is not None:
        raise ManifestError("requested_count must be null for original and whole plans")
    if value["cleaned_text_sha256"] != cleaned_hash:
        raise ManifestError("chapter plan cleaned text hash mismatch")
    _validate_sha(value["cleaned_text_sha256"], "cleaned_text_sha256")
    if not isinstance(value["chapters"], list):
        raise ManifestError("chapter plan chapters must be a list")
    if not isinstance(value["warnings"], list) or any(not isinstance(item, str) for item in value["warnings"]):
        raise ManifestError("chapter plan warnings must be a list of strings")
    return value


def _validate_cleaned_map(value: Any, cleaned_text: str) -> list[dict[str, Any]]:
    """Validate the ranges required by the chapter planner."""

    if not isinstance(value, list) or not value:
        raise ManifestError("cleaned map must be a non-empty list")
    previous_start = -1
    previous_end = 0
    for item in value:
        if not isinstance(item, dict):
            raise ManifestError("cleaned map entries must be objects")
        try:
            source_page = item["source_page"]
            start = item["cleaned_start"]
            end = item["cleaned_end"]
        except KeyError as exc:
            raise ManifestError("cleaned map entries require source_page, cleaned_start, and cleaned_end") from exc
        if (
            any(type(number) is not int for number in (source_page, start, end))
            or source_page < 1
            or start < 0
            or end <= start
            or end > len(cleaned_text)
        ):
            raise ManifestError("cleaned map ranges are invalid")
        if start < previous_start or start < previous_end:
            raise ManifestError("cleaned map ranges must be ordered and non-overlapping")
        previous_start = start
        previous_end = end
    if value[0]["cleaned_start"] != 0 or value[-1]["cleaned_end"] != len(cleaned_text):
        raise ManifestError("cleaned map must cover the cleaned text")
    return value


def _validate_analysis_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("analysis artifact must be an object")
    return value


def _replace_with_retries(source: Path, target: Path) -> None:
    """Replace a same-directory target, tolerating brief Windows lock races."""

    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(source, target)
        except PermissionError:
            if attempt + 1 >= _REPLACE_RETRY_ATTEMPTS:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)
        else:
            return


def _atomic_write(path: Path, writer: Any) -> Path:
    """Write through a same-directory temporary file and atomically replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (target.is_symlink() or target.is_dir()):
        raise UnsafePathError(f"refusing to write through {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retries(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(path: Path, value: Any) -> Path:
    """Atomically write UTF-8 JSON to ``path``."""

    canonical = _canonical_json_text(value)
    return _atomic_write(path, lambda handle: handle.write(canonical))


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def atomic_write_text(path: Path, value: str) -> Path:
    """Atomically write UTF-8 text to ``path``."""

    if not isinstance(value, str):
        raise TypeError("text value must be a string")
    return _atomic_write(path, lambda handle: handle.write(value))


def read_json_manifest(path: Path, validator: Any) -> dict[str, Any]:
    """Read a regular JSON manifest and apply its strict validator."""

    _validate_regular_manifest_file(Path(path), "manifest")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError("malformed manifest JSON") from exc
    return validator(value)


def copy_source_pdf(source: Path, destination: Path, *, chunk_size: int = COPY_CHUNK_SIZE) -> tuple[str, int]:
    """Stream-copy ``source`` to ``destination`` while calculating SHA-256."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(source)
    destination = Path(destination)
    try:
        source_mode = source.lstat().st_mode
    except OSError as exc:
        raise WorkspaceError("source PDF does not exist") from exc
    if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
        raise WorkspaceError("source PDF must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise WorkspaceError("destination source.pdf already exists")
    digest = hashlib.sha256()
    count = 0
    try:
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            while chunk := input_file.read(chunk_size):
                output_file.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), count


def _hash_file(path: Path, *, chunk_size: int = COPY_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class Workspace:
    """Own one application data root and its conversion directories."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.work_root = self.root / "work"
        self.active_path = self.root / "active.json"

    def _ensure_work_root(self) -> None:
        if self.work_root.is_symlink():
            raise UnsafePathError("work directory must not be a symlink")
        self.work_root.mkdir(parents=True, exist_ok=True)
        if self.work_root.is_symlink() or not self.work_root.is_dir():
            raise UnsafePathError("invalid work directory")

    def _conversion_id(self, value: str | uuid.UUID | None) -> str:
        if value is None:
            return str(uuid.uuid4())
        if isinstance(value, uuid.UUID):
            return str(value)
        try:
            return _validate_conversion_id(value)
        except ManifestError as exc:
            raise UnsafePathError(str(exc)) from exc

    def conversion_path(self, conversion_id: str | uuid.UUID) -> Path:
        """Return a contained conversion path, rejecting traversal."""

        self._ensure_work_root()
        normalized = self._conversion_id(conversion_id)
        raw_candidate = self.work_root / normalized
        if raw_candidate.is_symlink():
            raise UnsafePathError("conversion directory must not be a symlink")
        candidate = raw_candidate.resolve()
        try:
            candidate.relative_to(self.work_root.resolve())
        except ValueError as exc:
            raise UnsafePathError("conversion path escapes workspace") from exc
        if candidate == self.work_root.resolve():
            raise UnsafePathError("conversion path must be a child directory")
        return candidate

    def _write_active(self, conversion_id: str) -> None:
        atomic_write_json(
            self.active_path,
            {
                "schema_version": ACTIVE_SCHEMA_VERSION,
                "conversion_id": conversion_id,
                "updated_at": _timestamp(),
            },
        )

    def job_path(self, conversion_id: str | uuid.UUID) -> Path:
        """Return the validated path to one conversion's job manifest."""

        return self.conversion_path(conversion_id) / "job.json"

    def read_job(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        path = self.job_path(conversion_id)
        raw = read_json_manifest(path, _validate_job_for_read)
        if raw["schema_version"] == JOB_SCHEMA_VERSION:
            # A migrated object is distinguishable from a native v2 object by
            # the legacy artifact's missing chapter plan field on disk.
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ManifestError("malformed manifest JSON") from exc
            if isinstance(on_disk, dict) and "chapter_plan_sha256" not in on_disk:
                atomic_write_json(path, raw)
        if raw["schema_version"] == LEGACY_GENERATION_SCHEMA_VERSION:
            upgraded = dict(raw)
            upgraded["_conversion_path"] = str(path.parent)
            migrated = _upgrade_v3_generation_manifest(upgraded)
            atomic_write_json(path, migrated)
            return migrated
        if raw["schema_version"] == INTERACTIVE_GENERATION_SCHEMA_VERSION:
            return validate_interactive_generation_manifest(raw)
        return validate_generation_manifest(raw) if raw["schema_version"] == GENERATION_SCHEMA_VERSION else validate_job_manifest(raw)

    def chunks_path(self, conversion_id: str | uuid.UUID) -> Path:
        """Return/create the contained chunks directory."""

        directory = self.conversion_path(conversion_id)
        chunks = directory / "chunks"
        if chunks.exists() or chunks.is_symlink():
            try:
                info = chunks.stat(follow_symlinks=False)
            except OSError as exc:
                raise UnsafePathError("chunks directory is unavailable") from exc
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
                raise UnsafePathError("chunks directory is unsafe")
        else:
            chunks.mkdir(parents=False)
        return chunks

    def chatterbox_reference_path(self, conversion_id: str | uuid.UUID) -> Path:
        return self.conversion_path(conversion_id) / REFERENCE_WAV_FILENAME

    def chatterbox_reference_descriptor_path(self, conversion_id: str | uuid.UUID) -> Path:
        return self.conversion_path(conversion_id) / REFERENCE_DESCRIPTOR_FILENAME

    def _read_chatterbox_reference_descriptor(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        target = self.chatterbox_reference_descriptor_path(conversion_id)
        try:
            _validate_regular_manifest_file(target, "Chatterbox reference descriptor")
            return validate_reference_descriptor(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ManifestError("invalid Chatterbox reference descriptor") from exc

    def load_chatterbox_reference(self, conversion_id: str | uuid.UUID) -> ReferenceArtifact:
        """Load the controlled reference and fail closed on descriptor/file drift."""

        descriptor = self._read_chatterbox_reference_descriptor(conversion_id)
        path = self.chatterbox_reference_path(conversion_id)
        try:
            wav, digest = validate_reference_file(path)
        except ValueError as exc:
            raise ManifestError("invalid Chatterbox reference WAV") from exc
        if digest != descriptor["reference_sha256"] or wav.sample_rate != descriptor["sample_rate"] or wav.file_bytes != descriptor["file_bytes"]:
            raise ManifestError("Chatterbox reference identity mismatch")
        return ReferenceArtifact(path, descriptor, wav)

    def chatterbox_reference_status(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Return safe public metadata without exposing the controlled path."""

        descriptor_path = self.chatterbox_reference_descriptor_path(conversion_id)
        if not descriptor_path.exists() and not descriptor_path.is_symlink():
            if self.chatterbox_reference_path(conversion_id).exists() or self.chatterbox_reference_path(conversion_id).is_symlink():
                raise ManifestError("invalid Chatterbox reference descriptor")
            return public_reference_status(None)
        return public_reference_status(self.load_chatterbox_reference(conversion_id).descriptor)

    def _invalidate_chatterbox_generation(self, conversion_id: str | uuid.UUID) -> None:
        job = self.read_job(conversion_id)
        if job.get("schema_version") != GENERATION_SCHEMA_VERSION or job.get("tts", {}).get("engine") != "chatterbox" or job.get("tts", {}).get("voice") != "reference-wav":
            return
        chunks = self.conversion_path(conversion_id) / "chunks"
        if chunks.exists() or chunks.is_symlink():
            info = chunks.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
                raise UnsafePathError("chunks directory is unsafe")
            for entry in list(chunks.iterdir()):
                child_info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(child_info.st_mode) or _is_reparse(child_info) or not stat.S_ISREG(child_info.st_mode):
                    raise UnsafePathError("chunk entry is unsafe")
                entry.unlink()
        self.update_generation(
            conversion_id,
            status="cancelled", stage="reference_changed", worker=None,
            completed_chunks=[], progress={"completed": 0, "current": 0, "total": job["total_chunks"]},
            output=None, error="Chatterbox reference changed", last_safe_error="Chatterbox reference changed",
        )

    def store_chatterbox_reference(
        self,
        conversion_id: str | uuid.UUID,
        source: str | Path,
        *,
        consent_confirmed: bool,
        consent_evidence: str = "user-confirmed-local-reference",
        consent_recorded_at: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Copy, validate, and atomically own one Chatterbox reference WAV."""

        if consent_confirmed is not True:
            raise WorkspaceError("explicit reference consent is required")
        if not isinstance(consent_evidence, str) or not consent_evidence or len(consent_evidence) > 256 or any(ord(char) < 32 for char in consent_evidence):
            raise WorkspaceError("reference consent evidence is invalid")
        source_path = Path(source)
        descriptor_path = self.chatterbox_reference_descriptor_path(conversion_id)
        reference_path = self.chatterbox_reference_path(conversion_id)
        exists = descriptor_path.exists() or reference_path.exists() or descriptor_path.is_symlink() or reference_path.is_symlink()
        if exists and not replace:
            raise WorkspaceError("Chatterbox reference already exists")
        for target in (descriptor_path, reference_path):
            if target.exists() or target.is_symlink():
                try:
                    _validate_regular_manifest_file(target, "Chatterbox reference artifact")
                except ManifestError as exc:
                    raise UnsafePathError("Chatterbox reference artifact is unsafe") from exc
        directory = reference_path.parent
        fd, temporary_name = tempfile.mkstemp(prefix=".chatterbox-reference.", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        os.close(fd)
        try:
            try:
                copy_reference_file(source_path, temporary, chunk_size=COPY_CHUNK_SIZE)
            except ValueError as exc:
                raise WorkspaceError("reference WAV is invalid or unsafe") from exc
            try:
                wav, digest = validate_reference_file(temporary)
            except ValueError as exc:
                raise WorkspaceError("reference WAV is invalid or unsafe") from exc
            now = consent_recorded_at or _timestamp()
            descriptor = build_reference_descriptor(
                wav=wav, reference_sha256=digest, engine="chatterbox", model=CHATTERBOX_NANO_MODEL,
                model_revision=CHATTERBOX_SOURCE_COMMIT, model_checksum="unrecorded", voice="reference-wav",
                voice_version=CHATTERBOX_SOURCE_COMMIT, speed=1.0, chunk_cap=300,
                consent_confirmed=True, consent_evidence=consent_evidence, consent_recorded_at=now, created_at=now, updated_at=now,
            )
            _replace_with_retries(temporary, reference_path)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_write_json(descriptor_path, descriptor)
        if replace:
            self._invalidate_chatterbox_generation(conversion_id)
        return public_reference_status(descriptor)

    def replace_chatterbox_reference(self, conversion_id: str | uuid.UUID, source: str | Path, *, consent_confirmed: bool, consent_evidence: str = "user-confirmed-local-reference", consent_recorded_at: str | None = None) -> dict[str, Any]:
        return self.store_chatterbox_reference(conversion_id, source, consent_confirmed=consent_confirmed, consent_evidence=consent_evidence, consent_recorded_at=consent_recorded_at, replace=True)

    def delete_chatterbox_reference(self, conversion_id: str | uuid.UUID) -> bool:
        descriptor_path = self.chatterbox_reference_descriptor_path(conversion_id)
        reference_path = self.chatterbox_reference_path(conversion_id)
        if not descriptor_path.exists() and not reference_path.exists() and not descriptor_path.is_symlink() and not reference_path.is_symlink():
            return False
        unsafe = False
        removed = False
        for target in (descriptor_path, reference_path):
            if target.exists() or target.is_symlink():
                try:
                    _validate_regular_manifest_file(target, "Chatterbox reference artifact")
                except ManifestError as exc:
                    unsafe = True
                    continue
                target.unlink()
                removed = True
        if removed:
            self._invalidate_chatterbox_generation(conversion_id)
        if unsafe:
            raise UnsafePathError("Chatterbox reference artifact is unsafe")
        return removed

    def cancel_marker_path(self, conversion_id: str | uuid.UUID) -> Path:
        return self.conversion_path(conversion_id) / "cancel.request"

    def request_cancel(self, conversion_id: str | uuid.UUID) -> Path:
        marker = self.cancel_marker_path(conversion_id)
        try:
            info = marker.lstat()
        except FileNotFoundError:
            info = None
        except OSError as exc:
            raise UnsafePathError("cancel marker is unavailable") from exc
        if info is not None and (stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode)):
            raise UnsafePathError("cancel marker is unsafe")
        fd, temporary_name = tempfile.mkstemp(prefix=".cancel.", dir=marker.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write("cancel\n")
                handle.flush(); os.fsync(handle.fileno())
            _replace_with_retries(temporary, marker)
        finally:
            temporary.unlink(missing_ok=True)
        return marker

    def cancellation_requested(self, conversion_id: str | uuid.UUID) -> bool:
        marker = self.cancel_marker_path(conversion_id)
        try:
            info = marker.lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise UnsafePathError("cancel marker is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("cancel marker is unsafe")
        return True

    def clear_cancel_request(self, conversion_id: str | uuid.UUID) -> None:
        marker = self.cancel_marker_path(conversion_id)
        if not marker.exists() and not marker.is_symlink():
            return
        try:
            info = marker.stat(follow_symlinks=False)
        except OSError as exc:
            raise UnsafePathError("cancel marker is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("cancel marker is unsafe")
        marker.unlink()

    def voice_analysis_cancel_marker_path(self, conversion_id: str | uuid.UUID) -> Path:
        return self.conversion_path(conversion_id) / "voice-analysis.cancel"

    def request_voice_analysis_cancel(self, conversion_id: str | uuid.UUID) -> Path:
        marker = self.voice_analysis_cancel_marker_path(conversion_id)
        try:
            info = marker.lstat()
        except FileNotFoundError:
            info = None
        except OSError as exc:
            raise UnsafePathError("voice analysis cancel marker is unavailable") from exc
        if info is not None and (stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode)):
            raise UnsafePathError("voice analysis cancel marker is unsafe")
        fd, temporary_name = tempfile.mkstemp(prefix=".voice-analysis.cancel.", dir=marker.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write("cancel\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_retries(temporary, marker)
        finally:
            temporary.unlink(missing_ok=True)
        return marker

    def voice_analysis_cancellation_requested(self, conversion_id: str | uuid.UUID) -> bool:
        marker = self.voice_analysis_cancel_marker_path(conversion_id)
        try:
            info = marker.lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise UnsafePathError("voice analysis cancel marker is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("voice analysis cancel marker is unsafe")
        return True

    def clear_voice_analysis_cancel_request(self, conversion_id: str | uuid.UUID) -> None:
        marker = self.voice_analysis_cancel_marker_path(conversion_id)
        if not marker.exists() and not marker.is_symlink():
            return
        try:
            info = marker.stat(follow_symlinks=False)
        except OSError as exc:
            raise UnsafePathError("voice analysis cancel marker is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("voice analysis cancel marker is unsafe")
        marker.unlink()

    def configure_generation(self, conversion_id: str | uuid.UUID, *, tts: dict[str, Any], total_chunks: int) -> dict[str, Any]:
        """Upgrade a planned manifest to the current generation schema."""

        if type(total_chunks) is not int or total_chunks < 0:
            raise ManifestError("total_chunks must be non-negative")
        if isinstance(tts, dict) and tts.get("engine") == "chatterbox":
            if tts.get("voice") == CHATTERBOX_BUILTIN_VOICE:
                settings = tts.get("settings") if isinstance(tts.get("settings"), dict) else {}
                if (tts.get("model") != CHATTERBOX_NANO_MODEL or tts.get("model_revision") != CHATTERBOX_SOURCE_COMMIT or tts.get("model_checksum") != "unrecorded" or tts.get("voice_version") != "bundled" or tts.get("voice_checksum") != "unrecorded" or tts.get("sample_rate") != CHATTERBOX_SAMPLE_RATE or tts.get("speed") != 1.0 or tts.get("chunk_cap") != 300 or "reference_descriptor_sha256" in settings):
                    raise ManifestError("Chatterbox built-in voice identity is invalid")
            else:
                try:
                    reference = self.load_chatterbox_reference(conversion_id)
                except (ManifestError, UnsafePathError) as exc:
                    raise ManifestError("Chatterbox generation requires a valid reference") from exc
                if (tts.get("voice") != "reference-wav" or tts.get("model") != reference.descriptor["model"] or tts.get("model_revision") != reference.descriptor["model_revision"] or tts.get("model_checksum") != reference.descriptor["model_checksum"] or tts.get("voice_version") != reference.descriptor["voice_version"] or tts.get("voice_checksum") != reference.descriptor["voice_checksum"] or tts.get("sample_rate") != CHATTERBOX_SAMPLE_RATE or tts.get("speed") != 1.0 or tts.get("chunk_cap") != 300):
                    raise ManifestError("Chatterbox reference identity does not match generation settings")
                tts = dict(tts)
                settings = dict(tts.get("settings") or {})
                captured_descriptor = settings.get("reference_descriptor_sha256")
                if captured_descriptor is not None and captured_descriptor != reference.descriptor["descriptor_sha256"]:
                    raise ManifestError("Chatterbox reference descriptor does not match generation settings")
                settings["reference_descriptor_sha256"] = reference.descriptor["descriptor_sha256"]
                tts["settings"] = settings
        current = self.read_job(conversion_id)
        if current["schema_version"] == INTERACTIVE_GENERATION_SCHEMA_VERSION:
            raise ManifestError("interactive generation requires configure_interactive_generation")
        if current["schema_version"] == GENERATION_SCHEMA_VERSION:
            if current["status"] == "completed" and current.get("stage") == "synthesis_complete" and current.get("output") is None:
                if current["tts"] != tts or current["total_chunks"] != total_chunks:
                    raise ManifestError("synthesis-complete settings cannot change")
                return current
            if current["status"] == "completed":
                raise ManifestError("completed generation cannot be resumed")
            if current["status"] in {"synthesizing", "cancelling"}:
                if current["tts"] != tts or current["total_chunks"] != total_chunks:
                    raise ManifestError("active generation settings cannot change")
            elif current["tts"] != tts:
                current["tts"] = dict(tts)
                current["total_chunks"] = total_chunks
                current["status"] = "planned"
                current["stage"] = "chapter_review"
                current["error"] = None
                current["last_safe_error"] = None
                current["worker"] = None
                current["completed_chunks"] = []
                current["progress"] = {"completed": 0, "current": 0, "total": total_chunks}
                current["output"] = None
                current["updated_at"] = _timestamp()
                current = validate_generation_manifest(current)
                atomic_write_json(self.job_path(conversion_id), current)
                self.clear_cancel_request(conversion_id)
            elif current["total_chunks"] != total_chunks:
                raise ManifestError("generation chunk plan does not match settings")
            if current["status"] not in {"synthesizing", "cancelling"}:
                self.clear_cancel_request(conversion_id)
            return current
        if current["status"] not in {"planned", "analyzed", "pending", "cancelled", "failed", "synthesizing", "cancelling"}:
            raise ManifestError("job is not ready for generation")
        upgraded = _upgrade_generation_manifest(current, tts=tts, total_chunks=total_chunks)
        atomic_write_json(self.job_path(conversion_id), upgraded)
        self.chunks_path(conversion_id)
        self.clear_cancel_request(conversion_id)
        return upgraded

    def configure_interactive_generation(
        self,
        conversion_id: str | uuid.UUID,
        *,
        tts: dict[str, Any],
        total_chunks: int,
        voice_registry_revision: str,
    ) -> dict[str, Any]:
        """Bind an approved voice plan and current analysis to schema-v5 generation."""

        if type(total_chunks) is not int or total_chunks < 0:
            raise ManifestError("total_chunks must be non-negative")
        _validate_sha(voice_registry_revision, "voice_registry_revision")
        current = self.read_job(conversion_id)
        plan = self.load_voice_plan(conversion_id)
        if plan["approval"]["state"] != "approved":
            raise ManifestError("interactive generation requires an approved voice plan")
        analysis = self.load_speaker_analysis(conversion_id)
        cast_voice_ids: list[str] = []
        for cast_entry in plan["cast"]:
            voice_id = cast_entry["voice_id"]
            if voice_id not in cast_voice_ids:
                cast_voice_ids.append(voice_id)
        facts = {
            "mode": "interactive_voices",
            "tts": tts,
            "total_chunks": total_chunks,
            "voice_plan_sha256": plan["canonical_artifact_sha256"],
            "voice_plan_revision": plan["revision"],
            "speaker_analysis_sha256": analysis["canonical_artifact_sha256"],
            "cast_voice_ids": cast_voice_ids,
            "voice_registry_revision": voice_registry_revision,
        }
        if current.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
            facts_identical = all(current.get(key) == value for key, value in facts.items())
            if current["status"] == "completed":
                if current.get("stage") == "synthesis_complete" and current.get("output") is None and facts_identical:
                    return current
                raise ManifestError("completed interactive generation cannot be reconfigured")
            if facts_identical:
                if current["status"] not in {"synthesizing", "cancelling"}:
                    self.clear_cancel_request(conversion_id)
                return current
            if current["status"] in {"synthesizing", "cancelling"}:
                raise ManifestError("active interactive generation settings cannot change")
            if current["status"] not in {"planned", "analyzed", "pending", "cancelled", "failed"}:
                raise ManifestError("job is not ready for interactive generation")
        elif current["status"] not in {"planned", "analyzed", "pending", "cancelled", "failed", "synthesizing", "cancelling"}:
            raise ManifestError("job is not ready for interactive generation")

        candidates: list[dict[str, Any]] = []
        if current.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION and current["status"] in {"planned", "analyzed", "pending", "cancelled", "failed"}:
            candidates = [
                dict(record)
                for record in current["completed_chunks"]
                if record["global_index"] < total_chunks
            ]
        configured = {
            **current,
            "schema_version": INTERACTIVE_GENERATION_SCHEMA_VERSION,
            **facts,
            "status": "planned",
            "stage": "chapter_review",
            "error": None,
            "last_safe_error": None,
            "completed_chunks": candidates,
            "progress": {"completed": len(candidates), "current": 0, "total": total_chunks},
            "worker": None,
            "output": None,
        }
        validated = validate_interactive_generation_manifest(configured)
        atomic_write_json(self.job_path(conversion_id), validated)
        self.chunks_path(conversion_id)
        self.clear_cancel_request(conversion_id)
        return validated

    def update_generation(self, conversion_id: str | uuid.UUID, **updates: Any) -> dict[str, Any]:
        current = self.read_job(conversion_id)
        if current["schema_version"] not in {GENERATION_SCHEMA_VERSION, INTERACTIVE_GENERATION_SCHEMA_VERSION}:
            raise ManifestError("job is not a generation manifest")
        current.update(updates)
        current["updated_at"] = _timestamp()
        validated = (
            validate_interactive_generation_manifest(current)
            if current["schema_version"] == INTERACTIVE_GENERATION_SCHEMA_VERSION
            else validate_generation_manifest(current)
        )
        atomic_write_json(self.job_path(conversion_id), validated)
        return validated

    def load_analysis(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Load a trustworthy analysis artifact for the current source PDF."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        source = directory / "source.pdf"
        _validate_regular_manifest_file(source, "source.pdf")
        source_hash = _hash_file(source)
        if source_hash != job["source_pdf_sha256"]:
            raise ManifestError("source PDF hash mismatch")
        artifact_path = directory / "analysis.json"
        artifact = read_json_manifest(artifact_path, _validate_analysis_manifest)
        if artifact.get("source_pdf_sha256") != source_hash:
            raise ManifestError("analysis source hash mismatch")
        return artifact

    def load_cleaned_artifacts(self, conversion_id: str | uuid.UUID) -> tuple[str, list[dict[str, Any]]]:
        """Load cleaned text and mapping after verifying their job binding."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        cleaned_hash = job["cleaned_text_sha256"]
        if cleaned_hash is None:
            raise ManifestError("cleaned text is not persisted")
        cleaned_path = directory / "cleaned.txt"
        map_path = directory / "cleaned-map.json"
        _validate_regular_manifest_file(cleaned_path, "cleaned text")
        _validate_regular_manifest_file(map_path, "cleaned map")
        try:
            cleaned_text = cleaned_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ManifestError("malformed cleaned text") from exc
        if hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() != cleaned_hash:
            raise ManifestError("cleaned text hash mismatch")
        try:
            cleaned_map = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError("malformed cleaned map JSON") from exc
        return cleaned_text, _validate_cleaned_map(cleaned_map, cleaned_text)

    def persist_chapter_plan(self, conversion_id: str | uuid.UUID, plan: dict[str, Any]) -> dict[str, Any]:
        """Persist a validated chapter plan and bind it to the job manifest."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        cleaned_hash = job["cleaned_text_sha256"]
        if cleaned_hash is None:
            raise ManifestError("chapter plan requires cleaned text")
        _validate_chapter_plan(plan, cleaned_hash)
        canonical = _canonical_json_text(plan).encode("utf-8")
        plan_hash = hashlib.sha256(canonical).hexdigest()
        atomic_write_json(directory / "chapters.json", plan)
        return self.update_job(
            conversion_id,
            status="planned",
            stage="chapter_review",
            chapter_plan_sha256=plan_hash,
        )

    def load_chapter_plan(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Load and verify the persisted chapter plan artifact."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        expected_hash = job["chapter_plan_sha256"]
        if expected_hash is None:
            raise ManifestError("chapter plan is not persisted")
        artifact_path = directory / "chapters.json"
        _validate_regular_manifest_file(artifact_path, "chapter plan")
        try:
            raw_bytes = artifact_path.read_bytes()
            plan = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError("malformed chapter plan JSON") from exc
        if hashlib.sha256(_canonical_json_text(plan).encode("utf-8")).hexdigest() != expected_hash:
            raise ManifestError("chapter plan hash mismatch")
        _validate_chapter_plan(plan, job["cleaned_text_sha256"])
        return plan

    def persist_voice_plan(self, conversion_id: str | uuid.UUID, plan: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist a draft or approved Interactive Voices plan."""

        directory = self.conversion_path(conversion_id)
        target = directory / "voice-plan.json"
        if target.exists() or target.is_symlink():
            _validate_regular_manifest_file(target, "voice plan")
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        try:
            validated = validate_voice_plan(
                plan,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except VoicePlanError as exc:
            raise ManifestError("invalid voice plan artifact") from exc
        atomic_write_json(target, validated)
        return validated

    def load_voice_plan(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Load and validate a draft or approved Interactive Voices plan and bindings."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        target = directory / "voice-plan.json"
        _validate_regular_manifest_file(target, "voice plan")
        try:
            artifact = json.loads(target.read_bytes().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError("malformed voice plan JSON") from exc
        try:
            return validate_voice_plan(
                artifact,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except VoicePlanError as exc:
            raise ManifestError("invalid voice plan artifact") from exc

    def persist_speaker_analysis(self, conversion_id: str | uuid.UUID, analysis: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist replaceable machine-only analysis."""

        directory = self.conversion_path(conversion_id)
        target = directory / "speaker-analysis.json"
        if target.exists() or target.is_symlink():
            _validate_regular_manifest_file(target, "speaker analysis")
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        try:
            validated = validate_speaker_analysis(
                analysis,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except SpeakerAnalysisError as exc:
            raise ManifestError("invalid speaker analysis artifact") from exc
        atomic_write_json(target, validated)
        return validated

    def load_speaker_analysis(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Load and validate current replaceable machine-only analysis."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        target = directory / "speaker-analysis.json"
        _validate_regular_manifest_file(target, "speaker analysis")
        if target.stat(follow_symlinks=False).st_size > MAX_ARTIFACT_BYTES:
            raise ManifestError("speaker analysis artifact too large")
        try:
            artifact = json.loads(target.read_bytes().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError("malformed speaker analysis JSON") from exc
        try:
            return validate_speaker_analysis(
                artifact,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except SpeakerAnalysisError as exc:
            raise ManifestError("invalid speaker analysis artifact") from exc

    def persist_voice_analysis_status(self, conversion_id: str | uuid.UUID, status: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist the current voice-analysis status."""

        directory = self.conversion_path(conversion_id)
        target = directory / "voice-analysis-status.json"
        if target.exists() or target.is_symlink():
            _validate_regular_manifest_file(target, "voice-analysis status")
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        try:
            validated = validate_voice_analysis_status(
                status,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except VoiceAnalysisError as exc:
            raise ManifestError("invalid voice-analysis status") from exc
        atomic_write_json(target, validated)
        return validated

    def load_voice_analysis_status(self, conversion_id: str | uuid.UUID) -> dict[str, Any]:
        """Load and validate the current voice-analysis status."""

        directory = self.conversion_path(conversion_id)
        job = self.read_job(conversion_id)
        self.load_analysis(conversion_id)
        cleaned_text, _ = self.load_cleaned_artifacts(conversion_id)
        chapter_plan = self.load_chapter_plan(conversion_id)
        target = directory / "voice-analysis-status.json"
        _validate_regular_manifest_file(target, "voice-analysis status")
        if target.stat(follow_symlinks=False).st_size > MAX_VOICE_ANALYSIS_STATUS_BYTES:
            raise ManifestError("voice-analysis status too large")
        try:
            artifact = json.loads(target.read_bytes().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError("malformed voice-analysis status JSON") from exc
        try:
            return validate_voice_analysis_status(
                artifact,
                cleaned_text,
                chapter_plan,
                expected_source_pdf_sha256=job["source_pdf_sha256"],
                expected_chapter_plan_sha256=job["chapter_plan_sha256"],
            )
        except VoiceAnalysisError as exc:
            raise ManifestError("invalid voice-analysis status") from exc

    def update_job(self, conversion_id: str | uuid.UUID, **updates: Any) -> dict[str, Any]:
        """Atomically update strict job fields without changing its identity."""

        current = self.read_job(conversion_id)
        current.update(updates)
        current["updated_at"] = _timestamp()
        if current.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
            validated = validate_interactive_generation_manifest(current)
        elif current.get("schema_version") == GENERATION_SCHEMA_VERSION:
            validated = validate_generation_manifest(current)
        else:
            validated = validate_job_manifest(current)
        atomic_write_json(self.job_path(conversion_id), validated)
        return validated

    def persist_analysis(self, conversion_id: str | uuid.UUID, analysis: dict[str, Any]) -> dict[str, Any]:
        """Persist cleaned text, mapping, and compact analysis evidence atomically."""

        directory = self.conversion_path(conversion_id)
        cleaned_text = analysis.get("cleaned_text")
        cleaned_map = analysis.get("cleaned_map")
        if not isinstance(cleaned_text, str) or not isinstance(cleaned_map, list):
            raise WorkspaceError("analysis artifacts are incomplete")
        chapter_plan_path = directory / "chapters.json"
        if chapter_plan_path.exists() or chapter_plan_path.is_symlink():
            try:
                chapter_plan_info = chapter_plan_path.stat(follow_symlinks=False)
            except OSError as exc:
                raise UnsafePathError("chapter plan artifact could not be inspected") from exc
            if (
                stat.S_ISLNK(chapter_plan_info.st_mode)
                or _is_reparse(chapter_plan_info)
                or not stat.S_ISREG(chapter_plan_info.st_mode)
            ):
                raise UnsafePathError("refusing to remove unsafe chapter plan artifact")
        atomic_write_text(directory / "cleaned.txt", cleaned_text)
        atomic_write_json(directory / "cleaned-map.json", cleaned_map)
        artifact = {key: value for key, value in analysis.items() if key not in {"cleaned_text", "cleaned_map"}}
        atomic_write_json(directory / "analysis.json", artifact)
        if chapter_plan_path.exists():
            chapter_plan_path.unlink()
        return self.update_job(
            conversion_id,
            status="analyzed",
            stage="review",
            cleaned_text_sha256=hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest(),
            chapter_plan_sha256=None,
            warnings=list(analysis.get("warnings", [])),
            error=None,
        )

    def create_conversion(
        self,
        source_pdf: Path,
        *,
        original_display_filename: str | None = None,
        conversion_id: str | uuid.UUID | None = None,
        status: str = "pending",
        stage: str = "workspace",
    ) -> dict[str, Any]:
        """Create a conversion, copy its source, and mark it active."""

        if self.active_path.exists() or self.active_path.is_symlink():
            raise WorkspaceError("an active conversion already exists; inspect or delete it explicitly")
        normalized_id = self._conversion_id(conversion_id)
        directory = self.conversion_path(normalized_id)
        if directory.exists():
            raise WorkspaceError("conversion directory already exists")
        directory.mkdir(parents=False)
        display_name = Path(source_pdf).name if original_display_filename is None else original_display_filename
        created = _timestamp()
        try:
            source_hash, _ = copy_source_pdf(Path(source_pdf), directory / "source.pdf")
            manifest = {
                "schema_version": JOB_SCHEMA_VERSION,
                "conversion_id": normalized_id,
                "original_display_filename": display_name,
                "source_pdf_sha256": source_hash,
                "status": status,
                "stage": stage,
                "created_at": created,
                "updated_at": _timestamp(),
                "cleaned_text_sha256": None,
                "chapter_plan_sha256": None,
                "warnings": [],
                "error": None,
            }
            validate_job_manifest(manifest)
            atomic_write_json(directory / "job.json", manifest)
            self._write_active(normalized_id)
        except Exception:
            # A failed create is not a startup cleanup operation; removing the
            # directory here only rolls back this operation before it is exposed.
            self._remove_validated_tree(directory)
            raise
        return manifest

    def inspect_startup(self) -> StartupInspection:
        """Inspect active state without deleting or repairing any files."""

        if self.active_path.is_symlink() or not self.active_path.exists():
            if self.active_path.is_symlink():
                return StartupInspection("invalid", reason="active manifest must not be a symlink")
            return StartupInspection("no_active")
        conversion_id: str | None = None
        try:
            active = read_json_manifest(self.active_path, validate_active_manifest)
            conversion_id = active["conversion_id"]
            directory = self.conversion_path(conversion_id)
            if not directory.exists() or not directory.is_dir() or directory.is_symlink():
                raise ManifestError("active conversion directory is missing or unsafe")
            job = self.read_job(conversion_id)
            if job["conversion_id"] != conversion_id:
                raise ManifestError("job conversion_id does not match active manifest")
            source = directory / "source.pdf"
            _validate_regular_manifest_file(source, "source.pdf")
            digest = _hash_file(source)
            if digest != job["source_pdf_sha256"]:
                raise ManifestError("source PDF hash mismatch")
            cleaned_hash = job["cleaned_text_sha256"]
            if cleaned_hash is not None:
                cleaned = directory / "cleaned.txt"
                _validate_regular_manifest_file(cleaned, "cleaned.txt")
                if _hash_file(cleaned) != cleaned_hash:
                    raise ManifestError("cleaned text hash mismatch")
            if job["chapter_plan_sha256"] is not None:
                self.load_chapter_plan(conversion_id)
            output = job.get("output")
            if output is not None:
                output_path = Path(output["path"])
                _validate_regular_manifest_file(output_path, "published output")
                if output_path.stat().st_size != output["size_bytes"] or _hash_file(output_path) != output["sha256"]:
                    raise ManifestError("published output hash or size mismatch")
            return StartupInspection("resumable", conversion_id, job)
        except (OSError, WorkspaceError, json.JSONDecodeError) as exc:
            return StartupInspection("invalid", conversion_id=conversion_id, reason=str(exc))

    def _validate_tree_for_delete(self, path: Path) -> None:
        try:
            info = path.lstat()
        except OSError as exc:
            raise UnsafePathError("conversion directory is missing") from exc
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise UnsafePathError("refusing symlinked or non-directory conversion")
        with os.scandir(path) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                entry_info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_info.st_mode) or _is_reparse(entry_info):
                    raise UnsafePathError("refusing cleanup through a symlink or reparse point")
                if stat.S_ISDIR(entry_info.st_mode):
                    self._validate_tree_for_delete(entry_path)
                elif not stat.S_ISREG(entry_info.st_mode):
                    raise UnsafePathError("refusing cleanup of a special file")

    def _remove_validated_tree(self, path: Path) -> None:
        self._validate_tree_for_delete(path)
        with os.scandir(path) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            info = child.lstat()
            if stat.S_ISDIR(info.st_mode):
                self._remove_validated_tree(child)
            else:
                child.unlink()
        path.rmdir()

    def delete_conversion(self, conversion_id: str | uuid.UUID) -> bool:
        """Explicitly delete one validated conversion and nothing else."""

        directory = self.conversion_path(conversion_id)
        if not directory.exists():
            return False
        self._remove_validated_tree(directory)
        try:
            active = read_json_manifest(self.active_path, validate_active_manifest)
        except ManifestError:
            return True
        if active["conversion_id"] == self._conversion_id(conversion_id):
            self.active_path.unlink()
        return True

    def delete_active_state(self) -> bool:
        """Explicitly reset active state, deleting only a validated conversion."""

        try:
            mode = self.active_path.lstat().st_mode
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise UnsafePathError("active manifest could not be inspected") from exc
        if stat.S_ISLNK(mode) or _is_reparse(self.active_path.stat(follow_symlinks=False)) or not stat.S_ISREG(mode):
            raise UnsafePathError("active manifest must be a regular file")
        try:
            active = read_json_manifest(self.active_path, validate_active_manifest)
        except ManifestError:
            self.active_path.unlink()
            return True
        self.delete_conversion(active["conversion_id"])
        if self.active_path.exists() and not self.active_path.is_symlink():
            self.active_path.unlink()
        return True


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


__all__ = [
    "ACTIVE_SCHEMA_VERSION",
    "COPY_CHUNK_SIZE",
    "JOB_SCHEMA_VERSION",
    "LEGACY_GENERATION_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION",
    "INTERACTIVE_GENERATION_SCHEMA_VERSION",
    "ManifestError",
    "StartupInspection",
    "UnsafePathError",
    "Workspace",
    "WorkspaceError",
    "atomic_write_json",
    "atomic_write_text",
    "copy_source_pdf",
    "read_json_manifest",
    "validate_active_manifest",
    "validate_job_manifest",
    "validate_generation_manifest",
    "validate_generation_manifest_v3",
    "validate_generation_manifest_v5",
    "validate_interactive_generation_manifest",
]

"""Workspace-owned Chatterbox Nano reference-WAV descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from .audio import WavInfo, validate_wav

REFERENCE_DESCRIPTOR_SCHEMA_VERSION = 1
REFERENCE_DESCRIPTOR_FILENAME = "chatterbox-reference.json"
REFERENCE_WAV_FILENAME = "chatterbox-reference.wav"
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
MAX_REFERENCE_SECONDS = 60.0
MIN_REFERENCE_SECONDS = 5.0
RECOMMENDED_REFERENCE_SECONDS = 10.0
_DESCRIPTOR_FIELDS = {
    "schema_version", "engine", "model", "model_revision", "model_checksum",
    "voice", "voice_version", "voice_checksum", "sample_rate", "speed",
    "chunk_cap", "reference_sha256", "file_bytes", "duration_seconds",
    "consent_confirmed", "consent_evidence", "consent_recorded_at", "created_at", "updated_at",
    "descriptor_sha256",
}


@dataclass(frozen=True)
class ReferenceArtifact:
    path: Path
    descriptor: dict[str, Any]
    wav: WavInfo


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, message: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(message) from exc
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or (flag and getattr(info, "st_file_attributes", 0) & flag) or not stat.S_ISREG(info.st_mode):
        raise ValueError(message)


def validate_reference_file(path: str | Path) -> tuple[WavInfo, str]:
    """Validate a reference file and return its WAV facts and exact hash."""

    source = Path(path)
    try:
        _regular(source, "reference WAV is missing or unsafe")
        if source.stat(follow_symlinks=False).st_size <= 0 or source.stat(follow_symlinks=False).st_size > MAX_REFERENCE_BYTES:
            raise ValueError("reference WAV size is invalid")
        wav = validate_wav(source)
        if wav.duration_seconds <= MIN_REFERENCE_SECONDS or wav.duration_seconds > MAX_REFERENCE_SECONDS:
            raise ValueError("reference WAV duration must be longer than five seconds and no more than sixty seconds")
        return wav, _sha256(source)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("reference WAV is invalid or unsafe") from exc


def copy_reference_file(source: str | Path, destination: str | Path, *, chunk_size: int = 1024 * 1024) -> None:
    """Stream a source into a same-directory temporary file with a hard size cap."""

    source_path, destination_path = Path(source), Path(destination)
    try:
        _regular(source_path, "reference WAV is missing or unsafe")
        if source_path.stat(follow_symlinks=False).st_size > MAX_REFERENCE_BYTES:
            raise ValueError("reference WAV size is invalid")
        count = 0
        # The workspace creates this destination with mkstemp before handing it
        # here. Recheck it as a regular non-link file, then write that exact
        # inode without following a replacement symlink.
        _regular(destination_path, "reference WAV temporary is unsafe")
        with source_path.open("rb") as input_file, destination_path.open("r+b") as output_file:
            while block := input_file.read(chunk_size):
                count += len(block)
                if count > MAX_REFERENCE_BYTES:
                    raise ValueError("reference WAV size is invalid")
                output_file.write(block)
            output_file.flush()
            os.fsync(output_file.fileno())
    except (OSError, TypeError, ValueError) as exc:
        destination_path.unlink(missing_ok=True)
        raise ValueError("reference WAV copy failed") from exc


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def descriptor_digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "descriptor_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def validate_reference_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _DESCRIPTOR_FIELDS:
        raise ValueError("reference descriptor schema mismatch")
    if value["schema_version"] != REFERENCE_DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError("unsupported reference descriptor schema")
    for field in ("engine", "model", "model_revision", "model_checksum", "voice", "voice_version", "voice_checksum", "consent_evidence", "consent_recorded_at", "created_at", "updated_at"):
        if not isinstance(value[field], str) or not value[field] or "\x00" in value[field]:
            raise ValueError("reference descriptor identity is invalid")
    if value["engine"] != "chatterbox" or value["voice"] != "reference-wav":
        raise ValueError("reference descriptor engine or voice is invalid")
    if not isinstance(value["reference_sha256"], str) or len(value["reference_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in value["reference_sha256"]):
        raise ValueError("reference descriptor hash is invalid")
    if not isinstance(value["descriptor_sha256"], str) or value["descriptor_sha256"] != descriptor_digest(value):
        raise ValueError("reference descriptor digest is invalid")
    if value["consent_confirmed"] is not True:
        raise ValueError("reference consent is required")
    if type(value["sample_rate"]) is not int or value["sample_rate"] <= 0 or value["speed"] != 1.0 or value["chunk_cap"] != 300:
        raise ValueError("reference descriptor settings are invalid")
    if type(value["file_bytes"]) is not int or not 0 < value["file_bytes"] <= MAX_REFERENCE_BYTES:
        raise ValueError("reference descriptor file size is invalid")
    if type(value["duration_seconds"]) not in {int, float} or not math.isfinite(float(value["duration_seconds"])) or not MIN_REFERENCE_SECONDS < value["duration_seconds"] <= MAX_REFERENCE_SECONDS:
        raise ValueError("reference descriptor duration is invalid")
    return dict(value)


def build_reference_descriptor(*, wav: WavInfo, reference_sha256: str, engine: str, model: str, model_revision: str, model_checksum: str, voice: str, voice_version: str, speed: float, chunk_cap: int, consent_confirmed: bool, consent_evidence: str, consent_recorded_at: str, created_at: str, updated_at: str) -> dict[str, Any]:
    descriptor = {
        "schema_version": REFERENCE_DESCRIPTOR_SCHEMA_VERSION,
        "engine": engine, "model": model, "model_revision": model_revision, "model_checksum": model_checksum,
        "voice": voice, "voice_version": voice_version, "voice_checksum": reference_sha256,
        "sample_rate": wav.sample_rate, "speed": speed, "chunk_cap": chunk_cap,
        "reference_sha256": reference_sha256, "file_bytes": wav.file_bytes, "duration_seconds": wav.duration_seconds,
        "consent_confirmed": consent_confirmed, "consent_evidence": consent_evidence, "consent_recorded_at": consent_recorded_at,
        "created_at": created_at, "updated_at": updated_at,
    }
    descriptor["descriptor_sha256"] = descriptor_digest(descriptor)
    return validate_reference_descriptor(descriptor)


def public_reference_status(descriptor: dict[str, Any] | None) -> dict[str, Any]:
    if descriptor is None:
        return {"available": False, "recommended_duration_seconds": RECOMMENDED_REFERENCE_SECONDS}
    value = validate_reference_descriptor(descriptor)
    return {
        "available": True,
        "engine": value["engine"], "model": value["model"], "model_revision": value["model_revision"],
        "voice": value["voice"], "voice_checksum": value["voice_checksum"], "reference_sha256": value["reference_sha256"],
        "sample_rate": value["sample_rate"], "file_bytes": value["file_bytes"], "duration_seconds": value["duration_seconds"],
        "recommended_duration_seconds": RECOMMENDED_REFERENCE_SECONDS,
        "consent_confirmed": True, "updated_at": value["updated_at"],
    }


__all__ = [
    "MAX_REFERENCE_BYTES", "MAX_REFERENCE_SECONDS", "MIN_REFERENCE_SECONDS", "RECOMMENDED_REFERENCE_SECONDS",
    "REFERENCE_DESCRIPTOR_FILENAME", "REFERENCE_DESCRIPTOR_SCHEMA_VERSION", "REFERENCE_WAV_FILENAME",
    "ReferenceArtifact", "build_reference_descriptor", "copy_reference_file", "descriptor_digest", "public_reference_status",
    "validate_reference_descriptor", "validate_reference_file",
]

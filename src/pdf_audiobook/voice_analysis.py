"""Strict, durable status artifacts for Interactive Voices analysis runs."""

from __future__ import annotations

from datetime import datetime
import hashlib
import re
import uuid
from typing import Any

from .voice_plan import VoicePlanError, canonical_json_bytes, verify_canonical_artifact_hash


MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_TEXT = 512
MAX_WARNING_BYTES = 8 * 1024
MAX_WARNINGS_BYTES = 8 * 1024 * 1024
MAX_ERROR_CODE = 128
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_FIELDS = {
    "schema_version", "artifact", "analysis_id", "revision", "source_pdf_sha256",
    "cleaned_text_sha256", "chapter_plan_sha256", "chapter_plan_schema_version",
    "analyzer", "status", "stage", "progress", "cancel_requested", "warnings",
    "error", "started_at", "updated_at", "finished_at", "canonical_artifact_sha256",
}
_STATES = {
    "queued": "queued",
    "running": {"preparing", "analyzing", "validating", "persisting"},
    "completed": "completed",
    "cancelled": "cancelled",
    "failed": "failed",
}


class VoiceAnalysisError(ValueError):
    """Stable validation failure for a voice-analysis status artifact."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> VoiceAnalysisError:
    return VoiceAnalysisError(code, message, details=details)


def _text(value: Any, name: str, maximum: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise _fail("INVALID_" + name.upper(), f"{name} is invalid")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _fail("INVALID_" + name.upper(), f"{name} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: Any, code: str, message: str) -> int:
    if type(value) is not int or value <= 0:
        raise _fail(code, message)
    return value


def _nonnegative_int(value: Any, code: str, message: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(code, message)
    return value


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail("INVALID_" + name.upper(), f"{name} schema mismatch")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise _fail("INVALID_TIMESTAMP", f"{name} must be timezone-aware ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("INVALID_TIMESTAMP", f"{name} must be timezone-aware ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _fail("INVALID_TIMESTAMP", f"{name} must be timezone-aware ISO-8601")
    return parsed


def _analyzer(value: Any) -> None:
    analyzer = _object(value, {"id", "version", "model_hash"}, "analyzer")
    _text(analyzer["id"], "analyzer_id")
    _text(analyzer["version"], "analyzer_version")
    if analyzer["model_hash"] is not None:
        _sha(analyzer["model_hash"], "model_hash")


def _error(value: Any) -> None:
    error = _object(value, {"code", "message"}, "error")
    if not isinstance(error["code"], str) or not _ERROR_CODE.fullmatch(error["code"]):
        raise _fail("INVALID_ERROR", "error code is invalid")
    _text(error["message"], "error_message", MAX_TEXT)


def validate_voice_analysis_status(
    artifact: Any,
    cleaned_text: str,
    chapter_plan: Any,
    *,
    expected_source_pdf_sha256: str,
    expected_chapter_plan_sha256: str,
) -> dict[str, Any]:
    """Validate and return a status artifact without repairing it."""

    if not isinstance(artifact, dict) or set(artifact) != _FIELDS:
        raise _fail("INVALID_STATUS", "voice-analysis status schema mismatch")
    try:
        if len(canonical_json_bytes(artifact)) > MAX_ARTIFACT_BYTES:
            raise _fail("STATUS_TOO_LARGE", "voice-analysis status exceeds the size limit")
    except VoicePlanError as exc:
        raise VoiceAnalysisError(exc.code, exc.message, details=exc.details) from exc
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise _fail("INVALID_SCHEMA_VERSION", "status schema_version must be 1")
    if artifact["artifact"] != "voice-analysis-status":
        raise _fail("INVALID_ARTIFACT_TYPE", "artifact must be voice-analysis-status")
    analysis_id = artifact["analysis_id"]
    if not isinstance(analysis_id, str):
        raise _fail("INVALID_ANALYSIS_ID", "analysis_id must be a canonical UUID")
    try:
        if str(uuid.UUID(analysis_id)) != analysis_id:
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise _fail("INVALID_ANALYSIS_ID", "analysis_id must be a canonical UUID") from exc
    _positive_int(artifact["revision"], "INVALID_REVISION", "revision must be positive")
    source_hash = _sha(artifact["source_pdf_sha256"], "source_pdf_sha256")
    cleaned_hash = _sha(artifact["cleaned_text_sha256"], "cleaned_text_sha256")
    chapter_hash = _sha(artifact["chapter_plan_sha256"], "chapter_plan_sha256")
    expected_source = _sha(expected_source_pdf_sha256, "expected_source_pdf_sha256")
    expected_chapter = _sha(expected_chapter_plan_sha256, "expected_chapter_plan_sha256")
    if source_hash != expected_source:
        raise _fail("SOURCE_HASH_MISMATCH", "status source hash does not match current source")
    if chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "status chapter-plan hash does not match current plan")
    try:
        current_chapter_hash = hashlib.sha256(canonical_json_bytes(chapter_plan)).hexdigest()
    except VoicePlanError as exc:
        raise _fail("INVALID_CHAPTER_PLAN", "current chapter plan is invalid") from exc
    if current_chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "current chapter plan hash does not match expected binding")
    if type(artifact["chapter_plan_schema_version"]) is not int or artifact["chapter_plan_schema_version"] != 1:
        raise _fail("CHAPTER_PLAN_SCHEMA_MISMATCH", "chapter plan schema version must be 1")
    try:
        actual_cleaned_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() if isinstance(cleaned_text, str) else None
    except UnicodeError as exc:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "cleaned text hash does not match") from exc
    if actual_cleaned_hash != cleaned_hash:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "cleaned text hash does not match")
    try:
        verify_canonical_artifact_hash(artifact)
    except VoicePlanError as exc:
        raise VoiceAnalysisError(exc.code, exc.message, details=exc.details) from exc

    _analyzer(artifact["analyzer"])
    status = artifact["status"]
    stage = artifact["stage"]
    if not isinstance(status, str) or status not in _STATES:
        raise _fail("INVALID_STATUS_VALUE", "status is invalid")
    if not isinstance(stage, str):
        raise _fail("INVALID_STAGE", "stage is invalid")
    allowed_stages = _STATES[status]
    if isinstance(allowed_stages, set):
        if stage not in allowed_stages:
            raise _fail("INVALID_STAGE", "stage is invalid for status")
    elif stage != allowed_stages:
        raise _fail("INVALID_STAGE", "stage is invalid for status")
    progress = _object(artifact["progress"], {"completed", "total"}, "progress")
    completed = _nonnegative_int(progress["completed"], "INVALID_PROGRESS", "progress completed must be nonnegative")
    total = _nonnegative_int(progress["total"], "INVALID_PROGRESS", "progress total must be nonnegative")
    if completed > total:
        raise _fail("INVALID_PROGRESS", "progress completed cannot exceed total")
    cancel_requested = artifact["cancel_requested"]
    if type(cancel_requested) is not bool:
        raise _fail("INVALID_CANCEL_REQUEST", "cancel_requested must be boolean")
    expected_cancel = status == "cancelled" or status == "running"
    if cancel_requested and not expected_cancel:
        raise _fail("INVALID_CANCEL_REQUEST", "cancel_requested is invalid for this status")
    if status == "cancelled" and not cancel_requested:
        raise _fail("INVALID_CANCEL_REQUEST", "cancelled status requires cancel_requested")
    if status == "completed" and completed != total:
        raise _fail("INVALID_PROGRESS", "completed status requires complete progress")
    warnings = artifact["warnings"]
    if not isinstance(warnings, list):
        raise _fail("INVALID_WARNINGS", "warnings must be a list")
    if sum(len(item) for item in warnings if isinstance(item, str)) > MAX_WARNINGS_BYTES:
        raise _fail("INVALID_WARNINGS", "warnings exceed the size limit")
    for warning in warnings:
        if not isinstance(warning, str) or len(warning) > MAX_WARNING_BYTES or any(ord(char) < 32 for char in warning):
            raise _fail("INVALID_WARNINGS", "warning is invalid")
    if status == "failed":
        if artifact["error"] is None:
            raise _fail("INVALID_ERROR", "failed status requires an error")
        _error(artifact["error"])
    elif artifact["error"] is not None:
        raise _fail("INVALID_ERROR", "non-failed status must not include an error")
    started = _timestamp(artifact["started_at"], "started_at")
    updated = _timestamp(artifact["updated_at"], "updated_at")
    finished_value = artifact["finished_at"]
    if status in {"queued", "running"}:
        if finished_value is not None:
            raise _fail("INVALID_TIMESTAMP", "active status must not have finished_at")
    else:
        if finished_value is None:
            raise _fail("INVALID_TIMESTAMP", "terminal status requires finished_at")
    if finished_value is not None:
        finished = _timestamp(finished_value, "finished_at")
        if not started <= updated <= finished:
            raise _fail("INVALID_TIMESTAMP_ORDER", "status timestamps are out of order")
    elif started > updated:
        raise _fail("INVALID_TIMESTAMP_ORDER", "status timestamps are out of order")
    return artifact


__all__ = ["MAX_ARTIFACT_BYTES", "MAX_ERROR_CODE", "MAX_WARNING_BYTES", "MAX_WARNINGS_BYTES", "VoiceAnalysisError", "validate_voice_analysis_status"]

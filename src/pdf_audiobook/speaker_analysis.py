"""Strict machine-only speaker-analysis artifact validation and bindings."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from . import speakers
from .voice_plan import VoicePlanError, canonical_json_bytes, verify_canonical_artifact_hash


MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_ENTITIES = 100_000
MAX_SPANS = 2_000_000
MAX_TEXT = 512
MAX_REASON_SOURCE = 8 * 1024
MAX_WARNINGS_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,512}$")
_TOP_FIELDS = {
    "schema_version", "artifact", "revision", "source_pdf_sha256", "cleaned_text_sha256",
    "chapter_plan_sha256", "chapter_plan_schema_version", "analyzer", "characters", "spans",
    "warnings", "canonical_artifact_sha256",
}


class SpeakerAnalysisError(ValueError):
    """Stable validation failure for machine-only speaker analysis."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> SpeakerAnalysisError:
    return SpeakerAnalysisError(code, message, details=details)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _fail("INVALID_" + name.upper(), f"{name} must be lowercase SHA-256 hex")
    return value


def _text(value: Any, name: str, maximum: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise _fail("INVALID_" + name.upper(), f"{name} is invalid")
    return value


def _stable_id(value: Any, name: str, *, allow_narrator: bool = True) -> str:
    if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
        raise _fail("INVALID_" + name.upper(), f"{name} has an invalid stable ID")
    if not allow_narrator and value == "narrator":
        raise _fail("INVALID_CHARACTER_ID", "character_id cannot be narrator")
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


def _confidence(value: Any, name: str = "confidence") -> speakers.Confidence:
    value = _object(value, {"score", "band", "reasons"}, name)
    score = value["score"]
    try:
        score_float = float(score)
    except (TypeError, OverflowError) as exc:
        raise _fail("INVALID_CONFIDENCE", "confidence score is invalid") from exc
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score_float) or not 0 <= score <= 1:
        raise _fail("INVALID_CONFIDENCE", "confidence score is invalid")
    if not isinstance(value["band"], str) or value["band"] not in speakers.CONFIDENCE_BANDS:
        raise _fail("INVALID_CONFIDENCE_BAND", "confidence band is invalid")
    reasons = value["reasons"]
    if not isinstance(reasons, list):
        raise _fail("INVALID_CONFIDENCE_REASONS", "confidence reasons must be a list")
    if sum(len(item) for item in reasons if isinstance(item, str)) > MAX_REASON_SOURCE:
        raise _fail("INVALID_CONFIDENCE_REASONS", "confidence reasons exceed the size limit")
    for item in reasons:
        _text(item, "confidence_reason", MAX_REASON_SOURCE)
    try:
        return speakers.Confidence(score, value["band"], tuple(reasons))
    except speakers.SpeakerPlanError as exc:
        raise SpeakerAnalysisError(exc.code, exc.message, details=exc.details) from exc


def _provenance(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict) or set(value) not in ({"source"}, {"source", "quote_id"}):
        raise _fail("INVALID_PROVENANCE", "provenance schema mismatch")
    source = _text(value["source"], "provenance_source", MAX_REASON_SOURCE)
    result = [source]
    if "quote_id" in value:
        result.append(_stable_id(value["quote_id"], "quote_id"))
    return tuple(result)


def _validate_alias(value: Any, character_id: str) -> tuple[Any, ...]:
    alias = _object(value, {"alias", "kind", "confidence", "provenance"}, "alias")
    alias_text = _text(alias["alias"], "alias", MAX_TEXT)
    if not isinstance(alias["kind"], str) or alias["kind"] not in {"proper", "nominal", "pronoun"}:
        raise _fail("INVALID_ALIAS_KIND", "alias kind is invalid")
    confidence = alias["confidence"]
    try:
        score = float(confidence)
    except (TypeError, OverflowError) as exc:
        raise _fail("INVALID_ALIAS_CONFIDENCE", "alias confidence is invalid") from exc
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise _fail("INVALID_ALIAS_CONFIDENCE", "alias confidence is invalid")
    provenance = _object(alias["provenance"], {"source", "token_start", "token_end"}, "alias_provenance")
    source = _text(provenance["source"], "alias_provenance_source", MAX_REASON_SOURCE)
    start = _nonnegative_int(provenance["token_start"], "INVALID_TOKEN_RANGE", "token_start must be nonnegative")
    end = _nonnegative_int(provenance["token_end"], "INVALID_TOKEN_RANGE", "token_end must be nonnegative")
    if end <= start:
        raise _fail("INVALID_TOKEN_RANGE", "token range must be non-empty")
    return (alias_text, alias["kind"], score, source, start, end, character_id)


def validate_speaker_analysis(
    artifact: Any,
    cleaned_text: str,
    chapter_plan: Any,
    *,
    expected_source_pdf_sha256: str,
    expected_chapter_plan_sha256: str,
) -> dict[str, Any]:
    """Validate and return a machine-only speaker-analysis artifact unchanged."""

    if not isinstance(artifact, dict) or set(artifact) != _TOP_FIELDS:
        raise _fail("INVALID_ANALYSIS", "speaker analysis schema mismatch")
    try:
        if len(canonical_json_bytes(artifact)) > MAX_ARTIFACT_BYTES:
            raise _fail("ANALYSIS_TOO_LARGE", "speaker analysis artifact exceeds size limit")
    except VoicePlanError as exc:
        raise _fail("INVALID_ANALYSIS", "speaker analysis is not canonical JSON") from exc
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise _fail("INVALID_SCHEMA_VERSION", "speaker analysis schema_version must be 1")
    if artifact["artifact"] != "speaker-analysis":
        raise _fail("INVALID_ARTIFACT_TYPE", "artifact must be speaker-analysis")
    revision = _positive_int(artifact["revision"], "INVALID_REVISION", "revision must be positive")
    source_hash = _sha(artifact["source_pdf_sha256"], "source_pdf_sha256")
    cleaned_hash = _sha(artifact["cleaned_text_sha256"], "cleaned_text_sha256")
    chapter_hash = _sha(artifact["chapter_plan_sha256"], "chapter_plan_sha256")
    expected_source = _sha(expected_source_pdf_sha256, "expected_source_pdf_sha256")
    expected_chapter = _sha(expected_chapter_plan_sha256, "expected_chapter_plan_sha256")
    if source_hash != expected_source:
        raise _fail("SOURCE_HASH_MISMATCH", "speaker analysis source hash does not match current source")
    if chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "speaker analysis chapter-plan hash does not match current plan")
    try:
        current_chapter_hash = hashlib.sha256(canonical_json_bytes(chapter_plan)).hexdigest()
    except VoicePlanError as exc:
        raise _fail("INVALID_CHAPTER_PLAN", "current chapter plan is invalid") from exc
    if current_chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "current chapter plan hash does not match expected binding")
    if type(artifact["chapter_plan_schema_version"]) is not int or artifact["chapter_plan_schema_version"] != 1:
        raise _fail("CHAPTER_PLAN_SCHEMA_MISMATCH", "chapter plan schema version must be 1")
    try:
        text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() if isinstance(cleaned_text, str) else None
    except UnicodeError as exc:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "cleaned text hash does not match") from exc
    if text_hash != cleaned_hash:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "cleaned text hash does not match")
    try:
        verify_canonical_artifact_hash(artifact)
    except VoicePlanError as exc:
        raise SpeakerAnalysisError(exc.code, exc.message, details=exc.details) from exc

    analyzer = _object(artifact["analyzer"], {"id", "version", "model_hash"}, "analyzer")
    _text(analyzer["id"], "analyzer_id")
    _text(analyzer["version"], "analyzer_version")
    if analyzer["model_hash"] is not None:
        _sha(analyzer["model_hash"], "model_hash")

    characters = artifact["characters"]
    if not isinstance(characters, list):
        raise _fail("INVALID_CHARACTERS", "characters must be a list")
    character_ids: set[str] = set()
    total_entities = 0
    for raw in characters:
        character = _object(raw, {"character_id", "canonical_label", "aliases", "line_count", "quote_count"}, "character")
        character_id = _stable_id(character["character_id"], "character_id", allow_narrator=False)
        if character_id in character_ids:
            raise _fail("DUPLICATE_CHARACTER_ID", "character IDs must be unique")
        character_ids.add(character_id)
        _text(character["canonical_label"], "canonical_label")
        for field in ("line_count", "quote_count"):
            _nonnegative_int(character[field], "INVALID_CHARACTER_COUNT", f"{field} must be nonnegative")
        aliases = character["aliases"]
        if not isinstance(aliases, list):
            raise _fail("INVALID_ALIASES", "character aliases must be a list")
        total_entities += 1 + len(aliases)
        if total_entities > MAX_ENTITIES:
            raise _fail("ANALYSIS_TOO_LARGE", "characters and aliases exceed the size limit")
        seen_aliases: set[tuple[Any, ...]] = set()
        for raw_alias in aliases:
            key = _validate_alias(raw_alias, character_id)
            comparable = key[:-1]
            if comparable in seen_aliases:
                raise _fail("DUPLICATE_ALIAS", "identical alias records must not repeat")
            seen_aliases.add(comparable)

    spans = artifact["spans"]
    if not isinstance(spans, list):
        raise _fail("INVALID_SPANS", "spans must be a list")
    if len(spans) > MAX_SPANS:
        raise _fail("ANALYSIS_TOO_LARGE", "spans exceed the size limit")
    span_values: list[speakers.SpeakerSpan] = []
    for raw in spans:
        value = _object(raw, {"span_id", "chapter_index", "source_start", "source_end", "type", "speaker_id", "confidence", "provenance"}, "span")
        span_id = _stable_id(value["span_id"], "span_id")
        if type(value["chapter_index"]) is not int or value["chapter_index"] < 1:
            raise _fail("INVALID_CHAPTER_INDEX", "chapter_index must be one-based integer")
        for field in ("source_start", "source_end"):
            if type(value[field]) is not int:
                raise _fail("INVALID_SPAN_RANGE", "span offsets must be integers")
        if value["speaker_id"] is not None:
            speaker_id = _stable_id(value["speaker_id"], "speaker_id")
            if speaker_id != "narrator" and speaker_id not in character_ids:
                raise _fail("UNKNOWN_SPEAKER_REFERENCE", "span speaker_id is not declared")
        else:
            speaker_id = None
        confidence = _confidence(value["confidence"])
        provenance = _provenance(value["provenance"])
        try:
            span_values.append(speakers.SpeakerSpan(span_id, value["chapter_index"], value["source_start"], value["source_end"], value["type"], speaker_id, confidence, provenance))
        except speakers.SpeakerPlanError as exc:
            raise SpeakerAnalysisError(exc.code, exc.message, details=exc.details) from exc
    try:
        speakers.validate_machine_spans(span_values, cleaned_text, chapter_plan)
    except speakers.SpeakerPlanError as exc:
        raise SpeakerAnalysisError(exc.code, exc.message, details=exc.details) from exc

    warnings = artifact["warnings"]
    if not isinstance(warnings, list):
        raise _fail("INVALID_WARNINGS", "warnings must be a list")
    if sum(len(item) for item in warnings if isinstance(item, str)) > MAX_WARNINGS_BYTES:
        raise _fail("INVALID_WARNINGS", "warnings exceed the size limit")
    for warning in warnings:
        if not isinstance(warning, str) or len(warning) > MAX_REASON_SOURCE or any(ord(char) < 32 for char in warning):
            raise _fail("INVALID_WARNINGS", "warning is invalid")
    return artifact


__all__ = ["MAX_ARTIFACT_BYTES", "MAX_ENTITIES", "MAX_SPANS", "SpeakerAnalysisError", "validate_speaker_analysis"]

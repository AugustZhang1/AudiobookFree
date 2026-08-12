"""Strict Interactive Voices approval artifacts and canonical primitives.

This module defines the approved persisted voice-plan schema while excluding
drafts, API transport, and generation concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Sequence

from . import speakers
from .voice_settings import VoiceSettingsError, canonical_voice_settings


ROLES = frozenset({"narrator", "character"})
RELATIONSHIPS = frozenset({"third_person", "same_as_narrator", "separate_from_narrator"})
MIN_CHARACTER_QUOTE_COUNT = 10
_ELIGIBLE_SPEAKING_TYPES = frozenset({"dialogue", "unknown"})
_PRONOUN_LABELS = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_HASH_FIELD = "canonical_artifact_sha256"
_VOICE_PLAN_FIELDS = {
    "schema_version",
    "artifact",
    "revision",
    "source_pdf_sha256",
    "cleaned_text_sha256",
    "chapter_plan_sha256",
    "chapter_plan_schema_version",
    "analyzer",
    "cast",
    "aliases",
    "chapters",
    "unresolved_policy",
    "approval",
    _CANONICAL_HASH_FIELD,
}


class VoicePlanError(ValueError):
    """Stable validation failure for cast and canonical-artifact boundaries."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> VoicePlanError:
    return VoicePlanError(code, message, details=details)


def _text(value: Any, name: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise _fail("INVALID_" + name.upper(), f"{name} must be a non-empty string without controls")
    if maximum is not None and len(value) > maximum:
        raise _fail("INVALID_" + name.upper(), f"{name} exceeds its maximum length")
    return value


@dataclass(frozen=True, slots=True)
class CastEntry:
    """Immutable neutral cast identity and voice assignment."""

    cast_id: str
    display_label: str
    role: str
    relationship: str
    voice_id: str
    speed: float
    pitch_semitones: int = 0
    tone_preset: str = "neutral"

    def __post_init__(self) -> None:
        _text(self.cast_id, "cast_id")
        _text(self.display_label, "display_label", maximum=512)
        if not isinstance(self.role, str) or self.role not in ROLES:
            raise _fail("INVALID_ROLE", "role must be narrator or character")
        if not isinstance(self.relationship, str) or self.relationship not in RELATIONSHIPS:
            raise _fail("INVALID_RELATIONSHIP", "relationship is not supported")
        _text(self.voice_id, "voice_id")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise _fail("INVALID_SPEED", "speed must be numeric")
        try:
            speed = float(self.speed)
        except (TypeError, OverflowError) as exc:
            raise _fail("INVALID_SPEED", "speed must be finite and positive") from exc
        if not math.isfinite(speed) or speed <= 0:
            raise _fail("INVALID_SPEED", "speed must be finite and positive")
        object.__setattr__(self, "speed", speed)
        try:
            settings = canonical_voice_settings({"speed": speed, "pitch_semitones": self.pitch_semitones, "tone_preset": self.tone_preset})
        except VoiceSettingsError as exc:
            raise _fail(exc.code, exc.message) from exc
        object.__setattr__(self, "pitch_semitones", settings["pitch_semitones"])
        object.__setattr__(self, "tone_preset", settings["tone_preset"])


def _cast_sequence(cast: Any) -> tuple[CastEntry, ...]:
    if isinstance(cast, (str, bytes)) or not isinstance(cast, Sequence):
        raise _fail("INVALID_CAST", "cast must be an ordered sequence of CastEntry values")
    values = tuple(cast)
    if any(not isinstance(entry, CastEntry) for entry in values):
        raise _fail("INVALID_CAST", "cast must contain CastEntry values")
    seen: set[str] = set()
    for entry in values:
        if entry.cast_id in seen:
            raise _fail("DUPLICATE_CAST_ID", "cast IDs must be unique", cast_id=entry.cast_id)
        seen.add(entry.cast_id)
    return values


def validate_voice_plan_core(
    cast: Any,
    spans: Any,
    cleaned_text: str,
    chapter_plan: Any,
    *,
    narrator_fallback_accepted: bool = False,
    allow_unresolved: bool = False,
) -> tuple[tuple[CastEntry, ...], tuple[speakers.SpeakerSpan, ...]]:
    """Validate approved spans and their cast assignments without changing either."""

    try:
        validated_spans = speakers.validate_approved_spans(
            spans,
            cleaned_text,
            chapter_plan,
            narrator_fallback_accepted=narrator_fallback_accepted,
            allow_unresolved=allow_unresolved,
        )
    except speakers.SpeakerPlanError as exc:
        raise VoicePlanError(exc.code, exc.message, details=exc.details) from exc

    values = _cast_sequence(cast)
    by_id = {entry.cast_id: entry for entry in values}
    narrator_by_id = by_id.get("narrator")
    narrator_roles = [entry for entry in values if entry.role == "narrator"]
    if narrator_by_id is None and not narrator_roles:
        raise _fail("MISSING_NARRATOR", "cast must contain a narrator entry")
    if narrator_by_id is None or narrator_by_id.role != "narrator" or len(narrator_roles) != 1:
        raise _fail("MISIDENTIFIED_NARRATOR", "exactly cast_id narrator must be the sole narrator role")
    narrator = narrator_by_id

    for span in validated_spans:
        if span.speaker_id not in by_id:
            raise _fail("UNKNOWN_SPEAKER_REFERENCE", "span speaker_id does not resolve to cast", speaker_id=span.speaker_id)
    for entry in values:
        if entry.role == "character" and entry.relationship == "same_as_narrator" and entry.voice_id != narrator.voice_id:
            raise _fail("NARRATOR_VOICE_MISMATCH", "same_as_narrator character must use the narrator voice")
    return values, tuple(validated_spans)


def canonical_json_text(value: Any) -> str:
    """Serialize a JSON-compatible value using the project canonical form."""

    try:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        result.encode("utf-8")
        return result
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise _fail("INVALID_ARTIFACT", "value is not finite, JSON-compatible data") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes with a trailing newline."""

    try:
        return canonical_json_text(value).encode("utf-8")
    except UnicodeError as exc:
        raise _fail("INVALID_ARTIFACT", "value is not valid UTF-8 JSON data") from exc


def _artifact_without_hash(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise _fail("INVALID_ARTIFACT", "artifact must be a JSON object")
    copy = dict(artifact)
    copy.pop(_CANONICAL_HASH_FIELD, None)
    return copy


def canonical_artifact_digest(artifact: Any) -> str:
    """Hash an artifact after excluding only its top-level canonical digest."""

    return hashlib.sha256(canonical_json_bytes(_artifact_without_hash(artifact))).hexdigest()


def with_canonical_artifact_hash(artifact: Any) -> dict[str, Any]:
    """Return a shallow copy carrying its correct canonical artifact digest."""

    result = _artifact_without_hash(artifact)
    result[_CANONICAL_HASH_FIELD] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_canonical_artifact_hash(artifact: Any) -> bool:
    """Verify a canonical artifact's lowercase SHA-256 digest."""

    if not isinstance(artifact, dict):
        raise _fail("INVALID_ARTIFACT", "artifact must be a JSON object")
    actual = artifact.get(_CANONICAL_HASH_FIELD)
    if not isinstance(actual, str) or not _SHA256.fullmatch(actual):
        raise _fail("INVALID_ARTIFACT_HASH", "artifact hash must be lowercase SHA-256 hex")
    expected = canonical_artifact_digest(artifact)
    if actual != expected:
        raise _fail("ARTIFACT_HASH_MISMATCH", "artifact hash does not match canonical content")
    return True


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _fail("INVALID_" + name.upper(), f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _positive_int(value: Any, code: str, message: str) -> int:
    if type(value) is not int or value <= 0:
        raise _fail(code, message)
    return value


def _bounded(value: Any, name: str, maximum: int = 512, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise _fail("INVALID_" + name.upper(), f"{name} is invalid")
    return value


def _timestamp(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str):
        raise _fail("INVALID_" + name.upper(), f"{name} must be timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("INVALID_" + name.upper(), f"{name} must be timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise _fail("INVALID_" + name.upper(), f"{name} must be timezone-aware ISO timestamp")
    return value


def _strict_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail("INVALID_" + name.upper(), f"{name} schema mismatch")
    return value


def _strict_string_list(value: Any, name: str, *, item_limit: int = 8192) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail("INVALID_" + name.upper(), f"{name} must be a list of strings")
    if sum(len(item) for item in value if isinstance(item, str)) > item_limit:
        raise _fail("INVALID_" + name.upper(), f"{name} exceeds its size limit")
    result: list[str] = []
    for item in value:
        result.append(_bounded(item, name, item_limit))
    return tuple(result)


def validate_voice_plan(
    artifact: Any,
    cleaned_text: str,
    chapter_plan: Any,
    *,
    expected_source_pdf_sha256: str,
    expected_chapter_plan_sha256: str,
) -> dict[str, Any]:
    """Strictly validate an approved voice-plan artifact and current bindings."""

    root = _strict_object(artifact, _VOICE_PLAN_FIELDS, "voice_plan")
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise _fail("INVALID_SCHEMA_VERSION", "voice plan schema_version must be 1")
    if root["artifact"] != "voice-plan":
        raise _fail("INVALID_ARTIFACT_TYPE", "voice plan artifact must be voice-plan")
    revision = _positive_int(root["revision"], "INVALID_REVISION", "revision must be positive")
    source_hash = _sha(root["source_pdf_sha256"], "source_pdf_sha256")
    cleaned_hash = _sha(root["cleaned_text_sha256"], "cleaned_text_sha256")
    chapter_hash = _sha(root["chapter_plan_sha256"], "chapter_plan_sha256")
    expected_source = _sha(expected_source_pdf_sha256, "expected_source_pdf_sha256")
    expected_chapter = _sha(expected_chapter_plan_sha256, "expected_chapter_plan_sha256")
    if source_hash != expected_source:
        raise _fail("SOURCE_HASH_MISMATCH", "voice plan source hash does not match current source")
    if chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "voice plan chapter-plan hash does not match current plan")
    try:
        current_chapter_hash = hashlib.sha256(canonical_json_bytes(chapter_plan)).hexdigest()
    except VoicePlanError as exc:
        raise _fail("INVALID_CHAPTER_PLAN", "current chapter plan is invalid") from exc
    if current_chapter_hash != expected_chapter:
        raise _fail("CHAPTER_PLAN_HASH_MISMATCH", "current chapter plan hash does not match expected binding")
    if type(root["chapter_plan_schema_version"]) is not int or root["chapter_plan_schema_version"] != 1:
        raise _fail("CHAPTER_PLAN_SCHEMA_MISMATCH", "voice plan chapter-plan schema version is unsupported")
    try:
        cleaned_digest = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() if isinstance(cleaned_text, str) else None
    except UnicodeError as exc:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "voice plan cleaned-text hash does not match current text") from exc
    if cleaned_digest != cleaned_hash:
        raise _fail("CLEANED_TEXT_HASH_MISMATCH", "voice plan cleaned-text hash does not match current text")
    verify_canonical_artifact_hash(root)

    analyzer = _strict_object(root["analyzer"], {"id", "version", "model_hash"}, "analyzer")
    _bounded(analyzer["id"], "analyzer_id")
    _bounded(analyzer["version"], "analyzer_version")
    if analyzer["model_hash"] is not None:
        _sha(analyzer["model_hash"], "model_hash")

    cast_raw = root["cast"]
    if not isinstance(cast_raw, list):
        raise _fail("INVALID_CAST", "cast must be a list")
    aliases_raw = root["aliases"]
    if not isinstance(aliases_raw, list):
        raise _fail("INVALID_ALIASES", "aliases must be a list")
    if len(cast_raw) + len(aliases_raw) > 100_000:
        raise _fail("VOICE_PLAN_TOO_LARGE", "cast and aliases exceed the size limit")
    cast_entries: list[CastEntry] = []
    for raw in cast_raw:
        entry = _strict_object(raw, {"cast_id", "display_label", "role", "relationship", "voice_id", "voice_settings"}, "cast_entry")
        try:
            settings = canonical_voice_settings(entry["voice_settings"])
        except VoiceSettingsError as exc:
            raise _fail(exc.code, exc.message) from exc
        try:
            cast_entries.append(CastEntry(entry["cast_id"], entry["display_label"], entry["role"], entry["relationship"], entry["voice_id"], settings["speed"], settings["pitch_semitones"], settings["tone_preset"]))
        except VoicePlanError:
            raise
        except (TypeError, ValueError) as exc:
            raise _fail("INVALID_CAST", "cast entry is invalid") from exc

    cast_by_id = {entry.cast_id: entry for entry in cast_entries}
    aliases_seen: set[str] = set()
    for raw in aliases_raw:
        alias = _strict_object(raw, {"alias_id", "text", "character_id", "override_state"}, "alias")
        alias_id = _bounded(alias["alias_id"], "alias_id")
        if alias_id in aliases_seen:
            raise _fail("DUPLICATE_ALIAS_ID", "alias IDs must be unique")
        aliases_seen.add(alias_id)
        _bounded(alias["text"], "alias_text", 512)
        if alias["override_state"] != "accepted":
            raise _fail("INVALID_OVERRIDE_STATE", "approved aliases must be accepted")
        character_id = _bounded(alias["character_id"], "character_id")
        character = cast_by_id.get(character_id)
        if character is None or character.role != "character":
            raise _fail("INVALID_ALIAS_CHARACTER", "alias character_id must resolve to a character")

    plan_chapters = chapter_plan.get("chapters") if isinstance(chapter_plan, dict) else None
    if not isinstance(plan_chapters, list) or any(not isinstance(item, dict) for item in plan_chapters) or type(chapter_plan.get("schema_version")) is not int or chapter_plan["schema_version"] != 1:
        raise _fail("INVALID_CHAPTER_PLAN", "current chapter plan is invalid")
    artifact_chapters = root["chapters"]
    if not isinstance(artifact_chapters, list) or len(artifact_chapters) != len(plan_chapters):
        raise _fail("CHAPTER_COVERAGE_MISMATCH", "voice plan chapters must match the complete chapter plan")
    span_values: list[speakers.SpeakerSpan] = []
    if len(artifact_chapters) == 0:
        raise _fail("CHAPTER_COVERAGE_MISMATCH", "voice plan must contain chapters")
    total_spans = 0
    for raw_chapter, current in zip(artifact_chapters, plan_chapters):
        chapter = _strict_object(raw_chapter, {"chapter_index", "source_start", "source_end", "source_page_start", "source_page_end", "spans"}, "voice_plan_chapter")
        for field in ("chapter_index", "source_start", "source_end", "source_page_start", "source_page_end"):
            if type(chapter[field]) is not int:
                raise _fail("INVALID_CHAPTER_RANGE", "chapter range fields must be integers")
        if chapter["chapter_index"] != current.get("index") or chapter["source_start"] != current.get("start_offset") or chapter["source_end"] != current.get("end_offset") or chapter["source_page_start"] != current.get("start_page") or chapter["source_page_end"] != current.get("end_page"):
            raise _fail("CHAPTER_COVERAGE_MISMATCH", "voice plan chapter does not match current chapter plan")
        spans_raw = chapter["spans"]
        if not isinstance(spans_raw, list):
            raise _fail("INVALID_SPANS", "chapter spans must be a list")
        total_spans += len(spans_raw)
        if total_spans > 2_000_000:
            raise _fail("VOICE_PLAN_TOO_LARGE", "voice plan spans exceed the size limit")
        for raw_span in spans_raw:
            value = _strict_object(raw_span, {"span_id", "source_start", "source_end", "type", "speaker_id", "confidence", "provenance", "override"}, "span")
            for field in ("source_start", "source_end"):
                if type(value[field]) is not int:
                    raise _fail("INVALID_SPAN_RANGE", "span source offsets must be integers")
            confidence = _strict_object(value["confidence"], {"score", "band", "reasons"}, "confidence")
            score = confidence["score"]
            try:
                score_value = float(score)
            except (TypeError, OverflowError) as exc:
                raise _fail("INVALID_CONFIDENCE", "span confidence score is invalid") from exc
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score_value) or not 0 <= score <= 1:
                raise _fail("INVALID_CONFIDENCE", "span confidence score is invalid")
            if not isinstance(confidence["band"], str) or confidence["band"] not in speakers.CONFIDENCE_BANDS:
                raise _fail("INVALID_CONFIDENCE_BAND", "span confidence band is invalid")
            reasons = _strict_string_list(confidence["reasons"], "confidence_reasons")
            provenance = _strict_object(value["provenance"], {"source", "analysis_revision"}, "provenance")
            _bounded(provenance["source"], "provenance_source", 8192)
            analysis_revision = _positive_int(provenance["analysis_revision"], "INVALID_ANALYSIS_REVISION", "analysis_revision must be positive")
            override = value["override"]
            if override is not None:
                override = _strict_object(override, {"kind", "from", "to", "actor", "reason"}, "override")
                if not isinstance(override["kind"], str) or override["kind"] not in {"speaker", "type"}:
                    raise _fail("INVALID_OVERRIDE", "override kind must be speaker or type")
                for field in ("kind", "from", "to", "actor", "reason"):
                    _bounded(override[field], "override_" + field, 8192)
                effective = value["speaker_id"] if override["kind"] == "speaker" else value["type"]
                if override["to"] != effective:
                    raise _fail("INVALID_OVERRIDE", "override to must equal effective span value")
            try:
                confidence_value = speakers.Confidence(score, confidence["band"], reasons)
                span_values.append(speakers.SpeakerSpan(value["span_id"], chapter["chapter_index"], value["source_start"], value["source_end"], value["type"], value["speaker_id"], confidence_value, (provenance["source"], str(analysis_revision))))
            except speakers.SpeakerPlanError as exc:
                raise VoicePlanError(exc.code, exc.message, details=exc.details) from exc
    unresolved = _strict_object(root["unresolved_policy"], {"mode", "accepted_by_user", "accepted_at"}, "unresolved_policy")
    if unresolved["mode"] != "narrator" or not isinstance(unresolved["accepted_by_user"], bool):
        raise _fail("INVALID_UNRESOLVED_POLICY", "unresolved policy is invalid")
    if unresolved["accepted_by_user"]:
        _timestamp(unresolved["accepted_at"], "accepted_at")
    elif unresolved["accepted_at"] is not None:
        raise _fail("INVALID_ACCEPTED_AT", "accepted_at must be null until policy acceptance")
    approval = _strict_object(root["approval"], {"state", "approved_at", "approved_revision"}, "approval")
    state = approval["state"]
    if state not in {"draft", "approved"}:
        raise _fail("INVALID_APPROVAL", "voice plan approval state is unsupported")
    if state == "draft":
        if approval["approved_at"] is not None or approval["approved_revision"] is not None:
            raise _fail("INVALID_DRAFT_APPROVAL", "draft approval must have null timestamp and revision")
    else:
        _timestamp(approval["approved_at"], "approved_at")
        approved_revision = _positive_int(approval["approved_revision"], "INVALID_APPROVED_REVISION", "approved_revision must be positive")
        if approved_revision != revision:
            raise _fail("APPROVAL_REVISION_MISMATCH", "approval revision must equal plan revision")
    try:
        validate_voice_plan_core(
            cast_entries,
            span_values,
            cleaned_text,
            chapter_plan,
            narrator_fallback_accepted=unresolved["accepted_by_user"],
            allow_unresolved=state == "draft",
        )
    except VoicePlanError:
        raise
    return artifact


def _source_hash_from_analysis(analysis: dict[str, Any]) -> str:
    for key in ("source_pdf_sha256", "source_hash", "pdf_sha256"):
        value = analysis.get(key)
        if isinstance(value, str) and _SHA256.fullmatch(value):
            return value
    raise _fail("MISSING_SOURCE_HASH", "speaker analysis must include source_pdf_sha256")


def _safe_id(value: Any, prefix: str) -> str:
    raw = str(value).strip() if value is not None else ""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
    if not slug:
        slug = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{slug}"


def _analysis_span(raw: Any, index: int, analyzer_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _fail("INVALID_ANALYSIS", "speaker analysis spans must be objects")
    start = raw.get("source_start", raw.get("start", raw.get("start_offset")))
    end = raw.get("source_end", raw.get("end", raw.get("end_offset")))
    if type(start) is not int or type(end) is not int:
        raise _fail("INVALID_ANALYSIS_SPAN", "analysis span offsets must be integers")
    confidence = raw.get("confidence", {})
    if isinstance(confidence, dict):
        score = confidence.get("score", 0.0)
        band = confidence.get("band", "low")
        reasons = confidence.get("reasons", [])
    else:
        score, band, reasons = confidence, "low", []
    if not isinstance(reasons, list):
        reasons = list(reasons) if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) else []
    span_type = raw.get("type", raw.get("span_type", "narration"))
    raw_speaker_id = raw.get("speaker_id", raw.get("speaker"))
    unresolved = raw_speaker_id is None or str(raw_speaker_id).strip().lower() in {"", "unknown"}
    if unresolved:
        span_type = "unknown"
    return {
        "span_id": str(raw.get("span_id", f"machine-{index + 1}")),
        "source_start": start,
        "source_end": end,
        "type": span_type if isinstance(span_type, str) and span_type in speakers.SPAN_TYPES else "unknown",
        # A draft needs an effective cast binding for structural validation, but
        # the unknown type preserves that this is still unresolved for review.
        "speaker_id": "narrator" if unresolved else raw_speaker_id,
        "confidence": {"score": score, "band": band, "reasons": reasons},
        "provenance": raw.get("provenance", {"source": analyzer_id, "analysis_revision": 1}),
        "override": raw.get("override"),
    }


def _speaker_key(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _speaking_span_counts(raw_spans: Sequence[Any]) -> dict[str, int]:
    """Count attributed machine quote spans without changing analyzer data."""

    counts: dict[str, int] = {}
    for raw in raw_spans:
        if not isinstance(raw, dict):
            continue
        speaker = _speaker_key(raw.get("speaker_id", raw.get("speaker")))
        if not speaker or speaker.lower() in {"narrator", "unknown"}:
            continue
        span_type = raw.get("type", raw.get("span_type", "narration"))
        if span_type not in _ELIGIBLE_SPEAKING_TYPES:
            continue
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def _alias_values(raw_aliases: Any) -> list[Any]:
    if isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, (str, bytes)):
        return list(raw_aliases)
    return []


def _alias_text(raw_alias: Any) -> str:
    if isinstance(raw_alias, dict):
        value = raw_alias.get("text", raw_alias.get("alias", ""))
    else:
        value = raw_alias
    return value.strip() if isinstance(value, str) else ""


def _character_identity_keys(item: dict[str, Any], original_id: Any, aliases: Sequence[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for value in (
        original_id,
        item.get("canonical_label"),
        item.get("display_label"),
        item.get("name"),
    ):
        key = _speaker_key(value)
        if key and key not in values:
            values.append(key)
    for alias in aliases:
        key = _alias_text(alias)
        if key and key not in values:
            values.append(key)
    return tuple(values)


def _is_pronoun_label(value: Any) -> bool:
    return _speaker_key(value).casefold() in _PRONOUN_LABELS


_TITLE_ONLY_LABELS = frozenset({
    "admiral", "captain", "chief", "coach", "colonel", "commander", "doctor", "dr",
    "general", "judge", "king", "lady", "lieutenant", "lord", "major", "marshal",
    "miss", "mr", "mrs", "ms", "officer", "private", "professor", "reverend",
    "sergeant", "sir", "the captain", "the doctor", "the judge", "the officer",
})


def _looks_like_proper_name(value: Any) -> bool:
    """Conservative fallback name heuristic for untyped analyzer output."""

    label = _speaker_key(value)
    if not label or _is_pronoun_label(label) or label.casefold() in _TITLE_ONLY_LABELS:
        return False
    # Analyzer IDs and rank-like labels are commonly lower-case or opaque.
    if label != label.casefold() and label[0].isupper() and any(char.isalpha() for char in label):
        normalized = label.casefold()
        machine_like = (
            re.fullmatch(r"(?:character|speaker|person|entity|id)[-_][a-z0-9]+(?:[-_][a-z0-9]+)*", normalized)
            or re.fullmatch(r"[a-z]+[-_]\d+", normalized)
            or normalized.endswith(("-id", "_id", "-uuid", "_uuid"))
        )
        return not bool(machine_like)
    return False


def _has_proper_name_evidence(item: dict[str, Any], aliases: Sequence[Any], fallback_label: Any) -> bool:
    typed = [alias for alias in aliases if isinstance(alias, dict) and _alias_text(alias)]
    if typed:
        return any(alias.get("kind") == "proper" for alias in typed)
    return _looks_like_proper_name(item.get("canonical_label", item.get("display_label", item.get("name", fallback_label))))


def build_voice_plan(
    speaker_analysis: dict[str, Any],
    cleaned_text: str,
    chapter_plan: dict[str, Any],
    voice_ids: Sequence[str],
    *,
    source_pdf_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, complete schema-1 draft from analyzer output."""

    if not isinstance(speaker_analysis, dict):
        raise _fail("INVALID_ANALYSIS", "speaker_analysis must be an object")
    if not isinstance(cleaned_text, str):
        raise _fail("INVALID_TEXT", "cleaned_text must be a string")
    try:
        if not isinstance(chapter_plan, dict) or not isinstance(chapter_plan.get("chapters"), list):
            raise _fail("INVALID_CHAPTER_PLAN", "chapter plan must contain chapters")
        source_hash = source_pdf_sha256 or _source_hash_from_analysis(speaker_analysis)
        _sha(source_hash, "source_pdf_sha256")
        chapter_hash = hashlib.sha256(canonical_json_bytes(chapter_plan)).hexdigest()
        cleaned_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise _fail("INVALID_TEXT", "cleaned_text must be valid UTF-8") from exc
    if isinstance(voice_ids, (str, bytes)) or not isinstance(voice_ids, Sequence):
        raise _fail("INVALID_VOICE_IDS", "voice_ids must be an ordered sequence")
    ordered_voices = tuple(voice_ids)
    if any(not isinstance(value, str) or not value or any(ord(char) < 32 for char in value) for value in ordered_voices):
        raise _fail("INVALID_VOICE_IDS", "voice_ids must contain non-empty strings")
    analyzer_raw = speaker_analysis.get("analyzer", {})
    if not isinstance(analyzer_raw, dict):
        analyzer_raw = {}
    analyzer_id = str(analyzer_raw.get("id", speaker_analysis.get("analyzer_id", "speaker-analysis")))
    analyzer_version = str(analyzer_raw.get("version", speaker_analysis.get("analysis_revision", "1")))
    model_hash = analyzer_raw.get("model_hash")
    if model_hash is not None:
        _sha(model_hash, "model_hash")

    characters_raw = speaker_analysis.get("characters", [])
    if not isinstance(characters_raw, Sequence) or isinstance(characters_raw, (str, bytes)):
        raise _fail("INVALID_CHARACTERS", "analysis characters must be an ordered sequence")
    raw_spans = speaker_analysis.get("spans", speaker_analysis.get("segments", []))
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise _fail("INVALID_ANALYSIS", "speaker analysis spans must be an ordered sequence")
    speaking_counts = _speaking_span_counts(raw_spans)
    character_specs: list[dict[str, Any]] = []
    used_ids = {"narrator"}
    known_originals: set[str] = set()
    for item in characters_raw:
        if not isinstance(item, dict):
            raise _fail("INVALID_CHARACTERS", "analysis characters must contain objects")
        original_id = item.get("character_id", item.get("id", item.get("canonical_label", item.get("name"))))
        aliases = _alias_values(item.get("aliases", []))
        identity_keys = _character_identity_keys(item, original_id, aliases)
        known_originals.update(identity_keys)
        observed_count = sum(speaking_counts.get(key, 0) for key in identity_keys)
        if observed_count < MIN_CHARACTER_QUOTE_COUNT:
            continue
        cast_id = _safe_id(original_id, "character")
        if cast_id in used_ids:
            continue
        proper_alias = next(
            (_alias_text(alias) for alias in aliases if isinstance(alias, dict) and alias.get("kind") == "proper" and _alias_text(alias)),
            "",
        )
        fallback_label = item.get("canonical_label", item.get("display_label", item.get("name", original_id or cast_id)))
        if not _has_proper_name_evidence(item, aliases, fallback_label):
            continue
        used_ids.add(cast_id)
        label = proper_alias or _speaker_key(fallback_label) or cast_id
        relationship = item.get("relationship", "separate_from_narrator")
        character_specs.append({"cast_id": cast_id, "source_id": _speaker_key(original_id), "label": label, "relationship": relationship, "aliases": aliases, "identity_keys": identity_keys})
    # Include speakers that were detected in spans but omitted from characters.
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        speaker = _speaker_key(item.get("speaker_id", item.get("speaker")))
        if not speaker or speaker.lower() in {"narrator", "unknown"} or speaker in known_originals:
            continue
        if speaking_counts.get(speaker, 0) < MIN_CHARACTER_QUOTE_COUNT:
            continue
        if not _looks_like_proper_name(speaker):
            continue
        cast_id = _safe_id(speaker, "character")
        if cast_id not in used_ids:
            used_ids.add(cast_id)
            character_specs.append({"cast_id": cast_id, "source_id": speaker, "label": speaker, "relationship": "separate_from_narrator", "aliases": [], "identity_keys": (speaker,)})

    if not ordered_voices:
        raise _fail("INVALID_VOICE_IDS", "voice_ids must provide at least one voice")
    default_settings = canonical_voice_settings({"speed": 1.0})
    cast: list[dict[str, Any]] = [{"cast_id": "narrator", "display_label": "Narrator", "role": "narrator", "relationship": "third_person", "voice_id": ordered_voices[0], "voice_settings": dict(default_settings)}]
    aliases: list[dict[str, Any]] = []
    original_to_cast: dict[str, str] = {"narrator": "narrator"}
    for position, spec in enumerate(character_specs, start=1):
        cast_id, label = spec["cast_id"], spec["label"]
        relationship, alias_values = spec["relationship"], spec["aliases"]
        cast_voice = ordered_voices[position % len(ordered_voices)]
        if relationship == "same_as_narrator":
            cast_voice = ordered_voices[0]
        cast.append({"cast_id": cast_id, "display_label": label, "role": "character", "relationship": relationship, "voice_id": cast_voice, "voice_settings": dict(default_settings)})
        original_to_cast[cast_id] = cast_id
        original_to_cast[label] = cast_id
        original_to_cast[spec["source_id"]] = cast_id
        for identity_key in spec.get("identity_keys", ()):
            original_to_cast[identity_key] = cast_id
        for alias_index, alias in enumerate(alias_values):
            if isinstance(alias, dict):
                alias_text = alias.get("text", alias.get("alias", ""))
                alias_id = alias.get("alias_id")
            else:
                alias_text, alias_id = alias, None
            if not isinstance(alias_text, str) or not alias_text:
                continue
            aliases.append({"alias_id": str(alias_id or f"alias-{cast_id}-{alias_index + 1}"), "text": alias_text, "character_id": cast_id, "override_state": "accepted"})
            original_to_cast[alias_text] = cast_id
        original_to_cast[str(label)] = cast_id
    for alias_index, raw_alias in enumerate(speaker_analysis.get("aliases", []) if isinstance(speaker_analysis.get("aliases", []), list) else []):
        if not isinstance(raw_alias, dict):
            continue
        alias_text = raw_alias.get("text", raw_alias.get("alias", ""))
        target = original_to_cast.get(str(raw_alias.get("character_id", "")), _safe_id(raw_alias.get("character_id"), "character"))
        if isinstance(alias_text, str) and alias_text and target in used_ids:
            aliases.append({"alias_id": str(raw_alias.get("alias_id", f"alias-{target}-{alias_index + 1}")), "text": alias_text, "character_id": target, "override_state": "accepted"})

    chapters: list[dict[str, Any]] = []
    analysis_values = [_analysis_span(item, index, analyzer_id) for index, item in enumerate(raw_spans)]
    seen_span_ids: set[str] = set()
    chapter_ranges = [(item.get("start_offset"), item.get("end_offset")) for item in chapter_plan["chapters"] if isinstance(item, dict)]
    for item in analysis_values:
        if not any(type(start) is int and type(end) is int and start <= item["source_start"] and item["source_end"] <= end for start, end in chapter_ranges):
            raise _fail("INVALID_ANALYSIS_SPAN", "analysis span is outside the chapter plan")
    for index, current in enumerate(chapter_plan["chapters"], start=1):
        if not isinstance(current, dict):
            raise _fail("INVALID_CHAPTER_PLAN", "chapter entries must be objects")
        start, end = current.get("start_offset"), current.get("end_offset")
        start_page, end_page = current.get("start_page"), current.get("end_page")
        chapter_index = current.get("index")
        if type(chapter_index) is not int or chapter_index != index or type(start) is not int or type(end) is not int or start < 0 or end <= start or end > len(cleaned_text) or type(start_page) is not int or type(end_page) is not int:
            raise _fail("INVALID_CHAPTER_PLAN", "chapter offsets are invalid")
        candidates = [item for item in analysis_values if start <= item["source_start"] and item["source_end"] <= end]
        for item in analysis_values:
            if item["source_start"] < 0 or item["source_end"] > len(cleaned_text) or item["source_end"] <= item["source_start"]:
                raise _fail("INVALID_ANALYSIS_SPAN", "analysis span range is outside cleaned text")
            if item not in candidates and item["source_start"] < end and item["source_end"] > start:
                raise _fail("INVALID_ANALYSIS_SPAN", "analysis span crosses a chapter boundary or is outside the chapter plan")
        candidates.sort(key=lambda item: (item["source_start"], item["source_end"], item["span_id"]))
        built: list[dict[str, Any]] = []
        cursor = start
        for item in candidates:
            if item["source_start"] < cursor or item["source_end"] > end:
                raise _fail("INVALID_ANALYSIS_SPAN", "analysis spans overlap or cross chapter boundaries")
            if item["source_start"] > cursor:
                built.append({"span_id": f"gap-{index}-{len(built) + 1}", "source_start": cursor, "source_end": item["source_start"], "type": "narration", "speaker_id": "narrator", "confidence": {"score": 0.0, "band": "low", "reasons": ["coverage_gap"]}, "provenance": {"source": analyzer_id, "analysis_revision": 1}, "override": None})
            if item["span_id"] in seen_span_ids:
                raise _fail("DUPLICATE_SPAN_ID", "analysis span IDs must be unique")
            seen_span_ids.add(item["span_id"])
            item = dict(item)
            item["speaker_id"] = original_to_cast.get(str(item["speaker_id"]), "narrator")
            item["type"] = item["type"] if item["type"] in speakers.SPAN_TYPES else "unknown"
            built.append(item)
            cursor = item["source_end"]
        if cursor < end:
            built.append({"span_id": f"gap-{index}-{len(built) + 1}", "source_start": cursor, "source_end": end, "type": "narration", "speaker_id": "narrator", "confidence": {"score": 0.0, "band": "low", "reasons": ["coverage_gap"]}, "provenance": {"source": analyzer_id, "analysis_revision": 1}, "override": None})
        chapters.append({"chapter_index": chapter_index, "source_start": start, "source_end": end, "source_page_start": start_page, "source_page_end": end_page, "spans": built})
    artifact = {"schema_version": 1, "artifact": "voice-plan", "revision": 1, "source_pdf_sha256": source_hash, "cleaned_text_sha256": cleaned_hash, "chapter_plan_sha256": chapter_hash, "chapter_plan_schema_version": 1, "analyzer": {"id": analyzer_id, "version": analyzer_version, "model_hash": model_hash}, "cast": cast, "aliases": aliases, "chapters": chapters, "unresolved_policy": {"mode": "narrator", "accepted_by_user": False, "accepted_at": None}, "approval": {"state": "draft", "approved_at": None, "approved_revision": None}}
    result = with_canonical_artifact_hash(artifact)
    validate_voice_plan(result, cleaned_text, chapter_plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash)
    return result


def build_draft_voice_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_voice_plan(*args, **kwargs)


def _edit_copy(artifact: Any, expected_revision: int) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise _fail("INVALID_ARTIFACT", "artifact must be an object")
    if type(expected_revision) is not int or expected_revision <= 0:
        raise _fail("INVALID_EXPECTED_REVISION", "expected_revision must be positive")
    if artifact.get("revision") != expected_revision:
        raise _fail("STALE_REVISION", "voice plan revision is stale", expected_revision=expected_revision, actual_revision=artifact.get("revision"))
    import copy
    result = copy.deepcopy(artifact)
    result["revision"] = expected_revision + 1
    result["approval"] = {"state": "draft", "approved_at": None, "approved_revision": None}
    return result


def _finish_edit(artifact: dict[str, Any]) -> dict[str, Any]:
    return with_canonical_artifact_hash(artifact)


def rename_cast(artifact: dict[str, Any], cast_id: str, display_label: str, *, expected_revision: int, actor: str = "user", reason: str = "rename") -> dict[str, Any]:
    result = _edit_copy(artifact, expected_revision)
    entries = [entry for entry in result.get("cast", []) if isinstance(entry, dict) and entry.get("cast_id") == cast_id]
    if len(entries) != 1:
        raise _fail("UNKNOWN_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    _bounded(display_label, "display_label", 512)
    entries[0]["display_label"] = display_label
    return _finish_edit(result)


rename_character = rename_cast


def assign_cast(
    artifact: dict[str, Any],
    cast_id: str,
    *,
    expected_revision: int,
    voice_id: str | None = None,
    speed: float | None = None,
    pitch_semitones: int | None = None,
    tone_preset: str | None = None,
    voice_settings: dict[str, Any] | None = None,
    relationship: str | None = None,
) -> dict[str, Any]:
    result = _edit_copy(artifact, expected_revision)
    entries = [entry for entry in result.get("cast", []) if isinstance(entry, dict) and entry.get("cast_id") == cast_id]
    if len(entries) != 1:
        raise _fail("UNKNOWN_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    entry = entries[0]
    if voice_id is not None:
        _text(voice_id, "voice_id")
        entry["voice_id"] = voice_id
    if speed is not None or pitch_semitones is not None or tone_preset is not None or voice_settings is not None:
        current = canonical_voice_settings(entry.get("voice_settings", {"speed": 1.0}))
        if voice_settings is not None:
            current = canonical_voice_settings(voice_settings)
        else:
            if speed is not None:
                current["speed"] = speed
            if pitch_semitones is not None:
                current["pitch_semitones"] = pitch_semitones
            if tone_preset is not None:
                current["tone_preset"] = tone_preset
            current = canonical_voice_settings(current, allow_legacy=False)
        entry["voice_settings"] = current
    if relationship is not None:
        if relationship not in RELATIONSHIPS:
            raise _fail("INVALID_RELATIONSHIP", "relationship is not supported")
        entry["relationship"] = relationship
        if relationship == "same_as_narrator":
            narrator = next((item for item in result.get("cast", []) if item.get("cast_id") == "narrator"), None)
            if narrator is not None:
                entry["voice_id"] = narrator.get("voice_id")
    return _finish_edit(result)


update_cast = assign_cast


def _resolve_aliases(result: dict[str, Any], alias_ids: Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(alias_ids, (str, bytes)) or not isinstance(alias_ids, Sequence) or not alias_ids:
        raise _fail("INVALID_ALIAS_IDS", "alias_ids must be a non-empty sequence")
    aliases = [item for item in result.get("aliases", []) if isinstance(item, dict) and item.get("alias_id") in alias_ids]
    if len(aliases) != len(tuple(alias_ids)):
        raise _fail("UNKNOWN_ALIAS_ID", "one or more alias IDs are unknown")
    return aliases


def merge_aliases(artifact: dict[str, Any], target_character_id: str, alias_ids: Sequence[str], *, expected_revision: int) -> dict[str, Any]:
    result = _edit_copy(artifact, expected_revision)
    target = [item for item in result.get("cast", []) if item.get("cast_id") == target_character_id and item.get("role") == "character"]
    if len(target) != 1:
        raise _fail("UNKNOWN_CAST_ID", "target character ID does not resolve uniquely")
    for alias in _resolve_aliases(result, alias_ids):
        alias["character_id"] = target_character_id
    return _finish_edit(result)


def _unique_cast_entry(result: dict[str, Any], cast_id: str, *, role: str | None = None) -> dict[str, Any]:
    entries = [
        item for item in result.get("cast", [])
        if isinstance(item, dict) and item.get("cast_id") == cast_id and (role is None or item.get("role") == role)
    ]
    if not entries:
        raise _fail("UNKNOWN_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    if len(entries) != 1:
        raise _fail("AMBIGUOUS_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    return entries[0]


def remove_cast(artifact: dict[str, Any], cast_id: str, *, expected_revision: int) -> dict[str, Any]:
    """Remove a character and route its aliases/spans to Narrator."""

    result = _edit_copy(artifact, expected_revision)
    matches = [item for item in result.get("cast", []) if isinstance(item, dict) and item.get("cast_id") == cast_id]
    if not matches:
        raise _fail("UNKNOWN_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    if len(matches) != 1:
        raise _fail("AMBIGUOUS_CAST_ID", "cast ID does not resolve uniquely", cast_id=cast_id)
    if matches[0].get("role") == "narrator" or cast_id == "narrator":
        raise _fail("CANNOT_REMOVE_NARRATOR", "the narrator cast entry cannot be removed", cast_id=cast_id)
    if matches[0].get("role") != "character":
        raise _fail("INVALID_CAST_ROLE", "only character cast entries can be removed", cast_id=cast_id)
    result["cast"] = [item for item in result.get("cast", []) if item is not matches[0]]
    result["aliases"] = [item for item in result.get("aliases", []) if item.get("character_id") != cast_id]
    for chapter in result.get("chapters", []):
        for span in chapter.get("spans", []):
            if isinstance(span, dict) and span.get("speaker_id") == cast_id:
                span["speaker_id"] = "narrator"
                span["override"] = {"kind": "speaker", "from": cast_id, "to": "narrator", "actor": "user", "reason": "cast_removed"}
    return _finish_edit(result)


def merge_cast(artifact: dict[str, Any], source_cast_id: str, target_cast_id: str, *, expected_revision: int) -> dict[str, Any]:
    """Merge one character cast entry into another, preserving the target voice."""

    result = _edit_copy(artifact, expected_revision)
    if source_cast_id == target_cast_id:
        raise _fail("CANNOT_MERGE_SELF", "a cast entry cannot be merged into itself", cast_id=source_cast_id)
    source = _unique_cast_entry(result, source_cast_id)
    target = _unique_cast_entry(result, target_cast_id)
    if source.get("role") != "character" or target.get("role") != "character":
        raise _fail("CANNOT_MERGE_NARRATOR", "the narrator cast entry cannot be merged", source_cast_id=source_cast_id, target_cast_id=target_cast_id)
    for alias in result.get("aliases", []):
        if isinstance(alias, dict) and alias.get("character_id") == source_cast_id:
            alias["character_id"] = target_cast_id
    for chapter in result.get("chapters", []):
        for span in chapter.get("spans", []):
            if isinstance(span, dict) and span.get("speaker_id") == source_cast_id:
                span["speaker_id"] = target_cast_id
                span["override"] = {"kind": "speaker", "from": source_cast_id, "to": target_cast_id, "actor": "user", "reason": "cast_merged"}
    result["cast"] = [item for item in result.get("cast", []) if item is not source]
    return _finish_edit(result)


def split_aliases(
    artifact: dict[str, Any],
    alias_ids: Sequence[str],
    *,
    expected_revision: int,
    target_character_id: str | None = None,
    new_character_id: str | None = None,
    display_label: str | None = None,
    voice_id: str | None = None,
) -> dict[str, Any]:
    result = _edit_copy(artifact, expected_revision)
    aliases = _resolve_aliases(result, alias_ids)
    if target_character_id is None:
        target_character_id = new_character_id or _safe_id(display_label or aliases[0]["text"], "character")
        if any(item.get("cast_id") == target_character_id for item in result.get("cast", [])):
            raise _fail("AMBIGUOUS_CAST_ID", "new character ID already exists", cast_id=target_character_id)
        narrator = next(item for item in result["cast"] if item.get("cast_id") == "narrator")
        result["cast"].append({"cast_id": target_character_id, "display_label": display_label or target_character_id, "role": "character", "relationship": "separate_from_narrator", "voice_id": voice_id or narrator["voice_id"], "voice_settings": canonical_voice_settings({"speed": 1.0})})
    elif not any(item.get("cast_id") == target_character_id and item.get("role") == "character" for item in result.get("cast", [])):
        raise _fail("UNKNOWN_CAST_ID", "target character ID does not resolve uniquely")
    for alias in aliases:
        alias["character_id"] = target_character_id
    return _finish_edit(result)


def override_span(
    artifact: dict[str, Any],
    span_id: str,
    *,
    expected_revision: int,
    kind: str,
    to: str,
    actor: str = "user",
    reason: str,
) -> dict[str, Any]:
    result = _edit_copy(artifact, expected_revision)
    if kind not in {"speaker", "type"}:
        raise _fail("INVALID_OVERRIDE", "override kind must be speaker or type")
    matches: list[dict[str, Any]] = []
    for chapter in result.get("chapters", []):
        matches.extend(span for span in chapter.get("spans", []) if isinstance(span, dict) and span.get("span_id") == span_id)
    if not matches:
        raise _fail("UNKNOWN_SPAN_ID", "span ID is unknown", span_id=span_id)
    if len(matches) != 1:
        raise _fail("AMBIGUOUS_SPAN_ID", "span ID is not unique", span_id=span_id)
    if kind == "speaker" and not any(item.get("cast_id") == to for item in result.get("cast", [])):
        raise _fail("UNKNOWN_CAST_ID", "override speaker ID is unknown", speaker_id=to)
    if kind == "type" and to not in speakers.SPAN_TYPES:
        raise _fail("INVALID_SPAN_TYPE", "override type is unsupported")
    span = matches[0]
    previous = span.get("speaker_id") if kind == "speaker" else span.get("type")
    span[kind if kind == "type" else "speaker_id"] = to
    span["override"] = {"kind": kind, "from": str(previous), "to": to, "actor": actor, "reason": reason}
    return _finish_edit(result)


override_span_assignment = override_span


def approve_voice_plan(
    artifact: dict[str, Any],
    cleaned_text: str,
    chapter_plan: dict[str, Any],
    *,
    expected_source_pdf_sha256: str,
    expected_chapter_plan_sha256: str,
    accept_narrator_fallback: bool = False,
    approved_at: str | None = None,
) -> dict[str, Any]:
    """Approve a draft after complete validation and optional fallback acceptance."""

    if not isinstance(artifact, dict):
        raise _fail("INVALID_APPROVAL", "only draft voice plans can be approved")
    approval = artifact.get("approval")
    if isinstance(approval, dict) and approval.get("state") == "approved":
        validate_voice_plan(artifact, cleaned_text, chapter_plan, expected_source_pdf_sha256=expected_source_pdf_sha256, expected_chapter_plan_sha256=expected_chapter_plan_sha256)
        import copy
        return copy.deepcopy(artifact)
    if not isinstance(approval, dict) or approval.get("state") != "draft":
        raise _fail("INVALID_APPROVAL", "only draft voice plans can be approved")
    validate_voice_plan(artifact, cleaned_text, chapter_plan, expected_source_pdf_sha256=expected_source_pdf_sha256, expected_chapter_plan_sha256=expected_chapter_plan_sha256)
    import copy
    result = copy.deepcopy(artifact)
    unresolved = any(span.get("type") == "unknown" for chapter in result["chapters"] for span in chapter["spans"])
    if unresolved and not accept_narrator_fallback:
        raise _fail("UNRESOLVED_SPANS", "approval requires explicit narrator fallback acceptance")
    if unresolved:
        stamp = approved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        result["unresolved_policy"] = {"mode": "narrator", "accepted_by_user": True, "accepted_at": stamp}
    elif result["unresolved_policy"]["accepted_by_user"]:
        result["unresolved_policy"]["accepted_at"] = result["unresolved_policy"].get("accepted_at") or (approved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    stamp = approved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result["approval"] = {"state": "approved", "approved_at": stamp, "approved_revision": result["revision"]}
    result = with_canonical_artifact_hash(result)
    validate_voice_plan(result, cleaned_text, chapter_plan, expected_source_pdf_sha256=expected_source_pdf_sha256, expected_chapter_plan_sha256=expected_chapter_plan_sha256)
    return result


def review_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    cast = artifact.get("cast", []) if isinstance(artifact, dict) else []
    chapters = artifact.get("chapters", []) if isinstance(artifact, dict) else []
    spans = [span for chapter in chapters if isinstance(chapter, dict) for span in chapter.get("spans", []) if isinstance(span, dict)]
    bands = {band: 0 for band in speakers.CONFIDENCE_BANDS}
    for span in spans:
        band = span.get("confidence", {}).get("band")
        if band in bands:
            bands[band] += 1
    return {"cast_count": len(cast), "span_count": len(spans), "confidence": bands, "unresolved_count": sum(span.get("type") == "unknown" for span in spans), "override_count": sum(span.get("override") is not None for span in spans), "revision": artifact.get("revision"), "approval_state": artifact.get("approval", {}).get("state")}


__all__ = [
    "MIN_CHARACTER_QUOTE_COUNT",
    "assign_cast",
    "approve_voice_plan",
    "build_draft_voice_plan",
    "build_voice_plan",
    "CastEntry",
    "VoicePlanError",
    "canonical_artifact_digest",
    "canonical_json_bytes",
    "canonical_json_text",
    "merge_aliases",
    "merge_cast",
    "override_span",
    "override_span_assignment",
    "rename_cast",
    "rename_character",
    "remove_cast",
    "review_summary",
    "split_aliases",
    "validate_voice_plan",
    "validate_voice_plan_core",
    "verify_canonical_artifact_hash",
    "with_canonical_artifact_hash",
]

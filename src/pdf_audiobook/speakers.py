"""Analyzer-neutral, exact-text speaker span validation.

This module deliberately knows nothing about a particular speaker analyzer,
HTTP endpoint, persistence format, or synthesis worker.  Machine spans are
suggestions and may have gaps; an approved plan is the stricter, complete
representation used by later generation work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


SPAN_TYPES = frozenset({"narration", "dialogue", "thought", "unknown"})
CONFIDENCE_BANDS = ("low", "medium", "high")
MACHINE_WARNING_LIMIT = 8 * 1024


class SpeakerPlanError(ValueError):
    """Stable validation failure suitable for an API or artifact boundary."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> SpeakerPlanError:
    return SpeakerPlanError(code, message, details=details)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _validate_text(cleaned_text: Any) -> str:
    if not isinstance(cleaned_text, str):
        raise _fail("INVALID_TEXT", "cleaned_text must be a string")
    return cleaned_text


def _normal_strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail("INVALID_" + name.upper(), f"{name} must be an ordered sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item or any(ord(char) < 32 for char in item) for item in result):
        raise _fail("INVALID_" + name.upper(), f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class Confidence:
    """Immutable numeric confidence and analyzer-provided coarse band."""

    score: float
    band: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise _fail("INVALID_CONFIDENCE", "confidence score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise _fail("INVALID_CONFIDENCE", "confidence score must be finite and in [0, 1]")
        if not isinstance(self.band, str) or self.band not in CONFIDENCE_BANDS:
            raise _fail("INVALID_CONFIDENCE_BAND", "confidence band must be low, medium, or high")
        reasons = _normal_strings(self.reasons, "reasons") if self.reasons else ()
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "band", self.band)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class SpeakerSpan:
    """An exact half-open source range and its speaker metadata."""

    span_id: str
    chapter_index: int
    source_start: int
    source_end: int
    span_type: str
    speaker_id: str | None
    confidence: Confidence
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id or any(ord(char) < 32 for char in self.span_id):
            raise _fail("INVALID_SPAN_ID", "span ID must be a non-empty string")
        if not _is_int(self.chapter_index) or self.chapter_index < 1:
            raise _fail("INVALID_CHAPTER_INDEX", "chapter index must be a one-based integer")
        if not _is_int(self.source_start) or not _is_int(self.source_end) or self.source_start < 0 or self.source_end <= self.source_start:
            raise _fail("INVALID_RANGE", "span source range must be a non-empty half-open range")
        if not isinstance(self.span_type, str) or self.span_type not in SPAN_TYPES:
            raise _fail("INVALID_SPAN_TYPE", "span type is not supported")
        if self.speaker_id is not None and (not isinstance(self.speaker_id, str) or not self.speaker_id or any(ord(char) < 32 for char in self.speaker_id)):
            raise _fail("INVALID_SPEAKER_ID", "speaker ID must be null or a non-empty string")
        if not isinstance(self.confidence, Confidence):
            raise _fail("INVALID_CONFIDENCE", "span confidence must be a Confidence value")
        object.__setattr__(self, "provenance", _normal_strings(self.provenance, "provenance") if self.provenance else ())


@dataclass(frozen=True, slots=True)
class MachineAnalysis:
    """Optional convenience result for whole-book analyzer adapters."""

    spans: tuple[SpeakerSpan, ...]
    source_hash: str | None = None
    provenance: tuple[str, ...] = ()
    characters: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.spans, (str, bytes)):
            raise _fail("INVALID_ANALYSIS", "analysis spans must be a sequence")
        spans = tuple(self.spans)
        if any(not isinstance(span, SpeakerSpan) for span in spans):
            raise _fail("INVALID_ANALYSIS", "analysis spans must contain SpeakerSpan values")
        provenance = _normal_strings(self.provenance, "provenance") if self.provenance else ()
        if isinstance(self.characters, (str, bytes)) or not isinstance(self.characters, Sequence):
            raise _fail("INVALID_CHARACTERS", "analysis characters must be an ordered sequence")
        characters = tuple(dict(item) if isinstance(item, dict) else item for item in self.characters)
        if any(not isinstance(character, dict) for character in characters):
            raise _fail("INVALID_CHARACTERS", "analysis characters must contain objects")
        if isinstance(self.warnings, (str, bytes)) or not isinstance(self.warnings, Sequence):
            raise _fail("INVALID_WARNINGS", "analysis warnings must be an ordered sequence")
        warnings = tuple(self.warnings)
        if any(
            not isinstance(warning, str)
            or len(warning) > MACHINE_WARNING_LIMIT
            or any(ord(char) < 32 for char in warning)
            for warning in warnings
        ):
            raise _fail("INVALID_WARNINGS", "analysis warnings must be bounded strings without controls")
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "characters", characters)
        object.__setattr__(self, "warnings", warnings)


@runtime_checkable
class SpeakerAnalyzer(Protocol):
    """Whole-book analyzer boundary; implementations may be in-process or isolated."""

    def analyze(
        self,
        cleaned_text: str,
        chapter_plan: dict[str, Any],
        source_hash: str,
        options: Any = None,
    ) -> MachineAnalysis | Sequence[SpeakerSpan]:
        ...


def _chapter_ranges(chapter_plan: Any, text_length: int) -> dict[int, tuple[int, int]]:
    if not isinstance(chapter_plan, dict) or not isinstance(chapter_plan.get("chapters"), list) or not chapter_plan["chapters"]:
        raise _fail("INVALID_CHAPTER_PLAN", "chapter plan must contain a non-empty chapters list")
    ranges: dict[int, tuple[int, int]] = {}
    expected_index = 1
    expected_start = 0
    for chapter in chapter_plan["chapters"]:
        if not isinstance(chapter, dict):
            raise _fail("INVALID_CHAPTER_PLAN", "chapter entries must be objects")
        index, start, end = chapter.get("index"), chapter.get("start_offset"), chapter.get("end_offset")
        if not _is_int(index) or index != expected_index:
            raise _fail("INVALID_CHAPTER_PLAN", "chapter indices must be ordered from one")
        if not _is_int(start) or not _is_int(end) or start != expected_start or end <= start or end > text_length:
            raise _fail("INVALID_CHAPTER_PLAN", "chapter offsets must be contiguous and in bounds")
        ranges[index] = (start, end)
        expected_index += 1
        expected_start = end
    if expected_start != text_length:
        raise _fail("INVALID_CHAPTER_PLAN", "chapters must cover the complete cleaned text")
    return ranges


def _span_sequence(spans: Any) -> tuple[SpeakerSpan, ...]:
    if isinstance(spans, MachineAnalysis):
        spans = spans.spans
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise _fail("INVALID_SPANS", "spans must be a sequence of SpeakerSpan values")
    result = tuple(spans)
    if any(not isinstance(span, SpeakerSpan) for span in result):
        raise _fail("INVALID_SPAN", "spans must contain SpeakerSpan values")
    return result


def _validate_common(spans: tuple[SpeakerSpan, ...], cleaned_text: str, chapter_plan: Any) -> dict[int, tuple[int, int]]:
    ranges = _chapter_ranges(chapter_plan, len(cleaned_text))
    seen_ids: set[str] = set()
    previous_start = -1
    previous_end = -1
    for span in spans:
        if span.span_id in seen_ids:
            raise _fail("DUPLICATE_SPAN_ID", "span IDs must be unique", span_id=span.span_id)
        seen_ids.add(span.span_id)
        if span.source_start < previous_start:
            raise _fail("UNORDERED_SPANS", "spans must be ordered by source start")
        if span.source_start < previous_end:
            raise _fail("OVERLAPPING_SPANS", "spans must not overlap")
        previous_start, previous_end = span.source_start, span.source_end
        if span.source_end > len(cleaned_text):
            raise _fail("OUT_OF_RANGE", "span source range exceeds cleaned text")
        chapter_range = ranges.get(span.chapter_index)
        if chapter_range is None:
            raise _fail("INVALID_CHAPTER_INDEX", "span chapter index is not in the chapter plan")
        if span.source_start < chapter_range[0] or span.source_end > chapter_range[1]:
            raise _fail("CROSS_CHAPTER", "span must be wholly contained in its chapter")
    return ranges


def validate_machine_spans(spans: Any, cleaned_text: str, chapter_plan: Any) -> tuple[SpeakerSpan, ...]:
    """Validate incomplete analyzer suggestions without requiring coverage."""

    text = _validate_text(cleaned_text)
    values = _span_sequence(spans)
    _validate_common(values, text, chapter_plan)
    return values


def validate_approved_spans(
    spans: Any,
    cleaned_text: str,
    chapter_plan: Any,
    *,
    narrator_fallback_accepted: bool = False,
    allow_unresolved: bool = False,
) -> tuple[SpeakerSpan, ...]:
    """Validate a complete approved plan and its exact text reconstruction."""

    if not isinstance(narrator_fallback_accepted, bool):
        raise _fail("INVALID_FALLBACK_POLICY", "narrator fallback acceptance must be boolean")
    text = _validate_text(cleaned_text)
    values = _span_sequence(spans)
    ranges = _validate_common(values, text, chapter_plan)
    expected_position = 0
    expected_chapter = 1
    chapter_fragments: dict[int, list[str]] = {index: [] for index in ranges}
    for span in values:
        if span.chapter_index != expected_chapter:
            raise _fail("INCOMPLETE_COVERAGE", "approved spans must cover chapters in order", chapter_index=span.chapter_index)
        chapter_end = ranges[span.chapter_index][1]
        if span.source_start != expected_position:
            raise _fail("INCOMPLETE_COVERAGE", "approved spans must be contiguous and gap-free")
        if span.span_type == "unknown":
            if not narrator_fallback_accepted:
                if not allow_unresolved:
                    raise _fail("UNRESOLVED_SPANS", "unknown spans require explicit narrator fallback acceptance")
            elif span.speaker_id != "narrator":
                raise _fail("UNRESOLVED_SPANS", "accepted unknown spans must be explicitly assigned to narrator")
        if span.speaker_id is None and not allow_unresolved:
            raise _fail("UNASSIGNED_SPEAKER", "approved spans require an assigned speaker")
        chapter_fragments[span.chapter_index].append(text[span.source_start : span.source_end])
        expected_position = span.source_end
        if expected_position == chapter_end:
            expected_chapter += 1
            expected_position = chapter_end
    if expected_position != len(text) or expected_chapter != len(ranges) + 1:
        raise _fail("INCOMPLETE_COVERAGE", "approved spans must cover every chapter and the complete book")
    for chapter_index, (start, end) in ranges.items():
        if "".join(chapter_fragments[chapter_index]) != text[start:end]:
            raise _fail("RECONSTRUCTION_MISMATCH", "approved span text does not exactly reconstruct its chapter")
    if "".join(text[span.source_start : span.source_end] for span in values) != text:
        raise _fail("RECONSTRUCTION_MISMATCH", "approved spans do not exactly reconstruct cleaned text")
    return values


def validate_draft_spans(
    spans: Any,
    cleaned_text: str,
    chapter_plan: Any,
) -> tuple[SpeakerSpan, ...]:
    """Validate complete draft coverage while retaining unresolved spans."""

    return validate_approved_spans(spans, cleaned_text, chapter_plan, allow_unresolved=True)


__all__ = [
    "CONFIDENCE_BANDS",
    "SPAN_TYPES",
    "Confidence",
    "MachineAnalysis",
    "SpeakerAnalyzer",
    "SpeakerPlanError",
    "SpeakerSpan",
    "validate_approved_spans",
    "validate_draft_spans",
    "validate_machine_spans",
]

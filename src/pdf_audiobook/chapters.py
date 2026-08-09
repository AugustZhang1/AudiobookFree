"""Pure, deterministic chapter planning over Phase 2 review artifacts."""

from __future__ import annotations

import copy
from bisect import bisect_left
import hashlib
import re
from typing import Any


SCHEMA_VERSION = 1
SHORT_CHAPTER_WORDS = 250
MIN_CUSTOM_COUNT = 2
MAX_CUSTOM_COUNT = 50

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’‑-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")
_HEADING = re.compile(r"^(?:chapter|part|section)\b(?:\s+.*)?$", re.IGNORECASE)
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "e.g", "i.e"}


class ChapterPlanError(ValueError):
    """Stable planner failure with JSON-safe details."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> ChapterPlanError:
    return ChapterPlanError(code, message, details=details)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_starts(text: str) -> list[int]:
    """Build the ordered word index once for Custom boundary scoring."""

    return [match.start() for match in _WORD.finditer(text)]


def _validate_inputs(cleaned_text: str, cleaned_map: Any) -> list[dict[str, int]]:
    if not isinstance(cleaned_text, str) or not cleaned_text:
        raise _fail("INVALID_TEXT", "cleaned_text must be a non-empty string")
    if not isinstance(cleaned_map, list) or not cleaned_map:
        raise _fail("INVALID_MAPPING", "cleaned_map must be a non-empty list")
    result: list[dict[str, int]] = []
    previous_start = -1
    previous_end = 0
    for item in cleaned_map:
        if not isinstance(item, dict):
            raise _fail("INVALID_MAPPING", "each cleaned-map entry must be an object")
        try:
            source_page = item["source_page"]
            start = item["cleaned_start"]
            end = item["cleaned_end"]
        except KeyError as exc:
            raise _fail("INVALID_MAPPING", "cleaned-map entries require source_page, cleaned_start, and cleaned_end") from exc
        if any(type(value) is not int for value in (source_page, start, end)) or source_page < 1 or start < 0 or end <= start or end > len(cleaned_text):
            raise _fail("INVALID_MAPPING", "cleaned-map ranges are invalid")
        if start < previous_start or start < previous_end:
            raise _fail("INVALID_MAPPING", "cleaned-map ranges must be ordered and non-overlapping")
        previous_start = start
        previous_end = end
        result.append({"source_page": source_page, "cleaned_start": start, "cleaned_end": end})
    if result[0]["cleaned_start"] != 0 or result[-1]["cleaned_end"] != len(cleaned_text):
        raise _fail("INVALID_MAPPING", "cleaned-map must cover the cleaned text")
    return result


def _segments_for_page(mapping: list[dict[str, int]], page: int) -> tuple[int, int] | None:
    segments = [item for item in mapping if item["source_page"] == page]
    if not segments:
        return None
    return min(item["cleaned_start"] for item in segments), max(item["cleaned_end"] for item in segments)


def _page_at(mapping: list[dict[str, int]], offset: int) -> int:
    if offset >= mapping[-1]["cleaned_end"]:
        return mapping[-1]["source_page"]
    for item in mapping:
        if item["cleaned_start"] <= offset < item["cleaned_end"]:
            return item["source_page"]
    nearest = min(mapping, key=lambda item: min(abs(offset - item["cleaned_start"]), abs(offset - item["cleaned_end"])))
    return nearest["source_page"]


def _page_range(mapping: list[dict[str, int]], start: int, end: int) -> tuple[int, int]:
    return _page_at(mapping, start), _page_at(mapping, max(start, end - 1))


def _source_type(value: Any) -> str:
    source = str(value or "").lower()
    if "outline" in source or "bookmark" in source:
        return "outline"
    if source in {"heading", "explicit"}:
        return source
    if "heading" in source:
        return "heading"
    if "layout" in source:
        return "layout"
    return "candidate"


def _candidate_source(candidate: dict[str, Any]) -> Any:
    return candidate.get("source_type", candidate.get("source"))


def _family_rank(source_type: str) -> int:
    if source_type == "outline":
        return 0
    if source_type in {"heading", "explicit"}:
        return 1
    if source_type == "layout":
        return 2
    return 3


def _document_title(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _fail("INVALID_TITLE", "document_title must be a string")
    return value.strip()


def _candidate_offset(text: str, mapping: list[dict[str, int]], candidate: Any) -> int | None:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("title"), str):
        return None
    title = candidate["title"].strip()
    if not title:
        return None
    direct = candidate.get("cleaned_offset", candidate.get("offset"))
    if type(direct) is int and 0 <= direct < len(text) and text[direct : direct + len(title)].casefold() == title.casefold() and _validate_boundary_safety(text, mapping, direct, allow_line_start=True):
        return direct
    page = candidate.get("source_page")
    if type(page) is not int or page < 1:
        return None
    segment = _segments_for_page(mapping, page)
    if segment is None:
        return None
    start, end = segment
    if _source_type(_candidate_source(candidate)) == "outline":
        return start
    found = text.casefold().find(title.casefold(), start, end)
    if found >= 0 and (_validate_boundary_safety(text, mapping, found) or _line_start(text, found)):
        return found
    # Tolerate title punctuation/whitespace differences while remaining page-local.
    compact_title = re.sub(r"\s+", " ", title).casefold()
    for line_match in re.finditer(r"[^\n]+", text[start:end]):
        line = re.sub(r"\s+", " ", line_match.group(0).strip()).casefold()
        if line == compact_title or compact_title in line:
            offset = start + line_match.start() + len(line_match.group(0)) - len(line_match.group(0).lstrip())
            if _validate_boundary_safety(text, mapping, offset) or _line_start(text, offset):
                return offset
    return None


def _sentence_boundaries(text: str) -> set[int]:
    boundaries: set[int] = set()
    for match in re.finditer(r"\n\s*\n+", text):
        boundaries.add(match.end())
    for index, char in enumerate(text):
        if char not in ".?!":
            continue
        next_index = index + 1
        while next_index < len(text) and text[next_index] in "\"'”’)]}":
            next_index += 1
        if next_index < len(text) and not text[next_index].isspace():
            continue
        if char == ".":
            token_start = index - 1
            while token_start >= 0 and (text[token_start].isalpha() or text[token_start] == "."):
                token_start -= 1
            token = text[token_start + 1 : index].lower()
            if index and next_index < len(text) and text[index - 1].isdigit() and next_index < len(text) and text[next_index].isdigit():
                continue
            multi_initial = bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token))
            if token in _ABBREVIATIONS or multi_initial or (len(token) == 1 and token.isalpha()):
                continue
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        boundaries.add(next_index)
    return boundaries


def _heading_boundaries(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for match in re.finditer(r"(?m)^[^\n]+", text):
        line = match.group(0).strip()
        if _HEADING.fullmatch(line) or (line.isupper() and 2 <= len(_WORD.findall(line)) <= 10):
            result[match.start() + len(match.group(0)) - len(match.group(0).lstrip())] = line
    return result


def _line_start(text: str, offset: int) -> bool:
    if offset <= 0:
        return offset == 0
    line_start = text.rfind("\n", 0, offset) + 1
    return not text[line_start:offset].strip()


def _safe_custom_candidates(text: str, mapping: list[dict[str, int]], supplied: Any) -> list[tuple[int, int, str | None, str]]:
    headings = _heading_boundaries(text)
    sentences = _sentence_boundaries(text)
    candidates: dict[int, tuple[int, str | None, str]] = {
        offset: (0, title, "heading") for offset, title in headings.items() if 0 < offset < len(text)
    }
    if isinstance(supplied, list):
        for candidate in supplied:
            if not isinstance(candidate, dict):
                continue
            offset = _candidate_offset(text, mapping, candidate)
            if offset is None or not 0 < offset < len(text):
                continue
            title = candidate.get("title") if isinstance(candidate.get("title"), str) else None
            source_type = _source_type(_candidate_source(candidate))
            candidates[offset] = (0, title, source_type)
    for offset in sentences:
        if 0 < offset < len(text):
            existing = candidates.get(offset)
            if existing is None:
                candidates[offset] = (2, None, "custom")
    # Paragraph starts are preferable to sentence boundaries and are safe.
    for match in re.finditer(r"\n\s*\n+", text):
        offset = match.end()
        if 0 < offset < len(text):
            existing = candidates.get(offset)
            if existing is None or existing[0] > 1:
                candidates[offset] = (1, None, "custom")
    return [(offset, priority, title, source_type) for offset, (priority, title, source_type) in sorted(candidates.items())]


def _word_position(word_starts: list[int], offset: int) -> int:
    return bisect_left(word_starts, offset)


def _validate_boundary_safety(text: str, mapping: list[dict[str, int]], offset: int, *, allow_line_start: bool = False, sentence_boundaries: set[int] | None = None, heading_boundaries: dict[int, str] | None = None) -> bool:
    sentence_boundaries = _sentence_boundaries(text) if sentence_boundaries is None else sentence_boundaries
    heading_boundaries = _heading_boundaries(text) if heading_boundaries is None else heading_boundaries
    if offset in {0, len(text)} or offset in sentence_boundaries or offset in heading_boundaries or (allow_line_start and _line_start(text, offset)):
        return True
    return any(item["cleaned_start"] == offset for item in mapping)


def _build_chapter(text: str, mapping: list[dict[str, int]], index: int, start: int, end: int, title: str, source_type: str) -> dict[str, Any]:
    start_page, end_page = _page_range(mapping, start, end)
    return {
        "index": index,
        "title": title,
        "start_offset": start,
        "end_offset": end,
        "start_page": start_page,
        "end_page": end_page,
        "source_type": source_type,
        "word_count": len(_WORD.findall(text[start:end])),
    }


def _assemble(text: str, mapping: list[dict[str, int]], boundaries: list[tuple[int, str, str]], warnings: list[str]) -> dict[str, Any]:
    chapters = []
    for index, (start, title, source_type) in enumerate(boundaries, 1):
        end = boundaries[index][0] if index < len(boundaries) else len(text)
        chapters.append(_build_chapter(text, mapping, index, start, end, title, source_type))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": "original",
        "requested_count": None,
        "cleaned_text_sha256": _hash(text),
        "chapters": chapters,
        "warnings": list(warnings),
    }
    return plan


def _original_plan(text: str, mapping: list[dict[str, int]], candidates: Any, document_title: str | None) -> dict[str, Any]:
    if not isinstance(candidates, list):
        raise _fail("INVALID_CANDIDATES", "chapter_candidates must be a list")
    resolved_candidates: list[tuple[int, str, str, int]] = []
    for candidate in candidates:
        offset = _candidate_offset(text, mapping, candidate)
        if offset is None or offset < 0 or offset >= len(text):
            continue
        title = str(candidate.get("title", "")).strip()
        if not title:
            continue
        source_type = _source_type(_candidate_source(candidate))
        resolved_candidates.append((offset, title, source_type, _family_rank(source_type)))
    best_family = min((item[3] for item in resolved_candidates), default=None)
    resolved: dict[int, tuple[str, str]] = {}
    if best_family is not None:
        for offset, title, source_type, family in resolved_candidates:
            if family != best_family:
                continue
            existing = resolved.get(offset)
            if existing is None or (source_type, title) < (_source_type(existing[1]), existing[0]):
                resolved[offset] = (title, source_type)
    if not resolved:
        title = _document_title(document_title) or "Whole Book"
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "original",
            "requested_count": None,
            "cleaned_text_sha256": _hash(text),
            "chapters": [_build_chapter(text, mapping, 1, 0, len(text), title, "whole")],
            "warnings": ["No reliable chapter candidates; using Whole Book."],
        }
    ordered = sorted(resolved.items())
    boundaries: list[tuple[int, str, str]] = []
    for offset, (title, source_type) in ordered:
        if boundaries and boundaries[-1][0] == offset:
            continue
        boundaries.append((offset, title, source_type))
    # Front matter belongs to the first titled chapter, preserving complete coverage.
    first_offset, first_title, first_source = boundaries[0]
    boundaries[0] = (0, first_title, first_source)
    return _assemble(text, mapping, boundaries, [])


def _custom_plan(text: str, mapping: list[dict[str, int]], candidates_input: Any, count: int) -> dict[str, Any]:
    if type(count) is not int or count < MIN_CUSTOM_COUNT or count > MAX_CUSTOM_COUNT:
        raise _fail("INVALID_COUNT", "custom count must be an integer from 2 through 50")
    candidates = _safe_custom_candidates(text, mapping, candidates_input)
    if len(candidates) < count - 1:
        raise _fail("COUNT_TOO_HIGH", "The requested chapter count cannot be reached safely.", recommended_maximum=len(candidates) + 1, mode="whole")
    word_starts = _word_starts(text)
    total_words = len(word_starts)
    selected: list[tuple[int, int, str | None, str]] = []
    previous_index = -1
    for chapter_number in range(1, count):
        target = total_words * chapter_number / count
        available = []
        for candidate_index, (offset, priority, title, source_type) in enumerate(candidates):
            if candidate_index <= previous_index:
                continue
            if len(candidates) - candidate_index - 1 < count - chapter_number - 1:
                continue
            available.append((abs(_word_position(word_starts, offset) - target), priority, offset, candidate_index, title, source_type))
        if not available:
            raise _fail("COUNT_TOO_HIGH", "The requested chapter count cannot be reached safely.", recommended_maximum=chapter_number, mode="whole")
        window = max(10, min(total_words / count * 0.5, 40))
        near = [item for item in available if item[0] <= window]
        preferred = [item for item in near if item[1] == 0]
        chosen_pool = preferred or near or available
        _, priority, offset, previous_index, title, source_type = min(chosen_pool, key=lambda item: (item[1], item[0], item[2]))
        selected.append((offset, previous_index, title, source_type))
    headings = _heading_boundaries(text)
    first_title = headings.get(0)
    first_source = "heading" if first_title else "custom"
    if isinstance(candidates_input, list):
        for candidate in candidates_input:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("title"), str):
                continue
            offset = _candidate_offset(text, mapping, candidate)
            if offset == 0:
                first_title = candidate["title"].strip() or first_title
                first_source = _source_type(_candidate_source(candidate))
                break
    starts = [0] + [offset for offset, _, _, _ in selected]
    boundaries: list[tuple[int, str, str]] = []
    for index, start in enumerate(starts, 1):
        selected_item = next((item for item in selected if item[0] == start), None)
        title = selected_item[2] if selected_item and selected_item[2] else ((first_title or f"Chapter {index:02d}") if index == 1 else headings.get(start, f"Chapter {index:02d}"))
        source_type = selected_item[3] if selected_item else (first_source if index == 1 else ("heading" if start in headings else "custom"))
        boundaries.append((start, title, source_type))
    warnings: list[str] = []
    if total_words / count < SHORT_CHAPTER_WORDS:
        recommended = total_words // SHORT_CHAPTER_WORDS
        if recommended >= MIN_CUSTOM_COUNT:
            recommendation = f"Recommended lower count: {recommended}."
        else:
            recommendation = "Recommended: Whole Book (one chapter)."
        warnings.append(f"Some chapters average fewer than {SHORT_CHAPTER_WORDS} words. {recommendation}")
    plan = _assemble(text, mapping, boundaries, warnings)
    plan["mode"] = "custom"
    plan["requested_count"] = count
    return plan


def create_chapter_plan(cleaned_text: str, cleaned_map: Any, chapter_candidates: Any, *, mode: str, count: int | None = None, document_title: str | None = None) -> dict[str, Any]:
    """Create a deterministic original, custom, or whole-book plan."""

    mapping = _validate_inputs(cleaned_text, cleaned_map)
    if mode == "whole":
        title = _document_title(document_title) or "Whole Book"
        plan = {
            "schema_version": SCHEMA_VERSION,
            "mode": "whole",
            "requested_count": None,
            "cleaned_text_sha256": _hash(cleaned_text),
            "chapters": [_build_chapter(cleaned_text, mapping, 1, 0, len(cleaned_text), title, "whole")],
            "warnings": [],
        }
    elif mode == "original":
        plan = _original_plan(cleaned_text, mapping, chapter_candidates, document_title)
    elif mode == "custom":
        plan = _custom_plan(cleaned_text, mapping, chapter_candidates, count if count is not None else 0)
    else:
        raise _fail("INVALID_MODE", "mode must be original, custom, or whole")
    return validate_chapter_plan(plan, cleaned_text, mapping)


def validate_chapter_plan(plan: Any, cleaned_text: str, cleaned_map: Any) -> dict[str, Any]:
    """Strictly validate plan schema, coverage, hashes, pages, and safe boundaries."""

    mapping = _validate_inputs(cleaned_text, cleaned_map)
    sentence_boundaries = _sentence_boundaries(cleaned_text)
    heading_boundaries = _heading_boundaries(cleaned_text)
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "mode", "requested_count", "cleaned_text_sha256", "chapters", "warnings"}:
        raise _fail("INVALID_PLAN", "chapter plan schema mismatch")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != SCHEMA_VERSION:
        raise _fail("INVALID_PLAN", "unsupported chapter plan schema")
    if not isinstance(plan["mode"], str) or plan["mode"] not in {"original", "custom", "whole"}:
        raise _fail("INVALID_PLAN", "invalid chapter plan mode")
    if plan["cleaned_text_sha256"] != _hash(cleaned_text):
        raise _fail("HASH_MISMATCH", "chapter plan cleaned-text hash does not match")
    if not isinstance(plan["warnings"], list) or any(not isinstance(item, str) for item in plan["warnings"]):
        raise _fail("INVALID_PLAN", "warnings must be a list of strings")
    chapters = plan["chapters"]
    if not isinstance(chapters, list) or not chapters:
        raise _fail("INVALID_PLAN", "chapters must be a non-empty list")
    requested = plan["requested_count"]
    if plan["mode"] == "custom":
        if type(requested) is not int or not MIN_CUSTOM_COUNT <= requested <= MAX_CUSTOM_COUNT or len(chapters) != requested:
            raise _fail("INVALID_PLAN", "custom plan count does not match chapters")
    elif requested is not None or (plan["mode"] == "whole" and len(chapters) != 1):
        raise _fail("INVALID_PLAN", "requested_count is invalid for this mode")
    chapter_fields = {"index", "title", "start_offset", "end_offset", "start_page", "end_page", "source_type", "word_count"}
    expected_start = 0
    for index, chapter in enumerate(chapters, 1):
        if not isinstance(chapter, dict) or set(chapter) != chapter_fields:
            raise _fail("INVALID_PLAN", "chapter schema mismatch")
        if chapter["index"] != index or type(chapter["index"]) is not int:
            raise _fail("INVALID_PLAN", "chapter indices must be ordered from one")
        if not isinstance(chapter["title"], str) or not chapter["title"].strip() or any(ord(char) < 32 for char in chapter["title"]):
            raise _fail("INVALID_PLAN", "chapter title is invalid")
        if not isinstance(chapter["source_type"], str) or not chapter["source_type"]:
            raise _fail("INVALID_PLAN", "chapter source_type is invalid")
        starts, ends = chapter["start_offset"], chapter["end_offset"]
        if any(type(value) is not int for value in (starts, ends, chapter["start_page"], chapter["end_page"], chapter["word_count"])) or starts != expected_start or ends <= starts or ends > len(cleaned_text):
            raise _fail("INVALID_PLAN", "chapter offsets are not contiguous and positive")
        trusted_line_start = chapter["source_type"] in {"outline", "heading", "explicit", "layout"}
        if not _validate_boundary_safety(cleaned_text, mapping, starts, allow_line_start=trusted_line_start, sentence_boundaries=sentence_boundaries, heading_boundaries=heading_boundaries):
            raise _fail("UNSAFE_BOUNDARY", "chapter boundary would split a sentence")
        start_page, end_page = _page_range(mapping, starts, ends)
        if chapter["start_page"] != start_page or chapter["end_page"] != end_page or chapter["start_page"] < 1 or chapter["end_page"] < chapter["start_page"]:
            raise _fail("INVALID_PLAN", "chapter page range does not match cleaned-map")
        if chapter["word_count"] != len(_WORD.findall(cleaned_text[starts:ends])):
            raise _fail("INVALID_PLAN", "chapter word count does not match text")
        expected_start = ends
    if expected_start != len(cleaned_text):
        raise _fail("INVALID_PLAN", "chapters do not cover the complete cleaned text")
    return plan


def select_chapter_range(plan: Any, start: int | None = None, end: int | None = None) -> list[dict[str, Any]]:
    """Return a deep-copied, one-based chapter selection for downstream work.

    The persisted plan always remains the complete text-integrity artifact.  A
    selected range is therefore represented only as a reindexed list consumed
    by chunking and assembly.  ``None`` endpoints mean the corresponding edge
    of the current plan, which keeps legacy full-plan callers compatible.
    """

    _validate_plan_shape(plan)
    chapters = plan["chapters"]
    total = len(chapters)
    start = 1 if start is None else start
    end = total if end is None else end
    if type(start) is not int or type(end) is not int:
        raise _fail("INVALID_CHAPTER_RANGE", "chapter range endpoints must be integers")
    if start < 1:
        raise _fail("INVALID_CHAPTER_RANGE", "chapter range start must be at least 1")
    if end < start:
        raise _fail("INVALID_CHAPTER_RANGE", "chapter range end must not precede start")
    if end > total:
        raise _fail("INVALID_CHAPTER_RANGE", "chapter range end exceeds the chapter plan", chapter_count=total)
    selected = copy.deepcopy(chapters[start - 1 : end])
    for index, chapter in enumerate(selected, 1):
        chapter["index"] = index
    return selected


def _validate_plan_shape(plan: Any) -> None:
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "mode", "requested_count", "cleaned_text_sha256", "chapters", "warnings"}:
        raise _fail("INVALID_PLAN", "chapter plan schema mismatch")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != SCHEMA_VERSION or not isinstance(plan["mode"], str) or plan["mode"] not in {"original", "custom", "whole"}:
        raise _fail("INVALID_PLAN", "invalid chapter plan schema or mode")
    if not isinstance(plan["cleaned_text_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", plan["cleaned_text_sha256"]):
        raise _fail("INVALID_PLAN", "cleaned-text hash is invalid")
    if not isinstance(plan["warnings"], list) or any(not isinstance(item, str) for item in plan["warnings"]):
        raise _fail("INVALID_PLAN", "warnings must be a list of strings")
    chapters = plan["chapters"]
    if not isinstance(chapters, list) or not chapters:
        raise _fail("INVALID_PLAN", "chapters must be a non-empty list")
    if plan["mode"] == "custom":
        if type(plan["requested_count"]) is not int or not MIN_CUSTOM_COUNT <= plan["requested_count"] <= MAX_CUSTOM_COUNT or len(chapters) != plan["requested_count"]:
            raise _fail("INVALID_PLAN", "custom plan count does not match chapters")
    elif plan["requested_count"] is not None or (plan["mode"] == "whole" and len(chapters) != 1):
        raise _fail("INVALID_PLAN", "requested_count is invalid for this mode")
    fields = {"index", "title", "start_offset", "end_offset", "start_page", "end_page", "source_type", "word_count"}
    expected = 0
    for index, chapter in enumerate(chapters, 1):
        if not isinstance(chapter, dict) or set(chapter) != fields or chapter["index"] != index:
            raise _fail("INVALID_PLAN", "chapter schema or index is invalid")
        if not isinstance(chapter["title"], str) or not chapter["title"].strip() or any(ord(char) < 32 for char in chapter["title"]):
            raise _fail("INVALID_PLAN", "chapter title is invalid")
        if not isinstance(chapter["source_type"], str) or not chapter["source_type"]:
            raise _fail("INVALID_PLAN", "chapter source_type is invalid")
        start, end = chapter["start_offset"], chapter["end_offset"]
        if any(type(chapter[key]) is not int for key in ("index", "start_offset", "end_offset", "start_page", "end_page", "word_count")) or start != expected or end <= start or chapter["start_page"] < 1 or chapter["end_page"] < chapter["start_page"] or chapter["word_count"] < 0:
            raise _fail("INVALID_PLAN", "chapter shape is invalid")
        expected = end


def rename_chapters(plan: dict[str, Any], titles: Any) -> dict[str, Any]:
    """Return a plan with only validated chapter titles changed."""

    _validate_plan_shape(plan)
    if not isinstance(titles, list) or len(titles) != len(plan["chapters"]):
        raise _fail("INVALID_TITLES", "exactly one title is required for each chapter")
    replacement = copy.deepcopy(plan)
    for index, title in enumerate(titles):
        if not isinstance(title, str):
            raise _fail("INVALID_TITLES", "chapter titles must be strings")
        title = title.strip()
        if not title or len(title) > 200 or any(ord(char) < 32 for char in title):
            raise _fail("INVALID_TITLES", "chapter titles must be non-empty, at most 200 characters, and free of controls")
        replacement["chapters"][index]["title"] = title
    return replacement


__all__ = ["ChapterPlanError", "create_chapter_plan", "rename_chapters", "select_chapter_range", "validate_chapter_plan"]

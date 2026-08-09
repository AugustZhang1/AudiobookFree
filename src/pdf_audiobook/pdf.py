"""Deterministic, local-only PDF validation, extraction, and review analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
from typing import Any
import unicodedata

import pdfplumber
from pypdf import PdfReader


MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PAGES = 2_000
REQUIRED_DISK_MULTIPLIER = 3
REQUIRED_DISK_RESERVE = 256 * 1024 * 1024
WORDS_PER_MINUTE = 150
PREVIEW_CHARACTERS = 4_000
# Mixed pages below this selectable-word count require OCR for safe narration.
MIXED_MIN_WORDS = 5

ERROR_INVALID_SIGNATURE = "INVALID_SIGNATURE"
ERROR_SIZE_LIMIT = "SIZE_LIMIT"
ERROR_INSUFFICIENT_DISK = "INSUFFICIENT_DISK"
ERROR_PARSER_FAILURE = "PARSER_FAILURE"
ERROR_ENCRYPTED = "ENCRYPTED_PASSWORD_REQUIRED"
ERROR_PAGE_LIMIT = "PAGE_LIMIT"
ERROR_NO_USABLE_TEXT = "NO_USABLE_TEXT"
ERROR_OCR_REQUIRED = "OCR_REQUIRED"
ERROR_UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_ANY_WORD = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)?", re.UNICODE)
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+$", re.IGNORECASE)
_HEADING = re.compile(r"^(chapter|part|section)\b\s*(.*)$", re.IGNORECASE)
_ENGLISH_MARKERS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "as", "with", "was", "on", "by", "this", "from", "or", "an", "be", "are", "at", "not", "which", "but", "have", "has", "their", "they", "you", "we", "he", "she", "his", "her", "one", "all", "can", "will", "more", "would", "there", "what", "when", "who", "how", "were", "been", "into", "than", "then", "so", "if", "about", "out", "up", "do", "no", "my", "me", "our", "your",
}


class PdfAnalysisError(ValueError):
    """Stable machine-readable PDF validation or analysis failure."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class PageEvidence:
    number: int
    text: str
    classification: str
    has_images: bool
    warnings: tuple[str, ...] = ()


def _error(code: str, message: str, **details: Any) -> PdfAnalysisError:
    return PdfAnalysisError(code, message, details=details)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_file_size(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _error(ERROR_PARSER_FAILURE, "The staged PDF could not be read.") from exc
    if size > MAX_PDF_BYTES:
        raise _error(ERROR_SIZE_LIMIT, "The PDF exceeds the 100 MiB size limit.", maximum_bytes=MAX_PDF_BYTES)
    return size


def _ensure_limits(path: Path) -> int:
    size = _ensure_file_size(path)
    required = REQUIRED_DISK_MULTIPLIER * size + REQUIRED_DISK_RESERVE
    try:
        available = shutil.disk_usage(path.parent).free
    except OSError as exc:
        raise _error(ERROR_INSUFFICIENT_DISK, "Available disk space could not be determined.") from exc
    if available < required:
        raise _error(ERROR_INSUFFICIENT_DISK, "Not enough free disk space for this conversion.", required_bytes=required, available_bytes=available)
    return size


def _signature(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            signature = handle.read(5)
    except OSError as exc:
        raise _error(ERROR_PARSER_FAILURE, "The PDF could not be read.") from exc
    if signature != b"%PDF-":
        raise _error(ERROR_INVALID_SIGNATURE, "The selected file is not a PDF.")


def preflight_pdf(path: Path) -> int:
    """Check the upload before a workspace is allocated or parsed."""

    path = Path(path)
    size = _ensure_file_size(path)
    _signature(path)
    required = REQUIRED_DISK_MULTIPLIER * size + REQUIRED_DISK_RESERVE
    try:
        available = shutil.disk_usage(path.parent).free
    except OSError as exc:
        raise _error(ERROR_INSUFFICIENT_DISK, "Available disk space could not be determined.") from exc
    if available < required:
        raise _error(ERROR_INSUFFICIENT_DISK, "Not enough free disk space for this conversion.", required_bytes=required, available_bytes=available)
    return size


def _page_has_images(page: Any) -> bool:
    try:
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        return any(obj.get_object().get("/Subtype") == "/Image" for obj in xobjects.values())
    except Exception:
        return False


def _extract_pages(path: Path) -> tuple[PdfReader, list[PageEvidence]]:
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        raise _error(ERROR_PARSER_FAILURE, "The PDF parser could not read this file.") from exc
    if reader.is_encrypted:
        raise _error(ERROR_ENCRYPTED, "This PDF is encrypted and requires a password.")
    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise _error(ERROR_PARSER_FAILURE, "The PDF page tree could not be read.") from exc
    if page_count > MAX_PAGES:
        raise _error(ERROR_PAGE_LIMIT, "The PDF exceeds the 2,000 page limit.", maximum_pages=MAX_PAGES)
    evidence: list[PageEvidence] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            evidence.append(PageEvidence(index + 1, "", "unsupported", False, ("text extraction failed",)))
            continue
        text = _normalize_page(text)
        has_images = _page_has_images(page)
        words = len(_WORD.findall(text))
        if words:
            classification = "mixed" if has_images else "text"
        elif has_images:
            classification = "scanned"
        elif text:
            classification = "unsupported"
        else:
            classification = "blank"
        page_warnings: list[str] = []
        if classification == "unsupported":
            page_warnings.append("page text could not be classified")
        evidence.append(PageEvidence(index + 1, text, classification, has_images, tuple(page_warnings)))
    return reader, evidence


def _normalize_page(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(char if char in "\n\t" or not unicodedata.category(char).startswith("C") else " " for char in value)
    value = re.sub(r"[ \t]+", " ", value)
    return "\n".join(line.strip() for line in value.split("\n"))


def _repeated_lines(pages: list[PageEvidence]) -> set[str]:
    counts: Counter[str] = Counter()
    considered = 0
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip()]
        if not lines:
            continue
        considered += 1
        candidates = {lines[0], lines[-1]}
        counts.update(candidates)
    threshold = max(2, math.ceil(considered * 0.6))
    return {line for line, count in counts.items() if count >= threshold and len(line) > 2}


def _clean_pages(pages: list[PageEvidence]) -> tuple[str, list[dict[str, int]], list[str]]:
    repeated = _repeated_lines(pages)
    warnings: list[str] = []
    pieces: list[str] = []
    mapping: list[dict[str, int]] = []
    offset = 0
    for evidence in pages:
        lines = []
        for line in evidence.text.splitlines():
            stripped = line.strip()
            if stripped in repeated or _PAGE_NUMBER.fullmatch(stripped):
                continue
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            lines.append(stripped)
        page_text = "\n".join(lines)
        page_text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", page_text)
        page_lines = page_text.splitlines()
        joined: list[str] = []
        for line in page_lines:
            if not line:
                if joined and joined[-1] != "":
                    joined.append("")
            elif joined and joined[-1] and not re.search(r"[.!?:;]$", joined[-1]) and line[0].islower():
                joined[-1] += " " + line
            else:
                joined.append(line)
        page_text = "\n".join(joined).strip()
        if page_text:
            if pieces:
                pieces.append("\n\n")
                offset += 2
            start = offset
            pieces.append(page_text)
            offset += len(page_text)
            mapping.append({"source_page": evidence.number, "cleaned_start": start, "cleaned_end": offset})
        if evidence.classification in {"blank", "unsupported"}:
            warnings.append(f"Page {evidence.number} has no reliably selectable text.")
        if evidence.classification == "mixed":
            warnings.append(f"Page {evidence.number} contains both selectable text and images.")
        if evidence.classification == "scanned" and evidence.number in {1, len(pages)}:
            warnings.append(f"Page {evidence.number} is scanned but treated as a decorative page; verify before narration.")
    return "".join(pieces), mapping, warnings


def _language(text: str) -> tuple[str, list[str]]:
    words = [word.lower().replace("’", "'") for word in _ANY_WORD.findall(text)]
    if len(words) < 8:
        return "uncertain", ["Language confidence is low because the document has little text."]
    marker_ratio = sum(word in _ENGLISH_MARKERS for word in words) / len(words)
    ascii_ratio = sum(all(ord(char) < 128 for char in word) for word in words) / len(words)
    if marker_ratio >= 0.08 and ascii_ratio >= 0.75:
        return "English", []
    if marker_ratio >= 0.02 and ascii_ratio >= 0.45:
        return "uncertain", ["Language appears mixed or uncertain; review before narration."]
    raise _error(ERROR_UNSUPPORTED_LANGUAGE, "The document does not appear to be supported English text.")


def _outline_candidates(reader: PdfReader) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        outline = reader.outline
    except Exception:
        return result

    def visit(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, list):
                visit(item)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                page = None
            result.append({"title": str(title).strip(), "source_page": page, "source": "outline"})

    visit(outline)
    return result


def _layout_heading_candidates(path: Path, page_count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages[:page_count]):
                chars = page.chars or []
                sizes = sorted(float(char.get("size", 0)) for char in chars if char.get("size") is not None)
                baseline = sizes[len(sizes) // 2] if sizes else 0
                if not baseline:
                    continue
                grouped: dict[float, list[dict[str, Any]]] = {}
                for char in chars:
                    top = round(float(char.get("top", 0)), 1)
                    grouped.setdefault(top, []).append(char)
                for line_chars in grouped.values():
                    text = "".join(str(char.get("text", "")) for char in sorted(line_chars, key=lambda item: float(item.get("x0", 0)))).strip()
                    words = _WORD.findall(text)
                    if not (2 <= len(words) <= 10 and len(text) <= 100):
                        continue
                    max_size = max(float(char.get("size", 0)) for char in line_chars)
                    bold = any("bold" in str(char.get("fontname", "")).lower() for char in line_chars)
                    if max_size >= baseline * 1.25 or bold:
                        candidates.append({"title": text, "source_page": index + 1, "source": "layout"})
    except Exception:
        return []
    return candidates


def _chapter_candidates(reader: PdfReader, pages: list[PageEvidence], path: Path) -> list[dict[str, Any]]:
    candidates = _outline_candidates(reader)
    if candidates:
        return candidates
    for page in pages:
        for line in page.text.splitlines():
            match = _HEADING.match(line.strip())
            if match:
                candidates.append({"title": line.strip(), "source_page": page.number, "source": "heading"})
    if candidates:
        return candidates
    candidates.extend(_layout_heading_candidates(path, len(pages)))
    if candidates:
        return candidates
    for page in pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            words = _WORD.findall(stripped)
            if 2 <= len(words) <= 10 and len(stripped) <= 80 and stripped.isupper():
                candidates.append({"title": stripped, "source_page": page.number, "source": "layout"})
    return candidates


def _layout_warnings(path: Path, page_count: int) -> list[str]:
    warnings: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages[:page_count]):
                words = page.extract_words() or []
                lines = page.extract_text_lines() or []
                midpoint = page.width / 2
                left = [line for line in lines if float(line.get("x1", 0)) < midpoint - 20]
                right = [line for line in lines if float(line.get("x0", 0)) > midpoint + 20]
                if len(left) >= 3 and len(right) >= 3 and len(words) >= 20:
                    warnings.append(f"Page {index + 1} may use multiple columns; reading order was inferred.")
                try:
                    if page.find_tables():
                        warnings.append(f"Page {index + 1} contains table-like layout; reading order was inferred.")
                except Exception:
                    pass
                extracted = page.extract_text() or ""
                if re.search(r"(?:=\s*[A-Za-z0-9]|\b(?:eq|equation)\b)", extracted, re.IGNORECASE):
                    warnings.append(f"Page {index + 1} may contain equations; review the extracted text.")
                if any("top" in word and "bottom" in word and float(word["bottom"]) - float(word["top"]) > page.height * 0.15 for word in words):
                    warnings.append(f"Page {index + 1} contains unusually large layout elements.")
    except Exception:
        warnings.append("Layout evidence was unavailable for one or more pages.")
    return warnings


def _validate_page_coverage(pages: list[PageEvidence], cleaned_text: str) -> None:
    usable_pages = [page for page in pages if len(_WORD.findall(page.text)) >= 2]
    if not usable_pages or not cleaned_text.strip():
        if any(page.classification == "unsupported" for page in pages):
            raise _error(ERROR_NO_USABLE_TEXT, "The PDF has no usable selectable text.")
        raise _error(ERROR_OCR_REQUIRED, "This PDF has no usable selectable text; OCR is required.")
    interior = pages[1:-1]
    if any(page.classification == "unsupported" for page in interior):
        raise _error(ERROR_PARSER_FAILURE, "An interior page could not be reliably extracted.")
    if any(page.classification == "scanned" for page in interior) or any(page.classification == "mixed" and len(_WORD.findall(page.text)) < MIXED_MIN_WORDS for page in interior):
        raise _error(ERROR_OCR_REQUIRED, "An interior page requires OCR; the PDF was not silently omitted.")


def analyze_pdf(path: Path, *, fallback_title: str | None = None, check_disk: bool = True) -> dict[str, Any]:
    """Validate and analyze a workspace PDF, returning JSON-safe review data."""

    path = Path(path)
    if check_disk:
        _ensure_limits(path)
    else:
        _ensure_file_size(path)
    _signature(path)
    reader, pages = _extract_pages(path)
    cleaned_text, cleaned_map, warnings = _clean_pages(pages)
    _validate_page_coverage(pages, cleaned_text)
    language, language_warnings = _language(cleaned_text)
    warnings.extend(language_warnings)
    warnings.extend(_layout_warnings(path, len(pages)))
    words = len(_WORD.findall(cleaned_text))
    metadata = reader.metadata or {}
    title = str(getattr(metadata, "title", None) or metadata.get("/Title") or fallback_title or path.name)
    return {
        "title": title,
        "page_count": len(pages),
        "word_count": words,
        "detected_language": language,
        "estimated_duration_minutes": round(words / WORDS_PER_MINUTE, 2),
        "words_per_minute": WORDS_PER_MINUTE,
        "warnings": list(dict.fromkeys(warnings)),
        "preview": cleaned_text[:PREVIEW_CHARACTERS],
        "page_classifications": [
            {"page": page.number, "classification": page.classification, "has_images": page.has_images, "warnings": list(page.warnings)}
            for page in pages
        ],
        "chapter_candidates": _chapter_candidates(reader, pages, path),
        "cleaned_text": cleaned_text,
        "cleaned_map": cleaned_map,
        "source_pdf_sha256": _sha256(path),
    }


__all__ = [
    "ERROR_ENCRYPTED",
    "ERROR_INSUFFICIENT_DISK",
    "ERROR_INVALID_SIGNATURE",
    "ERROR_NO_USABLE_TEXT",
    "ERROR_OCR_REQUIRED",
    "ERROR_PAGE_LIMIT",
    "ERROR_PARSER_FAILURE",
    "ERROR_SIZE_LIMIT",
    "ERROR_UNSUPPORTED_LANGUAGE",
    "MAX_PAGES",
    "MAX_PDF_BYTES",
    "MIXED_MIN_WORDS",
    "PdfAnalysisError",
    "REQUIRED_DISK_MULTIPLIER",
    "REQUIRED_DISK_RESERVE",
    "WORDS_PER_MINUTE",
    "analyze_pdf",
    "preflight_pdf",
]

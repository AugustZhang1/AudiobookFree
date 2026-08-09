"""Deterministic text normalization and counting used by every engine."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CONTROL_EXCEPTIONS = {"\n", "\t"}
_WORD_RE = re.compile(r"\b[\w]+(?:['’.-][\w]+)*\b", re.UNICODE)


@dataclass(frozen=True)
class TextFacts:
    """Normalized text and its stable measurements."""

    text: str
    characters: int
    words: int


def normalize_text(text: str) -> str:
    """Return canonical, line-oriented text without destructive word edits."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch for ch in value if not unicodedata.category(ch).startswith("C") or ch in _CONTROL_EXCEPTIONS)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def count_words(text: str) -> int:
    """Count words using the same normalized input used for synthesis."""

    return len(_WORD_RE.findall(normalize_text(text)))


def text_facts(text: str) -> TextFacts:
    """Normalize text and return character/word counts in one operation."""

    normalized = normalize_text(text)
    return TextFacts(normalized, len(normalized), len(_WORD_RE.findall(normalized)))

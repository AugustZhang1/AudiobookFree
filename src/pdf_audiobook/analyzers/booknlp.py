"""BookNLP 1.0.8 adapter.

BookNLP is deliberately absent from the application environment.  This module
only starts the checked/configured isolated runner and consumes its small JSON
contract; it never imports the ML package itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from .. import speakers
from . import booknlp_runner
from .booknlp_runner import parse_booknlp_output


MAX_INPUT_BYTES = 64 * 1024 * 1024
# The runner owns this bound; retaining the adapter alias prevents drift while
# preserving the public constant used by callers and tests.
MAX_RESULT_BYTES = booknlp_runner.MAX_RESULT_BYTES
MAX_TOKENS = 2_000_000
MAX_QUOTES = 250_000
MAX_CHARACTERS = 100_000
MAX_WARNING_LENGTH = 8 * 1024
RUNNER_SCHEMA_VERSION = 1
BOOKNLP_VERSION = "1.0.8"
OUTPUT_TOO_LARGE_MESSAGE = "analyzer output exceeded the size limit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BookNLPAnalyzerError(ValueError):
    """Safe, stable adapter failure (details never include book text)."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> BookNLPAnalyzerError:
    return BookNLPAnalyzerError(code, message, details=details)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _clean_string(value: Any, name: str, maximum: int = MAX_WARNING_LENGTH) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise _fail("ANALYZER_OUTPUT_INVALID", f"runner {name} is invalid")
    return value


def _chapter_ranges(chapter_plan: Any, text_length: int) -> dict[int, tuple[int, int]]:
    if not isinstance(chapter_plan, dict) or not isinstance(chapter_plan.get("chapters"), list) or not chapter_plan["chapters"]:
        raise _fail("INVALID_CHAPTER_PLAN", "chapter plan is invalid")
    ranges: dict[int, tuple[int, int]] = {}
    expected = 1
    offset = 0
    for chapter in chapter_plan["chapters"]:
        if not isinstance(chapter, dict):
            raise _fail("INVALID_CHAPTER_PLAN", "chapter plan is invalid")
        index, start, end = chapter.get("index"), chapter.get("start_offset"), chapter.get("end_offset")
        if not _is_int(index) or index != expected or not _is_int(start) or not _is_int(end) or start != offset or end <= start or end > text_length:
            raise _fail("INVALID_CHAPTER_PLAN", "chapter plan is invalid")
        ranges[index] = (start, end)
        expected += 1
        offset = end
    if offset != text_length:
        raise _fail("INVALID_CHAPTER_PLAN", "chapter plan is invalid")
    return ranges


def _parse_json(stdout: Any) -> dict[str, Any]:
    if isinstance(stdout, bytes):
        try:
            stdout = stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner output was not UTF-8") from exc
    if not isinstance(stdout, str):
        raise _fail("ANALYZER_OUTPUT_INVALID", "runner output was not valid text")
    if len(stdout.encode("utf-8", "surrogatepass")) > MAX_RESULT_BYTES:
        raise _fail("OUTPUT_TOO_LARGE", OUTPUT_TOO_LARGE_MESSAGE)
    try:
        value = json.loads(stdout, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail("ANALYZER_OUTPUT_INVALID", "runner output was not valid JSON") from exc
    if not isinstance(value, dict):
        raise _fail("ANALYZER_OUTPUT_INVALID", "runner output must be an object")
    required = {"schema_version", "booknlp_version", "tokens", "quotes", "characters", "warnings"}
    if set(value) != required or value["schema_version"] != RUNNER_SCHEMA_VERSION or value["booknlp_version"] != BOOKNLP_VERSION:
        raise _fail("ANALYZER_OUTPUT_INVALID", "runner output schema mismatch")
    for field, maximum in (("tokens", MAX_TOKENS), ("quotes", MAX_QUOTES), ("characters", MAX_CHARACTERS)):
        if not isinstance(value[field], list) or len(value[field]) > maximum:
            raise _fail("ANALYZER_OUTPUT_INVALID", f"runner {field} are invalid")
    warnings = value["warnings"]
    if not isinstance(warnings, list) or len(warnings) > 256 or any(not isinstance(item, str) or len(item) > MAX_WARNING_LENGTH or any(ord(char) < 32 for char in item) for item in warnings):
        raise _fail("ANALYZER_OUTPUT_INVALID", "runner warnings are invalid")
    return value


def _byte_boundaries(text: str) -> tuple[dict[int, int], bytes]:
    encoded = text.encode("utf-8")
    boundaries = {0: 0}
    byte_position = 0
    for index, char in enumerate(text, 1):
        byte_position += len(char.encode("utf-8"))
        boundaries[byte_position] = index
    return boundaries, encoded


def _token_table(raw_tokens: Any, text: str) -> tuple[dict[int, tuple[int, int]], dict[int, str]]:
    boundaries, encoded = _byte_boundaries(text)
    positions: dict[int, tuple[int, int]] = {}
    words: dict[int, str] = {}
    for expected_id, token in enumerate(raw_tokens):
        if not isinstance(token, dict) or set(token) != {"id", "word", "byte_start", "byte_end"}:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner token schema mismatch")
        token_id = token["id"]
        word = token["word"]
        start = token["byte_start"]
        end = token["byte_end"]
        if not _is_int(token_id) or token_id != expected_id or not isinstance(word, str) or not word or len(word) > 4096 or any(ord(char) < 32 for char in word):
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner token is invalid")
        if not _is_int(start) or not _is_int(end) or start < 0 or end <= start or end > len(encoded) or start not in boundaries or end not in boundaries:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner token offset is invalid")
        if encoded[start:end] != word.encode("utf-8"):
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner token offset does not match cleaned text")
        positions[token_id] = (boundaries[start], boundaries[end])
        words[token_id] = word
    return positions, words


def _characters(raw_characters: Any, token_count: int) -> dict[int, tuple[str, dict[str, Any]]]:
    result: dict[int, tuple[str, dict[str, Any]]] = {}
    for character in raw_characters:
        if not isinstance(character, dict) or set(character) != {"coref_id", "canonical_label", "aliases", "quote_count"}:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner character schema mismatch")
        coref_id, label, aliases, quote_count = (character[key] for key in ("coref_id", "canonical_label", "aliases", "quote_count"))
        if not _is_int(coref_id) or coref_id < 0 or coref_id in result:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner character ID is invalid")
        label = _clean_string(label, "character label", 512)
        if not isinstance(aliases, list) or len(aliases) > 10_000 or not _is_int(quote_count) or quote_count < 0:
            raise _fail("ANALYZER_OUTPUT_INVALID", "runner character is invalid")
        normalized_aliases: list[dict[str, Any]] = []
        seen_aliases: set[tuple[str, str]] = set()
        for alias in aliases:
            if not isinstance(alias, dict) or set(alias) != {"alias", "kind", "token_start", "token_end"}:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner alias schema mismatch")
            text_value, kind, token_start, token_end = (alias[key] for key in ("alias", "kind", "token_start", "token_end"))
            if kind not in {"proper", "nominal", "pronoun"} or not _is_int(token_start) or not _is_int(token_end) or token_start < 0 or token_end <= token_start or token_end > token_count:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner alias is invalid")
            text_value = _clean_string(text_value, "alias", 512)
            key = (text_value, kind)
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            normalized_aliases.append({"alias": text_value, "kind": kind, "confidence": 0.7, "provenance": {"source": "booknlp", "token_start": token_start, "token_end": token_end}})
        character_id = f"booknlp:{coref_id}"
        result[coref_id] = (character_id, {"character_id": character_id, "canonical_label": label, "aliases": normalized_aliases, "line_count": 0, "quote_count": quote_count})
    return result


class BookNLPAnalyzer:
    """Run the BookNLP 1.0.8 runner in a separate Python environment."""

    descriptor = {"id": "booknlp", "version": BOOKNLP_VERSION, "model_hash": None}

    def __init__(self, python_executable: str | Path | None = None, runner_path: str | Path | None = None, *, timeout: float = 900.0, command_runner: Callable[..., Any] | None = None, temp_directory: str | Path | None = None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        root = Path(__file__).resolve().parents[3]
        env_root = root / "analyzer_envs" / "booknlp" / ".venv"
        default_python = env_root / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        self.python_executable = Path(python_executable) if python_executable is not None else default_python
        self.runner_path = Path(runner_path) if runner_path is not None else Path(__file__).with_name("booknlp_runner.py")
        self.timeout = float(timeout)
        self.command_runner = command_runner
        self.temp_directory = Path(temp_directory) if temp_directory is not None else None

    def _check_cancelled(self, control: Any) -> None:
        if control is not None and callable(getattr(control, "check_cancelled", None)):
            control.check_cancelled()

    def _run(self, argv: list[str]) -> Any:
        runner = self.command_runner or subprocess.run
        try:
            return runner(argv, shell=False, check=False, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise _fail("ANALYZER_TIMEOUT", "BookNLP analysis timed out") from exc
        except OSError as exc:
            raise _fail("ANALYZER_UNAVAILABLE", "BookNLP isolated runner could not start") from exc
        except Exception as exc:
            raise _fail("ANALYZER_FAILED", "BookNLP isolated runner failed") from exc

    @staticmethod
    def _stdout_exceeds_limit(stdout: Any) -> bool:
        if isinstance(stdout, bytes):
            return len(stdout) > MAX_RESULT_BYTES
        if isinstance(stdout, str):
            return len(stdout.encode("utf-8", "surrogatepass")) > MAX_RESULT_BYTES
        return False

    def analyze(self, cleaned_text: str, chapter_plan: dict[str, Any], source_hash: str, options: Any = None) -> speakers.MachineAnalysis:
        if not isinstance(cleaned_text, str):
            raise _fail("INVALID_TEXT", "cleaned text is invalid")
        if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
            raise _fail("INVALID_SOURCE_HASH", "source hash is invalid")
        try:
            input_bytes = cleaned_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _fail("INVALID_TEXT", "cleaned text is invalid") from exc
        if len(input_bytes) > MAX_INPUT_BYTES:
            raise _fail("INPUT_TOO_LARGE", "cleaned text exceeds the analyzer size limit")
        ranges = _chapter_ranges(chapter_plan, len(cleaned_text))
        control = options.get("analysis_control") if isinstance(options, Mapping) else None
        self._check_cancelled(control)
        if self.command_runner is None and (not self.python_executable.is_file() or not self.runner_path.is_file()):
            raise _fail("ANALYZER_UNAVAILABLE", "BookNLP isolated environment is unavailable")
        if self.command_runner is not None:
            # Injected runners are an in-process test boundary.  They receive
            # no book path and therefore cannot accidentally launch the real
            # runner or require filesystem writes.
            self._check_cancelled(control)
            result = self._run([str(self.python_executable), "-X", "utf8", str(self.runner_path), "<injected-cleaned-text>"])
            self._check_cancelled(control)
        else:
            with tempfile.TemporaryDirectory(prefix="booknlp-", dir=str(self.temp_directory) if self.temp_directory else None) as work:
                input_path = Path(work) / "cleaned.txt"
                input_path.write_bytes(input_bytes)
                self._check_cancelled(control)
                result = self._run([str(self.python_executable), "-X", "utf8", str(self.runner_path), str(input_path)])
                self._check_cancelled(control)
        stdout = getattr(result, "stdout", None)
        if self._stdout_exceeds_limit(stdout) or getattr(result, "returncode", None) == booknlp_runner.OUTPUT_TOO_LARGE_EXIT_CODE:
            raise _fail("OUTPUT_TOO_LARGE", OUTPUT_TOO_LARGE_MESSAGE)
        if getattr(result, "returncode", None) != 0:
            raise _fail("ANALYZER_FAILED", "BookNLP isolated runner failed")
        payload = _parse_json(stdout)
        positions, _ = _token_table(payload["tokens"], cleaned_text)
        self._check_cancelled(control)
        characters = _characters(payload["characters"], len(positions))
        spans: list[speakers.SpeakerSpan] = []
        previous_end = -1
        seen_quote_ids: set[str] = set()
        for quote in payload["quotes"]:
            if not isinstance(quote, dict) or set(quote) != {"quote_id", "start_token", "end_token", "coref_id"}:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner quote schema mismatch")
            quote_id, start_token, end_token, coref_id = (quote[key] for key in ("quote_id", "start_token", "end_token", "coref_id"))
            quote_id = _clean_string(quote_id, "quote ID", 256)
            if quote_id in seen_quote_ids or not _is_int(start_token) or not _is_int(end_token) or not _is_int(coref_id) or coref_id < -1 or start_token > end_token or start_token not in positions or end_token not in positions:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner quote is invalid")
            seen_quote_ids.add(quote_id)
            start = positions[start_token][0]
            end = positions[end_token][1]
            if start < previous_end or end <= start:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner quotes overlap or are invalid")
            chapter = next((index for index, (chapter_start, chapter_end) in ranges.items() if chapter_start <= start and end <= chapter_end), None)
            if chapter is None:
                raise _fail("ANALYZER_OUTPUT_INVALID", "runner quote crosses a chapter boundary")
            previous_end = end
            character = characters.get(coref_id) if _is_int(coref_id) and coref_id >= 0 else None
            speaker_id = character[0] if character else None
            span_type = "dialogue" if speaker_id else "unknown"
            score, band, reason = (0.85, "high", "booknlp quote attribution") if speaker_id else (0.2, "low", "speaker unresolved")
            spans.append(speakers.SpeakerSpan(f"booknlp:{quote_id}", chapter, start, end, span_type, speaker_id, speakers.Confidence(score, band, (reason,)), ("booknlp", quote_id)))
        analysis = speakers.MachineAnalysis(tuple(spans), source_hash, ("booknlp",), tuple(item[1] for item in characters.values()), tuple(payload["warnings"]))
        self._check_cancelled(control)
        speakers.validate_machine_spans(analysis, cleaned_text, chapter_plan)
        return analysis


__all__ = ["BOOKNLP_VERSION", "BookNLPAnalyzer", "BookNLPAnalyzerError", "parse_booknlp_output"]

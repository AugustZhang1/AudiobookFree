"""Isolated BookNLP 1.0.8 execution entry point.

The application invokes this file with exactly one input path.  BookNLP output
files are always created below a private temporary directory and only the
bounded contract JSON is written to stdout.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

MAX_INPUT_BYTES = 64 * 1024 * 1024
# This is the result transport ceiling shared with the adapter.  Keep the
# historical name as an alias because external checks may import it.
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = MAX_RESULT_BYTES
OUTPUT_TOO_LARGE_EXIT_CODE = 3
RESULT_TOO_LARGE_EXIT_CODE = OUTPUT_TOO_LARGE_EXIT_CODE
MAX_TOKENS = 2_000_000
MAX_QUOTES = 250_000
MAX_CHARACTERS = 100_000


class _ResultTooLarge(ValueError):
    """Internal marker for the stable oversized-result runner exit path."""


def _read_tsv(path: Path, maximum: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        # BookNLP writes literal tab-separated text, not CSV-quoted fields.
        # Disable csv's quote handling so a token/quote containing `"` remains
        # in its own field instead of shifting subsequent columns.
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        if not reader.fieldnames:
            raise ValueError("missing BookNLP header")
        for row in reader:
            if len(rows) >= maximum:
                raise ValueError("BookNLP output exceeds limit")
            rows.append(dict(row))
    return rows


def _int(row: dict[str, str], key: str, *, minimum: int = 0) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid BookNLP numeric field") from exc
    if value < minimum:
        raise ValueError("invalid BookNLP numeric field")
    return value


def _optional_coref(value: Any) -> int:
    # EnglishBookNLP writes the Python None value as the literal string
    # ``None`` when a quote has no attributed mention.
    if value is None or value in {"", "None"}:
        return -1
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid BookNLP coreference ID") from exc
    if result < -1:
        raise ValueError("invalid BookNLP coreference ID")
    return result


def parse_booknlp_output(output_directory: str | Path) -> dict[str, Any]:
    """Parse official BookNLP TSV outputs without importing BookNLP.

    BookNLP labels ``byte_onset``/``byte_offset`` as byte positions, but its
    Python writer emits character indices.  The runner normalizes those raw
    positions against the input text before returning this JSON contract.
    """

    output_dir = Path(output_directory)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("BookNLP output directory is invalid")
    token_rows = _read_tsv(output_dir / "book.tokens", MAX_TOKENS)
    quote_rows = _read_tsv(output_dir / "book.quotes", MAX_QUOTES)
    entity_rows = _read_tsv(output_dir / "book.entities", MAX_TOKENS)

    tokens: list[dict[str, Any]] = []
    for expected, row in enumerate(token_rows):
        token_id = _int(row, "token_ID_within_document")
        if token_id != expected:
            raise ValueError("BookNLP token IDs are not contiguous")
        word = row.get("word")
        if not isinstance(word, str) or not word:
            raise ValueError("BookNLP token text is invalid")
        tokens.append({"id": token_id, "word": word, "byte_start": _int(row, "byte_onset"), "byte_end": _int(row, "byte_offset")})

    grouped: dict[int, dict[str, Any]] = {}
    for row in entity_rows:
        # Non-PER rows can contain fields that are irrelevant to this MVP and
        # must be ignored before attempting to parse a coreference ID.
        if row.get("cat") != "PER":
            continue
        try:
            coref_id = int(row["COREF"])
            start = _int(row, "start_token")
            end_inclusive = _int(row, "end_token")
        except (KeyError, TypeError, ValueError):
            # Unclustered/invalid PER mentions are not characters.  Keep the
            # quote attribution reviewable rather than inventing an ID.
            continue
        if coref_id < 0 or start > end_inclusive or end_inclusive >= len(tokens):
            continue
        alias = row.get("text") or ""
        if not alias:
            continue
        char = grouped.setdefault(coref_id, {"coref_id": coref_id, "canonical_label": alias, "aliases": [], "quote_count": 0})
        kind = {"PROP": "proper", "NOM": "nominal", "PRON": "pronoun"}.get(row.get("prop", ""), "nominal")
        if len(alias) > len(char["canonical_label"]) and kind == "proper":
            char["canonical_label"] = alias
        record = {"alias": alias, "kind": kind, "token_start": start, "token_end": end_inclusive + 1}
        if record not in char["aliases"]:
            char["aliases"].append(record)

    quotes: list[dict[str, Any]] = []
    for index, row in enumerate(quote_rows):
        start = _int(row, "quote_start")
        end = _int(row, "quote_end")
        if end < start or end >= len(tokens):
            raise ValueError("BookNLP quote range is invalid")
        char_id = _optional_coref(row.get("char_id"))
        if char_id in grouped:
            grouped[char_id]["quote_count"] += 1
        # quote_end is an inclusive token ID in the official .quotes file;
        # retain it here and convert to a character-exclusive endpoint later.
        quotes.append({"quote_id": str(index), "start_token": start, "end_token": end, "coref_id": char_id})
    result = {"schema_version": 1, "booknlp_version": "1.0.8", "tokens": tokens, "quotes": quotes, "characters": list(grouped.values()), "warnings": []}
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        raise _ResultTooLarge()
    return result


def _normalize_token_offsets(result: dict[str, Any], text: str) -> None:
    character_cursor = 0
    byte_cursor = 0
    for token in result["tokens"]:
        start = token["byte_start"]
        end = token["byte_end"]
        word = token["word"]
        if type(start) is not int or type(end) is not int or start < character_cursor or end <= start or end > len(text):
            raise ValueError("BookNLP token offsets are not ordered")
        token_text = text[start:end]
        if not isinstance(word, str) or token_text != word:
            raise ValueError("BookNLP token offset does not match token text")
        # Gaps between tokens are valid.  Encode each gap and token once while
        # advancing the cursors, without retaining a full boundary array.
        byte_cursor += len(text[character_cursor:start].encode("utf-8"))
        token["byte_start"] = byte_cursor
        byte_cursor += len(token_text.encode("utf-8"))
        token["byte_end"] = byte_cursor
        character_cursor = end


def _import_booknlp() -> Any:
    runner_directory = Path(__file__).resolve().parent
    original_sys_path = sys.path[:]
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if not _resolves_to_runner_directory(entry, runner_directory)
        ]
        from booknlp.booknlp import BookNLP  # type: ignore[import-not-found]

        return BookNLP
    finally:
        sys.path[:] = original_sys_path


def _resolves_to_runner_directory(entry: str, runner_directory: Path) -> bool:
    try:
        return Path(entry or ".").resolve() == runner_directory
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _run(input_path: Path) -> dict[str, Any]:
    # Import only after entering the isolated execution process.  The main app
    # package never imports this module or BookNLP.
    BookNLP = _import_booknlp()

    data = input_path.read_bytes()
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds limit")
    text = data.decode("utf-8")
    with tempfile.TemporaryDirectory(prefix="booknlp-output-") as output:
        input_file = input_path.resolve()
        output_dir = Path(output).resolve()
        model_directory = (Path(sys.prefix) / "booknlp_models").resolve()
        model_directory.mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(model_directory):
            BookNLP("en", {"pipeline": "entity,quote,coref", "model": "small", "model_path": ""}).process(str(input_file), str(output_dir), "book")
        result = parse_booknlp_output(output_dir)
    _normalize_token_offsets(result, text)
    # Keep a reference to text so a future runner change cannot accidentally
    # emit offsets for a different input; all validation remains in the adapter.
    if not isinstance(text, str):
        raise ValueError("input is invalid")
    return result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    path = Path(args[0])
    if path.is_symlink() or not path.is_file():
        return 2
    try:
        # BookNLP and its dependencies can be noisy.  Suppress both streams so
        # stdout remains exactly one bounded JSON document.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = _run(path)
        output = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        if len(output.encode("utf-8")) > MAX_RESULT_BYTES:
            raise _ResultTooLarge()
        sys.stdout.write(output)
        return 0
    except _ResultTooLarge:
        return OUTPUT_TOO_LARGE_EXIT_CODE
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

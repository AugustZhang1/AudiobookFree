"""Deterministic character-to-voice attribution accuracy benchmark.

The benchmark consumes small synthetic JSON artifacts.  It intentionally has
no dependency on the application or on model runtimes: predictions are
pre-generated data, and scoring is purely a comparison of exact source
ranges and speaker identities.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
GOLD_CATEGORIES = (
    "explicit",
    "anaphoric",
    "implicit",
    "nested",
    "alternating",
    "role_named",
    "surname_conflict",
    "minor_character",
    "selected_range",
)
PREDICTION_STATES = (
    "auto_approved",
    "reviewed",
    "manual_correction",
    "exception",
    "unresolved",
)
EXCEPTION_STATES = frozenset(("exception", "unresolved"))


class BenchmarkError(ValueError):
    """A safe, user-facing input or scoring error."""


def _expect_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{where} must be an object")
    return value


def _expect_keys(value: Mapping[str, Any], required: set[str], optional: set[str], where: str) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise BenchmarkError(f"{where} missing required field(s): {', '.join(missing)}")
    if unknown:
        raise BenchmarkError(f"{where} has unknown field(s): {', '.join(unknown)}")


def _expect_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "\n" in value or "\r" in value:
        raise BenchmarkError(f"{where} must be a non-empty single-line string")
    return value


def _expect_id(value: Any, where: str) -> str:
    return _expect_string(value, where)


def _expect_source_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{where} must be a non-empty string")
    unsafe_controls = {
        character
        for character in value
        if unicodedata.category(character) == "Cc" and character not in "\t\n\r"
    }
    if unsafe_controls:
        raise BenchmarkError(f"{where} contains unsafe control characters")
    return value


def _expect_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{where} must be an integer")
    return value


def _expect_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{where} must be an array")
    return value


def _validate_cast(cast_value: Any, where: str) -> tuple[dict[str, str], str]:
    cast = _expect_list(cast_value, where)
    if not cast:
        raise BenchmarkError(f"{where} must contain a narrator and at least one speaker")
    identities: dict[str, str] = {}
    narrator_ids: list[str] = []
    for index, raw in enumerate(cast):
        item = _expect_object(raw, f"{where}[{index}]")
        _expect_keys(item, {"speaker_id", "role"}, set(), f"{where}[{index}]")
        speaker_id = _expect_id(item["speaker_id"], f"{where}[{index}].speaker_id")
        role = _expect_string(item["role"], f"{where}[{index}].role")
        if role not in {"narrator", "character"}:
            raise BenchmarkError(f"{where}[{index}].role must be narrator or character")
        if speaker_id in identities:
            raise BenchmarkError(f"duplicate speaker_id: {speaker_id}")
        identities[speaker_id] = role
        if role == "narrator":
            narrator_ids.append(speaker_id)
    if len(narrator_ids) != 1:
        raise BenchmarkError(f"{where} must contain exactly one narrator")
    return identities, narrator_ids[0]


def _validate_range(raw: Any, where: str, text_length: int) -> tuple[int, int]:
    item = _expect_object(raw, where)
    if "source_start" not in item or "source_end" not in item:
        raise BenchmarkError(f"{where} missing required field(s): source_start, source_end")
    start = _expect_int(item["source_start"], f"{where}.source_start")
    end = _expect_int(item["source_end"], f"{where}.source_end")
    if start < 0 or end <= start or end > text_length:
        raise BenchmarkError(f"{where} has invalid half-open source range [{start}, {end})")
    return start, end


def _validate_gold_quotes(raw_quotes: Any, identities: Mapping[str, str], text_length: int, where: str) -> list[dict[str, Any]]:
    quotes = _expect_list(raw_quotes, where)
    seen_ids: set[str] = set()
    seen_ranges: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(quotes):
        item = _expect_object(raw, f"{where}[{index}]")
        _expect_keys(item, {"quote_id", "source_start", "source_end", "speaker_id", "category"}, set(), f"{where}[{index}]")
        quote_id = _expect_id(item["quote_id"], f"{where}[{index}].quote_id")
        if quote_id in seen_ids:
            raise BenchmarkError(f"duplicate gold quote_id: {quote_id}")
        seen_ids.add(quote_id)
        start, end = _validate_range(item, f"{where}[{index}]", text_length)
        if (start, end) in seen_ranges:
            raise BenchmarkError(f"duplicate gold source range: [{start}, {end})")
        seen_ranges.add((start, end))
        speaker_id = _expect_id(item["speaker_id"], f"{where}[{index}].speaker_id")
        if speaker_id not in identities:
            raise BenchmarkError(f"{where}[{index}] references unknown speaker_id: {speaker_id}")
        category = _expect_string(item["category"], f"{where}[{index}].category")
        if category not in GOLD_CATEGORIES:
            raise BenchmarkError(f"{where}[{index}].category must be one of: {', '.join(GOLD_CATEGORIES)}")
        normalized.append({"quote_id": quote_id, "source_start": start, "source_end": end, "speaker_id": speaker_id, "category": category})
    return normalized


def _validate_predictions(raw_predictions: Any, identities: Mapping[str, str], text_length: int, where: str) -> list[dict[str, Any]]:
    predictions = _expect_list(raw_predictions, where)
    seen_ids: set[str] = set()
    seen_ranges: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(predictions):
        item = _expect_object(raw, f"{where}[{index}]")
        _expect_keys(item, {"prediction_id", "source_start", "source_end", "predicted_speaker_id", "original_speaker_id", "effective_speaker_id", "decision_state"}, set(), f"{where}[{index}]")
        prediction_id = _expect_id(item["prediction_id"], f"{where}[{index}].prediction_id")
        if prediction_id in seen_ids:
            raise BenchmarkError(f"duplicate prediction_id: {prediction_id}")
        seen_ids.add(prediction_id)
        start, end = _validate_range(item, f"{where}[{index}]", text_length)
        if (start, end) in seen_ranges:
            raise BenchmarkError(f"duplicate prediction source range: [{start}, {end})")
        seen_ranges.add((start, end))
        speakers: dict[str, str | None] = {}
        for field in ("predicted_speaker_id", "original_speaker_id", "effective_speaker_id"):
            value = item[field]
            if value is not None:
                value = _expect_id(value, f"{where}[{index}].{field}")
                if value not in identities:
                    raise BenchmarkError(f"{where}[{index}] references unknown speaker_id: {value}")
            speakers[field] = value
        state = _expect_string(item["decision_state"], f"{where}[{index}].decision_state")
        if state not in PREDICTION_STATES:
            raise BenchmarkError(f"{where}[{index}].decision_state must be one of: {', '.join(PREDICTION_STATES)}")
        normalized.append({"prediction_id": prediction_id, "source_start": start, "source_end": end, **speakers, "decision_state": state})
    return normalized


def _validate_book(raw: Any, where: str, *, require_gold: bool, require_predictions: bool) -> dict[str, Any]:
    item = _expect_object(raw, where)
    required = {"book_id", "text", "cast"}
    optional: set[str] = set()
    if require_gold:
        required.add("gold_quotes")
    else:
        optional.add("gold_quotes")
    if require_predictions:
        required.add("predictions")
    else:
        optional.add("predictions")
    _expect_keys(item, required, optional, where)
    book_id = _expect_id(item["book_id"], f"{where}.book_id")
    text = _expect_source_text(item["text"], f"{where}.text")
    identities, narrator_id = _validate_cast(item["cast"], f"{where}.cast")
    gold_quotes = _validate_gold_quotes(item.get("gold_quotes", []), identities, len(text), f"{where}.gold_quotes")
    predictions = _validate_predictions(item.get("predictions", []), identities, len(text), f"{where}.predictions")
    return {"book_id": book_id, "text": text, "cast": item["cast"], "identities": identities, "narrator_id": narrator_id, "gold_quotes": gold_quotes, "predictions": predictions}


def _validate_root(raw: Any, *, require_gold: bool, require_predictions: bool, artifact: str) -> dict[str, Any]:
    root = _expect_object(raw, "corpus")
    _expect_keys(root, {"schema_version", "artifact", "books"}, set(), "corpus")
    version = _expect_int(root["schema_version"], "corpus.schema_version")
    if version != SCHEMA_VERSION:
        raise BenchmarkError(f"unsupported schema_version: {version}")
    if root["artifact"] != artifact:
        raise BenchmarkError(f"corpus.artifact must be {artifact!r}")
    books = _expect_list(root["books"], "corpus.books")
    if not books:
        raise BenchmarkError("corpus.books must not be empty")
    normalized_books: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_book in enumerate(books):
        book = _validate_book(raw_book, f"corpus.books[{index}]", require_gold=require_gold, require_predictions=require_predictions)
        if book["book_id"] in seen_ids:
            raise BenchmarkError(f"duplicate book_id: {book['book_id']}")
        seen_ids.add(book["book_id"])
        normalized_books.append(book)
    normalized_books.sort(key=lambda book: book["book_id"])
    return {"schema_version": SCHEMA_VERSION, "artifact": "interactive-accuracy-corpus", "books": normalized_books}


def validate_corpus(raw: Any) -> dict[str, Any]:
    """Validate and normalize a combined corpus containing gold and predictions."""
    return _validate_root(raw, require_gold=True, require_predictions=True, artifact="interactive-accuracy-corpus")


def load_gold_and_predictions(gold_raw: Any, predictions_raw: Any) -> dict[str, Any]:
    """Validate separate gold/prediction artifacts and combine them."""
    gold = _validate_root(gold_raw, require_gold=True, require_predictions=False, artifact="interactive-accuracy-gold")
    predictions = _validate_root(predictions_raw, require_gold=False, require_predictions=True, artifact="interactive-accuracy-predictions")
    gold_books = {book["book_id"]: book for book in gold["books"]}
    prediction_books = {book["book_id"]: book for book in predictions["books"]}
    if set(gold_books) != set(prediction_books):
        missing_predictions = sorted(set(gold_books) - set(prediction_books))
        extra_predictions = sorted(set(prediction_books) - set(gold_books))
        details = []
        if missing_predictions:
            details.append("missing predictions for " + ", ".join(missing_predictions))
        if extra_predictions:
            details.append("unknown prediction books " + ", ".join(extra_predictions))
        raise BenchmarkError("book sets differ: " + "; ".join(details))
    combined_books: list[dict[str, Any]] = []
    for book_id in sorted(gold_books):
        left = gold_books[book_id]
        right = prediction_books[book_id]
        if left["text"] != right["text"] or left["identities"] != right["identities"]:
            raise BenchmarkError(f"gold and predictions disagree on text or cast for book_id: {book_id}")
        combined_books.append({
            "book_id": book_id,
            "text": left["text"],
            "cast": left["cast"],
            "gold_quotes": left["gold_quotes"],
            "predictions": right["predictions"],
        })
    return {"schema_version": SCHEMA_VERSION, "artifact": "interactive-accuracy-corpus", "books": combined_books}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _metrics_from_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    gold_count = counts["gold_count"]
    prediction_count = counts["prediction_count"]
    matched_count = counts["matched_count"]
    precision = _ratio(matched_count, prediction_count)
    recall = _ratio(matched_count, gold_count)
    f1 = _ratio(2 * precision * recall, precision + recall)
    category_metrics = {
        category: {"correct": counts["category_correct"][category], "gold": counts["category_gold"][category], "accuracy": _ratio(counts["category_correct"][category], counts["category_gold"][category])}
        for category in GOLD_CATEGORIES
    }
    return {
        "quote_detection": {"matched": matched_count, "predicted": prediction_count, "gold": gold_count, "precision": precision, "recall": recall, "f1": f1},
        "correct_speaker": {"correct": counts["speaker_correct"], "gold": gold_count, "accuracy": _ratio(counts["speaker_correct"], gold_count)},
        "attribution_by_category": category_metrics,
        "cast_recall": {"recovered": len(counts["recovered_speakers"]), "gold_speakers": len(counts["gold_non_narrator_speakers"]), "recall": _ratio(len(counts["recovered_speakers"]), len(counts["gold_non_narrator_speakers"]))},
        "false_narrator_demotion": {"count": counts["false_narrator_demotion"], "denominator": counts["non_narrator_matched"], "rate": _ratio(counts["false_narrator_demotion"], counts["non_narrator_matched"])},
        "auto_approval_errors": {"count": counts["auto_approval_errors"], "denominator": counts["auto_approval_count"], "rate": _ratio(counts["auto_approval_errors"], counts["auto_approval_count"])},
        "exception_capture": {"captured": counts["exception_captured"], "errors": counts["attribution_errors"], "recall": _ratio(counts["exception_captured"], counts["attribution_errors"])},
        "exception_rate": {"count": counts["exception_count"], "denominator": matched_count, "rate": _ratio(counts["exception_count"], matched_count)},
        "corrections_per_100_gold": round(100 * counts["correction_count"] / gold_count, 6) if gold_count else 0.0,
        "spurious_predictions": prediction_count - matched_count,
        "missed_gold_quotes": gold_count - matched_count,
    }


def _score_book(book: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_quotes = book["gold_quotes"]
    predictions = book["predictions"]
    gold_by_range = {(quote["source_start"], quote["source_end"]): quote for quote in gold_quotes}
    prediction_by_range = {(prediction["source_start"], prediction["source_end"]): prediction for prediction in predictions}
    matched_ranges = sorted(set(gold_by_range) & set(prediction_by_range))
    category_gold = {category: 0 for category in GOLD_CATEGORIES}
    category_correct = {category: 0 for category in GOLD_CATEGORIES}
    speaker_correct = 0
    attribution_errors = 0
    exception_captured = 0
    exception_count = 0
    non_narrator_matched = 0
    false_narrator_demotion = 0
    recovered_speakers: set[str] = set()
    gold_non_narrator_speakers = {quote["speaker_id"] for quote in gold_quotes if quote["speaker_id"] != book["narrator_id"]}
    for quote in gold_quotes:
        category_gold[quote["category"]] += 1
    for source_range in matched_ranges:
        gold = gold_by_range[source_range]
        prediction = prediction_by_range[source_range]
        effective = prediction["effective_speaker_id"]
        if effective == gold["speaker_id"]:
            speaker_correct += 1
            category_correct[gold["category"]] += 1
            if gold["speaker_id"] != book["narrator_id"]:
                recovered_speakers.add(effective)
        else:
            attribution_errors += 1
            if prediction["decision_state"] in EXCEPTION_STATES:
                exception_captured += 1
        if gold["speaker_id"] != book["narrator_id"]:
            non_narrator_matched += 1
            if prediction["original_speaker_id"] not in (None, book["narrator_id"]) and effective == book["narrator_id"]:
                false_narrator_demotion += 1
        if prediction["decision_state"] in EXCEPTION_STATES:
            exception_count += 1
    auto_approval_count = sum(prediction["decision_state"] == "auto_approved" for prediction in predictions)
    auto_approval_errors = 0
    for prediction in predictions:
        source_range = (prediction["source_start"], prediction["source_end"])
        if prediction["decision_state"] == "auto_approved" and (source_range not in gold_by_range or prediction["effective_speaker_id"] != gold_by_range[source_range]["speaker_id"]):
            auto_approval_errors += 1
    counts: dict[str, Any] = {
        "gold_count": len(gold_quotes), "prediction_count": len(predictions), "matched_count": len(matched_ranges),
        "speaker_correct": speaker_correct, "category_gold": category_gold, "category_correct": category_correct,
        "gold_non_narrator_speakers": gold_non_narrator_speakers, "recovered_speakers": recovered_speakers,
        "non_narrator_matched": non_narrator_matched, "false_narrator_demotion": false_narrator_demotion,
        "auto_approval_count": auto_approval_count, "auto_approval_errors": auto_approval_errors,
        "attribution_errors": attribution_errors, "exception_captured": exception_captured, "exception_count": exception_count,
        "correction_count": sum(prediction["decision_state"] == "manual_correction" for prediction in predictions),
    }
    return _metrics_from_counts(counts), counts


def score_corpus(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic per-book and pooled attribution metrics."""
    normalized = validate_corpus(corpus)
    scored_books: list[dict[str, Any]] = []
    pooled: dict[str, Any] | None = None
    pooled_counts: dict[str, Any] | None = None
    for book in normalized["books"]:
        metrics, counts = _score_book(book)
        scored_books.append({"book_id": book["book_id"], "metrics": metrics})
        if pooled_counts is None:
            pooled_counts = counts.copy()
            pooled_counts["category_gold"] = counts["category_gold"].copy()
            pooled_counts["category_correct"] = counts["category_correct"].copy()
            pooled_counts["gold_non_narrator_speakers"] = {
                (book["book_id"], speaker_id) for speaker_id in counts["gold_non_narrator_speakers"]
            }
            pooled_counts["recovered_speakers"] = {
                (book["book_id"], speaker_id) for speaker_id in counts["recovered_speakers"]
            }
        else:
            for key in ("gold_count", "prediction_count", "matched_count", "speaker_correct", "non_narrator_matched", "false_narrator_demotion", "auto_approval_count", "auto_approval_errors", "attribution_errors", "exception_captured", "exception_count", "correction_count"):
                pooled_counts[key] += counts[key]
            for category in GOLD_CATEGORIES:
                pooled_counts["category_gold"][category] += counts["category_gold"][category]
                pooled_counts["category_correct"][category] += counts["category_correct"][category]
            pooled_counts["gold_non_narrator_speakers"].update(
                (book["book_id"], speaker_id) for speaker_id in counts["gold_non_narrator_speakers"]
            )
            pooled_counts["recovered_speakers"].update(
                (book["book_id"], speaker_id) for speaker_id in counts["recovered_speakers"]
            )
    assert pooled_counts is not None
    pooled = _metrics_from_counts(pooled_counts)
    return {"schema_version": SCHEMA_VERSION, "books": scored_books, "pooled": pooled}


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read JSON input {path!r}: {error}") from None


def _load_cli_corpus(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.gold) != bool(args.predictions):
        raise BenchmarkError("--gold and --predictions must be provided together")
    if args.input and (args.gold or args.predictions):
        raise BenchmarkError("use either positional corpus input or --gold/--predictions")
    if args.gold:
        return load_gold_and_predictions(_read_json(args.gold), _read_json(args.predictions))
    if not args.input:
        raise BenchmarkError("provide a corpus path or both --gold and --predictions")
    return validate_corpus(_read_json(args.input))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score exact-range character-to-voice attribution predictions.")
    parser.add_argument("input", nargs="?", help="combined corpus JSON")
    parser.add_argument("--gold", help="separate gold JSON artifact")
    parser.add_argument("--predictions", help="separate prediction JSON artifact")
    parser.add_argument("--output", help="write results JSON to this path as well as stdout")
    args = parser.parse_args(argv)
    try:
        result = score_corpus(_load_cli_corpus(args))
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        if args.output:
            try:
                Path(args.output).write_text(encoded, encoding="utf-8")
            except OSError as error:
                raise BenchmarkError(f"cannot write output {args.output!r}: {error}") from None
        sys.stdout.write(encoded)
        return 0
    except BenchmarkError as error:
        sys.stderr.write(f"error: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

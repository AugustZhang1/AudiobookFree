import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_interactive_accuracy.py"
FIXTURES = Path(__file__).parent / "fixtures" / "interactive_accuracy"
sys.path.insert(0, str(SCRIPT.parent))

from benchmark_interactive_accuracy import BenchmarkError, load_gold_and_predictions, score_corpus  # noqa: E402


def fixture_corpus() -> dict:
    gold = json.loads((FIXTURES / "gold.json").read_text(encoding="utf-8"))
    predictions = json.loads((FIXTURES / "predictions.json").read_text(encoding="utf-8"))
    return load_gold_and_predictions(gold, predictions)


def test_fixture_metrics_cover_detection_speaker_categories_and_review_states() -> None:
    result = score_corpus(fixture_corpus())
    pooled = result["pooled"]
    assert pooled["quote_detection"] == {"matched": 9, "predicted": 10, "gold": 10, "precision": 0.9, "recall": 0.9, "f1": 0.9}
    assert pooled["correct_speaker"] == {"correct": 5, "gold": 10, "accuracy": 0.5}
    assert pooled["attribution_by_category"]["explicit"] == {"correct": 1, "gold": 2, "accuracy": 0.5}
    assert pooled["attribution_by_category"]["anaphoric"]["accuracy"] == 0.0
    assert pooled["cast_recall"] == {"recovered": 4, "gold_speakers": 6, "recall": 0.666667}
    assert pooled["false_narrator_demotion"] == {"count": 1, "denominator": 9, "rate": 0.111111}
    assert pooled["auto_approval_errors"] == {"count": 2, "denominator": 4, "rate": 0.5}
    assert pooled["exception_capture"] == {"captured": 2, "errors": 4, "recall": 0.5}
    assert pooled["exception_rate"] == {"count": 2, "denominator": 9, "rate": 0.222222}
    assert pooled["corrections_per_100_gold"] == 10.0
    assert pooled["spurious_predictions"] == 1
    assert pooled["missed_gold_quotes"] == 1


def test_per_book_metrics_do_not_leak_into_each_other() -> None:
    result = score_corpus(fixture_corpus())
    books = {book["book_id"]: book["metrics"] for book in result["books"]}
    assert books["book-alpha"]["quote_detection"]["matched"] == 7
    assert books["book-alpha"]["quote_detection"]["gold"] == 8
    assert books["book-beta"]["quote_detection"] == {"matched": 2, "predicted": 2, "gold": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert books["book-beta"]["false_narrator_demotion"]["count"] == 0


def test_pooled_cast_recall_scopes_repeated_speaker_ids_by_book() -> None:
    corpus = fixture_corpus()
    second_book = corpus["books"][1]
    for entry in second_book["cast"]:
        if entry["speaker_id"] == "doctor":
            entry["speaker_id"] = "alice"
    for quote in second_book["gold_quotes"]:
        if quote["speaker_id"] == "doctor":
            quote["speaker_id"] = "alice"
    for prediction in second_book["predictions"]:
        for field in ("predicted_speaker_id", "original_speaker_id", "effective_speaker_id"):
            if prediction[field] == "doctor":
                prediction[field] = "alice"
    metrics = score_corpus(corpus)["pooled"]["cast_recall"]
    assert metrics == {"recovered": 4, "gold_speakers": 6, "recall": 0.666667}


def test_multiline_source_text_with_boundary_whitespace_is_valid() -> None:
    corpus = {
        "schema_version": 1,
        "artifact": "interactive-accuracy-corpus",
        "books": [{
            "book_id": "multiline",
            "text": "\n  \"Hi\" said Alice.\r\n",
            "cast": [
                {"speaker_id": "narrator", "role": "narrator"},
                {"speaker_id": "alice", "role": "character"},
            ],
            "gold_quotes": [{"quote_id": "q1", "source_start": 3, "source_end": 7, "speaker_id": "alice", "category": "explicit"}],
            "predictions": [{"prediction_id": "p1", "source_start": 3, "source_end": 7, "predicted_speaker_id": "alice", "original_speaker_id": "alice", "effective_speaker_id": "alice", "decision_state": "reviewed"}],
        }],
    }
    assert score_corpus(corpus)["pooled"]["correct_speaker"]["accuracy"] == 1.0


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_range", "bad_range"])
def test_malformed_duplicate_ids_and_ranges_are_rejected(mutation: str) -> None:
    corpus = fixture_corpus()
    broken = copy.deepcopy(corpus)
    if mutation == "duplicate_id":
        broken["books"][0]["gold_quotes"][1]["quote_id"] = broken["books"][0]["gold_quotes"][0]["quote_id"]
    elif mutation == "duplicate_range":
        broken["books"][0]["gold_quotes"][1]["source_start"] = broken["books"][0]["gold_quotes"][0]["source_start"]
        broken["books"][0]["gold_quotes"][1]["source_end"] = broken["books"][0]["gold_quotes"][0]["source_end"]
    else:
        broken["books"][0]["gold_quotes"][0]["source_end"] = len(broken["books"][0]["text"]) + 1
    with pytest.raises(BenchmarkError):
        score_corpus(broken)


def test_zero_denominators_are_safe() -> None:
    empty = {
        "schema_version": 1,
        "artifact": "interactive-accuracy-corpus",
        "books": [{
            "book_id": "empty",
            "text": "Synthetic narration.",
            "cast": [{"speaker_id": "narrator", "role": "narrator"}],
            "gold_quotes": [],
            "predictions": [],
        }],
    }
    metrics = score_corpus(empty)["pooled"]
    assert metrics["quote_detection"] == {"matched": 0, "predicted": 0, "gold": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert metrics["correct_speaker"]["accuracy"] == 0.0
    assert metrics["cast_recall"] == {"recovered": 0, "gold_speakers": 0, "recall": 0.0}
    assert metrics["auto_approval_errors"]["rate"] == 0.0
    assert metrics["exception_capture"]["recall"] == 0.0
    assert metrics["corrections_per_100_gold"] == 0.0


def test_cli_json_is_deterministic_and_does_not_import_or_launch_application() -> None:
    command = [sys.executable, str(SCRIPT), "--gold", str(FIXTURES / "gold.json"), "--predictions", str(FIXTURES / "predictions.json")]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["schema_version"] == 1
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pdf_audiobook" not in source
    assert "subprocess" not in source


def test_cli_invalid_input_is_sanitized_and_nonzero() -> None:
    completed = subprocess.run([sys.executable, str(SCRIPT), str(FIXTURES / "does-not-exist.json")], capture_output=True, text=True)
    assert completed.returncode != 0
    assert completed.stderr.startswith("error: ")
    assert "Traceback" not in completed.stderr

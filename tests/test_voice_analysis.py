from __future__ import annotations

import copy
import hashlib
import uuid

import pytest

import pdf_audiobook.voice_analysis as voice_analysis_module
from pdf_audiobook.voice_analysis import (
    VoiceAnalysisError,
    validate_voice_analysis_status,
)
from pdf_audiobook.voice_plan import canonical_json_bytes, with_canonical_artifact_hash


def _base(status: str = "queued", stage: str = "queued") -> tuple[str, dict, dict]:
    text = "One chapter."
    plan = {
        "schema_version": 1,
        "mode": "whole",
        "requested_count": None,
        "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}],
        "warnings": [],
    }
    artifact = {
        "schema_version": 1,
        "artifact": "voice-analysis-status",
        "analysis_id": str(uuid.UUID("12345678-1234-5678-9234-567812345678")),
        "revision": 1,
        "source_pdf_sha256": "a" * 64,
        "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapter_plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "status": status,
        "stage": stage,
        "progress": {"completed": 0, "total": 0},
        "cancel_requested": status in {"running", "cancelled"},
        "warnings": [],
        "error": {"code": "ANALYZER_FAILED", "message": "failed"} if status == "failed" else None,
        "started_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:01Z",
        "finished_at": "2026-01-01T00:00:02Z" if status in {"completed", "cancelled", "failed"} else None,
    }
    return text, plan, with_canonical_artifact_hash(artifact)


def _validate(artifact: dict, text: str, plan: dict) -> dict:
    return validate_voice_analysis_status(
        artifact,
        text,
        plan,
        expected_source_pdf_sha256="a" * 64,
        expected_chapter_plan_sha256=artifact["chapter_plan_sha256"],
    )


def test_valid_status_and_all_state_stage_pairs() -> None:
    text, plan, artifact = _base()
    assert _validate(artifact, text, plan) is artifact
    for status, stage in (("queued", "queued"), ("running", "preparing"), ("running", "analyzing"), ("running", "validating"), ("running", "persisting"), ("cancelled", "cancelled"), ("failed", "failed")):
        _, _, candidate = _base(status, stage)
        assert _validate(candidate, text, plan) is candidate
    _, _, completed = _base("completed", "completed")
    completed["progress"] = {"completed": 0, "total": 0}
    completed = with_canonical_artifact_hash(completed)
    assert _validate(completed, text, plan) is completed


def test_analyzed_range_fields_validate_and_legacy_artifacts_are_full_plan() -> None:
    text, plan, legacy = _base()
    assert "chapter_start" not in legacy and _validate(legacy, text, plan) is legacy
    ranged = with_canonical_artifact_hash({**legacy, "chapter_start": 1, "chapter_end": 1})
    assert _validate(ranged, text, plan) is ranged
    partial = with_canonical_artifact_hash({**legacy, "chapter_start": 1})
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(partial, text, plan)
    assert exc.value.code == "INVALID_STATUS"
    out_of_bounds = with_canonical_artifact_hash({**legacy, "chapter_start": 1, "chapter_end": 2})
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(out_of_bounds, text, plan)
    assert exc.value.code == "INVALID_CHAPTER_RANGE"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("analysis_id", True, "INVALID_ANALYSIS_ID"),
        ("revision", True, "INVALID_REVISION"),
        ("status", "bad", "INVALID_STATUS_VALUE"),
        ("stage", "analyzing", "INVALID_STAGE"),
        ("cancel_requested", 1, "INVALID_CANCEL_REQUEST"),
    ],
)
def test_status_scalar_and_enum_validation(field: str, value: object, code: str) -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad[field] = value
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == code


def test_progress_error_timestamps_and_error_contract() -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad["progress"] = {"completed": 2, "total": 1}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_PROGRESS"
    bad = copy.deepcopy(artifact)
    bad["updated_at"] = "2025-01-01T00:00:00Z"
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_TIMESTAMP_ORDER"
    bad = copy.deepcopy(artifact)
    bad["error"] = {"code": "oops", "message": "no"}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_ERROR"


def test_status_bindings_unknown_fields_hash_and_size() -> None:
    text, plan, artifact = _base()
    bad = {**artifact, "extra": 1}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "INVALID_STATUS"
    bad = with_canonical_artifact_hash({**artifact, "source_pdf_sha256": "b" * 64})
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "SOURCE_HASH_MISMATCH"
    with pytest.raises(VoiceAnalysisError) as exc:
        validate_voice_analysis_status(artifact, text, plan, expected_source_pdf_sha256="a" * 64, expected_chapter_plan_sha256="b" * 64)
    assert exc.value.code == "CHAPTER_PLAN_HASH_MISMATCH"
    bad = dict(artifact)
    bad["canonical_artifact_sha256"] = "0" * 64
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "ARTIFACT_HASH_MISMATCH"


@pytest.mark.parametrize("nested, code", [("analyzer", "INVALID_ANALYZER"), ("progress", "INVALID_PROGRESS"), ("error", "INVALID_ERROR")])
def test_status_rejects_unknown_nested_fields(nested: str, code: str) -> None:
    text, plan, artifact = _base("failed", "failed") if nested == "error" else _base()
    bad = copy.deepcopy(artifact)
    bad[nested]["extra"] = 1
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == code


def test_status_rejects_bool_progress_and_invalid_cancel_combinations() -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad["progress"]["completed"] = True
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_PROGRESS"
    for status, stage, cancel in (("queued", "queued", True), ("completed", "completed", True), ("failed", "failed", True), ("cancelled", "cancelled", False)):
        _, _, candidate = _base(status, stage)
        candidate["cancel_requested"] = cancel
        if status == "failed" and candidate["error"] is None:
            candidate["error"] = {"code": "ANALYZER_FAILED", "message": "failed"}
        with pytest.raises(VoiceAnalysisError) as exc:
            _validate(with_canonical_artifact_hash(candidate), text, plan)
        assert exc.value.code == "INVALID_CANCEL_REQUEST"


def test_status_rejects_error_and_finished_timestamp_contracts() -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad["error"] = {"code": "ANALYZER_FAILED", "message": "failed"}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_ERROR"
    bad = copy.deepcopy(artifact)
    bad["finished_at"] = "2026-01-01T00:00:02Z"
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_TIMESTAMP"
    bad = _base("failed", "failed")[2]
    bad["finished_at"] = None
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_TIMESTAMP"


def test_status_cleaned_and_current_chapter_bindings_are_checked() -> None:
    text, plan, artifact = _base()
    bad = with_canonical_artifact_hash({**artifact, "cleaned_text_sha256": "b" * 64})
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "CLEANED_TEXT_HASH_MISMATCH"
    changed_plan = {**plan, "warnings": ["changed"]}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(artifact, text, changed_plan)
    assert exc.value.code == "CHAPTER_PLAN_HASH_MISMATCH"


def test_status_warning_and_canonical_size_limits_are_enforced(monkeypatch) -> None:
    text, plan, artifact = _base()
    monkeypatch.setattr(voice_analysis_module, "MAX_WARNINGS_BYTES", 3)
    warning_limited = with_canonical_artifact_hash({**artifact, "warnings": ["warn"]})
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(warning_limited, text, plan)
    assert exc.value.code == "INVALID_WARNINGS"
    monkeypatch.setattr(voice_analysis_module, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(artifact, text, plan)
    assert exc.value.code == "STATUS_TOO_LARGE"


def test_status_error_code_requires_uppercase_identifier() -> None:
    text, plan, artifact = _base("failed", "failed")
    artifact["error"] = {"code": "bad-code", "message": "failed"}
    with pytest.raises(VoiceAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(artifact), text, plan)
    assert exc.value.code == "INVALID_ERROR"

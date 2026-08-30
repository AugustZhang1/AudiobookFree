from __future__ import annotations

import copy
import hashlib

import pytest

from pdf_audiobook.speaker_analysis import (
    MAX_ENTITIES,
    MAX_SPANS,
    SpeakerAnalysisError,
    validate_speaker_analysis,
)
from pdf_audiobook.voice_plan import (
    canonical_json_bytes,
    with_canonical_artifact_hash,
)


def _base() -> tuple[str, dict, dict]:
    text = "Alice says, \"Hi,  世界!\"\nBob waits."
    split = text.index("Bob")
    chapter_plan = {
        "schema_version": 1,
        "mode": "whole",
        "requested_count": None,
        "cleaned_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chapters": [
            {"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 4},
            {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2},
        ],
        "warnings": [],
    }
    artifact = {
        "schema_version": 1,
        "artifact": "speaker-analysis",
        "revision": 1,
        "source_pdf_sha256": "a" * 64,
        "cleaned_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chapter_plan_sha256": hashlib.sha256(canonical_json_bytes(chapter_plan)).hexdigest(),
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "characters": [
            {
                "character_id": "alice",
                "canonical_label": "Alice",
                "aliases": [
                    {
                        "alias": "Alice",
                        "kind": "proper",
                        "confidence": 0.9,
                        "provenance": {"source": "fake", "token_start": 0, "token_end": 1},
                    }
                ],
                "line_count": 1,
                "quote_count": 1,
            },
            {
                "character_id": "bob",
                "canonical_label": "Bob",
                "aliases": [],
                "line_count": 0,
                "quote_count": 0,
            },
        ],
        "spans": [
            {
                "span_id": "s1",
                "chapter_index": 1,
                "source_start": 0,
                "source_end": split,
                "type": "dialogue",
                "speaker_id": "alice",
                "confidence": {"score": 0.1, "band": "high", "reasons": ["explicit quote"]},
                "provenance": {"source": "fake", "quote_id": "q1"},
            },
            {
                "span_id": "s2",
                "chapter_index": 2,
                "source_start": split + 1,
                "source_end": len(text),
                "type": "narration",
                "speaker_id": None,
                "confidence": {"score": 0.8, "band": "low", "reasons": []},
                "provenance": {"source": "fake"},
            },
        ],
        "warnings": ["machine attribution is incomplete"],
        "canonical_artifact_sha256": "",
    }
    return text, chapter_plan, with_canonical_artifact_hash(artifact)


def _validate(artifact: dict, text: str, plan: dict) -> dict:
    return validate_speaker_analysis(
        artifact,
        text,
        plan,
        expected_source_pdf_sha256="a" * 64,
        expected_chapter_plan_sha256=artifact["chapter_plan_sha256"],
    )


def test_valid_unicode_disjoint_machine_analysis_and_canonical_round_trip() -> None:
    text, plan, artifact = _base()
    assert _validate(artifact, text, plan) is artifact
    assert canonical_json_bytes(artifact).endswith(b"\n")
    assert canonical_json_bytes({"z": "世界", "a": 1}) == '{"a":1,"z":"世界"}\n'.encode("utf-8")
    assert artifact["spans"][0]["confidence"] == {"score": 0.1, "band": "high", "reasons": ["explicit quote"]}


def test_analyzed_range_fields_validate_and_legacy_artifacts_are_full_plan() -> None:
    text, plan, legacy = _base()
    assert "chapter_start" not in legacy and _validate(legacy, text, plan) is legacy
    ranged = with_canonical_artifact_hash({**legacy, "chapter_start": 1, "chapter_end": 2})
    assert _validate(ranged, text, plan) is ranged
    partial = with_canonical_artifact_hash({**legacy, "chapter_start": 1})
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(partial, text, plan)
    assert exc.value.code == "INVALID_ANALYSIS"
    for start, end, expected in ((True, 2, "INVALID_CHAPTER_RANGE"), (1, 3, "INVALID_CHAPTER_RANGE")):
        invalid = with_canonical_artifact_hash({**legacy, "chapter_start": start, "chapter_end": end})
        with pytest.raises(SpeakerAnalysisError) as exc:
            _validate(invalid, text, plan)
        assert exc.value.code == expected


def test_strict_top_and_nested_fields_are_rejected() -> None:
    text, plan, artifact = _base()
    for path in (("extra",), ("analyzer", "extra"), ("characters", 0, "extra"), ("spans", 0, "extra"), ("spans", 0, "provenance", "extra")):
        bad = copy.deepcopy(artifact)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = 1
        bad = with_canonical_artifact_hash(bad)
        with pytest.raises(SpeakerAnalysisError):
            _validate(bad, text, plan)


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("schema_version",), True, "INVALID_SCHEMA_VERSION"),
        (("revision",), True, "INVALID_REVISION"),
        (("chapter_plan_schema_version",), True, "CHAPTER_PLAN_SCHEMA_MISMATCH"),
        (("characters", 0, "character_id"), "bad id!", "INVALID_CHARACTER_ID"),
        (("characters", 0, "line_count"), True, "INVALID_CHARACTER_COUNT"),
        (("characters", 0, "aliases", 0, "kind"), "other", "INVALID_ALIAS_KIND"),
        (("characters", 0, "aliases", 0, "provenance", "token_start"), True, "INVALID_TOKEN_RANGE"),
        (("spans", 0, "chapter_index"), True, "INVALID_CHAPTER_INDEX"),
        (("spans", 0, "type"), "other", "INVALID_SPAN_TYPE"),
        (("spans", 0, "confidence", "score"), 2, "INVALID_CONFIDENCE"),
        (("spans", 0, "confidence", "band"), "certain", "INVALID_CONFIDENCE_BAND"),
        (("spans", 0, "confidence", "reasons"), "not-a-list", "INVALID_CONFIDENCE_REASONS"),
    ],
)
def test_scalar_and_enum_validation(path: tuple, value: object, code: str) -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bad = with_canonical_artifact_hash(bad)
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == code


def test_duplicate_alias_and_unknown_character_reference_rejected() -> None:
    text, plan, artifact = _base()
    duplicate = copy.deepcopy(artifact)
    duplicate["characters"][0]["aliases"].append(copy.deepcopy(duplicate["characters"][0]["aliases"][0]))
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(duplicate), text, plan)
    assert exc.value.code == "DUPLICATE_ALIAS"
    unknown = copy.deepcopy(artifact)
    unknown["spans"][0]["speaker_id"] = "nobody"
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(unknown), text, plan)
    assert exc.value.code == "UNKNOWN_SPEAKER_REFERENCE"


def test_duplicate_character_id_rejected() -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad["characters"][1]["character_id"] = bad["characters"][0]["character_id"]
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "DUPLICATE_CHARACTER_ID"


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("characters", 0, "aliases", 0, "confidence"), True, "INVALID_ALIAS_CONFIDENCE"),
        (("characters", 0, "aliases", 0, "confidence"), 2, "INVALID_ALIAS_CONFIDENCE"),
    ],
)
def test_alias_confidence_is_finite_bounded_and_not_boolean(path: tuple, value: object, code: str) -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == code


def test_alias_provenance_rejects_unknown_fields() -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    bad["characters"][0]["aliases"][0]["provenance"]["extra"] = "nope"
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == "INVALID_ALIAS_PROVENANCE"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda a: a["spans"].__setitem__(1, {**a["spans"][1], "source_start": a["spans"][0]["source_start"]}), "OVERLAPPING_SPANS"),
        (lambda a: a["spans"].__setitem__(0, {**a["spans"][0], "source_end": len(_base()[0]) + 1}), "OUT_OF_RANGE"),
        (lambda a: a["spans"].__setitem__(0, {**a["spans"][0], "chapter_index": 2}), "CROSS_CHAPTER"),
    ],
)
def test_span_range_validation(change, code: str) -> None:
    text, plan, artifact = _base()
    bad = copy.deepcopy(artifact)
    change(bad)
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(with_canonical_artifact_hash(bad), text, plan)
    assert exc.value.code == code


def test_hash_bindings_and_canonical_tamper_rejected() -> None:
    text, plan, artifact = _base()
    bad = dict(artifact, canonical_artifact_sha256="0" * 64)
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "ARTIFACT_HASH_MISMATCH"
    bad = with_canonical_artifact_hash({**artifact, "source_pdf_sha256": "b" * 64})
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "SOURCE_HASH_MISMATCH"
    bad = with_canonical_artifact_hash({**artifact, "cleaned_text_sha256": "b" * 64})
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "CLEANED_TEXT_HASH_MISMATCH"
    bad = with_canonical_artifact_hash({**artifact, "chapter_plan_sha256": "b" * 64})
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "CHAPTER_PLAN_HASH_MISMATCH"
    with pytest.raises(SpeakerAnalysisError) as exc:
        validate_speaker_analysis(
            artifact,
            text,
            plan,
            expected_source_pdf_sha256="a" * 64,
            expected_chapter_plan_sha256="b" * 64,
        )
    assert exc.value.code == "CHAPTER_PLAN_HASH_MISMATCH"


def test_warning_and_count_limits_can_be_exercised_without_large_allocations(monkeypatch) -> None:
    text, plan, artifact = _base()
    monkeypatch.setattr("pdf_audiobook.speaker_analysis.MAX_WARNINGS_BYTES", 3)
    warning_limited = with_canonical_artifact_hash({**artifact, "warnings": ["warn"]})
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(warning_limited, text, plan)
    assert exc.value.code == "INVALID_WARNINGS"
    monkeypatch.setattr("pdf_audiobook.speaker_analysis.MAX_ENTITIES", 1)
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(artifact, text, plan)
    assert exc.value.code == "ANALYSIS_TOO_LARGE"
    monkeypatch.setattr("pdf_audiobook.speaker_analysis.MAX_ENTITIES", MAX_ENTITIES)
    monkeypatch.setattr("pdf_audiobook.speaker_analysis.MAX_SPANS", 1)
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(artifact, text, plan)
    assert exc.value.code == "ANALYSIS_TOO_LARGE"


def test_unpaired_surrogate_is_rejected_as_invalid_canonical_artifact() -> None:
    text, plan, artifact = _base()
    bad = {**artifact, "warnings": ["\ud800"]}
    with pytest.raises(SpeakerAnalysisError) as exc:
        _validate(bad, text, plan)
    assert exc.value.code == "INVALID_ANALYSIS"

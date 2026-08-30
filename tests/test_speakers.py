import copy
import math

import pytest

from pdf_audiobook.speakers import (
    Confidence,
    MachineAnalysis,
    SpeakerAnalyzer,
    SpeakerPlanError,
    SpeakerSpan,
    validate_approved_spans,
    validate_draft_spans,
    validate_machine_spans,
)


def plan_for(text: str, split: int) -> dict:
    return {
        "schema_version": 1,
        "mode": "original",
        "requested_count": None,
        "cleaned_text_sha256": "unused-by-core",
        "chapters": [
            {"index": 1, "start_offset": 0, "end_offset": split},
            {"index": 2, "start_offset": split, "end_offset": len(text)},
        ],
    }


def span(
    span_id: str,
    chapter: int,
    start: int,
    end: int,
    span_type: str = "narration",
    speaker: str | None = "narrator",
    score: float = 0.9,
) -> SpeakerSpan:
    return SpeakerSpan(
        span_id=span_id,
        chapter_index=chapter,
        source_start=start,
        source_end=end,
        span_type=span_type,
        speaker_id=speaker,
        confidence=Confidence(score, "high", reasons=["fixture"]),
        provenance=("fake",),
    )


def test_approved_spans_reconstruct_unicode_punctuation_and_whitespace_exactly() -> None:
    text = "“Come here, ”  Zoë said.\n第二章 — déjà vu"
    split = text.index("第二章")
    plan = plan_for(text, split)
    values = [
        span("a", 1, 0, text.index("Zoë"), "dialogue", "zoe"),
        span("b", 1, text.index("Zoë"), split),
        span("c", 2, split, len(text), "thought", "narrator"),
    ]
    validated = validate_approved_spans(values, text, plan)
    assert validated == tuple(values)
    assert "".join(text[item.source_start : item.source_end] for item in validated) == text
    assert text[values[0].source_start : values[0].source_end] == "“Come here, ”  "


def test_machine_gaps_are_allowed_but_overlap_range_chapter_and_duplicate_id_are_rejected() -> None:
    text = "one two\nthree four"
    plan = plan_for(text, text.index("three"))
    assert validate_machine_spans([span("one", 1, 0, 3)], text, plan)
    cases = [
        ([span("a", 1, 0, 4), span("b", 1, 3, 5)], "OVERLAPPING_SPANS"),
        ([span("a", 1, 0, 100)], "OUT_OF_RANGE"),
        ([span("a", 1, 0, len(text))], "CROSS_CHAPTER"),
        ([span("a", 1, 0, 3), span("a", 1, 4, 7)], "DUPLICATE_SPAN_ID"),
        ([span("b", 1, 4, 7), span("a", 1, 0, 3)], "UNORDERED_SPANS"),
    ]
    for values, code in cases:
        with pytest.raises(SpeakerPlanError) as error:
            validate_machine_spans(values, text, plan)
        assert error.value.code == code


def test_approved_plan_rejects_gaps_overlaps_reordering_missing_or_extra_chapters() -> None:
    text = "abcdefghij"
    plan = plan_for(text, 5)
    complete = [span("a", 1, 0, 2), span("b", 1, 2, 5), span("c", 2, 5, 10)]
    assert validate_approved_spans(complete, text, plan)
    invalid = [
        ([span("a", 1, 0, 2), span("b", 1, 3, 5), span("c", 2, 5, 10)], "INCOMPLETE_COVERAGE"),
        ([span("a", 1, 0, 3), span("b", 1, 2, 5), span("c", 2, 5, 10)], "OVERLAPPING_SPANS"),
        ([span("c", 2, 5, 10), span("a", 1, 0, 5)], "UNORDERED_SPANS"),
        ([span("a", 1, 0, 5)], "INCOMPLETE_COVERAGE"),
        ([span("d", 3, 9, 10)], "INVALID_CHAPTER_INDEX"),
    ]
    for values, code in invalid:
        with pytest.raises(SpeakerPlanError) as error:
            validate_approved_spans(values, text, plan)
        assert error.value.code == code


def test_invalid_ids_types_indices_and_reconstruction_are_rejected() -> None:
    text = "abcdefghij"
    plan = plan_for(text, 5)
    with pytest.raises(SpeakerPlanError) as error:
        validate_machine_spans([span("a", 1, 0, 5), span("a", 2, 5, 10)], text, plan)
    assert error.value.code == "DUPLICATE_SPAN_ID"
    with pytest.raises(SpeakerPlanError) as error:
        SpeakerSpan("", 1, 0, 5, "narration", "narrator", Confidence(0.9, "high"), ())
    assert error.value.code == "INVALID_SPAN_ID"
    with pytest.raises(SpeakerPlanError) as error:
        SpeakerSpan("a", True, 0, 5, "narration", "narrator", Confidence(0.9, "high"), ())
    assert error.value.code == "INVALID_CHAPTER_INDEX"
    with pytest.raises(SpeakerPlanError) as error:
        SpeakerSpan("a", 1, 0, 5, "bad", "narrator", Confidence(0.9, "high"), ())
    assert error.value.code == "INVALID_SPAN_TYPE"
    bad_plan = copy.deepcopy(plan)
    bad_plan["chapters"][1]["start_offset"] = 6
    with pytest.raises(SpeakerPlanError):
        validate_machine_spans([span("a", 1, 0, 5)], text, bad_plan)


def test_confidence_is_finite_bounded_banded_and_immutable() -> None:
    assert Confidence(0, "high").band == "high"
    assert Confidence(0.5, "low").band == "low"
    confidence = Confidence(0.9, "medium", reasons=["explicit_tag"])
    assert confidence.reasons == ("explicit_tag",)
    with pytest.raises(TypeError):
        Confidence(0.9)
    with pytest.raises(SpeakerPlanError):
        Confidence(math.inf, "high")
    with pytest.raises(SpeakerPlanError):
        Confidence(-0.01, "low")
    with pytest.raises(SpeakerPlanError):
        Confidence(0.9, "invalid")
    with pytest.raises(SpeakerPlanError) as error:
        Confidence(0.9, "high", reasons={"set"})
    assert error.value.code == "INVALID_REASONS"
    with pytest.raises((AttributeError, TypeError)):
        confidence.score = 0.2


def test_unknown_requires_explicit_narrator_fallback_assignment() -> None:
    text = "unknown"
    plan = {"chapters": [{"index": 1, "start_offset": 0, "end_offset": len(text)}]}
    unresolved = [span("u", 1, 0, len(text), "unknown", None)]
    with pytest.raises(SpeakerPlanError) as error:
        validate_approved_spans(unresolved, text, plan)
    assert error.value.code == "UNRESOLVED_SPANS"
    with pytest.raises(SpeakerPlanError) as error:
        validate_approved_spans(unresolved, text, plan, narrator_fallback_accepted=True)
    assert error.value.code == "UNRESOLVED_SPANS"
    accepted = [span("u", 1, 0, len(text), "unknown", "narrator")]
    assert validate_approved_spans(accepted, text, plan, narrator_fallback_accepted=True) == tuple(accepted)


def test_draft_validation_allows_unresolved_speaker_but_approved_defaults_reject() -> None:
    text = "unknown"
    plan = {"chapters": [{"index": 1, "start_offset": 0, "end_offset": len(text)}]}
    unresolved = [span("u", 1, 0, len(text), "narration", None)]

    assert validate_approved_spans(unresolved, text, plan, allow_unresolved=True) == tuple(unresolved)
    assert validate_draft_spans(unresolved, text, plan) == tuple(unresolved)
    with pytest.raises(SpeakerPlanError) as error:
        validate_approved_spans(unresolved, text, plan)
    assert error.value.code == "UNASSIGNED_SPEAKER"


def test_fake_analyzer_satisfies_protocol_in_process() -> None:
    text = "hello"
    plan = {"chapters": [{"index": 1, "start_offset": 0, "end_offset": len(text)}]}

    class FakeAnalyzer:
        def analyze(self, cleaned_text: str, chapter_plan: dict, source_hash: str, options=None) -> MachineAnalysis:
            assert cleaned_text == text
            assert chapter_plan is plan
            assert source_hash == "source"
            return MachineAnalysis((span("machine", 1, 0, 2, speaker=None),))

    analyzer = FakeAnalyzer()
    assert isinstance(analyzer, SpeakerAnalyzer)
    result = analyzer.analyze(text, plan, "source", {"offline": True})
    assert validate_machine_spans(result, text, plan)[0].speaker_id is None


def test_machine_analysis_normalizes_characters_and_bounded_warnings() -> None:
    character = {"character_id": "alice"}
    result = MachineAnalysis((span("machine", 1, 0, 2, speaker=None),), "source", (), (character,), ("warning",))
    character["character_id"] = "changed"
    assert result.characters == ({"character_id": "alice"},)
    assert result.warnings == ("warning",)
    with pytest.raises(SpeakerPlanError) as error:
        MachineAnalysis((), characters=({"character_id": "alice"},), warnings=("bad\nwarning",))
    assert error.value.code == "INVALID_WARNINGS"

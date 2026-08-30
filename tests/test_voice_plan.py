import hashlib
import math
import copy
from types import SimpleNamespace

import pytest

from bisect import bisect_right as real_bisect_right
from pdf_audiobook import voice_plan as voice_plan_module
from pdf_audiobook.speakers import Confidence, SpeakerSpan
from pdf_audiobook.voice_plan import (
    approve_voice_plan,
    assign_cast,
    build_voice_plan,
    CastEntry,
    VoicePlanError,
    canonical_artifact_digest,
    canonical_json_bytes,
    canonical_json_text,
    merge_aliases,
    merge_cast,
    override_span,
    remove_cast,
    rename_cast,
    review_summary,
    split_aliases,
    validate_voice_plan,
    validate_voice_plan_core,
    verify_canonical_artifact_hash,
    with_canonical_artifact_hash,
)


def lifecycle_inputs(characters: list[dict] | None = None, spans: list[dict] | None = None) -> tuple[dict, str, dict, str, str]:
    text = "abcdefghij"
    plan = {"schema_version": 1, "chapters": [{"index": 1, "start_offset": 0, "end_offset": 5, "start_page": 1, "end_page": 1}, {"index": 2, "start_offset": 5, "end_offset": 10, "start_page": 2, "end_page": 2}]}
    source = "b" * 64
    analysis = {"source_pdf_sha256": source, "analyzer": {"id": "fixture", "version": "1", "model_hash": None}, "characters": characters or [], "spans": spans or []}
    chapter_hash = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    return analysis, text, plan, source, chapter_hash


def observed_lifecycle_inputs(characters: list[dict], speaker_ids: list[str]) -> tuple[dict, str, dict, str, str]:
    text = "x" * len(speaker_ids)
    plan = {"schema_version": 1, "chapters": [{"index": 1, "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1}]}
    source = "b" * 64
    spans = [{"span_id": f"s-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": speaker} for index, speaker in enumerate(speaker_ids)]
    analysis = {"source_pdf_sha256": source, "analyzer": {"id": "fixture", "version": "1", "model_hash": None}, "characters": characters, "spans": spans}
    chapter_hash = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    return analysis, text, plan, source, chapter_hash


def chapter_plan(text: str, split: int) -> dict:
    return {
        "chapters": [
            {"index": 1, "start_offset": 0, "end_offset": split},
            {"index": 2, "start_offset": split, "end_offset": len(text)},
        ]
    }


def span(
    span_id: str,
    chapter: int,
    start: int,
    end: int,
    speaker: str | None,
    span_type: str = "narration",
) -> SpeakerSpan:
    return SpeakerSpan(
        span_id=span_id,
        chapter_index=chapter,
        source_start=start,
        source_end=end,
        span_type=span_type,
        speaker_id=speaker,
        confidence=Confidence(0.9, "high", reasons=("fixture",)),
        provenance=("fixture",),
    )


def valid_fixture() -> tuple[str, dict, list[CastEntry], list[SpeakerSpan]]:
    text = "Narrator says hello. Alice replies! Bob waits."
    split = text.index("Bob")
    plan = chapter_plan(text, split)
    cast = [
        CastEntry("narrator", "Narrator", "narrator", "third_person", "voice-neutral", 1.0),
        CastEntry("alice", "Alice", "character", "same_as_narrator", "voice-neutral", 1.1),
        CastEntry("bob", "Bob", "character", "separate_from_narrator", "voice-neutral", 0.9),
    ]
    alice_start = text.index("Alice")
    bob_start = text.index("Bob")
    spans = [
        span("s1", 1, 0, alice_start, "narrator"),
        span("s2", 1, alice_start, bob_start, "alice", "dialogue"),
        span("s3", 2, bob_start, len(text), "bob"),
    ]
    return text, plan, cast, spans


def approved_artifact_fixture() -> tuple[dict, str, dict, str, str]:
    text = "Narrator speaks. Alice replies! Bob waits."
    bob_start = text.index("Bob")
    plan = {
        "schema_version": 1,
        "chapters": [
            {"index": 1, "start_offset": 0, "end_offset": bob_start, "start_page": 1, "end_page": 1},
            {"index": 2, "start_offset": bob_start, "end_offset": len(text), "start_page": 2, "end_page": 2},
        ],
    }
    source_hash = "a" * 64
    chapter_hash = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    cleaned_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    artifact = {
        "schema_version": 1,
        "artifact": "voice-plan",
        "revision": 3,
        "source_pdf_sha256": source_hash,
        "cleaned_text_sha256": cleaned_hash,
        "chapter_plan_sha256": chapter_hash,
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "cast": [
            {"cast_id": "narrator", "display_label": "Narrator", "role": "narrator", "relationship": "third_person", "voice_id": "voice-neutral", "voice_settings": {"speed": 1.0}},
            {"cast_id": "alice", "display_label": "Alice", "role": "character", "relationship": "same_as_narrator", "voice_id": "voice-neutral", "voice_settings": {"speed": 1.1}},
            {"cast_id": "bob", "display_label": "Bob", "role": "character", "relationship": "separate_from_narrator", "voice_id": "voice-neutral", "voice_settings": {"speed": 0.9}},
        ],
        "aliases": [{"alias_id": "a1", "text": "Al", "character_id": "alice", "override_state": "accepted"}],
        "chapters": [
            {"chapter_index": 1, "source_start": 0, "source_end": bob_start, "source_page_start": 1, "source_page_end": 1, "spans": [
                {"span_id": "s1", "source_start": 0, "source_end": text.index("Alice"), "type": "narration", "speaker_id": "narrator", "confidence": {"score": 0.2, "band": "high", "reasons": ["fixture"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
                {"span_id": "s2", "source_start": text.index("Alice"), "source_end": bob_start, "type": "dialogue", "speaker_id": "alice", "confidence": {"score": 0.9, "band": "low", "reasons": ["explicit_tag"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": {"kind": "speaker", "from": "machine", "to": "alice", "actor": "user", "reason": "reviewed"}},
            ]},
            {"chapter_index": 2, "source_start": bob_start, "source_end": len(text), "source_page_start": 2, "source_page_end": 2, "spans": [
                {"span_id": "s3", "source_start": bob_start, "source_end": len(text), "type": "narration", "speaker_id": "bob", "confidence": {"score": 0.5, "band": "medium", "reasons": []}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
            ]},
        ],
        "unresolved_policy": {"mode": "narrator", "accepted_by_user": False, "accepted_at": None},
        "approval": {"state": "approved", "approved_at": "2026-01-01T00:00:00Z", "approved_revision": 3},
    }
    return with_canonical_artifact_hash(artifact), text, plan, source_hash, chapter_hash


def test_validate_voice_plan_strict_structured_artifact_and_bindings() -> None:
    artifact, text, plan, source_hash, chapter_hash = approved_artifact_fixture()
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_validate_voice_plan_rejects_boolean_approved_revision() -> None:
    artifact, text, plan, source_hash, chapter_hash = approved_artifact_fixture()
    changed = dict(artifact)
    changed["approval"] = {**artifact["approval"], "approved_revision": True}
    changed = with_canonical_artifact_hash(changed)
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan(changed, text, plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash)
    assert error.value.code == "INVALID_APPROVED_REVISION"


@pytest.mark.parametrize("change", [
    lambda value: value.update({"extra": True}),
    lambda value: value["analyzer"].update({"extra": True}),
    lambda value: value["cast"][0].update({"extra": True}),
    lambda value: value["chapters"][0]["spans"][0]["confidence"].update({"extra": True}),
])
def test_validate_voice_plan_rejects_unknown_fields_after_rehash(change) -> None:
    artifact, text, plan, source_hash, chapter_hash = approved_artifact_fixture()
    changed = dict(artifact)
    change(changed)
    changed = with_canonical_artifact_hash(changed)
    with pytest.raises(VoicePlanError):
        validate_voice_plan(changed, text, plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash)


def test_validate_voice_plan_rejects_numeric_only_confidence_and_binding_tamper() -> None:
    artifact, text, plan, source_hash, chapter_hash = approved_artifact_fixture()
    numeric = dict(artifact)
    numeric["chapters"] = [dict(item) for item in artifact["chapters"]]
    numeric["chapters"][0]["spans"] = [dict(item) for item in artifact["chapters"][0]["spans"]]
    numeric["chapters"][0]["spans"][0]["confidence"] = {"score": 0.5}
    numeric = with_canonical_artifact_hash(numeric)
    with pytest.raises(VoicePlanError):
        validate_voice_plan(numeric, text, plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash)
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan(artifact, text + "x", plan, expected_source_pdf_sha256=source_hash, expected_chapter_plan_sha256=chapter_hash)
    assert error.value.code == "CLEANED_TEXT_HASH_MISMATCH"


def test_valid_core_returns_immutable_cast_and_exact_spans_with_voice_reuse() -> None:
    text, plan, cast, spans = valid_fixture()
    normalized_cast, normalized_spans = validate_voice_plan_core(cast, spans, text, plan)
    assert normalized_cast == tuple(cast)
    assert normalized_spans == tuple(spans)
    assert normalized_cast[0].voice_id == normalized_cast[1].voice_id == normalized_cast[2].voice_id
    assert isinstance(normalized_cast, tuple)
    assert isinstance(normalized_spans, tuple)


@pytest.mark.parametrize(
    ("cast", "code"),
    [
        ([], "MISSING_NARRATOR"),
        ([CastEntry("narrator", "Narrator", "character", "third_person", "v", 1.0)], "MISIDENTIFIED_NARRATOR"),
        ([CastEntry("speaker", "Speaker", "narrator", "third_person", "v", 1.0)], "MISIDENTIFIED_NARRATOR"),
    ],
)
def test_narrator_identity_is_required(cast: list[CastEntry], code: str) -> None:
    text, plan, _, spans = valid_fixture()
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan_core(cast, spans, text, plan)
    assert error.value.code == code


def test_duplicate_cast_id_unknown_speaker_and_same_as_narrator_mismatch_reject() -> None:
    text, plan, cast, spans = valid_fixture()
    duplicate = [*cast, CastEntry("alice", "Second Alice", "character", "third_person", "other", 1.0)]
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan_core(duplicate, spans, text, plan)
    assert error.value.code == "DUPLICATE_CAST_ID"

    unknown = [span("s1", 1, 0, text.index("Alice"), "ghost"), *spans[1:]]
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan_core(cast, unknown, text, plan)
    assert error.value.code == "UNKNOWN_SPEAKER_REFERENCE"

    mismatch = [entry if entry.cast_id != "alice" else CastEntry(entry.cast_id, entry.display_label, entry.role, entry.relationship, "other", entry.speed) for entry in cast]
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan_core(mismatch, spans, text, plan)
    assert error.value.code == "NARRATOR_VOICE_MISMATCH"


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (lambda: CastEntry("", "Label", "character", "third_person", "v", 1.0), "INVALID_CAST_ID"),
        (lambda: CastEntry("id", "", "character", "third_person", "v", 1.0), "INVALID_DISPLAY_LABEL"),
        (lambda: CastEntry("id", "x" * 513, "character", "third_person", "v", 1.0), "INVALID_DISPLAY_LABEL"),
        (lambda: CastEntry("id", "Label", "bad", "third_person", "v", 1.0), "INVALID_ROLE"),
        (lambda: CastEntry("id", "Label", "character", "bad", "v", 1.0), "INVALID_RELATIONSHIP"),
        (lambda: CastEntry("id\n", "Label", "character", "third_person", "v", 1.0), "INVALID_CAST_ID"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "v", True), "INVALID_SPEED"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "v", "fast"), "INVALID_SPEED"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "v", math.inf), "INVALID_SPEED"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "v", 0), "INVALID_SPEED"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "", 1.0), "INVALID_VOICE_ID"),
        (lambda: CastEntry("id", "Label", "character", "third_person", "voice\n", 1.0), "INVALID_VOICE_ID"),
    ],
)
def test_cast_entry_rejects_invalid_fields(factory, code: str) -> None:
    with pytest.raises(VoicePlanError) as error:
        factory()
    assert error.value.code == code


def test_unknown_fallback_is_forwarded_and_keeps_stable_unresolved_code() -> None:
    text, plan, cast, spans = valid_fixture()
    unknown = [*spans]
    unknown[-1] = span("s3", 2, unknown[-1].source_start, unknown[-1].source_end, "narrator", "unknown")
    with pytest.raises(VoicePlanError) as error:
        validate_voice_plan_core(cast, unknown, text, plan)
    assert error.value.code == "UNRESOLVED_SPANS"
    normalized_cast, normalized_spans = validate_voice_plan_core(cast, unknown, text, plan, narrator_fallback_accepted=True)
    assert normalized_cast == tuple(cast)
    assert normalized_spans == tuple(unknown)


def test_canonical_json_matches_workspace_convention_and_is_deterministic() -> None:
    first = {"z": "é", "a": 1, "nested": {"b": 2, "a": "✓"}}
    second = {"nested": {"a": "✓", "b": 2}, "a": 1, "z": "é"}
    expected = '{"a":1,"nested":{"a":"✓","b":2},"z":"é"}\n'
    assert canonical_json_text(first) == expected
    assert canonical_json_bytes(first) == expected.encode("utf-8")
    assert canonical_json_text(first) == canonical_json_text(second)
    assert canonical_json_bytes(first).endswith(b"\n")


def test_canonical_digest_excludes_only_top_level_hash_without_mutating() -> None:
    artifact = {"z": "é", "nested": {"canonical_artifact_sha256": "keep"}, "canonical_artifact_sha256": "old"}
    before = dict(artifact)
    expected_payload = '{"nested":{"canonical_artifact_sha256":"keep"},"z":"é"}\n'.encode("utf-8")
    assert canonical_artifact_digest(artifact) == hashlib.sha256(expected_payload).hexdigest()
    assert artifact == before
    stamped = with_canonical_artifact_hash(artifact)
    assert artifact == before
    assert stamped["canonical_artifact_sha256"] == canonical_artifact_digest(stamped)
    assert verify_canonical_artifact_hash(stamped) is True


def test_canonical_digest_rejects_malformed_hash_tamper_and_invalid_values() -> None:
    stamped = with_canonical_artifact_hash({"value": "ok"})
    malformed = {**stamped, "canonical_artifact_sha256": "ABC"}
    with pytest.raises(VoicePlanError) as error:
        verify_canonical_artifact_hash(malformed)
    assert error.value.code == "INVALID_ARTIFACT_HASH"
    tampered = {**stamped, "value": "changed"}
    with pytest.raises(VoicePlanError) as error:
        verify_canonical_artifact_hash(tampered)
    assert error.value.code == "ARTIFACT_HASH_MISMATCH"
    for value in ([], {"nan": math.nan}, {"infinity": math.inf}, {"set": {"x"}}):
        with pytest.raises(VoicePlanError) as error:
            canonical_artifact_digest(value)
        assert error.value.code == "INVALID_ARTIFACT"
    with pytest.raises(VoicePlanError) as error:
        canonical_artifact_digest({"surrogate": "\ud800"})
    assert error.value.code == "INVALID_ARTIFACT"
    with pytest.raises(VoicePlanError) as error:
        canonical_json_bytes({"set": {"x"}})
    assert error.value.code == "INVALID_ARTIFACT"


def test_builder_reuses_ordered_voices_and_prefers_canonical_labels() -> None:
    characters = [{"character_id": f"id-{index}", "canonical_label": f"Name {index}", "quote_count": 10} for index in range(5)]
    analysis, text, plan, source, chapter_hash = lifecycle_inputs(characters)
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1", "v2", "v3"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_builder_handles_span_only_character_and_rejects_bad_ranges() -> None:
    spans = [{"span_id": f"m-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": "solo"} for index in range(10)]
    analysis, text, plan, _, _ = lifecycle_inputs(spans=spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    outside = copy.deepcopy(analysis)
    outside["spans"][0]["source_start"], outside["spans"][0]["source_end"] = 11, 12
    with pytest.raises(VoicePlanError) as error:
        build_voice_plan(outside, text, plan, ["v0"])
    assert error.value.code == "INVALID_ANALYSIS_SPAN"
    crossing = copy.deepcopy(analysis)
    crossing["spans"][0]["source_start"], crossing["spans"][0]["source_end"] = 4, 7
    with pytest.raises(VoicePlanError) as error:
        build_voice_plan(crossing, text, plan, ["v0"])
    assert error.value.code == "INVALID_ANALYSIS_SPAN"


def test_builder_promotes_span_only_hyphenated_proper_name() -> None:
    analysis, text, plan, _, _ = observed_lifecycle_inputs([], ["Mary-Jane"] * 10)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert artifact["cast"][1]["display_label"] == "Mary-Jane"
    assert all(span["speaker_id"] == "character-Mary-Jane" for span in artifact["chapters"][0]["spans"])


@pytest.mark.parametrize(("quote_count", "qualifies"), [(10, False), (9, False)])
def test_builder_cast_threshold_is_inclusive(quote_count: int, qualifies: bool) -> None:
    analysis, text, plan, _, _ = lifecycle_inputs([{"character_id": "alice", "canonical_label": "Alice", "quote_count": quote_count}])
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert any(entry["cast_id"] == "character-alice" for entry in artifact["cast"]) is qualifies


def test_builder_resolves_pronoun_alias_to_named_character() -> None:
    characters = [{
        "character_id": "alice-id",
        "canonical_label": "she",
        "quote_count": 10,
        "aliases": [{"alias": "she", "kind": "pronoun"}, {"alias": "Alice", "kind": "proper"}],
    }]
    analysis, text, plan, _, _ = observed_lifecycle_inputs(characters, ["she", "Alice"] * 5)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert artifact["cast"][1]["display_label"] == "Alice"
    assert all(span["speaker_id"] == "character-alice-id" for span in artifact["chapters"][0]["spans"])


@pytest.mark.parametrize("pronoun", ["I", "you", "she"])
def test_builder_keeps_unresolved_pronoun_character_with_narrator(pronoun: str) -> None:
    characters = [{
        "character_id": f"{pronoun}-id",
        "canonical_label": pronoun,
        "quote_count": 10,
        "aliases": [{"alias": pronoun, "kind": "pronoun"}],
    }]
    spans = [{"span_id": "pronoun-1", "source_start": 0, "source_end": 1, "type": "dialogue", "speaker_id": pronoun}]
    analysis, text, plan, _, _ = lifecycle_inputs(characters, spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    assert artifact["chapters"][0]["spans"][0]["speaker_id"] == "narrator"


def test_builder_drops_top_level_alias_for_excluded_pronoun_identity() -> None:
    characters = [{
        "character_id": "she-id",
        "canonical_label": "she",
        "quote_count": 10,
        "aliases": [{"alias": "she", "kind": "pronoun"}],
    }]
    spans = [{"span_id": "she-1", "source_start": 0, "source_end": 1, "type": "dialogue", "speaker_id": "she"}]
    analysis, text, plan, source, chapter_hash = lifecycle_inputs(characters, spans)
    analysis["aliases"] = [{"alias": "her", "character_id": "she-id"}]
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    assert artifact["aliases"] == []
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_builder_keeps_frequent_span_only_pronoun_with_narrator() -> None:
    spans = [{"span_id": f"she-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": "she"} for index in range(10)]
    analysis, text, plan, _, _ = lifecycle_inputs(spans=spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    assert all(span["speaker_id"] == "narrator" for chapter in artifact["chapters"] for span in chapter["spans"] if span["span_id"].startswith("she-"))


def test_builder_maps_below_threshold_dialogue_to_narrator_without_alias_or_cast() -> None:
    spans = [{"span_id": f"a-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": "alice"} for index in range(9)]
    analysis, text, plan, _, _ = lifecycle_inputs([{"character_id": "alice", "canonical_label": "Alice", "quote_count": 9, "aliases": ["Al"]}], spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]
    assert artifact["aliases"] == []
    assert all(span["speaker_id"] == "narrator" for chapter in artifact["chapters"] for span in chapter["spans"] if span["span_id"].startswith("a-"))


def test_builder_prefers_first_proper_alias_for_display_label() -> None:
    analysis, text, plan, _, _ = observed_lifecycle_inputs([{"character_id": "alice", "canonical_label": "the woman", "quote_count": 10, "aliases": [{"alias": "the woman", "kind": "nominal"}, {"alias": "Alice", "kind": "proper"}, {"alias": "Al", "kind": "proper"}]}], ["alice"] * 10)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert artifact["cast"][1]["display_label"] == "Alice"


def test_builder_sums_attributed_spans_across_character_id_and_proper_alias() -> None:
    characters = [{"character_id": "alice-id", "canonical_label": "the woman", "aliases": [{"alias": "Alice", "kind": "proper"}]}]
    spans = [
        {"span_id": f"a-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": "alice-id" if index < 5 else "Alice"}
        for index in range(10)
    ]
    analysis, text, plan, _, _ = lifecycle_inputs(characters, spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert artifact["cast"][1]["display_label"] == "Alice"
    assert all(span["speaker_id"] == "character-alice-id" for chapter in artifact["chapters"] for span in chapter["spans"] if span["span_id"].startswith("a-"))


def test_builder_excludes_nine_combined_identity_spans() -> None:
    characters = [{"character_id": "alice-id", "canonical_label": "the woman", "aliases": [{"alias": "Alice", "kind": "proper"}]}]
    spans = [
        {"span_id": f"a-{index}", "source_start": index, "source_end": index + 1, "type": "dialogue", "speaker_id": "alice-id" if index < 5 else "Alice"}
        for index in range(9)
    ]
    analysis, text, plan, _, _ = lifecycle_inputs(characters, spans)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]


def test_builder_keeps_canonical_label_for_frequent_unnamed_role() -> None:
    analysis, text, plan, _, _ = lifecycle_inputs([{"character_id": "doctor", "canonical_label": "the doctor", "quote_count": 10, "aliases": [{"alias": "doctor", "kind": "nominal"}]}])
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator"]


def test_unknown_speaker_is_reviewable_and_requires_approval_fallback() -> None:
    analysis, text, plan, source, chapter_hash = lifecycle_inputs(spans=[{"span_id": "u", "source_start": 0, "source_end": 2, "type": "dialogue", "speaker_id": None}])
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    assert artifact["chapters"][0]["spans"][0]["type"] == "unknown"
    with pytest.raises(VoicePlanError) as error:
        approve_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash)
    assert error.value.code == "UNRESOLVED_SPANS"
    approved = approve_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash, accept_narrator_fallback=True, approved_at="2026-01-01T00:00:00Z")
    assert approved["approval"]["state"] == "approved"
    retry = approve_voice_plan(approved, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash)
    assert retry == approved and retry is not approved
    assert (retry["revision"], retry["canonical_artifact_sha256"], retry["approval"], retry["unresolved_policy"]) == (approved["revision"], approved["canonical_artifact_sha256"], approved["approval"], approved["unresolved_policy"])


def test_cast_remove_and_merge_reassign_aliases_spans_without_mutating_source() -> None:
    characters = [
        {"character_id": "bob", "canonical_label": "Bob", "aliases": [{"alias": "Bobby", "kind": "proper"}]},
        {"character_id": "alice", "canonical_label": "Alice", "aliases": [{"alias": "Al", "kind": "proper"}]},
    ]
    analysis, text, plan, _, _ = observed_lifecycle_inputs(characters, ["bob"] * 10 + ["alice"] * 10)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    original = copy.deepcopy(artifact)
    merged = merge_cast(artifact, "character-bob", "character-alice", expected_revision=1)
    assert artifact == original and merged["revision"] == 2
    assert all(alias["character_id"] == "character-alice" for alias in merged["aliases"])
    merged_spans = merged["chapters"][0]["spans"][:10]
    assert all(span["speaker_id"] == "character-alice" and span["override"]["reason"] == "cast_merged" for span in merged_spans)
    removed = remove_cast(merged, "character-alice", expected_revision=2)
    assert removed["revision"] == 3 and [entry["cast_id"] for entry in removed["cast"]] == ["narrator"]
    assert not removed["aliases"] and all(span["speaker_id"] == "narrator" for span in removed["chapters"][0]["spans"])
    assert all(span["override"]["reason"] == "cast_removed" for span in removed["chapters"][0]["spans"])
    verify_canonical_artifact_hash(removed)


@pytest.mark.parametrize(
    ("operation", "args", "code"),
    [
        (remove_cast, ("narrator",), "CANNOT_REMOVE_NARRATOR"),
        (merge_cast, ("character-alice", "character-alice"), "CANNOT_MERGE_SELF"),
        (merge_cast, ("narrator", "character-alice"), "CANNOT_MERGE_NARRATOR"),
        (remove_cast, ("missing",), "UNKNOWN_CAST_ID"),
    ],
)
def test_cast_mutations_reject_protected_unknown_and_self_ids(operation, args, code) -> None:
    analysis, text, plan, _, _ = observed_lifecycle_inputs([{"character_id": "alice", "canonical_label": "Alice", "aliases": [{"alias": "Al", "kind": "proper"}]}], ["alice"] * 10)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    with pytest.raises(VoicePlanError) as error:
        operation(artifact, *args, expected_revision=1)
    assert error.value.code == code


def test_edits_are_non_mutating_revision_bound_and_support_alias_and_span_review() -> None:
    characters = [{"character_id": "alice", "canonical_label": "Alice", "aliases": [{"alias": "Al", "kind": "proper"}, {"alias": "A", "kind": "proper"}], "quote_count": 10}]
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(characters, ["alice"] * 10)
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1"])
    original = copy.deepcopy(artifact)
    renamed = rename_cast(artifact, "character-alice", "Alice Prime", expected_revision=1)
    assert artifact == original and renamed["revision"] == 2 and renamed["approval"]["state"] == "draft"
    with pytest.raises(VoicePlanError) as error:
        assign_cast(renamed, "character-alice", expected_revision=1, speed=1.2)
    assert error.value.code == "STALE_REVISION"
    assigned = assign_cast(renamed, "character-alice", expected_revision=2, speed=1.2)
    merged = merge_aliases(assigned, "character-alice", ["alias-character-alice-1"], expected_revision=3)
    split = split_aliases(merged, ["alias-character-alice-2"], expected_revision=4, new_character_id="character-other", display_label="Other")
    span_id = split["chapters"][0]["spans"][0]["span_id"]
    overridden = override_span(split, span_id, expected_revision=5, kind="type", to="thought", reason="review")
    assert overridden["revision"] == 6
    assert validate_voice_plan(overridden, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == overridden
    merge_aliases,
    override_span,
    rename_cast,
    split_aliases,


def test_speaker_override_resolves_unknown_type_and_approves() -> None:
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(
        [{"character_id": "alice", "canonical_label": "Alice", "quote_count": 10}],
        ["unknown"] + ["alice"] * 10,
    )
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1"])
    overridden = override_span(artifact, "s-0", expected_revision=1, kind="speaker", to="character-alice", reason="review")
    changed = overridden["chapters"][0]["spans"][0]
    assert changed["type"] == "dialogue"
    assert changed["speaker_id"] == "character-alice"
    assert changed["override"] == {"kind": "speaker", "from": "narrator", "to": "character-alice", "actor": "user", "reason": "review"}
    approved = approve_voice_plan(overridden, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash, approved_at="2026-01-01T00:00:00Z")
    assert approved["approval"]["state"] == "approved"


def test_narration_type_override_clears_character_speaker() -> None:
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(
        [{"character_id": "alice", "canonical_label": "Alice", "quote_count": 10}],
        ["alice"] * 10,
    )
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1"])
    overridden = override_span(artifact, "s-0", expected_revision=1, kind="type", to="narration", reason="review")
    changed = overridden["chapters"][0]["spans"][0]
    assert changed["type"] == "narration"
    assert changed["speaker_id"] == "narrator"
    assert changed["override"] == {"kind": "type", "from": "dialogue", "to": "narration", "actor": "user", "reason": "review"}
    approved = approve_voice_plan(overridden, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash, approved_at="2026-01-01T00:00:00Z")
    assert approved["approval"]["state"] == "approved"


def test_unregistered_speaker_is_marked_unknown_and_counted_for_review() -> None:
    analysis, text, plan, source, chapter_hash = lifecycle_inputs(
        spans=[{"span_id": "unregistered", "source_start": 0, "source_end": 2, "type": "dialogue", "speaker_id": "unlisted", "confidence": {"score": 0.7, "band": "medium", "reasons": ["machine"]}}]
    )
    artifact = build_voice_plan(analysis, text, plan, ["v0"])
    marked = artifact["chapters"][0]["spans"][0]
    assert marked["speaker_id"] == "narrator"
    assert marked["type"] == "unknown"
    assert marked["confidence"] == {"score": 0.7, "band": "medium", "reasons": ["machine", "speaker_not_cast"]}
    assert review_summary(artifact)["speaker_not_cast_count"] == 1
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_colliding_cast_slugs_merge_identity_mapping_into_first_entry() -> None:
    characters = [
        {"character_id": "alpha/one", "canonical_label": "First", "quote_count": 10, "aliases": [{"alias": "First", "kind": "proper"}]},
        {"character_id": "alpha-one", "canonical_label": "Second", "quote_count": 10, "aliases": [{"alias": "Second", "kind": "proper"}]},
    ]
    speaker_ids = ["alpha/one"] * 10 + ["alpha-one"] * 10
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(characters, speaker_ids)
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1", "v2"])
    assert [entry["cast_id"] for entry in artifact["cast"]] == ["narrator", "character-alpha-one"]
    assert artifact["cast"][1]["display_label"] == "First"
    assert artifact["cast"][1]["voice_id"] == "v1"
    spans = [span for chapter in artifact["chapters"] for span in chapter["spans"] if span["span_id"].startswith("s-")]
    assert all(span["speaker_id"] == "character-alpha-one" for span in spans)
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_type_override_to_unknown_resets_speaker_and_stays_approvable() -> None:
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(
        [{"character_id": "alice", "canonical_label": "Alice", "quote_count": 10}],
        ["alice"] * 10,
    )
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1"])
    overridden = override_span(artifact, "s-0", expected_revision=1, kind="type", to="unknown", reason="review")
    changed = overridden["chapters"][0]["spans"][0]
    assert changed["type"] == "unknown"
    assert changed["speaker_id"] == "narrator"
    approved = approve_voice_plan(
        overridden,
        text,
        plan,
        expected_source_pdf_sha256=source,
        expected_chapter_plan_sha256=chapter_hash,
        accept_narrator_fallback=True,
        approved_at="2026-01-01T00:00:00Z",
    )
    assert approved["approval"]["state"] == "approved"


def test_colliding_cast_slugs_preserve_duplicate_aliases() -> None:
    characters = [
        {"character_id": "alpha/one", "canonical_label": "First", "quote_count": 10, "aliases": [{"alias": "First", "kind": "proper"}]},
        {"character_id": "alpha-one", "canonical_label": "Second", "quote_count": 10, "aliases": [{"alias": "Second", "kind": "proper"}]},
    ]
    speaker_ids = ["alpha/one"] * 10 + ["alpha-one"] * 10
    analysis, text, plan, source, chapter_hash = observed_lifecycle_inputs(characters, speaker_ids)
    artifact = build_voice_plan(analysis, text, plan, ["v0", "v1", "v2"])
    alias_texts = {entry["text"] for entry in artifact["aliases"]}
    assert {"First", "Second"} <= alias_texts
    assert all(entry["character_id"] == "character-alpha-one" for entry in artifact["aliases"])
    assert validate_voice_plan(artifact, text, plan, expected_source_pdf_sha256=source, expected_chapter_plan_sha256=chapter_hash) == artifact


def test_builder_reserves_narrator_voice_slot_when_character_assignment_wraps() -> None:
    characters = [
        {"character_id": "alice", "canonical_label": "Alice"},
        {"character_id": "bob", "canonical_label": "Bob"},
        {"character_id": "carol", "canonical_label": "Carol"},
    ]
    analysis, text, plan, _, _ = observed_lifecycle_inputs(
        characters,
        ["alice"] * 10 + ["bob"] * 10 + ["carol"] * 10,
    )

    artifact = build_voice_plan(analysis, text, plan, ["narrator-voice", "alice-voice", "bob-voice"])

    assert [entry["voice_id"] for entry in artifact["cast"]] == [
        "narrator-voice",
        "alice-voice",
        "bob-voice",
        "alice-voice",
    ]


def test_review_summary_counts_non_narrator_voice_collisions_only() -> None:
    characters = [
        {"character_id": "alice", "canonical_label": "Alice", "relationship": "same_as_narrator"},
        {"character_id": "bob", "canonical_label": "Bob", "relationship": "separate_from_narrator"},
        {"character_id": "carol", "canonical_label": "Carol", "relationship": "separate_from_narrator"},
    ]
    analysis, text, plan, _, _ = observed_lifecycle_inputs(
        characters,
        ["alice"] * 10 + ["bob"] * 10 + ["carol"] * 10,
    )
    artifact = build_voice_plan(analysis, text, plan, ["narrator-voice"])

    summary = review_summary(artifact)

    assert summary["voice_collision_count"] == 2
    assert summary["has_voice_collisions"] is True


def test_builder_assigns_sorted_spans_with_one_bisect_lookup_per_analysis_span(monkeypatch) -> None:
    analysis, text, plan, _, _ = lifecycle_inputs(
        spans=[
            {"span_id": "late", "source_start": 6, "source_end": 7, "type": "dialogue", "speaker_id": "narrator"},
            {"span_id": "early", "source_start": 0, "source_end": 1, "type": "dialogue", "speaker_id": "narrator"},
            {"span_id": "boundary", "source_start": 5, "source_end": 6, "type": "dialogue", "speaker_id": "narrator"},
        ]
    )
    calls: list[int] = []

    def counted_bisect(values, target):
        calls.append(target)
        return real_bisect_right(values, target)

    monkeypatch.setattr(voice_plan_module, "bisect_right", counted_bisect, raising=False)
    artifact = build_voice_plan(analysis, text, plan, ["v0"])

    assert calls == [0, 5, 6]
    assert [span["span_id"] for span in artifact["chapters"][0]["spans"] if span["span_id"] == "early"] == ["early"]
    assert [span["span_id"] for span in artifact["chapters"][1]["spans"] if span["span_id"] in {"boundary", "late"}] == ["boundary", "late"]


def test_builder_prefers_registry_gender_and_retained_pronoun_evidence_with_rotation_fallback(monkeypatch) -> None:
    analysis, text, plan, _, _ = observed_lifecycle_inputs(
        [
            {"character_id": "alice", "canonical_label": "Alice", "gender": "female", "quote_count": 10},
            {"character_id": "bob", "canonical_label": "Bob", "aliases": [{"alias": "he", "kind": "pronoun"}, {"alias": "Bob", "kind": "proper"}], "quote_count": 10},
            {"character_id": "casey", "canonical_label": "Casey", "quote_count": 10},
        ],
        ["alice"] * 10 + ["bob"] * 10 + ["casey"] * 10,
    )
    monkeypatch.setattr(
        voice_plan_module,
        "voice_registry",
        SimpleNamespace(
            list_public_entries=lambda: (
                {"id": "am_adam", "gender": "male"},
                {"id": "af_heart", "gender": "female"},
                {"id": "am_echo", "gender": "male"},
            )
        ),
        raising=False,
    )

    artifact = build_voice_plan(analysis, text, plan, ["narrator", "am_adam", "af_heart", "am_echo"])

    assert [entry["voice_id"] for entry in artifact["cast"]] == ["narrator", "af_heart", "am_adam", "am_echo"]

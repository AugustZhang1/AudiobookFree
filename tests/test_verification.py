"""Pure decision-record tests; no application, model, or process boundary."""

from __future__ import annotations

import copy
import json

import pytest

from pdf_audiobook import verification, verifier


IDENTITY = {"id": "fixture-verifier", "version": "1.0", "model_hash": None}


def request(*, primary: str | None = "alice", confidence_state: str = "scored", confidence: float | None = 0.4) -> dict:
    context = "Before Hello, said Alice. After."
    context_start = 100
    quote = "Hello"
    quote_start = context_start + context.index(quote)
    return {
        "schema_version": 1,
        "span_id": "booknlp:412",
        "quote": {"text": quote, "start_offset": quote_start, "end_offset": quote_start + len(quote)},
        "context": {"text": context, "start_offset": context_start, "end_offset": context_start + len(context)},
        "nearby_assignments": [],
        "allowed_speaker_ids": ["alice", "jane", "unknown"],
        "primary_proposal": {
            "speaker_id": primary,
            "confidence": confidence,
            "confidence_state": confidence_state,
            "classification": "weak-nearby",
            "evidence_offsets": [],
            "untrusted": True,
        },
    }


def response(**changes: object) -> dict:
    value: dict[str, object] = {
        "span_id": "booknlp:412",
        "speaker_id": "alice",
        "alternative_speaker_id": None,
        "decision": "correct_primary",
        "confidence": 0.91,
        "evidence_offsets": [[108, 113]],
        "reason_code": "EXPLICIT_SPEECH_TAG",
    }
    value.update(changes)
    return value


def make(req: dict | None = None, res: dict | None = None, identity: dict | None = None) -> dict:
    return verification.build_verification_record(req or request(), res or response(), identity or IDENTITY)


def test_exact_mapping_and_invariants_for_all_decisions() -> None:
    cases = [
        (response(), "verified_consensus", "alice", None, "alice", False, True),
        (response(decision="override_primary", speaker_id="jane"), "auto_corrected", "jane", None, "jane", False, True),
        (response(decision="ambiguous", alternative_speaker_id="unknown"), "ambiguous", "alice", "unknown", None, True, False),
        (response(decision="unresolved", speaker_id="unknown", evidence_offsets=[]), "unresolved", "unknown", None, None, True, False),
    ]
    for result, state, proposed, alternative, effective, review, approved in cases:
        record = make(res=result)
        assert record["decision_state"] == state
        assert record["proposed_speaker_id"] == proposed
        assert record["alternative_speaker_id"] == alternative
        assert record["effective_speaker_id"] == effective
        assert record["review_required"] is review
        assert record["auto_approved"] is approved
        assert record["verifier_decision"] == result["decision"]


def test_original_proposal_is_preserved_through_auto_correction() -> None:
    record = make(res=response(decision="override_primary", speaker_id="jane"))
    assert record["original_speaker_id"] == "alice"
    assert record["proposed_speaker_id"] == "jane"
    assert record["effective_speaker_id"] == "jane"


def test_scored_and_unscored_primary_state_is_preserved_without_fabrication() -> None:
    scored = make()
    unscored = make(request(confidence_state="unscored", confidence=None))
    assert (scored["primary_confidence_state"], scored["primary_confidence"]) == ("scored", 0.4)
    assert (unscored["primary_confidence_state"], unscored["primary_confidence"]) == ("unscored", None)


@pytest.mark.parametrize(
    "identity",
    [
        {"id": "bad\nidentity", "version": "1", "model_hash": None},
        {"id": "fixture", "version": "", "model_hash": None},
        {"id": "fixture", "version": "1", "model_hash": "A" * 64},
        {"id": "fixture", "version": "1", "model_hash": "0" * 63},
        {"id": "fixture", "version": "1", "model_hash": None, "extra": 1},
    ],
)
def test_verifier_identity_is_exact_and_sanitized(identity: dict) -> None:
    with pytest.raises(verification.VerificationRecordError) as error:
        make(identity=identity)
    assert error.value.code == "INVALID_VERIFIER_IDENTITY"
    assert error.value.details["field"].startswith("verifier")


def test_mismatched_response_is_safely_delegated_and_sanitized() -> None:
    bad = response(span_id="different")
    with pytest.raises(verification.VerificationRecordError) as error:
        make(res=bad)
    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.details == {"field": "response"}
    assert "Hello" not in str(error.value)


def test_record_is_detached_and_contains_no_source_text() -> None:
    req = request()
    res = response()
    identity = copy.deepcopy(IDENTITY)
    record = make(req, res, identity)
    req["primary_proposal"]["speaker_id"] = "jane"
    res["evidence_offsets"][0][0] = 999
    identity["id"] = "mutated"
    assert record["original_speaker_id"] == "alice"
    assert record["evidence_offsets"] == [[108, 113]]
    assert record["verifier"]["id"] == "fixture-verifier"
    assert "Hello" not in json.dumps(record)
    assert "Before" not in json.dumps(record)


def test_canonical_json_serialization_is_deterministic_and_finite() -> None:
    record = make()
    first = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    second = json.dumps(copy.deepcopy(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert first == second


def test_summary_exact_counts_empty_and_input_order_independent() -> None:
    records = [
        make(),
        make(res=response(decision="override_primary", speaker_id="jane")),
        make(res=response(decision="ambiguous", alternative_speaker_id="unknown")),
        make(res=response(decision="unresolved", speaker_id="unknown", evidence_offsets=[])),
    ]
    for index, record in enumerate(records):
        record["span_id"] = f"span:{index}"
    expected = {"total": 4, "verified_consensus": 1, "auto_corrected": 1, "ambiguous": 1, "unresolved": 1, "auto_approved": 2, "review_required": 2}
    assert verification.summarize_verification_records(records) == expected
    assert verification.summarize_verification_records(list(reversed(records))) == expected
    assert verification.summarize_verification_records([]) == {key: 0 for key in expected}


def test_summary_rejects_duplicate_and_malformed_records() -> None:
    record = make()
    with pytest.raises(verification.VerificationRecordError) as duplicate:
        verification.summarize_verification_records([record, copy.deepcopy(record)])
    assert duplicate.value.code == "DUPLICATE_SPAN_ID"
    malformed = copy.deepcopy(record)
    malformed["extra"] = "not allowed"
    with pytest.raises(verification.VerificationRecordError):
        verification.summarize_verification_records([malformed])


@pytest.mark.parametrize("mutate", [
    lambda record: record.update({"original_speaker_id": "jane", "proposed_speaker_id": "jane", "effective_speaker_id": "jane"}),
    lambda record: record.update({"evidence_offsets": [[-1, 2]]}),
    lambda record: record.update({"evidence_offsets": [[index * 2, index * 2 + 1] for index in range(verifier.MAX_EVIDENCE_RANGES + 1)]}),
    lambda record: record.update({"alternative_speaker_id": record["proposed_speaker_id"]}),
])
def test_summary_rejects_records_that_cannot_come_from_builder(mutate) -> None:
    record = make(res=response(decision="override_primary", speaker_id="jane"))
    mutate(record)
    with pytest.raises(verification.VerificationRecordError):
        verification.summarize_verification_records([record])


def test_record_layer_has_no_application_model_or_process_imports() -> None:
    source = open(verification.__file__, encoding="utf-8").read()
    assert "import app" not in source
    assert "import worker" not in source
    assert "subprocess" not in source
    assert "multiprocessing" not in source
    assert verifier.validate_request(request())["span_id"] == "booknlp:412"

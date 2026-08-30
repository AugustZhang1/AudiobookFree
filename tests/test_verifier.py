"""Focused contract tests for the dependency-free verifier boundary."""

from __future__ import annotations

import json

import pytest

from pdf_audiobook import verifier


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
        "nearby_assignments": [
            {"assignment_id": "booknlp:411", "speaker_id": "alice", "relation": "previous", "position": 0},
            {"assignment_id": "booknlp:413", "speaker_id": None, "relation": "following", "position": 1},
        ],
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


def run(result: dict | str | bytes, req: dict | None = None) -> dict:
    return verifier.verify(req or request(), lambda _: json.dumps(result))


def test_valid_request_and_each_decision_shape() -> None:
    assert run(response())["decision"] == "correct_primary"
    assert run(response(decision="override_primary", speaker_id="jane"))["decision"] == "override_primary"
    assert run(response(decision="ambiguous", alternative_speaker_id="jane"))["decision"] == "ambiguous"
    assert run(response(decision="unresolved", speaker_id="unknown", evidence_offsets=[]))["decision"] == "unresolved"


def test_unscored_primary_proposal_is_explicit() -> None:
    seen: list[dict] = []

    def fake(payload: dict) -> str:
        seen.append(payload)
        return json.dumps(response())

    verifier.verify(request(confidence_state="unscored", confidence=None), fake)
    assert seen[0]["primary_proposal"]["confidence_state"] == "unscored"
    with pytest.raises(verifier.VerifierError):
        verifier.validate_request(request(confidence_state="unscored", confidence=0.4))
    with pytest.raises(verifier.VerifierError):
        verifier.validate_request(request(confidence_state="scored", confidence=None))


def test_backend_receives_canonical_plain_request_and_untrusted_proposal() -> None:
    received: list[dict] = []

    def fake(payload: dict) -> str:
        received.append(payload)
        assert type(payload) is dict
        assert payload["primary_proposal"]["untrusted"] is True
        assert payload["quote"]["text"] == "Hello"
        return json.dumps(response())

    verifier.verify(request(), fake)
    assert received and all(type(item) is dict for item in received[0].values()) is False
    assert isinstance(received[0]["allowed_speaker_ids"], list)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update({"extra": 1}),
        lambda r: r["allowed_speaker_ids"].append("alice"),
        lambda r: r["quote"].update({"text": "Wrong"}),
        lambda r: r["nearby_assignments"][0].update({"speaker_id": "not-allowed"}),
        lambda r: r.update({"span_id": "bad\x00id"}),
    ],
)
def test_malformed_request_is_rejected(mutate) -> None:
    bad = request()
    mutate(bad)
    with pytest.raises(verifier.VerifierError):
        verifier.validate_request(bad)


def test_request_controls_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = request()
    bad["span_id"] = "bad\x00id"
    with pytest.raises(verifier.VerifierError):
        verifier.validate_request(bad)
    monkeypatch.setattr(verifier, "MAX_REQUEST_BYTES", 10)
    with pytest.raises(verifier.VerifierError) as error:
        verifier.validate_request(request())
    assert error.value.code == "REQUEST_TOO_LARGE"


@pytest.mark.parametrize(
    "raw",
    ["not json", b"\xff", "[]", '{"span_id":"booknlp:412","speaker_id":"alice","alternative_speaker_id":null,"decision":"correct_primary","confidence":0.9,"evidence_offsets":[],"reason_code":"OK","extra":1}', '{"span_id":"booknlp:412","speaker_id":"alice","alternative_speaker_id":null,"decision":"correct_primary","confidence":NaN,"evidence_offsets":[],"reason_code":"OK"}'],
)
def test_malformed_response_is_rejected(raw: str | bytes) -> None:
    with pytest.raises(verifier.VerifierError):
        verifier.verify(request(), lambda _: raw)


def test_duplicate_response_keys_and_python_object_output_are_rejected() -> None:
    duplicate = '{"span_id":"booknlp:412","span_id":"booknlp:412","speaker_id":"alice","alternative_speaker_id":null,"decision":"correct_primary","confidence":0.9,"evidence_offsets":[],"reason_code":"OK"}'
    with pytest.raises(verifier.VerifierError):
        verifier.verify(request(), lambda _: duplicate)
    with pytest.raises(verifier.VerifierError):
        verifier.verify(request(), lambda _: response())


@pytest.mark.parametrize(
    "changes",
    [
        {"span_id": "other"},
        {"speaker_id": "not-allowed"},
        {"alternative_speaker_id": "alice", "decision": "ambiguous"},
        {"decision": "unresolved", "speaker_id": "alice"},
        {"decision": "correct_primary", "speaker_id": "jane"},
        {"decision": "override_primary", "speaker_id": "alice"},
        {"decision": "ambiguous"},
        {"evidence_offsets": [[100, 102], [101, 104]]},
        {"evidence_offsets": [[99, 102]]},
        {"reason_code": "not uppercase"},
    ],
)
def test_cross_field_response_validation(changes: dict) -> None:
    with pytest.raises(verifier.VerifierError):
        run(response(**changes))


@pytest.mark.parametrize(
    "req,changes",
    [
        (request(primary="unknown"), {"decision": "correct_primary", "speaker_id": "unknown"}),
        (request(), {"decision": "override_primary", "speaker_id": "unknown"}),
        (request(), {"decision": "ambiguous", "speaker_id": "unknown", "alternative_speaker_id": "alice"}),
    ],
)
def test_unknown_is_only_valid_for_unresolved_selection(req: dict, changes: dict) -> None:
    with pytest.raises(verifier.VerifierError):
        run(response(**changes), req)


def test_evidence_count_and_output_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier, "MAX_EVIDENCE_RANGES", 1)
    with pytest.raises(verifier.VerifierError):
        run(response(evidence_offsets=[[100, 101], [102, 103]]))
    monkeypatch.setattr(verifier, "MAX_RESULT_BYTES", 5)
    with pytest.raises(verifier.VerifierError) as error:
        run(response())
    assert error.value.code == "RESPONSE_TOO_LARGE"


def test_cancellation_before_and_after_backend() -> None:
    class Control:
        def __init__(self, after: bool = False):
            self.calls = 0
            self.after = after

        def check_cancelled(self) -> None:
            self.calls += 1
            if self.after and self.calls == 2:
                raise RuntimeError("cancelled")

    before = Control()
    invoked: list[bool] = []

    class CancelBefore(Control):
        def check_cancelled(self) -> None:
            super().check_cancelled()
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError):
        verifier.verify(request(), lambda _: invoked.append(True) or json.dumps(response()), CancelBefore())
    assert not invoked
    after = Control(after=True)
    with pytest.raises(RuntimeError):
        verifier.verify(request(), lambda _: json.dumps(response()), after)


def test_backend_failure_is_sanitized() -> None:
    book_text = "SECRET BOOK TEXT"

    def failed(_: dict) -> str:
        raise RuntimeError(book_text)

    with pytest.raises(verifier.VerifierError) as error:
        verifier.verify(request(), failed)
    assert error.value.code == "BACKEND_FAILED"
    assert book_text not in str(error.value)
    assert book_text not in error.value.message
    assert book_text not in str(error.value.details)
    assert error.value.__cause__ is None


def test_validate_response_returns_detached_evidence() -> None:
    validated_request = verifier.validate_request(request())
    caller_response = response()
    result = verifier.validate_response(caller_response, validated_request)
    caller_response["evidence_offsets"][0][0] = 109
    caller_response["evidence_offsets"].append([114, 115])
    assert result["evidence_offsets"] == [[108, 113]]


def test_fake_injection_has_no_application_or_model_dependency() -> None:
    adapter = verifier.VerifierAdapter(lambda _: json.dumps(response()))
    assert adapter.run(request())["speaker_id"] == "alice"
    assert verifier.validate_request(request())["primary_proposal"]["untrusted"] is True

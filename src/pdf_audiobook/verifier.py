"""Strict, model-neutral boundary for interactive speaker verification.

Schema version 1 deliberately contains only bounded JSON data.  A request
binds one quote to a context range, a constrained candidate list, nearby
assignments, and an explicitly untrusted primary proposal.  The injected
backend receives a plain JSON object and must return an object with exactly
the response fields documented by :func:`verify`.

No model, process, filesystem, or application module is imported here.  This
keeps the boundary useful with deterministic fakes as well as future models.
Source offsets are Python-character, half-open offsets.  Evidence ranges are
ordered, non-overlapping, and contained in the context range.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import re
from typing import Any


SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 512 * 1024
MAX_RESULT_BYTES = 128 * 1024
MAX_TEXT_LENGTH = 64 * 1024
MAX_CONTEXT_LENGTH = 256 * 1024
MAX_TEXT_BYTES = 256 * 1024
MAX_CONTEXT_BYTES = 1024 * 1024
MAX_CANDIDATES = 128
MAX_NEARBY_ASSIGNMENTS = 64
MAX_EVIDENCE_RANGES = 16
MAX_ID_LENGTH = 256
MAX_REASON_LENGTH = 256

DECISIONS = frozenset({"correct_primary", "override_primary", "ambiguous", "unresolved"})
_RELATIONS = frozenset({"previous", "following"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,%d}$" % (MAX_ID_LENGTH - 1))
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,%d}$" % (MAX_REASON_LENGTH - 1))
_REQUEST_FIELDS = {"schema_version", "span_id", "quote", "context", "nearby_assignments", "allowed_speaker_ids", "primary_proposal"}
_RANGE_FIELDS = {"text", "start_offset", "end_offset"}
_ASSIGNMENT_FIELDS = {"assignment_id", "speaker_id", "relation", "position"}
_PROPOSAL_FIELDS = {"speaker_id", "confidence", "confidence_state", "classification", "evidence_offsets", "untrusted"}
_RESPONSE_FIELDS = {"span_id", "speaker_id", "alternative_speaker_id", "decision", "confidence", "evidence_offsets", "reason_code"}


class VerifierError(ValueError):
    """Stable adapter failure whose message never contains source or backend text."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _fail(code: str, message: str, **details: Any) -> VerifierError:
    return VerifierError(code, message, details=details)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_float(value: Any) -> bool:
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _safe_control(value: Any) -> bool:
    return not any((ord(char) < 32 and char not in "\t\n\r") or 127 <= ord(char) <= 159 for char in value)


def _string(value: Any, *, field: str, maximum: int, identifier: bool = False, reason: bool = False, code: str = "INVALID_REQUEST") -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or not _safe_control(value):
        raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
    if identifier and not _ID.fullmatch(value):
        raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
    if reason and not _REASON.fullmatch(value):
        raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
    return value


def _exact(value: Any, fields: set[str], *, code: str, message: str, field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise _fail(code, message, field=field)
    return value


def _range(value: Any, *, field: str, maximum: int, code: str = "INVALID_REQUEST") -> tuple[str, int, int]:
    item = _exact(value, _RANGE_FIELDS, code=code, message="verification request is invalid", field=field)
    text = item["text"]
    if not isinstance(text, str) or not text or len(text) > maximum or not _safe_control(text):
        raise _fail(code, "verification request is invalid", field=field)
    try:
        encoded_length = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise _fail(code, "verification request is invalid", field=field)
    if encoded_length > (MAX_CONTEXT_BYTES if field == "context" else MAX_TEXT_BYTES):
        raise _fail(code, "verification request is invalid", field=field)
    start, end = item["start_offset"], item["end_offset"]
    if not _is_int(start) or not _is_int(end) or start < 0 or end <= start or end - start != len(text):
        raise _fail(code, "verification request is invalid", field=field)
    return text, start, end


def _evidence(value: Any, context_start: int, context_end: int, *, code: str, field: str) -> list[list[int]]:
    if type(value) is not list or len(value) > MAX_EVIDENCE_RANGES:
        raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
    result: list[list[int]] = []
    previous_end = context_start
    for item in value:
        if type(item) is not list or len(item) != 2 or not _is_int(item[0]) or not _is_int(item[1]):
            raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
        start, end = item
        if start < context_start or end > context_end or end <= start or start < previous_end:
            raise _fail(code, "verification response is invalid" if code == "INVALID_RESPONSE" else "verification request is invalid", field=field)
        previous_end = end
        result.append([start, end])
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _fail("INVALID_REQUEST", "verification request is invalid") from exc


def validate_request(request: Any) -> dict[str, Any]:
    """Validate and return a detached canonical request object."""

    root = _exact(request, _REQUEST_FIELDS, code="INVALID_REQUEST", message="verification request is invalid", field="request")
    if not _is_int(root["schema_version"]) or root["schema_version"] != SCHEMA_VERSION:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="schema_version")
    span_id = _string(root["span_id"], field="span_id", maximum=MAX_ID_LENGTH, identifier=True)
    quote_text, quote_start, quote_end = _range(root["quote"], field="quote", maximum=MAX_TEXT_LENGTH)
    context_text, context_start, context_end = _range(root["context"], field="context", maximum=MAX_CONTEXT_LENGTH)
    if quote_start < context_start or quote_end > context_end or context_text[quote_start - context_start:quote_end - context_start] != quote_text:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="quote")

    allowed = root["allowed_speaker_ids"]
    if type(allowed) is not list or not allowed or len(allowed) > MAX_CANDIDATES:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="allowed_speaker_ids")
    normalized_allowed: list[str] = []
    seen_allowed: set[str] = set()
    for index, speaker_id in enumerate(allowed):
        speaker_id = _string(speaker_id, field="allowed_speaker_ids", maximum=MAX_ID_LENGTH, identifier=True)
        if speaker_id in seen_allowed:
            raise _fail("INVALID_REQUEST", "verification request is invalid", field="allowed_speaker_ids")
        seen_allowed.add(speaker_id)
        normalized_allowed.append(speaker_id)
    if "unknown" not in seen_allowed:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="allowed_speaker_ids")

    nearby = root["nearby_assignments"]
    if type(nearby) is not list or len(nearby) > MAX_NEARBY_ASSIGNMENTS:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="nearby_assignments")
    normalized_nearby: list[dict[str, Any]] = []
    seen_assignments: set[str] = set()
    for item in nearby:
        assignment = _exact(item, _ASSIGNMENT_FIELDS, code="INVALID_REQUEST", message="verification request is invalid", field="nearby_assignments")
        assignment_id = _string(assignment["assignment_id"], field="nearby_assignments", maximum=MAX_ID_LENGTH, identifier=True)
        if assignment_id in seen_assignments or not isinstance(assignment["relation"], str) or assignment["relation"] not in _RELATIONS or not _is_int(assignment["position"]) or assignment["position"] < 0 or assignment["position"] >= MAX_CONTEXT_LENGTH:
            raise _fail("INVALID_REQUEST", "verification request is invalid", field="nearby_assignments")
        speaker_id = assignment["speaker_id"]
        if speaker_id is not None and (not isinstance(speaker_id, str) or speaker_id not in seen_allowed):
            raise _fail("INVALID_REQUEST", "verification request is invalid", field="nearby_assignments")
        seen_assignments.add(assignment_id)
        normalized_nearby.append({"assignment_id": assignment_id, "speaker_id": speaker_id, "relation": assignment["relation"], "position": assignment["position"]})

    proposal = _exact(root["primary_proposal"], _PROPOSAL_FIELDS, code="INVALID_REQUEST", message="verification request is invalid", field="primary_proposal")
    proposal_speaker = proposal["speaker_id"]
    if proposal_speaker is not None and (not isinstance(proposal_speaker, str) or proposal_speaker not in seen_allowed):
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    confidence_state = proposal["confidence_state"]
    if not isinstance(confidence_state, str) or confidence_state not in {"scored", "unscored"}:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    if confidence_state == "scored" and (not _is_float(proposal["confidence"]) or not 0 <= proposal["confidence"] <= 1):
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    if confidence_state == "unscored" and proposal["confidence"] is not None:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    classification = _string(proposal["classification"], field="primary_proposal", maximum=MAX_REASON_LENGTH)
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in classification):
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    if proposal["untrusted"] is not True:
        raise _fail("INVALID_REQUEST", "verification request is invalid", field="primary_proposal")
    proposal_evidence = _evidence(proposal["evidence_offsets"], context_start, context_end, code="INVALID_REQUEST", field="primary_proposal")

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "span_id": span_id,
        "quote": {"text": quote_text, "start_offset": quote_start, "end_offset": quote_end},
        "context": {"text": context_text, "start_offset": context_start, "end_offset": context_end},
        "nearby_assignments": normalized_nearby,
        "allowed_speaker_ids": normalized_allowed,
        "primary_proposal": {"speaker_id": proposal_speaker, "confidence": proposal["confidence"], "confidence_state": confidence_state, "classification": classification, "evidence_offsets": proposal_evidence, "untrusted": True},
    }
    if len(_canonical_json(canonical)) > MAX_REQUEST_BYTES:
        raise _fail("REQUEST_TOO_LARGE", "verification request exceeded the size limit")
    return json.loads(_canonical_json(canonical))


class _DuplicateJSON(ValueError):
    pass


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSON()
        result[key] = value
    return result


def _parse_response(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > MAX_RESULT_BYTES:
            raise _fail("RESPONSE_TOO_LARGE", "verification response exceeded the size limit")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail("INVALID_RESPONSE", "verification response is invalid") from exc
    elif isinstance(raw, str):
        try:
            if len(raw.encode("utf-8")) > MAX_RESULT_BYTES:
                raise _fail("RESPONSE_TOO_LARGE", "verification response exceeded the size limit")
        except UnicodeEncodeError as exc:
            raise _fail("INVALID_RESPONSE", "verification response is invalid") from exc
    else:
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    try:
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()), object_pairs_hook=_object_no_duplicates)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail("INVALID_RESPONSE", "verification response is invalid") from exc
    if type(value) is not dict:
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    return value


def validate_response(response: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a response against a previously validated request."""

    root = _exact(response, _RESPONSE_FIELDS, code="INVALID_RESPONSE", message="verification response is invalid", field="response")
    if not isinstance(root["span_id"], str) or root["span_id"] != request["span_id"]:
        raise _fail("INVALID_RESPONSE", "verification response is invalid", field="span_id")
    allowed = set(request["allowed_speaker_ids"])
    speaker_id, alternative = root["speaker_id"], root["alternative_speaker_id"]
    if not isinstance(speaker_id, str) or speaker_id not in allowed:
        raise _fail("INVALID_RESPONSE", "verification response is invalid", field="speaker_id")
    if alternative is not None and (not isinstance(alternative, str) or alternative not in allowed or alternative == speaker_id):
        raise _fail("INVALID_RESPONSE", "verification response is invalid", field="alternative_speaker_id")
    decision = root["decision"]
    if not isinstance(decision, str) or decision not in DECISIONS or not _is_float(root["confidence"]) or not 0 <= root["confidence"] <= 1:
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    proposal_speaker = request["primary_proposal"]["speaker_id"]
    # ``unknown`` is a sentinel for an unresolved selection, never a proposed
    # or selected character for a successful/ambiguous decision.  Keep this
    # check separate from the decision-specific rules so it also covers an
    # untrusted primary proposal that was itself ``unknown``.
    if speaker_id == "unknown" and decision != "unresolved":
        raise _fail("INVALID_RESPONSE", "verification response is invalid", field="speaker_id")
    if decision == "unresolved" and (speaker_id != "unknown" or alternative is not None):
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    if decision == "correct_primary" and (proposal_speaker is None or speaker_id != proposal_speaker or alternative is not None):
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    if decision == "override_primary" and speaker_id == proposal_speaker:
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    if decision == "ambiguous" and alternative is None:
        raise _fail("INVALID_RESPONSE", "verification response is invalid")
    evidence_offsets = _evidence(root["evidence_offsets"], request["context"]["start_offset"], request["context"]["end_offset"], code="INVALID_RESPONSE", field="evidence_offsets")
    _string(root["reason_code"], field="reason_code", maximum=MAX_REASON_LENGTH, reason=True, code="INVALID_RESPONSE")
    return {"span_id": root["span_id"], "speaker_id": speaker_id, "alternative_speaker_id": alternative, "decision": decision, "confidence": root["confidence"], "evidence_offsets": evidence_offsets, "reason_code": root["reason_code"]}


def verify(request: Any, backend: Callable[[dict[str, Any]], str | bytes], control: Any = None) -> dict[str, Any]:
    """Run an injected backend and strictly validate its UTF-8 JSON response."""

    if not callable(backend):
        raise _fail("INVALID_BACKEND", "verification backend is invalid")
    canonical_request = validate_request(request)
    if control is not None:
        check = getattr(control, "check_cancelled", None)
        if not callable(check):
            raise _fail("INVALID_CONTROL", "verification control is invalid")
        check()
    try:
        raw = backend(canonical_request)
    except Exception:
        raise _fail("BACKEND_FAILED", "verification backend failed") from None
    if control is not None:
        check()
    parsed = _parse_response(raw)
    return validate_response(parsed, canonical_request)


run_verification = verify


class VerifierAdapter:
    """Small injectable adapter convenience wrapper."""

    def __init__(self, backend: Callable[[dict[str, Any]], str | bytes], control: Any = None):
        if not callable(backend):
            raise _fail("INVALID_BACKEND", "verification backend is invalid")
        self.backend = backend
        self.control = control

    def verify(self, request: Any) -> dict[str, Any]:
        return verify(request, self.backend, self.control)

    def run(self, request: Any) -> dict[str, Any]:
        return self.verify(request)


__all__ = [
    "DECISIONS", "MAX_CANDIDATES", "MAX_CONTEXT_BYTES", "MAX_CONTEXT_LENGTH", "MAX_EVIDENCE_RANGES",
    "MAX_ID_LENGTH", "MAX_NEARBY_ASSIGNMENTS", "MAX_REASON_LENGTH", "MAX_REQUEST_BYTES", "MAX_RESULT_BYTES",
    "MAX_TEXT_BYTES", "MAX_TEXT_LENGTH", "SCHEMA_VERSION", "VerifierAdapter", "VerifierError",
    "run_verification", "validate_request", "validate_response", "verify",
]

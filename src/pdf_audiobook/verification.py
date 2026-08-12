"""Pure, detached decision records for the strict verifier boundary.

This module deliberately does not know about books, plans, applications, or
models.  It converts one validated verifier exchange into a small canonical
record that can later be persisted or rendered by another layer.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import math
import re
from typing import Any

from . import verifier


RECORD_SCHEMA_VERSION = 1
DECISION_STATES = frozenset({"verified_consensus", "auto_corrected", "ambiguous", "unresolved"})
SUMMARY_KEYS = (
    "total",
    "verified_consensus",
    "auto_corrected",
    "ambiguous",
    "unresolved",
    "auto_approved",
    "review_required",
)
_RECORD_FIELDS = {
    "schema_version",
    "span_id",
    "original_speaker_id",
    "proposed_speaker_id",
    "effective_speaker_id",
    "alternative_speaker_id",
    "decision_state",
    "review_required",
    "auto_approved",
    "primary_confidence_state",
    "primary_confidence",
    "confidence",
    "evidence_offsets",
    "reason_code",
    "verifier",
    "verifier_decision",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,255}$")
_IDENTITY_STRING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/+@-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,255}$")


class VerificationRecordError(ValueError):
    """Stable, source-text-free failure from record construction or summary."""

    def __init__(self, code: str, field: str):
        super().__init__("verification record is invalid")
        self.code = code
        self.message = "verification record is invalid"
        self.details = {"field": field}


def _fail(code: str, field: str) -> VerificationRecordError:
    return VerificationRecordError(code, field)


def _finite_number(value: Any) -> bool:
    return (type(value) is int) or (type(value) is float and math.isfinite(value))


def _safe_identifier(value: Any, *, allow_none: bool, field: str) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise _fail("INVALID_RECORD", field)


def _validate_identity(identity: Any) -> dict[str, Any]:
    if type(identity) is not dict or set(identity) != {"id", "version", "model_hash"}:
        raise _fail("INVALID_VERIFIER_IDENTITY", "verifier")
    for field in ("id", "version"):
        value = identity[field]
        if not isinstance(value, str) or not _IDENTITY_STRING.fullmatch(value):
            raise _fail("INVALID_VERIFIER_IDENTITY", f"verifier.{field}")
    model_hash = identity["model_hash"]
    if model_hash is not None and (not isinstance(model_hash, str) or not _SHA256.fullmatch(model_hash)):
        raise _fail("INVALID_VERIFIER_IDENTITY", "verifier.model_hash")
    return {"id": identity["id"], "version": identity["version"], "model_hash": model_hash}


def _detach(value: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        detached = json.loads(encoded)
    except (TypeError, ValueError, UnicodeEncodeError):
        raise _fail("INVALID_RECORD", "record") from None
    if type(detached) is not dict:
        raise _fail("INVALID_RECORD", "record")
    return detached


def _validated_exchange(request: Any, response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        canonical_request = verifier.validate_request(request)
    except verifier.VerifierError:
        raise _fail("INVALID_REQUEST", "request") from None
    try:
        canonical_response = verifier.validate_response(response, canonical_request)
    except verifier.VerifierError:
        raise _fail("INVALID_RESPONSE", "response") from None
    return canonical_request, canonical_response


def build_verification_record(request: Any, response: Any, verifier_identity: Any) -> dict[str, Any]:
    """Build one detached decision record from a strict verifier exchange."""

    identity = _validate_identity(verifier_identity)
    canonical_request, canonical_response = _validated_exchange(request, response)
    original = canonical_request["primary_proposal"]["speaker_id"]
    proposed = canonical_response["speaker_id"]
    alternative = canonical_response["alternative_speaker_id"]
    decision = canonical_response["decision"]

    if decision == "correct_primary":
        state = "verified_consensus"
        effective = original
        auto_approved, review_required = True, False
    elif decision == "override_primary":
        state = "auto_corrected"
        effective = proposed
        auto_approved, review_required = True, False
    elif decision == "ambiguous":
        state = "ambiguous"
        effective = None
        auto_approved, review_required = False, True
    elif decision == "unresolved":
        state = "unresolved"
        proposed, alternative, effective = "unknown", None, None
        auto_approved, review_required = False, True
    else:  # Defensive: validate_response currently owns this contract.
        raise _fail("INVALID_RESPONSE", "decision")

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "span_id": canonical_request["span_id"],
        "original_speaker_id": original,
        "proposed_speaker_id": proposed,
        "effective_speaker_id": effective,
        "alternative_speaker_id": alternative,
        "decision_state": state,
        "review_required": review_required,
        "auto_approved": auto_approved,
        "primary_confidence_state": canonical_request["primary_proposal"]["confidence_state"],
        "primary_confidence": canonical_request["primary_proposal"]["confidence"],
        "confidence": canonical_response["confidence"],
        "evidence_offsets": canonical_response["evidence_offsets"],
        "reason_code": canonical_response["reason_code"],
        "verifier": identity,
        "verifier_decision": decision,
    }
    return _detach(record)


def _validate_record(record: Any) -> dict[str, Any]:
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        raise _fail("INVALID_RECORD", "record")
    if type(record["schema_version"]) is not int or record["schema_version"] != RECORD_SCHEMA_VERSION:
        raise _fail("INVALID_RECORD", "schema_version")
    _safe_identifier(record["span_id"], allow_none=False, field="span_id")
    for field in ("original_speaker_id", "proposed_speaker_id", "effective_speaker_id", "alternative_speaker_id"):
        _safe_identifier(record[field], allow_none=True, field=field)
    state = record["decision_state"]
    if not isinstance(state, str) or state not in DECISION_STATES:
        raise _fail("INVALID_RECORD", "decision_state")
    if type(record["review_required"]) is not bool or type(record["auto_approved"]) is not bool:
        raise _fail("INVALID_RECORD", "review_required")
    expected = {
        "verified_consensus": (False, True),
        "auto_corrected": (False, True),
        "ambiguous": (True, False),
        "unresolved": (True, False),
    }[state]
    if (record["review_required"], record["auto_approved"]) != expected:
        raise _fail("INVALID_RECORD", "decision_state")
    confidence_state = record["primary_confidence_state"]
    if confidence_state not in {"scored", "unscored"}:
        raise _fail("INVALID_RECORD", "primary_confidence_state")
    primary_confidence = record["primary_confidence"]
    if confidence_state == "scored":
        if not _finite_number(primary_confidence) or not 0 <= primary_confidence <= 1:
            raise _fail("INVALID_RECORD", "primary_confidence")
    elif primary_confidence is not None:
        raise _fail("INVALID_RECORD", "primary_confidence")
    if not _finite_number(record["confidence"]) or not 0 <= record["confidence"] <= 1:
        raise _fail("INVALID_RECORD", "confidence")
    evidence = record["evidence_offsets"]
    # Context containment cannot be rechecked here because decision records
    # intentionally omit quote/context text.  Shape, ordering, and safe
    # half-open bounds remain strict so hand-crafted records cannot masquerade
    # as builder output.
    if type(evidence) is not list or len(evidence) > verifier.MAX_EVIDENCE_RANGES:
        raise _fail("INVALID_RECORD", "evidence_offsets")
    previous_end = None
    for item in evidence:
        if type(item) is not list or len(item) != 2 or type(item[0]) is not int or type(item[1]) is not int:
            raise _fail("INVALID_RECORD", "evidence_offsets")
        start, end = item
        if start < 0 or end <= start or (previous_end is not None and start < previous_end):
            raise _fail("INVALID_RECORD", "evidence_offsets")
        previous_end = end
    if not isinstance(record["reason_code"], str) or not _REASON.fullmatch(record["reason_code"]):
        raise _fail("INVALID_RECORD", "reason_code")
    _validate_identity(record["verifier"])
    if not isinstance(record["verifier_decision"], str) or record["verifier_decision"] not in verifier.DECISIONS:
        raise _fail("INVALID_RECORD", "verifier_decision")
    expected_decision = {
        "verified_consensus": "correct_primary",
        "auto_corrected": "override_primary",
        "ambiguous": "ambiguous",
        "unresolved": "unresolved",
    }[state]
    if record["verifier_decision"] != expected_decision:
        raise _fail("INVALID_RECORD", "verifier_decision")
    if record["alternative_speaker_id"] is not None and record["alternative_speaker_id"] == record["proposed_speaker_id"]:
        raise _fail("INVALID_RECORD", "alternative_speaker_id")
    if state == "verified_consensus":
        if record["original_speaker_id"] in {None, "unknown"} or record["proposed_speaker_id"] != record["original_speaker_id"] or record["effective_speaker_id"] != record["original_speaker_id"] or record["alternative_speaker_id"] is not None:
            raise _fail("INVALID_RECORD", "decision_state")
    elif state == "auto_corrected":
        if record["proposed_speaker_id"] in {None, "unknown"} or record["original_speaker_id"] == record["proposed_speaker_id"] or record["effective_speaker_id"] != record["proposed_speaker_id"]:
            raise _fail("INVALID_RECORD", "effective_speaker_id")
    elif state == "ambiguous":
        if record["proposed_speaker_id"] in {None, "unknown"} or record["alternative_speaker_id"] is None or record["alternative_speaker_id"] == record["proposed_speaker_id"] or record["effective_speaker_id"] is not None:
            raise _fail("INVALID_RECORD", "decision_state")
    else:
        if record["proposed_speaker_id"] != "unknown" or record["alternative_speaker_id"] is not None or record["effective_speaker_id"] is not None:
            raise _fail("INVALID_RECORD", "decision_state")
    return record


def summarize_verification_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return deterministic exception-queue counts for generated records.

    Records are counted by state, so input order has no effect.  Only strict
    record dictionaries produced by :func:`build_verification_record` are
    accepted; duplicate span IDs are rejected.
    """

    if type(records) not in (list, tuple):
        raise _fail("INVALID_RECORD", "records")
    summary = {key: 0 for key in SUMMARY_KEYS}
    seen: set[str] = set()
    for record in records:
        value = _validate_record(record)
        span_id = value["span_id"]
        if span_id in seen:
            raise _fail("DUPLICATE_SPAN_ID", "span_id")
        seen.add(span_id)
        state = value["decision_state"]
        summary["total"] += 1
        summary[state] += 1
        if value["auto_approved"]:
            summary["auto_approved"] += 1
        if value["review_required"]:
            summary["review_required"] += 1
    return summary


__all__ = [
    "DECISION_STATES",
    "RECORD_SCHEMA_VERSION",
    "SUMMARY_KEYS",
    "VerificationRecordError",
    "build_verification_record",
    "summarize_verification_records",
]

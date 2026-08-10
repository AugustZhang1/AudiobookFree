"""Synchronous, injectable lifecycle runner for whole-book speaker analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from . import speakers
from .analyzers.booknlp import BookNLPAnalyzerError
from .voice_analysis import VoiceAnalysisError, validate_voice_analysis_status
from .voice_plan import VoicePlanError, with_canonical_artifact_hash


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class VoiceAnalysisCancelled(Exception):
    """Cooperative cancellation requested by the caller or adapter control."""


@dataclass(frozen=True, slots=True)
class AnalyzerDescriptor:
    id: str
    version: str
    model_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or len(self.id) > 512 or any(ord(char) < 32 for char in self.id):
            raise VoiceAnalysisError("INVALID_ANALYZER", "analyzer id is invalid")
        if not isinstance(self.version, str) or not self.version or len(self.version) > 512 or any(ord(char) < 32 for char in self.version):
            raise VoiceAnalysisError("INVALID_ANALYZER", "analyzer version is invalid")
        if self.model_hash is not None and (not isinstance(self.model_hash, str) or not _SHA256.fullmatch(self.model_hash)):
            raise VoiceAnalysisError("INVALID_ANALYZER", "analyzer model_hash is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "model_hash": self.model_hash}


class AnalysisControl:
    """Adapter-facing progress and cancellation boundary."""

    def __init__(self, cancelled: Any, on_progress: Any):
        self._cancelled = cancelled
        self._on_progress = on_progress
        self._completed = 0
        self._total = 0

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def total(self) -> int:
        return self._total

    def check_cancelled(self) -> None:
        if self._cancelled():
            raise VoiceAnalysisCancelled()

    def report(self, completed: int, total: int) -> None:
        self.check_cancelled()
        if type(completed) is not int or type(total) is not int or completed < 0 or total < 0 or completed > total:
            raise VoiceAnalysisError("INVALID_PROGRESS", "progress must be bounded nonnegative integers")
        if completed < self._completed:
            raise VoiceAnalysisError("PROGRESS_REGRESSION", "progress completed cannot regress")
        if self._total and total != self._total:
            raise VoiceAnalysisError("PROGRESS_TOTAL_CHANGED", "progress total cannot change")
        if total == 0 and completed != 0:
            raise VoiceAnalysisError("INVALID_PROGRESS", "zero total requires zero completed")
        if total:
            self._total = total
        self._completed = completed
        self._on_progress(self._completed, self._total)
        self.check_cancelled()


@dataclass(frozen=True, slots=True)
class AnalysisRunResult:
    analysis_id: str
    revision: int
    status: str
    status_artifact: dict[str, Any]
    speaker_artifact: dict[str, Any] | None = None


def _timestamp() -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z"), now


class VoiceAnalysisRunner:
    """Run one injected whole-book analyzer synchronously in-process."""

    def __init__(
        self,
        workspace: Any,
        conversion_id: str,
        adapter: speakers.SpeakerAnalyzer,
        descriptor: AnalyzerDescriptor,
        analysis_id: str,
        revision: int,
        options: Mapping[str, Any] | None = None,
    ):
        try:
            canonical_analysis_id = str(uuid.UUID(analysis_id)) if isinstance(analysis_id, str) else ""
        except (ValueError, AttributeError):
            canonical_analysis_id = ""
        if not isinstance(analysis_id, str) or not _UUID.fullmatch(analysis_id) or analysis_id != analysis_id.lower() or canonical_analysis_id != analysis_id:
            raise VoiceAnalysisError("INVALID_ANALYSIS_ID", "analysis_id must be a canonical UUID")
        if type(revision) is not int or revision <= 0:
            raise VoiceAnalysisError("INVALID_REVISION", "revision must be positive")
        if not callable(getattr(adapter, "analyze", None)):
            raise VoiceAnalysisError("INVALID_ANALYZER", "adapter must implement analyze")
        if options is not None and (not isinstance(options, Mapping) or "analysis_control" in options):
            raise VoiceAnalysisError("INVALID_OPTIONS", "options must be a mapping without analysis_control")
        self.workspace = workspace
        self.conversion_id = conversion_id
        self.adapter = adapter
        self.descriptor = descriptor
        self.analysis_id = analysis_id
        self.revision = revision
        self.options = dict(options or {})
        self._started_at: datetime | None = None
        self._last_time: datetime | None = None
        self._job: dict[str, Any] | None = None
        self._cleaned_text = ""
        self._chapter_plan: dict[str, Any] = {}
        self._control: AnalysisControl | None = None

    def _check_cancelled(self) -> None:
        if self.workspace.voice_analysis_cancellation_requested(self.conversion_id):
            raise VoiceAnalysisCancelled()

    def _status_artifact(
        self,
        status: str,
        stage: str,
        completed: int,
        total: int,
        *,
        cancel_requested: bool = False,
        warnings: list[str] | None = None,
        error: dict[str, str] | None = None,
        finished: bool = False,
        advance_clock: bool = True,
    ) -> dict[str, Any]:
        assert self._job is not None and self._started_at is not None
        timestamp, now = _timestamp()
        if self._last_time is not None and now < self._last_time:
            now = self._last_time
            timestamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")
        if advance_clock:
            self._last_time = now
        finished_at = timestamp if finished else None
        artifact = {
            "schema_version": 1,
            "artifact": "voice-analysis-status",
            "analysis_id": self.analysis_id,
            "revision": self.revision,
            "source_pdf_sha256": self._job["source_pdf_sha256"],
            "cleaned_text_sha256": self._job["cleaned_text_sha256"],
            "chapter_plan_sha256": self._job["chapter_plan_sha256"],
            "chapter_plan_schema_version": 1,
            "analyzer": self.descriptor.as_dict(),
            "status": status,
            "stage": stage,
            "progress": {"completed": completed, "total": total},
            "cancel_requested": cancel_requested,
            "warnings": list(warnings or []),
            "error": error,
            "started_at": self._started_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "updated_at": timestamp,
            "finished_at": finished_at,
        }
        return with_canonical_artifact_hash(artifact)

    def _set_status(
        self,
        status: str,
        stage: str,
        completed: int,
        total: int,
        *,
        cancel_requested: bool = False,
        warnings: list[str] | None = None,
        error: dict[str, str] | None = None,
        finished: bool = False,
    ) -> dict[str, Any]:
        artifact = self._status_artifact(
            status,
            stage,
            completed,
            total,
            cancel_requested=cancel_requested,
            warnings=warnings,
            error=error,
            finished=finished,
        )
        return self.workspace.persist_voice_analysis_status(self.conversion_id, artifact)

    def _progress(self, completed: int, total: int) -> None:
        self._set_status("running", "analyzing", completed, total, cancel_requested=False)

    def _normalize_analysis(self, result: Any) -> speakers.MachineAnalysis:
        if isinstance(result, speakers.MachineAnalysis):
            analysis = result
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            try:
                analysis = speakers.MachineAnalysis(tuple(result))
            except speakers.SpeakerPlanError as exc:
                raise VoiceAnalysisError(exc.code, exc.message, details=exc.details) from exc
        else:
            raise VoiceAnalysisError("ANALYZER_OUTPUT_INVALID", "analyzer output must be MachineAnalysis or speaker spans")
        if analysis.source_hash is not None and analysis.source_hash != self._job["source_pdf_sha256"]:
            raise VoiceAnalysisError("ANALYZER_OUTPUT_INVALID", "analyzer output source binding is invalid")
        return analysis

    def _analysis_artifact(self, analysis: speakers.MachineAnalysis) -> dict[str, Any]:
        assert self._job is not None
        spans: list[dict[str, Any]] = []
        for span in analysis.spans:
            if len(span.provenance) not in {1, 2}:
                raise VoiceAnalysisError("ANALYZER_OUTPUT_INVALID", "span provenance is invalid")
            provenance: dict[str, str] = {"source": span.provenance[0]}
            if len(span.provenance) == 2:
                provenance["quote_id"] = span.provenance[1]
            spans.append({
                "span_id": span.span_id,
                "chapter_index": span.chapter_index,
                "source_start": span.source_start,
                "source_end": span.source_end,
                "type": span.span_type,
                "speaker_id": span.speaker_id,
                "confidence": {"score": span.confidence.score, "band": span.confidence.band, "reasons": list(span.confidence.reasons)},
                "provenance": provenance,
            })
        artifact = {
            "schema_version": 1,
            "artifact": "speaker-analysis",
            "revision": self.revision,
            "source_pdf_sha256": self._job["source_pdf_sha256"],
            "cleaned_text_sha256": self._job["cleaned_text_sha256"],
            "chapter_plan_sha256": self._job["chapter_plan_sha256"],
            "chapter_plan_schema_version": 1,
            "analyzer": self.descriptor.as_dict(),
            "characters": [dict(character) for character in analysis.characters],
            "spans": spans,
            "warnings": list(analysis.warnings),
        }
        try:
            return with_canonical_artifact_hash(artifact)
        except VoicePlanError as exc:
            raise VoiceAnalysisError("ANALYZER_OUTPUT_INVALID", "analyzer output is not canonical JSON", details=exc.details) from exc

    def _terminal_error(self, exc: Exception) -> tuple[str, str]:
        safe_messages = {
            "INVALID_PROGRESS": "invalid analyzer progress",
            "PROGRESS_REGRESSION": "analyzer progress regressed",
            "PROGRESS_TOTAL_CHANGED": "analyzer progress total changed",
            "INCOMPLETE_PROGRESS": "analyzer progress was incomplete",
            "ANALYZER_OUTPUT_INVALID": "analyzer output was invalid",
        }
        if isinstance(exc, BookNLPAnalyzerError) and exc.code == "OUTPUT_TOO_LARGE":
            return "OUTPUT_TOO_LARGE", "analyzer output exceeded the size limit"
        code = exc.code if isinstance(exc, VoiceAnalysisError) else "ANALYZER_FAILED"
        if code in safe_messages:
            return code, safe_messages[code]
        return "ANALYZER_FAILED", "voice analysis failed"

    def run(self, options: Mapping[str, Any] | None = None) -> AnalysisRunResult:
        if options is not None and (not isinstance(options, Mapping) or "analysis_control" in options):
            raise VoiceAnalysisError("INVALID_OPTIONS", "options must be a mapping without analysis_control")
        run_options = dict(self.options)
        if options:
            if "analysis_control" in options:
                raise VoiceAnalysisError("INVALID_OPTIONS", "analysis_control is reserved")
            run_options.update(options)
        self._job = self.workspace.read_job(self.conversion_id)
        self.workspace.load_analysis(self.conversion_id)
        self._cleaned_text, _ = self.workspace.load_cleaned_artifacts(self.conversion_id)
        self._chapter_plan = self.workspace.load_chapter_plan(self.conversion_id)
        self.workspace.clear_voice_analysis_cancel_request(self.conversion_id)
        self._started_at = datetime.now(timezone.utc)
        self._last_time = self._started_at
        try:
            self._set_status("queued", "queued", 0, 0)
            self._set_status("running", "preparing", 0, 0)
            self._control = AnalysisControl(
                lambda: self.workspace.voice_analysis_cancellation_requested(self.conversion_id),
                self._progress,
            )
            self._check_cancelled()
            adapter_options = dict(run_options)
            adapter_options["analysis_control"] = self._control
            output = self.adapter.analyze(
                self._cleaned_text,
                self._chapter_plan,
                self._job["source_pdf_sha256"],
                adapter_options,
            )
            self._check_cancelled()
            analysis = self._normalize_analysis(output)
            self._check_cancelled()
            total = self._control.total if self._control else 0
            completed = self._control.completed if self._control else 0
            self._set_status("running", "validating", completed, total)
            self._check_cancelled()
            if total and completed != total:
                raise VoiceAnalysisError("INCOMPLETE_PROGRESS", "analyzer progress was incomplete")
            speaker_artifact = self._analysis_artifact(analysis)
            self._check_cancelled()
            completed_candidate = self._status_artifact(
                "completed",
                "completed",
                completed,
                total,
                warnings=list(analysis.warnings),
                finished=True,
                advance_clock=False,
            )
            validate_voice_analysis_status(
                completed_candidate,
                self._cleaned_text,
                self._chapter_plan,
                expected_source_pdf_sha256=self._job["source_pdf_sha256"],
                expected_chapter_plan_sha256=self._job["chapter_plan_sha256"],
            )
            self._set_status("running", "persisting", completed, total)
            self._check_cancelled()
            self.workspace.persist_speaker_analysis(self.conversion_id, speaker_artifact)
            completed_candidate = self._status_artifact(
                "completed",
                "completed",
                completed,
                total,
                warnings=list(analysis.warnings),
                finished=True,
            )
            validate_voice_analysis_status(
                completed_candidate,
                self._cleaned_text,
                self._chapter_plan,
                expected_source_pdf_sha256=self._job["source_pdf_sha256"],
                expected_chapter_plan_sha256=self._job["chapter_plan_sha256"],
            )
            completed_status = self.workspace.persist_voice_analysis_status(self.conversion_id, completed_candidate)
            self.workspace.clear_voice_analysis_cancel_request(self.conversion_id)
            return AnalysisRunResult(self.analysis_id, self.revision, "completed", completed_status, speaker_artifact)
        except VoiceAnalysisCancelled:
            try:
                cancelled = self._set_status("cancelled", "cancelled", self._control.completed if self._control else 0, self._control.total if self._control else 0, cancel_requested=True, finished=True)
            finally:
                self.workspace.clear_voice_analysis_cancel_request(self.conversion_id)
            return AnalysisRunResult(self.analysis_id, self.revision, "cancelled", cancelled)
        except Exception as exc:
            code, message = self._terminal_error(exc)
            try:
                try:
                    self._set_status("failed", "failed", self._control.completed if self._control else 0, self._control.total if self._control else 0, error={"code": code, "message": message}, finished=True)
                except Exception:
                    pass
            finally:
                self.workspace.clear_voice_analysis_cancel_request(self.conversion_id)
            raise


class DeterministicFakeAnalyzer:
    """Small in-process analyzer used for deterministic integration tests."""

    def analyze(self, cleaned_text: str, chapter_plan: dict[str, Any], source_hash: str, options: Any = None) -> speakers.MachineAnalysis:
        control = options.get("analysis_control") if isinstance(options, Mapping) else None
        chapters = chapter_plan.get("chapters", [])
        spans: list[speakers.SpeakerSpan] = []
        total = len(chapters)
        for position, chapter in enumerate(chapters, 1):
            if control is not None:
                control.report(position, total)
            spans.append(speakers.SpeakerSpan(
                f"fake:{position}",
                chapter["index"],
                chapter["start_offset"],
                chapter["end_offset"],
                "narration",
                "narrator",
                speakers.Confidence(1.0, "high", ("deterministic_fake",)),
                ("deterministic_fake",),
            ))
        return speakers.MachineAnalysis(tuple(spans), source_hash, (), (), ("deterministic fake analyzer",))


FakeSpeakerAnalyzer = DeterministicFakeAnalyzer


__all__ = [
    "AnalysisControl", "AnalysisRunResult", "AnalyzerDescriptor", "DeterministicFakeAnalyzer",
    "FakeSpeakerAnalyzer", "VoiceAnalysisCancelled", "VoiceAnalysisRunner",
]

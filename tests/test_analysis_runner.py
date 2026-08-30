from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
import shutil
import uuid

import pytest

from pdf_audiobook import speakers
import pdf_audiobook.voice_analysis as voice_analysis_module
from pdf_audiobook.analyzers.booknlp import BookNLPAnalyzerError
from pdf_audiobook.analysis_runner import (
    AnalysisControl,
    AnalyzerDescriptor,
    DeterministicFakeAnalyzer,
    VoiceAnalysisError,
    VoiceAnalysisCancelled,
    VoiceAnalysisRunner,
)
from pdf_audiobook.workspace import ManifestError, Workspace


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / f".pytest-runner-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _prepared(tmp_path: Path) -> tuple[Workspace, dict, str, dict]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    text = "One chapter. Two chapter."
    split = text.index("Two")
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": split}, {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    return workspace, workspace.read_job(manifest["conversion_id"]), text, plan


def _runner(workspace: Workspace, job: dict, adapter, options=None) -> VoiceAnalysisRunner:
    return VoiceAnalysisRunner(workspace, job["conversion_id"], adapter, AnalyzerDescriptor("fake", "1"), "12345678-1234-5678-9234-567812345678", 7, options)


def _prepared_three(tmp_path: Path) -> tuple[Workspace, dict, str, dict]:
    source = tmp_path / "three.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data-three")
    manifest = workspace.create_conversion(source)
    text = "One chapter. Two chapter. Three chapter."
    first = text.index("Two")
    second = text.index("Three")
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": first}, {"source_page": 2, "cleaned_start": first, "cleaned_end": second}, {"source_page": 3, "cleaned_start": second, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": first, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}, {"index": 2, "title": "Two", "start_offset": first, "end_offset": second, "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}, {"index": 3, "title": "Three", "start_offset": second, "end_offset": len(text), "start_page": 3, "end_page": 3, "source_type": "whole", "word_count": 2}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    return workspace, workspace.read_job(manifest["conversion_id"]), text, plan


def test_runner_scopes_selected_chapters_and_remaps_full_book_spans(tmp_path: Path) -> None:
    workspace, job, text, plan = _prepared_three(tmp_path)
    calls: dict[str, object] = {}

    class Capturing:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            calls["text"] = cleaned_text
            calls["plan"] = chapter_plan
            control = options["analysis_control"]
            spans = []
            for chapter in chapter_plan["chapters"]:
                spans.append(speakers.SpeakerSpan(f"captured:{chapter['index']}", chapter["index"], chapter["start_offset"], chapter["end_offset"], "narration", "narrator", speakers.Confidence(1.0, "high", ("captured",)), ("captured",)))
            control.report(len(spans), len(spans))
            return speakers.MachineAnalysis(tuple(spans), source_hash)

    result = _runner(workspace, job, Capturing(), {"chapter_start": 2, "chapter_end": 3}).run()
    scoped_plan = calls["plan"]
    assert calls["text"] == text[plan["chapters"][1]["start_offset"] :]
    assert [chapter["index"] for chapter in scoped_plan["chapters"]] == [1, 2]
    assert scoped_plan["chapters"][0]["start_offset"] == 0
    assert scoped_plan["chapters"][-1]["end_offset"] == len(calls["text"])
    assert result.status_artifact["chapter_start"] == 2 and result.status_artifact["chapter_end"] == 3
    assert result.speaker_artifact["chapter_start"] == 2 and result.speaker_artifact["chapter_end"] == 3
    assert [span["chapter_index"] for span in result.speaker_artifact["spans"]] == [2, 3]
    assert [span["source_start"] for span in result.speaker_artifact["spans"]] == [plan["chapters"][1]["start_offset"], plan["chapters"][2]["start_offset"]]


def test_fake_runner_success_history_artifact_and_job_immutability(tmp_path: Path, monkeypatch) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    before = workspace.job_path(job["conversion_id"]).read_bytes()
    history: list[dict] = []
    original = workspace.persist_voice_analysis_status

    def capture(conversion_id, status):
        history.append(status)
        return original(conversion_id, status)

    monkeypatch.setattr(workspace, "persist_voice_analysis_status", capture)
    result = _runner(workspace, job, DeterministicFakeAnalyzer()).run()
    assert result.status == "completed"
    assert result.status_artifact["analysis_id"] == "12345678-1234-5678-9234-567812345678"
    assert result.status_artifact["revision"] == 7
    assert result.status_artifact["progress"] == {"completed": 2, "total": 2}
    assert result.speaker_artifact is not None
    assert len(result.speaker_artifact["spans"]) == 2
    assert [item["stage"] for item in history] == ["queued", "preparing", "analyzing", "analyzing", "validating", "persisting", "completed"]
    updated = [datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")) for item in history]
    assert updated == sorted(updated)
    persisting = next(item for item in history if item["stage"] == "persisting")
    completed = next(item for item in history if item["stage"] == "completed")
    assert completed["updated_at"] >= persisting["updated_at"]
    assert workspace.job_path(job["conversion_id"]).read_bytes() == before
    assert not workspace.voice_analysis_cancellation_requested(job["conversion_id"])


def test_runner_cancellation_preserves_prior_artifacts_and_terminal_evidence(tmp_path: Path) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    conversion = workspace.conversion_path(job["conversion_id"])
    speaker_path = conversion / "speaker-analysis.json"
    voice_path = conversion / "voice-plan.json"
    speaker_path.write_bytes(b"old speaker")
    voice_path.write_bytes(b"old voice")

    class Cancelling:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            control = options["analysis_control"]
            control.report(0, 2)
            workspace.request_voice_analysis_cancel(job["conversion_id"])
            control.report(1, 2)

    result = _runner(workspace, job, Cancelling()).run()
    assert result.status == "cancelled"
    assert speaker_path.read_bytes() == b"old speaker"
    assert voice_path.read_bytes() == b"old voice"
    assert not workspace.voice_analysis_cancellation_requested(job["conversion_id"])
    assert workspace.load_voice_analysis_status(job["conversion_id"])["status"] == "cancelled"


def test_runner_failure_preserves_artifacts_and_sanitizes_error(tmp_path: Path) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    conversion = workspace.conversion_path(job["conversion_id"])
    (conversion / "speaker-analysis.json").write_bytes(b"old speaker")
    (conversion / "voice-plan.json").write_bytes(b"old voice")

    class Failing:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            raise RuntimeError("book text must not leak")

    with pytest.raises(RuntimeError):
        _runner(workspace, job, Failing()).run()
    status = workspace.load_voice_analysis_status(job["conversion_id"])
    assert status["status"] == "failed"
    assert status["error"] == {"code": "ANALYZER_FAILED", "message": "voice analysis failed"}
    assert (conversion / "speaker-analysis.json").read_bytes() == b"old speaker"
    assert (conversion / "voice-plan.json").read_bytes() == b"old voice"


def test_runner_maps_booknlp_output_size_error_safely_and_preserves_artifacts(tmp_path: Path) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    conversion = workspace.conversion_path(job["conversion_id"])
    speaker_path = conversion / "speaker-analysis.json"
    voice_path = conversion / "voice-plan.json"
    speaker_path.write_bytes(b"old speaker")
    voice_path.write_bytes(b"old voice")

    class TooLarge:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            raise BookNLPAnalyzerError("OUTPUT_TOO_LARGE", "secret book content")

    with pytest.raises(BookNLPAnalyzerError):
        _runner(workspace, job, TooLarge()).run()
    status = workspace.load_voice_analysis_status(job["conversion_id"])
    assert status["error"] == {"code": "OUTPUT_TOO_LARGE", "message": "analyzer output exceeded the size limit"}
    assert speaker_path.read_bytes() == b"old speaker"
    assert voice_path.read_bytes() == b"old voice"


def test_runner_rejects_reserved_options_and_progress_regression() -> None:
    with pytest.raises(VoiceAnalysisError) as exc:
        AnalysisControl(lambda: False, lambda *_: None).report(True, 1)
    assert exc.value.code == "INVALID_PROGRESS"
    control = AnalysisControl(lambda: False, lambda *_: None)
    control.report(1, 2)
    with pytest.raises(VoiceAnalysisError) as exc:
        control.report(0, 2)
    assert exc.value.code == "PROGRESS_REGRESSION"


def test_runner_rejects_reserved_option_and_malformed_output(tmp_path: Path) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    with pytest.raises(VoiceAnalysisError) as exc:
        _runner(workspace, job, DeterministicFakeAnalyzer(), {"analysis_control": "caller"})
    assert exc.value.code == "INVALID_OPTIONS"

    class Malformed:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            return {"not": "spans"}

    with pytest.raises(VoiceAnalysisError) as exc:
        _runner(workspace, job, Malformed()).run()
    assert exc.value.code == "ANALYZER_OUTPUT_INVALID"
    assert workspace.load_voice_analysis_status(job["conversion_id"])["error"]["code"] == "ANALYZER_OUTPUT_INVALID"


def test_runner_incomplete_progress_fails_before_machine_visibility(tmp_path: Path) -> None:
    workspace, job, _, plan = _prepared(tmp_path)
    conversion = workspace.conversion_path(job["conversion_id"])
    speaker_path = conversion / "speaker-analysis.json"
    voice_path = conversion / "voice-plan.json"
    speaker_path.write_bytes(b"prior speaker")
    voice_path.write_bytes(b"prior voice")

    class Incomplete:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            options["analysis_control"].report(1, 2)
            return DeterministicFakeAnalyzer().analyze(cleaned_text, chapter_plan, source_hash, {"analysis_control": None})

    with pytest.raises(VoiceAnalysisError) as exc:
        _runner(workspace, job, Incomplete()).run()
    assert exc.value.code == "INCOMPLETE_PROGRESS"
    assert workspace.load_voice_analysis_status(job["conversion_id"])["error"] == {"code": "INCOMPLETE_PROGRESS", "message": "analyzer progress was incomplete"}
    assert speaker_path.read_bytes() == b"prior speaker"
    assert voice_path.read_bytes() == b"prior voice"


def test_runner_completed_status_size_failure_precedes_machine_persistence(tmp_path: Path, monkeypatch) -> None:
    workspace, job, _, _ = _prepared(tmp_path)
    conversion = workspace.conversion_path(job["conversion_id"])
    speaker_path = conversion / "speaker-analysis.json"
    voice_path = conversion / "voice-plan.json"
    speaker_path.write_bytes(b"prior speaker")
    voice_path.write_bytes(b"prior voice")
    class WarningAnalyzer:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            result = DeterministicFakeAnalyzer().analyze(cleaned_text, chapter_plan, source_hash, options)
            return type(result)(result.spans, result.source_hash, result.provenance, result.characters, ("warning",))

    runner = _runner(workspace, job, WarningAnalyzer())
    original = runner._analysis_artifact
    def build_with_small_limit(analysis):
        result = original(analysis)
        monkeypatch.setattr(voice_analysis_module, "MAX_ARTIFACT_BYTES", 1)
        return result
    runner._analysis_artifact = build_with_small_limit
    with pytest.raises(VoiceAnalysisError):
        runner.run()
    assert speaker_path.read_bytes() == b"prior speaker"
    assert voice_path.read_bytes() == b"prior voice"


def test_runner_never_persists_adapter_exception_message(tmp_path: Path) -> None:
    workspace, job, _, _ = _prepared(tmp_path)

    class Leaky:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            raise VoiceAnalysisError("ANALYZER_OUTPUT_INVALID", "cleaned_text secret must not persist")

    with pytest.raises(VoiceAnalysisError):
        _runner(workspace, job, Leaky()).run()
    error = workspace.load_voice_analysis_status(job["conversion_id"])["error"]
    assert error == {"code": "ANALYZER_OUTPUT_INVALID", "message": "analyzer output was invalid"}


def test_analysis_control_total_change_and_cancellation_before_and_after_progress() -> None:
    control = AnalysisControl(lambda: False, lambda *_: None)
    control.report(1, 2)
    with pytest.raises(VoiceAnalysisError) as exc:
        control.report(1, 3)
    assert exc.value.code == "PROGRESS_TOTAL_CHANGED"
    cancelled = [True]
    with pytest.raises(VoiceAnalysisCancelled):
        AnalysisControl(lambda: cancelled[0], lambda *_: None).report(0, 1)
    cancelled[0] = False
    def callback(*_):
        cancelled[0] = True
    with pytest.raises(VoiceAnalysisCancelled):
        AnalysisControl(lambda: cancelled[0], callback).report(1, 1)


def test_runner_never_leaves_a_run_reporting_running(tmp_path: Path) -> None:
    """If the terminal status write itself fails, the run must not stay "running".

    The cancel marker is cleared regardless, so a swallowed write left the UI
    polling a permanently in-flight analysis with no worker behind it.
    """

    workspace, job, _, _ = _prepared(tmp_path)

    class Failing:
        def analyze(self, cleaned_text, chapter_plan, source_hash, options):
            raise RuntimeError("analyzer exploded")

    runner = _runner(workspace, job, Failing())
    original = runner._set_status
    calls = {"n": 0}

    def flaky(status, stage, completed, total, **kwargs):
        # Fail only the first terminal write, exactly as a transient
        # analyzer-derived-field problem would.
        if status == "failed" and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("status write failed")
        return original(status, stage, completed, total, **kwargs)

    runner._set_status = flaky
    with pytest.raises(RuntimeError):
        runner.run()

    status = workspace.load_voice_analysis_status(job["conversion_id"])
    assert status["status"] == "failed", f"run left durably reporting {status['status']!r}"
    assert calls["n"] == 1

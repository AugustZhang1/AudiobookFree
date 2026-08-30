from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

import pdf_audiobook.app as app_module
from pdf_audiobook.app import create_app
from pdf_audiobook.analysis_runner import DeterministicFakeAnalyzer
from pdf_audiobook.audio import write_pcm_wav
from pdf_audiobook.tts import FakeVoice, plan_interactive_chunks
from pdf_audiobook.voice_plan import build_voice_plan, with_canonical_artifact_hash
from pdf_audiobook.voice_shaping import ShapingCapability
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.workspace import Workspace, atomic_write_json
from pdf_audiobook.security import atomic_write_instance, build_instance
from test_pdf import make_pdf


def _app(root: Path, port: int, *, opener=None, worker_poll=1, worker_launcher=None, voice_analyzer=None, preview_generator=None):
    preview = root / "previews"
    preview.mkdir(parents=True)
    app = create_app(
        port=port,
        session_token="phase6-token",
        instance_file=root / "instance.json",
        data_root=root / "data",
        preview_root=preview,
        preview_generator=preview_generator,
        path_opener=opener,
        voice_analyzer=voice_analyzer,
        worker_launcher=worker_launcher or (lambda *_: type("P", (), {"poll": lambda self: worker_poll})()),
    )
    client = TestClient(app, base_url=f"http://127.0.0.1:{port}")
    headers = {"Origin": f"http://127.0.0.1:{port}"}
    assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
    return app, client, headers, preview


def test_cancel_vs_start_race_returns_active_worker_contention() -> None:
    root = Path("tests") / f".pytest-phase6-cancel-start-race-{uuid.uuid4().hex}"
    root.mkdir()
    entered = threading.Event()
    release = threading.Event()

    class StartingProcess:
        def poll(self):
            return 0

    def launch(*_args):
        entered.set()
        release.wait(5)
        return StartingProcess()

    start_client = None
    cancel_client = None
    start_result: list[object] = []
    start_error: list[BaseException] = []
    start_thread = None
    try:
        app, start_client, headers, _ = _app(root, 19879, worker_launcher=launch)
        cancel_client = TestClient(app, base_url="http://127.0.0.1:19879")
        assert cancel_client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert start_client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert start_client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200

        def start_generation() -> None:
            try:
                start_result.append(start_client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers))
            except BaseException as exc:
                start_error.append(exc)

        start_thread = threading.Thread(target=start_generation)
        start_thread.start()
        assert entered.wait(5), "generation start did not reach the worker launcher"

        cancelled = cancel_client.post("/api/generation/cancel", json={}, headers=headers)
        assert cancelled.status_code == 409
        assert cancelled.json() == {"error": {"code": "ACTIVE_WORKER", "message": "another generation operation is in progress"}}
    finally:
        release.set()
        if start_thread is not None:
            start_thread.join(5)
        if start_client is not None:
            start_client.close()
        if cancel_client is not None:
            cancel_client.close()
        shutil.rmtree(root, ignore_errors=True)
    assert not start_thread.is_alive()
    assert not start_error
    assert start_result and start_result[0].status_code == 200
    app.state.phase1.worker_process = None


def test_shutdown_vs_start_race_returns_active_worker_contention() -> None:
    root = Path("tests") / f".pytest-phase6-shutdown-start-race-{uuid.uuid4().hex}"
    root.mkdir()
    entered = threading.Event()
    release = threading.Event()

    class StartingProcess:
        def poll(self):
            return 0

    def launch(*_args):
        entered.set()
        release.wait(5)
        return StartingProcess()

    start_client = None
    shutdown_client = None
    start_result: list[object] = []
    start_error: list[BaseException] = []
    start_thread = None
    try:
        app, start_client, headers, _ = _app(root, 19876, worker_launcher=launch)
        shutdown_client = TestClient(app, base_url="http://127.0.0.1:19876")
        assert shutdown_client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert start_client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert start_client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200

        def start_generation() -> None:
            try:
                start_result.append(start_client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers))
            except BaseException as exc:
                start_error.append(exc)

        start_thread = threading.Thread(target=start_generation)
        start_thread.start()
        assert entered.wait(5), "generation start did not reach the worker launcher"

        shutting_down = shutdown_client.post("/api/shutdown", headers=headers)
        assert shutting_down.status_code == 409
        assert shutting_down.json() == {"error": {"code": "ACTIVE_WORKER", "message": "another generation operation is in progress"}}
        assert not app.state.phase1.shutdown_event.is_set()
    finally:
        release.set()
        if start_thread is not None:
            start_thread.join(5)
        if start_client is not None:
            start_client.close()
        if shutdown_client is not None:
            shutdown_client.close()
        shutil.rmtree(root, ignore_errors=True)
    assert not start_thread.is_alive()
    assert not start_error
    assert start_result and start_result[0].status_code == 200
    app.state.phase1.worker_process = None


def test_shutdown_requests_cancel_and_reaps_live_worker_before_cleanup() -> None:
    root = Path("tests") / f".pytest-phase6-shutdown-worker-{uuid.uuid4().hex}"
    root.mkdir()
    instance_file = root / "instance.json"
    events: list[object] = []

    class ShutdownProcess:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")
            assert instance_file.is_file()
            assert conversion_id is not None
            assert Workspace(root / "data").cancel_marker_path(conversion_id).is_file()

        def wait(self, timeout):
            events.append(("wait", timeout))
            assert timeout > 0
            assert instance_file.is_file()
            self.returncode = 0

    process = ShutdownProcess()
    client = None
    try:
        port = 19877
        app, client, headers, _ = _app(root, port, worker_launcher=lambda *_: process)
        state = app.state.phase1
        state.session_token = "phase6-shutdown-session-token-012345678901234567890123"
        client.cookies.clear()
        assert client.post("/api/session/bootstrap", json={"token": state.session_token}, headers=headers).status_code == 200
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        started = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert started.status_code == 200
        conversion_id = Workspace(root / "data").inspect_startup().conversion_id
        assert conversion_id is not None
        atomic_write_instance(build_instance(pid=os.getpid(), port=port, launch_id=state.launch_id, token=state.session_token), instance_file)

        response = client.post("/api/shutdown", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"shutting_down": True}
        assert events == ["terminate", ("wait", app_module.WORKER_SHUTDOWN_TIMEOUT_SECONDS)]
        assert process.poll() == 0
        assert state.worker_process is None
        assert state.shutdown_event.is_set()
        assert not instance_file.exists()
    finally:
        if client is not None:
            client.close()
        shutil.rmtree(root, ignore_errors=True)


class _BlockingVoiceAnalyzer:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def analyze(self, cleaned_text, chapter_plan, source_hash, options=None):
        self.started.set()
        self.release.wait(5)
        return DeterministicFakeAnalyzer().analyze(cleaned_text, chapter_plan, source_hash, options)


class _FailingVoiceAnalyzer:
    def analyze(self, *_args, **_kwargs):
        raise RuntimeError("secret-book-text-should-never-leak")


def _prepare_voice_analysis(root: Path, client, headers) -> None:
    pdf = make_pdf(root / "voice-book.pdf", ["One sentence. Two sentence."])
    upload_headers = {**headers, "X-PDF-Filename": "voice-book.pdf", "Content-Type": "application/pdf"}
    assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
    assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200


def _wait_voice_terminal(app, client, headers) -> dict:
    for _ in range(200):
        response = client.get("/api/voice-analysis/status", headers=headers)
        if response.status_code == 200 and response.json()["status"] in {"completed", "failed", "cancelled"}:
            while app.state.phase1.voice_analysis_thread is not None and app.state.phase1.voice_analysis_thread.is_alive():
                time.sleep(0.005)
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"voice analysis did not reach terminal state: {response.text}")


def _prepare_approved_voice_plan(root: Path, client, headers, app=None) -> tuple[Workspace, str, dict]:
    """Create a deterministic approved plan with aliases and two cast voices."""

    workspace, manifest, text, artifact, path = _prepare_speaker_review(root)
    if app is not None and app.state.phase1.voice_analyzer is not None:
        started = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert started.status_code == 200, started.text
        _wait_voice_terminal(app, client, headers)
    enriched = {**artifact, "characters": [
        {"character_id": "alice", "canonical_label": "Alice", "aliases": [{"alias": "Al", "kind": "proper", "confidence": 1.0, "provenance": {"source": "fake", "token_start": 0, "token_end": 1}}], "line_count": 1, "quote_count": 10},
        {"character_id": "bob", "canonical_label": "Bob", "aliases": [{"alias": "Bobby", "kind": "proper", "confidence": 1.0, "provenance": {"source": "fake", "token_start": 0, "token_end": 1}}], "line_count": 1, "quote_count": 10},
    ]}
    enriched["spans"] = [
        {**span, "speaker_id": "alice" if span["span_id"] == "alice" else ("bob" if span["span_id"] == "bob" else "narrator")}
        for span in artifact["spans"]
    ]
    workspace.persist_speaker_analysis(manifest["conversion_id"], with_canonical_artifact_hash(enriched))
    draft = client.post("/api/voice-plan/draft", json={"analysis_revision": 3}, headers=headers)
    assert draft.status_code == 200, draft.text
    identity = draft.json()
    # This helper exercises cast API mutations independently of the analyzer's
    # promotion threshold; seed two valid character entries explicitly.
    draft_plan = workspace.load_voice_plan(manifest["conversion_id"])
    draft_plan["cast"].extend([
        {"cast_id": "character-alice", "display_label": "Alice", "role": "character", "relationship": "separate_from_narrator", "voice_id": "af_bella", "voice_settings": {"speed": 1.0}},
        {"cast_id": "character-bob", "display_label": "Bob", "role": "character", "relationship": "separate_from_narrator", "voice_id": "af_nicole", "voice_settings": {"speed": 1.0}},
    ])
    draft_plan["aliases"] = [
        {"alias_id": "alias-character-alice-1", "text": "Al", "character_id": "character-alice", "override_state": "accepted"},
        {"alias_id": "alias-character-bob-1", "text": "Bobby", "character_id": "character-bob", "override_state": "accepted"},
    ]
    for chapter in draft_plan["chapters"]:
        for span in chapter["spans"]:
            if span["span_id"] == "alice":
                span["speaker_id"] = "character-alice"
                span["type"] = "dialogue"
            elif span["span_id"] == "bob":
                span["speaker_id"] = "character-bob"
                span["type"] = "dialogue"
    workspace.persist_voice_plan(manifest["conversion_id"], with_canonical_artifact_hash(draft_plan))
    approved = client.post("/api/voice-plan/approve", json={"expected_revision": identity["revision"], "accept_narrator_fallback": True}, headers=headers)
    assert approved.status_code == 200, approved.text
    return workspace, manifest["conversion_id"], approved.json()


def _prepare_speaker_review(root: Path) -> tuple[Workspace, dict, str, dict, Path]:
    workspace = Workspace(root / "data")
    source = root / "speaker-book.pdf"
    source.write_bytes(b"speaker source")
    manifest = workspace.create_conversion(source)
    text = "A" * 300 + ".\nÅlice says.\nBob waits."
    first_end = text.index("\n")
    alice_start = first_end + 1
    alice_end = text.index("\n", alice_start)
    split = alice_end + 1
    assert split == text.index("Bob")
    workspace.persist_analysis(manifest["conversion_id"], {
        "source_pdf_sha256": manifest["source_pdf_sha256"],
        "title": "Speaker book",
        "cleaned_text": text,
        "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}],
        "warnings": [],
    })
    plan = {
        "schema_version": 1,
        "mode": "original",
        "requested_count": None,
        "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapters": [
            {"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 3},
            {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2},
        ],
        "warnings": [],
    }
    persisted = workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    spans = [
        {"span_id": "long", "chapter_index": 1, "source_start": 0, "source_end": first_end, "type": "narration", "speaker_id": None, "confidence": {"score": 0.9, "band": "high", "reasons": ["machine"]}, "provenance": {"source": "fake"}},
        {"span_id": "alice", "chapter_index": 1, "source_start": alice_start, "source_end": split, "type": "dialogue", "speaker_id": None, "confidence": {"score": 0.5, "band": "medium", "reasons": []}, "provenance": {"source": "fake", "quote_id": "q1"}},
        {"span_id": "bob", "chapter_index": 2, "source_start": split, "source_end": len(text), "type": "narration", "speaker_id": None, "confidence": {"score": 0.1, "band": "low", "reasons": []}, "provenance": {"source": "fake"}},
    ]
    artifact = with_canonical_artifact_hash({
        "schema_version": 1,
        "artifact": "speaker-analysis",
        "revision": 3,
        "source_pdf_sha256": manifest["source_pdf_sha256"],
        "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapter_plan_sha256": persisted["chapter_plan_sha256"],
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "characters": [],
        "spans": spans,
        "warnings": [f"review warning {index}" for index in range(55)],
    })
    path = workspace.conversion_path(manifest["conversion_id"]) / "speaker-analysis.json"
    workspace.persist_speaker_analysis(manifest["conversion_id"], artifact)
    return workspace, workspace.read_job(manifest["conversion_id"]), text, artifact, path


def test_generation_performance_mode_defaults_forwards_and_stays_out_of_tts() -> None:
    root = Path("tests") / f".pytest-phase6-performance-{uuid.uuid4().hex}"; root.mkdir()
    launches: list[tuple[object, ...]] = []
    try:
        launcher = lambda *args: launches.append(args) or type("P", (), {"poll": lambda self: 1})()
        _, client, headers, _ = _app(root, 19880, worker_launcher=launcher)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        first = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert first.status_code == 200 and launches[-1][2] == "background"
        first_tts = workspace.read_job(conversion_id)["tts"]
        assert "performance_mode" not in first_tts and "performance_mode" not in first_tts["settings"]
        workspace.update_generation(conversion_id, status="cancelled", stage="cancelled", worker=None)
        resumed = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, "performance_mode": "maximum_speed"}, headers=headers)
        assert resumed.status_code == 200 and launches[-1][2] == "maximum_speed"
        launch_count = len(launches)
        invalid = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, "performance_mode": "turbo"}, headers=headers)
        assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "INVALID_INPUT"
        assert len(launches) == launch_count
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stale_runtime_generation_cancel_persists_terminal_state() -> None:
    root = Path("tests") / f".pytest-phase6-stale-cancel-{uuid.uuid4().hex}"; root.mkdir()
    try:
        app, client, headers, _ = _app(root, 19891)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        planned = workspace.read_job(conversion_id)
        stale = workspace.update_generation(
            conversion_id,
            status="synthesizing",
            stage="synthesis",
            worker={"pid": 4_000_000_000, "started_at": planned["created_at"], "updated_at": planned["updated_at"]},
        )
        app.state.phase1.worker_process = None

        cancelled = client.post("/api/generation/cancel", json={"conversion_id": conversion_id}, headers=headers)
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json() == {"conversion_id": conversion_id, "status": "cancelled", "cancel_requested": False}

        persisted = workspace.read_job(conversion_id)
        assert persisted["status"] == "cancelled" and persisted["stage"] == "cancelled"
        assert persisted["worker"] is None and persisted["last_safe_error"] == "cancelled"
        for key in ("tts", "total_chunks", "completed_chunks", "progress", "output"):
            assert persisted[key] == stale[key]
        assert not workspace.cancellation_requested(conversion_id)

        current = client.get("/api/status", headers=headers)
        assert current.status_code == 200
        assert current.json()["state"] == "cancelled" and current.json()["job"]["status"] == "cancelled"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_auth_validation_and_no_store() -> None:
    root = Path("tests") / f".pytest-phase6-preview-{uuid.uuid4().hex}"
    try:
        _, client, headers, preview = _app(root, 19881)
        client.cookies.clear()
        assert client.get("/api/voice-preview/af_heart", headers=headers).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.get("/api/voice-preview/not-a-voice", headers=headers).status_code == 404
        write_pcm_wav(preview / "sample-kokoro-af_heart.wav", b"\0\0" * 240, 24000, overwrite=True)
        response = client.get("/api/voice-preview/af_heart", headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert client.get("/favicon.ico", headers=headers).status_code == 204
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_legacy_download_is_reused_by_get_prepare_and_catalog() -> None:
    root = Path("tests") / f".pytest-phase6-preview-legacy-{uuid.uuid4().hex}"
    calls: list[tuple[str, Path]] = []

    def generate(voice: str, target: Path) -> None:
        calls.append((voice, target))
        raise AssertionError("valid legacy preview must not be regenerated")

    try:
        _, client, headers, preview = _app(root, 19887, preview_generator=generate)
        legacy = preview / "20260808T231337Z-kokoro-bf_emma.wav"
        write_pcm_wav(legacy, b"\0\0" * 240, 24000, overwrite=True)

        catalog = client.get("/api/voices", headers=headers)
        assert catalog.status_code == 200
        emma = next(entry for entry in catalog.json()["voices"] if entry["id"] == "bf_emma")
        assert emma["preview_available"] is True

        prepared = client.post("/api/voice-preview/bf_emma/prepare", headers=headers)
        assert prepared.status_code == 200 and prepared.json() == {"voice": "bf_emma", "status": "ready"}
        response = client.get("/api/voice-preview/bf_emma", headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert response.content[:4] == b"RIFF" and not calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_get_lazily_generates_once_and_caches() -> None:
    root = Path("tests") / f".pytest-phase6-preview-get-{uuid.uuid4().hex}"
    calls: list[tuple[str, Path]] = []

    def generate(voice: str, target: Path) -> None:
        calls.append((voice, target))
        write_pcm_wav(target, b"\0\0" * 240, 24000, overwrite=True)

    try:
        _, client, headers, _ = _app(root, 19884, preview_generator=generate)
        first = client.get("/api/voice-preview/af_heart", headers=headers)
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-store"
        assert first.headers["content-type"].startswith("audio/wav")
        assert first.content[:4] == b"RIFF"
        assert len(calls) == 1 and calls[0][0] == "af_heart" and calls[0][1].is_file()

        cached = client.get("/api/voice-preview/af_heart", headers=headers)
        assert cached.status_code == 200 and cached.content == first.content and len(calls) == 1
        assert client.get("/api/voice-preview/not-a-voice", headers=headers).status_code == 404
        assert len(calls) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_get_generation_failure_is_bounded() -> None:
    root = Path("tests") / f".pytest-phase6-preview-get-failed-{uuid.uuid4().hex}"

    def generate(_voice: str, _target: Path) -> None:
        raise RuntimeError("preview generation secret")

    try:
        _, client, headers, _ = _app(root, 19886, preview_generator=generate)
        response = client.get("/api/voice-preview/af_heart", headers=headers)
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "VOICE_PREVIEW_FAILED", "message": "voice preview generation failed"}}
        assert "preview generation secret" not in response.text and str(root) not in response.text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_prepare_generates_once_and_requires_exact_origin() -> None:
    root = Path("tests") / f".pytest-phase6-preview-prepare-{uuid.uuid4().hex}"
    calls: list[tuple[str, Path]] = []

    def generate(voice: str, target: Path) -> None:
        calls.append((voice, target))
        write_pcm_wav(target, b"\0\0" * 240, 24000, overwrite=True)

    try:
        _, client, headers, _ = _app(root, 19882, preview_generator=generate)
        client.cookies.clear()
        assert client.post("/api/voice-preview/af_heart/prepare", headers=headers).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.post("/api/voice-preview/af_heart/prepare", headers={**headers, "Origin": "http://localhost:9"}).status_code == 403
        assert client.post("/api/voice-preview/not-a-voice/prepare", headers=headers).status_code == 404
        prepared = client.post("/api/voice-preview/af_heart/prepare", headers=headers)
        assert prepared.status_code == 200 and prepared.json() == {"voice": "af_heart", "status": "ready"}
        assert len(calls) == 1
        assert client.get("/api/voice-preview/af_heart", headers=headers).status_code == 200
        cached = client.post("/api/voice-preview/af_heart/prepare", headers=headers)
        assert cached.status_code == 200 and len(calls) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_preview_prepare_rejects_malformed_generator_output_without_details() -> None:
    root = Path("tests") / f".pytest-phase6-preview-failed-{uuid.uuid4().hex}"

    def generate(_voice: str, target: Path) -> None:
        target.write_bytes(b"malformed preview")

    try:
        _, client, headers, _ = _app(root, 19883, preview_generator=generate)
        response = client.post("/api/voice-preview/af_heart/prepare", headers=headers)
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "VOICE_PREVIEW_FAILED", "message": "voice preview generation failed"}}
        assert "malformed" not in response.text and str(root) not in response.text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cancelled_generation_labels_resume_without_resetting_chunks() -> None:
    root = Path("tests") / f".pytest-phase6-rename-{uuid.uuid4().hex}"
    try:
        app, client, headers, _ = _app(root, 19885)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence. Three sentence. Four sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        before = workspace.read_job(conversion_id)
        assert before["completed_chunks"]
        first_record = before["completed_chunks"][0]
        chunk_path = workspace.conversion_path(conversion_id) / first_record["relative_path"]
        chunk_bytes = chunk_path.read_bytes()
        chunk_mtime = chunk_path.stat().st_mtime_ns
        preserved = {key: before[key] for key in ("tts", "total_chunks", "completed_chunks", "progress", "output")}
        old_plan_hash = before["chapter_plan_sha256"]
        cancelled = workspace.update_generation(conversion_id, status="cancelled", stage="cancelled", worker=None)
        assert cancelled["status"] == "cancelled"
        original_plan = workspace.load_chapter_plan(conversion_id)
        titles = ["Renamed chapter"]
        response = client.post("/api/chapter-plan/titles", json={"titles": titles}, headers=headers)
        assert response.status_code == 200
        renamed = response.json()
        assert renamed["job"]["status"] == "planned" and renamed["job"]["stage"] == "chapter_review"
        assert [chapter["title"] for chapter in renamed["chapter_plan"]["chapters"]] == titles
        assert [(chapter["start_offset"], chapter["end_offset"]) for chapter in renamed["chapter_plan"]["chapters"]] == [(chapter["start_offset"], chapter["end_offset"]) for chapter in original_plan["chapters"]]
        after = workspace.read_job(conversion_id)
        assert after["chapter_plan_sha256"] != old_plan_hash
        assert all(after[key] == preserved[key] for key in ("tts", "total_chunks", "completed_chunks", "progress", "output"))
        assert chunk_path.read_bytes() == chunk_bytes and chunk_path.stat().st_mtime_ns == chunk_mtime
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        assert chunk_path.read_bytes() == chunk_bytes and chunk_path.stat().st_mtime_ns == chunk_mtime
        app.state.worker_process = None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generation_start_reports_starting_until_worker_claims_job() -> None:
    root = Path("tests") / f".pytest-phase6-starting-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19887, worker_poll=None)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        started = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert started.status_code == 200 and started.json()["status"] == "starting"
        current = client.get("/api/status", headers=headers)
        assert current.status_code == 200 and current.json()["state"] == "starting"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cancelled_generation_resume_reports_starting_until_worker_claims_job() -> None:
    root = Path("tests") / f".pytest-phase6-resume-starting-{uuid.uuid4().hex}"
    try:
        app, client, headers, _ = _app(root, 19888, worker_poll=None)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        analyzed = client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers)
        assert analyzed.status_code == 200, analyzed.text
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        first_start = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert first_start.status_code == 200
        app.state.phase1.worker_process = None
        workspace.update_generation(conversion_id, status="cancelled", stage="cancelled", worker=None)
        resumed = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert resumed.status_code == 200 and resumed.json()["status"] == "starting", resumed.text
        current = client.get("/api/status", headers=headers)
        assert current.status_code == 200 and current.json()["state"] == "starting"
        cancelled = client.post("/api/generation/cancel", json={"conversion_id": conversion_id}, headers=headers)
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelling"
        app.state.phase1.worker_process = None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generation_title_rename_rejects_live_worker_and_final_output() -> None:
    root = Path("tests") / f".pytest-phase6-rename-guards-{uuid.uuid4().hex}"
    try:
        app, client, headers, _ = _app(root, 19886)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        job = workspace.read_job(conversion_id)
        job.update(status="synthesizing", stage="synthesis", worker={"pid": os.getpid(), "started_at": job["created_at"], "updated_at": job["updated_at"]})
        atomic_write_json(workspace.job_path(conversion_id), job)
        live = client.post("/api/chapter-plan/titles", json={"titles": ["Blocked"]}, headers=headers)
        assert live.status_code == 409 and live.json()["error"]["code"] == "ACTIVE_WORKER"
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        synthesis_complete = workspace.read_job(conversion_id)
        assert synthesis_complete["status"] == "completed" and synthesis_complete["stage"] == "synthesis_complete"
        output = (root / "published" / "book.m4b").resolve()
        output.parent.mkdir()
        output.write_bytes(b"verified")
        workspace.update_generation(conversion_id, status="completed", stage="completed", output={"filename": output.name, "path": str(output), "size_bytes": output.stat().st_size, "duration_seconds": 1.0, "chapter_count": 1, "codec": "aac", "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        final = client.post("/api/chapter-plan/titles", json={"titles": ["Blocked final"]}, headers=headers)
        assert final.status_code == 409 and final.json()["error"]["code"] == "NOT_READY"
        app.state.worker_process = None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generation_summary_maps_multi_chapter_progress() -> None:
    root = Path("tests") / f".pytest-phase6-progress-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19882)
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence. Three sentence. Four sentence. Five sentence. Six sentence."])
        headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "custom", "count": 2}, headers=headers).status_code == 200
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        job = workspace.read_job(conversion_id)
        job.update(status="synthesizing", stage="synthesis", worker={"pid": os.getpid(), "started_at": job["created_at"], "updated_at": job["updated_at"]})
        atomic_write_json(workspace.job_path(conversion_id), job)
        assert client.post("/api/generation/cancel", json={"conversion_id": conversion_id}, headers=headers).status_code == 200
        assert client.post("/api/generation/cancel", json={"conversion_id": "wrong"}, headers=headers).status_code == 409
        assert client.post("/api/generation/cancel", json={"conversion_id": conversion_id, "extra": True}, headers=headers).status_code == 422
        workspace.clear_cancel_request(conversion_id)
        response = client.get("/api/status", headers=headers)
        summary = response.json()["generation_summary"]
        assert summary["total_chapters"] == 2 and summary["completed_chunks"] == 0 and summary["current_chapter"] == 1
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        summary = client.get("/api/status", headers=headers).json()["generation_summary"]
        assert summary["completed_chunks"] == summary["total_chunks"] == 2
        assert summary["completed_chapters"] == 2 and summary["current_chapter"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_output_open_uses_only_validated_manifest_targets() -> None:
    root = Path("tests") / f".pytest-phase6-open-{uuid.uuid4().hex}"
    opened: list[Path] = []
    try:
        _, client, headers, _ = _app(root, 19883, opener=lambda target: opened.append(target))
        client.cookies.clear()
        assert client.post("/api/output/open", json={"target": "audiobook"}, headers=headers).status_code == 403
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.post("/api/output/open", json={"target": "audiobook", "path": "C:\\secret"}, headers=headers).status_code == 422
        assert client.post("/api/output/open", json={"target": "audiobook"}, headers=headers).status_code == 409
        pdf = make_pdf(root / "book.pdf", ["One sentence."])
        headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        client.post("/api/analyze", content=pdf.read_bytes(), headers=headers)
        client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers)
        client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        output = (root / "published" / "book.m4b").resolve()
        output.parent.mkdir()
        output.write_bytes(b"verified")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        workspace.update_generation(conversion_id, status="completed", stage="completed", output={"filename": output.name, "path": str(output), "size_bytes": output.stat().st_size, "duration_seconds": 1.0, "chapter_count": 1, "codec": "aac", "sha256": digest})
        assert client.post("/api/output/open", json={"target": "audiobook"}, headers=headers).json() == {"opened": "audiobook"}
        assert client.post("/api/output/open", json={"target": "folder"}, headers=headers).json() == {"opened": "folder"}
        assert client.post("/api/output/open", json={"target": "folder"}, headers={**headers, "Origin": "http://localhost:9"}).status_code == 403
        assert opened == [output, output.parent]
        def fail_opener(_target):
            raise OSError("private path details")

        failing = create_app(
            port=19884,
            session_token="phase6-token",
            instance_file=root / "instance-failing.json",
            data_root=root / "data",
            path_opener=fail_opener,
        )
        with TestClient(failing, base_url="http://127.0.0.1:19884") as failing_client:
            failing_headers = {"Origin": "http://127.0.0.1:19884"}
            assert failing_client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=failing_headers).status_code == 200
            failed = failing_client.post("/api/output/open", json={"target": "audiobook"}, headers=failing_headers)
            assert failed.status_code == 503 and str(output) not in failed.text
        output.write_bytes(b"tampered")
        unavailable = client.post("/api/output/open", json={"target": "audiobook"}, headers=headers)
        assert unavailable.status_code == 409 and str(output) not in unavailable.text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _range_plan_text() -> str:
    return " ".join(f"Sentence {index} has enough words for a safe boundary." for index in range(1, 16))


def test_generation_range_persists_and_invalid_requests_are_safe() -> None:
    root = Path("tests") / f".pytest-phase6-range-api-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19889)
        pdf = make_pdf(root / "book.pdf", [_range_plan_text()])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "custom", "count": 3}, headers=headers).status_code == 200
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id

        full = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert full.status_code == 200, full.text
        assert not {"chapter_start", "chapter_end"}.intersection(workspace.read_job(conversion_id)["tts"]["settings"])

        invalid = [
            ({"chapter_start": True, "chapter_end": 2}, "INVALID_CHAPTER_RANGE"),
            ({"chapter_start": 0, "chapter_end": 2}, "INVALID_CHAPTER_RANGE"),
            ({"chapter_start": 2, "chapter_end": 1}, "INVALID_CHAPTER_RANGE"),
            ({"chapter_start": 1, "chapter_end": 4}, "INVALID_CHAPTER_RANGE"),
            ({"chapter_start": 1, "chapter_end": 2, "unexpected": 1}, "INVALID_INPUT"),
        ]
        for extra, code in invalid:
            response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, **extra}, headers=headers)
            assert response.status_code == 422 and response.json()["error"]["code"] == code, response.text

        selected = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, "chapter_start": 2, "chapter_end": 3}, headers=headers)
        assert selected.status_code == 200, selected.text
        settings = workspace.read_job(conversion_id)["tts"]["settings"]
        assert settings["chapter_start"] == 2 and settings["chapter_end"] == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generation_range_worker_and_summary_are_relative_to_selected_plan() -> None:
    root = Path("tests") / f".pytest-phase6-range-summary-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19890)
        pdf = make_pdf(root / "book.pdf", [_range_plan_text()])
        upload_headers = {**headers, "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=upload_headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "custom", "count": 3}, headers=headers).status_code == 200
        started = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, "chapter_start": 2, "chapter_end": 3}, headers=headers)
        assert started.status_code == 200, started.text
        workspace = Workspace(root / "data")
        conversion_id = workspace.inspect_startup().conversion_id
        assert conversion_id
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice(), full_pipeline=False)
        job = workspace.read_job(conversion_id)
        assert [record["chapter_index"] for record in job["completed_chunks"]] == [1, 2]
        summary = client.get("/api/status", headers=headers).json()["generation_summary"]
        assert summary["total_chapters"] == 2 and summary["current_chapter"] == 2
        assert summary["completed_chapters"] == 2 and summary["completed_chunks"] == summary["total_chunks"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_requires_auth_origin_strict_body_and_injected_analyzer() -> None:
    root = Path("tests") / f".pytest-phase6-voice-input-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19891, voice_analyzer=object())
        client.cookies.clear()
        assert client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers).status_code == 403
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.post("/api/voice-analysis", json={"mode": "interactive", "unexpected": True}, headers=headers).status_code == 422
        assert client.post("/api/voice-analysis", json={"mode": "interactive"}, headers={**headers, "Origin": "http://localhost:9"}).status_code == 403
        unavailable = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert unavailable.status_code == 503 and unavailable.json()["error"]["code"] == "ANALYZER_UNAVAILABLE"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_start_status_projection_and_revision() -> None:
    root = Path("tests") / f".pytest-phase6-voice-status-{uuid.uuid4().hex}"
    try:
        app, client, headers, _ = _app(root, 19892, voice_analyzer=DeterministicFakeAnalyzer())
        _prepare_voice_analysis(root, client, headers)
        started = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert started.status_code == 200
        first = started.json()
        assert first["revision"] == 1 and first["status"] == "queued"
        status = _wait_voice_terminal(app, client, headers)
        assert status["status"] == "completed"
        assert status["analysis_id"] == first["analysis_id"]
        assert "spans" not in status and status["canonical_artifact_sha256"]
        second = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert second.status_code == 200 and second.json()["revision"] == 2
        _wait_voice_terminal(app, client, headers)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_cancellation_conflicts_with_generation_and_is_identity_bound() -> None:
    root = Path("tests") / f".pytest-phase6-voice-cancel-{uuid.uuid4().hex}"
    analyzer = _BlockingVoiceAnalyzer()
    try:
        app, client, headers, _ = _app(root, 19893, voice_analyzer=analyzer)
        _prepare_voice_analysis(root, client, headers)
        started = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers).json()
        assert analyzer.started.wait(2)
        generation = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert generation.status_code == 409 and generation.json()["error"]["code"] == "ANALYSIS_CONFLICT"
        stale = client.post("/api/voice-analysis/cancel", json={"analysis_id": started["analysis_id"], "revision": 2}, headers=headers)
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "ANALYSIS_CONFLICT"
        cancelled = client.post("/api/voice-analysis/cancel", json={"analysis_id": started["analysis_id"], "revision": 1}, headers=headers)
        assert cancelled.status_code == 200 and cancelled.json()["cancel_requested"] is True
        pending_status = client.get("/api/voice-analysis/status", headers=headers)
        assert pending_status.status_code == 200 and pending_status.json()["cancel_requested"] is True
        analyzer.release.set()
        status = _wait_voice_terminal(app, client, headers)
        assert status["status"] == "cancelled"
        terminal = client.post("/api/voice-analysis/cancel", json={"analysis_id": started["analysis_id"], "revision": 1}, headers=headers)
        assert terminal.status_code == 409 and terminal.json()["error"]["code"] == "NOT_ANALYZING"
    finally:
        analyzer.release.set()
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_allows_valid_stale_queued_and_running_statuses() -> None:
    root = Path("tests") / f".pytest-phase6-voice-stale-status-{uuid.uuid4().hex}"
    try:
        app, client, headers, _ = _app(root, 19894, voice_analyzer=DeterministicFakeAnalyzer())
        _prepare_voice_analysis(root, client, headers)
        first = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers).json()
        completed = _wait_voice_terminal(app, client, headers)
        workspace = Workspace(root / "data")
        conversion_id = completed["conversion_id"]
        for stale_status, stale_stage in (("queued", "queued"), ("running", "preparing")):
            durable = workspace.load_voice_analysis_status(conversion_id)
            stale = {key: value for key, value in durable.items() if key != "canonical_artifact_sha256"}
            stale.update({
                "status": stale_status,
                "stage": stale_stage,
                "progress": {"completed": 0, "total": 0},
                "cancel_requested": False,
                "warnings": [],
                "error": None,
                "finished_at": None,
            })
            workspace.persist_voice_analysis_status(conversion_id, with_canonical_artifact_hash(stale))
            restarted = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
            assert restarted.status_code == 200, restarted.text
            next_run = restarted.json()
            assert next_run["revision"] == first["revision"] + 1 and next_run["analysis_id"] != first["analysis_id"]
            _wait_voice_terminal(app, client, headers)
            first = next_run
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_failure_status_redacts_analyzer_exception() -> None:
    root = Path("tests") / f".pytest-phase6-voice-failure-{uuid.uuid4().hex}"
    secret = "secret-book-text-should-never-leak"
    try:
        app, client, headers, _ = _app(root, 19895, voice_analyzer=_FailingVoiceAnalyzer())
        _prepare_voice_analysis(root, client, headers)
        started = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert started.status_code == 200
        status = _wait_voice_terminal(app, client, headers)
        assert status["status"] == "failed"
        assert secret not in client.get("/api/voice-analysis/status", headers=headers).text
        assert status["error"] == {"code": "ANALYZER_FAILED", "message": "voice analysis failed"}
        durable = Workspace(root / "data").load_voice_analysis_status(status["conversion_id"])
        assert durable["error"] == status["error"] and secret not in str(durable)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_rejects_live_single_voice_generation() -> None:
    root = Path("tests") / f".pytest-phase6-voice-generation-conflict-{uuid.uuid4().hex}"
    process = type("LiveProcess", (), {"poll": lambda self: None})()
    try:
        app, client, headers, _ = _app(root, 19896, worker_launcher=lambda *_: process, voice_analyzer=DeterministicFakeAnalyzer())
        _prepare_voice_analysis(root, client, headers)
        generation = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert generation.status_code == 200 and generation.json()["status"] == "starting"
        blocked = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "ACTIVE_WORKER"
    finally:
        app.state.phase1.worker_process = None
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_api_auth_origin_and_cancel_body_boundaries() -> None:
    root = Path("tests") / f".pytest-phase6-voice-boundaries-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19897, voice_analyzer=DeterministicFakeAnalyzer())
        client.cookies.clear()
        assert client.get("/api/voice-analysis/status", headers=headers).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.get("/api/voice-analysis/status").status_code == 404
        client.cookies.clear()
        assert client.post("/api/voice-analysis/cancel", json={"analysis_id": "x", "revision": 1}, headers=headers).status_code == 403
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        invalid_bodies = [
            {"analysis_id": "x", "revision": 1, "extra": True},
            {"analysis_id": "x", "revision": True},
            {"analysis_id": "x", "revision": 0},
            {"analysis_id": "x", "revision": -1},
        ]
        for body in invalid_bodies:
            response = client.post("/api/voice-analysis/cancel", json=body, headers=headers)
            assert response.status_code == 422, response.text
        assert client.post("/api/voice-analysis/cancel", json={"analysis_id": "x", "revision": 1}, headers={**headers, "Origin": "http://localhost:9"}).status_code == 403
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_speaker_analysis_review_is_bounded_authenticated_and_filterable() -> None:
    root = Path("tests") / f".pytest-phase6-speaker-review-{uuid.uuid4().hex}"
    try:
        _, client, headers, _ = _app(root, 19898)
        workspace, manifest, text, artifact, path = _prepare_speaker_review(root)
        client.cookies.clear()
        assert client.get("/api/speaker-analysis", headers=headers).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200

        response = client.get("/api/speaker-analysis")
        assert response.status_code == 200
        body = response.json()
        assert body["conversion_id"] == manifest["conversion_id"]
        assert body["revision"] == 3 and body["character_count"] == 0
        assert body["warning_count"] == 55 and len(body["warnings"]) == 50 and body["warnings_truncated"]
        assert body["total"] == 3 and body["offset"] == 0 and body["limit"] == 50 and not body["has_more"]
        assert "characters" not in body and text not in response.text
        assert body["spans"][0]["excerpt"] == text[:240]
        assert body["spans"][0]["excerpt_truncated"] is True
        assert body["spans"][0]["confidence"]["band"] == "high"
        alice_start = text.index("Ålice")
        split = text.index("Bob")
        assert body["spans"][1]["excerpt"] == text[alice_start:split] == "Ålice says.\n"

        page = client.get("/api/speaker-analysis?offset=1&limit=1")
        assert page.status_code == 200 and page.json()["spans"][0]["span_id"] == "alice" and page.json()["has_more"]
        assert client.get("/api/speaker-analysis?chapter=2").json()["spans"][0]["span_id"] == "bob"
        assert client.get("/api/speaker-analysis?confidence=low").json()["spans"][0]["span_id"] == "bob"
        assert client.get("/api/speaker-analysis?limit=200").status_code == 200
        for query in ("unknown=1", "chapter=1&chapter=1", "chapter=true", "offset=-1", "limit=0", "limit=201", "confidence=highx"):
            invalid = client.get(f"/api/speaker-analysis?{query}")
            assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "INVALID_INPUT"

        path.write_text("{not-json", encoding="utf-8")
        unavailable = client.get("/api/speaker-analysis")
        assert unavailable.status_code == 404 and unavailable.json()["error"]["code"] == "SPEAKER_ANALYSIS_UNAVAILABLE"
        assert str(path) not in unavailable.text and text not in unavailable.text
        stale = with_canonical_artifact_hash({**artifact, "source_pdf_sha256": "0" * 64})
        atomic_write_json(path, stale)
        stale_response = client.get("/api/speaker-analysis")
        assert stale_response.status_code == 404 and stale_response.json()["error"]["code"] == "SPEAKER_ANALYSIS_UNAVAILABLE"
        assert str(path) not in stale_response.text and text not in stale_response.text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_voice_registry_auth_and_preview_projection() -> None:
    root = Path("tests") / f".pytest-phase6-voices-{uuid.uuid4().hex}"; root.mkdir()
    try:
        _, client, headers, preview = _app(root, 19899)
        client.cookies.clear()
        assert client.get("/api/voices").status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        catalog = client.get("/api/voices")
        assert catalog.status_code == 200
        payload = catalog.json()
        expected_ids = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ]
        assert payload["registry_revision"] == payload["revision"] and [entry["id"] for entry in payload["voices"]] == expected_ids
        assert all(("American English" in entry["description"] or "British English" in entry["description"]) and ("female" in entry["description"] or "male" in entry["description"]) for entry in payload["voices"])
        assert all(entry["preview_available"] is False for entry in payload["voices"])
        write_pcm_wav(preview / "sample-kokoro-af_heart.wav", b"\0\0" * 24, 24000, overwrite=True)
        updated = client.get("/api/voices").json()
        assert updated["voices"][0]["preview_available"] is True
        assert client.get("/api/voice-preview/af_heart", headers={**headers, "Origin": "http://localhost:9"}).status_code == 200
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_shaping_capability_is_path_free_and_rejects_unsupported_preview_settings(monkeypatch) -> None:
    root = Path("tests") / f".pytest-phase6-shaping-capability-{uuid.uuid4().hex}"; root.mkdir()
    calls: list[tuple] = []

    def generate(*args) -> None:
        calls.append(args)
        raise AssertionError("unsupported shaping must not reach the generator")

    capability = ShapingCapability("C:\\secret\\ffmpeg.exe", False, "ffmpeg-test", "", "fingerprint-test", False)
    monkeypatch.setattr(app_module, "shaping_capability", lambda: capability)
    monkeypatch.setattr(app_module, "shaping_fingerprint", lambda: capability.fingerprint)
    try:
        _, client, headers, _ = _app(root, 19898, preview_generator=generate)
        catalog = client.get("/api/voices", headers=headers)
        assert catalog.status_code == 200
        projected = catalog.json()["voice_shaping"]
        assert projected["pitch_available"] is False and projected["tone_available"] is False
        assert projected["preview_settings_available"] is True
        assert projected["preview_settings_implementation"] == "voice-preview-settings-v1"
        assert "ffmpeg" not in projected and "C:\\secret" not in catalog.text
        script = Path("src/pdf_audiobook/static/app.js").read_text(encoding="utf-8")
        css = Path("src/pdf_audiobook/static/styles.css").read_text(encoding="utf-8")
        assert "settingsAwarePreviewAvailable" in script
        assert "Number(settings.speed ?? 1) !== 1" in script
        assert "Restart the local backend to preview voice shaping." in script
        assert 'button.textContent = "Cancel preview"' in script
        assert 'audio.addEventListener("playing", markPlaying)' in script
        assert 'audio.addEventListener("waiting", markBuffering)' in script
        assert 'audio.addEventListener("stalled", markBuffering)' in script
        assert 'button.textContent = previewLabel(button, true)' in script
        assert 'if (activePreviewButton === preview) stopPreview(previewStatus);' in script
        assert 'button.setAttribute("aria-busy", "true")' in script
        assert "previewAudioError" in script
        assert ".preview-loading,.preview-buffering" in css
        assert "prefers-reduced-motion:reduce" in css
        assert 'previewControls.className = "cast-preview cast-audition-row"' in script
        assert "previewControls.append(preview, previewStatus); voiceField.append(previewControls)" in script
        assert "article.append(fields)" in script
        pitch = client.get("/api/voice-preview/af_heart?pitch_semitones=1", headers=headers)
        tone = client.get("/api/voice-preview/af_heart?tone_preset=warm", headers=headers)
        assert pitch.status_code == 422 and pitch.json()["error"]["code"] == "PITCH_UNAVAILABLE"
        assert tone.status_code == 422 and tone.json()["error"]["code"] == "TONE_UNAVAILABLE"
        assert not calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_voice_analysis_range_validation_projection_and_full_mode_compatibility() -> None:
    root = Path("tests") / f".pytest-phase6-analysis-range-{uuid.uuid4().hex}"; root.mkdir()
    try:
        analyzer = DeterministicFakeAnalyzer()
        app, client, headers, _ = _app(root, 19897, voice_analyzer=analyzer)
        _prepare_speaker_review(root)
        partial = client.post("/api/voice-analysis", json={"mode": "interactive", "chapter_start": 2}, headers=headers)
        assert partial.status_code == 422
        boolean = client.post("/api/voice-analysis", json={"mode": "interactive", "chapter_start": True, "chapter_end": 2}, headers=headers)
        assert boolean.status_code == 422
        out_of_bounds = client.post("/api/voice-analysis", json={"mode": "interactive", "chapter_start": 1, "chapter_end": 3}, headers=headers)
        assert out_of_bounds.status_code == 422
        selected = client.post("/api/voice-analysis", json={"mode": "interactive", "chapter_start": 2, "chapter_end": 2}, headers=headers)
        assert selected.status_code == 200
        status = _wait_voice_terminal(app, client, headers)
        assert status["chapter_start"] == 2 and status["chapter_end"] == 2
        full = client.post("/api/voice-analysis", json={"mode": "interactive"}, headers=headers)
        assert full.status_code == 200
        status = _wait_voice_terminal(app, client, headers)
        assert status["chapter_start"] == 1 and status["chapter_end"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_voice_plan_pagination_mutations_and_revision_conflicts() -> None:
    root = Path("tests") / f".pytest-phase6-plan-api-{uuid.uuid4().hex}"; root.mkdir()
    try:
        _, client, headers, _ = _app(root, 19900)
        _, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        page = client.get("/api/voice-plan?limit=1", headers=headers)
        assert page.status_code == 200 and page.json()["has_more"] and page.json()["spans"]
        assert client.put("/api/voice-plan", json={"expected_revision": 1, "cast_id": "narrator"}, headers=headers).status_code == 422
        renamed = client.put("/api/voice-plan", json={"expected_revision": 1, "cast_id": "narrator", "display_label": "Storyteller"}, headers=headers)
        assert renamed.status_code == 200 and renamed.json()["revision"] == 2
        assert client.put("/api/voice-plan", json={"expected_revision": 1, "cast_id": "narrator", "speed": 1.1}, headers=headers).json()["error"]["code"] == "PLAN_CONFLICT"
        aliases = client.get("/api/voice-plan", headers=headers).json()["aliases"]
        merge_body = {"expected_revision": 2, "target_character_id": "character-bob", "alias_ids": [aliases[0]["alias_id"]]}
        assert client.post("/api/voice-plan/aliases/merge", json={**merge_body, "extra": True}, headers=headers).status_code == 422
        merged = client.post("/api/voice-plan/aliases/merge", json=merge_body, headers=headers); assert merged.status_code == 200
        split_alias = aliases[1]["alias_id"]
        split = client.post("/api/voice-plan/aliases/split", json={"expected_revision": 3, "alias_ids": [split_alias], "new_character_id": "character-new", "display_label": "New Voice", "voice_id": "af_bella"}, headers=headers)
        assert split.status_code == 200
        assert client.post("/api/voice-plan/spans/override", json={"expected_revision": 4, "span_id": "long", "kind": "speaker", "to": "narrator"}, headers=headers).status_code == 422
        override = client.post("/api/voice-plan/spans/override", json={"expected_revision": 4, "span_id": "long", "kind": "speaker", "to": "narrator", "reason": "manual review"}, headers=headers)
        assert override.status_code == 200
        stale_approval = client.post("/api/voice-plan/approve", json={"expected_revision": 4, "accept_narrator_fallback": True}, headers=headers)
        assert stale_approval.status_code == 409 and stale_approval.json()["error"]["code"] == "PLAN_CONFLICT"
        approved_again = client.post("/api/voice-plan/approve", json={"expected_revision": 5, "accept_narrator_fallback": True}, headers=headers)
        assert approved_again.status_code == 200 and approved_again.json()["approval"]["state"] == "approved"
        retry = client.post("/api/voice-plan/approve", json={"expected_revision": 5, "accept_narrator_fallback": True}, headers=headers)
        assert retry.status_code == 200
        assert retry.json()["revision"] == approved_again.json()["revision"]
        assert retry.json()["canonical_artifact_sha256"] == approved_again.json()["canonical_artifact_sha256"]
        assert retry.json()["approval"] == approved_again.json()["approval"]
        assert conversion_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_cast_merge_remove_endpoints_are_strict_and_persisted() -> None:
    root = Path("tests") / f".pytest-phase6-cast-api-{uuid.uuid4().hex}"; root.mkdir()
    try:
        _, client, headers, _ = _app(root, 19902)
        workspace, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        revision = approved["revision"]
        assert client.post("/api/voice-plan/cast/merge", json={"expected_revision": revision, "source_cast_id": "character-bob", "target_cast_id": "character-alice", "extra": True}, headers=headers).status_code == 422
        merged = client.post("/api/voice-plan/cast/merge", json={"expected_revision": revision, "source_cast_id": "character-bob", "target_cast_id": "character-alice"}, headers=headers)
        assert merged.status_code == 200 and merged.json()["revision"] == revision + 1
        plan = workspace.load_voice_plan(conversion_id)
        assert {entry["cast_id"] for entry in plan["cast"]} == {"narrator", "character-alice"}
        assert all(alias["character_id"] == "character-alice" for alias in plan["aliases"])
        assert any((span.get("override") or {}).get("reason") == "cast_merged" for chapter in plan["chapters"] for span in chapter["spans"])
        removed = client.post("/api/voice-plan/cast/remove", json={"expected_revision": revision + 1, "cast_id": "character-alice"}, headers=headers)
        assert removed.status_code == 200 and removed.json()["revision"] == revision + 2
        plan = workspace.load_voice_plan(conversion_id)
        assert [entry["cast_id"] for entry in plan["cast"]] == ["narrator"] and not plan["aliases"]
        assert all(span["speaker_id"] == "narrator" for chapter in plan["chapters"] for span in chapter["spans"])
        assert any((span.get("override") or {}).get("reason") == "cast_removed" for chapter in plan["chapters"] for span in chapter["spans"])
        narrator = client.post("/api/voice-plan/cast/remove", json={"expected_revision": revision + 2, "cast_id": "narrator"}, headers=headers)
        assert narrator.status_code == 422 and narrator.json()["error"]["code"] == "CANNOT_REMOVE_NARRATOR"
        stale = client.post("/api/voice-plan/cast/remove", json={"expected_revision": revision, "cast_id": "narrator"}, headers=headers)
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "PLAN_CONFLICT"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_generation_accepts_approved_unknown_narrator_fallback() -> None:
    root = Path("tests") / f".pytest-phase6-unknown-fallback-{uuid.uuid4().hex}"; root.mkdir()
    try:
        launcher = lambda *_args: type("P", (), {"poll": lambda self: None})()
        _, client, headers, _ = _app(root, 19903, worker_launcher=launcher)
        workspace, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        plan = workspace.load_voice_plan(conversion_id)
        span = plan["chapters"][0]["spans"][0]
        span["type"] = "unknown"
        span["speaker_id"] = "narrator"
        plan["unresolved_policy"] = {"mode": "narrator", "accepted_by_user": True, "accepted_at": "2026-01-01T00:00:00Z"}
        plan = with_canonical_artifact_hash(plan)
        workspace.persist_voice_plan(conversion_id, plan)
        started = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": plan["canonical_artifact_sha256"], "voice_plan_revision": plan["revision"]}, headers=headers)
        assert started.status_code == 200 and started.json()["job"]["mode"] == "interactive_voices"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_generation_preserves_requested_chapter_range() -> None:
    root = Path("tests") / f".pytest-phase6-range-interactive-{uuid.uuid4().hex}"; root.mkdir()
    try:
        launcher = lambda *_args: type("P", (), {"poll": lambda self: None})()
        _, client, headers, _ = _app(root, 19904, worker_launcher=launcher)
        workspace, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        with patch("pdf_audiobook.app.plan_interactive_chunks", wraps=plan_interactive_chunks) as planner:
            started = client.post(
                "/api/generation/start",
                json={
                    "mode": "interactive_voices",
                    "voice_plan_sha256": approved["canonical_artifact_sha256"],
                    "voice_plan_revision": approved["revision"],
                    "chapter_start": 2,
                    "chapter_end": 2,
                },
                headers=headers,
            )
        assert started.status_code == 200, started.text
        assert planner.call_args.args[4] == (2, 2)
        settings = workspace.read_job(conversion_id)["tts"]["settings"]
        assert settings["chapter_start"] == 2 and settings["chapter_end"] == 2
        summary = client.get("/api/status", headers=headers).json()["generation_summary"]
        assert summary["total_chapters"] == 1 and summary["current_chapter"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_generation_is_bounded_by_reviewed_speaker_analysis_range() -> None:
    root = Path("tests") / f".pytest-phase6-reviewed-range-{uuid.uuid4().hex}"; root.mkdir()
    try:
        launcher = lambda *_args: type("P", (), {"poll": lambda self: None})()
        _, client, headers, _ = _app(root, 19896, worker_launcher=launcher)
        workspace, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        artifact = workspace.load_speaker_analysis(conversion_id)
        reviewed = with_canonical_artifact_hash({**artifact, "chapter_start": 2, "chapter_end": 2, "spans": [span for span in artifact["spans"] if span["chapter_index"] == 2]})
        workspace.persist_speaker_analysis(conversion_id, reviewed)
        blocked = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": approved["canonical_artifact_sha256"], "voice_plan_revision": approved["revision"], "chapter_start": 1, "chapter_end": 1}, headers=headers)
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "ANALYSIS_RANGE_CONFLICT"
        assert blocked.json()["error"]["analyzed_chapter_start"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interactive_v5_start_status_cancel_and_launch_failure() -> None:
    root = Path("tests") / f".pytest-phase6-v5-api-{uuid.uuid4().hex}"; root.mkdir()
    try:
        live = type("LiveProcess", (), {"poll": lambda self: None})()
        app, client, headers, _ = _app(root, 19901, worker_launcher=lambda *_: live, voice_analyzer=DeterministicFakeAnalyzer())
        workspace, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers, app)
        invalid = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": approved["canonical_artifact_sha256"], "voice_plan_revision": approved["revision"], "voice": "af_heart"}, headers=headers)
        assert invalid.status_code == 422
        invalid_plan = workspace.load_voice_plan(conversion_id)
        invalid_plan["cast"][1]["voice_id"] = "disabled-voice"
        invalid_identity = with_canonical_artifact_hash(invalid_plan)
        workspace.persist_voice_plan(conversion_id, invalid_identity)
        invalid_voice = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": invalid_identity["canonical_artifact_sha256"], "voice_plan_revision": invalid_identity["revision"]}, headers=headers)
        assert invalid_voice.status_code == 422 and invalid_voice.json()["error"]["code"] == "INVALID_VOICE"
        approved_plan = {**invalid_plan, "cast": [dict(entry) for entry in invalid_plan["cast"]]}
        approved_plan["cast"][1]["voice_id"] = "af_bella"
        workspace.persist_voice_plan(conversion_id, with_canonical_artifact_hash(approved_plan))
        approved = workspace.load_voice_plan(conversion_id)
        started = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": approved["canonical_artifact_sha256"], "voice_plan_revision": approved["revision"]}, headers=headers)
        assert started.status_code == 200 and started.json()["job"]["schema_version"] == 5
        job = workspace.read_job(conversion_id)
        assert job["mode"] == "interactive_voices" and job["voice_plan_revision"] == approved["revision"]
        status = client.get("/api/status", headers=headers).json()
        assert status["interactive_voices"]["available"] and status["generation_summary"]["total_chunks"] == job["total_chunks"]
        cancelling = client.post("/api/generation/cancel", json={"conversion_id": conversion_id}, headers=headers)
        assert cancelling.status_code == 200 and cancelling.json()["status"] == "cancelling"
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = Path("tests") / f".pytest-phase6-v5-launch-failure-{uuid.uuid4().hex}"; root.mkdir()
    try:
        app, client, headers, _ = _app(root, 19902, worker_launcher=lambda *_: (_ for _ in ()).throw(RuntimeError("launch secret")))
        _, conversion_id, approved = _prepare_approved_voice_plan(root, client, headers)
        failed = client.post("/api/generation/start", json={"mode": "interactive_voices", "voice_plan_sha256": approved["canonical_artifact_sha256"], "voice_plan_revision": approved["revision"]}, headers=headers)
        assert failed.status_code == 503 and failed.json()["error"]["code"] == "WORKER_START_FAILED" and "launch secret" not in failed.text
        assert Workspace(root / "data").read_job(conversion_id)["status"] == "failed"
    finally:
        shutil.rmtree(root, ignore_errors=True)

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import uuid

import pytest

from fastapi.testclient import TestClient

from pdf_audiobook.app import _spawn_worker, create_app
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.tts import FakeVoice
from pdf_audiobook.workspace import Workspace, atomic_write_json
from test_pdf import make_pdf


def test_phase5_module_imports_without_media_tools() -> None:
    from pdf_audiobook.m4b import finalize_conversion, verify_m4b

    assert callable(finalize_conversion) and callable(verify_m4b)


def test_spawn_worker_uses_checkout_kokoro_interpreter_by_default(monkeypatch) -> None:
    captured = {}
    class FakePopen:
        def __init__(self, argv, **kwargs): captured.update(argv=argv, kwargs=kwargs)
    import pdf_audiobook.app as app
    monkeypatch.delenv("PDF_AUDIOBOOK_KOKORO_PYTHON", raising=False)
    monkeypatch.setattr(app, "_safe_regular_file", lambda _path: True)
    monkeypatch.setattr(app.subprocess, "Popen", FakePopen)
    _spawn_worker(Path("data"), "conversion")
    expected = Path(app.__file__).resolve().parents[2] / "benchmark" / "environments" / "kokoro" / ".venv" / "Scripts" / "python.exe"
    assert Path(captured["argv"][0]).resolve() == expected
    assert captured["argv"][0] != sys.executable
    assert captured["kwargs"]["shell"] is False
    assert str(Path(app.__file__).resolve().parents[1]) in captured["kwargs"]["env"]["PYTHONPATH"]
    assert captured["argv"][1:3] == ["-m", "pdf_audiobook.worker"]


@pytest.mark.parametrize(("mode", "cpu_count", "expected_threads"), (("background", 16, 8), ("maximum_speed", 16, 16), ("background", 4, 4)))
def test_spawn_worker_sets_authoritative_torch_threads(monkeypatch, mode: str, cpu_count: int, expected_threads: int) -> None:
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured.update(argv=argv, kwargs=kwargs)

    import pdf_audiobook.app as app
    monkeypatch.setenv("PDF_AUDIOBOOK_TORCH_THREADS", "99")
    monkeypatch.setattr(app.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(app, "_safe_regular_file", lambda _path: True)
    monkeypatch.setattr(app.subprocess, "Popen", FakePopen)
    _spawn_worker(Path("data"), "conversion", mode)
    assert captured["kwargs"]["env"]["PDF_AUDIOBOOK_TORCH_THREADS"] == str(expected_threads)


def test_spawn_worker_rejects_invalid_performance_mode_before_popen(monkeypatch) -> None:
    called = False

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            nonlocal called
            called = True

    import pdf_audiobook.app as app
    monkeypatch.setattr(app.subprocess, "Popen", FakePopen)
    with pytest.raises(ValueError, match="invalid performance_mode"):
        _spawn_worker(Path("data"), "conversion", "turbo")
    assert not called


def test_spawn_worker_honors_valid_explicit_interpreter_override(monkeypatch) -> None:
    captured = {}
    class FakePopen:
        def __init__(self, argv, **kwargs): captured.update(argv=argv, kwargs=kwargs)
    import pdf_audiobook.app as app
    root = Path("tests") / f".pytest-phase5-interpreter-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        override = root / "python.exe"
        override.write_bytes(b"test interpreter")
        monkeypatch.setenv("PDF_AUDIOBOOK_KOKORO_PYTHON", str(override))
        monkeypatch.setattr(app.subprocess, "Popen", FakePopen)
        _spawn_worker(Path("data"), "conversion")
        assert Path(captured["argv"][0]) == override
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("selection", ["missing", "directory"])
def test_spawn_worker_rejects_missing_or_unsafe_interpreter_before_popen(monkeypatch, selection: str) -> None:
    called = False
    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            nonlocal called
            called = True
    import pdf_audiobook.app as app
    root = Path("tests") / f".pytest-phase5-interpreter-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        selected = root / "missing.exe" if selection == "missing" else root
        monkeypatch.setenv("PDF_AUDIOBOOK_KOKORO_PYTHON", str(selected))
        monkeypatch.setattr(app.subprocess, "Popen", FakePopen)
        with pytest.raises(OSError, match="Kokoro worker interpreter is unavailable"):
            _spawn_worker(Path("data"), "conversion")
        assert not called
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _prepared_app(root: Path, port: int):
    pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence. Three sentence."])
    launches = []
    app = create_app(port=port, session_token="phase5-token", instance_file=root / "instance.json", data_root=root / "data", worker_launcher=lambda *args: launches.append(args) or type("P", (), {"poll": lambda self: 1})())
    headers = {"Origin": f"http://127.0.0.1:{port}", "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
    client = TestClient(app, base_url=f"http://127.0.0.1:{port}"); assert client.post("/api/session/bootstrap", json={"token": "phase5-token"}, headers=headers).status_code == 200
    assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
    assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
    assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
    conversion_id = Workspace(root / "data").inspect_startup().conversion_id
    assert conversion_id
    return client, headers, Workspace(root / "data"), conversion_id, launches


def test_live_phase5_pid_rejected_and_stale_pid_can_resume() -> None:
    root = Path("tests") / f".pytest-phase5-app-{uuid.uuid4().hex}"; root.mkdir()
    try:
        client, headers, workspace, conversion_id, launches = _prepared_app(root, 19891)
        job = workspace.read_job(conversion_id); job.update(status="assembling", stage="assembling", worker={"pid": os.getpid(), "started_at": job["created_at"], "updated_at": job["updated_at"]}); atomic_write_json(workspace.job_path(conversion_id), job)
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 409
        job["worker"]["pid"] = 999999999; atomic_write_json(workspace.job_path(conversion_id), job)
        assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_synthesis_complete_fast_path_rejects_changed_settings() -> None:
    root = Path("tests") / f".pytest-phase5-settings-{uuid.uuid4().hex}"; root.mkdir()
    try:
        client, headers, workspace, conversion_id, _ = _prepared_app(root, 19890)
        ConversionWorker(workspace, conversion_id).run(engine=FakeVoice())
        job = workspace.read_job(conversion_id); job.update(status="completed", stage="synthesis_complete", worker=None, output=None); atomic_write_json(workspace.job_path(conversion_id), job)
        response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.1}, headers=headers)
        assert response.status_code == 409 and response.json()["error"]["code"] == "SETTINGS_CHANGED"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_phase5_cancel_marks_marker_for_active_stage_without_output() -> None:
    root = Path("tests") / f".pytest-phase5-cancel-{uuid.uuid4().hex}"; root.mkdir()
    try:
        client, headers, workspace, conversion_id, _ = _prepared_app(root, 19889)
        job = workspace.read_job(conversion_id); job.update(status="verifying", stage="verifying", worker={"pid": os.getpid(), "started_at": job["created_at"], "updated_at": job["updated_at"]}); atomic_write_json(workspace.job_path(conversion_id), job)
        response = client.post("/api/generation/cancel", json={}, headers=headers)
        assert response.status_code == 200 and workspace.cancel_marker_path(conversion_id).is_file()
        assert workspace.read_job(conversion_id)["output"] is None
    finally:
        shutil.rmtree(root, ignore_errors=True)

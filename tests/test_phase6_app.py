from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient

from pdf_audiobook.app import create_app
from pdf_audiobook.audio import write_pcm_wav
from pdf_audiobook.tts import FakeVoice
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.workspace import Workspace, atomic_write_json
from test_pdf import make_pdf


def _app(root: Path, port: int, *, opener=None, worker_poll=1, worker_launcher=None):
    preview = root / "previews"
    preview.mkdir(parents=True)
    app = create_app(
        port=port,
        session_token="phase6-token",
        instance_file=root / "instance.json",
        data_root=root / "data",
        preview_root=preview,
        path_opener=opener,
        worker_launcher=worker_launcher or (lambda *_: type("P", (), {"poll": lambda self: worker_poll})()),
    )
    client = TestClient(app, base_url=f"http://127.0.0.1:{port}")
    headers = {"Origin": f"http://127.0.0.1:{port}"}
    assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
    return app, client, headers, preview


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


def test_voice_preview_auth_validation_and_no_store() -> None:
    root = Path("tests") / f".pytest-phase6-preview-{uuid.uuid4().hex}"
    try:
        _, client, headers, preview = _app(root, 19881)
        client.cookies.clear()
        assert client.get("/api/voice-preview/af_heart", headers=headers).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": "phase6-token"}, headers=headers).status_code == 200
        assert client.get("/api/voice-preview/not-a-voice", headers=headers).status_code == 404
        (preview / "sample-kokoro-af_heart.wav").write_bytes(b"not wav")
        assert client.get("/api/voice-preview/af_heart", headers=headers).status_code == 404
        write_pcm_wav(preview / "sample-kokoro-af_heart.wav", b"\0\0" * 240, 24000, overwrite=True)
        response = client.get("/api/voice-preview/af_heart", headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert client.get("/favicon.ico", headers=headers).status_code == 204
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

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient

from pdf_audiobook.audio import write_pcm_wav
import pdf_audiobook.app as app_module
from pdf_audiobook.app import create_app
from pdf_audiobook.workspace import Workspace, atomic_write_json
from test_pdf import make_pdf


def test_preview_generators_keep_kokoro_bound_and_allow_chatterbox_cold_start(monkeypatch) -> None:
    calls = []
    root = Path("tests")

    monkeypatch.setattr(app_module, "_safe_regular_file", lambda _path: True)
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    app_module._default_preview_generator("af_heart", root / "kokoro.wav")
    app_module._default_chatterbox_preview_generator(root / "reference.wav", root / "reference-preview.wav")
    app_module._default_chatterbox_builtin_preview_generator(root / "builtin-preview.wav")

    assert [entry[1]["timeout"] for entry in calls] == [120, 600, 600]
    assert calls[0][0][0][1:3] == ["-m", "pdf_audiobook.preview_worker"]
    assert calls[1][0][0][1:3] == ["-m", "pdf_audiobook.chatterbox_preview_worker"]
    assert calls[2][0][0][1:4] == ["-m", "pdf_audiobook.chatterbox_preview_worker", "--builtin"]
    for _args, options in calls[1:]:
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["TRANSFORMERS_OFFLINE"] == "1"


def _prepared_generation(root: Path, port: int) -> tuple[str, dict[str, str]]:
    pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence. Three sentence."])
    app = create_app(port=port, session_token="phase4-token", instance_file=root / "instance.json", data_root=root / "data", worker_launcher=lambda *_: type("P", (), {"poll": lambda self: 1})())
    headers = {"Origin": f"http://127.0.0.1:{port}", "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
    with TestClient(app, base_url=f"http://127.0.0.1:{port}") as client:
        assert client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers).status_code == 200
        assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
        response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
        assert response.status_code == 200
        return response.json()["conversion_id"], headers


def _prepared_chatterbox(root: Path, port: int, *, preview_generator=None):
    text = "This is an English audiobook chapter with the narrator and story. " * 5
    pdf = make_pdf(root / "chatterbox.pdf", [text])
    reference = root / "reference.wav"
    write_pcm_wav(reference, b"\0\0" * (24000 * 6), 24000, overwrite=True)
    launches = []
    process = type("P", (), {"poll": lambda self: 1})()
    app = create_app(
        port=port,
        session_token="chatterbox-token",
        instance_file=root / "instance.json",
        data_root=root / "data",
        worker_launcher=lambda *args: launches.append(args) or process,
        preview_root=root / "previews",
        chatterbox_preview_generator=preview_generator,
    )
    headers = {"Origin": f"http://127.0.0.1:{port}", "X-PDF-Filename": "chatterbox.pdf", "Content-Type": "application/pdf"}
    return app, pdf, reference, headers, launches


def test_chatterbox_reference_routes_and_fixed_single_voice_contract() -> None:
    root = Path("tests") / f".pytest-phase4-chatterbox-{uuid.uuid4().hex}"; root.mkdir()
    calls = []

    def preview(reference, target, descriptor):
        calls.append((reference, target, descriptor))
        write_pcm_wav(target, b"\0\0" * 240, 24000, overwrite=True)

    try:
        app, pdf, reference, headers, launches = _prepared_chatterbox(root, 19990, preview_generator=preview)
        with TestClient(app, base_url="http://127.0.0.1:19990") as client:
            assert client.post("/api/session/bootstrap", json={"token": "chatterbox-token"}, headers=headers).status_code == 200
            assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
            assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
            assert client.post("/api/chatterbox/reference", content=reference.read_bytes(), headers=headers).status_code == 415
            raw_headers = {"Origin": headers["Origin"], "Content-Type": "audio/wav"}
            assert client.post("/api/chatterbox/reference", content=reference.read_bytes(), headers=raw_headers).status_code == 422
            raw_headers["X-Reference-Consent"] = "confirmed"
            created = client.post("/api/chatterbox/reference", content=reference.read_bytes(), headers=raw_headers)
            assert created.status_code == 200 and created.headers["cache-control"] == "no-store"
            assert "path" not in created.text and "reference_sha256" in created.text
            status = client.get("/api/chatterbox/reference")
            assert status.status_code == 200 and status.headers["cache-control"] == "no-store"
            lock = app.state.phase1.generation_lock
            assert lock.acquire(False)
            try:
                assert client.put("/api/chatterbox/reference", content=reference.read_bytes(), headers=raw_headers).status_code == 409
            finally:
                lock.release()
            preview = client.get("/api/chatterbox/preview")
            assert preview.status_code == 200 and preview.headers["cache-control"] == "no-store" and calls
            bad_model = client.post("/api/generation/start", json={"engine": "chatterbox", "model": "turbo", "voice": "reference-wav", "speed": 1}, headers=headers)
            bad_voice = client.post("/api/generation/start", json={"engine": "chatterbox", "model": "nano", "voice": "af_heart", "speed": 1}, headers=headers)
            invalid = client.post("/api/generation/start", json={"engine": "chatterbox", "model": "nano", "voice": "reference-wav", "speed": 1.1}, headers=headers)
            assert bad_model.status_code == 422 and bad_voice.status_code == 422 and invalid.status_code == 422
            started = client.post("/api/generation/start", json={"engine": "chatterbox", "model": "nano", "voice": "reference-wav", "speed": 1}, headers=headers)
            assert started.status_code == 200
            tts = started.json()["job"]["tts"]
            assert tts["engine"] == "chatterbox" and tts["model"] == "ResembleAI/chatterbox-nano" and tts["voice"] == "reference-wav" and tts["speed"] == 1.0 and tts["chunk_cap"] == 300
            assert tts["settings"]["chunk_mode"] == "legacy" and started.json()["total_chunks"] > 1
            assert launches
            job = Workspace(root / "data").read_job(started.json()["conversion_id"])
            job.update({"status": "failed", "stage": "generation_start", "error": "failed", "last_safe_error": "failed", "worker": None})
            atomic_write_json(Workspace(root / "data").job_path(started.json()["conversion_id"]), job)
            assert client.get("/api/chatterbox/reference").status_code == 200
            revoked = client.delete("/api/chatterbox/reference", headers={"Origin": headers["Origin"]})
            assert revoked.status_code == 200 and revoked.json()["available"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_builtin_starts_without_reference() -> None:
    root = Path("tests") / f".pytest-phase4-chatterbox-builtin-{uuid.uuid4().hex}"; root.mkdir()
    try:
        app, pdf, _reference, headers, launches = _prepared_chatterbox(root, 19991)
        with TestClient(app, base_url="http://127.0.0.1:19991") as client:
            assert client.post("/api/session/bootstrap", json={"token": "chatterbox-token"}, headers=headers).status_code == 200
            assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
            assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
            started = client.post("/api/generation/start", json={"engine": "chatterbox", "model": "nano", "voice": "builtin", "speed": 1}, headers=headers)
            assert started.status_code == 200
            tts = started.json()["job"]["tts"]
            assert tts["voice"] == "builtin" and tts["voice_version"] == "bundled" and tts["voice_checksum"] == "unrecorded"
            assert "reference_descriptor_sha256" not in tts["settings"]
            assert not list((root / "data" / "work").glob("*/chatterbox-reference.wav"))
            assert launches
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_builtin_preview_is_authenticated_cached_and_independent_of_conversion() -> None:
    root = Path("tests") / f".pytest-phase4-chatterbox-builtin-preview-{uuid.uuid4().hex}"; root.mkdir()
    calls = []
    def preview(target):
        calls.append(target)
        write_pcm_wav(target, b"\0\0" * 240, 24000, overwrite=True)
    try:
        app = create_app(port=19992, session_token="chatterbox-token", instance_file=root / "instance.json", data_root=root / "data", preview_root=root / "previews", chatterbox_builtin_preview_generator=preview)
        headers = {"Origin": "http://127.0.0.1:19992"}
        with TestClient(app, base_url="http://127.0.0.1:19992") as client:
            assert client.get("/api/chatterbox/preview/builtin").status_code == 401
            assert client.post("/api/session/bootstrap", json={"token": "chatterbox-token"}, headers=headers).status_code == 200
            first = client.get("/api/chatterbox/preview/builtin")
            second = client.get("/api/chatterbox/preview/builtin")
            assert first.status_code == second.status_code == 200
            assert first.headers["cache-control"] == second.headers["cache-control"] == "no-store"
            assert first.content.startswith(b"RIFF") and second.content.startswith(b"RIFF")
            assert len(calls) == 1 and not (root / "data" / "work").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_builtin_preview_failure_is_bounded() -> None:
    root = Path("tests") / f".pytest-phase4-chatterbox-builtin-preview-failure-{uuid.uuid4().hex}"; root.mkdir()
    def preview(_target):
        raise OSError("secret local path")
    try:
        app = create_app(port=19993, session_token="chatterbox-token", instance_file=root / "instance.json", data_root=root / "data", preview_root=root / "previews", chatterbox_builtin_preview_generator=preview)
        headers = {"Origin": "http://127.0.0.1:19993"}
        with TestClient(app, base_url="http://127.0.0.1:19993") as client:
            assert client.post("/api/session/bootstrap", json={"token": "chatterbox-token"}, headers=headers).status_code == 200
            response = client.get("/api/chatterbox/preview/builtin")
            assert response.status_code == 503
            assert response.json() == {"error": {"code": "PREVIEW_FAILED", "message": "built-in preview generation failed"}}
            assert "secret local path" not in response.text and str(root) not in response.text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generation_routes_auth_origin_and_strict_input() -> None:
    root = Path("tests") / f".pytest-phase4-app-{uuid.uuid4().hex}"; root.mkdir()
    try:
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        app = create_app(port=19998, session_token="phase4-token", instance_file=root / "instance.json", data_root=root / "data", worker_launcher=lambda *_: type("P", (), {"poll": lambda self: 0})())
        headers = {"Origin": "http://127.0.0.1:19998", "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        with TestClient(app, base_url="http://127.0.0.1:19998") as client:
            assert client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers).status_code == 200
            assert client.post("/api/analyze", content=pdf.read_bytes(), headers=headers).status_code == 200
            assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers).status_code == 200
            assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0, "extra": True}, headers=headers).status_code == 422
            assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers={**headers, "Origin": "http://localhost:9"}).status_code == 403
            assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
            cancel = client.post("/api/generation/cancel", json={}, headers=headers)
            assert cancel.status_code == 409 and cancel.json()["error"]["code"] == "NOT_GENERATING"
            assert not list((root / "data" / "work").glob("*/cancel.request"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stale_manifest_worker_pid_can_resume_once() -> None:
    root = Path("tests") / f".pytest-phase4-app-stale-{uuid.uuid4().hex}"; root.mkdir()
    try:
        conversion_id, headers = _prepared_generation(root, 19997)
        workspace = Workspace(root / "data"); job = workspace.read_job(conversion_id)
        job["status"] = "synthesizing"; job["worker"] = {"pid": 999999999, "started_at": job["created_at"], "updated_at": job["updated_at"]}; atomic_write_json(workspace.job_path(conversion_id), job)
        launches: list[str] = []
        app = create_app(port=19997, session_token="phase4-token", instance_file=root / "instance-2.json", data_root=root / "data", worker_launcher=lambda _root, cid, _mode: launches.append(cid) or type("P", (), {"poll": lambda self: 1})())
        with TestClient(app, base_url="http://127.0.0.1:19997") as client:
            assert client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers).status_code == 200
            response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
            assert response.status_code == 200 and launches == [conversion_id]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_live_manifest_worker_pid_is_rejected_without_launch() -> None:
    root = Path("tests") / f".pytest-phase4-app-live-{uuid.uuid4().hex}"; root.mkdir()
    try:
        conversion_id, headers = _prepared_generation(root, 19996)
        workspace = Workspace(root / "data"); job = workspace.read_job(conversion_id)
        job["status"] = "synthesizing"; job["worker"] = {"pid": os.getpid(), "started_at": job["created_at"], "updated_at": job["updated_at"]}; atomic_write_json(workspace.job_path(conversion_id), job)
        launches: list[str] = []
        app = create_app(port=19996, session_token="phase4-token", instance_file=root / "instance-2.json", data_root=root / "data", worker_launcher=lambda _root, cid, _mode: launches.append(cid))
        with TestClient(app, base_url="http://127.0.0.1:19996") as client:
            assert client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers).status_code == 200
            response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
            assert response.status_code == 409 and response.json()["error"]["code"] == "ACTIVE_WORKER" and launches == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_worker_launch_failure_persists_safe_failed_state() -> None:
    root = Path("tests") / f".pytest-phase4-app-launch-fail-{uuid.uuid4().hex}"; root.mkdir()
    try:
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        def launch(*_args): raise RuntimeError("secret worker details")
        app = create_app(port=19995, session_token="phase4-token", instance_file=root / "instance.json", data_root=root / "data", worker_launcher=launch)
        headers = {"Origin": "http://127.0.0.1:19995", "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        with TestClient(app, base_url="http://127.0.0.1:19995") as client:
            client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers)
            client.post("/api/analyze", content=pdf.read_bytes(), headers=headers)
            client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers)
            response = client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers)
            assert response.status_code == 503 and "secret worker details" not in response.text
        job = Workspace(root / "data").read_job(Workspace(root / "data").inspect_startup().conversion_id or "")
        assert job["status"] == "failed" and job["worker"] is None and job["last_safe_error"] == "worker launch failed: RuntimeError"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cancel_accepts_live_local_process_before_manifest_pid_claim() -> None:
    root = Path("tests") / f".pytest-phase4-app-cancel-race-{uuid.uuid4().hex}"; root.mkdir()
    try:
        pdf = make_pdf(root / "book.pdf", ["One sentence. Two sentence."])
        class LiveProcess:
            def poll(self): return None
        app = create_app(port=19994, session_token="phase4-token", instance_file=root / "instance.json", data_root=root / "data", worker_launcher=lambda *_: LiveProcess())
        headers = {"Origin": "http://127.0.0.1:19994", "X-PDF-Filename": "book.pdf", "Content-Type": "application/pdf"}
        with TestClient(app, base_url="http://127.0.0.1:19994") as client:
            client.post("/api/session/bootstrap", json={"token": "phase4-token"}, headers=headers)
            client.post("/api/analyze", content=pdf.read_bytes(), headers=headers)
            client.post("/api/chapter-plan", json={"mode": "whole"}, headers=headers)
            assert client.post("/api/generation/start", json={"voice": "af_heart", "speed": 1.0}, headers=headers).status_code == 200
            cancel = client.post("/api/generation/cancel", json={}, headers=headers)
            assert cancel.status_code == 200 and cancel.json()["cancel_requested"] is True
            conversion_id = Workspace(root / "data").inspect_startup().conversion_id
            assert conversion_id is not None
            assert Workspace(root / "data").cancel_marker_path(conversion_id).is_file()
    finally:
        shutil.rmtree(root, ignore_errors=True)

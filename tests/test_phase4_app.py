from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient

from pdf_audiobook.app import create_app
from pdf_audiobook.workspace import Workspace, atomic_write_json
from test_pdf import make_pdf


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

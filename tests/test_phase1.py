from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdf_audiobook.app import COOKIE_NAME, _store_chatterbox_reference_locked, create_app
import pdf_audiobook.launcher as launcher
from pdf_audiobook.launcher import InstanceLock, existing_instance, run_launcher
from pdf_audiobook.security import (
    atomic_write_instance,
    build_instance,
    read_instance,
    remove_instance_if_matches,
    validate_instance,
    pid_is_alive,
)


PORT = 18765


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / f".pytest-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def client_for(tmp_path: Path, token: str = "phase1-test-token-012345678901234567890123"):
    app = create_app(port=PORT, launch_id="launch-test", session_token=token, instance_file=tmp_path / "instance.json")
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def local_headers(**extra: str) -> dict[str, str]:
    return {"Host": f"127.0.0.1:{PORT}", **extra}


def test_static_shell_and_host_guard(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        response = client.get("/", headers=local_headers())
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "af_heart" in response.text and "bf_isabella" in response.text
        assert all(url not in response.text for url in ("https://", "http://"))
        styles = client.get("/styles.css", headers=local_headers())
        script = client.get("/app.js", headers=local_headers())
        assert styles.status_code == 200 and styles.headers["cache-control"] == "no-store"
        assert script.status_code == 200 and script.headers["cache-control"] == "no-store"
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_chatterbox_reference_publish_waits_for_preview_lock() -> None:
    lock = threading.Lock()
    state = type("State", (), {"preview_lock": lock, "preview_root": None})()
    started = threading.Event()
    completed = threading.Event()
    results: list[dict] = []

    class WorkspaceStub:
        def store_chatterbox_reference(self, conversion_id, source, *, consent_confirmed):
            return {"conversion_id": conversion_id, "consent_confirmed": consent_confirmed}

    lock.acquire()
    try:
        def publish() -> None:
            started.set()
            results.append(_store_chatterbox_reference_locked(state, WorkspaceStub(), "conversion", Path("reference.wav"), replace=False))
            completed.set()

        thread = threading.Thread(target=publish)
        thread.start()
        assert started.wait(1)
        assert not completed.wait(0.05)
    finally:
        lock.release()
    thread.join(timeout=1)
    assert not thread.is_alive() and results == [{"conversion_id": "conversion", "consent_confirmed": True}]


def test_chatterbox_reference_lock_contention_reports_busy_reason(tmp_path: Path) -> None:
    token = "phase1-lock-token-012345678901234567890123"
    app = create_app(port=PORT, session_token=token, instance_file=tmp_path / "instance.json", data_root=tmp_path / "data")
    headers = {"Origin": f"http://127.0.0.1:{PORT}", "Content-Type": "audio/wav", "X-Reference-Consent": "confirmed"}
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as client:
        assert client.post("/api/session/bootstrap", json={"token": token}, headers=headers).status_code == 200
        generation_lock = app.state.phase1.generation_lock
        analysis_lock = app.state.phase1.analysis_lock
        generation_lock.acquire()
        try:
            assert client.post("/api/chatterbox/reference", content=b"", headers=headers).json()["error"]["code"] == "ACTIVE_WORKER"
        finally:
            generation_lock.release()
        analysis_lock.acquire()
        try:
            assert client.post("/api/chatterbox/reference", content=b"", headers=headers).json()["error"]["code"] == "BUSY"
        finally:
            analysis_lock.release()


def test_bootstrap_cookie_and_mutation_origin_checks(tmp_path: Path) -> None:
    token = "phase1-test-token-012345678901234567890123"
    with client_for(tmp_path, token) as client:
        headers = local_headers(Origin="http://127.0.0.1:18765")
        response = client.post("/api/session/bootstrap", json={"token": token}, headers=headers)
        assert response.status_code == 200
        cookie = client.cookies.get(COOKIE_NAME)
        assert cookie
        cookie_header = response.headers.get("set-cookie", "").lower()
        assert "httponly" in cookie_header and "samesite=strict" in cookie_header
        assert client.post("/api/shutdown", headers=headers).status_code == 200


def test_health_identity_and_session_lifecycle(tmp_path: Path) -> None:
    token = "phase1-test-token-012345678901234567890123"
    with client_for(tmp_path, token) as client:
        headers = local_headers()
        assert client.get("/health", headers=headers).json()["launch_id"] is None
        assert client.get("/health", headers={**headers, "X-Instance-Token": token}).json()["launch_id"] == "launch-test"
        assert client.get("/api/session", headers=headers).status_code == 401
        origin = {**headers, "Origin": "http://127.0.0.1:18765"}
        assert client.post("/api/session/bootstrap", json={"token": "wrong-token"}, headers=origin).status_code == 401
        assert client.post("/api/session/bootstrap", json={"token": token}, headers=origin).status_code == 200
        assert client.get("/api/session", headers=headers).status_code == 200



def test_missing_or_foreign_origin_and_unauthenticated_mutation_fail(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        headers = local_headers()
        assert client.post("/api/session/bootstrap", json={"token": "wrong"}, headers=headers).status_code == 403
        assert client.post("/api/shutdown", headers=headers).status_code == 403
        assert client.post("/api/session/bootstrap", json={}, headers={**headers, "Origin": "https://evil.example"}).status_code == 403


def test_instance_atomic_schema_and_exact_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "instance.json"
    value = build_instance(pid=123, port=PORT, launch_id="launch", token="phase1-test-token-012345678901234567890123")
    atomic_write_instance(value, path)
    assert read_instance(path) == value
    assert remove_instance_if_matches(path, launch_id="wrong", pid=123, token=value["session_token"]) is False
    assert remove_instance_if_matches(path, launch_id="launch", pid=123, token=value["session_token"]) is True
    assert not path.exists()
    with (tmp_path / "bad.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 999}, handle)
    try:
        validate_instance(json.loads((tmp_path / "bad.json").read_text(encoding="utf-8")))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid schema accepted")


def test_pid_liveness_rejects_invalid_values_and_keeps_current_process_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    import pdf_audiobook.security as security

    calls: list[int] = []
    monkeypatch.setattr(security.os, "name", "nt")
    monkeypatch.setattr(security, "_windows_pid_is_alive", lambda pid: calls.append(pid) or True)
    for invalid_pid in (True, False, 0, -1, "123", None):
        assert pid_is_alive(invalid_pid) is False  # type: ignore[arg-type]
    assert pid_is_alive(123) is True
    assert calls == [123]
    monkeypatch.undo()
    assert pid_is_alive(os.getpid()) is True


def test_frontend_has_three_views_and_four_voice_ids() -> None:
    html = Path("src/pdf_audiobook/static/index.html").read_text(encoding="utf-8")
    assert all(f'id="view-{name}"' in html for name in ("add", "configure", "progress"))
    assert all(voice in html for voice in ("af_heart", "af_bella", "bf_emma", "bf_isabella"))


def test_existing_instance_requires_live_pid_and_authenticated_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "instance-root"
    value = build_instance(pid=123, port=PORT, launch_id="launch", token="phase1-test-token-012345678901234567890123")
    atomic_write_instance(value, root / "instance.json")
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "_health", lambda _instance: True)
    assert existing_instance(root) == value
    monkeypatch.setattr(launcher, "_health", lambda _instance: False)
    assert existing_instance(root) is None
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(launcher, "_health", lambda _instance: True)
    assert existing_instance(root) is None


def test_instance_lock_stale_recovery_and_ownership_safe_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "instance.lock"
    path.write_text("999999:stale", encoding="ascii")
    monkeypatch.setattr(launcher, "pid_is_alive", lambda pid: pid != 999999)
    lock = InstanceLock(path)
    assert lock.acquire() is True
    lock.marker = "foreign-marker"
    lock.close()
    assert path.exists()
    path.unlink()


def test_existing_instance_reuse_opens_exact_fragment_without_starting_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "reuse"
    root.mkdir()
    value = build_instance(pid=123, port=PORT, launch_id="launch", token="phase1-test-token-012345678901234567890123")
    atomic_write_instance(value, root / "instance.json")
    (root / "instance.lock").write_text("456:live", encoding="ascii")
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "_health", lambda _instance: True)
    opened: list[str] = []
    monkeypatch.setattr(launcher.uvicorn, "Server", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("server started")))
    assert run_launcher(root=root, open_browser=lambda url: opened.append(url) or True) == 0
    assert opened == [f"http://127.0.0.1:{PORT}/#session={value['session_token']}"]


def test_new_launch_binds_loopback_waits_readiness_and_joins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "new"
    config_calls: list[tuple[str, int]] = []
    servers: list[object] = []

    class FakeConfig:
        def __init__(self, _app: object, *, host: str, port: int, **_kwargs: object) -> None:
            config_calls.append((host, port))

    class FakeServer:
        def __init__(self, _config: object) -> None:
            self.should_exit = False
            servers.append(self)

        def run(self) -> None:
            while not self.should_exit:
                time.sleep(0.001)

    import time

    monkeypatch.setattr(launcher, "choose_port", lambda *_args: 18888)
    monkeypatch.setattr(launcher.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(launcher.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(launcher, "_health", lambda _instance: True)
    assert run_launcher(root=root, open_browser=lambda _url: True, run_server=False) == 0
    assert config_calls == [("127.0.0.1", 18888)]
    assert servers and servers[0].should_exit is True
    assert not (root / "instance.json").exists()
    assert not (root / "instance.lock").exists()


def test_launcher_reaps_live_worker_before_instance_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "live-worker"
    instance_file = root / "instance.json"
    events: list[str] = []

    class FakeWorker:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False
            self.killed = False
            self.returncode: int | None = None

        def poll(self) -> int | None:
            events.append("worker.poll")
            return None if self.alive else self.returncode

        def terminate(self) -> None:
            events.append("worker.terminate")
            assert self.alive
            self.terminated = True

        def kill(self) -> None:
            events.append("worker.kill")
            assert self.terminated and self.alive
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            events.append("worker.wait")
            assert self.terminated
            assert timeout is not None and timeout > 0
            assert instance_file.exists()
            if self.killed:
                self.alive = False
                self.returncode = 0
            return self.returncode

    worker = FakeWorker()
    phase1 = type("Phase1State", (), {"shutdown_event": threading.Event(), "worker_process": worker, "workspace_root": root})()
    app = type("FakeApp", (), {"state": type("FakeState", (), {"phase1": phase1})()})()

    class FakeConfig:
        def __init__(self, _app: object, **_kwargs: object) -> None:
            pass

    class FakeServer:
        should_exit = False

        def __init__(self, _config: object) -> None:
            pass

        def run(self) -> None:
            return

    class FakeWorkspace:
        def __init__(self, _root: Path) -> None:
            events.append("workspace.init")

        def inspect_startup(self) -> object:
            events.append("workspace.inspect")
            return type("Inspection", (), {"conversion_id": "conversion-id"})()

        def request_cancel(self, conversion_id: str) -> None:
            events.append(f"workspace.cancel:{conversion_id}")

    original_remove = launcher.remove_instance_if_matches

    def fake_remove_instance(path: Path, *, launch_id: str, pid: int, token: str) -> bool:
        events.append("instance.cleanup")
        assert not worker.alive
        return original_remove(path, launch_id=launch_id, pid=pid, token=token)

    monkeypatch.setattr(launcher, "create_app", lambda **_kwargs: app)
    monkeypatch.setattr(launcher, "choose_port", lambda *_args: 18889)
    monkeypatch.setattr(launcher.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(launcher.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(launcher, "_health", lambda _instance: True)
    monkeypatch.setattr(launcher, "Workspace", FakeWorkspace)
    monkeypatch.setattr(launcher, "remove_instance_if_matches", fake_remove_instance)

    assert run_launcher(root=root, open_browser=lambda _url: True, run_server=False) == 0
    assert events.index("workspace.cancel:conversion-id") < events.index("worker.terminate")
    assert events.index("worker.terminate") < events.index("worker.wait") < events.index("worker.kill") < events.index("instance.cleanup")
    assert events.count("worker.wait") == 2
    assert phase1.shutdown_event.is_set()
    assert worker.terminated and worker.killed and not worker.alive and worker.returncode == 0
    assert not instance_file.exists()
    assert not (root / "instance.lock").exists()


def test_runtime_assets_have_no_remote_references() -> None:
    for path in Path("src/pdf_audiobook/static").glob("*"):
        if path.suffix not in {".html", ".css", ".js"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "http://" not in text and "https://" not in text and "//host" not in text
        assert "cdn" not in text and "@import" not in text and "url(" not in text
    script = Path("src/pdf_audiobook/static/app.js").read_text(encoding="utf-8")
    assert "response.ok" in script
    assert "Could not establish a secure local session" in script

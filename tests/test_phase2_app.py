from __future__ import annotations

from pathlib import Path
import shutil
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

import pdf_audiobook.app as app_module
import pdf_audiobook.pdf as pdf_module
from pdf_audiobook.app import create_app
from pdf_audiobook.pdf import ERROR_OCR_REQUIRED
from test_pdf import make_pdf


PORT = 18766
TOKEN = "phase2-test-token-012345678901234567890123"


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / f".pytest-phase2-app-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def client_for(tmp_path: Path) -> TestClient:
    app = create_app(port=PORT, launch_id="phase2", session_token=TOKEN, instance_file=tmp_path / "instance.json", data_root=tmp_path / "data")
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def auth(client: TestClient) -> None:
    response = client.post("/api/session/bootstrap", json={"token": TOKEN}, headers={"Origin": f"http://127.0.0.1:{PORT}"})
    assert response.status_code == 200


def upload_headers() -> dict[str, str]:
    return {"Origin": f"http://127.0.0.1:{PORT}", "X-PDF-Filename": "My%20Book.pdf", "Content-Type": "application/pdf"}


def test_raw_upload_analyzes_persists_and_reports_status(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["Chapter 1\nThe quick brown fox jumps over the dog. This is useful English text for review."])
    with client_for(tmp_path) as client:
        assert client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers()).status_code == 403
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "analyzed"
        assert body["analysis"]["detected_language"] == "English"
        assert body["analysis"]["chapter_candidates"][0]["title"] == "Chapter 1"
        assert "cleaned_text" not in body["analysis"]
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["state"] == "analyzed"
        conversion = Path(tmp_path / "data" / "work" / body["conversion_id"])
        assert conversion.joinpath("source.pdf").is_file()
        assert conversion.joinpath("cleaned.txt").is_file()
        assert conversion.joinpath("cleaned-map.json").is_file()
        assert conversion.joinpath("analysis.json").is_file()


def test_upload_errors_are_stable_and_staging_is_cleaned(tmp_path: Path) -> None:
    scanned = make_pdf(tmp_path / "scanned.pdf", [None])
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=scanned.read_bytes(), headers=upload_headers())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == ERROR_OCR_REQUIRED
        assert not list((tmp_path / "data" / ".staging").glob("*"))

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    with client_for(bad_root) as client:
        auth(client)
        response = client.post("/api/analyze", content=b"not a pdf", headers=upload_headers())
        assert response.json()["error"]["code"] == "INVALID_SIGNATURE"


def test_state_changing_routes_require_exact_origin_and_delete_is_explicit(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers={**upload_headers(), "Origin": "http://localhost:9999"})
        assert response.status_code == 403
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        conversion_id = response.json()["conversion_id"]
        assert client.delete(f"/api/workspace/{conversion_id}", headers={"Origin": "http://localhost:9999"}).status_code == 403
        assert client.delete(f"/api/workspace/{conversion_id}", headers={"Origin": f"http://127.0.0.1:{PORT}"}).json() == {"deleted": True}
        assert client.get("/api/status").json() == {"state": "no_active"}


def test_second_upload_is_refused_while_first_active(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    with client_for(tmp_path) as client:
        auth(client)
        assert client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers()).status_code == 200
        second = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert second.status_code == 422
        assert second.json()["error"]["code"] == "ACTIVE_JOB"


def test_missing_conversion_directory_can_be_deleted_and_replaced(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    with client_for(tmp_path) as client:
        auth(client)
        first = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert first.status_code == 200
        conversion_id = first.json()["conversion_id"]
        shutil.rmtree(tmp_path / "data" / "work" / conversion_id)
        invalid = client.get("/api/status")
        assert invalid.status_code == 200 and invalid.json() == {"state": "invalid", "conversion_id": conversion_id, "reason": "active conversion directory is missing or unsafe"}
        deleted = client.delete("/api/workspace/active", headers={"Origin": f"http://127.0.0.1:{PORT}"})
        assert deleted.status_code == 200 and deleted.json() == {"deleted": True}
        assert client.get("/api/status").json() == {"state": "no_active"}
        replacement = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert replacement.status_code == 200


def test_streaming_size_limit_returns_stable_error_and_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "MAX_PDF_BYTES", 4)
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=b"%PDF-oversized", headers=upload_headers())
        assert response.status_code == 422
        assert response.json()["error"] == {"code": "SIZE_LIMIT", "message": "The PDF exceeds the 100 MiB size limit.", "maximum_bytes": 4}
        assert not (tmp_path / "data" / "active.json").exists()
        assert not list((tmp_path / "data" / ".staging").glob("*"))


def test_disk_preflight_rejects_before_workspace_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    monkeypatch.setattr(pdf_module.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(0, 0, 0))
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INSUFFICIENT_DISK"
        assert not (tmp_path / "data" / "active.json").exists()
        assert not (tmp_path / "data" / "work").exists()


def test_filename_is_title_fallback_when_pdf_has_no_metadata_title(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "source.pdf", ["The quick brown fox and the useful English review text are here."])
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 200
        assert response.json()["analysis"]["title"] == "My Book.pdf"


def test_malformed_active_can_be_reset_and_deletion_reports_busy(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    client = client_for(tmp_path)
    with client:
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 200
        active = tmp_path / "data" / "active.json"
        active.write_text("{not-json", encoding="utf-8")
        assert client.delete("/api/workspace/active", headers={"Origin": f"http://127.0.0.1:{PORT}"}).json() == {"deleted": True}
        assert active.exists() is False
        assert (tmp_path / "data" / "work").exists()
        state = client.app.state.phase1
        assert state.analysis_lock.acquire(blocking=False)
        try:
            busy = client.delete("/api/workspace/active", headers={"Origin": f"http://127.0.0.1:{PORT}"})
        finally:
            state.analysis_lock.release()
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "BUSY"


def test_regular_staging_path_is_rejected_without_overwrite(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    with client_for(tmp_path) as client:
        auth(client)
        staging = tmp_path / "data" / ".staging"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text("preserve", encoding="utf-8")
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "STAGING_UNAVAILABLE"
        assert staging.read_text(encoding="utf-8") == "preserve"
        assert not (tmp_path / "data" / "active.json").exists()


def test_symlink_staging_path_is_rejected_without_touching_target(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = tmp_path / "data" / ".staging"
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "STAGING_UNAVAILABLE"
        assert not list(outside.iterdir())


def test_analysis_runs_in_worker_and_health_remains_responsive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The quick brown fox and the useful English review text are here."])
    started = threading.Event()
    release = threading.Event()
    original = app_module.analyze_pdf

    def blocking_analyze(source: Path, **kwargs):
        started.set()
        release.wait(timeout=10)
        return original(source, **kwargs)

    monkeypatch.setattr(app_module, "analyze_pdf", blocking_analyze)
    with client_for(tmp_path) as client:
        auth(client)
        result: dict[str, object] = {}

        def submit() -> None:
            try:
                result["response"] = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
            except BaseException as error:  # pragma: no cover - surfaced below
                result["error"] = error

        worker = threading.Thread(target=submit, daemon=True)
        worker.start()
        try:
            assert started.wait(timeout=5)
            health = client.get("/health", headers={"X-Instance-Token": TOKEN})
            assert health.status_code == 200
        finally:
            release.set()
            worker.join(timeout=10)
        assert not worker.is_alive()
        assert "error" not in result
        assert result["response"].status_code == 200


def test_ui_keeps_local_three_view_phase2_review_evidence(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        html = client.get("/").text
        script = client.get("/app.js").text
        assert all(f'id="view-{name}"' in html for name in ("add", "configure", "progress"))
        assert "cleaned preview" in html.lower()
        assert "api/analyze" in script
        assert "OCR_REQUIRED" in script
        assert "No warnings reported" in script
        assert "/api/workspace/active" in script
        assert "name === " + '"progress"' in script
        assert "http://" not in html.lower().replace(f"http://127.0.0.1:{PORT}", "")

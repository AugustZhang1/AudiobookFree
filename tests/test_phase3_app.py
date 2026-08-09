from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from pdf_audiobook.app import create_app
from test_pdf import make_pdf


PORT = 18767
TOKEN = "phase3-test-token-012345678901234567890123"


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / f".pytest-phase3-app-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def client_for(tmp_path: Path) -> TestClient:
    app = create_app(port=PORT, launch_id="phase3", session_token=TOKEN, instance_file=tmp_path / "instance.json", data_root=tmp_path / "data")
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def auth(client: TestClient) -> None:
    response = client.post("/api/session/bootstrap", json={"token": TOKEN}, headers={"Origin": f"http://127.0.0.1:{PORT}"})
    assert response.status_code == 200


def upload_headers(origin: str | None = None) -> dict[str, str]:
    return {"Origin": origin or f"http://127.0.0.1:{PORT}", "X-PDF-Filename": "Phase%203%20Book.pdf", "Content-Type": "application/pdf"}


def analyzed_client(tmp_path: Path) -> tuple[TestClient, str]:
    path = make_pdf(tmp_path / "book.pdf", ["Chapter 1\nAlpha starts here. It has enough text for a deterministic review boundary.\n\nChapter 2\nBeta continues here. It also has enough text for local planning and labels."])
    client = client_for(tmp_path)
    auth(client)
    response = client.post("/api/analyze", content=path.read_bytes(), headers=upload_headers())
    assert response.status_code == 200
    return client, response.json()["conversion_id"]


def test_chapter_plan_requires_authentication_and_exact_origin(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers={"Origin": f"http://127.0.0.1:{PORT}"}).status_code == 403
        auth(client)
        response = client.post("/api/chapter-plan", json={"mode": "whole"}, headers={"Origin": "http://localhost:9999"})
        assert response.status_code == 403


def test_no_active_and_busy_state_errors_are_stable(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        auth(client)
        response = client.post("/api/chapter-plan", json={"mode": "whole"}, headers=upload_headers())
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NO_ACTIVE"
        state = client.app.state.phase1
        assert state.analysis_lock.acquire(blocking=False)
        try:
            response = client.post("/api/chapter-plan", json={"mode": "whole"}, headers=upload_headers())
        finally:
            state.analysis_lock.release()
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "BUSY"


def test_all_modes_persist_and_status_recovers(tmp_path: Path) -> None:
    client, conversion_id = analyzed_client(tmp_path)
    with client:
        for mode, payload, expected in (("original", {"mode": "original"}, None), ("whole", {"mode": "whole"}, 1), ("custom", {"mode": "custom", "count": 2}, 2)):
            response = client.post("/api/chapter-plan", json=payload, headers=upload_headers())
            assert response.status_code == 200
            plan = response.json()["chapter_plan"]
            assert plan["mode"] == mode
            if expected is not None:
                assert len(plan["chapters"]) == expected
            if mode == "original":
                first = plan
        assert client.get("/api/status").json()["state"] == "planned"
        assert client.get("/api/status").json()["chapter_plan"]["cleaned_text_sha256"] == first["cleaned_text_sha256"]
        assert conversion_id == client.get("/api/status").json()["conversion_id"]

    recovered = client_for(tmp_path)
    with recovered:
        auth(recovered)
        status = recovered.get("/api/status")
        assert status.status_code == 200
        assert status.json()["state"] == "planned"
        assert status.json()["chapter_plan"]["mode"] == "custom"


def test_custom_plan_is_deterministic_and_impossible_count_is_explained(tmp_path: Path) -> None:
    client, _ = analyzed_client(tmp_path)
    with client:
        headers = upload_headers()
        first = client.post("/api/chapter-plan", json={"mode": "custom", "count": 2}, headers=headers)
        second = client.post("/api/chapter-plan", json={"mode": "custom", "count": 2}, headers=headers)
        assert first.status_code == second.status_code == 200
        assert first.json()["chapter_plan"] == second.json()["chapter_plan"]
        too_high = client.post("/api/chapter-plan", json={"mode": "custom", "count": 50}, headers=headers)
        assert too_high.status_code == 422
        assert too_high.json()["error"]["code"] == "COUNT_TOO_HIGH"
        assert "recommended_maximum" in too_high.json()["error"]


def test_plan_request_validation_and_rename_only_behavior(tmp_path: Path) -> None:
    client, _ = analyzed_client(tmp_path)
    with client:
        headers = upload_headers()
        for payload, code in (({"mode": "bogus"}, "INVALID_MODE"), ({"mode": "custom", "count": 1}, "INVALID_COUNT"), ({"mode": "whole", "count": 2}, "INVALID_COUNT")):
            response = client.post("/api/chapter-plan", json=payload, headers=headers)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == code
        assert client.post("/api/chapter-plan", content=b"not-json", headers={**headers, "Content-Type": "application/json"}).json()["error"]["code"] == "INVALID_BODY"
        not_ready = client.post("/api/chapter-plan/titles", json={"titles": ["Not ready"]}, headers=headers)
        assert not_ready.status_code == 409
        assert not_ready.json()["error"]["code"] == "NOT_READY"
        plan_response = client.post("/api/chapter-plan", json={"mode": "original"}, headers=headers)
        original = plan_response.json()["chapter_plan"]
        titles = [f"Renamed {chapter['index']}" for chapter in original["chapters"]]
        renamed_response = client.post("/api/chapter-plan/titles", json={"titles": titles}, headers=headers)
        assert renamed_response.status_code == 200
        renamed = renamed_response.json()["chapter_plan"]
        assert [chapter["title"] for chapter in renamed["chapters"]] == titles
        assert [(chapter["start_offset"], chapter["end_offset"]) for chapter in renamed["chapters"]] == [(chapter["start_offset"], chapter["end_offset"]) for chapter in original["chapters"]]
        invalid_titles = client.post("/api/chapter-plan/titles", json={"titles": []}, headers=headers)
        assert invalid_titles.status_code == 422
        assert invalid_titles.json()["error"]["code"] == "INVALID_TITLES"


def test_ui_exposes_phase3_controls_and_truthful_plan_gate(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        html = client.get("/").text
        script = client.get("/app.js").text
        assert 'value="original"' in html and 'value="custom"' in html and 'value="whole"' in html
        assert 'min="2" max="50"' in html
        assert 'class="primary next"' in html and 'disabled' in html
        assert "Phase 6" in html
        assert "reliable layout headings" in html
        assert "if (await generateOriginal())" in script
        assert "Chapter planning needs attention" in script
        assert 'document.querySelector("#plan-status").textContent = "Generate and review' in script
        assert 'runPlanRequest(mode);' in script
        assert 'Choose a count from 2–50, then select Generate plan.' in script
        assert 'if (planRequestInFlight) return false;' in script
        assert 'setPlanControlsDisabled(true);' in script and 'setPlanControlsDisabled(false);' in script
        assert 'planMatchesSelection' in script


def test_tampered_plan_is_invalid_in_status(tmp_path: Path) -> None:
    client, conversion_id = analyzed_client(tmp_path)
    with client:
        assert client.post("/api/chapter-plan", json={"mode": "whole"}, headers=upload_headers()).status_code == 200
        chapters_path = tmp_path / "data" / "work" / conversion_id / "chapters.json"
        chapters_path.write_text(chapters_path.read_text(encoding="utf-8").replace("Phase 3 Book.pdf", "Tampered"), encoding="utf-8")
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["state"] == "invalid"


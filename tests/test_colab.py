from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import uuid

import pytest

from pdf_audiobook import colab
from pdf_audiobook.worker import WorkerResult


@pytest.fixture
def sandbox_path():
    path = Path.cwd() / f".colab-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _progress_manifest(*, stage: str = "synthesis", completed: int = 0, total: int = 4, chapter: int | None = None) -> dict:
    records = [] if chapter is None else [{"global_index": completed - 1, "chapter_index": chapter}]
    return {
        "status": "synthesizing" if stage == "synthesis" else stage,
        "stage": stage,
        "progress": {"completed": completed, "current": completed, "total": total},
        "completed_chunks": records,
    }


def test_progress_display_formats_percentage_chunk_count_and_chapter() -> None:
    lines: list[str] = []
    display = colab.ColabProgressDisplay({"chapters": [{"index": 1}, {"index": 2}]}, output=lines.append, bar_width=8)

    display.render(_progress_manifest(completed=2, total=4, chapter=2))

    assert lines == ["[####----]  50.0% (2/4 chunks | chapter 2/2) | stage: synthesis"]
    assert lines[0].isascii()


def test_progress_display_shows_resumed_count_immediately() -> None:
    lines: list[str] = []
    display = colab.ColabProgressDisplay({"chapters": [{"index": 1}]}, output=lines.append)

    display.render(_progress_manifest(completed=3, total=4, chapter=1))

    assert "75.0%" in lines[0]
    assert "3/4 chunks" in lines[0]


def test_progress_proxy_emits_each_new_chunk_and_suppresses_duplicates() -> None:
    lines: list[str] = []
    display = colab.ColabProgressDisplay({"chapters": [{"index": 1}]}, output=lines.append)
    state = {"manifest": _progress_manifest()}

    class Workspace:
        def update_generation(self, *args, **kwargs):
            state["manifest"] = _progress_manifest(completed=kwargs["completed"], total=4, chapter=1)
            return state["manifest"]

        def marker(self):
            return "delegated"

    proxy = colab._ProgressWorkspaceProxy(Workspace(), display)
    first = proxy.update_generation("id", completed=1)
    second = proxy.update_generation("id", completed=1)
    proxy.update_generation("id", completed=2)

    assert first["progress"]["completed"] == second["progress"]["completed"] == 1
    assert proxy.marker() == "delegated"
    assert len(lines) == 2
    assert "1/4 chunks" in lines[0] and "2/4 chunks" in lines[1]


def test_progress_display_keeps_phase5_stages_visible_at_full_synthesis() -> None:
    lines: list[str] = []
    display = colab.ColabProgressDisplay({"chapters": [{"index": 1}]}, output=lines.append)

    for stage in ("synthesis_complete", "assembling", "encoding", "verifying", "publishing", "completed"):
        display.render(_progress_manifest(stage=stage, completed=4, total=4, chapter=1))

    assert all("100.0%" in line and "4/4 chunks" in line for line in lines)
    assert [line.rsplit("stage: ", 1)[1] for line in lines] == [
        "synthesis_complete", "assembling", "encoding", "verifying", "publishing", "completed"
    ]


def test_cuda_refusal_happens_before_workspace_mutation(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = sandbox_path
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created = []

    class NeverWorkspace:
        def __init__(self, root):
            created.append(root)

    monkeypatch.setattr(colab, "Workspace", NeverWorkspace)
    with pytest.raises(colab.ColabError, match="CUDA is unavailable"):
        colab.run_conversion(
            source,
            workspace_root=tmp_path / "workspace",
            output_dir=tmp_path / "output",
            cuda_check=lambda: (_ for _ in ()).throw(colab.ColabError("CUDA is unavailable")),
        )
    assert created == []


def test_cuda_factory_passes_device_cuda() -> None:
    calls = []

    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def inference_mode():
            return nullcontext()

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    factory = colab.make_cuda_kokoro_factory(torch_loader=lambda: Torch, kokoro_loader=lambda: SimpleNamespace(KPipeline=Pipeline))
    voice = factory("af_heart")
    assert calls == [{"lang_code": "a", "device": "cuda"}]
    assert voice.metadata.voice == "af_heart"


def test_fresh_run_uses_headless_pipeline_and_prints_output(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = sandbox_path
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    output = tmp_path / "out" / "book.m4b"
    state = {"job": {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending"}, "plan": None}

    class FakeWorkspace:
        def __init__(self, root):
            self.root = Path(root)

        def inspect_startup(self):
            return SimpleNamespace(state="no_active")

        def create_conversion(self, source_pdf, **kwargs):
            state["job"] = {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending", "conversion_id": "id"}
            return {"conversion_id": "id"}

        def conversion_path(self, conversion_id):
            path = self.root / "source.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            return self.root

        def persist_analysis(self, conversion_id, analysis):
            state["job"].update({"status": "analyzed", "chapter_plan_sha256": None})

        def read_job(self, conversion_id):
            return dict(state["job"])

        def load_analysis(self, conversion_id):
            return {"title": "Book", "chapter_candidates": []}

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]

        def persist_chapter_plan(self, conversion_id, plan):
            state["plan"] = plan
            state["job"]["chapter_plan_sha256"] = "present"

        def load_chapter_plan(self, conversion_id):
            return state["plan"]

        def configure_generation(self, conversion_id, *, tts, total_chunks):
            state["job"].update({"schema_version": 4, "tts": tts, "total_chunks": total_chunks, "status": "planned"})

    monkeypatch.setattr(colab, "Workspace", FakeWorkspace)
    monkeypatch.setattr(colab, "create_chapter_plan", lambda *args, **kwargs: {"mode": kwargs["mode"], "requested_count": kwargs.get("count"), "chapters": [{"index": 1, "start_offset": 0, "end_offset": 12}]})

    class FakeWorker:
        def __init__(self, workspace, conversion_id, *, engine_factory):
            assert callable(engine_factory)

        def run(self, *, full_pipeline):
            assert full_pipeline is True
            state["job"].update({"status": "completed", "stage": "completed", "output": {"path": str(output)}})
            return WorkerResult("completed", 1, 1, 1)

    result = colab.run_conversion(source, workspace_root=tmp_path / "workspace", output_dir=tmp_path / "out", cuda_check=lambda: None, analyzer=lambda *args, **kwargs: {"title": "Book", "chapter_candidates": [], "cleaned_text": "Hello world.", "cleaned_map": [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]}, worker_class=FakeWorker, engine_factory=lambda *args, **kwargs: None)
    assert result == output


def test_resume_conflicting_source_does_not_start_worker(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = sandbox_path
    source = tmp_path / "other.pdf"
    source.write_bytes(b"different")

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "planned"})

        def read_job(self, conversion_id):
            return {"source_pdf_sha256": "0" * 64}

    monkeypatch.setattr(colab, "Workspace", Workspace)

    with pytest.raises(colab.ColabConflictError, match="different PDF"):
        colab.run_conversion(source, workspace_root=tmp_path / "workspace", output_dir=tmp_path / "output", cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))


def _matching_request_state(source: Path, *, status: str = "failed") -> tuple[dict, dict, int]:
    plan = {"mode": "original", "requested_count": None, "chapters": [{"index": 1, "start_offset": 0, "end_offset": 12}]}
    tts, total = colab._expected_tts("Hello world.", plan, "af_heart", 1.0)
    return {
        "schema_version": 4,
        "conversion_id": "id",
        "source_pdf_sha256": colab._sha256(source),
        "chapter_plan_sha256": "present",
        "cleaned_text_sha256": "present",
        "tts": tts,
        "total_chunks": total,
        "status": status,
        "stage": "synthesis",
        "output": None,
    }, plan, total


def test_matching_completed_output_returns_without_reconfiguration(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "complete.pdf"
    source.write_bytes(b"%PDF-test")
    job, plan, _ = _matching_request_state(source, status="completed")
    output = sandbox_path / "existing.m4b"
    job.update({"stage": "completed", "output": {"path": str(output)}})
    calls = []

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "completed"})

        def read_job(self, conversion_id):
            return dict(job)

        def load_chapter_plan(self, conversion_id):
            return plan

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", []

        def configure_generation(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("completed workspaces must not be reconfigured")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    assert colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", cuda_check=lambda: None) == output
    assert calls == []


def test_matching_interrupted_resume_reuses_conversion_and_skips_analysis(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "resume.pdf"
    source.write_bytes(b"%PDF-test")
    job, plan, total = _matching_request_state(source)
    configured = []
    ran = []
    output = sandbox_path / "resumed.m4b"

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "failed"})

        def read_job(self, conversion_id):
            assert conversion_id == "id"
            return dict(job)

        def load_chapter_plan(self, conversion_id):
            return plan

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", []

        def configure_generation(self, conversion_id, *, tts, total_chunks):
            configured.append((conversion_id, tts, total_chunks))

        def conversion_path(self, conversion_id):
            return sandbox_path

    class Worker:
        def __init__(self, workspace, conversion_id, *, engine_factory):
            assert conversion_id == "id"

        def run(self, *, full_pipeline):
            ran.append(full_pipeline)
            job.update({"status": "completed", "stage": "completed", "output": {"path": str(output)}})
            return WorkerResult("completed", total, total, 1)

    monkeypatch.setattr(colab, "Workspace", Workspace)
    analyzer = lambda *args, **kwargs: pytest.fail("matching resume must not analyze again")
    result = colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", cuda_check=lambda: None, analyzer=analyzer, worker_class=Worker, engine_factory=lambda *args, **kwargs: None)
    assert result == output
    assert len(configured) == 1 and configured[0][0] == "id" and configured[0][2] == total
    assert ran == [True]


def test_resume_voice_conflict_does_not_configure_or_run(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "conflict.pdf"
    source.write_bytes(b"%PDF-test")
    job, plan, _ = _matching_request_state(source)
    configured = []

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "failed"})

        def read_job(self, conversion_id):
            return dict(job)

        def load_chapter_plan(self, conversion_id):
            return plan

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", []

        def configure_generation(self, *args, **kwargs):
            configured.append(True)

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabConflictError, match="different voice"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", voice="am_adam", cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert configured == []


def test_start_new_deletes_reinspects_then_creates_in_order(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    output = sandbox_path / "out" / "book.m4b"
    events: list[str] = []
    state = {"job": {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending"}, "plan": None, "active": True}

    class Workspace:
        def __init__(self, root):
            self.root = Path(root)

        def inspect_startup(self):
            events.append("inspect_startup")
            if state["active"]:
                return SimpleNamespace(state="resumable", conversion_id="old", manifest={"status": "failed"}, reason=None)
            return SimpleNamespace(state="no_active", conversion_id=None, manifest=None, reason=None)

        def delete_active_state(self):
            events.append("delete_active_state")
            state["active"] = False
            return True

        def create_conversion(self, source_pdf, **kwargs):
            events.append("create_conversion")
            state["job"] = {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending", "conversion_id": "new"}
            return {"conversion_id": "new"}

        def conversion_path(self, conversion_id):
            path = self.root / "source.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            return self.root

        def persist_analysis(self, conversion_id, analysis):
            state["job"].update({"status": "analyzed", "chapter_plan_sha256": None})

        def read_job(self, conversion_id):
            if conversion_id == "old":
                return {"conversion_id": "old", "worker": None, "completed_chunks": [{"global_index": 0}]}
            return dict(state["job"])

        def load_analysis(self, conversion_id):
            return {"title": "Book", "chapter_candidates": []}

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]

        def persist_chapter_plan(self, conversion_id, plan):
            state["plan"] = plan
            state["job"]["chapter_plan_sha256"] = "present"

        def load_chapter_plan(self, conversion_id):
            return state["plan"]

        def configure_generation(self, conversion_id, *, tts, total_chunks):
            state["job"].update({"schema_version": 4, "tts": tts, "total_chunks": total_chunks, "status": "planned"})

    class Worker:
        def __init__(self, workspace, conversion_id, *, engine_factory):
            assert conversion_id == "new"

        def run(self, *, full_pipeline):
            state["job"].update({"status": "completed", "stage": "completed", "output": {"path": str(output)}})
            return WorkerResult("completed", 1, 1, 1)

    monkeypatch.setattr(colab, "Workspace", Workspace)
    monkeypatch.setattr(colab, "create_chapter_plan", lambda *args, **kwargs: {"mode": kwargs["mode"], "requested_count": kwargs.get("count"), "chapters": [{"index": 1, "start_offset": 0, "end_offset": 12}]})

    result = colab.run_conversion(
        source,
        workspace_root=sandbox_path / "workspace",
        output_dir=sandbox_path / "out",
        start_new=True,
        cuda_check=lambda: None,
        analyzer=lambda *args, **kwargs: {"title": "Book", "chapter_candidates": [], "cleaned_text": "Hello world.", "cleaned_map": [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]},
        worker_class=Worker,
        engine_factory=lambda *args, **kwargs: None,
    )

    assert result == output
    assert events == ["inspect_startup", "delete_active_state", "inspect_startup", "create_conversion"]


def test_start_new_rejects_bad_voice_before_deleting_anything(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[Path] = []

    class Workspace:
        def __init__(self, root):
            created.append(Path(root))

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "failed"}, reason=None)

        def delete_active_state(self):
            pytest.fail("an invalid request must never discard the active conversion")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="not approved"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", voice="zz_nobody", start_new=True, cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert created == []


def test_start_new_rejects_out_of_range_speed_before_deleting_anything(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[Path] = []

    class Workspace:
        def __init__(self, root):
            created.append(Path(root))

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "failed"}, reason=None)

        def delete_active_state(self):
            pytest.fail("an out-of-range speed must never discard the active conversion")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", speed=5.0, start_new=True, cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert created == []


def test_start_new_refuses_while_a_recorded_worker_is_alive(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    deleted: list[bool] = []

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "synthesizing"}, reason=None)

        def read_job(self, conversion_id):
            return {"conversion_id": "id", "worker": {"pid": 4321}, "completed_chunks": [{"global_index": 0}]}

        def delete_active_state(self):
            deleted.append(True)
            pytest.fail("a live worker must block the discard")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    monkeypatch.setattr(colab, "pid_is_alive", lambda pid: True)
    with pytest.raises(colab.ColabError, match="still running"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", start_new=True, cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert deleted == []


def test_start_new_refused_delete_raises_without_creating(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pdf_audiobook.workspace import WorkspaceError

    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[bool] = []

    class Workspace:
        def __init__(self, root):
            pass

        def inspect_startup(self):
            return SimpleNamespace(state="resumable", conversion_id="id", manifest={"status": "failed"}, reason=None)

        def read_job(self, conversion_id):
            return {"conversion_id": "id", "worker": None, "completed_chunks": []}

        def delete_active_state(self):
            raise WorkspaceError("active manifest must be a regular file")

        def create_conversion(self, *args, **kwargs):
            created.append(True)
            pytest.fail("a refused discard must not create a conversion")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="could not be discarded"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", start_new=True, cuda_check=lambda: None, worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert created == []


def test_start_new_must_be_a_real_boolean(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[Path] = []

    class Workspace:
        def __init__(self, root):
            created.append(Path(root))

        def inspect_startup(self):
            pytest.fail("a malformed start_new must not reach the workspace")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="boolean"):
        colab.run_conversion(source, workspace_root=sandbox_path / "workspace", output_dir=sandbox_path / "output", start_new="yes", cuda_check=lambda: pytest.fail("CUDA must not be probed"), worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"))
    assert created == []


def test_start_new_on_empty_workspace_is_a_no_op(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    output = sandbox_path / "out" / "book.m4b"
    events: list[str] = []
    state = {"job": {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending"}, "plan": None}

    class Workspace:
        def __init__(self, root):
            self.root = Path(root)

        def inspect_startup(self):
            events.append("inspect_startup")
            return SimpleNamespace(state="no_active", conversion_id=None, manifest=None, reason=None)

        def delete_active_state(self):
            pytest.fail("an empty workspace has nothing to discard")

        def create_conversion(self, source_pdf, **kwargs):
            events.append("create_conversion")
            state["job"] = {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending", "conversion_id": "id"}
            return {"conversion_id": "id"}

        def conversion_path(self, conversion_id):
            path = self.root / "source.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            return self.root

        def persist_analysis(self, conversion_id, analysis):
            state["job"].update({"status": "analyzed", "chapter_plan_sha256": None})

        def read_job(self, conversion_id):
            return dict(state["job"])

        def load_analysis(self, conversion_id):
            return {"title": "Book", "chapter_candidates": []}

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]

        def persist_chapter_plan(self, conversion_id, plan):
            state["plan"] = plan
            state["job"]["chapter_plan_sha256"] = "present"

        def load_chapter_plan(self, conversion_id):
            return state["plan"]

        def configure_generation(self, conversion_id, *, tts, total_chunks):
            state["job"].update({"schema_version": 4, "tts": tts, "total_chunks": total_chunks, "status": "planned"})

    class Worker:
        def __init__(self, workspace, conversion_id, *, engine_factory):
            assert conversion_id == "id"

        def run(self, *, full_pipeline):
            state["job"].update({"status": "completed", "stage": "completed", "output": {"path": str(output)}})
            return WorkerResult("completed", 1, 1, 1)

    monkeypatch.setattr(colab, "Workspace", Workspace)
    monkeypatch.setattr(colab, "create_chapter_plan", lambda *args, **kwargs: {"mode": kwargs["mode"], "requested_count": kwargs.get("count"), "chapters": [{"index": 1, "start_offset": 0, "end_offset": 12}]})

    result = colab.run_conversion(
        source,
        workspace_root=sandbox_path / "workspace",
        output_dir=sandbox_path / "out",
        start_new=True,
        cuda_check=lambda: None,
        analyzer=lambda *args, **kwargs: {"title": "Book", "chapter_candidates": [], "cleaned_text": "Hello world.", "cleaned_map": [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]},
        worker_class=Worker,
        engine_factory=lambda *args, **kwargs: None,
    )

    assert result == output
    assert events == ["inspect_startup", "create_conversion"]


def test_preview_voice_forwards_request_and_creates_the_parent_directory(sandbox_path: Path) -> None:
    target = sandbox_path / "previews" / "sample.wav"
    captured: dict = {}
    factory = lambda *args, **kwargs: None

    def fake_preview(voice, destination, *, settings, voice_loader):
        assert Path(destination).parent.is_dir()
        captured.update({"voice": voice, "destination": Path(destination), "settings": settings, "loader": voice_loader})
        return Path(destination)

    result = colab.preview_voice("bf_emma", target, speed=1.25, cuda_check=lambda: None, engine_factory=factory, preview=fake_preview)

    assert result == target
    assert captured["voice"] == "bf_emma"
    assert captured["settings"] == {"speed": 1.25}
    assert captured["loader"] is factory
    assert target.parent.is_dir()


def test_preview_voice_rejects_an_unapproved_voice_before_cuda(sandbox_path: Path) -> None:
    with pytest.raises(colab.ColabError, match="not approved"):
        colab.preview_voice(
            "zz_nobody",
            sandbox_path / "previews" / "sample.wav",
            cuda_check=lambda: pytest.fail("CUDA must not be probed for an unapproved voice"),
            engine_factory=lambda *args, **kwargs: pytest.fail("no engine may be built"),
            preview=lambda *args, **kwargs: pytest.fail("no preview may be rendered"),
        )
    assert not (sandbox_path / "previews").exists()


def test_preview_voice_rejects_an_out_of_range_speed_before_cuda(sandbox_path: Path) -> None:
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.preview_voice(
            "bf_emma",
            sandbox_path / "previews" / "sample.wav",
            speed=5.0,
            cuda_check=lambda: pytest.fail("CUDA must not be probed for an invalid speed"),
            engine_factory=lambda *args, **kwargs: pytest.fail("no engine may be built"),
            preview=lambda *args, **kwargs: pytest.fail("no preview may be rendered"),
        )
    assert not (sandbox_path / "previews").exists()


def test_real_cuda_factory_drives_generate_preview_to_a_valid_wav(sandbox_path: Path) -> None:
    from pdf_audiobook.audio import validate_wav
    from pdf_audiobook.preview_worker import generate_preview

    pipelines: list[dict] = []
    synthesis: list[dict] = []

    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def inference_mode():
            return nullcontext()

    class Pipeline:
        def __init__(self, **kwargs):
            pipelines.append(kwargs)

        def __call__(self, text, **kwargs):
            synthesis.append({"text": text, **kwargs})
            yield SimpleNamespace(audio=[0.05] * 2400)

    factory = colab.make_cuda_kokoro_factory(torch_loader=lambda: Torch, kokoro_loader=lambda: SimpleNamespace(KPipeline=Pipeline))
    target = sandbox_path / "previews" / "bf_emma.wav"

    result = colab.preview_voice("bf_emma", target, cuda_check=lambda: None, engine_factory=factory, preview=generate_preview)

    assert result == target and target.is_file()
    info = validate_wav(target, expected_sample_rate=24000)
    assert (info.sample_rate, info.channels, info.sample_width) == (24000, 1, 2)
    assert info.frames == 2400
    assert pipelines == [{"lang_code": "b", "device": "cuda"}]
    assert synthesis and synthesis[0]["voice"] == "bf_emma" and synthesis[0]["speed"] == 1.0


def test_main_preview_out_needs_no_pdf(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    target = sandbox_path / "preview.wav"
    calls: list[tuple] = []

    def fake_preview_voice(voice, out, *, speed=1.0):
        calls.append((voice, Path(out), speed))
        return Path(out)

    monkeypatch.setattr(colab, "preview_voice", fake_preview_voice)
    assert colab.main(["--preview-out", str(target), "--voice", "bf_emma"]) == 0
    assert capsys.readouterr().out.strip() == f"Preview WAV: {target}"
    assert calls == [("bf_emma", target, 1.0)]


def test_main_without_pdf_or_preview_reports_a_required_pdf(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(colab, "run_conversion", lambda *args, **kwargs: pytest.fail("no conversion may start without a PDF"))
    assert colab.main(["--voice", "bf_emma"]) == 2
    assert "a PDF path is required" in capsys.readouterr().out


def test_main_threads_start_new_into_run_conversion(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: list[dict] = []

    def fake_run_conversion(pdf, **kwargs):
        captured.append({"pdf": Path(pdf), **kwargs})
        return sandbox_path / "book.m4b"

    monkeypatch.setattr(colab, "run_conversion", fake_run_conversion)
    pdf = sandbox_path / "book.pdf"

    assert colab.main([str(pdf), "--start-new", "--voice", "bf_emma"]) == 0
    assert colab.main([str(pdf), "--voice", "bf_emma"]) == 0

    assert [call["start_new"] for call in captured] == [True, False]
    assert captured[0]["pdf"] == pdf and captured[0]["voice"] == "bf_emma"
    assert "Verified M4B:" in capsys.readouterr().out


def test_notebook_parses_and_generate_cell_does_not_reassign_chapter_count() -> None:
    import json

    notebook = Path(__file__).resolve().parents[1] / "colab" / "PDF_Audiobook_Colab.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    generate = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell.get("cell_type") == "code" and "pdf_audiobook.colab" in "".join(cell["source"]) and "--chapter-mode" in "".join(cell["source"])
    ]

    assert len(generate) == 1
    assert "chapter_count = None" not in generate[0]
    assert "chapter_count=None" not in generate[0]
    assert "--chapter-count" in generate[0]
    derivation = re.search(r"^(\w+) = chapter_count if chapter_mode == 'custom' else None$", generate[0], re.MULTILINE)
    assert derivation is not None
    assert derivation.group(1) != "chapter_count"
    assert f"'--chapter-count', str({derivation.group(1)})" in generate[0]


def _fresh_run_doubles(source: Path, output: Path, *, startup: list, events: list[str]):
    """Build Workspace/Worker doubles that carry one fresh conversion to completion."""

    state = {"job": {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending"}, "plan": None}
    remaining = list(startup)

    class Workspace:
        def __init__(self, root):
            self.root = Path(root)

        def inspect_startup(self):
            events.append("inspect_startup")
            return remaining.pop(0)

        def delete_active_state(self):
            events.append("delete_active_state")
            return True

        def create_conversion(self, source_pdf, **kwargs):
            events.append("create_conversion")
            state["job"] = {"schema_version": 2, "chapter_plan_sha256": None, "status": "pending", "conversion_id": "new"}
            return {"conversion_id": "new"}

        def conversion_path(self, conversion_id):
            path = self.root / "source.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            return self.root

        def persist_analysis(self, conversion_id, analysis):
            state["job"].update({"status": "analyzed", "chapter_plan_sha256": None})

        def read_job(self, conversion_id):
            return dict(state["job"])

        def load_analysis(self, conversion_id):
            return {"title": "Book", "chapter_candidates": []}

        def load_cleaned_artifacts(self, conversion_id):
            return "Hello world.", [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]

        def persist_chapter_plan(self, conversion_id, plan):
            state["plan"] = plan
            state["job"]["chapter_plan_sha256"] = "present"

        def load_chapter_plan(self, conversion_id):
            return state["plan"]

        def configure_generation(self, conversion_id, *, tts, total_chunks):
            state["job"].update({"schema_version": 4, "tts": tts, "total_chunks": total_chunks, "status": "planned"})

    class Worker:
        def __init__(self, workspace, conversion_id, *, engine_factory):
            assert conversion_id == "new"

        def run(self, *, full_pipeline):
            state["job"].update({"status": "completed", "stage": "completed", "output": {"path": str(output)}})
            return WorkerResult("completed", 1, 1, 1)

    return Workspace, Worker


def _run_fresh_conversion(source: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch, *, startup: list, events: list[str], **kwargs) -> Path:
    workspace_class, worker_class = _fresh_run_doubles(source, output_dir / "book.m4b", startup=startup, events=events)
    monkeypatch.setattr(colab, "Workspace", workspace_class)
    monkeypatch.setattr(colab, "create_chapter_plan", lambda *args, **options: {"mode": options["mode"], "requested_count": options.get("count"), "chapters": [{"index": 1, "start_offset": 0, "end_offset": 12}]})
    return colab.run_conversion(
        source,
        workspace_root=output_dir.parent / "workspace",
        output_dir=output_dir,
        cuda_check=lambda: None,
        analyzer=lambda *args, **options: {"title": "Book", "chapter_candidates": [], "cleaned_text": "Hello world.", "cleaned_map": [{"page": 1, "start": 0, "end": 12, "text": "Hello world."}]},
        worker_class=worker_class,
        engine_factory=lambda *args, **options: None,
        **kwargs,
    )


def test_start_new_on_an_invalid_workspace_discards_reinspects_and_warns(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    events: list[str] = []
    startup = [
        SimpleNamespace(state="invalid", conversion_id=None, manifest=None, reason="active.json is not valid JSON"),
        SimpleNamespace(state="no_active", conversion_id=None, manifest=None, reason=None),
    ]

    result = _run_fresh_conversion(source, sandbox_path / "out", monkeypatch, startup=startup, events=events, start_new=True)

    assert result == sandbox_path / "out" / "book.m4b"
    assert events == ["inspect_startup", "delete_active_state", "inspect_startup", "create_conversion"]
    printed = capsys.readouterr().out
    assert "The live-worker check could not be performed" in printed
    assert "may leave the old conversion tree orphaned on disk" in printed


def test_resumable_discard_message_names_the_conversion_and_completed_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    deleted: list[bool] = []

    class Workspace:
        def read_job(self, conversion_id):
            assert conversion_id == "conv-77"
            return {"worker": None, "completed_chunks": [{"global_index": 0}, {"global_index": 1}, {"global_index": 2}]}

        def delete_active_state(self):
            deleted.append(True)
            return True

    inspection = SimpleNamespace(state="resumable", conversion_id="conv-77", manifest={"status": "failed"}, reason=None)
    colab._discard_active_conversion(Workspace(), inspection)

    printed = capsys.readouterr().out
    assert "conv-77" in printed
    assert "3 completed chunk(s)" in printed
    assert deleted == [True]


class _CountingSpeed:
    """A stateful speed whose __float__ calls are counted."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        return self.value


def test_run_conversion_coerces_the_requested_speed_exactly_once(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    speed = _CountingSpeed(1.0)
    startup = [SimpleNamespace(state="no_active", conversion_id=None, manifest=None, reason=None)]

    result = _run_fresh_conversion(source, sandbox_path / "out", monkeypatch, startup=startup, events=[], speed=speed)

    assert result == sandbox_path / "out" / "book.m4b"
    assert speed.calls == 1


def test_preview_voice_coerces_the_requested_speed_exactly_once(sandbox_path: Path) -> None:
    speed = _CountingSpeed(1.0)
    captured: dict = {}

    def fake_preview(voice, destination, *, settings, voice_loader):
        captured.update(settings)
        return Path(destination)

    target = sandbox_path / "previews" / "sample.wav"
    assert colab.preview_voice("bf_emma", target, speed=speed, cuda_check=lambda: None, engine_factory=lambda *args, **kwargs: None, preview=fake_preview) == target
    assert speed.calls == 1
    assert captured == {"speed": 1.0}


def test_run_conversion_rejects_a_non_numeric_speed_without_touching_the_workspace(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[Path] = []

    class Workspace:
        def __init__(self, root):
            created.append(Path(root))

        def inspect_startup(self):
            pytest.fail("a non-numeric speed must not reach the workspace")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.run_conversion(
            source,
            workspace_root=sandbox_path / "workspace",
            output_dir=sandbox_path / "output",
            speed="fast",
            start_new=True,
            cuda_check=lambda: pytest.fail("CUDA must not be probed for a non-numeric speed"),
            worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"),
        )
    assert created == []


def test_preview_voice_rejects_a_non_numeric_speed_before_cuda(sandbox_path: Path) -> None:
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.preview_voice(
            "bf_emma",
            sandbox_path / "previews" / "sample.wav",
            speed=None,
            cuda_check=lambda: pytest.fail("CUDA must not be probed for a non-numeric speed"),
            engine_factory=lambda *args, **kwargs: pytest.fail("no engine may be built"),
            preview=lambda *args, **kwargs: pytest.fail("no preview may be rendered"),
        )
    assert not (sandbox_path / "previews").exists()


_UNCOERCIBLE_SPEEDS = [pytest.param(True, id="bool"), pytest.param(10**10000, id="huge_int")]


@pytest.mark.parametrize("speed", _UNCOERCIBLE_SPEEDS)
def test_run_conversion_rejects_an_uncoercible_speed_without_touching_the_workspace(speed, sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = sandbox_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    created: list[Path] = []

    class Workspace:
        def __init__(self, root):
            created.append(Path(root))

        def inspect_startup(self):
            pytest.fail("an uncoercible speed must not reach the workspace")

    monkeypatch.setattr(colab, "Workspace", Workspace)
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.run_conversion(
            source,
            workspace_root=sandbox_path / "workspace",
            output_dir=sandbox_path / "output",
            speed=speed,
            start_new=True,
            cuda_check=lambda: pytest.fail("CUDA must not be probed for an uncoercible speed"),
            worker_class=lambda *args, **kwargs: pytest.fail("worker must not run"),
        )
    assert created == []


@pytest.mark.parametrize("speed", _UNCOERCIBLE_SPEEDS)
def test_preview_voice_rejects_an_uncoercible_speed_before_cuda(speed, sandbox_path: Path) -> None:
    with pytest.raises(colab.ColabError, match="speed must be between"):
        colab.preview_voice(
            "bf_emma",
            sandbox_path / "previews" / "sample.wav",
            speed=speed,
            cuda_check=lambda: pytest.fail("CUDA must not be probed for an uncoercible speed"),
            engine_factory=lambda *args, **kwargs: pytest.fail("no engine may be built"),
            preview=lambda *args, **kwargs: pytest.fail("no preview may be rendered"),
        )
    assert not (sandbox_path / "previews").exists()

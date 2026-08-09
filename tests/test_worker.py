from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import uuid

from pdf_audiobook.tts import EngineMetadata, FakeVoice, SynthesisSettings, plan_chunks
from pdf_audiobook.chapters import select_chapter_range
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.workspace import Workspace


def _prepared() -> tuple[Path, Workspace, str, EngineMetadata]:
    root = Path("tests") / f".pytest-phase4-worker-extra-{uuid.uuid4().hex}"; root.mkdir()
    source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
    workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
    text = "First sentence. Second sentence. Third sentence."
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    settings = SynthesisSettings(chunk_mode="legacy"); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase4", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
    total = len(plan_chunks(text, plan["chapters"], metadata)); workspace.configure_generation(manifest["conversion_id"], tts={**metadata.as_dict(), "speed": 1.0, "chunk_cap": 900}, total_chunks=total)
    return root, workspace, manifest["conversion_id"], metadata


def _small_chunks(workspace: Workspace, conversion_id: str, metadata: EngineMetadata) -> int:
    text, _ = workspace.load_cleaned_artifacts(conversion_id); plan = workspace.load_chapter_plan(conversion_id)
    settings = {**metadata.settings, "chunk_cap": 16}; changed = {**metadata.as_dict(), "settings": settings, "chunk_cap": 16, "speed": 1.0}
    total = len(plan_chunks(text, plan["chapters"], changed, cap=16)); workspace.configure_generation(conversion_id, tts=changed, total_chunks=total)
    return total


class _CountingEngine:
    def __init__(self, workspace: Workspace | None = None, conversion_id: str | None = None):
        self.inner = FakeVoice(); self.calls = 0; self.workspace = workspace; self.conversion_id = conversion_id
    def synthesize(self, text: str):
        self.calls += 1; result = self.inner.synthesize(text)
        if self.calls == 1 and self.workspace is not None and self.conversion_id is not None:
            self.workspace.request_cancel(self.conversion_id)
        return result
    def close_voice(self): self.inner.close_voice()


def test_fake_worker_resumes_without_rewriting_completed_chunk() -> None:
    root = Path("tests") / f".pytest-phase4-worker-{uuid.uuid4().hex}"; root.mkdir()
    try:
        source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
        workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
        text = "First sentence. Second sentence."
        workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
        plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}], "warnings": []}
        workspace.persist_chapter_plan(manifest["conversion_id"], plan)
        settings = SynthesisSettings(); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase4", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
        total = len(plan_chunks(text, plan["chapters"], metadata)); workspace.configure_generation(manifest["conversion_id"], tts={**metadata.as_dict(), "speed": 1.0, "chunk_cap": 900}, total_chunks=total)
        worker = ConversionWorker(workspace, manifest["conversion_id"]); assert worker.run(engine=FakeVoice()).status == "completed"
        wav = next((root / "data" / "work" / manifest["conversion_id"] / "chunks").glob("*.wav")); before = wav.read_bytes(); mtime = wav.stat().st_mtime_ns
        assert worker.run(engine=FakeVoice()).status == "completed"
        assert wav.read_bytes() == before and wav.stat().st_mtime_ns == mtime
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_setting_change_resets_progress() -> None:
    root, workspace, conversion_id, metadata = _prepared()
    try:
        worker = ConversionWorker(workspace, conversion_id); worker.run(engine=FakeVoice())
        job = workspace.read_job(conversion_id); job["status"] = "cancelled"; job["worker"] = None
        from pdf_audiobook.workspace import atomic_write_json
        atomic_write_json(workspace.job_path(conversion_id), job)
        text, _ = workspace.load_cleaned_artifacts(conversion_id); plan = workspace.load_chapter_plan(conversion_id)
        changed = {**metadata.as_dict(), "settings": {**metadata.settings, "speed": 1.1, "chunk_cap": 16}, "speed": 1.1, "chunk_cap": 16}
        new_total = len(plan_chunks(text, plan["chapters"], changed, cap=16))
        assert new_total != job["total_chunks"]
        reset = workspace.configure_generation(conversion_id, tts=changed, total_chunks=new_total)
        assert reset["total_chunks"] == new_total and reset["completed_chunks"] == [] and reset["progress"] == {"completed": 0, "current": 0, "total": new_total}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_retry_cap_leaves_factual_failed_state() -> None:
    root, workspace, conversion_id, _ = _prepared()
    try:
        class Failing:
            def __init__(self): self.calls = 0
            def synthesize(self, _text): self.calls += 1; raise RuntimeError("expected")
            def close_voice(self): pass
        engine = Failing()
        try: ConversionWorker(workspace, conversion_id).run(engine=engine)
        except RuntimeError: pass
        else: raise AssertionError("expected bounded failure")
        assert engine.calls == 3
        assert workspace.read_job(conversion_id)["status"] == "failed"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unsafe_cancel_marker_is_rejected() -> None:
    root, workspace, conversion_id, _ = _prepared()
    try:
        marker = workspace.cancel_marker_path(conversion_id); marker.mkdir()
        from pdf_audiobook.workspace import UnsafePathError
        try: workspace.cancellation_requested(conversion_id)
        except UnsafePathError: pass
        else: raise AssertionError("directory marker was accepted")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wrong_path_and_tampered_same_path_are_regenerated() -> None:
    for tamper in (False, True):
        root, workspace, conversion_id, _ = _prepared()
        try:
            ConversionWorker(workspace, conversion_id).run(engine=FakeVoice())
            job = workspace.read_job(conversion_id); record = dict(job["completed_chunks"][0]); expected = workspace.conversion_path(conversion_id) / record["relative_path"]
            if tamper:
                expected.write_bytes(b"tampered")
            else:
                wrong = workspace.chunks_path(conversion_id) / "wrong.wav"; shutil.copy2(expected, wrong); record["relative_path"] = "chunks/wrong.wav"
                job["completed_chunks"][0] = record
                expected.unlink()
            job["status"] = "cancelled"; job["worker"] = None
            from pdf_audiobook.workspace import atomic_write_json
            atomic_write_json(workspace.job_path(conversion_id), job)
            engine = _CountingEngine(); ConversionWorker(workspace, conversion_id).run(engine=engine)
            assert engine.calls >= 1 and expected.is_file() and workspace.read_job(conversion_id)["status"] == "completed"
        finally:
            shutil.rmtree(root, ignore_errors=True)


def test_cancel_then_resume_preserves_first_chunk_bytes_and_mtime() -> None:
    root, workspace, conversion_id, metadata = _prepared()
    try:
        assert _small_chunks(workspace, conversion_id, metadata) >= 2
        first_engine = _CountingEngine(workspace, conversion_id)
        assert ConversionWorker(workspace, conversion_id).run(engine=first_engine).status == "cancelled"
        job = workspace.read_job(conversion_id); first = job["completed_chunks"][0]; path = workspace.conversion_path(conversion_id) / first["relative_path"]
        before, mtime = path.read_bytes(), path.stat().st_mtime_ns
        workspace.configure_generation(conversion_id, tts=job["tts"], total_chunks=job["total_chunks"])
        assert ConversionWorker(workspace, conversion_id).run(engine=_CountingEngine()).status == "completed"
        assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chapter_mode_cancel_then_resume_checkpoints_by_chapter() -> None:
    root = Path("tests") / f".pytest-phase4-worker-chapters-{uuid.uuid4().hex}"; root.mkdir()
    try:
        source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
        workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
        text = "First chapter. Second chapter. Third chapter."
        splits = [0, text.index("Second"), text.index("Third"), len(text)]
        chapters = [{"index": index, "title": f"Chapter {index}", "start_offset": splits[index - 1], "end_offset": splits[index], "start_page": 1, "end_page": 1, "source_type": "heading", "word_count": len(text[splits[index - 1]:splits[index]].split())} for index in range(1, 4)]
        workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
        plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": chapters, "warnings": []}
        workspace.persist_chapter_plan(manifest["conversion_id"], plan)
        settings = SynthesisSettings(chunk_mode="chapter"); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase4", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
        total = len(plan_chunks(text, chapters, metadata, cap=1)); tts = {**metadata.as_dict(), "speed": 1.0, "chunk_cap": 1}
        workspace.configure_generation(manifest["conversion_id"], tts=tts, total_chunks=total)
        first_engine = _CountingEngine(workspace, manifest["conversion_id"])
        assert ConversionWorker(workspace, manifest["conversion_id"]).run(engine=first_engine).status == "cancelled"
        job = workspace.read_job(manifest["conversion_id"])
        assert [(record["chapter_index"], record["global_index"], record["local_index"]) for record in job["completed_chunks"]] == [(1, 0, 0)]
        workspace.configure_generation(manifest["conversion_id"], tts=job["tts"], total_chunks=job["total_chunks"])
        assert ConversionWorker(workspace, manifest["conversion_id"]).run(engine=FakeVoice()).status == "completed"
        assert [(record["chapter_index"], record["global_index"], record["local_index"]) for record in workspace.read_job(manifest["conversion_id"])["completed_chunks"]] == [(1, 0, 0), (2, 1, 0), (3, 2, 0)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_worker_synthesizes_only_selected_reindexed_chapters() -> None:
    root = Path("tests") / f".pytest-phase6-worker-range-{uuid.uuid4().hex}"
    try:
        source = root / "book.pdf"; root.mkdir(); source.write_bytes(b"%PDF-1")
        workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
        text = "First sentence. Second sentence. Third sentence."
        workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
        splits = [0, text.index("Second"), text.index("Third"), len(text)]
        chapters = [{"index": index, "title": f"Chapter {index}", "start_offset": splits[index - 1], "end_offset": splits[index], "start_page": 1, "end_page": 1, "source_type": "heading", "word_count": len(text[splits[index - 1]:splits[index]].split())} for index in range(1, 4)]
        plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": chapters, "warnings": []}
        workspace.persist_chapter_plan(manifest["conversion_id"], plan)
        settings = SynthesisSettings(); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase6", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
        selected = select_chapter_range(plan, 2, 3)
        tts = {**metadata.as_dict(), "speed": 1.0, "chunk_cap": 900, "settings": {**metadata.settings, "chapter_start": 2, "chapter_end": 3}}
        total = len(plan_chunks(text, selected, metadata)); workspace.configure_generation(manifest["conversion_id"], tts=tts, total_chunks=total)

        class RecordingEngine:
            def __init__(self): self.inner = FakeVoice(); self.texts: list[str] = []
            def synthesize(self, value: str): self.texts.append(value); return self.inner.synthesize(value)
            def close_voice(self): self.inner.close_voice()

        worker = ConversionWorker(workspace, manifest["conversion_id"])
        assert [chunk.chapter_index for chunk in worker._planned_chunks(workspace.read_job(manifest["conversion_id"]))] == [1, 2]
        engine = RecordingEngine(); assert worker.run(engine=engine, full_pipeline=False).status == "completed"
        assert engine.texts == [text[splits[1]:splits[2]], text[splits[2]:splits[3]]]
    finally:
        shutil.rmtree(root, ignore_errors=True)

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import uuid
import wave

import pytest

import pdf_audiobook.worker as worker_module
from pdf_audiobook.tts import EngineMetadata, FakeVoice, SynthesisSettings, plan_chunks, plan_interactive_chunks
from pdf_audiobook.chapters import select_chapter_range
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.voice_plan import with_canonical_artifact_hash
from pdf_audiobook.workspace import ManifestError, Workspace, atomic_write_json
from pdf_audiobook.engine_catalog import CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT


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


def _prepared_chatterbox() -> tuple[Path, Workspace, str, dict, Path]:
    root = Path("tests") / f".pytest-chatterbox-worker-{uuid.uuid4().hex}"; root.mkdir()
    source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
    workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
    text = "First sentence. Second sentence."
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    reference = root / "reference.wav"
    with wave.open(str(reference), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b"\x00\x00" * (16000 * 6))
    status = workspace.store_chatterbox_reference(manifest["conversion_id"], reference, consent_confirmed=True)
    descriptor = workspace.load_chatterbox_reference(manifest["conversion_id"]).descriptor
    settings = SynthesisSettings(sample_rate=24000, chunk_cap=300, chunk_mode="legacy")
    metadata = EngineMetadata("chatterbox", "0.1.7", CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT, "unrecorded", "reference-wav", CHATTERBOX_SOURCE_COMMIT, descriptor["voice_checksum"], 24000, settings.as_dict())
    total = len(plan_chunks(text, plan["chapters"], metadata, cap=300))
    tts = {**metadata.as_dict(), "speed": 1.0, "chunk_cap": 300}
    workspace.configure_generation(manifest["conversion_id"], tts=tts, total_chunks=total)
    return root, workspace, manifest["conversion_id"], tts, reference


def test_v4_chatterbox_worker_binds_controlled_reference_and_input_hash() -> None:
    root, workspace, conversion_id, tts, reference = _prepared_chatterbox()
    try:
        calls: list[dict] = []
        class RecordingVoice:
            def synthesize(self, _text: str) -> bytes:
                return b"\x00\x00" * 1600
            def close_voice(self) -> None:
                pass
        def factory(voice: str, settings: SynthesisSettings, *, engine: str, **kwargs: object) -> RecordingVoice:
            calls.append({"voice": voice, "settings": settings, "engine": engine, **kwargs})
            return RecordingVoice()
        result = ConversionWorker(workspace, conversion_id, engine_factory=factory).run(full_pipeline=False)
        assert result.status == "completed" and len(calls) == 1
        assert calls[0]["engine"] == "chatterbox" and calls[0]["reference_wav"] == workspace.chatterbox_reference_path(conversion_id)
        job = workspace.read_job(conversion_id)
        planned = ConversionWorker(workspace, conversion_id, engine_factory=factory)._planned_chunks(job)
        assert all(record["input_hash"] == chunk.input_hash for record, chunk in zip(job["completed_chunks"], planned))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v4_chatterbox_worker_rejects_reference_mutation_before_factory() -> None:
    root, workspace, conversion_id, _tts, _reference = _prepared_chatterbox()
    try:
        workspace.chatterbox_reference_path(conversion_id).write_bytes(b"mutated")
        calls: list[object] = []
        with pytest.raises(ManifestError, match="reference"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *args, **kwargs: calls.append((args, kwargs))).run(full_pipeline=False)
        assert calls == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v4_chatterbox_builtin_worker_needs_no_reference_or_prompt() -> None:
    root = Path("tests") / f".pytest-chatterbox-worker-builtin-{uuid.uuid4().hex}"; root.mkdir()
    try:
        source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
        workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source); conversion_id = manifest["conversion_id"]
        text = "First sentence. Second sentence."
        workspace.persist_analysis(conversion_id, {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
        plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}], "warnings": []}
        workspace.persist_chapter_plan(conversion_id, plan)
        settings = SynthesisSettings(sample_rate=24000, chunk_cap=300, chunk_mode="legacy")
        metadata = EngineMetadata("chatterbox", "0.1.7", CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT, "unrecorded", "builtin", "bundled", "unrecorded", 24000, settings.as_dict())
        tts = {**metadata.as_dict(), "speed": 1.0, "chunk_cap": 300}
        workspace.configure_generation(conversion_id, tts=tts, total_chunks=1)
        calls = []
        class Voice:
            def synthesize(self, _text): return b"\0\0" * 1600
            def close_voice(self): pass
        def factory(voice, settings, *, engine, **kwargs): calls.append((voice, engine, kwargs)); return Voice()
        result = ConversionWorker(workspace, conversion_id, engine_factory=factory).run(full_pipeline=False)
        assert result.status == "completed" and calls == [("builtin", "chatterbox", {})]
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _CountingEngine:
    def __init__(self, workspace: Workspace | None = None, conversion_id: str | None = None):
        self.inner = FakeVoice(); self.calls = 0; self.workspace = workspace; self.conversion_id = conversion_id
    def synthesize(self, text: str):
        self.calls += 1; result = self.inner.synthesize(text)
        if self.calls == 1 and self.workspace is not None and self.conversion_id is not None:
            self.workspace.request_cancel(self.conversion_id)
        return result
    def close_voice(self): self.inner.close_voice()


def _prepared_v5() -> tuple[Path, Workspace, str, str, dict, dict, dict[str, dict], dict]:
    root = Path("tests") / f".pytest-phase5-worker-{uuid.uuid4().hex}"; root.mkdir()
    source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
    workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
    text = "Narrator speaks. Alice replies! Bob waits."
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    chapter_plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], chapter_plan)
    alice_start, bob_start = text.index("Alice"), text.index("Bob")
    voice_plan = with_canonical_artifact_hash({
        "schema_version": 1, "artifact": "voice-plan", "revision": 1,
        "source_pdf_sha256": manifest["source_pdf_sha256"], "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapter_plan_sha256": workspace.read_job(manifest["conversion_id"])["chapter_plan_sha256"], "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "cast": [
            {"cast_id": "narrator", "display_label": "Narrator", "role": "narrator", "relationship": "third_person", "voice_id": "voice-a", "voice_settings": {"speed": 1.0}},
            {"cast_id": "alice", "display_label": "Alice", "role": "character", "relationship": "separate_from_narrator", "voice_id": "voice-b", "voice_settings": {"speed": 1.2}},
            {"cast_id": "bob", "display_label": "Bob", "role": "character", "relationship": "separate_from_narrator", "voice_id": "voice-c", "voice_settings": {"speed": 0.8}},
        ], "aliases": [],
        "chapters": [{"chapter_index": 1, "source_start": 0, "source_end": len(text), "source_page_start": 1, "source_page_end": 1, "spans": [
            {"span_id": "s1", "source_start": 0, "source_end": alice_start, "type": "narration", "speaker_id": "narrator", "confidence": {"score": 0.2, "band": "high", "reasons": ["fixture"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
            {"span_id": "s2", "source_start": alice_start, "source_end": bob_start, "type": "dialogue", "speaker_id": "alice", "confidence": {"score": 0.9, "band": "low", "reasons": ["fixture"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
            {"span_id": "s3", "source_start": bob_start, "source_end": len(text), "type": "narration", "speaker_id": "bob", "confidence": {"score": 0.5, "band": "medium", "reasons": []}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
        ]}],
        "unresolved_policy": {"mode": "narrator", "accepted_by_user": False, "accepted_at": None},
        "approval": {"state": "approved", "approved_at": "2026-01-01T00:00:00Z", "approved_revision": 1},
    })
    workspace.persist_voice_plan(manifest["conversion_id"], voice_plan)
    analysis = with_canonical_artifact_hash({
        "schema_version": 1, "artifact": "speaker-analysis", "revision": 1,
        "source_pdf_sha256": manifest["source_pdf_sha256"], "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chapter_plan_sha256": workspace.read_job(manifest["conversion_id"])["chapter_plan_sha256"], "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "characters": [{"character_id": "alice", "canonical_label": "Alice", "aliases": [], "line_count": 1, "quote_count": 0}, {"character_id": "bob", "canonical_label": "Bob", "aliases": [], "line_count": 1, "quote_count": 0}],
        "spans": [{"span_id": "m1", "chapter_index": 1, "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": None, "confidence": {"score": 0.5, "band": "medium", "reasons": []}, "provenance": {"source": "fake"}}],
        "warnings": [],
    })
    workspace.persist_speaker_analysis(manifest["conversion_id"], analysis)
    facts = {voice: {"id": voice, "engine": "fake", "package": "builtin", "package_version": "builtin", "model": "fake-model", "model_revision": "1", "model_checksum": "model", "voice_version": "1", "voice_checksum": "voice", "sample_rate": 24000, "enabled": True} for voice in ("voice-a", "voice-b", "voice-c")}
    tts = {"engine": "fake", "package_version": "builtin", "model": "fake-model", "model_revision": "1", "model_checksum": "model", "voice": "voice-a", "voice_version": "1", "voice_checksum": "voice", "sample_rate": 24000, "settings": {}, "speed": 1.0, "chunk_cap": 900}
    return root, workspace, manifest["conversion_id"], text, voice_plan, facts, tts


def _configure_v5(workspace: Workspace, conversion_id: str, text: str, voice_plan: dict, facts: dict[str, dict], tts: dict, revision: str = "a" * 64) -> int:
    total = len(plan_interactive_chunks(text, voice_plan, facts, revision))
    workspace.configure_interactive_generation(conversion_id, tts=tts, total_chunks=total, voice_registry_revision=revision)
    return total


class _RecordedVoice:
    def __init__(self, voice: str, settings: SynthesisSettings):
        self.inner = FakeVoice(voice, settings); self.voice = voice; self.speed = settings.speed; self.closed = 0
    @property
    def metadata(self): return self.inner.metadata
    def synthesize(self, text: str): return self.inner.synthesize(text)
    def close_voice(self): self.closed += 1; self.inner.close_voice()


def test_v5_dispatches_cast_settings_and_releases_lru_cache(monkeypatch) -> None:
    root, workspace, conversion_id, text, voice_plan, facts, tts = _prepared_v5()
    try:
        monkeypatch.setattr(worker_module, "get_generation_facts", lambda voice: facts[voice]); monkeypatch.setattr(worker_module, "registry_revision", lambda: "a" * 64)
        _configure_v5(workspace, conversion_id, text, voice_plan, facts, tts)
        made: list[_RecordedVoice] = []
        def factory(voice, settings, *, engine):
            loaded = _RecordedVoice(voice, settings); made.append(loaded); return loaded
        result = ConversionWorker(workspace, conversion_id, engine_factory=factory).run(full_pipeline=False)
        assert result.status == "completed" and [(item.voice, item.speed) for item in made] == [("voice-a", 1.0), ("voice-b", 1.2), ("voice-c", 0.8)]
        assert [item.closed for item in made] == [1, 1, 1]
        assert [record["global_index"] for record in workspace.read_job(conversion_id)["completed_chunks"]] == [0, 1, 2]
    finally: shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("cancel_point", ["before_load", "after_load", "synthesis", "before_eviction"])
def test_v5_cancellation_checks_surround_load_eviction_and_synthesis(monkeypatch, cancel_point: str) -> None:
    root, workspace, conversion_id, text, voice_plan, facts, tts = _prepared_v5()
    try:
        monkeypatch.setattr(worker_module, "get_generation_facts", lambda voice: facts[voice]); monkeypatch.setattr(worker_module, "registry_revision", lambda: "a" * 64)
        _configure_v5(workspace, conversion_id, text, voice_plan, facts, tts)
        made: list[_RecordedVoice] = []
        if cancel_point == "before_load": workspace.request_cancel(conversion_id)
        def factory(voice, settings, *, engine):
            loaded = _RecordedVoice(voice, settings); made.append(loaded)
            if cancel_point == "after_load" or (cancel_point == "before_eviction" and voice == "voice-c"):
                workspace.request_cancel(conversion_id)
            return loaded
        class SynthCancel(_RecordedVoice):
            def synthesize(self, text):
                result = super().synthesize(text); workspace.request_cancel(conversion_id); return result
        def synth_factory(voice, settings, *, engine):
            if cancel_point == "synthesis":
                loaded = SynthCancel(voice, settings); made.append(loaded); return loaded
            return factory(voice, settings, engine=engine)
        result = ConversionWorker(workspace, conversion_id, engine_factory=synth_factory).run(full_pipeline=False)
        assert result.status == "cancelled" and all(item.closed == 1 for item in made)
        if cancel_point == "before_load": assert made == []
    finally: shutil.rmtree(root, ignore_errors=True)


def test_v5_retries_and_preserves_safe_failure(monkeypatch) -> None:
    root, workspace, conversion_id, text, voice_plan, facts, tts = _prepared_v5()
    try:
        monkeypatch.setattr(worker_module, "get_generation_facts", lambda voice: facts[voice]); monkeypatch.setattr(worker_module, "registry_revision", lambda: "a" * 64)
        _configure_v5(workspace, conversion_id, text, voice_plan, facts, tts)
        class Failing:
            def __init__(self): self.calls = 0; self.closed = 0; self.metadata = FakeVoice("voice-a", SynthesisSettings()).metadata
            def synthesize(self, _text): self.calls += 1; raise RuntimeError("secret detail")
            def close_voice(self): self.closed += 1
        failing = Failing()
        with pytest.raises(RuntimeError, match="chunk 0 failed after 3 attempts"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *_args, **_kwargs: failing).run(full_pipeline=False)
        job = workspace.read_job(conversion_id); assert failing.calls == 3 and failing.closed == 1 and job["status"] == "failed" and job["error"] == "chunk 0 failed after 3 attempts"
    finally: shutil.rmtree(root, ignore_errors=True)


def test_v5_strict_bindings_and_approved_plan(monkeypatch) -> None:
    root, workspace, conversion_id, text, voice_plan, facts, tts = _prepared_v5()
    try:
        monkeypatch.setattr(worker_module, "get_generation_facts", lambda voice: facts[voice]); monkeypatch.setattr(worker_module, "registry_revision", lambda: "b" * 64)
        _configure_v5(workspace, conversion_id, text, voice_plan, facts, tts, "a" * 64)
        with pytest.raises(ManifestError, match="voice registry revision"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *_args, **_kwargs: None).run(full_pipeline=False)
        job = workspace.read_job(conversion_id)
        atomic_write_json(workspace.job_path(conversion_id), {**job, "voice_plan_sha256": "b" * 64})
        monkeypatch.setattr(worker_module, "registry_revision", lambda: "a" * 64)
        with pytest.raises(ManifestError, match="voice plan"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *_args, **_kwargs: None).run(full_pipeline=False)
        atomic_write_json(workspace.job_path(conversion_id), {**job, "speaker_analysis_sha256": "c" * 64})
        with pytest.raises(ManifestError, match="speaker analysis"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *_args, **_kwargs: None).run(full_pipeline=False)
        draft = {**voice_plan, "approval": {**voice_plan["approval"], "state": "draft", "approved_at": None, "approved_revision": None}}
        atomic_write_json(workspace.conversion_path(conversion_id) / "voice-plan.json", with_canonical_artifact_hash(draft))
        monkeypatch.setattr(worker_module, "registry_revision", lambda: "a" * 64)
        with pytest.raises(ManifestError, match="approved voice plan"):
            ConversionWorker(workspace, conversion_id, engine_factory=lambda *_args, **_kwargs: None).run(full_pipeline=False)
    finally: shutil.rmtree(root, ignore_errors=True)


def test_v5_semantic_reuse_rewrites_current_manifest_and_rejects_bad_audio(monkeypatch) -> None:
    root, workspace, conversion_id, text, voice_plan, facts, tts = _prepared_v5()
    try:
        monkeypatch.setattr(worker_module, "get_generation_facts", lambda voice: facts[voice]); revision = ["a" * 64]; monkeypatch.setattr(worker_module, "registry_revision", lambda: revision[0])
        _configure_v5(workspace, conversion_id, text, voice_plan, facts, tts, revision[0])
        made: list[_RecordedVoice] = []
        def factory(voice, settings, *, engine):
            loaded = _RecordedVoice(voice, settings); made.append(loaded); return loaded
        worker = ConversionWorker(workspace, conversion_id, engine_factory=factory); assert worker.run(full_pipeline=False).status == "completed"
        previous = workspace.read_job(conversion_id); paths = [workspace.conversion_path(conversion_id) / r["relative_path"] for r in previous["completed_chunks"]]
        revised_plan = with_canonical_artifact_hash({**voice_plan, "revision": 2, "approval": {**voice_plan["approval"], "approved_revision": 2}})
        workspace.persist_voice_plan(conversion_id, revised_plan); atomic_write_json(workspace.job_path(conversion_id), {**previous, "status": "cancelled", "stage": "cancelled", "worker": None})
        revision[0] = "b" * 64; _configure_v5(workspace, conversion_id, text, revised_plan, facts, tts, revision[0]); made.clear()
        assert worker.run(full_pipeline=False).attempts == 0 and not made
        current = workspace.read_job(conversion_id); assert all(record["input_hash"] == chunk.input_hash for record, chunk in zip(current["completed_chunks"], worker._planned_chunks(current)))
        tampered_records = [{**current["completed_chunks"][0], "wav_sha256": "0" * 64}, *current["completed_chunks"][1:]]
        atomic_write_json(workspace.job_path(conversion_id), {**current, "status": "cancelled", "stage": "cancelled", "worker": None, "completed_chunks": tampered_records, "progress": {"completed": 3, "current": 3, "total": 3}})
        paths[0].write_bytes(b"bad")
        _configure_v5(workspace, conversion_id, text, revised_plan, facts, tts, revision[0]); made.clear(); assert worker.run(full_pipeline=False).attempts == 1 and len(made) == 1
    finally: shutil.rmtree(root, ignore_errors=True)


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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import uuid

import pytest

from pdf_audiobook.m4b import (
    AssemblyResult,
    ChapterTiming,
    M4BError,
    Phase5Cancelled,
    ToolUnavailable,
    assemble_chapters,
    build_ffmetadata,
    encode_m4b,
    escape_ffmetadata,
    publish_verified_output,
    verify_m4b,
    finalize_conversion,
)
from pdf_audiobook.audio import write_pcm_wav
from pdf_audiobook.tts import EngineMetadata, FakeVoice, SynthesisSettings, plan_chunks
from pdf_audiobook.chapters import select_chapter_range
from pdf_audiobook.tts import plan_interactive_chunks
from pdf_audiobook.voice_plan import with_canonical_artifact_hash
from pdf_audiobook.worker import ConversionWorker
from pdf_audiobook.workspace import Workspace
from pdf_audiobook.workspace import atomic_write_json


def _prepared(text: str, chapters: list[dict[str, object]], *, cap: int = 900, chunk_mode: str = "legacy", chapter_start: int | None = None, chapter_end: int | None = None) -> tuple[Path, Workspace, str]:
    root = Path("tests") / f".pytest-phase5-m4b-{uuid.uuid4().hex}"
    try:
        return _prepared_inner(root, text, chapters, cap=cap, chunk_mode=chunk_mode, chapter_start=chapter_start, chapter_end=chapter_end)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _prepared_inner(root: Path, text: str, chapters: list[dict[str, object]], *, cap: int = 900, chunk_mode: str = "legacy", chapter_start: int | None = None, chapter_end: int | None = None) -> tuple[Path, Workspace, str]:
    root.mkdir()
    source = root / "book.pdf"; source.write_bytes(b"%PDF-1")
    workspace = Workspace(root / "data"); manifest = workspace.create_conversion(source)
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": chapters, "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    settings = SynthesisSettings(chunk_cap=cap, chunk_mode=chunk_mode); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase5", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
    selected = select_chapter_range(plan, chapter_start, chapter_end)
    tts = {**metadata.as_dict(), "speed": 1.0, "chunk_cap": cap}
    if chapter_start is not None or chapter_end is not None:
        tts["settings"] = {**tts["settings"], "chapter_start": chapter_start, "chapter_end": chapter_end}
    total = len(plan_chunks(text, selected, metadata, cap=cap)); workspace.configure_generation(manifest["conversion_id"], tts=tts, total_chunks=total)
    ConversionWorker(workspace, manifest["conversion_id"]).run(engine=FakeVoice(settings=settings))
    return root, workspace, manifest["conversion_id"]


def test_frame_exact_pauses_and_measured_chapter_boundaries() -> None:
    text = "One sentence.\n\nTwo sentence."
    split = text.index("Two")
    chapters = [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "original", "word_count": 2}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "original", "word_count": 2}]
    root, workspace, conversion_id = _prepared(text, chapters)
    try:
        assembly = assemble_chapters(workspace, conversion_id)
        measured = __import__("pdf_audiobook.m4b", fromlist=["_recorded_chunks"])._recorded_chunks(workspace, conversion_id)[0]
        first_ms = round(measured[0][2].frames * 1000 / 24000); second_ms = round(measured[1][2].frames * 1000 / 24000)
        assert assembly.frames == sum((item[2].frames for item in measured)) + round(24000 * 750 / 1000)
        assert [(item.start_ms, item.end_ms) for item in assembly.chapters] == [(0, first_ms), (first_ms + 750, first_ms + 750 + second_ms)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_m4b_assembly_uses_only_selected_chapter_titles_and_count() -> None:
    text = "One sentence. Two sentence. Three sentence."
    splits = [0, text.index("Two"), text.index("Three"), len(text)]
    chapters = [{"index": index, "title": f"Title {index}", "start_offset": splits[index - 1], "end_offset": splits[index], "start_page": 1, "end_page": 1, "source_type": "original", "word_count": 2} for index in range(1, 4)]
    root, workspace, conversion_id = _prepared(text, chapters, chapter_start=2, chapter_end=3)
    try:
        assembly = assemble_chapters(workspace, conversion_id)
        assert [chapter.title for chapter in assembly.chapters] == ["Title 2", "Title 3"]
        assert len(assembly.chapters) == 2
        recorded = __import__("pdf_audiobook.m4b", fromlist=["_recorded_chunks"])._recorded_chunks(workspace, conversion_id)
        assert [chunk.chapter_index for chunk, _, _ in recorded[0]] == [1, 2]
        assert [chapter["title"] for chapter in recorded[1]["chapters"]] == ["Title 2", "Title 3"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chapter_mode_has_no_intra_chapter_pause() -> None:
    text = "One sentence. Two sentence."
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}]
    root, workspace, conversion_id = _prepared(text, chapters, cap=1, chunk_mode="chapter")
    try:
        assembly = assemble_chapters(workspace, conversion_id)
        records = __import__("pdf_audiobook.m4b", fromlist=["_recorded_chunks"])._recorded_chunks(workspace, conversion_id)[0]
        assert len(records) == 1 and assembly.frames == records[0][2].frames
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_metadata_escape_and_pause_markers() -> None:
    assert escape_ffmetadata(r"A\B=C;#D\nE") == r"A\\B\=C\;\#D\\nE"
    assert "START=0" in build_ffmetadata([ChapterTiming(1, "A", 0, 24000, 24000)])
    with pytest.raises(M4BError):
        build_ffmetadata([ChapterTiming(1, "bad", 10, 10, 1000)])


def test_metadata_and_verification_use_contiguous_pause_boundary(monkeypatch) -> None:
    chapters = [ChapterTiming(1, "One", 0, 39150, 1000), ChapterTiming(2, "Two", 39900, 50000, 1000)]
    metadata = build_ffmetadata(chapters)
    assert "START=0" in metadata and "END=39900" in metadata
    assert "START=39900" in metadata and "END=50000" in metadata

    root = Path("tests") / f".pytest-phase5-contiguous-{uuid.uuid4().hex}"; root.mkdir(); output = root / "out.m4b"; output.write_bytes(b"m4b")
    payload = {"format": {"duration": "50.0"}, "streams": [{"codec_name": "aac"}], "chapters": [{"start_time": "0", "end_time": "39.9", "tags": {"title": "One"}}, {"start_time": "39.9", "end_time": "50.0", "tags": {"title": "Two"}}]}
    monkeypatch.setattr("pdf_audiobook.m4b.discover_tool", lambda name, *_: name)
    def runner(argv: list[str], **_kwargs):
        if "ffprobe" in argv[0]: return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="")
    try:
        assert verify_m4b(output, chapters, command_runner=runner).chapter_count == 2
        payload["chapters"][0]["end_time"] = "39.8"
        with pytest.raises(M4BError): verify_m4b(output, chapters, command_runner=runner)
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(("text", "cap", "pause_ms"), [("One sentence. Two sentence.", 16, 150), ("One sentence.\n\nTwo sentence.", 100, 400)])
def test_ordinary_and_paragraph_pause_frames_are_exact(text: str, cap: int, pause_ms: int) -> None:
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split())}]
    root, workspace, conversion_id = _prepared(text, chapters, cap=cap)
    try:
        assembly = assemble_chapters(workspace, conversion_id)
        records = __import__("pdf_audiobook.m4b", fromlist=["_recorded_chunks"])._recorded_chunks(workspace, conversion_id)[0]
        assert assembly.frames == sum(item[2].frames for item in records) + round(24000 * pause_ms / 1000)
        assert assembly.chapters[0].start_frame == 0 and assembly.chapters[0].end_frame == assembly.frames
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_encode_argv_and_tool_failure_are_injected() -> None:
    assembly = AssemblyResult(Path("assembly.wav"), 24000, 24000, (ChapterTiming(1, "Book", 0, 24000, 24000),))
    root = Path("tests") / f".pytest-phase5-encode-{uuid.uuid4().hex}"; root.mkdir(); (root / "assembly.wav").write_bytes(b"wav"); metadata = root / "metadata.txt"; metadata.write_text(";FFMETADATA1\n", encoding="utf-8")
    calls: list[list[str]] = []
    def runner(argv: list[str], **_kwargs):
        calls.append(argv); Path(argv[-1]).write_bytes(b"m4b"); return SimpleNamespace(returncode=0, stdout="")
    import pdf_audiobook.m4b as m4b
    old = m4b.discover_tool; m4b.discover_tool = lambda *_: "ffmpeg.exe"
    try:
        output = encode_m4b(AssemblyResult(root / "assembly.wav", 24000, 24000, assembly.chapters), metadata, root / "out.m4b", command_runner=runner)
        assert output.is_file() and "-f" in calls[0] and "ffmetadata" in calls[0] and "+faststart" in calls[0] and "loudnorm=I=-18:TP=-3:LRA=11" in calls[0]
    finally:
        m4b.discover_tool = old; shutil.rmtree(root, ignore_errors=True)


def test_verify_success_and_malformed_probe_decode_or_chapter_failures() -> None:
    root = Path("tests") / f".pytest-phase5-verify-{uuid.uuid4().hex}"; root.mkdir(); output = root / "out.m4b"; output.write_bytes(b"m4b")
    chapter = ChapterTiming(1, "Book", 0, 24000, 24000)
    payload = {"format": {"duration": "1.0"}, "streams": [{"codec_name": "aac"}], "chapters": [{"start_time": "0", "end_time": "1", "tags": {"title": "Book"}}]}
    def runner(argv: list[str], **_kwargs):
        if "ffprobe" in argv[0]: return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="")
    import pdf_audiobook.m4b as m4b
    old = m4b.discover_tool; m4b.discover_tool = lambda name, *_: name
    try:
        assert verify_m4b(output, [chapter], command_runner=runner).codec == "aac"
        for bad in ("{", json.dumps({**payload, "streams": [{"codec_name": "mp3"}]}), json.dumps({**payload, "chapters": []}), json.dumps({**payload, "format": {"duration": "nan"}})):
            payload["streams"] = [{"codec_name": "aac"}]; payload["chapters"] = [{"start_time": "0", "end_time": "1", "tags": {"title": "Book"}}]; payload["format"] = {"duration": "1.0"}
            def bad_runner(argv: list[str], **_kwargs):
                if "ffprobe" in argv[0]: return SimpleNamespace(returncode=0, stdout=bad)
                return SimpleNamespace(returncode=1, stdout="")
            with pytest.raises(M4BError): verify_m4b(output, [chapter], command_runner=bad_runner)
    finally:
        m4b.discover_tool = old; shutil.rmtree(root, ignore_errors=True)


def test_missing_tool_and_atomic_publication_facts(monkeypatch) -> None:
    with pytest.raises(ToolUnavailable):
        import pdf_audiobook.m4b as m4b
        old = m4b.shutil.which; m4b.shutil.which = lambda *_: None
        try: m4b.discover_tool("ffmpeg", "PDF_AUDIOBOOK_NO_SUCH")
        finally: m4b.shutil.which = old
    root = Path("tests") / f".pytest-phase5-publish-{uuid.uuid4().hex}"; root.mkdir(); source = root / "source.m4b"; source.write_bytes(b"verified")
    try:
        output = publish_verified_output(source, title="Book", conversion_id="12345678", duration_seconds=2.0, chapter_count=1, destination=root / "dest")
        assert output["size_bytes"] == source.stat().st_size and output["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest() and output["duration_seconds"] == 2.0
        assert Path(output["path"]).parent == (root / "dest").absolute()
        assert [path.absolute() for path in (root / "dest").glob("*.m4b")] == [Path(output["path"])]
        assert not (root / "dest" / "Book").exists()
        monkeypatch.setattr("pdf_audiobook.m4b.shutil.copyfileobj", lambda _source, target: target.write(b"corrupt"))
        with pytest.raises(M4BError): publish_verified_output(source, title="Corrupt", conversion_id="12345678", duration_seconds=2.0, chapter_count=1, destination=root / "corrupt-dest")
        assert not list((root / "corrupt-dest").rglob("*.m4b"))
        monkeypatch.undo()
        monkeypatch.delenv("PDF_AUDIOBOOK_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("USERPROFILE", str(root / "profile"))
        default_output = publish_verified_output(source, title="Default", conversion_id="87654321", duration_seconds=2.0, chapter_count=1)
        assert Path(default_output["path"]).parent == (root / "profile" / "Downloads").absolute()
        assert [path.absolute() for path in (root / "profile" / "Downloads").glob("*.m4b")] == [Path(default_output["path"])]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_explicit_destination_symlink_is_rejected_when_supported() -> None:
    root = Path("tests") / f".pytest-phase5-publish-link-{uuid.uuid4().hex}"; root.mkdir(); source = root / "source.m4b"; source.write_bytes(b"verified"); outside = root / "outside"; outside.mkdir(); link = root / "link"
    try:
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        with pytest.raises(M4BError): publish_verified_output(source, title="Book", conversion_id="12345678", duration_seconds=2.0, chapter_count=1, destination=link)
        assert not list(outside.rglob("*.m4b"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cancellation_and_unsafe_phase5_directory_fail_closed() -> None:
    text = "One sentence."
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}]
    root, workspace, conversion_id = _prepared(text, chapters)
    try:
        workspace.request_cancel(conversion_id)
        with pytest.raises(Phase5Cancelled): finalize_conversion(workspace, conversion_id, destination=root / "dest")
        assert workspace.read_job(conversion_id)["status"] == "cancelled" and not (root / "dest").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unsafe_phase5_working_path_is_rejected() -> None:
    text = "One sentence."
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}]
    root, workspace, conversion_id = _prepared(text, chapters)
    try:
        (workspace.conversion_path(conversion_id) / ".phase5").write_text("not a directory", encoding="utf-8")
        with pytest.raises(M4BError): assemble_chapters(workspace, conversion_id)
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(("field", "value"), [("input_hash", "0" * 64), ("relative_path", "chunks/missing.wav"), ("duration_seconds", 9.0), ("wav_sha256", "0" * 64)])
def test_tampered_chunk_record_is_rejected(field: str, value: object) -> None:
    text = "One sentence."
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}]
    root, workspace, conversion_id = _prepared(text, chapters)
    try:
        job = workspace.read_job(conversion_id); job["completed_chunks"][0][field] = value; atomic_write_json(workspace.job_path(conversion_id), job)
        with pytest.raises(M4BError): assemble_chapters(workspace, conversion_id)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_encode_failures_and_runner_start_failure_are_fail_closed(monkeypatch) -> None:
    root = Path("tests") / f".pytest-phase5-encode-fail-{uuid.uuid4().hex}"; root.mkdir()
    try:
        assembly_path = root / "assembly.wav"; assembly_path.write_bytes(b"wav"); metadata = root / "metadata.txt"; metadata.write_text(";FFMETADATA1\n", encoding="utf-8")
        assembly = AssemblyResult(assembly_path, 24000, 24000, (ChapterTiming(1, "Book", 0, 24000, 24000),))
        import pdf_audiobook.m4b as m4b
        monkeypatch.setattr(m4b, "discover_tool", lambda *_: "ffmpeg")
        for result in (SimpleNamespace(returncode=1), SimpleNamespace(returncode="0"), SimpleNamespace(returncode=0)):
            with pytest.raises(M4BError): encode_m4b(assembly, metadata, root / f"{uuid.uuid4().hex}.m4b", command_runner=lambda *_args, _result=result, **_kwargs: _result)
        def start_failure(*_args, **_kwargs): raise OSError("missing")
        with pytest.raises(M4BError): encode_m4b(assembly, metadata, root / "start-fail.m4b", command_runner=start_failure)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_probe_nonobject_decode_failure_and_timestamp_errors(monkeypatch) -> None:
    root = Path("tests") / f".pytest-phase5-probe-{uuid.uuid4().hex}"; root.mkdir()
    try:
        output = root / "out.m4b"; output.write_bytes(b"m4b"); chapter = ChapterTiming(1, "Book", 0, 24000, 24000)
        payload = {"format": {"duration": "1.0"}, "streams": [{"codec_name": "aac"}], "chapters": [{"start_time": "0", "end_time": "1", "tags": {"title": "Book"}}]}
        monkeypatch.setattr("pdf_audiobook.m4b.discover_tool", lambda name, *_: name)
        def runner(argv: list[str], **_kwargs):
            if "ffprobe" in argv[0]: return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
            return SimpleNamespace(returncode=1, stdout="")
        with pytest.raises(M4BError): verify_m4b(output, [chapter], command_runner=lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout="[]") if "ffprobe" in argv[0] else SimpleNamespace(returncode=0, stdout=""))
        with pytest.raises(M4BError): verify_m4b(output, [chapter], command_runner=runner)
        for replacement in ({"tags": {"title": "Wrong"}, "start_time": "0", "end_time": "1"}, {"tags": {"title": "Book"}, "start_time": "-1", "end_time": "1"}, {"tags": {"title": "Book"}, "start_time": "0.2", "end_time": "1"}):
            payload["chapters"] = [replacement]
            with pytest.raises(M4BError): verify_m4b(output, [chapter], command_runner=lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)) if "ffprobe" in argv[0] else SimpleNamespace(returncode=0, stdout=""))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_failure_does_not_publish(monkeypatch) -> None:
    text = "One sentence."
    chapters = [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}]
    root, workspace, conversion_id = _prepared(text, chapters)
    try:
        import pdf_audiobook.m4b as m4b
        assembly = AssemblyResult(root / "assembly.wav", 24000, 24000, (ChapterTiming(1, "Book", 0, 24000, 24000),))
        (root / "assembly.wav").write_bytes(b"wav")
        monkeypatch.setattr(m4b, "assemble_chapters", lambda *_: assembly)
        monkeypatch.setattr(m4b, "encode_m4b", lambda _a, _m, destination, **_kwargs: destination.write_bytes(b"m4b") or destination)
        monkeypatch.setattr(m4b, "verify_m4b", lambda *_args, **_kwargs: (_ for _ in ()).throw(M4BError("bad probe")))
        destination = root / "published"
        with pytest.raises(M4BError): m4b.finalize_conversion(workspace, conversion_id, destination=destination)
        assert not destination.exists() or not list(destination.rglob("*.m4b"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _v5_fixture(monkeypatch, *, thought: bool = False) -> tuple[object, dict, list[object], Path]:
    import pdf_audiobook.m4b as m4b

    text = "One. Two. Three."
    split = text.index("Two")
    chapter_plan = {
        "schema_version": 1,
        "mode": "original",
        "requested_count": None,
        "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "warnings": [],
        "chapters": [
            {"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "original", "word_count": 1},
            {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "original", "word_count": 2},
        ]
    }
    selected_span = {"span_id": "s2", "source_start": split, "source_end": len(text), "type": "thought" if thought else "narration", "speaker_id": "narrator"}
    if thought:
        selected_span["override"] = {}
    voice_plan = with_canonical_artifact_hash({
        "schema_version": 1,
        "artifact": "voice-plan",
        "revision": 2,
        "approval": {"state": "approved", "approved_revision": 2},
        "cast": [{"cast_id": "narrator", "voice_id": "af_heart", "voice_settings": {"speed": 1.0}}],
        "chapters": [
            {"chapter_index": 1, "source_start": 0, "source_end": split, "spans": [{"span_id": "s1", "source_start": 0, "source_end": split, "type": "narration", "speaker_id": "narrator"}]},
            {"chapter_index": 2, "source_start": split, "source_end": len(text), "spans": [selected_span]},
        ],
    })
    facts = {"af_heart": {"id": "af_heart", "engine": "kokoro", "package": "kokoro", "package_version": "0.9.4", "model": "model", "model_revision": "r1", "model_checksum": "m1", "voice_version": "v1", "voice_checksum": "w1", "sample_rate": 24000, "enabled": True}}
    registry = "a" * 64
    monkeypatch.setattr(m4b, "get_generation_facts", lambda voice_id: facts[voice_id])
    monkeypatch.setattr(m4b, "registry_revision", lambda: registry)
    chunks = plan_interactive_chunks(text, voice_plan, facts, registry, chapter_range=(2, 2), cap=5)
    root = Path("tests") / f".pytest-phase5-v5-{uuid.uuid4().hex}"
    root.mkdir()
    records = []
    for chunk in chunks:
        path = root / f"chunks/chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"
        info = write_pcm_wav(path, b"\0\0" * 240, 24000)
        relative = path.relative_to(root).as_posix()
        records.append({**chunk.manifest_record(relative, info.duration_seconds), "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    job = {
        "schema_version": 5,
        "tts": {"sample_rate": 24000, "chunk_cap": 5, "settings": {"chapter_start": 2, "chapter_end": 2}},
        "total_chunks": len(chunks),
        "completed_chunks": records,
        "voice_plan_sha256": voice_plan["canonical_artifact_sha256"],
        "voice_plan_revision": voice_plan["revision"],
        "speaker_analysis_sha256": "b" * 64,
        "cast_voice_ids": ["af_heart"],
        "voice_registry_revision": registry,
    }

    class FakeWorkspace:
        def read_job(self, _conversion_id): return job
        def load_cleaned_artifacts(self, _conversion_id): return text, []
        def load_chapter_plan(self, _conversion_id): return chapter_plan
        def load_voice_plan(self, _conversion_id): return voice_plan
        def load_speaker_analysis(self, _conversion_id): return {"canonical_artifact_sha256": "b" * 64, "revision": 3}
        def conversion_path(self, _conversion_id): return root

    return FakeWorkspace(), job, chunks, root


def test_v5_assembly_replans_selected_chapters_and_preserves_global_pcm_order(monkeypatch) -> None:
    workspace, job, chunks, root = _v5_fixture(monkeypatch)
    try:
        assembly = assemble_chapters(workspace, "conversion")
        assert [chunk.global_index for chunk in chunks] == [0, 1]
        assert [chapter.title for chapter in assembly.chapters] == ["Two"]
        assert len(assembly.chapters) == 1
        assert build_ffmetadata(assembly.chapters).count("[CHAPTER]") == 1
        assert assembly.frames == sum(240 for _ in chunks) + round(24000 * 150 / 1000)
        assert job["completed_chunks"][0]["source_start"] == chunks[0].source_start
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v5_approved_manual_thought_record_is_accepted(monkeypatch) -> None:
    workspace, job, _chunks, root = _v5_fixture(monkeypatch, thought=True)
    try:
        assemble_chapters(workspace, "conversion")
        assert job["completed_chunks"][0]["segment_type"] == "thought"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("field", ["input_hash", "audio_input_hash", "span_id", "speaker_id", "voice_id", "segment_type", "source_start", "source_end"])
def test_v5_tampered_chunk_metadata_is_rejected(monkeypatch, field: str) -> None:
    workspace, job, _chunks, root = _v5_fixture(monkeypatch)
    try:
        value = job["completed_chunks"][0][field]
        job["completed_chunks"][0][field] = (value + "x") if isinstance(value, str) else value + 1
        with pytest.raises(M4BError):
            assemble_chapters(workspace, "conversion")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v5_registry_and_wav_tampering_are_rejected(monkeypatch) -> None:
    workspace, job, _chunks, root = _v5_fixture(monkeypatch)
    try:
        job["voice_registry_revision"] = "c" * 64
        with pytest.raises(M4BError):
            assemble_chapters(workspace, "conversion")
        job["voice_registry_revision"] = "a" * 64
        path = root / job["completed_chunks"][0]["relative_path"]
        path.write_bytes(b"not-wav")
        with pytest.raises((M4BError, ValueError)):
            assemble_chapters(workspace, "conversion")
    finally:
        shutil.rmtree(root, ignore_errors=True)

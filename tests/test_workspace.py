from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid
import wave

import pytest

import pdf_audiobook.workspace as workspace_module
import pdf_audiobook.speaker_analysis as speaker_analysis_module

from pdf_audiobook.workspace import (
    INTERACTIVE_GENERATION_SCHEMA_VERSION,
    JOB_SCHEMA_VERSION,
    ManifestError,
    UnsafePathError,
    Workspace,
    WorkspaceError,
    atomic_write_json,
    atomic_write_text,
    copy_source_pdf,
    validate_job_manifest,
    validate_interactive_generation_manifest,
)
from pdf_audiobook.chatterbox_reference import REFERENCE_DESCRIPTOR_FILENAME, REFERENCE_WAV_FILENAME
from pdf_audiobook.voice_plan import canonical_json_bytes, canonical_json_text, with_canonical_artifact_hash
from pdf_audiobook.tts import FakeVoice
from pdf_audiobook.worker import ConversionWorker


def _interactive_tts() -> dict[str, object]:
    return {
        "engine": "fake",
        "package_version": "builtin",
        "model": "fake-model",
        "model_revision": "1",
        "model_checksum": None,
        "voice": "voice-neutral",
        "voice_version": "1",
        "voice_checksum": None,
        "sample_rate": 24000,
        "settings": {},
        "speed": 1.0,
        "chunk_cap": 900,
    }


@pytest.fixture
def tmp_path() -> Path:
    """Use a repository-local temporary path on restricted Windows hosts."""

    path = Path("tests") / f".pytest-workspace-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_create_conversion_streams_source_and_writes_manifests(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    payload = (b"%PDF-1.7\n" + bytes(range(256))) * 20000
    source.write_bytes(payload)
    workspace = Workspace(tmp_path / "data")

    manifest = workspace.create_conversion(source)

    conversion = workspace.work_root / manifest["conversion_id"]
    assert conversion.joinpath("source.pdf").read_bytes() == payload
    assert manifest["source_pdf_sha256"] == hashlib.sha256(payload).hexdigest()
    assert json.loads((conversion / "job.json").read_text(encoding="utf-8")) == manifest
    assert workspace.inspect_startup().state == "resumable"
    assert workspace.inspect_startup().manifest == manifest
    assert source.read_bytes() == payload


def test_create_refuses_second_conversion_while_active_state_exists(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"first")
    workspace = Workspace(tmp_path / "data")
    first = workspace.create_conversion(source)
    conversion = workspace.work_root / first["conversion_id"]
    active_before = workspace.active_path.read_bytes()
    job_before = (conversion / "job.json").read_bytes()
    source_before = (conversion / "source.pdf").read_bytes()

    source.write_bytes(b"second")
    with pytest.raises(WorkspaceError, match="active conversion"):
        workspace.create_conversion(source, conversion_id=str(uuid.uuid4()))

    assert workspace.active_path.read_bytes() == active_before
    assert (conversion / "job.json").read_bytes() == job_before
    assert (conversion / "source.pdf").read_bytes() == source_before


def test_copy_source_reports_bytes_and_hash_without_requiring_whole_read(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "nested" / "source.pdf"
    source.write_bytes(b"abcdef" * 100)
    digest, count = copy_source_pdf(source, destination, chunk_size=7)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert count == source.stat().st_size
    assert destination.read_bytes() == source.read_bytes()


def test_atomic_json_and_text_writes_are_utf8_and_replace_existing(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "state.json"
    text_path = tmp_path / "nested" / "cleaned.txt"
    atomic_write_json(json_path, {"title": "Café", "n": 1})
    atomic_write_json(json_path, {"title": "更新", "n": 2})
    atomic_write_text(text_path, "Résumé — chapter\n")
    assert json.loads(json_path.read_text(encoding="utf-8"))["title"] == "更新"
    assert text_path.read_text(encoding="utf-8") == "Résumé — chapter\n"
    assert not list(json_path.parent.glob(".*.tmp"))


def test_atomic_replace_retries_transient_permission_error(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.txt"
    real_replace = workspace_module.os.replace
    attempts = 0

    def replace_with_transient_error(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary lock")
        real_replace(source, destination)

    monkeypatch.setattr(workspace_module.os, "replace", replace_with_transient_error)
    monkeypatch.setattr(workspace_module.time, "sleep", lambda _: None)

    atomic_write_text(target, "updated\n")

    assert target.read_text(encoding="utf-8") == "updated\n"
    assert attempts == 2


@pytest.mark.parametrize(
    ("request_method", "marker_name"),
    [("request_cancel", "cancel.request"), ("request_voice_analysis_cancel", "voice-analysis.cancel")],
)
def test_cancel_marker_replace_retries_transient_permission_error(
    tmp_path: Path, monkeypatch, request_method: str, marker_name: str
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    real_replace = workspace_module.os.replace
    attempts = 0

    def replace_with_transient_error(source_path: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary lock")
        real_replace(source_path, destination)

    monkeypatch.setattr(workspace_module.os, "replace", replace_with_transient_error)
    monkeypatch.setattr(workspace_module.time, "sleep", lambda _: None)

    marker = getattr(workspace, request_method)(manifest["conversion_id"])

    assert marker.name == marker_name
    assert marker.read_text(encoding="ascii") == "cancel\n"
    assert attempts == 2


def test_atomic_replace_only_retries_permission_errors_and_is_bounded(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.txt"
    attempts = 0

    def always_permission_error(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("persistent lock")

    monkeypatch.setattr(workspace_module.os, "replace", always_permission_error)
    monkeypatch.setattr(workspace_module.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError, match="persistent lock"):
        atomic_write_text(target, "never committed\n")

    assert attempts == workspace_module._REPLACE_RETRY_ATTEMPTS
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp"))

    attempts = 0

    def non_permission_error(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("non-retryable failure")

    monkeypatch.setattr(workspace_module.os, "replace", non_permission_error)
    with pytest.raises(OSError, match="non-retryable failure"):
        atomic_write_text(target, "never committed\n")
    assert attempts == 1


def test_strict_manifest_rejects_malformed_unknown_and_unknown_schema() -> None:
    base = {
        "schema_version": JOB_SCHEMA_VERSION,
        "conversion_id": str(uuid.uuid4()),
        "original_display_filename": "book.pdf",
        "source_pdf_sha256": "0" * 64,
        "status": "pending",
        "stage": "workspace",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "cleaned_text_sha256": None,
        "chapter_plan_sha256": None,
        "warnings": [],
        "error": None,
    }
    assert validate_job_manifest(base) == base
    with pytest.raises(ManifestError):
        validate_job_manifest({**base, "extra": True})
    with pytest.raises(ManifestError):
        validate_job_manifest({**base, "schema_version": 999})
    with pytest.raises(ManifestError):
        validate_job_manifest({**base, "conversion_id": "../escape"})


def test_v1_job_manifest_migrates_atomically_on_read(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    legacy = {key: value for key, value in manifest.items() if key != "chapter_plan_sha256"}
    legacy["schema_version"] = 1
    atomic_write_json(conversion / "job.json", legacy)

    loaded = workspace.read_job(manifest["conversion_id"])

    assert loaded["schema_version"] == JOB_SCHEMA_VERSION == 2
    assert loaded["chapter_plan_sha256"] is None
    assert json.loads((conversion / "job.json").read_text(encoding="utf-8")) == loaded
    assert workspace.inspect_startup().state == "resumable"


def test_v1_extra_field_and_unknown_schema_fail_without_migration(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    legacy = {key: value for key, value in manifest.items() if key != "chapter_plan_sha256"}
    legacy["schema_version"] = 1
    atomic_write_json(conversion / "job.json", {**legacy, "extra": True})
    before = (conversion / "job.json").read_bytes()
    with pytest.raises(ManifestError):
        workspace.read_job(manifest["conversion_id"])
    assert (conversion / "job.json").read_bytes() == before
    atomic_write_json(conversion / "job.json", {**legacy, "schema_version": 99})
    with pytest.raises(ManifestError):
        workspace.read_job(manifest["conversion_id"])


def _analysis_for_workspace() -> dict[str, object]:
    return {
        "source_pdf_sha256": "0" * 64,
        "title": "Book",
        "cleaned_text": "cleaned text\n",
        "cleaned_map": [],
        "warnings": [],
    }


def _analysis_with_mapping() -> dict[str, object]:
    analysis = _analysis_for_workspace()
    analysis["cleaned_map"] = [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(analysis["cleaned_text"])}]
    return analysis


def _voice_plan_for_workspace(source_hash: str, chapter_hash: str, text: str, plan: dict) -> dict:
    alice_start = text.index("Alice")
    bob_start = text.index("Bob")
    artifact = {
        "schema_version": 1,
        "artifact": "voice-plan",
        "revision": 1,
        "source_pdf_sha256": source_hash,
        "cleaned_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chapter_plan_sha256": chapter_hash,
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "cast": [
            {"cast_id": "narrator", "display_label": "Narrator", "role": "narrator", "relationship": "third_person", "voice_id": "voice-neutral", "voice_settings": {"speed": 1.0}},
            {"cast_id": "alice", "display_label": "Alice", "role": "character", "relationship": "same_as_narrator", "voice_id": "voice-neutral", "voice_settings": {"speed": 1.1}},
            {"cast_id": "bob", "display_label": "Bob", "role": "character", "relationship": "separate_from_narrator", "voice_id": "voice-neutral", "voice_settings": {"speed": 0.9}},
        ],
        "aliases": [],
        "chapters": [
            {"chapter_index": 1, "source_start": 0, "source_end": bob_start, "source_page_start": 1, "source_page_end": 1, "spans": [
                {"span_id": "s1", "source_start": 0, "source_end": alice_start, "type": "narration", "speaker_id": "narrator", "confidence": {"score": 0.2, "band": "high", "reasons": ["fixture"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
                {"span_id": "s2", "source_start": alice_start, "source_end": bob_start, "type": "dialogue", "speaker_id": "alice", "confidence": {"score": 0.9, "band": "low", "reasons": ["fixture"]}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
            ]},
            {"chapter_index": 2, "source_start": bob_start, "source_end": len(text), "source_page_start": 2, "source_page_end": 2, "spans": [
                {"span_id": "s3", "source_start": bob_start, "source_end": len(text), "type": "narration", "speaker_id": "bob", "confidence": {"score": 0.5, "band": "medium", "reasons": []}, "provenance": {"source": "fake", "analysis_revision": 1}, "override": None},
            ]},
        ],
        "unresolved_policy": {"mode": "narrator", "accepted_by_user": False, "accepted_at": None},
        "approval": {"state": "approved", "approved_at": "2026-01-01T00:00:00Z", "approved_revision": 1},
    }
    return with_canonical_artifact_hash(artifact)


def _speaker_analysis_for_workspace(source_hash: str, chapter_hash: str, text: str, plan: dict) -> dict:
    split = text.index("Bob")
    artifact = {
        "schema_version": 1,
        "artifact": "speaker-analysis",
        "revision": 1,
        "source_pdf_sha256": source_hash,
        "cleaned_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chapter_plan_sha256": chapter_hash,
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "characters": [{"character_id": "alice", "canonical_label": "Alice", "aliases": [], "line_count": 1, "quote_count": 0}],
        "spans": [
            {"span_id": "machine-1", "chapter_index": 1, "source_start": 0, "source_end": 8, "type": "narration", "speaker_id": None, "confidence": {"score": 0.2, "band": "high", "reasons": ["fixture"]}, "provenance": {"source": "fake"}},
            {"span_id": "machine-2", "chapter_index": 2, "source_start": split, "source_end": len(text), "type": "narration", "speaker_id": None, "confidence": {"score": 0.8, "band": "low", "reasons": []}, "provenance": {"source": "fake", "quote_id": "q1"}},
        ],
        "warnings": ["incomplete machine attribution"],
    }
    return with_canonical_artifact_hash(artifact)


def _voice_analysis_status_for_workspace(job: dict, *, status: str = "queued", stage: str = "queued") -> dict:
    artifact = {
        "schema_version": 1,
        "artifact": "voice-analysis-status",
        "analysis_id": "12345678-1234-5678-9234-567812345678",
        "revision": 1,
        "source_pdf_sha256": job["source_pdf_sha256"],
        "cleaned_text_sha256": job["cleaned_text_sha256"],
        "chapter_plan_sha256": job["chapter_plan_sha256"],
        "chapter_plan_schema_version": 1,
        "analyzer": {"id": "fake", "version": "1", "model_hash": None},
        "status": status,
        "stage": stage,
        "progress": {"completed": 0, "total": 0},
        "cancel_requested": status in {"running", "cancelled"},
        "warnings": [],
        "error": {"code": "ANALYZER_FAILED", "message": "failed"} if status == "failed" else None,
        "started_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:01Z",
        "finished_at": "2026-01-01T00:00:02Z" if status in {"completed", "cancelled", "failed"} else None,
    }
    return with_canonical_artifact_hash(artifact)


def test_voice_plan_round_trip_is_canonical_atomic_and_does_not_mutate_job(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    text = "Narrator speaks. Alice replies! Bob waits."
    split = text.index("Bob")
    analysis = {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": split}, {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)}], "warnings": []}
    workspace.persist_analysis(manifest["conversion_id"], analysis)
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 3}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}], "warnings": []}
    planned = workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    job_before = workspace.read_job(manifest["conversion_id"])
    artifact = _voice_plan_for_workspace(manifest["source_pdf_sha256"], planned["chapter_plan_sha256"], text, plan)
    returned = workspace.persist_voice_plan(manifest["conversion_id"], artifact)
    path = workspace.conversion_path(manifest["conversion_id"]) / "voice-plan.json"
    assert returned == artifact
    assert path.read_bytes() == canonical_json_text(artifact).encode("utf-8")
    assert workspace.load_voice_plan(manifest["conversion_id"]) == artifact
    assert workspace.read_job(manifest["conversion_id"]) == job_before
    replacement = dict(artifact)
    replacement["revision"] = 2
    replacement["approval"] = {**artifact["approval"], "approved_revision": 2}
    replacement = with_canonical_artifact_hash(replacement)
    workspace.persist_voice_plan(manifest["conversion_id"], replacement)
    assert workspace.load_voice_plan(manifest["conversion_id"]) == replacement
    previous_bytes = path.read_bytes()
    invalid_replacement = dict(replacement)
    invalid_replacement["approval"] = {**replacement["approval"], "approved_revision": True}
    invalid_replacement = with_canonical_artifact_hash(invalid_replacement)
    with pytest.raises(ManifestError, match="invalid voice plan artifact"):
        workspace.persist_voice_plan(manifest["conversion_id"], invalid_replacement)
    assert path.read_bytes() == previous_bytes


def test_voice_plan_load_rejects_tampered_malformed_stale_and_unsafe_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    text = "Narrator speaks. Alice replies! Bob waits."
    split = text.index("Bob")
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": split}, {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 3}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}], "warnings": []}
    planned = workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    artifact = _voice_plan_for_workspace(manifest["source_pdf_sha256"], planned["chapter_plan_sha256"], text, plan)
    workspace.persist_voice_plan(manifest["conversion_id"], artifact)
    path = workspace.conversion_path(manifest["conversion_id"]) / "voice-plan.json"
    tampered = dict(artifact); tampered["revision"] = 99; atomic_write_json(path, tampered)
    with pytest.raises(ManifestError, match="invalid voice plan artifact"):
        workspace.load_voice_plan(manifest["conversion_id"])
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="malformed voice plan"):
        workspace.load_voice_plan(manifest["conversion_id"])
    workspace.persist_voice_plan(manifest["conversion_id"], artifact)
    previous_bytes = path.read_bytes()
    stale_source = {**artifact, "source_pdf_sha256": "c" * 64}
    stale_source = with_canonical_artifact_hash(stale_source)
    with pytest.raises(ManifestError, match="invalid voice plan artifact"):
        workspace.persist_voice_plan(manifest["conversion_id"], stale_source)
    assert path.read_bytes() == previous_bytes
    stale_chapter = {**artifact, "chapter_plan_sha256": "d" * 64}
    stale_chapter = with_canonical_artifact_hash(stale_chapter)
    with pytest.raises(ManifestError, match="invalid voice plan artifact"):
        workspace.persist_voice_plan(manifest["conversion_id"], stale_chapter)
    assert path.read_bytes() == previous_bytes
    atomic_write_json(path, stale_source)
    with pytest.raises(ManifestError, match="invalid voice plan artifact"):
        workspace.load_voice_plan(manifest["conversion_id"])


def _prepared_voice_workspace(tmp_path: Path) -> tuple[Workspace, dict, Path, dict, Path]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    text = "Narrator speaks. Alice replies! Bob waits."
    split = text.index("Bob")
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": split}, {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 3}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}], "warnings": []}
    planned = workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    artifact = _voice_plan_for_workspace(manifest["source_pdf_sha256"], planned["chapter_plan_sha256"], text, plan)
    workspace.persist_voice_plan(manifest["conversion_id"], artifact)
    return workspace, manifest, workspace.conversion_path(manifest["conversion_id"]) / "source.pdf", artifact, workspace.conversion_path(manifest["conversion_id"]) / "voice-plan.json"


def test_voice_plan_source_tamper_fails_closed(tmp_path: Path) -> None:
    workspace, manifest, source_path, _, _ = _prepared_voice_workspace(tmp_path)
    source_path.write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="source PDF hash mismatch"):
        workspace.load_voice_plan(manifest["conversion_id"])


def test_voice_plan_unsafe_targets_are_rejected_without_skipping_other_checks(tmp_path: Path) -> None:
    workspace, manifest, source_path, artifact, path = _prepared_voice_workspace(tmp_path)
    path.unlink()
    path.mkdir()
    with pytest.raises(ManifestError):
        workspace.persist_voice_plan(manifest["conversion_id"], artifact)
    with pytest.raises(ManifestError):
        workspace.load_voice_plan(manifest["conversion_id"])
    path.rmdir()
    try:
        path.symlink_to(source_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ManifestError):
        workspace.load_voice_plan(manifest["conversion_id"])
    with pytest.raises(ManifestError):
        workspace.persist_voice_plan(manifest["conversion_id"], artifact)


def _prepared_speaker_workspace(tmp_path: Path) -> tuple[Workspace, dict, dict, dict, Path]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    text = "Narrator speaks. Alice replies! Bob waits."
    split = text.index("Bob")
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": split}, {"source_page": 2, "cleaned_start": split, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "original", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "One", "start_offset": 0, "end_offset": split, "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 3}, {"index": 2, "title": "Two", "start_offset": split, "end_offset": len(text), "start_page": 2, "end_page": 2, "source_type": "whole", "word_count": 2}], "warnings": []}
    planned = workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    artifact = _speaker_analysis_for_workspace(manifest["source_pdf_sha256"], planned["chapter_plan_sha256"], text, plan)
    manifest = workspace.read_job(manifest["conversion_id"])
    return workspace, manifest, plan, artifact, workspace.conversion_path(manifest["conversion_id"]) / "speaker-analysis.json"


def test_speaker_analysis_round_trip_replacement_and_voice_plan_preservation(tmp_path: Path) -> None:
    workspace, manifest, plan, artifact, path = _prepared_speaker_workspace(tmp_path)
    text = "Narrator speaks. Alice replies! Bob waits."
    voice = _voice_plan_for_workspace(manifest["source_pdf_sha256"], manifest["chapter_plan_sha256"], text, plan)
    workspace.persist_voice_plan(manifest["conversion_id"], voice)
    voice_bytes = workspace.conversion_path(manifest["conversion_id"]).joinpath("voice-plan.json").read_bytes()
    job_before = workspace.job_path(manifest["conversion_id"]).read_bytes()

    assert workspace.persist_speaker_analysis(manifest["conversion_id"], artifact) == artifact
    assert path.read_bytes() == canonical_json_bytes(artifact)
    assert workspace.load_speaker_analysis(manifest["conversion_id"]) == artifact
    assert workspace.job_path(manifest["conversion_id"]).read_bytes() == job_before
    assert workspace.conversion_path(manifest["conversion_id"]).joinpath("voice-plan.json").read_bytes() == voice_bytes

    replacement = with_canonical_artifact_hash({**artifact, "revision": 2})
    assert workspace.persist_speaker_analysis(manifest["conversion_id"], replacement) == replacement
    assert workspace.load_speaker_analysis(manifest["conversion_id"]) == replacement
    previous_bytes = path.read_bytes()
    invalid = with_canonical_artifact_hash({**replacement, "artifact": "wrong"})
    with pytest.raises(ManifestError, match="invalid speaker analysis artifact"):
        workspace.persist_speaker_analysis(manifest["conversion_id"], invalid)
    assert path.read_bytes() == previous_bytes


def test_speaker_analysis_persist_replaces_oversized_regular_previous_artifact(tmp_path: Path, monkeypatch) -> None:
    workspace, manifest, _, artifact, path = _prepared_speaker_workspace(tmp_path)
    path.write_bytes(b"xx")
    monkeypatch.setattr("pdf_audiobook.workspace.MAX_ARTIFACT_BYTES", 1)
    assert workspace.persist_speaker_analysis(manifest["conversion_id"], artifact) == artifact
    monkeypatch.undo()
    assert workspace.load_speaker_analysis(manifest["conversion_id"]) == artifact
    assert path.read_bytes() == canonical_json_bytes(artifact)


def test_speaker_analysis_rejects_malformed_tampered_and_stale_bindings(tmp_path: Path) -> None:
    workspace, manifest, plan, artifact, path = _prepared_speaker_workspace(tmp_path)
    workspace.persist_speaker_analysis(manifest["conversion_id"], artifact)
    previous_bytes = path.read_bytes()

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="malformed speaker analysis JSON"):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    path.write_bytes(previous_bytes)
    tampered = json.loads(previous_bytes.decode("utf-8"))
    tampered["warnings"] = ["changed"]
    atomic_write_json(path, tampered)
    with pytest.raises(ManifestError, match="invalid speaker analysis artifact"):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    atomic_write_json(path, artifact)

    stale_source = with_canonical_artifact_hash({**artifact, "source_pdf_sha256": "b" * 64})
    atomic_write_json(path, stale_source)
    with pytest.raises(ManifestError, match="invalid speaker analysis artifact"):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    assert path.read_bytes() != previous_bytes
    atomic_write_json(path, artifact)
    with pytest.raises(ManifestError, match="invalid speaker analysis artifact"):
        workspace.persist_speaker_analysis(manifest["conversion_id"], stale_source)
    assert path.read_bytes() == previous_bytes

    cleaned_path = workspace.conversion_path(manifest["conversion_id"]) / "cleaned.txt"
    cleaned_path.write_text("stale cleaned text", encoding="utf-8")
    with pytest.raises(ManifestError, match="cleaned text hash mismatch"):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    cleaned_path.write_text("Narrator speaks. Alice replies! Bob waits.", encoding="utf-8")

    chapter_path = workspace.conversion_path(manifest["conversion_id"]) / "chapters.json"
    atomic_write_json(chapter_path, {**plan, "warnings": ["stale plan"]})
    with pytest.raises(ManifestError, match="chapter plan hash mismatch"):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    atomic_write_json(chapter_path, plan)

    source_path = workspace.conversion_path(manifest["conversion_id"]) / "source.pdf"
    source_path.write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="source PDF hash mismatch"):
        workspace.load_speaker_analysis(manifest["conversion_id"])


def test_speaker_analysis_unsafe_targets_are_rejected_independently(tmp_path: Path) -> None:
    workspace, manifest, _, artifact, path = _prepared_speaker_workspace(tmp_path)
    workspace.persist_speaker_analysis(manifest["conversion_id"], artifact)
    path.unlink()
    path.mkdir()
    with pytest.raises(ManifestError):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    with pytest.raises(ManifestError):
        workspace.persist_speaker_analysis(manifest["conversion_id"], artifact)
    path.rmdir()
    source_path = workspace.conversion_path(manifest["conversion_id"]) / "source.pdf"
    try:
        path.symlink_to(source_path)
    except (OSError, NotImplementedError) as exc:
        return
    with pytest.raises(ManifestError):
        workspace.load_speaker_analysis(manifest["conversion_id"])
    with pytest.raises(ManifestError):
        workspace.persist_speaker_analysis(manifest["conversion_id"], artifact)


def test_speaker_analysis_size_limit_is_checked_before_parse(tmp_path: Path, monkeypatch) -> None:
    workspace, manifest, _, artifact, path = _prepared_speaker_workspace(tmp_path)
    path.write_bytes(canonical_json_bytes(artifact))
    monkeypatch.setattr(speaker_analysis_module, "MAX_ARTIFACT_BYTES", 1)
    monkeypatch.setattr("pdf_audiobook.workspace.MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ManifestError, match="speaker analysis artifact too large"):
        workspace.load_speaker_analysis(manifest["conversion_id"])


def test_voice_analysis_status_round_trip_replacement_and_bindings(tmp_path: Path) -> None:
    workspace, manifest, _, _, _ = _prepared_speaker_workspace(tmp_path)
    job_before = workspace.job_path(manifest["conversion_id"]).read_bytes()
    status = _voice_analysis_status_for_workspace(manifest)
    returned = workspace.persist_voice_analysis_status(manifest["conversion_id"], status)
    path = workspace.conversion_path(manifest["conversion_id"]) / "voice-analysis-status.json"
    assert returned == status
    assert path.read_bytes() == canonical_json_bytes(status)
    assert workspace.load_voice_analysis_status(manifest["conversion_id"]) == status
    assert workspace.job_path(manifest["conversion_id"]).read_bytes() == job_before
    previous = path.read_bytes()
    invalid = with_canonical_artifact_hash({**status, "analysis_id": "not-a-uuid"})
    with pytest.raises(ManifestError, match="invalid voice-analysis status"):
        workspace.persist_voice_analysis_status(manifest["conversion_id"], invalid)
    assert path.read_bytes() == previous
    stale = with_canonical_artifact_hash({**status, "chapter_plan_sha256": "b" * 64})
    with pytest.raises(ManifestError, match="invalid voice-analysis status"):
        workspace.persist_voice_analysis_status(manifest["conversion_id"], stale)
    assert path.read_bytes() == previous


def test_voice_analysis_status_persist_replaces_oversized_regular_previous(tmp_path: Path, monkeypatch) -> None:
    workspace, manifest, _, _, _ = _prepared_speaker_workspace(tmp_path)
    status = _voice_analysis_status_for_workspace(manifest)
    path = workspace.conversion_path(manifest["conversion_id"]) / "voice-analysis-status.json"
    path.write_bytes(b"xx")
    monkeypatch.setattr("pdf_audiobook.workspace.MAX_VOICE_ANALYSIS_STATUS_BYTES", 1)
    assert workspace.persist_voice_analysis_status(manifest["conversion_id"], status) == status
    monkeypatch.undo()
    assert workspace.load_voice_analysis_status(manifest["conversion_id"]) == status


def test_voice_analysis_status_malformed_tampered_oversized_and_unsafe(tmp_path: Path, monkeypatch) -> None:
    workspace, manifest, _, _, _ = _prepared_speaker_workspace(tmp_path)
    status = _voice_analysis_status_for_workspace(manifest)
    workspace.persist_voice_analysis_status(manifest["conversion_id"], status)
    path = workspace.conversion_path(manifest["conversion_id"]) / "voice-analysis-status.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="malformed voice-analysis status"):
        workspace.load_voice_analysis_status(manifest["conversion_id"])
    path.write_bytes(canonical_json_bytes(status))
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["warnings"] = ["changed"]
    atomic_write_json(path, tampered)
    with pytest.raises(ManifestError, match="invalid voice-analysis status"):
        workspace.load_voice_analysis_status(manifest["conversion_id"])
    atomic_write_json(path, status)
    monkeypatch.setattr("pdf_audiobook.workspace.MAX_VOICE_ANALYSIS_STATUS_BYTES", 1)
    with pytest.raises(ManifestError, match="voice-analysis status too large"):
        workspace.load_voice_analysis_status(manifest["conversion_id"])
    monkeypatch.undo()
    path.unlink()
    path.mkdir()
    with pytest.raises(ManifestError):
        workspace.load_voice_analysis_status(manifest["conversion_id"])
    with pytest.raises(ManifestError):
        workspace.persist_voice_analysis_status(manifest["conversion_id"], status)


def test_voice_analysis_cancel_marker_is_distinct_safe_and_idempotent(tmp_path: Path) -> None:
    workspace, manifest, _, _, _ = _prepared_speaker_workspace(tmp_path)
    marker = workspace.voice_analysis_cancel_marker_path(manifest["conversion_id"])
    assert marker.name == "voice-analysis.cancel"
    assert workspace.voice_analysis_cancellation_requested(manifest["conversion_id"]) is False
    workspace.request_voice_analysis_cancel(manifest["conversion_id"])
    workspace.request_voice_analysis_cancel(manifest["conversion_id"])
    assert workspace.voice_analysis_cancellation_requested(manifest["conversion_id"]) is True
    assert not workspace.cancellation_requested(manifest["conversion_id"])
    workspace.clear_voice_analysis_cancel_request(manifest["conversion_id"])
    workspace.clear_voice_analysis_cancel_request(manifest["conversion_id"])
    assert workspace.voice_analysis_cancellation_requested(manifest["conversion_id"]) is False
    marker.mkdir()
    with pytest.raises(UnsafePathError):
        workspace.request_voice_analysis_cancel(manifest["conversion_id"])
    with pytest.raises(UnsafePathError):
        workspace.voice_analysis_cancellation_requested(manifest["conversion_id"])
    with pytest.raises(UnsafePathError):
        workspace.clear_voice_analysis_cancel_request(manifest["conversion_id"])


def test_voice_analysis_status_and_cancel_symlink_targets_are_rejected_when_supported(tmp_path: Path) -> None:
    workspace, manifest, _, _, _ = _prepared_speaker_workspace(tmp_path)
    status = _voice_analysis_status_for_workspace(manifest)
    status_path = workspace.conversion_path(manifest["conversion_id"]) / "voice-analysis-status.json"
    status_path.write_bytes(canonical_json_bytes(status))
    target = workspace.conversion_path(manifest["conversion_id"]) / "source.pdf"
    try:
        status_path.unlink()
        status_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pass
    else:
        with pytest.raises(ManifestError):
            workspace.load_voice_analysis_status(manifest["conversion_id"])
        with pytest.raises(ManifestError):
            workspace.persist_voice_analysis_status(manifest["conversion_id"], status)
        status_path.unlink()
    marker = workspace.voice_analysis_cancel_marker_path(manifest["conversion_id"])
    try:
        marker.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(UnsafePathError):
        workspace.voice_analysis_cancellation_requested(manifest["conversion_id"])
    with pytest.raises(UnsafePathError):
        workspace.request_voice_analysis_cancel(manifest["conversion_id"])
    with pytest.raises(UnsafePathError):
        workspace.clear_voice_analysis_cancel_request(manifest["conversion_id"])


def test_load_cleaned_artifacts_verifies_text_and_mapping(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    workspace.persist_analysis(manifest["conversion_id"], _analysis_with_mapping())

    cleaned_text, cleaned_map = workspace.load_cleaned_artifacts(manifest["conversion_id"])

    assert cleaned_text == "cleaned text\n"
    assert cleaned_map == [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(cleaned_text)}]


def test_load_cleaned_artifacts_rejects_text_hash_and_map_errors(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    workspace.persist_analysis(manifest["conversion_id"], _analysis_with_mapping())
    conversion = workspace.work_root / manifest["conversion_id"]
    cleaned_path = conversion / "cleaned.txt"
    cleaned_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="cleaned text hash"):
        workspace.load_cleaned_artifacts(manifest["conversion_id"])
    cleaned_path.write_text("cleaned text\n", encoding="utf-8")
    map_path = conversion / "cleaned-map.json"
    map_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="malformed cleaned map"):
        workspace.load_cleaned_artifacts(manifest["conversion_id"])
    atomic_write_json(map_path, [{"source_page": 1, "cleaned_start": 1, "cleaned_end": 2}])
    with pytest.raises(ManifestError, match="cleaned map"):
        workspace.load_cleaned_artifacts(manifest["conversion_id"])


def test_load_cleaned_artifacts_rejects_symlinked_inputs(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    workspace.persist_analysis(manifest["conversion_id"], _analysis_with_mapping())
    cleaned_path = workspace.work_root / manifest["conversion_id"] / "cleaned.txt"
    try:
        cleaned_path.unlink()
        cleaned_path.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ManifestError):
        workspace.load_cleaned_artifacts(manifest["conversion_id"])


def test_chapter_plan_round_trip_binds_exact_canonical_hash(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    analyzed = workspace.persist_analysis(conversion_id, _analysis_for_workspace())
    cleaned_hash = analyzed["cleaned_text_sha256"]
    plan = {
        "schema_version": 1,
        "mode": "custom",
        "requested_count": 2,
        "cleaned_text_sha256": cleaned_hash,
        "chapters": [{"title": "One", "start": 0}, {"title": "Two", "start": 10}],
        "warnings": [],
    }

    planned = workspace.persist_chapter_plan(conversion_id, plan)

    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    assert planned["status"] == "planned"
    assert planned["stage"] == "chapter_review"
    assert planned["chapter_plan_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert (conversion := workspace.work_root / conversion_id).joinpath("chapters.json").read_bytes() == canonical.encode("utf-8")
    assert workspace.load_chapter_plan(conversion_id) == plan
    assert workspace.inspect_startup().state == "resumable"


@pytest.mark.parametrize("mode", ["original", "whole"])
def test_chapter_plan_round_trip_accepts_original_and_whole_modes(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    analyzed = workspace.persist_analysis(conversion_id, _analysis_for_workspace())
    plan = {
        "schema_version": 1,
        "mode": mode,
        "requested_count": None,
        "cleaned_text_sha256": analyzed["cleaned_text_sha256"],
        "chapters": [{"title": "Whole Book"}],
        "warnings": [],
    }

    workspace.persist_chapter_plan(conversion_id, plan)

    assert workspace.load_chapter_plan(conversion_id) == plan


def test_chapter_plan_cleaned_hash_mismatch_does_not_mutate_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    analyzed = workspace.persist_analysis(manifest["conversion_id"], _analysis_for_workspace())
    plan = {
        "schema_version": 1,
        "mode": "custom",
        "requested_count": 2,
        "cleaned_text_sha256": "f" * 64,
        "chapters": [],
        "warnings": [],
    }
    job_before = (conversion / "job.json").read_bytes()
    with pytest.raises(ManifestError, match="cleaned text hash"):
        workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    assert not (conversion / "chapters.json").exists()
    assert (conversion / "job.json").read_bytes() == job_before
    assert analyzed["chapter_plan_sha256"] is None


def test_analysis_replacement_clears_prior_plan_and_removes_artifact(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    analyzed = workspace.persist_analysis(conversion_id, _analysis_for_workspace())
    plan = {
        "schema_version": 1,
        "mode": "custom",
        "requested_count": 2,
        "cleaned_text_sha256": analyzed["cleaned_text_sha256"],
        "chapters": [],
        "warnings": [],
    }
    workspace.persist_chapter_plan(conversion_id, plan)
    replaced = workspace.persist_analysis(conversion_id, {**_analysis_for_workspace(), "cleaned_text": "new text\n"})
    assert replaced["status"] == "analyzed"
    assert replaced["chapter_plan_sha256"] is None
    assert not (workspace.work_root / conversion_id / "chapters.json").exists()


def test_startup_reports_no_active_and_invalid_states_without_deleting(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "data")
    assert workspace.inspect_startup().state == "no_active"
    workspace.root.mkdir()
    workspace.active_path.write_text("{not-json", encoding="utf-8")
    result = workspace.inspect_startup()
    assert result.state == "invalid"
    assert workspace.active_path.exists()


def test_startup_detects_missing_source_and_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    conversion.joinpath("source.pdf").unlink()
    assert workspace.inspect_startup().state == "invalid"
    workspace.delete_conversion(manifest["conversion_id"])

    manifest = workspace.create_conversion(source, conversion_id=str(uuid.uuid4()))
    conversion = workspace.work_root / manifest["conversion_id"]
    conversion.joinpath("source.pdf").write_bytes(b"changed")
    result = workspace.inspect_startup()
    assert result.state == "invalid"
    assert "hash mismatch" in (result.reason or "")


def test_startup_validates_optional_cleaned_text_hash_and_file(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    cleaned = b"cleaned text\n"
    manifest = {
        **manifest,
        "cleaned_text_sha256": hashlib.sha256(cleaned).hexdigest(),
    }
    atomic_write_json(conversion / "job.json", manifest)

    assert workspace.inspect_startup().state == "invalid"
    cleaned_path = conversion / "cleaned.txt"
    cleaned_path.write_bytes(cleaned)
    assert workspace.inspect_startup().state == "resumable"
    cleaned_path.unlink()
    assert workspace.inspect_startup().state == "invalid"
    cleaned_path.write_bytes(b"changed")
    result = workspace.inspect_startup()
    assert result.state == "invalid"
    assert "cleaned text hash mismatch" in (result.reason or "")


def test_conversion_traversal_and_noncanonical_ids_are_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "data")
    with pytest.raises(UnsafePathError):
        workspace.conversion_path("../escape")
    with pytest.raises(UnsafePathError):
        workspace.conversion_path("not-a-uuid")
    with pytest.raises(UnsafePathError):
        workspace.conversion_path(str(uuid.uuid4()).upper())


def test_delete_refuses_symlink_anywhere_and_preserves_other_conversions(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    first = workspace.create_conversion(source)
    workspace.active_path.unlink()
    second = workspace.create_conversion(source, conversion_id=str(uuid.uuid4()))
    first_dir = workspace.work_root / first["conversion_id"]
    second_dir = workspace.work_root / second["conversion_id"]
    linked_target = tmp_path / "outside.txt"
    linked_target.write_text("keep", encoding="utf-8")
    try:
        first_dir.joinpath("linked").symlink_to(linked_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(UnsafePathError):
        workspace.delete_conversion(first["conversion_id"])
    assert first_dir.exists()
    assert second_dir.exists()
    assert linked_target.read_text(encoding="utf-8") == "keep"


def test_explicit_delete_removes_only_validated_conversion_and_active_marker(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    first = workspace.create_conversion(source)
    workspace.active_path.unlink()
    second = workspace.create_conversion(source, conversion_id=str(uuid.uuid4()))
    first_dir = workspace.work_root / first["conversion_id"]
    first_dir.joinpath("cleaned.txt").write_text("clean", encoding="utf-8")
    assert workspace.delete_conversion(first["conversion_id"]) is True
    assert not first_dir.exists()
    assert (workspace.work_root / second["conversion_id"]).exists()
    assert workspace.active_path.exists()
    assert workspace.delete_conversion(second["conversion_id"]) is True
    assert not workspace.active_path.exists()
    assert workspace.delete_conversion(second["conversion_id"]) is False


def test_analysis_load_requires_regular_artifact_and_matching_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    directory = workspace.work_root / manifest["conversion_id"]
    atomic_write_json(directory / "analysis.json", {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book"})
    assert workspace.load_analysis(manifest["conversion_id"])["title"] == "Book"
    atomic_write_json(directory / "analysis.json", {"source_pdf_sha256": "0" * 64})
    with pytest.raises(ManifestError, match="analysis source hash"):
        workspace.load_analysis(manifest["conversion_id"])
    atomic_write_json(directory / "analysis.json", {"source_pdf_sha256": manifest["source_pdf_sha256"]})
    try:
        directory.joinpath("analysis.json").unlink()
        directory.joinpath("analysis.json").symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ManifestError):
        workspace.load_analysis(manifest["conversion_id"])


def _write_reference(path: Path, *, sample_rate: int = 16000, seconds: float = 6.0, value: int = 0) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((value.to_bytes(2, "little", signed=True)) * frames)


def test_chatterbox_reference_lifecycle_is_owned_strict_and_path_free(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference)
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    with pytest.raises(WorkspaceError, match="consent"):
        workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=False)
    status = workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    assert status["available"] is True and "path" not in status
    directory = workspace.conversion_path(conversion_id)
    assert (directory / REFERENCE_WAV_FILENAME).is_file() and (directory / REFERENCE_DESCRIPTOR_FILENAME).is_file()
    loaded = workspace.load_chatterbox_reference(conversion_id)
    assert loaded.descriptor["consent_evidence"] == "user-confirmed-local-reference"
    assert loaded.descriptor["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert workspace.chatterbox_reference_status(conversion_id)["reference_sha256"] == loaded.descriptor["reference_sha256"]
    replacement = tmp_path / "replacement.wav"; _write_reference(replacement, value=1)
    replaced = workspace.replace_chatterbox_reference(conversion_id, replacement, consent_confirmed=True)
    assert replaced["reference_sha256"] != status["reference_sha256"]
    assert workspace.delete_chatterbox_reference(conversion_id) is True
    assert workspace.chatterbox_reference_status(conversion_id)["available"] is False
    assert workspace.delete_chatterbox_reference(conversion_id) is False


def test_chatterbox_reference_limits_and_mutation_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference, seconds=61)
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    with pytest.raises(WorkspaceError, match="invalid"):
        workspace.store_chatterbox_reference(manifest["conversion_id"], reference, consent_confirmed=True)
    _write_reference(reference)
    workspace.store_chatterbox_reference(manifest["conversion_id"], reference, consent_confirmed=True)
    workspace.chatterbox_reference_path(manifest["conversion_id"]).write_bytes(b"changed")
    with pytest.raises(ManifestError, match="reference"):
        workspace.load_chatterbox_reference(manifest["conversion_id"])


def test_chatterbox_reference_requires_more_than_five_seconds(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    _write_reference(reference, seconds=5.0)
    with pytest.raises(WorkspaceError, match="invalid"):
        workspace.store_chatterbox_reference(manifest["conversion_id"], reference, consent_confirmed=True)
    _write_reference(reference, seconds=5.01)
    status = workspace.store_chatterbox_reference(manifest["conversion_id"], reference, consent_confirmed=True)
    assert status["available"] is True and status["duration_seconds"] > 5.0


def test_builtin_chatterbox_generation_is_not_invalidated_by_reference_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source); conversion_id = manifest["conversion_id"]
    text = "Built-in voice."
    workspace.persist_analysis(conversion_id, {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 2}], "warnings": []}
    workspace.persist_chapter_plan(conversion_id, plan)
    from pdf_audiobook.tts import CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT, EngineMetadata, SynthesisSettings
    metadata = EngineMetadata("chatterbox", "0.1.7", CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT, "unrecorded", "builtin", "bundled", "unrecorded", 24000, SynthesisSettings(sample_rate=24000, chunk_cap=300, chunk_mode="legacy").as_dict())
    with pytest.raises(ManifestError):
        workspace.configure_generation(conversion_id, tts={**metadata.as_dict(), "speed": 1.0, "chunk_cap": 300, "settings": {"reference_descriptor_sha256": "x"}}, total_chunks=1)
    workspace.configure_generation(conversion_id, tts={**metadata.as_dict(), "speed": 1.0, "chunk_cap": 300}, total_chunks=1)
    reference = tmp_path / "reference.wav"; _write_reference(reference)
    workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    workspace.replace_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    workspace.delete_chatterbox_reference(conversion_id)
    assert workspace.read_job(conversion_id)["status"] == "planned"


def test_chatterbox_invalid_replacement_preserves_previous_reference(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference)
    invalid = tmp_path / "invalid.wav"; invalid.write_bytes(b"not wav")
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    before = workspace.load_chatterbox_reference(conversion_id).descriptor["reference_sha256"]
    with pytest.raises(WorkspaceError, match="invalid"):
        workspace.replace_chatterbox_reference(conversion_id, invalid, consent_confirmed=True)
    assert workspace.load_chatterbox_reference(conversion_id).descriptor["reference_sha256"] == before


def test_chatterbox_descriptor_describes_exact_copy_after_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference, value=0)
    replacement = tmp_path / "replacement.wav"; _write_reference(replacement, value=1)
    original_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    original_copy = workspace_module.copy_reference_file

    def copy_then_drift(source_path: Path, destination_path: Path, *, chunk_size: int) -> None:
        original_copy(source_path, destination_path, chunk_size=chunk_size)
        Path(source_path).write_bytes(replacement.read_bytes())

    monkeypatch.setattr(workspace_module, "copy_reference_file", copy_then_drift)
    status = workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    assert status["reference_sha256"] == original_digest
    assert status["reference_sha256"] != hashlib.sha256(replacement.read_bytes()).hexdigest()
    assert workspace.load_chatterbox_reference(conversion_id).descriptor["reference_sha256"] == status["reference_sha256"]


def test_chatterbox_status_rejects_missing_or_mutated_controlled_side(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference)
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    controlled = workspace.chatterbox_reference_path(conversion_id)
    controlled.unlink()
    with pytest.raises(ManifestError, match="reference"):
        workspace.chatterbox_reference_status(conversion_id)


def test_chatterbox_revoke_removes_corrupt_safe_artifacts_but_refuses_links(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    reference = tmp_path / "reference.wav"; _write_reference(reference)
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    conversion_id = manifest["conversion_id"]
    workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    workspace.chatterbox_reference_descriptor_path(conversion_id).write_text("{broken", encoding="utf-8")
    workspace.chatterbox_reference_path(conversion_id).write_bytes(b"corrupt but regular")
    assert workspace.delete_chatterbox_reference(conversion_id) is True
    assert not workspace.chatterbox_reference_descriptor_path(conversion_id).exists()
    assert not workspace.chatterbox_reference_path(conversion_id).exists()

    workspace.store_chatterbox_reference(conversion_id, reference, consent_confirmed=True)
    linked = tmp_path / "outside-reference.wav"; linked.write_bytes(b"keep")
    target = workspace.chatterbox_reference_path(conversion_id)
    target.unlink()
    try:
        target.symlink_to(linked)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(UnsafePathError):
        workspace.delete_chatterbox_reference(conversion_id)
    assert linked.read_bytes() == b"keep"
    assert not workspace.chatterbox_reference_descriptor_path(conversion_id).exists()


def test_malformed_active_state_can_be_explicitly_reset_without_deleting_work(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data")
    manifest = workspace.create_conversion(source)
    conversion = workspace.work_root / manifest["conversion_id"]
    workspace.active_path.write_text("{not-json", encoding="utf-8")
    assert workspace.delete_active_state() is True
    assert not workspace.active_path.exists()
    assert conversion.exists()


def test_v3_synthesis_manifest_migrates_strictly_and_missing_chunk_stays_unverified(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    text = "One sentence. Two sentence."
    workspace.persist_analysis(manifest["conversion_id"], {"source_pdf_sha256": manifest["source_pdf_sha256"], "title": "Book", "cleaned_text": text, "cleaned_map": [{"source_page": 1, "cleaned_start": 0, "cleaned_end": len(text)}], "warnings": []})
    plan = {"schema_version": 1, "mode": "whole", "requested_count": None, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chapters": [{"index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text), "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": 4}], "warnings": []}
    workspace.persist_chapter_plan(manifest["conversion_id"], plan)
    from pdf_audiobook.tts import EngineMetadata, SynthesisSettings, plan_chunks
    settings = SynthesisSettings(); metadata = EngineMetadata("fake", "builtin", "deterministic-fake", "phase5", "builtin", "fake-neutral", "builtin", "builtin", 24000, settings.as_dict())
    workspace.configure_generation(manifest["conversion_id"], tts={**metadata.as_dict(), "speed": 1.0, "chunk_cap": 900}, total_chunks=len(plan_chunks(text, plan["chapters"], metadata)))
    ConversionWorker(workspace, manifest["conversion_id"]).run(engine=FakeVoice())
    job_path = workspace.job_path(manifest["conversion_id"]); raw = json.loads(job_path.read_text(encoding="utf-8")); raw["schema_version"] = 3; raw.pop("output")
    for record in raw["completed_chunks"]: record.pop("wav_sha256")
    (workspace.conversion_path(manifest["conversion_id"]) / raw["completed_chunks"][0]["relative_path"]).unlink()
    atomic_write_json(job_path, raw)
    migrated = workspace.read_job(manifest["conversion_id"])
    assert migrated["schema_version"] == 4 and migrated["output"] is None and migrated["completed_chunks"][0]["wav_sha256"] == "0" * 64
    from pdf_audiobook.m4b import M4BError, assemble_chapters
    with pytest.raises(M4BError): assemble_chapters(workspace, manifest["conversion_id"])
    malformed = json.loads(json.dumps(migrated)); malformed["completed_chunks"][0]["duration_seconds"] = float("nan")
    atomic_write_json(job_path, malformed)
    with pytest.raises(ManifestError): workspace.read_job(manifest["conversion_id"])
    partial = json.loads(json.dumps(migrated)); partial["progress"]["completed"] = 0
    atomic_write_json(job_path, partial)
    with pytest.raises(ManifestError): workspace.read_job(manifest["conversion_id"])
    migrated["stage"] = "completed"; migrated["status"] = "completed"
    atomic_write_json(job_path, migrated)
    with pytest.raises(ManifestError): workspace.read_job(manifest["conversion_id"])


def test_output_manifest_rejects_wrong_filename_or_nonfinite_facts(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"; source.write_bytes(b"source")
    workspace = Workspace(tmp_path / "data"); manifest = workspace.create_conversion(source)
    base = workspace.read_job(manifest["conversion_id"])
    atomic_write_json(workspace.job_path(manifest["conversion_id"]), {**base, "schema_version": 4, "output": {"filename": "wrong.m4b", "path": str(tmp_path / "right.m4b"), "size_bytes": 1, "duration_seconds": 1.0, "chapter_count": 1, "codec": "aac", "sha256": "0" * 64}, "status": "completed", "stage": "completed"})
    with pytest.raises(ManifestError): workspace.read_job(manifest["conversion_id"])


def _prepare_interactive_generation(tmp_path: Path) -> tuple[Workspace, dict, dict, dict]:
    workspace, manifest, plan, analysis, _ = _prepared_speaker_workspace(tmp_path)
    text = "Narrator speaks. Alice replies! Bob waits."
    voice_plan = _voice_plan_for_workspace(manifest["source_pdf_sha256"], manifest["chapter_plan_sha256"], text, plan)
    workspace.persist_voice_plan(manifest["conversion_id"], voice_plan)
    workspace.persist_speaker_analysis(manifest["conversion_id"], analysis)
    return workspace, manifest, voice_plan, analysis


def test_interactive_generation_v5_strict_round_trip_and_update(tmp_path: Path) -> None:
    workspace, manifest, voice_plan, analysis = _prepare_interactive_generation(tmp_path)
    registry_revision = "a" * 64
    configured = workspace.configure_interactive_generation(
        manifest["conversion_id"],
        tts=_interactive_tts(),
        total_chunks=1,
        voice_registry_revision=registry_revision,
    )
    assert configured["schema_version"] == INTERACTIVE_GENERATION_SCHEMA_VERSION == 5
    assert configured["mode"] == "interactive_voices"
    assert configured["voice_plan_sha256"] == voice_plan["canonical_artifact_sha256"]
    assert configured["voice_plan_revision"] == voice_plan["revision"]
    assert configured["speaker_analysis_sha256"] == analysis["canonical_artifact_sha256"]
    assert configured["cast_voice_ids"] == ["voice-neutral"]
    assert configured["voice_registry_revision"] == registry_revision

    record = {
        "chapter_index": 1,
        "global_index": 0,
        "local_index": 0,
        "span_id": "s2",
        "audio_input_hash": "d" * 64,
        "input_hash": "b" * 64,
        "relative_path": "chunks/0000.wav",
        "duration_seconds": 1.0,
        "wav_sha256": "c" * 64,
        "speaker_id": "alice",
        "voice_id": "voice-neutral",
        "segment_type": "dialogue",
        "source_start": 10,
        "source_end": 20,
    }
    updated = workspace.update_generation(
        manifest["conversion_id"],
        completed_chunks=[record],
        progress={"completed": 1, "current": 1, "total": 1},
    )
    assert validate_interactive_generation_manifest(updated) == updated
    assert workspace.read_job(manifest["conversion_id"]) == updated

    thought = {**record, "segment_type": "thought"}
    thought_manifest = {**updated, "completed_chunks": [thought]}
    assert validate_interactive_generation_manifest(thought_manifest) == thought_manifest

    unknown = dict(updated)
    unknown["unexpected"] = True
    with pytest.raises(ManifestError):
        validate_interactive_generation_manifest(unknown)
    malformed = dict(updated)
    malformed["completed_chunks"] = [{**record, "span_id": "", "segment_type": []}]
    with pytest.raises(ManifestError):
        validate_interactive_generation_manifest(malformed)
    malformed_hash = dict(updated)
    malformed_hash["completed_chunks"] = [{**record, "audio_input_hash": "D" * 64}]
    with pytest.raises(ManifestError, match="audio_input_hash"):
        validate_interactive_generation_manifest(malformed_hash)
    wrong_voice = {**updated, "completed_chunks": [{**record, "voice_id": "not-cast"}]}
    with pytest.raises(ManifestError, match="cast_voice_ids"):
        validate_interactive_generation_manifest(wrong_voice)
    wrong_tts_voice = {**updated, "tts": {**updated["tts"], "voice": "not-cast"}}
    with pytest.raises(ManifestError, match="cast_voice_ids"):
        validate_interactive_generation_manifest(wrong_tts_voice)


def test_interactive_generation_requires_approval_and_reconfigure_resets_changed_facts(tmp_path: Path) -> None:
    workspace, manifest, voice_plan, _ = _prepare_interactive_generation(tmp_path)
    draft = {**voice_plan, "approval": {"state": "draft", "approved_at": None, "approved_revision": None}}
    workspace.persist_voice_plan(manifest["conversion_id"], with_canonical_artifact_hash(draft))
    with pytest.raises(ManifestError, match="approved voice plan"):
        workspace.configure_interactive_generation(
            manifest["conversion_id"], tts=_interactive_tts(), total_chunks=1, voice_registry_revision="a" * 64
        )

    workspace.persist_voice_plan(manifest["conversion_id"], voice_plan)
    workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=1, voice_registry_revision="a" * 64
    )
    before = workspace.read_job(manifest["conversion_id"])
    evidence = {"chapter_index": 1, "global_index": 0, "local_index": 0, "span_id": "s2", "audio_input_hash": "d" * 64, "input_hash": "b" * 64, "relative_path": "chunks/0000.wav", "duration_seconds": 1.0, "wav_sha256": "c" * 64, "speaker_id": "alice", "voice_id": "voice-neutral", "segment_type": "dialogue", "source_start": 10, "source_end": 20}
    workspace.update_generation(manifest["conversion_id"], completed_chunks=[evidence], progress={"completed": 1, "current": 1, "total": 1})
    same = workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=1, voice_registry_revision="a" * 64
    )
    assert same["completed_chunks"] == [evidence]
    changed = workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=1, voice_registry_revision="d" * 64
    )
    assert changed["completed_chunks"] == [evidence]
    assert changed["progress"] == {"completed": 1, "current": 0, "total": 1}
    assert before["completed_chunks"] == []


def test_interactive_generation_reconfigure_rejects_active_and_completed_state_changes(tmp_path: Path) -> None:
    workspace, manifest, _, _ = _prepare_interactive_generation(tmp_path)
    workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=0, voice_registry_revision="a" * 64
    )
    workspace.update_generation(manifest["conversion_id"], status="synthesizing")
    with pytest.raises(ManifestError, match="active interactive"):
        workspace.configure_interactive_generation(
            manifest["conversion_id"], tts=_interactive_tts(), total_chunks=0, voice_registry_revision="b" * 64
        )
    workspace.update_generation(manifest["conversion_id"], status="completed", stage="synthesis_complete")
    with pytest.raises(ManifestError, match="completed interactive"):
        workspace.configure_interactive_generation(
            manifest["conversion_id"], tts=_interactive_tts(), total_chunks=0, voice_registry_revision="b" * 64
        )


def test_interactive_generation_changed_facts_filter_out_of_range_candidates(tmp_path: Path) -> None:
    workspace, manifest, _, _ = _prepare_interactive_generation(tmp_path)
    workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=2, voice_registry_revision="a" * 64
    )
    first = {"chapter_index": 1, "global_index": 0, "local_index": 0, "span_id": "s2", "audio_input_hash": "d" * 64, "input_hash": "b" * 64, "relative_path": "chunks/0000.wav", "duration_seconds": 1.0, "wav_sha256": "c" * 64, "speaker_id": "alice", "voice_id": "voice-neutral", "segment_type": "dialogue", "source_start": 10, "source_end": 20}
    second = {**first, "global_index": 1, "local_index": 1, "span_id": "s3", "audio_input_hash": "e" * 64, "relative_path": "chunks/0001.wav", "source_start": 20, "source_end": 30}
    workspace.update_generation(manifest["conversion_id"], completed_chunks=[first, second], progress={"completed": 2, "current": 2, "total": 2})
    changed = workspace.configure_interactive_generation(
        manifest["conversion_id"], tts=_interactive_tts(), total_chunks=1, voice_registry_revision="b" * 64
    )
    assert changed["completed_chunks"] == [first]
    assert changed["progress"] == {"completed": 1, "current": 0, "total": 1}

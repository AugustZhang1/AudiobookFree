from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from pdf_audiobook.workspace import (
    JOB_SCHEMA_VERSION,
    ManifestError,
    UnsafePathError,
    Workspace,
    WorkspaceError,
    atomic_write_json,
    atomic_write_text,
    copy_source_pdf,
    validate_job_manifest,
)
from pdf_audiobook.tts import FakeVoice
from pdf_audiobook.worker import ConversionWorker


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

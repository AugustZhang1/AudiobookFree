from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import pytest

from pdf_audiobook import voice_registry


EXPECTED_VOICE_IDS = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)


def test_catalog_is_ordered_path_free_and_returns_fresh_snapshots() -> None:
    entries = voice_registry.list_public_entries()
    assert tuple(entry["id"] for entry in entries) == EXPECTED_VOICE_IDS
    assert voice_registry.APPROVED_VOICE_IDS == EXPECTED_VOICE_IDS
    assert len(set(EXPECTED_VOICE_IDS)) == 28
    assert all(
        ("American English" in entry["description"] or "British English" in entry["description"])
        and ("female" in entry["description"] or "male" in entry["description"])
        for entry in entries
    )
    assert all("path" not in entry and not any(Path(str(value)).is_absolute() for value in entry.values() if isinstance(value, str)) for entry in entries)
    entries[0]["display_label"] = "changed"
    assert voice_registry.list_public_entries()[0]["display_label"] != "changed"


def test_public_catalog_exposes_stable_gender_accent_metadata_without_generation_impact() -> None:
    expected_by_family = {
        "af": ("female", "American"),
        "am": ("male", "American"),
        "bf": ("female", "British"),
        "bm": ("male", "British"),
    }
    entries = voice_registry.list_public_entries()
    assert [(entry["gender"], entry["accent"]) for entry in entries] == [expected_by_family[entry["id"][:2]] for entry in entries]
    assert all("gender" not in voice_registry.get_generation_facts(entry["id"]) for entry in entries)
    assert all("accent" not in voice_registry.get_generation_facts(entry["id"]) for entry in entries)
    assert voice_registry._GENERATION_FIELDS == (
        "id", "language", "engine", "package", "package_version", "model", "model_revision",
        "model_checksum", "voice_version", "voice_checksum", "sample_rate", "enabled",
    )
    assert voice_registry.REGISTRY_REVISION == "43fe7f8ff4bd54b164a7eaf5b86a5fc1b1b67282c40ff5bfe2e46b6a1741b7be"


def test_revision_is_lowercase_sha256_of_canonical_generation_entries() -> None:
    entries = voice_registry.list_public_entries()
    fields = (
        "id", "language", "engine", "package", "package_version", "model", "model_revision",
        "model_checksum", "voice_version", "voice_checksum", "sample_rate", "enabled",
    )
    payload = [{field: entry[field] for field in fields} for entry in entries]
    expected = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert voice_registry.REGISTRY_REVISION == expected
    assert voice_registry.registry_revision() == expected


@pytest.mark.parametrize("value", [None, "", " af_heart", "not-a-voice", 3, True])
def test_invalid_voice_ids_raise_bounded_registry_error(value: object) -> None:
    with pytest.raises(voice_registry.VoiceRegistryError):
        voice_registry.require_enabled_voice_id(value)
    with pytest.raises(voice_registry.VoiceRegistryError):
        voice_registry.get_generation_facts(value)


def test_generation_facts_are_fresh_and_match_tts_metadata_shape() -> None:
    facts = voice_registry.get_generation_facts("af_heart")
    assert facts["voice"] == "af_heart"
    assert facts["engine"] == "kokoro"
    assert facts["package"] == "kokoro"
    assert facts["package_version"] == "0.9.4"
    assert facts["model"] == "hexgrad/Kokoro-82M"
    facts["model"] = "changed"
    assert voice_registry.get_generation_facts("af_heart")["model"] == "hexgrad/Kokoro-82M"


def test_preview_resolution_distinguishes_missing_from_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    preview_root = Path(__file__).resolve().parents[1] / "src" / "pdf_audiobook"
    assert voice_registry.resolve_preview_path("af_heart", preview_root) is None
    assert voice_registry.resolve_preview_target("af_heart", preview_root) == preview_root / "sample-kokoro-af_heart.wav"
    original = voice_registry._VOICE_BY_ID["af_heart"]
    monkeypatch.setattr(voice_registry, "_VOICE_BY_ID", {**voice_registry._VOICE_BY_ID, "af_heart": original[:17] + ("voice_registry.py",) + original[18:]})
    preview = preview_root / "voice_registry.py"
    assert voice_registry.resolve_preview_path("af_heart", preview_root) == preview

    with pytest.raises(voice_registry.VoiceRegistryError):
        voice_registry.resolve_preview_path("af_heart", Path(__file__))
    monkeypatch.setattr(voice_registry, "_VOICE_BY_ID", {**voice_registry._VOICE_BY_ID, "af_heart": original[:17] + ("../voice_registry.py",) + original[18:]})
    with pytest.raises(voice_registry.VoiceRegistryError):
        voice_registry.resolve_preview_path("af_heart", preview_root)


def test_preview_target_rejects_existing_non_regular_target(monkeypatch: pytest.MonkeyPatch) -> None:
    original = voice_registry._VOICE_BY_ID["af_heart"]
    root = Path("tests") / f".pytest-voice-target-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        monkeypatch.setattr(voice_registry, "_VOICE_BY_ID", {**voice_registry._VOICE_BY_ID, "af_heart": original[:17] + ("nested/target.wav",) + original[18:]})
        assert voice_registry.resolve_preview_target("af_heart", root) == (root / "nested" / "target.wav").resolve()
        (root / "nested").mkdir()
        (root / "nested" / "target.wav").mkdir()
        with pytest.raises(voice_registry.VoiceRegistryError):
            voice_registry.resolve_preview_target("af_heart", root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preview_non_regular_targets_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    preview_root = Path(__file__).resolve().parents[1] / "src" / "pdf_audiobook"
    original = voice_registry._VOICE_BY_ID["af_heart"]
    monkeypatch.setattr(voice_registry, "_VOICE_BY_ID", {**voice_registry._VOICE_BY_ID, "af_heart": original[:17] + (".",) + original[18:]})
    with pytest.raises(voice_registry.VoiceRegistryError):
        voice_registry.resolve_preview_path("af_heart", preview_root)


def test_preview_resolution_prefers_canonical_target_over_legacy_download() -> None:
    root = Path("tests") / f".pytest-voice-legacy-priority-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        root = root.resolve()
        canonical = root / "sample-kokoro-af_heart.wav"
        legacy = root / "20260808T231337Z-kokoro-af_heart.wav"
        canonical.write_bytes(b"canonical")
        legacy.write_bytes(b"legacy")

        assert voice_registry.resolve_preview_path("af_heart", root) == canonical
        assert voice_registry.resolve_preview_target("af_heart", root) == canonical
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preview_resolution_selects_newest_legacy_and_skips_unsafe_or_nonregular() -> None:
    root = Path("tests") / f".pytest-voice-legacy-selection-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        root = root.resolve()
        older = root / "20260808T231337Z-kokoro-bf_emma.wav"
        newer = root / "20260809T231337Z-kokoro-bf_emma.wav"
        tied_a = root / "a-kokoro-bf_isabella.wav"
        tied_z = root / "z-kokoro-bf_isabella.wav"
        older.write_bytes(b"older")
        newer.write_bytes(b"newer")
        tied_a.write_bytes(b"a")
        tied_z.write_bytes(b"z")
        os.utime(older, (100, 100))
        os.utime(newer, (200, 200))
        os.utime(tied_a, (300, 300))
        os.utime(tied_z, (300, 300))
        (root / "-kokoro-bf_emma.wav").write_bytes(b"empty prefix")
        (root / "directory-kokoro-bf_emma.wav").mkdir()
        nested = root / "nested"
        nested.mkdir()
        (nested / "nested-kokoro-bf_emma.wav").write_bytes(b"nested")

        assert voice_registry.resolve_preview_path("bf_emma", root) == newer
        assert voice_registry.resolve_preview_path("bf_isabella", root) == tied_z
    finally:
        shutil.rmtree(root, ignore_errors=True)

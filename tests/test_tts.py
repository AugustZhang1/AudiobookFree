from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import uuid
import wave

import pdf_audiobook.tts as tts
import pytest
from pdf_audiobook.audio import write_pcm_wav
from pdf_audiobook.m4b import _recorded_chunks
from pdf_audiobook.tts import EngineMetadata, SynthesisSettings, chunk_input_hash, plan_chunks, plan_interactive_chunks
from pdf_audiobook import voice_registry
from pdf_audiobook.engine_catalog import catalog_revision, get_capability, list_capabilities, require_enabled
from pdf_audiobook.voice_plan import with_canonical_artifact_hash
from pdf_audiobook.voice_settings import VoiceSettingsError, canonical_voice_settings, voice_settings_digest
from pdf_audiobook.worker import ConversionWorker


def _metadata() -> EngineMetadata:
    settings = SynthesisSettings(chunk_mode="legacy")
    return EngineMetadata("fake", "builtin", "fake", "r1", "c1", "fake-neutral", "v1", "c2", 24000, settings.as_dict())


def test_speed_only_legacy_settings_normalize_to_complete_neutral_defaults() -> None:
    assert canonical_voice_settings({"speed": 1.2}) == {"speed": 1.2, "pitch_semitones": 0, "tone_preset": "neutral"}


def test_empty_voice_settings_are_malformed() -> None:
    with pytest.raises(VoiceSettingsError, match="schema mismatch"):
        canonical_voice_settings({})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.49, 2.01])
def test_speed_is_finite_and_bounded(value: float) -> None:
    with pytest.raises(VoiceSettingsError):
        canonical_voice_settings({"speed": value})


def test_pitch_and_tone_are_canonical_and_digest_changes() -> None:
    settings = canonical_voice_settings({"speed": 1, "pitch_semitones": -3, "tone_preset": "warm"})
    assert settings == {"speed": 1.0, "pitch_semitones": -3, "tone_preset": "warm"}
    assert voice_settings_digest(settings) != voice_settings_digest({"speed": 1})


@pytest.mark.parametrize("value", [{"speed": 1, "pitch_semitones": 1.5, "tone_preset": "neutral"}, {"speed": 1, "pitch_semitones": 4, "tone_preset": "neutral"}, {"speed": 1, "pitch_semitones": 0, "tone_preset": "muddy"}])
def test_invalid_pitch_and_tone_are_rejected(value: dict) -> None:
    with pytest.raises(VoiceSettingsError):
        canonical_voice_settings(value)


def test_approved_voices_match_registry_and_all_are_accepted() -> None:
    assert tts.APPROVED_VOICES == voice_registry.APPROVED_VOICE_IDS
    assert len(tts.APPROVED_VOICES) == 28 and len(set(tts.APPROVED_VOICES)) == 28
    for voice_id in tts.APPROVED_VOICES:
        tts.load_voice(voice_id, engine="fake").close_voice()
    with pytest.raises(ValueError, match="not approved"):
        tts.load_voice("not-a-voice", engine="fake")


def test_chapter_mode_plans_one_exact_chunk_per_chapter() -> None:
    text = "Intro sentence. Chapter two is deliberately longer than its cap."
    split = text.index("Chapter")
    chapters = [
        {"index": 1, "start_offset": 0, "end_offset": split},
        {"index": 2, "start_offset": split, "end_offset": len(text)},
    ]
    metadata = EngineMetadata("fake", "builtin", "fake", "r1", "c1", "fake-neutral", "v1", "c2", 24000, SynthesisSettings(chunk_mode="chapter").as_dict())
    chunks = plan_chunks(text, chapters, metadata, cap=1)
    assert [(chunk.chapter_index, chunk.global_index, chunk.local_index, chunk.source_start, chunk.source_end, chunk.text) for chunk in chunks] == [
        (1, 0, 0, 0, split, text[:split]),
        (2, 1, 0, split, len(text), text[split:]),
    ]


def test_metadata_without_chunk_mode_retains_legacy_planning() -> None:
    text = "One. Two. Three."
    metadata = _metadata().as_dict()
    del metadata["settings"]["chunk_mode"]
    chunks = plan_chunks(text, [{"index": 1, "start_offset": 0, "end_offset": len(text)}], metadata, cap=6)
    assert len(chunks) > 1


@pytest.mark.parametrize(
    ("settings", "expected_chunk_count"),
    [
        pytest.param(None, 2, id="settings-absent"),
        pytest.param({"speed": 1.0, "chunk_cap": 7}, 10, id="settings-partial-with-cap"),
        pytest.param({"speed": 1.0}, 1, id="settings-partial-without-cap"),
    ],
)
def test_v4_worker_and_assembler_share_worker_authoritative_planning_inputs(
    settings: dict[str, object] | None, expected_chunk_count: int,
) -> None:
    """Legacy v4 records must be planned identically before assembly.

    This intentionally uses the old ``model_id``-only metadata shape at the
    planner boundary.  A strict current Workspace validator is not the subject
    of this regression; the worker and assembler must still agree when reading
    a legacy recovery manifest.
    """

    root = Path("tests") / f".pytest-v4-planning-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        chapters = [{
            "index": 1, "title": "Book", "start_offset": 0, "end_offset": len(text),
            "start_page": 1, "end_page": 1, "source_type": "whole", "word_count": len(text.split()),
        }]
        tts = {
            "engine": "fake",
            "package_version": "builtin",
            "model_id": "worker-authoritative-model",
            "model_revision": "r1",
            "model_checksum": "c1",
            "voice": "fake-neutral",
            "voice_version": "v1",
            "voice_checksum": "c2",
            "sample_rate": 24000,
            "speed": 1.0,
            "chunk_cap": 31,
        }
        if settings is not None:
            tts["settings"] = settings
        job = {"schema_version": 4, "tts": tts, "total_chunks": 0, "completed_chunks": []}
        conversion_root = root / "conversion"
        conversion_root.mkdir()

        chapter_plan = {
            "schema_version": 1,
            "mode": "whole",
            "requested_count": None,
            "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chapters": chapters,
            "warnings": [],
        }

        class LegacyWorkspace:
            def read_job(self, _conversion_id: str) -> dict:
                return job

            def load_cleaned_artifacts(self, _conversion_id: str) -> tuple[str, list]:
                return text, []

            def load_chapter_plan(self, _conversion_id: str) -> dict:
                return chapter_plan

            def conversion_path(self, _conversion_id: str) -> Path:
                return conversion_root

        workspace = LegacyWorkspace()
        worker_chunks = ConversionWorker(workspace, "legacy-conversion")._planned_chunks(job)
        assert len(worker_chunks) == expected_chunk_count

        records = []
        for chunk in worker_chunks:
            path = conversion_root / f"chunks/chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"
            info = write_pcm_wav(path, b"\0\0" * 240, 24000)
            relative = path.relative_to(conversion_root).as_posix()
            records.append({
                **chunk.manifest_record(relative, info.duration_seconds),
                "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        job["total_chunks"] = len(records)
        job["completed_chunks"] = records

        assembled, _plan, _job = _recorded_chunks(workspace, "legacy-conversion")
        assert [chunk.text for chunk, _path, _info in assembled] == [chunk.text for chunk in worker_chunks]
        assert [chunk.input_hash for chunk, _path, _info in assembled] == [chunk.input_hash for chunk in worker_chunks]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_plan_chunks_rejects_explicit_unsupported_chunk_mode() -> None:
    metadata = _metadata().as_dict()
    metadata["settings"]["chunk_mode"] = "invalid"
    with pytest.raises(ValueError, match="unsupported chunk mode"):
        plan_chunks("One sentence.", [{"index": 1, "start_offset": 0, "end_offset": 14}], metadata)


def test_load_voice_rejects_unsupported_chunk_mode() -> None:
    with pytest.raises(ValueError, match="unsupported chunk mode"):
        tts.load_voice("fake-neutral", SynthesisSettings(chunk_mode="invalid"), engine="fake")


@pytest.fixture
def reference_dir() -> Path:
    root = Path(__file__).with_name(f".tts-chatterbox-{os.getpid()}")
    root.mkdir(exist_ok=True)
    try:
        yield root
    finally:
        for child in root.iterdir():
            child.unlink(missing_ok=True)
        root.rmdir()


def _reference_wav(tmp_path: Path, *, channels: int = 1, payload: bytes = b"\x00\x00\x01\x00") -> Path:
    path = tmp_path / "reference.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(payload * max(1, (24000 * 6) // max(1, len(payload) // 2)))
    return path


def test_chatterbox_catalog_is_static_and_only_nano_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: (_ for _ in ()).throw(AssertionError("model import")))
    entries = list_capabilities()
    assert [entry.model_id for entry in entries] == ["nano", "turbo", "base", "multilingual"]
    assert entries[0].enabled and not entries[0].reference_wav_required and entries[0].chunk_cap == 300
    assert entries[0].model_revision == "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
    assert all(not entry.enabled for entry in entries[1:])
    assert catalog_revision() == catalog_revision() and len(catalog_revision()) == 64
    with pytest.raises(ValueError, match="disabled"):
        require_enabled("chatterbox", "turbo")
    assert get_capability("chatterbox", "nano").runtime == "cpu"


def test_chatterbox_requires_reference_and_fixed_settings(reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)
    with pytest.raises(ValueError, match="reference WAV is required"):
        tts.load_voice("reference-wav", engine="chatterbox")
    with pytest.raises(ValueError, match="speed 1.0"):
        tts.load_voice("reference-wav", SynthesisSettings(speed=1.1, chunk_cap=300), engine="chatterbox", reference_wav=reference)
    with pytest.raises(ValueError, match="neutral pitch"):
        tts.load_voice("reference-wav", SynthesisSettings(pitch_semitones=1, chunk_cap=300), engine="chatterbox", reference_wav=reference)
    with pytest.raises(ValueError, match="does not use"):
        tts.load_voice("builtin", engine="chatterbox", reference_wav=reference)
    with pytest.raises(ValueError, match="24000 Hz"):
        tts.load_voice("builtin", SynthesisSettings(sample_rate=16000), engine="chatterbox")


def test_chatterbox_builtin_voice_uses_bundled_conditionals_without_reference(monkeypatch) -> None:
    events: list[object] = []
    class Model:
        sr = 24000
        def generate(self, text: str, **kwargs: object) -> list[float]: events.append((text, kwargs)); return [0.0, 0.5]
        def close(self) -> None: events.append("close")
    class ModelClass:
        @staticmethod
        def from_pretrained(**kwargs: object) -> Model: events.append(("from_pretrained", kwargs)); return Model()
    class Module: ChatterboxTurboTTS = ModelClass
    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: Module)
    voice = tts.load_voice("builtin", SynthesisSettings(), engine="chatterbox")
    assert voice.metadata.voice == "builtin" and voice.metadata.voice_checksum == "unrecorded"
    assert voice.synthesize("hello") == tts.pcm_from_audio([0.0, 0.5])
    assert events[-1] == ("hello", {})
    voice.close_voice(); assert events[-1] == "close"


def test_chatterbox_rejects_invalid_or_unsafe_reference_wav(reference_dir: Path) -> None:
    invalid = reference_dir / "invalid.wav"
    invalid.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="reference WAV is invalid or unsafe") as error:
        tts.load_voice("reference-wav", engine="chatterbox", reference_wav=invalid)
    assert str(invalid) not in str(error.value)
    stereo = _reference_wav(reference_dir, channels=2)
    with pytest.raises(ValueError, match="reference WAV is invalid or unsafe"):
        tts.load_voice("reference-wav", engine="chatterbox", reference_wav=stereo)


def test_chatterbox_loads_lazily_with_cpu_nano_and_binds_reference_hash(monkeypatch, reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)
    events: list[object] = []

    class Model:
        sr = 24000

        def generate(self, text: str, **kwargs: object) -> list[float]:
            events.append((text, kwargs))
            return [0.0, 0.5]

        def release(self) -> None:
            events.append("release")

    class ModelClass:
        @staticmethod
        def from_pretrained(**kwargs: object) -> Model:
            events.append(("from_pretrained", kwargs))
            return Model()

    class Module:
        ChatterboxTurboTTS = ModelClass

    def import_module(name: str) -> object:
        events.append(("import", name))
        return Module

    monkeypatch.setattr(tts.importlib, "import_module", import_module)
    voice = tts.load_voice("reference-wav", SynthesisSettings(), engine="chatterbox", reference_wav=reference)
    assert events[:2] == [("import", "chatterbox.tts_turbo"), ("from_pretrained", {"device": "cpu", "nano": True})]
    assert voice.metadata.model_revision == "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
    assert voice.metadata.voice_checksum == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert voice.metadata.sample_rate == 24000
    assert voice.metadata.settings["chunk_cap"] == 300
    assert voice.synthesize("hello") == tts.pcm_from_audio([0.0, 0.5])
    assert events[-1][0] == "hello" and events[-1][1]["audio_prompt_path"] == str(reference)
    assert voice.synthesize("!?\u2026") == b"\x00\x00" * 1200
    _reference_wav(reference_dir, payload=b"\x02\x00\x03\x00")
    generate_calls = sum(1 for event in events if isinstance(event, tuple) and event[0] == "hello")
    with pytest.raises(RuntimeError, match="reference WAV has changed") as error:
        voice.synthesize("!?\u2026")
    assert str(reference) not in str(error.value)
    assert sum(1 for event in events if isinstance(event, tuple) and event[0] == "hello") == generate_calls
    with pytest.raises(RuntimeError, match="reference WAV has changed") as error:
        voice.synthesize("changed")
    assert str(reference) not in str(error.value)
    assert sum(1 for event in events if isinstance(event, tuple) and event[0] == "hello") == generate_calls
    voice.close_voice()
    voice.close_voice()
    assert events.count("release") == 1


def test_chatterbox_load_failure_does_not_expose_reference_path(monkeypatch, reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)
    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: (_ for _ in ()).throw(RuntimeError(str(reference))))
    with pytest.raises(RuntimeError, match="Chatterbox Nano is unavailable") as error:
        tts.load_voice("reference-wav", engine="chatterbox", reference_wav=reference)
    assert str(reference) not in str(error.value)


def test_chatterbox_generation_failure_is_bounded_and_path_free(monkeypatch, reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)

    class Model:
        sr = 24000

        def generate(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(f"backend failed for {reference}")

    class ModelClass:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> Model:
            return Model()

    class Module:
        ChatterboxTurboTTS = ModelClass

    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: Module)
    voice = tts.load_voice("reference-wav", engine="chatterbox", reference_wav=reference)
    with pytest.raises(RuntimeError, match="^Chatterbox generation failed$") as error:
        voice.synthesize("hello")
    assert str(reference) not in str(error.value)


def test_chatterbox_wrapper_failure_releases_loaded_model(monkeypatch, reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)
    events: list[str] = []

    class Model:
        sr = 0

        def release(self) -> None:
            events.append("release")

    class ModelClass:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> Model:
            return Model()

    class Module:
        ChatterboxTurboTTS = ModelClass

    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: Module)
    with pytest.raises(RuntimeError, match="Chatterbox Nano is unavailable"):
        tts.load_voice("reference-wav", engine="chatterbox", reference_wav=reference)
    assert events == ["release"]


def test_chatterbox_rejects_model_sample_rate_mismatch(monkeypatch, reference_dir: Path) -> None:
    reference = _reference_wav(reference_dir)
    events: list[str] = []

    class Model:
        sr = 16000

        def release(self) -> None:
            events.append("release")

    class ModelClass:
        @staticmethod
        def from_pretrained(**_kwargs: object) -> Model:
            return Model()

    class Module:
        ChatterboxTurboTTS = ModelClass

    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: Module)
    with pytest.raises(RuntimeError, match="Chatterbox Nano is unavailable") as error:
        tts.load_voice("reference-wav", engine="chatterbox", reference_wav=reference)
    assert "sample rate" not in str(error.value).lower()
    assert events == ["release"]


def test_chatterbox_environment_pins_resolved_perth_commit() -> None:
    declaration = Path("engine_envs/chatterbox/pyproject.toml").read_text(encoding="utf-8")
    assert "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@ce86c49d029f42272c1902eccb675556b9ed2330" in declaration


def test_chunks_are_ordered_complete_and_sentence_safe() -> None:
    text = "A short sentence. " + ("A very long sentence " + "word " * 250 + ".") + " Final sentence."
    chunks = plan_chunks(text, [{"index": 1, "start_offset": 0, "end_offset": len(text)}], _metadata(), cap=30)
    assert [chunk.global_index for chunk in chunks] == list(range(len(chunks)))
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks)
    assert any(len(chunk.text) > 30 for chunk in chunks)


def test_chunk_hash_changes_for_every_output_setting() -> None:
    metadata = _metadata().as_dict()
    original = chunk_input_hash("same", metadata)
    for key, value in (("engine", "other"), ("package_version", "2"), ("model", "other"), ("model_revision", "r2"), ("model_checksum", "c3"), ("voice", "other"), ("voice_version", "v2"), ("voice_checksum", "c4"), ("sample_rate", 22050), ("settings", {"speed": 1.1})):
        changed = dict(metadata); changed[key] = value
        assert chunk_input_hash("same", changed) != original
    assert chunk_input_hash("changed", metadata) != original


def test_chunk_hash_binds_kokoro_implementation_only(monkeypatch) -> None:
    metadata_by_engine = {}
    for engine in ("kokoro", "chatterbox", "fake"):
        metadata = _metadata().as_dict()
        metadata["engine"] = engine
        metadata_by_engine[engine] = metadata
    original = {engine: chunk_input_hash("same", metadata) for engine, metadata in metadata_by_engine.items()}
    monkeypatch.setattr(tts, "KOKORO_SYNTHESIS_IMPLEMENTATION", "kokoro-synthesis-v3")
    assert chunk_input_hash("same", metadata_by_engine["kokoro"]) != original["kokoro"]
    assert chunk_input_hash("same", metadata_by_engine["chatterbox"]) == original["chatterbox"]
    assert chunk_input_hash("same", metadata_by_engine["fake"]) == original["fake"]


def test_short_sentences_group_to_soft_cap_but_long_sentence_stays_intact() -> None:
    text = "One. Two. Three. Four. " + ("Long " * 80) + ". Tail."
    chunks = plan_chunks(text, [{"index": 1, "start_offset": 0, "end_offset": len(text)}], _metadata(), cap=24)
    assert len(chunks) >= 3
    assert chunks[0].text == "One. Two. Three. Four. "
    assert any(len(chunk.text) > 24 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def _fake_torch(events: list[str]) -> object:
    class Context:
        def __enter__(self) -> None:
            events.append("inference-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("inference-exit")

    class Torch:
        @staticmethod
        def set_num_threads(value: int) -> None:
            events.append(f"threads:{value}")

        @staticmethod
        def inference_mode() -> Context:
            return Context()

    return Torch


def test_kokoro_configures_threads_before_pipeline_and_wraps_inference(monkeypatch) -> None:
    events: list[str] = []

    class Pipeline:
        def __init__(self, *, lang_code: str) -> None:
            events.append(f"pipeline:{lang_code}")

        def __call__(self, *_args: object, **_kwargs: object):
            events.append("pipeline-call")
            return iter([[0.0, 0.25]])

    class Kokoro:
        KPipeline = Pipeline

    def import_module(name: str) -> object:
        events.append(f"import:{name}")
        return _fake_torch(events) if name == "torch" else Kokoro

    monkeypatch.setenv(tts.TORCH_THREADS_ENV, "2")
    monkeypatch.setattr(tts.importlib, "import_module", import_module)
    voice = tts.load_voice("af_heart")
    assert len(voice.synthesize("hello")) == 4
    assert events.index("threads:2") < events.index("pipeline:a")
    assert events.index("inference-enter") < events.index("pipeline-call") < events.index("inference-exit")


def _fake_kokoro_voice(pipeline: object, events: list[str], *, sample_rate: int = 1000) -> tts.KokoroVoice:
    class Context:
        def __enter__(self) -> None:
            events.append("inference-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("inference-exit")

    return tts.KokoroVoice(
        pipeline,
        "af_heart",
        SynthesisSettings(sample_rate=sample_rate),
        inference_context=lambda: Context(),
    )


@pytest.mark.parametrize("text", [" \n\t", "!?…"])
def test_kokoro_silences_non_speakable_chunks_without_pipeline_call(text: str) -> None:
    events: list[str] = []

    class Pipeline:
        def __call__(self, *_args: object, **_kwargs: object):
            events.append("pipeline-call")
            return iter(())

    pcm = _fake_kokoro_voice(Pipeline(), events).synthesize(text)
    assert pcm == b"\x00\x00" * 50
    assert pcm and not any(pcm)
    assert events == []


def test_kokoro_sends_unicode_alphanumeric_text_to_pipeline() -> None:
    calls: list[str] = []
    events: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), events).synthesize("é١")
    assert pcm == b"\x00\x00"
    assert calls == ["é١"]
    assert events == ["inference-enter", "inference-exit"]


def test_kokoro_speakable_text_without_pipeline_output_still_raises() -> None:
    calls: list[str] = []
    events: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter(())

    with pytest.raises(RuntimeError, match="^Kokoro returned no audio$"):
        _fake_kokoro_voice(Pipeline(), events).synthesize("hello")
    assert calls == ["hello"]
    assert events == ["inference-enter", "inference-exit"]


def test_flowed_paragraphs_flows_wrapped_lines() -> None:
    assert tts.flowed_paragraphs("A wrapped\nline.") == ["A wrapped line."]


def test_flowed_paragraphs_preserves_blank_line_breaks() -> None:
    assert tts.flowed_paragraphs("First.\n\nSecond.") == ["First.", "Second."]


def test_flowed_paragraphs_drops_whitespace_only_blocks() -> None:
    assert tts.flowed_paragraphs(" \n\t\n\n First. \n\n \t ") == ["First."]


def test_flowed_paragraphs_preserves_leading_chapter_heading() -> None:
    assert tts.flowed_paragraphs("Chapter 1\nThe opening\ncontinues.") == ["Chapter 1", "The opening continues."]


def test_flowed_paragraphs_peels_consecutive_leading_headings() -> None:
    assert tts.flowed_paragraphs("Part I\nChapter 1\nThe opening.") == ["Part I", "Chapter 1", "The opening."]


def test_flowed_paragraphs_keeps_heading_only_block_single() -> None:
    assert tts.flowed_paragraphs("Chapter 1") == ["Chapter 1"]
    assert tts.flowed_paragraphs("Part I\nChapter 1") == ["Part I", "Chapter 1"]


def test_flowed_paragraphs_flows_nonmatching_heading_into_body() -> None:
    assert tts.flowed_paragraphs("PROLOGUE\nThe opening continues.") == ["PROLOGUE The opening continues."]


def test_kokoro_synthesizes_one_flowed_paragraph_per_pipeline_call() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    events: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **kwargs: object):
            calls.append((text, kwargs))
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), events).synthesize("First wrapped\nline.\n\nSecond")
    assert calls == [
        ("First wrapped line.", {"voice": "af_heart", "speed": 1.0, "split_pattern": None}),
        ("Second", {"voice": "af_heart", "speed": 1.0, "split_pattern": None}),
    ]
    assert pcm == b"\x00\x00" * 402
    assert events == ["inference-enter", "inference-exit"]


@pytest.mark.parametrize("ending", ['."', "!)", '\u2026"'])
def test_kokoro_adds_400ms_pause_after_sentence_final_paragraph(ending: str) -> None:
    calls: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), []).synthesize(f"First{ending}\n\nSecond")
    assert pcm == b"\x00\x00" + (b"\x00\x00" * 400) + b"\x00\x00"
    assert calls == [f"First{ending}", "Second"]


def test_kokoro_does_not_pause_after_non_sentence_final_paragraph() -> None:
    calls: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), []).synthesize("First\n\nSecond")
    assert pcm == b"\x00\x00" * 2
    assert calls == ["First", "Second"]


@pytest.mark.parametrize(
    ("text", "expected_length"),
    [("Chapter 1\nBody", 4), ("Chapter 1.\nBody", 804)],
)
def test_kokoro_heading_sentence_final_controls_pause(text: str, expected_length: int) -> None:
    calls: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), []).synthesize(text)
    assert len(pcm) == expected_length
    assert calls == ["Chapter 1" if text.startswith("Chapter 1\n") else "Chapter 1.", "Body"]


def test_kokoro_does_not_add_trailing_paragraph_pause() -> None:
    class Pipeline:
        def __call__(self, *_args: object, **_kwargs: object):
            return iter([[0.0]])

    assert _fake_kokoro_voice(Pipeline(), []).synthesize("First.\n\n") == b"\x00\x00"


def test_kokoro_filters_non_alphanumeric_scene_break_blocks() -> None:
    calls: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]])

    pcm = _fake_kokoro_voice(Pipeline(), []).synthesize("First.\n\n***\n\nSecond.")
    assert calls == ["First.", "Second."]
    assert len(pcm) == 804


def test_kokoro_surviving_no_audio_paragraph_raises_exact_message() -> None:
    calls: list[str] = []

    class Pipeline:
        def __call__(self, text: str, **_kwargs: object):
            calls.append(text)
            return iter([[0.0]]) if text == "First." else iter(())

    with pytest.raises(RuntimeError) as error:
        _fake_kokoro_voice(Pipeline(), []).synthesize("First.\n\nSecond.")
    assert str(error.value) == "Kokoro returned no audio"
    assert calls == ["First.", "Second."]


@pytest.mark.parametrize(("cpu_count", "expected_threads"), ((16, 8), (4, 4)))
def test_kokoro_unset_threads_uses_adaptive_default(monkeypatch, cpu_count: int, expected_threads: int) -> None:
    events: list[str] = []
    monkeypatch.delenv(tts.TORCH_THREADS_ENV, raising=False)
    monkeypatch.setattr(tts.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(tts.importlib, "import_module", lambda name: _fake_torch(events) if name == "torch" else type("Kokoro", (), {"KPipeline": lambda **_kwargs: lambda *_args, **_kwargs: iter([[0.0]])}))
    voice = tts.load_voice("af_heart")
    voice.synthesize("hello")
    assert f"threads:{expected_threads}" in events


def test_invalid_threads_fail_before_model_load(monkeypatch) -> None:
    monkeypatch.setenv(tts.TORCH_THREADS_ENV, "0")
    monkeypatch.setattr(tts.importlib, "import_module", lambda _name: (_ for _ in ()).throw(AssertionError("model import must not run")))
    with pytest.raises(ValueError, match="positive integer"):
        tts.load_voice("af_heart")


def _interactive_facts() -> dict[str, dict[str, object]]:
    facts = {
        "id": "af_heart", "engine": "kokoro", "package": "kokoro", "package_version": "0.9.4",
        "model": "model", "model_revision": "r1", "model_checksum": "m1",
        "voice_version": "v1", "voice_checksum": "w1", "sample_rate": 24000, "enabled": True,
    }
    return {"af_heart": facts}


def _interactive_plan(text: str, spans: list[dict[str, object]], *, revision: int = 1, state: str = "approved") -> dict[str, object]:
    plan = {
        "schema_version": 1,
        "artifact": "voice-plan",
        "revision": revision,
        "approval": {"state": state, "approved_revision": revision if state == "approved" else None},
        "cast": [
            {"cast_id": "narrator", "voice_id": "af_heart", "voice_settings": {"speed": 1.0}},
            {"cast_id": "alice", "voice_id": "af_heart", "voice_settings": {"speed": 1.2}},
        ],
        "chapters": [{"chapter_index": 1, "source_start": 0, "source_end": len(text), "spans": spans}],
    }
    return with_canonical_artifact_hash(plan)


def test_interactive_chunks_reconstruct_exactly_and_never_cross_span() -> None:
    text = "Narrátor says: “Áé.”  Alice replies!\n\n" + ("Long sentence " * 35) + "done."
    split = text.index("Alice")
    long_start = text.index("Long")
    plan = _interactive_plan(text, [
        {"span_id": "n", "source_start": 0, "source_end": split, "type": "narration", "speaker_id": "narrator"},
        {"span_id": "a", "source_start": split, "source_end": long_start, "type": "dialogue", "speaker_id": "alice"},
        {"span_id": "n2", "source_start": long_start, "source_end": len(text), "type": "narration", "speaker_id": "narrator"},
    ])
    chunks = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=40)
    assert "".join(chunk.text for chunk in chunks) == text
    assert any(chunk.span_id == "n2" for chunk in chunks)
    assert all(chunk.source_start >= (split if chunk.span_id != "n" else 0) for chunk in chunks)
    assert all(chunk.voice_id == "af_heart" for chunk in chunks)
    record = chunks[0].manifest_record("chunks/a.wav", 1.0)
    assert record["segment_type"] == "narration"
    assert record["span_id"] == "n"


def test_interactive_accepted_unknown_narrator_fallback_plans_narration_without_mutation() -> None:
    text = "Unresolved line."
    plan = _interactive_plan(text, [{"span_id": "u", "source_start": 0, "source_end": len(text), "type": "unknown", "speaker_id": "narrator"}])
    plan["unresolved_policy"] = {"mode": "narrator", "accepted_by_user": True, "accepted_at": "2026-01-01T00:00:00Z"}
    plan = with_canonical_artifact_hash(plan)
    original = json.loads(json.dumps(plan))
    chunks = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100)
    assert len(chunks) == 1 and chunks[0].segment_type == "narration" and chunks[0].voice_id == "af_heart"
    assert plan == original


@pytest.mark.parametrize(
    "policy, speaker_id",
    [
        (None, "narrator"),
        ({"mode": "narrator", "accepted_by_user": False, "accepted_at": None}, "narrator"),
        ({"mode": "narrator", "accepted_by_user": True, "accepted_at": None}, "narrator"),
        ({"mode": "narrator", "accepted_by_user": True, "accepted_at": "not-a-timestamp"}, "narrator"),
        ({"mode": "narrator", "accepted_by_user": True, "accepted_at": "2026-01-01T00:00:00Z"}, "alice"),
    ],
)
def test_interactive_rejects_unaccepted_or_non_narrator_unknown_fallback(policy, speaker_id: str) -> None:
    text = "Unresolved line."
    plan = _interactive_plan(text, [{"span_id": "u", "source_start": 0, "source_end": len(text), "type": "unknown", "speaker_id": speaker_id}])
    if policy is not None:
        plan["unresolved_policy"] = policy
    plan = with_canonical_artifact_hash(plan)
    with pytest.raises(ValueError, match="unknown spans"):
        plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100)


def test_interactive_selected_chapter_range_preserves_order_and_reconstructs_selection() -> None:
    parts = ["Chapter one. ", "Chapter two — naïve.  ", "Chapter three!\n"]
    text = "".join(parts)
    chapters = []
    cursor = 0
    for index, part in enumerate(parts, start=1):
        end = cursor + len(part)
        chapters.append({
            "chapter_index": index,
            "source_start": cursor,
            "source_end": end,
            "spans": [{
                "span_id": f"s{index}",
                "source_start": cursor,
                "source_end": end,
                "type": "narration",
                "speaker_id": "narrator",
            }],
        })
        cursor = end
    plan = with_canonical_artifact_hash({
        "schema_version": 1,
        "artifact": "voice-plan",
        "revision": 1,
        "approval": {"state": "approved", "approved_revision": 1},
        "cast": [{"cast_id": "narrator", "voice_id": "af_heart", "voice_settings": {"speed": 1.0}}],
        "chapters": chapters,
    })
    chunks = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, (2, 3), cap=100)
    assert "".join(chunk.text for chunk in chunks) == parts[1] + parts[2]
    assert [chunk.chapter_index for chunk in chunks] == [2, 3]
    assert [chunk.global_index for chunk in chunks] == [0, 1]
    assert [chunk.local_index for chunk in chunks] == [0, 0]


def test_interactive_hashes_bind_plan_registry_voice_and_offsets() -> None:
    text = "One. Two."
    base_span = {"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}
    plan = _interactive_plan(text, [base_span])
    baseline = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100)[0].input_hash
    changed_facts = _interactive_facts()
    changed_facts["af_heart"] = {**changed_facts["af_heart"], "model_revision": "r2"}
    assert plan_interactive_chunks(text, plan, changed_facts, "a" * 64, cap=100)[0].input_hash != baseline
    assert plan_interactive_chunks(text, plan, _interactive_facts(), "b" * 64, cap=100)[0].input_hash != baseline
    changed_plan = _interactive_plan(text, [base_span], revision=2)
    assert plan_interactive_chunks(text, changed_plan, _interactive_facts(), "a" * 64, cap=100)[0].input_hash != baseline
    changed_speed = _interactive_plan(text, [base_span])
    changed_speed["cast"][0]["voice_settings"]["speed"] = 1.1  # type: ignore[index]
    changed_speed = with_canonical_artifact_hash(changed_speed)
    assert plan_interactive_chunks(text, changed_speed, _interactive_facts(), "a" * 64, cap=100)[0].input_hash != baseline


def test_interactive_audio_hash_reuse_excludes_plan_and_registry_revisions() -> None:
    text = "One. Two."
    span = {"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}
    plan = _interactive_plan(text, [span])
    baseline = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100)[0]
    revised = plan_interactive_chunks(text, _interactive_plan(text, [span], revision=2), _interactive_facts(), "a" * 64, cap=100)[0]
    registry_changed = plan_interactive_chunks(text, plan, _interactive_facts(), "b" * 64, cap=100)[0]
    assert revised.input_hash != baseline.input_hash
    assert registry_changed.input_hash != baseline.input_hash
    assert revised.audio_input_hash == baseline.audio_input_hash == registry_changed.audio_input_hash


def test_interactive_hash_binds_kokoro_implementation_only(monkeypatch) -> None:
    text = "One."
    span = {"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}
    kokoro = plan_interactive_chunks(text, _interactive_plan(text, [span]), _interactive_facts(), "a" * 64, cap=100)[0]
    non_kokoro_facts = _interactive_facts()
    non_kokoro_facts["af_heart"] = {**non_kokoro_facts["af_heart"], "engine": "fake"}
    non_kokoro = plan_interactive_chunks(text, _interactive_plan(text, [span]), non_kokoro_facts, "a" * 64, cap=100)[0]
    monkeypatch.setattr(tts, "KOKORO_SYNTHESIS_IMPLEMENTATION", "kokoro-synthesis-v3")
    changed_kokoro = plan_interactive_chunks(text, _interactive_plan(text, [span]), _interactive_facts(), "a" * 64, cap=100)[0]
    changed_non_kokoro = plan_interactive_chunks(text, _interactive_plan(text, [span]), non_kokoro_facts, "a" * 64, cap=100)[0]
    assert changed_kokoro.audio_input_hash != kokoro.audio_input_hash
    assert changed_kokoro.input_hash != kokoro.input_hash
    assert changed_non_kokoro.audio_input_hash == non_kokoro.audio_input_hash
    assert changed_non_kokoro.input_hash == non_kokoro.input_hash


def test_interactive_audio_hash_changes_for_voice_settings_model_text_and_offsets() -> None:
    text = "One. Two."
    span = {"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}
    plan = _interactive_plan(text, [span])
    baseline = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100)[0].audio_input_hash

    voice_facts = _interactive_facts()
    voice_facts["af_bella"] = {**voice_facts["af_heart"], "id": "af_bella"}
    voice_plan = _interactive_plan(text, [span])
    voice_plan["cast"][0]["voice_id"] = "af_bella"  # type: ignore[index]
    voice_plan = with_canonical_artifact_hash(voice_plan)
    assert plan_interactive_chunks(text, voice_plan, voice_facts, "a" * 64, cap=100)[0].audio_input_hash != baseline

    settings_plan = _interactive_plan(text, [span])
    settings_plan["cast"][0]["voice_settings"]["speed"] = 1.1  # type: ignore[index]
    settings_plan = with_canonical_artifact_hash(settings_plan)
    assert plan_interactive_chunks(text, settings_plan, _interactive_facts(), "a" * 64, cap=100)[0].audio_input_hash != baseline

    model_facts = _interactive_facts()
    model_facts["af_heart"] = {**model_facts["af_heart"], "model_revision": "r2"}
    assert plan_interactive_chunks(text, plan, model_facts, "a" * 64, cap=100)[0].audio_input_hash != baseline

    changed_text = "One. Three."
    changed_span = {**span, "source_end": len(changed_text)}
    changed_text_plan = _interactive_plan(changed_text, [changed_span])
    assert plan_interactive_chunks(changed_text, changed_text_plan, _interactive_facts(), "a" * 64, cap=100)[0].audio_input_hash != baseline

    offset_plan = _interactive_plan(text, [
        {"span_id": "s1", "source_start": 0, "source_end": 4, "type": "narration", "speaker_id": "narrator"},
        {"span_id": "s2", "source_start": 4, "source_end": len(text), "type": "narration", "speaker_id": "narrator"},
    ])
    assert plan_interactive_chunks(text, offset_plan, _interactive_facts(), "a" * 64, cap=100)[0].audio_input_hash != baseline


def test_interactive_audio_hash_binds_shaping_but_not_plan_revision() -> None:
    text = "One. Two."
    span = {"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}
    plan = _interactive_plan(text, [span])
    baseline = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100, shaping_identity="shape-a")[0]
    changed_shape = plan_interactive_chunks(text, plan, _interactive_facts(), "a" * 64, cap=100, shaping_identity="shape-b")[0]
    revised = plan_interactive_chunks(text, _interactive_plan(text, [span], revision=2), _interactive_facts(), "a" * 64, cap=100, shaping_identity="shape-a")[0]
    assert changed_shape.audio_input_hash != baseline.audio_input_hash
    assert revised.audio_input_hash == baseline.audio_input_hash


@pytest.mark.parametrize("mutation", ["draft", "missing-facts", "disabled-facts", "missing-cast"])
def test_interactive_rejects_unapproved_or_unresolvable_inputs(mutation: str) -> None:
    text = "One."
    plan = _interactive_plan(text, [{"span_id": "s", "source_start": 0, "source_end": len(text), "type": "narration", "speaker_id": "narrator"}], state="draft" if mutation == "draft" else "approved")
    facts = _interactive_facts()
    if mutation == "missing-facts":
        facts = {}
    elif mutation == "disabled-facts":
        facts["af_heart"]["enabled"] = False
    elif mutation == "missing-cast":
        plan["chapters"][0]["spans"][0]["speaker_id"] = "unknown"  # type: ignore[index]
        plan = with_canonical_artifact_hash(plan)
    with pytest.raises(ValueError):
        plan_interactive_chunks(text, plan, facts, "a" * 64)

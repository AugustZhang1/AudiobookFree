from __future__ import annotations

import pdf_audiobook.tts as tts
import pytest
from pdf_audiobook.tts import EngineMetadata, SynthesisSettings, chunk_input_hash, plan_chunks


def _metadata() -> EngineMetadata:
    settings = SynthesisSettings(chunk_mode="legacy")
    return EngineMetadata("fake", "builtin", "fake", "r1", "c1", "fake-neutral", "v1", "c2", 24000, settings.as_dict())


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


def test_plan_chunks_rejects_explicit_unsupported_chunk_mode() -> None:
    metadata = _metadata().as_dict()
    metadata["settings"]["chunk_mode"] = "invalid"
    with pytest.raises(ValueError, match="unsupported chunk mode"):
        plan_chunks("One sentence.", [{"index": 1, "start_offset": 0, "end_offset": 14}], metadata)


def test_load_voice_rejects_unsupported_chunk_mode() -> None:
    with pytest.raises(ValueError, match="unsupported chunk mode"):
        tts.load_voice("fake-neutral", SynthesisSettings(chunk_mode="invalid"), engine="fake")


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

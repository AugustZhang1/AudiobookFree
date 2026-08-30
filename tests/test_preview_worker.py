from __future__ import annotations

from pathlib import Path
import shutil
import uuid
import time

import pytest

from pdf_audiobook.audio import validate_wav
from pdf_audiobook.audio import write_pcm_wav
from pdf_audiobook.chatterbox_preview_worker import BUILTIN_PREVIEW_TEXT as CHATTERBOX_BUILTIN_PREVIEW_TEXT, PREVIEW_TEXT as CHATTERBOX_PREVIEW_TEXT, generate_builtin_preview as generate_chatterbox_builtin_preview, generate_preview as generate_chatterbox_preview
from pdf_audiobook.preview_worker import PREVIEW_TEXT, cleanup_preview_cache, generate_preview, preview_cache_key, preview_cache_target


class _FakeVoice:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.closed = False
        self.text = None

    def synthesize(self, text: str) -> bytes:
        self.text = text
        if self.fail:
            raise RuntimeError("fake synthesis failure")
        return b"\0\0" * 240

    def close_voice(self) -> None:
        self.closed = True


def test_generate_preview_uses_fixed_sentence_and_publishes_valid_wav() -> None:
    root = Path("tests") / f".pytest-preview-worker-{uuid.uuid4().hex}"
    root.mkdir()
    target = root / "sample-kokoro-af_heart.wav"
    fake = _FakeVoice()
    try:
        result = generate_preview("af_heart", target, voice_loader=lambda voice, settings: fake)
        assert result == target
        assert fake.text == PREVIEW_TEXT and fake.closed
        assert validate_wav(target, expected_sample_rate=24000).frames == 240
        assert not list(root.glob(".*.tmp"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generate_preview_closes_voice_and_cleans_temp_on_failure() -> None:
    root = Path("tests") / f".pytest-preview-worker-failure-{uuid.uuid4().hex}"
    root.mkdir()
    target = root / "sample-kokoro-af_heart.wav"
    fake = _FakeVoice(fail=True)
    try:
        with pytest.raises(RuntimeError, match="fake synthesis failure"):
            generate_preview("af_heart", target, voice_loader=lambda voice, settings: fake)
        assert fake.closed and not target.exists() and not list(root.glob(".*.tmp"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generate_preview_validates_voice_before_loading() -> None:
    loaded = False

    def loader(voice, settings):
        nonlocal loaded
        loaded = True
        raise AssertionError("loader should not run")

    with pytest.raises(ValueError):
        generate_preview("not-a-voice", Path("tests") / "preview.wav", voice_loader=loader)
    assert not loaded


def test_preview_cache_identity_includes_settings_facts_and_shaping(monkeypatch) -> None:
    root = Path("tests") / f".pytest-preview-cache-{uuid.uuid4().hex}"
    root.mkdir()
    facts = {"id": "af_heart", "engine": "kokoro", "model": "m", "model_checksum": "c", "voice_checksum": "v"}
    monkeypatch.setattr("pdf_audiobook.preview_worker.get_generation_facts", lambda _voice: facts)
    monkeypatch.setattr("pdf_audiobook.preview_worker.shaping_fingerprint", lambda: "shape-a")
    neutral = {"speed": 1, "pitch_semitones": 0, "tone_preset": "neutral"}
    changed = {"speed": 1.1, "pitch_semitones": 0, "tone_preset": "neutral"}
    baseline = preview_cache_key("af_heart", neutral)
    assert baseline != preview_cache_key("af_heart", changed)
    facts["voice_checksum"] = "changed"
    assert baseline != preview_cache_key("af_heart", neutral)
    assert preview_cache_target(root, "af_heart", neutral).name == "sample-kokoro-af_heart.wav"
    assert preview_cache_target(root, "af_heart", changed).parent.name == ".voice-preview-cache"
    shutil.rmtree(root, ignore_errors=True)


def test_preview_cache_cleanup_is_bounded() -> None:
    root = Path("tests") / f".pytest-preview-cleanup-{uuid.uuid4().hex}"
    cache = root / ".voice-preview-cache"
    cache.mkdir(parents=True)
    for index in range(3):
        (cache / f"{index}.wav").write_bytes(b"RIFF")
        time.sleep(0.01)
    assert cleanup_preview_cache(root, max_files=1, max_age_seconds=3600) == 2
    assert len(list(cache.glob("*.wav"))) == 1
    shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_preview_worker_uses_reference_voice_and_closes_fake() -> None:
    root = Path("tests") / f".pytest-chatterbox-preview-{uuid.uuid4().hex}"; root.mkdir()
    reference = root / "reference.wav"; target = root / "preview.wav"
    write_pcm_wav(reference, b"\0\0" * (24000 * 6), 24000, overwrite=True)
    class FakeVoice:
        closed = False
        text = None
        def synthesize(self, text):
            self.text = text
            return b"\0\0" * 240
        def close_voice(self):
            self.closed = True
    fake = FakeVoice(); calls = []
    try:
        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return fake
        assert generate_chatterbox_preview(reference, target, voice_loader=loader) == target
        assert validate_wav(target, expected_sample_rate=24000).frames == 240
        assert fake.closed and fake.text == CHATTERBOX_PREVIEW_TEXT
        assert calls[0][0][0] == "reference-wav" and calls[0][1]["engine"] == "chatterbox"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_preview_worker_cleans_temp_on_failure_and_rejects_unsafe_reference() -> None:
    root = Path("tests") / f".pytest-chatterbox-preview-failure-{uuid.uuid4().hex}"; root.mkdir()
    reference = root / "reference.wav"; target = root / "preview.wav"
    write_pcm_wav(reference, b"\0\0" * (24000 * 6), 24000, overwrite=True)
    class FakeVoice:
        closed = False
        def synthesize(self, _text):
            raise RuntimeError("fake failure")
        def close_voice(self):
            self.closed = True
    fake = FakeVoice()
    try:
        with pytest.raises(RuntimeError, match="fake failure"):
            generate_chatterbox_preview(reference, target, voice_loader=lambda *args, **kwargs: fake)
        assert fake.closed and not target.exists() and not list(root.glob(".*.tmp"))
        with pytest.raises(ValueError):
            generate_chatterbox_preview(root, target, voice_loader=lambda *args, **kwargs: fake)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_builtin_preview_omits_reference_and_publishes_atomically() -> None:
    root = Path("tests") / f".pytest-chatterbox-builtin-preview-{uuid.uuid4().hex}"; root.mkdir()
    target = root / "builtin.wav"
    fake = _FakeVoice(); calls = []
    try:
        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return fake
        assert generate_chatterbox_builtin_preview(target, voice_loader=loader) == target
        assert validate_wav(target, expected_sample_rate=24000).frames == 240
        assert fake.text == CHATTERBOX_BUILTIN_PREVIEW_TEXT and fake.closed
        assert calls[0][0][0] == "builtin" and calls[0][1] == {"engine": "chatterbox"}
        assert calls[0][0][1].sample_rate == 24000 and calls[0][0][1].chunk_cap == 300
        assert not list(root.glob(".*.tmp"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_chatterbox_builtin_preview_cleans_temp_on_failure() -> None:
    root = Path("tests") / f".pytest-chatterbox-builtin-preview-failure-{uuid.uuid4().hex}"; root.mkdir()
    target = root / "builtin.wav"
    fake = _FakeVoice(fail=True)
    try:
        with pytest.raises(RuntimeError, match="fake synthesis failure"):
            generate_chatterbox_builtin_preview(target, voice_loader=lambda *args, **kwargs: fake)
        assert fake.closed and not target.exists() and not list(root.glob(".*.tmp"))
    finally:
        shutil.rmtree(root, ignore_errors=True)

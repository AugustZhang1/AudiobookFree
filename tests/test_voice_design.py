from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import uuid
import wave

import pytest

from pdf_audiobook import cast, voice_design
from pdf_audiobook.chatterbox_reference import validate_reference_file

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg converts the designed sample")


@pytest.fixture
def work():
    path = Path.cwd() / f".design-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _FakeDesigner:
    def __init__(self, seconds: float = 12.0, rate: int = 24000):
        self.calls: list[dict] = []
        self.seconds = seconds
        self.rate = rate

    def generate_voice_design(self, *, text, language, instruct):
        self.calls.append({"text": text, "language": language, "instruct": instruct})
        return [[0.3 * math.sin(2 * math.pi * 180 * n / self.rate) for n in range(int(self.rate * self.seconds))]], self.rate


def test_designed_voice_is_a_valid_reference_clip_with_its_description(work: Path) -> None:
    model = _FakeDesigner(rate=48000)
    target = voice_design.design_voice("  A gruff   old man ", work / "Gruff.wav", model=model)
    assert model.calls == [{"text": voice_design.SAMPLE_TEXT, "language": "English", "instruct": "A gruff old man"}]
    with wave.open(str(target)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 24000)
    validate_reference_file(target)  # what the library and Chatterbox require
    assert target.with_suffix(".txt").read_text(encoding="utf-8") == "A gruff old man"


def test_too_short_sample_is_rejected(work: Path) -> None:
    with pytest.raises(voice_design.VoiceDesignError, match="only 3.0 seconds"):
        voice_design.design_voice("A whisper", work / "Short.wav", model=_FakeDesigner(seconds=3))
    assert not (work / "Short.wav").exists()


def test_batch_loads_the_model_once_and_skips_voices_already_made(work: Path) -> None:
    model = _FakeDesigner()
    loads: list[int] = []

    def loader():
        loads.append(1)
        return model

    (work / "Mara.wav").write_bytes(b"already designed")
    lines: list[str] = []
    done = voice_design.design_voices([("A young woman", work / "Mara.wav"), ("An old man", work / "Tom.wav"), ("A boy", work / "Kid.wav")], model_loader=loader, progress=lines.append)
    assert done == [work / "Mara.wav", work / "Tom.wav", work / "Kid.wav"]
    assert len(loads) == 1 and [call["instruct"] for call in model.calls] == ["An old man", "A boy"]
    assert lines[0] == f"Designed voice 1/3: {work / 'Mara.wav'}"
    assert voice_design.design_voices([("x y z", work / "Tom.wav")], model_loader=lambda: pytest.fail("nothing to design")) == [work / "Tom.wav"]


def test_presets_are_usable_descriptions() -> None:
    assert {"Gruff old man", "Charming woman"} <= set(voice_design.PRESETS)
    assert all(3 <= len(text) <= 500 for text in voice_design.PRESETS.values())


def test_gemini_casting_sees_a_designed_voices_description(work: Path) -> None:
    clip = work / "Gruff old man.wav"
    clip.write_bytes(b"x")
    clip.with_suffix(".txt").write_text("A gruff old man with a gravelly voice.", encoding="utf-8")
    choices = dict(cast.voice_choices([("★ Gruff old man · your voice", str(clip))], "af_heart"))
    assert choices[str(clip)] == "Gruff old man: A gruff old man with a gravelly voice."


def test_cli_batch(work: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(voice_design, "_load_model", lambda: _FakeDesigner())
    batch = work / "jobs.json"
    batch.write_text(json.dumps([{"description": "A charming woman", "out": str(work / "Charm.wav")}]), encoding="utf-8")
    assert voice_design.main(["--batch", str(batch)]) == 0
    assert capsys.readouterr().out.strip() == f"Designed voice 1/1: {work / 'Charm.wav'}"

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

from pdf_audiobook.audio import deterministic_pcm, validate_wav, write_pcm_wav


@pytest.fixture
def workdir() -> Path:
    path = Path("tests") / f".pytest-phase4-audio-{uuid.uuid4().hex}"; path.mkdir()
    try: yield path
    finally: shutil.rmtree(path, ignore_errors=True)


def test_atomic_pcm_wav_round_trip_and_rejection(workdir: Path) -> None:
    path = workdir / "chunk.wav"
    info = write_pcm_wav(path, deterministic_pcm("x", 2400), 24000)
    assert info.duration_seconds == pytest.approx(0.1)
    assert validate_wav(path, expected_sample_rate=24000).frames == 2400
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError): validate_wav(path)


def test_existing_valid_wav_is_not_overwritten_by_default(workdir: Path) -> None:
    path = workdir / "chunk.wav"; write_pcm_wav(path, deterministic_pcm("x", 10), 24000)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): write_pcm_wav(path, deterministic_pcm("y", 10), 24000)
    assert path.read_bytes() == before

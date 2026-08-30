from __future__ import annotations

from array import array
from pathlib import Path
import shutil
import sys
import subprocess
import uuid

import pytest

from pdf_audiobook.audio import deterministic_pcm, pcm_from_audio, validate_wav, write_pcm_wav
import pdf_audiobook.voice_shaping as shaping
from pdf_audiobook.voice_shaping import BRIGHT_FILTER, SHAPING_IMPLEMENTATION, WARM_FILTER, ShapingCapability, VoiceShapingError, VoiceShapingUnavailable, shape_pcm


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


def test_pcm_from_audio_uses_optional_numpy_dtype_fast_path_and_keeps_list_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeDType:
        kind = "f"

    class FakeArray:
        dtype = FakeDType()
        size = 2

        def reshape(self, *shape: int) -> "FakeArray":
            calls.append(("reshape", shape))
            return self

        def __mul__(self, scale: float) -> "FakeArray":
            calls.append(("scale", scale))
            return self

        def astype(self, dtype: str) -> "FakeArray":
            calls.append(("astype", dtype))
            return self

        def tobytes(self) -> bytes:
            calls.append(("tobytes", None))
            return b"numpy-fast-path"

        def tolist(self) -> list[float]:
            raise AssertionError("fast path must not materialize a Python list")

    class FakeAll:
        @staticmethod
        def all() -> bool:
            return True

    class FakeNumpy:
        ndarray = FakeArray

        @staticmethod
        def clip(value: FakeArray, lower: int, upper: int) -> FakeArray:
            calls.append(("clip", (lower, upper)))
            return value

        @staticmethod
        def isfinite(value: FakeArray) -> FakeAll:
            calls.append(("isfinite", None))
            return FakeAll()

        @staticmethod
        def rint(value: FakeArray) -> FakeArray:
            # Round-half-to-even, matching the Python fallback's int(round(...)).
            calls.append(("rint", None))
            return value

    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())
    assert pcm_from_audio(FakeArray()) == b"numpy-fast-path"
    assert ("scale", 32767.0) in calls
    assert ("isfinite", None) in calls
    assert ("rint", None) in calls
    assert ("clip", (-32768, 32767)) in calls
    assert ("astype", "<i2") in calls
    assert ("tobytes", None) in calls
    assert pcm_from_audio([0.5, -1.0]) == array("h", [16384, -32767]).tobytes()


def _shaping_capability(*, rubberband: bool = True, tone: bool = True) -> ShapingCapability:
    return ShapingCapability("ffmpeg.exe", rubberband, "ffmpeg-test", "rubberband-test", "fingerprint-test", tone)


def test_neutral_shape_bypasses_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(shaping, "shaping_capability", lambda: (_ for _ in ()).throw(AssertionError("neutral must bypass")))
    pcm = b"\x01\x00" * 4
    assert shape_pcm(pcm, 24000, {"speed": 1, "pitch_semitones": 0, "tone_preset": "neutral"}) == pcm


def test_pitch_and_tone_build_valid_filter_argv(monkeypatch) -> None:
    monkeypatch.setattr(shaping, "shaping_capability", lambda: _shaping_capability())
    calls = []

    class Result:
        returncode = 0
        stdout = b"\x00\x00" * 4
        stderr = b""

    monkeypatch.setattr(shaping.subprocess, "run", lambda argv, **kwargs: (calls.append((argv, kwargs)) or Result()))
    output = shape_pcm(b"\x01\x00" * 4, 24000, {"speed": 1, "pitch_semitones": 2, "tone_preset": "warm"})
    assert output and calls
    argv = calls[0][0]
    filtergraph = argv[argv.index("-af") + 1]
    assert argv[0] == "ffmpeg.exe" and filtergraph.startswith("rubberband=pitch=")
    assert "pitchq=quality" in filtergraph and "quality=quality" not in filtergraph
    assert WARM_FILTER == "volume=0.70794578,lowshelf=f=180:g=3:w=0.7,highshelf=f=3600:g=-2:w=0.7"
    assert WARM_FILTER in filtergraph and "volume=0.70794578" in filtergraph
    assert calls[0][1]["shell"] is False


@pytest.mark.parametrize("tone, expected", [("warm", WARM_FILTER), ("bright", BRIGHT_FILTER)])
def test_tone_presets_use_opposing_broad_shelves(monkeypatch, tone: str, expected: str) -> None:
    assert SHAPING_IMPLEMENTATION == "voice-shaping-v3"
    monkeypatch.setattr(shaping, "shaping_capability", lambda: _shaping_capability())

    class Result:
        returncode = 0
        stdout = b"\x00\x00" * 4
        stderr = b""

    calls = []
    monkeypatch.setattr(shaping.subprocess, "run", lambda argv, **kwargs: (calls.append(argv) or Result()))
    shape_pcm(b"\x01\x00" * 4, 24000, {"speed": 1, "pitch_semitones": 0, "tone_preset": tone})
    filtergraph = calls[0][calls[0].index("-af") + 1]
    assert filtergraph == expected
    assert "g=3" in filtergraph and "g=-2" in filtergraph


def test_missing_capabilities_reject_requested_effects(monkeypatch) -> None:
    monkeypatch.setattr(shaping, "shaping_capability", lambda: _shaping_capability(rubberband=False, tone=False))
    with pytest.raises(VoiceShapingUnavailable, match="rubberband"):
        shape_pcm(b"\x00\x00", 24000, {"speed": 1, "pitch_semitones": 1, "tone_preset": "neutral"})
    with pytest.raises(VoiceShapingUnavailable, match="shelf"):
        shape_pcm(b"\x00\x00", 24000, {"speed": 1, "pitch_semitones": 0, "tone_preset": "warm"})


@pytest.mark.parametrize("result", [b"", b"\x00"])
def test_empty_or_misaligned_shaping_output_is_rejected(monkeypatch, result: bytes) -> None:
    monkeypatch.setattr(shaping, "shaping_capability", lambda: _shaping_capability())

    class Result:
        returncode = 0
        stdout = result
        stderr = b""

    monkeypatch.setattr(shaping.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(VoiceShapingError, match="empty or misaligned"):
        shape_pcm(b"\x00\x00", 24000, {"speed": 1, "pitch_semitones": 1, "tone_preset": "neutral"})


def test_shaping_timeout_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(shaping, "shaping_capability", lambda: _shaping_capability())

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 20)

    monkeypatch.setattr(shaping.subprocess, "run", timeout)
    with pytest.raises(VoiceShapingError, match="timed out"):
        shape_pcm(b"\x00\x00", 24000, {"speed": 1, "pitch_semitones": 1, "tone_preset": "neutral"})


def test_numpy_pcm_fast_path_matches_the_python_fallback() -> None:
    """The fast path must be numerically identical, not merely faster.

    astype() truncates toward zero while the fallback uses int(round(...)),
    which is round-half-to-even: 0.5*32767 = 16383.5 differed by one LSB on
    every half-way sample before np.rint was used.
    """

    np = pytest.importorskip("numpy")
    values = [0.0, 0.5 / 32767, 1.5 / 32767, 2.5 / 32767, 0.5, -0.5, 1.0, -1.0, 0.99999, -0.99999]

    fallback = array("h", [max(-32768, min(32767, int(round(v * 32767.0)))) for v in values]).tobytes()
    fast = pcm_from_audio(np.array(values, dtype=np.float64))

    assert fast == fallback


def test_numpy_pcm_fast_path_rejects_non_finite_samples() -> None:
    """The fallback raises on NaN/inf; casting them would emit silent garbage."""

    np = pytest.importorskip("numpy")
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            pcm_from_audio(np.array([0.1, bad, 0.2], dtype=np.float64))

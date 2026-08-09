"""Tiny engine boundary with lazy official candidate adapters and a fake engine."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from importlib import metadata
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .audio import pcm_from_audio, deterministic_pcm

KOKORO_MODEL_ID = "hexgrad/Kokoro-82M"
CHATTERBOX_MODEL_ID = "ResembleAI/chatterbox-nano"
CHATTERBOX_SOURCE_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"

KOKORO_VOICES = (
    ("af_heart", "American English — Heart"),
    ("af_bella", "American English — Bella"),
    ("af_nicole", "American English — Nicole"),
    ("af_aoede", "American English — Aoede"),
    ("af_kore", "American English — Kore"),
    ("af_sarah", "American English — Sarah"),
    ("af_nova", "American English — Nova"),
    ("af_sky", "American English — Sky"),
    ("af_alloy", "American English — Alloy"),
    ("af_jessica", "American English — Jessica"),
    ("af_river", "American English — River"),
    ("am_michael", "American English — Michael"),
    ("am_fenrir", "American English — Fenrir"),
    ("am_puck", "American English — Puck"),
    ("am_echo", "American English — Echo"),
)
KOKORO_BRITISH_VOICES = (("bf_emma", "British English - Emma"), ("bf_isabella", "British English - Isabella"))
KOKORO_VOICES = KOKORO_VOICES + KOKORO_BRITISH_VOICES
CHATTERBOX_VOICES = (("builtin", "Chatterbox Nano built-in/default"), ("reference-wav", "Reference WAV (path supplied at run time)"))


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str


class LoadedEngine(Protocol):
    engine_id: str
    package_version: str
    model_id: str
    model_revision: str | None
    model_checksum: str | None
    sample_rate: int
    runtime_facts: dict[str, Any]

    def synthesize(self, text: str, voice: str, speed: float, reference_wav: Path | None = None) -> bytes: ...

    def close(self) -> None: ...


def voices_for_engine(engine: str) -> tuple[Voice, ...]:
    """Return static voice metadata without importing or loading any model."""

    if engine == "fake":
        return (Voice("fake-neutral", "Deterministic fake voice"),)
    if engine == "kokoro":
        return tuple(Voice(*voice) for voice in KOKORO_VOICES)
    if engine == "chatterbox":
        return tuple(Voice(*voice) for voice in CHATTERBOX_VOICES)
    raise ValueError(f"unknown engine: {engine}")


class FakeEngine:
    engine_id = "fake"
    package_version = "builtin"
    model_id = "deterministic-fake"
    model_revision = "phase0"
    model_checksum = "builtin"
    sample_rate = 22050
    runtime_facts = {"configured_torch_threads": None, "effective_torch_threads": None, "inference_mode": False, "compile": False}

    def synthesize(self, text: str, voice: str, speed: float, reference_wav: Path | None = None) -> bytes:
        if voice != "fake-neutral":
            raise ValueError(f"unknown fake voice: {voice}")
        if speed <= 0:
            raise ValueError("speed must be positive")
        words = max(1, len(text.split()))
        frames = max(1, int((0.32 * words + 0.25) * self.sample_rate / speed))
        return deterministic_pcm(f"{voice}|{speed:.6f}|{text}", frames)

    def close(self) -> None:
        return None


class KokoroEngine:
    engine_id = "kokoro"
    package_version = "unresolved until package load"
    model_id = KOKORO_MODEL_ID
    model_revision = "captured-at-download"
    model_checksum = None
    sample_rate = 24000

    def __init__(self, pipeline: Any, *, inference_context: Any = None, runtime_facts: dict[str, Any] | None = None) -> None:
        self.pipeline = pipeline
        self._inference_context = nullcontext if inference_context is None else inference_context
        self.runtime_facts = runtime_facts or {"configured_torch_threads": None, "effective_torch_threads": None, "inference_mode": False, "compile": False}
        self.package_version = _package_version("kokoro", "0.9.4 candidate declaration; package metadata unavailable")

    def synthesize(self, text: str, voice: str, speed: float, reference_wav: Path | None = None) -> bytes:
        if reference_wav is not None:
            raise ValueError("Kokoro does not use reference WAV prompts")
        chunks: list[bytes] = []
        with self._inference_context():
            for result in self.pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
                if hasattr(result, "audio"):
                    audio = result.audio
                elif isinstance(result, tuple):
                    audio = result[2] if len(result) >= 3 else result[0] if result else None
                else:
                    audio = result
                if audio is None:
                    raise ValueError("Kokoro returned a result without audio")
                pcm = pcm_from_audio(audio)
                if not pcm:
                    raise ValueError("Kokoro returned empty audio")
                chunks.append(pcm)
        if not chunks:
            raise ValueError("Kokoro returned no audio")
        return b"".join(chunks)

    def close(self) -> None:
        return None


class ChatterboxEngine:
    engine_id = "chatterbox"
    package_version = "unresolved until package load"
    model_id = CHATTERBOX_MODEL_ID
    model_revision = CHATTERBOX_SOURCE_COMMIT
    model_checksum = None
    runtime_facts = {"configured_torch_threads": None, "effective_torch_threads": None, "inference_mode": False, "compile": False}

    def __init__(self, model: Any) -> None:
        self.model = model
        self.package_version = _package_version("chatterbox-tts", "0.1.7 / source commit for Nano support")
        self.sample_rate = int(getattr(model, "sr", getattr(model, "sampling_rate", 24000)))

    def synthesize(self, text: str, voice: str, speed: float, reference_wav: Path | None = None) -> bytes:
        if speed != 1.0:
            raise ValueError("Chatterbox Nano adapter currently supports speed 1.0 only")
        if voice == "reference-wav" and reference_wav is None:
            raise ValueError("--reference-wav is required for reference-wav voice")
        if voice not in {"builtin", "reference-wav"}:
            raise ValueError(f"unknown Chatterbox voice: {voice}")
        kwargs = {"audio_prompt_path": str(reference_wav)} if reference_wav else {}
        result = self.model.generate(text, **kwargs)
        return pcm_from_audio(result)

    def close(self) -> None:
        return None


def load_engine(
    engine: str,
    voice: str | None = None,
    *,
    torch_threads: int | None = None,
    torch_compile: bool = False,
    inference_mode: bool = True,
) -> tuple[LoadedEngine, float]:
    """Load one engine lazily and return it with model-init elapsed seconds."""

    started = time.perf_counter()
    if engine == "fake":
        if torch_threads is not None or torch_compile:
            raise ValueError("--torch-threads and --torch-compile are only valid with Kokoro")
        loaded: LoadedEngine = FakeEngine()
    elif engine == "kokoro":
        validate_torch_threads(torch_threads)
        if voice is None:
            raise ValueError("Kokoro voice is required to select its language pipeline")
        if voice.startswith("a"):
            lang_code = "a"
        elif voice.startswith("b"):
            lang_code = "b"
        else:
            raise ValueError(f"cannot derive Kokoro language from voice {voice!r}")
        try:
            torch = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(f"PyTorch import failed; Kokoro runtime tuning is unavailable: {exc}") from exc
        if torch_threads is not None:
            setter = getattr(torch, "set_num_threads", None)
            if not callable(setter):
                raise RuntimeError("PyTorch does not expose set_num_threads; cannot configure --torch-threads")
            setter(torch_threads)
        get_threads = getattr(torch, "get_num_threads", None)
        effective_threads = get_threads() if callable(get_threads) else None
        if inference_mode:
            inference_context = getattr(torch, "inference_mode", None)
            if not callable(inference_context):
                raise RuntimeError("PyTorch does not expose inference_mode; use --no-inference-mode only for an explicit baseline")
        else:
            inference_context = nullcontext
        try:
            module = importlib.import_module("kokoro")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(f"Kokoro package import failed or is not installed: {exc}") from exc
        try:
            pipeline_class = getattr(module, "KPipeline")
        except AttributeError as exc:
            raise RuntimeError(f"Kokoro package has no KPipeline: {exc}") from exc
        try:
            pipeline = pipeline_class(lang_code=lang_code)
            if torch_compile:
                compiler = getattr(torch, "compile", None)
                model = getattr(pipeline, "model", None)
                if not callable(compiler):
                    raise RuntimeError("PyTorch does not expose torch.compile")
                if model is None:
                    raise RuntimeError("Kokoro pipeline has no model to compile")
                forward_with_tokens = getattr(model, "forward_with_tokens", None)
                if not callable(forward_with_tokens):
                    raise RuntimeError("Kokoro model has no callable forward_with_tokens boundary to compile")
                try:
                    model.forward_with_tokens = compiler(forward_with_tokens)
                except Exception as exc:
                    raise RuntimeError(f"torch.compile failed for Kokoro model.forward_with_tokens: {exc}") from exc
            runtime_facts = {
                "configured_torch_threads": torch_threads,
                "effective_torch_threads": effective_threads,
                "inference_mode": inference_mode,
                "compile": torch_compile,
            }
            loaded = KokoroEngine(pipeline, inference_context=inference_context, runtime_facts=runtime_facts)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and ("torch.compile" in str(exc) or "inference_mode" in str(exc) or "set_num_threads" in str(exc) or "no model" in str(exc)):
                raise
            raise RuntimeError(f"Kokoro pipeline initialization failed: {exc}") from exc
    elif engine == "chatterbox":
        if torch_threads is not None or torch_compile:
            raise ValueError("--torch-threads and --torch-compile are only valid with Kokoro")
        try:
            module = importlib.import_module("chatterbox.tts_turbo")
            model_class = getattr(module, "ChatterboxTurboTTS")
            loaded = ChatterboxEngine(model_class.from_pretrained(device="cpu", nano=True))
        except Exception as exc:
            raise RuntimeError("Chatterbox Nano is not installed in the isolated candidate environment") from exc
    else:
        raise ValueError(f"unknown engine: {engine}")
    return loaded, time.perf_counter() - started


def _package_version(distribution: str, fallback: str) -> str:
    """Capture installed distribution version, retaining an honest source fallback."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def validate_torch_threads(value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError("--torch-threads must be a positive integer")
    cpu_count = os.cpu_count()
    if cpu_count is not None and value > cpu_count:
        raise ValueError(f"--torch-threads must be no greater than os.cpu_count() ({cpu_count})")

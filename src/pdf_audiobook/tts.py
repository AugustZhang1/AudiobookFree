"""Small, lazy TTS boundary and deterministic sentence-safe chunk planning.

The application only knows about this narrow contract.  Kokoro is imported on
the worker side, so normal application startup and tests never import a model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, ContextManager, Protocol

from .audio import pcm_from_audio

APPROVED_VOICES = ("af_heart", "af_bella", "bf_emma", "bf_isabella")
KOKORO_PACKAGE = "kokoro"
KOKORO_PACKAGE_VERSION = "0.9.4"
KOKORO_MODEL = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_CHUNK_CAP = 900
DEFAULT_TORCH_THREADS = 8
TORCH_THREADS_ENV = "PDF_AUDIOBOOK_TORCH_THREADS"
CHUNK_MODES = ("chapter", "legacy")


@dataclass(frozen=True)
class SynthesisSettings:
    speed: float = 1.0
    sample_rate: int = KOKORO_SAMPLE_RATE
    chunk_cap: int = DEFAULT_CHUNK_CAP
    chunk_mode: str = "chapter"
    paragraph_pause_ms: int = 0
    sentence_pause_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineMetadata:
    engine: str
    package_version: str
    model: str
    model_revision: str
    model_checksum: str
    voice: str
    voice_version: str
    voice_checksum: str
    sample_rate: int
    settings: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoadedVoice(Protocol):
    metadata: EngineMetadata

    def synthesize(self, text: str) -> bytes: ...

    def close_voice(self) -> None: ...


class FakeVoice:
    """Deterministic test engine; intentionally not exposed by the UI."""

    def __init__(self, voice: str = "fake-neutral", settings: SynthesisSettings | None = None):
        from .audio import deterministic_pcm

        self.settings = settings or SynthesisSettings(sample_rate=24000)
        self.metadata = EngineMetadata(
            "fake", "builtin", "deterministic-fake", "phase4", "builtin",
            voice, "builtin", "builtin", self.settings.sample_rate, self.settings.as_dict(),
        )
        self._pcm = deterministic_pcm

    def synthesize(self, text: str) -> bytes:
        words = max(1, len(text.split()))
        frames = max(1, int((0.32 * words + 0.25) * self.settings.sample_rate / self.settings.speed))
        return self._pcm(f"{self.metadata.voice}|{self.settings.speed:.6f}|{text}", frames)

    def close_voice(self) -> None:
        return None


class KokoroVoice:
    def __init__(
        self,
        pipeline: Any,
        voice: str,
        settings: SynthesisSettings,
        *,
        inference_context: Callable[[], ContextManager[Any]],
    ):
        self.pipeline = pipeline
        self._inference_context = inference_context
        self.metadata = EngineMetadata(
            "kokoro", KOKORO_PACKAGE_VERSION, KOKORO_MODEL,
            "captured-at-download", "unrecorded", voice, "captured-at-download",
            "unrecorded", settings.sample_rate, settings.as_dict(),
        )

    def synthesize(self, text: str) -> bytes:
        pieces: list[bytes] = []
        with self._inference_context():
            for result in self.pipeline(text, voice=self.metadata.voice, speed=self.metadata.settings["speed"], split_pattern=r"\n+"):
                audio = getattr(result, "audio", result[2] if isinstance(result, tuple) and len(result) >= 3 else result)
                pcm = pcm_from_audio(audio)
                if not pcm:
                    raise RuntimeError("Kokoro returned empty audio")
                pieces.append(pcm)
        if not pieces:
            raise RuntimeError("Kokoro returned no audio")
        return b"".join(pieces)

    def close_voice(self) -> None:
        return None


def load_voice(voice: str, settings: SynthesisSettings | None = None, *, engine: str = "kokoro") -> LoadedVoice:
    """Load a voice lazily.  No model import occurs until this function runs."""

    settings = settings or SynthesisSettings()
    if voice not in APPROVED_VOICES and not (engine == "fake" and voice == "fake-neutral"):
        raise ValueError(f"voice is not approved: {voice}")
    if not 0.5 <= settings.speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if settings.sample_rate <= 0 or settings.chunk_cap <= 0:
        raise ValueError("invalid synthesis settings")
    if settings.chunk_mode not in CHUNK_MODES:
        raise ValueError(f"unsupported chunk mode: {settings.chunk_mode}")
    if engine == "fake":
        return FakeVoice(voice, settings)
    if engine != "kokoro":
        raise ValueError("only Kokoro production engine is supported")
    torch_threads = _configured_torch_threads()
    try:
        torch = importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch is unavailable; Kokoro requires torch") from exc
    if torch_threads is not None:
        setter = getattr(torch, "set_num_threads", None)
        if not callable(setter):
            raise RuntimeError("PyTorch does not expose set_num_threads; cannot apply PDF_AUDIOBOOK_TORCH_THREADS")
        try:
            setter(torch_threads)
        except Exception as exc:
            raise RuntimeError(f"PyTorch thread configuration failed for {torch_threads}") from exc
    inference_context = getattr(torch, "inference_mode", None)
    if not callable(inference_context):
        raise RuntimeError("PyTorch does not expose inference_mode; cannot run Kokoro safely")
    try:
        module = importlib.import_module(KOKORO_PACKAGE)
        pipeline_class = getattr(module, "KPipeline")
        language = "a" if voice.startswith("a") else "b"
        return KokoroVoice(pipeline_class(lang_code=language), voice, settings, inference_context=inference_context)
    except Exception as exc:
        raise RuntimeError("Kokoro 0.9.4 is unavailable in the isolated worker environment") from exc


def _configured_torch_threads() -> int:
    raw = os.environ.get(TORCH_THREADS_ENV)
    if raw is None:
        cpu_count = os.cpu_count()
        return min(DEFAULT_TORCH_THREADS, cpu_count) if cpu_count is not None and cpu_count > 0 else DEFAULT_TORCH_THREADS
    if not raw or not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{TORCH_THREADS_ENV} must be a positive integer")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{TORCH_THREADS_ENV} must be a positive integer")
    cpu_count = os.cpu_count()
    if cpu_count is not None and value > cpu_count:
        raise ValueError(f"{TORCH_THREADS_ENV} must be no greater than os.cpu_count() ({cpu_count})")
    return value


def close_voice(voice: LoadedVoice) -> None:
    voice.close_voice()


def synthesize(text: str, voice: str, settings: SynthesisSettings | None = None, *, engine: str = "kokoro") -> bytes:
    """Convenience form of the boundary for small callers and previews."""

    loaded = load_voice(voice, settings, engine=engine)
    try:
        return loaded.synthesize(text)
    finally:
        close_voice(loaded)


def _sentence_ends(text: str) -> list[int]:
    ends: list[int] = []
    for match in re.finditer(r"(?s).*?(?:[.!?](?:[\"'’”\])]*)|\n\s*\n+)(?=\s|$)", text):
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        if end > (ends[-1] if ends else 0):
            ends.append(end)
    if not ends or ends[-1] < len(text):
        ends.append(len(text))
    return ends


def _paragraph_ends(text: str) -> set[int]:
    ends: set[int] = set()
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        ends.add(end)
    return ends


@dataclass(frozen=True)
class TextChunk:
    chapter_index: int
    global_index: int
    local_index: int
    source_start: int
    source_end: int
    text: str
    input_hash: str

    def manifest_record(self, relative_path: str, duration: float) -> dict[str, Any]:
        return {
            "chapter_index": self.chapter_index, "global_index": self.global_index,
            "local_index": self.local_index, "input_hash": self.input_hash,
            "relative_path": relative_path, "duration_seconds": duration,
        }


def chunk_input_hash(text: str, metadata: EngineMetadata | dict[str, Any]) -> str:
    """Hash all output-affecting inputs using canonical JSON."""

    facts = metadata.as_dict() if isinstance(metadata, EngineMetadata) else metadata
    payload = {"text": text, "metadata": facts}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_settings(metadata: EngineMetadata | dict[str, Any]) -> dict[str, Any]:
    settings = metadata.settings if isinstance(metadata, EngineMetadata) else metadata.get("settings", {})
    return settings if isinstance(settings, dict) else {}


def _plan_legacy_chunks(cleaned_text: str, chapters: list[dict[str, Any]], metadata: EngineMetadata | dict[str, Any], cap: int) -> list[TextChunk]:
    """Plan the original paragraph/sentence-safe chunks for existing jobs."""

    if not isinstance(cleaned_text, str) or not cleaned_text:
        raise ValueError("cleaned_text must be non-empty")
    if cap <= 0:
        raise ValueError("cap must be positive")
    result: list[TextChunk] = []
    for chapter in chapters:
        chapter_index = chapter["index"]
        start, end = chapter["start_offset"], chapter["end_offset"]
        if not isinstance(chapter_index, int) or not (0 <= start < end <= len(cleaned_text)):
            raise ValueError("invalid chapter range")
        text = cleaned_text[start:end]
        sentence_ends = _sentence_ends(text)
        paragraph_ends = _paragraph_ends(text)
        cursor = 0
        local = 0
        while cursor < len(text):
            target = min(len(text), cursor + cap)
            eligible = [pos for pos in sentence_ends if cursor < pos <= target]
            paragraph_eligible = [pos for pos in eligible if pos in paragraph_ends]
            boundary = max(paragraph_eligible or eligible, default=None)
            if boundary is None:
                boundary = next((pos for pos in sentence_ends if pos > target), len(text))
            piece = text[cursor:boundary]
            actual_start = start + cursor
            actual_end = start + boundary
            if piece:
                result.append(TextChunk(chapter_index, len(result), local, actual_start, actual_end, piece, chunk_input_hash(piece, metadata)))
                local += 1
            cursor = boundary
    return result


def plan_chunks(cleaned_text: str, chapters: list[dict[str, Any]], metadata: EngineMetadata | dict[str, Any], *, cap: int = DEFAULT_CHUNK_CAP) -> list[TextChunk]:
    """Plan one chapter chunk for new jobs, or legacy chunks for old metadata."""

    if not isinstance(cleaned_text, str) or not cleaned_text:
        raise ValueError("cleaned_text must be non-empty")
    if cap <= 0:
        raise ValueError("cap must be positive")
    settings = _metadata_settings(metadata)
    mode = settings.get("chunk_mode", "legacy")
    if mode == "chapter":
        result: list[TextChunk] = []
        for chapter in chapters:
            chapter_index = chapter["index"]
            start, end = chapter["start_offset"], chapter["end_offset"]
            if not isinstance(chapter_index, int) or not (0 <= start < end <= len(cleaned_text)):
                raise ValueError("invalid chapter range")
            text = cleaned_text[start:end]
            result.append(TextChunk(chapter_index, len(result), 0, start, end, text, chunk_input_hash(text, metadata)))
        return result
    if mode != "legacy":
        raise ValueError(f"unsupported chunk mode: {mode}")
    return _plan_legacy_chunks(cleaned_text, chapters, metadata, cap)


__all__ = ["APPROVED_VOICES", "CHUNK_MODES", "DEFAULT_CHUNK_CAP", "DEFAULT_TORCH_THREADS", "EngineMetadata", "FakeVoice", "KOKORO_MODEL", "KOKORO_PACKAGE_VERSION", "KOKORO_SAMPLE_RATE", "LoadedVoice", "SynthesisSettings", "TextChunk", "TORCH_THREADS_ENV", "chunk_input_hash", "close_voice", "load_voice", "plan_chunks", "synthesize"]

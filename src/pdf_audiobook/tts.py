"""Small, lazy TTS boundary and deterministic sentence-safe chunk planning.

The application only knows about this narrow contract.  Kokoro is imported on
the worker side, so normal application startup and tests never import a model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, ContextManager, Protocol

from .audio import pcm_from_audio
from .voice_registry import APPROVED_VOICE_IDS
from .voice_settings import VoiceSettingsError, canonical_voice_settings
from .voice_shaping import shaping_capability, shaping_fingerprint

APPROVED_VOICES = APPROVED_VOICE_IDS
KOKORO_PACKAGE = "kokoro"
KOKORO_PACKAGE_VERSION = "0.9.4"
KOKORO_MODEL = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_CHUNK_CAP = 900
DEFAULT_TORCH_THREADS = 8
TORCH_THREADS_ENV = "PDF_AUDIOBOOK_TORCH_THREADS"
CHUNK_MODES = ("chapter", "legacy")
_KOKORO_SILENCE_DURATION_SECONDS = 0.05


def _silence_pcm(sample_rate: int) -> bytes:
    """Return 50 ms of deterministic mono signed-16-bit silence."""

    frames = max(1, round(sample_rate * _KOKORO_SILENCE_DURATION_SECONDS))
    return b"\x00\x00" * frames


@dataclass(frozen=True)
class SynthesisSettings:
    speed: float = 1.0
    pitch_semitones: int = 0
    tone_preset: str = "neutral"
    sample_rate: int = KOKORO_SAMPLE_RATE
    chunk_cap: int = DEFAULT_CHUNK_CAP
    chunk_mode: str = "chapter"
    paragraph_pause_ms: int = 0
    sentence_pause_ms: int = 0

    def __post_init__(self) -> None:
        try:
            settings = canonical_voice_settings({"speed": self.speed, "pitch_semitones": self.pitch_semitones, "tone_preset": self.tone_preset})
        except VoiceSettingsError as exc:
            raise ValueError(exc.message) from exc
        object.__setattr__(self, "speed", settings["speed"])
        object.__setattr__(self, "pitch_semitones", settings["pitch_semitones"])
        object.__setattr__(self, "tone_preset", settings["tone_preset"])

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
        if not any(character.isalnum() for character in text):
            return _silence_pcm(self.metadata.sample_rate)
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
    if not math.isfinite(settings.speed) or not 0.5 <= settings.speed <= 2.0:
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


@dataclass(frozen=True)
class InteractiveTextChunk:
    """A speaker- and span-bounded chunk for Interactive Voices jobs."""

    chapter_index: int
    global_index: int
    local_index: int
    source_start: int
    source_end: int
    text: str
    input_hash: str
    audio_input_hash: str
    span_id: str
    speaker_id: str
    voice_id: str
    segment_type: str

    def manifest_record(self, relative_path: str, duration: float) -> dict[str, Any]:
        """Return the v5 fields while leaving the v4 TextChunk record unchanged."""

        return {
            "chapter_index": self.chapter_index,
            "global_index": self.global_index,
            "local_index": self.local_index,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "input_hash": self.input_hash,
            "audio_input_hash": self.audio_input_hash,
            "relative_path": relative_path,
            "duration_seconds": duration,
            "span_id": self.span_id,
            "speaker_id": self.speaker_id,
            "voice_id": self.voice_id,
            "segment_type": self.segment_type,
        }


def chunk_input_hash(text: str, metadata: EngineMetadata | dict[str, Any]) -> str:
    """Hash all output-affecting inputs using canonical JSON."""

    facts = metadata.as_dict() if isinstance(metadata, EngineMetadata) else metadata
    payload = {"text": text, "metadata": facts}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _interactive_input_hash(payload: dict[str, Any]) -> str:
    """Hash interactive inputs using the same canonical JSON rules as v4."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ValueError("interactive chunk inputs are not canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _interactive_generation_facts(source: Any, voice_id: str) -> dict[str, Any]:
    """Resolve and validate one enabled registry generation record."""

    try:
        if callable(source):
            facts = source(voice_id)
        elif hasattr(source, "get_generation_facts") and callable(source.get_generation_facts):
            facts = source.get_generation_facts(voice_id)
        elif isinstance(source, Mapping) and "id" in source:
            facts = source if source.get("id") == voice_id else None
        elif isinstance(source, Mapping):
            facts = source.get(voice_id)
        else:
            facts = None
    except Exception as exc:
        raise ValueError(f"unknown or disabled voice facts: {voice_id}") from exc
    if not isinstance(facts, Mapping):
        raise ValueError(f"unknown or disabled voice facts: {voice_id}")
    result = dict(facts)
    required = {
        "id", "engine", "package", "package_version", "model", "model_revision",
        "model_checksum", "voice_version", "voice_checksum", "sample_rate", "enabled",
    }
    if set(result) < required or result.get("id") != voice_id or result.get("enabled") is not True:
        raise ValueError(f"unknown or disabled voice facts: {voice_id}")
    if type(result["sample_rate"]) is not int or result["sample_rate"] <= 0:
        raise ValueError(f"invalid generation facts: {voice_id}")
    for field in required - {"id", "sample_rate", "enabled"}:
        if not isinstance(result[field], str) or not result[field]:
            raise ValueError(f"invalid generation facts: {voice_id}")
    return result


def _interactive_chapter_selection(
    chapters: list[dict[str, Any]], selected: Any,
) -> list[int]:
    available = [chapter["chapter_index"] for chapter in chapters]
    if selected is None:
        return available
    if type(selected) is int:
        values = [selected]
    elif isinstance(selected, range):
        values = list(selected)
    elif isinstance(selected, tuple) and len(selected) == 2 and all(type(item) is int for item in selected):
        start, end = selected
        if end < start:
            raise ValueError("invalid selected chapter range")
        values = list(range(start, end + 1))
    elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
        values = list(selected)
    else:
        raise ValueError("selected chapters must be a range or list")
    if not values or any(type(item) is not int for item in values) or len(set(values)) != len(values):
        raise ValueError("invalid selected chapters")
    if any(item not in available for item in values):
        raise ValueError("selected chapter is out of range")
    order = {chapter: index for index, chapter in enumerate(available)}
    if values != sorted(values, key=order.__getitem__):
        raise ValueError("selected chapters must be in plan order")
    return values


def plan_interactive_chunks(
    cleaned_text: str,
    voice_plan: dict[str, Any],
    generation_facts: Any,
    registry_revision: str,
    selected_chapters: Sequence[int] | range | tuple[int, int] | int | None = None,
    cap: int = DEFAULT_CHUNK_CAP,
    *,
    chapter_range: tuple[int, int] | None = None,
    shaping_identity: str | None = None,
) -> list[InteractiveTextChunk]:
    """Plan exact, speaker-bounded chunks from an approved voice plan."""

    if not isinstance(cleaned_text, str) or not cleaned_text:
        raise ValueError("cleaned_text must be non-empty")
    if type(cap) is not int or cap <= 0:
        raise ValueError("cap must be positive")
    if not isinstance(registry_revision, str) or re.fullmatch(r"[0-9a-f]{64}", registry_revision) is None:
        raise ValueError("registry revision is required")
    shaping_identity = shaping_identity or shaping_fingerprint()
    if not isinstance(shaping_identity, str) or not shaping_identity:
        raise ValueError("shaping identity is required")
    if chapter_range is not None:
        if selected_chapters is not None:
            raise ValueError("provide only one selected chapter range")
        selected_chapters = chapter_range
    if not isinstance(voice_plan, dict):
        raise ValueError("voice plan must be an object")
    if voice_plan.get("schema_version") != 1 or voice_plan.get("artifact") != "voice-plan":
        raise ValueError("voice plan is not schema-1")
    approval = voice_plan.get("approval")
    revision = voice_plan.get("revision")
    if not isinstance(approval, dict) or approval.get("state") != "approved":
        raise ValueError("voice plan must be approved")
    if type(revision) is not int or revision <= 0 or approval.get("approved_revision") != revision:
        raise ValueError("voice plan approval revision is invalid")
    canonical_hash = voice_plan.get("canonical_artifact_sha256")
    try:
        from .voice_plan import verify_canonical_artifact_hash
        verify_canonical_artifact_hash(voice_plan)
    except Exception as exc:
        raise ValueError("voice plan canonical hash is invalid") from exc
    if not isinstance(canonical_hash, str) or len(canonical_hash) != 64:
        raise ValueError("voice plan canonical hash is invalid")

    unresolved_policy = voice_plan.get("unresolved_policy")
    accepted_narrator_fallback = False
    if isinstance(unresolved_policy, dict) and set(unresolved_policy) == {"mode", "accepted_by_user", "accepted_at"} and unresolved_policy.get("mode") == "narrator" and unresolved_policy.get("accepted_by_user") is True:
        accepted_at = unresolved_policy.get("accepted_at")
        if isinstance(accepted_at, str):
            try:
                accepted_narrator_fallback = datetime.fromisoformat(accepted_at.replace("Z", "+00:00")).tzinfo is not None
            except ValueError:
                accepted_narrator_fallback = False

    cast_raw = voice_plan.get("cast")
    if not isinstance(cast_raw, list) or not cast_raw:
        raise ValueError("voice plan cast is missing")
    cast: dict[str, dict[str, Any]] = {}
    facts_by_voice: dict[str, dict[str, Any]] = {}
    for entry in cast_raw:
        if not isinstance(entry, dict):
            raise ValueError("voice plan cast is invalid")
        cast_id = entry.get("cast_id")
        voice_id = entry.get("voice_id")
        settings = entry.get("voice_settings")
        if not isinstance(cast_id, str) or not cast_id or cast_id in cast:
            raise ValueError("voice plan cast is invalid")
        if not isinstance(voice_id, str) or not voice_id:
            raise ValueError(f"voice is invalid: {voice_id}")
        if not isinstance(settings, dict) or "speed" not in settings:
            raise ValueError(f"missing cast settings: {cast_id}")
        try:
            canonical_settings = canonical_voice_settings(settings)
        except VoiceSettingsError as exc:
            raise ValueError(f"invalid cast settings: {cast_id}") from exc
        if canonical_settings["pitch_semitones"] and not shaping_capability().rubberband:
            raise ValueError("pitch shaping is unavailable: FFmpeg rubberband support is required")
        if voice_id not in facts_by_voice:
            facts_by_voice[voice_id] = _interactive_generation_facts(generation_facts, voice_id)
        cast[cast_id] = {**entry, "voice_settings": canonical_settings}

    artifact_chapters = voice_plan.get("chapters")
    if not isinstance(artifact_chapters, list) or not artifact_chapters:
        raise ValueError("voice plan chapters are missing")
    chapters: list[dict[str, Any]] = []
    expected_chapter_start = 0
    for expected_index, chapter in enumerate(artifact_chapters, start=1):
        if not isinstance(chapter, dict):
            raise ValueError("voice plan chapter is invalid")
        chapter_index = chapter.get("chapter_index")
        start, end = chapter.get("source_start"), chapter.get("source_end")
        if type(chapter_index) is not int or chapter_index != expected_index or type(start) is not int or type(end) is not int or start != expected_chapter_start or start < 0 or end <= start or end > len(cleaned_text):
            raise ValueError("invalid voice plan chapter range")
        spans = chapter.get("spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("voice plan spans are missing")
        chapters.append(chapter)
        expected_chapter_start = end
    if expected_chapter_start != len(cleaned_text):
        raise ValueError("voice plan chapters do not cover cleaned text")

    selected = _interactive_chapter_selection(chapters, selected_chapters)
    selected_set = set(selected)
    result: list[InteractiveTextChunk] = []
    seen_span_ids: set[str] = set()
    for chapter in chapters:
        chapter_index = chapter["chapter_index"]
        cursor = chapter["source_start"]
        local = 0
        for span in chapter["spans"]:
            if not isinstance(span, dict):
                raise ValueError("voice plan span is invalid")
            span_id = span.get("span_id")
            start, end = span.get("source_start"), span.get("source_end")
            segment_type = span.get("type")
            speaker_id = span.get("speaker_id")
            if not isinstance(span_id, str) or not span_id or span_id in seen_span_ids:
                raise ValueError("voice plan span ID is invalid")
            seen_span_ids.add(span_id)
            if type(start) is not int or type(end) is not int or start != cursor or end <= start or end > chapter["source_end"]:
                raise ValueError("voice plan spans have gaps or overlap")
            if segment_type == "unknown":
                if not accepted_narrator_fallback or speaker_id != "narrator":
                    raise ValueError("unknown spans require accepted narrator fallback")
                segment_type = "narration"
            if segment_type not in {"narration", "dialogue", "thought"}:
                raise ValueError("interactive spans must have an approved type")
            if not isinstance(speaker_id, str) or not speaker_id or speaker_id not in cast:
                raise ValueError("interactive span speaker is missing from cast")
            if segment_type == "thought" and not isinstance(span.get("override"), dict):
                raise ValueError("thought spans require a manual override")
            entry = cast[speaker_id]
            voice_id = entry["voice_id"]
            canonical_settings = entry["voice_settings"]
            facts = facts_by_voice[voice_id]
            if chapter_index in selected_set:
                span_text = cleaned_text[start:end]
                sentence_ends = _sentence_ends(span_text)
                paragraph_ends = _paragraph_ends(span_text)
                span_cursor = 0
                while span_cursor < len(span_text):
                    target = min(len(span_text), span_cursor + cap)
                    eligible = [pos for pos in sentence_ends if span_cursor < pos <= target]
                    paragraph_eligible = [pos for pos in eligible if pos in paragraph_ends]
                    boundary = max(paragraph_eligible or eligible, default=None)
                    if boundary is None:
                        boundary = next((pos for pos in sentence_ends if pos > target), len(span_text))
                    piece = span_text[span_cursor:boundary]
                    actual_start, actual_end = start + span_cursor, start + boundary
                    if piece:
                        audio_payload = {
                            "mode": "interactive_voices",
                            "chapter_index": chapter_index,
                            "source_start": actual_start,
                            "source_end": actual_end,
                            "text": piece,
                            "span_id": span_id,
                            "segment_type": segment_type,
                            "speaker_id": speaker_id,
                            "voice_id": voice_id,
                            "voice_settings": canonical_settings,
                            "generation_facts": facts,
                            "chunk_cap": cap,
                            "boundary_policy": "sentence_paragraph_safe",
                            "shaping_fingerprint": shaping_identity,
                        }
                        payload = {
                            **audio_payload,
                            "voice_plan_canonical_hash": canonical_hash,
                            "voice_plan_revision": revision,
                            "registry_revision": registry_revision,
                            "shaping_fingerprint": shaping_identity,
                        }
                        result.append(InteractiveTextChunk(
                            chapter_index, len(result), local, actual_start, actual_end, piece,
                            _interactive_input_hash(payload), _interactive_input_hash(audio_payload),
                            span_id, speaker_id, voice_id, segment_type,
                        ))
                        local += 1
                    span_cursor = boundary
            cursor = end
        if cursor != chapter["source_end"]:
            raise ValueError("voice plan spans have gaps or overlap")
    return result


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


__all__ = ["APPROVED_VOICES", "CHUNK_MODES", "DEFAULT_CHUNK_CAP", "DEFAULT_TORCH_THREADS", "EngineMetadata", "FakeVoice", "InteractiveTextChunk", "KOKORO_MODEL", "KOKORO_PACKAGE_VERSION", "KOKORO_SAMPLE_RATE", "LoadedVoice", "SynthesisSettings", "TextChunk", "TORCH_THREADS_ENV", "chunk_input_hash", "close_voice", "load_voice", "plan_chunks", "plan_interactive_chunks", "synthesize"]

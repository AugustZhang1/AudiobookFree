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

from .audio import pcm_from_audio, validate_wav
from .engine_catalog import (
    CHATTERBOX_NANO_MODEL,
    CHATTERBOX_NANO_MODEL_ID,
    CHATTERBOX_PACKAGE_VERSION,
    CHATTERBOX_SOURCE_COMMIT,
    require_enabled,
)
from .voice_registry import APPROVED_VOICE_IDS
from .voice_settings import VoiceSettingsError, canonical_voice_settings
from .voice_shaping import shaping_capability, shaping_fingerprint

APPROVED_VOICES = APPROVED_VOICE_IDS
KOKORO_PACKAGE = "kokoro"
KOKORO_PACKAGE_VERSION = "0.9.4"
KOKORO_MODEL = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
CHATTERBOX_PACKAGE = "chatterbox-tts"
CHATTERBOX_SAMPLE_RATE = 24000
CHATTERBOX_REFERENCE_VOICE = "reference-wav"
CHATTERBOX_BUILTIN_VOICE = "builtin"
CHATTERBOX_CHUNK_CAP = 300
DEFAULT_CHUNK_CAP = 900
DEFAULT_TORCH_THREADS = 8
TORCH_THREADS_ENV = "PDF_AUDIOBOOK_TORCH_THREADS"
CHUNK_MODES = ("chapter", "legacy")
_KOKORO_SILENCE_DURATION_SECONDS = 0.05
KOKORO_SYNTHESIS_IMPLEMENTATION = "kokoro-synthesis-v2"
_KOKORO_PARAGRAPH_PAUSE_SECONDS = 0.4  # matches m4b.PARAGRAPH_PAUSE_MS
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_WRAPPED_LINE = re.compile(r"[ \t]*\n[ \t]*")
_SENTENCE_FINAL = re.compile(r"""[.!?…]["'’”»›)\]}]*\Z""")
# Mirrors chapters.py:18 deliberately: importing it would couple narration
# hashes to the chapter planner's heading rule.
_HEADING_LINE = re.compile(r"^(?:chapter|part|section)\b(?:\s+.*)?$", re.IGNORECASE)


def flowed_paragraphs(text: str) -> list[str]:
    """Split on blank lines; flow a surviving single newline into a space.

    A single newline in cleaned PDF text is a hard wrap, not a spoken boundary,
    so it becomes a space -- except that a leading `Chapter`/`Part`/`Section`
    line stays a paragraph of its own so it keeps its own prosody. Other
    line-oriented material -- poetry, lists, dialogue, and headings that do not
    match that pattern -- is flowed into the following sentence.
    """

    flowed: list[str] = []
    for block in _PARAGRAPH_SPLIT.split(text):
        lines = block.split("\n")
        while len(lines) > 1 and _HEADING_LINE.match(lines[0].strip()):
            flowed.append(lines[0].strip())
            lines = lines[1:]
        block = "\n".join(lines)
        joined = _WRAPPED_LINE.sub(" ", block).strip()
        if joined:
            flowed.append(joined)
    return flowed


def _silence_pcm(sample_rate: int, seconds: float = _KOKORO_SILENCE_DURATION_SECONDS) -> bytes:
    """Return deterministic mono signed-16-bit silence for the requested duration."""

    frames = max(1, round(sample_rate * seconds))
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


@dataclass(frozen=True)
class ChunkPlanInputs:
    settings: dict[str, Any]
    model: str
    cap: int


def derive_chunk_plan_inputs(tts: dict[str, Any], *, interactive: bool) -> ChunkPlanInputs:
    """Normalize manifest values shared by worker and assembler planning."""

    model = str(tts.get("model", tts.get("model_id", "")))
    if interactive:
        raw_settings = tts.get("settings")
        settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
        cap = int(tts.get("chunk_cap", settings.get("chunk_cap", 900)))
    else:
        settings = dict(tts.get("settings", {"speed": tts["speed"], "sample_rate": tts["sample_rate"], "chunk_cap": tts.get("chunk_cap", 900)}))
        cap = int(settings.get("chunk_cap", 900))
    return ChunkPlanInputs(settings=settings, model=model, cap=cap)


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
        paragraphs = [paragraph for paragraph in flowed_paragraphs(text) if any(character.isalnum() for character in paragraph)]
        pieces: list[bytes] = []
        with self._inference_context():
            for index, paragraph in enumerate(paragraphs):
                paragraph_pcm: list[bytes] = []
                for result in self.pipeline(paragraph, voice=self.metadata.voice, speed=self.metadata.settings["speed"], split_pattern=None):
                    audio = getattr(result, "audio", result[2] if isinstance(result, tuple) and len(result) >= 3 else result)
                    pcm = pcm_from_audio(audio)
                    if not pcm:
                        raise RuntimeError("Kokoro returned empty audio")
                    paragraph_pcm.append(pcm)
                if not paragraph_pcm:
                    raise RuntimeError("Kokoro returned no audio")
                if index >= 1 and _SENTENCE_FINAL.search(paragraphs[index - 1]):
                    pieces.append(_silence_pcm(self.metadata.sample_rate, _KOKORO_PARAGRAPH_PAUSE_SECONDS))
                pieces.extend(paragraph_pcm)
        return b"".join(pieces)

    def close_voice(self) -> None:
        return None


class ChatterboxVoice:
    """Lazy-boundary wrapper around Chatterbox Nano's reference prompt API."""

    def __init__(self, model: Any, settings: SynthesisSettings, voice: str, reference_wav: Path | None = None, reference_sha256: str = "unrecorded"):
        self.model = model
        self._voice = voice
        self._reference_wav = reference_wav
        self._reference_sha256 = reference_sha256
        self._closed = False
        sample_rate = getattr(model, "sr", getattr(model, "sampling_rate", CHATTERBOX_SAMPLE_RATE))
        if sample_rate != CHATTERBOX_SAMPLE_RATE:
            raise RuntimeError("Chatterbox Nano returned an unsupported sample rate")
        self.metadata = EngineMetadata(
            "chatterbox", CHATTERBOX_PACKAGE_VERSION, CHATTERBOX_NANO_MODEL,
            CHATTERBOX_SOURCE_COMMIT, "unrecorded", voice,
            CHATTERBOX_SOURCE_COMMIT if voice == CHATTERBOX_REFERENCE_VOICE else "bundled", reference_sha256, sample_rate,
            {**settings.as_dict(), "chunk_cap": CHATTERBOX_CHUNK_CAP},
        )

    def synthesize(self, text: str) -> bytes:
        if self._voice == CHATTERBOX_REFERENCE_VOICE:
            try:
                reference_wav, reference_sha256 = _reference_wav_identity(self._reference_wav)
            except ValueError:
                raise RuntimeError("Chatterbox reference WAV is invalid or unavailable") from None
            if reference_sha256 != self._reference_sha256:
                raise RuntimeError("Chatterbox reference WAV has changed")
        if not any(character.isalnum() for character in text):
            return _silence_pcm(self.metadata.sample_rate)
        try:
            result = self.model.generate(text, **({"audio_prompt_path": str(reference_wav)} if self._voice == CHATTERBOX_REFERENCE_VOICE else {}))
        except Exception:
            raise RuntimeError("Chatterbox generation failed") from None
        try:
            pcm = pcm_from_audio(result)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Chatterbox returned empty audio") from exc
        if not pcm:
            raise RuntimeError("Chatterbox returned empty audio")
        return pcm

    def close_voice(self) -> None:
        if self._closed:
            return None
        self._closed = True
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
            return None
        release = getattr(self.model, "release", None)
        if callable(release):
            release()
        return None


def _reference_wav_identity(reference_wav: str | os.PathLike[str] | Path) -> tuple[Path, str]:
    """Validate and hash a caller-provided WAV without putting its path in errors."""

    try:
        source = Path(reference_wav)
        wav = validate_wav(source)
        if wav.duration_seconds <= 5.0 or wav.duration_seconds > 60.0:
            raise ValueError("reference WAV duration is invalid")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return source, digest.hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("reference WAV is invalid or unsafe") from exc


def _load_chatterbox_nano(settings: SynthesisSettings, voice: str, reference_wav: Path | None = None, reference_sha256: str = "unrecorded") -> ChatterboxVoice:
    model: Any | None = None
    try:
        module = importlib.import_module("chatterbox.tts_turbo")
        model_class = getattr(module, "ChatterboxTurboTTS")
        model = model_class.from_pretrained(device="cpu", nano=True)
        return ChatterboxVoice(model, settings, voice, reference_wav, reference_sha256)
    except Exception as exc:
        if model is not None:
            _release_model(model)
        raise RuntimeError("Chatterbox Nano is unavailable in the isolated worker environment") from exc


def _release_model(model: Any) -> None:
    """Release a model at most once, preferring its explicit close boundary."""

    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
        return
    release = getattr(model, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass


def load_voice(
    voice: str,
    settings: SynthesisSettings | None = None,
    *,
    engine: str = "kokoro",
    reference_wav: str | os.PathLike[str] | Path | None = None,
) -> LoadedVoice:
    """Load a voice lazily.  No model import occurs until this function runs."""

    if engine == "chatterbox":
        settings = settings or SynthesisSettings(chunk_cap=CHATTERBOX_CHUNK_CAP)
    else:
        settings = settings or SynthesisSettings()
    if engine == "chatterbox":
        require_enabled("chatterbox", CHATTERBOX_NANO_MODEL_ID)
        if voice not in {CHATTERBOX_BUILTIN_VOICE, CHATTERBOX_REFERENCE_VOICE}:
            raise ValueError("Chatterbox Nano supports builtin or reference-wav voices")
        if voice == CHATTERBOX_REFERENCE_VOICE and reference_wav is None:
            raise ValueError("reference WAV is required for the custom Chatterbox voice")
        if voice == CHATTERBOX_BUILTIN_VOICE and reference_wav is not None:
            raise ValueError("builtin Chatterbox voice does not use a reference WAV")
        if settings.speed != 1.0:
            raise ValueError("Chatterbox Nano supports speed 1.0 only")
        if settings.pitch_semitones != 0 or settings.tone_preset != "neutral":
            raise ValueError("Chatterbox Nano supports neutral pitch and tone only")
        if settings.sample_rate != CHATTERBOX_SAMPLE_RATE:
            raise ValueError("Chatterbox Nano requires a 24000 Hz sample rate")
        if settings.chunk_mode not in CHUNK_MODES:
            raise ValueError(f"unsupported chunk mode: {settings.chunk_mode}")
        if voice == CHATTERBOX_REFERENCE_VOICE:
            source, reference_sha256 = _reference_wav_identity(reference_wav)
            return _load_chatterbox_nano(settings, voice, source, reference_sha256)
        return _load_chatterbox_nano(settings, voice)
    if voice not in APPROVED_VOICES and not (engine == "fake" and voice == "fake-neutral"):
        raise ValueError(f"voice is not approved: {voice}")
    if not math.isfinite(settings.speed) or not 0.5 <= settings.speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if settings.sample_rate <= 0 or settings.chunk_cap <= 0:
        raise ValueError("invalid synthesis settings")
    if settings.chunk_mode not in CHUNK_MODES:
        raise ValueError(f"unsupported chunk mode: {settings.chunk_mode}")
    if engine == "fake":
        if reference_wav is not None:
            raise ValueError("Fake does not use reference WAV prompts")
        return FakeVoice(voice, settings)
    if engine != "kokoro":
        raise ValueError("only Kokoro and Chatterbox production engines are supported")
    if reference_wav is not None:
        raise ValueError("Kokoro does not use reference WAV prompts")
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
    if facts.get("engine") == "kokoro":
        payload["synthesis_implementation"] = KOKORO_SYNTHESIS_IMPLEMENTATION
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
                        if facts.get("engine") == "kokoro":
                            audio_payload["synthesis_implementation"] = KOKORO_SYNTHESIS_IMPLEMENTATION
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


__all__ = ["APPROVED_VOICES", "CHUNK_MODES", "ChunkPlanInputs", "DEFAULT_CHUNK_CAP", "DEFAULT_TORCH_THREADS", "EngineMetadata", "FakeVoice", "InteractiveTextChunk", "KOKORO_MODEL", "KOKORO_PACKAGE_VERSION", "KOKORO_SAMPLE_RATE", "LoadedVoice", "SynthesisSettings", "TextChunk", "TORCH_THREADS_ENV", "chunk_input_hash", "close_voice", "derive_chunk_plan_inputs", "flowed_paragraphs", "load_voice", "plan_chunks", "plan_interactive_chunks", "synthesize"]

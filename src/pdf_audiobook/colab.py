"""Headless Google Colab runner for the optional Kokoro / Chatterbox GPU workflow.

This module is deliberately separate from the desktop launcher.  Colab calls
the same workspace, chapter planning, resumable worker, and M4B finalizer as
the local app, while its engine factory opts into CUDA explicitly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from array import array
from typing import Any, Callable, Iterator
import wave

from .audio import write_pcm_wav
from .chapters import create_chapter_plan
from .chatterbox_reference import validate_reference_file
from .pdf import analyze_pdf
from .preview_worker import PREVIEW_TEXT, generate_preview
from .security import pid_is_alive
from .tts import (
    CHATTERBOX_CHUNK_CAP,
    CHATTERBOX_NANO_MODEL,
    CHATTERBOX_PACKAGE_VERSION,
    CHATTERBOX_REFERENCE_VOICE,
    CHATTERBOX_SAMPLE_RATE,
    CHATTERBOX_SOURCE_COMMIT,
    ChatterboxVoice,
    EngineMetadata,
    KokoroVoice,
    SynthesisSettings,
    plan_chunks,
)
from .voice_registry import APPROVED_VOICE_IDS
from .worker import ConversionWorker, WorkerResult
from .workspace import ManifestError, Workspace, WorkspaceError


class ColabError(RuntimeError):
    """Base error for a headless Colab conversion."""


class ColabConflictError(ColabError):
    """The active conversion belongs to a different request."""


def _import(name: str) -> Any:
    return importlib.import_module(name)


def _cuda_module() -> Any:
    try:
        torch = _import("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ColabError("CUDA is unavailable: PyTorch is not installed in this Colab runtime") from exc
    if not bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
        raise ColabError("CUDA is unavailable. In Colab, choose Runtime > Change runtime type > T4 GPU, then rerun.")
    return torch


def make_cuda_kokoro_factory(*, torch_loader: Callable[[], Any] | None = None, kokoro_loader: Callable[[], Any] | None = None) -> Callable[..., Any]:
    """Return a worker-compatible Kokoro factory that always requests CUDA."""

    load_torch = torch_loader or _cuda_module
    load_kokoro = kokoro_loader or (lambda: _import("kokoro"))
    pipelines: dict[str, Any] = {}

    def factory(voice: str, settings: SynthesisSettings | None = None, *, engine: str = "kokoro") -> KokoroVoice:
        if engine != "kokoro":
            raise ColabError("the Colab runner supports Kokoro only")
        settings = settings or SynthesisSettings()
        torch = load_torch()
        if not bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
            raise ColabError("CUDA became unavailable before Kokoro loaded")
        inference_mode = getattr(torch, "inference_mode", None)
        if not callable(inference_mode):
            raise ColabError("this PyTorch build does not provide inference_mode")
        try:
            kokoro = load_kokoro()
            pipeline_class = getattr(kokoro, "KPipeline")
            language = "a" if voice.startswith("a") else "b"
            if language not in pipelines:
                pipelines[language] = pipeline_class(lang_code=language, device="cuda")
            pipeline = pipelines[language]
            return KokoroVoice(pipeline, voice, settings, inference_context=inference_mode)
        except ColabError:
            raise
        except Exception as exc:
            raise ColabError("Kokoro 0.9.4 could not be loaded in the Colab GPU runtime") from exc

    return factory


# Custom-voice models, best first. "index" is IndexTTS-2: it copies the voice
# from the clip and reads each line's emotion from its words (Expression sets
# how strongly), in its own environment (the index-tts repo, checkpoints in
# INDEXTTS_DIR). "qwen" is Qwen3-TTS Base: it is given a transcript of the
# clip, so it copies accent and delivery closely, and runs in the
# voice-design environment (qwen-tts) instead of Chatterbox's.
# "original" is the largest Chatterbox and the only model with an exaggeration
# (expression) control; the Turbo family uses inline tags such as [laugh] instead.
CLONE_MODELS = ("index", "qwen", "original", "turbo", "nano")
DEFAULT_CLONE_MODEL = "index"
DEFAULT_EXAGGERATION = 0.5
QWEN_CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"


def transcript_path(clip: str | os.PathLike[str]) -> Path:
    """Where a clip's transcript lives: ``name.transcript.txt`` beside ``name.wav``; edit it to fix a mistake."""

    return Path(clip).with_suffix(".transcript.txt")


def _wav_floats(clip: Path) -> tuple[list[float], int]:
    with wave.open(str(clip), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ColabError("custom voice clips must be 16-bit WAV")
    samples = array("h", frames)
    if sys.byteorder == "big":
        samples.byteswap()
    mono = [sum(samples[i:i + channels]) / channels / 32768.0 for i in range(0, len(samples), channels)]
    return mono, rate


def _whisper_transcribe(clip: Path) -> str:
    torch = _import("torch")
    numpy = _import("numpy")
    samples, rate = _wav_floats(clip)
    asr = _import("transformers").pipeline("automatic-speech-recognition", model=WHISPER_MODEL_ID, device=0)
    try:
        result = asr({"raw": numpy.asarray(samples, dtype=numpy.float32), "sampling_rate": rate}, chunk_length_s=30, generate_kwargs={"language": "english"})
    finally:
        del asr
        torch.cuda.empty_cache()
    return str(result["text"])


def reference_transcript(clip: str | os.PathLike[str], *, transcriber: Callable[[Path], str] | None = None) -> str:
    """Return what is said in ``clip``, transcribing it with Whisper the first time and saving it beside the clip."""

    path = transcript_path(clip)
    if path.is_file():
        text = " ".join(path.read_text(encoding="utf-8").split())
        if text:
            return text
    print(f"Transcribing {Path(clip).stem} for Qwen3 cloning (first time only)…", flush=True)
    text = " ".join(str((transcriber or _whisper_transcribe)(Path(clip))).split())
    if not text:
        raise ColabError(f"no speech was found in {Path(clip).stem}; use a clip of someone talking")
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Transcript saved to {path.name}: \"{text}\" (fix it if a word is wrong; a mismatch makes the clip leak into lines)", flush=True)
    return text


class _QwenCloneModel:
    """Give Qwen3-TTS Base voice cloning ChatterboxVoice's call shape (24 kHz audio from a reference clip)."""

    sr = CHATTERBOX_SAMPLE_RATE

    def __init__(self, model: Any, transcriber: Callable[[Path], str] | None = None) -> None:
        self._model = model
        self._transcriber = transcriber
        self._prompts: dict[str, tuple[str, str]] = {}
        self._scratch = tempfile.TemporaryDirectory(prefix="qwen-ref-")

    def _prompt(self, clip: str) -> tuple[str, str]:
        """A copy of the clip with half a second of silence on each end, and its transcript ending in punctuation.

        Qwen3 continues straight on from the clip, so a clip that stops mid-word
        or a transcript without a full stop leaks bits of the clip into the
        start of every line; a clean, silent boundary prevents that.
        """
        if clip not in self._prompts:
            text = reference_transcript(clip, transcriber=self._transcriber)
            if text[-1] not in ".!?…\"'":
                text += "."
            padded = Path(self._scratch.name) / f"{len(self._prompts)}.wav"
            with wave.open(clip, "rb") as source:
                params, frames = source.getparams(), source.readframes(source.getnframes())
            silence = b"\x00" * (params.sampwidth * params.nchannels * (params.framerate // 2))
            with wave.open(str(padded), "wb") as target:
                target.setparams(params)
                target.writeframes(silence + frames + silence)
            self._prompts[clip] = (str(padded), text)
        return self._prompts[clip]

    def generate(self, text: str, *, audio_prompt_path: str) -> Any:
        ref_audio, ref_text = self._prompt(audio_prompt_path)
        wavs, rate = self._model.generate_voice_clone(text=text, language="English", ref_audio=ref_audio, ref_text=ref_text)
        return _resample(wavs[0], int(rate), self.sr)


def _resample(audio: Any, rate: int, target: int) -> Any:
    if rate == target:
        return audio
    numpy = _import("numpy")
    audio = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    positions = numpy.arange(round(len(audio) * target / rate)) * (rate / target)
    return numpy.interp(positions, numpy.arange(len(audio)), audio).astype(numpy.float32)


def _load_qwen_clone_model() -> Any:
    torch = _import("torch")
    # bfloat16 needs an Ampere-or-newer GPU; a T4 gets float32 (the 1.7B model still fits).
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    model = _import("qwen_tts").Qwen3TTSModel.from_pretrained(QWEN_CLONE_MODEL_ID, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa")
    return _QwenCloneModel(model)


class _IndexCloneModel:
    """Give IndexTTS-2 ChatterboxVoice's call shape: the voice from the clip, the emotion from the line's own words."""

    sr = CHATTERBOX_SAMPLE_RATE

    def __init__(self, model: Any, exaggeration: float) -> None:
        self._model = model
        self._emotion = min(1.0, exaggeration)  # IndexTTS-2's emotion strength runs 0-1

    def generate(self, text: str, *, audio_prompt_path: str) -> Any:
        rate, pcm = self._model.infer(spk_audio_prompt=audio_prompt_path, text=text, output_path=None, use_emo_text=True, emo_alpha=self._emotion)
        numpy = _import("numpy")
        return _resample(numpy.asarray(pcm, dtype=numpy.float32).reshape(-1) / 32768.0, int(rate), self.sr)


def _load_index_clone_model(exaggeration: float) -> Any:
    checkpoints = os.environ.get("INDEXTTS_DIR", "/content/index-tts/checkpoints")
    model = _import("indextts.infer_v2").IndexTTS2(cfg_path=f"{checkpoints}/config.yaml", model_dir=checkpoints, use_fp16=True, use_cuda_kernel=False)
    return _IndexCloneModel(model, exaggeration)


class _ExpressiveModel:
    """Give the original Chatterbox model ChatterboxVoice's call shape, with a fixed exaggeration."""

    def __init__(self, model: Any, exaggeration: float) -> None:
        self._model = model
        self._exaggeration = exaggeration
        self.sr = getattr(model, "sr", CHATTERBOX_SAMPLE_RATE)

    def generate(self, text: str, *, audio_prompt_path: str) -> Any:
        return self._model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=self._exaggeration)


def _load_clone_model(model: str, exaggeration: float) -> Any:
    if model == "index":
        return _load_index_clone_model(exaggeration)
    if model == "qwen":
        return _load_qwen_clone_model()
    if model == "original":
        return _ExpressiveModel(_import("chatterbox.tts").ChatterboxTTS.from_pretrained(device="cuda"), exaggeration)
    return _import("chatterbox.tts_turbo").ChatterboxTurboTTS.from_pretrained(device="cuda", nano=model == "nano")


def _clone_options(model: Any, exaggeration: Any) -> tuple[str, float]:
    """Validate the custom-voice model and expression before anything is touched."""

    if model not in CLONE_MODELS:
        raise ColabError(f"custom voice model must be one of: {', '.join(CLONE_MODELS)}")
    if isinstance(exaggeration, bool):
        raise ColabError("expression must be between 0.25 and 2.0")
    try:
        value = float(exaggeration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColabError("expression must be between 0.25 and 2.0") from exc
    if not math.isfinite(value) or not 0.25 <= value <= 2.0:
        raise ColabError("expression must be between 0.25 and 2.0")
    return model, value


def make_cuda_chatterbox_factory(
    *,
    model: str = DEFAULT_CLONE_MODEL,
    exaggeration: float = DEFAULT_EXAGGERATION,
    torch_loader: Callable[[], Any] | None = None,
    model_loader: Callable[[], Any] | None = None,
) -> Callable[..., Any]:
    """Return a worker-compatible Chatterbox factory that clones a reference WAV on CUDA.

    The model is loaded once and shared, so previewing several library voices
    does not reload it.
    """

    model, exaggeration = _clone_options(model, exaggeration)
    load_torch = torch_loader or _cuda_module
    load_model = model_loader or (lambda: _load_clone_model(model, exaggeration))
    cache: dict[str, Any] = {}

    def factory(voice: str, settings: SynthesisSettings | None = None, *, engine: str = "chatterbox", reference_wav: Any = None) -> ChatterboxVoice:
        if engine != "chatterbox" or voice != CHATTERBOX_REFERENCE_VOICE or reference_wav is None:
            raise ColabError("the Colab Chatterbox runner needs a custom voice clip")
        load_torch()
        clip = Path(reference_wav)
        digest = _clip_digest(clip)
        if "model" not in cache:
            try:
                cache["model"] = load_model()
            except Exception as exc:
                if model == "index":
                    raise ColabError(f"IndexTTS-2 could not be loaded in the Colab GPU runtime: {exc}") from exc
                if model == "qwen":
                    raise ColabError(f"Qwen3-TTS could not be loaded in the Colab GPU runtime: {exc}") from exc
                raise ColabError("Chatterbox could not be loaded in the Colab GPU runtime; run the voice-cloning setup cell") from exc
        settings = settings or SynthesisSettings(sample_rate=CHATTERBOX_SAMPLE_RATE, chunk_cap=CHATTERBOX_CHUNK_CAP)
        return ChatterboxVoice(cache["model"], settings, CHATTERBOX_REFERENCE_VOICE, clip, digest)

    return factory


def _clip_digest(clip: Any) -> str:
    """Validate a custom voice clip and return its SHA-256."""

    try:
        _wav, digest = validate_reference_file(Path(clip).expanduser())
    except (TypeError, ValueError) as exc:
        name = Path(clip).name if isinstance(clip, (str, os.PathLike)) else "clip"
        raise ColabError(f"voice clip {name} must be a mono 16-bit WAV between 5 and 60 seconds long") from exc
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_mode(mode: str, count: int | None) -> tuple[str, int | None]:
    if mode not in {"original", "whole", "custom"}:
        raise ColabError("chapter mode must be original, whole, or custom")
    if mode == "custom":
        if type(count) is not int or not 2 <= count <= 50:
            raise ColabError("custom chapter count must be an integer from 2 to 50")
        return mode, count
    if count is not None:
        raise ColabError("chapter count is only valid with custom chapter mode")
    return mode, None


def _normalized_speed(speed: Any) -> float:
    """Coerce a caller-supplied speed exactly once, so validation binds the value that is used."""

    if isinstance(speed, bool):
        raise ColabError("speed must be between 0.5 and 2.0")
    try:
        return float(speed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColabError("speed must be between 0.5 and 2.0") from exc


def _validate_voice_speed(voice: str, speed: float) -> None:
    """Reject an unapproved voice or an out-of-range speed before any mutation."""

    if voice not in APPROVED_VOICE_IDS:
        raise ColabError(f"voice is not approved: {voice}")
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ColabError("speed must be between 0.5 and 2.0")


MAX_PREVIEW_CHARS = 10000


def _preview_text(text: Any) -> str:
    if text is None:
        return PREVIEW_TEXT
    if not isinstance(text, str):
        raise ColabError(f"preview text must be 1 to {MAX_PREVIEW_CHARS} characters")
    cleaned = text.strip()
    if not (1 <= len(cleaned) <= MAX_PREVIEW_CHARS):
        raise ColabError(f"preview text must be 1 to {MAX_PREVIEW_CHARS} characters")
    return cleaned


def _expected_tts(cleaned_text: str, plan: dict[str, Any], voice: str, speed: float) -> tuple[dict[str, Any], int]:
    _validate_voice_speed(voice, speed)
    # Whole-paragraph chunks sound identical to the desktop's chapter mode but
    # keep memory and checkpoints bounded for very long books.
    settings = SynthesisSettings(speed=speed, chunk_mode="paragraph", chunk_cap=5000)
    metadata = EngineMetadata(
        "kokoro", "0.9.4", "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded",
        voice, "captured-at-download", "unrecorded", settings.sample_rate, settings.as_dict(),
    )
    chunks = plan_chunks(cleaned_text, plan["chapters"], metadata, cap=settings.chunk_cap)
    return {**metadata.as_dict(), "speed": speed, "chunk_cap": settings.chunk_cap}, len(chunks)


def _expected_chatterbox_tts(cleaned_text: str, plan: dict[str, Any], descriptor: dict[str, Any], model: str = DEFAULT_CLONE_MODEL, exaggeration: float = DEFAULT_EXAGGERATION) -> tuple[dict[str, Any], int]:
    """Mirror the desktop app's Chatterbox reference-voice generation identity.

    The model fields keep the desktop's validated Nano identity; the model that
    actually narrates (and its expression) is recorded in the settings, so a
    resume with a different model is refused like any other settings change.
    """

    settings = SynthesisSettings(speed=1.0, sample_rate=CHATTERBOX_SAMPLE_RATE, chunk_cap=CHATTERBOX_CHUNK_CAP, chunk_mode="legacy")
    settings_dict = {**settings.as_dict(), "reference_descriptor_sha256": descriptor["descriptor_sha256"], "clone_model": model}
    if model in ("original", "index"):
        settings_dict["exaggeration"] = exaggeration
    metadata = EngineMetadata(
        "chatterbox", CHATTERBOX_PACKAGE_VERSION, CHATTERBOX_NANO_MODEL, CHATTERBOX_SOURCE_COMMIT, "unrecorded",
        CHATTERBOX_REFERENCE_VOICE, CHATTERBOX_SOURCE_COMMIT, descriptor["voice_checksum"], CHATTERBOX_SAMPLE_RATE, settings_dict,
    )
    chunks = plan_chunks(cleaned_text, plan["chapters"], metadata, cap=CHATTERBOX_CHUNK_CAP)
    return {**metadata.as_dict(), "speed": 1.0, "chunk_cap": CHATTERBOX_CHUNK_CAP}, len(chunks)


def _ensure_reference(workspace: Workspace, conversion_id: str, clip: Path, digest: str) -> dict[str, Any]:
    """Copy the library clip into the conversion once; refuse a different clip on resume."""

    if workspace.chatterbox_reference_descriptor_path(conversion_id).exists():
        reference = workspace.load_chatterbox_reference(conversion_id)
        if reference.descriptor["reference_sha256"] != digest:
            raise ColabConflictError("active conversion uses a different custom voice; no files were changed. Pass start_new (--start-new) to discard it")
        return reference.descriptor
    if workspace.read_job(conversion_id).get("schema_version") == 4:
        raise ColabConflictError("active conversion has different voice, speed, or chunk settings; no files were changed. Pass start_new (--start-new) to discard it")
    workspace.store_chatterbox_reference(conversion_id, clip, consent_confirmed=True, consent_evidence="colab-voice-library")
    return workspace.load_chatterbox_reference(conversion_id).descriptor


def _ensure_chapter_plan(workspace: Workspace, conversion_id: str, mode: str, count: int | None) -> dict[str, Any]:
    job = workspace.read_job(conversion_id)
    if job.get("chapter_plan_sha256") is not None:
        plan = workspace.load_chapter_plan(conversion_id)
        if plan.get("mode") != mode or plan.get("requested_count") != count:
            raise ColabConflictError("active conversion has a different chapter configuration; resume it with the original settings, or pass start_new (--start-new) to discard it")
        return plan
    analysis = workspace.load_analysis(conversion_id)
    cleaned_text, cleaned_map = workspace.load_cleaned_artifacts(conversion_id)
    plan = create_chapter_plan(
        cleaned_text,
        cleaned_map,
        analysis.get("chapter_candidates", []),
        mode=mode,
        count=count,
        document_title=analysis.get("title"),
    )
    workspace.persist_chapter_plan(conversion_id, plan)
    return plan


def _print_progress(line: str) -> None:
    print(line, flush=True)


def _discard_active_conversion(workspace: Workspace, inspection: Any) -> None:
    """Delete the active conversion, refusing while a recorded worker is alive."""

    if inspection.state == "resumable":
        conversion_id = inspection.conversion_id
        job = workspace.read_job(conversion_id) if conversion_id else {}
        worker = job.get("worker")
        pid = worker.get("pid") if isinstance(worker, dict) else None
        if type(pid) is int and pid_is_alive(pid):
            raise ColabError("a conversion worker is still running for the active conversion; stop it before starting a new one")
        completed = job.get("completed_chunks")
        done = len(completed) if isinstance(completed, list) else 0
        _print_progress(
            f"Starting new: discarding active conversion {conversion_id} with {done} completed chunk(s). "
            "This deletes the whole active conversion tree - source copy, analysis, chapter plan and chunk WAVs. "
            "An already-published M4B is not affected."
        )
    else:
        _print_progress(
            "Starting new: discarding an invalid active workspace. The live-worker check could not be performed "
            "because the manifest did not validate. If active.json itself is malformed, only the active pointer is "
            "removed, which frees the workspace but may leave the old conversion tree orphaned on disk. "
            "An already-published M4B is not affected."
        )
    try:
        workspace.delete_active_state()
    except WorkspaceError as exc:
        raise ColabError(f"the previous conversion could not be discarded: {exc}") from exc


class ColabProgressDisplay:
    """Render validated generation manifests as bounded, line-oriented output."""

    def __init__(
        self,
        chapter_plan: dict[str, Any],
        *,
        output: Callable[[str], Any] | None = None,
        bar_width: int = 24,
    ) -> None:
        chapters = chapter_plan.get("chapters") if isinstance(chapter_plan, dict) else None
        self._chapter_total = len(chapters) if isinstance(chapters, list) else 0
        self._output = output if output is not None else _print_progress
        self._bar_width = max(8, min(int(bar_width), 48))
        self._last_key: tuple[Any, ...] | None = None
        self._last_milestone: tuple[Any, ...] | None = None

    @staticmethod
    def _ascii(value: Any, limit: int = 32) -> str:
        text = "".join(char if 32 <= ord(char) < 127 else "?" for char in str(value))
        return text[:limit]

    def _snapshot_key(self, manifest: dict[str, Any]) -> tuple[Any, ...]:
        progress = manifest["progress"]
        completed = manifest["completed_chunks"]
        latest = completed[-1] if completed else None
        return (
            manifest["status"],
            manifest["stage"],
            progress["completed"],
            progress["current"],
            progress["total"],
            (latest["global_index"], latest["chapter_index"]) if latest else None,
        )

    def _format(self, manifest: dict[str, Any]) -> str:
        progress = manifest["progress"]
        completed = progress["completed"]
        total = progress["total"]
        ratio = completed / total if total else 1.0
        filled = round(self._bar_width * ratio)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        details = f"{completed}/{total} chunks"
        records = manifest["completed_chunks"]
        if records and self._chapter_total:
            chapter = records[-1]["chapter_index"]
            details += f" | chapter {chapter}/{self._chapter_total}"
        stage = self._ascii(manifest["stage"])
        return f"[{bar}] {ratio * 100:5.1f}% ({details}) | stage: {stage}"

    def render(self, manifest: dict[str, Any]) -> None:
        """Render one already-validated manifest without affecting conversion."""

        try:
            key = self._snapshot_key(manifest)
            if key == self._last_key:
                return
            # Long books have thousands of chunks; print only when the stage,
            # chapter, or whole percentage changes.
            status, stage, completed, _current, total, latest = key
            milestone = (status, stage, latest[1] if latest else None, completed * 100 // total if total else 100)
            if milestone == self._last_milestone:
                return
            line = self._format(manifest)
            self._output(line)
            self._last_key = key
            self._last_milestone = milestone
        except Exception:
            # Display is best-effort.  The workspace update has already
            # succeeded, and an output/formatting failure must not change it.
            return


class _ProgressWorkspaceProxy:
    """Delegate a Workspace while observing its validated generation updates."""

    def __init__(self, workspace: Workspace, display: ColabProgressDisplay) -> None:
        self._workspace = workspace
        self._display = display

    def __getattr__(self, name: str) -> Any:
        return getattr(self._workspace, name)

    def update_generation(self, *args: Any, **updates: Any) -> dict[str, Any]:
        manifest = self._workspace.update_generation(*args, **updates)
        self._display.render(manifest)
        return manifest


@contextmanager
def _output_directory(path: Path) -> Iterator[None]:
    """Temporarily direct the existing finalizer to an explicit directory."""

    previous = os.environ.get("PDF_AUDIOBOOK_OUTPUT_DIR")
    os.environ["PDF_AUDIOBOOK_OUTPUT_DIR"] = str(Path(path).expanduser().absolute())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PDF_AUDIOBOOK_OUTPUT_DIR", None)
        else:
            os.environ["PDF_AUDIOBOOK_OUTPUT_DIR"] = previous


def run_conversion(
    pdf: str | os.PathLike[str],
    *,
    workspace_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    voice: str = "af_heart",
    speed: float = 1.0,
    clone: str | os.PathLike[str] | None = None,
    clone_model: str = DEFAULT_CLONE_MODEL,
    exaggeration: float = DEFAULT_EXAGGERATION,
    chapter_mode: str = "original",
    chapter_count: int | None = None,
    start_new: bool = False,
    cuda_check: Callable[[], Any] = _cuda_module,
    analyzer: Callable[..., dict[str, Any]] = functools.partial(analyze_pdf, layout_warnings=False),
    worker_class: Callable[..., Any] = ConversionWorker,
    engine_factory: Callable[..., Any] | None = None,
) -> Path:
    """Analyze, synthesize, and publish one PDF, resuming only a match.

    ``clone`` narrates with a custom voice clip through the Chatterbox
    ``clone_model`` instead of the Kokoro ``voice``; Chatterbox only supports
    speed 1.0.
    """

    if type(start_new) is not bool:
        raise ColabError("start_new must be a boolean")
    source = Path(pdf).expanduser().resolve()
    if not source.is_file():
        raise ColabError("PDF path does not exist or is not a regular file")
    mode, count = _request_mode(chapter_mode, chapter_count)
    # Every request-level check runs before the workspace is touched, so a bad
    # voice or speed can never destroy checkpoints on the start_new path.  The
    # speed is coerced exactly once here, so validation binds the same value the
    # rest of the run uses even for an object with a stateful __float__.
    requested_speed = _normalized_speed(speed)
    clip: Path | None = None
    clip_digest = ""
    if clone is None:
        _validate_voice_speed(voice, requested_speed)
    else:
        clone_model, exaggeration = _clone_options(clone_model, exaggeration)
        clip = Path(clone).expanduser().absolute()
        clip_digest = _clip_digest(clip)
        if requested_speed != 1.0:
            raise ColabError("custom voices only support speed 1.0")
    cuda_check()
    workspace = Workspace(Path(workspace_root).expanduser())
    output = Path(output_dir).expanduser().absolute()
    inspection = workspace.inspect_startup()
    if start_new and inspection.state != "no_active":
        _discard_active_conversion(workspace, inspection)
        inspection = workspace.inspect_startup()
    if inspection.state == "invalid":
        raise ColabError(f"active workspace is invalid: {inspection.reason or 'manifest validation failed'}; pass start_new (--start-new) to discard it and start over")

    resumed = inspection.state == "resumable"
    if resumed:
        if not inspection.conversion_id or not inspection.manifest:
            raise ColabError("active workspace did not identify a conversion")
        conversion_id = inspection.conversion_id
        job = workspace.read_job(conversion_id)
        if job["source_pdf_sha256"] != _sha256(source):
            raise ColabConflictError("active conversion uses a different PDF; no files were changed. Pass start_new (--start-new) to discard it")
    else:
        try:
            manifest = workspace.create_conversion(source, original_display_filename=source.name)
        except WorkspaceError as exc:
            raise ColabError(str(exc)) from exc
        conversion_id = manifest["conversion_id"]
        analysis = analyzer(workspace.conversion_path(conversion_id) / "source.pdf", fallback_title=source.name)
        workspace.persist_analysis(conversion_id, analysis)

    # A runtime can be reclaimed between source copy and PDF analysis. Finish
    # that safe, pre-generation state before asking for a chapter plan.
    job = workspace.read_job(conversion_id)
    if resumed and job.get("cleaned_text_sha256") is None:
        analysis = analyzer(
            workspace.conversion_path(conversion_id) / "source.pdf",
            fallback_title=job.get("original_display_filename"),
        )
        workspace.persist_analysis(conversion_id, analysis)

    plan = _ensure_chapter_plan(workspace, conversion_id, mode, count)
    cleaned_text, _ = workspace.load_cleaned_artifacts(conversion_id)
    if clip is None:
        tts, total_chunks = _expected_tts(cleaned_text, plan, voice, requested_speed)
    else:
        descriptor = _ensure_reference(workspace, conversion_id, clip, clip_digest)
        tts, total_chunks = _expected_chatterbox_tts(cleaned_text, plan, descriptor, clone_model, exaggeration)
    job = workspace.read_job(conversion_id)
    if job.get("schema_version") == 4 and (job.get("tts") != tts or job.get("total_chunks") != total_chunks):
        raise ColabConflictError("active conversion has different voice, speed, or chunk settings; no files were changed. Pass start_new (--start-new) to discard it")
    # A fully completed generation is immutable. Validation above still binds
    # this request to its source, plan, voice, speed, and chunk count, but the
    # existing verified publication must be returned without reconfiguration.
    if job.get("schema_version") == 4 and job.get("status") == "completed" and job.get("stage") == "completed" and job.get("output"):
        return Path(job["output"]["path"])
    workspace.configure_generation(conversion_id, tts=tts, total_chunks=total_chunks)

    factory = engine_factory or (make_cuda_kokoro_factory() if clip is None else make_cuda_chatterbox_factory(model=clone_model, exaggeration=exaggeration))
    display = ColabProgressDisplay(plan)
    progress_workspace = _ProgressWorkspaceProxy(workspace, display)
    worker = worker_class(progress_workspace, conversion_id, engine_factory=factory)
    with _output_directory(output):
        display.render(workspace.read_job(conversion_id))
        result: WorkerResult = worker.run(full_pipeline=True)
    if result.status != "completed":
        raise ColabError(f"conversion did not complete (status: {result.status})")
    final_job = workspace.read_job(conversion_id)
    final_output = final_job.get("output")
    if not isinstance(final_output, dict) or not final_output.get("path"):
        raise ColabError("worker completed without a verified M4B output")
    return Path(final_output["path"])


def preview_voice(
    voice: str,
    target: str | os.PathLike[str],
    *,
    speed: float = 1.0,
    text: str | None = None,
    cuda_check: Callable[[], Any] = _cuda_module,
    engine_factory: Callable[..., Any] | None = None,
    preview: Callable[..., Path] = generate_preview,
) -> Path:
    """Render one short preview WAV for a voice without touching a workspace."""

    requested_speed = _normalized_speed(speed)
    _validate_voice_speed(voice, requested_speed)
    sample_text = _preview_text(text)
    cuda_check()
    destination = Path(target).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    factory = engine_factory or make_cuda_kokoro_factory()
    try:
        return preview(voice, destination, settings={"speed": requested_speed}, voice_loader=factory, text=sample_text)
    except ColabError:
        raise
    except Exception as exc:
        raise ColabError(f"voice preview failed for {voice}") from exc


def preview_voices(
    voices: list[str],
    out_dir: str | os.PathLike[str],
    *,
    speed: float = 1.0,
    text: str | None = None,
    cuda_check: Callable[[], Any] = _cuda_module,
    engine_factory: Callable[..., Any] | None = None,
    preview: Callable[..., Path] = generate_preview,
) -> dict[str, Path]:
    """Render preview WAVs for multiple voices using custom or default text."""

    voice_list = list(voices)
    if not voice_list:
        raise ColabError("choose at least one voice to preview")
    # Validate the whole request before CUDA is probed or any file is written.

    requested_speed = _normalized_speed(speed)
    sample_text = _preview_text(text)
    for voice in voice_list:
        _validate_voice_speed(voice, requested_speed)

    unique_voices = list(dict.fromkeys(voice_list))
    digest = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()[:12]
    directory = Path(out_dir).expanduser().absolute()

    results: dict[str, Path] = {}
    missing: list[tuple[str, Path]] = []
    for voice in unique_voices:
        target = directory / f"{voice}-{requested_speed:g}-{digest}.wav"
        results[voice] = target
        if not target.is_file():
            missing.append((voice, target))

    if missing:
        cuda_check()
        factory = engine_factory or make_cuda_kokoro_factory()
        for voice, target in missing:
            results[voice] = preview_voice(
                voice,
                target,
                speed=requested_speed,
                text=sample_text,
                cuda_check=lambda: None,
                engine_factory=factory,
                preview=preview,
            )

    return results


def _split_for_chatterbox(text: str) -> list[str]:
    """Pack whole sentences into pieces no longer than the Chatterbox chunk cap."""

    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if pieces and len(pieces[-1]) + 1 + len(sentence) <= CHATTERBOX_CHUNK_CAP:
            pieces[-1] += " " + sentence
        elif sentence:
            pieces.append(sentence)
    return pieces


def preview_clones(
    clips: list[str | os.PathLike[str]],
    out_dir: str | os.PathLike[str],
    *,
    text: str | None = None,
    model: str = DEFAULT_CLONE_MODEL,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cuda_check: Callable[[], Any] = _cuda_module,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, Path]:
    """Render preview WAVs for custom voice clips, cached by clip, model and text."""

    model, exaggeration = _clone_options(model, exaggeration)
    clip_list = list(dict.fromkeys(Path(clip).expanduser().absolute() for clip in clips))
    if not clip_list:
        raise ColabError("choose at least one voice to preview")
    sample_text = _preview_text(text)
    text_digest = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()[:12]
    directory = Path(out_dir).expanduser().absolute()

    results: dict[str, Path] = {}
    missing: list[tuple[Path, Path]] = []
    for clip in clip_list:
        style = f"{model}-{exaggeration:g}" if model in ("original", "index") else ("qwen-icl" if model == "qwen" else model)
        target = directory / f"clone-{style}-{_clip_digest(clip)[:16]}-{text_digest}.wav"
        results[str(clip)] = target
        if not target.is_file():
            missing.append((clip, target))

    if missing:
        cuda_check()
        directory.mkdir(parents=True, exist_ok=True)
        factory = engine_factory or make_cuda_chatterbox_factory(model=model, exaggeration=exaggeration, torch_loader=lambda: None)
        for clip, target in missing:
            try:
                loaded = factory(CHATTERBOX_REFERENCE_VOICE, None, engine="chatterbox", reference_wav=clip)
                pcm = b"".join(loaded.synthesize(piece) for piece in _split_for_chatterbox(sample_text))
                write_pcm_wav(target, pcm, CHATTERBOX_SAMPLE_RATE, overwrite=True)
            except ColabError:
                raise
            except Exception as exc:
                raise ColabError(f"voice preview failed for {clip.stem}") from exc
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Single Voice Kokoro audiobook conversion on a CUDA runtime")
    parser.add_argument("pdf", type=Path, nargs="?")
    parser.add_argument("--workspace-root", type=Path, default=Path("/content/pdf-audiobook-workspace"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/pdf-audiobook-output"))
    parser.add_argument(
        "--voice",
        action="append",
        choices=APPROVED_VOICE_IDS,
        help="Kokoro voice; repeat with --preview-dir to preview several voices",
    )
    parser.add_argument(
        "--clone",
        action="append",
        type=Path,
        help="Custom voice clip (mono 16-bit WAV, 5-60 s) narrated with Chatterbox instead of a Kokoro voice; repeat with --preview-dir",
    )
    parser.add_argument("--clone-model", choices=CLONE_MODELS, default=DEFAULT_CLONE_MODEL, help="Chatterbox model for --clone voices")
    parser.add_argument("--exaggeration", type=float, default=DEFAULT_EXAGGERATION, help="Expression for the original model, 0.25-2.0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--chapter-mode", choices=("original", "whole", "custom"), default="original")
    parser.add_argument("--chapter-count", type=int)
    parser.add_argument(
        "--start-new",
        action="store_true",
        help=(
            "Discard the entire active conversion - source copy, analysis, chapter plan and chunk WAVs - "
            "before starting, so a new voice, speed or chapter configuration can be used. "
            "An already-published M4B is not affected."
        ),
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Render a short preview WAV for each --voice into this directory and exit; no PDF is needed",
    )
    parser.add_argument(
        "--preview-text",
        type=str,
        help="Custom preview text (1-10000 characters); defaults to the built-in sample sentence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    clones = args.clone or []
    voices = args.voice or ([] if clones else ["af_heart"])
    if args.preview_dir is not None:
        try:
            previews: dict[str, Path] = {}
            if voices:
                previews.update(preview_voices(voices, args.preview_dir, speed=args.speed, text=args.preview_text))
            if clones:
                previews.update(preview_clones(clones, args.preview_dir, text=args.preview_text, model=args.clone_model, exaggeration=args.exaggeration))
        except (ColabError, OSError, ValueError) as exc:
            print(f"Colab voice preview failed: {exc}")
            return 2
        for voice, path in previews.items():
            print(f"Preview WAV {voice}: {path}")
        return 0
    if len(voices) + len(clones) > 1:
        print("Colab conversion failed: a conversion uses exactly one --voice or --clone")
        return 2
    if args.pdf is None:
        print("Colab conversion failed: a PDF path is required unless --preview-dir is used")
        return 2
    try:
        result = run_conversion(
            args.pdf,
            workspace_root=args.workspace_root,
            output_dir=args.output_dir,
            voice=voices[0] if voices else "af_heart",
            speed=args.speed,
            clone=clones[0] if clones else None,
            clone_model=args.clone_model,
            exaggeration=args.exaggeration,
            chapter_mode=args.chapter_mode,
            chapter_count=args.chapter_count,
            start_new=args.start_new,
        )
    except (ColabError, ManifestError, WorkspaceError, OSError, ValueError) as exc:
        print(f"Colab conversion failed: {exc}")
        return 2
    print(f"Verified M4B: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ColabConflictError", "ColabError", "ColabProgressDisplay", "main", "CLONE_MODELS", "make_cuda_chatterbox_factory", "make_cuda_kokoro_factory", "preview_clones", "preview_voice", "preview_voices", "run_conversion"]

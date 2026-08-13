"""Headless Google Colab runner for the optional Kokoro GPU workflow.

This module is deliberately separate from the desktop launcher.  Colab calls
the same workspace, chapter planning, resumable worker, and M4B finalizer as
the local app, while its engine factory opts into CUDA explicitly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from .chapters import create_chapter_plan
from .pdf import analyze_pdf
from .tts import EngineMetadata, KokoroVoice, SynthesisSettings, plan_chunks
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
            pipeline = pipeline_class(lang_code=language, device="cuda")
            return KokoroVoice(pipeline, voice, settings, inference_context=inference_mode)
        except ColabError:
            raise
        except Exception as exc:
            raise ColabError("Kokoro 0.9.4 could not be loaded in the Colab GPU runtime") from exc

    return factory


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


def _expected_tts(cleaned_text: str, plan: dict[str, Any], voice: str, speed: float) -> tuple[dict[str, Any], int]:
    if voice not in APPROVED_VOICE_IDS:
        raise ColabError(f"voice is not approved: {voice}")
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ColabError("speed must be between 0.5 and 2.0")
    settings = SynthesisSettings(speed=speed)
    metadata = EngineMetadata(
        "kokoro", "0.9.4", "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded",
        voice, "captured-at-download", "unrecorded", settings.sample_rate, settings.as_dict(),
    )
    chunks = plan_chunks(cleaned_text, plan["chapters"], metadata, cap=settings.chunk_cap)
    return {**metadata.as_dict(), "speed": speed, "chunk_cap": settings.chunk_cap}, len(chunks)


def _ensure_chapter_plan(workspace: Workspace, conversion_id: str, mode: str, count: int | None) -> dict[str, Any]:
    job = workspace.read_job(conversion_id)
    if job.get("chapter_plan_sha256") is not None:
        plan = workspace.load_chapter_plan(conversion_id)
        if plan.get("mode") != mode or plan.get("requested_count") != count:
            raise ColabConflictError("active conversion has a different chapter configuration; resume it with the original settings")
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
            line = self._format(manifest)
            self._output(line)
            self._last_key = key
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
    chapter_mode: str = "original",
    chapter_count: int | None = None,
    cuda_check: Callable[[], Any] = _cuda_module,
    analyzer: Callable[..., dict[str, Any]] = analyze_pdf,
    worker_class: Callable[..., Any] = ConversionWorker,
    engine_factory: Callable[..., Any] | None = None,
) -> Path:
    """Analyze, synthesize, and publish one PDF, resuming only a match."""

    source = Path(pdf).expanduser().resolve()
    if not source.is_file():
        raise ColabError("PDF path does not exist or is not a regular file")
    mode, count = _request_mode(chapter_mode, chapter_count)
    cuda_check()
    workspace = Workspace(Path(workspace_root).expanduser())
    output = Path(output_dir).expanduser().absolute()
    inspection = workspace.inspect_startup()
    if inspection.state == "invalid":
        raise ColabError(f"active workspace is invalid: {inspection.reason or 'manifest validation failed'}")

    resumed = inspection.state == "resumable"
    if resumed:
        if not inspection.conversion_id or not inspection.manifest:
            raise ColabError("active workspace did not identify a conversion")
        conversion_id = inspection.conversion_id
        job = workspace.read_job(conversion_id)
        if job["source_pdf_sha256"] != _sha256(source):
            raise ColabConflictError("active conversion uses a different PDF; no files were changed")
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
    tts, total_chunks = _expected_tts(cleaned_text, plan, voice, float(speed))
    job = workspace.read_job(conversion_id)
    if job.get("schema_version") == 4 and (job.get("tts") != tts or job.get("total_chunks") != total_chunks):
        raise ColabConflictError("active conversion has different voice, speed, or chunk settings; no files were changed")
    # A fully completed generation is immutable. Validation above still binds
    # this request to its source, plan, voice, speed, and chunk count, but the
    # existing verified publication must be returned without reconfiguration.
    if job.get("schema_version") == 4 and job.get("status") == "completed" and job.get("stage") == "completed" and job.get("output"):
        return Path(job["output"]["path"])
    workspace.configure_generation(conversion_id, tts=tts, total_chunks=total_chunks)

    factory = engine_factory or make_cuda_kokoro_factory()
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Single Voice Kokoro audiobook conversion on a CUDA runtime")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("/content/pdf-audiobook-workspace"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/pdf-audiobook-output"))
    parser.add_argument("--voice", choices=APPROVED_VOICE_IDS, default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--chapter-mode", choices=("original", "whole", "custom"), default="original")
    parser.add_argument("--chapter-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_conversion(
            args.pdf,
            workspace_root=args.workspace_root,
            output_dir=args.output_dir,
            voice=args.voice,
            speed=args.speed,
            chapter_mode=args.chapter_mode,
            chapter_count=args.chapter_count,
        )
    except (ColabError, ManifestError, WorkspaceError, OSError, ValueError) as exc:
        print(f"Colab conversion failed: {exc}")
        return 2
    print(f"Verified M4B: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ColabConflictError", "ColabError", "ColabProgressDisplay", "main", "make_cuda_kokoro_factory", "run_conversion"]

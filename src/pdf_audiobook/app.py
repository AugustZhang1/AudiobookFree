"""Authenticated localhost application and Phase 4 generation routes."""

from __future__ import annotations

import hmac
import asyncio
from datetime import datetime, timezone
import os
import secrets
import stat
import subprocess
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from .audio import validate_wav
from .analysis_runner import AnalyzerDescriptor, VoiceAnalysisRunner
from .analyzers.booknlp import BookNLPAnalyzer
from .pdf import MAX_PDF_BYTES, PdfAnalysisError, analyze_pdf, preflight_pdf
from .chapters import ChapterPlanError, create_chapter_plan, rename_chapters, select_chapter_range, validate_chapter_plan
from .tts import APPROVED_VOICES, DEFAULT_CHUNK_CAP, DEFAULT_TORCH_THREADS, EngineMetadata, SynthesisSettings, TORCH_THREADS_ENV, plan_chunks, plan_interactive_chunks
from .security import instance_path, pid_is_alive, remove_instance_if_matches, token_hash
from .voice_plan import VoicePlanError, approve_voice_plan, assign_cast, build_voice_plan, merge_aliases, merge_cast, override_span, remove_cast, rename_cast, review_summary, split_aliases, with_canonical_artifact_hash
from .voice_registry import VoiceRegistryError, get_generation_facts, list_public_entries, registry_revision, require_enabled_voice_id, resolve_preview_path, resolve_preview_target
from .workspace import GENERATION_SCHEMA_VERSION, INTERACTIVE_GENERATION_SCHEMA_VERSION, ManifestError, Workspace, WorkspaceError

STATIC_DIR = Path(__file__).with_name("static")
COOKIE_NAME = "pdf_audiobook_session"
PREVIEW_GENERATION_TIMEOUT_SECONDS = 120


@dataclass
class AppState:
    port: int
    launch_id: str
    session_token: str
    instance_file: Path
    workspace_root: Path
    shutdown_event: threading.Event
    analysis_lock: threading.Lock
    generation_lock: threading.Lock
    preview_lock: threading.Lock
    worker_launcher: Any = None
    voice_analyzer: Any = None
    voice_analysis_thread: threading.Thread | None = None
    voice_analysis_conversion_id: str | None = None
    worker_process: Any = None
    uvicorn_server: Any = None
    preview_root: Path | None = None
    preview_generator: Any = None
    path_opener: Any = None

    @property
    def session_digest(self) -> str:
        return token_hash(self.session_token)


def _expected_host(request: Request, port: int) -> bool:
    host = request.headers.get("host", "")
    return host in {f"127.0.0.1:{port}", f"localhost:{port}"}


def _exact_origin(request: Request, port: int) -> bool:
    origin = request.headers.get("origin")
    return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


def _authenticated(request: Request, state: AppState) -> bool:
    supplied = request.cookies.get(COOKIE_NAME, "")
    return bool(supplied) and hmac.compare_digest(supplied, state.session_digest)


def _error_response(error: PdfAnalysisError) -> JSONResponse:
    return JSONResponse({"error": {"code": error.code, "message": error.message, **error.details}}, status_code=422)


def _chapter_error(error: ChapterPlanError, *, status_code: int | None = None) -> JSONResponse:
    if status_code is None:
        status_code = 409 if error.code in {"NO_ACTIVE", "NOT_READY"} else 422
    return JSONResponse({"error": {"code": error.code, "message": error.message, **error.details}}, status_code=status_code)


def _workspace_error(error: Exception, *, status_code: int = 422) -> JSONResponse:
    return JSONResponse({"error": {"code": "INVALID_WORKSPACE", "message": str(error)}}, status_code=status_code)


def _spawn_worker(workspace_root: Path, conversion_id: str, performance_mode: str = "background") -> Any:
    """Launch the isolated worker with a fixed argument vector and no shell."""

    if performance_mode not in {"background", "maximum_speed"}:
        raise ValueError("invalid performance_mode")
    repo_root = Path(__file__).resolve().parents[2]
    configured_interpreter = os.environ.get("PDF_AUDIOBOOK_KOKORO_PYTHON")
    interpreter = Path(configured_interpreter).expanduser() if configured_interpreter else repo_root / "benchmark" / "environments" / "kokoro" / ".venv" / "Scripts" / "python.exe"
    if not _safe_regular_file(interpreter):
        raise OSError("Kokoro worker interpreter is unavailable")
    package_src = str(repo_root / "src")
    child_env = os.environ.copy()
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = package_src + (os.pathsep + existing if existing else "")
    cpu_count = os.cpu_count()
    detected_cpus = cpu_count if cpu_count is not None and cpu_count > 0 else 8
    child_env[TORCH_THREADS_ENV] = str(detected_cpus if performance_mode == "maximum_speed" else min(DEFAULT_TORCH_THREADS, detected_cpus))
    return subprocess.Popen([str(interpreter), "-m", "pdf_audiobook.worker", str(workspace_root), conversion_id], shell=False, env=child_env)


def _default_preview_generator(voice: str, target: Path) -> None:
    """Generate a preview in the isolated Kokoro interpreter."""

    repo_root = Path(__file__).resolve().parents[2]
    configured_interpreter = os.environ.get("PDF_AUDIOBOOK_KOKORO_PYTHON")
    interpreter = Path(configured_interpreter).expanduser() if configured_interpreter else repo_root / "benchmark" / "environments" / "kokoro" / ".venv" / "Scripts" / "python.exe"
    if not _safe_regular_file(interpreter):
        raise RuntimeError("preview generator is unavailable")
    package_src = str(repo_root / "src")
    child_env = os.environ.copy()
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = package_src + (os.pathsep + existing if existing else "")
    cpu_count = os.cpu_count()
    detected_cpus = cpu_count if cpu_count is not None and cpu_count > 0 else 8
    child_env[TORCH_THREADS_ENV] = str(min(DEFAULT_TORCH_THREADS, detected_cpus))
    try:
        subprocess.run(
            [str(interpreter), "-m", "pdf_audiobook.preview_worker", voice, str(target)],
            shell=False,
            check=True,
            env=child_env,
            timeout=PREVIEW_GENERATION_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError("preview generation failed") from exc


class _PreviewGenerationError(RuntimeError):
    pass


def _prepare_preview(state: AppState, voice: str) -> None:
    """Serialize cache writes and publish only a validated completed WAV."""

    with state.preview_lock:
        if state.preview_root is None:
            raise VoiceRegistryError("preview root is unavailable")
        existing = resolve_preview_path(voice, state.preview_root)
        if existing is not None:
            try:
                validate_wav(existing, expected_sample_rate=24000)
                return
            except (OSError, ValueError):
                pass
        target = resolve_preview_target(voice, state.preview_root)
        if target is None:
            raise VoiceRegistryError("preview root is unavailable")
        generator = state.preview_generator or _default_preview_generator
        try:
            generator(voice, target)
            validate_wav(target, expected_sample_rate=24000)
        except Exception as exc:
            raise _PreviewGenerationError("preview generation failed") from exc


async def _json_body(request: Request) -> dict[str, Any] | None:
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _display_filename(request: Request) -> str:
    encoded = request.headers.get("x-pdf-filename", "upload.pdf")
    name = urllib.parse.unquote(encoded).strip()
    if not name or "\x00" in name or "/" in name or "\\" in name or name in {".", ".."}:
        raise PdfAnalysisError("INVALID_FILENAME", "The upload filename is invalid.")
    return name


def _public_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in analysis.items() if key not in {"cleaned_text", "cleaned_map"}}


def _selected_chapters(plan: dict[str, Any], settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Resolve the durable range settings, defaulting legacy jobs to all chapters."""

    settings = settings or {}
    return select_chapter_range(plan, settings.get("chapter_start"), settings.get("chapter_end"))


def _is_reparse(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _safe_regular_file(path: Path) -> bool:
    """Return whether ``path`` is a regular, non-link, non-reparse file."""

    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not _is_reparse(path)


def _safe_real_directory(path: Path) -> bool:
    """Return whether ``path`` is a real directory rather than a link."""

    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not _is_reparse(path)


def _analyzer_descriptor(analyzer: Any, *, injected: bool) -> AnalyzerDescriptor:
    """Normalize optional adapter descriptor facts at the application boundary."""

    raw = getattr(analyzer, "descriptor", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = None
    if hasattr(raw, "as_dict") and callable(raw.as_dict):
        try:
            raw = raw.as_dict()
        except Exception:
            raw = None
    if not isinstance(raw, dict):
        raw = {}
    default_id = "injected" if injected else "booknlp"
    try:
        return AnalyzerDescriptor(str(raw.get("id", default_id)), str(raw.get("version", "1")), raw.get("model_hash"))
    except Exception:
        return AnalyzerDescriptor(default_id, "1", None)


def _voice_plan_error(error: VoicePlanError, *, status_code: int | None = None) -> JSONResponse:
    if status_code is None:
        status_code = 409 if error.code == "STALE_REVISION" else 422
    code = "PLAN_CONFLICT" if error.code == "STALE_REVISION" else error.code
    return JSONResponse({"error": {"code": code, "message": error.message, **error.details}}, status_code=status_code)


def _default_path_opener(path: Path) -> None:
    opener = getattr(os, "startfile", None)
    if not callable(opener):
        raise OSError("path opener is unavailable")
    opener(str(path))


def _generation_summary(cleaned_text: str, chapter_plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Build factual generation progress from the bound plan and manifest."""

    if manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
        return None
    try:
        tts = manifest["tts"]
        metadata = EngineMetadata(
            str(tts["engine"]), str(tts.get("package_version", "")), str(tts.get("model", "")),
            str(tts.get("model_revision", "")), str(tts.get("model_checksum", "")), str(tts["voice"]),
            str(tts.get("voice_version", "")), str(tts.get("voice_checksum", "")), int(tts["sample_rate"]),
            dict(tts.get("settings", {})),
        )
        chapters = _selected_chapters(chapter_plan, dict(tts.get("settings", {})))
        chunks = plan_chunks(cleaned_text, chapters, metadata, cap=int(tts["chunk_cap"]))
        if len(chunks) != manifest["total_chunks"]:
            return None
        by_global = {chunk.global_index: chunk for chunk in chunks}
        completed_globals: set[int] = set()
        for record in manifest["completed_chunks"]:
            chunk = by_global.get(record["global_index"])
            if chunk is None or (record["chapter_index"], record["local_index"], record["input_hash"]) != (chunk.chapter_index, chunk.local_index, chunk.input_hash):
                return None
            expected_path = f"chunks/chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"
            if record.get("relative_path") != expected_path:
                return None
            completed_globals.add(chunk.global_index)
        chapters = _selected_chapters(chapter_plan, dict(tts.get("settings", {})))
        chapter_indexes = [int(chapter["index"]) for chapter in chapters]
        chapter_chunks = {index: [chunk.global_index for chunk in chunks if chunk.chapter_index == index] for index in chapter_indexes}
        completed_chapter_indexes = {index for index in chapter_indexes if chapter_chunks[index] and all(global_index in completed_globals for global_index in chapter_chunks[index])}
        completed_chapters = len(completed_chapter_indexes)
        current_chapter = next((index for index in chapter_indexes if index not in completed_chapter_indexes), None)
        if current_chapter is None and chapter_indexes:
            current_chapter = chapter_indexes[-1]
        worker = manifest.get("worker")
        run_started_at = worker.get("started_at") if isinstance(worker, dict) else None
        return {
            "stage": manifest.get("stage"),
            "total_chunks": len(chunks),
            "completed_chunks": len(completed_globals),
            "total_chapters": len(chapters),
            "current_chapter": current_chapter,
            "completed_chapters": completed_chapters,
            "run_started_at": run_started_at,
        }
    except (KeyError, TypeError, ValueError, ChapterPlanError):
        return None


def _interactive_generation_summary(workspace: Workspace, conversion_id: str, cleaned_text: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Build factual progress for a schema-v5 plan-bound generation."""

    if manifest.get("schema_version") != INTERACTIVE_GENERATION_SCHEMA_VERSION:
        return None
    try:
        plan = workspace.load_voice_plan(conversion_id)
        if plan.get("approval", {}).get("state") != "approved":
            return None
        if plan.get("canonical_artifact_sha256") != manifest.get("voice_plan_sha256") or plan.get("revision") != manifest.get("voice_plan_revision"):
            return None
        if manifest.get("voice_registry_revision") != registry_revision():
            return None
        settings = manifest["tts"].get("settings", {})
        chapter_range = None
        if settings.get("chapter_start") is not None or settings.get("chapter_end") is not None:
            if type(settings.get("chapter_start")) is not int or type(settings.get("chapter_end")) is not int:
                return None
            chapter_range = (settings["chapter_start"], settings["chapter_end"])
        chunks = plan_interactive_chunks(
            cleaned_text,
            plan,
            get_generation_facts,
            registry_revision(),
            chapter_range=chapter_range,
            cap=int(manifest["tts"].get("chunk_cap", settings.get("chunk_cap", DEFAULT_CHUNK_CAP))),
        )
        if len(chunks) != manifest.get("total_chunks"):
            return None
        by_global = {chunk.global_index: chunk for chunk in chunks}
        completed_globals: set[int] = set()
        for record in manifest.get("completed_chunks", []):
            chunk = by_global.get(record.get("global_index"))
            expected_path = f"chunks/chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav" if chunk is not None else None
            if chunk is None or record.get("chapter_index") != chunk.chapter_index or record.get("local_index") != chunk.local_index or record.get("source_start") != chunk.source_start or record.get("source_end") != chunk.source_end or record.get("input_hash") != chunk.input_hash or record.get("relative_path") != expected_path:
                return None
            if record.get("span_id") != chunk.span_id or record.get("speaker_id") != chunk.speaker_id or record.get("voice_id") != chunk.voice_id or record.get("segment_type") != chunk.segment_type:
                return None
            completed_globals.add(chunk.global_index)
        chapter_indexes = sorted({chunk.chapter_index for chunk in chunks})
        chapter_chunks = {index: [chunk.global_index for chunk in chunks if chunk.chapter_index == index] for index in chapter_indexes}
        completed_chapters = sum(bool(values) and all(value in completed_globals for value in values) for values in chapter_chunks.values())
        current_chapter = next((ordinal for ordinal, index in enumerate(chapter_indexes, 1) if not all(value in completed_globals for value in chapter_chunks[index])), None)
        if current_chapter is None and chapter_indexes:
            current_chapter = len(chapter_indexes)
        worker = manifest.get("worker")
        return {
            "stage": manifest.get("stage"),
            "total_chunks": len(chunks),
            "completed_chunks": len(completed_globals),
            "total_chapters": len(chapter_indexes),
            "current_chapter": current_chapter,
            "completed_chapters": completed_chapters,
            "run_started_at": worker.get("started_at") if isinstance(worker, dict) else None,
        }
    except (KeyError, TypeError, ValueError, WorkspaceError):
        return None


def _staging_directory(root: Path) -> Path:
    """Return a safe, contained staging directory under the resolved root."""

    staging = root / ".staging"
    if staging.exists() or staging.is_symlink():
        try:
            info = staging.lstat()
        except OSError as exc:
            raise PdfAnalysisError("STAGING_UNAVAILABLE", "The upload staging directory is unavailable.") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(staging) or not stat.S_ISDIR(info.st_mode):
            raise PdfAnalysisError("STAGING_UNAVAILABLE", "The upload staging directory is unsafe.")
    else:
        try:
            staging.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PdfAnalysisError("STAGING_UNAVAILABLE", "The upload staging directory is unavailable.") from exc
    try:
        staging.resolve().relative_to(root)
    except ValueError as exc:
        raise PdfAnalysisError("STAGING_UNAVAILABLE", "The upload staging directory is unsafe.") from exc
    return staging


def _process_staged_pdf(staged: Path, workspace: Workspace, display_name: str, lock: threading.Lock) -> dict[str, Any]:
    with lock:
        preflight_pdf(staged)
        try:
            manifest = workspace.create_conversion(staged, original_display_filename=display_name)
        except WorkspaceError as exc:
            raise PdfAnalysisError("ACTIVE_JOB", str(exc)) from exc
        source = workspace.work_root / manifest["conversion_id"] / "source.pdf"
        try:
            analysis = analyze_pdf(source, fallback_title=display_name, check_disk=False)
        except PdfAnalysisError as exc:
            workspace.update_job(manifest["conversion_id"], status="failed", stage="analysis", error=f"{exc.code}: {exc.message}")
            exc.details["conversion_id"] = manifest["conversion_id"]
            raise
        job = workspace.persist_analysis(manifest["conversion_id"], analysis)
        return {"conversion_id": manifest["conversion_id"], "status": job["status"], "job": job, "analysis": _public_analysis(analysis)}


def create_app(*, port: int, launch_id: str | None = None, session_token: str | None = None, instance_file: Path | None = None, data_root: Path | None = None, worker_launcher: Any | None = None, preview_root: Path | None = None, preview_generator: Any | None = None, path_opener: Any | None = None, voice_analyzer: Any | None = None) -> FastAPI:
    instance_file = instance_file or instance_path()
    workspace_root = Path(data_root or instance_file.parent).expanduser().resolve()
    preview_root = Path(preview_root or (Path(__file__).resolve().parents[2] / "benchmark" / "previews")).expanduser()
    state = AppState(
        port=port,
        launch_id=launch_id or secrets.token_hex(16),
        session_token=session_token or secrets.token_urlsafe(32),
        instance_file=instance_file,
        workspace_root=workspace_root,
        shutdown_event=threading.Event(),
        analysis_lock=threading.Lock(),
        generation_lock=threading.Lock(),
        preview_lock=threading.Lock(),
        worker_launcher=worker_launcher or _spawn_worker,
        voice_analyzer=voice_analyzer,
        preview_root=preview_root,
        preview_generator=preview_generator or _default_preview_generator,
        path_opener=path_opener or _default_path_opener,
    )
    app = FastAPI(title="PDF to Audiobook", docs_url=None, redoc_url=None)
    app.state.phase1 = state

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        if not _expected_host(request, state.port):
            return JSONResponse({"detail": "invalid host"}, status_code=400)
        return await call_next(request)

    @app.get("/health")
    async def health(request: Request):
        authorized = hmac.compare_digest(request.headers.get("x-instance-token", ""), state.session_token)
        return {"status": "ok", "ready": True, "launch_id": state.launch_id if authorized else None}

    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/styles.css", response_class=FileResponse)
    async def styles():
        return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js", response_class=FileResponse)
    async def script():
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @app.post("/api/session/bootstrap")
    async def bootstrap(request: Request, response: Response):
        if not _exact_origin(request, state.port):
            raise HTTPException(403, "exact local Origin required")
        body = await request.json()
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not hmac.compare_digest(token, state.session_token):
            raise HTTPException(401, "invalid session bootstrap")
        response.set_cookie(COOKIE_NAME, state.session_digest, httponly=True, samesite="strict", secure=False, path="/")
        return {"authenticated": True, "launch_id": state.launch_id}

    @app.get("/api/session")
    async def session(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        return {"authenticated": True, "launch_id": state.launch_id}

    @app.get("/api/voice-preview/{voice}")
    async def voice_preview(voice: str, request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        unavailable = {"error": {"code": "VOICE_PREVIEW_UNAVAILABLE", "message": "voice preview unavailable"}}
        if state.preview_root is None:
            return JSONResponse(unavailable, status_code=404)
        try:
            validated = require_enabled_voice_id(voice)
        except VoiceRegistryError:
            return JSONResponse(unavailable, status_code=404)
        try:
            await asyncio.to_thread(_prepare_preview, state, validated)
        except VoiceRegistryError:
            return JSONResponse(unavailable, status_code=404)
        except _PreviewGenerationError:
            return JSONResponse({"error": {"code": "VOICE_PREVIEW_FAILED", "message": "voice preview generation failed"}}, status_code=503)
        try:
            candidate = resolve_preview_path(validated, state.preview_root)
            if candidate is None:
                return JSONResponse(unavailable, status_code=404)
            validate_wav(candidate, expected_sample_rate=24000)
        except (OSError, ValueError, VoiceRegistryError):
            return JSONResponse(unavailable, status_code=404)
        return FileResponse(candidate, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.post("/api/voice-preview/{voice}/prepare")
    async def prepare_voice_preview(voice: str, request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        if not _exact_origin(request, state.port):
            raise HTTPException(403, "exact local Origin required")
        unavailable = {"error": {"code": "VOICE_PREVIEW_UNAVAILABLE", "message": "voice preview unavailable"}}
        try:
            validated = require_enabled_voice_id(voice)
        except VoiceRegistryError:
            return JSONResponse(unavailable, status_code=404)
        try:
            await asyncio.to_thread(_prepare_preview, state, validated)
        except VoiceRegistryError:
            return JSONResponse(unavailable, status_code=404)
        except _PreviewGenerationError:
            return JSONResponse({"error": {"code": "VOICE_PREVIEW_FAILED", "message": "voice preview generation failed"}}, status_code=503)
        return {"voice": validated, "status": "ready"}

    @app.get("/api/voices")
    async def voices(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        entries: list[dict[str, Any]] = []
        for entry in list_public_entries():
            projected = dict(entry)
            try:
                projected["preview_available"] = bool(state.preview_root is not None and resolve_preview_path(entry["id"], state.preview_root) is not None)
            except (OSError, ValueError, VoiceRegistryError):
                projected["preview_available"] = False
            entries.append(projected)
        revision = registry_revision()
        return {"revision": revision, "registry_revision": revision, "voices": entries}

    @app.get("/api/status")
    async def status(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        workspace = Workspace(state.workspace_root)
        inspection = await asyncio.to_thread(workspace.inspect_startup)
        if inspection.state == "no_active":
            return {"state": "no_active"}
        if inspection.state == "invalid":
            return {"state": "invalid", "conversion_id": inspection.conversion_id, "reason": inspection.reason}
        manifest = inspection.manifest or {}
        status_value = manifest.get("status")
        local_process = state.worker_process
        local_live = local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
        launchable_before_claim = status_value == "planned" or (
            manifest.get("schema_version") in {GENERATION_SCHEMA_VERSION, INTERACTIVE_GENERATION_SCHEMA_VERSION} and status_value in {"cancelled", "failed"}
        ) or (
            status_value == "completed" and manifest.get("stage") == "synthesis_complete" and manifest.get("output") is None
        )
        inferred_starting = local_live and launchable_before_claim
        result: dict[str, Any] = {"state": "starting" if inferred_starting else status_value if status_value in {"synthesizing", "cancelling", "cancelled", "failed", "assembling", "encoding", "verifying", "publishing", "completed"} else ("analyzed" if status_value == "analyzed" else "resumable"), "conversion_id": inspection.conversion_id, "job": manifest}
        if inspection.conversion_id:
            result["interactive_voices"] = await _interactive_status_projection(workspace, inspection.conversion_id)
        if inspection.conversion_id and status_value in {"analyzed", "planned", "synthesizing", "cancelling", "cancelled", "failed", "assembling", "encoding", "verifying", "publishing", "completed"}:
            try:
                analysis = await asyncio.to_thread(workspace.load_analysis, inspection.conversion_id)
                result["analysis"] = _public_analysis(analysis)
                if manifest.get("status") in {"planned", "synthesizing", "cancelling", "cancelled", "failed", "assembling", "encoding", "verifying", "publishing", "completed"}:
                    text, cleaned_map = await asyncio.to_thread(workspace.load_cleaned_artifacts, inspection.conversion_id)
                    plan = await asyncio.to_thread(workspace.load_chapter_plan, inspection.conversion_id)
                    result["chapter_plan"] = await asyncio.to_thread(validate_chapter_plan, plan, text, cleaned_map)
                    if manifest.get("status") == "planned" and not inferred_starting:
                        result["state"] = "planned"
                    summary = _generation_summary(text, result["chapter_plan"], manifest)
                    if summary is None and manifest.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
                        summary = _interactive_generation_summary(workspace, inspection.conversion_id, text, manifest)
                    if summary is not None:
                        result["generation_summary"] = summary
            except (WorkspaceError, OSError) as exc:
                result["state"] = "invalid"
                result["reason"] = str(exc)
            except ChapterPlanError as exc:
                result["state"] = "invalid"
                result["reason"] = exc.message
        return result

    @app.post("/api/output/open")
    async def output_open(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or set(body) != {"target"} or body.get("target") not in {"audiobook", "folder"}:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "target must be audiobook or folder"}}, status_code=422)
        try:
            inspection = await asyncio.to_thread(Workspace(state.workspace_root).inspect_startup)
        except Exception:
            inspection = None
        manifest = inspection.manifest if inspection is not None and inspection.state == "resumable" else None
        output = manifest.get("output") if isinstance(manifest, dict) and manifest.get("status") == "completed" and manifest.get("stage") == "completed" else None
        output_path = Path(output["path"]) if isinstance(output, dict) and isinstance(output.get("path"), str) else None
        if output_path is None or not _safe_regular_file(output_path) or not _safe_real_directory(output_path.parent):
            return JSONResponse({"error": {"code": "OUTPUT_UNAVAILABLE", "message": "completed output is unavailable"}}, status_code=409)
        target_path = output_path if body["target"] == "audiobook" else output_path.parent
        try:
            await asyncio.to_thread(state.path_opener, target_path)
        except Exception:
            return JSONResponse({"error": {"code": "OUTPUT_OPEN_FAILED", "message": "completed output could not be opened"}}, status_code=503)
        return {"opened": body["target"]}

    def _active_plan_inputs(workspace: Workspace) -> tuple[str, dict[str, Any], str, list[dict[str, Any]], dict[str, Any]]:
        inspection = workspace.inspect_startup()
        if inspection.state == "no_active":
            raise ChapterPlanError("NO_ACTIVE", "There is no active conversion to plan.")
        if inspection.state == "invalid" or not inspection.conversion_id or not inspection.manifest:
            raise ChapterPlanError("INVALID_ACTIVE", "The active conversion is invalid.")
        if inspection.manifest.get("status") not in {"analyzed", "planned"}:
            raise ChapterPlanError("NOT_READY", "The active conversion is not ready for chapter planning.")
        conversion_id = inspection.conversion_id
        try:
            analysis = workspace.load_analysis(conversion_id)
            cleaned_text, cleaned_map = workspace.load_cleaned_artifacts(conversion_id)
        except (WorkspaceError, OSError) as exc:
            raise ChapterPlanError("INVALID_ACTIVE", "The active conversion artifacts are invalid.") from exc
        return conversion_id, analysis, cleaned_text, cleaned_map, inspection.manifest

    def _plan_and_persist(workspace: Workspace, *, mode: str, count: int | None) -> dict[str, Any]:
        conversion_id, analysis, cleaned_text, cleaned_map, _ = _active_plan_inputs(workspace)
        plan = create_chapter_plan(
            cleaned_text,
            cleaned_map,
            analysis.get("chapter_candidates", []),
            mode=mode,
            count=count,
            document_title=analysis.get("title"),
        )
        plan = validate_chapter_plan(plan, cleaned_text, cleaned_map)
        job = workspace.persist_chapter_plan(conversion_id, plan)
        return {"conversion_id": conversion_id, "status": job["status"], "job": job, "analysis": _public_analysis(analysis), "chapter_plan": plan}

    def _rename_and_persist(workspace: Workspace, titles: list[Any]) -> dict[str, Any]:
        inspection = workspace.inspect_startup()
        if inspection.state == "no_active":
            raise ChapterPlanError("NO_ACTIVE", "There is no active conversion to rename.")
        if inspection.state == "invalid" or not inspection.conversion_id or not inspection.manifest:
            raise ChapterPlanError("INVALID_ACTIVE", "The active conversion is invalid.")
        conversion_id = inspection.conversion_id
        manifest = inspection.manifest
        generation_manifest = manifest.get("schema_version") == GENERATION_SCHEMA_VERSION
        status = manifest.get("status")
        if generation_manifest:
            inactive_generation = status in {"planned", "cancelled", "failed"}
            synthesis_complete = status == "completed" and manifest.get("stage") == "synthesis_complete" and manifest.get("output") is None
            if not inactive_generation and not synthesis_complete:
                raise ChapterPlanError("NOT_READY", "Labels can only be changed while generation is inactive.")
        elif status != "planned":
            raise ChapterPlanError("NOT_READY", "Generate a chapter plan before renaming labels.")
        try:
            analysis = workspace.load_analysis(conversion_id)
            cleaned_text, cleaned_map = workspace.load_cleaned_artifacts(conversion_id)
        except (WorkspaceError, OSError) as exc:
            raise ChapterPlanError("INVALID_ACTIVE", "The active conversion artifacts are invalid.") from exc
        try:
            plan = workspace.load_chapter_plan(conversion_id)
        except (WorkspaceError, OSError) as exc:
            raise ChapterPlanError("INVALID_ACTIVE", "The active chapter plan is invalid.") from exc
        plan = validate_chapter_plan(plan, cleaned_text, cleaned_map)
        renamed = rename_chapters(plan, titles)
        renamed = validate_chapter_plan(renamed, cleaned_text, cleaned_map)
        job = workspace.persist_chapter_plan(conversion_id, renamed)
        return {"conversion_id": conversion_id, "status": job["status"], "job": job, "analysis": _public_analysis(analysis), "chapter_plan": renamed}

    @app.post("/api/chapter-plan")
    async def chapter_plan(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None:
            return _chapter_error(ChapterPlanError("INVALID_BODY", "request body must be a JSON object"))
        mode = body.get("mode")
        count = body.get("count")
        if mode not in {"original", "custom", "whole"}:
            return _chapter_error(ChapterPlanError("INVALID_MODE", "mode must be original, custom, or whole"))
        if mode == "custom" and (type(count) is not int or not 2 <= count <= 50):
            return _chapter_error(ChapterPlanError("INVALID_COUNT", "custom count must be an integer from 2 through 50"))
        if mode != "custom" and count is not None:
            return _chapter_error(ChapterPlanError("INVALID_COUNT", "count is only valid for custom mode"))
        workspace = Workspace(state.workspace_root)
        acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
        if not acquired:
            return JSONResponse({"error": {"code": "BUSY", "message": "Another workspace operation is in progress."}}, status_code=409)
        try:
            try:
                return await asyncio.to_thread(_plan_and_persist, workspace, mode=mode, count=count)
            except ChapterPlanError as exc:
                return _chapter_error(exc)
            except (WorkspaceError, OSError) as exc:
                return _workspace_error(exc)
        finally:
            state.analysis_lock.release()

    @app.post("/api/chapter-plan/titles")
    async def chapter_plan_titles(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or not isinstance(body.get("titles"), list):
            return _chapter_error(ChapterPlanError("INVALID_TITLES", "titles must be a JSON array"))
        workspace = Workspace(state.workspace_root)
        generation_acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not generation_acquired:
            return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation operation is in progress"}}, status_code=409)
        acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
        if not acquired:
            state.generation_lock.release()
            return JSONResponse({"error": {"code": "BUSY", "message": "Another workspace operation is in progress."}}, status_code=409)
        try:
            try:
                inspection = await asyncio.to_thread(workspace.inspect_startup)
                if inspection.state == "resumable" and inspection.manifest:
                    recorded = inspection.manifest.get("worker") or {}
                    recorded_live = bool(recorded and pid_is_alive(int(recorded.get("pid", 0))))
                    local_process = state.worker_process
                    local_live = local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
                    if recorded_live or local_live:
                        return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
                return await asyncio.to_thread(_rename_and_persist, workspace, body["titles"])
            except ChapterPlanError as exc:
                return _chapter_error(exc)
            except (WorkspaceError, OSError) as exc:
                return _workspace_error(exc)
        finally:
            state.analysis_lock.release()
            state.generation_lock.release()

    @app.post("/api/analyze")
    async def analyze(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        try:
            display_name = _display_filename(request)
            staging_dir = _staging_directory(state.workspace_root)
            staged = staging_dir / f"{uuid.uuid4()}.upload"
            staged_created = False
            total = 0
            with staged.open("xb") as handle:
                staged_created = True
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise PdfAnalysisError("SIZE_LIMIT", "The PDF exceeds the 100 MiB size limit.", details={"maximum_bytes": MAX_PDF_BYTES})
                    handle.write(chunk)
                handle.flush()
            result = await asyncio.to_thread(_process_staged_pdf, staged, Workspace(state.workspace_root), display_name, state.analysis_lock)
            return result
        except PdfAnalysisError as exc:
            return _error_response(exc)
        except OSError as exc:
            return _error_response(PdfAnalysisError("STAGING_UNAVAILABLE", "The upload staging directory is unavailable."))
        finally:
            if locals().get("staged_created", False):
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pass

    def _generation_inputs(workspace: Workspace, conversion_id: str, voice: str, speed: float, chapter_start: int | None = None, chapter_end: int | None = None) -> tuple[dict[str, Any], int]:
        job = workspace.read_job(conversion_id)
        if job.get("schema_version") != GENERATION_SCHEMA_VERSION and job.get("status") != "planned":
            raise WorkspaceError("generation requires a persisted chapter plan")
        text, _ = workspace.load_cleaned_artifacts(conversion_id)
        plan = workspace.load_chapter_plan(conversion_id)
        chapters = select_chapter_range(plan, chapter_start, chapter_end)
        normalized_start = 1 if chapter_start is None else chapter_start
        normalized_end = len(plan["chapters"]) if chapter_end is None else chapter_end
        settings = SynthesisSettings(speed=speed)
        metadata = EngineMetadata("kokoro", "0.9.4", "hexgrad/Kokoro-82M", "captured-at-download", "unrecorded", voice, "captured-at-download", "unrecorded", settings.sample_rate, settings.as_dict())
        chunks = plan_chunks(text, chapters, metadata, cap=settings.chunk_cap)
        tts = {**metadata.as_dict(), "speed": speed, "chunk_cap": settings.chunk_cap}
        if normalized_start != 1 or normalized_end != len(plan["chapters"]):
            tts["settings"] = {**tts["settings"], "chapter_start": normalized_start, "chapter_end": normalized_end}
        return tts, len(chunks)

    def _voice_generation_live(manifest: dict[str, Any]) -> bool:
        recorded = manifest.get("worker") or {}
        try:
            recorded_live = bool(recorded and pid_is_alive(int(recorded.get("pid", 0))))
        except (TypeError, ValueError):
            recorded_live = False
        local_process = state.worker_process
        local_live = local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
        return recorded_live or local_live

    def _voice_status_projection(conversion_id: str, status: dict[str, Any], *, cancel_requested: bool | None = None, chapter_count: int | None = None) -> dict[str, Any]:
        requested = status["cancel_requested"] if cancel_requested is None else cancel_requested
        chapter_start = status.get("chapter_start", 1)
        chapter_end = status.get("chapter_end", chapter_count)
        result: dict[str, Any] = {
            "conversion_id": conversion_id,
            "analysis_id": status["analysis_id"],
            "revision": status["revision"],
            "status": status["status"],
            "stage": status["stage"],
            "progress": dict(status["progress"]),
            "analyzer": dict(status["analyzer"]),
            "warnings": list(status["warnings"]),
            "cancel_requested": requested,
            "cancelable": status["status"] in {"queued", "running"},
            "started_at": status["started_at"],
            "updated_at": status["updated_at"],
            "finished_at": status["finished_at"],
            "source_pdf_sha256": status["source_pdf_sha256"],
            "cleaned_text_sha256": status["cleaned_text_sha256"],
            "chapter_plan_sha256": status["chapter_plan_sha256"],
            "chapter_start": chapter_start,
            "chapter_end": chapter_end,
            "canonical_artifact_sha256": status["canonical_artifact_sha256"],
        }
        if status["error"] is not None:
            result["error"] = dict(status["error"])
        return result

    def _voice_active_inputs(workspace: Workspace) -> tuple[str, dict[str, Any]]:
        inspection = workspace.inspect_startup()
        if inspection.state != "resumable" or not inspection.conversion_id or not inspection.manifest:
            raise WorkspaceError("an analyzed chapter plan is required")
        manifest = inspection.manifest
        if not manifest.get("chapter_plan_sha256"):
            raise WorkspaceError("an analyzed chapter plan is required")
        workspace.load_chapter_plan(inspection.conversion_id)
        return inspection.conversion_id, manifest

    def _voice_analyzer() -> tuple[Any, AnalyzerDescriptor] | None:
        if state.voice_analyzer is not None:
            if callable(getattr(state.voice_analyzer, "analyze", None)):
                return state.voice_analyzer, _analyzer_descriptor(state.voice_analyzer, injected=True)
            return None
        try:
            analyzer = BookNLPAnalyzer()
            if not _safe_regular_file(analyzer.python_executable) or not _safe_regular_file(analyzer.runner_path):
                return None
            return analyzer, _analyzer_descriptor(analyzer, injected=False)
        except Exception:
            return None

    async def _interactive_status_projection(workspace: Workspace, conversion_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"available": False, "analysis": None, "plan": None}
        try:
            analysis_status = await asyncio.to_thread(workspace.load_voice_analysis_status, conversion_id)
            chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
            result["analysis"] = {
                "status": analysis_status["status"],
                "stage": analysis_status["stage"],
                "progress": dict(analysis_status["progress"]),
                "cancelable": analysis_status["status"] in {"queued", "running"},
                "revision": analysis_status["revision"],
                "sha256": analysis_status["canonical_artifact_sha256"],
                "chapter_start": analysis_status.get("chapter_start", 1),
                "chapter_end": analysis_status.get("chapter_end", len(chapter_plan["chapters"])),
            }
        except (WorkspaceError, OSError):
            return result
        try:
            plan = await asyncio.to_thread(workspace.load_voice_plan, conversion_id)
            result["plan"] = {
                "revision": plan["revision"],
                "sha256": plan["canonical_artifact_sha256"],
                "canonical_artifact_sha256": plan["canonical_artifact_sha256"],
                "approval": plan["approval"]["state"],
                "review": review_summary(plan),
            }
        except (WorkspaceError, OSError):
            return result
        result["available"] = True
        return result

    async def _interactive_generation_start(body: dict[str, Any], workspace: Workspace):
        allowed = {"mode", "voice_plan_sha256", "voice_plan_revision", "chapter_start", "chapter_end", "performance_mode"}
        required = {"mode", "voice_plan_sha256", "voice_plan_revision"}
        if set(body) - allowed or not required.issubset(body) or body.get("mode") != "interactive_voices":
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "interactive generation body is invalid"}}, status_code=422)
        if not isinstance(body.get("voice_plan_sha256"), str) or len(body["voice_plan_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in body["voice_plan_sha256"]):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "voice_plan_sha256 is invalid"}}, status_code=422)
        if type(body.get("voice_plan_revision")) is not int or body["voice_plan_revision"] <= 0:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "voice_plan_revision must be a positive integer"}}, status_code=422)
        performance_mode = body.get("performance_mode", "background")
        if not isinstance(performance_mode, str) or performance_mode not in {"background", "maximum_speed"}:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "performance_mode must be background or maximum_speed"}}, status_code=422)
        for key in ("chapter_start", "chapter_end"):
            if key in body and type(body[key]) is not int:
                return _chapter_error(ChapterPlanError("INVALID_CHAPTER_RANGE", "chapter range endpoints must be integers"))
        generation_acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not generation_acquired:
            return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "another generation start is in progress"}}, status_code=409)
        analysis_acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
        if not analysis_acquired:
            state.generation_lock.release()
            return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
        try:
            if state.voice_analysis_thread is not None and state.voice_analysis_thread.is_alive():
                return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
            try:
                conversion_id, manifest = await asyncio.to_thread(_voice_active_inputs, workspace)
                plan = await asyncio.to_thread(workspace.load_voice_plan, conversion_id)
                text, _ = await asyncio.to_thread(workspace.load_cleaned_artifacts, conversion_id)
                chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
                try:
                    scoped_analysis = await asyncio.to_thread(workspace.load_speaker_analysis, conversion_id)
                except (WorkspaceError, OSError):
                    # Existing approved plans may predate persisted speaker
                    # artifacts; those plans represent the complete book.
                    scoped_analysis = {}
                if plan["canonical_artifact_sha256"] != body["voice_plan_sha256"] or plan["revision"] != body["voice_plan_revision"] or plan["approval"]["state"] != "approved":
                    return JSONResponse({"error": {"code": "PLAN_CONFLICT", "message": "approved voice plan identity is stale"}}, status_code=409)
                if manifest.get("status") == "completed" and manifest.get("output") is not None:
                    return JSONResponse({"error": {"code": "ACTIVE_JOB", "message": "generation is complete"}}, status_code=409)
                if _voice_generation_live(manifest):
                    return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
                try:
                    for cast_entry in plan.get("cast", []):
                        require_enabled_voice_id(cast_entry.get("voice_id"))
                except (VoiceRegistryError, AttributeError):
                    return JSONResponse({"error": {"code": "INVALID_VOICE", "message": "voice plan contains an invalid voice"}}, status_code=422)
                start = body["chapter_start"] if "chapter_start" in body else 1
                end = body["chapter_end"] if "chapter_end" in body else len(chapter_plan["chapters"])
                select_chapter_range(chapter_plan, start, end)
                analyzed_start = scoped_analysis.get("chapter_start", 1)
                analyzed_end = scoped_analysis.get("chapter_end", len(chapter_plan["chapters"]))
                if start < analyzed_start or end > analyzed_end:
                    return JSONResponse({"error": {"code": "ANALYSIS_RANGE_CONFLICT", "message": "requested chapters are outside the analyzed range", "analyzed_chapter_start": analyzed_start, "analyzed_chapter_end": analyzed_end}}, status_code=409)
                narrator = next(item for item in plan["cast"] if item["cast_id"] == "narrator")
                voice_id = require_enabled_voice_id(narrator["voice_id"])
                facts = get_generation_facts(voice_id)
                speed = float(narrator["voice_settings"]["speed"])
                settings = SynthesisSettings(speed=speed, sample_rate=facts["sample_rate"], chunk_cap=DEFAULT_CHUNK_CAP)
                metadata = EngineMetadata(facts["engine"], facts["package_version"], facts["model"], facts["model_revision"], facts["model_checksum"], voice_id, facts["voice_version"], facts["voice_checksum"], facts["sample_rate"], settings.as_dict())
                chunks = plan_interactive_chunks(text, plan, get_generation_facts, registry_revision(), (start, end), cap=settings.chunk_cap)
                tts = {**metadata.as_dict(), "speed": speed, "chunk_cap": settings.chunk_cap}
                if start != 1 or end != len(chapter_plan["chapters"]):
                    tts["settings"] = {**tts["settings"], "chapter_start": start, "chapter_end": end}
                job = await asyncio.to_thread(workspace.configure_interactive_generation, conversion_id, tts=tts, total_chunks=len(chunks), voice_registry_revision=registry_revision())
                try:
                    process = state.worker_launcher(state.workspace_root, conversion_id, performance_mode)
                except Exception as exc:
                    message = "worker launch failed: " + type(exc).__name__
                    await asyncio.to_thread(workspace.update_generation, conversion_id, status="failed", stage="generation_start", error=message, last_safe_error=message, worker=None)
                    return JSONResponse({"error": {"code": "WORKER_START_FAILED", "message": message}}, status_code=503)
                state.worker_process = process
                return {"conversion_id": conversion_id, "status": "starting", "job": job, "total_chunks": len(chunks)}
            except VoiceRegistryError:
                return JSONResponse({"error": {"code": "INVALID_VOICE", "message": "voice is invalid"}}, status_code=422)
            except (ValueError, StopIteration) as exc:
                return JSONResponse({"error": {"code": "INVALID_PLAN", "message": "voice plan cannot be generated"}}, status_code=409)
            except ChapterPlanError as exc:
                return _chapter_error(exc)
            except (WorkspaceError, ManifestError, OSError):
                return JSONResponse({"error": {"code": "INVALID_GENERATION", "message": "interactive generation is unavailable"}}, status_code=409)
        finally:
            state.analysis_lock.release()
            state.generation_lock.release()

    @app.post("/api/voice-analysis")
    async def voice_analysis_start(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or set(body) - {"mode", "chapter_start", "chapter_end"} or body.get("mode") != "interactive" or ("chapter_start" in body) != ("chapter_end" in body):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "mode must be interactive"}}, status_code=422)
        for key in ("chapter_start", "chapter_end"):
            if key in body and type(body[key]) is not int:
                return _chapter_error(ChapterPlanError("INVALID_CHAPTER_RANGE", "chapter range endpoints must be integers"))
        analyzer_binding = _voice_analyzer()
        if analyzer_binding is None:
            return JSONResponse({"error": {"code": "ANALYZER_UNAVAILABLE", "message": "voice analysis is unavailable"}}, status_code=503)
        analyzer, descriptor = analyzer_binding

        generation_acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not generation_acquired:
            return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation operation is in progress"}}, status_code=409)
        analysis_acquired = False
        try:
            current_thread = state.voice_analysis_thread
            if current_thread is not None and current_thread.is_alive():
                return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is already running"}}, status_code=409)
            if current_thread is not None:
                state.voice_analysis_thread = None
                state.voice_analysis_conversion_id = None
            analysis_acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
            if not analysis_acquired:
                return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis conflicts with another workspace operation"}}, status_code=409)
            workspace = Workspace(state.workspace_root)
            try:
                conversion_id, manifest = await asyncio.to_thread(_voice_active_inputs, workspace)
            except (WorkspaceError, OSError):
                return JSONResponse({"error": {"code": "INVALID_ACTIVE", "message": "an analyzed chapter plan is required"}}, status_code=409)
            if _voice_generation_live(manifest):
                return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
            try:
                chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
                start = body.get("chapter_start")
                end = body.get("chapter_end")
                select_chapter_range(chapter_plan, start, end)
                normalized_start = int(body.get("chapter_start") or 1)
                normalized_end = int(body.get("chapter_end") or len(chapter_plan["chapters"]))
            except ChapterPlanError as exc:
                return _chapter_error(exc)

            previous: dict[str, Any] | None = None
            try:
                previous = await asyncio.to_thread(workspace.load_voice_analysis_status, conversion_id)
            except (WorkspaceError, OSError):
                previous = None
            revision = previous["revision"] + 1 if previous is not None and type(previous.get("revision")) is int and previous["revision"] > 0 else 1
            analysis_id = str(uuid.uuid4())
            queued_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            queued_status = with_canonical_artifact_hash({
                "schema_version": 1,
                "artifact": "voice-analysis-status",
                "analysis_id": analysis_id,
                "revision": revision,
                "source_pdf_sha256": manifest["source_pdf_sha256"],
                "cleaned_text_sha256": manifest["cleaned_text_sha256"],
                "chapter_plan_sha256": manifest["chapter_plan_sha256"],
                "chapter_plan_schema_version": 1,
                "chapter_start": normalized_start,
                "chapter_end": normalized_end,
                "analyzer": descriptor.as_dict(),
                "status": "queued",
                "stage": "queued",
                "progress": {"completed": 0, "total": 0},
                "cancel_requested": False,
                "warnings": [],
                "error": None,
                "started_at": queued_at,
                "updated_at": queued_at,
                "finished_at": None,
            })
            try:
                await asyncio.to_thread(workspace.persist_voice_analysis_status, conversion_id, queued_status)
            except (WorkspaceError, OSError):
                return JSONResponse({"error": {"code": "ANALYSIS_START_FAILED", "message": "voice analysis could not start"}}, status_code=503)
            runner = VoiceAnalysisRunner(workspace, conversion_id, analyzer, descriptor, analysis_id, revision, {"chapter_start": normalized_start, "chapter_end": normalized_end})

            def run_analysis() -> None:
                try:
                    runner.run()
                except Exception:
                    # VoiceAnalysisRunner persists a sanitized terminal failure.
                    pass
                finally:
                    try:
                        state.analysis_lock.release()
                    finally:
                        if state.voice_analysis_thread is threading.current_thread():
                            state.voice_analysis_thread = None
                            state.voice_analysis_conversion_id = None

            thread = threading.Thread(target=run_analysis, name="voice-analysis", daemon=True)
            state.voice_analysis_thread = thread
            state.voice_analysis_conversion_id = conversion_id
            try:
                thread.start()
            except Exception:
                state.voice_analysis_thread = None
                state.voice_analysis_conversion_id = None
                state.analysis_lock.release()
                analysis_acquired = False
                return JSONResponse({"error": {"code": "ANALYSIS_START_FAILED", "message": "voice analysis could not start"}}, status_code=503)
            analysis_acquired = False
            return {
                "conversion_id": conversion_id,
                "analysis_id": analysis_id,
                "revision": revision,
                "status": "queued",
                "stage": "queued",
                "source_pdf_sha256": manifest.get("source_pdf_sha256"),
                "cleaned_text_sha256": manifest.get("cleaned_text_sha256"),
                "chapter_plan_sha256": manifest.get("chapter_plan_sha256"),
                "chapter_start": normalized_start,
                "chapter_end": normalized_end,
            }
        finally:
            if analysis_acquired:
                state.analysis_lock.release()
            state.generation_lock.release()

    @app.get("/api/voice-analysis/status")
    async def voice_analysis_status(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        workspace = Workspace(state.workspace_root)
        try:
            conversion_id, _ = await asyncio.to_thread(_voice_active_inputs, workspace)
            status = await asyncio.to_thread(workspace.load_voice_analysis_status, conversion_id)
            chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
            cancel_requested = await asyncio.to_thread(workspace.voice_analysis_cancellation_requested, conversion_id)
        except (WorkspaceError, OSError):
            return JSONResponse({"error": {"code": "NO_ANALYSIS", "message": "voice analysis status is unavailable"}}, status_code=404)
        return _voice_status_projection(conversion_id, status, cancel_requested=cancel_requested or status["cancel_requested"], chapter_count=len(chapter_plan["chapters"]))

    @app.get("/api/speaker-analysis")
    async def speaker_analysis_review(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")

        allowed = {"chapter", "confidence", "offset", "limit"}
        values: dict[str, str] = {}
        for key, value in request.query_params.multi_items():
            if key not in allowed or key in values:
                return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "query parameters are invalid"}}, status_code=422)
            values[key] = value

        def _query_integer(name: str, *, minimum: int, maximum: int | None = None, default: int | None = None) -> int | None:
            value = values.get(name)
            if value is None:
                return default
            if not value or not value.isascii() or not value.isdigit():
                return None
            try:
                parsed = int(value)
            except ValueError:
                return None
            if parsed < minimum or (maximum is not None and parsed > maximum):
                return None
            return parsed

        chapter = _query_integer("chapter", minimum=1)
        offset = _query_integer("offset", minimum=0, default=0)
        limit = _query_integer("limit", minimum=1, maximum=200, default=50)
        confidence = values.get("confidence")
        if (("chapter" in values and chapter is None) or ("offset" in values and offset is None)
                or ("limit" in values and limit is None) or (confidence is not None and confidence not in {"high", "medium", "low"})):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "query parameters are invalid"}}, status_code=422)

        workspace = Workspace(state.workspace_root)
        try:
            conversion_id, _ = await asyncio.to_thread(_voice_active_inputs, workspace)
            artifact = await asyncio.to_thread(workspace.load_speaker_analysis, conversion_id)
            cleaned_text, _ = await asyncio.to_thread(workspace.load_cleaned_artifacts, conversion_id)
        except (WorkspaceError, OSError):
            return JSONResponse({"error": {"code": "SPEAKER_ANALYSIS_UNAVAILABLE", "message": "speaker analysis is unavailable"}}, status_code=404)

        filtered = [
            span for span in artifact["spans"]
            if (chapter is None or span["chapter_index"] == chapter)
            and (confidence is None or span["confidence"]["band"] == confidence)
        ]
        total = len(filtered)
        page = filtered[offset:offset + limit]
        spans: list[dict[str, Any]] = []
        for span in page:
            start, end = span["source_start"], span["source_end"]
            excerpt_end = min(end, start + 240)
            spans.append({
                "span_id": span["span_id"],
                "chapter_index": span["chapter_index"],
                "source_start": start,
                "source_end": end,
                "type": span["type"],
                "speaker_id": span["speaker_id"],
                "confidence": dict(span["confidence"]),
                "provenance": dict(span["provenance"]),
                "excerpt": cleaned_text[start:excerpt_end],
                "excerpt_truncated": excerpt_end < end,
            })
        warning_count = len(artifact["warnings"])
        return {
            "conversion_id": conversion_id,
            "revision": artifact["revision"],
            "canonical_artifact_sha256": artifact["canonical_artifact_sha256"],
            "source_pdf_sha256": artifact["source_pdf_sha256"],
            "cleaned_text_sha256": artifact["cleaned_text_sha256"],
            "chapter_plan_sha256": artifact["chapter_plan_sha256"],
            "analyzer": dict(artifact["analyzer"]),
            "warnings": list(artifact["warnings"][:50]),
            "warning_count": warning_count,
            "warnings_truncated": warning_count > 50,
            "character_count": len(artifact["characters"]),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
            "spans": spans,
        }

    @app.post("/api/voice-analysis/cancel")
    async def voice_analysis_cancel(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or set(body) != {"analysis_id", "revision"} or not isinstance(body.get("analysis_id"), str) or not body["analysis_id"] or type(body.get("revision")) is not int or body["revision"] <= 0:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "analysis_id and positive revision are required"}}, status_code=422)
        workspace = Workspace(state.workspace_root)
        try:
            conversion_id, _ = await asyncio.to_thread(_voice_active_inputs, workspace)
            status = await asyncio.to_thread(workspace.load_voice_analysis_status, conversion_id)
            chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
        except (WorkspaceError, OSError):
            return JSONResponse({"error": {"code": "NO_ANALYSIS", "message": "voice analysis status is unavailable"}}, status_code=409)
        if status["analysis_id"] != body["analysis_id"] or status["revision"] != body["revision"]:
            return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "analysis identity is stale"}}, status_code=409)
        if status["status"] not in {"queued", "running"}:
            return JSONResponse({"error": {"code": "NOT_ANALYZING", "message": "voice analysis is not cancellable"}}, status_code=409)
        try:
            await asyncio.to_thread(workspace.request_voice_analysis_cancel, conversion_id)
        except (WorkspaceError, OSError):
            return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis cancellation failed"}}, status_code=409)
        return _voice_status_projection(conversion_id, status, cancel_requested=True, chapter_count=len(chapter_plan["chapters"]))

    async def _acquire_plan_locks() -> tuple[bool, JSONResponse | None]:
        generation_acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not generation_acquired:
            return False, JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation operation is in progress"}}, status_code=409)
        analysis_acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
        if not analysis_acquired:
            state.generation_lock.release()
            return False, JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
        current = state.voice_analysis_thread
        if current is not None and current.is_alive():
            state.analysis_lock.release()
            state.generation_lock.release()
            return False, JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
        return True, None

    def _plan_identity(conversion_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return {
            "conversion_id": conversion_id,
            "revision": plan["revision"],
            "canonical_artifact_sha256": plan["canonical_artifact_sha256"],
            "approval": dict(plan["approval"]),
            "review": review_summary(plan),
        }

    async def _plan_context(workspace: Workspace) -> tuple[str, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
        conversion_id, manifest = await asyncio.to_thread(_voice_active_inputs, workspace)
        analysis = await asyncio.to_thread(workspace.load_speaker_analysis, conversion_id)
        cleaned_text, _ = await asyncio.to_thread(workspace.load_cleaned_artifacts, conversion_id)
        chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
        return conversion_id, manifest, analysis, cleaned_text, chapter_plan

    @app.post("/api/voice-plan/draft")
    async def voice_plan_draft(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or set(body) != {"analysis_revision"} or type(body.get("analysis_revision")) is not int or body["analysis_revision"] <= 0:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "analysis_revision must be a positive integer"}}, status_code=422)
        acquired, failure = await _acquire_plan_locks()
        if not acquired:
            return failure
        try:
            workspace = Workspace(state.workspace_root)
            try:
                conversion_id, manifest, analysis, cleaned_text, chapter_plan = await _plan_context(workspace)
                if analysis.get("revision") != body["analysis_revision"]:
                    return JSONResponse({"error": {"code": "PLAN_CONFLICT", "message": "speaker analysis revision is stale"}}, status_code=409)
                # Older/injected analyzer artifacts may omit the normalized
                # analysis revision from provenance; bind it at this API edge.
                analysis_for_plan = dict(analysis)
                normalized_spans: list[dict[str, Any]] = []
                for raw_span in analysis.get("spans", []):
                    span = dict(raw_span)
                    provenance = span.get("provenance")
                    if isinstance(provenance, dict):
                        span["provenance"] = {"source": str(provenance.get("source", "speaker-analysis")), "analysis_revision": analysis["revision"]}
                    else:
                        span["provenance"] = {"source": "speaker-analysis", "analysis_revision": analysis["revision"]}
                    normalized_spans.append(span)
                analysis_for_plan["spans"] = normalized_spans
                plan = build_voice_plan(
                    analysis_for_plan,
                    cleaned_text,
                    chapter_plan,
                    [entry["id"] for entry in list_public_entries() if entry.get("enabled") is True],
                    source_pdf_sha256=manifest["source_pdf_sha256"],
                )
                plan = await asyncio.to_thread(workspace.persist_voice_plan, conversion_id, plan)
            except VoicePlanError as exc:
                return _voice_plan_error(exc)
            except (WorkspaceError, OSError) as exc:
                return JSONResponse({"error": {"code": "PLAN_UNAVAILABLE", "message": "voice plan could not be created"}}, status_code=409)
            return _plan_identity(conversion_id, plan)
        finally:
            state.analysis_lock.release()
            state.generation_lock.release()

    @app.get("/api/voice-plan")
    async def voice_plan_read(request: Request):
        if not _authenticated(request, state):
            raise HTTPException(401, "authentication required")
        allowed = {"chapter", "confidence", "offset", "limit"}
        values: dict[str, str] = {}
        for key, value in request.query_params.multi_items():
            if key not in allowed or key in values:
                return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "query parameters are invalid"}}, status_code=422)
            values[key] = value
        def query_integer(name: str, minimum: int, maximum: int | None = None, default: int | None = None) -> int | None:
            value = values.get(name)
            if value is None:
                return default
            if not value or not value.isascii() or not value.isdigit():
                return None
            parsed = int(value)
            return parsed if parsed >= minimum and (maximum is None or parsed <= maximum) else None
        chapter = query_integer("chapter", 1)
        offset = query_integer("offset", 0, default=0)
        limit = query_integer("limit", 1, 200, 50)
        confidence = values.get("confidence")
        if (chapter is None and "chapter" in values) or (offset is None and "offset" in values) or (limit is None and "limit" in values) or (confidence is not None and confidence not in {"high", "medium", "low"}):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "query parameters are invalid"}}, status_code=422)
        workspace = Workspace(state.workspace_root)
        try:
            conversion_id, _ = await asyncio.to_thread(_voice_active_inputs, workspace)
            plan = await asyncio.to_thread(workspace.load_voice_plan, conversion_id)
            cleaned_text, _ = await asyncio.to_thread(workspace.load_cleaned_artifacts, conversion_id)
        except (WorkspaceError, OSError):
            return JSONResponse({"error": {"code": "PLAN_UNAVAILABLE", "message": "voice plan is unavailable"}}, status_code=404)
        cast = list(plan.get("cast", []))
        aliases = list(plan.get("aliases", []))
        chapters = list(plan.get("chapters", []))
        all_spans = [(item.get("chapter_index"), span) for item in chapters if isinstance(item, dict) for span in item.get("spans", []) if isinstance(span, dict)]
        filtered = [(chapter_index, span) for chapter_index, span in all_spans if (chapter is None or chapter_index == chapter) and (confidence is None or span.get("confidence", {}).get("band") == confidence)]
        total = len(filtered)
        page = filtered[offset:offset + limit]
        spans: list[dict[str, Any]] = []
        for chapter_index, span in page:
            start, end = span["source_start"], span["source_end"]
            excerpt_end = min(end, start + 240)
            spans.append({key: span[key] for key in ("span_id", "source_start", "source_end", "type", "speaker_id", "confidence", "provenance", "override") if key in span} | {"chapter_index": chapter_index, "excerpt": cleaned_text[start:excerpt_end], "excerpt_truncated": excerpt_end < end})
        return {
            **_plan_identity(conversion_id, plan),
            "analysis": dict(plan["analyzer"]),
            "analyzer": dict(plan["analyzer"]),
            "cast": cast[:256], "cast_count": len(cast), "cast_truncated": len(cast) > 256,
            "aliases": aliases[:2048], "alias_count": len(aliases), "aliases_truncated": len(aliases) > 2048,
            "total": total, "offset": offset, "limit": limit, "has_more": offset + len(page) < total, "spans": spans,
        }

    async def _mutate_voice_plan(request: Request, operation: str):
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "request body must be a JSON object"}}, status_code=422)
        acquired, failure = await _acquire_plan_locks()
        if not acquired:
            return failure
        try:
            workspace = Workspace(state.workspace_root)
            try:
                conversion_id, manifest = await asyncio.to_thread(_voice_active_inputs, workspace)
                plan = await asyncio.to_thread(workspace.load_voice_plan, conversion_id)
                cleaned_text, _ = await asyncio.to_thread(workspace.load_cleaned_artifacts, conversion_id)
                chapter_plan = await asyncio.to_thread(workspace.load_chapter_plan, conversion_id)
                if manifest.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION and manifest.get("status") not in {"planned", "cancelled", "failed", "analyzed", "pending"}:
                    code = "ACTIVE_WORKER" if manifest.get("status") in {"synthesizing", "cancelling", "assembling", "encoding", "verifying", "publishing"} else "NOT_READY"
                    message = "a generation worker is already active" if code == "ACTIVE_WORKER" else "voice plan cannot be edited after generation completion"
                    return JSONResponse({"error": {"code": code, "message": message}}, status_code=409)
                if _voice_generation_live(manifest):
                    return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
                expected = body.get("expected_revision")
                if type(expected) is not int or expected <= 0:
                    return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "expected_revision must be a positive integer"}}, status_code=422)
                if plan.get("revision") != expected:
                    return JSONResponse({"error": {"code": "PLAN_CONFLICT", "message": "voice plan revision is stale", "expected_revision": expected, "actual_revision": plan.get("revision")}}, status_code=409)
                if operation == "edit":
                    allowed = {"expected_revision", "cast_id", "display_label", "voice_id", "speed", "relationship"}
                    if set(body) - allowed or set(body) <= {"expected_revision", "cast_id"} or not isinstance(body.get("cast_id"), str):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "cast edit body is invalid"}}, status_code=422)
                    if "voice_id" in body:
                        require_enabled_voice_id(body["voice_id"])
                    if "speed" in body and (isinstance(body["speed"], bool) or not isinstance(body["speed"], (int, float)) or not 0.5 <= float(body["speed"]) <= 2.0):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "speed is invalid"}}, status_code=422)
                    if "relationship" in body and body["relationship"] not in {"third_person", "same_as_narrator", "separate_from_narrator"}:
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "relationship is invalid"}}, status_code=422)
                    updates = {key: body[key] for key in ("voice_id", "speed", "relationship") if key in body}
                    if updates:
                        result = assign_cast(plan, body["cast_id"], expected_revision=expected, **updates)
                        if "display_label" in body:
                            if not isinstance(body["display_label"], str) or not body["display_label"] or len(body["display_label"]) > 512:
                                return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "display_label is invalid"}}, status_code=422)
                            match = next(item for item in result["cast"] if item["cast_id"] == body["cast_id"])
                            match["display_label"] = body["display_label"]
                            result = with_canonical_artifact_hash(result)
                    else:
                        result = rename_cast(plan, body["cast_id"], body["display_label"], expected_revision=expected)
                elif operation == "cast_remove":
                    if set(body) != {"expected_revision", "cast_id"} or not isinstance(body.get("cast_id"), str) or not body["cast_id"]:
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "cast removal body is invalid"}}, status_code=422)
                    result = remove_cast(plan, body["cast_id"], expected_revision=expected)
                elif operation == "cast_merge":
                    if set(body) != {"expected_revision", "source_cast_id", "target_cast_id"} or not all(isinstance(body.get(key), str) and body[key] for key in ("source_cast_id", "target_cast_id")):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "cast merge body is invalid"}}, status_code=422)
                    result = merge_cast(plan, body["source_cast_id"], body["target_cast_id"], expected_revision=expected)
                elif operation == "merge":
                    if set(body) != {"expected_revision", "target_character_id", "alias_ids"} or not isinstance(body.get("target_character_id"), str) or not isinstance(body.get("alias_ids"), list):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "alias merge body is invalid"}}, status_code=422)
                    result = merge_aliases(plan, body["target_character_id"], body["alias_ids"], expected_revision=expected)
                elif operation == "split":
                    allowed = {"expected_revision", "alias_ids", "target_character_id", "new_character_id", "display_label", "voice_id"}
                    if set(body) - allowed or "alias_ids" not in body or not isinstance(body["alias_ids"], list):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "alias split body is invalid"}}, status_code=422)
                    if "voice_id" in body:
                        require_enabled_voice_id(body["voice_id"])
                    result = split_aliases(plan, body["alias_ids"], expected_revision=expected, target_character_id=body.get("target_character_id"), new_character_id=body.get("new_character_id"), display_label=body.get("display_label"), voice_id=body.get("voice_id"))
                elif operation == "override":
                    if set(body) != {"expected_revision", "span_id", "kind", "to", "reason"} or not all(isinstance(body.get(key), str) and body[key] for key in ("span_id", "kind", "to", "reason")):
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "span override body is invalid"}}, status_code=422)
                    result = override_span(plan, body["span_id"], expected_revision=expected, kind=body["kind"], to=body["to"], reason=body["reason"])
                else:
                    if set(body) != {"expected_revision", "accept_narrator_fallback"} or type(body["accept_narrator_fallback"]) is not bool:
                        return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "approval body is invalid"}}, status_code=422)
                    result = approve_voice_plan(plan, cleaned_text, chapter_plan, expected_source_pdf_sha256=manifest["source_pdf_sha256"], expected_chapter_plan_sha256=manifest["chapter_plan_sha256"], accept_narrator_fallback=body["accept_narrator_fallback"])
                persisted = await asyncio.to_thread(workspace.persist_voice_plan, conversion_id, result)
            except VoiceRegistryError as exc:
                return JSONResponse({"error": {"code": "INVALID_VOICE", "message": str(exc)}}, status_code=422)
            except VoicePlanError as exc:
                return _voice_plan_error(exc)
            except (WorkspaceError, OSError):
                return JSONResponse({"error": {"code": "PLAN_UNAVAILABLE", "message": "voice plan is unavailable"}}, status_code=409)
            return _plan_identity(conversion_id, persisted)
        finally:
            state.analysis_lock.release()
            state.generation_lock.release()

    @app.put("/api/voice-plan")
    async def voice_plan_edit(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "edit")

    @app.post("/api/voice-plan/aliases/merge")
    async def voice_plan_merge(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "merge")

    @app.post("/api/voice-plan/cast/remove")
    async def voice_plan_cast_remove(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "cast_remove")

    @app.post("/api/voice-plan/cast/merge")
    async def voice_plan_cast_merge(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "cast_merge")

    @app.post("/api/voice-plan/aliases/split")
    async def voice_plan_split(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "split")

    @app.post("/api/voice-plan/spans/override")
    async def voice_plan_override(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "override")

    @app.post("/api/voice-plan/approve")
    async def voice_plan_approve(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        return await _mutate_voice_plan(request, "approve")

    @app.post("/api/generation/start")
    async def generation_start(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if isinstance(body, dict) and "mode" in body:
            return await _interactive_generation_start(body, Workspace(state.workspace_root))
        allowed = {"voice", "speed", "chapter_start", "chapter_end", "performance_mode"}
        if body is None or not {"voice", "speed"}.issubset(body) or not set(body).issubset(allowed):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "voice and speed are required"}}, status_code=422)
        voice, speed = body.get("voice"), body.get("speed")
        performance_mode = body.get("performance_mode", "background")
        if not isinstance(performance_mode, str) or performance_mode not in {"background", "maximum_speed"}:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "performance_mode must be background or maximum_speed"}}, status_code=422)
        if voice not in APPROVED_VOICES or isinstance(speed, bool) or type(speed) not in {int, float} or not 0.5 <= float(speed) <= 2.0:
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "voice or speed is invalid"}}, status_code=422)
        for key in ("chapter_start", "chapter_end"):
            if key in body and type(body[key]) is not int:
                return _chapter_error(ChapterPlanError("INVALID_CHAPTER_RANGE", "chapter range endpoints must be integers"))
        workspace = Workspace(state.workspace_root)
        generation_acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not generation_acquired:
            return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "another generation start is in progress"}}, status_code=409)
        analysis_acquired = await asyncio.to_thread(state.analysis_lock.acquire, False)
        if not analysis_acquired:
            state.generation_lock.release()
            return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
        current_analysis = state.voice_analysis_thread
        if current_analysis is not None and current_analysis.is_alive():
            state.analysis_lock.release()
            state.generation_lock.release()
            return JSONResponse({"error": {"code": "ANALYSIS_CONFLICT", "message": "voice analysis is in progress"}}, status_code=409)
        try:
            inspection = await asyncio.to_thread(workspace.inspect_startup)
            if inspection.state != "resumable" or not inspection.conversion_id or not inspection.manifest:
                return JSONResponse({"error": {"code": "INVALID_ACTIVE", "message": "an analyzed chapter plan is required"}}, status_code=409)
            manifest = inspection.manifest
            if manifest.get("status") == "completed" and manifest.get("output") is not None:
                return JSONResponse({"error": {"code": "ACTIVE_JOB", "message": "generation is complete"}}, status_code=409)
            recorded = manifest.get("worker") or {}
            recorded_live = bool(recorded and pid_is_alive(int(recorded.get("pid", 0))))
            if recorded_live:
                return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
            local_process = state.worker_process
            local_live = local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
            if local_live:
                return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "a generation worker is already active"}}, status_code=409)
            if manifest.get("status") == "completed" and manifest.get("stage") == "synthesis_complete" and manifest.get("output") is None:
                tts, total = await asyncio.to_thread(_generation_inputs, workspace, inspection.conversion_id, voice, float(speed), body.get("chapter_start"), body.get("chapter_end"))
                if tts != manifest.get("tts") or total != manifest.get("total_chunks"):
                    return JSONResponse({"error": {"code": "SETTINGS_CHANGED", "message": "synthesis-complete audio must be finalized with its recorded settings"}}, status_code=409)
                try:
                    process = state.worker_launcher(state.workspace_root, inspection.conversion_id, performance_mode)
                except Exception as exc:
                    message = "worker launch failed: " + type(exc).__name__
                    await asyncio.to_thread(workspace.update_generation, inspection.conversion_id, status="failed", stage="generation_start", error=message, last_safe_error=message, worker=None)
                    return JSONResponse({"error": {"code": "WORKER_START_FAILED", "message": message}}, status_code=503)
                state.worker_process = process
                return {"conversion_id": inspection.conversion_id, "status": "starting", "job": manifest, "total_chunks": manifest["total_chunks"]}
            tts, total = await asyncio.to_thread(_generation_inputs, workspace, inspection.conversion_id, voice, float(speed), body.get("chapter_start"), body.get("chapter_end"))
            job = await asyncio.to_thread(workspace.configure_generation, inspection.conversion_id, tts=tts, total_chunks=total)
            try:
                process = state.worker_launcher(state.workspace_root, inspection.conversion_id, performance_mode)
            except Exception as exc:
                message = "worker launch failed: " + type(exc).__name__
                await asyncio.to_thread(workspace.update_generation, inspection.conversion_id, status="failed", stage="generation_start", error=message, last_safe_error=message, worker=None)
                return JSONResponse({"error": {"code": "WORKER_START_FAILED", "message": message}}, status_code=503)
            state.worker_process = process
            return {"conversion_id": inspection.conversion_id, "status": "starting", "job": job, "total_chunks": total}
        except ChapterPlanError as exc:
            return _chapter_error(exc)
        except (WorkspaceError, ManifestError, OSError) as exc:
            return JSONResponse({"error": {"code": "INVALID_GENERATION", "message": str(exc)}}, status_code=409)
        finally:
            state.analysis_lock.release()
            state.generation_lock.release()

    @app.post("/api/generation/cancel")
    async def generation_cancel(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
        if body is None or (body != {} and (set(body) != {"conversion_id"} or type(body.get("conversion_id")) is not str or not body["conversion_id"])):
            return JSONResponse({"error": {"code": "INVALID_INPUT", "message": "cancel accepts an empty body or the active conversion_id"}}, status_code=422)
        workspace = Workspace(state.workspace_root)
        try:
            inspection = await asyncio.to_thread(workspace.inspect_startup)
            if inspection.state != "resumable" or not inspection.conversion_id or not inspection.manifest:
                return JSONResponse({"error": {"code": "NO_ACTIVE", "message": "no active generation"}}, status_code=409)
            if body and body.get("conversion_id") != inspection.conversion_id:
                return JSONResponse({"error": {"code": "INVALID_CONVERSION", "message": "conversion_id does not match the active conversion"}}, status_code=409)
            if inspection.manifest.get("schema_version") not in {GENERATION_SCHEMA_VERSION, INTERACTIVE_GENERATION_SCHEMA_VERSION}:
                return JSONResponse({"error": {"code": "NOT_GENERATING", "message": "generation has not started"}}, status_code=409)
            status = inspection.manifest.get("status")
            recorded = inspection.manifest.get("worker") or {}
            runtime_active_statuses = {"synthesizing", "cancelling", "assembling", "encoding", "verifying", "publishing"}
            active_statuses = {"planned", "starting", "synthesizing", "assembling", "encoding", "verifying", "publishing"}
            live = status in active_statuses | {"cancelling"} and pid_is_alive(int(recorded.get("pid", 0)))
            local_process = state.worker_process
            synthesis_preclaim = status == "completed" and inspection.manifest.get("stage") == "synthesis_complete" and inspection.manifest.get("output") is None
            local_live = (status in (active_statuses | {"cancelling", "cancelled", "failed"}) or synthesis_preclaim) and local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
            live = live or local_live
            if not live:
                if status in runtime_active_statuses:
                    await asyncio.to_thread(
                        workspace.update_generation,
                        inspection.conversion_id,
                        status="cancelled",
                        stage="cancelled",
                        worker=None,
                        last_safe_error="cancelled",
                    )
                    await asyncio.to_thread(workspace.clear_cancel_request, inspection.conversion_id)
                    return {"conversion_id": inspection.conversion_id, "status": "cancelled", "cancel_requested": False}
                return JSONResponse({"error": {"code": "NOT_GENERATING", "message": "generation is complete"}}, status_code=409)
            await asyncio.to_thread(workspace.request_cancel, inspection.conversion_id)
            return {"conversion_id": inspection.conversion_id, "status": "cancelling", "cancel_requested": True}
        except (WorkspaceError, ManifestError, OSError) as exc:
            return JSONResponse({"error": {"code": "INVALID_GENERATION", "message": str(exc)}}, status_code=409)

    @app.delete("/api/workspace/active")
    async def delete_active_workspace(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        if not state.analysis_lock.acquire(blocking=False):
            return JSONResponse({"error": {"code": "BUSY", "message": "An analysis is in progress."}}, status_code=409)
        try:
            deleted = await asyncio.to_thread(Workspace(state.workspace_root).delete_active_state)
        except (WorkspaceError, OSError) as exc:
            return JSONResponse({"error": {"code": "INVALID_ACTIVE", "message": str(exc)}}, status_code=422)
        finally:
            state.analysis_lock.release()
        return {"deleted": deleted}

    @app.delete("/api/workspace/{conversion_id}")
    async def delete_workspace(conversion_id: str, request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        if not state.analysis_lock.acquire(blocking=False):
            return JSONResponse({"error": {"code": "BUSY", "message": "An analysis is in progress."}}, status_code=409)
        try:
            deleted = await asyncio.to_thread(Workspace(state.workspace_root).delete_conversion, conversion_id)
        except (WorkspaceError, OSError) as exc:
            return JSONResponse({"error": {"code": "INVALID_CONVERSION", "message": str(exc)}}, status_code=422)
        finally:
            state.analysis_lock.release()
        return {"deleted": deleted}

    @app.post("/api/shutdown")
    async def shutdown(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        state.shutdown_event.set()
        if state.uvicorn_server is not None:
            state.uvicorn_server.should_exit = True
        remove_instance_if_matches(state.instance_file, launch_id=state.launch_id, pid=__import__("os").getpid(), token=state.session_token)
        return {"shutting_down": True}

    return app

"""Authenticated localhost application and Phase 4 generation routes."""

from __future__ import annotations

import hmac
import asyncio
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
from .pdf import MAX_PDF_BYTES, PdfAnalysisError, analyze_pdf, preflight_pdf
from .chapters import ChapterPlanError, create_chapter_plan, rename_chapters, select_chapter_range, validate_chapter_plan
from .tts import APPROVED_VOICES, DEFAULT_TORCH_THREADS, EngineMetadata, SynthesisSettings, TORCH_THREADS_ENV, plan_chunks
from .security import instance_path, pid_is_alive, remove_instance_if_matches, token_hash
from .workspace import GENERATION_SCHEMA_VERSION, ManifestError, Workspace, WorkspaceError

STATIC_DIR = Path(__file__).with_name("static")
COOKIE_NAME = "pdf_audiobook_session"


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
    worker_launcher: Any = None
    worker_process: Any = None
    uvicorn_server: Any = None
    preview_root: Path | None = None
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


def create_app(*, port: int, launch_id: str | None = None, session_token: str | None = None, instance_file: Path | None = None, data_root: Path | None = None, worker_launcher: Any | None = None, preview_root: Path | None = None, path_opener: Any | None = None) -> FastAPI:
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
        worker_launcher=worker_launcher or _spawn_worker,
        preview_root=preview_root,
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
        if voice not in APPROVED_VOICES or state.preview_root is None:
            return JSONResponse(unavailable, status_code=404)
        root = state.preview_root
        if not _safe_real_directory(root):
            return JSONResponse(unavailable, status_code=404)
        try:
            resolved_root = root.resolve()
            suffix = f"-kokoro-{voice}.wav"
            matches: list[tuple[int, str, Path]] = []
            for candidate in root.iterdir():
                if not candidate.name.endswith(suffix) or len(candidate.name) == len(suffix):
                    continue
                try:
                    resolved_candidate = candidate.resolve()
                    resolved_candidate.relative_to(resolved_root)
                    info = candidate.stat(follow_symlinks=False)
                except (OSError, ValueError):
                    continue
                if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not _is_reparse(candidate):
                    matches.append((info.st_mtime_ns, candidate.name, candidate))
            if not matches:
                return JSONResponse(unavailable, status_code=404)
            candidate = max(matches, key=lambda item: (item[0], item[1]))[2]
            validate_wav(candidate, expected_sample_rate=24000)
        except (OSError, ValueError):
            return JSONResponse(unavailable, status_code=404)
        return FileResponse(candidate, media_type="audio/wav", headers={"Cache-Control": "no-store"})

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
            manifest.get("schema_version") == GENERATION_SCHEMA_VERSION and status_value in {"cancelled", "failed"}
        ) or (
            status_value == "completed" and manifest.get("stage") == "synthesis_complete" and manifest.get("output") is None
        )
        inferred_starting = local_live and launchable_before_claim
        result: dict[str, Any] = {"state": "starting" if inferred_starting else status_value if status_value in {"synthesizing", "cancelling", "cancelled", "failed", "assembling", "encoding", "verifying", "publishing", "completed"} else ("analyzed" if status_value == "analyzed" else "resumable"), "conversion_id": inspection.conversion_id, "job": manifest}
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

    @app.post("/api/generation/start")
    async def generation_start(request: Request):
        if not _authenticated(request, state) or not _exact_origin(request, state.port):
            raise HTTPException(403, "authenticated exact local Origin required")
        body = await _json_body(request)
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
        acquired = await asyncio.to_thread(state.generation_lock.acquire, False)
        if not acquired:
            return JSONResponse({"error": {"code": "ACTIVE_WORKER", "message": "another generation start is in progress"}}, status_code=409)
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
            if inspection.manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
                return JSONResponse({"error": {"code": "NOT_GENERATING", "message": "generation has not started"}}, status_code=409)
            status = inspection.manifest.get("status")
            recorded = inspection.manifest.get("worker") or {}
            active_statuses = {"planned", "starting", "synthesizing", "assembling", "encoding", "verifying", "publishing"}
            live = status in active_statuses | {"cancelling"} and pid_is_alive(int(recorded.get("pid", 0)))
            local_process = state.worker_process
            synthesis_preclaim = status == "completed" and inspection.manifest.get("stage") == "synthesis_complete" and inspection.manifest.get("output") is None
            local_live = (status in (active_statuses | {"cancelled", "failed"}) or synthesis_preclaim) and local_process is not None and (not hasattr(local_process, "poll") or local_process.poll() is None)
            live = live or local_live
            if not live:
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

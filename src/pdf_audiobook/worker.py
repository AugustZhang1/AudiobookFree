"""One-process, resumable chunk synthesis worker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Callable

from .audio import validate_wav, write_pcm_wav
from .chapters import select_chapter_range
from .security import pid_is_alive
from .tts import EngineMetadata, SynthesisSettings, TextChunk, close_voice, load_voice, plan_chunks
from .workspace import ManifestError, Workspace, UnsafePathError
from .m4b import Phase5Cancelled, finalize_conversion

MAX_ATTEMPTS = 3


class CancellationRequested(Exception):
    pass


class WorkerBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerResult:
    status: str
    completed: int
    total: int
    attempts: int


class ConversionWorker:
    """Synthesize chunks with atomic progress after every successful chunk."""

    def __init__(self, workspace: Workspace, conversion_id: str, *, engine_factory: Callable[..., Any] = load_voice, max_attempts: int = MAX_ATTEMPTS):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.workspace = workspace
        self.conversion_id = conversion_id
        self.engine_factory = engine_factory
        self.max_attempts = max_attempts

    def _claim(self, job: dict[str, Any]) -> dict[str, Any]:
        worker = job.get("worker")
        if worker and worker.get("pid") != os.getpid() and pid_is_alive(worker.get("pid", -1)):
            raise WorkerBusyError("a conversion worker is already active")
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return self.workspace.update_generation(self.conversion_id, status="synthesizing", stage="synthesis", worker={"pid": os.getpid(), "started_at": now, "updated_at": now}, error=None, last_safe_error=None)

    def _claim_phase5(self, job: dict[str, Any]) -> dict[str, Any]:
        worker = job.get("worker")
        if worker and worker.get("pid") != os.getpid() and pid_is_alive(worker.get("pid", -1)):
            raise WorkerBusyError("a conversion worker is already active")
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        started = worker.get("started_at", now) if worker else now
        return self.workspace.update_generation(self.conversion_id, status="assembling", stage="assembling", worker={"pid": os.getpid(), "started_at": started, "updated_at": now}, error=None, last_safe_error=None)

    def _planned_chunks(self, job: dict[str, Any]) -> list[TextChunk]:
        text, _ = self.workspace.load_cleaned_artifacts(self.conversion_id)
        plan = self.workspace.load_chapter_plan(self.conversion_id)
        tts = job["tts"]
        metadata = EngineMetadata(
            str(tts["engine"]), str(tts.get("package_version", "")), str(tts.get("model", tts.get("model_id", ""))),
            str(tts.get("model_revision", "")), str(tts.get("model_checksum", "")), str(tts["voice"]),
            str(tts.get("voice_version", "")), str(tts.get("voice_checksum", "")), int(tts["sample_rate"]),
            dict(tts.get("settings", {"speed": tts["speed"], "sample_rate": tts["sample_rate"], "chunk_cap": tts.get("chunk_cap", 900)})),
        )
        settings = metadata.settings
        chapters = select_chapter_range(plan, settings.get("chapter_start"), settings.get("chapter_end"))
        return plan_chunks(text, chapters, metadata, cap=int(settings.get("chunk_cap", 900)))

    def _chunk_path(self, chunk: TextChunk) -> Path:
        return self.workspace.chunks_path(self.conversion_id) / f"chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"

    def _existing(self, job: dict[str, Any], chunk: TextChunk) -> dict[str, Any] | None:
        expected_path = self._chunk_path(chunk).relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
        for record in job["completed_chunks"]:
            if (
                record["chapter_index"] != chunk.chapter_index
                or record["local_index"] != chunk.local_index
                or record["global_index"] != chunk.global_index
                or record["relative_path"] != expected_path
                or record["input_hash"] != chunk.input_hash
            ):
                continue
            path = self.workspace.conversion_path(self.conversion_id) / record["relative_path"]
            try:
                info = validate_wav(path, expected_sample_rate=job["tts"]["sample_rate"])
            except (OSError, ValueError):
                return None
            if info.duration_seconds <= 0:
                return None
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if record.get("wav_sha256") not in {None, "0" * 64, digest}:
                return None
            return {**record, "duration_seconds": info.duration_seconds, "wav_sha256": digest}
        return None

    def run(self, *, engine: Any | None = None, full_pipeline: bool | None = None) -> WorkerResult:
        if full_pipeline is None:
            full_pipeline = engine is None
        job = self.workspace.read_job(self.conversion_id)
        if job.get("schema_version") != 4:
            raise ManifestError("job must be configured for generation")
        chunks = self._planned_chunks(job)
        if len(chunks) != job["total_chunks"]:
            raise ManifestError("planned chunk count does not match manifest")
        if job.get("status") in {"assembling", "encoding", "verifying", "publishing"}:
            if not full_pipeline:
                return WorkerResult(job["status"], len(job["completed_chunks"]), len(chunks), 0)
            self._claim_phase5(job)
            try:
                finalize_conversion(self.workspace, self.conversion_id)
            except Phase5Cancelled:
                return WorkerResult("cancelled", len(job["completed_chunks"]), len(chunks), 0)
            return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
        if job.get("status") == "completed" and job.get("stage") == "synthesis_complete" and job.get("output") is None:
            if full_pipeline:
                self._claim_phase5(job)
                try:
                    finalize_conversion(self.workspace, self.conversion_id)
                except Phase5Cancelled:
                    return WorkerResult("cancelled", len(job["completed_chunks"]), len(chunks), 0)
                return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
            return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
        job = self._claim(job)
        attempts_total = 0
        loaded = engine
        try:
            if loaded is None:
                settings = SynthesisSettings(**{key: job["tts"].get("settings", {}).get(key, value) for key, value in {"speed": job["tts"]["speed"], "sample_rate": job["tts"]["sample_rate"], "chunk_cap": job["tts"].get("chunk_cap", 900), "chunk_mode": "legacy", "paragraph_pause_ms": 0, "sentence_pause_ms": 0}.items()})
                loaded = self.engine_factory(job["tts"]["voice"], settings, engine=job["tts"]["engine"])
            completed = {record["global_index"]: record for record in job["completed_chunks"]}
            for chunk in chunks:
                if self.workspace.cancellation_requested(self.conversion_id):
                    raise CancellationRequested
                reused = self._existing(job, chunk)
                if reused is not None:
                    completed[chunk.global_index] = reused
                    continue
                path = self._chunk_path(chunk)
                # A mismatched regular file is replaced only after inspection;
                # unsafe links are rejected by write_pcm_wav.
                pcm = None
                error: Exception | None = None
                for attempt in range(self.max_attempts):
                    attempts_total += 1
                    try:
                        pcm = loaded.synthesize(chunk.text)
                        error = None
                        break
                    except Exception as exc:  # bounded, factual retry
                        error = exc
                if pcm is None:
                    message = f"chunk {chunk.global_index} failed after {self.max_attempts} attempts"
                    self.workspace.update_generation(self.conversion_id, status="failed", stage="synthesis", error=message, last_safe_error=message, worker=None)
                    raise RuntimeError(message) from error
                info = write_pcm_wav(path, pcm, int(job["tts"]["sample_rate"]), overwrite=True)
                relative = path.relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
                completed[chunk.global_index] = {**chunk.manifest_record(relative, info.duration_seconds), "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                ordered = [completed[index] for index in sorted(completed)]
                job = self.workspace.update_generation(self.conversion_id, status="synthesizing", stage="synthesis", completed_chunks=ordered, progress={"completed": len(ordered), "current": chunk.global_index + 1, "total": len(chunks)}, worker={"pid": os.getpid(), "started_at": job["worker"]["started_at"], "updated_at": job["updated_at"]})
            final_records = [completed[index] for index in sorted(completed)]
            self.workspace.update_generation(self.conversion_id, status="completed", stage="synthesis_complete", completed_chunks=final_records, progress={"completed": len(final_records), "current": len(chunks), "total": len(chunks)}, worker=None, output=None)
            if full_pipeline:
                self._claim_phase5(self.workspace.read_job(self.conversion_id))
                try:
                    finalize_conversion(self.workspace, self.conversion_id)
                except Phase5Cancelled:
                    return WorkerResult("cancelled", len(final_records), len(chunks), attempts_total)
            return WorkerResult("completed", len(final_records), len(chunks), attempts_total)
        except CancellationRequested:
            self.workspace.update_generation(self.conversion_id, status="cancelled", stage="cancelled", worker=None, last_safe_error="cancelled")
            return WorkerResult("cancelled", len(self.workspace.read_job(self.conversion_id)["completed_chunks"]), len(chunks), attempts_total)
        except Exception as exc:
            message = "worker failed: " + type(exc).__name__
            try:
                current = self.workspace.read_job(self.conversion_id)
                if current.get("stage") not in {"assembling", "encoding", "verifying", "publishing", "phase5"}:
                    self.workspace.update_generation(self.conversion_id, status="failed", stage="synthesis", error=message, last_safe_error=message, worker=None)
            except Exception:
                pass
            raise
        finally:
            if loaded is not None:
                try:
                    close_voice(loaded)
                except Exception:
                    pass


def run_worker(workspace_root: str | Path, conversion_id: str) -> WorkerResult:
    return ConversionWorker(Workspace(Path(workspace_root)), conversion_id).run(full_pipeline=True)


__all__ = ["MAX_ATTEMPTS", "CancellationRequested", "ConversionWorker", "WorkerBusyError", "WorkerResult", "run_worker"]

if __name__ == "__main__":  # pragma: no cover - exercised by the launcher
    import argparse
    parser = argparse.ArgumentParser(description="PDF audiobook conversion worker")
    parser.add_argument("workspace_root")
    parser.add_argument("conversion_id")
    args = parser.parse_args()
    raise SystemExit(0 if run_worker(args.workspace_root, args.conversion_id).status == "completed" else 2)

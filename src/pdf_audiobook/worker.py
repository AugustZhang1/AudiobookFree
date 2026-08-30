"""One-process, resumable chunk synthesis worker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable
import uuid

from .audio import validate_wav, write_pcm_wav
from .chapters import select_chapter_range
from .security import PID_ALIVE, PID_DEAD, pid_is_alive, pid_liveness
from .tts import CHATTERBOX_BUILTIN_VOICE, CHATTERBOX_CHUNK_CAP, CHATTERBOX_NANO_MODEL, CHATTERBOX_REFERENCE_VOICE, CHATTERBOX_SAMPLE_RATE, CHATTERBOX_SOURCE_COMMIT, EngineMetadata, InteractiveTextChunk, SynthesisSettings, TextChunk, close_voice, derive_chunk_plan_inputs, load_voice, plan_chunks, plan_interactive_chunks
from .voice_shaping import shape_pcm, shaping_fingerprint
from .voice_settings import canonical_voice_settings
from .voice_registry import get_generation_facts, registry_revision
from .workspace import ManifestError, Workspace, UnsafePathError
from .m4b import Phase5Cancelled, finalize_conversion

MAX_ATTEMPTS = 3
VOICE_CACHE_CAP = 2
WORKER_LOCK_FILENAME = ".conversion-worker.lock"
_WORKER_LOCK_RETRIES = 8
_WORKER_LOCK_RETRY_DELAY_SECONDS = 0.05
RETRY_BACKOFF_SECONDS = 0.25
_WORKER_LOCK_STALE_SECONDS = 300.0
_WORKER_LOCK_HEARTBEAT_SECONDS = 10.0


@dataclass
class _WorkerLease:
    path: Path
    fd: int
    token: str
    stop: threading.Event
    heartbeat_thread: threading.Thread | None = None

    def start_heartbeat(self) -> None:
        self.heartbeat_thread = threading.Thread(target=self._heartbeat, name="conversion-worker-lock-heartbeat", daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat(self) -> None:
        while not self.stop.wait(_WORKER_LOCK_HEARTBEAT_SECONDS):
            try:
                os.utime(self.path, None)
            except OSError:
                return

    def release(self) -> None:
        self.stop.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=max(1.0, _WORKER_LOCK_HEARTBEAT_SECONDS))
        owned = False
        try:
            info = self.path.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                marker = json.loads(self.path.read_text(encoding="utf-8"))
                owned = isinstance(marker, dict) and marker.get("pid") == os.getpid() and marker.get("token") == self.token
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            pass
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass
        if owned:
            try:
                self.path.unlink()
            except OSError:
                pass


def _validate_captured_voice_facts(tts: dict[str, Any], facts: dict[str, Any], voice_id: str) -> None:
    """Reject registry drift before a voice can produce audio."""

    if not isinstance(facts, dict) or facts.get("id") != voice_id or facts.get("enabled") is not True:
        raise RuntimeError("voice facts do not match captured generation")
    for field in ("engine", "package_version", "model", "model_revision", "model_checksum", "sample_rate"):
        if facts.get(field) != tts.get(field):
            raise RuntimeError("voice facts do not match captured generation")
    if type(facts.get("sample_rate")) is not int or facts["sample_rate"] != tts.get("sample_rate"):
        raise RuntimeError("voice sample rate does not match captured generation")
    # Voice version/checksum are voice-specific.  The captured top-level voice
    # metadata binds those fields only for that voice; other cast voices share
    # the model/engine/sample-rate capture.
    if voice_id == tts.get("voice") and any(facts.get(field) != tts.get(field) for field in ("voice_version", "voice_checksum")):
        raise RuntimeError("voice facts do not match captured generation")


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
        self.retry_backoff_seconds = RETRY_BACKOFF_SECONDS

    @staticmethod
    def _timestamp() -> str:
        return __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _worker_lock_is_stale(self, path: Path) -> bool:
        try:
            info = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return False
        age = max(0.0, time.time() - info.st_mtime)
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            pid = marker.get("pid") if isinstance(marker, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            pid = None
        if isinstance(pid, int) and pid > 0:
            liveness = pid_liveness(pid)
            if liveness == PID_DEAD:
                return True
            if liveness == PID_ALIVE:
                # Never steal a lock from a confirmed-live worker: a heartbeat
                # delayed by long synthesis or a suspended process would
                # otherwise let a second worker overwrite the same chunks.
                return False
        # The owner cannot be queried (Windows ACCESS_DENIED). Only the bounded
        # lease can free it, or an inaccessible crashed owner wedges this
        # conversion forever.
        return age >= _WORKER_LOCK_STALE_SECONDS

    def _acquire_worker_lock(self) -> _WorkerLease:
        path = self.workspace.conversion_path(self.conversion_id) / WORKER_LOCK_FILENAME
        token = uuid.uuid4().hex
        marker = {"pid": os.getpid(), "started_at": self._timestamp(), "token": token}
        encoded = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        retries = max(1, _WORKER_LOCK_RETRIES)
        reclaimed_stale = False
        self._reclaimed_stale_lock = False
        for attempt in range(retries):
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if self._worker_lock_is_stale(path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
                    else:
                        # Note that THIS attempt removed a stale lock.  The flag is
                        # only promoted to the instance once the following O_EXCL
                        # acquisition actually wins, so a worker that loses the race
                        # cannot carry a bypass it did not earn.
                        reclaimed_stale = True
                    continue
                if attempt + 1 >= retries:
                    raise WorkerBusyError("a conversion worker is already active")
                time.sleep(_WORKER_LOCK_RETRY_DELAY_SECONDS)
            except PermissionError:
                if attempt + 1 >= retries:
                    raise WorkerBusyError("a conversion worker lock is unavailable")
                time.sleep(_WORKER_LOCK_RETRY_DELAY_SECONDS)
            else:
                try:
                    written = 0
                    while written < len(encoded):
                        written += os.write(fd, encoded[written:])
                    os.fsync(fd)
                except Exception:
                    os.close(fd)
                    path.unlink(missing_ok=True)
                    raise
                lease = _WorkerLease(path, fd, token, threading.Event())
                lease.start_heartbeat()
                self._reclaimed_stale_lock = reclaimed_stale
                return lease
        raise WorkerBusyError("a conversion worker is already active")

    def _retryable(self, error: Exception) -> bool:
        """Retrying a deterministic failure just burns identical model invocations."""

        return not isinstance(error, (ValueError, TypeError, KeyError, ManifestError))

    def _backoff(self, attempt: int) -> bool:
        """Pause between attempts. Returns False when no attempts remain."""

        if attempt + 1 >= self.max_attempts:
            return False
        if self.retry_backoff_seconds > 0:
            time.sleep(self.retry_backoff_seconds * (attempt + 1))
        return True

    def _refuse_if_busy(self, job: dict[str, Any]) -> None:
        # The exclusive conversion lease is the authority on ownership.  Once this
        # process holds it, a recorded PID belonging to a dead-but-inaccessible
        # owner (pid_is_alive() reports True on Windows ACCESS_DENIED) must not be
        # able to wedge the conversion.
        if self._holds_conversion_lease():
            return
        worker = job.get("worker")
        if worker and worker.get("pid") != os.getpid() and pid_is_alive(worker.get("pid", -1)):
            raise WorkerBusyError("a conversion worker is already active")

    def _holds_conversion_lease(self) -> bool:
        return getattr(self, "_lease", None) is not None and getattr(self, "_reclaimed_stale_lock", False)

    def _claim(self, job: dict[str, Any]) -> dict[str, Any]:
        self._refuse_if_busy(job)
        now = self._timestamp()
        # Only a fresh claim after configure/resume may explicitly leave a
        # cancelled manifest.  All progress, completion, and error writes use
        # the ordinary terminal-cancellation behavior.
        return self.workspace.update_generation(self.conversion_id, allow_cancelled_resume=job.get("status") == "cancelled", status="synthesizing", stage="synthesis", worker={"pid": os.getpid(), "started_at": now, "updated_at": now}, error=None, last_safe_error=None)

    def _claim_phase5(self, job: dict[str, Any]) -> dict[str, Any]:
        self._refuse_if_busy(job)
        worker = job.get("worker")
        now = self._timestamp()
        started = worker.get("started_at", now) if worker else now
        return self.workspace.update_generation(self.conversion_id, status="assembling", stage="assembling", worker={"pid": os.getpid(), "started_at": started, "updated_at": now}, error=None, last_safe_error=None)

    def _planned_chunks(self, job: dict[str, Any]) -> list[TextChunk | InteractiveTextChunk]:
        if job.get("schema_version") == 5:
            return self._planned_interactive_chunks(job)
        text, _ = self.workspace.load_cleaned_artifacts(self.conversion_id)
        plan = self.workspace.load_chapter_plan(self.conversion_id)
        tts = job["tts"]
        plan_inputs = derive_chunk_plan_inputs(tts, interactive=False)
        metadata = EngineMetadata(
            str(tts["engine"]), str(tts.get("package_version", "")), plan_inputs.model,
            str(tts.get("model_revision", "")), str(tts.get("model_checksum", "")), str(tts["voice"]),
            str(tts.get("voice_version", "")), str(tts.get("voice_checksum", "")), int(tts["sample_rate"]),
            plan_inputs.settings,
        )
        settings = plan_inputs.settings
        chapters = select_chapter_range(plan, settings.get("chapter_start"), settings.get("chapter_end"))
        return plan_chunks(text, chapters, metadata, cap=plan_inputs.cap)

    def _planned_interactive_chunks(self, job: dict[str, Any]) -> list[InteractiveTextChunk]:
        text, _ = self.workspace.load_cleaned_artifacts(self.conversion_id)
        plan = self.workspace.load_voice_plan(self.conversion_id)
        analysis = self.workspace.load_speaker_analysis(self.conversion_id)
        approval = plan.get("approval")
        if not isinstance(approval, dict) or approval.get("state") != "approved":
            raise ManifestError("interactive generation requires an approved voice plan")
        if plan.get("canonical_artifact_sha256") != job.get("voice_plan_sha256") or plan.get("revision") != job.get("voice_plan_revision"):
            raise ManifestError("voice plan does not match generation manifest")
        if analysis.get("canonical_artifact_sha256") != job.get("speaker_analysis_sha256"):
            raise ManifestError("speaker analysis does not match generation manifest")
        current_registry_revision = registry_revision()
        if current_registry_revision != job.get("voice_registry_revision"):
            raise ManifestError("voice registry revision does not match generation manifest")
        tts = job["tts"]
        plan_inputs = derive_chunk_plan_inputs(tts, interactive=True)
        settings = plan_inputs.settings
        chapter_start, chapter_end = settings.get("chapter_start"), settings.get("chapter_end")
        chapter_range = None
        if chapter_start is not None or chapter_end is not None:
            if type(chapter_start) is not int or type(chapter_end) is not int:
                raise ManifestError("captured chapter range is invalid")
            chapter_range = (chapter_start, chapter_end)
        return plan_interactive_chunks(
            text,
            plan,
            get_generation_facts,
            current_registry_revision,
            chapter_range=chapter_range,
            cap=plan_inputs.cap,
        )

    def _chunk_path(self, chunk: TextChunk) -> Path:
        return self.workspace.chunks_path(self.conversion_id) / f"chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"

    def _existing(self, job: dict[str, Any], chunk: TextChunk, completed_by_index: dict[int, dict[str, Any]] | None = None) -> dict[str, Any] | None:
        expected_path = self._chunk_path(chunk).relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
        if completed_by_index is None:
            records = job["completed_chunks"]
        else:
            record = completed_by_index.get(chunk.global_index)
            records = (record,) if record is not None else ()
        for record in records:
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

    def _existing_interactive(self, job: dict[str, Any], chunk: InteractiveTextChunk) -> dict[str, Any] | None:
        expected_path = self._chunk_path(chunk).relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
        for record in job["completed_chunks"]:
            if (
                record["chapter_index"] != chunk.chapter_index
                or record["local_index"] != chunk.local_index
                or record["global_index"] != chunk.global_index
                or record["relative_path"] != expected_path
                or record.get("audio_input_hash") != chunk.audio_input_hash
            ):
                continue
            path = self.workspace.conversion_path(self.conversion_id) / expected_path
            try:
                info = validate_wav(path, expected_sample_rate=job["tts"]["sample_rate"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                return None
            if record.get("wav_sha256") != digest or info.duration_seconds <= 0:
                return None
            # Always project the current planner metadata onto a semantically
            # reusable WAV.  This drops stale plan/input metadata while
            # retaining the verified audio and its measured duration.
            return {**chunk.manifest_record(expected_path, info.duration_seconds), "wav_sha256": digest}
        return None

    def _sanitize_interactive(self, job: dict[str, Any], chunks: list[InteractiveTextChunk]) -> dict[str, Any]:
        verified: list[dict[str, Any]] = []
        for chunk in chunks:
            candidate = self._existing_interactive(job, chunk)
            if candidate is not None:
                verified.append(candidate)
        if verified == job["completed_chunks"] and job["progress"]["completed"] == len(verified):
            return job
        current = verified[-1]["global_index"] + 1 if verified else 0
        # status/stage move with the truncation: a resume out of a completed or
        # phase-5 state must not leave "completed" over a shortened chunk list.
        return self.workspace.update_generation(
            self.conversion_id,
            status="synthesizing",
            stage="synthesis",
            completed_chunks=verified,
            progress={"completed": len(verified), "current": current, "total": len(chunks)},
        )

    def _phase5_plan_matches(self, job: dict[str, Any], chunks: list[Any], *, interactive: bool) -> bool:
        """Report whether the recorded chunks still equal the freshly planned identity."""

        records = job["completed_chunks"]
        if len(records) != len(chunks):
            return False
        for record, chunk in zip(records, chunks):
            expected_path = self._chunk_path(chunk).relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
            if record.get("global_index") != chunk.global_index:
                return False
            if record.get("chapter_index") != chunk.chapter_index:
                return False
            if record.get("local_index") != chunk.local_index:
                return False
            if record.get("input_hash") != chunk.input_hash:
                return False
            if record.get("relative_path") != expected_path:
                return False
            if interactive and any(record.get(field) != getattr(chunk, field) for field in ("audio_input_hash", "span_id", "speaker_id", "voice_id", "segment_type", "source_start", "source_end")):
                return False
        return True

    def _run_v4(self, job: dict[str, Any], *, engine: Any | None = None, full_pipeline: bool = True) -> WorkerResult:
        """Run the legacy schema-v4 flow without interactive voice behavior."""

        if job.get("schema_version") != 4:
            raise ManifestError("job must be configured for generation")
        chunks = self._planned_chunks(job)
        if len(chunks) != job["total_chunks"]:
            raise ManifestError("planned chunk count does not match manifest")
        # A phase-5 resume is only safe while the recorded chunks still match the
        # planner; otherwise fall through and regenerate against the new identity.
        resumable = self._phase5_plan_matches(job, chunks, interactive=False)
        if resumable and job.get("status") in {"assembling", "encoding", "verifying", "publishing"}:
            if not full_pipeline:
                return WorkerResult(job["status"], len(job["completed_chunks"]), len(chunks), 0)
            self._claim_phase5(job)
            try:
                finalize_conversion(self.workspace, self.conversion_id)
            except Phase5Cancelled:
                return WorkerResult("cancelled", len(job["completed_chunks"]), len(chunks), 0)
            return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
        if resumable and job.get("status") == "completed" and job.get("stage") == "synthesis_complete" and job.get("output") is None:
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
        settings = SynthesisSettings(**{key: job["tts"].get("settings", {}).get(key, value) for key, value in {"speed": job["tts"]["speed"], "pitch_semitones": 0, "tone_preset": "neutral", "sample_rate": job["tts"]["sample_rate"], "chunk_cap": job["tts"].get("chunk_cap", 900), "chunk_mode": "legacy", "paragraph_pause_ms": 0, "sentence_pause_ms": 0}.items()})
        reference_wav: Path | None = None
        if job["tts"].get("engine") == "chatterbox":
            if (settings.speed != 1.0 or settings.chunk_cap != CHATTERBOX_CHUNK_CAP or job["tts"].get("voice") not in {CHATTERBOX_BUILTIN_VOICE, CHATTERBOX_REFERENCE_VOICE} or job["tts"].get("model") != CHATTERBOX_NANO_MODEL or job["tts"].get("model_revision") != CHATTERBOX_SOURCE_COMMIT or job["tts"].get("model_checksum") != "unrecorded" or job["tts"].get("sample_rate") != CHATTERBOX_SAMPLE_RATE):
                raise ManifestError("Chatterbox generation settings are invalid")
            if job["tts"].get("voice") == CHATTERBOX_BUILTIN_VOICE and (job["tts"].get("voice_version") != "bundled" or job["tts"].get("voice_checksum") != "unrecorded" or "reference_descriptor_sha256" in (job["tts"].get("settings") or {})):
                raise ManifestError("Chatterbox built-in voice identity is invalid")
            if job["tts"].get("voice") == "reference-wav":
                try:
                    reference = self.workspace.load_chatterbox_reference(self.conversion_id)
                except (ManifestError, UnsafePathError) as exc:
                    raise ManifestError("Chatterbox reference is missing or invalid") from exc
                tts = job["tts"]
                descriptor = reference.descriptor
                if any(tts.get(field) != descriptor.get(expected) for field, expected in (("model", "model"), ("model_revision", "model_revision"), ("model_checksum", "model_checksum"), ("voice_checksum", "voice_checksum"), ("voice_version", "voice_version"))):
                    raise ManifestError("Chatterbox reference identity does not match generation")
                if (tts.get("settings") or {}).get("reference_descriptor_sha256") != descriptor["descriptor_sha256"]:
                    raise ManifestError("Chatterbox reference descriptor does not match generation")
                reference_wav = reference.path
        try:
            if loaded is None:
                if job["tts"].get("engine") == "chatterbox" and job["tts"].get("voice") == "reference-wav":
                    loaded = self.engine_factory(job["tts"]["voice"], settings, engine=job["tts"]["engine"], reference_wav=reference_wav)
                else:
                    loaded = self.engine_factory(job["tts"]["voice"], settings, engine=job["tts"]["engine"])
            completed_by_index = {record["global_index"]: record for record in job["completed_chunks"]}
            completed: dict[int, dict[str, Any]] = {}
            for chunk in chunks:
                if self.workspace.cancellation_requested(self.conversion_id):
                    raise CancellationRequested
                reused = self._existing(job, chunk, completed_by_index)
                if reused is not None:
                    completed[chunk.global_index] = reused
            for chunk in chunks:
                if self.workspace.cancellation_requested(self.conversion_id):
                    raise CancellationRequested
                if chunk.global_index in completed:
                    continue
                path = self._chunk_path(chunk)
                # A mismatched regular file is replaced only after inspection;
                # unsafe links are rejected by write_pcm_wav.
                pcm = None
                error: Exception | None = None
                for attempt in range(self.max_attempts):
                    attempts_total += 1
                    try:
                        pcm = shape_pcm(loaded.synthesize(chunk.text), settings.sample_rate, settings.as_dict())
                        error = None
                        break
                    except Exception as exc:  # bounded, factual retry
                        error = exc
                        if not self._retryable(exc) or not self._backoff(attempt):
                            break
                if pcm is None:
                    message = f"chunk {chunk.global_index} failed after {self.max_attempts} attempts"
                    self.workspace.update_generation(self.conversion_id, status="failed", stage="synthesis", error=message, last_safe_error=message, worker=None)
                    raise RuntimeError(message) from error
                info = write_pcm_wav(path, pcm, int(job["tts"]["sample_rate"]), overwrite=True)
                relative = path.relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
                completed[chunk.global_index] = {**chunk.manifest_record(relative, info.duration_seconds), "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                ordered = [completed[index] for index in sorted(completed)]
                job = self.workspace.update_generation(self.conversion_id, status="synthesizing", stage="synthesis", completed_chunks=ordered, progress={"completed": len(ordered), "current": chunk.global_index + 1, "total": len(chunks)}, worker={"pid": os.getpid(), "started_at": job["worker"]["started_at"], "updated_at": self._timestamp()})
                if job["status"] == "cancelled":
                    raise CancellationRequested
                if self.workspace.cancellation_requested(self.conversion_id):
                    raise CancellationRequested
            final_records = [completed[index] for index in sorted(completed)]
            final_job = self.workspace.update_generation(self.conversion_id, status="completed", stage="synthesis_complete", completed_chunks=final_records, progress={"completed": len(final_records), "current": len(chunks), "total": len(chunks)}, worker=None, output=None)
            if final_job["status"] == "cancelled":
                return WorkerResult("cancelled", len(final_job["completed_chunks"]), len(chunks), attempts_total)
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
                if current.get("status") != "failed" and current.get("stage") not in {"assembling", "encoding", "verifying", "publishing", "phase5"}:
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

    def _run_v5(self, job: dict[str, Any], *, full_pipeline: bool = True) -> WorkerResult:
        chunks = self._planned_chunks(job)
        if len(chunks) != job["total_chunks"]:
            raise ManifestError("planned chunk count does not match manifest")
        # A phase-5 resume is only safe while the recorded chunks still match the
        # planner; otherwise fall through and regenerate against the new identity.
        resumable = self._phase5_plan_matches(job, chunks, interactive=True)
        if resumable and job.get("status") in {"assembling", "encoding", "verifying", "publishing"}:
            if not full_pipeline:
                return WorkerResult(job["status"], len(job["completed_chunks"]), len(chunks), 0)
            self._claim_phase5(job)
            try:
                finalize_conversion(self.workspace, self.conversion_id)
            except Phase5Cancelled:
                return WorkerResult("cancelled", len(job["completed_chunks"]), len(chunks), 0)
            return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
        if resumable and job.get("status") == "completed" and job.get("stage") == "synthesis_complete" and job.get("output") is None:
            if full_pipeline:
                self._claim_phase5(job)
                try:
                    finalize_conversion(self.workspace, self.conversion_id)
                except Phase5Cancelled:
                    return WorkerResult("cancelled", len(job["completed_chunks"]), len(chunks), 0)
                return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)
            return WorkerResult("completed", len(job["completed_chunks"]), len(chunks), 0)

        # Remove every unverified candidate before claiming the worker.  The
        # planner's semantic audio hash permits metadata-only reuse.
        voice_plan = self.workspace.load_voice_plan(self.conversion_id)
        captured_shaping = (job["tts"].get("settings") or {}).get("shaping_fingerprint")
        if captured_shaping is not None and captured_shaping != shaping_fingerprint():
            raise ManifestError("voice shaping capability does not match captured generation")
        cast_by_id = {entry["cast_id"]: entry for entry in voice_plan["cast"]}
        self._refuse_if_busy(job)
        job = self._sanitize_interactive(job, chunks)
        job = self._claim(job)
        attempts_total = 0
        completed = {record["global_index"]: record for record in job["completed_chunks"]}
        cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        closed: set[int] = set()

        def close_cached(loaded: Any) -> None:
            marker = id(loaded)
            if marker in closed:
                return
            closed.add(marker)
            try:
                close_voice(loaded)
            except Exception:
                pass

        def cancellation_check() -> None:
            if self.workspace.cancellation_requested(self.conversion_id):
                raise CancellationRequested

        def load_for(chunk: InteractiveTextChunk) -> Any:
            settings_data = job["tts"].get("settings", {})
            if not isinstance(settings_data, dict):
                settings_data = {}
            cast_entry = cast_by_id.get(chunk.speaker_id)
            if cast_entry is None or cast_entry.get("voice_id") != chunk.voice_id:
                raise ManifestError("interactive chunk speaker binding is invalid")
            cast_settings = canonical_voice_settings(cast_entry.get("voice_settings", {"speed": 1.0}))
            settings = SynthesisSettings(
                speed=float(cast_settings["speed"]),
                pitch_semitones=int(cast_settings["pitch_semitones"]),
                tone_preset=str(cast_settings["tone_preset"]),
                sample_rate=int(job["tts"]["sample_rate"]),
                chunk_cap=int(job["tts"].get("chunk_cap", 900)),
                chunk_mode=str(settings_data.get("chunk_mode", "chapter")),
                paragraph_pause_ms=int(settings_data.get("paragraph_pause_ms", 0)),
                sentence_pause_ms=int(settings_data.get("sentence_pause_ms", 0)),
            )
            key = (chunk.voice_id, tuple(sorted(settings.as_dict().items())))
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
            cancellation_check()
            facts = get_generation_facts(chunk.voice_id)
            _validate_captured_voice_facts(job["tts"], facts, chunk.voice_id)
            cancellation_check()
            loaded = self.engine_factory(chunk.voice_id, settings, engine=facts["engine"])
            try:
                cancellation_check()
                metadata = getattr(loaded, "metadata", None)
                if metadata is not None and getattr(metadata, "sample_rate", settings.sample_rate) != settings.sample_rate:
                    raise RuntimeError("loaded voice sample rate does not match captured generation")
                while len(cache) >= VOICE_CACHE_CAP:
                    cancellation_check()
                    _, evicted = cache.popitem(last=False)
                    close_cached(evicted)
                cache[key] = loaded
                return loaded
            except Exception:
                close_cached(loaded)
                raise

        try:
            for chunk in chunks:
                if not isinstance(chunk, InteractiveTextChunk):
                    raise ManifestError("interactive planner returned an invalid chunk")
                cancellation_check()
                reused = self._existing_interactive(job, chunk)
                if reused is not None:
                    completed[chunk.global_index] = reused
                    continue
                loaded = load_for(chunk)
                active_settings = SynthesisSettings(**canonical_voice_settings(cast_by_id[chunk.speaker_id].get("voice_settings", {"speed": 1.0})), sample_rate=int(job["tts"]["sample_rate"]), chunk_cap=int(job["tts"].get("chunk_cap", 900)))
                path = self._chunk_path(chunk)
                pcm = None
                error: Exception | None = None
                for _attempt in range(self.max_attempts):
                    cancellation_check()
                    attempts_total += 1
                    try:
                        pcm = shape_pcm(loaded.synthesize(chunk.text), active_settings.sample_rate, active_settings.as_dict())
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        if not self._retryable(exc) or not self._backoff(_attempt):
                            break
                if pcm is None:
                    message = f"chunk {chunk.global_index} failed after {self.max_attempts} attempts"
                    self.workspace.update_generation(self.conversion_id, status="failed", stage="synthesis", error=message, last_safe_error=message, worker=None)
                    raise RuntimeError(message) from error
                info = write_pcm_wav(path, pcm, int(job["tts"]["sample_rate"]), overwrite=True)
                relative = path.relative_to(self.workspace.conversion_path(self.conversion_id)).as_posix()
                completed[chunk.global_index] = {**chunk.manifest_record(relative, info.duration_seconds), "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                ordered = [completed[index] for index in sorted(completed)]
                job = self.workspace.update_generation(self.conversion_id, status="synthesizing", stage="synthesis", completed_chunks=ordered, progress={"completed": len(ordered), "current": chunk.global_index + 1, "total": len(chunks)}, worker={"pid": os.getpid(), "started_at": job["worker"]["started_at"], "updated_at": self._timestamp()})
                if job["status"] == "cancelled":
                    raise CancellationRequested
                if self.workspace.cancellation_requested(self.conversion_id):
                    raise CancellationRequested
            final_records = [completed[index] for index in sorted(completed)]
            final_job = self.workspace.update_generation(self.conversion_id, status="completed", stage="synthesis_complete", completed_chunks=final_records, progress={"completed": len(final_records), "current": len(chunks), "total": len(chunks)}, worker=None, output=None)
            if final_job["status"] == "cancelled":
                return WorkerResult("cancelled", len(final_job["completed_chunks"]), len(chunks), attempts_total)
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
                if current.get("status") != "failed" and current.get("stage") not in {"assembling", "encoding", "verifying", "publishing", "phase5"}:
                    self.workspace.update_generation(self.conversion_id, status="failed", stage="synthesis", error=message, last_safe_error=message, worker=None)
            except Exception:
                pass
            raise
        finally:
            for loaded in list(cache.values()):
                close_cached(loaded)

    def run(self, *, engine: Any | None = None, full_pipeline: bool | None = None) -> WorkerResult:
        lease = self._acquire_worker_lock()
        self._lease = lease
        try:
            if full_pipeline is None:
                full_pipeline = engine is None
            job = self.workspace.read_job(self.conversion_id)
            if job.get("schema_version") == 5:
                if engine is not None:
                    raise ManifestError("schema-v5 requires an engine factory")
                return self._run_v5(job, full_pipeline=full_pipeline)
            if job.get("schema_version") != 4:
                raise ManifestError("job must be configured for generation")
            return self._run_v4(job, engine=engine, full_pipeline=full_pipeline)
        finally:
            self._lease = None
            self._reclaimed_stale_lock = False
            lease.release()


def conversion_worker_lock_is_reclaimable(workspace: Workspace, conversion_id: str) -> bool:
    """Report whether a conversion's worker lock exists but is provably abandoned.

    A crashed owner whose PID is inaccessible (Windows ACCESS_DENIED) reports as
    alive forever, so a recorded PID alone can wedge a conversion permanently.
    This returns True only when a lock file is actually present AND stale, so a
    manifest with no lock at all still falls back to the recorded-PID check.
    """

    path = workspace.conversion_path(conversion_id) / WORKER_LOCK_FILENAME
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    return ConversionWorker(workspace, conversion_id)._worker_lock_is_stale(path)


def run_worker(workspace_root: str | Path, conversion_id: str) -> WorkerResult:
    return ConversionWorker(Workspace(Path(workspace_root)), conversion_id).run(full_pipeline=True)


__all__ = ["MAX_ATTEMPTS", "VOICE_CACHE_CAP", "CancellationRequested", "ConversionWorker", "WorkerBusyError", "WorkerResult", "run_worker"]

if __name__ == "__main__":  # pragma: no cover - exercised by the launcher
    import argparse
    parser = argparse.ArgumentParser(description="PDF audiobook conversion worker")
    parser.add_argument("workspace_root")
    parser.add_argument("conversion_id")
    args = parser.parse_args()
    raise SystemExit(0 if run_worker(args.workspace_root, args.conversion_id).status == "completed" else 2)

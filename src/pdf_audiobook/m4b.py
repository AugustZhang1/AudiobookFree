"""Measured chapter assembly, AAC/M4B encoding, and verified publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import wave
from typing import Any, Callable, Iterable

from .audio import validate_wav
from .chapters import select_chapter_range
from .tts import EngineMetadata, plan_chunks, plan_interactive_chunks
from .voice_registry import get_generation_facts, registry_revision
from .workspace import INTERACTIVE_GENERATION_SCHEMA_VERSION, Workspace, atomic_write_text

ORDINARY_PAUSE_MS = 150
PARAGRAPH_PAUSE_MS = 400
CHAPTER_PAUSE_MS = 750
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class M4BError(RuntimeError):
    """A conversion could not be assembled, encoded, verified, or published."""


class ToolUnavailable(M4BError):
    pass


class Phase5Cancelled(M4BError):
    pass


@dataclass(frozen=True)
class ChapterTiming:
    index: int
    title: str
    start_frame: int
    end_frame: int
    sample_rate: int

    @property
    def start_ms(self) -> int:
        return round(self.start_frame * 1000 / self.sample_rate)

    @property
    def end_ms(self) -> int:
        return round(self.end_frame * 1000 / self.sample_rate)


@dataclass(frozen=True)
class AssemblyResult:
    path: Path
    sample_rate: int
    frames: int
    chapters: tuple[ChapterTiming, ...]

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


@dataclass(frozen=True)
class VerifiedOutput:
    path: Path
    duration_seconds: float
    chapter_count: int
    codec: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise M4BError(f"{label} is missing or unsafe") from exc
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    reparse = bool(flag and getattr(info, "st_file_attributes", 0) & flag)
    if path.is_symlink() or reparse or not stat.S_ISREG(info.st_mode):
        raise M4BError(f"{label} is missing or unsafe")
    return path


def _paragraph_boundary(text: str) -> bool:
    return bool(re.search(r"\n\s*\n+\s*$", text))


def _safe_directory(path: Path, *, create: bool = False) -> Path:
    if path.exists() or path.is_symlink():
        try:
            info = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise M4BError(f"unsafe working directory: {path}") from exc
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        reparse = bool(flag and getattr(info, "st_file_attributes", 0) & flag)
        if path.is_symlink() or reparse or not stat.S_ISDIR(info.st_mode):
            raise M4BError(f"unsafe working directory: {path}")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
        return _safe_directory(path, create=False)
    return path


def _prepare_working_file(path: Path) -> None:
    if path.exists() or path.is_symlink():
        info = path.stat(follow_symlinks=False)
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        reparse = bool(flag and getattr(info, "st_file_attributes", 0) & flag)
    else:
        info = None; reparse = False
    if path.is_symlink() or reparse or (path.exists() and not path.is_file()):
        raise M4BError(f"unsafe working output: {path}")
    if path.exists():
        path.unlink()


def _recorded_chunks(workspace: Workspace, conversion_id: str) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    job = workspace.read_job(conversion_id)
    text, _ = workspace.load_cleaned_artifacts(conversion_id)
    plan = workspace.load_chapter_plan(conversion_id)
    tts = job["tts"]
    if job.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
        try:
            voice_plan = workspace.load_voice_plan(conversion_id)
            speaker_analysis = workspace.load_speaker_analysis(conversion_id)
        except Exception as exc:
            raise M4BError("interactive voice artifacts are unavailable") from exc
        if voice_plan.get("approval", {}).get("state") != "approved":
            raise M4BError("interactive generation requires an approved voice plan")
        if voice_plan.get("canonical_artifact_sha256") != job.get("voice_plan_sha256") or voice_plan.get("revision") != job.get("voice_plan_revision"):
            raise M4BError("voice plan does not match the bound generation")
        if speaker_analysis.get("canonical_artifact_sha256") != job.get("speaker_analysis_sha256"):
            raise M4BError("speaker analysis does not match the bound generation")
        if registry_revision() != job.get("voice_registry_revision"):
            raise M4BError("voice registry revision does not match the bound generation")
        plan_voice_ids = [entry.get("voice_id") for entry in voice_plan.get("cast", []) if isinstance(entry, dict)]
        plan_voice_ids = list(dict.fromkeys(plan_voice_ids))
        if plan_voice_ids != job.get("cast_voice_ids"):
            raise M4BError("voice cast does not match the bound generation")
        settings = tts.get("settings", {}) if isinstance(tts.get("settings", {}), dict) else {}
        chapter_start = settings.get("chapter_start")
        chapter_end = settings.get("chapter_end")
        start = 1 if chapter_start is None else chapter_start
        end = len(plan["chapters"]) if chapter_end is None else chapter_end
        chapters = select_chapter_range(plan, chapter_start, chapter_end)
        # Interactive chunks retain their original chapter indexes. Preserve
        # those indexes on the selected chapter metadata used for assembly.
        for chapter, chapter_index in zip(chapters, range(start, end + 1)):
            chapter["index"] = chapter_index
        try:
            chunks = plan_interactive_chunks(
                text,
                voice_plan,
                get_generation_facts,
                job["voice_registry_revision"],
                chapter_range=(start, end),
                cap=int(tts.get("chunk_cap", 900)),
            )
        except Exception as exc:
            raise M4BError("interactive chunk plan is invalid") from exc
    else:
        metadata = EngineMetadata(
            str(tts["engine"]), str(tts.get("package_version", "")), str(tts.get("model", "")),
            str(tts.get("model_revision", "")), str(tts.get("model_checksum", "")), str(tts["voice"]),
            str(tts.get("voice_version", "")), str(tts.get("voice_checksum", "")), int(tts["sample_rate"]),
            dict(tts.get("settings", {})),
        )
        chapters = select_chapter_range(plan, metadata.settings.get("chapter_start"), metadata.settings.get("chapter_end"))
        chunks = plan_chunks(text, chapters, metadata, cap=int(tts.get("chunk_cap", 900)))
    records = job["completed_chunks"]
    if not chunks or len(chunks) != job["total_chunks"] or len(records) != len(chunks):
        raise M4BError("completed chunk set does not match the current plan")
    ordered: list[Any] = []
    conversion = workspace.conversion_path(conversion_id)
    for chapter in chapters:
        if not isinstance(chapter.get("title"), str) or not chapter["title"]:
            raise M4BError("chapter title is missing")
    for chunk, record in zip(chunks, records):
        expected = conversion / f"chunks/chapter-{chunk.chapter_index:03d}-chunk-{chunk.local_index:04d}.wav"
        if (record["chapter_index"], record["global_index"], record["local_index"]) != (chunk.chapter_index, chunk.global_index, chunk.local_index):
            raise M4BError("completed chunks are not in planned global order")
        if record["input_hash"] != chunk.input_hash or record["relative_path"] != expected.relative_to(conversion).as_posix():
            raise M4BError("completed chunk input or path does not match the bound plan")
        if job.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
            for field in ("audio_input_hash", "span_id", "speaker_id", "voice_id", "segment_type", "source_start", "source_end"):
                if record.get(field) != getattr(chunk, field):
                    raise M4BError(f"completed chunk {field} does not match the bound plan")
        path = conversion / record["relative_path"]
        _safe_regular_file(path, "completed chunk WAV")
        try:
            info = validate_wav(path, expected_sample_rate=int(tts["sample_rate"]))
        except (OSError, ValueError) as exc:
            if job.get("schema_version") == INTERACTIVE_GENERATION_SCHEMA_VERSION:
                raise M4BError("completed chunk WAV is invalid") from exc
            raise
        digest = _sha256(path)
        if "wav_sha256" in record and (record["wav_sha256"] == "0" * 64 or record["wav_sha256"] != digest):
            raise M4BError("completed chunk WAV hash mismatch")
        if float(record["duration_seconds"]) != info.frames / info.sample_rate:
            raise M4BError("completed chunk duration does not match WAV frames")
        ordered.append((chunk, path, info))
    return ordered, {**plan, "chapters": chapters}, job


def assemble_chapters(workspace: Workspace, conversion_id: str) -> AssemblyResult:
    """Re-plan and concatenate validated PCM chunks with frame-exact pauses."""

    ordered, plan, job = _recorded_chunks(workspace, conversion_id)
    rate = int(job["tts"]["sample_rate"])
    destination_dir = workspace.conversion_path(conversion_id) / ".phase5"
    _safe_directory(destination_dir, create=True)
    destination = destination_dir / "assembled.wav"
    _prepare_working_file(destination)
    chapters: list[ChapterTiming] = []
    frames_total = 0
    chapter_start: int | None = None
    chapter_index: int | None = None
    chapter_title = ""
    chapter_count = len(plan["chapters"])
    chapter_by_index = {int(item["index"]): str(item["title"]) for item in plan["chapters"]}

    fd, temporary_name = tempfile.mkstemp(prefix=".assembled.", suffix=".wav", dir=destination_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with wave.open(raw, "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate)
                for position, (chunk, path, info) in enumerate(ordered):
                    if chapter_index != chunk.chapter_index:
                        if chapter_index is not None and chapter_start is not None and (not chapters or chapters[-1].index != chapter_index):
                            chapters.append(ChapterTiming(chapter_index, chapter_title, chapter_start, frames_total, rate))
                        chapter_index = chunk.chapter_index
                        chapter_title = chapter_by_index.get(chapter_index, f"Chapter {chapter_index}")
                        chapter_start = frames_total
                    with wave.open(str(path), "rb") as source:
                        output.writeframes(source.readframes(info.frames))
                    frames_total += info.frames
                    next_item = ordered[position + 1] if position + 1 < len(ordered) else None
                    if next_item is not None:
                        pause_ms = CHAPTER_PAUSE_MS if next_item[0].chapter_index != chunk.chapter_index else (PARAGRAPH_PAUSE_MS if _paragraph_boundary(chunk.text) else ORDINARY_PAUSE_MS)
                        if next_item[0].chapter_index != chunk.chapter_index and chapter_start is not None:
                            chapters.append(ChapterTiming(chunk.chapter_index, chapter_title, chapter_start, frames_total, rate))
                        pause_frames = round(rate * pause_ms / 1000)
                        output.writeframes(b"\0\0" * pause_frames)
                        frames_total += pause_frames
                if chapter_index is not None and chapter_start is not None:
                    chapters.append(ChapterTiming(chapter_index, chapter_title, chapter_start, frames_total, rate))
            raw.flush(); os.fsync(raw.fileno())
        os.replace(temporary, destination)
        return AssemblyResult(destination, rate, frames_total, tuple(chapters))
    finally:
        temporary.unlink(missing_ok=True)


def escape_ffmetadata(value: str) -> str:
    """Escape the FFmpeg metadata characters without interpreting user text."""

    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("metadata value is invalid")
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")


def _chapter_bounds_ms(chapters: Iterable[ChapterTiming]) -> tuple[tuple[int, int], ...]:
    """Return the contiguous chapter bounds used by metadata and verification."""

    chapters = tuple(chapters)
    if not chapters:
        raise M4BError("at least one chapter is required")
    bounds: list[tuple[int, int]] = []
    previous_start = -1
    previous_measured_end: int | None = None
    for index, chapter in enumerate(chapters):
        try:
            start = chapter.start_ms
            measured_end = chapter.end_ms
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise M4BError("chapter timestamps are malformed") from exc
        if start < 0 or measured_end <= start or start <= previous_start:
            raise M4BError("chapter timestamps are not strictly increasing")
        if previous_measured_end is not None and previous_measured_end > start:
            raise M4BError("chapter timestamps overlap")
        end = chapters[index + 1].start_ms if index + 1 < len(chapters) else measured_end
        if end <= start:
            raise M4BError("chapter timestamps are not strictly increasing")
        bounds.append((start, end))
        previous_start = start
        previous_measured_end = measured_end
    return tuple(bounds)


def build_ffmetadata(chapters: Iterable[ChapterTiming]) -> str:
    chapters = tuple(chapters)
    bounds = _chapter_bounds_ms(chapters)
    result = [";FFMETADATA1"]
    for chapter, (start, end) in zip(chapters, bounds):
        result.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={escape_ffmetadata(chapter.title)}"])
    return "\n".join(result) + "\n"


def discover_tool(name: str, env_var: str) -> str:
    configured = os.environ.get(env_var)
    found = configured if configured and Path(configured).is_file() else (shutil.which(configured) if configured else None)
    if not found:
        found = shutil.which(name)
    if not found:
        raise ToolUnavailable(f"{name} executable is unavailable; configure {env_var} or PATH")
    return str(found)


def _run(command_runner: Callable[..., Any] | None, argv: list[str]) -> Any:
    runner = command_runner or subprocess.run
    try:
        return runner(argv, shell=False, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise M4BError("external media tool failed to start") from exc


def encode_m4b(assembly: AssemblyResult, metadata_path: Path, destination: Path, *, command_runner: Callable[..., Any] | None = None) -> Path:
    _prepare_working_file(destination)
    ffmpeg = discover_tool("ffmpeg", "PDF_AUDIOBOOK_FFMPEG")
    argv = [ffmpeg, "-y", "-i", str(assembly.path), "-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-af", "loudnorm=I=-18:TP=-3:LRA=11", "-movflags", "+faststart", str(destination)]
    result = _run(command_runner, argv)
    if type(getattr(result, "returncode", None)) is not int or result.returncode != 0:
        raise M4BError("ffmpeg encoding failed")
    _safe_regular_file(destination, "encoded M4B")
    if destination.stat().st_size <= 0:
        raise M4BError("ffmpeg did not produce an output file")
    return destination


def verify_m4b(path: Path, chapters: Iterable[ChapterTiming], *, command_runner: Callable[..., Any] | None = None) -> VerifiedOutput:
    _safe_regular_file(Path(path), "output file")
    ffprobe = discover_tool("ffprobe", "PDF_AUDIOBOOK_FFPROBE")
    ffmpeg = discover_tool("ffmpeg", "PDF_AUDIOBOOK_FFMPEG")
    probe_result = _run(command_runner, [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-show_chapters", str(path)])
    if type(getattr(probe_result, "returncode", None)) is not int or probe_result.returncode != 0:
        raise M4BError("ffprobe failed")
    try:
        payload = json.loads(getattr(probe_result, "stdout", ""))
    except (TypeError, ValueError) as exc:
        raise M4BError("ffprobe returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise M4BError("ffprobe JSON must be an object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not any(isinstance(item, dict) and item.get("codec_name") == "aac" for item in streams):
        raise M4BError("output does not contain AAC audio")
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
        for stream in streams or []:
            try:
                duration = max(duration, float(stream.get("duration", 0)))
            except (TypeError, ValueError):
                continue
    if not math.isfinite(duration) or duration <= 0:
        raise M4BError("output duration is invalid")
    expected = tuple(chapters)
    expected_bounds = _chapter_bounds_ms(expected)
    actual = payload.get("chapters")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise M4BError("output chapter count is incorrect")
    previous_end = -1.0
    previous_start = -1.0
    for item, wanted, (wanted_start_ms, wanted_end_ms) in zip(actual, expected, expected_bounds):
        if not isinstance(item, dict) or not isinstance(item.get("tags"), dict) or item["tags"].get("title") != wanted.title:
            raise M4BError("output chapter title or order is incorrect")
        try:
            start, end = float(item["start_time"]), float(item["end_time"])
        except (KeyError, TypeError, ValueError):
            raise M4BError("output chapter timestamps are malformed") from None
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or start <= previous_start or start < previous_end or end > duration + 0.001:
            raise M4BError("output chapter timestamps are invalid")
        if abs(start - wanted_start_ms / 1000) > 0.01 or abs(end - wanted_end_ms / 1000) > 0.01:
            raise M4BError("output chapter timestamps do not match the measured plan")
        previous_start = start
        previous_end = end
    decode = _run(command_runner, [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"])
    if type(getattr(decode, "returncode", None)) is not int or decode.returncode != 0:
        raise M4BError("ffmpeg decode verification failed")
    return VerifiedOutput(path, duration, len(expected), "aac")


# Small compatibility spellings for callers that name stages directly.
create_ffmetadata = build_ffmetadata
verify_output = verify_m4b


def _safe_name(value: str) -> str:
    value = _SAFE_NAME.sub("_", value).strip(" .")
    return value[:120] or "Audiobook"


def publish_verified_output(source: Path, *, title: str, conversion_id: str, duration_seconds: float, chapter_count: int, codec: str = "aac", destination: Path | None = None) -> dict[str, Any]:
    if codec != "aac" or type(chapter_count) is not int or chapter_count <= 0 or type(duration_seconds) not in {int, float} or not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise M4BError("verified output facts are invalid")
    _safe_regular_file(source, "publication source")
    if destination is None:
        profile = Path(os.environ["USERPROFILE"]) if os.environ.get("USERPROFILE") else Path.home()
        destination = Path(os.environ.get("PDF_AUDIOBOOK_OUTPUT_DIR") or profile / "Downloads")
    destination = Path(destination).expanduser().absolute()
    _safe_directory(destination, create=True)
    stem = _safe_name(title)
    target = destination / f"{stem}-{conversion_id[:8]}.m4b"
    suffix = 2
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise M4BError("unsafe publication target")
    while target.exists() or target.is_symlink():
        target = destination / f"{stem}-{conversion_id[:8]}-{suffix}.m4b"; suffix += 1
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=destination)
    temporary = Path(temporary_name)
    source_size = source.stat().st_size
    source_hash = _sha256(source)
    try:
        os.close(fd)
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush(); os.fsync(output_file.fileno())
        if temporary.stat().st_size != source_size or _sha256(temporary) != source_hash:
            raise M4BError("temporary publication does not match verified source")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if target.stat().st_size != source_size or _sha256(target) != source_hash:
        if target.is_file() and not target.is_symlink():
            target.unlink(missing_ok=True)
        raise M4BError("published output does not match verified source")
    return {"filename": target.name, "path": str(target), "size_bytes": source_size, "duration_seconds": duration_seconds, "chapter_count": chapter_count, "codec": codec, "sha256": source_hash}


def finalize_conversion(workspace: Workspace, conversion_id: str, *, command_runner: Callable[..., Any] | None = None, destination: Path | None = None) -> dict[str, Any]:
    """Run all Phase 5 stages and publish only after both verifications pass."""

    def stage(name: str) -> None:
        current = workspace.read_job(conversion_id)
        if workspace.cancellation_requested(conversion_id):
            workspace.update_generation(conversion_id, status="cancelled", stage="cancelled", worker=None, last_safe_error="cancelled")
            raise Phase5Cancelled("conversion cancelled")
        worker = current.get("worker")
        if worker is None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            worker = {"pid": os.getpid(), "started_at": now, "updated_at": now}
        else:
            from datetime import datetime, timezone
            worker = {**worker, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}
        workspace.update_generation(conversion_id, status=name, stage=name, worker=worker, error=None)

    try:
        stage("assembling")
        assembly = assemble_chapters(workspace, conversion_id)
        metadata_path = assembly.path.with_name("metadata.txt")
        atomic_write_text(metadata_path, build_ffmetadata(assembly.chapters))
        stage("encoding")
        encoded = assembly.path.with_name("encoded.m4b")
        _prepare_working_file(encoded)
        encode_m4b(assembly, metadata_path, encoded, command_runner=command_runner)
        stage("verifying")
        verified = verify_m4b(encoded, assembly.chapters, command_runner=command_runner)
        stage("publishing")
        title = (workspace.load_analysis(conversion_id).get("title") or Path(workspace.read_job(conversion_id)["original_display_filename"]).stem)
        output = publish_verified_output(encoded, title=str(title), conversion_id=conversion_id, duration_seconds=verified.duration_seconds, chapter_count=verified.chapter_count, codec=verified.codec, destination=destination)
        workspace.update_generation(conversion_id, status="completed", stage="completed", output=output, error=None, last_safe_error=None, worker=None)
        return output
    except Phase5Cancelled:
        raise
    except Exception as exc:
        try:
            stage = workspace.read_job(conversion_id).get("stage", "phase5")
            safe = "phase 5 failed: " + type(exc).__name__
            workspace.update_generation(conversion_id, status="failed", stage=stage, error=safe, last_safe_error=safe, worker=None)
        except Exception:
            pass
        raise


__all__ = ["AssemblyResult", "ChapterTiming", "M4BError", "Phase5Cancelled", "ToolUnavailable", "VerifiedOutput", "assemble_chapters", "build_ffmetadata", "create_ffmetadata", "discover_tool", "encode_m4b", "escape_ffmetadata", "finalize_conversion", "publish_verified_output", "verify_m4b", "verify_output"]

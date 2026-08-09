"""Typed command-line benchmark runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .audio import WavInfo, calculate_rtf, write_pcm_wav
from .engines import CHATTERBOX_MODEL_ID, KOKORO_MODEL_ID, load_engine, validate_torch_threads, voices_for_engine
from .text import text_facts


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}-{index:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free output name near {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local TTS engines with deterministic fake and lazy official adapters.")
    parser.add_argument("--engine", choices=("fake", "kokoro", "chatterbox"), default="fake")
    parser.add_argument("--voice", help="voice ID; omit with --list-voices")
    parser.add_argument("--list-voices", action="store_true", help="list static voices without loading/downloading a model")
    parser.add_argument("--text", type=Path, default=Path("benchmark/sample.txt"), help="UTF-8 benchmark text file")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/previews"), help="directory for measured WAV output")
    parser.add_argument("--output", type=Path, help="explicit WAV path; existing files require --overwrite")
    parser.add_argument("--result-dir", type=Path, default=Path("benchmark/results"), help="directory for JSON result")
    parser.add_argument("--result", type=Path, help="explicit JSON result path; existing files require --overwrite")
    parser.add_argument("--warmups", "--warmup", dest="warmups", type=int, default=1, help="warmup synthesis runs (default: 1)")
    parser.add_argument("--runs", type=int, default=2, help="measured synthesis runs (default: 2)")
    parser.add_argument("--speed", type=float, default=1.0, help="engine speaking speed")
    parser.add_argument("--sample-rate", type=int, default=22050, help="fake-engine output sample rate")
    parser.add_argument("--torch-threads", type=int, help="Kokoro PyTorch intra-op thread count")
    parser.add_argument("--torch-compile", action="store_true", help="experimental Kokoro-only torch.compile(pipeline.model)")
    parser.add_argument("--no-inference-mode", action="store_true", help="Kokoro-only baseline without torch.inference_mode()")
    parser.add_argument("--reference-wav", type=Path, help="explicit Chatterbox reference WAV path")
    parser.add_argument("--short", action="store_true", help="use the first ~60-90 seconds of the common sample")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing explicitly named output/result files")
    return parser


def _facts_template(engine: Any, voice: str, facts: Any, model_init: float) -> dict[str, Any]:
    logical_cpu_count = _logical_cpu_count()
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "machine": {"system": platform.system(), "release": platform.release(), "processor": platform.processor(), "logical_cpu_count": logical_cpu_count},
        "engine": {
            "id": engine.engine_id,
            "package_version": engine.package_version,
            "model_id": engine.model_id,
            "model_revision": engine.model_revision,
            "model_checksum": engine.model_checksum,
        },
        "voice": {"id": voice},
        "text": {"characters": facts.characters, "words": facts.words},
        "sample_rate": engine.sample_rate,
        "model_init_seconds": model_init,
        "runtime": {**getattr(engine, "runtime_facts", {}), "cpu_time_basis": "process_time seconds; CPU-time/utilization evidence, not energy"},
        "warmup_runs": [],
        "measured_runs": [],
        "summary": {},
        "output_wav": None,
        "errors": [],
    }


def _intended_failure_template(args: argparse.Namespace, facts: Any) -> dict[str, Any]:
    """Create honest metadata when installation/model initialization fails."""

    model_id = {
        "fake": "deterministic-fake",
        "kokoro": KOKORO_MODEL_ID,
        "chatterbox": CHATTERBOX_MODEL_ID,
    }.get(args.engine)
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "machine": {"system": platform.system(), "release": platform.release(), "processor": platform.processor(), "logical_cpu_count": _logical_cpu_count()},
        "engine": {"id": args.engine, "package_version": "unresolved: engine initialization failed", "model_id": model_id, "model_revision": None, "model_checksum": None},
        "voice": {"id": args.voice},
        "text": {"characters": facts.characters, "words": facts.words},
        "sample_rate": args.sample_rate if args.engine == "fake" else None,
        "model_init_seconds": None,
        "runtime": {
            "configured_torch_threads": getattr(args, "torch_threads", None),
            "effective_torch_threads": None,
            "inference_mode": args.engine == "kokoro" and not getattr(args, "no_inference_mode", False),
            "compile": bool(getattr(args, "torch_compile", False)),
            "cpu_time_basis": "process_time seconds; CPU-time/utilization evidence, not energy",
        },
        "warmup_runs": [],
        "measured_runs": [],
        "summary": {},
        "output_wav": None,
        "errors": [],
    }


def _run_once(engine: Any, text: str, voice: str, speed: float, reference: Path | None) -> tuple[bytes, float, float]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    audio = engine.synthesize(text, voice, speed, reference)
    return audio, time.perf_counter() - started, time.process_time() - cpu_started


def _logical_cpu_count() -> int | None:
    return os.cpu_count()


def _cpu_utilization(cpu_seconds: float, wall_seconds: float, logical_cpu_count: int | None) -> float | None:
    if logical_cpu_count is None or wall_seconds <= 0:
        return None
    return cpu_seconds / (wall_seconds * logical_cpu_count)


def _validate_runtime_options(args: argparse.Namespace) -> None:
    torch_threads = getattr(args, "torch_threads", None)
    validate_torch_threads(torch_threads)
    if args.engine != "kokoro" and torch_threads is not None:
        raise ValueError("--torch-threads is only valid with --engine kokoro")
    if args.engine != "kokoro" and getattr(args, "torch_compile", False):
        raise ValueError("--torch-compile is only valid with --engine kokoro")
    if args.engine != "kokoro" and getattr(args, "no_inference_mode", False):
        raise ValueError("--no-inference-mode is only valid with --engine kokoro")


def run_benchmark(args: argparse.Namespace) -> int:
    _validate_runtime_options(args)
    if args.warmups < 0 or args.runs < 1:
        raise ValueError("--warmups must be >= 0 and --runs must be >= 1")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.voice is None:
        raise ValueError("--voice is required (omit it only with --list-voices)")
    if args.engine == "chatterbox":
        if args.voice == "reference-wav" and args.reference_wav is None:
            raise ValueError("--reference-wav is required with Chatterbox voice reference-wav")
        if args.voice == "builtin" and args.reference_wav is not None:
            raise ValueError("--reference-wav is only valid with Chatterbox voice reference-wav")
    elif args.reference_wav is not None:
        raise ValueError("--reference-wav is only valid with Chatterbox")
    if args.reference_wav is not None and not args.reference_wav.is_file():
        raise FileNotFoundError(args.reference_wav)
    available = {item.id for item in voices_for_engine(args.engine)}
    if args.voice not in available:
        raise ValueError(f"unknown voice {args.voice!r}; use --list-voices")
    source = args.text
    if not source.is_file():
        raise FileNotFoundError(source)
    source_facts = text_facts(source.read_text(encoding="utf-8"))
    normalized = source_facts.text
    if args.short:
        words = normalized.split()
        normalized = " ".join(words[:220])
        source_facts = text_facts(normalized)
    try:
        torch_threads = getattr(args, "torch_threads", None)
        torch_compile = bool(getattr(args, "torch_compile", False))
        inference_mode = args.engine == "kokoro" and not getattr(args, "no_inference_mode", False)
        if torch_threads is None and not torch_compile and (args.engine != "kokoro" or inference_mode):
            engine, model_init = load_engine(args.engine, args.voice)
        else:
            engine, model_init = load_engine(
                args.engine,
                args.voice,
                torch_threads=torch_threads,
                torch_compile=torch_compile,
                inference_mode=inference_mode,
            )
    except Exception as exc:
        failure = _intended_failure_template(args, source_facts)
        failure["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_failure_evidence(failure, args)
        raise
    if args.engine == "fake":
        engine.sample_rate = args.sample_rate
    result = _facts_template(engine, args.voice, source_facts, model_init)
    try:
        for _ in range(args.warmups):
            audio, elapsed, cpu_seconds = _run_once(engine, normalized, args.voice, args.speed, args.reference_wav)
            result["warmup_runs"].append({"generation_seconds": elapsed, "cpu_seconds": cpu_seconds, "audio_bytes": len(audio)})
        measured: list[tuple[bytes, float, float, WavInfo]] = []
        for run_number in range(1, args.runs + 1):
            audio, elapsed, cpu_seconds = _run_once(engine, normalized, args.voice, args.speed, args.reference_wav)
            temporary = args.output_dir / f".benchmark-{args.engine}-{args.voice}-{run_number}.wav"
            wav_info = write_pcm_wav(temporary, audio, engine.sample_rate, overwrite=True)
            measured.append((audio, elapsed, cpu_seconds, wav_info))
            utilization = _cpu_utilization(cpu_seconds, elapsed, result["machine"]["logical_cpu_count"])
            result["measured_runs"].append({"run": run_number, "generation_seconds": elapsed, "cpu_seconds": cpu_seconds, "cpu_utilization_fraction": utilization, "cpu_utilization_percent": utilization * 100 if utilization is not None else None, "audio_seconds": wav_info.duration_seconds, "rtf": calculate_rtf(elapsed, wav_info.duration_seconds), "audio_bytes": len(audio)})
            temporary.unlink(missing_ok=True)
        chosen_audio, _, _, _ = measured[-1]
        output = args.output or args.output_dir / f"{_utc_stamp()}-{args.engine}-{args.voice}.wav"
        if args.output is not None and output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing WAV: {output}")
        if args.output is None:
            output = _unique_path(output, False)
        output_info = write_pcm_wav(output, chosen_audio, engine.sample_rate, overwrite=args.overwrite)
        result["output_wav"] = {"path": str(output), "sample_rate": output_info.sample_rate, "duration_seconds": output_info.duration_seconds, "bytes": output_info.file_bytes}
        average_generation = sum(item[1] for item in measured) / len(measured)
        average_cpu = sum(item[2] for item in measured) / len(measured)
        average_audio = sum(item[3].duration_seconds for item in measured) / len(measured)
        utilization = _cpu_utilization(average_cpu, average_generation, result["machine"]["logical_cpu_count"])
        result["summary"] = {"average_generation_seconds": average_generation, "average_cpu_seconds": average_cpu, "average_cpu_utilization_fraction": utilization, "average_cpu_utilization_percent": utilization * 100 if utilization is not None else None, "average_audio_seconds": average_audio, "average_rtf": calculate_rtf(average_generation, average_audio)}
        result_path = args.result or args.result_dir / f"{_utc_stamp()}-{args.engine}-{args.voice}.json"
        if args.result is not None and result_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing result: {result_path}")
        if args.result is None:
            result_path = _unique_path(result_path, False)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing result: {result_path}")
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"result": str(result_path), "output": str(output), "rtf": result["summary"]["average_rtf"]}, indent=2))
        return 0
    except Exception as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_failure_evidence(result, args)
        raise
    finally:
        engine.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_options(args)
        if args.list_voices:
            for voice in voices_for_engine(args.engine):
                print(f"{voice.id}\t{voice.display_name}")
            return 0
        return run_benchmark(args)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


def _write_failure_evidence(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Best-effort timestamped evidence for failures after engine initialization."""

    target = args.result or args.result_dir / f"{_utc_stamp()}-{args.engine}-{args.voice or 'unknown'}-failed.json"
    if target.exists() and not args.overwrite:
        target = _unique_path(target, False)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"partial result: {target}", file=sys.stderr)
    except OSError as exc:
        print(f"could not write partial result: {exc}", file=sys.stderr)

"""Design a brand-new voice from a description with Qwen3-TTS VoiceDesign.

Runs in its own Colab environment (``pip install qwen-tts``). It speaks a short
neutral passage in the described voice and saves it as a mono 16-bit 24 kHz
WAV: a reference clip the voice library and Chatterbox cloning already accept.
Only the standard library is imported at module level.
"""

from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable
import wave

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
SAMPLE_TEXT = (
    "I have walked these roads for longer than I care to remember. Some days the wind is kind, "
    "and some days it cuts right through you. Still, there is always another town over the next hill, "
    "and another story waiting to be told. Sit down, and I will tell you one."
)

# Ready-made descriptions for the panel's preset menu.
PRESETS = {
    "Gruff old man": "A gruff old man with a deep, gravelly, weathered voice, speaking slowly and bluntly.",
    "Charming woman": "A charming young woman with a warm, smooth, playful voice and a hint of a smile.",
    "Menacing villain": "A cold, menacing man with a low, quiet, controlled voice that sounds dangerous.",
    "Wise elderly woman": "A wise elderly woman with a soft, slightly raspy, kind voice, speaking gently and patiently.",
    "Cheerful girl": "A cheerful teenage girl with a bright, high, energetic voice, speaking quickly.",
    "Young boy": "A curious young boy of about ten with a light, clear, eager voice.",
    "Booming warrior": "A strong warrior with a booming, deep, commanding voice full of confidence.",
    "Nervous young man": "A nervous young man with a slightly shaky, hesitant voice who speaks a little too fast.",
    "Regal queen": "A regal queen with a poised, elegant, measured voice and a British accent.",
    "Raspy pirate": "A rough, raspy sea captain with a loud, hearty, gravelly voice and a West Country accent.",
    "Sultry storyteller": "A woman with a low, husky, velvety voice, speaking slowly and intimately.",
    "Jolly innkeeper": "A jolly, round-bellied innkeeper with a hearty, booming laugh in his warm voice.",
}


class VoiceDesignError(RuntimeError):
    """A user-facing voice design failure."""


def _load_model() -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        raise VoiceDesignError("voice design needs a GPU; choose Runtime > Change runtime type > T4 GPU")
    # bfloat16 needs an Ampere-or-newer GPU; a T4 gets float32 (the 1.7B model still fits).
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    return Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa")


def _pcm16(samples: Any) -> bytes:
    values = samples.reshape(-1).tolist() if hasattr(samples, "reshape") else list(samples)
    if not values:
        raise VoiceDesignError("the voice designer returned no audio")
    peak = max(1.0, max(abs(value) for value in values))
    pcm = array("h", (round(max(-1.0, min(1.0, value / peak)) * 32767) for value in values))
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm.tobytes()


def design_voices(
    jobs: list[tuple[str, str | os.PathLike[str]]],
    *,
    model_loader: Callable[[], Any] | None = None,
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """Design every ``(description, target)`` voice not already saved, loading the model once."""

    model = None
    done = []
    for number, (description, target) in enumerate(jobs, 1):
        destination = Path(target)
        if not destination.is_file():
            model = model or (model_loader or _load_model)()
            design_voice(description, destination, model=model)
        progress(f"Designed voice {number}/{len(jobs)}: {destination}")
        done.append(destination)
    return done


def design_voice(
    description: str,
    target: str | os.PathLike[str],
    *,
    text: str = SAMPLE_TEXT,
    model: Any = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Speak ``text`` in the described voice and save a 24 kHz mono 16-bit reference WAV."""

    description = " ".join(str(description or "").split())
    if not 3 <= len(description) <= 500:
        raise VoiceDesignError("describe the voice in 3 to 500 characters")
    wavs, rate = (model or _load_model()).generate_voice_design(text=text, language="English", instruct=description)
    pcm = _pcm16(wavs[0])
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "raw.wav"
        with wave.open(str(raw), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(rate))
            handle.writeframes(pcm)
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-t", "30", str(destination)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )
    if result.returncode != 0:
        raise VoiceDesignError(f"could not save the designed voice: {result.stderr.strip()}")
    with wave.open(str(destination), "rb") as clip:
        seconds = clip.getnframes() / clip.getframerate()
    if seconds <= 5:
        destination.unlink()
        raise VoiceDesignError(f"the designed sample was only {seconds:.1f} seconds; try again")
    destination.with_suffix(".txt").write_text(description, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Design voices from descriptions and save them as reference WAVs")
    parser.add_argument("--description", help="one voice: its description (with --out)")
    parser.add_argument("--out", help="one voice: where to save it")
    parser.add_argument("--batch", help='JSON file with a list of {"description": ..., "out": ...}')
    args = parser.parse_args(argv)
    try:
        if args.batch:
            jobs = [(job["description"], job["out"]) for job in json.loads(Path(args.batch).read_text(encoding="utf-8"))]
        elif args.description and args.out:
            jobs = [(args.description, args.out)]
        else:
            parser.error("give --description and --out, or --batch")
        design_voices(jobs, progress=lambda line: print(line, flush=True))
    except Exception as exc:
        print(f"Voice design failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

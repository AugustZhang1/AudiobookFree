"""Warm the Kokoro model and selected voice tensors into the Hugging Face cache."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


APPROVED_VOICES = (
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)
DEFAULT_VOICE = "af_heart"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Kokoro's model/configuration and selected voice tensors "
            "into the normal Hugging Face cache."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--voice",
        action="append",
        dest="voices",
        metavar="VOICE_ID",
        help="voice ID to warm; may be repeated (default: af_heart)",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="warm all 28 approved voices (large download)",
    )
    return parser


def _selected_voices(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    if args.all:
        return APPROVED_VOICES
    voices = tuple(args.voices or (DEFAULT_VOICE,))
    invalid = tuple(voice for voice in voices if voice not in APPROVED_VOICES)
    if invalid:
        parser.error(
            "unknown voice ID(s): "
            + ", ".join(invalid)
            + "; choose from the approved Kokoro voice IDs"
        )
    return voices


def _voices_by_pipeline(voices: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    grouped: dict[str, list[str]] = {"a": [], "b": []}
    for voice in voices:
        grouped["a" if voice.startswith("a") else "b"].append(voice)
    return tuple((language, tuple(grouped[language])) for language in ("a", "b") if grouped[language])


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    voices = _selected_voices(args, parser)

    try:
        from kokoro import KPipeline
    except Exception as exc:  # Import failures vary with optional native dependencies.
        print(
            "Kokoro could not be imported in this interpreter. Run the isolated "
            "environment setup first (uv sync --project benchmark/environments/kokoro --python 3.11). "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Preparing Kokoro assets for {len(voices)} voice(s): {', '.join(voices)}")
    try:
        for language, pipeline_voices in _voices_by_pipeline(voices):
            print(f"Loading {'American' if language == 'a' else 'British'} English pipeline (lang_code={language})...")
            pipeline = KPipeline(lang_code=language)
            for voice in pipeline_voices:
                print(f"  Downloading/warming {voice}...")
                pipeline.load_voice(voice)
    except Exception as exc:  # Kokoro and Hugging Face expose varied download exceptions.
        print(
            "Kokoro asset download failed. Check network access and the Hugging Face cache, "
            f"then retry. Error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Kokoro model/configuration and selected voice assets are ready in the normal Hugging Face cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

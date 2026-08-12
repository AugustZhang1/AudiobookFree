"""Explicit opt-in Chatterbox Nano cache/setup helper."""

from __future__ import annotations

import os
import sys


def main() -> int:
    previous_offline = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
        try:
            print(f"Chatterbox Nano cache ready on CPU (Python {sys.version.split()[0]}).")
        finally:
            for name in ("close", "release"):
                boundary = getattr(model, name, None)
                if callable(boundary):
                    boundary()
                    break
    finally:
        for name, value in previous_offline.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

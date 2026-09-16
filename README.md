# AudiobookFree

Turn a PDF into an M4B audiobook with chapters, for free, on your own computer. Everything runs locally in your browser at `http://127.0.0.1` — no accounts, no API keys, no uploads.

- **Narration:** [Kokoro](https://github.com/hexgrad/kokoro) (28 English voices, CPU-friendly). Optional: Chatterbox Nano, Interactive Voices (BookNLP) for per-character voices.
- **Output:** M4B with embedded chapters, verified after encoding.
- **No GPU needed.** A GPU only makes it faster.

## Quick start (Windows)

1. Install [Git](https://git-scm.com/download/win) if you don't have it.
2. Open **PowerShell** and run:

   ```powershell
   git clone https://github.com/AugustZhang1/AudiobookFree.git
   cd AudiobookFree
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -InstallTools
   ```

   `-InstallTools` uses WinGet to install [uv](https://docs.astral.sh/uv/) and [FFmpeg](https://ffmpeg.org/) if they're missing, then creates the Python 3.11 environments and downloads the default voice (`af_heart`). The first run takes several minutes.

3. Start the app:

   ```powershell
   uv run pdf-audiobook
   ```

   Your browser opens automatically. Upload a PDF, pick a voice and speed, choose how to split chapters, and click generate.

### Optional setup flags

| Flag | What it does |
| --- | --- |
| `-AllVoices` | Download all 28 Kokoro voices (large download) |
| `-Voice af_bella,am_adam` | Download only specific voices |
| `-WithChatterbox` | Add the Chatterbox Nano engine (large download) |
| `-WithBookNLP` | Add Interactive Voices (per-character voice assignment) |

Re-run the setup script with any flag later to add it.

## Quick start (macOS / Linux)

```bash
# Prerequisites: git, ffmpeg, espeak-ng, uv
#   macOS:  brew install ffmpeg espeak-ng uv
#   Ubuntu: sudo apt install ffmpeg espeak-ng && curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/AugustZhang1/AudiobookFree.git
cd AudiobookFree
uv python install 3.11
uv sync
uv sync --project benchmark/environments/kokoro --python 3.11
benchmark/environments/kokoro/.venv/bin/python scripts/download_kokoro_assets.py --voice af_heart
uv run pdf-audiobook
```

## Stuck? Let an AI install it for you

Paste this into ChatGPT, Claude, Gemini, or any coding assistant (Claude Code, Codex, Cursor, etc.):

> I want to install and run this project on my computer: https://github.com/AugustZhang1/AudiobookFree
> Read its README, check what I already have installed (git, uv, ffmpeg, Python 3.11), walk me through the setup one step at a time, and help me fix any errors. My OS is: ______

If you use a terminal-based assistant like Claude Code or Codex, open it inside the cloned folder and say: *"Follow the README to set this project up and launch it."*

## Run it in Google Colab instead (no install)

If your computer is slow or you can't install anything, run the Kokoro pipeline on a free Colab GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustZhang1/AudiobookFree/blob/main/colab/PDF_Audiobook_Colab.ipynb)

1. Open the link above and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells top to bottom. Pick a voice in the **Settings** cell (you can preview it before converting).
3. Upload your PDF when prompted and download the finished M4B at the end.

Free Colab sessions can disconnect; the notebook checkpoints after every chapter and resumes where it left off if you rerun it with the same settings.

## Requirements

| | |
| --- | --- |
| Python | 3.11 (installed automatically by `uv`) |
| Tools | `uv`, `ffmpeg` + `ffprobe`, `espeak-ng` (macOS/Linux only) |
| Disk | ~3 GB for the base install and one voice; more for extra voices/engines |
| RAM | 8 GB recommended |

If FFmpeg isn't on your PATH, point the app at it instead:

```powershell
$env:PDF_AUDIOBOOK_FFMPEG  = 'C:\path\to\ffmpeg.exe'
$env:PDF_AUDIOBOOK_FFPROBE = 'C:\path\to\ffprobe.exe'
```

## Troubleshooting

- **`uv` / `ffmpeg` not found right after installing** — close and reopen PowerShell so PATH refreshes, then rerun.
- **"PDF Audiobook is already running"** — an instance is open; the launcher just reopens that tab.
- **Slow generation** — normal on CPU (roughly real-time or slower). Use the Colab notebook for a GPU.
- **Scanned PDF with no text** — OCR isn't supported yet; the app needs a PDF with selectable text.

## Development

```powershell
uv sync --group test
uv run pytest
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses of bundled models and libraries.

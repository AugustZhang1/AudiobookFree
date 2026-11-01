# AudiobookFree

Turn a PDF into an M4B audiobook with chapters, for free, on your own computer. Everything runs locally in your browser at `http://127.0.0.1` — no accounts, no API keys, no uploads.

- **Narration:** [Kokoro](https://github.com/hexgrad/kokoro) (28 English voices, CPU-friendly). Optional: Chatterbox Nano, voice cloning from your own clips (in Colab), Interactive Voices (BookNLP) for per-character voices.
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

Re-run the setup script with any flag later to add it. Optional Chatterbox Nano is used only for single-voice narration; Interactive Voices remains Kokoro-only.

While a book is generating you can click **Cancel** to stop it. When it finishes, click **Convert Another PDF** to start over.

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

If your computer is slow or you can't install anything, run it on a free Colab GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustZhang1/AudiobookFree/blob/main/colab/PDF_Audiobook_Colab.ipynb)

1. Open the link above and choose **Runtime → Change runtime type → T4 GPU**.
2. Click **Runtime → Run all**. Nothing asks for a file, so the cells run straight through to the panel.
3. In the panel, pick a voice and click **Preview voice** to hear it on your own sample text (up to 10,000 characters). To compare, pick more voices under **Compare with** and click **Compare voices**.
4. Click **Choose book** in the panel and pick a PDF or EPUB, then run the last cell. The finished M4B downloads at the end.

### Your own voices (voice cloning)

The notebook can narrate with any voice you have a clip of. By default it uses [IndexTTS-2](https://github.com/index-tts/index-tts), which copies the voice from your clip and reads the emotion of each line from its words; [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Base and [Chatterbox](https://github.com/resemble-ai/chatterbox) models are also available under **Custom voice model**:

1. Under **Add a voice**, click **Choose clips**, pick an audio or video file (up to 20 MB) of one person talking, with no music, and click **Add to library**. The first 30 seconds are used, and the clip must have more than 5 seconds of audio. The file name becomes the voice name.
2. The voice is saved to Google Drive in `MyDrive/AudiobookFree/voices` and shows up in the voice list as **★ name · your voice** in every future session.
3. Pick a **Custom voice model**:
   - **IndexTTS-2** (default): close voice match with emotion taken from each line. **Expression** (0.25-1.0) sets how strong the emotion is. Slower; the first use installs it (about 10 minutes, 6 GB).
   - **Qwen3**: closest match to the clip's delivery, using a transcript of the clip.
   - **Turbo**: good quality, and it understands tags in the text such as `[laugh]` and `[sigh]`.
   - **Original**: the largest and slowest model. Use its **Expression** slider instead of tags.
   - **Nano**: the fastest model, with the lowest quality.

Qwen3 needs to know what is said in the clip: the first time a voice is used, Whisper transcribes it and saves `name.transcript.txt` beside the clip. Check that file and fix any wrong word: if the transcript does not match the clip exactly, bits of the clip leak into the lines. A short clip (5-15 s) that ends on a full sentence works best. Custom voices always narrate at speed 1.0. Clean, English, single-speaker clips clone best. Only clone voices you have permission to use.

### Design a voice

Under **Design a voice**, describe any voice ("a gruff old man with a deep, gravelly voice") or pick a preset such as *Gruff old man*, *Charming woman* or *Menacing villain*, then click **Design voice**. [Qwen3-TTS VoiceDesign](https://github.com/QwenLM/Qwen3-TTS) invents the voice and speaks a short sample, which is saved to your library. Chatterbox then clones that fixed sample to narrate, so the voice stays the same for the whole book. Each design comes out a little different, so design again if you want another take. The designer installs and downloads its model (about 4 GB) the first time you use it in a session.

### Character voices (Gemini)

Give each character in the book their own voice, with the narrator reading everything else:

1. Set **Narration** at the top of the panel to **Character voices (Gemini)**. It is off (**One voice**) by default.
2. Set the chapter mode, then choose your book. The script starts by itself (or click **Make script** to redo it): Gemini works out who says each quote. The book's words are never changed: the app splits the text at the quotation marks and Gemini only names the speaker of each quote.
3. Each character gets a short summary of who they are in the story and a description of how they sound, written by Gemini from the book. By default every main character gets a new voice designed from that description (see **Design a voice**). Untick **Design a unique voice for each character** to have Gemini pick from your existing voices instead. Change any voice, or pick **Narrator voice**. Designed voices are read by your custom voice model (Qwen3 by default; Chatterbox models need the voice-cloning setup cell).
4. Characters with fewer lines than **Own voice from** (5 by default) are read by the narrator, so a large book doesn't need a voice for every minor character.
5. Once the cast is chosen, each character reads one of their own lines below it. Change any voice and click **Preview cast** to hear it again, then click **Confirm and generate**. The finished M4B downloads at the end.

It uses Colab's built-in Gemini through your Google login, so no key is needed. If that is unavailable or hits its limit, paste a free API key from [aistudio.google.com](https://aistudio.google.com/apikey). The script is saved to `MyDrive/AudiobookFree/cast` as a text file (`[Mara] "Are you coming?"`) that you can edit to fix a speaker. Every rendered line is cached there too, so a rerun after a disconnect continues where it stopped. Dialogue without quotation marks (dashes, italic thoughts) stays with the narrator.

Free Colab sessions can disconnect; the notebook checkpoints every few pages and resumes where it left off if you rerun it with the same settings. There is no page limit on the PDF (the file-size limit is 100 MiB).

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

"""Character voices for Colab: Gemini labels who says each quote, then each
character is narrated with their own voice.

The book's words are never changed. Gemini only answers "who speaks quote N";
everything outside quotes stays with the narrator. The script is a plain text
file that can be edited by hand::

    ## Chapter One
    [narrator] The rain had not stopped for three days.

    [Mara] "Are you coming with us?"
    [narrator] Mara asked.

A blank line is a paragraph break. Work happens in three steps so it can be
split across the two Colab environments and resumed after a disconnect:

``extract`` (Kokoro env) reads the PDF into chapters, ``make_script`` (the
notebook kernel, where Colab's built-in Gemini lives) writes the script, and
``render``/``build`` (Kokoro env, plus the Chatterbox env for custom voices)
cache one WAV per line and assemble the M4B.

Only the standard library is imported at module level so the notebook kernel
can import this file without the project's dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import wave

NARRATOR = "narrator"
SAMPLE_RATE = 24000
CHUNK_CHARS = 12000
LINE_PAUSE_SECONDS = 0.15
PARAGRAPH_PAUSE_SECONDS = 0.4
CHAPTER_PAUSE_SECONDS = 1.0
GEMINI_MODEL = "gemini-flash-latest"

_PARAGRAPH = re.compile(r"\n\s*\n")
_DOUBLE_QUOTE = re.compile(r'“(?:[^“”])+(?:”|$)|"[^"]+"')
_SINGLE_QUOTE = re.compile(r"(?<!\w)‘(?:[^‘’]|’(?=\w))+(?:’(?!\w)|$)")
_LINE = re.compile(r"^\[([^\[\]\n]+)\]\s?(.*)$")
# Same rule as tts.flowed_paragraphs: a leading Chapter/Part/Section line is its own paragraph.
_HEADING_LINE = re.compile(r"^(?:chapter|part|section)\b(?:\s+.*)?$", re.IGNORECASE)
# Kokoro voices from best to weakest (Kokoro's own quality grades), for default casting.
KOKORO_BEST_FIRST = [
    "af_heart", "af_bella", "af_nicole", "bf_emma", "af_aoede", "af_kore", "af_sarah", "af_nova", "bf_isabella",
    "af_alloy", "af_sky", "bf_alice", "bf_lily", "af_jessica", "af_river",
    "am_fenrir", "am_michael", "am_puck", "bm_george", "bm_fable", "bm_lewis", "am_echo", "am_eric", "am_liam",
    "am_onyx", "bm_daniel", "am_santa", "am_adam",
]


class CastError(RuntimeError):
    """A user-facing failure in the character-voice pipeline."""


# --- script -------------------------------------------------------------------


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in _PARAGRAPH.split(text):
        lines = block.strip().split("\n")
        while len(lines) > 1 and _HEADING_LINE.match(lines[0].strip()):
            paragraphs.append(lines.pop(0).strip())
        flowed = " ".join(" ".join(lines).split())
        if flowed:
            paragraphs.append(flowed)
    return paragraphs


def _quotes(paragraph: str, single: bool) -> list[tuple[int, int]]:
    pattern = _SINGLE_QUOTE if single else _DOUBLE_QUOTE
    return [match.span() for match in pattern.finditer(paragraph)]


def _uses_single_quotes(text: str) -> bool:
    return '"' not in text and "“" not in text and "‘" in text


def _speakable(text: str) -> bool:
    return any(character.isalnum() for character in text)


def clean_name(name: Any) -> str:
    """Normalise a speaker name; anything unusable becomes the narrator."""

    if not isinstance(name, str):
        return NARRATOR
    name = " ".join(re.sub(r"[\[\]]", "", name).split())[:40]
    if not name or name.lower() in {NARRATOR, "unknown", "none", "nobody"}:
        return NARRATOR
    return name


def _chunks(paragraphs: list[str], limit: int = CHUNK_CHARS) -> list[list[int]]:
    """Group paragraph indexes into requests of at most about ``limit`` characters."""

    groups: list[list[int]] = []
    size = 0
    for index, paragraph in enumerate(paragraphs):
        if groups and size + len(paragraph) <= limit:
            groups[-1].append(index)
            size += len(paragraph) + 2
        else:
            groups.append([index])
            size = len(paragraph) + 2
    return groups


def build_prompt(passage: str, known: dict[str, dict[str, str]]) -> str:
    cast = ", ".join(f"{name} ({about['gender']})" for name, about in known.items())
    return (
        "You are labelling who speaks each quotation in a passage from a book, for a multi-voice audiobook.\n"
        "Each quotation is marked with an ID such as [Q3] just before its opening quote mark.\n"
        "For every ID, give the name of the character who says it out loud. Use \"narrator\" when it is not "
        "spoken by a character (a title, a sign, a letter being read, a quoted word or phrase).\n"
        "Use one consistent name per character: their usual name, never a pronoun or a description like "
        "\"the boy\" when the name is known.\n"
        + (f"Characters already named earlier in the book (reuse these names exactly): {cast}.\n" if cast else "")
        + "For every character you name, give their gender (\"female\", \"male\" or \"unknown\"); \"voice\": a "
        "description a voice designer can turn into their speaking voice, in under 35 words, based on how the "
        "book portrays them (gender, age, pitch, texture such as gravelly, breathy or smooth, pace, accent, and "
        "manner); and \"about\": who they are in the story, in under 25 words.\n"
        "Reply with JSON only, in this shape:\n"
        '{"speakers": {"Q1": "Mara", "Q2": "narrator"}, '
        '"characters": {"Mara": {"gender": "female", "voice": "A woman in her late twenties with a clear, '
        'mid-pitched, slightly husky voice; speaks quickly and firmly, warm but impatient, neutral American accent.", '
        '"about": "Tom\'s older sister, leading her family to safety"}}}\n\n'
        "Passage:\n" + passage
    )


def parse_reply(reply: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Read Gemini's JSON reply; tolerate code fences and stray text around it."""

    data = _json_object(reply)
    speakers = data.get("speakers")
    speakers = {str(key): clean_name(value) for key, value in speakers.items()} if isinstance(speakers, dict) else {}
    characters: dict[str, dict[str, str]] = {}
    raw = data.get("characters")
    for name, about in (raw.items() if isinstance(raw, dict) else ()):
        name = clean_name(name)
        if name == NARRATOR or not isinstance(about, dict):
            continue
        gender = about.get("gender") if about.get("gender") in {"female", "male", "unknown"} else "unknown"
        voice = " ".join(str(about.get("voice") or "").split())[:300]
        summary = " ".join(str(about.get("about") or "").split())[:200]
        characters[name] = {"gender": gender, "voice": voice, "about": summary}
    return speakers, characters


def _json_object(reply: str) -> dict[str, Any]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise CastError("Gemini did not reply with JSON")
    try:
        data = json.loads(reply[start : end + 1])
    except ValueError as exc:
        raise CastError("Gemini replied with malformed JSON") from exc
    if not isinstance(data, dict):
        raise CastError("Gemini replied with malformed JSON")
    return data


def _ask_cached(cache: Path, ask: Callable[[str], str], prompt: str) -> str:
    """Ask once per distinct prompt; only replies that parse are kept."""

    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / (hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24] + ".json")
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    reply = ask(prompt)
    _json_object(reply)
    cached.write_text(reply, encoding="utf-8")
    return reply


def make_script(
    work_dir: str | os.PathLike[str],
    ask: Callable[[str], str],
    *,
    progress: Callable[[str], None] = print,
) -> Path:
    """Write ``script.txt`` and ``characters.json`` for the extracted book.

    Each Gemini reply is cached in ``gemini/`` so an interrupted run picks up
    where it stopped.
    """

    work = Path(work_dir)
    book = json.loads((work / "book.json").read_text(encoding="utf-8"))
    cache = work / "gemini"
    characters: dict[str, dict[str, str]] = {}
    out: list[str] = []
    chapters = book["chapters"]
    for chapter_number, chapter in enumerate(chapters, 1):
        paragraphs = _paragraphs(chapter["text"])
        single = _uses_single_quotes(chapter["text"])
        spans = [_quotes(paragraph, single) for paragraph in paragraphs]
        speakers: dict[tuple[int, int], str] = {}
        for group in _chunks(paragraphs):
            ids: dict[str, tuple[int, int]] = {}
            marked = []
            for index in group:
                paragraph = paragraphs[index]
                keys = []
                for quote in range(len(spans[index])):
                    keys.append(f"Q{len(ids) + 1}")
                    ids[keys[-1]] = (index, quote)
                for key, (start, _end) in reversed(list(zip(keys, spans[index]))):
                    paragraph = paragraph[:start] + f"[{key}]" + paragraph[start:]
                marked.append(paragraph)
            if not ids:
                continue
            reply = _ask_cached(cache, ask, build_prompt("\n\n".join(marked), characters))
            labels, found = parse_reply(reply)
            for name, about in found.items():
                known = characters.setdefault(name, {"gender": "unknown", "voice": "", "about": ""})
                if known["gender"] == "unknown":
                    known["gender"] = about["gender"]
                known["voice"] = known["voice"] or about["voice"]
                known["about"] = known["about"] or about["about"]
            for key, position in ids.items():
                speakers[position] = labels.get(key, NARRATOR)
                if speakers[position] != NARRATOR:
                    characters.setdefault(speakers[position], {"gender": "unknown", "voice": "", "about": ""})
        out.append(f"## {chapter['title']}")
        for index, paragraph in enumerate(paragraphs):
            pieces: list[tuple[str, str]] = []
            cursor = 0
            for quote, (start, end) in enumerate(spans[index]):
                pieces.append((NARRATOR, paragraph[cursor:start]))
                pieces.append((speakers.get((index, quote), NARRATOR), paragraph[start:end]))
                cursor = end
            pieces.append((NARRATOR, paragraph[cursor:]))
            # Join pieces by slicing, never re-spacing, so the words and punctuation
            # stay exactly as printed; a bare "." after a quote joins the line before it.
            lines: list[list[str]] = []
            for speaker, text in pieces:
                if lines and not _speakable(lines[-1][1]):
                    lines[-1] = [speaker, lines[-1][1] + text]
                elif lines and (lines[-1][0] == speaker or not _speakable(text)):
                    lines[-1][1] += text
                else:
                    lines.append([speaker, text])
            out.extend(f"[{speaker}] {text.strip()}" for speaker, text in lines if _speakable(text))
            out.append("")
        progress(f"Script: chapter {chapter_number}/{len(chapters)}")
    script = work / "script.txt"
    script.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    (work / "characters.json").write_text(json.dumps(characters, indent=1, ensure_ascii=False), encoding="utf-8")
    return script


def read_script(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Parse a script into chapters of paragraphs of ``(speaker, text)`` lines."""

    chapters: list[dict[str, Any]] = []
    paragraph: list[tuple[str, str]] = []

    def close_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            if not chapters:
                chapters.append({"title": "Chapter 1", "paragraphs": []})
            chapters[-1]["paragraphs"].append(paragraph)
        paragraph = []

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            close_paragraph()
            chapters.append({"title": line[3:].strip() or f"Chapter {len(chapters) + 1}", "paragraphs": []})
        elif not line:
            close_paragraph()
        else:
            match = _LINE.match(line)
            speaker, text = (clean_name(match.group(1)), match.group(2).strip()) if match else (NARRATOR, line)
            if _speakable(text):
                paragraph.append((speaker, text))
    close_paragraph()
    chapters = [chapter for chapter in chapters if chapter["paragraphs"]]
    if not chapters:
        raise CastError("the script has no lines to read")
    return chapters


def _lines_by_speaker(work: Path, script: str = "script.txt") -> dict[str, list[str]]:
    lines: dict[str, list[str]] = {}
    for chapter in read_script(work / script):
        for paragraph in chapter["paragraphs"]:
            for speaker, text in paragraph:
                lines.setdefault(speaker, []).append(text)
    return lines


def cast_list(work_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every character, busiest first: ``name``, ``gender``, ``voice`` (Gemini's note) and ``lines``."""

    work = Path(work_dir)
    path = work / "characters.json"
    about = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    people = [
        {
            "name": name,
            "gender": about.get(name, {}).get("gender", "unknown"),
            "voice": about.get(name, {}).get("voice", ""),
            "about": about.get(name, {}).get("about", ""),
            "lines": len(lines),
        }
        for name, lines in _lines_by_speaker(work).items()
        if name != NARRATOR
    ]
    return sorted(people, key=lambda person: (-person["lines"], person["name"].lower()))


def suggest_voices(people: list[dict[str, Any]], narrator: str, kokoro_voices: list[str]) -> dict[str, str]:
    """Give each character a different Kokoro voice, best first, matching their gender where known."""

    others = [voice for voice in kokoro_voices if voice != narrator]
    used = set()
    voices = {}
    for person in people:
        letter = {"female": "f", "male": "m"}.get(person["gender"])
        pool = [voice for voice in others if letter is None or voice[1:2] == letter] or others
        fresh = [voice for voice in pool if voice not in used]
        voices[person["name"]] = (fresh or pool or [narrator])[0]
        used.add(voices[person["name"]])
    return voices


def voice_choices(library: list[tuple[str, str]], narrator: str) -> list[tuple[str, str]]:
    """Describe every voice but the narrator's for Gemini: ``(value, description)``.

    ``library`` is the notebook's ``(label, path)`` list of custom voices.
    """

    choices = []
    for label, path in library:
        if path == narrator:
            continue
        # A designed voice keeps its description beside it (name.txt), which tells Gemini how it sounds.
        note = Path(path).with_suffix(".txt")
        described = " ".join(note.read_text(encoding="utf-8").split())[:300] if note.is_file() else "the user's own cloned voice"
        choices.append((path, f"{label.lstrip('★ ').split(' · ')[0]}: {described}"))
    for voice in KOKORO_BEST_FIRST:
        if voice == narrator:
            continue
        rank = [other for other in KOKORO_BEST_FIRST if other[1] == voice[1]].index(voice)
        quality = "best quality" if rank < 3 else "good quality" if rank < 7 else "basic quality"
        accent = "American" if voice[0] == "a" else "British"
        gender = "female" if voice[1] == "f" else "male"
        choices.append((voice, f"{voice.split('_', 1)[1].title()}: {accent} {gender}, {quality}"))
    return choices


def recommend_voices(
    work_dir: str | os.PathLike[str],
    ask: Callable[[str], str],
    people: list[dict[str, Any]],
    choices: list[tuple[str, str]],
) -> dict[str, str]:
    """Ask Gemini to cast each character from ``choices`` (``(value, description)`` pairs).

    Returns ``{name: value}`` for the characters it cast with a valid choice.
    """

    if not people or not choices:
        return {}
    work = Path(work_dir)
    samples = _lines_by_speaker(work)
    ids = {f"V{index}": value for index, (value, _about) in enumerate(choices, 1)}
    voice_list = "\n".join(f"V{index}: {about}" for index, (_value, about) in enumerate(choices, 1))
    character_list = "\n".join(
        f"- {person['name']} ({person['gender']}; {person['voice'] or 'no description'}; {person['lines']} lines). {person['about']} "
        f"Sample: {max(samples.get(person['name'], ['']), key=len)[:200]}"
        for person in people
    )
    prompt = (
        "You are casting a multi-voice audiobook. Choose the best voice for each character from the list.\n"
        "Match gender, age, personality and accent; give main characters different voices where possible.\n\n"
        f"Voices:\n{voice_list}\n\nCharacters:\n{character_list}\n\n"
        'Reply with JSON only, mapping each character to a voice ID, e.g. {"Mara": "V3"}'
    )
    reply = _json_object(_ask_cached(work / "gemini", ask, prompt))
    names = {person["name"] for person in people}
    return {name: ids[value] for name, value in reply.items() if name in names and isinstance(value, str) and value in ids}


def write_preview(work_dir: str | os.PathLike[str], speakers: list[str]) -> Path:
    """Write ``preview.txt``: one line from the book for each speaker, so the cast can be heard."""

    work = Path(work_dir)
    lines = _lines_by_speaker(work)
    out = ["## Preview"]
    for speaker in speakers:
        theirs = lines.get(speaker) or ["This is how I sound."]
        fitting = [line for line in theirs if len(line) <= 250] or [theirs[0][:250]]
        out += [f"[{speaker}] {max(fitting, key=len)}", ""]
    path = work / "preview.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def preview_lines(work_dir: str | os.PathLike[str], settings_path: str | os.PathLike[str]) -> list[tuple[str, Path]]:
    """Return ``(speaker, rendered WAV)`` for each line of ``preview.txt``."""

    work = Path(work_dir)
    settings = _settings(settings_path)
    key = _line_keys(settings)
    return [
        (speaker, work / "lines" / f"{key(_voice_for(settings, speaker), text)}.wav")
        for chapter in read_script(work / "preview.txt")
        for paragraph in chapter["paragraphs"]
        for speaker, text in paragraph
    ]


# --- Gemini -------------------------------------------------------------------


def gemini_api(api_key: str, *, model: str = GEMINI_MODEL, retries: int = 5, sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """Return an ``ask(prompt)`` that calls the Gemini API with a key from aistudio.google.com."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def ask(prompt: str) -> str:
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}).encode("utf-8")
        for attempt in range(retries):
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return "".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"])
            except urllib.error.HTTPError as exc:
                if exc.code in {400, 401, 403}:
                    raise CastError(f"Gemini rejected the request (HTTP {exc.code}); check the API key") from exc
                if attempt == retries - 1:
                    raise CastError(f"Gemini is unavailable (HTTP {exc.code}); try again later") from exc
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
                if attempt == retries - 1:
                    raise CastError("Gemini did not answer; try again later") from exc
            sleep(min(60, 10 * (attempt + 1)))  # free-tier rate limits reset within a minute
        raise CastError("Gemini did not answer; try again later")

    return ask


def gemini_colab(*, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """Return an ``ask(prompt)`` using Colab's built-in Gemini (no API key needed)."""

    try:
        from google.colab import ai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CastError("Colab's built-in Gemini is not available here; paste a Gemini API key from aistudio.google.com") from exc

    def ask(prompt: str) -> str:
        for attempt in range(retries):
            try:
                return ai.generate_text(prompt)
            except Exception as exc:
                if attempt == retries - 1:
                    raise CastError(f"Colab's built-in Gemini failed ({exc}); paste a Gemini API key and try again") from exc
                sleep(20 * (attempt + 1))
        raise CastError("Colab's built-in Gemini failed")

    return ask


# --- audio --------------------------------------------------------------------


def _is_custom(voice: str) -> bool:
    return voice.endswith(".wav")


def _file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _settings(path: str | os.PathLike[str]) -> dict[str, Any]:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    voices = settings.get("voices")
    if not isinstance(voices, dict) or not isinstance(voices.get(NARRATOR), str):
        raise CastError("settings need a narrator voice")
    return settings


def _voice_for(settings: dict[str, Any], speaker: str) -> str:
    """A character with no voice of their own (or set to "narrator") uses the narrator's."""

    voice = settings["voices"].get(speaker)
    return settings["voices"][NARRATOR] if not voice or voice == NARRATOR else voice


def _line_keys(settings: dict[str, Any]) -> Callable[[str, str], str]:
    """Return ``key(voice, text)``: a cache name that changes with anything that changes the audio."""

    clip_shas: dict[str, str] = {}

    def key(voice: str, text: str) -> str:
        if _is_custom(voice):
            if voice not in clip_shas:
                clip_shas[voice] = _file_sha(voice)
            model = settings.get("clone_model", "turbo")
            style = f"clone:{model}:{settings.get('exaggeration', 0.5) if model == 'original' else ''}:{clip_shas[voice]}"
        else:
            style = f"kokoro:{voice}:{float(settings.get('speed', 1.0)):g}"
        return hashlib.sha256(f"{style}\n{text}".encode("utf-8")).hexdigest()[:32]

    return key


def render(work_dir: str | os.PathLike[str], settings_path: str | os.PathLike[str], *, engine: str, script: str = "script.txt", voice_loader: Callable[[str], Any] | None = None) -> int:
    """Synthesize every not-yet-cached line whose voice belongs to ``engine``.

    ``engine`` is ``kokoro`` for the built-in voices or ``clone`` for custom
    WAV voices; the notebook runs each in its own environment. Returns the
    number of lines rendered.
    """

    work = Path(work_dir)
    settings = _settings(settings_path)
    key = _line_keys(settings)
    lines_dir = work / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    todo: dict[str, tuple[str, str]] = {}
    for chapter in read_script(work / script):
        for paragraph in chapter["paragraphs"]:
            for speaker, text in paragraph:
                voice = _voice_for(settings, speaker)
                if _is_custom(voice) != (engine == "clone"):
                    continue
                name = key(voice, text)
                if not (lines_dir / f"{name}.wav").is_file():
                    todo[name] = (voice, text)
    if not todo:
        return 0
    load = voice_loader or _default_loader(settings, engine)
    from .audio import write_pcm_wav

    loaded: dict[str, Any] = {}
    started = time.time()
    for done, (name, (voice, text)) in enumerate(todo.items(), 1):
        if voice not in loaded:
            loaded[voice] = load(voice)
        pcm = loaded[voice](text)
        partial = lines_dir / f"{name}.part.wav"
        write_pcm_wav(partial, pcm, SAMPLE_RATE, overwrite=True)
        os.replace(partial, lines_dir / f"{name}.wav")
        if done == len(todo) or done % 25 == 0:
            print(f"Rendered {done}/{len(todo)} {'custom-voice' if engine == 'clone' else 'Kokoro'} lines ({time.time() - started:.0f}s)", flush=True)
    return len(todo)


def _default_loader(settings: dict[str, Any], engine: str) -> Callable[[str], Callable[[str], bytes]]:
    from . import colab
    from .tts import CHATTERBOX_REFERENCE_VOICE, SynthesisSettings

    if engine == "clone":
        factory = colab.make_cuda_chatterbox_factory(model=settings.get("clone_model", "turbo"), exaggeration=settings.get("exaggeration", 0.5))

        def load_clone(voice: str) -> Callable[[str], bytes]:
            loaded = factory(CHATTERBOX_REFERENCE_VOICE, None, engine="chatterbox", reference_wav=voice)
            return lambda text: b"".join(loaded.synthesize(piece) for piece in colab._split_for_chatterbox(text))

        return load_clone
    factory = colab.make_cuda_kokoro_factory()
    speed = float(settings.get("speed", 1.0))
    return lambda voice: factory(voice, SynthesisSettings(speed=speed)).synthesize


def build(work_dir: str | os.PathLike[str], settings_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], *, command_runner: Callable[..., Any] | None = None) -> Path:
    """Join the rendered lines into one chaptered, verified M4B."""

    from .m4b import AssemblyResult, ChapterTiming, _safe_name, build_ffmetadata, encode_m4b, verify_m4b

    work = Path(work_dir)
    settings = _settings(settings_path)
    key = _line_keys(settings)
    book = json.loads((work / "book.json").read_text(encoding="utf-8"))
    chapters = read_script(work / "script.txt")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joined = output / "cast-assembly.wav"

    def silence(seconds: float) -> bytes:
        return b"\x00\x00" * round(SAMPLE_RATE * seconds)

    timings = []
    frames = 0
    with wave.open(str(joined), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        for number, chapter in enumerate(chapters):
            start = frames
            for p_index, paragraph in enumerate(chapter["paragraphs"]):
                for l_index, (speaker, text) in enumerate(paragraph):
                    path = work / "lines" / f"{key(_voice_for(settings, speaker), text)}.wav"
                    if not path.is_file():
                        raise CastError(f"a line for {speaker} has not been rendered yet; rerun generation")
                    with wave.open(str(path), "rb") as clip:
                        if clip.getframerate() != SAMPLE_RATE or clip.getnchannels() != 1 or clip.getsampwidth() != 2:
                            raise CastError("a rendered line has the wrong audio format")
                        pcm = clip.readframes(clip.getnframes())
                    if l_index:
                        pcm = silence(LINE_PAUSE_SECONDS) + pcm
                    elif p_index:
                        pcm = silence(PARAGRAPH_PAUSE_SECONDS) + pcm
                    out.writeframes(pcm)
                    frames += len(pcm) // 2
            if number < len(chapters) - 1:
                pause = silence(CHAPTER_PAUSE_SECONDS)
                out.writeframes(pause)
                frames += len(pause) // 2
            timings.append(ChapterTiming(number + 1, chapter["title"], start, frames, SAMPLE_RATE))
    assembly = AssemblyResult(joined, SAMPLE_RATE, frames, tuple(timings))
    metadata = output / "cast-chapters.txt"
    metadata.write_text(build_ffmetadata(assembly.chapters), encoding="utf-8")
    destination = output / f"{_safe_name(str(book.get('title') or 'Audiobook'))} (character voices).m4b"
    encode_m4b(assembly, metadata, destination, command_runner=command_runner)
    verify_m4b(destination, assembly.chapters, command_runner=command_runner)
    joined.unlink()
    metadata.unlink()
    return destination


# --- extract ------------------------------------------------------------------


def extract(pdf: str | os.PathLike[str], work_dir: str | os.PathLike[str], *, chapter_mode: str = "original", chapter_count: int | None = None) -> Path:
    """Read the PDF into ``book.json``: its title and each chapter's text."""

    from .chapters import create_chapter_plan
    from .pdf import analyze_pdf

    work = Path(work_dir)
    target = work / "book.json"
    if target.is_file():
        return target
    source = Path(pdf)
    analysis = analyze_pdf(source, fallback_title=source.stem, layout_warnings=False)
    text = analysis["cleaned_text"]
    plan = create_chapter_plan(text, analysis["cleaned_map"], analysis.get("chapter_candidates", []), mode=chapter_mode, count=chapter_count, document_title=analysis.get("title"))
    chapters = [{"title": item["title"], "text": text[item["start_offset"] : item["end_offset"]]} for item in plan["chapters"]]
    work.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"title": analysis.get("title"), "chapters": chapters}, ensure_ascii=False), encoding="utf-8")
    return target


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Character voices: extract a book, render its script's lines, build the M4B")
    steps = parser.add_subparsers(dest="step", required=True)
    step = steps.add_parser("extract")
    step.add_argument("pdf")
    step.add_argument("work_dir")
    step.add_argument("--chapter-mode", choices=("original", "whole", "custom"), default="original")
    step.add_argument("--chapter-count", type=int)
    step = steps.add_parser("render")
    step.add_argument("work_dir")
    step.add_argument("--settings", required=True)
    step.add_argument("--engine", choices=("kokoro", "clone"), required=True)
    step.add_argument("--script", default="script.txt", help="script file in the work folder (preview.txt for the cast preview)")
    step = steps.add_parser("build")
    step.add_argument("work_dir")
    step.add_argument("--settings", required=True)
    step.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.step == "extract":
            print(f"Book: {extract(args.pdf, args.work_dir, chapter_mode=args.chapter_mode, chapter_count=args.chapter_count)}")
        elif args.step == "render":
            render(args.work_dir, args.settings, engine=args.engine, script=args.script)
        else:
            print(f"Verified M4B: {build(args.work_dir, args.settings, args.output_dir)}")
    except Exception as exc:
        print(f"Character voices failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

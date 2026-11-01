from __future__ import annotations

import io
import json
from pathlib import Path
import re
import shutil
import urllib.error
import uuid
import wave

import pytest

from pdf_audiobook import cast


@pytest.fixture
def work():
    path = Path.cwd() / f".cast-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


CHAPTER_ONE = (
    "The rain had not stopped.\n\n"
    "“Are you coming with us?” Mara asked, pulling on her\ncoat. “It’s late.”\n\n"
    "Tom glanced at the door. \"Not tonight,\" he said.\n\n"
    "The sign read \"Closed\"."
)


def _book(work: Path, chapters: list[tuple[str, str]]) -> None:
    (work / "book.json").write_text(json.dumps({"title": "Storm", "chapters": [{"title": t, "text": x} for t, x in chapters]}), encoding="utf-8")


def _fake_gemini(calls: list[str]):
    names = {"Are you coming": "Mara", "It’s late": "Mara", "Not tonight": "Tom", "Closed": "narrator"}

    def ask(prompt: str) -> str:
        calls.append(prompt)
        passage = prompt.split("Passage:\n", 1)[1]
        speakers = {}
        for key, quote in re.findall(r"\[(Q\d+)\]([“\"][^”\"]*)", passage):
            speakers[key] = next(name for start, name in names.items() if start in quote)
        return "```json\n" + json.dumps({"speakers": speakers, "characters": json.loads(MARA)}) + "\n```"

    return ask


MARA = '{"Mara": {"gender": "female", "voice": "young, warm", "about": "Tom\'s sister"}, "Tom": {"gender": "male", "voice": "gruff", "about": "a fisherman"}}'


def _words(text: str) -> str:
    return "".join(text.split())


def test_script_labels_quotes_and_keeps_every_word(work: Path) -> None:
    _book(work, [("Chapter One", CHAPTER_ONE), ("Chapter Two", "No one spoke after that.")])
    calls: list[str] = []
    script = cast.make_script(work, _fake_gemini(calls), progress=lambda _line: None)

    assert script.read_text(encoding="utf-8") == (
        "## Chapter One\n"
        "[narrator] The rain had not stopped.\n\n"
        "[Mara] “Are you coming with us?”\n"
        "[narrator] Mara asked, pulling on her coat.\n"
        "[Mara] “It’s late.”\n\n"
        "[narrator] Tom glanced at the door.\n"
        "[Tom] \"Not tonight,\"\n"
        "[narrator] he said.\n\n"
        "[narrator] The sign read \"Closed\".\n\n"
        "## Chapter Two\n"
        "[narrator] No one spoke after that.\n"
    )
    spoken = "".join(text for chapter in cast.read_script(script) for paragraph in chapter["paragraphs"] for _speaker, text in paragraph)
    assert _words(spoken) == _words(CHAPTER_ONE + "No one spoke after that.")
    # One request for the chapter with quotes; none for the chapter without; IDs in reading order.
    assert len(calls) == 1
    assert calls[0].index("[Q1]“Are") < calls[0].index("[Q2]“It") < calls[0].index("[Q3]\"Not")
    assert cast.cast_list(work) == [
        {"name": "Mara", "gender": "female", "voice": "young, warm", "about": "Tom's sister", "lines": 2},
        {"name": "Tom", "gender": "male", "voice": "gruff", "about": "a fisherman", "lines": 1},
    ]


def test_script_reuses_cached_gemini_replies(work: Path) -> None:
    _book(work, [("Chapter One", CHAPTER_ONE)])
    calls: list[str] = []
    first = cast.make_script(work, _fake_gemini(calls), progress=lambda _line: None).read_text(encoding="utf-8")
    second = cast.make_script(work, lambda _prompt: pytest.fail("cached replies must not call Gemini"), progress=lambda _line: None)
    assert second.read_text(encoding="utf-8") == first


def test_bad_gemini_reply_is_not_cached(work: Path) -> None:
    _book(work, [("Chapter One", CHAPTER_ONE)])
    with pytest.raises(cast.CastError, match="JSON"):
        cast.make_script(work, lambda _prompt: "Sorry, I can't help.", progress=lambda _line: None)
    assert list((work / "gemini").iterdir()) == []


def test_known_characters_are_passed_to_later_requests(work: Path) -> None:
    _book(work, [("One", CHAPTER_ONE), ("Two", "“Wait,” Mara said.")])
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"speakers": {"Q1": "Mara", "Q2": "Mara", "Q3": "Tom", "Q4": "narrator"}, "characters": json.loads(MARA)})

    cast.make_script(work, ask, progress=lambda _line: None)
    assert "Mara (female), Tom (male)" in prompts[1]


def test_british_single_quotes_are_found_but_apostrophes_are_not() -> None:
    text = "‘I don’t know,’ he said. ‘Ask Sam’s mother.’"
    assert cast._uses_single_quotes(text)
    spans = cast._quotes(text, single=True)
    assert [text[a:b] for a, b in spans] == ["‘I don’t know,’", "‘Ask Sam’s mother.’"]


def test_unclosed_quote_runs_to_the_end_of_the_paragraph() -> None:
    text = "“It went on for pages and pages"
    assert cast._quotes(text, single=False) == [(0, len(text))]
    text = '"A speech that goes on into the next paragraph.'
    assert cast._quotes(text, single=False) == [(0, len(text))]
    assert cast._quotes('the end of a quote." Then narration.', single=False) == []


def test_single_line_breaks_split_paragraphs_but_never_inside_a_quote() -> None:
    # EPUB-converted PDFs keep paragraphs apart with one line break, which also
    # appears where a line wraps before a capital letter.
    text = (
        "Sunny looked at the\nSin of Solace. It laughed.\n"
        '"Let\'s go.\nWe\'ll escape," he said.\n'
        '"First paragraph of a long speech.\n'
        '"And the second."\n'
        "The mist rolled in."
    )
    assert cast._paragraphs(text) == [
        "Sunny looked at the Sin of Solace. It laughed.",
        '"Let\'s go. We\'ll escape," he said.',
        '"First paragraph of a long speech.',
        '"And the second."',
        "The mist rolled in.",
    ]


def test_parse_reply_cleans_names() -> None:
    speakers, characters = cast.parse_reply(
        'noise {"speakers": {"Q1": " [Mara]  Lee ", "Q2": "Unknown", "Q3": 7}, '
        '"characters": {"Mara Lee": {"gender": "female", "voice": " soft ", "about": "the heroine"}, "x": {"gender": "robot"}, "narrator": {}}} noise'
    )
    assert speakers == {"Q1": "Mara Lee", "Q2": "narrator", "Q3": "narrator"}
    assert characters == {"Mara Lee": {"gender": "female", "voice": "soft", "about": "the heroine"}, "x": {"gender": "unknown", "voice": "", "about": ""}}


def test_hand_edited_script_is_read_leniently(work: Path) -> None:
    (work / "script.txt").write_text("A line with no tag.\n[Mara] Hello.\n[ ] Who?\n\n...\n## Next\n[Tom] Bye.\n## Empty\n", encoding="utf-8")
    assert cast.read_script(work / "script.txt") == [
        {"title": "Chapter 1", "paragraphs": [[("narrator", "A line with no tag."), ("Mara", "Hello."), ("narrator", "Who?")]]},
        {"title": "Next", "paragraphs": [[("Tom", "Bye.")]]},
    ]


def test_suggested_voices_match_gender_and_avoid_the_narrator() -> None:
    voices = ["af_heart", "af_bella", "af_nova", "am_adam", "am_echo"]
    people = [{"name": n, "gender": g} for n, g in (("Mara", "female"), ("Ann", "female"), ("Tom", "male"), ("It", "unknown"))]
    suggested = cast.suggest_voices(people, "af_heart", voices)
    # Nobody shares a voice while unused ones remain.
    assert suggested == {"Mara": "af_bella", "Ann": "af_nova", "Tom": "am_adam", "It": "am_echo"}


def test_gemini_api_retries_rate_limits_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[dict] = []

    def urlopen(request, timeout):
        attempts.append(dict(request.header_items()))
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BytesIO())
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": '{"speakers": {}}'}]}}]}).encode())

    monkeypatch.setattr(cast.urllib.request, "urlopen", urlopen)
    sleeps: list[float] = []
    assert cast.gemini_api("secret", sleep=sleeps.append)("hi") == '{"speakers": {}}'
    assert len(attempts) == 2 and sleeps == [10]
    assert attempts[0]["X-goog-api-key"] == "secret"


def test_gemini_api_bad_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO())

    monkeypatch.setattr(cast.urllib.request, "urlopen", urlopen)
    with pytest.raises(cast.CastError, match="check the API key"):
        cast.gemini_api("bad", sleep=lambda _s: pytest.fail("no retry for a bad key"))("hi")


def _clip(path: Path, seconds: float = 6.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x01\x00" * int(24000 * seconds))
    return path


def test_render_splits_lines_by_engine_and_caches(work: Path) -> None:
    (work / "script.txt").write_text("## One\n[narrator] Hello there.\n[Mara] “Hi.”\n[Tom] “Yo.”\n", encoding="utf-8")
    clip = _clip(work / "Tom.wav")
    settings = work / "settings.json"
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart", "Mara": "af_bella", "Tom": str(clip)}, "speed": 1.0}), encoding="utf-8")
    heard: list[tuple[str, str]] = []

    def loader(voice):
        return lambda text: heard.append((voice, text)) or b"\x01\x00" * 2400

    assert cast.render(work, settings, engine="kokoro", voice_loader=loader) == 2
    assert heard == [("af_heart", "Hello there."), ("af_bella", "“Hi.”")]
    assert cast.render(work, settings, engine="clone", voice_loader=loader) == 1
    assert heard[-1] == (str(clip), "“Yo.”")
    assert cast.render(work, settings, engine="kokoro", voice_loader=lambda _v: pytest.fail("lines are cached")) == 0
    # A different speed is different audio, so those lines render again.
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart", "Mara": "af_bella", "Tom": str(clip)}, "speed": 1.2}), encoding="utf-8")
    assert cast.render(work, settings, engine="kokoro", voice_loader=loader) == 2


def test_build_needs_every_line_rendered(work: Path) -> None:
    _book(work, [("One", "x")])
    (work / "script.txt").write_text("## One\n[Mara] Hi.\n", encoding="utf-8")
    settings = work / "settings.json"
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart"}}), encoding="utf-8")
    with pytest.raises(cast.CastError, match="Mara has not been rendered"):
        cast.build(work, settings, work / "out")


def test_main_builds_a_chaptered_m4b(work: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is required for the M4B")
    _book(work, [("One", "x"), ("Two", "y")])
    (work / "script.txt").write_text("## One\n[narrator] Hello.\n[Mara] “Hi.”\n\n[narrator] Bye.\n## Two\n[Mara] “Again.”\n", encoding="utf-8")
    settings = work / "settings.json"
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart", "Mara": "af_bella"}}), encoding="utf-8")
    cast.render(work, settings, engine="kokoro", voice_loader=lambda _v: lambda _t: b"\x10\x00\xf0\xff" * 12000)

    assert cast.main(["build", str(work), "--settings", str(settings), "--output-dir", str(work / "out")]) == 0
    m4b = work / "out" / "Storm (character voices).m4b"
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"Verified M4B: {m4b}"
    assert sorted(path.name for path in (work / "out").iterdir()) == [m4b.name]


def test_heading_line_is_its_own_paragraph() -> None:
    assert cast._paragraphs("Chapter 3\nThe end\nof it.\n\nPart Two") == ["Chapter 3", "The end of it.", "Part Two"]


def test_default_casting_prefers_the_best_kokoro_voices() -> None:
    from pdf_audiobook.voice_registry import APPROVED_VOICE_IDS

    assert sorted(cast.KOKORO_BEST_FIRST) == sorted(APPROVED_VOICE_IDS)
    suggested = cast.suggest_voices([{"name": "Mara", "gender": "female"}, {"name": "Tom", "gender": "male"}], "af_heart", cast.KOKORO_BEST_FIRST)
    assert suggested == {"Mara": "af_bella", "Tom": "am_fenrir"}


def test_minor_and_narrator_assigned_characters_use_the_narrator_voice(work: Path) -> None:
    (work / "script.txt").write_text("## One\n[Mara] “Hi.”\n[Guard] “Halt.”\n[Tom] “Yo.”\n", encoding="utf-8")
    settings = work / "settings.json"
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart", "Mara": "narrator", "Tom": "am_fenrir"}}), encoding="utf-8")
    heard: list[tuple[str, str]] = []
    cast.render(work, settings, engine="kokoro", voice_loader=lambda voice: lambda text: heard.append((voice, text)) or b"\x01\x00")
    assert heard == [("af_heart", "“Hi.”"), ("af_heart", "“Halt.”"), ("am_fenrir", "“Yo.”")]


def test_voice_choices_describe_voices_and_skip_the_narrator() -> None:
    choices = dict(cast.voice_choices([("★ Saber · your voice", "/lib/Saber.wav")], "af_heart"))
    assert "af_heart" not in choices and len(choices) == 28
    assert choices["/lib/Saber.wav"] == "Saber: the user's own cloned voice"
    assert choices["af_bella"] == "Bella: American female, best quality"
    assert choices["bm_george"] == "George: British male, good quality"
    assert choices["am_adam"] == "Adam: American male, basic quality"


def test_gemini_recommends_voices_from_the_list(work: Path) -> None:
    (work / "script.txt").write_text("## One\n[Mara] “Run!”\n[Tom] “Never.”\n", encoding="utf-8")
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return '{"Mara": "V2", "Tom": "V99", "Ghost": "V1"}'

    people = [{"name": "Mara", "gender": "female", "voice": "young", "about": "the hero", "lines": 1}, {"name": "Tom", "gender": "male", "voice": "", "about": "", "lines": 1}]
    choices = [("af_bella", "Bella: American female"), ("/lib/Saber.wav", "Saber: cloned")]
    assert cast.recommend_voices(work, ask, people, choices) == {"Mara": "/lib/Saber.wav"}
    assert "V2: Saber: cloned" in prompts[0] and "Mara (female; young; 1 lines). the hero Sample: “Run!”" in prompts[0]
    # Cached: the same question is not asked twice.
    assert cast.recommend_voices(work, lambda _p: pytest.fail("cached"), people, choices) == {"Mara": "/lib/Saber.wav"}


def test_cast_preview_uses_each_speakers_own_line_and_the_generation_cache(work: Path) -> None:
    (work / "script.txt").write_text("## One\n[narrator] It was late.\n[Mara] “Hi.”\n[Mara] “A longer line, this one.”\n", encoding="utf-8")
    settings = work / "settings.json"
    settings.write_text(json.dumps({"voices": {"narrator": "af_heart", "Mara": "af_bella", "Tom": "narrator"}}), encoding="utf-8")
    cast.write_preview(work, ["narrator", "Mara", "Tom"])
    heard: list[tuple[str, str]] = []
    cast.render(work, settings, engine="kokoro", script="preview.txt", voice_loader=lambda v: lambda t: heard.append((v, t)) or b"\x01\x00")
    assert heard == [("af_heart", "It was late."), ("af_bella", "“A longer line, this one.”"), ("af_heart", "This is how I sound.")]
    lines = cast.preview_lines(work, settings)
    assert [speaker for speaker, _path in lines] == ["narrator", "Mara", "Tom"] and all(path.is_file() for _s, path in lines)
    # Generation reuses the preview clips: only Mara's short line is left to render.
    assert cast.render(work, settings, engine="kokoro", voice_loader=lambda v: lambda t: heard.append((v, t)) or b"\x01\x00") == 1
    assert heard[-1] == ("af_bella", "“Hi.”")

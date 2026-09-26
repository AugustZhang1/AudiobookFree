from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any
import uuid

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

import pdf_audiobook.pdf as pdf
from pdf_audiobook.pdf import PageEvidence, PdfAnalysisError, analyze_pdf


@pytest.fixture
def tmp_path() -> Path:
    path = Path("tests") / f".pytest-pdf-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_pdf(path: Path, pages: list[str | None], *, encrypted: bool = False, title: str | None = None) -> Path:
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_ref = writer._add_object(font)
    for content in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
        if content is not None:
            stream = DecodedStreamObject()
            encoded = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")
            stream.set_data(b"BT /F1 12 Tf 72 700 Td (" + encoded + b") Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
        else:
            image = DecodedStreamObject()
            image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"), NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1), NameObject("/ColorSpace"): NameObject("/DeviceGray"), NameObject("/BitsPerComponent"): NumberObject(8)})
            image.set_data(b"\x00")
            image_ref = writer._add_object(image)
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): image_ref})})
    if title:
        writer.add_metadata({"/Title": title})
    if encrypted:
        writer.encrypt("secret")
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_normal_english_extraction_cleanup_mapping_and_chapters(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["Header\nChapter 1\nThe quick brown fox jumps over the dog.\nPage 1", "Header\nChapter 2\nThis is a second useful English paragraph for review.\nPage 2"], title="A Test Book")
    result = analyze_pdf(path)
    assert result["title"] == "A Test Book"
    assert result["page_count"] == 2
    assert result["detected_language"] == "English"
    assert "Page 1" not in result["cleaned_text"]
    assert [item["source_page"] for item in result["cleaned_map"]] == [1, 2]
    assert result["cleaned_map"][0]["cleaned_start"] == 0
    assert result["cleaned_map"][0]["cleaned_end"] < result["cleaned_map"][1]["cleaned_start"]
    assert result["cleaned_map"][1]["cleaned_end"] == len(result["cleaned_text"])
    assert [item["title"] for item in result["chapter_candidates"]] == ["Chapter 1", "Chapter 2"]
    assert result["source_pdf_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_signature_corrupt_encrypted_and_ocr_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PdfAnalysisError, match="not a PDF") as signature:
        analyze_pdf(bad)
    assert signature.value.code == pdf.ERROR_INVALID_SIGNATURE

    encrypted = make_pdf(tmp_path / "encrypted.pdf", ["The English text is here for a useful review document."], encrypted=True)
    with pytest.raises(PdfAnalysisError) as encrypted_error:
        analyze_pdf(encrypted)
    assert encrypted_error.value.code == pdf.ERROR_ENCRYPTED

    scanned = make_pdf(tmp_path / "scanned.pdf", [None, None])
    with pytest.raises(PdfAnalysisError) as ocr_error:
        analyze_pdf(scanned)
    assert ocr_error.value.code == pdf.ERROR_OCR_REQUIRED


def test_limits_and_disk_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The English text is sufficient for analysis and review."])
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 1)
    with pytest.raises(PdfAnalysisError) as size_error:
        analyze_pdf(path)
    assert size_error.value.code == pdf.ERROR_SIZE_LIMIT
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 100 * 1024 * 1024)
    monkeypatch.setattr(pdf.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(0, 0, 0))
    with pytest.raises(PdfAnalysisError) as disk_error:
        analyze_pdf(path)
    assert disk_error.value.code == pdf.ERROR_INSUFFICIENT_DISK


def test_unsupported_language_is_explicit(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "foreign.pdf", ["bonjour monde maison livre voiture soleil arbre musique voyage"])
    with pytest.raises(PdfAnalysisError) as error:
        analyze_pdf(path)
    assert error.value.code == pdf.ERROR_UNSUPPORTED_LANGUAGE


def test_cleanup_preserves_paragraphs_and_dehyphenates_only_lowercase() -> None:
    pages = [PageEvidence(1, "First para line\n\nSecond para\nword-\ncontinued\nUpper-\nCase", "text", False)]
    cleaned, _, _ = pdf._clean_pages(pages)
    assert "First para line\n\nSecond para" in cleaned
    assert "wordcontinued" in cleaned
    assert "Upper-\nCase" in cleaned


def test_leading_scanned_page_is_explicitly_warned(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "decorative.pdf", [None, "The English text is sufficient for analysis and review."])
    result = analyze_pdf(path)
    assert any("scanned" in warning and "decorative" in warning for warning in result["warnings"])


def test_interior_mixed_page_with_few_words_requires_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "mixed.pdf", ["The English text is sufficient for review.", "A few words", "More English text is sufficient for review."])
    reader = pdf.PdfReader(str(path), strict=False)
    pages = [
        PageEvidence(1, "The English text is sufficient for review.", "text", False),
        PageEvidence(2, "A few words", "mixed", True),
        PageEvidence(3, "More English text is sufficient for review.", "text", False),
    ]
    monkeypatch.setattr(pdf, "_extract_pages", lambda _path: (reader, pages))
    with pytest.raises(PdfAnalysisError) as error:
        analyze_pdf(path)
    assert error.value.code == pdf.ERROR_OCR_REQUIRED


def test_header_valid_structural_corruption_is_parser_failure(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a valid object tree\n%%EOF\n")
    with pytest.raises(PdfAnalysisError) as error:
        analyze_pdf(path)
    assert error.value.code == pdf.ERROR_PARSER_FAILURE


@pytest.mark.parametrize("interior", ["42", "* * *"])
def test_interior_page_without_letters_is_blank(tmp_path: Path, interior: str) -> None:
    path = make_pdf(
        tmp_path / "blank.pdf",
        [
            "The English text is sufficient for analysis and review of this book.",
            interior,
            "A different English passage continues the story for the reader.",
        ],
    )
    result = analyze_pdf(path)
    assert result["page_classifications"][1]["classification"] == "blank"


def test_interior_unsupported_page_names_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(
        tmp_path / "unsupported.pdf",
        [
            "The English text is sufficient for review.",
            "unsupported",
            "More English text is sufficient for review.",
        ],
    )
    reader = pdf.PdfReader(str(path), strict=False)
    pages = [
        PageEvidence(1, "The English text is sufficient for review.", "text", False),
        PageEvidence(2, "", "unsupported", False, ("text extraction failed",)),
        PageEvidence(3, "More English text is sufficient for review.", "text", False),
    ]
    monkeypatch.setattr(pdf, "_extract_pages", lambda _path: (reader, pages))
    with pytest.raises(PdfAnalysisError) as error:
        analyze_pdf(path)
    assert error.value.code == pdf.ERROR_PARSER_FAILURE
    assert error.value.details["pages"] == [2]
    assert "2" in error.value.message


def test_long_pdf_has_no_page_limit(tmp_path: Path) -> None:
    pages = [f"Chapter {n}\nThe quick brown fox jumps over the lazy dog on page {n}." for n in range(1, 2002)]
    path = make_pdf(tmp_path / "long.pdf", pages)
    result = analyze_pdf(path, layout_warnings=False)
    assert result["page_count"] == 2001


def test_layout_warnings_can_be_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "normal.pdf", ["Chapter 1\nThe quick brown fox jumps over the lazy dog.", "Chapter 2\nAnother page for testing."])
    monkeypatch.setattr(pdf, "_layout_warnings", lambda *args, **kwargs: pytest.fail("_layout_warnings should not be called"))
    result = analyze_pdf(path, layout_warnings=False)
    assert result["page_count"] == 2


def test_pdfplumber_pages_are_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "test.pdf"

    class FakePage:
        def __init__(self) -> None:
            self.chars: list[dict[str, Any]] = []
            self.width = 612.0
            self.height = 792.0
            self.closed = False

        def extract_words(self) -> list[dict[str, Any]]:
            return []

        def extract_text_lines(self) -> list[dict[str, Any]]:
            return []

        def find_tables(self) -> list[Any]:
            return []

        def extract_text(self) -> str:
            return ""

        def close(self) -> None:
            self.closed = True

    created_pages: list[FakePage] = []

    class FakePdf:
        def __init__(self) -> None:
            self.pages = [FakePage(), FakePage()]
            created_pages.extend(self.pages)

        def __enter__(self) -> FakePdf:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(pdf.pdfplumber, "open", lambda _path: FakePdf())
    pdf._layout_heading_candidates(path, 2)
    assert len(created_pages) == 2
    assert all(page.closed for page in created_pages)
    pdf._layout_warnings(path, 2)
    assert len(created_pages) == 4
    assert all(page.closed for page in created_pages)


def _assert_page_clean_mapping(text: str, mapping: list[dict[str, int]], page2_expected: str) -> None:
    assert len(mapping) == 2
    assert mapping[0]["source_page"] == 1
    assert mapping[1]["source_page"] == 2
    assert mapping[0]["cleaned_start"] == 0
    assert mapping[0]["cleaned_start"] < mapping[0]["cleaned_end"]
    assert mapping[0]["cleaned_end"] <= mapping[1]["cleaned_start"]
    assert mapping[1]["cleaned_start"] < mapping[1]["cleaned_end"]
    assert mapping[-1]["cleaned_end"] == len(text)
    assert text[mapping[1]["cleaned_start"]:mapping[1]["cleaned_end"]] == page2_expected


def test_clean_pages_lowercase_continuation() -> None:
    p1 = "It was a bright day and the wealthy families hired"
    p2 = "private tutors for their children."
    text, mapping, _ = pdf._clean_pages([PageEvidence(1, p1, "text", False), PageEvidence(2, p2, "text", False)])
    assert text == "It was a bright day and the wealthy families hired private tutors for their children."
    assert "\n\n" not in text
    _assert_page_clean_mapping(text, mapping, p2)
    assert text[mapping[0]["cleaned_start"]:mapping[0]["cleaned_end"]] == p1


def test_clean_pages_lowercase_word_next_capitalized() -> None:
    p1 = "He studied the details of your"
    p2 = "Appraisal before the trial began."
    text, mapping, _ = pdf._clean_pages([PageEvidence(1, p1, "text", False), PageEvidence(2, p2, "text", False)])
    assert text == "He studied the details of your Appraisal before the trial began."
    assert "\n\n" not in text
    _assert_page_clean_mapping(text, mapping, p2)
    assert text[mapping[0]["cleaned_start"]:mapping[0]["cleaned_end"]] == p1


def test_clean_pages_ends_with_comma() -> None:
    p1 = "When the bell rang at noon,"
    p2 = "Sunny left the hall."
    text, mapping, _ = pdf._clean_pages([PageEvidence(1, p1, "text", False), PageEvidence(2, p2, "text", False)])
    assert text == "When the bell rang at noon, Sunny left the hall."
    assert "\n\n" not in text
    _assert_page_clean_mapping(text, mapping, p2)
    assert text[mapping[0]["cleaned_start"]:mapping[0]["cleaned_end"]] == p1


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        (
            "He slept well.",
            "Early in the morning he woke.",
            "He slept well.\n\nEarly in the morning he woke.",
        ),
        (
            "He slept well.\n***",
            "Early in the morning he woke.",
            "He slept well.\n***\n\nEarly in the morning he woke.",
        ),
        (
            "and so the long night ended",
            "Chapter 2\nThe next day came.",
            "and so the long night ended\n\nChapter 2\nThe next day came.",
        ),
        (
            "The end came.\nChapter 7: the fall",
            "It began at night.",
            "The end came.\nChapter 7: the fall\n\nIt began at night.",
        ),
    ],
)
def test_clean_pages_keeps_double_newline(p1: str, p2: str, expected: str) -> None:
    text, mapping, _ = pdf._clean_pages([PageEvidence(1, p1, "text", False), PageEvidence(2, p2, "text", False)])
    assert text == expected
    _assert_page_clean_mapping(text, mapping, p2)
    assert text[mapping[0]["cleaned_start"]:mapping[0]["cleaned_end"]] == p1


def test_clean_pages_hyphenation_across_pages() -> None:
    p1 = "It was an extraordinary compre-"
    p2 = "hension of the matter."
    text, mapping, _ = pdf._clean_pages([PageEvidence(1, p1, "text", False), PageEvidence(2, p2, "text", False)])
    assert "comprehension" in text
    assert text == "It was an extraordinary comprehension of the matter."
    assert text[0:mapping[0]["cleaned_end"]] == "It was an extraordinary compre"
    assert mapping[1]["cleaned_start"] == mapping[0]["cleaned_end"]
    _assert_page_clean_mapping(text, mapping, p2)


def test_page_separator_rules() -> None:
    assert pdf._page_separator("families hired", "private tutors") == " "
    assert pdf._page_separator("details of your", "Appraisal before") == " "
    assert pdf._page_separator("noon,", "Sunny left") == " "
    assert pdf._page_separator("He slept well.", "Early in") == "\n\n"
    assert pdf._page_separator("He slept well.\n***", "Early in") == "\n\n"
    assert pdf._page_separator("night ended", "Chapter 2\nNext") == "\n\n"
    assert pdf._page_separator("End.\nChapter 7: the fall", "It began") == "\n\n"
    assert pdf._page_separator("compre-", "hension") == ""
    assert pdf._page_separator("Upper-", "Case") == "\n\n"
    assert pdf._page_separator('said, "Wait!"', "Next line") == "\n\n"
    assert pdf._page_separator("Waiting…", "Next line") == "\n\n"


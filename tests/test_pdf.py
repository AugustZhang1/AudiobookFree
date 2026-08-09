from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
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


def test_limits_disk_and_page_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "book.pdf", ["The English text is sufficient for analysis and review."])
    real_disk_usage = shutil.disk_usage
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 1)
    with pytest.raises(PdfAnalysisError) as size_error:
        analyze_pdf(path)
    assert size_error.value.code == pdf.ERROR_SIZE_LIMIT
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 100 * 1024 * 1024)
    monkeypatch.setattr(pdf.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(0, 0, 0))
    with pytest.raises(PdfAnalysisError) as disk_error:
        analyze_pdf(path)
    assert disk_error.value.code == pdf.ERROR_INSUFFICIENT_DISK

    monkeypatch.setattr(pdf.shutil, "disk_usage", real_disk_usage)
    monkeypatch.setattr(pdf, "MAX_PAGES", 0)
    with pytest.raises(PdfAnalysisError) as page_error:
        analyze_pdf(path)
    assert page_error.value.code == pdf.ERROR_PAGE_LIMIT


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

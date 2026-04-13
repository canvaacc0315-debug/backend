import tempfile
import os

# Lazy-loaded globals — initialized on first use so the app
# can start even if paddleocr / pdf2image aren't installed.
_ocr = None
_convert_from_path = None


def _ensure_ocr():
    """Lazily import PaddleOCR and pdf2image on first call."""
    global _ocr, _convert_from_path
    if _ocr is None:
        from paddleocr import PaddleOCR
        from pdf2image import convert_from_path

        _ocr = PaddleOCR(use_angle_cls=True, lang="en")
        _convert_from_path = convert_from_path


def run_ocr_on_page(pdf_path: str, page_number: int) -> str:
    """
    Run OCR on a single page of a PDF.
    page_number is 0-indexed (matching PyMuPDF convention).
    pdf2image uses 1-indexed pages internally, so we convert.
    Returns the extracted text for that page, or empty string on failure.
    """
    try:
        _ensure_ocr()

        # pdf2image uses 1-based page numbers
        images = _convert_from_path(
            pdf_path,
            first_page=page_number + 1,
            last_page=page_number + 1,
        )

        if not images:
            return ""

        img = images[0]
        page_text = ""

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            result = _ocr.ocr(f.name, cls=True)
            os.unlink(f.name)

        # result can be [None] for blank pages
        if result and result[0]:
            for line in result[0]:
                page_text += line[1][0] + "\n"

        return page_text.strip()

    except Exception as e:
        print(f"OCR failed on page {page_number}: {e}")
        return ""


def run_ocr_on_pdf(pdf_path: str) -> str:
    """Run OCR on the entire PDF. Returns all extracted text."""
    try:
        _ensure_ocr()
    except Exception as e:
        print(f"OCR unavailable: {e}")
        return ""

    images = _convert_from_path(pdf_path)
    full_text = ""

    for img in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            result = _ocr.ocr(f.name, cls=True)
            os.unlink(f.name)

        if result and result[0]:
            for line in result[0]:
                full_text += line[1][0] + "\n"

        full_text += "\n"

    return full_text

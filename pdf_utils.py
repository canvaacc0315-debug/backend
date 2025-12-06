import os
from typing import List
import fitz  # PyMuPDF
from fpdf import FPDF
from fastapi import UploadFile


def save_uploaded_pdf(file: UploadFile, upload_dir: str, pdf_id: str) -> str:
    """
    Save uploaded PDF to disk and return file path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{pdf_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def edit_pdf_add_text(
    input_pdf_path: str,
    output_pdf_path: str,
    page_number: int,
    text: str,
    x: float,
    y: float,
    font_size: int = 12,
):
    """
    Add text to the given PDF page at coordinates (x, y).
    """
    doc = fitz.open(input_pdf_path)

    if page_number < 0 or page_number >= len(doc):
        raise ValueError("Invalid page number")

    page = doc[page_number]
    page.insert_text(
        (x, y),
        text,
        fontsize=font_size,
    )

    doc.save(output_pdf_path)
    doc.close()


def edit_pdf_add_image(
    input_pdf_path: str,
    output_pdf_path: str,
    page_number: int,
    image_path: str,
    x: float,
    y: float,
    width: float | None = None,
    height: float | None = None,
):
    """
    Add an image to a given page at (x, y) with optional width/height.
    """
    doc = fitz.open(input_pdf_path)

    if page_number < 0 or page_number >= len(doc):
        raise ValueError("Invalid page number")

    page = doc[page_number]

    # Set default size if not provided
    w = width or 200
    h = height or 200

    rect = fitz.Rect(x, y, x + w, y + h)
    page.insert_image(rect, filename=image_path)

    doc.save(output_pdf_path)
    doc.close()


def create_custom_pdf_with_images(
    output_pdf_path: str,
    title: str,
    body_text: str,
    image_paths: List[str],
):
    """
    Create a new PDF with a title, body text, and optional images.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)

    pdf.ln(5)

    # Body text
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, body_text)

    # Images (each on its own page)
    for img_path in image_paths:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Image", ln=True)
        pdf.ln(5)
        # Place image roughly full width
        pdf.image(img_path, x=10, y=25, w=180)

    pdf.output(output_pdf_path)

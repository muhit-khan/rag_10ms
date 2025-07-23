"""
PDF text extraction logic
"""
import pdfminer.high_level
import pytesseract
from PIL import Image
import os
from pathlib import Path
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = pdfminer.high_level.extract_text(pdf_path, laparams=None)
        if text.strip():
            return text
    except Exception:
        pass
    # Fallback to OCR
    images = pdf_to_images(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img, lang="ben") for img in images])
    return text

def pdf_to_images(pdf_path: str):
    return convert_from_path(pdf_path)

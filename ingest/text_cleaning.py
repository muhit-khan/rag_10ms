"""
Text cleaning and normalization logic
"""
import re
import unicodedata
import logging
from typing import Any

logger = logging.getLogger("ingest.text_cleaning")

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    - Normalize Unicode (NFKC)
    - Remove headers and footers
    - Remove page numbers
    - Remove excessive whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r"^\s*\d+\s*$", line):
            continue
        if re.match(r"^\s*(header|footer|www\.|http|copyright)", line.lower()):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def save_text_to_file(text: str, filename: str):
    """
    Save the extracted text to a .txt file in data/ingest/simple directory.
    Args:
        text: The text to save.
        filename: The base filename (without extension) to use for the .txt file.
    """
    import os
    from pathlib import Path
    output_dir = Path("data/ingest/simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{filename}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved extracted text to {file_path}")

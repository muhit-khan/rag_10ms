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

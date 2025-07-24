"""
Metadata extraction logic
"""
import re
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger("ingest.metadata_extraction")

def extract_metadata(pdf_path: Path, text: str) -> Dict[str, str]:
    """
    Extract metadata from PDF path and content.
    Attempts to identify chapter, section, and page information.
    Falls back to filename-based metadata if extraction fails.
    """
    metadata = {
        "source": pdf_path.name,
        "path": str(pdf_path),
    }
    filename = pdf_path.stem
    chapter_match = re.search(r"chapter[_-]?(\d+)", filename, re.IGNORECASE)
    if chapter_match:
        metadata["chapter"] = chapter_match.group(1)
    section_match = re.search(r"section[_-]?(\d+)", filename, re.IGNORECASE)
    if section_match:
        metadata["section"] = section_match.group(1)
    page_matches = re.findall(r"page\s+(\d+)", text, re.IGNORECASE)
    if page_matches:
        metadata["page"] = page_matches[0]
    return metadata

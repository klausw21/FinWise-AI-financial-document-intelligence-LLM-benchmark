"""PDF text-layer extraction (born-digital docs in this dataset have a real text layer).

`pdftotext -layout` (poppler) preserves column alignment, which the transaction-table
parser relies on. Falls back to pdfplumber if the CLI is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_PDFTOTEXT = shutil.which("pdftotext")


def pdf_text_layout(pdf_path: str | Path) -> str:
    """Return the text layer of a PDF, preserving layout (columns/rows aligned)."""
    pdf_path = str(pdf_path)
    if _PDFTOTEXT:
        out = subprocess.run(
            [_PDFTOTEXT, "-layout", pdf_path, "-"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    # Fallback: pdfplumber
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text(layout=True) or "")
    return "\n".join(parts)


def pdf_text_plain(pdf_path: str | Path) -> str:
    """Plain reading-order text (no layout), useful as an LLM text input."""
    pdf_path = str(pdf_path)
    if _PDFTOTEXT:
        out = subprocess.run(
            [_PDFTOTEXT, pdf_path, "-"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

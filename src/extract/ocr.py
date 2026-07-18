"""Tesseract OCR over a page image (input for the OCR-based methods M1/M2)."""
from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image


def ocr_image(image_path: str | Path, psm: int = 4) -> str:
    """OCR a page image to text. psm=4 = 'assume a single column of text of
    variable sizes', which suits statement/invoice layouts reasonably well."""
    img = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(img, config=f"--psm {psm}")


if __name__ == "__main__":
    from src import dataset as ds

    d = ds.list_docs("bank_statement")[0]
    txt = ocr_image(d.image_path)
    print(f"OCR of {d.image_path.name} ({len(txt)} chars):\n")
    print(txt[:1200])

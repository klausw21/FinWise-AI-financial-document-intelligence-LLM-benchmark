"""Heuristic document-type detection (free) so uploads don't need a manual pick.

Scores keyword hits per type over the document text (PDF layer or OCR) plus the
filename; the user can still override in the UI.
"""
from __future__ import annotations

from pathlib import Path

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "credit_card_statement": ["credit card", "card statement", "minimum payment",
                              "new balance", "previous balance", "credit limit", "cardholder"],
    "bank_statement": ["bank statement", "account holder", "opening balance",
                       "closing balance", "account number"],
    "invoice": ["invoice", "bill to", "invoice no", "invoice number", "amount due",
                "balance due", "po number"],
    "receipt": ["receipt", "merchant", "subtotal", "cashier", "thank you for"],
}


def detect_scored(text: str | None = None, filename: str | None = None) -> tuple[str, int]:
    """(best_type, keyword_hits). hits==0 means the guess is only the fallback —
    the caller should ask the user to confirm rather than trust it."""
    blob = f"{filename or ''} {text or ''}".lower()
    scores = {t: sum(k in blob for k in kws) for t, kws in _TYPE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else ("invoice", 0)


def detect_type(text: str | None = None, filename: str | None = None) -> str:
    return detect_scored(text=text, filename=filename)[0]


def _text_from_paths(pdf_path=None, image_path=None) -> str:
    text = ""
    if pdf_path:
        try:
            from src.extract.pdf_text import pdf_text_plain
            text = pdf_text_plain(pdf_path)
        except Exception:
            text = ""
    if not text.strip() and image_path:
        try:
            from src.extract.ocr import ocr_image
            text = ocr_image(image_path)
        except Exception:
            text = ""
    return text


def detect_scored_from_paths(pdf_path=None, image_path=None, filename=None) -> tuple[str, int]:
    text = _text_from_paths(pdf_path, image_path)
    name = filename or (Path(pdf_path or image_path).name if (pdf_path or image_path) else None)
    return detect_scored(text=text, filename=name)


def detect_from_paths(pdf_path=None, image_path=None, filename=None) -> str:
    return detect_scored_from_paths(pdf_path, image_path, filename)[0]


if __name__ == "__main__":
    from src import dataset as ds
    from src.extract.pdf_text import pdf_text_plain

    ok = 0
    types = ds.available_types()
    for t in types:
        d = ds.list_docs(t)[0]
        guess = detect_type(text=pdf_text_plain(d.pdf_path), filename=d.stem)
        hit = "✅" if guess == t else "❌"
        ok += guess == t
        print(f"  {t:22s} -> {guess:22s} {hit}")
    print(f"\n{ok}/{len(types)} correct")

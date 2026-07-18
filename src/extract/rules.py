"""Rule-based extraction (the traditional OCR+regex baseline, no LLM).

Parses text (from pdftotext -layout or Tesseract OCR) into schema-shaped dicts.
Deliberately simple: strong on labelled header fields and the bank/credit
transaction tables, best-effort on invoice/receipt line items. This is the
baseline the LLM methods are meant to beat.
"""
from __future__ import annotations

import re

_MONEY = r"\$?-?[\d,]+\.?\d*"


def _f(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    if s in ("", "-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _kv(text: str, label: str) -> str | None:
    """Value after 'Label:' on the same line."""
    m = re.search(rf"{re.escape(label)}\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).split("  ")[0].strip().strip("|").strip()


def _clean_row(line: str) -> str:
    return re.sub(r"\s{2,}", " ", line.replace("|", " ")).strip()


# ---------- bank statement ----------
_BANK_ROW = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(.*?)\s+(\$?[\d,]+\.\d{2}|-)\s+(\$?[\d,]+\.\d{2}|-)\s+(\$?[\d,]+\.\d{2}|-)\s*$"
)


def parse_bank(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        m = _BANK_ROW.match(_clean_row(line))
        if m:
            rows.append({
                "date": m.group(1), "description": m.group(2).strip(),
                "debit": _f(m.group(3)), "credit": _f(m.group(4)), "balance": _f(m.group(5)),
            })
    period = _kv(text, "Period") or ""
    pm = re.search(r"(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})", period)
    return {
        "category": "bank_statement",
        "statement_id": _kv(text, "Statement ID"),
        "account_holder": _kv(text, "Account Holder"),
        "account_number_masked": _kv(text, "Account Number"),
        "period_start": pm.group(1) if pm else None,
        "period_end": pm.group(2) if pm else None,
        "opening_balance": _f(_kv(text, "Opening Balance")),
        "closing_balance": _f(_kv(text, "Closing Balance")),
        "transactions": len(rows),
        "transaction_rows": rows,
    }


# ---------- credit card ----------
_CC_ROW = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(.*?)\s+(charge|payment)\s+(\$?[\d,]+\.\d{2})\s*$", re.IGNORECASE
)


def parse_credit(text: str) -> dict:
    txns = []
    for line in text.splitlines():
        m = _CC_ROW.match(_clean_row(line))
        if m:
            txns.append({"date": m.group(1), "description": m.group(2).strip(),
                         "type": m.group(3).lower(), "amount": _f(m.group(4))})
    period = _kv(text, "Period") or ""
    pm = re.search(r"(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})", period)
    return {
        "category": "credit_card_statement",
        "statement_id": _kv(text, "Statement ID"),
        "card_number_masked": _kv(text, "Card Number"),
        "cardholder": _kv(text, "Cardholder"),
        "period_start": pm.group(1) if pm else None,
        "period_end": pm.group(2) if pm else None,
        "previous_balance": _f(_kv(text, "Previous Balance")),
        "new_balance": _f(_kv(text, "New Balance")),
        "minimum_payment": _f(_kv(text, "Minimum Payment")),
        "payment_due_date": _kv(text, "Payment Due Date"),
        "credit_limit": _f(_kv(text, "Credit Limit")),
        "transactions": txns,
    }


# ---------- receipt ----------
def parse_receipt(text: str) -> dict:
    return {
        "category": "receipt",
        "receipt_id": _kv(text, "Receipt") or _kv(text, "Receipt ID"),
        "merchant_name": _kv(text, "Merchant"),
        "date": _kv(text, "Date"),
        "subtotal": _f(_kv(text, "Subtotal")),
        "tax": _f(_kv(text, "Tax")),
        "discount": _f(_kv(text, "Discount")),
        "total": _f(_kv(text, "Total")),
        "currency": _kv(text, "Currency"),
        "items": [],  # item table parsing is left to the LLM methods (baseline: header only)
    }


# ---------- invoice ----------
def parse_invoice(text: str) -> dict:
    return {
        "invoice_id": _kv(text, "Invoice No") or _kv(text, "Invoice Number"),
        "invoice_number": _kv(text, "Invoice No") or _kv(text, "Invoice Number"),
        "company_name": None,
        "customer_name": _kv(text, "Bill To"),
        "invoice_date": _kv(text, "Invoice Date"),
        "due_date": _kv(text, "Due Date"),
        "payment_terms": _kv(text, "Payment Terms"),
        "po_number": _kv(text, "PO Number"),
        "subtotal": _f(_kv(text, "Subtotal")),
        "tax": _f(_kv(text, "Tax")),
        "shipping": _f(_kv(text, "Shipping")),
        "total": _f(_kv(text, "Total")),
        "amount_paid": _f(_kv(text, "Amount Paid")),
        "balance_due": _f(_kv(text, "Balance Due")),
        "status": _kv(text, "Status"),
        "items": [],
    }


PARSERS = {
    "bank_statement": parse_bank,
    "credit_card_statement": parse_credit,
    "receipt": parse_receipt,
    "invoice": parse_invoice,
}


def parse(doc_type: str, text: str) -> dict:
    return PARSERS[doc_type](text)


if __name__ == "__main__":
    from src import dataset as ds
    from src.extract.pdf_text import pdf_text_layout

    for t in ("bank_statement", "credit_card_statement", "receipt"):
        docs = ds.list_docs(t)
        if not docs:
            continue
        d = docs[0]
        parsed = parse(t, pdf_text_layout(d.pdf_path))
        n_rows = len(parsed.get("transaction_rows", parsed.get("transactions", [])) or []) \
            if isinstance(parsed.get("transaction_rows", parsed.get("transactions")), list) else "-"
        print(f"[{t}] parsed keys={len(parsed)}  rows/items={n_rows}")
        print("   sample:", {k: parsed[k] for k in list(parsed)[1:5]})

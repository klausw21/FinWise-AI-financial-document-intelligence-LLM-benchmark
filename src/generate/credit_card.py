"""Synthetic credit-card statement generator (fills the type missing from the dataset).

Produces the same 4-subfolder layout as the rest of the dataset
(labels/ image/ texts/ pdfs/) with matching conventions:
  labels/<stem>.json, image/<stem>_page_1.png (A4 @200 DPI), texts/<stem>.txt, pdfs/<stem>.pdf

Unlike the provided bank statements, the label carries the FULL transaction list
(reconciling: previous_balance + sum(charges) - sum(payments) == new_balance),
so it is clean transaction-level gold with no PDF parsing needed.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import fitz  # PyMuPDF (render PDF -> PNG)
from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from src import dataset as ds

DOC_TYPE = "credit_card_statement"
OUT_ROOT = ds.DATASET_ROOT / DOC_TYPE
RENDER_DPI = 200
BANNER = "SYNTHETIC DOCUMENT - NOT A REAL GOVERNMENT, BANK, MEDICAL, OR LEGAL DOCUMENT"

CHARGE_MERCHANTS = [
    "Amazon", "Whole Foods", "Shell Gas", "Starbucks", "Uber", "Netflix",
    "Apple Store", "Target", "Delta Airlines", "Best Buy", "Costco",
    "Spotify", "Home Depot", "Walmart", "IKEA", "Restaurant", "Pharmacy",
]
PAYMENT_DESCS = ["Payment - Thank You", "Online Payment", "AutoPay Payment"]
ISSUERS = ["Summit", "Meridian", "Cascade", "Atlas", "Beacon", "Horizon"]


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def _gen_record(idx: int, fake: Faker, rng: random.Random) -> dict:
    stem = f"{DOC_TYPE}_{idx:04d}"
    cardholder = fake.name()
    issuer = f"{rng.choice(ISSUERS)} Bank"
    card_last4 = f"{rng.randint(0, 9999):04d}"
    period_end = fake.date_between(start_date="-1y", end_date="today")
    period_start = period_end - timedelta(days=30)
    due = period_end + timedelta(days=21)
    credit_limit = float(rng.choice([2000, 5000, 8000, 10000, 15000, 25000]))
    previous_balance = _round2(rng.uniform(0, credit_limit * 0.4))

    n = rng.randint(6, 16)
    txns = []
    charges_total = 0.0
    payments_total = 0.0
    for _ in range(n):
        d = fake.date_between(start_date=period_start, end_date=period_end)
        if rng.random() < 0.25:
            amt = _round2(rng.uniform(50, max(60, previous_balance)))
            txns.append({"date": d.isoformat(), "description": rng.choice(PAYMENT_DESCS),
                         "amount": amt, "type": "payment"})
            payments_total += amt
        else:
            amt = _round2(rng.uniform(5, 900))
            txns.append({"date": d.isoformat(), "description": rng.choice(CHARGE_MERCHANTS),
                         "amount": amt, "type": "charge"})
            charges_total += amt
    txns.sort(key=lambda t: t["date"])

    new_balance = _round2(previous_balance + charges_total - payments_total)
    minimum_payment = _round2(max(25.0, new_balance * 0.03)) if new_balance > 0 else 0.0

    return {
        "category": DOC_TYPE,
        "statement_id": f"CCS-{fake.bothify('?????????').upper()}",
        "card_number_masked": f"****{card_last4}",
        "cardholder": cardholder,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "previous_balance": previous_balance,
        "new_balance": new_balance,
        "minimum_payment": minimum_payment,
        "payment_due_date": due.isoformat(),
        "credit_limit": credit_limit,
        "transactions": txns,
        "sample_id": stem,
        "_issuer": issuer,  # render-only, stripped from label
    }


def _render_pdf(rec: dict, pdf_path: Path) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    y = H - 1.5 * cm
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.grey)
    c.drawCentredString(W / 2, y, BANNER)
    c.setFillColor(colors.black)
    y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(W / 2, y, "Credit Card Statement")
    y -= 0.9 * cm
    c.setFont("Helvetica", 10)
    left = 2 * cm

    def line(label, value):
        nonlocal y
        c.setFont("Helvetica-Bold", 10)
        c.drawString(left, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(left + 4.2 * cm, y, str(value))
        y -= 0.52 * cm

    line("Issuer", rec["_issuer"])
    line("Statement ID", rec["statement_id"])
    line("Cardholder", rec["cardholder"])
    line("Card Number", rec["card_number_masked"])
    line("Period", f'{rec["period_start"]} to {rec["period_end"]}')
    line("Payment Due Date", rec["payment_due_date"])
    line("Credit Limit", f'${rec["credit_limit"]:.2f}')
    line("Previous Balance", f'${rec["previous_balance"]:.2f}')
    line("New Balance", f'${rec["new_balance"]:.2f}')
    line("Minimum Payment", f'${rec["minimum_payment"]:.2f}')

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 10)
    cols = [left, left + 3.2 * cm, left + 10.5 * cm, left + 13.0 * cm]
    for x, h in zip(cols, ["Date", "Description", "Type", "Amount"]):
        c.drawString(x, y, h)
    y -= 0.15 * cm
    c.line(left, y, W - 2 * cm, y)
    y -= 0.45 * cm
    c.setFont("Helvetica", 9)
    for t in rec["transactions"]:
        c.drawString(cols[0], y, t["date"])
        c.drawString(cols[1], y, t["description"])
        c.drawString(cols[2], y, t["type"])
        c.drawRightString(W - 2 * cm, y, f'${t["amount"]:.2f}')
        y -= 0.5 * cm
    y -= 0.3 * cm
    c.setFont("Helvetica-Oblique", 7)
    c.setFillColor(colors.grey)
    c.drawString(left, y, "This is a synthetic credit card statement generated for OCR / "
                          "document-AI experiments. No real person or account data is used.")
    c.showPage()
    c.save()


def _render_png(pdf_path: Path, png_path: Path, dpi: int = RENDER_DPI) -> None:
    doc = fitz.open(str(pdf_path))
    pix = doc[0].get_pixmap(dpi=dpi)
    pix.save(str(png_path))
    doc.close()


def _write_text(rec: dict, txt_path: Path) -> None:
    lines = [
        f"Credit Card Statement {rec['statement_id']}",
        f"Cardholder: {rec['cardholder']}",
        f"New Balance: ${rec['new_balance']:.2f}",
        "",
        json.dumps({k: v for k, v in rec.items() if k != "_issuer"}, ensure_ascii=False, indent=2),
    ]
    txt_path.write_text("\n".join(lines))


def generate(n: int = 200, seed: int = 20260701, progress=None) -> None:
    """Generate n credit-card statements. `progress(i, n)` is called per document
    (used by the UI); prints every 50 when no callback is given."""
    for sub in ("labels", "image", "texts", "pdfs"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    for i in range(1, n + 1):
        rec = _gen_record(i, fake, rng)
        stem = rec["sample_id"]
        pdf_path = OUT_ROOT / "pdfs" / f"{stem}.pdf"
        _render_pdf(rec, pdf_path)
        _render_png(pdf_path, OUT_ROOT / "image" / f"{stem}_page_1.png")
        _write_text(rec, OUT_ROOT / "texts" / f"{stem}.txt")
        label = {k: v for k, v in rec.items() if k != "_issuer"}
        (OUT_ROOT / "labels" / f"{stem}.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2))
        if progress:
            progress(i, n)
        elif i % 50 == 0:
            print(f"  generated {i}/{n}")
    if not progress:
        print(f"done: {n} credit_card_statement docs -> {OUT_ROOT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    generate(n)

"""Normalization + business logic run on extracted JSON.

Normalization (dates, money, currency, strings) is shared by metrics and the demo.
Business logic (reconciliation, duplicate/anomaly detection, cash-flow) powers the
demo; reconciliation additionally feeds the transaction-level benchmark.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

# ---------------- normalization ----------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d",
]


def norm_date(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # leave as-is if unparseable


def norm_money(x: Any) -> Optional[float]:
    """'$1,234.56' / 'QAR 1,234.56' / 1234.56 -> 1234.56."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = re.sub(r"[A-Za-z$,€£¥]", "", s).strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def norm_str(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# ---------------- reconciliation ----------------
def reconcile_ledger(rows, opening: float, get_delta, closing: float, tol: float = 0.011) -> dict:
    """Generic running-balance reconciliation.

    get_delta(row) -> signed change to the balance for that row.
    Checks each stated balance (if present) and the final == closing.
    """
    running = float(opening)
    per_row_ok = []
    for r in rows:
        running += get_delta(r)
        stated = r.get("balance")
        if stated is not None:
            per_row_ok.append(abs(float(stated) - running) <= tol)
            running = float(stated)
    final_ok = abs(running - float(closing)) <= tol
    return {"reconciles": bool(rows) and final_ok and all(per_row_ok) if per_row_ok else final_ok,
            "final_balance": round(running, 2), "expected_closing": round(float(closing), 2),
            "per_row_ok": per_row_ok}


def reconcile_bank(data: dict) -> dict:
    rows = data.get("transaction_rows") or []
    op, cl = data.get("opening_balance"), data.get("closing_balance")
    if op is None or cl is None:
        return {"reconciles": False, "reason": "missing opening/closing"}
    return reconcile_ledger(rows, op, lambda r: (r.get("credit") or 0) - (r.get("debit") or 0), cl)


def reconcile_credit(data: dict) -> dict:
    txns = data.get("transactions") or []
    prev, new = data.get("previous_balance"), data.get("new_balance")
    if prev is None or new is None or not isinstance(txns, list):
        return {"reconciles": False, "reason": "missing balances"}
    chg = sum((t.get("amount") or 0) for t in txns if t.get("type") == "charge")
    pay = sum((t.get("amount") or 0) for t in txns if t.get("type") == "payment")
    calc = round(prev + chg - pay, 2)
    return {"reconciles": abs(calc - new) <= 0.011, "computed_new_balance": calc,
            "stated_new_balance": new, "charges": round(chg, 2), "payments": round(pay, 2)}


def reconcile(data: dict, doc_type: str) -> dict:
    if doc_type == "bank_statement":
        return reconcile_bank(data)
    if doc_type == "credit_card_statement":
        return reconcile_credit(data)
    return {"reconciles": None, "reason": "no ledger for this type"}


# ---------------- duplicate / anomaly / cash-flow (demo features) ----------------
def _row_amount(r: dict) -> float:
    if "amount" in r:
        return abs(norm_money(r.get("amount")) or 0)
    return abs((r.get("credit") or 0) - (r.get("debit") or 0))


def detect_duplicates(rows: list[dict]) -> list[tuple[int, int]]:
    """Same date + same |amount| + similar description => candidate duplicates."""
    dups = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if (rows[i].get("date") == rows[j].get("date")
                    and abs(_row_amount(rows[i]) - _row_amount(rows[j])) < 0.011
                    and norm_str(rows[i].get("description")) == norm_str(rows[j].get("description"))):
                dups.append((i, j))
    return dups


def detect_anomalies(rows: list[dict], z: float = 2.5) -> list[int]:
    amts = [_row_amount(r) for r in rows]
    if len(amts) < 3:
        return []
    mean = sum(amts) / len(amts)
    var = sum((a - mean) ** 2 for a in amts) / len(amts)
    std = var ** 0.5 or 1.0
    return [i for i, a in enumerate(amts) if abs(a - mean) / std > z]


def cashflow_summary(data: dict, doc_type: str) -> dict:
    """Inflow/outflow/net for a ledger-bearing document."""
    if doc_type == "bank_statement":
        rows = data.get("transaction_rows") or []
        inflow = sum((r.get("credit") or 0) for r in rows)
        outflow = sum((r.get("debit") or 0) for r in rows)
    elif doc_type == "credit_card_statement":
        txns = data.get("transactions") or []
        outflow = sum((t.get("amount") or 0) for t in txns if t.get("type") == "charge")
        inflow = sum((t.get("amount") or 0) for t in txns if t.get("type") == "payment")
    else:
        return {}
    return {"inflow": round(inflow, 2), "outflow": round(outflow, 2),
            "net": round(inflow - outflow, 2)}


if __name__ == "__main__":
    from src import dataset as ds
    from src.gold.tx_from_pdf import parse_bank_rows

    d = ds.list_docs("bank_statement")[0]
    label = d.load_label()
    rows = [r.model_dump() for r in parse_bank_rows(d.pdf_path)]
    data = {**label, "transaction_rows": rows}
    print("reconcile_bank:", reconcile_bank(data))
    print("cashflow      :", cashflow_summary(data, "bank_statement"))
    print("duplicates    :", detect_duplicates(rows))
    print("anomalies idx :", detect_anomalies(rows))
    cc = ds.list_docs("credit_card_statement")[0].load_label()
    print("reconcile_credit:", reconcile_credit(cc))

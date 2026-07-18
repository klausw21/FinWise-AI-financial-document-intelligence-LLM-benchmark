"""Structured financial insights — the one-line tip upgraded to explainable cards.

Every card has a type, a severity, an evidence trail (real rows/figures from the
document), an estimated impact amount, and a suggested action. Grounded only in the
analysis bundle (src/postprocess.py) — no ungrounded "advice", no over-certain
imperatives (we say "review", never "cancel"), so it can't read as hallucination.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src import postprocess as pp

# card severities drive the UI colour
INFO, WARN, ALERT = "info", "warn", "alert"


def _money(x) -> str:
    return f"${abs(pp.norm_money(x) or 0):,.2f}"


def _row_desc(r: dict) -> str:
    return str(r.get("description") or r.get("name") or "—")


def _recurring(rows: list[dict], lang: str) -> list[dict]:
    """Same merchant + same amount appearing 2+ times = recurring / duplicate charge."""
    zh = lang == "zh"
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        desc = pp.norm_str(_row_desc(r))
        amt = pp._row_amount(r)
        if desc and amt > 0:
            groups[(desc, round(amt, 2))].append(r)
    cards = []
    for (desc, amt), grp in groups.items():
        if len(grp) < 2:
            continue
        raw_desc = _row_desc(grp[0])
        annual = amt * 12
        cards.append({
            "type": "recurring_payment", "severity": WARN,
            "title": (f"疑似周期性扣款:{raw_desc}" if zh else f"Recurring charge: {raw_desc}"),
            "evidence": [(f"{_row_desc(r)} · {r.get('date') or '—'} · {_money(pp._row_amount(r))}") for r in grp],
            "impact": (f"每笔 {_money(amt)},共 {len(grp)} 笔;按月计年化约 {_money(annual)}" if zh
                       else f"{_money(amt)} each, {len(grp)} times; ~{_money(annual)}/yr if monthly"),
            "action": ("核对这些扣款,确认是否为预期订阅" if zh else "Review these charges — confirm they're expected"),
        })
    return sorted(cards, key=lambda c: c["title"])[:4]


def _anomalies(rows: list[dict], anomalies: list[int], lang: str) -> Optional[dict]:
    if not anomalies:
        return None
    zh = lang == "zh"
    ev = []
    for i in anomalies[:5]:
        if 0 <= i < len(rows):
            ev.append(f"{_row_desc(rows[i])} · {rows[i].get('date') or '—'} · {_money(pp._row_amount(rows[i]))}")
    biggest = max((pp._row_amount(rows[i]) for i in anomalies if 0 <= i < len(rows)), default=0)
    return {
        "type": "anomaly_spend", "severity": WARN,
        "title": (f"{len(anomalies)} 笔异常大额交易" if zh else f"{len(anomalies)} unusually large transaction(s)"),
        "evidence": ev,
        "impact": (f"最大一笔约 {_money(biggest)}" if zh else f"largest ~{_money(biggest)}"),
        "action": ("核对这些异常支出是否为预期" if zh else "Check these outliers are expected"),
    }


def _duplicates(rows: list[dict], dups: list, lang: str) -> Optional[dict]:
    if not dups:
        return None
    zh = lang == "zh"
    ev = []
    for i, j in dups[:5]:
        if 0 <= i < len(rows) and 0 <= j < len(rows):
            ev.append(f"{_row_desc(rows[i])} · {rows[i].get('date') or '—'} · {_money(pp._row_amount(rows[i]))} ×2")
    total = sum(pp._row_amount(rows[i]) for i, j in dups if 0 <= i < len(rows))
    return {
        "type": "duplicate_charge", "severity": ALERT,
        "title": (f"{len(dups)} 组疑似重复扣款" if zh else f"{len(dups)} possible duplicate charge(s)"),
        "evidence": ev,
        "impact": (f"涉及约 {_money(total)}" if zh else f"~{_money(total)} at risk"),
        "action": ("核对是否被重复计费或重复付款" if zh else "Verify you weren't billed or paid twice"),
    }


def _large_bill(data: dict, doc_type: str, lang: str) -> Optional[dict]:
    zh = lang == "zh"
    if doc_type == "invoice":
        bal = pp.norm_money(data.get("balance_due"))
        if bal and bal > 0:
            due = pp.norm_date(data.get("due_date"))
            return {
                "type": "large_bill", "severity": ALERT if bal >= 1000 else WARN,
                "title": (f"发票有未付余额" if zh else "Invoice has a balance due"),
                "evidence": [f"{('供应商' if zh else 'Vendor')}: {data.get('company_name') or '—'}",
                             f"{('到期日' if zh else 'Due')}: {due or '—'}"],
                "impact": (f"应付 {_money(bal)}" if zh else f"{_money(bal)} due"),
                "action": ("在到期前安排付款以免滞纳金" if zh else "Schedule payment before the due date to avoid late fees"),
            }
    return None


def _missing_info(data: dict, doc_type: str, review: list[str], lang: str) -> Optional[dict]:
    if not review:
        return None
    zh = lang == "zh"
    shown = review[:6]
    return {
        "type": "missing_info", "severity": INFO,
        "title": (f"{len(review)} 个字段需人工核对" if zh else f"{len(review)} field(s) need review"),
        "evidence": shown,
        "impact": (f"低置信度字段:{', '.join(shown)}" if zh else f"low-confidence: {', '.join(shown)}"),
        "action": ("在下方逐一确认或修正这些字段" if zh else "Confirm or correct these fields below"),
    }


_LEDGER = ("bank_statement", "credit_card_statement")


def build_insights(analysis: dict, data: dict, doc_type: str, lang: str = "en",
                   review: Optional[list[str]] = None) -> list[dict]:
    """Ordered list of insight cards (most actionable first)."""
    rows = analysis.get("rows") or []
    cards: list[dict] = []
    # transaction-level insights only apply to ledgers (receipt/invoice line items
    # legitimately repeat and carry no per-row signed amount).
    if doc_type in _LEDGER:
        d = _duplicates(rows, analysis.get("duplicates") or [], lang)
        if d:
            cards.append(d)
    lb = _large_bill(data, doc_type, lang)
    if lb:
        cards.append(lb)
    if doc_type in _LEDGER:
        cards += _recurring(rows, lang)
        an = _anomalies(rows, analysis.get("anomalies") or [], lang)
        if an:
            cards.append(an)
    mi = _missing_info(data, doc_type, review or [], lang)
    if mi:
        cards.append(mi)
    return cards

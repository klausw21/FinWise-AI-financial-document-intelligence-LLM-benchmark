"""A one-line financial tip shown after each single analysis (upload or sample).

Two tiers, so every analyze run gets a tip:
  * LLM tip — a cheap Haiku call in the UI language, grounded in the extracted data,
    used when an API key is set.
  * Rules tip — a deterministic sentence derived from the analysis bundle
    (cash-flow / reconciliation / categories); needs no key and no network, so it
    also serves as the fallback when the LLM call fails.

Not used by the benchmark — only the interactive /api/analyze path.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from src import postprocess as pp

# categories that are inflows / settlements, not discretionary spend
_NON_SPEND = {"Income", "Transfer"}

# realistic trim % per discretionary category — mirrors freedom.py's plan (kept local to
# avoid a circular import: freedom.py already imports _NON_SPEND from this module)
_TRIM_HINT = {
    "Fees": 0.80, "Entertainment": 0.40, "Dining": 0.40, "Shopping": 0.35,
    "Cash": 0.25, "Transport": 0.15, "Groceries": 0.10,
}
_NON_TRIM = _NON_SPEND | {"Payment"}


def _money(x) -> str:
    return f"${abs(pp.norm_money(x) or 0):,.2f}"


def _top_spend(category_totals: dict) -> Optional[tuple[str, float]]:
    """Largest non-income category (category_totals is already sorted desc)."""
    for cat, amt in category_totals.items():
        if cat not in _NON_SPEND and (amt or 0) > 0:
            return cat, amt
    return None


def _trim_tips(cats: dict, lang: str, limit: int = 2) -> list[str]:
    """Up to `limit` 'cut your biggest spend' tips with a concrete amount saved."""
    zh = lang == "zh"
    tips = []
    for cat, amt in cats.items():
        if len(tips) >= limit:
            break
        if cat in _NON_TRIM or (amt or 0) <= 0:
            continue
        pct = _TRIM_HINT.get(cat, 0.20)
        saved = (amt or 0) * pct
        if saved < 1:
            continue
        tips.append(
            (f"本期在「{cat}」花了 {_money(amt)},削减约 {int(pct*100)}% 每期可省 {_money(saved)},长期加速财务自由。"
             if zh else
             f"You spent {_money(amt)} on {cat} this period — trimming ~{int(pct*100)}% frees up "
             f"{_money(saved)} to invest toward financial freedom."))
    return tips


def financial_tips(analysis: dict, data: dict, doc_type: str, lang: str) -> list[str]:
    """Deterministic 3-5 multi-dimension financial tips from the analysis bundle (no LLM)."""
    zh = lang == "zh"
    cf = analysis.get("cashflow") or {}
    rec = analysis.get("reconcile") or {}
    cats = analysis.get("category_totals") or {}
    dups = analysis.get("duplicates") or []
    anoms = analysis.get("anomalies") or []
    norm = analysis.get("normalized") or {}
    tips: list[str] = []

    # 1) reconciliation failed -> data-integrity warning first
    if rec.get("reconciles") is False:
        tips.append("账目未对平,建议先逐笔核对交易与期初/期末余额,再据此做预算决策。" if zh else
                    "The balances don't reconcile — verify each transaction against the "
                    "opening/closing balance before budgeting on these numbers.")

    # 2) ledger cash-flow (bank / credit-card)
    if cf:
        net = cf.get("net", 0) or 0
        if net < 0:
            tips.append(f"本期净现金流为 -{_money(net)},支出超过收入,优先压缩下面几类非必要开销。" if zh else
                        f"Your net cash-flow is -{_money(net)} this period — you spent more than you "
                        f"earned; start by trimming the discretionary categories below.")
        else:
            tips.append(f"本期净现金流为正 +{_money(net)},把结余的一部分定投可显著缩短财务自由时间。" if zh else
                        f"Your net cash-flow is +{_money(net)} — investing part of that surplus regularly "
                        f"can meaningfully shorten your path to financial freedom.")
        # 3) biggest discretionary categories, with a concrete amount saved
        tips += _trim_tips(cats, lang, limit=2)
        # 4) possible duplicates / anomalies
        if dups:
            tips.append(f"发现 {len(dups)} 组疑似重复扣款,建议逐一核对,避免重复付款。" if zh else
                        f"Spotted {len(dups)} possible duplicate charge(s) — review them so you aren't billed twice.")
        if anoms:
            tips.append(f"有 {len(anoms)} 笔金额明显偏离常态,核对是否为预期支出。" if zh else
                        f"{len(anoms)} transaction(s) look unusually large — double-check they're expected.")

    # 5) invoice — outstanding balance vs paid
    elif doc_type == "invoice":
        bal = pp.norm_money(norm.get("balance_due"))
        if bal and bal > 0:
            due = norm.get("due_date")
            tips.append(
                (f"这张发票还有 {_money(bal)} 未付{('(到期 ' + str(due) + ')') if due else ''},请在到期前安排付款以免滞纳金。"
                 if zh else
                 f"This invoice has {_money(bal)} still due{(' by ' + str(due)) if due else ''} — "
                 f"schedule payment before the due date to avoid late fees."))
        total = pp.norm_money(norm.get("total"))
        if total:
            tips.append(f"发票金额 {_money(total)},记得留存作报销或记账凭证。" if zh else
                        f"Invoice total is {_money(total)} — keep it on file for expense or bookkeeping records.")

    # 6) receipt
    elif doc_type == "receipt":
        total = pp.norm_money(norm.get("total"))
        if total:
            tips.append(f"本次消费 {_money(total)},建议归入对应预算类别并留存收据。" if zh else
                        f"This receipt totals {_money(total)} — file it under the right budget category and keep the receipt.")

    if not tips:   # generic fallback so there is always at least one tip
        tips.append("数据已抽取并核验,定期回顾这类记录有助于掌握现金流与预算。" if zh else
                    "Your data is extracted and verified — reviewing statements like this regularly helps "
                    "you stay on top of cash-flow and budgeting.")

    # de-dup preserving order, cap at 5
    seen, out = set(), []
    for tp in tips:
        if tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out[:5]


def financial_tip(analysis: dict, data: dict, doc_type: str, lang: str) -> str:
    """Single-tip convenience wrapper (kept for compatibility)."""
    return financial_tips(analysis, data, doc_type, lang)[0]


def _summary_for_llm(analysis: dict, doc_type: str) -> str:
    """Compact analysis digest handed to the LLM alongside the raw data."""
    cf = analysis.get("cashflow") or {}
    rec = analysis.get("reconcile") or {}
    cats = analysis.get("category_totals") or {}
    parts = [f"doc_type={doc_type}"]
    if cf:
        parts.append(f"cash-flow inflow={cf.get('inflow')} outflow={cf.get('outflow')} net={cf.get('net')}")
    if rec.get("reconciles") is not None:
        parts.append(f"reconciles={rec.get('reconciles')}")
    if cats:
        parts.append("top categories=" + ", ".join(f"{k}:{v}" for k, v in list(cats.items())[:5]))
    if analysis.get("duplicates"):
        parts.append(f"possible_duplicates={len(analysis['duplicates'])}")
    if analysis.get("anomalies"):
        parts.append(f"anomalies={len(analysis['anomalies'])}")
    return "; ".join(parts)


def _split_tips(text: str, limit: int = 5) -> list[str]:
    """Parse the LLM's multi-line reply into a clean list of tips (strip bullets/numbering)."""
    import re
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out[:limit]


def build_advice(analysis: dict, data: dict, doc_type: str, lang: str,
                 have_key: bool) -> dict:
    """Return {tips, text, source, cost_usd, latency_s}. Always yields usable tips.

    `tips` is a list of 3-5 multi-dimension suggestions; `text` mirrors the first tip
    (back-compat for older consumers). LLM tips when a key is set (falling back to the
    rules tips on any error/empty), else the deterministic rules tips.
    """
    t0 = time.perf_counter()
    if have_key:
        try:
            from src.extract import llm
            ctx = json.dumps(data, ensure_ascii=False)[:6000]
            r = llm.advise(doc_type, ctx, _summary_for_llm(analysis, doc_type), lang)
            tips = _split_tips(r.get("text") or "")
            if tips and "error" not in r:
                return {"tips": tips, "text": tips[0], "source": "llm",
                        "cost_usd": r.get("cost_usd", 0.0),
                        "latency_s": r.get("latency_s", round(time.perf_counter() - t0, 2))}
        except Exception:
            pass  # fall through to the rules tips
    tips = financial_tips(analysis, data, doc_type, lang)
    return {"tips": tips, "text": tips[0], "source": "rules",
            "cost_usd": 0.0, "latency_s": round(time.perf_counter() - t0, 2)}

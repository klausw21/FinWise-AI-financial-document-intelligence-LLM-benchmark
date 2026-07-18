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


def _money(x) -> str:
    return f"${abs(pp.norm_money(x) or 0):,.2f}"


def _top_spend(category_totals: dict) -> Optional[tuple[str, float]]:
    """Largest non-income category (category_totals is already sorted desc)."""
    for cat, amt in category_totals.items():
        if cat not in _NON_SPEND and (amt or 0) > 0:
            return cat, amt
    return None


def financial_tip(analysis: dict, data: dict, doc_type: str, lang: str) -> str:
    """A single deterministic financial tip from the analysis bundle (no LLM)."""
    zh = lang == "zh"
    cf = analysis.get("cashflow") or {}
    rec = analysis.get("reconcile") or {}
    cats = analysis.get("category_totals") or {}
    dups = analysis.get("duplicates") or []
    anoms = analysis.get("anomalies") or []
    norm = analysis.get("normalized") or {}

    # 1) reconciliation failed -> data-integrity warning first
    if rec.get("reconciles") is False:
        return ("账目未对平,建议先逐笔核对交易与期初/期末余额,再据此做预算决策。" if zh else
                "The balances don't reconcile — verify each transaction against the "
                "opening/closing balance before budgeting on these numbers.")

    # 2) possible duplicate charges (ledger only — receipts/invoices repeat line items legitimately)
    if dups and doc_type in ("bank_statement", "credit_card_statement"):
        n = len(dups)
        return (f"发现 {n} 组疑似重复扣款,建议逐一核对,避免重复付款或被重复计费。" if zh else
                f"Spotted {n} possible duplicate charge(s) — review them so you aren't "
                f"paying or being billed twice.")

    # 3) ledger cash-flow (bank / credit-card)
    if cf:
        net = cf.get("net", 0) or 0
        top = _top_spend(cats)
        if net < 0:
            if top and zh:
                return f"本期净现金流为 -{_money(net)},支出超过收入,最大支出是「{top[0]}」({_money(top[1])}),可优先从这里节省。"
            if top:
                return (f"Your net cash-flow is -{_money(net)} this period — you spent more than "
                        f"you earned; the biggest outflow is {top[0]} ({_money(top[1])}), a good place to trim.")
            return ("本期净现金流为负,支出超过收入,建议压缩非必要开销。" if zh else
                    f"Your net cash-flow is -{_money(net)} this period — consider trimming non-essential spending.")
        return (f"本期净现金流为正 +{_money(net)},可考虑把结余的一部分转入储蓄或投资。" if zh else
                f"Your net cash-flow is positive at +{_money(net)} — consider moving part of the "
                f"surplus into savings or investing it.")

    # 4) invoice — outstanding balance vs paid
    if doc_type == "invoice":
        bal = pp.norm_money(norm.get("balance_due"))
        if bal and bal > 0:
            due = norm.get("due_date")
            if zh:
                return f"这张发票还有 {_money(bal)} 未付{('(到期 ' + str(due) + ')') if due else ''},请在到期前安排付款以免滞纳金。"
            return (f"This invoice has {_money(bal)} still due{(' by ' + str(due)) if due else ''} — "
                    f"schedule payment before the due date to avoid late fees.")
        total = pp.norm_money(norm.get("total"))
        if total:
            return (f"发票已结清,金额 {_money(total)},记得留存作报销或记账凭证。" if zh else
                    f"This invoice looks settled ({_money(total)}) — keep it on file for expense or bookkeeping records.")

    # 5) receipt
    if doc_type == "receipt":
        total = pp.norm_money(norm.get("total"))
        if total:
            return (f"本次消费 {_money(total)},建议归入对应预算类别并留存收据。" if zh else
                    f"This receipt totals {_money(total)} — file it under the right budget category and keep the receipt.")

    # 6) generic fallbacks
    if anoms and doc_type in ("bank_statement", "credit_card_statement"):
        return (f"有 {len(anoms)} 笔金额明显偏离常态,建议核对是否为预期支出。" if zh else
                f"{len(anoms)} transaction(s) look unusually large — double-check they're expected.")
    return ("数据已抽取并核验,定期回顾这类记录有助于掌握现金流与预算。" if zh else
            "Your data is extracted and verified — reviewing statements like this regularly helps "
            "you stay on top of cash-flow and budgeting.")


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
        parts.append("top categories=" + ", ".join(f"{k}:{v}" for k, v in list(cats.items())[:3]))
    if analysis.get("duplicates"):
        parts.append(f"possible_duplicates={len(analysis['duplicates'])}")
    if analysis.get("anomalies"):
        parts.append(f"anomalies={len(analysis['anomalies'])}")
    return "; ".join(parts)


def build_advice(analysis: dict, data: dict, doc_type: str, lang: str,
                 have_key: bool) -> dict:
    """Return {text, source, cost_usd, latency_s}. Always yields a usable one-liner.

    LLM tip when a key is set (falling back to the rules tip on any error/empty),
    else the deterministic rules tip.
    """
    t0 = time.perf_counter()
    if have_key:
        try:
            from src.extract import llm
            ctx = json.dumps(data, ensure_ascii=False)[:6000]
            r = llm.advise(doc_type, ctx, _summary_for_llm(analysis, doc_type), lang)
            text = (r.get("text") or "").strip()
            if text and "error" not in r:
                return {"text": text, "source": "llm", "cost_usd": r.get("cost_usd", 0.0),
                        "latency_s": r.get("latency_s", round(time.perf_counter() - t0, 2))}
        except Exception:
            pass  # fall through to the rules tip
    return {"text": financial_tip(analysis, data, doc_type, lang), "source": "rules",
            "cost_usd": 0.0, "latency_s": round(time.perf_counter() - t0, 2)}

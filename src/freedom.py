"""Financial Freedom Plan — a FIRE (Financial Independence / Retire Early) projection
built from a single bank or credit-card statement.

Design mirrors src/advice.py exactly: ALL money and projection math is deterministic
Python (grounded, reproducible, cited from real rows); the LLM only writes the warm
"story" around numbers we already computed (src/extract/llm.py `story`). It never
invents a figure.

Only meaningful for ledgers (bank_statement, credit_card_statement) — they carry a
cash-flow. A bank statement exposes income (credits) and spending (debits) so we can
project a full timeline; a credit-card statement exposes only spending, so we build an
expenses-only plan (opportunities + a lower "freedom number") and note that the income
side / timeline needs a bank statement. We never fake income from card payments.

Not used by the benchmark — only the interactive /api/analyze path.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from typing import Optional

from src import postprocess as pp
from src.advice import _NON_SPEND   # {"Income", "Transfer"}

_LEDGER = ("bank_statement", "credit_card_statement")

DEFAULT_RETURN = 0.05          # assumed real (after-inflation) annual return
SWR = 0.04                     # 4% safe-withdrawal rule
FI_MULTIPLE = 1.0 / SWR        # 25x annual expenses
_DAYS_PER_MONTH = 30.437
_MONTHS_FLOOR = 0.5            # never annualize a sub-2-week period by a huge factor
_YEARS_CAP = 100.0            # display ceiling for years-to-FI

# card payments pay down a balance — not discretionary spend, never "income"
_NON_TRIM = _NON_SPEND | {"Payment"}

# "Aggressive" trim % per category — the reduction the recommended plan applies.
# Discretionary categories are cut harder; essentials only lightly.
_TRIM = {
    "Fees": 0.80, "Entertainment": 0.40, "Dining": 0.40, "Shopping": 0.35,
    "Cash": 0.25, "Transport": 0.15, "Groceries": 0.10,
}


def _money(x) -> str:
    return f"${abs(pp.norm_money(x) or 0):,.2f}"


def _row_desc(r: dict) -> str:
    return str(r.get("description") or r.get("name") or "—")


# ---------------- period / annualization ----------------
def _parse_date(s) -> Optional[datetime]:
    iso = pp.norm_date(s)
    if not iso:
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _months_from_rows(rows: list[dict]) -> Optional[float]:
    """Months covered, inferred from the transaction dates themselves — robust when
    period_start/period_end are missing or describe only one of several months."""
    dates = sorted(d for d in (_parse_date(r.get("date")) for r in (rows or [])) if d)
    if len(dates) < 2:
        return None
    span = (dates[-1] - dates[0]).days / _DAYS_PER_MONTH
    distinct = len({(d.year, d.month) for d in dates})
    # a multi-month report: trust the distinct calendar-month count, but only when the
    # span really covers it (guards a few days straddling a month boundary)
    if distinct >= 2 and span >= 1.0:
        return float(distinct)
    return round(max(span, _MONTHS_FLOOR), 3)


def _period_months(data: dict, rows: list[dict], doc_type: str) -> tuple[float, bool, bool]:
    """(months, derived, multi_month). Prefer period_start/period_end, but cross-check
    against the transaction dates: when the rows span materially more than the stated
    period (or the period is missing), trust the rows — so a multi-month report is never
    collapsed into one month (which would inflate the monthly figures)."""
    from_rows = _months_from_rows(rows)
    start = _parse_date(data.get("period_start"))
    end = _parse_date(data.get("period_end"))
    period_months = None
    if start and end and end > start:
        period_months = round(max((end - start).days / _DAYS_PER_MONTH, _MONTHS_FLOOR), 3)

    if from_rows is not None and (period_months is None or from_rows >= 1.5 * period_months):
        months, derived = from_rows, True          # rows span more than stated -> trust rows
    elif period_months is not None:
        months, derived = period_months, True
    else:
        months, derived = 1.0, False
    return months, derived, months >= 1.5


# ---------------- baseline ----------------
def _category_rows(rows: list[dict], category: str) -> list[dict]:
    return [r for r in rows if r.get("category") == category]


def _baseline(analysis: dict, doc_type: str, months: float,
              income_override: Optional[float] = None,
              fixed_costs: Optional[float] = None) -> dict:
    """Monthly income / expenses / net / savings-rate + per-category expense breakdown.
    A user-supplied income_override wins for any doc type — it corrects a misread bank
    income and, for a credit-card statement (spending-only), unlocks a full timeline.
    fixed_costs (rent etc. not captured by the statement) are added to monthly expenses
    so net savings actually deduct them and the freedom number covers them."""
    cf = analysis.get("cashflow") or {}
    cats = analysis.get("category_totals") or {}
    outflow = cf.get("outflow") or 0
    inflow = cf.get("inflow") or 0

    doc_expenses = round(outflow / months, 2)
    expenses_monthly = round(doc_expenses + (fixed_costs or 0.0), 2)   # rent folded into total spend

    if income_override is not None:
        income_monthly = round(income_override, 2)
    elif doc_type == "bank_statement":
        income_monthly = round(inflow / months, 2)
    else:  # credit card: "inflow" is card payments, not income -> unknown income
        income_monthly = None
    if income_monthly is not None:
        net_monthly = round(income_monthly - expenses_monthly, 2)     # already net of fixed_costs
        savings_rate = round(net_monthly / income_monthly, 4) if income_monthly > 0 else None
    else:
        net_monthly = savings_rate = None

    by_cat = []
    for cat, total in cats.items():
        if cat in _NON_TRIM or (total or 0) <= 0:
            continue
        by_cat.append({"category": cat, "monthly": round((total or 0) / months, 2),
                       "trim": _TRIM.get(cat, 0.0)})
    by_cat.sort(key=lambda c: -c["monthly"])
    return {"income_monthly": income_monthly, "expenses_monthly": expenses_monthly,
            "doc_expenses_monthly": doc_expenses, "fixed_costs": (fixed_costs or None),
            "net_monthly": net_monthly, "savings_rate": savings_rate,
            "expenses_by_category": by_cat}


# ---------------- opportunities ----------------
def _category_opps(baseline: dict, rows: list[dict], lang: str) -> list[dict]:
    zh = lang == "zh"
    opps = []
    for c in baseline["expenses_by_category"]:
        pct = c["trim"]
        if pct <= 0:
            continue
        saved = round(c["monthly"] * pct, 2)
        if saved < 1:
            continue
        cat = c["category"]
        ev = []
        for r in sorted(_category_rows(rows, cat), key=lambda r: -pp._row_amount(r))[:3]:
            ev.append(f"{_row_desc(r)} · {r.get('date') or '—'} · {_money(pp._row_amount(r))}")
        opps.append({
            "type": "category_trim", "category": cat, "counted": True,
            "current_monthly": c["monthly"], "trim_pct": pct, "monthly_savings": saved,
            "evidence": ev,
            "action": (f"复核你的「{cat}」开销 —— 削减约 {int(pct * 100)}% 是可行的"
                       if zh else
                       f"Review your {cat} spending — a ~{int(pct * 100)}% trim is realistic"),
        })
    return opps


# ---------------- projection ----------------
def _years_to_fi(fi_number: float, annual_savings: float, real_return: float,
                 starting_assets: float = 0.0) -> Optional[float]:
    """Closed-form years for assets S0 + a yearly contribution P at real return r to
    reach FI:  n = ln((FI + P/r)/(S0 + P/r)) / ln(1+r).  None if unreachable."""
    if fi_number <= 0:
        return 0.0                          # no expenses -> already free
    if starting_assets >= fi_number:
        return 0.0
    if annual_savings <= 0:
        return None                         # not saving -> never reaches FI
    r, P, S0 = real_return, annual_savings, starting_assets
    if r <= 0:                              # no growth -> linear
        n = (fi_number - S0) / P
    else:
        n = math.log((fi_number + P / r) / (S0 + P / r)) / math.log(1 + r)
    return round(min(n, _YEARS_CAP), 1)


def _projection(baseline: dict, extra_monthly: float, doc_type: str,
                real_return: float, starting_assets: float) -> dict:
    exp_m = baseline["expenses_monthly"]
    annual_expenses_now = round(exp_m * 12, 2)
    annual_expenses_opt = round(max(0.0, exp_m - extra_monthly) * 12, 2)
    fi_now = round(annual_expenses_now * FI_MULTIPLE, 2)
    fi_opt = round(annual_expenses_opt * FI_MULTIPLE, 2)

    if baseline["net_monthly"] is not None:   # income known (bank, or user-supplied for a card)
        net_m = baseline["net_monthly"]
        years_now = _years_to_fi(fi_now, net_m * 12, real_return, starting_assets)
        years_opt = _years_to_fi(fi_opt, (net_m + extra_monthly) * 12, real_return, starting_assets)
        years_saved = (round(years_now - years_opt, 1)
                       if years_now is not None and years_opt is not None else None)
    else:  # spending-only (credit card, no income entered) -> no timeline
        years_now = years_opt = years_saved = None

    return {"annual_expenses_now": annual_expenses_now, "annual_expenses_opt": annual_expenses_opt,
            "fi_number_now": fi_now, "fi_number_opt": fi_opt,
            "years_now": years_now, "years_opt": years_opt, "years_saved": years_saved,
            "reachable_now": years_now is not None}


# ---------------- deterministic story + headline ----------------
def _headline(baseline: dict, comparison: dict, projection: dict, doc_type: str, lang: str) -> str:
    zh = lang == "zh"
    extra = comparison["extra_monthly_savings"]
    saved = projection.get("years_saved")
    if projection.get("reachable_now") and saved and saved > 0:
        return (f"大约提前 {saved} 年实现财务自由。" if zh else
                f"Reach financial freedom about {saved} years sooner.")
    if doc_type == "credit_card_statement" and baseline["net_monthly"] is None:
        drop = round(projection["fi_number_now"] - projection["fi_number_opt"], 2)
        return (f"每月多挤出 {_money(extra)},自由数字降低 {_money(drop)}。" if zh else
                f"Free up {_money(extra)}/mo and cut your freedom number by {_money(drop)}.")
    return (f"先让现金流转正 —— 每月约 {_money(extra)} 触手可及。" if zh else
            f"Turn cash-flow positive first — about {_money(extra)}/mo is within reach.")


def financial_story(baseline: dict, comparison: dict, projection: dict,
                    doc_type: str, lang: str) -> str:
    """Deterministic, grounded, bilingual narrative — the fallback when no LLM key."""
    zh = lang == "zh"
    cur = comparison["current"]
    opt = comparison["optimized"]
    extra = comparison["extra_monthly_savings"]
    saved = projection.get("years_saved")

    # 1) bank, reaching FI with a surplus
    if projection.get("reachable_now") and saved and saved > 0 and cur.get("savings_rate") is not None:
        cr = round(cur["savings_rate"] * 100)
        orr = round(opt["savings_rate"] * 100)
        if zh:
            return (f"你每月约结余 {_money(cur['monthly_surplus'])},约占收入的 {cr}%。"
                    f"对几类非必要开销大胆削减,储蓄率可提升到 {orr}%,让财务自由大约提前 {saved} 年到来。"
                    f"省下的每一块,都会随复利变成你人生中实实在在的岁月。")
        return (f"You're saving about {_money(cur['monthly_surplus'])} a month — roughly {cr}% of your "
                f"income. Bolder cuts to a few discretionary categories could lift that to {orr}% and "
                f"bring financial freedom about {saved} years sooner. Small, steady changes compound "
                f"into years of your life back.")

    # 2) credit card — spending only, no income/timeline (skip once income is supplied)
    if doc_type == "credit_card_statement" and baseline["net_monthly"] is None:
        drop = round(projection["fi_number_now"] - projection["fi_number_opt"], 2)
        if zh:
            return (f"这张卡本期约有 {_money(baseline['expenses_monthly'])}/月 的支出。"
                    f"削减这几类开销,能把你的「自由数字」降低约 {_money(drop)},每月挤出约 {_money(extra)} 去投资 —— "
                    f"连上一份银行账单,就能看到你完整的时间线。")
        return (f"This card shows about {_money(baseline['expenses_monthly'])}/month of spending. "
                f"Trimming these categories could lower your freedom number by {_money(drop)} and free up "
                f"about {_money(extra)} a month to invest — connect a bank statement to see your full timeline.")

    # 3) no surplus yet (net <= 0)
    if zh:
        return (f"目前支出略高于收入,暂时还没有可投资的结余。好消息是:复核几类非必要开销大约能挤出每月 "
                f"{_money(extra)} —— 这是扭转局面的第一步。")
    return (f"Right now spending edges out income, so there's no surplus to invest yet. The good news: "
            f"reviewing a few discretionary categories could free up about {_money(extra)} a month — "
            f"the first step toward turning the corner.")


def _assumptions(months: float, derived: bool, multi_month: bool, real_return: float,
                 starting_assets: float, doc_type: str, lang: str,
                 user_income: Optional[float] = None, fixed_costs: Optional[float] = None) -> dict:
    zh = lang == "zh"
    notes = []
    if not derived:
        notes.append("已按账单覆盖约 1 个月估算(未能从账单推断周期)。" if zh else
                     "Assumed the statement covers about one month (period not derivable).")
    elif multi_month:
        notes.append((f"已识别账单覆盖约 {months:.0f} 个月,所有数值均已折算为「平均每月」。" if zh else
                      f"Detected a ~{months:.0f}-month span; all figures are shown as a monthly average."))
    else:
        notes.append((f"已按账单周期约 {months:.1f} 个月折算为月度数值。" if zh else
                      f"Annualized from the ~{months:.1f}-month statement period."))
    notes.append((f"假设投资实际年化收益 {round(real_return * 100)}%、按「25× 年支出」(4% 提取率)定义财务自由。"
                  if zh else
                  f"Assumes a {round(real_return * 100)}% real annual return and defines freedom as "
                  f"25× annual expenses (the 4% safe-withdrawal rule)."))
    # user-supplied inputs (each optional) — surfaced so the plan is transparent
    if user_income is not None:
        notes.append((f"按你填写的月收入 {_money(user_income)} 计算(而非从账单推断)。" if zh else
                      f"Using the monthly income you entered ({_money(user_income)}), not the statement's."))
    if starting_assets and starting_assets > 0:
        notes.append((f"从你现有的储蓄/投资 {_money(starting_assets)} 起算时间线。" if zh else
                      f"Timeline starts from your current savings/investments of {_money(starting_assets)}."))
    if fixed_costs is not None:
        notes.append((f"已把你填写的固定支出/房租 {_money(fixed_costs)}/月 计入月支出(结余与自由数字已相应调整)。" if zh else
                      f"Added your fixed costs / rent ({_money(fixed_costs)}/mo) to monthly expenses (net savings and freedom number adjusted)."))
    if doc_type == "credit_card_statement" and user_income is None:
        notes.append(("信用卡账单只含支出;收入与时间线需要一份银行账单,或在上方填写月收入。" if zh else
                      "A credit-card statement shows spending only; add a bank statement or enter your monthly income for a timeline."))
    return {"months": months, "period_derived": derived, "multi_month": multi_month,
            "real_return": real_return, "fi_multiple": FI_MULTIPLE,
            "starting_assets": starting_assets, "notes": notes}


def _disclaimer(lang: str) -> str:
    return ("教育性估算,非投资建议。基于本期账单、固定的假设收益率与「25× 年支出」规则,实际情况会因收入、"
            "市场与生活变化而不同。建议在做重大财务决策前咨询专业人士。" if lang == "zh" else
            "Educational estimate, not financial advice. Based on this statement, a fixed assumed return and "
            "the 25× (4%) rule; your real path will differ with income, markets and life. Consult a professional "
            "before major financial decisions.")


# ---------------- orchestration ----------------
def build_plan(analysis: dict, data: dict, doc_type: str, lang: str = "en",
               have_key: bool = False, *, real_return: float = DEFAULT_RETURN,
               starting_assets: float = 0.0, user_income: Optional[float] = None,
               fixed_costs: Optional[float] = None) -> dict:
    """Full Financial Freedom Plan dict for the UI. Deterministic math + a two-tier
    story (LLM when a key is set, deterministic template otherwise). Never raises.

    Optional user inputs sharpen the plan: `user_income` overrides the statement's income
    (and unlocks a timeline for a spending-only card), `starting_assets` seeds the timeline
    from existing savings, `fixed_costs` (rent etc. not in the statement) are added to
    monthly expenses so net savings deduct them. All default to the prior behaviour when
    omitted."""
    lang = "zh" if lang == "zh" else "en"
    if doc_type not in _LEDGER:
        return {"available": False, "doc_type": doc_type}

    rows = analysis.get("rows") or []
    months, derived, multi_month = _period_months(data, rows, doc_type)
    baseline = _baseline(analysis, doc_type, months, income_override=user_income,
                         fixed_costs=fixed_costs)

    # not enough to plan on (empty / zero-spend statement)
    if baseline["expenses_monthly"] <= 0 and not baseline["expenses_by_category"]:
        return {"available": True, "insufficient": True, "doc_type": doc_type,
                "assumptions": _assumptions(months, derived, multi_month, real_return, starting_assets,
                                            doc_type, lang, user_income, fixed_costs),
                "note": ("交易数据不足,无法生成规划。" if lang == "zh" else
                         "Not enough transaction data to build a plan.")}

    opportunities = _category_opps(baseline, rows, lang)   # discretionary category trims only
    extra_monthly = round(sum(o["monthly_savings"] for o in opportunities if o["counted"]), 2)

    projection = _projection(baseline, extra_monthly, doc_type, real_return, starting_assets)

    if baseline["income_monthly"] and baseline["income_monthly"] > 0:
        opt_surplus = round((baseline["net_monthly"] or 0) + extra_monthly, 2)
        comparison = {
            "current": {"savings_rate": baseline["savings_rate"], "monthly_surplus": baseline["net_monthly"]},
            "optimized": {"savings_rate": round(opt_surplus / baseline["income_monthly"], 4),
                          "monthly_surplus": opt_surplus},
            "extra_monthly_savings": extra_monthly}
    else:
        comparison = {
            "current": {"savings_rate": None, "monthly_surplus": baseline["net_monthly"]},
            "optimized": {"savings_rate": None,
                          "monthly_surplus": (round((baseline["net_monthly"] or 0) + extra_monthly, 2)
                                              if baseline["net_monthly"] is not None else None)},
            "extra_monthly_savings": extra_monthly}

    headline = _headline(baseline, comparison, projection, doc_type, lang)

    # two-tier story: LLM prose over the computed figures, else deterministic template
    t0 = time.perf_counter()
    story = None
    if have_key:
        try:
            from src.extract import llm
            figures = {
                "doc_type": doc_type,
                "income_monthly": baseline["income_monthly"],
                "expenses_monthly": baseline["expenses_monthly"],
                "net_monthly": baseline["net_monthly"],
                "savings_rate": baseline["savings_rate"],
                "optimized_savings_rate": comparison["optimized"]["savings_rate"],
                "extra_monthly_savings": extra_monthly,
                "fi_number_now": projection["fi_number_now"],
                "fi_number_opt": projection["fi_number_opt"],
                "years_now": projection["years_now"],
                "years_opt": projection["years_opt"],
                "years_saved": projection["years_saved"],
                "reachable_now": projection["reachable_now"],
            }
            r = llm.story(figures, lang)
            text = (r.get("text") or "").strip()
            if text and "error" not in r:
                story = {"text": text, "source": "llm", "cost_usd": r.get("cost_usd", 0.0),
                         "latency_s": r.get("latency_s", round(time.perf_counter() - t0, 2))}
        except Exception:
            pass
    if story is None:
        story = {"text": financial_story(baseline, comparison, projection, doc_type, lang),
                 "source": "rules", "cost_usd": 0.0, "latency_s": round(time.perf_counter() - t0, 2)}

    return {
        "available": True, "doc_type": doc_type,
        "income_available": baseline["income_monthly"] is not None,
        "assumptions": _assumptions(months, derived, multi_month, real_return, starting_assets,
                                    doc_type, lang, user_income, fixed_costs),
        "baseline": baseline, "opportunities": opportunities,
        "comparison": comparison, "projection": projection,
        "headline": headline, "story": story, "disclaimer": _disclaimer(lang),
    }


if __name__ == "__main__":
    from src import analyze as analyze_mod
    from src import dataset as ds
    from src.gold.build import gold_for

    d = ds.list_docs("bank_statement")[0]
    g = gold_for(d)
    a = analyze_mod.analyze(g, "bank_statement")
    plan = build_plan(a, g, "bank_statement", "en", have_key=False)
    print("available:", plan["available"], "| income_available:", plan["income_available"])
    print("baseline:", plan["baseline"]["income_monthly"], plan["baseline"]["expenses_monthly"],
          plan["baseline"]["net_monthly"], plan["baseline"]["savings_rate"])
    print("extra/mo:", plan["comparison"]["extra_monthly_savings"])
    print("years now/opt/saved:", plan["projection"]["years_now"],
          plan["projection"]["years_opt"], plan["projection"]["years_saved"])
    print("headline:", plan["headline"])
    print("story:", plan["story"]["text"])

    # ---- synthetic multi-month regression (no dataset / no key needed) ----
    print("\n--- multi-month synthetic check ---")
    syn_rows = [
        {"date": "2026-01-05", "description": "Diner",  "debit": 100, "category": "Dining"},
        {"date": "2026-01-20", "description": "Store",  "debit": 100, "category": "Shopping"},
        {"date": "2026-02-05", "description": "Diner",  "debit": 100, "category": "Dining"},
        {"date": "2026-02-20", "description": "Store",  "debit": 100, "category": "Shopping"},
        {"date": "2026-03-05", "description": "Diner",  "debit": 100, "category": "Dining"},
        {"date": "2026-03-20", "description": "Store",  "debit": 100, "category": "Shopping"},
    ]
    syn_analysis = {
        "rows": syn_rows,
        "cashflow": {"inflow": 900.0, "outflow": 600.0, "net": 300.0},
        "category_totals": {"Dining": 300.0, "Shopping": 300.0},
    }
    syn_data = {"period_start": None, "period_end": None, "transaction_rows": syn_rows}
    m, drv, mm = _period_months(syn_data, syn_rows, "bank_statement")
    syn_plan = build_plan(syn_analysis, syn_data, "bank_statement", "en", have_key=False)
    exp_m = syn_plan["baseline"]["expenses_monthly"]
    print(f"months={m} derived={drv} multi_month={mm}  expenses_monthly={exp_m}")
    assert abs(m - 3.0) < 0.01 and mm is True, "expected 3-month span detected from rows"
    assert abs(exp_m - 200.0) < 0.01, "expected 600 outflow / 3 months = 200/mo (was 600 pre-fix)"
    print("PASS: 3-month report averaged to $200/mo (pre-fix would show $600/mo)")

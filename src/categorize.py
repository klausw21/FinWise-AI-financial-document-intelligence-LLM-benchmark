"""Transaction / line-item categorization (an intro core feature).

Rules-based keyword mapping by default (free, offline, deterministic). Optionally
one LLM call to categorize when higher accuracy is wanted (use_llm=True).
"""
from __future__ import annotations

from src import postprocess as pp

CATEGORIES = ["Income", "Payment", "Transfer", "Groceries", "Dining", "Transport",
              "Utilities", "Shopping", "Entertainment", "Health", "Cash", "Fees", "Other"]

# ordered: first matching keyword wins (put specific before generic)
_RULES: list[tuple[str, list[str]]] = [
    ("Income", ["salary", "payroll", "interest credit", "interest earned", "dividend",
                "deposit", "refund", "reimburs", "credit received"]),
    ("Fees", ["overdraft", "service charge", "interest charged", "finance charge",
              "late fee", "annual fee", "atm fee", " fee"]),
    ("Payment", ["card payment", "payment - thank you", "online payment", "autopay",
                 "bill payment", "minimum payment"]),
    ("Transfer", ["transfer", "wire", "zelle", "venmo", "ach"]),
    ("Groceries", ["grocery", "whole foods", "walmart", "costco", "safeway", "aldi",
                   "trader joe", "kroger", "supermarket"]),
    ("Dining", ["restaurant", "cafe", "coffee", "starbucks", "mcdonald", "dining",
                "pizza", "doordash", "uber eats", "grubhub"]),
    ("Transport", ["uber", "lyft", "shell", "chevron", "gas ", "fuel", "parking",
                   "transit", "delta", "airline", "flight", "taxi"]),
    ("Utilities", ["utility", "electric", "water bill", "internet", "comcast",
                   "verizon", "at&t", "phone bill", "gas bill"]),
    ("Shopping", ["amazon", "target", "best buy", "ikea", "home depot", "apple store",
                  "online purchase", "cartridge", "printer", "laptop", "notebook", "store"]),
    ("Entertainment", ["netflix", "spotify", "hulu", "disney", "cinema", "movie",
                       "steam", "game"]),
    ("Health", ["pharmacy", "cvs", "walgreens", "doctor", "medical", "hospital",
                "clinic", "dental"]),
    ("Cash", ["atm", "withdrawal", "cash"]),
]

# which text field to categorize on, per doc type
_TEXT_FIELD = {"bank_statement": "description", "credit_card_statement": "description",
               "invoice": "description", "receipt": "name"}


def _rule_category(text: str) -> str:
    t = pp.norm_str(text)
    for cat, kws in _RULES:
        if any(k in t for k in kws):
            return cat
    return "Other"


def _rows_and_field(data: dict, doc_type: str) -> tuple[list[dict], str | None]:
    from src.schemas import LIST_FIELD
    if doc_type not in LIST_FIELD:
        return [], None
    field = LIST_FIELD[doc_type][0]
    rows = data.get(field) or []
    return (rows if isinstance(rows, list) else []), field


def categorize(data: dict, doc_type: str, use_llm: bool = False) -> list[dict]:
    """Add a `category` to each row of the doc's list field; returns the rows."""
    rows, field = _rows_and_field(data, doc_type)
    if not rows:
        return []
    tf = _TEXT_FIELD.get(doc_type, "description")
    if use_llm:
        cats = _llm_categorize([str(r.get(tf, "")) for r in rows])
    else:
        cats = [_rule_category(str(r.get(tf, ""))) for r in rows]
    for r, c in zip(rows, cats):
        r["category"] = c
    return rows


def category_totals(rows: list[dict]) -> dict[str, float]:
    """Sum of |amount| per category — for the breakdown chart."""
    out: dict[str, float] = {}
    for r in rows:
        amt = abs((pp.norm_money(r.get("amount")) if r.get("amount") is not None
                   else (r.get("credit") or 0) - (r.get("debit") or 0)) or 0)
        out[r.get("category", "Other")] = round(out.get(r.get("category", "Other"), 0) + amt, 2)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _llm_categorize(descriptions: list[str]) -> list[str]:
    """One LLM call classifying every description into CATEGORIES; falls back to rules."""
    try:
        import json
        from src.extract.llm import _get_client
        prompt = ("Classify each transaction description into exactly one category from "
                  f"{CATEGORIES}. Return a JSON array of category strings, same length and "
                  f"order as the input.\nDescriptions:\n" + json.dumps(descriptions))
        resp = _get_client().messages.create(
            model="claude-haiku-4-5", max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "array", "items": {"type": "string", "enum": CATEGORIES}}}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "[]")
        cats = json.loads(text)
        if len(cats) == len(descriptions):
            return [c if c in CATEGORIES else "Other" for c in cats]
    except Exception:
        pass
    return [_rule_category(d) for d in descriptions]


if __name__ == "__main__":
    from src import dataset as ds

    for t in ("bank_statement", "credit_card_statement"):
        docs = ds.list_docs(t)
        if not docs:
            continue
        data = docs[0].load_label()
        if t == "bank_statement":
            from src.gold.tx_from_pdf import parse_bank_rows
            data = {**data, "transaction_rows": [r.model_dump() for r in parse_bank_rows(docs[0].pdf_path)]}
        rows = categorize(data, t)
        print(f"\n[{t}] categories:")
        for r in rows[:8]:
            tf = _TEXT_FIELD[t]
            print(f"  {r.get(tf, ''):20s} -> {r['category']}")
        print("  totals:", category_totals(rows))

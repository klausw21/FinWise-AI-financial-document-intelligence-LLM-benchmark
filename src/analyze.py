"""One-stop analysis for the product UI.

Takes an extracted document dict + its type and returns the full analysis bundle
the intro asks for: normalized fields, categorized rows, balance reconciliation,
duplicate & anomaly detection, and cash-flow — all reusing src/postprocess.py.
"""
from __future__ import annotations

from src import categorize as cat
from src import postprocess as pp
from src.metrics import DATE, NUM, FIELD_SPEC
from src.schemas import LIST_FIELD


def _normalized_header(data: dict, doc_type: str) -> dict:
    """Normalized (display) view of scalar fields: ISO dates, numeric amounts."""
    out = {}
    for k, kind in FIELD_SPEC.get(doc_type, {}).items():
        v = data.get(k)
        if v is None:
            continue
        out[k] = pp.norm_date(v) if kind == DATE else pp.norm_money(v) if kind == NUM else v
    return out


def analyze(data: dict, doc_type: str, use_llm_cat: bool = False) -> dict:
    rows = cat.categorize(data, doc_type, use_llm=use_llm_cat)  # adds `category` in place
    field = LIST_FIELD.get(doc_type, (None,))[0]
    return {
        "doc_type": doc_type,
        "list_field": field,
        "rows": rows,
        "category_totals": cat.category_totals(rows) if rows else {},
        "reconcile": pp.reconcile(data, doc_type),
        "cashflow": pp.cashflow_summary(data, doc_type),
        "duplicates": pp.detect_duplicates(rows) if rows else [],
        "anomalies": pp.detect_anomalies(rows) if rows else [],
        "normalized": _normalized_header(data, doc_type),
    }


if __name__ == "__main__":
    from src import dataset as ds
    from src.gold.build import gold_for

    d = ds.list_docs("bank_statement")[0]
    a = analyze(gold_for(d), "bank_statement")
    print("keys:", list(a))
    print("reconcile:", a["reconcile"]["reconciles"], "| cashflow:", a["cashflow"])
    print("category_totals:", a["category_totals"])
    print("row[0]:", a["rows"][0])
    print("duplicates:", a["duplicates"], "| anomalies:", a["anomalies"])

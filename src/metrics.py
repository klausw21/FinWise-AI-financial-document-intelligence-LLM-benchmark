"""Scoring: field accuracy, line-item / transaction matching, reconciliation.

`score_doc(pred, gold, doc_type)` returns a per-document record; the benchmark
aggregates these across docs and methods. `gold` must already carry the scorable
list (bank gold rows come from the PDF-derived gold, not the label).
"""
from __future__ import annotations

from typing import Any, Optional

from rapidfuzz import fuzz

from src import postprocess as pp

STR, NUM, DATE, INT = "str", "num", "date", "int"

# scalar (header) fields per type -> comparison kind
FIELD_SPEC: dict[str, dict[str, str]] = {
    "bank_statement": {
        "category": STR, "statement_id": STR, "account_holder": STR,
        "account_number_masked": STR, "period_start": DATE, "period_end": DATE,
        "opening_balance": NUM, "closing_balance": NUM, "transactions": INT,
    },
    "receipt": {
        "category": STR, "receipt_id": STR, "merchant_name": STR, "date": DATE,
        "subtotal": NUM, "tax": NUM, "discount": NUM, "total": NUM, "currency": STR,
    },
    "invoice": {
        "invoice_id": STR, "invoice_number": STR, "company_name": STR,
        "company_address": STR, "customer_name": STR, "billing_address": STR,
        "invoice_date": DATE, "due_date": DATE, "currency": STR, "payment_terms": STR,
        "status": STR, "subtotal": NUM, "discount": NUM, "tax_rate": NUM, "tax": NUM,
        "shipping": NUM, "total": NUM, "amount_paid": NUM, "balance_due": NUM,
        "vendor_email": STR, "vendor_phone": STR, "po_number": STR,
    },
    "credit_card_statement": {
        "category": STR, "statement_id": STR, "card_number_masked": STR, "cardholder": STR,
        "period_start": DATE, "period_end": DATE, "previous_balance": NUM, "new_balance": NUM,
        "minimum_payment": NUM, "payment_due_date": DATE, "credit_limit": NUM,
    },
}

# list field: (field_name, scored subfields with kinds, amount-key fn for matching)
LIST_SPEC: dict[str, dict] = {
    "bank_statement": {
        "field": "transaction_rows",
        "sub": {"date": DATE, "description": STR, "debit": NUM, "credit": NUM, "balance": NUM},
        "amount": lambda r: (pp.norm_money(r.get("credit")) or 0) - (pp.norm_money(r.get("debit")) or 0),
    },
    "credit_card_statement": {
        "field": "transactions",
        "sub": {"date": DATE, "description": STR, "amount": NUM, "type": STR},
        "amount": lambda r: pp.norm_money(r.get("amount")) or 0,
    },
    "invoice": {
        "field": "items",
        "sub": {"description": STR, "quantity": NUM, "unit_price": NUM, "amount": NUM},
        "amount": lambda r: pp.norm_money(r.get("amount")) or 0,
    },
    "receipt": {
        "field": "items",
        "sub": {"name": STR, "qty": NUM, "unit_price": NUM, "line_total": NUM},
        "amount": lambda r: pp.norm_money(r.get("line_total")) or 0,
    },
}

STR_THRESHOLD = 90.0
NUM_TOL = 0.02


def _cmp(kind: str, a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if kind == STR:
        return fuzz.ratio(pp.norm_str(a), pp.norm_str(b)) >= STR_THRESHOLD
    if kind == NUM:
        na, nb = pp.norm_money(a), pp.norm_money(b)
        return na is not None and nb is not None and abs(na - nb) <= NUM_TOL
    if kind == DATE:
        return pp.norm_date(a) == pp.norm_date(b)
    if kind == INT:
        try:
            return int(a) == int(b)
        except (TypeError, ValueError):
            return False
    return False


def score_fields(pred: dict, gold: dict, doc_type: str) -> dict:
    spec = FIELD_SPEC[doc_type]
    per_field = {k: _cmp(kind, pred.get(k), gold.get(k)) for k, kind in spec.items()}
    acc = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"per_field": per_field, "field_accuracy": acc}


def _match_rows(pred: list[dict], gold: list[dict], amount_fn) -> list[tuple[int, Optional[int]]]:
    """Greedy match gold->pred on (date, amount within tol). Returns (gold_i, pred_i|None)."""
    used = set()
    pairs = []
    for gi, g in enumerate(gold):
        gdate, gamt = pp.norm_date(g.get("date")), amount_fn(g)
        best = None
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            if pp.norm_date(p.get("date")) == gdate and abs(amount_fn(p) - gamt) <= NUM_TOL:
                best = pi
                break
        if best is not None:
            used.add(best)
        pairs.append((gi, best))
    return pairs


def score_list(pred: dict, gold: dict, doc_type: str) -> Optional[dict]:
    spec = LIST_SPEC.get(doc_type)
    if not spec:
        return None
    pred_rows = pred.get(spec["field"]) or []
    gold_rows = gold.get(spec["field"]) or []
    if not isinstance(pred_rows, list):
        pred_rows = []
    pairs = _match_rows(pred_rows, gold_rows, spec["amount"])
    matched = [(gi, pi) for gi, pi in pairs if pi is not None]
    n_gold, n_pred = len(gold_rows), len(pred_rows)
    precision = len(matched) / n_pred if n_pred else 0.0
    recall = len(matched) / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # field accuracy within matched pairs + fully-correct row rate
    sub = spec["sub"]
    field_hits = field_total = 0
    exact_rows = 0
    for gi, pi in matched:
        row_ok = True
        for k, kind in sub.items():
            ok = _cmp(kind, pred_rows[pi].get(k), gold_rows[gi].get(k))
            field_hits += ok
            field_total += 1
            row_ok = row_ok and ok
        exact_rows += row_ok
    return {
        "n_gold": n_gold, "n_pred": n_pred, "matched": len(matched),
        "precision": precision, "recall": recall, "f1": f1,
        "field_accuracy": field_hits / field_total if field_total else 0.0,
        "row_exact_rate": exact_rows / n_gold if n_gold else 0.0,  # ~ transaction accuracy
    }


def score_doc(pred: dict, gold: dict, doc_type: str) -> dict:
    out = {"doc_type": doc_type, "fields": score_fields(pred, gold, doc_type)}
    ls = score_list(pred, gold, doc_type)
    if ls is not None:
        out["list"] = ls
    if doc_type in ("bank_statement", "credit_card_statement"):
        out["reconciles"] = bool(pp.reconcile(pred, doc_type).get("reconciles"))
    return out


if __name__ == "__main__":
    from src import dataset as ds
    from src.gold.tx_from_pdf import parse_bank_rows

    # perfect-prediction sanity check: score the gold against itself -> ~1.0 everywhere
    d = ds.list_docs("bank_statement")[0]
    label = d.load_label()
    rows = [r.model_dump() for r in parse_bank_rows(d.pdf_path)]
    gold = {**label, "transaction_rows": rows}
    s = score_doc(gold, gold, "bank_statement")
    print("self-score bank:", {"field_acc": round(s["fields"]["field_accuracy"], 3),
                                "row_exact": round(s["list"]["row_exact_rate"], 3),
                                "f1": round(s["list"]["f1"], 3), "reconciles": s["reconciles"]})
    # a degraded prediction: drop 2 rows + corrupt a balance
    bad = {**gold, "closing_balance": 9999.0, "transaction_rows": rows[:-2]}
    s2 = score_doc(bad, gold, "bank_statement")
    print("degraded bank  :", {"field_acc": round(s2["fields"]["field_accuracy"], 3),
                                "recall": round(s2["list"]["recall"], 3),
                                "row_exact": round(s2["list"]["row_exact_rate"], 3),
                                "reconciles": s2["reconciles"]})

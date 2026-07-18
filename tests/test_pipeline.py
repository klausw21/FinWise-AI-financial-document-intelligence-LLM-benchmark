"""Fast, API-key-free tests for the extraction/benchmark pipeline invariants."""
from __future__ import annotations

import pytest

from src import dataset as ds
from src import postprocess as pp
from src.extract import rules
from src.extract.pdf_text import pdf_text_layout
from src.gold.build import gold_for
from src.gold.tx_from_pdf import parse_bank_rows
from src.gold.validate import reconcile
from src.metrics import score_doc
from src.schemas import schema_for, strict_json_schema


# ---------- dataset adapter ----------
def test_pairing_and_naming():
    for t in ("bank_statement", "invoice"):  # two naming schemes
        docs = ds.list_docs(t)
        assert len(docs) == 1000
        d = docs[0]
        assert all(d.exists().values()), (t, d.exists())
    # invoice uses the INV-2026 scheme
    assert ds.list_docs("invoice")[0].stem.startswith("INV-2026-")


def test_stratified_sample_deterministic():
    a = ds.stratified_sample(3, seed=1)
    b = ds.stratified_sample(3, seed=1)
    assert [d.stem for d in a] == [d.stem for d in b]


# ---------- schemas ----------
@pytest.mark.parametrize("t", ["bank_statement", "receipt", "invoice"])
def test_labels_validate(t):
    raw = ds.list_docs(t)[0].load_label()
    schema_for(t).model_validate(raw)  # must not raise


def _count_unions(node) -> int:
    """Parameters using anyOf or a type-array (['string','null']) — Claude caps at 16."""
    if not isinstance(node, dict):
        return 0
    n = 1 if ("anyOf" in node or isinstance(node.get("type"), list)) else 0
    for v in list(node.get("properties", {}).values()) + list(node.get("$defs", {}).values()):
        n += _count_unions(v)
    if isinstance(node.get("items"), dict):
        n += _count_unions(node["items"])
    return n


def _count_optionals(node) -> int:
    """Properties not listed in `required` — Claude caps these at 24."""
    if not isinstance(node, dict):
        return 0
    n = 0
    if node.get("type") == "object":
        req = set(node.get("required", []))
        n += sum(1 for k in node.get("properties", {}) if k not in req)
    for v in list(node.get("properties", {}).values()) + list(node.get("$defs", {}).values()):
        n += _count_optionals(v)
    if isinstance(node.get("items"), dict):
        n += _count_optionals(node["items"])
    return n


@pytest.mark.parametrize("t", ["bank_statement", "invoice", "credit_card_statement",
                               "receipt"])
def test_strict_schema_shape(t):
    s = strict_json_schema(t)
    assert s["additionalProperties"] is False
    # root: every property required -> fixed-order linear grammar (no subset explosion)
    assert set(s["required"]) == set(s["properties"].keys())
    # stay under Claude's two counted structured-output limits
    assert _count_unions(s) <= 16, f"{t} has too many nullable/union params"      # Invoice hit 26
    assert _count_optionals(s) <= 24, f"{t} has too many optional params"          # Invoice hit 27


# ---------- transaction gold ----------
def test_bank_gold_reconciles():
    for doc in ds.list_docs("bank_statement")[:25]:
        label = doc.load_label()
        rows = parse_bank_rows(doc.pdf_path)
        ok, reasons = reconcile(rows, label["opening_balance"], label["closing_balance"])
        assert ok, (doc.stem, reasons)
        assert len(rows) == label["transactions"]


def test_credit_card_reconciles():
    docs = ds.list_docs("credit_card_statement")
    assert docs, "generate credit cards first: python -m src.generate.credit_card 200"
    for doc in docs[:25]:
        assert pp.reconcile(doc.load_label(), "credit_card_statement")["reconciles"]


# ---------- rules baseline ----------
def test_rules_parse_bank_headers():
    d = ds.list_docs("bank_statement")[0]
    parsed = rules.parse("bank_statement", pdf_text_layout(d.pdf_path))
    label = d.load_label()
    assert parsed["statement_id"] == label["statement_id"]
    assert abs(parsed["opening_balance"] - label["opening_balance"]) < 0.01
    assert len(parsed["transaction_rows"]) == label["transactions"]


# ---------- metrics ----------
def test_self_score_is_perfect():
    d = ds.list_docs("bank_statement")[0]
    gold = gold_for(d)
    s = score_doc(gold, gold, "bank_statement")
    assert s["fields"]["field_accuracy"] == 1.0
    assert s["list"]["f1"] == 1.0
    assert s["list"]["row_exact_rate"] == 1.0
    assert s["reconciles"] is True


def test_degraded_score_drops():
    d = ds.list_docs("bank_statement")[0]
    gold = gold_for(d)
    bad = {**gold, "closing_balance": 9999.0,
           "transaction_rows": gold["transaction_rows"][:-2]}
    s = score_doc(bad, gold, "bank_statement")
    assert s["fields"]["field_accuracy"] < 1.0
    assert s["list"]["recall"] < 1.0
    assert s["reconciles"] is False

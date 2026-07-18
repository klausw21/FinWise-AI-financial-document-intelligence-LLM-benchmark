"""Tests for the product layer: detection, categorization, analysis, recommendation, web API."""
from __future__ import annotations

import pandas as pd
import pytest

from src import analyze as analyze_mod
from src import categorize as cat
from src import dataset as ds
from src import detect, methods
from src.extract.pdf_text import pdf_text_layout
from src.gold.build import gold_for


# ---------- detection ----------
@pytest.mark.parametrize("t", ["bank_statement", "invoice", "credit_card_statement",
                               "receipt"])
def test_detect_type(t):
    docs = ds.list_docs(t)
    if not docs:
        pytest.skip(f"{t} not present")
    assert detect.detect_type(text=pdf_text_layout(docs[0].pdf_path), filename=docs[0].stem) == t


# ---------- categorization ----------
def test_categorize_known_descriptions():
    rows = [{"description": "Salary Credit", "credit": 100, "debit": None},
            {"description": "Whole Foods", "debit": 40, "credit": None},
            {"description": "ATM Withdrawal", "debit": 60, "credit": None}]
    out = cat.categorize({"transaction_rows": rows}, "bank_statement")
    got = {r["description"]: r["category"] for r in out}
    assert got["Salary Credit"] == "Income"
    assert got["Whole Foods"] == "Groceries"
    assert got["ATM Withdrawal"] == "Cash"


# ---------- analysis bundle ----------
def test_analyze_bundle_complete():
    d = ds.list_docs("bank_statement")[0]
    a = analyze_mod.analyze(gold_for(d), "bank_statement")
    for k in ("normalized", "rows", "category_totals", "reconcile", "cashflow",
              "duplicates", "anomalies"):
        assert k in a, k
    assert a["reconcile"]["reconciles"] is True
    assert a["cashflow"]["net"] == pytest.approx(
        d.load_label()["closing_balance"] - d.load_label()["opening_balance"], abs=0.05)
    assert all("category" in r for r in a["rows"])


# ---------- financial advice ----------
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_advice_rules_without_key(lang):
    from src.advice import build_advice
    d = ds.list_docs("bank_statement")[0]
    g = gold_for(d)
    a = analyze_mod.analyze(g, "bank_statement")
    adv = build_advice(a, g, "bank_statement", lang, have_key=False)  # offline, no network
    assert adv["source"] == "rules"
    assert adv["cost_usd"] == 0.0
    assert isinstance(adv["text"], str) and adv["text"].strip()


# ---------- recommendation ----------
def test_recommend_rejects_worse_cheaper_model():
    """A genuinely worse model (accuracy far below best) must not be picked for being cheap."""
    from benchmark.run import recommend
    df = pd.DataFrame([
        {"perturbation": "clean", "method": "A", "doc_type": "x", "field_accuracy": 0.9,
         "row_exact": 0.9, "reconcile_rate": 1.0, "cost_per_100": 4.0},
        {"perturbation": "clean", "method": "B", "doc_type": "x", "field_accuracy": 0.7,
         "row_exact": 0.7, "reconcile_rate": 1.0, "cost_per_100": 0.0},
    ])
    assert recommend(df)["recommended"] == "A"


def test_recommend_prefers_cheaper_within_tolerance():
    """Near-equal quality (within 1pp) + big cost gap -> pick the cheaper model (Sonnet over Opus)."""
    from benchmark.run import recommend
    df = pd.DataFrame([
        {"perturbation": "clean", "method": "opus", "doc_type": "x", "field_accuracy": 0.978,
         "row_exact": 1.0, "reconcile_rate": 1.0, "cost_per_100": 2.37},
        {"perturbation": "clean", "method": "sonnet", "doc_type": "x", "field_accuracy": 0.971,
         "row_exact": 1.0, "reconcile_rate": 1.0, "cost_per_100": 0.95},
    ])
    assert recommend(df)["recommended"] == "sonnet"


def test_recommended_method_is_valid():
    assert methods.recommended_method(have_key=False) in methods.METHODS
    assert methods.recommended_method(have_key=True) in methods.METHODS


# ---------- confidence / present / insights / store ----------
def test_confidence_and_fields_on_gold():
    from src import present, confidence as cf
    from src.extract.pdf_text import pdf_text_layout
    d = ds.list_docs("bank_statement")[0]
    g = gold_for(d)
    a = analyze_mod.analyze(g, "bank_statement")
    src_text = pdf_text_layout(d.pdf_path) or ""
    fields, review = present.build_fields(g, "bank_statement", src_text, a)
    assert fields and all(f["level"] in ("high", "medium", "low") for f in fields)
    # gold values are exact and appear in the source text -> should be high-confidence, no review
    assert all(f["level"] == "high" for f in fields)
    assert review == []
    # a value not present in the source text is not high-confidence
    c = cf.field_confidence("ZZ-NOT-IN-DOC-999", cf.STR, src_text)
    assert c["level"] != "high"


def test_confidence_missing_is_low():
    from src import confidence as cf
    assert cf.field_confidence(None, cf.STR, "some text")["level"] == "low"
    assert cf.field_confidence("", cf.NUM, "some text")["level"] == "low"


def test_insights_grounded_cards():
    from src import insights
    d = ds.list_docs("invoice")[0]
    g = gold_for(d)
    a = analyze_mod.analyze(g, "invoice")
    cards = insights.build_insights(a, g, "invoice", "en", review=[])
    # an invoice with a balance due should surface a large_bill card with evidence + impact
    types = {c["type"] for c in cards}
    assert "large_bill" in types
    for c in cards:
        assert c["title"] and c["impact"] and c["action"] and "severity" in c


def test_insights_no_transaction_cards_for_receipt():
    from src import insights
    d = ds.list_docs("receipt")[0]
    g = gold_for(d)
    a = analyze_mod.analyze(g, "receipt")
    cards = insights.build_insights(a, g, "receipt", "en", review=[])
    # receipt line items must not trigger duplicate/recurring/anomaly (ledger-only) cards
    assert not ({"duplicate_charge", "recurring_payment", "anomaly_spend"} & {c["type"] for c in cards})


def _fake_llm_client(captured, text="{}", stop_reason="end_turn"):
    """Streaming client stub: records kwargs, returns a canned final message."""
    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self):
            block = type("B", (), {"type": "text", "text": text})()
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            return type("R", (), {"content": [block], "usage": usage, "stop_reason": stop_reason})()

    class _Msgs:
        def stream(self, **kw): captured.update(kw); return _Stream()
    return type("C", (), {"messages": _Msgs()})()


def test_vision_multipage_sends_all_pages(monkeypatch):
    """A multi-page upload must send every page image to the vision model."""
    from src.extract import llm
    captured = {}
    monkeypatch.setattr(llm, "_get_client", lambda: _fake_llm_client(captured))
    imgs = [d.image_path for d in ds.list_docs("bank_statement")[:2]]
    llm.extract_vision("bank_statement", imgs, "claude-haiku-4-5", "x", "M5", max_edge=700)
    content = captured["messages"][0]["content"]
    assert sum(1 for b in content if b.get("type") == "image") == 2  # both pages sent
    assert content[-1]["type"] == "text" and "pages" in content[-1]["text"].lower()
    assert captured["max_tokens"] >= 8000   # generous extraction budget (was 4096 -> truncated real docs)


def test_extraction_truncation_is_salvaged(monkeypatch):
    """max_tokens truncation -> salvage the complete rows instead of a hard JSON failure."""
    from src.extract import llm
    captured = {}
    truncated = ('{"category":"credit_card_statement","new_balance":100.0,"transactions":'
                 '[{"date":"2026-01-01","amount":10.0},{"date":"2026-01-02","amount":20.0},'
                 '{"date":"2026-01-03","amoun')
    monkeypatch.setattr(llm, "_get_client",
                        lambda: _fake_llm_client(captured, text=truncated, stop_reason="max_tokens"))
    d = ds.list_docs("credit_card_statement")[0]
    res = llm.extract_vision("credit_card_statement", d.image_path, "claude-opus-4-8", d.stem, "M4")
    assert res.truncated is True and res.error is None            # salvaged, not a crash
    assert len(res.data.get("transactions", [])) == 2            # two complete rows recovered


def test_extraction_truncation_unrecoverable_gives_code(monkeypatch):
    from src.extract import llm
    captured = {}
    monkeypatch.setattr(llm, "_get_client",
                        lambda: _fake_llm_client(captured, text='{"category":"cre', stop_reason="max_tokens"))
    d = ds.list_docs("credit_card_statement")[0]
    res = llm.extract_vision("credit_card_statement", d.image_path, "claude-opus-4-8", d.stem, "M4")
    assert res.truncated is True and res.error == "output_truncated"


def test_multipage_pdf_upload_rasterizes_all(monkeypatch):
    """A 3-page PDF upload rasterizes all pages and warns about free mode on real files."""
    import io
    import os
    from reportlab.pdfgen import canvas
    from fastapi.testclient import TestClient
    from webapp.main import app
    from src import store
    os.environ.pop("ANTHROPIC_API_KEY", None)
    buf = io.BytesIO(); c = canvas.Canvas(buf)
    for p in range(3):
        c.drawString(72, 720, f"Statement page {p + 1}"); c.showPage()
    c.save()
    store.clear()
    cl = TestClient(app); cl.cookies.set("fw_role", "admin")
    r = cl.post("/api/analyze", files={"file": ("s.pdf", buf.getvalue(), "application/pdf")},
                data={"doc_type": "credit_card_statement", "model": "M1_ocr_rules", "lang": "en"}).json()
    assert r["pages"] == 3   # all pages rasterized (admin explicit M1 baseline)
    store.clear()


def test_store_crud(tmp_path, monkeypatch):
    import src.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.init_db()
    rid = store.save({"filename": "x.pdf", "doc_type": "invoice", "source": "sample",
                      "sample": "INV-1", "model": "M3", "mode": "balanced", "status": "ok",
                      "confidence": 0.9, "needs_review": 0, "cost_usd": 0.02, "latency_s": 5.0,
                      "payload": {"doc_type": "invoice", "hi": 1}})
    assert isinstance(rid, int)
    assert len(store.list_records(doc_type="invoice")) == 1
    assert store.list_records(doc_type="receipt") == []
    assert store.get(rid)["payload"]["hi"] == 1
    assert store.delete(rid) is True and store.get(rid) is None


# ---------- product = always vision; no free garbage for users ----------
def test_user_never_runs_free_baseline():
    from webapp.main import _user_method
    m = _user_method()   # users always get the recommended vision model, never M1
    assert m != "M1_ocr_rules" and methods.METHODS[m].is_vision


def test_user_input_is_minimal_admin_keeps_controls():
    from fastapi.testclient import TestClient
    from webapp.main import app
    c = TestClient(app); c.cookies.set("fw_role", "user")
    h = c.get("/").text
    # users don't classify or pick a mode; no type-limiting chips
    assert 'id="docType"' not in h and 'id="mode"' not in h and 'hero-chips' not in h
    c.cookies.set("fw_role", "admin")
    ha = c.get("/").text
    assert 'id="docType"' in ha and 'id="model"' in ha   # admin keeps full controls


def test_detect_scored_flags_uncertain():
    from src import detect
    assert detect.detect_scored(text="lorem ipsum nothing relevant here")[1] == 0
    assert detect.detect_scored(text="Bank Statement — account holder, opening balance")[1] > 0


def test_user_sample_no_key_uses_reference():
    import os
    from fastapi.testclient import TestClient
    from webapp.main import app
    from src import store
    os.environ.pop("ANTHROPIC_API_KEY", None)
    store.clear()
    c = TestClient(app); c.cookies.set("fw_role", "user")
    r = c.post("/api/analyze", data={"sample": "bank_statement_0001",
                                     "doc_type": "bank_statement", "mode": "balanced"}).json()
    # no key -> sample demos on reference (gold) data, not the M1 baseline
    assert r["reference"] is True and r["cost_usd"] == 0.0
    assert r["status"] in ("ok", "review") and len(r["analysis"]["rows"]) == 12
    store.clear()


def test_user_upload_no_key_is_unconfigured():
    import io
    import os
    from reportlab.pdfgen import canvas
    from fastapi.testclient import TestClient
    from webapp.main import app
    os.environ.pop("ANTHROPIC_API_KEY", None)
    buf = io.BytesIO(); cv = canvas.Canvas(buf)
    cv.drawString(72, 720, "Bank Statement account holder opening balance closing balance")
    cv.showPage(); cv.save()
    c = TestClient(app); c.cookies.set("fw_role", "user")
    r = c.post("/api/analyze", files={"file": ("s.pdf", buf.getvalue(), "application/pdf")},
               data={"doc_type": "bank_statement", "mode": "accurate"}).json()
    assert r["status"] == "unconfigured"          # real upload w/o key -> setup prompt, no garbage


def test_upload_uncertain_type_needs_confirmation():
    import io
    import os
    from reportlab.pdfgen import canvas
    from fastapi.testclient import TestClient
    from webapp.main import app
    os.environ.pop("ANTHROPIC_API_KEY", None)
    buf = io.BytesIO(); cv = canvas.Canvas(buf)
    cv.drawString(72, 720, "lorem ipsum dolor sit amet consectetur"); cv.showPage(); cv.save()
    c = TestClient(app); c.cookies.set("fw_role", "user")
    r = c.post("/api/analyze", files={"file": ("x.pdf", buf.getvalue(), "application/pdf")},
               data={"mode": "balanced"}).json()  # no doc_type -> auto-detect fails -> ask
    assert r["status"] == "needs_type"


def test_user_can_bring_own_key():
    from fastapi.testclient import TestClient
    from webapp.main import app
    c = TestClient(app); c.cookies.set("fw_role", "user")
    h = c.get("/").text
    assert 'id="keyBtn"' in h and 'id="keyModal"' in h   # BYO: key UI available to users too


# ---------- web API ----------
def _admin_client():
    import os
    from fastapi.testclient import TestClient
    from webapp.main import app
    os.environ.pop("ANTHROPIC_API_KEY", None)  # force the offline rules path (no network)
    c = TestClient(app)
    c.cookies.set("fw_role", "admin")
    return c


def test_webapp_analyze_sample():
    c = _admin_client()
    assert c.get("/").status_code == 200
    r = c.post("/api/analyze", data={"sample": "bank_statement_0001",
                                     "doc_type": "bank_statement", "model": "M1_ocr_rules",
                                     "lang": "zh"})
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "bank_statement"
    assert len(j["analysis"]["rows"]) == 12
    assert j["analysis"]["reconcile"]["reconciles"] is True
    # every analyze run returns a one-line financial tip (rules tip when no key)
    assert j["advice"] and j["advice"]["text"].strip() and j["advice"]["source"] == "rules"
    # enriched product payload: labelled fields, confidence, insights, status, history id
    assert j["fields"] and all({"key", "label_en", "label_zh", "level"} <= set(f) for f in j["fields"])
    assert j["confidence"] is not None and j["status"] in ("ok", "review")
    assert isinstance(j["insights"], list) and isinstance(j["id"], int)


def test_login_and_role_gating():
    from fastapi.testclient import TestClient
    from webapp.main import app
    c = TestClient(app)
    # unauthenticated -> redirected to /login; the report page is admin-only
    assert c.get("/", follow_redirects=False).status_code == 303
    assert c.get("/login").status_code == 200
    c.cookies.set("fw_role", "user")
    assert c.get("/", follow_redirects=False).status_code == 200
    assert c.get("/report", follow_redirects=False).status_code == 303   # user can't see benchmark
    c.cookies.set("fw_role", "admin")
    assert c.get("/report", follow_redirects=False).status_code == 200


def test_history_roundtrip():
    from src import store
    c = _admin_client()
    store.clear()
    c.post("/api/analyze", data={"sample": "bank_statement_0001",
                                 "doc_type": "bank_statement", "model": "M1_ocr_rules"})
    recs = c.get("/api/history").json()["records"]
    assert len(recs) == 1 and recs[0]["doc_type"] == "bank_statement"
    rid = recs[0]["id"]
    assert c.get(f"/api/history/{rid}").json()["doc_type"] == "bank_statement"
    assert c.delete(f"/api/history/{rid}").json()["deleted"] is True
    assert c.get("/api/history").json()["records"] == []
    store.clear()


def test_webapp_chat_without_key_is_graceful():
    import os
    from fastapi.testclient import TestClient
    from webapp.main import app
    os.environ.pop("ANTHROPIC_API_KEY", None)
    c = TestClient(app)
    r = c.post("/api/chat", data={"doc_type": "bank_statement", "context": "{}", "messages": "[]"})
    assert r.status_code == 200 and "error" in r.json()


def test_webapp_analyze_thinking_flag_free_method():
    c = _admin_client()
    r = c.post("/api/analyze", data={"sample": "bank_statement_0001", "doc_type": "bank_statement",
                                     "model": "M1_ocr_rules", "thinking": "1"})
    assert r.status_code == 200
    # thinking is ignored on the free rules method (no error, no summary)
    assert r.json()["thinking"] is None
    import src.methods as m
    assert m.THINKING is False   # reset after the request


def test_report_compare_structure():
    from webapp.main import _report_compare
    comp = _report_compare()
    if comp is None:
        pytest.skip("no benchmark summary yet")
    assert comp["metrics"] and "rows" in comp["metrics"][0]
    assert "types" in comp["pivot"]


def test_report_page_is_readonly():
    c = _admin_client()
    h = c.get("/report").text
    assert 'id="dataPanel"' not in h and 'id="benchmarkPanel"' not in h
    # the data/benchmark-run endpoints are removed
    assert c.get("/api/data/status").status_code == 404
    assert c.post("/api/benchmark/run", data={}).status_code == 404


def test_i18n_key_parity():
    import re
    from pathlib import Path
    src = Path("webapp/static/i18n.js").read_text()
    en = set(re.findall(r'"([\w.]+)":', src.split('en:')[1].split('zh:')[0]))
    zh = set(re.findall(r'"([\w.]+)":', src.split('zh:')[1]))
    missing = en - zh
    assert not missing, f"zh missing keys: {missing}"

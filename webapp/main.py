"""FastAPI product: upload a financial document -> extract -> verify -> analyze.

Self-contained (server-rendered templates + bundled CSS/JS; no Node, no CDN).
Two roles share one pipeline: a User view (business results, confidence, evidence,
insights, history) and an Admin view (models, cost, benchmark, logs). Auth is
demo-grade (a role cookie, not real multi-tenant auth) — see the MVP roadmap.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import analyze as analyze_mod
from src import dataset as ds
from src import detect, insights, methods, present, store
from src.advice import build_advice
from src.freedom import build_plan
from src.extract.base import ExtractResult
from src.gold.build import gold_for

BASE = Path(__file__).resolve().parent
RESULTS = Path(__file__).resolve().parent.parent / "benchmark" / "results"

app = FastAPI(title="FinWise AI")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
if RESULTS.exists():
    app.mount("/results", StaticFiles(directory=RESULTS), name="results")
templates = Jinja2Templates(directory=str(BASE / "templates"))

MODEL_LABELS = {
    "M1_ocr_rules": "OCR + Rules · free",
    "M5_vision_haiku": "Haiku vision · $",
    "M0_pdftext_sonnet": "PDF-text + Sonnet · $",
    "M2_ocr_sonnet": "OCR + Sonnet · $$",
    "M3_vision_sonnet": "Sonnet vision · $$",
    "M4_vision_opus": "Opus vision · $$$",
}

# user-facing processing mode -> extraction method (hides model names from users)
MODE_MAP = {"fast": "M5_vision_haiku", "balanced": "M3_vision_sonnet", "accurate": "M4_vision_opus"}
_LEVEL_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.2}
ROLES = ("user", "admin")
MAX_PAGES = 10   # multi-page uploads: rasterize + send up to this many pages to the model

_NOTICES = {
    "pages_capped": {
        "en": "This document has {n} pages; only the first {m} were processed.",
        "zh": "该文档共 {n} 页,仅处理了前 {m} 页。"},
    "unconfigured": {
        "en": "Extraction isn't set up yet — an administrator needs to add an API key.",
        "zh": "抽取服务尚未启用 —— 需要管理员配置 API key。"},
    "needs_type": {
        "en": "Couldn't confidently detect the document type. Please pick the type and re-run.",
        "zh": "无法确定文档类型,请选择类型后重新分析。"},
    "partial": {
        "en": "This document is large — some rows may be missing (extraction hit the output limit). "
              "Results below are partial.",
        "zh": "文档较大,可能有部分行未抽全(达到输出上限)。下方为部分结果。"},
}


# ---------------- role / auth (demo-grade cookie) ----------------
def _role(request: Request) -> str | None:
    r = request.cookies.get("fw_role")
    return r if r in ROLES else None


def _have_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _avail_methods() -> list[str]:
    return list(methods.METHODS) if _have_key() else methods.FREE_METHODS


def _user_method() -> str:
    """The model every user run uses: the benchmark-recommended model, forced to
    vision so it is never the free M1 baseline (a benchmark-only baseline). Users
    don't pick a processing mode — they always get the best-value recommended model."""
    m = methods.recommended_method(have_key=True)
    return m if methods.METHODS[m].is_vision else "M3_vision_sonnet"


def _method_to_mode(method: str) -> str:
    if method == "M1_ocr_rules":
        return "free"
    return {v: k for k, v in MODE_MAP.items()}.get(method, "balanced")


# ---------------- login / onboarding ----------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if _role(request):
        return RedirectResponse(next or "/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"next": next})


@app.post("/login")
def login(role: str = Form("user"), next: str = Form("/")):
    role = role if role in ROLES else "user"
    resp = RedirectResponse(next or "/", status_code=303)
    resp.set_cookie("fw_role", role, max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("fw_role")
    return resp


# ---------------- analyze page ----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    role = _role(request)
    if not role:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", {
        "role": role,
        "types": ds.available_types(),
        "methods": [{"id": m, "label": MODEL_LABELS.get(m, m)} for m in _avail_methods()],
        "recommended": methods.recommended_method(),
        "have_key": _have_key(),
    })


@app.get("/api/samples")
def samples(type: str):
    return {"samples": [d.stem for d in ds.list_docs(type)[:60]]}


@app.get("/api/sample_image/{doc_type}/{stem}")
def sample_image(doc_type: str, stem: str):
    doc = ds.get_doc(doc_type, stem)
    if not doc.image_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(doc.image_path)


@app.post("/api/key")
def set_key(key: str = Form("")):
    if key.strip():
        os.environ["ANTHROPIC_API_KEY"] = key.strip()
        methods.llm._client = None  # rebuild client with the new key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        methods.llm._client = None
    return {"have_key": _have_key(),
            "methods": [{"id": m, "label": MODEL_LABELS.get(m, m)} for m in _avail_methods()],
            "recommended": methods.recommended_method()}


def _source_text(doc) -> str:
    """Document text (PDF text layer, else OCR) used for confidence + evidence."""
    try:
        if doc.text_path and Path(doc.text_path).exists():
            txt = Path(doc.text_path).read_text(errors="ignore")
            if txt.strip():
                return txt
    except Exception:
        pass
    try:
        from src.extract.pdf_text import pdf_text_layout
        if doc.pdf_path and Path(doc.pdf_path).exists():
            txt = pdf_text_layout(doc.pdf_path)
            if txt and txt.strip():
                return txt
    except Exception:
        pass
    try:
        from src.extract.ocr import ocr_image
        return ocr_image(doc.image_path) or ""
    except Exception:
        return ""


@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    file: UploadFile | None = File(None),
    sample: str = Form(""),
    doc_type: str = Form(""),
    model: str = Form(""),      # admin: explicit method (M0-M5)
    mode: str = Form(""),       # user: fast | balanced | accurate
    use_llm_cat: str = Form(""),
    thinking: str = Form(""),
    freedom: str = Form(""),    # opt-in Financial Freedom Plan (any role)
    lang: str = Form("en"),
):
    role = _role(request) or "user"
    lang = "zh" if lang == "zh" else "en"
    detected = doc_type
    filename = sample or (file.filename if file else "document")
    src_kind = "sample" if sample else "upload"
    n_pages = 1
    if sample:
        doc = ds.get_doc(doc_type, sample)
    else:
        if file is None:
            return JSONResponse({"error": "no file"}, status_code=400)
        tmp = Path(tempfile.mkdtemp())
        raw = tmp / (file.filename or "upload")
        raw.write_bytes(await file.read())
        img_path, pdf_path = raw, raw
        page_paths = (raw,)
        if raw.suffix.lower() == ".pdf":
            import fitz
            pdf = fitz.open(str(raw))
            n_pages = pdf.page_count
            imgs = []
            for i in range(min(n_pages, MAX_PAGES)):   # multi-page: rasterize every page
                pth = tmp / f"page_{i + 1}.png"
                pdf[i].get_pixmap(dpi=180).save(str(pth))
                imgs.append(pth)
            img_path, page_paths = imgs[0], tuple(imgs)
        else:
            pdf_path = None
        if not detected:
            detected, score = detect.detect_scored_from_paths(
                pdf_path=pdf_path, image_path=img_path, filename=file.filename)
            if score == 0:   # only the fallback guess -> ask the user instead of mis-extracting
                return JSONResponse({"status": "needs_type", "guess": detected,
                                     "notice": [_NOTICES["needs_type"][lang]]})
        doc = ds.DocPaths(stem=raw.stem, doc_type=detected, label_path=tmp / "x.json",
                          image_path=img_path, pdf_path=pdf_path or img_path, text_path=tmp / "x.txt",
                          image_paths=page_paths)

    # resolve method: admin can pick any method (incl. the M1 baseline for debug);
    # users don't choose — always the benchmark-recommended vision model, never M1
    mname = model if (model and role == "admin") else _user_method()
    uses_llm = methods.METHODS[mname].uses_llm

    # service-configuration gate: vision needs a key. No key -> reference-data demo for
    # samples, a clear "not configured" state for uploads. Never silently run M1 garbage.
    reference = False
    if uses_llm and not _have_key():
        if src_kind == "sample" and gold_for(doc):
            reference = True
            res = ExtractResult(method=mname, doc_type=detected, stem=doc.stem,
                                model="reference", data=gold_for(doc), latency_s=0.0, cost_usd=0.0)
        else:
            return JSONResponse({"status": "unconfigured", "role": role,
                                 "notice": [_NOTICES["unconfigured"][lang]]})
    else:
        methods.THINKING = bool(thinking) and role == "admin"
        try:
            res = methods.run_method(mname, doc)
        finally:
            methods.THINKING = False

    a = analyze_mod.analyze(res.data or {}, detected, use_llm_cat=bool(use_llm_cat))
    src_text = "" if res.error else _source_text(doc)
    fields, review = present.build_fields(res.data or {}, detected, src_text, a)
    cards = [] if res.error else insights.build_insights(a, res.data or {}, detected, lang, review)
    advice = None if res.error else build_advice(a, res.data or {}, detected, lang, have_key=_have_key())
    plan = (build_plan(a, res.data or {}, detected, lang, have_key=_have_key())
            if freedom and not res.error else None)
    conf_mean = (round(sum(_LEVEL_SCORE[f["level"]] for f in fields) / len(fields), 3)
                 if fields else None)
    status = "error" if res.error else ("review" if review else "ok")

    notice = []
    if src_kind == "upload" and n_pages > MAX_PAGES:
        notice.append(_NOTICES["pages_capped"][lang].format(n=n_pages, m=MAX_PAGES))
    if res.truncated:
        notice.append(_NOTICES["partial"][lang])

    payload = {
        "doc_type": detected,
        "model": mname, "model_label": MODEL_LABELS.get(mname, mname),
        "mode": _method_to_mode(mname), "pages": n_pages, "reference": reference,
        "truncated": res.truncated,
        "cost_usd": res.cost_usd, "latency_s": round(res.latency_s, 2),
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "error": res.error, "thinking": res.thinking, "notice": notice,
        "data": res.data or {}, "analysis": a,
        "fields": fields, "review": review, "insights": cards, "advice": advice,
        "freedom": plan,
        "confidence": conf_mean, "needs_review": len(review), "status": status,
    }
    rec_id = store.save({
        "filename": filename, "doc_type": detected, "source": src_kind,
        "sample": sample or None, "model": mname, "mode": _method_to_mode(mname),
        "status": status, "confidence": conf_mean, "needs_review": len(review),
        "cost_usd": res.cost_usd, "latency_s": round(res.latency_s, 2), "payload": payload,
    })
    payload["id"] = rec_id
    return JSONResponse(payload)


# ---------------- history ----------------
@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    role = _role(request)
    if not role:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "history.html",
                                      {"role": role, "types": ds.available_types(), "have_key": _have_key()})


@app.get("/api/history")
def api_history(doc_type: str = "", status: str = ""):
    return {"records": store.list_records(doc_type or None, status or None)}


@app.get("/api/history/{rec_id}")
def api_history_get(rec_id: int):
    rec = store.get(rec_id)
    if not rec:
        return JSONResponse({"error": "not found"}, status_code=404)
    return rec["payload"]


@app.delete("/api/history/{rec_id}")
def api_history_delete(rec_id: int):
    return {"deleted": store.delete(rec_id)}


# ---------------- chat ----------------
@app.post("/api/chat")
async def api_chat(
    doc_type: str = Form(""),
    context: str = Form("{}"),
    messages: str = Form("[]"),
    model: str = Form(""),
    thinking: str = Form(""),
):
    if not _have_key():
        return JSONResponse({"error": "Set an API key (top-right) to use chat."})
    try:
        history = json.loads(messages)
    except json.JSONDecodeError:
        history = []
    mdl = model if model in methods.llm.PRICING else "claude-sonnet-5"
    out = methods.llm.chat(doc_type, context, history, mdl, thinking=bool(thinking))
    return JSONResponse(out)


# ---------------- admin: model & benchmark ----------------
def _report_compare() -> dict | None:
    """Multi-model comparison built from benchmark/results/summary.csv (clean docs)."""
    import pandas as pd
    p = RESULTS / "summary.csv"
    if not p.exists():
        return None
    clean = pd.read_csv(p)
    clean = clean[clean["perturbation"] == "clean"]
    if clean.empty:
        return None
    macro = clean.groupby("method").agg(
        field_accuracy=("field_accuracy", "mean"), row_exact=("row_exact", "mean"),
        reconcile_rate=("reconcile_rate", "mean"), latency_s=("latency_s", "mean"),
        cost_per_100=("cost_per_100", "mean")).reset_index()
    meta = [("field_accuracy", "Field accuracy", True, "pct"),
            ("row_exact", "Transaction accuracy", True, "pct"),
            ("reconcile_rate", "Reconcile rate", True, "pct"),
            ("latency_s", "Latency", False, "sec"),
            ("cost_per_100", "Cost / 100 pages", False, "usd")]
    metrics = []
    for key, label, higher, fmt in meta:
        vals = macro[["method", key]].dropna()
        if vals.empty:
            continue
        mx = float(vals[key].max()) or 1.0
        best = vals.loc[(vals[key].idxmax() if higher else vals[key].idxmin()), "method"]
        rows = []
        for _, r in vals.iterrows():
            v = float(r[key])
            disp = (f"{v*100:.1f}%" if fmt == "pct" else f"{v:.2f}s" if fmt == "sec" else f"${v:.3f}")
            rows.append({"method": r["method"], "pct": round(v / mx * 100, 1),
                         "display": disp, "best": r["method"] == best})
        metrics.append({"label": label, "best": best, "rows": rows})
    piv = clean.pivot_table(index="method", columns="doc_type", values="field_accuracy")
    pivot = {"types": list(piv.columns),
             "rows": [{"method": m, "cells": [None if pd.isna(piv.loc[m, c]) else round(float(piv.loc[m, c]), 3)
                                              for c in piv.columns]} for m in piv.index]}
    return {"metrics": metrics, "pivot": pivot}


@app.get("/report", response_class=HTMLResponse)
def report(request: Request):
    if _role(request) != "admin":                    # admin-only page
        return RedirectResponse("/", status_code=303)
    rec = {}
    if (RESULTS / "recommendation.json").exists():
        rec = json.loads((RESULTS / "recommendation.json").read_text())
    report_md = Path("report/report.md").read_text() if Path("report/report.md").exists() else ""
    return templates.TemplateResponse(request, "report.html", {
        "role": "admin", "rec": rec, "compare": _report_compare(), "report_md": report_md,
        "have_key": _have_key(),
        "has_charts": (RESULTS / "accuracy_vs_cost.png").exists(),
        "has_robustness": (RESULTS / "robustness.png").exists(),
    })

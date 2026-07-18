# FinWise AI — Intelligent Personal-Finance Assistant

A web app that turns a financial document into **structured, verified data**, plus a
one-off **benchmark** that picks the model the app uses.

Upload a **bank statement / invoice / credit-card statement / receipt** →
auto-detected type → extracted to standard JSON with **transaction categorization,
debit/credit, date & currency normalization, balance reconciliation, duplicate & anomaly
detection, cash-flow, and a one-line financial tip**.

## Quick start — one command

```bash
./run.sh
```

Creates the venv, installs deps, and opens the app at **http://localhost:8000**.
- On first visit you land on a **login page** — pick a role (demo access, no signup):
  - **User** — business view: upload or try a sample → extract, verify, confidence,
    evidence, insights, export; history. Models/cost/benchmark are hidden.
  - **Admin** — everything above **plus** models, cost/latency, the **API Key**, and the
    **Model & Benchmark** page.
- Drop a PDF/PNG (or pick a built-in sample) → **Analyze** → see the results.
- **Extraction always uses a vision model** (Fast/Balanced/Best = Haiku/Sonnet/Opus) —
  no low-quality free fallback (the OCR+rules method is a *benchmark baseline only*).
- **The API key** can be provided three ways: `export ANTHROPIC_API_KEY=sk-ant-...` before
  `./run.sh` (operator), in-app via **API Key** top-right (works for **any user** —
  bring-your-own, session-only, never written to disk), or set once by Admin for everyone.
- **Without a key configured**: real uploads show a clear *"add your API key"* prompt, and
  built-in **samples still demo on reference data** (labeled).
- **Large documents**: extraction streams and allows a generous output budget; if a
  statement is so long that output is still truncated, you get the **partial rows plus a
  clear notice**, not a crash.

> Auth is **demo-grade** (a role cookie, not real multi-tenant auth) — see the MVP roadmap.

System prerequisites (once): `brew install tesseract poppler`.

## What the app does

- **Zero-config for users**: just upload → analyze. The **document type is auto-detected**
  (if genuinely unsure it asks; the result shows *"auto-detected as X · change"* to fix a
  wrong guess) — no manual classifying, no processing-mode dial.
- **Extracts** to a standard JSON schema using the **benchmark-recommended model** (currently
  Sonnet — measured equal to Opus at ~40% the cost). Admins can still pick the exact model,
  document type, and see **cost + latency**.
- **Verifies**: per-field **confidence** (High / Medium / Low) with **source evidence** —
  click a field to see where it came from in the document; low-confidence fields flag for
  review; **balance reconciliation** badge.
- **Explains**: structured **insight cards** (duplicate charges, recurring payments, spend
  anomalies, large bills, missing info) — each with evidence, impact, and a suggested action —
  plus a one-line financial tip.
- **Analyzes**: categorized transactions (chips), debit/credit, normalized dates/currency,
  and a cash-flow breakdown.
- **Remembers**: a **History** page (SQLite) of every analysis — filter, re-open, re-run, delete.

## Model selection & benchmark (run once)

The benchmark evaluates methods (OCR+rules baseline vs Haiku/Sonnet/Opus vision) on
field accuracy, transaction accuracy, reconciliation, latency, cost, and robustness,
then **recommends the model** the app uses by default. It is **pre-run once from the
CLI**; the **Model & Benchmark page just displays the comparison read-only** (no data
editing or running from the web). Results are persisted and reused.

```bash
# pre-run once (needs a key for the vision models), then open /report
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/run_benchmark.sh 20 5          # n per type, budget $5 (~$4–5 for 4 models)
```

That runs `M1 + Haiku + Sonnet + Opus`, writes `benchmark/results/`, and generates the
report. Re-run only when you want to refresh it.

Budget knobs: `--max-edge` (downscale images ~$7.6→$4.8), `--max-usd` (spend cap),
`--workers` (parallel, ~4× faster). Re-run only when needed: `--force`.

## Data prep (once)

```bash
python -m src.gold.validate 1000 --cache        # bank transaction gold (1000/1000 reconcile)
python -m src.generate.credit_card 200          # synthesize the missing credit-card type
```

## Tests

```bash
python -m pytest tests/ -q
```

## Layout

```
run.sh                  one-command launcher (venv + deps + uvicorn)
webapp/                 FastAPI product: main.py + templates/ + static/ (bundled CSS/JS)
src/detect.py           heuristic document-type detection
src/categorize.py       transaction categorization (rules + optional LLM)
src/analyze.py          one-stop analysis bundle for the UI
src/extract/            pdf_text, ocr, rules, llm (vision+text)
src/methods.py          M0-M5 registry + recommended_method()
src/postprocess.py      normalization, reconciliation, dedup, anomaly, cash-flow
src/metrics.py          field / line-item / transaction / reconciliation scoring
src/gold/               transaction gold from PDF text + reconciliation validator
src/generate/           credit-card statement generator
benchmark/run.py        parallel benchmark + recommend() + cache (recommendation.json)
report/generate.py      results -> report.md
```

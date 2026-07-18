"""Render benchmark results into report/report.md (tables + charts + narrative).

Run after a benchmark:  python report/generate.py
Reads benchmark/results/summary.csv (and raw.csv); degrades gracefully if only
the free-method baseline has been run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"
OUT = ROOT / "report" / "report.md"

METHOD_DESC = {
    "M0_pdftext_sonnet": "PDF text layer → Sonnet 5 (born-digital ceiling)",
    "M1_ocr_rules": "Tesseract OCR → regex rules (traditional baseline, $0)",
    "M2_ocr_sonnet": "Tesseract OCR → Sonnet 5",
    "M3_vision_sonnet": "Image → Sonnet 5 (vision, main)",
    "M4_vision_opus": "Image → Opus 4.8 (vision, quality ceiling)",
    "M5_vision_haiku": "Image → Haiku 4.5 (vision, cost floor)",
}


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    if not (RESULTS / "summary.csv").exists():
        raise SystemExit("no benchmark/results/summary.csv — run `python -m benchmark.run` first")
    summ = pd.read_csv(RESULTS / "summary.csv")
    clean = summ[summ["perturbation"] == "clean"].copy()

    lines: list[str] = []
    add = lines.append
    add("# FinWise AI — Financial Document Intelligence & LLM Benchmark — Report\n")
    add("## 1. Overview\n")
    add("We extract 4 financial document types (bank statement, credit-card statement, "
        "invoice, receipt) into a standard JSON schema and benchmark "
        "extraction methods on **field accuracy, transaction accuracy, balance "
        "reconciliation, latency, cost, and robustness**.\n")

    add("## 2. Methods compared\n")
    methods_present = sorted(summ["method"].unique())
    mtab = pd.DataFrame({"method": methods_present,
                         "description": [METHOD_DESC.get(m, "") for m in methods_present]})
    add(_md_table(mtab) + "\n")

    add("## 3. Field accuracy (clean documents)\n")
    if not clean.empty:
        piv = clean.pivot_table(index="method", columns="doc_type",
                                values="field_accuracy").round(3)
        add(_md_table(piv.reset_index()) + "\n")

    add("## 4. Transaction-level (bank & credit card)\n")
    ledger = clean[clean["doc_type"].isin(["bank_statement", "credit_card_statement"])]
    if not ledger.empty:
        lt = ledger[["method", "doc_type", "list_f1", "row_exact",
                     "reconcile_rate"]].round(3)
        add("Row F1, row-exact (≈ transaction accuracy), and balance-reconciliation rate:\n")
        add(_md_table(lt) + "\n")

    add("## 5. Cost & latency (per method, clean)\n")
    if not clean.empty:
        cl = clean.groupby("method").agg(
            field_accuracy=("field_accuracy", "mean"),
            latency_s=("latency_s", "mean"),
            cost_per_100=("cost_per_100", "mean")).round(4).reset_index()
        add(_md_table(cl) + "\n")
        add("![Accuracy vs Cost](../benchmark/results/accuracy_vs_cost.png)\n")

    if (RESULTS / "robustness.png").exists():
        add("## 6. Robustness (accuracy vs degradation)\n")
        add("![Robustness](../benchmark/results/robustness.png)\n")

    add("## 7. Notes & known issues\n")
    add("- Bank-statement labels store only a transaction *count*; per-transaction gold "
        "is derived from the born-digital PDF text layer (1000/1000 reconcile) and used "
        "as the transaction benchmark gold.\n")
    add("- Credit-card statements are synthetic (generated with reconciling ledgers); "
        "their labels carry full transaction rows.\n")
    add("- The rules baseline (M1) intentionally parses only header fields for "
        "invoice/receipt line items, so its item F1 is 0 there — the LLM methods handle items.\n")
    add("- Costs are computed from real `response.usage`; Sonnet 5 uses the introductory "
        "price ($2/$10 per 1M) valid through 2026-08-31.\n")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

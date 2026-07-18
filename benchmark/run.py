"""Benchmark runner: methods x perturbations x documents -> metrics.

Outputs benchmark/results/raw.csv (per-doc) and summary.csv (aggregated), plus
charts. LLM methods are skipped automatically when no ANTHROPIC_API_KEY is set
(free methods M1 still run); pass --methods to override.

Examples
--------
  # free methods only, small sample
  python -m benchmark.run --n 5 --methods M1_ocr_rules
  # full clean benchmark (needs API key)
  python -m benchmark.run --n 60
  # robustness sweep for vision on bank statements
  python -m benchmark.run --n 20 --types bank_statement --methods M3_vision_sonnet \
        --perturbations clean,rotate15,blur,noise,downscale100,scan_combo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import dataset as ds
from src import degrade, methods
from src.gold.build import gold_for
from src.metrics import score_doc

RESULTS = Path(__file__).resolve().parent / "results"
PERTURBED = Path("data/perturbed")


def _image_for(doc: ds.DocPaths, perturbation: str) -> Path:
    if perturbation == "clean":
        return doc.image_path
    out = PERTURBED / perturbation / f"{doc.stem}_page_1.png"
    if not out.exists():
        degrade.degrade_file(doc.image_path, perturbation, out)
    return out


def _row(res, score, perturbation) -> dict:
    fields = score.get("fields", {})
    lst = score.get("list") or {}
    return {
        "perturbation": perturbation, "method": res.method, "doc_type": res.doc_type,
        "stem": res.stem, "field_accuracy": fields.get("field_accuracy"),
        "list_f1": lst.get("f1"), "list_recall": lst.get("recall"),
        "row_exact": lst.get("row_exact_rate"),
        "reconciles": score.get("reconciles"),
        "latency_s": res.latency_s, "cost_usd": res.cost_usd,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "error": res.error,
    }


# --- rough token model for the pre-run cost estimate (actuals come from usage) ---
_AVG_OUT = {"bank_statement": 1200, "credit_card_statement": 1000, "invoice": 900,
            "receipt": 600}
_A4_RATIO = 0.707  # short/long edge


def _image_tokens(max_edge: int | None) -> int:
    e = max_edge or 2339
    return int(_A4_RATIO * e * e / 750)


def _est_doc_cost(mname: str, doc_type: str, max_edge: int | None) -> float:
    spec = methods.METHODS[mname]
    if not spec.uses_llm:
        return 0.0
    out = _AVG_OUT.get(doc_type, 800)
    if spec.is_vision:
        in_tok = _image_tokens(max_edge) + 300
    else:  # text methods (M0 pdf text, M2 ocr text)
        in_tok = 1300
    pin, pout = methods.llm.PRICING.get(spec.model, (0.0, 0.0))
    return (in_tok * pin + out * pout) / 1_000_000


def estimate_cost(n, types, method_names, perturbations, max_edge) -> float:
    docs = ds.stratified_sample(n, seed=0, types=types)
    total = 0.0
    for pert in perturbations:
        for doc in docs:
            for m in method_names:
                spec = methods.METHODS[m]
                if pert != "clean" and not (spec.is_vision or "ocr" in m):
                    continue
                total += _est_doc_cost(m, doc.doc_type, max_edge)
    return total


def _needs_image(mname: str) -> bool:
    spec = methods.METHODS[mname]
    return spec.is_vision or "ocr" in mname


def run_benchmark(n, types, method_names, perturbations, seed=42, max_edge=None,
                  max_usd=None, workers=8, on_result=None, on_progress=None,
                  verbose=True) -> pd.DataFrame:
    """Parallel benchmark. I/O-bound calls (API, tesseract subprocess) run in a
    thread pool; orchestration/state stays on the calling thread so on_result /
    on_progress callbacks are safe to drive a Streamlit UI.

    Budget note: with workers>1 the --max-usd cap is *soft* — up to (workers-1)
    paid calls may already be in flight when the cap trips (overshoot bounded by
    workers x per-call cost). Use workers=1 for a hard cap.
    """
    methods.VISION_MAX_EDGE = max_edge
    docs = ds.stratified_sample(n, seed=seed, types=types)

    gold_cache: dict[str, dict | None] = {}

    def get_gold(doc):
        if doc.stem not in gold_cache:
            gold_cache[doc.stem] = gold_for(doc)
        return gold_cache[doc.stem]

    # expand tasks (M0 skipped on degraded images); drop docs without reconciling gold
    tasks = []
    for pert in perturbations:
        for doc in docs:
            if get_gold(doc) is None:
                continue
            for mname in method_names:
                if pert != "clean" and not _needs_image(mname):
                    continue
                tasks.append((pert, doc, mname))

    # pre-generate degraded images single-threaded to avoid concurrent writes
    seen: set = set()
    for pert, doc, mname in tasks:
        if pert != "clean" and _needs_image(mname) and (pert, doc.stem) not in seen:
            seen.add((pert, doc.stem))
            _image_for(doc, pert)

    total = len(tasks)
    rows: list[dict] = []
    lock = threading.Lock()
    st = {"spent": 0.0, "handled": 0, "budget_hit": False}

    def work(task):
        pert, doc, mname = task
        img = _image_for(doc, pert) if _needs_image(mname) else None
        res = methods.run_method(mname, doc, img)
        return task, res, score_doc(res.data, get_gold(doc), doc.doc_type)

    def bump_progress():
        st["handled"] += 1
        if on_progress:
            on_progress(st["handled"], total)

    task_iter = iter(tasks)

    def next_task():
        """Next runnable task honoring the budget gate; None when exhausted."""
        for task in task_iter:
            spec = methods.METHODS[task[2]]
            if max_usd is not None and spec.uses_llm and st["spent"] >= max_usd:
                if not st["budget_hit"]:
                    st["budget_hit"] = True
                    if verbose:
                        print(f"  budget ${max_usd:.2f} reached; skipping further paid calls")
                bump_progress()  # count the skip so progress still reaches 100%
                continue
            return task
        return None

    def handle(fut):
        try:
            task, res, score = fut.result()
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"  ! task failed: {e}")
            bump_progress()
            return
        with lock:
            st["spent"] += res.cost_usd
        row = _row(res, score, task[0])
        rows.append(row)
        if on_result:
            on_result(row)
        bump_progress()
        if verbose:
            extra = (f"acc={score['fields']['field_accuracy']:.2f}"
                     if not res.error else f"ERR {res.error[:30]}")
            print(f"  [{st['handled']}/{total}] {task[0]} {res.method} {res.stem:26s} "
                  f"{extra}  {res.latency_s:.2f}s ${res.cost_usd:.4f} Σ${st['spent']:.2f}")

    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = set()
        for _ in range(workers):
            t = next_task()
            if t is None:
                break
            inflight.add(ex.submit(work, t))
        while inflight:
            done, pending = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = set(pending)
            for fut in done:
                handle(fut)
                t = next_task()
                if t is not None:
                    inflight.add(ex.submit(work, t))

    if verbose:
        print(f"\ntotal spend: ${st['spent']:.2f}  ({len(rows)} results)")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["perturbation", "doc_type", "stem", "method"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["perturbation", "method", "doc_type"])
    summ = g.agg(
        n=("stem", "count"),
        field_accuracy=("field_accuracy", "mean"),
        list_f1=("list_f1", "mean"),
        row_exact=("row_exact", "mean"),
        reconcile_rate=("reconciles", "mean"),
        latency_s=("latency_s", "mean"),
        cost_total=("cost_usd", "sum"),
    ).reset_index()
    summ["cost_per_100"] = summ["cost_total"] / summ["n"] * 100
    for c in ("field_accuracy", "list_f1", "row_exact", "reconcile_rate", "latency_s"):
        summ[c] = summ[c].round(4)
    summ["cost_per_100"] = summ["cost_per_100"].round(4)
    return summ


# Accuracy differences smaller than this are within-noise at ~20 docs/type, so we
# prefer the cheaper model rather than paying for a statistically-insignificant edge.
_ACC_TOL = 0.01  # 1 percentage point


def recommend(summ: pd.DataFrame) -> dict:
    """Pick the *best-value* model the product should use: the cheapest method whose
    quality is within _ACC_TOL of the best on clean docs — measured on field accuracy,
    transaction row-exact, AND reconciliation, so we never trade away real quality for
    price. Chasing the raw top accuracy would pick Opus over Sonnet for a <1pp edge at
    2.5x the cost; this picks Sonnet. Also returns the full ranking."""
    clean = summ[summ["perturbation"] == "clean"]
    if clean.empty:
        return {}
    macro = clean.groupby("method").agg(
        field_accuracy=("field_accuracy", "mean"),
        row_exact=("row_exact", "mean"),
        reconcile_rate=("reconcile_rate", "mean"),
        cost_per_100=("cost_per_100", "mean")).reset_index()

    best_acc = macro["field_accuracy"].max()
    best_row = macro["row_exact"].max()
    best_rec = macro["reconcile_rate"].max()
    # candidates: within tolerance of the best on all three quality metrics
    # (reconcile_rate is NaN for non-ledger-only method sets -> treat as passing).
    rec_ok = macro["reconcile_rate"].fillna(best_rec) >= best_rec - _ACC_TOL
    near_best = macro[(macro["field_accuracy"] >= best_acc - _ACC_TOL)
                      & (macro["row_exact"] >= best_row - _ACC_TOL) & rec_ok]
    if near_best.empty:  # no method clears all three; fall back to top field accuracy
        near_best = macro[macro["field_accuracy"] >= best_acc - _ACC_TOL]
    # cheapest among the near-best; break ties toward higher accuracy
    best = near_best.sort_values(["cost_per_100", "field_accuracy"],
                                 ascending=[True, False]).iloc[0]

    ranked = macro.sort_values(["field_accuracy", "cost_per_100"], ascending=[False, True])
    return {
        "recommended": str(best["method"]),
        "criterion": (f"best value: cheapest method within {_ACC_TOL:.0%} of the best on "
                      "field accuracy, transaction row-exact, and reconciliation"),
        "field_accuracy": round(float(best["field_accuracy"]), 4),
        "cost_per_100": round(float(best["cost_per_100"]), 4),
        "table": ranked[["method", "field_accuracy", "row_exact",
                         "reconcile_rate", "cost_per_100"]].round(4).to_dict("records"),
    }


def _fingerprint(n, types, method_names, perts, max_edge) -> str:
    """Hash of the inputs that would change the benchmark result (data + config)."""
    from src.gold.validate import GOLD_DIR
    counts = {t: v["count"] for t, v in ds.dataset_summary().items()}
    gold_n = len(list(GOLD_DIR.glob("*.json"))) if GOLD_DIR.exists() else 0
    payload = json.dumps({"counts": counts, "gold": gold_n, "n": n,
                          "types": sorted(types or ds.available_types()),
                          "methods": sorted(method_names), "perts": sorted(perts),
                          "max_edge": max_edge}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def charts(df: pd.DataFrame, summ: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    clean = summ[summ["perturbation"] == "clean"]
    # 1) accuracy vs cost per method (macro over types)
    if not clean.empty:
        macro = clean.groupby("method").agg(
            field_accuracy=("field_accuracy", "mean"),
            cost_per_100=("cost_per_100", "mean"),
            latency_s=("latency_s", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(macro["cost_per_100"], macro["field_accuracy"], s=80)
        for _, r in macro.iterrows():
            ax.annotate(r["method"], (r["cost_per_100"], r["field_accuracy"]),
                        fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Cost per 100 pages (USD)")
        ax.set_ylabel("Field accuracy (macro over types)")
        ax.set_title("Accuracy vs Cost")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "accuracy_vs_cost.png", dpi=120)
        plt.close(fig)
    # 2) robustness degradation curve (field accuracy by perturbation, per method)
    perts = list(df["perturbation"].unique())
    if len(perts) > 1:
        rob = df.groupby(["perturbation", "method"])["field_accuracy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        order = [p for p in ["clean", "jpeg40", "noise", "blur", "rotate5",
                             "downscale150", "rotate15", "downscale100", "scan_combo"]
                 if p in perts]
        for m in rob["method"].unique():
            sub = rob[rob["method"] == m].set_index("perturbation").reindex(order)
            ax.plot(order, sub["field_accuracy"], marker="o", label=m)
        ax.set_ylabel("Field accuracy")
        ax.set_title("Robustness: accuracy vs degradation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(RESULTS / "robustness.png", dpi=120)
        plt.close(fig)


def persist_results(df: pd.DataFrame, fingerprint: str, out: Path = RESULTS) -> dict:
    """Write raw/summary/charts/recommendation/fingerprint; return the recommendation.
    Shared by the CLI and the web-app benchmark job."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "summary.csv", index=False)
    charts(df, summ)
    rec = recommend(summ)
    (out / "recommendation.json").write_text(json.dumps(rec, indent=2))
    (out / "fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="docs per type")
    ap.add_argument("--types", default=None, help="comma list; default all available")
    ap.add_argument("--methods", default=None, help="comma list; default all (free-only w/o key)")
    ap.add_argument("--perturbations", default="clean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--max-edge", type=int, default=None,
                    help="downscale vision image long edge to N px (e.g. 1300) to cut token cost")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="budget cap: stop launching paid calls once spend reaches N "
                         "(soft with --workers>1; overshoot <= workers x per-call cost)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent calls (I/O-bound)")
    ap.add_argument("--yes", action="store_true", help="skip the cost-estimate confirmation")
    ap.add_argument("--force", "--rerun", action="store_true", dest="force",
                    help="re-run even if cached results match (default: reuse unchanged results)")
    args = ap.parse_args()

    types = args.types.split(",") if args.types else None
    perts = args.perturbations.split(",")
    methods.VISION_MAX_EDGE = args.max_edge
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if args.methods:
        method_names = args.methods.split(",")
    else:
        method_names = list(methods.METHODS) if have_key else methods.FREE_METHODS
        if not have_key:
            print("! No ANTHROPIC_API_KEY -> running free methods only:", method_names)
            print("  (set the key and pass --methods to include M0/M2-M5)\n")

    # persistence/cache: reuse unchanged results unless --force
    fp = _fingerprint(args.n, types, method_names, perts, args.max_edge)
    fp_file = RESULTS / "fingerprint.json"
    if (not args.force and (RESULTS / "summary.csv").exists() and fp_file.exists()
            and json.loads(fp_file.read_text()).get("fingerprint") == fp):
        print("reusing cached benchmark results (data + config unchanged). "
              "Pass --force to re-run.")
        if not (RESULTS / "recommendation.json").exists():
            rec = recommend(pd.read_csv(RESULTS / "summary.csv"))
            (RESULTS / "recommendation.json").write_text(json.dumps(rec, indent=2))
        rec = json.loads((RESULTS / "recommendation.json").read_text())
        print(f"recommended model: {rec.get('recommended')} "
              f"(field_acc {rec.get('field_accuracy')}, ${rec.get('cost_per_100')}/100pg)")
        return

    # pre-run cost estimate (paid methods only)
    est = estimate_cost(args.n, types, method_names, perts, args.max_edge)
    if est > 0:
        cap = f", cap ${args.max_usd:.2f}" if args.max_usd else ""
        edge = args.max_edge or "full(2339)"
        print(f"estimated cost: ~${est:.2f}  (max_edge={edge}{cap})")
        if args.max_usd and est > args.max_usd:
            print(f"  note: estimate exceeds cap; the run will stop at ${args.max_usd:.2f} "
                  f"(fewer docs covered).")
        if not args.yes and sys.stdin.isatty():
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted."); return

    df = run_benchmark(args.n, types, method_names, perts, seed=args.seed,
                       max_edge=args.max_edge, max_usd=args.max_usd, workers=args.workers)
    if df.empty:
        print("no results")
        return
    out = Path(args.out)
    rec = persist_results(df, fp, out)
    summ = summarize(df)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n===== SUMMARY (per perturbation x method x type) =====")
    print(summ.to_string(index=False))
    if rec:
        print(f"\n★ recommended model: {rec['recommended']} "
              f"(field_acc {rec['field_accuracy']}, ${rec['cost_per_100']}/100pg) — "
              f"the product uses this by default")
    print(f"\nwrote {out/'raw.csv'}, {out/'summary.csv'}, recommendation.json, charts -> {out}")


if __name__ == "__main__":
    main()

"""Per-field extraction confidence + source evidence (no bbox, text-based).

For a financial product the key question is not "did the model answer" but "can I
trust the answer". We derive a High / Medium / Low confidence per scalar field from
signals we actually have — presence, format validity, whether the value appears in
the document's own text (OCR / PDF text layer), and (for ledger balances) whether
the statement reconciles — and locate the matching source text so the UI can show
evidence. This is a heuristic, deliberately explainable, and needs no extra model.
"""
from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from src import postprocess as pp
from src.metrics import DATE, INT, NUM, STR, FIELD_SPEC

HIGH, MEDIUM, LOW = "high", "medium", "low"

# ledger balance fields whose confidence is backed by reconciliation
_BALANCE_FIELDS = {"opening_balance", "closing_balance", "previous_balance",
                   "new_balance", "credit_limit", "minimum_payment"}


def _value_candidates(value, kind: str) -> list[str]:
    """String forms of a value to look for in the source text."""
    cands = [str(value).strip()]
    if kind == NUM:
        n = pp.norm_money(value)
        if n is not None:
            cands += [f"{n:.2f}", f"{n:,.2f}", f"{n:g}", str(int(n)) if n == int(n) else f"{n}"]
    if kind == DATE:
        iso = pp.norm_date(value)
        if iso:
            cands.append(iso)
    # de-dup, drop empties/very short numeric-only noise
    out, seen = [], set()
    for c in cands:
        c = c.strip()
        if c and c not in seen and len(c) >= 2:
            seen.add(c); out.append(c)
    return out


def _locate(cands: list[str], source_lc: str, source: str) -> tuple[float, Optional[str], Optional[str]]:
    """Best (match_score 0-100, matched_substring, snippet) of any candidate in source."""
    best = (0.0, None, None)
    for c in cands:
        idx = source_lc.find(c.lower())
        if idx >= 0:  # exact substring hit -> strong evidence
            a, b = max(0, idx - 30), min(len(source), idx + len(c) + 30)
            return (100.0, source[idx:idx + len(c)], source[a:b].strip())
        r = fuzz.partial_ratio(c.lower(), source_lc) if source_lc else 0.0
        if r > best[0]:
            best = (r, None, None)
    return best


def field_confidence(value, kind: str, source: str, key: str = "",
                     reconciles: Optional[bool] = None) -> dict:
    """Confidence for one scalar field: {level, score, source, match, reason}."""
    if value is None or str(value).strip() == "":
        return {"level": LOW, "score": 0.0, "source": None, "match": None, "reason": "missing"}

    # format validity for the expected type
    fmt_ok = True
    if kind == NUM:
        fmt_ok = pp.norm_money(value) is not None
    elif kind == DATE:
        iso = pp.norm_date(value)
        fmt_ok = bool(iso) and iso != str(value) or bool(iso)  # parseable to a date
    elif kind == INT:
        try:
            int(value); fmt_ok = True
        except (TypeError, ValueError):
            fmt_ok = False

    source_lc = (source or "").lower()
    match_score, matched, snippet = _locate(_value_candidates(value, kind), source_lc, source or "")

    # reconciliation is strong evidence for balance fields
    if key in _BALANCE_FIELDS and reconciles is True and fmt_ok:
        return {"level": HIGH, "score": 0.99, "source": snippet, "match": matched,
                "reason": "reconciled"}

    if not fmt_ok:
        level, reason = LOW, "format"
    elif match_score >= 90:
        level, reason = HIGH, "found in source"
    elif match_score >= 60:
        level, reason = MEDIUM, "partial match"
    else:
        level, reason = MEDIUM if source_lc else HIGH, ("no source text" if not source_lc else "not found")
    return {"level": level, "score": round(match_score / 100, 2), "source": snippet,
            "match": matched, "reason": reason}


def assess(data: dict, doc_type: str, source: str, analysis: Optional[dict] = None) -> dict:
    """Confidence for every scalar field of a doc: {field: {level, score, source, ...}}."""
    reconciles = None
    if analysis:
        reconciles = (analysis.get("reconcile") or {}).get("reconciles")
    out = {}
    for key, kind in FIELD_SPEC.get(doc_type, {}).items():
        out[key] = field_confidence(data.get(key), kind, source, key=key, reconciles=reconciles)
    return out


def review_keys(conf: dict) -> list[str]:
    """Fields that need human review (low, or missing)."""
    return [k for k, c in conf.items() if c["level"] == LOW]

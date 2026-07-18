"""Presentation layer: turn a raw extraction into the User-facing `fields` payload.

Combines business labels (src/labels.py), normalized display values, and per-field
confidence + source evidence (src/confidence.py) into one list the frontend renders
as labelled rows with confidence badges and click-to-see-evidence.
"""
from __future__ import annotations

from typing import Optional

from src import confidence as conf_mod
from src import labels as lb
from src import postprocess as pp
from src.metrics import DATE, NUM, FIELD_SPEC


def _display(value, kind: str):
    if value is None:
        return None
    if kind == DATE:
        return pp.norm_date(value)
    if kind == NUM:
        n = pp.norm_money(value)
        return n if n is not None else value
    return value


def build_fields(data: dict, doc_type: str, source: str,
                 analysis: Optional[dict] = None) -> tuple[list[dict], list[str]]:
    """Returns (fields, review_keys).

    fields: [{key, label_en, label_zh, value, kind, level, source, match}] in schema order.
    review_keys: business labels (EN) of fields that need human review (low confidence).
    """
    conf = conf_mod.assess(data, doc_type, source, analysis)
    fields, review = [], []
    for key, kind in FIELD_SPEC.get(doc_type, {}).items():
        c = conf.get(key, {})
        en, zh = lb.field_label(doc_type, key)
        fields.append({
            "key": key, "label_en": en, "label_zh": zh,
            "value": _display(data.get(key), kind), "kind": kind,
            "level": c.get("level", "medium"), "source": c.get("source"),
            "match": c.get("match"),
        })
        if c.get("level") == conf_mod.LOW:
            review.append(en)
    return fields, review

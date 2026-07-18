"""The comparable extraction methods M0-M5.

Scored input is the IMAGE (or a degraded image) for M1-M5. M0 uses the PDF text
layer as a born-digital upper-bound / ceiling (clean docs only, not robustness).

    M0_pdftext_sonnet : PDF text layer -> Sonnet 5   (ceiling; ~free-ish text)
    M1_ocr_rules      : Tesseract OCR  -> regex rules (traditional baseline; $0)
    M2_ocr_sonnet     : Tesseract OCR  -> Sonnet 5
    M3_vision_sonnet  : image          -> Sonnet 5    (main)
    M4_vision_opus    : image          -> Opus 4.8    (quality ceiling)
    M5_vision_haiku   : image          -> Haiku 4.5   (cost floor)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src import dataset as ds
from src.extract import llm, rules
from src.extract.base import ExtractResult
from src.extract.ocr import ocr_image
from src.extract.pdf_text import pdf_text_plain


@dataclass(frozen=True)
class MethodSpec:
    name: str
    uses_llm: bool
    is_vision: bool
    model: Optional[str]
    run: Callable[[ds.DocPaths, Optional[Path]], ExtractResult]


def _m0(doc, image_path=None):
    text = pdf_text_plain(doc.pdf_path)
    return llm.extract_text(doc.doc_type, text, "claude-sonnet-5", doc.stem,
                            "M0_pdftext_sonnet", thinking=THINKING)


def _ocr_all(doc, image_path) -> str:
    """OCR the given image, or every page image of a multi-page upload."""
    if image_path:
        return ocr_image(image_path)
    pages = doc.image_paths or (doc.image_path,)
    return "\n".join(ocr_image(p) for p in pages)


def _m1(doc, image_path=None):
    t0 = time.perf_counter()
    text = _ocr_all(doc, image_path)
    data = rules.parse(doc.doc_type, text)
    return ExtractResult(method="M1_ocr_rules", doc_type=doc.doc_type, stem=doc.stem,
                         data=data, latency_s=time.perf_counter() - t0, cost_usd=0.0)


def _ocr_llm(doc, image_path, model, name):
    t0 = time.perf_counter()
    text = _ocr_all(doc, image_path)
    ocr_dt = time.perf_counter() - t0
    res = llm.extract_text(doc.doc_type, text, model, doc.stem, name, thinking=THINKING)
    res.latency_s += ocr_dt  # include OCR in method latency
    return res


def _m2(doc, image_path=None):
    return _ocr_llm(doc, image_path, "claude-sonnet-5", "M2_ocr_sonnet")


# Downscale the long edge of vision inputs to this many px before sending
# (None = full 2339px). Cuts image tokens ~quadratically; set by the runner.
VISION_MAX_EDGE: int | None = None
# Enable adaptive thinking (with summarized display) on the LLM-backed methods.
# Set per-request by the web app; ignored by the free rules method.
THINKING: bool = False


def _vision(doc, image_path, model, name):
    img = image_path or (list(doc.image_paths) if doc.image_paths else doc.image_path)
    return llm.extract_vision(doc.doc_type, img, model, doc.stem, name,
                              max_edge=VISION_MAX_EDGE, thinking=THINKING)


def _m3(doc, image_path=None):
    return _vision(doc, image_path, "claude-sonnet-5", "M3_vision_sonnet")


def _m4(doc, image_path=None):
    return _vision(doc, image_path, "claude-opus-4-8", "M4_vision_opus")


def _m5(doc, image_path=None):
    return _vision(doc, image_path, "claude-haiku-4-5", "M5_vision_haiku")


METHODS: dict[str, MethodSpec] = {
    "M0_pdftext_sonnet": MethodSpec("M0_pdftext_sonnet", True, False, "claude-sonnet-5", _m0),
    "M1_ocr_rules":      MethodSpec("M1_ocr_rules", False, False, None, _m1),
    "M2_ocr_sonnet":     MethodSpec("M2_ocr_sonnet", True, False, "claude-sonnet-5", _m2),
    "M3_vision_sonnet":  MethodSpec("M3_vision_sonnet", True, True, "claude-sonnet-5", _m3),
    "M4_vision_opus":    MethodSpec("M4_vision_opus", True, True, "claude-opus-4-8", _m4),
    "M5_vision_haiku":   MethodSpec("M5_vision_haiku", True, True, "claude-haiku-4-5", _m5),
}

FREE_METHODS = [n for n, s in METHODS.items() if not s.uses_llm]
LLM_METHODS = [n for n, s in METHODS.items() if s.uses_llm]


def run_method(name: str, doc: ds.DocPaths, image_path: Optional[Path] = None) -> ExtractResult:
    return METHODS[name].run(doc, image_path)


_RECOMMENDATION = Path(__file__).resolve().parent.parent / "benchmark" / "results" / "recommendation.json"


def recommended_method(have_key: Optional[bool] = None) -> str:
    """The method the product should use by default: the benchmark's recommendation
    when available and runnable, else a sensible fallback."""
    if have_key is None:
        have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if _RECOMMENDATION.exists():
        try:
            rec = json.loads(_RECOMMENDATION.read_text()).get("recommended")
            if rec in METHODS and (have_key or not METHODS[rec].uses_llm):
                return rec
        except Exception:
            pass
    return "M3_vision_sonnet" if have_key else "M1_ocr_rules"


if __name__ == "__main__":
    # Only the free method runs without an API key.
    d = ds.list_docs("bank_statement")[0]
    res = run_method("M1_ocr_rules", d)
    print("M1 on", d.stem, "-> latency %.2fs, cost $%.4f" % (res.latency_s, res.cost_usd))
    print("  fields:", {k: res.data.get(k) for k in ("statement_id", "opening_balance", "closing_balance", "transactions")})
    print("  rows extracted:", len(res.data.get("transaction_rows", [])))

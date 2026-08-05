"""Claude-based extraction (vision + text), with structured output and cost tracking.

Vision methods (M3/M4/M5) send the page image; text methods (M0/M2) send extracted
text. Both constrain the response to the per-type JSON schema via output_config.
Cost is computed from the real response.usage — never estimated.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from src.extract.base import ExtractResult
from src.schemas import strict_json_schema

# $ per 1M tokens (input, output). Sonnet 5 uses the introductory price
# ($2/$10, valid through 2026-08-31); switch to 3/15 afterwards.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_SYSTEM = (
    "You are a precise financial-document extraction engine. Extract the requested "
    "fields exactly as they appear in the document. Use the document's own values; "
    "do not invent, compute, or reformat numbers (keep amounts as shown). For dates "
    "use ISO YYYY-MM-DD. Return every transaction / line-item row present. Within a row, "
    "only fill the amount side that applies (e.g. a credit row has no debit) — omit the "
    "other rather than writing 0. Output only the structured JSON."
)

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:  # thread-safe single-init (the client itself is concurrency-safe)
            if _client is None:
                import anthropic  # lazy: only needed when actually calling the API
                _client = anthropic.Anthropic()
    return _client


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return (in_tok * pin + out_tok * pout) / 1_000_000


def _encode_image(image_path: str | Path, max_edge: Optional[int]) -> tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"


def _thinking_summary(resp) -> Optional[str]:
    parts = [b.thinking for b in resp.content
             if getattr(b, "type", "") == "thinking" and getattr(b, "thinking", "")]
    return "\n".join(parts) or None


# Output budget for extraction. Financial statements can hold hundreds of rows, so
# the old 4096 truncated real docs mid-array. All current models allow >=64k output
# (Haiku 64k; Sonnet 5 / Opus 4.8 128k), so a generous single value is safe.
_EXTRACT_MAX = 16000
_EXTRACT_MAX_THINKING = 24000


def _repair_truncated_json(s: str) -> Optional[dict]:
    """Best-effort salvage of JSON cut off by max_tokens: trim to the last complete
    object and balance the open brackets. Recovers partial rows instead of failing."""
    i = s.rfind("}")
    if i < 0:
        return None
    s = s[:i + 1]
    s += "]" * max(0, s.count("[") - s.count("]"))
    s += "}" * max(0, s.count("{") - s.count("}"))
    try:
        return json.loads(s)
    except Exception:
        return None


def _call(doc_type: str, model: str, content: list, stem: str, method: str,
          thinking: bool = False) -> ExtractResult:
    schema = strict_json_schema(doc_type)
    kwargs = {"model": model, "max_tokens": _EXTRACT_MAX_THINKING if thinking else _EXTRACT_MAX,
              "system": _SYSTEM, "messages": [{"role": "user", "content": content}],
              "output_config": {"format": {"type": "json_schema", "schema": schema}}}
    if thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    t0 = time.perf_counter()
    try:
        with _get_client().messages.stream(**kwargs) as stream:  # stream: long output, no timeout
            resp = stream.get_final_message()
    except Exception as e:  # network/auth/api errors surface as a failed result
        return ExtractResult(method=method, doc_type=doc_type, stem=stem, model=model,
                             latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    text = next((b.text for b in resp.content if b.type == "text"), "")
    truncated = getattr(resp, "stop_reason", None) == "max_tokens"
    err = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _repair_truncated_json(text) if truncated else None
        if data is None:                       # unrecoverable -> a clear, mappable code
            data, err = {}, ("output_truncated" if truncated else "output_unparseable")
    u = resp.usage
    return ExtractResult(
        method=method, doc_type=doc_type, stem=stem, model=model, data=data,
        latency_s=dt, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None),
        cost_usd=_cost(model, u.input_tokens, u.output_tokens), error=err,
        thinking=_thinking_summary(resp) if thinking else None, truncated=truncated,
    )


def extract_vision(doc_type, image_path, model: str, stem: str,
                   method: str, max_edge: Optional[int] = None,
                   thinking: bool = False) -> ExtractResult:
    """`image_path` may be a single path or a list/tuple of page images (multi-page)."""
    paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
    content = []
    for p in paths:
        b64, media = _encode_image(p, max_edge)
        content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
    instr = f"Extract all fields for this {doc_type.replace('_', ' ')}."
    if len(paths) > 1:
        instr += (f" This document has {len(paths)} pages shown above in order; read ALL of them and "
                  "combine — transactions and totals often span multiple pages.")
    instr += (" If the document spans multiple months or statement periods, keep each transaction row's "
              "date with its own correct month and year (do not collapse different months together).")
    content.append({"type": "text", "text": instr})
    return _call(doc_type, model, content, stem, method, thinking=thinking)


def extract_text(doc_type: str, text: str, model: str, stem: str, method: str,
                 thinking: bool = False) -> ExtractResult:
    content = [{"type": "text",
                "text": f"Document text of a {doc_type.replace('_', ' ')}:\n\n{text}\n\n"
                        f"Extract all fields."}]
    return _call(doc_type, model, content, stem, method, thinking=thinking)


_CHAT_SYSTEM = (
    "You are a helpful financial-document assistant. Answer the user's questions about the "
    "document below using ONLY its extracted data (and general financial knowledge for "
    "explanations). Be concise and precise; cite specific figures. If the data doesn't "
    "contain the answer, say so.\n\nExtracted data (JSON):\n{context}"
)


def chat(doc_type: str, context: str, messages: list, model: str,
         thinking: bool = False) -> dict:
    """Doc-grounded Q&A. `messages` is the multi-turn history [{role, content}]."""
    system = _CHAT_SYSTEM.format(context=context)
    kwargs = {"model": model, "max_tokens": 4096, "system": system, "messages": messages}
    if thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    t0 = time.perf_counter()
    try:
        resp = _get_client().messages.create(**kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t0, 2)}
    u = resp.usage
    return {
        "answer": next((b.text for b in resp.content if b.type == "text"), ""),
        "thinking": _thinking_summary(resp) if thinking else None,
        "latency_s": round(time.perf_counter() - t0, 2),
        "cost_usd": _cost(model, u.input_tokens, u.output_tokens),
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
    }


_ADVISE_SYSTEM = (
    "You are a concise personal-finance advisor. Given a financial document's extracted "
    "data and a short analysis summary, reply with EXACTLY ONE short, actionable financial "
    "tip (about 30 words max). Cite one specific figure from the data. No preamble, no "
    "lists, no markdown — just the single sentence. Respond in {lang}."
)


def advise(doc_type: str, context: str, summary: str, lang: str,
           model: str = "claude-haiku-4-5") -> dict:
    """One-sentence financial tip grounded in the extracted data. Cheap by design
    (short output on Haiku). Returns {text, cost_usd, latency_s, ...} or {error}."""
    lang_name = "Chinese (简体中文)" if lang == "zh" else "English"
    system = _ADVISE_SYSTEM.format(lang=lang_name)
    user = (f"Document type: {doc_type.replace('_', ' ')}\n"
            f"Analysis summary: {summary}\n\n"
            f"Extracted data (JSON):\n{context}\n\n"
            f"Give one short financial tip.")
    kwargs = {"model": model, "max_tokens": 200, "system": system,
              "messages": [{"role": "user", "content": user}]}
    t0 = time.perf_counter()
    try:
        resp = _get_client().messages.create(**kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t0, 2)}
    u = resp.usage
    return {
        "text": next((b.text for b in resp.content if b.type == "text"), "").strip(),
        "latency_s": round(time.perf_counter() - t0, 2),
        "cost_usd": _cost(model, u.input_tokens, u.output_tokens),
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
    }


_STORY_SYSTEM = (
    "You are a warm, encouraging personal-finance coach. Using ONLY the pre-computed "
    "figures provided, write a short motivating note (2-3 sentences, about 50 words) about "
    "the person's path to financial independence. Never invent, recompute, or add any "
    "figure not given. If there is no surplus, be kind and focus on the first step. No "
    "markdown, no lists, no preamble — just the sentences. Respond in {lang}."
)


def story(figures: dict, lang: str, model: str = "claude-haiku-4-5") -> dict:
    """A motivating 2-3 sentence financial-freedom note grounded in pre-computed figures.

    Only the computed scalars are passed in (never raw rows / extracted data), so the
    model can phrase the numbers we gave it but physically cannot invent new ones. Cheap
    by design (short output on Haiku). Returns {text, cost_usd, latency_s, ...} or {error}."""
    lang_name = "Chinese (简体中文)" if lang == "zh" else "English"
    system = _STORY_SYSTEM.format(lang=lang_name)
    user = ("Pre-computed figures (use verbatim, do not change or add any number):\n"
            + json.dumps(figures, ensure_ascii=False)
            + "\n\nWrite the motivating note.")
    kwargs = {"model": model, "max_tokens": 220, "system": system,
              "messages": [{"role": "user", "content": user}]}
    t0 = time.perf_counter()
    try:
        resp = _get_client().messages.create(**kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t0, 2)}
    u = resp.usage
    return {
        "text": next((b.text for b in resp.content if b.type == "text"), "").strip(),
        "latency_s": round(time.perf_counter() - t0, 2),
        "cost_usd": _cost(model, u.input_tokens, u.output_tokens),
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
    }


_DOC_TYPES = ("bank_statement", "credit_card_statement", "invoice", "receipt")

_CLASSIFY_SYSTEM = (
    "You classify a financial document image into exactly ONE of these type keys:\n"
    "- bank_statement: a bank account statement with deposits/withdrawals and a running balance\n"
    "- credit_card_statement: a credit-card statement or a spending report of card transactions "
    "(charges, minimum payment, new/previous balance, credit limit)\n"
    "- invoice: a bill issued to a customer (invoice number, bill to, amount due)\n"
    "- receipt: a store purchase receipt (merchant, line items, subtotal/total)\n"
    "Pick the single closest key even when the document is non-standard (e.g. a monthly "
    "spending summary of card activity -> credit_card_statement). Reply with ONLY the type "
    "key, lowercase, nothing else."
)


def classify_doc_type(image_paths, model: str = "claude-haiku-4-5",
                      max_edge: Optional[int] = 1024) -> Optional[str]:
    """Vision classification of a document into one of the four types — used to skip the
    manual type picker. Cheap by design (first pages, downscaled image, tiny output on
    Haiku). Returns a valid type key, or None on error / unrecognized output."""
    paths = list(image_paths) if isinstance(image_paths, (list, tuple)) else [image_paths]
    content = []
    for p in paths[:2]:   # first 1-2 pages is plenty to identify the type
        b64, media = _encode_image(p, max_edge)
        content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
    content.append({"type": "text", "text": "Classify this document. Reply with only the type key."})
    try:
        resp = _get_client().messages.create(
            model=model, max_tokens=12, system=_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": content}])
    except Exception:
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "").strip().lower()
    # credit_card_statement contains "statement" too, so match the most specific first
    for t in ("credit_card_statement", "bank_statement", "invoice", "receipt"):
        if t in text:
            return t
    return None

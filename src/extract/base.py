"""Shared extraction result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractResult:
    method: str
    doc_type: str
    stem: str
    data: dict[str, Any] = field(default_factory=dict)   # extracted fields (schema-shaped)
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cost_usd: float = 0.0
    model: Optional[str] = None
    error: Optional[str] = None
    thinking: Optional[str] = None   # summarized reasoning when thinking is enabled
    truncated: bool = False          # hit max_tokens; data may be partial (salvaged)

    def as_row(self) -> dict:
        return {
            "method": self.method, "doc_type": self.doc_type, "stem": self.stem,
            "latency_s": round(self.latency_s, 3),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": round(self.cost_usd, 6), "model": self.model, "error": self.error,
        }

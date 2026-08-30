from __future__ import annotations

import json
from typing import Any


class StructuredAnswerInterface:
    """Turn an architecture-owned structured answer/request into language."""

    def __init__(self, llm, trace: bool = True):
        self.llm = llm
        self.trace = trace

    def render(self, request: dict[str, Any]) -> str | None:
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True)
        try:
            out = self.llm.generate(payload)
        except Exception as exc:
            if self.trace:
                print(f"[LLM INTERFACE] render failed: {exc}", flush=True)
            return None
        text = str(out or "").strip()
        if not text:
            return None
        if text.startswith("{") or text.startswith("```"):
            return None
        return text

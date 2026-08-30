from __future__ import annotations

import json
from typing import Any


class StructuredAnswerInterface:
    """Render an architecture-owned structured request and expose the exact input for debugging."""

    def __init__(self, llm, trace: bool = True, show_request: bool = True):
        self.llm = llm
        self.trace = trace
        self.show_request = show_request

    def render(self, request: dict[str, Any]) -> str | None:
        payload = json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True)
        if self.trace and self.show_request:
            print("[LLM REQUEST BEGIN]", flush=True)
            print(payload, flush=True)
            print("[LLM REQUEST END]", flush=True)
        try:
            out = self.llm.generate(payload)
        except Exception as exc:
            if self.trace:
                print(f"[LLM INTERFACE] render failed: {exc}", flush=True)
            return None
        text = str(out or "").strip()
        if self.trace and self.show_request:
            print("[LLM RESPONSE BEGIN]", flush=True)
            print(text or "<empty>", flush=True)
            print("[LLM RESPONSE END]", flush=True)
        if not text:
            return None
        if text.startswith("{") or text.startswith("```"):
            return None
        return text

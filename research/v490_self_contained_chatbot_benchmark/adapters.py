
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_factory(path):
    path=Path(path).resolve()
    spec=importlib.util.spec_from_file_location(
        "benchmark_factory",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter factory: {path}")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    factory=getattr(mod,"create_adapter",None)
    if factory is None:
        raise RuntimeError(
            f"{path} must define create_adapter(name='...')"
        )
    return factory


class LLMOnlyAdapter:
    name="llm"

    def __init__(self,model_path,max_new_tokens=96):
        from llm_backend import LocalLLM
        self.llm=LocalLLM(
            model_path,
            max_new_tokens=max_new_tokens,
        )
        self.history=[]

    def reset(self):
        self.history=[]

    def respond(self,text):
        self.history.append(("user",text))
        context="\n".join(
            f"{speaker}: {message}"
            for speaker,message in self.history[-8:]
        )
        answer=self.llm.generate(
            "You are a conversational assistant. "
            "Respond naturally and use the conversation context. "
            "Return only the response.",
            f"CONVERSATION:\n{context}\n\nCURRENT TURN:\n{text}",
        ).strip()
        self.history.append(("assistant",answer))
        return answer


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    memory: Path = Path(
        r"C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite"
    )
    model: Path = Path(
        r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM2-1.7B-Instruct"
    )
    parser_model: str = "en_core_web_sm"

    freeze_learning: bool = True
    use_policies: bool = False

    recent_turns: int = 8
    memory_facts: int = 24
    context_words: int = 180
    participant_max_tokens: int = 80
    realizer_max_tokens: int = 80


from __future__ import annotations

import sqlite3

from memory import TypedMemory
from perception import Perceiver
from assistant_core import CognitiveAssistant


class Adapter:
    name="typed_architecture"

    def __init__(self,memory_path):
        self.con=sqlite3.connect(memory_path)
        self.memory=TypedMemory(self.con)
        self.assistant=CognitiveAssistant(
            self.memory,
            Perceiver(None),
            llm=None,
            trace=False,
        )

    def reset(self):
        self.assistant.reset()

    def respond(self,text):
        return self.assistant.respond(text)["response"]


def create_adapter(name="architecture"):
    import os
    path=os.environ.get(
        "STARK_MEMORY",
        r".\results\combined_cognitive_memory.sqlite",
    )
    return Adapter(path)

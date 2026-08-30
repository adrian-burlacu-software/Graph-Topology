from __future__ import annotations

import sqlite3
from pathlib import Path

from assistant_cli import CognitiveBridge, Fact
from schema import SCHEMA


def build_memory() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)

    def source(dataset: str, content_type: str) -> int:
        cur = con.execute(
            "INSERT INTO sources(dataset,source_path,record_key,content_type) VALUES(?,?,?,?)",
            (dataset, "test", dataset, content_type),
        )
        return cur.lastrowid

    # The simple fixture intentionally contains both useful semantic evidence
    # and lexical noise, to ensure ranking filters the latter.
    sid_sem = source("conceptnet", "semantic")
    sid_lex = source("wordnet", "lexical")

    def concept(name: str, typ: str = "noun") -> int:
        cur = con.execute(
            "INSERT INTO concepts(canonical,display,concept_type) VALUES(?,?,?)",
            (name, name, typ),
        )
        return cur.lastrowid

    dog = concept("dog")
    cat = concept("cat")
    universe = concept("universe")
    cosmos = concept("cosmos")
    infinite = concept("infinite")
    content = concept("content")

    rows = [
        (dog, "has_property", None, "red", "semantic", sid_sem),
        (universe, "has_property", infinite, "infinite", "semantic", sid_sem),
        (universe, "synonym", cosmos, "cosmos", "lexical", sid_lex),
        (universe, "hypernym", content, "content", "lexical", sid_lex),
    ]
    for subject_id, predicate, object_id, object_text, fact_type, source_id in rows:
        con.execute(
            """
            INSERT INTO facts(
                subject_id,predicate,object_id,object_text,fact_type,domain,
                source_id,confidence,frequency,answerable,metadata_json
            ) VALUES(?,?,?,?,?,?,?,1.0,1.0,1,'{}')
            """,
            (subject_id, predicate, object_id, object_text, fact_type, None, source_id),
        )
    con.commit()
    return con



class FakeTeacher:
    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "The teacher invented an unsupported fact."

def main() -> None:
    bridge = CognitiveBridge(memory_path=None, teacher=None, freeze_knowledge=True, trace=False)
    bridge.memory = build_memory()

    assert bridge.turn_once("the dog is red") == "The dog is red."
    assert bridge.turn_once("what color is the dog?") == "The dog is red."
    assert bridge.turn_once("and the cat is it also red?") == "I'm not certain yet."
    assert bridge.last_target["subject"] == "cat"

    bridge.reset()
    assert bridge.turn_once("the dog is red") == "The dog is red."
    assert bridge.turn_once("help me") == "Sure. What would you like me to do?"
    assert bridge.last_target["subject"] is None

    bridge.reset()
    assert bridge.turn_once("I want to know more about the universe.") == "The universe is infinite."
    assert bridge.last_target["subject"] == "universe"
    bridge.reset()
    bridge.memory = build_memory()
    assert bridge.turn_once("the universe is large") == "The universe is large."
    assert bridge.turn_once("is it black?") == "I'm not certain yet."

    bridge.reset()
    bridge.turn_once("the dog is red")
    bridge.turn_once("the dog is red")
    assert len(bridge.state.facts) == 1

    # Factual questions must not be delegated to the teacher merely because
    # long-term evidence is missing. The bridge falls back safely instead.
    teacher = FakeTeacher()
    bridge = CognitiveBridge(memory_path=None, teacher=teacher, freeze_knowledge=True, trace=False)
    assert bridge.turn_once("what is the universe?") == "I'm not sure yet."
    assert teacher.calls == []

    print("V515 smoke tests: PASS")


if __name__ == "__main__":
    main()

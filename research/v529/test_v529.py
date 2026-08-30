from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from topic_catalog import topic_rows, format_topics
from llm_interface import StructuredAnswerInterface


class DummyLLM:
    def __init__(self):
        self.payload = None
    def generate(self, payload):
        self.payload = payload
        return "The dog is red."


def main():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE facts(fact_id INTEGER,subject_id INTEGER,predicate TEXT,object_id INTEGER,object_text TEXT,fact_type TEXT,domain TEXT,source_id INTEGER,confidence REAL,frequency REAL,answerable INTEGER)")
    con.execute("CREATE TABLE concepts(concept_id INTEGER,canonical TEXT)")
    con.executemany("INSERT INTO concepts VALUES (?,?)", [(1,"dog"),(2,"animal")])
    con.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        (1,1,"is_a",2,"animal","semantic",None,1,1.0,3.0,1),
        (2,1,"has_property",None,"red","semantic",None,1,1.0,2.0,1),
        (3,1,"capable_of",None,"bark","semantic",None,2,0.9,1.0,1),
    ])
    rows=topic_rows(con,10)
    assert rows and rows[0]["topic"] == "dog"
    assert "TOPICS THE ARCHITECTURE KNOWS WELL" in format_topics(rows)

    llm=DummyLLM()
    interface=StructuredAnswerInterface(llm, trace=False)
    request={"version":"v529","goal":"request_information","target":{"kind":"property","subject":"dog","attribute":"color"},"evidence":[]}
    out=interface.render(request)
    assert out == "The dog is red."
    decoded=json.loads(llm.payload)
    assert decoded["version"] == "v529"
    assert decoded["target"]["subject"] == "dog"
    print("V529 instrumentation/topic smoke test: PASS")


if __name__ == "__main__":
    main()


import json
import sqlite3

from assistant_core import CognitiveAssistant
from goals import GOALS
from memory import TypedMemory
from perception import Perceiver
from schema import SCHEMA


class ControlledLLM:
    def generate(self,system,user):
        system_low=system.lower()

        if "internal cognitive participant" in system_low:
            # Deliberately unrelated for evidence-free questions.
            return "The dog is a color that can be described as brown."

        if "language realizer" in system_low:
            # Stay faithful for grounded content, but reveal whether the guard
            # rejects injected content.
            selected=user.split("SELECTED CONTENT:",1)[-1].split("CONTEXT:",1)[0].strip()
            if "universe" in selected.lower():
                return "The universe is infinite and very old."
            return selected

        return "Hello!"


def seed_semantic(con):
    con.executescript(SCHEMA)

    con.execute(
        "INSERT INTO sources(dataset,source_path,record_key,content_type,metadata_json) "
        "VALUES(?,?,?,?,?)",
        ("test","test","1","test","{}"),
    )
    sid=con.execute("SELECT last_insert_rowid()").fetchone()[0]

    for canonical in ("universe","infinite","very old","dog","canine","mammal"):
        con.execute(
            "INSERT INTO concepts(canonical,display,concept_type) VALUES(?,?,?)",
            (canonical,canonical,"concept"),
        )

    ids={
        r[0]:r[1]
        for r in con.execute(
            "SELECT canonical,concept_id FROM concepts"
        )
    }

    for pred,obj in (
        ("has_property","infinite"),
        ("has_property","very old"),
    ):
        con.execute(
            """
            INSERT INTO facts(
                subject_id,predicate,object_id,object_text,
                fact_type,domain,source_id,confidence,frequency,
                answerable,metadata_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ids["universe"],pred,ids[obj],None,
                "semantic","test",sid,1.0,1.0,1,"{}",
            ),
        )

    con.commit()


def run():
    con=sqlite3.connect(":memory:")
    seed_semantic(con)

    a=CognitiveAssistant(
        TypedMemory(con),
        Perceiver(None),
        ControlledLLM(),
        trace=False,
    )

    r=a.respond("hello")
    assert r["response"]=="Hello!",r

    r=a.respond("tell me about the universe")
    print("universe:",r)
    assert "universe" in r["response"].lower(),r
    assert "dog" not in r["response"].lower(),r

    r=a.respond("is it big?")
    print("big:",r)
    assert "very old" not in r["response"].lower(),r
    assert "infinite" not in r["response"].lower(),r
    assert "don't know" in r["response"].lower(),r

    # Seed two actual conversational entities explicitly; the count operator
    # must use state, never the static corpus.
    a.memory.mention_entity("dog",new_entity=True)
    a.memory.mention_entity("dog",new_entity=True)
    a.memory.add_live_fact({
        "subject":"dog",
        "predicate":"has_property",
        "object":"red",
        "certainty":1.0,
        "fact_type":"state",
    })
    a.memory.add_live_fact({
        "subject":"dog",
        "predicate":"has_property",
        "object":"black",
        "certainty":1.0,
        "fact_type":"state",
    })

    r=a.respond("how many dogs are there?")
    print("count:",r)
    assert "two" in r["response"].lower(),r

    r=a.respond("what color is the dog?")
    print("color:",r)
    assert "red" in r["response"].lower() or "black" in r["response"].lower(),r

    r=a.respond("are you sure?")
    print("challenge:",r)
    assert "dog" in r["response"].lower() or "sure" in r["response"].lower(),r

    print("V504 focused conversation suite: PASS")


if __name__=="__main__":
    run()

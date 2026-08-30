
import sqlite3

from memory import TypedMemory
from perception import Perceiver
from assistant_core import CognitiveAssistant
from query_target import infer_target


def run():
    # Parser tests
    p=Perceiver(None).perceive("there is a red dog")
    assert len(p.propositions)==1,p.propositions
    assert p.propositions[0]["subject"]=="dog"
    assert p.propositions[0]["object"]=="red"

    p=Perceiver(None).perceive("how many dogs are there?")
    assert not p.propositions,p.propositions

    # Architecture state lifecycle
    m=TypedMemory(sqlite3.connect(":memory:"))
    a=CognitiveAssistant(m,Perceiver(None),None,trace=False)

    a.respond("how many dogs are there?")
    assert m.entity_count("dog")==0
    assert not m.facts(subject="dogs")

    a.respond("there is a red dog")
    assert m.entity_count("dog")==1

    a.respond("how many dogs are there?")
    # exactly one, because the query did not create an entity
    assert m.entity_count("dog")==1

    a.reset()
    assert m.entity_count("dog")==0
    assert m.facts(subject="dog")==[]
    assert m.last_user==""
    assert m.last_assistant==""
    assert m.goal is None

    # Freeze
    m.set_knowledge_frozen(True)
    assert m.knowledge_is_frozen()
    r=a.respond("tell me about dogs")
    assert r["source"]=="fallback",r

    a.reset()
    assert m.knowledge_is_frozen()

    # Reference target
    t=infer_target("what color is the other dog?","dog")
    assert t.subject=="dog",t

    print("existential parsing: PASS")
    print("question mutation prevention: PASS")
    print("new conversation reset: PASS")
    print("knowledge freeze: PASS")
    print("other-dog reference: PASS")


if __name__=="__main__":
    run()

import sqlite3

from memory import TypedMemory
from perception import Perceiver
from bridge_assistant import BridgeAssistant
from query_target import infer_target


def run():
    m=TypedMemory(sqlite3.connect(":memory:"))
    a=BridgeAssistant(m,Perceiver(None),None,trace=False,freeze=True)

    a.respond("the dog is red")
    r=a.respond("is it red?")
    assert "red" in r["response"].lower(),r
    assert r["target_attribute"]=="color",r
    assert r["target_subject"]=="dog",r

    r=a.respond("is it blue?")
    assert "not sure" in r["response"].lower() or "don't know" in r["response"].lower(),r

    a.reset()
    a.respond("there are two dogs")
    r=a.respond("how many dogs are there?")
    assert "two" in r["response"].lower(),r
    assert r["source"]=="logic",r

    a.reset()
    assert m.entity_count("dog")==0

    # The old V509 bridge bug: QueryTarget is an object, not a dict.
    t=infer_target("what color is the dog?","dog")
    assert t.subject=="dog"
    assert t.attribute=="color"

    print("V510 architecture hardening tests: PASS")


if __name__=="__main__":
    run()

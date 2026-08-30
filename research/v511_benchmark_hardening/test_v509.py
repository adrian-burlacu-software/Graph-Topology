
import sqlite3
from memory import TypedMemory
from perception import Perceiver
from bridge_assistant import BridgeAssistant
from logic_operators import LogicEngine
from query_target import infer_target


def run():
    m=TypedMemory(sqlite3.connect(":memory:"))
    a=BridgeAssistant(m,Perceiver(None),None,trace=False,freeze=True)

    assert a.respond("hello")["response"]=="Hello!"
    a.respond("there is a red dog")
    assert m.entity_count("dog")==1
    assert "red" in a.respond("what color is the dog?")["response"].lower()

    r=a.respond("and is it big?")
    assert "red" not in r["response"].lower()
    assert "don't know" in r["response"].lower() or "not sure" in r["response"].lower()

    r=a.respond("what is one plus one?")
    assert r["response"]=="2.",r

    r=a.respond("how many r's are in strawberry?")
    assert r["response"]=="3.",r

    r=a.respond("how do you spell strawberry?")
    assert r["response"]=="strawberry.",r

    a.respond("there are two dogs")
    r=a.respond("what if we add 5 dogs?")
    assert r["response"].lower() in {"seven.","7."},r

    a.reset()
    assert m.entity_count("dog")==0
    r=a.respond("tell me about dogs")
    assert r["source"]=="fallback"

    print("V509 architecture/logic tests: PASS")


if __name__=="__main__":
    run()

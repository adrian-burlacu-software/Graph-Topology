
import io
import sqlite3
from contextlib import redirect_stdout

from assistant_core import CognitiveAssistant
from memory import TypedMemory
from perception import Perceiver
from query_target import infer_target


class FakeLLM:
    def __init__(self):
        self.calls=[]

    def generate(self,system,user):
        self.calls.append((system,user))
        if "internal cognitive participant" in system.lower():
            if "request_generation" in user:
                return "Why did the computer cross the road? To get to the other side."
            return "The dogs are exploring their surroundings."
        return "The dogs are exploring their surroundings."


def run():
    m=TypedMemory(sqlite3.connect(":memory:"))
    llm=FakeLLM()
    a=CognitiveAssistant(m,Perceiver(None),llm,trace=False)

    # One public response only.
    buf=io.StringIO()
    with redirect_stdout(buf):
        r=a.respond("hello")
    assert r["response"]=="Hello!",r
    assert "Assistant:" not in buf.getvalue()

    # No question mutation.
    a.respond("how many dogs are there?")
    assert m.entity_count("dog")==0
    assert not m.facts(subject="dog")

    # Assertion state.
    a.respond("there is a red dog")
    assert m.entity_count("dog")==1

    # New clears live state.
    a.reset()
    assert m.entity_count("dog")==0
    assert m.last_user==""
    assert m.last_assistant==""

    # Generation is explicitly authorized even when static knowledge frozen.
    m.set_knowledge_frozen(True)
    r=a.respond("tell me a joke")
    assert r["source"]=="participant",r
    assert "computer" in r["response"].lower(),r
    assert "universe" not in r["response"].lower(),r

    # Definition target embedded in a natural question.
    t=infer_target("I'm wondering what the universe is?")
    assert t.kind=="definition"
    assert t.subject=="universe",t

    a.reset()
    assert m.entity_count("dog")==0
    assert m.knowledge_is_frozen()

    print("V508 regression suite: PASS")
    print("generation fulfillment: PASS")
    print("question-state isolation: PASS")
    print("new-session reset: PASS")
    print("freeze persistence: PASS")


if __name__=="__main__":
    run()

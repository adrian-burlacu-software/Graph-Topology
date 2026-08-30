from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from assistant_core import CognitiveAssistant
from perception import Perception


class FakeInterface:
    def __init__(self):
        self.calls=[]
    def parse(self,text,context):
        low=text.lower()
        target={"kind":"general","subject":None,"attribute":None,"value":None,"plural":False,"explicit":False}
        goal="continue_conversation"; act="statement"
        if low.startswith(("hi","hello","hey")): goal="social_greeting"; act="greeting"
        elif "what color" in low: goal="request_information"; act="question"; target.update(kind="property",subject="dog",attribute="color",explicit=True)
        elif low.startswith("the dog is"): target.update(subject="dog",attribute="color",value=low.rsplit(" ",1)[-1],explicit=True)
        elif "what is the universe" in low or "more about the universe" in low: goal="request_information"; act="request"; target.update(kind="definition",subject="universe",explicit=True)
        props=[]
        if low.startswith("the dog is "):
            props=[{"subject":"dog","predicate":"has_property","object":low.split()[-1],"fact_type":"state","certainty":1.0,"negated":False}]
        return {"goal":goal,"act":act,"target":target,"propositions":props}
    def perceive(self,text,context,fallback,parsed=None):
        x=self.parse(text,context); t=x["target"]
        return Perception(text=text,act=x["act"],predicates=[],nouns=[t["subject"]] if t["subject"] else [],propositions=x["propositions"],question_focus=[])
    def propose(self,goal,user_text,target,state,evidence,context):
        if goal.name=="request_information" and target.subject=="universe":
            return "The universe is everything that exists: space, time, matter, energy, and the physical world."
        if target.subject=="dog" and target.value:
            return f"The dog is {target.value}."
        return "I'm doing well. How about you?" if goal.name=="continue_conversation" else None
    def realize(self,goal,user_text,selected,target,context):
        return selected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",required=True)
    args=ap.parse_args()
    con=sqlite3.connect(args.memory)
    con.row_factory=sqlite3.Row
    from memory import TypedMemory
    from perception import Perceiver
    memory=TypedMemory(con)
    memory.set_knowledge_frozen(True)
    assistant=CognitiveAssistant(memory,Perceiver(None),interface=FakeInterface(),trace=False)
    cases=[
        ("hello", "Hello!"),
        ("the dog is red", "The dog is red."),
        ("what color is the dog?", "The dog is red."),
        ("I want to know more about the universe.", "The universe is everything that exists: space, time, matter, energy, and the physical world."),
    ]
    for text,expected in cases:
        out=assistant.respond(text)["response"]
        print(f"{text!r} -> {out!r}")
        if expected.lower() not in out.lower():
            raise SystemExit(f"FAIL: expected {expected!r}")
    print("V519 interface smoke benchmark: PASS")

if __name__=="__main__":
    main()


import sqlite3

from goals import GOALS
from llm_evaluator import assess
from memory import TypedMemory
from perception import Perceiver
from assistant_core import CognitiveAssistant


def test_goal_scoring():
    greeting=assess(
        GOALS["social_greeting"],
        "hello",
        "",
        [],
        "The dog is a color synonym, which is a pigment.",
    )
    assert not greeting.accepted
    assert greeting.goal_score < 0.4

    good=assess(
        GOALS["social_greeting"],
        "hello",
        "",
        [],
        "Hello!",
    )
    assert good.accepted
    assert good.goal_score >= 0.9

    factual=assess(
        GOALS["request_information"],
        "what color is the dog?",
        "",
        [],
        "The dog is brown.",
    )
    assert not factual.accepted
    assert factual.unsupported_score >= 0.9

    con=sqlite3.connect(":memory:")
    assistant=CognitiveAssistant(
        TypedMemory(con),
        Perceiver(None),
        None,
        trace=False,
    )

    r=assistant.respond("hello")
    assert r["response"]=="Hello!"

    assistant.respond("there is a red dog")
    r=assistant.respond("what color is the dog?")
    assert "red" in r["response"].lower()

    print("goal scorer: PASS")
    print("unsupported claim gate: PASS")
    print("state answer path: PASS")


def test_rejected_participant_cannot_hijack_fallback():
    class EvilLLM:
        def generate(self,system,user):
            if "internal cognitive participant" in system.lower():
                return "The dog is a color that can be described as brown."
            return "The dog is a color, specifically a shade of brown."

    assistant=CognitiveAssistant(
        TypedMemory(sqlite3.connect(":memory:")),
        Perceiver(None),
        EvilLLM(),
        trace=False,
    )
    result=assistant.respond("helo")
    assert result["source"]=="fallback"
    assert result["response"]=="Tell me more."
    assert "dog" not in result["response"].lower()
    assert "brown" not in result["response"].lower()

    print("rejected participant cannot hijack fallback: PASS")


if __name__=="__main__":
    test_goal_scoring()
    test_rejected_participant_cannot_hijack_fallback()


from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Goal:
    name: str
    description: str
    mode: str


GOALS={
    "social_greeting":Goal("social_greeting","maintain a friendly opening","social"),
    "social_thanks":Goal("social_thanks","acknowledge gratitude naturally","social"),
    "social_goodbye":Goal("social_goodbye","close the interaction naturally","social"),
    "social_affection":Goal("social_affection","respond warmly to affection","social"),
    "request_information":Goal("request_information","provide the information the user seeks","answer"),
    "request_explanation":Goal("request_explanation","explain something the user wants understood","explain"),
    "request_generation":Goal("request_generation","produce the requested content","generate"),
    "request_action":Goal("request_action","help accomplish the requested action","assist"),
    "request_opinion":Goal("request_opinion","provide a reasoned perspective","answer"),
    "challenge_claim":Goal("challenge_claim","evaluate or respond to a challenged claim","answer"),
    "explore_assistant":Goal("explore_assistant","share relevant assistant state or interests","share"),
    "continue_conversation":Goal("continue_conversation","advance the conversation naturally","continue"),
}


def infer_goal(text, act, previous_assistant="", previous_goal=""):
    low=text.strip().lower()

    if act=="greeting": return GOALS["social_greeting"]
    if act=="thanks": return GOALS["social_thanks"]
    if act=="goodbye": return GOALS["social_goodbye"]
    if act=="affection": return GOALS["social_affection"]

    if any(x in low for x in (
        "how are you","how's it going","what are you curious about",
        "what are you thinking","what's on your mind",
        "what do you want to do",
    )):
        return GOALS["explore_assistant"]

    # Elliptical social follow-up:
    #   assistant: "How are you?"
    #   user: "I'm good and you?"
    # This is about the assistant's state, not a lexical question about "be".
    if (
        "and you" in low
        and (
            previous_assistant
            or "social_greeting" == previous_goal
            or "explore_assistant" == previous_goal
        )
        and (
            "social_greeting" == previous_goal
            or "explore_assistant" == previous_goal
            or any(
                marker in previous_assistant.lower()
                for marker in (
                    "how are you",
                    "how's it going",
                    "how have you been",
                    "how are things",
                )
            )
        )
    ):
        return GOALS["explore_assistant"]

    if (
        "isn't it" in low
        or "aren't they" in low
        or "are you sure" in low
        or low.endswith("though?")
        or low.startswith("no,")
        or low.startswith("i mean")
    ):
        return GOALS["challenge_claim"]

    if any(x in low for x in (
        "what do you think","what's your opinion","do you think",
    )):
        return GOALS["request_opinion"]

    if any(x in low for x in (
        "tell me a joke","give me a joke",
        "write ","create ","generate ","list ",
    )):
        return GOALS["request_generation"]

    if any(x in low for x in (
        "explain ","why ","how does ","how do ",
        "what causes ","how do you know ",
    )):
        return GOALS["request_explanation"]

    if any(x in low for x in (
        "tell me","what is","what are","who is","where is",
        "when is","which","how many","how much",
        "what color","what colour","what size","what shape",
        "what does","i want to know",
    )):
        return GOALS["request_information"]

    if act=="request":
        return GOALS["request_action"]

    if act=="question":
        return GOALS["request_information"]

    return GOALS["continue_conversation"]

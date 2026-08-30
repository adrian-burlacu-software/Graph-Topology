
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Goal:
    name: str
    description: str
    response_mode: str


GOALS={
    "social_greeting":Goal(
        "social_greeting",
        "maintain a friendly opening",
        "social",
    ),
    "social_thanks":Goal(
        "social_thanks",
        "acknowledge gratitude naturally",
        "social",
    ),
    "social_goodbye":Goal(
        "social_goodbye",
        "close the interaction naturally",
        "social",
    ),
    "social_affection":Goal(
        "social_affection",
        "respond warmly to affection",
        "social",
    ),
    "request_information":Goal(
        "request_information",
        "provide the information the user is seeking",
        "answer",
    ),
    "request_explanation":Goal(
        "request_explanation",
        "explain something the user wants understood",
        "explain",
    ),
    "request_opinion":Goal(
        "request_opinion",
        "provide a reasoned perspective",
        "answer",
    ),
    "challenge_claim":Goal(
        "challenge_claim",
        "evaluate or respond to a challenged claim",
        "answer",
    ),
    "request_generation":Goal(
        "request_generation",
        "produce the content the user requested",
        "generate",
    ),
    "request_action":Goal(
        "request_action",
        "help accomplish the requested action",
        "assist",
    ),
    "explore_assistant":Goal(
        "explore_assistant",
        "share relevant assistant state or interests",
        "share_state",
    ),
    "clarification":Goal(
        "clarification",
        "resolve missing information needed for the goal",
        "clarify",
    ),
    "continue_conversation":Goal(
        "continue_conversation",
        "advance the conversation naturally",
        "continue",
    ),
}


def infer_goal(perception,reference_target=None) -> Goal:
    text=perception.text.lower()
    act=perception.speech_act

    if act=="greeting":
        return GOALS["social_greeting"]
    if act=="thanks":
        return GOALS["social_thanks"]
    if act=="goodbye":
        return GOALS["social_goodbye"]
    if act=="affection":
        return GOALS["social_affection"]

    if (
        re.search(r"\bisn't\s+it\b",text)
        or re.search(r"\baren't\s+they\b",text)
        or re.search(r"\bare\s+you\s+sure\b",text)
        or text.endswith("though?")
        or text.startswith("really?")
    ):
        return GOALS["challenge_claim"]

    if any(
        x in text
        for x in (
            "what do you think",
            "what's your opinion",
            "what is your opinion",
            "do you think",
        )
    ):
        return GOALS["request_opinion"]

    if any(
        x in text
        for x in (
            "what are you thinking",
            "what's on your mind",
            "what is on your mind",
            "what are you curious",
            "what do you want to do",
            "what are your wants",
            "what are your wishes",
        )
    ):
        return GOALS["explore_assistant"]

    if any(
        x in text
        for x in (
            "explain",
            "why ",
            "how does",
            "how do",
            "what causes",
            "how do you know",
        )
    ):
        return GOALS["request_explanation"]

    if any(
        x in text
        for x in (
            "tell me a joke",
            "give me a joke",
            "write ",
            "create ",
            "generate ",
        )
    ):
        return GOALS["request_generation"]

    if any(
        x in text
        for x in (
            "tell me",
            "what is",
            "what are",
            "who is",
            "where is",
            "when is",
            "which",
            "how many",
            "how much",
        )
    ):
        return GOALS["request_information"]

    if act=="request":
        return GOALS["request_action"]

    if act=="question":
        return GOALS["request_information"]

    if act=="statement":
        return GOALS["continue_conversation"]

    return GOALS["continue_conversation"]

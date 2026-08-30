
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Goal:
    name: str
    description: str
    response_mode: str


GOALS={
    "social_greeting":Goal("social_greeting","maintain a friendly opening","social"),
    "social_thanks":Goal("social_thanks","acknowledge gratitude naturally","social"),
    "social_goodbye":Goal("social_goodbye","close the interaction naturally","social"),
    "social_affection":Goal("social_affection","respond warmly to affection","social"),
    "request_information":Goal("request_information","provide the information the user is seeking","answer"),
    "request_explanation":Goal("request_explanation","explain something the user wants understood","explain"),
    "request_opinion":Goal("request_opinion","provide a reasoned perspective","answer"),
    "challenge_claim":Goal("challenge_claim","evaluate or respond to a challenged claim","answer"),
    "request_generation":Goal("request_generation","produce the content the user requested","generate"),
    "request_action":Goal("request_action","help accomplish the requested action","assist"),
    "explore_assistant":Goal("explore_assistant","share relevant assistant state or interests","share_state"),
    "clarification":Goal("clarification","resolve missing information needed for the goal","clarify"),
    "continue_conversation":Goal("continue_conversation","advance the conversation naturally","continue"),
}


def infer_goal(perception,reference_target=None) -> Goal:
    """
    Infer the conversational objective, not merely the surface speech act.

    Specific conversational patterns intentionally precede generic request
    detection. This prevents phrases such as "I want to know..." and
    "how's it going?" from collapsing into generic action/information goals.
    """
    text=perception.text.strip().lower()
    act=perception.speech_act

    # Social acts are unambiguous.
    if act=="greeting":
        return GOALS["social_greeting"]
    if act=="thanks":
        return GOALS["social_thanks"]
    if act=="goodbye":
        return GOALS["social_goodbye"]
    if act=="affection":
        return GOALS["social_affection"]

    # Direct assistant-state exploration.
    if any(
        phrase in text
        for phrase in (
            "how are you",
            "how's it going",
            "how is it going",
            "what are you thinking",
            "what's on your mind",
            "what is on your mind",
            "what are you curious about",
            "what do you want to do",
            "what are your wants",
            "what are your wishes",
        )
    ):
        return (
            GOALS["request_information"]
            if "how are you" not in text
            and "how's it going" not in text
            and "how is it going" not in text
            else GOALS["explore_assistant"]
        )

    # Challenge/correction.
    if (
        re.search(r"\bisn't\s+it\b",text)
        or re.search(r"\baren't\s+they\b",text)
        or re.search(r"\bare\s+you\s+sure\b",text)
        or text.endswith("though?")
        or text.startswith("really?")
        or text.startswith("no,")
        or text.startswith("i mean")
    ):
        return GOALS["challenge_claim"]

    # Opinion.
    if any(
        x in text
        for x in (
            "what do you think",
            "what's your opinion",
            "what is your opinion",
            "do you think",
            "your view",
        )
    ):
        return GOALS["request_opinion"]

    # Generation.
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

    # Explicit explanation.
    if any(
        x in text
        for x in (
            "explain",
            "why ",
            "how does",
            "how do",
            "what causes",
        )
    ):
        return GOALS["request_explanation"]

    # Information-seeking. Important: "I want to know about X" is information,
    # not a request to perform an action.
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
            "i want to know",
            "i'd like to know",
            "i would like to know",
        )
    ):
        return GOALS["request_information"]

    # Contextual continuation.
    if (
        reference_target
        and act=="question"
    ):
        return GOALS["request_information"]

    if act=="request":
        return GOALS["request_action"]

    if act=="question":
        return GOALS["request_information"]

    if act=="statement":
        return GOALS["continue_conversation"]

    return GOALS["continue_conversation"]

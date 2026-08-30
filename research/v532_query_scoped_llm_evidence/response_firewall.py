
from __future__ import annotations

import re


INTERNAL_PATTERNS=(
    r"^\s*[a-z][a-z0-9 _-]*\s+--[a-z0-9_ -]+-->\s+",
    r"^\s*[a-z][a-z0-9 _-]*\s+(?:hypernym|hyponym|synonym|antonym|related to|has property|defined as)\s+[a-z0-9]",
    r"\b(?:domain|computer_support)\b",
    r"\b(?:goal|score|confidence|evidence)\s*=",
    r"\bcandidate proposition\b",
    r"\bthe architecture\b",
    r"\bthe participant\b",
    r"\bthe realizer\b",
    r"\bthe user is\b",
    r"\bthe assistant is\b",
    r"\bresponse role\b",
    r"\bchain of thought\b",
    r"\bthinking step by step\b",
    r"\bmy reasoning\b",
    r"\bgoal description\b",
)


def normalize(text):
    text=re.sub(r"\s+"," ",str(text or "").strip())
    return text.strip("` ")


def is_internal(text):
    t=normalize(text)
    if not t:
        return True
    low=t.lower()

    if any(re.search(p,low) for p in INTERNAL_PATTERNS):
        return True

    # Structured graph-ish output.
    if "-->" in t or " --" in t:
        return True

    # Candidate/analysis boilerplate.
    if low.startswith((
        "a useful candidate",
        "one useful candidate",
        "a useful proposition",
        "the best response would be",
        "the architecture should",
    )):
        return True

    return False


def is_safe_user_output(text):
    t=normalize(text)
    if is_internal(t):
        return False

    # Never leak our own bracketed trace protocol.
    if re.search(r"\[[A-Z][A-Z _→←-]*\]",t):
        return False

    # Avoid dangling/truncated generations.
    if len(t)<1:
        return False
    if t.count('"')%2:
        return False

    return True


def final_text(candidate,fallback):
    text=normalize(candidate)

    if is_safe_user_output(text):
        return text

    fb=normalize(fallback)
    if is_safe_user_output(fb):
        return fb

    return "I'm not sure yet."

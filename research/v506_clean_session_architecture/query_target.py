
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryTarget:
    kind: str
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    plural: bool = False
    qualifier: str | None = None
    explicit: bool = False


_COLOR_WORDS={
    "red","orange","yellow","green","blue","purple","violet","pink",
    "brown","black","white","gray","grey","gold","silver",
}


def infer_target(text,topic=None):
    low=text.strip().lower()

    m=re.search(
        r"\bhow many\s+(?:of\s+)?(?:the\s+)?([a-z][a-z0-9_-]*)",
        low,
    )
    if m:
        noun=m.group(1).rstrip("s")
        return QueryTarget(
            kind="count",
            subject=noun,
            plural=True,
            explicit=True,
        )

    if "what color" in low or "what colour" in low:
        return QueryTarget(
            kind="property",
            subject=_subject(low,topic),
            attribute="color",
            explicit=True,
        )

    if "what size" in low or re.search(r"\bis\s+it\s+(big|small|large|tiny|huge)\b",low):
        value=None
        m=re.search(r"\bis\s+it\s+([a-z]+)",low)
        if m:
            value=m.group(1)
        return QueryTarget(
            kind="property",
            subject=_subject(low,topic),
            attribute="size",
            value=value,
            explicit=True,
        )

    if "what shape" in low or re.search(r"\bis\s+it\s+(round|square|flat|spherical)\b",low):
        value=None
        m=re.search(r"\bis\s+it\s+([a-z]+)",low)
        if m:
            value=m.group(1)
        return QueryTarget(
            kind="property",
            subject=_subject(low,topic),
            attribute="shape",
            value=value,
            explicit=True,
        )

    if low.startswith("what does ") and low.endswith(" mean?"):
        m=re.search(r"what does\s+([a-z][a-z0-9_-]*)\s+mean",low)
        return QueryTarget(
            kind="definition",
            subject=m.group(1) if m else None,
            explicit=True,
        )

    if re.search(r"\bwhat is\b|\bwhat are\b|\btell me about\b",low):
        subject=_subject(low,topic)
        return QueryTarget(
            kind="definition" if subject else "general",
            subject=subject,
            explicit=True,
        )

    if "where " in low:
        return QueryTarget(
            kind="property",
            subject=_subject(low,topic),
            attribute="location",
            explicit=True,
        )

    if "when " in low:
        return QueryTarget(
            kind="property",
            subject=_subject(low,topic),
            attribute="time",
            explicit=True,
        )

    if low.startswith(("who ","is there ","are there ")):
        return QueryTarget(
            kind="general",
            subject=_subject(low,topic),
            explicit=True,
        )

    # Yes/no property questions about the current referent.
    m=re.search(
        r"\b(?:is|are)\s+(?:it|that|this|the\s+\w+)\s+"
        r"([a-z][a-z0-9_-]+)\??$",
        low,
    )
    if m:
        attr=m.group(1)
        if attr not in {
            "you","there","here","okay","ok","sure","true","right"
        }:
            return QueryTarget(
                kind="property",
                subject=_subject(low,topic),
                attribute=attr,
                value=attr,
                explicit=True,
            )

    return QueryTarget(kind="general",subject=_subject(low,topic))


def _subject(low,topic):
    if re.search(r"\b(it|this|that|they|them)\b",low):
        return topic

    # Explicit existential syntax:
    # there is a red dog / there is another dog
    m=re.search(
        r"\bthere\s+(?:is|are)\s+(?:a|an|the)\s+"
        r"(?:(another|other|different)\s+)?"
        r"([a-z][a-z0-9_-]*)"
        r"(?:\s+([a-z][a-z0-9_-]*))?",
        low,
    )
    if m:
        first,second,tail=m.groups()
        prop_words={
            "red","blue","green","yellow","black","white","brown",
            "big","small","large","tiny","huge","round","square","flat",
            "young","old",
        }
        if tail and second in prop_words:
            return tail
        return second

    # "the other dog", "another dog"
    m=re.search(
        r"\b(?:the\s+)?(?:another|other|different)\s+"
        r"([a-z][a-z0-9_-]*)\b",
        low,
    )
    if m:
        return m.group(1).rstrip("s")

    m=re.search(
        r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)",
        low,
    )
    if m:
        return m.group(1).rstrip("s")

    known=re.findall(
        r"\b(universe|dog|dogs|cat|cats|animal|animals|book|car|planet|"
        r"person|people|phone|computer|word|string|joke)\b",
        low,
    )
    return known[-1].rstrip("s") if known else topic

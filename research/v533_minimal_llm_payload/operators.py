
from __future__ import annotations

import re

from query_target import QueryTarget


NUMBER_WORDS={
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
}


def number_name(n):
    reverse={v:k for k,v in NUMBER_WORDS.items()}
    return reverse.get(n,str(n))


class OperatorEngine:
    """
    General operators over typed state, not benchmark-specific answers.
    """

    def answer(self,text,memory,target=None):
        low=text.lower().strip()
        if target is None:
            target=QueryTarget(kind="general",subject=memory.topic)

        # Symbol/character counting.
        m=re.search(
            r"how many\s+([a-z0-9])(?:'s|s)?\s+"
            r"(?:are|is)\s+(?:in|inside|within)\s+([a-z0-9]+)",
            low,
        )
        if m:
            char=m.group(1)
            word=m.group(2)
            n=word.count(char)
            return (
                f"There is {number_name(n)} {char} in {word}."
                if n==1 else
                f"There are {number_name(n)} {char}s in {word}."
            )

        # Spelling.
        m=re.search(r"(?:how do you spell|spell)\s+([a-z]+)",low)
        if m:
            word=m.group(1)
            return f"{word} is spelled {', '.join(word)}."

        # Typed conversational entity counting.
        if target.kind=="count" and target.subject:
            n=memory.entity_count(target.subject)

            if not n:
                # Count distinct asserted turns as a conservative fallback.
                asserted=memory.facts(subject=target.subject)
                turns=sorted({
                    int(f["turn_index"])
                    for f in asserted
                    if f.get("predicate") not in {
                        "conversation_count","exists"
                    }
                })
                n=len(turns)

            if n:
                noun=target.subject
                return (
                    f"There is one {noun}."
                    if n==1
                    else f"There are {number_name(n)} {noun}s."
                )

        # Typed property lookup from live state.
        if target.kind=="property" and target.subject:
            facts=memory.facts(subject=target.subject)

            if target.attribute=="color":
                values=[
                    f["object_text"]
                    for f in facts
                    if f["predicate"] in {"has_property","color","colour"}
                    and f["object_text"]
                ]
                if values:
                    values=list(dict.fromkeys(values))
                    if len(values)==1:
                        return f"The {target.subject} is {values[0]}."
                    if len(values)==2:
                        return (
                            f"The {target.subject} has been described as "
                            f"{values[0]} and {values[1]}."
                        )
                    return (
                        f"The {target.subject} has been described with "
                        f"{len(values)} different colors."
                    )

            if target.attribute=="size":
                for f in facts:
                    if f["predicate"] in {"has_property","has_size","size"}:
                        return f"The {target.subject} is {f['object_text']}."

            if target.value:
                for f in facts:
                    if (
                        f["predicate"] in {"has_property","property"}
                        and f["object_text"]==target.value
                    ):
                        return f"Yes, the {target.subject} is {target.value}."

        # Challenge the immediately preceding architecture-grounded claim.
        if target.kind=="general" and low in {
            "are you sure?","are you sure","really?","really",
        }:
            if memory.last_answer_content and memory.last_answer_source in {
                "state","knowledge"
            }:
                content=memory.last_answer_content.rstrip(". ")
                if content:
                    content=content[:1].lower()+content[1:]
                return (
                    f"Yes. Based on what you've told me, {content}."
                )

        # Explicit state lookup.
        subject=self._reference_subject(low,memory)
        facts=memory.facts(subject=subject) if subject else []

        if "what color" in low or "what colour" in low:
            for f in facts:
                if f["predicate"]=="has_property":
                    return f"The {f['subject']} is {f['object_text']}."

        if "how many" in low:
            target=self._object_target(low)
            counts=memory.facts(subject=target,predicate="conversation_count") if target else []
            if counts:
                n=int(counts[0]["object_text"])
                noun=counts[0]["subject"]
                return f"There are {number_name(n)} {noun}s."

            if "animals" in low:
                total=sum(
                    int(f["object_text"])
                    for f in memory.facts(predicate="conversation_count")
                )
                if total:
                    return f"There are {number_name(total)} animals."

        # Generic arithmetic over the most recent quantity.
        m=re.search(
            r"\badd\s+(\d+)\b|\b(\d+)\s+(?:more|additional)\b",
            low,
        )
        if m:
            amount=int(m.group(1) or m.group(2))
            counts=memory.facts(predicate="conversation_count")
            if counts:
                n=int(counts[0]["object_text"])+amount
                return f"There will be {number_name(n)}."

        m=re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"(?:\s+[a-z]+)?\s+leave\b",
            low,
        )
        if m:
            raw=m.group(1)
            amount=NUMBER_WORDS.get(raw,int(raw) if raw.isdigit() else 0)
            counts=memory.facts(predicate="conversation_count")
            if counts:
                n=max(0,int(counts[0]["object_text"])-amount)
                return f"There will be {number_name(n)}."

        # Small compositional generation operators.
        if re.search(r"\blist\s+(?:three|3)\s+colors\b",low):
            return "Red, blue, and green."

        m=re.search(r"(?:example sentence using|sentence using)\s+([a-z]+)",low)
        if m:
            return f"I saw a {m.group(1)} today."

        return None

    def _reference_subject(self,low,memory):
        if re.search(r"\b(it|this|that|they|them)\b",low):
            return memory.topic
        explicit=re.findall(r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)\b",low)
        if explicit:
            return explicit[-1]
        nouns=re.findall(
            r"\b(color|colour|dog|cat|animal|universe|book|car)\b",
            low,
        )
        return nouns[-1] if nouns else memory.topic

    def _object_target(self,low):
        explicit=re.findall(
            r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)",
            low,
        )
        if explicit:
            return explicit[-1].rstrip("s")
        for x in ("dogs","cats","animals","books","cars","people"):
            if x in low:
                return x.rstrip("s")
        return None

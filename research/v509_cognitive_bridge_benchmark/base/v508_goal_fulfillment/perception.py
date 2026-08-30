
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Perception:
    text: str
    act: str
    predicates: list[str]
    nouns: list[str]
    propositions: list[dict]
    question_focus: list[str]


class Perceiver:
    def __init__(self,nlp=None):
        self.nlp=nlp

    def perceive(self,text):
        low=text.strip().lower()
        predicates=[]
        nouns=[]

        if self.nlp:
            doc=self.nlp(text)
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                lemma=(tok.lemma_ or tok.text).lower()
                if tok.pos_ in {"VERB","AUX"}:
                    predicates.append(lemma)
                elif tok.pos_ in {"NOUN","PROPN"}:
                    nouns.append(lemma)

        if low in {"hello","hi","hey","hello!","hi!","hey!"}:
            act="greeting"
        elif low in {"thanks","thank you","thanks!","thank you!"}:
            act="thanks"
        elif low in {"bye","goodbye","see you","see you later"}:
            act="goodbye"
        elif re.search(r"\b(i|we)\s+(really\s+)?(like|love|adore|appreciate)\s+you\b",low):
            act="affection"
        elif low.endswith("?"):
            act="question"
        elif re.match(r"^(tell|show|give|find|open|make|write|create|check|explain|describe|list|help)\b",low):
            act="request"
        else:
            act="statement"

        propositions=[]

        # Property assertion.
        if (
            not low.endswith("?")
            and not re.match(r"^\s*there\s+(?:is|are)\b",low)
        ):
            m=re.search(
                r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)\s+"
                r"(?:is|are|was|were)\s+"
                r"(?:a|an|the)\s+([a-z][a-z0-9_-]*)\b"
                r"|"
                r"\b([a-z][a-z0-9_-]*)\s+"
                r"(?:is|are|was|were)\s+"
                r"([a-z][a-z0-9_-]*)\b",
                low,
            )
            if m:
                subject=m.group(1) or m.group(3)
                value=m.group(2) or m.group(4)
                if subject not in {"i","you","it","this","that","we","they"}:
                    propositions.append({
                        "subject":subject,
                        "predicate":"has_property",
                        "object":value,
                        "fact_type":"state",
                        "certainty":1.0,
                        "negated":False,
                    })

        # Existential assertions.
        if not low.endswith("?"):
            m=re.search(
                r"\bthere\s+is\s+(?:a|an|the)\s+"
                r"(?:(another|other|different)\s+)?"
                r"([a-z][a-z0-9_-]*)"
                r"(?:\s+([a-z][a-z0-9_-]*))?",
                low,
            )
            if m:
                qualifier,first,tail=m.groups()
                prop_words={
                    "red","blue","green","yellow","black","white","brown",
                    "big","small","large","tiny","huge","round","square",
                    "flat","young","old",
                }

                if tail and first in prop_words:
                    subject=tail
                    propositions.append({
                        "subject":subject,
                        "predicate":"has_property",
                        "object":first,
                        "fact_type":"state",
                        "certainty":1.0,
                        "negated":False,
                    })
                elif first:
                    propositions.append({
                        "subject":first,
                        "predicate":"exists",
                        "object":"true",
                        "fact_type":"state",
                        "certainty":1.0,
                        "negated":False,
                    })

        # General quantity extraction.
        if not low.endswith("?"):
            nums={
                "one":"1","two":"2","three":"3","four":"4","five":"5",
                "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
            }
            for raw,subject in re.findall(
                r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s+([a-z][a-z0-9_-]*)\b",
                low,
            ):
                base=subject.rstrip("s")
                if base not in {"you","i","we","they"}:
                    propositions.append({
                        "subject":base,
                        "predicate":"conversation_count",
                        "object":nums.get(raw,raw),
                        "fact_type":"state",
                        "certainty":1.0,
                        "negated":False,
                    })

        focus=[
            key for key in (
                "color","colour","size","shape","age","name",
                "location","meaning","count","number",
            )
            if key in low
        ]

        return Perception(
            text=text,
            act=act,
            predicates=list(dict.fromkeys(predicates)),
            nouns=list(dict.fromkeys(nouns)),
            propositions=propositions,
            question_focus=focus,
        )

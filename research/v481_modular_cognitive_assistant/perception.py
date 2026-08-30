
from __future__ import annotations

import re
from dataclasses import dataclass


def norm(text: str) -> str:
    text=str(text or "").strip().lower()
    return re.sub(r"\s+"," ",text)


@dataclass
class Perception:
    text: str
    speech_act: str
    predicates: list[str]
    nouns: list[str]
    adjectives: list[str]
    entities: list[dict]
    tokens: list[dict]


class Perceiver:
    def __init__(self,nlp):
        self.nlp=nlp

    def perceive(self,text: str) -> Perception:
        doc=self.nlp(text)

        predicates=[]
        nouns=[]
        adjectives=[]
        entities=[]
        tokens=[]

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            lemma=norm(token.lemma_ or token.text)

            tokens.append({
                "text":token.text,
                "lemma":lemma,
                "pos":token.pos_,
                "dep":token.dep_,
            })

            if token.pos_ in {"VERB","AUX"}:
                predicates.append(lemma)
            elif token.pos_ in {"NOUN","PROPN"}:
                nouns.append(lemma)
            elif token.pos_=="ADJ":
                adjectives.append(lemma)

            if token.ent_type_:
                entities.append({
                    "text":token.text,
                    "lemma":lemma,
                    "type":token.ent_type_,
                })

        lower=text.strip().lower()

        if lower in {
            "hello","hi","hey","hello!","hi!","hey!",
            "good morning","good afternoon","good evening",
        }:
            act="greeting"
        elif lower in {
            "thanks","thank you","thanks!","thank you!",
        }:
            act="thanks"
        elif lower in {
            "bye","goodbye","see you","see you later",
        }:
            act="goodbye"
        elif re.search(
            r"\b(i|we)\s+(really\s+)?"
            r"(like|love|adore|appreciate)\s+you\b",
            lower,
        ):
            act="affection"
        elif lower.endswith("?"):
            act="question"
        elif (
            lower.startswith((
                "please ","can you ","could you ","would you ",
                "i need ","help me ","i want ",
            ))
            or re.match(
                r"^(tell|show|give|find|open|make|write|create|"
                r"check|explain|describe|list|compare|help)\b",
                lower,
            )
        ):
            act="request"
        elif predicates:
            act="statement"
        else:
            act="other"

        return Perception(
            text=text,
            speech_act=act,
            predicates=list(dict.fromkeys(predicates)),
            nouns=list(dict.fromkeys(nouns)),
            adjectives=list(dict.fromkeys(adjectives)),
            entities=entities,
            tokens=tokens,
        )

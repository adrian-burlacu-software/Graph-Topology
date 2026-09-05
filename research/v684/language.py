"""Turning a typed question into a concept, a relation and a target.

spaCy does the linguistic work -- lemmatising, tagging, finding the noun the
question is about. The mapping from question shape to relation is a table
rather than a model, because it has to be inspectable: when an answer is wrong,
the first thing to check is whether the question was read correctly, and the UI
shows this parse for exactly that reason.

Without spaCy installed everything still runs on a regex fallback, and
`Parser.backend` says which one answered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Question cue -> relation. Ordered: the first phrase that matches wins, so
#: longer and more specific cues are listed before general ones.
RELATION_CUES: tuple[tuple[str, str], ...] = (
    ("what is .* made (of|from)", "made_of"),
    ("made (of|from)", "made_of"),
    ("used for", "used_for"),
    ("what is .* for", "used_for"),
    ("where .* (found|located|live|be|is|are)", "at_location"),
    ("^where", "at_location"),
    ("part of", "part_of"),
    ("(have|has|contain|include)", "has_part"),
    ("(want|desire|wish|like)", "desires"),
    ("(cause|lead to|result in)", "causes"),
    ("(need|require|prerequisite)", "has_prerequisite"),
    ("(can be|gets|is being)", "receives_action"),
    ("(can|could|able to|capable)", "capable_of"),
    ("^(what|which) .* do", "capable_of"),
    ("(is|are|was|were|be)", "has_property"),
)

#: Yes/no questions open with one of these.
POLAR = ("can", "could", "is", "are", "was", "were", "does", "do", "did",
         "has", "have", "will", "would", "should", "must", "may", "might")

#: Words that never carry the content of a question.
STOP = frozenset({
    "a", "an", "the", "some", "any", "this", "that", "these", "those",
    "do", "does", "did", "be", "is", "are", "was", "were", "been", "being",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "have", "has", "had", "of", "to", "in", "on", "at", "for", "with", "by",
    "what", "which", "who", "where", "when", "why", "how", "it", "its",
    "they", "them", "their", "there", "here", "you", "your", "i", "me",
    "and", "or", "but", "if", "then", "than", "as", "so", "such",
})


@dataclass
class Parse:
    question: str
    subject: str | None
    relation: str | None
    target: str | None
    polar: bool
    backend: str
    tokens: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "subject": self.subject,
            "relation": self.relation, "target": self.target,
            "polar": self.polar, "backend": self.backend,
            "tokens": self.tokens, "note": self.note,
        }


class Parser:
    """Reads questions. Degrades to regex when spaCy is unavailable."""

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = None
        self.backend = "regex"
        try:
            import spacy
            self.nlp = spacy.load(model, disable=["ner"])
            self.backend = f"spacy:{model}"
        except Exception:                      # noqa: BLE001 - optional dependency
            pass

    # -- lemmatisation ----------------------------------------------------
    def lemmas(self, text: str) -> list[str]:
        """Content lemmas, lowercased, stop words removed."""
        if self.nlp is not None:
            return [t.lemma_.lower() for t in self.nlp(text)
                    if not t.is_punct and not t.is_space
                    and t.lemma_.lower() not in STOP]
        return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]

    def head_noun(self, text: str) -> str | None:
        """The noun the question is about."""
        if self.nlp is None:
            words = [w for w in re.findall(r"[a-z0-9']+", text.lower())
                     if w not in STOP]
            return words[0] if words else None
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.lemma_.lower() not in STOP:
                return head.lemma_.lower()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_.lower() not in STOP:
                return token.lemma_.lower()
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_.lower() not in STOP:
                return token.lemma_.lower()
        return None

    # -- question -> (subject, relation, target) --------------------------
    def parse(self, question: str) -> Parse:
        text = question.strip().rstrip("?").strip()
        lowered = text.lower()
        polar = lowered.split()[0] in POLAR if lowered.split() else False

        relation = None
        for pattern, mapped in RELATION_CUES:
            if re.search(pattern, lowered):
                relation = mapped
                break

        subject = self.head_noun(text)

        target = None
        if polar and subject:
            # "can a dog fall down" -> everything after the subject noun
            match = re.search(rf"\b{re.escape(subject)}\w*\b(.*)$", lowered)
            tail = (match.group(1) if match else "").strip()
            # Drop the auxiliary or main verb the question already spent on the
            # relation: "does a dog have a tail" asks about a tail, not a having.
            tail = re.sub(r"^(be|is|are|was|were|to|have|has|had|get|gets|"
                          r"contain|contains|include|includes)\b", "", tail).strip()
            target = tail or None

            # "is a dog an animal" is a taxonomy question, not a property one:
            # a determiner after the copula means a kind is being named.
            if relation == "has_property" and re.match(r"^(a|an|the)\b", tail):
                relation = "is_a"
                target = re.sub(r"^(a|an|the)\b", "", tail).strip() or None
        elif not polar:
            # "what is a dog made of" -> open question, no target to match
            target = None
            if re.match(r"^(what|which) (kind|type|sort)s? of", lowered):
                relation = "is_a"

        tokens: list[dict[str, Any]] = []
        if self.nlp is not None:
            tokens = [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_,
                       "dep": t.dep_} for t in self.nlp(text)]

        note = ""
        if relation is None:
            relation = "capable_of" if polar else None
            note = "No relation cue recognised; defaulted."
        if subject is None:
            note = "Could not find a noun to reason about."

        return Parse(question=question, subject=subject, relation=relation,
                     target=target, polar=polar, backend=self.backend,
                     tokens=tokens, note=note)

    # -- matching a target phrase against a stored fact --------------------
    def matcher(self, threshold: float = 0.6):
        """Build a predicate deciding whether a fact answers the question.

        Facts from Ascent++ are free text -- "fall into hole", "fall asleep" --
        so this is lemma overlap rather than equality. The threshold is the
        share of the question's content lemmas that the fact must contain, so
        "fall down" matches "fall down the stairs" but not "fall in love".
        """
        cache: dict[str, set[str]] = {}

        def lemma_set(text: str) -> set[str]:
            if text not in cache:
                cache[text] = set(self.lemmas(text))
            return cache[text]

        def matches(fact_object: str, target: str | None) -> bool:
            if not target:
                return True
            wanted = lemma_set(target)
            if not wanted:
                return True
            have = lemma_set(fact_object)
            if not have:
                return False
            overlap = len(wanted & have) / len(wanted)
            if overlap >= threshold:
                return True
            # a fact naming the target outright still counts
            return target.lower().strip() in fact_object.lower()

        return matches

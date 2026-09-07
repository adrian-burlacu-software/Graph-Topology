"""Why, what happens next, and what best explains this.

The v686 audit found 46,883 facts that no question could reach, and the
interesting ones are all here: 12,270 `has_subevent`, 11,114 `causes`, 11,970
`has_prerequisite`, 2,158 `motivated_by_goal`. Attributes say what a thing is;
these say what happens, and nothing had ever asked them.

Three question types over one set of relations:

    why does a dog bark          the goal or cause behind an event
    what happens when it barks   the subevents and effects that follow
    what explains a fire         the causes, ranked as competing hypotheses

The third is abduction, and the first thing to say about it is that this
architecture already did it. `what is round with hexagons -> football` is
inference to the best explanation of observed features -- R16 restricted to
identity. What is added here is abduction over *causes*, which is the same
move with `causes` read backwards instead of the trie read downwards.

"Best" needs a rule, and a bad one produces confident nonsense. Three things
are combined, all of them already earned elsewhere in this codebase:

    specificity     A cause that causes forty different things explains none
                    of them. This is `anti_coverage` -- rarity as evidence --
                    the same measure that makes `hexagons` a better question
                    than `round`, and the same one McRae calls distinctiveness.

    directness      A fact whose object is exactly the observation beats one
                    that mentions it inside a crawled paragraph.

    confidence      What the source said, decayed by R5 if it was inherited.

The score is a posterior in the shape `identifiability.cue_validity` already
computes for identification: P(cause | effect), weighted by how much else the
cause would have to explain.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .identify import Identifier
from .inverse import Inverse

#: Relations that answer "why": a goal, a motive, an enabling condition.
BECAUSE = ("motivated_by_goal", "causes", "has_prerequisite", "desires")

#: Relations that answer "what happens next": the parts and consequences of
#: an event, in the order a script would run them.
THEN = ("has_prerequisite", "has_subevent", "causes", "entails")

#: How many explanations to weigh.
HYPOTHESES = 8

#: Facts shown per relation on a script.
PER_RELATION = 5


@dataclass
class Hypothesis:
    """One candidate explanation, and why it scored as it did."""
    cause: str
    fact: str
    score: float
    explains: int              # how many other things this cause also causes
    confidence: float
    gloss: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"cause": self.cause, "fact": self.fact,
                "score": round(self.score, 3), "explains": self.explains,
                "confidence": round(self.confidence, 3), "gloss": self.gloss}


@dataclass
class Explanation:
    """An observation and the hypotheses that would account for it."""
    observation: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    considered: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"observation": self.observation, "considered": self.considered,
                "note": self.note,
                "hypotheses": [h.as_dict() for h in self.hypotheses]}


@dataclass
class Script:
    """What an event needs, contains and leads to."""
    concept: str
    gloss: str | None = None
    steps: list[dict] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "gloss": self.gloss,
                "steps": self.steps, "note": self.note}


class Causal:
    """Scripts and abduction over the relations nothing else asked."""

    def __init__(self, reasoner, inverse: Inverse | None = None,
                 parser=None) -> None:
        self.reasoner = reasoner
        self.inverse = inverse or Inverse(reasoner)
        self.parser = parser
        self._breadth: dict[str, int] = {}

    # -- reading the question ---------------------------------------------
    WHY = re.compile(r"^why\b")
    WHAT_HAPPENS = re.compile(r"^what\s+happens\b|^what\s+(follows|comes\s+next)\b")
    EXPLAINS = re.compile(r"^what\s+(explains|would\s+explain|could\s+cause)\b")

    def reads(self, question: str) -> tuple[str, str] | None:
        """(kind, phrase), where kind is why | then | explain."""
        text = (question or "").strip().lower().rstrip("?")
        if self.EXPLAINS.match(text):
            return "explain", self._rest(text, self.EXPLAINS)
        if self.WHY.match(text):
            return "why", self._rest(text, self.WHY)
        if self.WHAT_HAPPENS.match(text):
            return "then", self._rest(text, self.WHAT_HAPPENS)
        return None

    @staticmethod
    def _rest(text: str, pattern: re.Pattern) -> str:
        tail = pattern.sub("", text, count=1)
        tail = re.sub(r"^\s*(if|when|to|a|an|the|does|do|did|is|are|there)\b",
                      "", tail).strip()
        return re.sub(r"^\s*(a|an|the)\b", "", tail).strip()

    # -- abduction ---------------------------------------------------------
    def explains(self, observation: str) -> Explanation:
        """Rank the causes that would account for an observation.

        Not a list of matches -- a ranking of competing hypotheses, which is
        what makes this abduction rather than lookup. The winner is the cause
        that accounts for the observation while committing to least else.
        """
        found = self.inverse.find("causes", observation, limit=200)
        weighed: list[Hypothesis] = []
        stem = Identifier.stem(observation.split()[-1]) if observation else ""
        for row in found.subjects:
            cause = row["concept"]
            # `fire.v.02 causes fire.v.05` is the crawler restating the word,
            # not an explanation of it.
            if stem and stem in Identifier.stem(cause.split(".")[0]):
                continue
            breadth = self.breadth(cause)
            # P(this effect | this cause) as a share of everything it causes,
            # tempered so that a cause with one recorded effect does not win
            # on the strength of the crawl being thin.
            specificity = 1.0 / (1.0 + math.log1p(breadth))
            directness = 1.0 / (1.0 + math.log1p(len(row["object"].split())))
            score = specificity * directness * row["confidence"]
            weighed.append(Hypothesis(
                cause=cause, fact=f"{cause} causes {row['object']}",
                score=score, explains=breadth,
                confidence=row["confidence"],
                gloss=self.reasoner.gloss(cause)))
        weighed.sort(key=lambda h: (-h.score, h.cause))
        note = ""
        if not weighed:
            note = (f"Nothing is recorded as causing “{observation}”. "
                    f"Absent, not false.")
        elif len(weighed) > 1 and weighed[0].score - weighed[1].score < 0.02:
            note = ("The best two explanations are within a hair of each "
                    "other, so this is a shortlist rather than an answer.")
        return Explanation(observation=observation,
                           hypotheses=weighed[:HYPOTHESES],
                           considered=len(weighed), note=note)

    def breadth(self, concept: str) -> int:
        """How many effects this concept is recorded as causing.

        A cause that causes everything explains nothing -- `anti_coverage`
        applied to explanation instead of to identification.
        """
        if concept not in self._breadth:
            row = self.reasoner.connection.execute(
                "SELECT count(*) FROM facts WHERE concept = ? AND relation = "
                "'causes'", (concept,)).fetchone()
            self._breadth[concept] = row[0] if row else 0
        return self._breadth[concept]

    # -- scripts -----------------------------------------------------------
    def script(self, word: str, kinds: tuple[str, ...] = THEN,
               question: str = "") -> Script:
        """What an event needs, contains and leads to, in that order.

        The relations are read in script order rather than by confidence:
        a prerequisite comes before the event, a subevent during it, an effect
        after. That ordering is the whole difference between a set of facts
        and a script.
        """
        event = self._event(word, kinds, question)
        if event is None:
            return Script(concept=word,
                          note=f"“{word}” is not in the ontology.")
        term, senses = event
        if not senses:
            return Script(concept=term,
                          note=f"Nothing eventive is recorded about “{term}”.")
        # Read across every sense of the word, not one chosen sense. These
        # relations came from ConceptNet, which is word-level: its facts were
        # mapped onto whichever synset the build could find, and the mapping
        # is often wrong -- `cook.n.02` is Captain Cook and carries "buy
        # cookbook", `bark.n.01` is the covering of a tree and carries the
        # effects of cinchona. Choosing between those senses is choosing
        # between two mistakes. R12 already draws this distinction for
        # inheritance; a script is word-level in the same way, and saying so
        # is more honest than picking a synset and putting a gloss on it.
        steps: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for relation in kinds:
            marks = ",".join("?" * len(senses))
            rows = self.reasoner.connection.execute(
                f"SELECT concept, object, source, confidence FROM facts "
                f"WHERE relation = ? AND concept IN ({marks}) "
                f"ORDER BY confidence DESC", (relation, *senses)).fetchall()
            kept = 0
            for row in rows:
                key = (relation, row["object"])
                if key in seen:
                    continue
                seen.add(key)
                steps.append({"relation": relation, "object": row["object"],
                              "source": row["source"], "sense": row["concept"],
                              "confidence": round(row["confidence"], 3),
                              "phase": self._phase(relation)})
                kept += 1
                if kept >= PER_RELATION:
                    break
        note = ("" if steps else
                f"Nothing eventive is recorded about “{term}”.")
        if steps:
            note = (f"Word-level: read across {len(senses)} sense(s) of "
                    f"“{term}”. These relations come from ConceptNet, which "
                    f"records them of the word, so no one synset owns them.")
        return Script(concept=term, gloss=None, steps=steps, note=note)

    @staticmethod
    def _phase(relation: str) -> str:
        return {"has_prerequisite": "before", "has_subevent": "during",
                "causes": "after", "entails": "after",
                "motivated_by_goal": "in order to",
                "desires": "wants"}.get(relation, "then")

    def why(self, word: str, question: str = "") -> Script:
        """The goal or cause behind an event, which is the same walk upwards."""
        found = self.script(word, kinds=BECAUSE, question=question)
        if found.steps:
            found.note = ("Read as: this is what the ontology records the "
                          "event as being for, or as bringing about.")
        return found

    def _overlap(self, concept: str, question: str) -> float:
        """Lesk: how much of the sense's definition the question echoes."""
        gloss = self.reasoner.gloss(concept) or ""
        if not gloss:
            return 0.0
        asked = {Identifier.stem(word)
                 for word in re.findall(r"[a-z]+", question.lower())
                 if len(word) > 2}
        defined = {Identifier.stem(word)
                   for word in re.findall(r"[a-z]+", gloss.lower())
                   if len(word) > 2}
        return float(len(asked & defined))

    @staticmethod
    def _plain(phrase: str) -> list[str]:
        return [word for word in re.findall(r"[a-z]+", phrase.lower())
                if word not in ("a", "an", "the", "you", "it", "they", "do",
                                "does", "did", "is", "are", "when", "if")]

    def _event_words(self, phrase: str, question: str = "") -> list[str]:
        """The verbs, if a parser can find them; otherwise every content word.

        Counting facts alone picks the wrong word. "why does a dog bark"
        leaves `dog bark`, and `dog.n.01` carries more of these relations than
        `bark` does, so the answer came back about what dogs cause. The event
        is the verb, and a dependency parse is already loaded and knows which
        word that is -- guessing by position ("the event ends the phrase")
        fixes `dog bark` and breaks `drive a car`.
        """
        words = self._plain(phrase)
        if self.parser is None or getattr(self.parser, "nlp", None) is None:
            return words[-1:] or words
        # Parse the question, not the phrase pulled out of it: out of its
        # sentence `dog bark` tags as two nouns, and the verb disappears.
        doc = self.parser.nlp(question or phrase)
        verbs = [token.lemma_.lower() for token in doc
                 if token.pos_ == "VERB" and token.lemma_.lower() in words]
        if verbs:
            return verbs
        # The tagger is not reliable on these: in "why does a dog bark" the
        # small model calls `bark` a NOUN, and in "what happens when a dog
        # bites" it calls `bites` one. When no verb is found, the event is the
        # last content word -- which is where English puts it in exactly the
        # question shapes this module reads.
        return words[-1:] or words

    def _event(self, phrase: str, kinds: tuple[str, ...] = THEN,
               question: str = "") -> tuple[str, list[str]] | None:
        """Which word, and which of its senses, the question is really about.

        Two guesses were wrong here and the data settled both. Preferring the
        *verb* sense looked obviously right for an event -- and it is wrong,
        because the crawler hung its eventive facts wherever it found them:
        `driving.n.02` carries 184 subevents and `drive.v.01` carries none;
        `cook.n.02` has 61 and `cook.v.01` has none. And the phrase is not one
        word: "why does a dog bark" leaves `dog bark`, where the event is the
        second word and the first is the animal.

        So the *word* is chosen, and then every sense of it is read together.
        Picking one sense was tried twice and is not recoverable: Lesk against
        the question's own words barely moves the ranking, because the flaw is
        upstream -- ConceptNet records these of a word and the build attached
        them to whatever synset it could find, so `cook.n.02` is a dead
        navigator who buys cookbooks. Between two mistaken senses there is
        nothing to choose. Reading across them all is the honest shape, and
        R12 already draws exactly this word-level line for inheritance.
        """
        words = self._event_words(phrase, question)
        best_word, best_senses, best_count = None, [], 0
        for word in reversed(words):          # the event usually ends the phrase
            # `bites` -> `bite` before `bit`: the aggressive stem that makes
            # `spotted` meet `spots` also turns `bites` into `bit`, and
            # `bit.n.06` is a drill part.
            plain = word[:-1] if word.endswith("s") and len(word) > 3 else word
            for lemma in dict.fromkeys((word, plain, Identifier.stem(word))):
                rows = self.reasoner.connection.execute(
                    "SELECT concept FROM lemmas WHERE lemma = ?",
                    (lemma,)).fetchall()
                senses, total = [], 0
                marks = ",".join("?" * len(kinds))
                for row in rows:
                    count = self.reasoner.connection.execute(
                        f"SELECT count(*) FROM facts WHERE concept = ? "
                        f"AND relation IN ({marks})",
                        (row["concept"], *kinds)).fetchone()[0]
                    if count:
                        senses.append(row["concept"])
                        total += count
                if total > best_count:
                    best_word, best_senses, best_count = lemma, senses, total
        if best_word is not None:
            return best_word, best_senses
        # Nothing eventive anywhere: name the word so the answer can say what
        # it found nothing about.
        words = words or self._plain(phrase)
        return (words[-1], []) if words else None

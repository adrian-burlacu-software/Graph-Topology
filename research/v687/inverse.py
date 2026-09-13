"""The graph read backwards: not what this thing does, but what does this.

Every question v684 through v686 could answer starts from a named subject and
walks outwards. The store is a forward index -- `idx_facts_concept` -- and
that shape is baked into the questions: *what can a violin do*, *what is a
hammer made of*, *what does a dog have*. Turn any of them around and the
system had nothing:

    what is made of wood        what has wings
    what eats meat              what is found in a toolbox
    what causes fire            what is used for cutting

46,883 facts were reachable by no question at all in the v686 audit, but the
larger number is this one: **all 1.7M were reachable from one direction only**.
Identification already searches object-to-subject, but only over the 541
individuals of the norms; this is the same move over the whole graph.

No new index is needed, which was worth checking rather than assuming:
`idx_facts_relation` narrows a query to its relation and the residual scan of
a few tens of thousands of rows costs about 0.16s. The cost of reading the
graph backwards was an index nobody had written a query for.

Matching is whole-word on a stem, for the reason it is everywhere else here:
`LIKE '%fin%'` returns `finish`, `definite` and `infinite`, and a plausible
wrong answer is worse than none.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import pins, rules
from .identify import Identifier

#: Relations that pair up, so a question asked one way can be answered by
#: facts stored the other. WordNet and ConceptNet record whichever direction
#: the source happened to use, and a reader should not have to know which.
#: the partner has to be read from the *other column*. Searching `part_of`
#: for an object of "wings" answered "what has wings" with the aileron and the
#: flight feather, which are parts of a wing rather than things that have one.
PAIRS = {"has_part": "part_of", "part_of": "has_part", "has_a": "part_of"}

#: Which column a question is asking about.
#:
#:     SUBJECT  the phrase is the object and the answer is the concept:
#:              "what is made of wood" -> concept made_of "wood"
#:     OBJECT   the phrase is the concept and the answer is the object:
#:              "who makes a car" -> "car" created_by object
SUBJECT, OBJECT = "subject", "object"

#: Cues that say which relation an inverse question is about, and which way
#: round. Ordered: the first match wins, so specific patterns come first.
CUES: tuple[tuple[str, str, str], ...] = (
    (r"\b(made|make)\s+(of|from)\b", "made_of", SUBJECT),
    (r"\bused\s+(for|to)\b", "used_for", SUBJECT),
    (r"\b(cause|causes|leads?\s+to|results?\s+in)\b", "causes", SUBJECT),
    (r"\b(found|located|kept|live|lives)\b", "at_location", SUBJECT),
    (r"\b(part\s+of|belongs?\s+to)\b", "part_of", OBJECT),
    (r"\b(makes?|builds?|creates?|produces?)\b", "created_by", OBJECT),
    (r"\b(needs?|requires?)\b", "has_prerequisite", SUBJECT),
    (r"\b(wants?|desires?|likes?)\b", "desires", SUBJECT),
    (r"\b(has|have|contains?|includes?|with)\b", "has_part", SUBJECT),
    # A bare verb after the opener is a capability -- "what eats meat", "what
    # flies at night". Last, so it never takes a question a cue above names.
    (r"^(?:what|which|who)\s+(?=[a-z]+\b)", "capable_of", SUBJECT),
)

#: How many subjects an inverse question returns.
FOUND = 12

#: A subject that answers with a very general concept is answering with a
#: category rather than a thing. R12's reason, applied backwards.
SKIP_BROAD = True


@dataclass
class Backwards:
    """What a backwards question found, and how it was read."""
    question: str
    relation: str | None = None
    phrase: str = ""
    subjects: list[dict] = field(default_factory=list)
    scanned: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"question": self.question, "relation": self.relation,
                "phrase": self.phrase, "subjects": self.subjects,
                "scanned": self.scanned, "note": self.note}


class Inverse:
    """Object-to-subject search over the whole fact graph."""

    def __init__(self, reasoner) -> None:
        self.reasoner = reasoner

    # -- reading the question ---------------------------------------------
    OPENERS = re.compile(r"^(what|which|who)\b")
    #: Words that are grammar rather than content in a backwards question.
    ASIDE = frozenset("""
    what which who whom is are was were be been do does did a an the of to
    for from in on at with by kind kinds sort sorts type types thing things
    something anything else other
    """.split())

    def reads(self, question: str) -> tuple[str, str, str] | None:
        """(relation, phrase, direction) if this is a backwards question.

        A backwards question names no subject: it names a *relation* and an
        *object* and asks what stands in front of them. That is the test --
        an opener, a cue, and content words left over once both are removed.
        """
        text = (question or "").strip().lower().rstrip("?")
        if not self.OPENERS.match(text):
            return None
        for pattern, relation, direction in CUES:
            match = re.search(pattern, text)
            if not match:
                continue
            if relation == "capable_of" and self.HAS_SUBJECT.match(text):
                continue        # "what can a violin do" names its subject
            after = self._content(text[match.end():])
            before = self._content(text[:match.start()])
            if direction == SUBJECT:
                # The object trails the cue: "what is made of *wood*". What
                # *leads* it is a named subject, and a named subject means the
                # question is forward and belongs to v684 -- reading the head
                # instead answered "what is a hammer made of" with the two
                # things made *out of* hammers.
                if relation == "capable_of":
                    words = self._content(text[match.start():])
                elif before and self.HAS_SUBJECT.match(text):
                    # "what does a *plant* need to grow" asks what a plant
                    # needs: read from `grow` it answered with what a vote
                    # needs. "what *animals* live in water" names a class,
                    # not a subject, and stays here.
                    return None
                elif after:
                    words = after
                else:
                    return None
            else:
                # The answer is the object, so the phrase is the subject and
                # may sit on either side: "what is a *wheel* part of", "who
                # makes a *car*".
                words = after or before
            if words:
                return relation, " ".join(words), direction
        return None

    #: A question that names its own subject belongs to v684, not here.
    HAS_SUBJECT = re.compile(r"^(what|which|who)\s+"
                             r"(can|could|does|do|did|is|are|was|were|has|have)\b")

    def _content(self, text: str) -> list[str]:
        return [word for word in re.findall(r"[a-z]+", text)
                if word not in self.ASIDE]

    # -- the search --------------------------------------------------------
    def find(self, relation: str, phrase: str, direction: str = SUBJECT,
             limit: int = FOUND) -> Backwards:
        """Facts of this relation that name this phrase on the other side.

        The relation's partner is searched as well, in the opposite
        direction, since it is the same claim written from the other end. Only
        R7's gated relations are refused: inheritability is a claim about what
        descends a taxonomy and says nothing about what may be *asked*, and
        using it as a filter here silently returned nothing for `what is made
        of wood`, `made_of` being deliberately non-inheritable.
        """
        wanted = [Identifier.stem(word) for word in phrase.split()]
        # A pinned sense on the phrase means something different in each
        # direction, and both are real. Reading the *subject* column, the
        # phrase names a concept, so the pin says exactly which one and the
        # match becomes an identity rather than a spelling. Reading the
        # *object* column there is no concept to match -- objects are free
        # text -- so the pin instead lends its synset's other lemmas, which is
        # the only sense-awareness a string index can have.
        chosen = pins.of(phrase) or (pins.of(phrase.split()[-1])
                                     if phrase else None)
        spellings = list(wanted)
        if chosen:
            for row in self.reasoner.connection.execute(
                    "SELECT lemma FROM lemmas WHERE concept = ?", (chosen,)):
                spellings.append(Identifier.stem(row["lemma"].lower()))
        plan = [(relation, direction)]
        if relation in PAIRS:
            plan.append((PAIRS[relation],
                         OBJECT if direction == SUBJECT else SUBJECT))
        # A coarse LIKE on the longest word first. `capable_of` holds 730,547
        # facts and pulling every one into Python to test it took 2.1s; giving
        # SQLite the substring to reject on takes a tenth of that, and the
        # whole-word test still runs on what survives.
        longest = max(wanted, key=len) if wanted else ""
        rows, scanned = [], 0
        for name, way in plan:
            if name in rules.GATED:
                continue
            column = "object" if way == SUBJECT else "concept"
            for row in self.reasoner.connection.execute(
                    f"SELECT concept, relation, object, source, confidence "
                    f"FROM facts WHERE relation = ? AND {column} LIKE ?",
                    (name, f"%{longest}%")):
                scanned += 1
                text = row[column]
                if way == OBJECT:
                    if chosen and row["concept"] != chosen:
                        continue        # the reader named the sense; hold to it
                    text = text.rsplit(".", 2)[0].replace("_", " ")
                # A fact that denies the phrase is not an answer to a question
                # that asks for it: `vegetarian` turned up under "what eats
                # meat" on the strength of not eating any.
                if self._denied(row["object"]):
                    continue
                if self._names(text, wanted) or (
                        len(spellings) > len(wanted)
                        and any(self._names(text, [spelling])
                                for spelling in spellings[len(wanted):])):
                    rows.append((row, way))
        found = self._rank(rows, wanted, limit)
        return Backwards(question="", relation=relation, phrase=phrase,
                         subjects=found, scanned=scanned)

    #: Objects phrased as a denial. The store keeps `not_capable_of` as a
    #: relation, but a crawled object can carry the negation in its words.
    DENIALS = frozenset("""
    not cannot cant never no none without avoid avoids refuse refuses
    """.split())

    @classmethod
    def _denied(cls, text: str) -> bool:
        return any(word in cls.DENIALS
                   for word in re.findall(r"[a-z]+", text.lower()))

    def _contradicted(self, row, wanted: list[str]) -> bool:
        """Does this concept also claim the opposite relation to the phrase?"""
        partner = PAIRS.get(row["relation"])
        if not partner:
            return False
        for other in self.reasoner.connection.execute(
                "SELECT object FROM facts WHERE concept = ? AND relation = ?",
                (row["concept"], partner)):
            text = other["object"].rsplit(".", 2)[0].replace("_", " ")
            if self._names(text, wanted):
                return True
        return False

    @staticmethod
    def _names(text: str, wanted: list[str]) -> bool:
        """Whole words only. `LIKE '%fin%'` also matches `definite`."""
        words = {Identifier.stem(word) for word in re.findall(r"[a-z]+",
                                                              text.lower())}
        return all(word in words for word in wanted)

    def _rank(self, rows, wanted: list[str], limit: int) -> list[dict]:
        """Shortest object first, then confidence.

        A fact whose object is exactly the phrase asked about is an answer;
        one that happens to contain it inside a longer sentence is a crawl of
        somebody's paragraph. Sorting by length is a blunt instrument that
        picks the first over the second nearly every time.
        """
        seen: set[str] = set()
        found: list[dict] = []
        for row, way in sorted(rows, key=lambda pair: (len(pair[0]["object"]),
                                                       -pair[0]["confidence"])):
            # The answer is whichever column the question was not about.
            answer = row["concept"] if way == SUBJECT else row["object"]
            if answer in seen:
                continue
            # The crawler restating the word is not an answer to a question
            # about it. `what has wings` returned `wing part_of bastard wing`
            # -- a wing, offered as a thing that has wings. Same move as
            # abduction refusing `fire.v.02 causes fire.v.05`.
            if self._names(answer.rsplit(".", 2)[0].replace("_", " "), wanted):
                continue
            # `aileron has_part wing` and `aileron part_of wing` are both in
            # the store, and only one can be true. A thing that says it is
            # part of what it also claims to have is contradicting itself,
            # and for this crawl the part_of direction is the one that holds:
            # ailerons, flaps and flight feathers are parts of wings. Without
            # this, `what has wings` offers them as things that have wings.
            if way == SUBJECT and self._contradicted(row, wanted):
                continue
            if SKIP_BROAD and self.reasoner.too_broad(answer):
                continue
            seen.add(answer)
            found.append({"concept": answer, "relation": row["relation"],
                          "object": row["object"] if way == SUBJECT
                                    else row["concept"],
                          "source": row["source"],
                          "confidence": round(row["confidence"], 3),
                          "gloss": self.reasoner.gloss(answer)})
            if len(found) >= limit:
                break
        return found

    def answer(self, question: str, limit: int = FOUND) -> Backwards | None:
        read = self.reads(question)
        if read is None:
            return None
        relation, phrase, direction = read
        found = self.find(relation, phrase, direction, limit)
        found.question = question
        if not found.subjects:
            found.note = (f"Nothing is recorded as {relation.replace('_', ' ')} "
                          f"“{phrase}”. Absent, not false.")
        return found

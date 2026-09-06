"""Identification: the trie read downwards.

V684 answers "what can a dog do" -- name a thing, get its predicates. This
answers the inverse, which is what Appendix 3's trie is actually shaped for:

    what kind of dog has spots      -> dalmatian
    what is red and flies           -> robin

Storing an individual walks *down* the trie until its predicate set is
exhausted. Identifying one walks down the same structure until only one
individual remains underneath. It is the same mechanism and the same ordering
question: the predicate that eliminates most candidates is the one worth
asking first, which is `adaptive_coverage` used as a question-picker instead
of a storage plan.

Two sources of predicates, because neither is enough alone:

    stated      What the feature norms say. AwA2's `dalmatian` carries
                `spots` directly, so "which dog has spots" needs nothing else.

    inherited   What the concept gets from being a kind of something. XCSLB's
                `robin` lists `has a red breast` but no flying property at
                all -- of its 14 properties, not one mentions flight. It is
                `a small bird`, and v684 knows birds fly. Without the join
                "what is red and flies" has no answer; with it, robin does.

The join is XCSLB's own `concept_senses.csv`: 511 of its 530 concepts carry a
WordNet sense key that resolves straight into v684's taxonomy, so no sense
guessing is needed here at all.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..v684 import rules
from ..v684.language import Parser
from ..v684.reason import Reasoner
from . import corpora

#: Words that carry no discriminating power in a query.
NOISE = frozenset("""
what which kind kinds sort sorts type types of a an the is are was were be
that this these those has have had can could do does did and or with it its
thing things something anything one ones me you i tell name
""".split())

#: How far up the taxonomy an inherited predicate may come from. Beyond this
#: the properties are `physical entity` generalities that identify nothing --
#: the same reason v684's R12 exists.
INHERIT_DEPTH = 6

#: How many rivals to hand the page. Thirty birds is already a crowded globe.
MAX_CONSIDERED = 24

#: How many rivals to show per attribute. More than a handful and the
#: interesting near misses are lost among things that failed at once.
RIVALS_SHOWN = 5


@dataclass
class Candidate:
    name: str
    source: str
    matched: dict[str, str] = field(default_factory=dict)
    predicates: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source,
                "matched": self.matched, "predicates": self.predicates}


@dataclass
class Identification:
    question: str
    terms: list[str] = field(default_factory=list)
    among: str | None = None
    among_concept: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    #: Everyone who was in the running, survivors and eliminated alike, with
    #: the synset each resolves to. The page draws the elimination from this;
    #: without it there is nothing to animate but the words of the question.
    considered: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"question": self.question, "terms": self.terms,
                "among": self.among, "among_concept": self.among_concept,
                "verdict": self.verdict, "note": self.note,
                "candidates": [c.as_dict() for c in self.candidates],
                "considered": self.considered, "steps": self.steps}


class Identifier:
    """Find the individual a description picks out."""

    def __init__(self, store: Path, reasoner: Reasoner | None = None,
                 inherit: bool = True, parser=None):
        self.reasoner = reasoner or Reasoner(store)
        # Routing and class extraction are grammatical, so a parser is not
        # optional in practice. One is built when the caller has none to
        # share, rather than silently falling back to the weaker patterns.
        self.parser = parser or Parser(vocabulary=self.reasoner.vocabulary())
        self._owns_reasoner = reasoner is None
        self.stated: dict[str, frozenset[str]] = {}
        self.origin: dict[str, str] = {}
        self.synset: dict[str, str] = {}

        for name, predicates in corpora.load_awa2().items:
            self.stated[name] = predicates
            self.origin[name] = "awa2"
        for name, predicates in corpora.load_xcslb().items:
            merged = self.stated.get(name, frozenset()) | predicates
            self.stated[name] = merged
            self.origin.setdefault(name, "xcslb")

        # the join: XCSLB ships the sense key, so this needs no guessing
        for concept, key in corpora.senses().items():
            if concept in self.stated:
                self._join(concept, key.split("%")[0].replace("_", " "))
        # AwA2 ships no sense keys, so its classes are looked up by name.
        # Without this `dalmatian` has no synset, and "what kind of dog has
        # spots" finds only the concept `dog` itself.
        for name in self.stated:
            if name not in self.synset:
                self._join(name, name)

        self.inherited: dict[str, frozenset[str]] = {}
        if inherit:
            for name in self.stated:
                self.inherited[name] = self._inherited(name)

    def _join(self, name: str, lemma: str) -> None:
        row = self.reasoner.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma = ? "
            "ORDER BY primary_sense DESC LIMIT 1", (lemma,)).fetchone()
        if row:
            self.synset[name] = row[0]

    def close(self) -> None:
        if self._owns_reasoner:
            self.reasoner.close()

    def _inherited(self, name: str) -> frozenset[str]:
        """Predicates the concept gets from what it is a kind of."""
        concept = self.synset.get(name)
        if not concept:
            return frozenset()
        out: set[str] = set()
        for node, distance, _ in self.reasoner.ascend(concept):
            if distance == 0 or distance > INHERIT_DEPTH:
                continue
            if self.reasoner.too_broad(node):
                break                      # R12: too general to say anything
            for fact in self.reasoner.facts_of(node):
                if rules.inheritable(fact.relation):
                    out.add(f"{fact.relation.replace('_', ' ')} {fact.object}")
        return frozenset(out)

    # -- reading the question ---------------------------------------------
    #: Roots that mean the question names a thing rather than describes one.
    #: "what can a violin *do*", "what does an owner *need*".
    NAMING_ROOTS = frozenset({"do", "need", "use", "mean", "call", "cost"})

    def describes(self, question: str) -> bool:
        """Is this a description of an unnamed thing, or a question about a
        named one?

        Grammar answers it better than a pattern. A description leaves the
        thing unnamed and says what it is like, which shows up as a relative
        clause (`an object *that is round*`), an adjectival complement (`what
        is *round* with spots`), or `what` used as a determiner (`*what
        animal* has stripes`). A naming question has a subject and asks what
        it does -- that is v684's, and it must not be taken.
        """
        text = (question or "").strip().lower().rstrip("?")
        if not re.match(r"^(what|which)", text):
            return False
        if re.search(r"(kind|type|sort)s?\s+of", text):
            return True
        if self.parser is None or self.parser.nlp is None:
            return bool(re.search(r"that\s+(is|are|has|have)", text))
        doc = self.parser.nlp(text)
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is not None and root.lemma_ in self.NAMING_ROOTS:
            return False
        if re.search(r"used\s+for", text):
            return False
        if any(t.dep_ == "poss" for t in doc):
            return False                       # `a dog's owner` is v685's
        if any(t.dep_ == "relcl" for t in doc):
            return True
        if any(t.dep_ == "acomp" for t in doc):
            return True
        opener = doc[0]
        if opener.dep_ == "det" and opener.head.pos_ in ("NOUN", "PROPN"):
            return True                        # "what animal has stripes"
        return False

    def terms_of(self, question: str) -> tuple[list[str], str | None]:
        """Content words to match on, and the class to search within.

        The class constrains the candidates rather than describing them, so it
        must not also be matched as a property. It can be named three ways:
        `what *kind of dog*`, `what is an *object* that ...`, `what *animal*
        has ...` -- and the properties are whatever is left.
        """
        text = question.lower().rstrip("?").strip()
        among = None
        kind = re.search(r"(?:kind|type|sort)s?\s+of\s+([a-z ]+?)\s+"
                         r"(?:has|have|is|are|can|does|do|that|with)", text)
        if kind:
            among = kind.group(1).strip()
        elif self.parser is not None and self.parser.nlp is not None:
            doc = self.parser.nlp(text)
            # `an object that is round`: the noun the relative clause hangs off
            relative = next((t for t in doc if t.dep_ == "relcl"), None)
            if relative is not None and relative.head.pos_ in ("NOUN", "PROPN"):
                among = relative.head.lemma_.lower()
            elif doc[0].dep_ == "det" and doc[0].head.pos_ in ("NOUN", "PROPN"):
                among = doc[0].head.lemma_.lower()
        if among and among not in self.stated and among not in self.synset:
            row = self.reasoner.connection.execute(
                "SELECT 1 FROM lemmas WHERE lemma = ? LIMIT 1", (among,)).fetchone()
            if row is None:
                among = None                   # not a class this data knows

        words = [w for w in re.findall(r"[a-z]+", text) if w not in NOISE]
        if among:
            for part in among.split():
                if part in words:
                    words.remove(part)
        return words, among

    def _within(self, among: str) -> set[str] | None:
        """The individuals that are a kind of `among`, via v684's taxonomy."""
        wanted = {row["concept"] for row in self.reasoner.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma = ?", (among,))}
        if not wanted:
            return None
        inside = set()
        for name, concept in self.synset.items():
            if concept in wanted:
                inside.add(name)
                continue
            for node, distance, _ in self.reasoner.ascend(concept):
                if node in wanted:
                    inside.add(name)
                    break
        return inside or None

    @staticmethod
    def stem(word: str) -> str:
        """Enough morphology to match `flies` to `fly` and `spots` to `spot`.

        Prefix matching was tried and is far too loose: at four characters
        `striven` matches `stripes` and `truncated` matches `trunk`, which put
        cheetahs under "has stripes" and desks under "has a trunk".
        """
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith(("ses", "xes", "zes", "ches", "shes")):
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word

    @classmethod
    def _hit(cls, term: str, predicates: frozenset[str]) -> str | None:
        """The predicate a query word names, if one does. Whole words only."""
        wanted = cls.stem(term)
        for predicate in predicates:
            if any(cls.stem(word) == wanted for word in predicate.split()):
                return predicate
        return None

    # -- the search --------------------------------------------------------
    def identify(self, question: str, limit: int = 8) -> Identification:
        terms, among = self.terms_of(question)
        result = Identification(question=question, terms=terms, among=among)
        if not terms:
            result.note = "No describing words in the question."
            return result

        pool = set(self.stated)
        depth_of: dict[str, int] = {}
        if among:
            row = self.reasoner.connection.execute(
                "SELECT concept FROM lemmas WHERE lemma = ? "
                "ORDER BY primary_sense DESC LIMIT 1", (among,)).fetchone()
            result.among_concept = row[0] if row else None
            inside = self._within(among)
            if inside is None:
                result.note = f"“{among}” is not a class this data covers."
            else:
                pool = inside
                result.steps.append({
                    "rule": "R1", "kind": "narrow", "term": among,
                    "detail": f"Restrict to kinds of {among}.",
                    "remaining": len(pool),
                    "examples": sorted(pool)[:6]})

        # Stated before inherited. v684's inherited facts are free text from
        # a corpus, and matching one word inside them is how `collie` gets
        # "spots" out of `capable of spot movement` and `marble` gets to fly
        # via `capable of fly through the air`. What a norm actually says
        # about the thing is better evidence than what its ancestors say, so
        # inheritance is only consulted when nothing is stated at all.
        for allow_inherited in (False, True):
            alive = dict.fromkeys(sorted(pool))
            matched_by = {name: {} for name in alive}
            # Which *step* removed each candidate -- the index into
            # `result.steps`, not a separate counter. The class restriction
            # occupies step 0 when there is one, so a depth counted from the
            # terms alone is off by one exactly when a class was named, and
            # the rivals of `stripes` were drawn hanging off `animal`.
            offset = len(result.steps)
            fell_at: dict[str, int] = {}
            trace: list[dict] = []
            # Order the questions the way the trie would: the term that
            # eliminates most candidates is asked first. That is
            # `adaptive_coverage` used to pick a question, not a storage slot.
            ranked = sorted(terms,
                            key=lambda t: self._breadth(t, alive, allow_inherited))
            for term in ranked:
                keep = {}
                for name in alive:
                    hit = self._hit(term, self.stated.get(name, frozenset()))
                    origin = "stated"
                    if hit is None and allow_inherited:
                        hit = self._hit(term, self.inherited.get(name, frozenset()))
                        origin = "inherited"
                    if hit is not None:
                        keep[name] = None
                        matched_by[name][term] = f"{hit} ({origin})"
                before = len(alive)
                for name in alive:
                    if name not in keep:
                        fell_at[name] = offset + len(trace)
                alive = keep
                trace.append({
                    "rule": "R16", "kind": "narrow", "term": term,
                    "detail": f"Keep only what is “{term}”"
                              + ("" if not allow_inherited
                                 else ", allowing inherited properties"),
                    "source": "stated" if not allow_inherited else "inherited",
                    "remaining": len(alive), "eliminated": before - len(alive),
                    "examples": sorted(alive)[:6]})
                if not alive:
                    break
            if alive:
                result.steps.extend(trace)
                depth_of = fell_at
                break
            if allow_inherited:
                result.steps.extend(trace)
                depth_of = fell_at

        # Best evidence first. When nothing satisfies every property from
        # what is stated, the fallback lets inherited facts in and they are
        # corpus free text -- so a candidate that really does state two of the
        # three properties should still outrank one that inherits all three.
        def quality(name: str) -> tuple:
            stated = sum(1 for hit in matched_by[name].values()
                         if hit.endswith("(stated)"))
            return (-stated, len(self.stated.get(name, ())))

        for name in sorted(alive, key=quality):
            result.candidates.append(Candidate(
                name=name, source=self.origin.get(name, "?"),
                matched=matched_by[name],
                predicates=len(self.stated.get(name, ()))))
        result.candidates = result.candidates[:limit]
        survivors = {c.name for c in result.candidates}
        # Which rivals to show. Sorting by name gave `ambulance, accordion,
        # antelope` -- the alphabet, not the near misses. The interesting
        # rival is the thing most like the answer that still failed, so they
        # are ranked by how many properties they share with it, and each
        # attribute shows only a few.
        best = result.candidates[0].name if result.candidates else None
        target = self.stated.get(best, frozenset()) if best else frozenset()

        def closeness(name: str) -> tuple:
            shared = len(self.stated.get(name, frozenset()) & target)
            return (-shared, name)

        deepest = max(depth_of.values(), default=0) + 1
        grouped: dict[int, list[str]] = {}
        for name in pool:
            if name in survivors:
                continue
            grouped.setdefault(depth_of.get(name, deepest), []).append(name)
        for depth in sorted(grouped):
            for name in sorted(grouped[depth], key=closeness)[:RIVALS_SHOWN]:
                result.considered.append({
                    "name": name,
                    "concept": self.synset.get(name),
                    "survived": False,
                    "depth": depth,
                    "matched": matched_by.get(name, {}),
                })
        for name in sorted(survivors):
            result.considered.append({
                "name": name, "concept": self.synset.get(name),
                "survived": True, "depth": deepest,
                "matched": matched_by.get(name, {}),
            })

        if len(result.candidates) == 1:
            result.verdict = "IDENTIFIED"
        elif result.candidates:
            result.verdict = "AMBIGUOUS"
        else:
            result.verdict = "NO_MATCH"
            if not result.note:
                result.note = ("Nothing carries all of those properties. "
                               "Absent, not false.")
        return result

    def _breadth(self, term: str, pool, allow_inherited: bool) -> int:
        """How many candidates a term leaves -- fewer is a better question."""
        total = 0
        for name in pool:
            if self._hit(term, self.stated.get(name, frozenset())) is not None:
                total += 1
            elif allow_inherited and self._hit(
                    term, self.inherited.get(name, frozenset())) is not None:
                total += 1
        return total

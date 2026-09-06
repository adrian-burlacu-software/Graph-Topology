"""Which sense do word-level facts belong to?

Ascent++ and ConceptNet state facts about *words*. WordNet stores *senses*.
Joining them needs a decision, and the first version decided it by sense
number: eponymous synset first, then noun before verb, then `.01`. For
`hammer` that picks

    hammer.n.01  the part of a gunlock that strikes the percussion cap

over

    hammer.n.02  a hand tool with a heavy rigid head and a handle

and 348 facts -- carpenter's toolbox, hardware store, toolbelt -- landed on a
gun part. WordNet's sense order is not a usefulness ranking, and 30% of the
words carrying facts had two or more eponymous noun senses, so for those the
tie-break was a coin flip.

The evidence for the right sense is already in hand: the facts themselves.
`hammer` is stated to be in a toolbelt and a hardware store; `hammer.n.02`
sits under `hand tool` and `tool`, `hammer.n.01` under `gunlock` and
`fastener`. Overlap decides it. This is Lesk, with the gloss extended by the
sense's place in the taxonomy, and it is the same scoring at two grains:

    choose()  one sense per word, from all of that word's facts
    route()   one sense per fact, when a single fact disagrees strongly
              enough with the word's own primary sense

No labels, no model, no new dependency -- WordNet's own text scoring
ConceptNet's own text.
"""
from __future__ import annotations

import collections
import math
import re
import sqlite3

WORD = re.compile(r"[a-z]+")

#: Function words plus the vocabulary of glossing itself ("a kind of ...",
#: "used for ..."), which otherwise matches every sense equally.
STOPWORDS = frozenset("""
a an the of to in on at for with by from as or and not is are was were be been
being it its this that these those any some each other another such no nor but
if then than so too very can could may might must shall should will would
one two three ones kind kinds sort sorts type types way ways form forms
something someone anything anyone thing things person people place places
used use uses using usually often typically especially esp generally chiefly
made make makes making having have has had who whom whose which what when
where while during before after above below over under up down out off again
also more most much many few less least own same
""".split())

#: How much each part of a sense's neighbourhood counts toward its signature.
#: The gloss and the sense's own synonyms are what the word means; the
#: taxonomy around it is context, worth less but worth something -- `tool` is
#: not in hammer.n.02's gloss, it is its grandparent.
WEIGHTS = {"gloss": 1.0, "synonym": 1.0, "parent": 0.75, "child": 0.55,
           "fact": 0.5}

MAX_PARENT_DEPTH = 2
MAX_CHILDREN = 24

#: How much better a rival sense must score before a word abandons the synset
#: named after it. At 1.0 `seal` left for `navy seal.n.01` on a 5% edge.
EPONYMOUS_FACTOR = 1.25


def tokens(text: str) -> list[str]:
    """Content words of a phrase. `carpenter's toolbox` -> carpenter, toolbox."""
    return [w for w in WORD.findall(text.lower())
            if len(w) > 2 and w not in STOPWORDS]


class Senses:
    """Sense signatures over a built store, and the two ways to use them."""

    def __init__(self, connection: sqlite3.Connection):
        self.gloss: dict[str, str] = {}
        for identifier, definition in connection.execute(
            "SELECT id, definition FROM concepts"
        ):
            self.gloss[identifier] = definition or ""

        self.synonyms: dict[str, list[str]] = collections.defaultdict(list)
        self.candidates: dict[str, list[str]] = collections.defaultdict(list)
        for lemma, concept in connection.execute("SELECT lemma, concept FROM lemmas"):
            self.synonyms[concept].append(lemma)
            self.candidates[lemma].append(concept)

        self.parents: dict[str, list[str]] = collections.defaultdict(list)
        self.children: dict[str, list[str]] = collections.defaultdict(list)
        for child, parent in connection.execute("SELECT child, parent FROM taxonomy"):
            self.parents[child].append(parent)
            self.children[parent].append(child)

        self.structural: dict[str, list[str]] = collections.defaultdict(list)
        for concept, obj in connection.execute(
            "SELECT concept, object FROM facts WHERE source='wordnet'"
        ):
            self.structural[concept].append(obj)

        # Inverse document frequency over every gloss in the ontology, so
        # `device` counts for almost nothing and `percussion` counts a lot.
        document_frequency: collections.Counter = collections.Counter()
        for text in self.gloss.values():
            document_frequency.update(set(tokens(text)))
        total = max(1, len(self.gloss))
        self.idf = {word: math.log(total / count)
                    for word, count in document_frequency.items()}
        self.default_idf = math.log(total)
        self._cache: dict[str, dict[str, float]] = {}

    # -- signatures -------------------------------------------------------
    def signature(self, concept: str) -> dict[str, float]:
        """Weighted bag of words for one sense and its immediate surroundings."""
        cached = self._cache.get(concept)
        if cached is not None:
            return cached
        bag: dict[str, float] = collections.defaultdict(float)

        def add(text: str, weight: float) -> None:
            for word in tokens(text):
                bag[word] = max(bag[word], weight)

        add(self.gloss.get(concept, ""), WEIGHTS["gloss"])
        for name in self.synonyms.get(concept, ()):
            add(name, WEIGHTS["synonym"])
        for obj in self.structural.get(concept, ()):
            add(str(obj).rsplit(".", 2)[0], WEIGHTS["fact"])

        frontier, seen = [concept], {concept}
        for depth in range(MAX_PARENT_DEPTH):
            nxt = []
            for node in frontier:
                for parent in self.parents.get(node, ()):
                    if parent in seen:
                        continue
                    seen.add(parent)
                    nxt.append(parent)
                    weight = WEIGHTS["parent"] * (0.7 ** depth)
                    for name in self.synonyms.get(parent, ()):
                        add(name, weight)
                    if depth == 0:
                        add(self.gloss.get(parent, ""), weight * 0.6)
            frontier = nxt
        for child in self.children.get(concept, ())[:MAX_CHILDREN]:
            for name in self.synonyms.get(child, ()):
                add(name, WEIGHTS["child"])

        self._cache[concept] = dict(bag)
        return self._cache[concept]

    # -- scoring ----------------------------------------------------------
    def score(self, concept: str, evidence: collections.Counter,
              skip: str = "") -> float:
        """How well one sense explains a bag of evidence words."""
        signature = self.signature(concept)
        skip_words = set(tokens(skip))
        total = 0.0
        for word, count in evidence.items():
            if word in skip_words:
                continue                    # every sense of `hammer` says hammer
            weight = signature.get(word)
            if weight:
                total += weight * min(count, 3) * self.idf.get(word, self.default_idf)
        return total

    def prior(self, lemma: str, concept: str) -> tuple[int, int, int]:
        """The old rule, kept as the tie-break when there is no evidence."""
        head, pos, sense = concept.rsplit(".", 2)
        return (0 if head == lemma else 1,
                {"n": 0, "v": 1, "a": 2, "r": 3}.get(pos, 9), int(sense))

    def choose(self, lemma: str, evidence: collections.Counter
               ) -> tuple[str | None, float, float]:
        """The sense of `lemma` that its own facts point at.

        Returns the sense, its score, and the margin over the runner-up. A
        margin of zero means the evidence did not decide it and the prior did.

        Two guards keep the evidence from overreaching, both found by reading
        what it changed. Without the first, `dog` went to `chase.v.01` and
        `spring` to `jump.v.01`: these sources state facts about things, so a
        noun sense wins whenever the word has one. Without the second, a thin
        margin was enough to leave the synset actually named for the word --
        `seal` to `navy seal.n.01`. Leaving it now takes a decisive score, so
        `drink` still reaches `beverage.n.01` at 12x but `seal` stays put.
        """
        options = self.candidates.get(lemma)
        if not options:
            return None, 0.0, 0.0
        nouns = [c for c in options if c.rsplit(".", 2)[1] == "n"]
        options = nouns or options
        ranked = sorted(
            ((self.score(c, evidence, skip=lemma), c) for c in options),
            key=lambda pair: (-pair[0], self.prior(lemma, pair[1])))
        best_score, best = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        eponymous = next((pair for pair in ranked
                          if pair[1].rsplit(".", 2)[0] == lemma), None)
        if (eponymous and eponymous[1] != best
                and best_score < eponymous[0] * EPONYMOUS_FACTOR):
            return eponymous[1], eponymous[0], 0.0
        return best, best_score, best_score - runner_up


class Ranges:
    """R13: does a fact's object name the kind of thing the relation takes?

    Built from the taxonomy and the lemma index, so it can be used during the
    build, before the store is finished. A phrase is resolved by trying the
    whole thing first and then its last word -- `carpenter's toolbox` is not a
    synset, `toolbox` is -- and a phrase that resolves to nothing passes
    unchecked.

    **Any** sense of the object may satisfy the range, not just the one the
    build chose for it. The rule rejects what it can prove wrong, and a word
    with a place among its senses is not proof of anything: checking only the
    chosen sense threw away `hammer at_location store`, because the corpus
    evidence for the word `store` had settled on `store.n.02`, a supply. It is
    a weaker filter that way, and correct rather than merely effective.
    """

    ARTICLE = re.compile(r"^(?:a|an|the|some|any|his|her|their|its|your|my)\s+")

    def __init__(self, parents: dict[str, set[str]],
                 senses_of: dict[str, list[str]], ranges: dict[str, str]):
        self.parents = parents
        self.senses_of = senses_of
        self.ranges = ranges
        self._ancestors: dict[str, set[str]] = {}

    def ancestors(self, concept: str, limit: int = 24) -> set[str]:
        cached = self._ancestors.get(concept)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = [concept]
        for _ in range(limit):
            nxt: set[str] = set()
            for node in frontier:
                for parent in self.parents.get(node, ()):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.add(parent)
            if not nxt:
                break
            frontier = list(nxt)
        self._ancestors[concept] = seen
        return seen

    def denotes(self, phrase: str) -> list[str]:
        """Every sense a free-text object could name; empty if unrecognised."""
        text = self.ARTICLE.sub("", str(phrase).strip().lower())
        if not text:
            return []
        head = text.rsplit(" ", 1)[-1] if " " in text else None
        for key in (text, head):
            if key and key in self.senses_of:
                return self.senses_of[key]
        return []

    def allows(self, relation: str, obj: str) -> bool:
        root = self.ranges.get(relation)
        if root is None:
            return True
        options = [c for c in self.denotes(obj) if c.rsplit(".", 2)[1] == "n"]
        if not options:
            return True                     # unknown, or not a noun at all
        return any(c == root or root in self.ancestors(c) for c in options)


# Routing each fact individually was tried and abandoned. The idea was that
# `hammer at_location carpenter's toolbox` names the tool by itself, so a fact
# could be sent to a sense other than the word's primary one. It moved 77,950
# of 1.9M facts and most of the moves were wrong:
#
#     drink   created_by  freezing rose wine   -> drink.v.02  (consume alcohol)
#     fire    capable_of  burn clothing        -> burn.v.01   (destroy by fire)
#     change  capable_of  become official      -> deepen.v.04
#
# The object of a fact describes the *predicate*, not the subject. Scoring a
# subject's senses against it measures the wrong thing, and a verb in the
# object reliably drags the subject to a verb sense. Pooled over hundreds of
# objects that noise averages out, which is why `choose` works and this did
# not. One object is not enough evidence, and no margin fixes that.

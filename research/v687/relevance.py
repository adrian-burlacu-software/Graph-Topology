"""R14: a fact about a sibling of the anchor is not a fact about the anchor.

"What can a violin player do" bridges to `musician.n.01` and then answers with
everything musicians do -- including *play drum*, *play trumpet*, *play
accordion*, *play cello*. Every one of those is true of musicians and false of
violin players. The bridge chose the right concept and then threw away the
thing that made it right: the violin.

The anchor has to constrain the answer, not just select the sense. And the
constraint is the one inheritance already uses, pointed sideways:

    an ancestor's property descends          (R2, v684)
    a *sibling's* property does not          (R14, here)

`drum` and `violin` are both musical instruments, so "musicians play drums"
says nothing about violin players -- it is evidence about a different member
of the same class. Whereas `instrument` is violin's ancestor and `string` is
its parent, so "tune the instrument" does transfer.

So a fact is set aside when its object names a **co-hyponym** of the anchor:
not the anchor, not above or below it, but joined to it by a class specific
enough that being a different member of it matters. Specificity is read off
the subtree size the build already stores:

    cello           lowest common ancestor  bowed stringed instrument      11
    piano, guitar                           stringed instrument            48
    drum, trumpet                           musical instrument            163
    ----------------------------------------------------- SIBLING_LIMIT ------
    bow, scale                              device                      2,764
    case, stage                             instrumentality             5,516
    note                                    artifact                   10,698
    teacher                                 whole                      31,542
    concert, music, orchestra               entity                     82,114

Everything a violin player actually does or uses sits below the line by an
order of magnitude, and every rival instrument sits above it. The threshold is
put in the gap, not tuned into it.

The object's sense is decided the same way R13 decides a range: by looking at
every sense the word could carry and taking the most specific evidence. It has
to be, because `drum` resolves to a barrel by default, `bass` to a fish and
`brass` to management -- and all three are instruments here.

Set-aside facts are returned, not deleted. "These are about other instruments"
is worth saying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .graph import FactGraph

#: How specific a shared class has to be before being a different member of it
#: counts against a fact. Measured: rival instruments join violin at 11-163
#: descendants, everything legitimately related at 2,764 or more.
SIBLING_LIMIT = 1000

#: How many senses of an object's head word to weigh.
SENSES_TRIED = 8

ANCHOR, KIN, SIBLING, UNRELATED = "anchor", "kin", "sibling", "unrelated"

#: Shown in the page's rule list beside v684's R1-R13. It is stated here
#: rather than in v684's rules.py because v684 does not implement it and must
#: keep working without v685.
RULE_TEXT: dict[str, str] = {
    "R15": "Bridging: a question with two subjects is answered about the "
           "second, but only after a route to it is found in the fact graph. "
           "Each hop is a stored fact, and the route is shown so it can be "
           "judged.",
    "R14": f"Sibling exclusion: a fact about a co-hyponym of the anchor does "
           f"not transfer to it. An ancestor's property descends (R2); a "
           f"sibling's does not. Two concepts are siblings when their nearest "
           f"shared class has fewer than {SIBLING_LIMIT:,} descendants -- "
           f"`violin` and `drum` meet at `musical instrument` (163), while "
           f"`violin` and `bow` only meet at `device` (2,764).",
}


@dataclass
class Judgement:
    """Why a fact was kept or set aside, in the terms of the rule."""
    verdict: str
    sense: str | None = None
    shared: str | None = None
    shared_size: int = 0

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "sense": self.sense,
                "shared": self.shared, "shared_size": self.shared_size}


class Relevance:
    """Judge a role's facts against the anchor the question came from."""

    def __init__(self, graph: FactGraph, limit: int = SIBLING_LIMIT):
        self.graph = graph
        self.limit = limit
        self.size = dict(graph.connection.execute(
            "SELECT id, descendants FROM concepts"))
        self._levels: dict[str, dict[str, int]] = {}

    def levels(self, concept: str, limit: int = 12) -> dict[str, int]:
        """Every ancestor with its distance, the concept itself at zero."""
        cached = self._levels.get(concept)
        if cached is not None:
            return cached
        out = {concept: 0}
        frontier = [concept]
        for depth in range(1, limit + 1):
            nxt: list[str] = []
            for node in frontier:
                for parent in self.graph.parents.get(node, ()):
                    if parent not in out:
                        out[parent] = depth
                        nxt.append(parent)
            frontier = nxt
            if not frontier:
                break
        self._levels[concept] = out
        return out

    def compare(self, concept: str, anchor: str) -> Judgement:
        """How one concept stands to the anchor."""
        if concept == anchor:
            return Judgement(ANCHOR, concept, concept, self.size.get(concept, 0))
        above_anchor = self.levels(anchor)
        above_concept = self.levels(concept)
        # An ancestor's facts descend and a descendant is a special case of us:
        # both transfer, so neither is a sibling.
        if concept in above_anchor or anchor in above_concept:
            return Judgement(KIN, concept, concept, self.size.get(concept, 0))
        shared = set(above_concept) & set(above_anchor)
        if not shared:
            return Judgement(UNRELATED, concept)
        nearest = min(shared, key=lambda c: (above_concept[c] + above_anchor[c],
                                             self.size.get(c, 0)))
        size = self.size.get(nearest, 0)
        if size < self.limit:
            return Judgement(SIBLING, concept, nearest, size)
        return Judgement(UNRELATED, concept, nearest, size)

    def judge(self, phrase: str, anchor: str) -> Judgement:
        """Judge a fact's object phrase against the anchor.

        Every sense of the head word is weighed and the most specific reading
        wins, because the default sense of `drum` is a barrel and of `bass` a
        fish, while the fact plainly means the instrument.
        """
        text = str(phrase).strip().lower()
        head = text.rsplit(" ", 1)[-1] if " " in text else None
        options: list[str] = []
        for key in (text, head):
            if key and key in self.graph.lemmas:
                options = [c for c in self.graph.lemmas[key][:SENSES_TRIED]
                           if c.rsplit(".", 2)[1] == "n"]
                if options:
                    break
        if not options:
            return Judgement(UNRELATED)
        best = Judgement(UNRELATED)
        for candidate in options:
            judgement = self.compare(candidate, anchor)
            if judgement.verdict in (ANCHOR, KIN):
                return judgement            # nothing outranks belonging
            if judgement.verdict == SIBLING and (
                    best.verdict != SIBLING or
                    judgement.shared_size < best.shared_size):
                best = judgement
        return best

    def split(self, facts: Iterable, anchor: str):
        """Facts that survive the anchor, and the ones it rules out.

        Kept facts are reordered so anything naming the anchor itself leads:
        for a violin player, `play violin` should not be the seventh row.
        """
        kept, excluded, judgements = [], [], {}
        for fact in facts:
            judgement = self.judge(fact.object, anchor)
            judgements[id(fact)] = judgement
            (excluded if judgement.verdict == SIBLING else kept).append(fact)
        kept.sort(key=lambda f: 0 if judgements[id(f)].verdict == ANCHOR else 1)
        return kept, excluded, judgements

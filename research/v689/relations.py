"""S. Where things are against each other, and how big.

Episodic memory kept `the kitchen is north of the office` as a quality of the
kitchen, `has_property "north of an office"`, with which office beside it. A
quality answers `is the kitchen north of the office` and nothing else: not
`is the office south of the kitchen`, not `what is north of the office`, not
`how do you go from the kitchen to the garden`. This is the part of memory
that reads such facts as what they are, relations between two individuals
along one dimension -- a projection of the same stream (`events.py`), like
story time.

    the kitchen is north of the office        north-south: kitchen above office
    the triangle is above the pink rectangle  vertical, and in line
    the square is to the left of the triangle lateral, and level
    the box is bigger than the chest          size
    the chocolate fits inside the box         size: the chocolate is smaller

## The dimensions

A relation places one individual on one side of another along a dimension:
the two compass axes, the vertical and lateral axes of a picture, and size.
The words for each side are WordNet antonyms -- north and south, east and west,
above and below, left and right, big and small -- which is what makes each
pair one relation. This table, the frame of reference, is the one thing here
that is not read out of a resource: which axis is at right angles to which is
geometry, not vocabulary.

## The rules

**S1. A relation and its converse are one fact.** The kitchen north of the
office is the office south of the kitchen. What fits inside something is
smaller than it: containing is being bigger.

**S2. Along one dimension, a relation is a partial order, walked like the
taxonomy.** `is the box bigger than the chocolate`, told the box is bigger
than the chest and the chest bigger than the chocolate, is a walk along
`bigger` edges -- as T2 walks `before` and R1 walks `is_a`: yes when the one is
reached from the other, no when the other is reached from the one, and not
told when neither is. Absent, not false.

**S3. A direction puts the two in line.** What is above something is directly
above it, so the two are level along the lateral axis; what is north of
something is due north. This is the prototype of a projective preposition
(Logan and Sadler's spatial templates: `above` is best directly above), and
the way bAbI's positional stories are drawn. Two individuals level along one
axis are compared along it through what they are level with: the square left
of the triangle and the rectangle below the triangle make the rectangle right
of the square. Level along an axis is neither side of it: no.

**S4. Directions make a map.** `how do you go from the kitchen to the garden`
is the shortest walk over what was told of the compass, each step said as the
direction it goes: south, then east.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687 import walks
from research.v687.links import LINKS

#: Words that are not part of a relation's phrase.
ARTICLES = frozenset({"a", "an", "the"})


@dataclass(frozen=True)
class Dimension:
    name: str
    #: the side up the order, and its antonym
    more: str
    less: str
    #: the dimension at right angles to it, which a relation along this one
    #: puts the two level on (S3); empty where there is none
    across: str = ""
    #: a map: its relations can be walked as steps (S4)
    compass: bool = False


#: One per relation `links.py` gives an axis: its row is the side up the
#: order, its converse the side down.
DIMENSIONS = {one.axis: Dimension(one.axis, one.name, one.converse,
                                  one.across, one.compass)
              for one in LINKS.values() if one.axis}

#: (words, dimension, side): `side` is +1 where the one said first is on the
#: `more` side of the other. Articles are left out, and the longest match is
#: taken.
PHRASES = tuple(sorted((
    (("north", "of"), "north-south", 1), (("south", "of"), "north-south", -1),
    (("east", "of"), "east-west", 1), (("west", "of"), "east-west", -1),
    (("above",), "vertical", 1), (("below",), "vertical", -1),
    (("to", "right", "of"), "lateral", 1), (("to", "left", "of"), "lateral", -1),
    (("bigger", "than"), "size", 1), (("larger", "than"), "size", 1),
    (("smaller", "than"), "size", -1),
    # S1: what fits inside is smaller than what it fits inside.
    (("fits", "inside"), "size", -1), (("fits", "in"), "size", -1),
    (("fit", "inside"), "size", -1), (("fit", "in"), "size", -1),
), key=lambda one: -len(one[0])))


@dataclass(frozen=True)
class Phrase:
    """A relation said at the start of a verb phrase."""

    dimension: str
    side: int
    #: how many words it took, articles included
    length: int
    words: tuple


def phrase(rest: list[str]) -> Phrase | None:
    """The relation a verb phrase opens with: `north of the office`, `to the
    left of the triangle`, `fits inside the box`."""
    kept = [(index, word) for index, word in enumerate(rest)
            if word not in ARTICLES]
    for words, dimension, side in PHRASES:
        if tuple(word for _, word in kept[:len(words)]) == words:
            length = kept[len(words) - 1][0] + 1
            return Phrase(dimension, side, length, words)
    return None


@dataclass
class Relation:
    """`first` is on `side` of `second` along `dimension`, as told."""

    first: str
    second: str
    dimension: str
    side: int
    said: str = ""
    seq: int = 0

    def as_dict(self) -> dict:
        return {"first": self.first, "second": self.second,
                "dimension": self.dimension, "side": self.side,
                "said": self.said, "seq": self.seq}


@dataclass
class Found:
    """What S2 and S3 came to: yes, no, or not told, and on what."""

    value: bool | None
    #: the relations the answer was reached through, in order
    path: list = field(default_factory=list)
    #: the two were level along the dimension (S3)
    level: bool = False


class Relations:
    """Every relation told between two individuals, and S1 to S4 over them."""

    def __init__(self, log) -> None:
        self.relations: list[Relation] = []
        log.on("told", self._on_told)

    def _on_told(self, event) -> None:
        data = event.data
        if data.get("relation") != "has_property":
            return      # `is not bigger than`: a denial places nothing
        bound = data.get("bound")
        found = phrase([word for word in (data.get("object") or "").split()])
        if not bound or found is None or bound == data.get("node"):
            return
        self.relations.append(Relation(data["node"], bound, found.dimension,
                                       found.side, data.get("said", ""),
                                       event.seq))

    # -- S3: level ---------------------------------------------------------
    def _levels(self, dimension: str) -> dict[str, str]:
        """individual -> the representative of everyone level with it along
        `dimension`: put in line by a relation along the axis across it."""
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for relation in self.relations:
            find(relation.first)
            find(relation.second)
            if DIMENSIONS[relation.dimension].across == dimension:
                parent[find(relation.first)] = find(relation.second)
        return {node: find(node) for node in list(parent)}

    # -- S2: order ---------------------------------------------------------
    def _upward(self, dimension: str, level: dict) -> dict[str, list]:
        """class -> [(class above it, relation)] along a dimension."""
        edges: dict[str, list] = {}
        for relation in self.relations:
            if relation.dimension != dimension:
                continue
            first = level.get(relation.first, relation.first)
            second = level.get(relation.second, relation.second)
            low, high = ((second, first) if relation.side > 0
                         else (first, second))
            edges.setdefault(low, []).append((high, relation))
        return edges

    @staticmethod
    def _walk(edges: dict, start: str, goal: str) -> list | None:
        """The relations on the way up from `start` to `goal`: the one path
        walk (`walks.path`) along the dimension's order."""
        route = walks.path(start, goal, lambda node: (
            (relation, higher) for higher, relation in edges.get(node, ())))
        return None if route is None else [relation for relation, _ in route]

    def compare(self, first: str, second: str, dimension: str,
                side: int) -> Found:
        """Is `first` on `side` of `second` along `dimension`? S2 over the
        classes S3 makes level."""
        level = self._levels(dimension)
        one = level.get(first, first)
        other = level.get(second, second)
        if one == other and first in level and second in level:
            return Found(False, level=True)
        edges = self._upward(dimension, level)
        low, high = (other, one) if side > 0 else (one, other)
        path = self._walk(edges, low, high)
        if path is not None:
            return Found(True, path)
        path = self._walk(edges, high, low)
        if path is not None:
            return Found(False, path)
        return Found(None)

    def beside(self, individual: str, dimension: str,
               side: int) -> list[tuple]:
        """(individual, relation) told directly on `side` of this one: `what
        is north of the office`, S1 reading each relation both ways."""
        out = []
        for relation in self.relations:
            if relation.dimension != dimension:
                continue
            if relation.second == individual and relation.side == side:
                out.append((relation.first, relation))
            elif relation.first == individual and relation.side == -side:
                out.append((relation.second, relation))
        return out

    # -- S4: a map ---------------------------------------------------------
    def route(self, start: str, goal: str) -> list[tuple] | None:
        """[(direction, individual reached, relation)] for the shortest walk
        over the compass from one individual to another; None if none."""
        steps: dict[str, list] = {}
        for relation in self.relations:
            dimension = DIMENSIONS[relation.dimension]
            if not dimension.compass:
                continue
            toward = dimension.more if relation.side > 0 else dimension.less
            away = dimension.less if relation.side > 0 else dimension.more
            steps.setdefault(relation.second, []).append(
                (toward, relation.first, relation))
            steps.setdefault(relation.first, []).append(
                (away, relation.second, relation))
        route = walks.path(start, goal, lambda node: (
            ((direction, relation), reached)
            for direction, reached, relation in steps.get(node, ())))
        return None if route is None else [
            (direction, reached, relation)
            for (direction, relation), reached in route]

    def as_dict(self) -> dict:
        return {"relations": [one.as_dict() for one in self.relations]}

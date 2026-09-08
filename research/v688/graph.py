"""Curiosity over the whole store, not the feature-norm slice of it.

`attention.Curiosity` reads `Profiles.plan`: 541 individuals and 3,677
predicates, which is the XCSLB and AwA2 feature norms. That is 1.2% of the
45,219 concepts the store has facts for, and it is why every run asked the
same six AwA2 columns -- fast, furry, active, tail, ground -- whatever the
subject was.

This asks a different question of the same store. Take the concept's
*siblings* out of the taxonomy, look at what they are recorded as having, and
ask whether the subject has it too. Where the norms give an attribute
checklist, this gives an **expectation**: every other hound is used for
hunting, so is a beagle? The question is a hypothesis rather than a slot, it
is different for every subject, and when the answer is no something has been
learned.

It is also R11 hoisting read backwards. R11 says a fact every child states
moves to the parent; this says a fact most children state is worth putting to
the child that has not stated it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gap import plain

#: Relations whose objects make a question worth asking. `related_to` is
#: 1.7M edges meaning "co-occurs" and R7 already refuses to reason with it;
#: `receives_action` and `has_property` carry most of the useful claims.
ASKABLE = ("capable_of", "has_property", "has_a", "has_part", "used_for",
           "made_of", "at_location", "receives_action", "eats", "desires")

#: Objects that say nothing when put to anything. Crawled corpora are full of
#: them and they make a question that cannot be wrong.
EMPTY = frozenset("""
thing things something anything one type kind sort way part place time
person people other others many some most all
""".split())

#: How many siblings to read. Beyond a dozen the query slows and the
#: expectation stops being about a family.
SIBLINGS = 12

#: An expectation has to be held by at least this share of the siblings read.
#: Below it the fact is one sibling's quirk, not something the family says.
SHARE_FLOOR = 0.34

#: And by at least this many, so a family of two cannot manufacture one.
HOLDERS_FLOOR = 2


@dataclass(frozen=True)
class Expectation:
    """Something the family says that the subject has not been asked about."""

    relation: str
    object: str
    holders: tuple[str, ...]     # the siblings that state it
    share: float                 # of the siblings read
    confidence: float
    parent: str

    @property
    def score(self) -> float:
        """How much this is worth asking.

        Share first -- an expectation only means something if the family
        actually holds it -- then confidence, so a crawl artefact three
        siblings happen to share does not outrank a solid fact four of them
        do.
        """
        return self.share * self.confidence

    def as_dict(self) -> dict:
        return {"relation": self.relation, "object": self.object,
                "holders": list(self.holders), "share": round(self.share, 3),
                "confidence": round(self.confidence, 3), "parent": self.parent}


class GraphCuriosity:
    """What a concept's family is recorded as having, that it is not."""

    def __init__(self, reasoner) -> None:
        self.reasoner = reasoner
        self._cache: dict[str, list[Expectation]] = {}

    # -- the family -------------------------------------------------------
    def family(self, sense: str) -> tuple[str, list[str]]:
        """The nearest ancestor with more than one child, and those children.

        Climbing matters: `penguin.n.01`, `shark.n.01` and `whale.n.02` are
        each an only child of their immediate parent, and stopping there left
        them with no family at all. Three levels is as far as it is worth
        going -- beyond that the siblings are not relatives.
        """
        seen, frontier = {sense}, [sense]
        for _ in range(3):
            parents: list[str] = []
            for child in frontier:
                for row in self.reasoner.connection.execute(
                        "SELECT parent FROM taxonomy WHERE child = ?",
                        (child,)):
                    parent = row["parent"]
                    if parent in seen:
                        continue
                    seen.add(parent)
                    parents.append(parent)
                    kin = [other["child"] for other
                           in self.reasoner.connection.execute(
                               "SELECT child FROM taxonomy WHERE parent = ? "
                               "AND child != ? ORDER BY child LIMIT ?",
                               (parent, sense, SIBLINGS))]
                    kin = [one for one in kin if one != sense]
                    if len(kin) >= HOLDERS_FLOOR:
                        return parent, kin
            if not parents:
                break
            frontier = parents
        return "", []

    # -- what they say that this one has not been asked -------------------
    def expectations(self, sense: str, limit: int = 8) -> list[Expectation]:
        if sense in self._cache:
            return self._cache[sense][:limit]
        parent, kin = self.family(sense)
        if not kin:
            self._cache[sense] = []
            return []

        marks = ",".join("?" * len(kin))
        rows = self.reasoner.connection.execute(
            f"SELECT relation, object, COUNT(DISTINCT concept) AS holders, "
            f"AVG(confidence) AS confidence, "
            f"GROUP_CONCAT(DISTINCT concept) AS who "
            f"FROM facts WHERE concept IN ({marks}) "
            f"AND relation IN ({','.join('?' * len(ASKABLE))}) "
            f"GROUP BY relation, object "
            f"HAVING holders >= ? "
            f"ORDER BY holders DESC, confidence DESC LIMIT 200",
            (*kin, *ASKABLE, HOLDERS_FLOOR)).fetchall()

        # What the subject already states, so the loop does not ask after
        # something the store would answer from the subject's own row.
        mine = {(row["relation"], row["object"]) for row in
                self.reasoner.connection.execute(
                    "SELECT relation, object FROM facts WHERE concept = ?",
                    (sense,))}

        found: list[Expectation] = []
        for row in rows:
            key = (row["relation"], row["object"])
            if key in mine:
                continue
            # `has_part pod.n.03` is how the store writes it, and a question
            # built straight from that asks `is a whale pod.n.03`.
            text = plain(row["object"] or "")
            if not text or text.lower() in EMPTY or len(text) > 40:
                continue
            if (row["relation"] == "has_property" and text.endswith("er")
                    and " " not in text):
                # A bare comparative has nothing to compare to: `is a violin
                # larger` is not a question anything can answer.
                continue
            share = row["holders"] / len(kin)
            if share < SHARE_FLOOR:
                continue
            found.append(Expectation(
                relation=row["relation"], object=text,
                holders=tuple((row["who"] or "").split(",")[:6]),
                share=share, confidence=float(row["confidence"] or 0.0),
                parent=parent))
        found.sort(key=lambda one: -one.score)
        self._cache[sense] = found
        return found[:limit]

    def sense_of(self, word: str) -> str:
        """The sense a question about this word is about, under its own name."""
        for sense in self.reasoner.senses_of(word) or []:
            name = sense.get("id") or ""
            if (name.split(".")[0].replace("_", " ") == word
                    and name.split(".")[-2:-1] == ["n"]):
                return name
        return ""


#: How many things that do a thing to read before deciding what it needs.
DOERS = 90

#: A part has to be held by this share of them at least. Below it, one
#: crawled row decides what running requires.
REQUIRES_SHARE = 0.12

#: And read at least this many doers, so a verb the store barely covers
#: cannot manufacture one. `sing` has thirteen and offered `tool`.
MIN_DOERS = 20

#: And it has to be this much commoner among them than among everything
#: else. `tooth` turns up for run, swim, climb and jump alike -- it is what
#: animals have, not what any of those needs. `wing` is 304 times commoner
#: among things that fly than among concepts at large.
REQUIRES_LIFT = 60.0


@dataclass(frozen=True)
class Requirement:
    """What the things that do something turn out to have in common."""

    action: str
    part: str
    holders: int
    doers: int
    lift: float

    @property
    def share(self) -> float:
        return self.holders / self.doers if self.doers else 0.0

    def as_dict(self) -> dict:
        return {"action": self.action, "part": self.part,
                "holders": self.holders, "doers": self.doers,
                "share": round(self.share, 3), "lift": round(self.lift, 1)}


class Requirements:
    """What an action needs, worked out rather than looked up.

    The store has `has_prerequisite`, but ConceptNet means it about people:
    reading requires `find book`, flying requires `get airline ticket`. What
    running needs of a body is not recorded anywhere.

    It is derivable. Take everything the store says can run, ask what those
    things have, and keep what is commoner among them than among concepts in
    general. Running gives **leg** (12 of 90, 102x), flying gives **wing**
    (31 of 87, 304x). It is cue validity, computed over the graph instead of
    over the feature norms.

    It does not always work, and the failure is informative rather than
    hidden: `walk` gives `friend` and `child`, because the crawl records
    walking a child; `swim` gives `tooth`. Where no part clears both floors
    the loop says the store could not tell it what the action needs, which is
    a better answer than a confident wrong one.
    """

    def __init__(self, reasoner) -> None:
        self.reasoner = reasoner
        self._cache: dict[str, Requirement | None] = {}
        self._overall: dict[str, int] = {}
        self._total = reasoner.connection.execute(
            "SELECT COUNT(DISTINCT concept) FROM facts").fetchone()[0] or 1

    def _commonness(self, part: str) -> float:
        if part not in self._overall:
            self._overall[part] = self.reasoner.connection.execute(
                "SELECT COUNT(DISTINCT concept) FROM facts "
                "WHERE relation IN ('has_a', 'has_part') AND object = ?",
                (part,)).fetchone()[0]
        return max(self._overall[part] / self._total, 1e-9)

    def _concrete(self, word: str) -> bool:
        """A requirement is a thing, not a standing. `run` otherwise needs
        `reputation`, `the power` and `the capability`, because the crawl is
        mostly about running a business."""
        senses = self.reasoner.senses_of(word) or []
        # Not a standing, and not a person. `walk` otherwise needs a `friend`
        # and `read` needs a `parent`, because the crawl records walking a
        # child and reading to one. A requirement is a part, not company.
        return bool(senses) and not any(
            self.reasoner.partition_of(sense["id"])
            in ("abstraction.n.06", "person.n.01")
            for sense in senses[:3])

    def of(self, action: str) -> Requirement | None:
        action = (action or "").strip().lower()
        if action in self._cache:
            return self._cache[action]
        self._cache[action] = None
        doers = [row["concept"] for row in self.reasoner.connection.execute(
            "SELECT DISTINCT concept FROM facts "
            "WHERE relation = 'capable_of' AND object = ? LIMIT ?",
            (action, DOERS))]
        if len(doers) < MIN_DOERS:
            return None
        marks = ",".join("?" * len(doers))
        rows = self.reasoner.connection.execute(
            f"SELECT object, COUNT(DISTINCT concept) AS holders FROM facts "
            f"WHERE concept IN ({marks}) "
            f"AND relation IN ('has_a', 'has_part') "
            f"GROUP BY object HAVING holders > 2 "
            f"ORDER BY holders DESC LIMIT 30", doers).fetchall()

        best: Requirement | None = None
        for row in rows:
            part = (row["object"] or "").strip()
            if " " in part or not self._concrete(part):
                continue
            share = row["holders"] / len(doers)
            lift = share / self._commonness(part)
            if share < REQUIRES_SHARE or lift < REQUIRES_LIFT:
                continue
            if best is None or lift > best.lift:
                best = Requirement(action, part, row["holders"], len(doers),
                                   lift)
        self._cache[action] = best
        return best


#: How many kinds a claim may be put to at once.
FAMILY_LIMIT = 8


class Kinds:
    """The kinds of a thing, taken from the whole taxonomy.

    `Profiles.subtypes` reads the feature norms, which cover 541 concepts, so
    every family the loop ever assembled came out of the same small pool:
    collie, dalmatian, german shepherd for dogs; carp, goldfish, minnow for
    fish. The taxonomy has 45,219 concepts with facts and knows that a dog is
    also a puppy, a pug, a poodle and a basenji.

    Both are used, and for different reasons. The norms are the only place a
    denial is *scored* -- `shark has legs: false` is recorded, not merely
    absent -- so they answer a family check decisively. The taxonomy has the
    actual relatives. Taking the union, ranked by how much the store holds
    about each, puts a claim to a family that is both real and answerable.
    """

    def __init__(self, reasoner, profiles) -> None:
        self.reasoner = reasoner
        self.profiles = profiles
        self._cache: dict[tuple[str, str], list[str]] = {}

    def of(self, word: str, sense: str = "") -> list[str]:
        key = (word, sense)
        if key in self._cache:
            return self._cache[key]

        norms = [name for name in self.profiles.subtypes(word)]
        below: list[str] = []
        if sense:
            below = [row["child"] for row in self.reasoner.connection.execute(
                "SELECT t.child, COUNT(f.concept) AS held FROM taxonomy t "
                "LEFT JOIN facts f ON f.concept = t.child "
                "WHERE t.parent = ? GROUP BY t.child "
                "HAVING held > 0 ORDER BY held DESC LIMIT ?",
                (sense, FAMILY_LIMIT))]

        found: list[str] = []
        for name in norms + [plain(one) for one in below]:
            name = (name or "").strip()
            if name and name != word and name not in found:
                found.append(name)
        self._cache[key] = found[:FAMILY_LIMIT]
        return self._cache[key]

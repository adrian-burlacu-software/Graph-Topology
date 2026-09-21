"""Problems to solve, over the domains that ship.

`world.py` says what a world is, `domains.py` says what one domain has in
it, and this is where the two meet: a start, a goal, and the ground actions
of a domain over the objects a problem happens to have.

The blocks suite is here because it is the one with an oracle and a measured
history, not because it is special: `of` builds a problem over any domain,
and `SAMPLERS` has one per domain so `acting --domain` can say whether the
planner works anywhere or only where it was tuned.
"""
from __future__ import annotations

import random

from research.v691 import world as W
from research.v691.domains import DOMAINS

BLOCKS = DOMAINS["blocks"]


def of(domain, objects: dict, start, goal, name: str = "") -> W.Problem:
    """One problem: the objects it has, what holds, and what is wanted."""
    return W.Problem(name or domain.name, frozenset(start), frozenset(goal),
                     tuple(domain.ground(objects)), dict(objects))


# -- the blocks suite ------------------------------------------------------

def start_of(towers) -> frozenset:
    """The facts of a configuration given as stacks, bottom block first."""
    facts = {"empty"}
    for stack in towers:
        facts.add(f"table {stack[0]}")
        for lower, upper in zip(stack, stack[1:]):
            facts.add(f"on {upper} {lower}")
        facts.add(f"clear {stack[-1]}")
    return frozenset(facts)


def blocks(name: str, names, towers, goal) -> W.Problem:
    return of(BLOCKS, {one: "block" for one in names},
              start_of(towers), goal, name)


#: The fixed suite. `sussman` is first because it is the reason v691 exists:
#: achieving `on a b` and then `on b c` in turn undoes the first, so a
#: planner that finishes one goal before starting the next cannot solve it
#: in fewer than the optimal six steps, and a careless one cannot solve it
#: at all. Everything after it is a size ladder, so the numbers say where
#: the method stops rather than only whether it works.
SUITE = [
    blocks("one step", "ab", [["a"], ["b"]], ["on a b"]),
    blocks("undo a tower", "ab", [["a", "b"]], ["on a b"]),
    blocks("three in a row", "abc", [["a"], ["b"], ["c"]],
           ["on a b", "on b c"]),
    blocks("sussman", "abc", [["c", "a"], ["b"]], ["on a b", "on b c"]),
    blocks("invert three", "abc", [["a", "b", "c"]], ["on b c", "on a b"]),
    blocks("four apart", "abcd", [["a"], ["b"], ["c"], ["d"]],
           ["on a b", "on b c", "on c d"]),
    blocks("four inverted", "abcd", [["a", "b", "c", "d"]],
           ["on a b", "on b c", "on c d"]),
    blocks("two towers", "abcd", [["a", "b"], ["c", "d"]],
           ["on b c", "on d a"]),
    blocks("five apart", "abcde", [[one] for one in "abcde"],
           ["on a b", "on b c", "on c d", "on d e"]),
    blocks("five inverted", "abcde", [["a", "b", "c", "d", "e"]],
           ["on a b", "on b c", "on c d", "on d e"]),
]


def sampled(count: int = 20, names: str = "abcd", seed: int = 0) -> list:
    """Random start and goal configurations, for numbers the fixed suite
    cannot give: it was chosen to be hard and so it says nothing about how
    often a method works."""
    rng = random.Random(seed)

    def configuration() -> list:
        order = list(names)
        rng.shuffle(order)
        towers, stack = [], [order[0]]
        for one in order[1:]:
            if rng.random() < 0.5:
                stack.append(one)
            else:
                towers.append(stack)
                stack = [one]
        towers.append(stack)
        return towers

    out = []
    for index in range(count):
        start, want = configuration(), configuration()
        goal = {f"on {upper} {lower}" for stack in want
                for lower, upper in zip(stack, stack[1:])}
        if not goal or start_of(start) >= goal:
            continue
        out.append(blocks(f"sampled {index}", names, start, sorted(goal)))
    return out


# -- the other domains, sampled the same way -------------------------------

def errands(count: int = 20, seed: int = 0) -> list:
    """Fetch things from places and leave them in others.

    Interference here is not a tower: the hands hold one thing, so two
    errands whose parts are in different places have to be interleaved, and
    a planner that finishes one before starting the next walks twice as far.
    """
    domain = DOMAINS["errands"]
    places = ["shop", "library", "office", "home"]
    things = ["book", "parcel", "keys", "cup"]
    rng = random.Random(seed)
    out = []
    for index in range(count):
        here = rng.sample(things, rng.randint(1, 3))
        objects = {one: "place" for one in places}
        objects.update({one: "thing" for one in here})
        start = set(domain.begin(objects)) | {"at home"}
        goal = set()
        for thing in here:
            was, wants = rng.sample(places, 2)
            start.add(f"in {thing} {was}")
            goal.add(f"in {thing} {wants}")
        out.append(of(domain, objects, start, goal, f"errands {index}"))
    return out


def delivery(count: int = 20, seed: int = 0) -> list:
    """Parcels, vans and towns: one van holds one parcel, so the ordering
    is over journeys rather than over a structure."""
    domain = DOMAINS["delivery"]
    towns = ["york", "leeds", "hull"]
    rng = random.Random(seed)
    out = []
    for index in range(count):
        parcels = [f"parcel{one}" for one in range(rng.randint(1, 2))]
        vans = ["blue"] if rng.random() < 0.5 else ["blue", "red"]
        objects = {one: "town" for one in towns}
        objects.update({one: "parcel" for one in parcels})
        objects.update({one: "van" for one in vans})
        start = set(domain.begin(objects))
        goal = set()
        for van in vans:
            start.add(f"parked {van} {rng.choice(towns)}")
            start.add(f"empty {van}")
        for parcel in parcels:
            was, wants = rng.sample(towns, 2)
            start.add(f"at {parcel} {was}")
            goal.add(f"at {parcel} {wants}")
        out.append(of(domain, objects, start, goal, f"delivery {index}"))
    return out


SAMPLERS = {"blocks": lambda count, seed: sampled(count, seed=seed),
            "errands": errands, "delivery": delivery}

"""How far a world with nothing declared about it gets.

`problems.py` measures the planner on worlds somebody wrote. This measures
it on the one nobody wrote: every problem is a few English sentences and a
goal, the actions come from VerbNet through `verbs.py`, and what a thing is
comes from the store. Nothing between the sentences and the plan was written
for these problems.

    python -m research.v691.errands

Two numbers matter and they are different questions. **Reached** is whether
a plan was found and executed to the goal -- which is what the planner is
being asked. **Sensible** is whether the actions it chose are ones a person
would have chosen, judged against `sense`, and that is a question about the
knowledge, not about the control. A plan can reach and not be sensible: a
book that rolls itself from the shop to the kitchen satisfies `at book
kitchen` and is not what anybody meant.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from research.v691 import acting, verbs, world as W


@dataclass
class Errand:
    """A situation, a goal, and what a person would have done."""

    name: str
    things: dict
    start: tuple
    goal: tuple
    #: verbs a person would accept in the plan; empty means any will do
    sense: tuple = ()
    steps: int = 1
    reached: bool = False
    sensible: bool = False
    plan: tuple = ()
    offered: int = 0


#: Written as situations, not as a domain. The `sense` lists are what a
#: person would accept, and they are generous on purpose -- the question is
#: whether the verb chosen is *a* way of doing it, not whether it is the one
#: the author had in mind.
ERRANDS = [
    Errand("fetch a book",
           {"john": "man", "book": "book", "kitchen": "kitchen",
            "shop": "shop"},
           ("at john kitchen", "at book shop", "alive john"),
           ("at book kitchen",),
           ("take", "bring", "carry", "get", "fetch", "move", "haul",
            "cart", "lug", "tote", "transport")),
    Errand("go somewhere",
           {"john": "man", "kitchen": "kitchen", "garden": "garden"},
           ("at john kitchen", "alive john"),
           ("at john garden",),
           ("go", "walk", "move", "travel", "run", "come", "enter",
            "proceed", "journey", "head")),
    Errand("open a door",
           {"john": "man", "door": "door", "hall": "hall"},
           ("at john hall", "at door hall", "alive john"),
           ("open door",),
           ("open", "unlock", "unfasten", "unbolt")),
    Errand("break a vase",
           {"john": "man", "vase": "vase", "hall": "hall"},
           ("at john hall", "at vase hall", "alive john"),
           ("break vase",),
           ("break", "smash", "shatter", "crack", "fracture", "splinter")),
    Errand("fill a glass",
           {"john": "man", "glass": "glass", "water": "water",
            "kitchen": "kitchen"},
           ("at john kitchen", "at glass kitchen", "at water kitchen"),
           ("fill glass",),
           ("fill", "load", "pack", "stuff", "cram", "pour")),
    Errand("two errands",
           {"john": "man", "book": "book", "cup": "cup",
            "kitchen": "kitchen", "shop": "shop"},
           ("at john kitchen", "at book shop", "at cup kitchen",
            "alive john"),
           ("at book kitchen", "at cup shop"),
           ("take", "bring", "carry", "get", "fetch", "move", "haul",
            "cart", "lug", "tote", "transport"), steps=2),
    Errand("move a chair",
           {"mary": "woman", "chair": "chair", "office": "office",
            "hall": "hall"},
           ("at mary office", "at chair office", "alive mary"),
           ("at chair hall",),
           ("move", "carry", "take", "push", "drag", "bring", "haul",
            "shift", "cart", "pull")),
    Errand("cook the meat",
           {"john": "man", "meat": "meat", "kitchen": "kitchen"},
           ("at john kitchen", "at meat kitchen", "alive john"),
           ("cook meat",),
           ("cook", "roast", "fry", "bake", "boil", "grill", "broil",
            "stew", "braise", "simmer")),
    Errand("wash the plate",
           {"john": "man", "plate": "plate", "kitchen": "kitchen"},
           ("at john kitchen", "at plate kitchen", "alive john"),
           ("wash plate",),
           ("wash", "clean", "rinse", "scrub", "launder", "bathe")),
    Errand("lock the box",
           {"john": "man", "box": "box", "hall": "hall"},
           ("at john hall", "at box hall", "alive john"),
           ("lock box",),
           ("lock", "bolt", "fasten", "secure", "latch")),
    Errand("kill the fly",
           {"john": "man", "fly": "fly", "kitchen": "kitchen"},
           ("at john kitchen", "at fly kitchen", "alive john", "alive fly"),
           ("kill fly",),
           ("kill", "slay", "murder", "swat", "destroy", "exterminate")),
    Errand("put it down",
           {"john": "man", "book": "book", "table": "table",
            "kitchen": "kitchen"},
           ("at john kitchen", "at table kitchen", "with book john"),
           ("at book table",),
           ("put", "place", "set", "lay", "drop", "leave", "position",
            "deposit", "rest", "stand")),
]


#: What people were heard saying they did, for `--seen`: other people,
#: other things, other places than any errand -- the verbs are what carry
#: over, and nothing else does.
SEEN = ("sam went to the park", "the girl walked to school",
        "he put the plate on the shelf", "mary took the cup to the office",
        "tom carried the bag to the car")


def preferences(sentences=SEEN) -> dict:
    """predicate -> verbs, most seen first, from what people said they did
    (`hearing.done`, the page's `seen_done`)."""
    from collections import Counter

    from research.v691 import hearing
    counts: dict = {}
    for said in sentences:
        for predicate, verb in hearing.hear(said, verbs.stated).done:
            counts.setdefault(predicate, Counter())[verb] += 1
    return {predicate: [verb for verb, _ in found.most_common()]
            for predicate, found in counts.items()}


def work(errand: Errand, resolver=None, per_verb: int = 8,
         prefer: dict | None = None) -> Errand:
    things = verbs.Things(resolver)
    for name, kind in errand.things.items():
        things.add(name, kind)
    actions = verbs.useful(errand.goal, things, per_verb=per_verb,
                           prefer=prefer)
    problem = W.Problem(errand.name, frozenset(errand.start),
                        frozenset(errand.goal), tuple(actions),
                        dict(things.kinds))
    got = acting.solve(problem, optimal=False)
    errand.offered = len(actions)
    errand.plan = tuple(str(one) for one in got.plan)
    errand.reached = got.solved
    used = {one.split()[0] for one in errand.plan}
    errand.sensible = bool(errand.reached and used and (
        not errand.sense or used <= set(errand.sense)))
    return errand


def measure(resolver=None, per_verb: int = 8,
            prefer: dict | None = None) -> list:
    return [work(Errand(**{**one.__dict__}), resolver, per_verb, prefer)
            for one in ERRANDS]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-verb", type=int, default=8,
                        help="how many verbs per wanted predicate are "
                             "ground (`verbs.useful`)")
    parser.add_argument("--seen", action="store_true",
                        help="prefer the verbs people were heard using "
                             "(`SEEN`, which shares nothing else with the "
                             "errands)")
    parser.add_argument("--no-store", action="store_true",
                        help="without the taxonomy, so nothing restricts "
                             "which thing may fill which role")
    options = parser.parse_args(argv)

    resolver = None
    if not options.no_store:
        from research.v691.openworld import resolver as of_store
        resolver = of_store()
        if resolver is None:
            print("no store: run `python -m regenerate`")
    prefer = preferences() if options.seen else None
    if prefer:
        print("seen:", prefer)
    rows = measure(resolver, options.per_verb, prefer)
    print(f"{'errand':<16} {'offered':>8} {'reached':>8} {'sensible':>9}  "
          f"plan")
    print("-" * 92)
    for one in rows:
        print(f"{one.name:<16} {one.offered:>8} "
              f"{'yes' if one.reached else 'no':>8} "
              f"{'yes' if one.sensible else 'no':>9}  "
              f"{'; '.join(one.plan)[:44]}")
    print("-" * 92)
    reached = sum(one.reached for one in rows)
    sensible = sum(one.sensible for one in rows)
    print(f"reached {reached}/{len(rows)}, sensible {sensible}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

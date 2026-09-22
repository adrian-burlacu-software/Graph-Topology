"""A world with rules the agent was not told: does it learn them by acting?

The agent plans in a model (`model`) and acts in a world (`real`) that has
two rules the model lacks:

    go      needs the place you are going to be lit    (a requirement)
    fetch   cannot be done to a thing that is wrapped  (a blocker)

Both remedies are in the model -- `light` lights a place, `unwrap` unwraps
a thing -- and what the model does not know is that they are ever needed.
So the only way to stop failing is to work out, from what happened, which
fact made the difference: nobody tells it, and nothing in the code names
`lit` or `wrapped` (`lessons.py`).

`--kinds` makes the blocker a fact about a kind of thing: a hat is a
garment, and a wrapped garment can be picked up. The actions are the same
for both kinds, so only what kind a thing is tells them apart -- and a
learner that compared a wrapped book's failure with a wrapped hat's success
would conclude that wrapping has nothing to do with it (`lessons.py`,
scope).

A run is a sequence of problems sharing one memory, and what is measured is
whether surprises stop: problems early in the sequence should fail and
surprise, problems late in it should not. Against the same sequence with no
learner, which fails the same way every time.

    python -m research.v691.hidden [--problems 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import random

from research.v691 import acting, domains, learned as L, lessons
from research.v691 import world as W
from research.v687.executive import Working

ROOMS = """
domain rooms
kind place
kind thing
object home place
world free, at home

goalish in

action go ?from:place ?to:place
  needs at ?from{go}
  adds  at ?to
  dels  at ?from
action light ?where:place
  adds  lit ?where
action fetch ?what:thing ?where:place
  needs at ?where, in ?what ?where, free
{fetch}  adds  carrying ?what
  dels  in ?what ?where, free
action unwrap ?what:thing ?where:place
  needs at ?where, in ?what ?where, wrapped ?what
  dels  wrapped ?what
action leave ?what:thing ?where:place
  needs at ?where, carrying ?what
  adds  in ?what ?where, free
  dels  carrying ?what

say at        I am at {{0}}
say in        the {{0}} is at {{1}}
"""

#: What the agent is told: the actions and what they do.
MODEL = domains.parse(ROOMS.format(go="", fetch=""))
#: What is so: the same, and two rules more.
REAL = domains.parse(ROOMS.format(go=", lit ?to",
                                  fetch="  forbids wrapped ?what\n"))

#: The same world with a second kind of thing, a garment, whose actions are
#: the thing's -- one text per kind, the domain language having no subkinds
#: -- and which the wrapping rule does not touch.
GARMENT = """
kind garment
action fetch ?what:garment ?where:place
  needs at ?where, in ?what ?where, free
  adds  carrying ?what
  dels  in ?what ?where, free
action unwrap ?what:garment ?where:place
  needs at ?where, in ?what ?where, wrapped ?what
  dels  wrapped ?what
action leave ?what:garment ?where:place
  needs at ?where, carrying ?what
  adds  in ?what ?where, free
  dels  carrying ?what
"""
MODEL_KINDS = domains.parse(ROOMS.format(go="", fetch="") + GARMENT)
REAL_KINDS = domains.parse(ROOMS.format(
    go=", lit ?to", fetch="  forbids wrapped ?what\n") + GARMENT)
#: Which things are garments, under `--kinds`.
GARMENTS = frozenset({"hat"})

PLACES = ("home", "shop", "park", "library")
THINGS = ("book", "cup", "keys", "hat")
#: Properties of places and of things that hold at random and matter to
#: nothing. They are what makes the evidence rule earn its keep: with them,
#: one failure beside one success usually leaves several candidates, and a
#: learner that took the first would learn that busy shops cannot be
#: entered.
DISTRACTING = (("busy", "quiet"), ("red", "old"))


def problems(count: int = 30, seed: int = 0, kinds: bool = False) -> list:
    """(model problem, real problem) pairs over the same start and goal."""
    rng = random.Random(seed)
    model, real = (MODEL_KINDS, REAL_KINDS) if kinds else (MODEL, REAL)
    out = []
    for index in range(count):
        here = rng.sample(THINGS, rng.randint(1, 2))
        objects = {one: "place" for one in PLACES}
        objects.update({one: "garment" if kinds and one in GARMENTS
                        else "thing" for one in here})
        start = {"free", "at home", "lit home"}
        goal = set()
        for place in PLACES[1:]:
            if rng.random() < 0.5:
                start.add(f"lit {place}")
        for place in PLACES:
            for quality in DISTRACTING[0]:
                if rng.random() < 0.5:
                    start.add(f"{quality} {place}")
        for thing in here:
            was = rng.choice(PLACES[1:])
            start.add(f"in {thing} {was}")
            goal.add(f"in {thing} home")
            if rng.random() < 0.4:
                start.add(f"wrapped {thing}")
            for quality in DISTRACTING[1]:
                if rng.random() < 0.5:
                    start.add(f"{quality} {thing}")
        name = f"rooms {index}"
        out.append((W.Problem(name, frozenset(start), frozenset(goal),
                              tuple(model.ground(objects)), objects),
                    W.Problem(name, frozenset(start), frozenset(goal),
                              tuple(real.ground(objects)), objects)))
    return out


def run(pairs, learner=None) -> list:
    """Each problem worked in the real world from the model, in order.
    Returns one Attempt per problem."""
    out = []
    for model, real in pairs:
        world = W.World(real.start)
        # The world refuses by its own actions: the one with the model's
        # name is the one that is really done.
        truly = {one.name: one for one in real.actions}
        world.do = _by_name(world, truly)
        report = acting.Attempt(name=model.name)
        if learner is not None and getattr(learner, "by_kind", False):
            # What kind each thing here is: the problem's own objects.
            learner.kind_of = model.objects.get
            learner.is_a = (lambda thing, kind, objects=model.objects:
                            objects.get(thing) == kind)
        acting.agent(model, world, None, report, learner=learner).run(
            Working(goal=f"solve {model.name}"))
        report.solved = world.solved(model.goal)
        report.plan = tuple(world.did)
        out.append(report)
    return out


def _by_name(world: W.World, truly: dict):
    do = W.World.do

    def doing(action) -> bool:
        return do(world, truly.get(action.name, action))
    return doing


def summary(reports, cut: int = 10) -> dict:
    first, last = reports[:cut], reports[-cut:]
    needless = sum(1 for report in last for one in report.plan
                   if one.name.startswith("unwrap ")
                   and one.name.split()[1] in GARMENTS)
    return {"needless last": needless,
            "solved first": sum(one.solved for one in first),
            "solved last": sum(one.solved for one in last),
            "surprises first": sum(one.surprises for one in first),
            "surprises last": sum(one.surprises for one in last),
            "lessons": [(one.kind, one.verb, one.literal)
                        for report in reports for one in report.lessons]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kinds", action="store_true",
                        help="hats are garments, and the wrapping rule is "
                             "not theirs")
    options = parser.parse_args(argv)
    pairs = problems(options.problems, options.seed, options.kinds)
    runs = [("no learning", None, None)]
    runs.append(("learning", L.Learned(None), False))
    if options.kinds:
        runs.append(("by kind", L.Learned(None), True))
    for label, memory, by_kind in runs:
        learner = None
        if memory is not None:
            learner = lessons.Learner(memory)
            learner.by_kind = by_kind
        found = summary(run(pairs, learner))
        print(f"{label:12} solved {found['solved first']}/10 first, "
              f"{found['solved last']}/10 last; surprises "
              f"{found['surprises first']} first, "
              f"{found['surprises last']} last"
              + (f"; hats unwrapped needlessly, last 10: "
                 f"{found['needless last']}" if options.kinds else ""))
        if memory is not None:
            print("   kept:", [(verb, literal, memory.scoped(
                table, verb, literal)) for table, rows in (
                ("requires", memory.requirements()),
                ("blocks", memory.blockings())) for verb, literal, _ in rows])
            memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

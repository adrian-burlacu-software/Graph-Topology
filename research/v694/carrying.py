"""Carrying a design out: act, be surprised, learn, design again.

A design is imagined. Carrying it out is done in a world (`W.World`) that
may refuse a step -- the fridge the design put the milk in is broken -- and
that is v691's surprise, with v691's learner to draw a lesson from it
(`lessons.Learner.failed`). What it learns is about the *step*, over its
positions: `put ?subject ?object` is blocked while `broken ?object` holds.
The designer folds what was learned into every step it imagines from then
on (`designing.checked`), so the next design does not use a broken fridge
-- and neither does any design after it, for anything put anywhere.

This is the loop v693 ran over forms, run over the world: nothing here
knows what a fridge is.

    carry_out(goal, world, learner)   design, act, learn, redesign

**What was remembered is tried first next time.** A way that worked is
kept (`Remembered`): the verb or state it met, the kind of thing it met it
for, and what it used. The next goal of the same shape is offered what
worked before first, whatever the store ranks highest -- and a way a person
*teaches* (`remember(..., said=...)`) is kept the same way, which is how a
way the store never heard of gets into a design at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from research.v691 import acting, world as W
from research.v694 import designing, knowing as K
from research.v694.goals import Clause, Goal


# -- what worked, kept -----------------------------------------------------

@dataclass
class Way:
    """One way remembered: `form` with `means`, for `predicate` done to a
    thing of `kind`."""

    predicate: str
    kind: str
    form: str
    means: str
    #: how many times it worked
    worked: int = 1
    #: who said so, when a person taught it
    said: str = ""


@dataclass
class Recipe:
    """How a thing is made: `product` from `parts`. Taught, or seen done,
    and kept -- the store's `made_of` lists what a thing *can* be made of
    (a blanket of wool, or cotton, or fleece), which is not a recipe."""

    product: str
    parts: tuple
    worked: int = 0
    said: str = ""


class Remembered:
    """Ways that worked or were taught, and how things are made.

    **What worked for one thing is tried for things like it.** A way is
    offered for the same predicate on the same kind of thing at full
    weight, and at a discount (`knowing.similar`, `similar_verb`) for a
    similar doing on a similar thing: a knife that cut a rope is worth
    trying on a cord, and a way to cut is worth trying to slice. What is
    offered is only offered: it still has to check in the imagined world,
    and Occam still decides."""

    def __init__(self, path=None) -> None:
        self.path = Path(path) if path else None
        self.ways: list = []
        self.recipes: list = []
        if self.path is not None and self.path.exists():
            try:
                kept = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(kept, list):
                    kept = {"ways": kept}
                self.ways = [Way(**one) for one in kept.get("ways", ())]
                self.recipes = [Recipe(one["product"], tuple(one["parts"]),
                                       one.get("worked", 0),
                                       one.get("said", ""))
                                for one in kept.get("recipes", ())]
            except (OSError, ValueError, TypeError, KeyError):
                self.ways, self.recipes = [], []

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "ways": [one.__dict__ for one in self.ways],
            "recipes": [dict(one.__dict__, parts=list(one.parts))
                        for one in self.recipes]}, indent=1),
            encoding="utf-8")

    def remember(self, predicate: str, kind: str, form: str, means: str,
                 said: str = "") -> Way:
        for one in self.ways:
            if (one.predicate, one.kind, one.form, one.means) == (
                    predicate, kind, form, means):
                one.worked += 1
                one.said = one.said or said
                self.save()
                return one
        one = Way(predicate, kind, form, means, said=said)
        self.ways.append(one)
        self.save()
        return one

    def forget(self, predicate: str, kind: str, means: str) -> None:
        self.ways = [one for one in self.ways if not (
            one.predicate == predicate and one.means == means
            and _alike(one.kind, kind))]
        self.save()

    def ways_for(self, clause: Clause, kind: str) -> list:
        """(way, how alike its goal is to this one) for what worked on a
        goal like this, the most alike and most used first."""
        found = []
        for one in self.ways:
            if one.predicate == clause.predicate:
                verb = 1.0
            elif clause.kind == "done":
                verb = K.similar_verb(one.predicate, clause.predicate)
            else:
                verb = 0.0
            weight = verb * K.similar(one.kind, kind)
            if weight > 0:
                found.append((one, weight))
        return sorted(found, key=lambda pair: (-pair[1], -pair[0].worked,
                                               pair[0].means))

    def learn_recipe(self, product: str, parts, said: str = "") -> Recipe:
        parts = tuple(sorted(parts))
        for one in self.recipes:
            if one.product == product and one.parts == parts:
                one.said = one.said or said
                self.save()
                return one
        one = Recipe(product, parts, said=said)
        self.recipes.append(one)
        self.save()
        return one

    def recipes_for(self, means: str) -> list:
        """Recipes for this thing, or a kind of it: a recipe for a torch
        makes something that serves wherever a torch does."""
        return sorted((one for one in self.recipes
                       if K.similar(one.product, means) >= 1.0),
                      key=lambda one: -one.worked)

    def made(self, product: str) -> None:
        for one in self.recipes:
            if one.product == product:
                one.worked += 1
        self.save()


def _alike(one: str, other: str) -> bool:
    return K.similar(one, other) >= 1.0


# -- carrying out ----------------------------------------------------------

@dataclass
class Outcome:
    goal: Goal
    done: bool = False
    #: every design made, in order
    designs: list = field(default_factory=list)
    #: every step done, in order
    did: list = field(default_factory=list)
    #: (step, what was learned from it) for each surprise
    surprises: list = field(default_factory=list)

    def said(self) -> str:
        if not self.designs:
            return "I could not find a way"
        return self.designs[-1].said()


#: How many times a goal is designed again after a surprise.
TRIES = 4


def carry_out(goal: Goal, world: W.World, learner=None, memory=None,
              order=None, told=None, tries: int = TRIES) -> Outcome:
    """Design a way to the goal, do it in `world`, and when a step is
    refused, learn from it and design again from where things stand.

    `told(step, world)` is what a person says came with a failure -- *the
    fridge is broken* -- as facts; what is said is the one case where a
    single failure is enough to learn from (`lessons`)."""
    out = Outcome(goal)
    for _ in range(tries):
        now = Goal(goal.wants, frozenset(world.facts), goal.names,
                   goal.doer, goal.said, memory=memory)
        design = designing.design(now, learner=learner, order=order)
        out.designs.append(design)
        if not design.done:
            return out
        steps = list(design.steps)
        if learner is not None:
            steps = learner.applied(steps)
        refused = None
        for step in steps:
            before = frozenset(world.facts)
            if world.do(step):
                out.did.append(step)
                if learner is not None:
                    learner.worked(step, before, world.facts)
                continue
            refused = (step, before)
            break
        if refused is None:
            out.done = all(clause.literal in world.facts
                           for clause in goal.clauses)
            if out.done and memory is not None:
                for part in design.parts:
                    if part.best is not None and part.best.means:
                        memory.remember(part.clause.predicate,
                                        goal.kind_of(part.clause.patient),
                                        part.best.form,
                                        _plain(part.best.means))
                    for step in (part.best.steps if part.best else ()):
                        if step.name.startswith("make "):
                            memory.made(step.name.split()[1])
            return out
        step, before = refused
        lessons: list = []
        if learner is not None:
            gap = acting.Gap(step, frozenset(step.adds) - before,
                             frozenset(), len(out.did), before, refused=True)
            revealed = told(step, world) if told is not None else ()
            lessons = learner.failed(gap, frozenset(world.facts),
                                     frozenset(revealed or ()))
        out.surprises.append((step, lessons))
        if memory is not None:
            for part in design.parts:
                if part.best is not None and step.name in {
                        one.name for one in part.best.steps}:
                    memory.forget(part.clause.predicate,
                                  goal.kind_of(part.clause.patient),
                                  part.best.means)
    return out


def _plain(name: str) -> str:
    """What a thing used is remembered as: *another fridge* is a fridge."""
    from research.v694.ways import OTHER
    return name[len(OTHER):] if name.startswith(OTHER) else name

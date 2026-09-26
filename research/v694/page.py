"""Designing on the page: an order in the open world that takes a way.

Importing this registers the designer with v691's page (`page.contributes`)
for two acts, before v691's own planner:

    want   *cut the rope*            designed, then done in the scene
    how    *how can I make the milk cold*   designed, and said, not done

and adds two of its own: `teach` (a way or a recipe, taught or seen) and
`recipe` (*do you have a recipe for cake* -- `asking.py`).

It takes an order only when the way to it is one the planner could not
have found: a tool, a place that keeps a thing so, something to give
someone, someone else to ask (`TAKES`). Getting the book to the kitchen
is a plan, not a design, and is left to v691 -- which is also what it does
when the designer finds nothing, so an order it cannot design is no worse
off than it was.

The scene's learner is the designer's: a step the world refuses, or that
a person says did not happen, teaches both. And what worked is remembered
for the conversation (`carrying.Remembered`), so the second time a thing
is wanted the way that worked is tried first.
"""
from __future__ import annotations

from research.v691 import page as v691_page
from research.v691.openworld import past
from research.v694 import asking, carrying, designing, teaching
from research.v694.goals import DOER, Goal

#: The ways that are the designer's to carry out. The rest -- doing it
#: oneself, taking a thing somewhere -- are the planner's.
TAKES = frozenset({"tool", "inside", "place", "wear", "bring",
                   "helper"})

#: Ways that worked, by conversation.
MEMORIES: dict = {}
#: The designs carried out, by conversation, newest last: what `why` is
#: answered from.
DONE: dict = {}

_PROPOSER: dict = {}


def _order():
    """The learned proposer's order (`proposer.py`), where it has been
    trained: which way to imagine first. Only how soon, never what."""
    if "one" not in _PROPOSER:
        from research.v694.proposer import Proposer
        _PROPOSER["one"] = Proposer.load()
    found = _PROPOSER["one"]
    return found.order if found is not None else None


def _memory(scene) -> carrying.Remembered:
    key = id(scene)
    if key not in MEMORIES:
        MEMORIES[key] = carrying.Remembered()
    return MEMORIES[key]


def _goal(scene, heard) -> Goal | None:
    """The order as a goal to design, or None when it is not the open
    world's or asks for nothing the designer designs."""
    if not scene.open or getattr(scene.domain, "things", None) is None:
        return None
    wants = [one for one in (heard.wants or ()) if len(one.split()) == 2]
    if not wants or len(wants) != len(heard.wants or ()):
        return None
    names = frozenset(getattr(scene.domain, "names", ()) or ())
    return Goal(tuple(wants), frozenset(scene.world.facts), names,
                said=heard.said, memory=_memory(scene))


def _designed(scene, heard):
    goal = _goal(scene, heard)
    if goal is None:
        return None, None
    found = designing.design(goal, learner=scene.learner,
                             order=_order())
    if not found.done or not any(part.best.form in TAKES
                                 for part in found.parts):
        return goal, None
    return goal, found


def _pasts(text: str) -> str:
    """`get a knife, then cut the rope` -> `got a knife, then cut the
    rope`: each step's verb in the past, as the planner narrates."""
    out = []
    for step in text.split(", then "):
        verb, _, rest = step.partition(" ")
        out.append(f"{past(verb)} {rest}".strip())
    return ", then ".join(out)


def _because(found) -> str:
    reasons = [part.best.why for part in found.parts
               if part.best is not None and part.best.why]
    return (" -- " + "; ".join(dict.fromkeys(reasons))) if reasons else ""


def _steps(found) -> str:
    return ", then ".join(part.best.text() for part in found.parts
                          if part.best is not None and part.best.steps)


def how(scene, heard) -> str:
    """What it would take, designed and not done."""
    _, found = _designed(scene, heard)
    if found is None:
        return ""
    return f"I would {_steps(found)}{_because(found)}."


def want(scene, heard) -> str:
    """Design a way and do it in the scene: the steps go into the world
    like the planner's do, and a refused one is learned from and the goal
    designed again (`carrying.carry_out`)."""
    goal, found = _designed(scene, heard)
    if found is None:
        return ""
    before = len(scene.world.did)
    start = frozenset(scene.world.facts)
    outcome = carrying.carry_out(goal, scene.world, scene.learner,
                                 goal.memory, order=_order())
    scene.wanted = list(goal.wants)
    scene.last_plan = list(scene.world.did[before:])
    scene.steps, facts = [], start
    for one in scene.last_plan:
        scene.steps.append((one, facts))
        facts = one.on(facts)
    scene.pending = None
    # What a design supposed is in the scene now: the knife it got.
    for fact in scene.world.facts - start:
        for name in fact.split()[1:]:
            if name not in scene.objects and name != DOER:
                scene.objects[name] = ""
                scene.told(name, "")
    agents = getattr(scene.domain, "agents", None)
    if agents is not None:
        agents.add(DOER)
    last = outcome.designs[-1] if outcome.designs else found
    if outcome.done:
        DONE.setdefault(id(scene), []).append(last)
    if not outcome.done:
        return ("I could not get that done: " + last.said()
                + (f" ({len(outcome.surprises)} step"
                   f"{'s' if len(outcome.surprises) != 1 else ''} did not "
                   f"work)" if outcome.surprises else ""))
    said = f"I {_pasts(_steps(last))}{_because(last)}."
    if outcome.surprises:
        step, lessons = outcome.surprises[-1]
        said = (f"That did not work at first ({step.name} was refused), so "
                f"I designed it again: " + said)
    return said


def why(scene, heard) -> str:
    """Why a thing was used, from the design that used it: *why did you
    get the knife* -- because the rope was to be cut with it, and a knife
    is what the store says cuts rope. Only about a thing a design used;
    anything else is the planner's to answer."""
    words = set(heard.said.lower().replace("?", " ").split())
    for design in reversed(DONE.get(id(scene), [])):
        for part in design.parts:
            best = part.best
            if best is None or not best.means:
                continue
            said = best.means.replace("-", " ")
            if not (set(said.split()) & words):
                continue
            use = best.said.get(best.steps[-1].name, "") if best.steps else ""
            because = f"to {use}" if use else f"for {part.clause.said()}"
            return (f"I used the {said} {because}"
                    + (f" -- {best.why}" if best.why else "") + ".")
    return ""


def _teaches(scene, text: str) -> bool:
    """Taught in general -- *you can*, *use X to*, *is made from* -- and
    so this turn's to answer. Something said as done is not: it is what
    happened, v689's to keep, and is learned from as it is noted
    (`observe`)."""
    if not _goal_world(scene):
        return False
    taught = teaching.read(text)
    return taught is not None and not taught.seen


def observe(scene, text: str) -> None:
    """A way or a recipe seen done -- *i cut the rope with a saw* --
    learned from quietly, as the page notes what was said."""
    if not _goal_world(scene):
        return
    taught = teaching.read(text)
    if taught is not None and taught.seen:
        teaching.learn(taught, _memory(scene))


def teach(scene, heard) -> str:
    """A way or a recipe, taught or seen done: kept for this conversation
    (`carrying.Remembered`) and offered first when a goal like it comes."""
    taught = teaching.read(heard.said)
    if taught is None:
        return ""
    return teaching.learn(taught, _memory(scene))


def _goal_world(scene) -> bool:
    return scene.open and getattr(scene.domain, "things", None) is not None


def _asks_recipe(scene, text: str) -> bool:
    """*Do you have a recipe for cake*, *how do you make a torch*: how a
    thing is made, asked (`asking.read`) -- and, when it is asked as how,
    only when there is something to say, so the rest stay v689's."""
    if not _goal_world(scene):
        return False
    asked = asking.read(text)
    return asked is not None and asking.knows(asked, _memory(scene))


def recipe(scene, heard) -> str:
    """What is known of making a thing: a recipe taught or seen, or only
    what the store says one can be made of."""
    asked = asking.read(heard.said)
    return asking.answer(asked, _memory(scene)) if asked else ""


v691_page.contributes("want", want)
v691_page.contributes("how", how)
v691_page.contributes("why", why)
v691_page.observes(observe)
v691_page.adds("teach", _teaches, teach,
               rule="a way or a recipe, taught or seen done: kept")
v691_page.adds("recipe", _asks_recipe, recipe, utility=v691_page.ASK,
               rule="how a thing is made: a recipe, or what it is made of")

"""The shapes a way can take: forms with holes, for the open world.

v693's forms were shapes of a mathematical object with unknowns where the
specification would decide -- *something times (x - 2)(x + 3)*. Here a
form is a shape of **a way of getting something done**, and its hole is
the thing the way uses, which the store decides:

    unaided   do it                       open the door
    tool      do it with ?T               cut the rope with a knife
    inside    do it with the thing in ?P  bake the bread in an oven
    place     put the thing in ?P         put the milk in the fridge
    bring     put ?M in the thing         put a heater in the room
    wear      give the one ?W             give john a blanket
    helper    have ?H do it               ask a mechanic to fix the car
    move      take the thing there        put the book in the kitchen

Each is general -- nothing in it names a knife, a fridge or a rope -- and
each says what it is built from (`needs`, over `goals.features`), so the
proposer and the learner can say when it is worth trying, as v693's did.

**How a hole is filled.** The store says what serves (`knowing.serving`);
the scene says what is at hand. A thing at hand that is one of what serves
-- a carving knife is a knife -- is used before anything is supposed; what
is not at hand is **supposed**, with where one is usually found (`found
in`), and getting it becomes a step. That is the composition: every tool,
place or thing to wear is itself a goal, *have one*, and this is where it
is met.

**When a thing may do it unaided** is the store's to say too: people open
doors, hang pictures and write letters (`capable_of`), and nothing says
they cut rope with their hands. And not where a tool or a place made for
doing it to this very thing is better attested than people doing it --
people cook eggs, in a pan -- unless a part of the thing does it: a door's
knob comes with the door (`_tool_for_it`).

**Built, as well as found.** A thing a recipe makes from what is at hand
(`_build`, `carrying.Recipe`) is obtained before anything is supposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from research.v691 import world as W
from research.v694 import knowing as K
from research.v694.goals import DOER, Clause, Goal, takes_instrument

#: How much the store must say for a thing to be offered for a hole: one
#: row about the goal's own thing (0.35 x 1), or better.
SUPPORT = 0.35

#: How many of the store's best are offered for one hole. A hole is a
#: search over things, and the first few the store stands behind are the
#: ones worth imagining.
OFFERED = 4


@dataclass
class Candidate:
    """One way to meet one clause: its steps, and what it takes for
    granted."""

    form: str
    clause: Clause
    #: the thing the way uses, as named in the steps; "" for none
    means: str = ""
    steps: list = field(default_factory=list)
    #: action name -> how it is said
    said: dict = field(default_factory=dict)
    #: (thing, where one is usually found) for each thing supposed
    supposed: list = field(default_factory=list)
    #: others who would have to do something
    helpers: list = field(default_factory=list)
    support: float = 0.0
    why: str = ""
    #: how exactly the store's rows fit the thing (`knowing.about`): 3 for
    #: the thing itself, 2 for a kind of it
    reach: float = 0.0
    #: (thing, the parts it was made from) for each thing made
    built: list = field(default_factory=list)

    def cost(self) -> tuple:
        """Occam, for ways: rely on nobody else, suppose nothing, take as
        few steps as possible -- then, among equals, do what was asked
        rather than something that comes to it (cleaning the floor, not
        laying a mat to keep it clean), what the store says of this very
        thing over what it says of its kind (a saucepan cooks rice; a grill
        cooks food), and the one the store stands behind most."""
        return (len(self.helpers), len(self.supposed), len(self.steps),
                self.form in INDIRECT, -min(self.reach, 3.0), -self.support)

    def text(self) -> str:
        return ", then ".join(self.said.get(one.name, one.name)
                              for one in self.steps)


#: Ways that bring a state about by something other than the doing asked
#: for: a place that keeps it, a thing brought or worn.
INDIRECT = frozenset({"place", "bring", "wear"})


@dataclass(frozen=True)
class Way:
    name: str
    #: the features it is built from (`goals.features`)
    needs: tuple
    #: (goal, clause, facts) -> candidates
    build: Callable
    said: str
    #: the least it could cost: helpers it must rely on
    helpers: int = 0


# -- the hole: what serves, and what is at hand ----------------------------

def _at_hand(goal: Goal, facts, means: str, avoid=()) -> str:
    """A thing in the scene that is one of `means`: the thing itself, or
    a kind of it (a carving knife is a knife)."""
    wanted = means.replace("-", " ")
    for thing in sorted(_things(facts) | goal.things()):
        if thing in avoid or thing in goal.names or thing == DOER:
            continue
        if thing == means or wanted in K.kinds_of(thing):
            return thing
    return ""


def _things(facts) -> set:
    out = set()
    for fact in facts:
        out.update(fact.split()[1:])
    return out


def _where(facts, thing: str) -> str:
    for fact in facts:
        parts = fact.split()
        if len(parts) == 3 and parts[0] == "at" and parts[1] == thing:
            return parts[2]
    return ""


def _held(facts, thing: str, who: str = DOER) -> bool:
    return f"with {thing} {who}" in facts


def the(name: str) -> str:
    return "the " + name.replace("-", " ")


def a(name: str) -> str:
    said = name.replace("-", " ")
    if said.startswith("other "):
        return "an" + said
    return ("an " if said[:1] in "aeiou" else "a ") + said


OTHER = "other-"


def fresh(means: str, facts) -> str:
    """A name for one of `means` that is not the one in the scene: what is
    true of the broken fridge is not true of another fridge."""
    if means not in _things(facts):
        return means
    return OTHER + means


def obtain(goal: Goal, facts, means: str, candidate: Candidate,
           avoid=(), another: bool = False) -> str:
    """Steps that end with the doer holding one of `means`, added to the
    candidate; the name of the one it will be. What is held already costs
    nothing, what is at hand is taken, and what is not is supposed -- got
    from where one usually is. `another` supposes a new one even where one
    is at hand: the one at hand may be the one that did not work."""
    thing = "" if another else _at_hand(goal, facts, means, avoid)
    thing = thing or (fresh(means, facts) if another else means)
    if thing.startswith(OTHER):
        means = thing
    if _held(facts, thing):
        return thing
    if thing != means or thing in _things(facts):
        where = _where(facts, thing)
        if where:
            step = W.Action(f"take {thing} {where}",
                            frozenset({f"at {thing} {where}"}),
                            frozenset({f"with {thing} {DOER}"}),
                            frozenset({f"at {thing} {where}"}))
            candidate.said[step.name] = (f"take {the(thing)} from "
                                         f"{the(where)}")
        else:
            step = W.Action(f"take {thing}", frozenset(),
                            frozenset({f"with {thing} {DOER}"}), frozenset())
            candidate.said[step.name] = f"take {the(thing)}"
        candidate.steps.append(step)
        return thing
    if not another:
        built = _build(goal, facts, means, candidate, avoid)
        if built:
            return built
    where = K.found_in(thing[len(OTHER):] if thing.startswith(OTHER)
                       else thing)
    step = W.Action(f"get {thing}", frozenset(),
                    frozenset({f"with {thing} {DOER}"}), frozenset())
    candidate.said[step.name] = (
        f"get {a(thing)}" + (f" (there is usually one in {a(where)})"
                             if where else ""))
    candidate.steps.append(step)
    candidate.supposed.append((thing, where))
    return thing


#: How alike a thing at hand must be to a recipe's part to stand in for it:
#: a close sibling will do (a rag for a cloth), and anything less will not.
STANDS_IN = 0.5


def _recipe(goal: Goal, facts, means: str, avoid=()):
    """(recipe, the things at hand it would use) for making one of
    `means`, or None."""
    memory = getattr(goal, "memory", None)
    if memory is None:
        return None
    things = [one for one in sorted(_things(facts) | goal.things())
              if one not in avoid and one not in goal.names and one != DOER]
    for recipe in memory.recipes_for(means):
        used: list = []
        for part in recipe.parts:
            ranked = sorted(((K.similar(part, one), one) for one in things
                             if one not in used), reverse=True)
            if not ranked or ranked[0][0] < STANDS_IN:
                break
            used.append(ranked[0][1])
        else:
            return recipe, used
    return None


def _build(goal: Goal, facts, means: str, candidate: Candidate,
           avoid=()) -> str:
    """Make one of `means` from what is at hand, by a recipe that was
    taught or seen (`carrying.Recipe`), before supposing one. Each part is
    matched by a thing at hand that is one, or like one (`knowing.similar`)
    -- a recipe learned with a stick and a cloth works with a branch and a
    rag -- and is used up. Nothing is supposed, so Occam prefers it to
    fetching one from somewhere."""
    found = _recipe(goal, facts, means, avoid)
    if found is None:
        return ""
    recipe, used = found
    for one in used:
        if _held(facts, one):
            continue
        where = _where(facts, one)
        placed = frozenset({f"at {one} {where}"} if where else ())
        step = W.Action(f"take {one} {where}".strip(), placed,
                        frozenset({f"with {one} {DOER}"}), placed)
        candidate.steps.append(step)
        candidate.said[step.name] = (f"take {the(one)}" + (
            f" from {the(where)}" if where else ""))
    # What is made is what the recipe makes -- a torch -- wherever it
    # stands in for what the store named: a light.
    product = fresh(K.name_of(recipe.product), facts)
    held = frozenset(f"with {one} {DOER}" for one in used)
    step = W.Action(f"make {product} " + " ".join(used), held,
                    frozenset({f"with {product} {DOER}"}), held)
    candidate.steps.append(step)
    candidate.said[step.name] = (
        f"make {a(product)} from " + " and ".join(the(one) for one in used)
        + (f" ({recipe.said})" if recipe.said else ""))
    candidate.built.append((product, tuple(used)))
    if K.name_of(recipe.product) != means:
        candidate.why = (f"{a(K.name_of(recipe.product))} is "
                         f"{a(means)}, and I know how to make one")
    return product


#: What a way that worked before is worth, as evidence: more than any one
#: row -- it was seen to work.
REMEMBERED = 2.0

#: How much less the store may stand behind a thing than behind its best
#: for the same hole, and the thing still be offered.
WEAKER = 4


def _remembered(goal: Goal, clause: Clause, form: str, found: list) -> list:
    """The store's candidates, with what worked before for a clause like
    this one put first -- added where the store never heard of it, which is
    how a way somebody taught gets into a design."""
    memory = getattr(goal, "memory", None)
    if memory is None:
        return found
    kind = goal.kind_of(clause.patient)
    first = []
    for way, alike in memory.ways_for(clause, kind):
        if way.form != form:
            continue
        have = next((one for one in found if one.name == way.means), None)
        if have is None:
            have = K.Means(way.means.replace("-", " "), fits=True,
                           reach=2.0, rows=[])
        if have in first:
            continue
        have.score = max(have.score, REMEMBERED * way.worked * alike)
        exact = alike >= 1.0 and way.predicate == clause.predicate
        # What worked for this very goal is about this very thing, as a
        # row naming it is; what worked for a goal like it, less so.
        have.reach = max(have.reach, 3.0 if exact else
                         2.0 if alike >= 1.0 else K.SIBLING)
        have.remembered = way.said or (
            "it worked before" if exact else
            f"it worked to {way.predicate} {K.name_of(way.kind)
                                            .replace('-', ' ')}, which is "
            f"like this")
        first.append(have)
    return first + [one for one in found if one not in first]


def _offered(goal: Goal, facts, found: list, patient: str = "") -> list:
    """The store's candidates for a hole, what is at hand first: a thing
    in the scene that serves is offered however far down the store's list
    it is, then the store's best few.

    Only what the store says is about *this* -- or about anything at all
    -- counts: a tractor pulls heavy objects, and nothing says it pulls
    teeth. And never the thing itself or a kind of it: a painting is not
    what one hangs a picture with."""
    kind = goal.kind_of(patient) if patient else ""
    # A category is not a thing to get: *a kitchen utensil* is dozens of
    # things, and the one that opens cans is among the store's own.
    good = [one for one in found if one.score >= SUPPORT
            and one.reach >= 1 and not (kind and _related(one.name, kind))
            and not K.category(one.name)]
    # The specific over the general, where the store stands behind it
    # nearly as well: a can opener is a kitchen utensil, and what is said
    # of kitchen utensils opening cans is said of it. A pocketknife is a
    # knife too, and far less is said of it cutting rope.
    # Not what is at hand or can be made from it, though: the torch one
    # can make is not less of a way because flashlights are torches too.
    def ready(one) -> bool:
        return bool(_at_hand(goal, facts, one.name)
                    or _recipe(goal, facts, one.name, {patient}))

    good = [one for one in good if ready(one) or not any(
        other is not one and other.score >= one.score / 2
        and one.name.replace("-", " ") in K.kinds_of(other.name)
        for other in good)]
    # Occam chooses among ways about as good as each other, not between a
    # shovel and the knife in one's hand because a knife was once said to
    # dig: what the store stands behind far less than its best is not
    # offered at all.
    best = max((one.score for one in good), default=0.0)
    good = [one for one in good if one.score * WEAKER >= best]
    # At hand, or makeable from what is: either way nothing is supposed.
    here = [one for one in good if ready(one)]
    rest = [one for one in good if one not in here][:OFFERED]
    return here + rest


def _why(means) -> str:
    remembered = getattr(means, "remembered", "")
    return remembered or means.why()


def _twice(goal: Goal, facts, offered: list, avoid) -> list:
    """(means, another) for each thing offered: the one at hand, and --
    where there is one at hand -- another like it, supposed. The second
    costs more and is only chosen when the first does not check: the fridge
    in the kitchen is broken, and a fridge is still the way."""
    out = []
    for means in offered:
        out.append((means, False))
        if _at_hand(goal, facts, means.name, avoid):
            out.append((means, True))
    return out


def _related(one: str, other: str) -> bool:
    """Whether one thing is a kind of the other, either way round."""
    a_, b_ = one.replace("-", " "), other.replace("-", " ")
    return a_ == b_ or a_ in K.kinds_of(other) or b_ in K.kinds_of(one)


def tool_like(word: str) -> bool:
    """What may be a tool: a made thing one can carry -- not a living
    one, not a room or a wall."""
    if not K.an_artifact(word) or K.lives(word) or fixed(word):
        return False
    # Somewhere things are kept -- a hallway, a car -- is where a picture
    # hangs or a family travels, not what one does it with. A container
    # one carries is still a tool: a bucket, a bag.
    return K.a_place(word) < PLACE or K.is_a(word, "container.n.01")


#: How many things the store must say are found at a thing for it to be a
#: place rather than something to use.
PLACE = 10


def helper_like(word: str) -> bool:
    """Who may be asked: someone -- a person by trade, not people at
    large, who are whoever is asking."""
    return K.is_a(word, "person.n.01") and word not in ("person", "people",
                                                        "human", "man",
                                                        "woman", "child")


# -- the forms -------------------------------------------------------------

def _unaided(goal: Goal, clause: Clause, facts) -> list:
    out = []
    kind = goal.kind_of(clause.patient)
    for verb in clause.verbs():
        support = K.person_can(verb, kind)
        if support < SUPPORT / 2:
            # Nothing says a person does this with nothing but their
            # hands. VerbNet will not say it either way -- it has `cut`
            # with and without an Instrument, as English does.
            continue
        if _tool_for_it(goal, clause, verb, support):
            # *People cook eggs* is in the store, and so is *a frying pan
            # is used for cooking eggs*: that people do it says they can,
            # not that they do it bare-handed. Where the store knows a
            # tool for this very thing, better attested than the person
            # doing it, the doing is the tool's -- unless the tool is part
            # of the thing, as a door's knob is, and comes with it.
            continue
        one = Candidate("unaided", clause, support=support, reach=2.0)
        step = W.Action(f"{verb} {clause.patient}", frozenset(),
                        frozenset({clause.literal}), frozenset())
        one.steps.append(step)
        one.said[step.name] = f"{verb} {goal_name(goal, clause.patient)}"
        one.why = f"people {verb} {kind}s themselves"
        out.append(one)
    return out


def _tool_for_it(goal: Goal, clause: Clause, verb: str,
                 support: float) -> bool:
    """Whether doing `verb` to this thing takes a tool: the language names
    one (`iron`), or the store knows one for this very thing better than
    it knows people doing it -- unless one of those is part of the thing
    and comes with it, as a door's knob does."""
    kind = goal.kind_of(clause.patient)
    parts = K.has_parts(kind)

    def part(word: str) -> bool:
        # One of its parts, or a kind of one: a doorknob is a knob.
        name = K.name_of(word)
        return name in parts or any(
            name.replace("-", " ") in K.kinds_of(one) for one in parts)

    if parts and any(one.reach >= 1 for one in K.serving(
            verb=verb, patient=kind, keep=part)):
        # The thing has a part that does it: a door's knob opens it.
        return False
    # The tools the tool form would offer -- no others: a tool it would
    # not use is no reason to refuse doing it by hand.
    named = set(K.named_tools(verb))
    if any(one.name in named or (one.reach >= 2 and one.score > support)
           for one in tools(goal, clause, verb)[:OFFERED]):
        return True
    # Or a place made for doing it to this very thing: people bake bread,
    # and they bake it in an oven.
    return any(one.reach >= 2 and one.score > support
               for one in places_for(goal, clause, verb)[:OFFERED])


def places_for(goal: Goal, clause: Clause, verb: str) -> list:
    """Places made for doing `verb` to this thing, best first: an oven for
    baking bread. Somewhere a thing is put, not a tool one holds."""
    kind = goal.kind_of(clause.patient)
    # Something a thing is put *into* -- an oven, a sink -- not a room:
    # cooking is done in a kitchen, and pasta is not put in one to cook.
    found = [one for one in K.serving(
        verb=verb, patient=kind, keep=lambda word: _placeable(word)
        and not tool_like(word) and not fixed(word), weights=K.AS_TOOL)
        if one.reach >= 2 or K.kept_at(kind, one.name)]
    return [one for one in found if one.score >= SUPPORT
            and not K.category(one.name)]


def tools(goal: Goal, clause: Clause, verb: str) -> list:
    """The tools that may do `verb` to this thing, best first: what the
    store says serves, and what the language names (`_named`).

    Where VerbNet does not make an instrument part of what the verb means
    -- *fix*, *unlock*, *write* -- a tool is offered only where the store
    says so of this very thing: a key is used for unlocking doors; duct
    tape fixing *everything* is not a way to fix a car."""
    kind = goal.kind_of(clause.patient)
    named = set(K.named_tools(verb))
    found = _named(verb, kind, K.serving(verb=verb, patient=kind,
                                         keep=tool_like, weights=K.AS_TOOL))
    if not takes_instrument(verb):
        found = [one for one in found if one.name in named or any(
            row.relation == "used_for"
            and K.about(row, kind, K.lemmas(verb)) >= 2
            for row in one.rows)]
    return [one for one in found if one.score >= SUPPORT and
            one.reach >= 1 and not _related(one.name, kind)
            and not K.category(one.name)]


def _named(verb: str, kind: str, found: list) -> list:
    """The store's candidates, and the tools the language names for the
    verb (`K.named_tools`) where the store has nothing to say of them: a
    verb that is the name of its tool is evidence as good as a row."""
    have = {one.name for one in found}
    out = list(found)
    for name in K.named_tools(verb):
        if name in have or _related(name, kind):
            continue
        out.append(K.Means(name.replace("-", " "), score=NAMED, fits=True,
                           reach=1.0, rows=[]))
    return out


#: What a tool the language names is worth, as evidence: a row about the
#: goal's own thing.
NAMED = 0.7


def _tool(goal: Goal, clause: Clause, facts) -> list:
    out = []
    kind = goal.kind_of(clause.patient)
    for verb in clause.verbs():
        for means, another in _twice(goal, facts, _offered(
                goal, facts, _remembered(goal, clause, "tool",
                                         tools(goal, clause, verb)),
                clause.patient), {clause.patient}):
            one = Candidate("tool", clause, support=means.score,
                        reach=means.reach,
                            why=_why(means))
            used = obtain(goal, facts, means.name, one,
                          avoid={clause.patient}, another=another)
            one.means = used
            step = W.Action(f"{verb} {clause.patient} {used}",
                            frozenset({f"with {used} {DOER}"}),
                            frozenset({clause.literal}), frozenset())
            one.steps.append(step)
            one.said[step.name] = (f"{verb} "
                                   f"{goal_name(goal, clause.patient)} "
                                   f"with {the(used)}")
            out.append(one)
    return out


def _placeable(word: str) -> bool:
    """Somewhere a thing can be put: a made thing -- a fridge, a room, an
    oven -- not a region or the weather."""
    return (K.an_artifact(word) or K.is_a(word, "room.n.01",
                                           "building.n.01")) and \
        not K.lives(word)


def fixed(word: str) -> bool:
    """A thing that stays where it is: a room, a house, a field -- or a
    part of one, a floor or a wall. A place form cannot put one
    somewhere."""
    return K.is_a(word, "location.n.01", "structure.n.01", "area.n.05",
                  "room.n.01", "geological_formation.n.01") or \
        K.part_of_something_fixed(word)


def _place(goal: Goal, clause: Clause, facts) -> list:
    out = []
    patient = clause.patient
    kind = goal.kind_of(patient)
    if fixed(kind):
        return out
    was = _where(facts, patient)
    # A place *for* keeping things so -- `used for keeping food cold` --
    # and not merely one that is so: a house is warm, and soup is not
    # warmed by being put in one. What it is adds to the evidence.
    found = [one for one in K.serving(
        state=clause.predicate, patient=kind,
        relations=K.SERVING + ("has_property",), keep=_placeable,
        weights=K.AS_TOOL)
        if K.kept_at(kind, one.name) and any(
            row.relation != "has_property" for row in one.rows)]
    found = _remembered(goal, clause, "place", found)
    for means, another in _twice(goal, facts, _offered(
            goal, facts, found, patient), {patient}):
        one = Candidate("place", clause, support=means.score,
                        reach=means.reach,
                        why=_why(means))
        place = "" if another else _at_hand(goal, facts, means.name,
                                            avoid={patient})
        if place and place == was:
            # It is there already, and it is not so: whatever the store
            # says of places like it, this one has not done it.
            continue
        needs = frozenset()
        if not place:
            # Supposed, as a tool is: one has to be found before anything
            # can be put in it, and finding it is a step like getting a
            # knife is.
            place = fresh(means.name, facts) if another else means.name
            where = K.found_in(means.name)
            one.supposed.append((place, where))
            find = W.Action(f"find {place}", frozenset(),
                            frozenset({f"found {place}"}), frozenset())
            one.steps.append(find)
            one.said[find.name] = (
                f"find {a(place)}" + (f" (there is usually one in "
                                      f"{a(where)})" if where else ""))
            needs = frozenset({f"found {place}"})
        one.means = place
        step = W.Action(f"put {patient} {place}", needs,
                        frozenset({f"at {patient} {place}", clause.literal}),
                        frozenset({f"at {patient} {was}"} if was else ()))
        one.steps.append(step)
        one.said[step.name] = (
            (f"take {goal_name(goal, patient)} to " if goal.lives(patient)
             else f"put {the(patient)} in ") + the(place))
        out.append(one)
    return out


def _wear(goal: Goal, clause: Clause, facts) -> list:
    out = []
    patient = clause.patient
    kind = goal.kind_of(patient)
    found = [one for one in K.serving(state=clause.predicate, patient=kind,
                                      keep=tool_like, weights=K.AS_TOOL)
             if not K.a_place(one.name) or one.fits]
    found = _remembered(goal, clause, "wear", found)
    for means, another in _twice(goal, facts, _offered(
            goal, facts, found, patient), {patient}):
        one = Candidate("wear", clause, support=means.score,
                        reach=means.reach,
                        why=_why(means))
        used = obtain(goal, facts, means.name, one, avoid={patient},
                      another=another)
        one.means = used
        step = W.Action(f"give {used} {patient}",
                        frozenset({f"with {used} {DOER}"}),
                        frozenset({f"with {used} {patient}",
                                   clause.literal}),
                        frozenset({f"with {used} {DOER}"}))
        one.steps.append(step)
        one.said[step.name] = f"give {goal_name(goal, patient)} {the(used)}"
        out.append(one)
    return out


def goal_name(goal: Goal, name: str) -> str:
    return name if name in goal.names else the(name)


def _helper(goal: Goal, clause: Clause, facts) -> list:
    out = []
    kind = goal.kind_of(clause.patient)
    for verb in clause.verbs():
        for means in _offered(goal, facts, _remembered(
                goal, clause, "helper", K.serving(
                    verb=verb, patient=kind, relations=("capable_of",),
                    keep=helper_like)), clause.patient):
            if means.reach < K.SIBLING:
                # Someone who does this to other things: a barber cuts
                # hair, and is not who to ask about a rope. A plumber who
                # fixes toilets is who to ask about a sink.
                continue
            one = Candidate("helper", clause, means=means.name,
                            support=means.score, why=_why(means),
                            reach=means.reach)
            here = _at_hand(goal, facts, means.name)
            if here:
                one.means = here
            else:
                one.helpers.append(means.name)
            step = W.Action(f"ask {one.means} {verb} {clause.patient}",
                            frozenset(), frozenset({clause.literal}),
                            frozenset())
            one.steps.append(step)
            one.said[step.name] = (
                f"ask {the(one.means) if here else a(one.means)} to "
                f"{verb} {the(clause.patient)}")
            out.append(one)
    return out


def _inside(goal: Goal, clause: Clause, facts) -> list:
    """A doing done with the thing *in* somewhere: bread is baked in an
    oven, clothes washed in a washing machine. The place is no tool -- it
    stays where it is (`tool_like`) -- and the thing goes to it, as in the
    place form; then the doing is done there."""
    out = []
    patient = clause.patient
    kind = goal.kind_of(patient)
    if fixed(kind) or goal.lives(patient):
        # A room is not put anywhere; and a person is taken somewhere,
        # which is the place form's to say, not put in a fireplace.
        return out
    was = _where(facts, patient)
    for verb in clause.verbs():
        found = _remembered(goal, clause, "inside",
                            places_for(goal, clause, verb))
        for means, another in _twice(goal, facts, _offered(
                goal, facts, found, patient), {patient}):
            one = Candidate("inside", clause, support=means.score,
                        reach=means.reach,
                            why=_why(means))
            place = "" if another else _at_hand(goal, facts, means.name,
                                                avoid={patient})
            needs = frozenset()
            if not place:
                place = fresh(means.name, facts) if another else means.name
                where = K.found_in(means.name)
                one.supposed.append((place, where))
                find = W.Action(f"find {place}", frozenset(),
                                frozenset({f"found {place}"}), frozenset())
                one.steps.append(find)
                one.said[find.name] = (
                    f"find {a(place)}" + (f" (there is usually one in "
                                          f"{a(where)})" if where else ""))
                needs = frozenset({f"found {place}"})
            one.means = place
            # One step, as using a tool is one: putting the bread in the
            # oven and baking it is baking it in the oven.
            do = W.Action(f"{verb} {patient} {place}", needs,
                          frozenset({f"at {patient} {place}",
                                     clause.literal}),
                          frozenset({f"at {patient} {was}"} if was else ()))
            one.steps.append(do)
            one.said[do.name] = (f"{verb} {the(patient)} in "
                                 f"{the(place)}")
            out.append(one)
    return out


def _bring(goal: Goal, clause: Clause, facts) -> list:
    """A state of a thing that stays where it is -- a room to be bright,
    or warm -- is brought about by bringing to it something that makes it
    so: a lamp, a heater. The place form the other way round."""
    out = []
    patient = clause.patient
    kind = goal.kind_of(patient)
    found = [one for one in K.serving(state=clause.predicate, patient=kind,
                                      keep=tool_like, weights=K.AS_TOOL)
             if one.fits and any(row.relation != "has_property"
                                 for row in one.rows)]
    found = _remembered(goal, clause, "bring", found)
    for means, another in _twice(goal, facts, _offered(
            goal, facts, found, patient), {patient}):
        one = Candidate("bring", clause, support=means.score,
                        reach=means.reach,
                        why=_why(means))
        used = obtain(goal, facts, means.name, one, avoid={patient},
                      another=another)
        one.means = used
        step = W.Action(f"install {used} {patient}",
                        frozenset({f"with {used} {DOER}"}),
                        frozenset({f"at {used} {patient}", clause.literal}),
                        frozenset({f"with {used} {DOER}"}))
        one.steps.append(step)
        one.said[step.name] = f"put {the(used)} in {the(patient)}"
        out.append(one)
    return out


def _move(goal: Goal, clause: Clause, facts) -> list:
    """`at book kitchen`, `with key john`: take the thing there."""
    thing, to = clause.patient, clause.other
    one = Candidate("hand" if clause.kind == "having" else "move", clause,
                    support=SUPPORT)
    if clause.kind == "having":
        used = obtain(goal, facts, thing, one, avoid={to})
        step = W.Action(f"give {used} {to}",
                        frozenset({f"with {used} {DOER}"}),
                        frozenset({f"with {used} {to}"}),
                        frozenset({f"with {used} {DOER}"}))
        one.said[step.name] = f"give {the(used)} to {goal_name(goal, to)}"
    else:
        was = _where(facts, thing)
        if goal.lives(thing):
            step = W.Action(f"go {thing} {to}", frozenset(),
                            frozenset({f"at {thing} {to}"}),
                            frozenset({f"at {thing} {was}"} if was else ()))
            one.said[step.name] = (f"take {goal_name(goal, thing)} to "
                                   f"{the(to)}")
        else:
            used = obtain(goal, facts, thing, one)
            step = W.Action(f"put {used} {to}",
                            frozenset({f"with {used} {DOER}"}),
                            frozenset({f"at {used} {to}"}),
                            frozenset({f"with {used} {DOER}"}))
            one.said[step.name] = f"put {the(used)} in {the(to)}"
    one.steps.append(step)
    one.why = "a thing goes where it is taken"
    return [one]


WAYS = (
    Way("unaided", ("has-verb",), _unaided, "doing it oneself"),
    Way("tool", ("has-verb",), _tool, "doing it with a tool"),
    Way("inside", ("has-verb",), _inside, "doing it with the thing in "
                                          "somewhere made for it"),
    Way("place", ("kind-state",), _place,
        "putting it somewhere that keeps it so"),
    Way("bring", ("kind-state", "patient-fixed"), _bring,
        "bringing it something that makes it so"),
    Way("wear", ("kind-state", "patient-lives"), _wear,
        "giving them something that keeps them so"),
    Way("helper", ("has-verb",), _helper, "asking someone who does it",
        helpers=1),
    Way("move", ("kind-place",), _move, "taking it there"),
    Way("hand", ("kind-having",), _move, "handing it over"),
)

BY_NAME = {way.name: way for way in WAYS}


def usable(features) -> list:
    return [way for way in WAYS if set(way.needs) <= set(features)]

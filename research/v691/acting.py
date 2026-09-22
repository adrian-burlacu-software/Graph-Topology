"""An agent that plans and then acts.

`DESIGN.md` (v690) §8c closed the executive as a *reasoner*. Three things
were missing before it could be an *agent*, and all three are here,
deliberately with no reader anywhere near them (`talking.py` is where one
gets added, and it is kept thin on purpose):

    a world           facts that change only by acting, and are not undone
                      by giving up on a goal (`world.World`)
    planning          the executive's own means-ends analysis, run over a
                      model of the world rather than over what it knows
    execution         one action at a time against the real world, looking
                      after each -- and a **surprise is an impasse**, so
                      what to do about it is a substate and not a branch

## Working memory is nearly a world, and the difference is one method

The striking thing is how little had to be written. An `Action`'s
preconditions are an `Operator`'s `needs`, its adds are that operator's
`gives`, and `Executive` then plans over actions with no translation at all:
`_means_ends` picking the most useful waiting operator and pushing a subgoal
for the slots it lacks **is** goal-stack planning, which is what a STRIPS
planner of the period did. One thing was added to `executive.py` and it is a
pure read -- `pursuing()`, so an operator can decline to undo what a goal
beneath it has got (see `operator_of`).

Two things did have to be added here, and both are the same point from
different sides -- *a world is not a belief*:

**Deletes.** `gives` grows working memory and nothing in the executive ever
shrinks it, because knowing something does not stop you knowing something
else. Acting does. `Situation.apply` retracts an action's `deletes`, and
`Executive.plan` -- pure regression over needs and gives -- cannot express
them at all. It is measured here anyway (`by_regression`) precisely so the
gap is a number: it finds a "plan" for nearly everything and most of them
cannot be executed past the first step or two.

**Frames.** `Working` scopes a subgoal so that one which fails leaves
nothing behind -- exactly right for belief, and wrong for a world: a subgoal
that unstacked a block and then failed has still unstacked it. `Situation`
overrides that in the one place it differs (`retract` reaches into every
frame beneath, and the facts are one set the whole stack shares). The rest
of `Working` is untouched and still does its job, which is the honest
version of the irreversibility problem at this scale. It is honest here
only because blocks can be put back; §2 of `DESIGN.md` is about where it
stops being.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass, field

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Chunks,
                                     Executive, Operator, Subgoal, Working,
                                     episode, pursuing)
from research.v691 import quantities as Q, world as W
from research.v691.problems import SAMPLERS, SUITE

#: How the ground actions are ordered before the executive ever sees them,
#: as utilities. Means-ends takes the most useful waiting operator and the
#: most useful way to what it lacks, so this ordering is the whole of the
#: planner's taste -- there is no evaluation function and no backtracking
#: across the choice.
#:
#: **Not one of them names a predicate, an action or a kind of thing.** The
#: first version did: it preferred an action that frees *a block the goal
#: buries* and penalised *stacking higher in the goal tower*, and because
#: every number had been measured on blocks, "the executive can plan" was a
#: claim about the one domain it had been fitted to. These three say the
#: same things about any domain, in terms only of what an action needs,
#: adds and deletes:
#:
#:     a goal fact that others wait on
#:         achieve what other goal facts are waiting on first -- the general
#:         form of *build from the bottom*
#:     undoing what a goal's achiever needs
#:         do not take away something a goal still needs, the more so the
#:         deeper that goal is -- the general form of *do not stack onto a
#:         block that has to move*
#:     something an achiever will need
#:         produce what a goal's achiever is going to want -- the general
#:         form of *clear the block the goal has to sit on*
#:
#: Measured on `problems.SUITE` plus 20 sampled four-block problems, and
#: held out on 97 sampled problems over three unseen seeds (solved,
#: shortest):
#:
#:     as shipped, general               76   67    train 23/24
#:     the blocks-specific version       78   68    train 21/24
#:     the first signal alone            70   60
#:     none (the domain's own order)     58   49
#:     as shipped, without protection    60   51
#:
#: **Generality costs two problems of the 97**, and that is the honest
#: price: the fitted signals knew `clear` was special, and these only know
#: that something is a precondition. What they buy is that the same numbers
#: can be asked of `errands` and `delivery` -- which the fitted ones could
#: not have been asked of at all.
#:
#: Four other signals were written first and measured out, `achieves a goal
#: fact` -- the most obvious of them -- worth nothing at all. What decides a
#: problem is the structure of the goal, not which action helps now.
#: Magnitudes do not matter, only the rank: nine pairs of coefficients in a
#: sweep gave identical results, because means-ends reads an order.
SIGNALS = {"a goal fact that others wait on": -0.5,
           "undoing what a goal's achiever needs": -0.5,
           "something an achiever will need": 0.25,
           "an action that assumes nothing": -1.0}

#: Whether an action may throw away what an open subgoal has achieved: see
#: `operator_of`. Turned off only by `--ablate`, which is where the last
#: line of the table in `SIGNALS` comes from.
PROTECT = True

#: A model run that has applied this many actions has not found a plan; it
#: has wandered. Blocks problems here are solved in at most ten.
BUDGET = 60
#: And one that has pushed this many subgoals has not found one either.
#: Applying is bounded by `BUDGET`; wanting is not, and counts made it
#: unbounded -- to give John 6 Mary needs 6, to have 6 she needs someone's
#: 3, and every holder at every amount is another way to each.
SUBGOALS = 3000


def achievers(goal, actions) -> dict:
    """goal fact -> the actions that would bring it about."""
    out = {}
    for fact in goal:
        out[fact] = [one for one in actions if fact in one.adds]
    return out


def wanted(goal, actions) -> frozenset:
    """Every precondition of every achiever of a goal fact, minus the goal
    facts themselves: what the plan is going to need on the way."""
    needs = set()
    for ways in achievers(goal, actions).values():
        for one in ways:
            needs |= set(one.needs)
    return frozenset(needs) - frozenset(goal)


def depths(goal, actions) -> dict:
    """goal fact -> how many other goal facts have to come first.

    The general form of *build the tower from the bottom*, and it
    generalises because the tower was never the point. One goal fact
    **enables** another when something achieving the first produces
    something the second's achiever needs: `stack c d` leaves `clear c`,
    which `stack b c` wants, so `on c d` comes before `on b c`. Nothing in
    that mentions a block, and in `errands` the same relation puts being at
    the shop before having what is kept there.

    Depth is the longest chain of enablings ending at a fact. Cycles are
    ignored rather than resolved: two goal facts that enable each other have
    no order between them, and saying so beats inventing one.
    """
    goal = frozenset(goal)
    by_goal = achievers(goal, actions)
    gives = {fact: frozenset().union(*[one.adds for one in ways]) if ways
             else frozenset() for fact, ways in by_goal.items()}
    takes = {fact: frozenset().union(*[one.needs for one in ways]) if ways
             else frozenset() for fact, ways in by_goal.items()}
    before = {fact: {other for other in goal
                     if other != fact and gives[other] & takes[fact]}
              for fact in goal}
    depth = {}

    def of(fact, seen=()):
        if fact in depth:
            return depth[fact]
        if fact in seen:
            return 0
        depth[fact] = max((1 + of(one, seen + (fact,))
                           for one in before[fact]), default=0)
        return depth[fact]

    for fact in goal:
        of(fact)
    return depth


def relied_on(goal, actions, deep: dict) -> dict:
    """fact -> the depth of the deepest goal fact whose achiever needs it.

    The general form of *do not stack onto a block that has to move*. In
    blocks it is reached through `clear y`: putting anything on `y` takes
    that away, and something the goal wants on `y` needs it back. The rule
    never mentions `clear` -- only that a goal's achiever has a precondition
    and this action would remove it.
    """
    out = {}
    for fact, ways in achievers(goal, actions).items():
        for one in ways:
            for need in one.needs:
                out[need] = max(out.get(need, 0), deep.get(fact, 0))
    return out


@dataclass(frozen=True)
class Taste:
    """Everything the utilities are read off, worked out once per problem."""

    goal: frozenset = frozenset()
    deep: dict = field(default_factory=dict)
    relied: dict = field(default_factory=dict)
    needed: frozenset = frozenset()

    @classmethod
    def of(cls, goal, actions) -> "Taste":
        goal = frozenset(goal)
        deep = depths(goal, actions)
        return cls(goal, deep, relied_on(goal, actions, deep),
                   wanted(goal, actions))


def utility_of(action: W.Action, taste: Taste) -> float:
    """Where this action sits in the order the executive will try things.

    Three signals, none of which knows what domain it is in: see `SIGNALS`.
    """
    score = 1.0
    mine = [taste.deep.get(one, 0) for one in action.adds
            if one in taste.goal]
    if mine:
        score += SIGNALS["a goal fact that others wait on"] * max(mine)
    undone = [taste.relied[one] for one in action.deletes
              if one in taste.relied]
    if undone:
        score += (SIGNALS["undoing what a goal's achiever needs"]
                  * max(undone))
    if action.adds & taste.needed:
        score += SIGNALS["something an achiever will need"]
    if not action.needs:
        # An operator that applies in every state decides nothing. It is
        # kept, because sometimes it is all there is, and tried last.
        # `domains.py`'s worlds have no such action; `verbs.py` has many,
        # because VerbNet writes one class's meaning on frames of differing
        # completeness and the least complete says only where things end up.
        score += SIGNALS["an action that assumes nothing"]
    return score


# -- the model the planner searches in -------------------------------------

class Situation(Working):
    """Working memory whose facts are a world: shared by every goal frame,
    and taken away as well as added.

    A fact is a slot, so an `Operator`'s `needs` are preconditions and its
    `gives` are adds with nothing in between. What `Working` does not do is
    let a slot stop being true, or let a subgoal's change outlive the
    subgoal -- and a world does both. So the facts live in one set the whole
    goal stack shares; they are mirrored into the top frame as they are
    asserted, because the executive's `until` and its check on `gives` both
    read the top frame directly and must see what was just achieved.
    """

    def __init__(self, facts=(), goal: str = "", conditions=()) -> None:
        super().__init__(goal=goal)
        self.facts = set(facts)
        #: the actions applied, in order: the plan, as it is being found
        self.did: list = []
        #: what the goal and the actions ask of counts -- `with>=5 apple
        #: mary` -- kept as slots that are there exactly when they hold, so
        #: means-ends can want one like any other fact (`refresh`)
        self.conditions = frozenset(conditions)
        #: subgoals pushed so far: what `spent` weighs against `SUBGOALS`
        self.pushed = 0
        for fact in self.facts:
            dict.__setitem__(self, fact, True)
        self.refresh()

    def push(self, goal: str, **slots) -> None:
        self.pushed += 1
        super().push(goal, **slots)

    def spent(self) -> bool:
        """Whether this search has wanted enough: the executive pushes no
        more subgoals once it has (`Executive._means_ends`)."""
        return self.pushed >= SUBGOALS

    def refresh(self) -> None:
        """Every condition on a count, true or not as the counts now are.
        A count is not a fact an action adds, so what depends on one is
        worked out again after each action rather than added by it."""
        for literal in self.conditions:
            if Q.holds(literal, self.facts):
                if literal not in self.facts:
                    self.assert_(literal)
            elif literal in self.facts and not Q.is_amount(literal):
                self.retract(literal)

    def __contains__(self, key) -> bool:
        return key in self.facts or Working.__contains__(self, key)

    def __missing__(self, key):
        if key in self.facts:
            return True
        return Working.__missing__(self, key)

    def keys(self):
        """Slots and facts together: what a chunk is keyed on. A chunk that
        ignored the world would offer the way out of an impasse that arose
        in some other arrangement of the blocks."""
        return set(dict.keys(self)) | self.facts

    def about(self, missing) -> set:
        """What an impasse over `missing` turned on: the facts about the
        things those facts are about, and nothing else in the world.

        A chunk is keyed on this (E6). Keyed on every fact, as `keys` is,
        it never comes round again -- a world is different after every
        action -- and 38 chunks were learned for one hit. `clear a` is
        about block `a`; what else is true of `a` is what decides how it
        was made clear, and where `d` is does not."""
        things = {word for literal in missing
                  for word in str(literal).split()[1:]}
        return ({fact for fact in self.facts
                 if things & set(fact.split()[1:])}
                | {str(one) for one in missing})

    def assert_(self, fact: str) -> None:
        self.facts.add(fact)
        dict.__setitem__(self, fact, True)

    def retract(self, fact: str) -> None:
        """Stop a fact being true -- everywhere, not just here. Reaching
        into the frames beneath is the one thing `Working` must not do for
        belief and must do for a world."""
        self.facts.discard(fact)
        dict.pop(self, fact, None)
        for _, frame in self._beneath:
            frame.pop(fact, None)

    def apply(self, action: W.Action) -> None:
        for fact in action.deletes:
            self.retract(fact)
        for fact in action.adds:
            self.assert_(fact)
        if action.changes:
            before = frozenset(self.facts)
            after = Q.changed(before, action.changes)
            for fact in before - after:
                self.retract(fact)
            for fact in after - before:
                self.assert_(fact)
            self.refresh()
        self.did.append(action)


def operator_of(action: W.Action, goal: frozenset,
                taste: Taste | None = None, promised=()) -> Operator:
    """An action as an operator. This is the whole of the translation.

    `promised` is what it is offered as a way to on a count: `give ... #2`
    for `with=5 apple mary` when she has 3. The executive holds an operator
    to what it `gives`, so an action that would not close its count -- the
    plan having moved it since -- declines rather than half-does it.
    """
    taste = Taste(frozenset(goal)) if taste is None else taste
    promised = tuple(sorted(promised))

    def proposes(memory) -> bool:
        """Never throw away the means. The ends may be undone and redone.

        `unstack b a` gets `held b`; the same subgoal then goes after
        `clear c`, and on the way `drop b` looks like progress and throws
        away the block it is holding. That is clobbering, and it is why
        conjunctive goals in this domain are hard. The executive already
        knows which slots its open means-ends subgoals are achieving; with
        `pursuing` it can be asked, and an action that would take away one
        that is *true now* does not propose -- taking away what is already
        false costs nothing.

        **Except a fact the goal itself wants**, and that exception is the
        Sussman anomaly. Protecting everything pursued costs five of the
        twenty-four problems and gains sussman's shape back, because the
        only way to `on b c` there is to take `on a b` apart again. So the
        rule is asymmetric on purpose: what the search built as a step is
        protected, what it was asked for is not.

        **And never do what would change nothing.** An action whose every
        effect already holds is not a step towards anything. In a declared
        domain this never comes up, because every action there moves
        something; in the open world it came up at once -- VerbNet has a
        dozen readings of verbs that put a thing somewhere, several with no
        preconditions at all, and after the cup was in the shop the planner
        put it there five more times by five different verbs before it
        turned to the book.
        """
        if action.adds and action.adds <= memory.facts:
            return False
        if not PROTECT:
            return True
        guarded = pursuing() - goal
        return not any(fact in guarded and fact in memory.facts
                       for fact in action.deletes)

    def apply(memory):
        if (not Q.satisfied(action.needs, memory.facts)
                or len(memory.did) >= BUDGET):
            return DECLINED
        if promised and not Q.satisfied(
                promised, action.on(frozenset(memory.facts))):
            return DECLINED
        memory.apply(action)
        return CONTINUE

    return Operator(name=action.name, apply=apply, proposes=proposes,
                    needs=tuple(sorted(action.needs)),
                    gives=tuple(sorted(action.adds)) + promised,
                    utility=utility_of(action, taste),
                    rule=f"{action.name}: needs "
                         f"{', '.join(sorted(action.needs))}")


def goal_operator(goal) -> Operator:
    """What the whole search hangs off.

    ProofWriter's lesson, unchanged: means-ends plans toward a *proposing
    operator's* unmet needs, never toward `until` on its own, so a run with
    no operator wanting the goal facts sits at an impasse and does nothing.
    """
    def apply(memory):
        dict.__setitem__(memory, "plan", tuple(memory.did))
        return ANSWERED

    return Operator(name="done", apply=apply, needs=tuple(sorted(goal)),
                    gives=("plan",), utility=1000.0,
                    rule="every fact the goal wants holds")


@dataclass
class Search:
    """What a model run cost, beside what it came to."""

    plan: tuple = ()
    fired: int = 0
    subgoals: int = 0
    depth: int = 1
    answered: bool = False
    #: the run itself. Kept because the means-ends subgoal names are the
    #: only record of *what each action was for* -- `achieve clear red for
    #: stack green red` -- and `talking.Table._why` reads them back.
    trace: object = None


def _tally(trace, at: int = 1) -> tuple:
    fired = len(trace.fired)
    subgoals = len(trace.subgoals)
    depth = at
    for inner in trace.subgoals:
        more, deeper, below = _tally(inner, at + 1)
        fired += more
        subgoals += deeper
        depth = max(depth, below)
    return fired, subgoals, depth


#: How a negative precondition is a slot: `not locked door`.
NOT = "not "


def negated(actions, facts) -> tuple:
    """Actions and facts with every negative precondition made positive.

    `needs` can only ask for presence, so *the door is not locked* is the
    slot `not locked door`: true at the start wherever `locked door` is
    not, taken away by whatever locks it and brought about by whatever
    unlocks it. That is the standard compilation of negative preconditions
    into STRIPS, and it means means-ends can plan to *remove* a blocker the
    same way it plans to bring about a need. Returns (actions, facts, back)
    where `back` maps each compiled action to the one it stands for, so the
    plan handed to the world is the world's own actions.
    """
    actions = list(actions)
    forbidden = set()
    for one in actions:
        forbidden |= set(getattr(one, "forbids", ()))
    if not forbidden:
        return actions, frozenset(facts), {}
    out, back = [], {}
    for one in actions:
        adds = set(one.adds) | {NOT + fact for fact in one.deletes
                                if fact in forbidden}
        deletes = set(one.deletes) | {NOT + fact for fact in one.adds
                                      if fact in forbidden}
        made = dataclasses.replace(
            one, needs=frozenset(one.needs) | {NOT + fact for fact in
                                               one.forbids},
            adds=frozenset(adds), deletes=frozenset(deletes - adds),
            forbids=frozenset())
        out.append(made)
        back[made] = one
    facts = frozenset(facts) | {NOT + fact for fact in forbidden
                                if fact not in facts}
    return out, facts, back


#: Whether a found plan is shortened before it is acted on (`shortened`).
SHORTEN = True


def valid(plan, facts, goal) -> bool:
    """Whether a sequence of actions applies, one after another, from
    `facts`, and leaves `goal` true."""
    state = frozenset(facts)
    for one in plan:
        if not one.holds_in(state):
            return False
        state = one.on(state)
    return Q.satisfied(goal, state)


def shortened(plan, facts, goal, actions=()) -> list:
    """A plan with what it did not need taken out -- checked in the model,
    so what is left is as sound as what was found.

    Means-ends counts nothing: it takes the first way to each subgoal it
    finds, so in errands `go` repeats and a van drives somewhere and back
    for nothing. Two cuts, both general. **A stretch that comes back to a
    state already passed through did nothing**, and goes. Then **each action
    is tried without**, last first, and dropped if the rest still applies
    and still reaches the goal. Neither knows a domain, and neither can make
    a plan wrong: every candidate is run in the model before it is kept.
    """
    plan = list(plan)
    if not valid(plan, facts, goal):
        return plan
    cut = True
    while cut:
        cut = False
        states = [frozenset(facts)]
        for one in plan:
            states.append(one.on(states[-1]))
        first: dict = {}
        for index, state in enumerate(states):
            if state in first:
                plan = plan[:first[state]] + plan[index:]
                cut = True
                break
            first[state] = index
    index = len(plan) - 1
    while index >= 0:
        trial = plan[:index] + plan[index + 1:]
        if valid(trial, facts, goal):
            plan = trial
        index -= 1
    # **A detour one action can make directly**: going home and then to the
    # shop, where going to the shop was on offer all along. Any stretch
    # that one available action takes from the same state to the same
    # state is that action, longest stretch first.
    cut = bool(actions)
    while cut:
        cut = False
        states = [frozenset(facts)]
        for one in plan:
            states.append(one.on(states[-1]))
        for start in range(len(plan)):
            for end in range(len(plan), start + 1, -1):
                direct = next((one for one in actions
                               if one.holds_in(states[start])
                               and one.on(states[start]) == states[end]),
                              None)
                if direct is not None:
                    plan = plan[:start] + [direct] + plan[end:]
                    cut = True
                    break
            if cut:
                break
    return plan


def promising(action, conditions, facts) -> tuple:
    """The conditions on counts an action is a way to: those its change
    would close from what is known now. Giving 2 apples is a way to Mary
    having 5 when she has 3, and not when she has 1 -- then it is a step,
    and something else has to be the rest."""
    out = []
    for fluent, delta in getattr(action, "changes", ()):
        for literal in conditions:
            found = Q.read(literal)
            if found.fluent == fluent and Q.closes(delta, literal, facts):
                out.append(literal)
    return tuple(out)


def smallest(actions, conditions, facts) -> dict:
    """action -> the conditions it is offered as a way to, where of one
    action at several amounts only the smallest that closes a count is:
    giving 4 closes *at least 4* and so do giving 5, 6 and 7, and a search
    offered all of them tries all of them."""
    if not conditions:
        return {}
    best: dict = {}
    offers: dict = {}
    for one in actions:
        offers[one] = promising(one, conditions, facts)
        base, amount = Q.action_words(one.name)
        for literal in offers[one]:
            key = (base, literal)
            if amount is not None and (key not in best
                                       or amount < best[key][0]):
                best[key] = (amount, one)
    out = {}
    for one, literals in offers.items():
        base, amount = Q.action_words(one.name)
        kept = tuple(literal for literal in literals
                     if amount is None or best[(base, literal)][1] is one)
        if kept:
            out[one] = kept
    return out


def think(actions, facts, goal, chunks: Chunks | None = None) -> Search:
    """Plan: means-ends over a model of the world, and the actions it
    applied there are the plan. Nothing outside the model is touched."""
    actions, facts, back = negated(actions, facts)
    goal = frozenset(goal)
    taste = Taste.of(goal, actions)
    counted = Q.conditions(set(goal).union(*[one.needs for one in actions]))
    promised = smallest(actions, counted, facts)
    means = sorted((operator_of(one, goal, taste, promised.get(one, ()))
                    for one in actions),
                   key=lambda one: -one.utility)
    memory = Situation(facts, goal=f"make {', '.join(sorted(goal))} true",
                       conditions=counted)
    executive = Executive([goal_operator(goal)], name="acting", plans=True,
                          means=means, chunks=chunks)
    trace = executive.run(memory)
    fired, subgoals, depth = _tally(trace)
    found = list(memory.get("plan", ()))
    if SHORTEN and found:
        found = shortened(found, facts, goal, actions)
    return Search(plan=tuple(back.get(one, one) for one in found),
                  fired=fired,
                  subgoals=subgoals, depth=depth, trace=trace,
                  answered=trace.answered_by is not None)


# -- the agent: plan, act, look --------------------------------------------

class Surprises(dict):
    """Every surprise opens the same subgoal, whatever it is numbered.

    The executive looks an impasse's name up here, and resolves each name
    once per run. Surprises need a name apiece -- the second one is not the
    first one still standing -- but they all want the same substate, so the
    lookup is by prefix and the dict holds the one subgoal it ever returns.
    """

    def __init__(self, sense: Subgoal) -> None:
        super().__init__(surprise=sense)
        self.sense = sense

    def get(self, name, default=None):
        return (self.sense if str(name).startswith("surprise") else default)


@dataclass
class Gap:
    """What an action was expected to leave, against what it did.

    The first thing in this project a learner could be given: everywhere
    else the signal was whether an answer was right, and this is a
    prediction the agent made itself, falsified by the world, with the
    action that made it still in hand. `lessons.Learner` learns from it.
    """

    action: object
    #: expected and not found
    missing: frozenset
    #: found and not expected
    extra: frozenset
    #: how many actions had been taken when it happened
    after: int
    #: the world just before the action: what a learner compares
    before: frozenset = frozenset()
    #: whether the world refused the action outright
    refused: bool = False


@dataclass
class Attempt:
    """One problem, worked."""

    name: str = ""
    solved: bool = False
    plan: tuple = ()
    acted: int = 0
    optimal: int | None = None
    plans: int = 0
    surprises: int = 0
    search: Search = field(default_factory=Search)
    #: a `Gap` for each surprise, in order -- the prediction against the
    #: observation, which is the first thing in this project that
    #: counterfactual credit could learn from
    gaps: list = field(default_factory=list)
    #: what the surprises taught (`lessons.Lesson`), in order
    lessons: list = field(default_factory=list)

    @property
    def shortest(self) -> bool:
        return (self.solved and self.optimal is not None
                and self.acted == self.optimal)


def agent(problem: W.Problem, real: W.World, chunks=None,
          attempt: Attempt | None = None, tries: int = 4,
          learner=None) -> Executive:
    """The agent as an executive: plan, act, look, and plan again when what
    it saw is not what it expected.

    Four operators, and the order is the point. `done` outranks everything,
    so the run ends the moment the goal holds however it came to. `look`
    outranks `act`, so nothing is done twice before the first is checked.
    `act` outranks `plan it`, so a plan in hand is followed rather than
    re-derived. Each repeats, and each clears its own condition -- the trap
    every repeating operator in this project has fallen into once.

    **A surprise is an impasse, not a flag.** When `look` finds the world
    is not what the action was expected to leave, it does not quietly plan
    again: it names the impasse, and nothing can be proposed until the
    subgoal `make sense of it` has run and handed back a plan. That is E2's
    mechanism doing the job v691 was built for, and it is the reason the
    gap between prediction and observation is somewhere a trace can find it
    rather than a branch inside a loop.
    """
    report = attempt if attempt is not None else Attempt()

    def plan_from(memory) -> str:
        report.plans += 1
        # What has been learned is folded in at every plan, not once: a
        # lesson from a surprise in this very run changes the replan.
        actions = (learner.applied(problem.actions) if learner is not None
                   else problem.actions)
        found = think(actions, real.facts, problem.goal, chunks)
        report.search = found
        if not found.plan or report.plans > tries:
            dict.__setitem__(memory, "plan", [])
            return DECLINED
        memory["plan"] = list(found.plan)
        return CONTINUE

    def acted(memory):
        plan = memory["plan"]
        action = plan.pop(0)
        memory["did"] = action
        memory["before"] = real.facts
        memory["expected"] = action.on(real.facts)
        if not real.do(action):
            # The world refused: it was not as the model had it.
            memory["expected"] = frozenset(real.facts)
            memory["refused"] = True
        else:
            report.acted += 1
        if not plan:
            del memory["plan"]
        return CONTINUE

    def looked(memory):
        expected = memory.pop("expected")
        refused = memory.pop("refused", False)
        before = memory.pop("before", frozenset())
        if not refused and expected == real.facts:
            if learner is not None:
                learner.worked(memory.get("did"), before, real.facts)
            return CONTINUE
        report.surprises += 1
        gap = Gap(memory.get("did"), frozenset(expected) - real.facts,
                  frozenset(real.facts) - frozenset(expected), report.acted,
                  frozenset(before), refused)
        report.gaps.append(gap)
        if learner is not None:
            report.lessons.extend(learner.failed(gap, real.facts))
        memory.pop("plan", None)
        # Numbered, because the executive resolves each impasse *name* once
        # per run, and a second surprise is a second impasse rather than
        # the first one still standing.
        memory["impasse"] = f"surprise {report.surprises}"
        return CONTINUE

    def finished(memory):
        report.solved = True
        memory["outcome"] = "solved"
        return ANSWERED

    # What the impasse opens. `noticed` writes down the difference so the
    # trace carries it; `plan again` replans from the world as it is now,
    # and it is a different operator from `plan it` only so that a trace
    # says which of the two a plan came from.
    def noticed(memory):
        memory["gap"] = report.gaps[-1] if report.gaps else None
        return CONTINUE

    def replanning(memory) -> bool:
        """Plan, unless a plan is in hand or a surprise is still open: the
        impasse is what decides what happens next, and this operator
        standing in for it would make the impasse cosmetic."""
        if dict.__contains__(memory, "plan"):
            return False
        named = dict.get(memory, "impasse")
        return not named or named in dict.get(memory, "resolved", [])

    sense = Subgoal(
        goal="make sense of it",
        executive=Executive([
            Operator(name="noticed", apply=noticed,
                     rule="what was expected, against what is",
                     gives=("gap",)),
            Operator(name="plan again", apply=plan_from, needs=("gap",),
                     rule="means-ends again, from the world as it is now",
                     gives=("plan",)),
        ], name="surprise"),
        returns=("plan",))

    watching = Executive([
        Operator(name="done", apply=finished,
                 proposes=lambda memory: real.solved(problem.goal),
                 rule="the goal holds in the world", gives=("outcome",)),
        Operator(name="look", apply=looked,
                 proposes=lambda memory: "expected" in memory,
                 rule="what the world is, against what was expected",
                 needs=("expected",), repeats=True),
        Operator(name="act", apply=acted,
                 proposes=lambda memory: bool(memory.get("plan")),
                 rule="do the next action of the plan",
                 needs=("plan",), gives=("expected",),
                 effects=(W.WORLD,), repeats=True),
        Operator(name="plan it", apply=plan_from, proposes=replanning,
                 rule="means-ends, against a model of the world",
                 gives=("plan",), repeats=True),
    ], name="agent", subgoals={"surprise": sense})
    # After construction, because `Executive` copies what it is given into a
    # plain dict -- rightly, since a caller's dict changing underneath it
    # would change which impasses it can resolve mid-run. Here the mapping
    # *is* the behaviour, so it is put back.
    watching.subgoals = Surprises(sense)
    return watching


def solve(problem: W.Problem, chunks=None, optimal: bool = True) -> Attempt:
    """Work one problem end to end, and say what it cost."""
    real = problem.world()
    report = Attempt(name=problem.name)
    executive = agent(problem, real, chunks, report)
    executive.run(Working(goal=f"solve {problem.name}"))
    report.plan = tuple(real.did)
    report.solved = real.solved(problem.goal)
    if optimal:
        found = W.shortest(problem)
        report.optimal = None if found is None else len(found)
    return report


# -- what it is measured against -------------------------------------------

def by_regression(problem: W.Problem) -> tuple:
    """`Executive.plan` used exactly as it is: regression over needs and
    gives, which is a STRIPS planner with no delete lists.

    It is here to be a number rather than an argument. The plan it returns
    is executed against a real world for as long as it applies, and `steps`
    is how far it got -- which for blocks is nearly always short, because
    the second action's preconditions were deleted by the first.
    """
    goal = frozenset(problem.goal)
    taste = Taste.of(goal, problem.actions)
    means = sorted((operator_of(one, goal, taste)
                    for one in problem.actions),
                   key=lambda one: -one.utility)
    executive = Executive([goal_operator(goal)], name="regression",
                          plans=True, means=means)
    names = executive.plan(tuple(sorted(goal)), tuple(sorted(problem.start)))
    if names is None:
        return None, 0, False
    by_name = {one.name: one for one in problem.actions}
    real = problem.world()
    steps = 0
    for name in names:
        if name not in by_name or not real.do(by_name[name]):
            break
        steps += 1
    return tuple(names), steps, real.solved(problem.goal)


@dataclass
class Report:
    solved: int = 0
    shortest: int = 0
    total: int = 0
    acted: int = 0
    fired: int = 0
    subgoals: int = 0
    deepest: int = 0
    surprises: int = 0
    rows: list = field(default_factory=list)


def measure(problems, chunks=None) -> Report:
    out = Report()
    for one in problems:
        with episode():
            got = solve(one, chunks)
        out.total += 1
        out.solved += got.solved
        out.shortest += got.shortest
        out.acted += got.acted
        out.fired += got.search.fired
        out.subgoals += got.search.subgoals
        out.deepest = max(out.deepest, got.search.depth)
        out.surprises += got.surprises
        out.rows.append(got)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled", type=int, default=20,
                        help="random four-block problems beside the suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default="blocks",
                        choices=sorted(SAMPLERS),
                        help="which world to solve problems in: the point "
                             "of there being more than one is that nothing "
                             "in the planner knows which")
    parser.add_argument("--ablate", action="store_true",
                        help="drop each signal in turn and solve again")
    parser.add_argument("--regression", type=float, default=30.0,
                        help="seconds to spend on the regression baseline; "
                             "it has no cutoff of its own")
    parser.add_argument("--chunks", action="store_true",
                        help="share one procedural memory across problems")
    options = parser.parse_args(argv)

    if options.domain == "blocks":
        problems = SUITE + SAMPLERS["blocks"](options.sampled, options.seed)
    else:
        problems = SAMPLERS[options.domain](options.sampled, options.seed)
    if options.ablate:
        shipped = dict(SIGNALS)

        def row(label: str) -> None:
            got = measure(problems)
            print(f"{label:<34} {got.solved:>4}/{got.total:<3} "
                  f"{got.shortest:>9} {got.subgoals:>10}")

        print(f"{len(problems)} problems, one thing dropped at a time\n")
        print(f"{'dropped':<34} {'solved':>8} {'shortest':>9} "
              f"{'subgoals':>10}")
        print("-" * 64)
        row("nothing (as shipped)")
        for name in shipped:
            SIGNALS[name] = 0.0
            try:
                row(name)
            finally:
                SIGNALS.update(shipped)
        for name in shipped:
            SIGNALS[name] = 0.0
        try:
            row("every signal (domain order)")
        finally:
            SIGNALS.update(shipped)
        global PROTECT
        PROTECT = False
        try:
            row("goal protection")
        finally:
            PROTECT = True
        return 0

    chunks = Chunks() if options.chunks else None
    print(f"{len(problems)} problems "
          f"({len(SUITE)} fixed, {len(problems) - len(SUITE)} sampled)\n")
    print(f"{'problem':<18} {'plan':>5} {'best':>5} {'fired':>6} "
          f"{'subgoals':>9} {'deep':>5}  outcome")
    print("-" * 70)
    got = measure(problems, chunks)
    for row in got.rows:
        best = "-" if row.optimal is None else row.optimal
        outcome = ("solved" if row.shortest else
                   "solved, longer" if row.solved else "NOT SOLVED")
        print(f"{row.name:<18} {row.acted:>5} {best:>5} "
              f"{row.search.fired:>6} {row.search.subgoals:>9} "
              f"{row.search.depth:>5}  {outcome}")
    print("-" * 70)
    print(f"solved {got.solved}/{got.total}, of which "
          f"{got.shortest} in the fewest actions; "
          f"{got.acted} actions, {got.fired} operators fired, "
          f"{got.subgoals} subgoals, deepest stack {got.deepest}, "
          f"{got.surprises} surprises")

    print()
    tried = found = whole = steps = 0
    started = time.time()
    # Four blocks and under. `Executive.plan` regresses with no cutoff of
    # its own and nothing ever becomes false in it, so every action always
    # *could* be added: four blocks takes up to nine seconds and five does
    # not return. That is the cost of the missing delete lists, before the
    # plans it does return are looked at at all.
    for one in problems:
        if len(one.objects) > 4 or time.time() - started > options.regression:
            continue
        names, got, solved = by_regression(one)
        tried += 1
        found += names is not None
        steps += got
        whole += solved
    print(f"`Executive.plan` regression, {tried} problems in "
          f"{time.time() - started:.0f}s: a plan for {found}, executable to "
          f"the end for {whole}, {steps} actions applied before one did not "
          f"-- it has no delete lists")

    if chunks is not None:
        print(f"chunks: {len(chunks.rules)} learned, {chunks.hits} hits, "
              f"{chunks.misses} misses, {chunks.forgotten} forgotten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

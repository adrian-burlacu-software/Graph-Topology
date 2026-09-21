"""An agent that plans and then acts: v691a.

`DESIGN.md` (v690) §8c closed the executive as a *reasoner*. Three things
were missing before it could be an *agent*, and this is the first of them
built end to end, deliberately with no reader anywhere near it.

    a world           facts that change only by acting, and are not undone
                      by giving up on a goal (`world.World`)
    planning          the executive's own means-ends analysis, run over a
                      model of the world rather than over what it knows
    execution         one action at a time against the real world, looking
                      after each, and planning again when what it sees is
                      not what it expected

## Working memory is nearly a world, and the difference is one method

The striking thing is how little had to be written. An `Action`'s
preconditions are an `Operator`'s `needs`, its adds are that operator's
`gives`, and `Executive` then plans over actions with no translation at all:
`_means_ends` picking the most useful waiting operator and pushing a subgoal
for the slots it lacks **is** goal-stack planning, which is what a STRIPS
planner of the period did. Nothing in `executive.py` was changed.

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
import time
from dataclasses import dataclass, field

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Chunks,
                                     Executive, Operator, Working, episode,
                                     pursuing)
from research.v691 import world as W

#: How the ground actions are ordered before the executive ever sees them,
#: as utilities. Means-ends takes the most useful waiting operator and the
#: most useful way to what it lacks, so this ordering is the whole of the
#: planner's taste -- there is no evaluation function and no backtracking
#: across the choice.
#:
#: **Two signals survived; three obvious ones did not.** Written first were
#: four local judgements: prefer an action that achieves a goal fact, avoid
#: one that undoes a goal fact, prefer one that frees a block the goal
#: buries, prefer putting a block on the table. Then the one that is not
#: local: build the goal tower from the bottom. An exhaustive search over
#: all sixteen
#: subsets of the four, on `world.SUITE` plus 20 sampled four-block
#: problems, found the
#: goal-tower ordering on its own as good as any combination of them -- and
#: then, held out on 97 sampled problems over three unseen seeds, `frees a
#: goal block` beside it was better again. Solved / shortest / subgoals on
#: the held-out 97:
#:
#:     both, as shipped                78   68   3052
#:     the tower ordering alone         72   65   2833
#:     all five signals                 67   57   3686
#:     none (the domain's own order)    58   49   4579
#:     as shipped, without protection   60   51   2731
#:
#: Two things in that table are worth more than the winner. **`frees a goal
#: block` was measured out and then back in**: with the other three it cost
#: points, with the tower ordering alone it is worth six problems, so the
#: subset search on its own would have thrown away the signal that solves
#: the Sussman anomaly. And **`achieves a goal fact`, the most obvious
#: of the local signals, is worth nothing at all** -- what matters is not
#: which action helps now but the structure of the goal: build from the
#: bottom, and clear what the goal has to sit on.
#:
#: Magnitudes do not matter: 0.25 and 2.0 for `frees a goal block` give
#: identical results, because means-ends reads an order and not a score.
SIGNALS = {"frees a goal block": 0.5,
           "a level higher in the goal tower": -0.5}

#: Whether an action may throw away what an open subgoal has achieved: see
#: `operator_of`. Turned off only by `--ablate`, which is where the last
#: line of the table in `SIGNALS` comes from.
PROTECT = True

#: A model run that has applied this many actions has not found a plan; it
#: has wandered. Blocks problems here are solved in at most ten.
BUDGET = 60


def levels(goal) -> dict:
    """How far above the bottom of the goal structure each block sits.

    The one piece of the domain that is not a local judgement. Achieving
    `on a b` before `on b c` is what makes a blocks problem unsolvable
    without taking the first apart again, and the fix known since the
    Sussman anomaly is to build from the bottom. This is that, as a number
    a utility can carry: `stack c d` where d rests on nothing the goal
    cares about is level 0, `stack b c` is level 1, `stack a b` is level 2.
    """
    under = {}
    for fact in goal:
        parts = fact.split()
        if parts[0] == "on":
            under[parts[1]] = parts[2]
    depth: dict = {}

    def of(block, seen=()):
        if block in depth:
            return depth[block]
        if block not in under or block in seen:
            return 0
        depth[block] = 1 + of(under[block], seen + (block,))
        return depth[block]

    for block in list(under):
        of(block)
    return depth


def utility_of(action: W.Action, goal: frozenset, deep: dict) -> float:
    """Where this action sits in the order the executive will try things.

    Two lines, after four other signals were measured out: see `SIGNALS`.
    Build the goal tower from the bottom, and clear a block the goal needs
    something to sit on.
    """
    score = 1.0
    parts = action.name.split()
    if parts[0] == "stack":
        score += (SIGNALS["a level higher in the goal tower"]
                  * deep.get(parts[2], 0))
    if any(fact.startswith("clear ")
           and any(one.endswith(" " + fact.split()[1]) for one in goal)
           for fact in action.adds):
        score += SIGNALS["frees a goal block"]
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

    def __init__(self, facts=(), goal: str = "") -> None:
        super().__init__(goal=goal)
        self.facts = set(facts)
        #: the actions applied, in order: the plan, as it is being found
        self.did: list = []
        for fact in self.facts:
            dict.__setitem__(self, fact, True)

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
        self.did.append(action)


def operator_of(action: W.Action, goal: frozenset,
                deep: dict | None = None) -> Operator:
    """An action as an operator. This is the whole of the translation."""
    deep = levels(goal) if deep is None else deep

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
        """
        if not PROTECT:
            return True
        guarded = pursuing() - goal
        return not any(fact in guarded and fact in memory.facts
                       for fact in action.deletes)

    def apply(memory):
        if not action.needs <= memory.facts or len(memory.did) >= BUDGET:
            return DECLINED
        memory.apply(action)
        return CONTINUE

    return Operator(name=action.name, apply=apply, proposes=proposes,
                    needs=tuple(sorted(action.needs)),
                    gives=tuple(sorted(action.adds)),
                    utility=utility_of(action, goal, deep),
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


def think(actions, facts, goal, chunks: Chunks | None = None) -> Search:
    """Plan: means-ends over a model of the world, and the actions it
    applied there are the plan. Nothing outside the model is touched."""
    goal = frozenset(goal)
    deep = levels(goal)
    means = sorted((operator_of(one, goal, deep) for one in actions),
                   key=lambda one: -one.utility)
    memory = Situation(facts, goal=f"make {', '.join(sorted(goal))} true")
    executive = Executive([goal_operator(goal)], name="acting", plans=True,
                          means=means, chunks=chunks)
    trace = executive.run(memory)
    fired, subgoals, depth = _tally(trace)
    return Search(plan=tuple(memory.get("plan", ())), fired=fired,
                  subgoals=subgoals, depth=depth,
                  answered=trace.answered_by is not None)


# -- the agent: plan, act, look --------------------------------------------

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

    @property
    def shortest(self) -> bool:
        return (self.solved and self.optimal is not None
                and self.acted == self.optimal)


def agent(problem: W.Problem, real: W.World, chunks=None,
          attempt: Attempt | None = None, tries: int = 4) -> Executive:
    """The agent as an executive: plan, act, look, and plan again when what
    it saw is not what it expected.

    Four operators, and the order is the point. `done` outranks everything,
    so the run ends the moment the goal holds however it came to. `look`
    outranks `act`, so nothing is done twice before the first is checked.
    `act` outranks `plan it`, so a plan in hand is followed rather than
    re-derived. Each repeats, and each clears its own condition -- the trap
    every repeating operator in this project has fallen into once.
    """
    report = attempt if attempt is not None else Attempt()

    def planned(memory):
        report.plans += 1
        found = think(problem.actions, real.facts, problem.goal, chunks)
        report.search = found
        if not found.plan or report.plans > tries:
            dict.__setitem__(memory, "plan", [])
            return DECLINED
        memory["plan"] = list(found.plan)
        return CONTINUE

    def acted(memory):
        plan = memory["plan"]
        action = plan.pop(0)
        memory["expected"] = action.on(real.facts)
        if not real.do(action):
            # The world refused: it was not as the model had it.
            memory["expected"] = None
        else:
            report.acted += 1
        if not plan:
            del memory["plan"]
        return CONTINUE

    def looked(memory):
        expected = memory.pop("expected")
        if expected is None or expected != real.facts:
            report.surprises += 1
            memory.pop("plan", None)
        return CONTINUE

    def finished(memory):
        report.solved = True
        memory["outcome"] = "solved"
        return ANSWERED

    return Executive([
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
        Operator(name="plan it", apply=planned,
                 proposes=lambda memory: not dict.__contains__(memory,
                                                               "plan"),
                 rule="means-ends, against a model of the world",
                 gives=("plan",), repeats=True),
    ], name="agent")


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
    deep = levels(goal)
    means = sorted((operator_of(one, goal, deep) for one in problem.actions),
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
    parser.add_argument("--ablate", action="store_true",
                        help="drop each signal in turn and solve again")
    parser.add_argument("--regression", type=float, default=30.0,
                        help="seconds to spend on the regression baseline; "
                             "it has no cutoff of its own")
    parser.add_argument("--chunks", action="store_true",
                        help="share one procedural memory across problems")
    options = parser.parse_args(argv)

    problems = W.SUITE + W.sampled(options.sampled, seed=options.seed)
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
          f"({len(W.SUITE)} fixed, {len(problems) - len(W.SUITE)} sampled)\n")
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
        if len(one.names) > 4 or time.time() - started > options.regression:
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

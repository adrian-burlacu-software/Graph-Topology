"""Search that goes through the architecture: one solver, a switch each.

`PLAN.md` Phase 2. The baseline is the architecture as it was: the
executive's means-ends (`v687.executive._means_ends`) as regression over
what operators need and give -- here the types -- from the wanted type
back to what is in hand, each complete candidate checked. Each mechanism
is a switch, measured on and off against it:

    recognition  2a  a trie of solved specs, walked down by a new spec's
                     features: what solved the specs it reaches is tried
                     first, re-bound to this one's parameters; one left is
                     the unique-answer shortcut
    meet         2b  two tries over one answer set: forward, what every
                     expression built so far *does* on the examples (a
                     trie keyed by its values -- a repeated vector is
                     access, not allocation, and is not grown again);
                     backward, which types can still lead to the goal.
                     An answer is where the forward values meet the
                     required outputs
    coarse       2c  types before values: only what the backward trie
                     says can still reach the goal is grown or evaluated
    repair       2d  a near miss is walked back: each of its subtrees in
                     turn is replaced by forward-trie entries of its type,
                     instead of searching again
    learned      2e  operators ordered by what solved specs with these
                     features (the executive's learned control)
    chunks       2f  subtrees of solved programs kept as operators with
                     holes, offered to every later spec

Every result says how it was found (`route`), what it cost, and whether
it also meets the hidden examples.

**Judged** (where the request has been read, `Spec.expected`): a few
examples do not say which of the programs that meet them was asked for --
a constant meets one example of anything. So the forward trie tells
expressions apart by what they do beyond the examples too
(`meaning.probes`), everything that meets the examples is kept with where
it came from, and the answer is the best founded (`RANK`): what the decoder
wrote for this request, then a near miss of its repaired by edits, then what
the search found, and last a value it began with.

**Risk first** (`risk.py`, `Spec.moves`): six estimators score the request
before it is searched, and the resolution matrix says what its search does
-- deeper and dearer where it is deep, probes at the edges and in pairs where
its rules and values are many, the behaviour most programs agree on where it
is open, four eyes where a wrong answer costs much (`_settled`, `_chosen`).
A request in no shaded corner is searched with half the budget.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

from research.v687.trie import PredicateTrie, ROOT
from research.v696 import cognition as C
from research.v696 import program as P
from research.v696.checker import checker
from research.v696.spec import Spec

#: How many candidates are sent to Node at once.
BATCH = 400
#: How far a program is grown, in operators.
DEPTH = 3
#: How many candidates may be evaluated for one spec.
BUDGET = 20000
#: How many of a spec's recurring values are taken as its constants.
LITERALS = 3
#: How many near misses repair may check: its own allowance.
REPAIRS = 20000
#: How many candidates folds by induction may spend: their own allowance.
INDUCTIONS = 12000
#: What the reader of meaning must be this sure of before a program that
#: meets the examples is refused for not showing it.
STRONG = 0.95
#: ... and how sure, where the risk matrix asks for golden files (S×X:
#: the request's own reading is the reference the program is held to)
GOLDEN = 0.8
#: How much an operator the reader expects is worth in the attention queue.
PRIOR = 1.0
#: Judged (`Solver._chosen`): how well founded a program that meets the
#: examples is, by where it came from -- lower first. What the search
#: finds itself (any other stage) is `FOUND`.
RANK = {"proposed": 0, "edited": 1, "part": 2, "pool": 3}
FOUND = 2
#: How many proposals are repaired by edits, and what each may run.
EDITED = 4
EDITS = 400


@dataclass
class Switches:
    recognition: bool = False
    meet: bool = False
    coarse: bool = False
    repair: bool = False
    learned: bool = False
    chunks: bool = False
    #: rung 2: control forms, their holes as subgoals (`forms.py`)
    forms: bool = False
    #: the decoder's programs as sketches (`sketcher.py`)
    proposals: bool = False

    @classmethod
    def all(cls) -> "Switches":
        return cls(True, True, True, True, True, True, True, True)

    def label(self) -> str:
        on = [name for name, value in vars(self).items() if value]
        return "+".join(on) or "baseline"


@dataclass
class Result:
    spec: str
    program: P.Expr | None = None
    #: recognized | deduced | meet | repaired | means-ends | unsolved --
    #: and with the decoder: proposed | edited | part
    route: str = "unsolved"
    evaluated: int = 0
    #: holes solved as specs of their own (rung 2)
    subgoals: int = 0
    seconds: float = 0.0
    #: meets the hidden examples too
    general: bool = False
    #: programs that met the examples and were refused: what they did
    #: beyond them contradicted what the request means
    rejected: int = 0
    #: judged: a second program, written or found apart from the answer,
    #: does what it does beyond the examples (`risk.py`, four eyes)
    confirmed: bool = False

    @property
    def solved(self) -> bool:
        return self.program is not None


def _key(values) -> str:
    return json.dumps(values, sort_keys=True)


def _matches(row, outputs) -> bool:
    return all("value" in one and _key(one["value"]) == _key(out)
               for one, out in zip(row, outputs))


def _relation(value, output) -> float:
    """How a value stands to a required output, whatever their types: the
    same, one inside the other, the same length, the same items. The
    backward trie at the level of values, not only types."""
    if _key(value) == _key(output):
        return 3.0
    score = 0.0
    if isinstance(value, (str, list)) and isinstance(output, (str, list)):
        if type(value) is type(output) and value and output:
            if isinstance(output, str) and (value in output
                                            or output in value):
                score += 1.5
            if isinstance(output, list) and all(one in value
                                                for one in output):
                score += 1.5
        if len(value) == len(output):
            score += 1.0
    if isinstance(output, (int, float)) and not isinstance(output, bool):
        if isinstance(value, (str, list)) and len(value) == output:
            score += 1.5
        if isinstance(value, list) and output in value:
            score += 1.5
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score += 1.0 / (1.0 + abs(value - output))
    if isinstance(output, str) and isinstance(value, list) and \
            output in [str(one) for one in value]:
        score += 1.5
    return score


def relevance(row, outputs) -> float:
    """Over every example: how far a value already resembles the answer --
    and, whatever its type, whether the answer is a function of it."""
    resembles = sum(_relation(one["value"], out) for one, out in
                    zip(row, outputs) if "value" in one) / max(len(outputs),
                                                               1)
    return resembles + determines(row, outputs)


def determines(row, outputs) -> float:
    """Functional dependence: the output is fixed by this value -- equal
    values never meet different outputs -- and the value varies with it.
    `x % 2` determines `is x even` though it resembles neither `true` nor
    `false`. The fewer distinct values, the more it has already done."""
    if len({_key(out) for out in outputs}) < 2:
        return 0.0
    seen: dict = {}
    for one, out in zip(row, outputs):
        if "value" not in one:
            return 0.0
        key = _key(one["value"])
        if seen.setdefault(key, _key(out)) != _key(out):
            return 0.0
    distinct = len(seen)
    if distinct < 2:
        return 0.0
    return 1.5 * (len({_key(out) for out in outputs}) / distinct)


def _near(row, outputs) -> int:
    return sum("value" in one and _key(one["value"]) == _key(out)
               for one, out in zip(row, outputs))


# -- what persists across specs: the architecture's memories ---------------

class Memory:
    """What solving leaves behind, for the next spec -- the general
    mechanisms of `cognition.py`, with programs as what is stored: the
    recognition trie, the learned control, and the chunks."""

    def __init__(self) -> None:
        self.recognizer = C.Recognizer()
        self.control = C.Control()
        #: chunk op key -> (op, template)
        self.chunks: dict = {}

    @property
    def specs(self) -> int:
        return self.control.problems

    def remember(self, spec: Spec, program: P.Expr, switches: Switches
                 ) -> None:
        features = spec.features()
        self.recognizer.remember(features, (spec, program))
        self.control.learn(features, [op.key for op in program.ops()])
        if switches.chunks:
            for sub in _subtrees(program):
                if sub.depth >= 2:
                    chunk = _chunk(sub)
                    if chunk is not None:
                        self.chunks.setdefault(chunk[0].key, chunk)

    def recognise(self, features) -> list:
        return self.recognizer.recognise(features)

    def utility(self, op: P.Op, features) -> float:
        """The learned control's score for an op here (2e)."""
        return self.control.utility(op.key, features)


def _subtrees(expr: P.Expr):
    yield expr
    for one in expr.args:
        yield from _subtrees(one)


def _has_lambda(expr: P.Expr) -> bool:
    return expr.kind == "lambda" or any(_has_lambda(one)
                                        for one in expr.args)


def _chunk(expr: P.Expr):
    """A subtree as an operator: its leaves become holes, in order. Not one
    with a lambda in it: a lambda's parameters are bound there, not free."""
    if _has_lambda(expr):
        return None
    holes = []

    def lift(one):
        if one.kind in ("param", "const"):
            holes.append(one.type)
            return P.Expr(one.type, "hole", name=f"${len(holes) - 1}")
        return P.Expr(one.type, one.kind, one.name, one.value, one.op,
                      tuple(lift(arg) for arg in one.args))

    template = lift(expr)
    if not holes:
        return None
    op = P.Op(f"chunk[{template.source()}]", "chunk", tuple(holes),
              expr.type)
    return op, template


def _instantiate(template: P.Expr, args) -> P.Expr:
    if template.kind == "hole":
        return args[int(template.name[1:])]
    return P.Expr(template.type, template.kind, template.name,
                  template.value, template.op,
                  tuple(_instantiate(one, args) for one in template.args))


# -- the solver ------------------------------------------------------------

class Solver:
    def __init__(self, switches: Switches | None = None,
                 memory: Memory | None = None, depth: int = DEPTH,
                 budget: int = BUDGET) -> None:
        self.switches = switches or Switches()
        self.memory = memory if memory is not None else Memory()
        self.depth = depth
        self.budget = budget
        self.inductions = INDUCTIONS
        #: what the risk matrix says of the spec being solved (`risk.py`)
        self._moves, self._behaving, self._spec = (), {}, None
        #: what the last solve did, as it happened (v697's page shows it):
        #: [{"event", "at" (candidates evaluated by then), ...}]
        self.events: list = []
        self._result = None
        #: judged: (program, where it came from, candidates so far)
        self._hits, self._stage = [], "meet"

    @property
    def _stage(self) -> str:
        return self.__stage

    @_stage.setter
    def _stage(self, stage: str) -> None:
        # every change of stage is an event: the search's account of itself
        if getattr(self, "_Solver__stage", None) != stage:
            self._note("stage", stage=stage)
        self.__stage = stage

    def _note(self, event: str, **said) -> None:
        """One thing the search did, kept for whoever asks what it did."""
        self.events.append({"event": event, "at": getattr(
            self._result, "evaluated", 0), **said})

    # the pieces every route shares
    def _ops(self, spec: Spec) -> list:
        ops = list(P.library().ops) + list(spec.library)
        if self.switches.chunks:
            ops += [op for op, _ in self.memory.chunks.values()]
        if self.switches.learned and self.memory.specs:
            features = spec.features()
            ops.sort(key=lambda op: -self.memory.utility(op, features))
        return ops

    def _build(self, op: P.Op, args) -> P.Expr:
        if op.kind == "chunk":
            template = next(one for key, (chunk, one) in
                            self.memory.chunks.items() if key == op.key)
            return _instantiate(template, args)
        return P.apply(op, args)

    def _pool(self, spec: Spec) -> list:
        pool = spec.inputs() + [P.const(value, kind)
                                for value, kind in P.CONSTANTS]
        seen = {(one.type, one.source()) for one in pool}
        # The spec's own constants: a value that recurs across its examples
        # is the task's; one that varies is data. Numbers and single
        # characters only -- a longer string taken from an output is the
        # output memorised, not a constant of the program.
        counts: dict = defaultdict(int)
        for args, output in spec.examples:
            here = set()
            for value in list(args) + [output]:
                if isinstance(value, bool) or not isinstance(value,
                                                             (int, str)):
                    continue
                if isinstance(value, str) and len(value) != 1:
                    continue
                kind = "number" if isinstance(value, int) else "string"
                here.add((kind, json.dumps(value)))
            for one in here:
                counts[one] += 1
        recurring = sorted((one for one, count in counts.items()
                            if count * 2 >= len(spec.examples)
                            and len(spec.examples) > 1),
                           key=lambda one: -counts[one])[:LITERALS]
        for kind, said in recurring:
            if (kind, said) not in seen:
                seen.add((kind, said))
                pool.append(P.const(json.loads(said), kind))
        return pool

    def _evaluate(self, spec: Spec, exprs: list, beyond=()) -> list:
        """Each expression's values on the examples' inputs -- and on
        `beyond`, inputs no example gives an output for."""
        rows = []
        for start in range(0, len(exprs), BATCH):
            batch = exprs[start:start + BATCH]
            rows += checker().values(spec.names, spec.cases + list(beyond),
                                     [one.source() for one in batch],
                                     prelude=P.prelude(batch))
        return rows

    def _general(self, spec: Spec, program: P.Expr) -> bool:
        if not spec.hidden:
            return True
        row = checker().values(spec.names, [a for a, _ in spec.hidden],
                               [program.source()],
                               prelude=P.prelude([program]))[0]
        return _matches(row, [o for _, o in spec.hidden])

    def solve(self, spec: Spec) -> Result:
        """What the risk matrix says of this request (`Spec.moves`) holds
        while it is searched: deeper, a budget of its own."""
        moves = spec.moves
        if not moves:
            return self._solve(spec)
        kept = self.depth, self.budget, self.inductions
        if "deeper" in moves:
            self.depth += 1
            self.inductions *= 2
        self.budget = int(self.budget * moves.budget)
        try:
            return self._solve(spec)
        finally:
            self.depth, self.budget, self.inductions = kept

    def _solve(self, spec: Spec) -> Result:
        started = time.time()
        self._deduced = {}
        result = Result(spec.name)
        self.events, self._result = [], result
        self._note("begin", spec=spec.name, depth=self.depth,
                   budget=self.budget, proposals=len(spec.proposals),
                   moves=sorted(getattr(spec.moves, "on", ()) or ()))
        self._hits, self._stage = [], "recognized"
        self._moves = spec.moves or ()
        self._behaving, self._spec = {}, spec
        found = None
        if self.switches.recognition:
            found = self._recognised(spec, result)
            if found is not None:
                result.route = "recognized"
        if found is None and self.switches.meet:
            found = self._meet(spec, result)
        elif found is None:
            self._stage = "means-ends"
            found = self._means_ends(spec, result)
            if found is not None:
                result.route = "means-ends"
        if self._hits:
            # judged: of everything that met the examples, the one the
            # request most likely asked for
            found, result.route, result.confirmed = self._chosen(spec)
        result.program = found
        result.seconds = time.time() - started
        self._note("end", route=result.route,
                   program=found.source() if found is not None else None,
                   confirmed=result.confirmed, seconds=round(
                       result.seconds, 2))
        if found is not None:
            result.general = self._general(spec, found)
            self.memory.remember(spec, found, self.switches)
        return result

    # 2a
    def _recognised(self, spec: Spec, result: Result) -> P.Expr | None:
        candidates = []
        for other, program in self.memory.recognise(spec.features()):
            rebound = _rebind(program, other, spec)
            if rebound is not None:
                candidates.append(rebound)
        if not candidates:
            return None
        rows = self._evaluate(spec, candidates)
        result.evaluated += len(candidates)
        for expr, row in zip(candidates, rows):
            if _matches(row, spec.outputs) and self._accepts(spec, expr,
                                                             result):
                return expr
        return None

    # the baseline: means-ends regression, iterative deepening
    def _means_ends(self, spec: Spec, result: Result) -> P.Expr | None:
        pool = self._pool(spec)
        ops = self._ops(spec)
        giving = defaultdict(list)
        for op in ops:
            giving[op.gives].append(op)

        def achieve(kind: str, depth: int):
            for one in pool:
                if one.type == kind:
                    yield one
            if depth == 0:
                return
            for op in giving[kind]:
                parts = [list(itertools.islice(achieve(need, depth - 1),
                                               60)) for need in op.needs]
                for args in itertools.product(*parts):
                    yield self._build(op, args)

        for depth in range(1, self.depth + 1):
            batch = []
            for expr in achieve(spec.returns, depth):
                batch.append(expr)
                if len(batch) == BATCH:
                    hit = self._check(spec, batch, result)
                    if hit is not None:
                        return hit
                    batch = []
                if result.evaluated >= self.budget:
                    return None
            hit = self._check(spec, batch, result)
            if hit is not None:
                return hit
        return None

    def _check(self, spec, batch, result) -> P.Expr | None:
        if not batch:
            return None
        rows = self._evaluate(spec, batch)
        result.evaluated += len(batch)
        for expr, row in zip(batch, rows):
            if _matches(row, spec.outputs) and self._accepts(spec, expr,
                                                             result):
                return expr
        return None

    def _judged(self, spec: Spec) -> bool:
        """Whether there is a reading of the request to judge by. Without
        one, the first program that meets the examples is the answer."""
        return bool(spec.expected) and "uses" in spec.expected

    def _beyond(self, spec: Spec, moved: bool = True) -> list:
        """Inputs varied from the examples' (`meaning.probes`): what a
        program does on them is part of what it is, though no example says
        what it should do there. Where the risk matrix asks (`moved`), more:
        the arguments at their edges (S), two varied at once (P), the two at
        their edges (P×S)."""
        from research.v696 import meaning as M
        probes = self.__dict__.setdefault("_probes", {})
        if spec.name not in probes:
            probes[spec.name] = M.probes(spec.examples)
        if not moved or not self._moves:
            return probes[spec.name]
        key = (spec.name, "moved")
        if key not in probes:
            more = list(probes[spec.name])
            if "edges" in self._moves:
                more += M.edge_probes(spec.examples)
            if "pairs" in self._moves:
                more += M.pair_probes(spec.examples)
            if "cross" in self._moves:
                more += M.pair_probes(spec.examples, edges=True)
            seen, probes[key] = set(), []
            for one in more:
                if _key(one) not in seen:
                    seen.add(_key(one))
                    probes[key].append(one)
        return probes[key]

    def _behaviour(self, spec: Spec, programs: list) -> dict:
        """source -> what each program does beyond the examples, a value or
        "!" where it throws, each run once."""
        todo = [one for one in programs
                if one.source() not in self._behaving]
        cases = self._beyond(spec)
        if todo and cases:
            rows = self._evaluate(spec, todo, cases)
            for one, row in zip(todo, rows):
                self._behaving[one.source()] = tuple(
                    _key(cell["value"]) if "value" in cell else "!"
                    for cell in row[len(spec.cases):])
        for one in todo:
            self._behaving.setdefault(one.source(), ())
        return {one.source(): self._behaving[one.source()]
                for one in programs}

    def _agreeing(self, spec: Spec, hits: list) -> dict:
        """What the hits do beyond the examples -> the distinct programs
        that do it (a value the search began with is no one's reading)."""
        groups: dict = defaultdict(dict)
        doing = self._behaviour(spec, [program for program, _, _ in hits])
        for program, stage, _ in hits:
            if stage != "pool":
                groups[doing[program.source()]].setdefault(
                    program.source(), (program, stage))
        return groups

    @staticmethod
    def _authors(spec: Spec, group: dict) -> set:
        """Who arrived at what a group does, apart: each decoder that wrote
        one of its programs (`Spec.authors`), the search for one it found,
        the edits for one they repaired. One program written by two
        decoders is two pairs of eyes; two programs by one, one."""
        out = set()
        for source, (_, stage) in group.items():
            if stage == "proposed":
                out |= spec.authors.get(source) or {"a decoder"}
            else:
                out.add({"edited": "edits", "part": "a decoder's part"}.get(
                    stage, "search"))
        return out

    def _confirmed(self, spec: Spec) -> bool:
        return any(len(self._authors(spec, group)) >= 2
                   for group in self._agreeing(spec, self._hits).values())

    def _accepts(self, spec: Spec, program: P.Expr, result: Result) -> bool:
        """Whether the search may stop on a program that meets the
        examples. With no reading of the request: yes, if the round trip
        does not refuse it. With one, the request is for a function of
        its inputs and a few examples are not the judge of that -- a
        constant meets one example of anything -- so what meets them is
        kept (`_hits`) with where it came from, and the search goes on
        until something better founded than it has is found (`_settled`);
        the answer is chosen among them (`_chosen`)."""
        if not self._shows(spec, program, result):
            self._note("refused", program=program.source(),
                       why="lacks what the reading of the request is sure of")
            return False
        if not self._judged(spec):
            self._note("meets", program=program.source(), stage=self._stage)
            return True
        if "strict" in self._moves and "!" in self._behaviour(
                spec, [program])[program.source()]:
            # fail loudly (X×B): one that throws beyond the examples is not
            # an answer to something that costs this much when wrong
            result.rejected += 1
            self._note("refused", program=program.source(),
                       why="throws beyond the examples (fail loudly)")
            return False
        # a parameter or a constant is what the search began with,
        # wherever it turns up (inside a proposal too) -- unless the
        # function takes nothing: then a constant is all it can be
        stage = ("pool" if program.kind in ("param", "const") and spec.params
                 else self._stage)
        self._hits.append((program, stage, result.evaluated))
        self._note("meets", program=program.source(), stage=stage)
        return self._settled()

    def _found(self) -> bool:
        """Judged, and something that meets the examples has been found
        (not only a value the search began with)."""
        return any(stage != "pool" for _, stage, _ in self._hits)

    def _settled(self, reached: int | None = None) -> bool:
        """Whether to stop looking. While the decoder's work is being read
        (whole proposals, then those repaired, then their parts) nothing
        stops part-way: at the end of each, if something founded at least
        that well (`RANK` <= `reached`) meets the examples. After them, at
        the first program the search finds.

        Four eyes (the risk matrix, B: a wrong answer costs much): not until
        two programs, apart, do the same beyond the examples."""
        if reached is not None:
            done = any(RANK.get(stage, FOUND) <= reached
                       for _, stage, _ in self._hits)
        else:
            done = self._stage not in ("pool", "proposed", "edited", "part")
        if done and "four_eyes" in self._moves:
            return self._confirmed(self._spec)
        return done

    def _chosen(self, spec: Spec) -> tuple:
        """(program, route, confirmed): of the programs that met the
        examples, the best founded (`RANK`) -- one the decoder wrote for
        this request, then one of its near misses repaired, then what the
        search found from the examples alone, and last a value it began
        with. Among equals, the first. Confirmed where another program,
        apart from it, does what it does beyond the examples.

        (Measured on MBPP dev and left out for every request: ranking them
        by how likely the reader of meaning finds what each is made of, or
        how it behaves beyond the examples, or taking the behaviour most of
        the decoder's samples agree on, chooses no better than the first.)

        Where the risk matrix asks: a program that throws on no probe over
        one that does (S, examples first), and the behaviour most programs
        agree on (U, clarify; B, four eyes)."""
        hits = self._hits
        moves = self._moves
        if "total" in moves:
            doing = self._behaviour(spec, [one[0] for one in hits])
            total = [one for one in hits if "!" not in doing[
                one[0].source()]]
            hits = total or hits
        groups = self._agreeing(spec, hits)

        def rank(one):
            return RANK.get(one[1], FOUND)

        if groups and ("agree" in moves or "four_eyes" in moves):
            group = max(groups.values(), key=lambda g: (
                len(self._authors(spec, g)),
                -min(rank(one) for one in g.values())))
            best = min(group.values(), key=rank)
        else:
            best = min(hits, key=rank)
        confirmed = any(best[0].source() in group
                        and len(self._authors(spec, group)) >= 2
                        for group in groups.values())
        # how it was chosen: every behaviour beyond the examples, and who
        # does it
        self._note("chosen", program=best[0].source(), stage=best[1],
                   confirmed=confirmed,
                   by=("agreement" if groups and (
                       "agree" in moves or "four_eyes" in moves)
                       else "where it came from"),
                   behaviours=[{"does": list(behaviour)[:16],
                                "authors": sorted(self._authors(spec, group)),
                                "programs": [{"program": source,
                                              "stage": stage}
                                             for source, (_, stage)
                                             in group.items()]}
                               for behaviour, group in groups.items()])
        return best[0], best[1] if best[1] != "pool" else "meet", confirmed

    def _edited(self, spec: Spec, result: Result) -> None:
        """No proposal meets the examples: each is a program that exists
        and is wrong, and is repaired as one (rung 4's edits, in its own
        text), the examples its cases. Only with more than one example:
        with one, an edit that fits it is rarely the fix (measured: 1 of
        12 right; with two, 4 of 10)."""
        from research.v696 import editing
        from research.v696.parse import parse
        for source in spec.sources[:EDITED]:
            bug = editing.Bug(spec.name, source, spec.entry,
                              list(spec.params), spec.returns,
                              [list(one) for one in spec.cases],
                              list(spec.outputs))
            fix = editing.repair(bug, budget=EDITS, rewrite=False)
            result.evaluated += fix.tried
            if fix.route != "fixed":
                continue
            tree = parse(fix.source, spec.entry, spec.params)
            if tree is not None:
                self._check(spec, [tree], result)
            if self._settled(RANK["edited"]):
                return

    def _shows(self, spec: Spec, program: P.Expr, result: Result) -> bool:
        """The round trip: a program that meets the examples is read back
        into behaviour exactly -- by running it on the examples and on
        inputs varied from them (`meaning.probes`) -- and refused if it
        lacks what the reading of the request is sure of."""
        if not spec.expected:
            return True
        from research.v696 import meaning as M
        bar = GOLDEN if "golden" in self._moves else STRONG
        sure = {one for one, p in spec.expected["behaviour"].items()
                if p >= bar and M.checkable(one)}
        if not sure:
            return True
        # the probes the reading was measured with: an edge (an empty
        # input) is where "longer than its input" need not hold
        cases = self._beyond(spec, moved=False)
        row = checker().values(spec.names, cases, [program.source()],
                               prelude=P.prelude([program]))[0] \
            if cases else []
        pairs = list(spec.examples) + [
            (case, one["value"]) for case, one in zip(cases, row)
            if "error" not in one]
        if sure <= M.behaviour(pairs):
            return True
        result.rejected += 1
        return False

    def _prior(self, spec: Spec, op) -> float:
        """How much the reading of the request expects this operator."""
        if not spec.expected:
            return 0.0
        from research.v696.meaning import word
        return PRIOR * spec.expected["uses"].get(word(op), 0.0)

    # 2b, 2c, 2d
    def _reachable(self, ops, goal: str) -> dict:
        """The backward trie at type level: for each type, how few steps
        from it the goal can be reached (regression over needs/gives)."""
        steps = {goal: 0}
        changed = True
        while changed:
            changed = False
            for op in ops:
                if op.gives not in steps:
                    continue
                for need in op.needs:
                    if steps.get(need, 99) > steps[op.gives] + 1:
                        steps[need] = steps[op.gives] + 1
                        changed = True
        return steps

    def _meet(self, spec: Spec, result: Result) -> P.Expr | None:
        ops = self._ops(spec)
        goal = spec.returns
        steps = self._reachable(ops, goal) if self.switches.coarse else {}
        # The forward trie: every expression kept is the first with its
        # values on the examples; a later one with the same is access.
        forward = C.Equivalence()
        kept: dict = defaultdict(list)          # type -> [(expr, row)]
        near = []
        # Judged, two expressions are one only if they also do the same
        # beyond the examples: a few examples tell few programs apart, and
        # the first with their values would hide every other.
        beyond = self._beyond(spec) if self._judged(spec) else []
        shown = len(spec.cases)
        self._stage = "pool"

        def admit(exprs: list) -> P.Expr | None:
            rows = self._evaluate(spec, exprs, beyond)
            result.evaluated += len(exprs)
            for expr, whole in zip(exprs, rows):
                row = whole[:shown]
                if any("error" in one for one in row):
                    continue
                signature = [expr.type] + [
                    _key(one["value"]) if "value" in one else "!"
                    for one in whole]
                if not forward.admit(signature):
                    # the same as something kept -- but what the decoder
                    # wrote for this request is weighed as its own, though
                    # a constant does the same wherever it was tried
                    if self._stage == "proposed" and expr.type == goal \
                            and _matches(row, spec.outputs) \
                            and self._accepts(spec, expr, result):
                        return expr
                    continue
                kept[expr.type].append((expr, row))
                if expr.type == goal:
                    if _matches(row, spec.outputs):
                        if self._accepts(spec, expr, result):
                            return expr
                        continue
                    hits = _near(row, spec.outputs)
                    if hits:
                        near.append((hits, expr))
            return None

        # What was kept, by type and by the level it was made at, so a
        # level is grown only from combinations that include the last one.
        levels: dict = defaultdict(lambda: defaultdict(list))
        admitted = {"level": 0}
        #: source -> how far it already resembles the answer
        promise: dict = {}
        base_admit = admit

        def admit(exprs: list) -> P.Expr | None:          # noqa: F811
            before = {kind: len(kept[kind]) for kind in list(kept)}
            found = base_admit(exprs)
            for kind in list(kept):
                fresh = kept[kind][before.get(kind, 0):]
                for expr, row in fresh:
                    promise[expr.source()] = relevance(row, spec.outputs)
                if self.switches.coarse:
                    # Attention: what already resembles the answer is grown
                    # first (the backward trie over values).
                    fresh = sorted(fresh, key=lambda item: -relevance(
                        item[1], spec.outputs))
                levels[kind][admitted["level"]].extend(
                    expr for expr, _ in fresh)
            return found

        found = admit(self._pool(spec))
        if found is not None:
            result.route = "meet"
            return found
        if self.switches.proposals and spec.proposals:
            # The decoder's programs, admitted like anything grown. Whole
            # first: one that meets the examples (and the round trip) is
            # the answer. Then -- judged, and none did -- each repaired by
            # edits, as a program that exists and is wrong. Then every
            # part of them: a part may meet the examples itself, one that
            # nearly does is a near miss for repair, and all are in the
            # forward trie for the search to compose with. What the
            # decoder got wrong is searched.
            whole, parts, seen = [], [], set()
            for tree in spec.proposals:
                if tree.kind in ("apply", "param", "const") \
                        and tree.source() not in seen:
                    seen.add(tree.source())
                    whole.append(tree)
            for tree in spec.proposals:
                for one in _subtrees(tree):
                    if one.kind in ("apply", "param", "const") \
                            and one.source() not in seen:
                        seen.add(one.source())
                        parts.append(one)
            self._stage = "proposed"
            found = admit(whole)
            if found is None and self._settled(RANK["proposed"]):
                return self._hits[-1][0]
            if found is None and self._judged(spec) \
                    and len(spec.examples) > 1 and spec.sources:
                self._stage = "edited"
                self._edited(spec, result)
                if self._settled(RANK["edited"]):
                    return self._hits[-1][0]
            if found is None:
                self._stage = "part"
                found = admit(parts)
            if found is not None:
                result.route = "proposed"
                return found
            if self._settled(FOUND):
                return self._hits[-1][0]
        self._stage = "meet"
        forward_forms = []
        if self.switches.forms:
            # Deduction pushes subgoals, and a subgoal is a search of its
            # own: it waits until the first level -- cheap, and enough for
            # much -- has been grown and checked (below, at level 2).
            # Checked forms over what is in hand, grown with the first
            # level: a form whose whole output meets the spec is found by
            # the meet, one that does not may be composed further.
            from research.v696 import forms as F
            literals = [one for one in self._pool(spec)
                        if one.kind == "const"]
            for kind in list(kept):
                if kind.endswith("[]"):
                    for expr, _ in kept[kind][:4]:
                        forward_forms += F.applied(spec, expr, literals)

        def combinations(op: P.Op, depth: int):
            """Argument tuples with at least one argument made at the last
            level: the first such is the pivot; those before it are older,
            those after it any level up to the last."""
            last = depth - 1
            for pivot in range(len(op.needs)):
                parts = []
                for index, need in enumerate(op.needs):
                    made = levels.get(need, {})
                    if index == pivot:
                        part = made.get(last, [])
                    elif index < pivot:
                        part = [one for lvl in range(last)
                                for one in made.get(lvl, [])]
                    else:
                        part = [one for lvl in range(last + 1)
                                for one in made.get(lvl, [])]
                    parts.append(part[:80])
                if all(parts):
                    yield from itertools.product(*parts)

        for depth in range(1, self.depth + 1):
            admitted["level"] = depth
            self._note("level", depth=depth, kept={
                kind: len(made) for kind, made in kept.items() if made})
            if forward_forms and depth == 1 and not self.switches.coarse:
                for start in range(0, len(forward_forms), BATCH):
                    found = admit(forward_forms[start:start + BATCH])
                    if found is not None:
                        result.route = "meet"
                        return found
                forward_forms = []
            if self.switches.forms and depth == 2:
                # Lists made at the first level (`s.split("")`) are
                # receivers too.
                self._stage = "deduced"
                found = self._deduce(spec, kept, result)
                self._stage = "meet"
                if found is not None:
                    result.route = "deduced"
                    return found
            if self.switches.coarse:
                # Attention across operators: every candidate of this
                # level in one queue, the most promising first -- arguments
                # that already resemble the answer, an operator that gives
                # the wanted type, one the learned control prefers.
                # Checked forms compete with every other candidate of the
                # first level for the budget, by the same attention.
                found = self._attend(spec, ops, steps, depth, combinations,
                                     promise, admit, result,
                                     forward_forms if depth == 1 else ())
                if found is not None:
                    result.route = "meet"
                    return found
                if result.evaluated >= self.budget:
                    break
                continue
            new = []
            for op in ops:
                if self.switches.coarse:
                    # Types before values: grown only if the goal can
                    # still be reached from what it gives.
                    if steps.get(op.gives, 99) > self.depth - depth:
                        continue
                for args in combinations(op, depth):
                    new.append(self._build(op, args))
                    if len(new) >= BATCH:
                        found = admit(new)
                        new = []
                        if found is not None:
                            result.route = "meet"
                            return found
                    if result.evaluated >= self.budget:
                        break
                if result.evaluated >= self.budget:
                    break
            found = admit(new)
            if found is not None:
                result.route = "meet"
                return found
            if result.evaluated >= self.budget:
                break
        if self._found():
            # judged, and something meets the examples: what repairing and
            # induction look for has been found
            return None
        if self.switches.repair and near:
            self._stage = "repaired"
            found = self._repair(spec, result, near, kept)
            if found is not None:
                result.route = "repaired"
                return found
        if self.switches.forms:
            # The dearest subgoal last: a fold's step by induction, once
            # growing and repairing have found nothing -- with its own
            # allowance, declared, as repair has: the budget is spent by
            # then.
            budget = self.budget
            self.budget = result.evaluated + self.inductions
            self._stage = "deduced"
            try:
                found = self._deduce(spec, kept, result, ("reduce",))
            finally:
                self.budget = budget
            if found is not None:
                result.route = "deduced"
                return found
        return None

#: How many of each argument's most promising are combined, per operator,
    #: when attention orders a level.
    ATTENDED = 12

    def _attend(self, spec, ops, steps, depth, combinations, promise, admit,
                result, formed=()) -> P.Expr | None:
        queue = []
        for count, expr in enumerate(formed):
            # A form is as promising as what it goes over, and more if it
            # gives the wanted type.
            score = promise.get(expr.args[0].source(), 0.0) + (
                1.0 if expr.type == spec.returns else 0.0) + self._prior(
                spec, expr.op)
            queue.append((-score, len(ops), count, None, expr))
        features = spec.features() if self.switches.learned else ()
        for rank, op in enumerate(ops):
            if steps.get(op.gives, 99) > self.depth - depth:
                continue
            bonus = (1.0 if op.gives == spec.returns else 0.0) +                 self._prior(spec, op)
            if self.switches.learned and self.memory.specs:
                bonus += 0.1 * self.memory.utility(op, features)
            count = 0
            for args in combinations(op, depth):
                # As promising as its most promising part: a constant
                # beside it (`=== 0`) resembles nothing, and costs nothing.
                score = max(promise.get(arg.source(), 0.0) for arg in args)
                queue.append((-(score + bonus), rank, count, op, args))
                count += 1
                if count >= self.ATTENDED ** len(op.needs):
                    break
        queue.sort(key=lambda one: one[:3])
        for start in range(0, len(queue), BATCH):
            batch = [args if op is None else self._build(op, args)
                     for _, _, _, op, args in queue[start:start + BATCH]]
            found = admit(batch)
            if found is not None:
                return found
            if result.evaluated >= self.budget:
                return None
        return None

#: How many list receivers a level offers to deduction.
    RECEIVERS = 6

    def _deduce(self, spec, kept, result, kinds=("map", "filter")
                ) -> P.Expr | None:
        """Rung 2: push the spec into a form's hole over each list in hand,
        those most like the output first (`forms.deduced`). `kinds` are
        the forms tried: element by element while the meet grows, folds by
        induction (rung 3) only when it has found nothing."""
        from research.v696 import forms as F
        tried = self.__dict__.setdefault("_deduced", {}).setdefault(
            kinds, set())
        offered = []
        for kind in list(kept):
            if not kind.endswith("[]"):
                continue
            for expr, row in kept[kind]:
                if expr.source() in tried or any("error" in one
                                                 for one in row):
                    continue
                offered.append((relevance(row, spec.outputs), expr, row))
        # Occam first: the simplest receiver, then the most promising --
        # `xs` before `xs.filter(...)`, which would stack a form on a form.
        offered.sort(key=lambda one: (one[1].size, -one[0]))
        chosen = offered[:self.RECEIVERS]
        for _, expr, _ in chosen:
            tried.add(expr.source())
        for _, expr, row in chosen:
            if result.evaluated >= self.budget:
                return None
            found = F.deduced(self, spec, expr,
                              [one["value"] for one in row], result, kinds)
            if found is not None:
                return found
        return None

    def _repair(self, spec, result, near, kept) -> P.Expr | None:
        """Backward error-correction: for the nearest misses, replace one
        subtree at a time by a kept expression of its type."""
        near.sort(key=lambda one: -one[0])
        # Its own allowance, declared: beyond what repairing what the
        # search grows ever needs, and a bound on a near miss the search
        # was given whole (a program read from a file: thousands of nodes,
        # each subtree times each kept expression). Made as checked, not
        # all at once.
        spent = 0
        for _, expr in near[:5]:
            batch = []
            for path, sub in _paths(expr):
                for other, _ in kept.get(sub.type, [])[:150]:
                    if other.source() == sub.source():
                        continue
                    batch.append(_replace(expr, path, other))
                    if len(batch) < BATCH:
                        continue
                    hit = self._check(spec, batch, result)
                    spent += len(batch)
                    batch = []
                    if hit is not None:
                        return hit
                    if spent >= REPAIRS:
                        return None
            if batch:
                hit = self._check(spec, batch, result)
                spent += len(batch)
                if hit is not None:
                    return hit
        return None


def _paths(expr: P.Expr, path=()):
    yield path, expr
    for index, one in enumerate(expr.args):
        yield from _paths(one, path + (index,))


def _replace(expr: P.Expr, path, new: P.Expr) -> P.Expr:
    if not path:
        return new
    args = list(expr.args)
    args[path[0]] = _replace(args[path[0]], path[1:], new)
    return P.Expr(expr.type, expr.kind, expr.name, expr.value, expr.op,
                  tuple(args))


def _rebind(program: P.Expr, old: Spec, new: Spec) -> P.Expr | None:
    """A solved program re-bound to another spec's parameters, by type in
    order: `s` of the old one is `t` of the new, where both are strings."""
    mapping, used = {}, set()
    for name, kind in old.params:
        match = next((other for other, other_kind in new.params
                      if other_kind == kind and other not in used), None)
        if match is None:
            return None
        mapping[name] = match
        used.add(match)

    def walk(one):
        if one.kind == "param":
            # A lambda's own parameters are not the spec's: kept as named.
            return P.param(mapping.get(one.name, one.name), one.type)
        return P.Expr(one.type, one.kind, one.name, one.value, one.op,
                      tuple(walk(arg) for arg in one.args))

    rebound = walk(program)
    return rebound if rebound.type == new.returns else None

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
#: What the reader of meaning must be this sure of before a program that
#: meets the examples is refused for not showing it.
STRONG = 0.95
#: How much an operator the reader expects is worth in the attention queue.
PRIOR = 1.0


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

    @classmethod
    def all(cls) -> "Switches":
        return cls(True, True, True, True, True, True, True)

    def label(self) -> str:
        on = [name for name, value in vars(self).items() if value]
        return "+".join(on) or "baseline"


@dataclass
class Result:
    spec: str
    program: P.Expr | None = None
    #: recognized | deduced | meet | repaired | means-ends | unsolved
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

    # the pieces every route shares
    def _ops(self, spec: Spec) -> list:
        ops = list(P.library().ops)
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

    def _evaluate(self, spec: Spec, exprs: list) -> list:
        rows = []
        for start in range(0, len(exprs), BATCH):
            rows += checker().values(spec.names, spec.cases,
                                     [one.source() for one in
                                      exprs[start:start + BATCH]])
        return rows

    def _general(self, spec: Spec, program: P.Expr) -> bool:
        if not spec.hidden:
            return True
        row = checker().values(spec.names, [a for a, _ in spec.hidden],
                               [program.source()])[0]
        return _matches(row, [o for _, o in spec.hidden])

    def solve(self, spec: Spec) -> Result:
        started = time.time()
        self._deduced = set()
        result = Result(spec.name)
        found = None
        if self.switches.recognition:
            found = self._recognised(spec, result)
            if found is not None:
                result.route = "recognized"
        if found is None and self.switches.meet:
            found = self._meet(spec, result)
        elif found is None:
            found = self._means_ends(spec, result)
            if found is not None:
                result.route = "means-ends"
        result.program = found
        result.seconds = time.time() - started
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

    def _accepts(self, spec: Spec, program: P.Expr, result: Result) -> bool:
        """The round trip: a program that meets the examples is read back
        into behaviour exactly -- by running it on the examples and on
        inputs varied from them (`meaning.probes`) -- and refused if it
        lacks what the reading of the request is sure of."""
        if not spec.expected:
            return True
        from research.v696 import meaning as M
        sure = {one for one, p in spec.expected["behaviour"].items()
                if p >= STRONG and M.checkable(one)}
        if not sure:
            return True
        probes = self.__dict__.setdefault("_probes", {})
        if spec.name not in probes:
            probes[spec.name] = M.probes(spec.examples)
        cases = probes[spec.name]
        row = checker().values(spec.names, cases, [program.source()])[0]             if cases else []
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

        def admit(exprs: list) -> P.Expr | None:
            rows = self._evaluate(spec, exprs)
            result.evaluated += len(exprs)
            for expr, row in zip(exprs, rows):
                if any("error" in one for one in row):
                    continue
                signature = [expr.type] + [_key(one["value"]) for one in row]
                if not forward.admit(signature):
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
                found = self._deduce(spec, kept, result)
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
        if self.switches.repair and near:
            found = self._repair(spec, result, near, kept)
            if found is not None:
                result.route = "repaired"
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

    def _deduce(self, spec, kept, result) -> P.Expr | None:
        """Rung 2: push the spec into a form's hole over each list in hand,
        those most like the output first (`forms.deduced`)."""
        from research.v696 import forms as F
        tried = self.__dict__.setdefault("_deduced", set())
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
        for _, expr, row in offered[:self.RECEIVERS]:
            tried.add(expr.source())
            found = F.deduced(self, spec, expr,
                              [one["value"] for one in row], result)
            if found is not None:
                return found
        return None

    def _repair(self, spec, result, near, kept) -> P.Expr | None:
        """Backward error-correction: for the nearest misses, replace one
        subtree at a time by a kept expression of its type."""
        near.sort(key=lambda one: -one[0])
        for _, expr in near[:5]:
            paths = list(_paths(expr))
            candidates = []
            for path, sub in paths:
                for other, _ in kept.get(sub.type, [])[:150]:
                    if other.source() != sub.source():
                        candidates.append(_replace(expr, path, other))
            for start in range(0, len(candidates), BATCH):
                hit = self._check(spec, candidates[start:start + BATCH],
                                  result)
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

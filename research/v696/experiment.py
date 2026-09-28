"""The measurements `PLAN.md` promised: switches, routes, learning.

For each configuration of switches, the same tasks, the same budget:

    solved      met the examples it was shown
    general     met the hidden ones too (or MultiPL-E's own tests)
    routes      recognized / meet / repaired / means-ends
    evaluated   candidates run -- what the search cost

The learning mechanisms (recognition, learned control, chunks) need
something to have been learned from, so each configuration first solves a
training sequence -- the same sequence for all -- with memory kept, and
is then measured on tasks it has not seen. Dev seeds may be looked at;
`--held` runs held seeds, and is run once per phase.

    python -m research.v696.experiment                 dev, every config
    python -m research.v696.experiment --held          held, once
    python -m research.v696.experiment --multipl-e     HumanEval-TS
    ... --multipl-e --meaning meaning-unixcoder         each request read
                                                        first (`reader.py`)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass

from research.v696 import generating as G
from research.v696 import search as S
from research.v696.checker import checker
from research.v696.spec import Spec

CONFIGS = {
    "baseline": S.Switches(),
    "meet": S.Switches(meet=True, coarse=True),
    "meet+repair": S.Switches(meet=True, coarse=True, repair=True),
    "meet+learned": S.Switches(meet=True, coarse=True, learned=True),
    "meet+recognition": S.Switches(meet=True, coarse=True,
                                   recognition=True),
    "meet+chunks": S.Switches(meet=True, coarse=True, chunks=True),
    "meet+repair+forms": S.Switches(meet=True, coarse=True, repair=True,
                                    forms=True),
    "meet+repair+forms+proposals": S.Switches(
        meet=True, coarse=True, repair=True, forms=True, proposals=True),
    "all": S.Switches.all(),
}

#: The tasks: how many at each depth, for training and for measuring.
DEPTHS = (1, 2, 3)
TRAIN_EACH = 10
TEST_EACH = 10
#: Where the dev test tasks start, past the training ones.
DEV_TEST = 100_000


@dataclass
class Row:
    config: str
    solved: int = 0
    general: int = 0
    total: int = 0
    evaluated: int = 0
    seconds: float = 0.0
    routes: Counter = None
    #: met the examples, refused by the round trip with the meaning
    rejected: int = 0

    def line(self) -> str:
        routes = ", ".join(f"{name} {count}" for name, count in
                           sorted(self.routes.items()))
        return (f"{self.config:<18} solved {self.solved:>3}/{self.total:<3}"
                f" general {self.general:>3}  evaluated "
                f"{self.evaluated:>7}  {self.seconds:>6.0f}s  [{routes}]"
                + (f"  refused {self.rejected}" if self.rejected else ""))


def generated(held: bool) -> tuple:
    """(training tasks, test tasks), each a mix of depths."""
    train, test = [], []
    for depth in DEPTHS:
        train += G.tasks(TRAIN_EACH, depth)
        start = G.HELD if held else DEV_TEST
        found, seed = [], start
        while len(found) < TEST_EACH:
            one = G.task(seed, depth)
            if one is not None:
                found.append(one)
            seed += 1
        test += found
    return train, test


def generated2(held: bool, count: int = 30) -> tuple:
    """Rung 2's (training, test) tasks: a form in each."""
    train = G.tasks2(count, 1)
    test = G.tasks2(count, 1, start=G.HELD if held else DEV_TEST)
    return train, test


def run(config: str, train: list, test: list, budget: int) -> Row:
    solver = S.Solver(CONFIGS[config], budget=budget)
    for spec in train:
        solver.solve(spec)
    row = Row(config, routes=Counter())
    started = time.time()
    for spec in test:
        got = solver.solve(spec)
        row.total += 1
        row.solved += got.solved
        row.general += got.general if not spec.tests else _passes(spec, got)
        row.evaluated += got.evaluated
        row.rejected += got.rejected
        row.routes[got.route] += 1
    row.seconds = time.time() - started
    return row


# -- MultiPL-E as specs ----------------------------------------------------

def _literal(text: str):
    """A TypeScript literal as the value Node makes of it."""
    row = checker().values([], [[]], [text])[0][0]
    if "error" in row:
        raise ValueError(text)
    return row["value"]


def multipl_e(config: str = "humaneval-ts") -> list:
    """HumanEval-TS tasks whose examples can be read, as specs: the
    examples the prompt shows to search with, its own tests to judge."""
    from research.v696 import tasks
    from research.v696.teach_meaning import _english
    out = []
    for task in tasks.load(config):
        try:
            examples = [([_literal(f"[{args}]")][0], _literal(value))
                        for args, value in task.examples]
        except ValueError:
            continue
        if not examples and config == "mbpp-ts":
            # MBPP's prompts show none: its first test is its example.
            # (HumanEval's are its own; a test is never shown there.)
            from research.v696.meaning import test_pairs, values_of
            examples = [tuple(one) for one in values_of(
                test_pairs(task.tests))[:1]]
        if not examples:
            continue
        out.append(Spec(task.name, task.params, task.returns, examples,
                        tests=task.tests, entry=task.entry,
                        english=_english(task.prompt)))
    return out


def _passes(spec: Spec, got: S.Result) -> bool:
    if got.program is None:
        return False
    return checker().tests(spec.function(got.program) + "\n"
                           + spec.tests) is None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--held", action="store_true")
    parser.add_argument("--multipl-e", action="store_true")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--out", default="")
    parser.add_argument("--rung", type=int, default=1)
    parser.add_argument("--meaning", default="",
                        help="a reader of meaning in llm/ to read each "
                             "request with before searching")
    parser.add_argument("--sketcher", default="",
                        help="a decoder in llm/ to propose programs with "
                             "(needs --meaning)")
    options = parser.parse_args(argv)
    if options.multipl_e:
        train, test = generated(False)[0], multipl_e()
        print(f"HumanEval-TS: {len(test)} tasks with readable examples")
    elif options.rung == 2:
        train, test = generated2(options.held)
        print(f"rung 2 {'held' if options.held else 'dev'}: {len(train)} "
              f"training, {len(test)} test tasks")
    else:
        train, test = generated(options.held)
        print(f"{'held' if options.held else 'dev'}: {len(train)} training, "
              f"{len(test)} test tasks, depths {DEPTHS}")
    if options.meaning:
        from research.v696 import reader
        reader.expect(test, reader.LLM / options.meaning)
    if options.sketcher:
        from research.v696 import sketcher
        sketcher.proposals(sketcher.Sketcher(sketcher.LLM / options.sketcher),
                           test)
    rows = []
    for config in options.configs:
        row = run(config, train, test, options.budget)
        print(row.line(), flush=True)
        rows.append(row)
    if options.out:
        with open(options.out, "a", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({**vars(row),
                                      "routes": dict(row.routes)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

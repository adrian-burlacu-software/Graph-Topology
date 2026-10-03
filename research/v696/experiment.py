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
    ... --sketcher proposer                             and the decoder's
                                                        programs proposed
                                                        (sketcher-functions2)
    ... --risk estimators                               each request's six
                                                        risks read first, and
                                                        searched as the
                                                        matrix says (`risk.py`)
    ... --dev                                           MBPP dev instead (two
                                                        examples shown)
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
    #: judged: answers another program confirmed, and how many passed;
    #: the same for those none did
    confirmed: int = 0
    confirmed_pass: int = 0
    unconfirmed: int = 0
    unconfirmed_pass: int = 0
    #: high factors -> [passed, of] (`risk.ranked`)
    ranked: dict = None

    def line(self) -> str:
        routes = ", ".join(f"{name} {count}" for name, count in
                           sorted(self.routes.items()))
        return (f"{self.config:<18} solved {self.solved:>3}/{self.total:<3}"
                f" general {self.general:>3}  evaluated "
                f"{self.evaluated:>7}  {self.seconds:>6.0f}s  [{routes}]"
                + (f"  refused {self.rejected}" if self.rejected else "")
                + (f"  confirmed {self.confirmed_pass}/{self.confirmed}"
                   f" unconfirmed {self.unconfirmed_pass}/"
                   f"{self.unconfirmed}" if self.confirmed
                   or self.unconfirmed else "")
                + (f"  by high factors {self.ranked}" if self.ranked
                   else ""))


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


def generated3(held: bool, count: int = 30) -> tuple:
    """Rung 3's (training, test) tasks: a loop over a counted range."""
    train = G.tasks3(count)
    test = G.tasks3(count, start=G.HELD if held else DEV_TEST)
    return train, test


def run(config: str, train: list, test: list, budget: int) -> Row:
    solver = S.Solver(CONFIGS[config], budget=budget)
    for spec in train:
        solver.solve(spec)
    row = Row(config, routes=Counter(), ranked={})
    started = time.time()
    passed = []
    for spec in test:
        got = solver.solve(spec)
        general = got.general if not spec.tests else _passes(spec, got)
        passed.append(general)
        row.total += 1
        row.solved += got.solved
        row.general += general
        row.evaluated += got.evaluated
        row.rejected += got.rejected
        row.routes[got.route] += 1
        if got.solved and solver._judged(spec):
            if got.confirmed:
                row.confirmed += 1
                row.confirmed_pass += general
            else:
                row.unconfirmed += 1
                row.unconfirmed_pass += general
        if spec.tests:
            # one line a task, as it goes: a long run shows where it is
            risk = ("" if spec.risk is None else " " + "".join(
                f"{factor}{spec.risk[factor]}" for factor in "DPSUXB"))
            print(f"  {row.total:>3}/{len(test)} {spec.name[:44]:44} "
                  f"{got.route:9} {'passes' if general else '-':6} "
                  f"{'confirmed' if got.confirmed else '':9}{risk} "
                  f"({time.time() - started:.0f}s)", flush=True)
    row.seconds = time.time() - started
    from research.v696 import risk as R
    row.ranked = R.ranked(test, passed)
    return row


# -- MultiPL-E as specs ----------------------------------------------------

def _literal(text: str):
    """A TypeScript literal as the value Node makes of it."""
    row = checker().values([], [[]], [text])[0][0]
    if "error" in row:
        raise ValueError(text)
    return row["value"]


def multipl_e(config: str = "humaneval-ts", shown: int = 1) -> list:
    """HumanEval-TS tasks whose examples can be read, as specs: the
    examples the prompt shows to search with, its own tests to judge.
    MBPP's prompts show none: its first `shown` tests are its examples."""
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
                test_pairs(task.tests))[:shown]]
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
                             "(needs --meaning); 'proposer' for the chosen "
                             "one (sketcher.PROPOSER)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="ask the decoder again, up to this many times, "
                             "where nothing it wrote meets the examples")
    parser.add_argument("--dev", action="store_true",
                        help="MBPP dev (two examples shown) for "
                             "--multipl-e: what may be looked at")
    parser.add_argument("--risk", default="",
                        choices=("", "estimators", "oracle"),
                        help="each request's six risks -- read by the "
                             "estimators, or its verified program's labels "
                             "-- and its search as the matrix says")
    parser.add_argument("--moves", default="",
                        help="only these of the matrix's moves "
                             "(comma-separated; risk.MOVES)")
    parser.add_argument("--cache", default="",
                        help="a file to keep what the decoders wrote in, "
                             "so a run measured again asks them nothing "
                             "it asked before (`sketcher.CACHE`)")
    parser.add_argument("--part", default="",
                        help="k/n: only the k-th of n equal runs of the "
                             "test tasks -- a long run in parts")
    options = parser.parse_args(argv)
    if options.rung == 4:
        main4(options.held)
        return 0
    if options.rung == 5:
        print(f"rung 5 {'held' if options.held else 'dev'}:",
              rung5(options.held), flush=True)
        return 0
    if options.multipl_e and options.dev:
        from research.v696.teach_meaning import split
        train = generated(False)[0]
        test = [one for one in multipl_e("mbpp-ts", shown=2)
                if split(one.name) == "dev"]
        print(f"MBPP dev: {len(test)} tasks, two examples shown")
    elif options.multipl_e:
        train, test = generated(False)[0], multipl_e()
        print(f"HumanEval-TS: {len(test)} tasks with readable examples")
    elif options.rung == 3:
        train, test = generated3(options.held)
        print(f"rung 3 {'held' if options.held else 'dev'}: {len(train)} "
              f"training, {len(test)} test tasks")
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
    if options.part:
        k, n = map(int, options.part.split("/"))
        test = test[(k - 1) * len(test) // n:k * len(test) // n]
        print(f"part {k} of {n}: {len(test)} tasks", flush=True)
    if options.risk:
        from research.v696 import risk as R
        if options.risk == "oracle":
            print(f"risk from labels: {R.oracle(test)} of {len(test)}")
        else:
            R.assess(test)
        allowed = (frozenset(options.moves.split(",")) if options.moves
                   else R.MOVES)
        for spec in test:
            spec.moves = R.moves(spec.risk, allowed)
        corners = Counter(len(spec.moves.cells) for spec in test)
        print(f"shaded corners per request: {dict(sorted(corners.items()))}",
              flush=True)
    if options.sketcher:
        from pathlib import Path
        from research.v696 import sketcher
        if options.cache:
            sketcher.CACHE = Path(options.cache)
        said = {"proposer": sketcher.PROPOSER,
                "proposers": sketcher.PROPOSERS}.get(options.sketcher,
                                                     options.sketcher)
        # several decoders: each after the last, where it found nothing
        for at, name in enumerate(said.split(",")):
            model = sketcher.Sketcher(sketcher.LLM / name)
            sketcher.proposals(model, test, rounds=options.rounds,
                               keep=at > 0)
            torch = model.torch
            del model
            torch.cuda.empty_cache()
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



# -- rung 4: editing a program that exists ---------------------------------

def rung4(bugs_: list, budget: int = 8000) -> dict:
    """Each bug repaired by edits (`editing.repair`), and solved from
    scratch by the search on the same cases, both judged by its tests."""
    from research.v696 import editing
    routes, edits_made, tried = Counter(), Counter(), 0
    scratch = scratch_ok = 0
    solver = S.Solver(CONFIGS["meet+repair+forms"], budget=budget)
    started = time.time()
    for bug in bugs_:
        fix = editing.repair(bug)
        routes[fix.route] += 1
        tried += fix.tried
        if fix.route == "fixed":
            edits_made[len(fix.edits)] += 1
        spec = Spec(bug.name, bug.params, bug.returns,
                    [(list(args), want) for args, want in
                     zip(bug.cases, bug.wanted)],
                    tests=bug.tests, entry=bug.entry)
        got = solver.solve(spec)
        scratch += got.solved
        if got.solved:
            scratch_ok += (_passes(spec, got) if bug.tests
                           else True)
        # one line a bug, as it goes: a long run shows where it is
        print(f"  {len(routes and bugs_[:sum(routes.values())]):>3}/"
              f"{len(bugs_)} {bug.name[:40]:40} {fix.route:9} "
              f"{len(fix.edits)} edit(s)  scratch "
              f"{'solved' if got.solved else '-':6} "
              f"({time.time() - started:.0f}s)", flush=True)
    return {"bugs": len(bugs_), "routes": dict(routes),
            "edits": dict(edits_made), "programs run": tried,
            "scratch solved": scratch, "scratch passes tests": scratch_ok,
            "seconds": round(time.time() - started)}


def main4(held: bool) -> None:
    from research.v696 import bugs
    suites = {"generated": bugs.generated(held)}
    if held:
        suites["humanevalfix"] = bugs.humanevalfix()
    for name, found in suites.items():
        print(f"rung 4 {'held' if held else 'dev'} {name}:",
              rung4(found), flush=True)



# -- rung 5: projects -------------------------------------------------------

def rung5(held: bool) -> dict:
    """Each project task by its route (5b use, 5c fix, 5d change), and by
    the baseline without the rung's mechanism, both judged by the
    project's own tests. One line a task, as it goes."""
    from research.v696 import changing as C, projects
    found = projects.load(held)
    out = {}
    started = time.time()
    for task in found:
        if task.kind == "use":
            mine, base = C.use(task), C.use(task, with_project=False)
        elif task.kind == "bug":
            mine, base = C.fix(task), C.use(task, with_project=False)
        else:
            mine, base = C.change(task), C.fix(task)
        row = out.setdefault(task.kind, Counter())
        row["tasks"] += 1
        row["solved"] += mine.route == "solved"
        row["baseline solved"] += base.route == "solved"
        row["edits"] += len(mine.edits) if mine.route == "solved" else 0
        row["impasses"] += mine.impasses
        row["files touched"] += mine.files_touched
        print(f"  {task.name:24} {mine.route:9} {len(mine.edits)} edit(s) "
              f"{mine.impasses} impasse(s) {mine.files_touched} file(s)  "
              f"baseline {base.route:9} ({time.time() - started:.0f}s)",
              flush=True)
    return {kind: dict(row) for kind, row in out.items()}


if __name__ == "__main__":
    sys.exit(main())

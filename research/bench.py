"""One command, one scoreboard, a history.

Every suite the architecture is measured by, run and printed as one table,
and appended to `research/BENCH.jsonl` with the commit it ran on, so a
change that makes something else worse shows up as a line that moved.

    python -m research.bench                   every quick suite
    python -m research.bench --only planning   one suite
    python -m research.bench --full            and the slow ones (PIQA dev)
    python -m research.bench --history         what each suite was, by commit

A suite returns `{"headline": "...", "metrics": {...}}`; the headline is
what the table shows, the metrics what the history keeps. The suites (as
`v696/PLAN.md` lists them):

    planning      v691's worlds from new seeds, against breadth-first
                  search's shortest plan
    designer      v694's banks: sensible, and also simplest
    proposer      v694's learned order: ways fitted with it and without, on
                  the same designs -- what it saves, not what it agrees with
    actions       v695's held-out movement goals
    piqa          v695 on PIQA dev (slow: --full)
    code-*        v696's code suites (added as they are built)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "research" / "BENCH.jsonl"

#: Seeds the planning suite samples from: fixed, so two runs compare, and
#: not the ones v691 was built on (0).
PLANNING_SEED = 696
PLANNING_COUNT = 20


def _planning() -> dict:
    from research.v691 import acting
    from research.v691.problems import SAMPLERS, SUITE
    metrics, parts = {}, []
    for domain in sorted(SAMPLERS):
        problems = SAMPLERS[domain](PLANNING_COUNT, PLANNING_SEED)
        if domain == "blocks":
            problems = SUITE + problems
        got = acting.measure(problems)
        metrics[domain] = {"solved": got.solved, "shortest": got.shortest,
                           "total": got.total, "subgoals": got.subgoals,
                           "fired": got.fired}
        parts.append(f"{domain} {got.solved}/{got.total} "
                     f"({got.shortest} shortest)")
    return {"headline": ", ".join(parts), "metrics": metrics}


def _designer() -> dict:
    from research.v694 import knowing as K
    from research.v694.bank import BANK, HELD, HELD2, run
    K.index()
    metrics, parts = {}, []
    for name, bank in (("bank", BANK), ("held", HELD), ("held2", HELD2)):
        got = run(bank, verbose=False)
        metrics[name] = {key: got[key] for key in ("right", "simple",
                                                   "total")}
        parts.append(f"{name} {got['right']}/{got['total']}")
    return {"headline": ", ".join(parts), "metrics": metrics}


def _fitted(found: dict) -> int:
    return sum(len(part.fitted) for _, design, _, _ in found["rows"]
               for part in design.parts)


def _proposer() -> dict:
    from research.v694 import knowing as K
    from research.v694.bank import BANK, run
    from research.v694.proposer import Proposer
    K.index()
    learned = Proposer.load()
    if learned is None:
        return {"headline": "no proposer trained", "metrics": {}}
    table = run(BANK, verbose=False)
    ordered = run(BANK, order=learned.order, verbose=False)
    same = sum(one[1].said() == two[1].said()
               for one, two in zip(table["rows"], ordered["rows"]))
    saved = _fitted(table) - _fitted(ordered)
    return {"headline": f"fitted {_fitted(ordered)} vs {_fitted(table)} "
                        f"(saves {saved}); same designs "
                        f"{same}/{len(BANK)}",
            "metrics": {"fitted": _fitted(ordered),
                        "fitted_table": _fitted(table),
                        "same": same, "total": len(BANK)}}


def _actions() -> dict:
    from research.v695 import achieving as A
    from research.v695 import held_out as H
    goals = H.goals()
    real = A._done_for
    hits = proposed = 0
    hidden = [""]

    def without(goal, found):
        real(goal, found)
        found.pop(hidden[0], None)

    A._done_for = without
    try:
        for goal, verb in goals:
            hidden[0] = verb
            found = A.actions(goal)
            if found:
                proposed += 1
                hits += verb in [one.verb for one in found[:3]]
    finally:
        A._done_for = real
    return {"headline": f"hit@3 {hits}/{len(goals)} held-out movement "
                        f"goals ({proposed} proposed)",
            "metrics": {"hits": hits, "proposed": proposed,
                        "total": len(goals)}}


def _piqa() -> dict:
    from research.v695 import piqa
    got = piqa.run(piqa.load("valid"), cascaded=True)
    return {"headline": f"dev {got.overall:.1%} overall "
                        f"({got.coverage:.0%} answered)",
            "metrics": {"overall": got.overall, "coverage": got.coverage,
                        "precision": got.precision, "total": got.total}}


#: name -> (run, slow)
SUITES = {
    "planning": (_planning, False),
    "designer": (_designer, False),
    "proposer": (_proposer, False),
    "actions": (_actions, False),
    "piqa": (_piqa, True),
}


def register(name: str, run, slow: bool = False) -> None:
    """A later layer's suite (v696's code suites)."""
    SUITES[name] = (run, slow)


def _commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, cwd=ROOT)
        return head.stdout.strip() + ("+" if dirty.stdout.strip() else "")
    except OSError:
        return "?"


def _load_layers() -> None:
    """Suites later layers register (`register`), imported if present."""
    try:
        import research.v696.suites  # noqa: F401
    except ImportError:
        return
    # Run as `-m`, this module is `__main__`, and the layers registered
    # with `research.bench`: the same table, under another name.
    import research.bench as registry
    SUITES.update(registry.SUITES)


def history() -> None:
    if not HISTORY.exists():
        print("no history yet")
        return
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        print(f"{row['when']}  {row['commit']:<10} {row['suite']:<14} "
              f"{row['headline']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--no-record", action="store_true",
                        help="print, and keep nothing")
    options = parser.parse_args(argv)
    if options.history:
        history()
        return 0
    _load_layers()
    commit = _commit()
    names = options.only or [name for name, (_, slow) in SUITES.items()
                             if options.full or not slow]
    print(f"commit {commit}\n")
    print(f"{'suite':<14} {'time':>7}  result")
    print("-" * 78)
    for name in names:
        run, _ = SUITES[name]
        started = time.time()
        try:
            found = run()
        except Exception as bad:                  # noqa: BLE001
            found = {"headline": f"ERROR {type(bad).__name__}: {bad}",
                     "metrics": {}}
        spent = time.time() - started
        print(f"{name:<14} {spent:>6.0f}s  {found['headline']}", flush=True)
        if not options.no_record:
            with HISTORY.open("a", encoding="utf-8") as out:
                out.write(json.dumps({
                    "when": time.strftime("%Y-%m-%d %H:%M"),
                    "commit": commit, "suite": name,
                    "headline": found["headline"],
                    "metrics": found["metrics"],
                    "seconds": round(spent, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

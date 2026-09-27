"""Phase 3: the planner, with recognition before search.

The same `cognition.Recognizer` the code search uses, over v691's
problems. A problem's features are its facts with the objects renamed by
the part they play -- the first object the goal mentions is `o0`, the next
`o1` -- so two problems of the same shape share features whatever their
objects are called. A solved plan is kept in the same terms. A new
problem walks the trie; each plan it reaches is re-bound to this
problem's objects and **checked exactly** (`acting.valid`) before it is
used. Only when nothing recognised checks does the executive search.

    python -m research.v696.planning      each domain, plain and recognising
"""
from __future__ import annotations

import sys
import time

from research.v691 import acting
from research.v691.problems import SAMPLERS
from research.v696.cognition import Recognizer

#: How many recognised plans are checked before searching instead.
CHECKED = 10


def roles(problem) -> dict:
    """object -> role, by first appearance in the goal, then the start."""
    objects = set(problem.objects) if problem.objects else set()
    order: dict = {}
    for fact in sorted(problem.goal) + sorted(problem.start):
        for word in fact.split()[1:]:
            if (not objects or word in objects) and word not in order:
                order[word] = f"o{len(order)}"
    return order


def _said(fact: str, mapping: dict) -> str:
    return " ".join(mapping.get(word, word) for word in fact.split())


def features(problem, mapping: dict) -> frozenset:
    return frozenset({f"goal {_said(one, mapping)}" for one in problem.goal}
                     | {f"start {_said(one, mapping)}"
                        for one in problem.start})


def recalled(problem, recognizer: Recognizer) -> tuple | None:
    """A recognised plan re-bound to this problem that checks, or None."""
    mapping = roles(problem)
    back = {role: name for name, role in mapping.items()}
    by_name = {action.name: action for action in problem.actions}
    for abstract in recognizer.recognise(features(problem, mapping)
                                         )[:CHECKED]:
        plan = []
        for step in abstract:
            action = by_name.get(_said(step, back))
            if action is None:
                break
            plan.append(action)
        else:
            if acting.valid(plan, problem.start, problem.goal):
                return tuple(plan)
    return None


def solve(problem, recognizer: Recognizer | None) -> dict:
    started = time.time()
    if recognizer is not None:
        plan = recalled(problem, recognizer)
        if plan is not None:
            return {"solved": True, "route": "recognized", "length":
                    len(plan), "fired": 0, "subgoals": 0,
                    "seconds": time.time() - started}
    got = acting.solve(problem, optimal=False)
    if recognizer is not None and got.solved:
        mapping = roles(problem)
        recognizer.remember(features(problem, mapping),
                            tuple(_said(one.name, mapping)
                                  for one in got.plan))
    return {"solved": got.solved, "route": "searched",
            "length": len(got.plan), "fired": got.search.fired,
            "subgoals": got.search.subgoals,
            "seconds": time.time() - started}


def measure(domain: str, count: int, seed: int, recognizer) -> dict:
    out = {"solved": 0, "recognized": 0, "fired": 0, "subgoals": 0,
           "length": 0, "total": 0, "seconds": 0.0}
    for problem in SAMPLERS[domain](count, seed):
        got = solve(problem, recognizer)
        out["total"] += 1
        out["solved"] += got["solved"]
        out["recognized"] += got["route"] == "recognized"
        for key in ("fired", "subgoals", "length", "seconds"):
            out[key] += got[key]
    return out


def main(argv=None) -> int:
    count = 60
    for domain in sorted(SAMPLERS):
        for label, recognizer in (("plain", None),
                                  ("recognising", Recognizer())):
            got = measure(domain, count, 696, recognizer)
            print(f"{domain:<9} {label:<12} solved {got['solved']}/"
                  f"{got['total']} recognized {got['recognized']:>3} "
                  f"fired {got['fired']:>6} subgoals {got['subgoals']:>5} "
                  f"steps {got['length']:>4} {got['seconds']:.1f}s",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""v696's suites on the scoreboard (`research/bench.py`).

    code-generated   dev test tasks at depths 1-3 after a training
                     sequence, every mechanism on: solved, general, routes
    code-humaneval   HumanEval-TS, every mechanism on (slow: --full)
"""
from __future__ import annotations

from research import bench


def _generated() -> dict:
    from research.v696 import experiment as E
    train, test = E.generated(held=False)
    row = E.run("all", train, test, budget=3000)
    routes = dict(row.routes)
    return {"headline": f"{row.solved}/{row.total} solved, {row.general} "
                        f"general; routes {routes}",
            "metrics": {"solved": row.solved, "general": row.general,
                        "total": row.total, "evaluated": row.evaluated,
                        "routes": routes}}


def _humaneval() -> dict:
    from research.v696 import experiment as E
    train = E.generated(held=False)[0]
    row = E.run("all", train, E.multipl_e(), budget=5000)
    return {"headline": f"passes its tests {row.general}/{row.total} "
                        f"(meets the shown examples {row.solved})",
            "metrics": {"passed": row.general, "examples": row.solved,
                        "total": row.total, "routes": dict(row.routes)}}


def _designer_recognition() -> dict:
    """v694's three banks in turn, recognition learning as it goes: the
    designs imagined, against the plain designer's."""
    from research.v694 import designing, knowing as K
    from research.v694.bank import BANK, HELD, HELD2, score
    from research.v694.goals import Goal
    from research.v696.cognition import Recognizer
    K.index()
    out = {}
    for label, recognizer in (("plain", None), ("recognising",
                                                Recognizer())):
        fitted = sensible = shortcuts = 0
        for said, wants, scene, names, ways, means in BANK + HELD + HELD2:
            goal = Goal(tuple(wants), frozenset(scene), frozenset(names),
                        said=said)
            found = designing.design(goal, recognizer=recognizer)
            fitted += sum(len(part.fitted) for part in found.parts)
            shortcuts += sum(part.route == "recognized"
                             for part in found.parts)
            sensible += score(found, ways, means)[0]
        out[label] = {"fitted": fitted, "sensible": sensible,
                      "shortcuts": shortcuts}
    plain, mine = out["plain"], out["recognising"]
    return {"headline": f"fitted {mine['fitted']} vs {plain['fitted']}, "
                        f"sensible {mine['sensible']} vs "
                        f"{plain['sensible']}, {mine['shortcuts']} "
                        f"shortcuts",
            "metrics": out}


def _planning_recognition() -> dict:
    from research.v691.problems import SAMPLERS
    from research.v696 import planning
    from research.v696.cognition import Recognizer
    metrics, parts = {}, []
    for domain in sorted(SAMPLERS):
        plain = planning.measure(domain, 60, 696, None)
        mine = planning.measure(domain, 60, 696, Recognizer())
        metrics[domain] = {"plain": plain, "recognising": mine}
        parts.append(f"{domain} fired {mine['fired']}/{plain['fired']} "
                     f"({mine['recognized']} recognized, solved "
                     f"{mine['solved']}/{plain['solved']})")
    return {"headline": "; ".join(parts), "metrics": metrics}


bench.register("code-generated", _generated)
bench.register("designer-recog", _designer_recognition)
bench.register("planning-recog", _planning_recognition)
bench.register("code-humaneval", _humaneval, slow=True)

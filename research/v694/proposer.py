"""The learned proposer for ways: which to try first.

    python -m research.v694.proposer train      # llm/open-designer-proposer2
    python -m research.v694.proposer evaluate

v693's proposer, over ways instead of forms: one logistic model per way,
over what a clause is (`features`), predicting whether that way is the
one the simplest design takes. Trained on goals made from the store's own
rows (`generating.labelled`), labelled by the designer itself.

What it buys is time, not answers: Occam still decides what is chosen
(`designing._simpler`), and a proposer that guesses wrong costs a fit or
two. Guessing right means the first way imagined is the one kept.

The model is its own folder, `llm/open-designer-proposer2`, and nothing
else writes there (`regenerate.py`'s `open-designer-proposer` step).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from research.v694 import ways as Ways
from research.v694.goals import Clause, Goal, features as clause_features

LLM = Path(__file__).resolve().parents[2] / "llm"
#: Its own folder, never written over (`llm/` rule): the first, trained
#: before the `inside` way existed, is `open-designer-proposer` and is
#: kept as it was.
MODEL = LLM / "open-designer-proposer2" / "model.json"


def features(goal: Goal, clause: Clause) -> set:
    """The clause's features, and what the store and the language say of
    its verbs: whether people do it themselves, whether the verb names its
    tool."""
    from research.v694 import knowing as K
    out = set(clause_features(goal, clause))
    kind = goal.kind_of(clause.patient)
    for verb in clause.verbs():
        if K.person_can(verb, kind) >= Ways.SUPPORT / 2:
            out.add("person-can")
        if K.named_tools(verb):
            out.add("named-tool")
    return out


class Proposer:
    def __init__(self, weights: dict | None = None) -> None:
        self.weights = weights or {}

    def chance(self, way: str, seen: set) -> float:
        model = self.weights.get(way)
        if model is None:
            return 0.5
        score = model["bias"] + sum(model["w"].get(one, 0.0) for one in seen)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    def order(self, goal: Goal, clause: Clause) -> list:
        seen = features(goal, clause)
        return sorted((way.name for way in Ways.WAYS),
                      key=lambda name: -self.chance(name, seen))

    @classmethod
    def trained(cls, rows: list, epochs: int = 60, rate: float = 0.3,
                decay: float = 1e-3, seed: int = 694) -> "Proposer":
        rng = random.Random(seed)
        data = [(features(goal, goal.clauses[0]), best)
                for goal, best in rows]
        weights: dict = {}
        for way in Ways.WAYS:
            mine = [(seen, best == way.name) for seen, best in data]
            model = {"bias": 0.0, "w": {}}
            for _ in range(epochs):
                rng.shuffle(mine)
                for seen, label in mine:
                    score = model["bias"] + sum(model["w"].get(one, 0.0)
                                                for one in seen)
                    guess = 1.0 / (1.0 + math.exp(-max(-30.0,
                                                       min(30.0, score))))
                    error = guess - float(label)
                    model["bias"] -= rate * error
                    for one in seen:
                        old = model["w"].get(one, 0.0)
                        model["w"][one] = old - rate * (error + decay * old)
            weights[way.name] = model
        return cls(weights)

    def save(self, path: Path = MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.weights, indent=1, sort_keys=True),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL) -> "Proposer | None":
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))


def first_right(proposer: Proposer, rows: list) -> dict:
    """How often the first way tried is the one the simplest design takes:
    the proposer's order against the table's."""
    out = {"possible": 0, "right": [0, 0]}
    for goal, best in rows:
        if best is None:
            continue
        clause = goal.clauses[0]
        usable = [way.name for way in Ways.usable(
            clause_features(goal, clause))]
        if best not in usable:
            continue
        out["possible"] += 1
        ranked = [one for one in proposer.order(goal, clause)
                  if one in usable]
        out["right"][0] += ranked[0] == best
        out["right"][1] += usable[0] == best
    return out


def bank_rows(bank) -> list:
    """The bank's single-clause goals, labelled by the designer."""
    from research.v694 import designing
    rows = []
    for said, wants, scene, names, _, _ in bank:
        if len(wants) != 1:
            continue
        goal = Goal(tuple(wants), frozenset(scene), frozenset(names),
                    said=said)
        found = designing.design(goal)
        best = found.parts[0].best if found.parts else None
        rows.append((goal, best.form if best is not None else None))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("train", "evaluate"))
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=694)
    options = parser.parse_args(argv)
    from research.v694 import knowing as K
    from research.v694.bank import BANK, HELD
    from research.v694.generating import labelled
    K.index()
    if options.what == "train":
        rows = labelled(options.count, options.seed)
        proposer = Proposer.trained(rows)
        proposer.save()
        counts: dict = {}
        for _, best in rows:
            counts[str(best)] = counts.get(str(best), 0) + 1
        (MODEL.parent / "stats.json").write_text(json.dumps({
            "rows": len(rows), "seed": options.seed, "labels": counts},
            indent=1), encoding="utf-8")
        print(f"trained on {len(rows)} goals {counts} -> {MODEL}")
        train = {goal.wants for goal, _ in rows}
    else:
        train = set()
    proposer = Proposer.load()
    if proposer is None:
        print("no proposer trained yet")
        return 1
    held = [row for row in labelled(options.count // 2, options.seed + 1)
            if row[0].wants not in train]
    for name, rows in (("held-out generated", held),
                       ("the bank", bank_rows(BANK)),
                       ("the held-out bank", bank_rows(HELD))):
        found = first_right(proposer, rows)
        mine, table = found["right"]
        total = found["possible"]
        print(f"{name}: first way tried is the one kept -- proposer "
              f"{mine}/{total}, table order {table}/{total}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())

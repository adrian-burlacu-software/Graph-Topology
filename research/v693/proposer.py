"""The learned proposer: which form to try first, from what worked before.

    python -m research.v693.proposer train        # llm/designer-proposer
    python -m research.v693.proposer evaluate

The designer tries forms in the order it is handed them
(`designing.design(order=...)`). Handed them in table order, it is right
first time only where the table happens to put the right form first. This
learns the order instead.

**What it learns from is the designer's own work.** `generating.labelled`
makes goals from objects and fits every form to each, so every row is a
specification, the forms that met it, and the simplest of those (fewest
unknowns) -- checked, not guessed. One logistic model per form, over the
specification's features (`features`: what `Spec.features` says, and how
many of each thing it gives), predicts whether that form is the one to
use -- the simplest that fits -- and the proposer ranks the forms by it.
Trained on *fits* instead, it learned to reach for the polynomial that
fits everything, which is right and useless.

It is small on purpose: a few dozen features, six forms, seconds to
train. Architecture over data -- the model only has to learn which
proposal suits which goal, because the proposal and the check are exact.

The model is its own folder, `llm/designer-proposer`, and nothing else
writes there (`regenerate.py`'s `designer-proposer` step rebuilds it).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from research.v693.forms import FORMS
from research.v693.spec import Spec

LLM = Path(__file__).resolve().parents[2] / "llm"
MODEL = LLM / "designer-proposer" / "model.json"


def features(spec: Spec) -> set:
    """What the proposer reads: the specification's features, and how
    many conditions of each sort it gives -- which is what decides whether
    a scale alone can meet them, or a polynomial of what degree."""
    out = set(spec.features())
    counts = {}
    for clause in spec.clauses:
        counts[clause.kind] = counts.get(clause.kind, 0) + 1
    for kind, count in counts.items():
        out.add(f"{kind}={min(count, 4)}")
    conditions = spec.conditions()
    out.add(f"conditions={min(conditions, 6)}")
    roots = sum(one.conditions() for one in spec.of("root"))
    out.add(f"beyond-roots={min(conditions - roots, 4)}")
    degree = spec.of("degree")
    if degree:
        room = int(degree[0].args[0]) + 1 - conditions
        out.add("degree-room=" + ("none" if room < 0 else
                                  "exact" if room == 0 else "spare"))
        out.add("roots-fill-degree" if roots == int(degree[0].args[0])
                else "roots-short-of-degree")
    return out


class Proposer:
    """One logistic model per form: P(this form is the one to use -- the
    simplest that fits | the features)."""

    def __init__(self, weights: dict | None = None) -> None:
        #: form -> {"bias": b, "w": {feature: weight}}
        self.weights = weights or {}

    def chance(self, form: str, spec: Spec) -> float:
        model = self.weights.get(form)
        if model is None:
            return 0.5
        score = model["bias"] + sum(model["w"].get(one, 0.0)
                                    for one in features(spec))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    def order(self, spec: Spec) -> list:
        """The forms of the kind, likeliest to be the one first."""
        forms = [form.name for form in FORMS if form.kind == spec.kind]
        return sorted(forms, key=lambda name: -self.chance(name, spec))

    @classmethod
    def trained(cls, rows: list, epochs: int = 60, rate: float = 0.3,
                decay: float = 1e-3, seed: int = 693) -> "Proposer":
        """Fitted by stochastic gradient descent on (spec, forms that
        fit, the simplest) rows, one model per form over its kind."""
        rng = random.Random(seed)
        weights: dict = {}
        for form in FORMS:
            data = [(features(spec), form.name == best)
                    for spec, best in rows if spec.kind == form.kind]
            if not data:
                continue
            model = {"bias": 0.0, "w": {}}
            for _ in range(epochs):
                rng.shuffle(data)
                for seen, label in data:
                    score = model["bias"] + sum(model["w"].get(one, 0.0)
                                                for one in seen)
                    guess = 1.0 / (1.0 + math.exp(-max(-30.0,
                                                       min(30.0, score))))
                    error = guess - float(label)
                    model["bias"] -= rate * error
                    for one in seen:
                        old = model["w"].get(one, 0.0)
                        model["w"][one] = old - rate * (error + decay * old)
            weights[form.name] = model
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


def first_right(proposer, rows: list) -> dict:
    """How often the first form tried fits, and is the simplest that
    fits: the proposer's order against table order, on labelled rows."""
    from research.v693 import designing
    from research.v693.forms import BY_NAME
    out = {"possible": 0, "fits": [0, 0], "simplest": [0, 0]}
    for spec, best in rows:
        if best is None:
            continue
        out["possible"] += 1
        forms = [form.name for form in FORMS if form.kind == spec.kind
                 and set(form.needs) <= spec.features()]
        ranked = [one for one in proposer.order(spec) if one in forms]
        for index, first in enumerate((ranked[0], forms[0])):
            out["fits"][index] += (first == best or designing.fit(
                BY_NAME[first], spec) is not None)
            out["simplest"][index] += first == best
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("train", "evaluate"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=693)
    options = parser.parse_args(argv)
    from research.v693.generating import labelled
    if options.what == "train":
        rows = labelled(options.count, options.seed)
        proposer = Proposer.trained(rows)
        proposer.save()
        (MODEL.parent / "stats.json").write_text(json.dumps({
            "rows": len(rows), "seed": options.seed,
            "designable": sum(best is not None for _, best in rows)},
            indent=1), encoding="utf-8")
        print(f"trained on {len(rows)} goals -> {MODEL}")
    proposer = Proposer.load()
    held = labelled(600, options.seed + 1)
    found = first_right(proposer, held)
    total = found["possible"]
    print(f"held-out generated goals ({total} designable of {len(held)}):")
    for name in ("fits", "simplest"):
        mine, table = found[name]
        print(f"  first form tried {name:9} proposer {mine}/{total}, "
              f"table order {table}/{total}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())

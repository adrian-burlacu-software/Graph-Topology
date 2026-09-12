"""Class-level guards for definitions memory: the questions the gold cannot
see, asked of v687 over each store.

    python -m research.v689.definition_guards data/v684_reasoning.sqlite \\
        data/v684_definitions.sqlite data/v684_definitions_genus.sqlite

XCSLB is almost all leaves, so `audit.py` cannot see a fact that makes a whole
class say yes to something false. These are the questions that would show
it: the over-affirmation guards `challenge.py` uses, and the facts definitions
memory was built to add, each with the answer a person would give.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research.v688 import confidence
from research.v688.challenge import GUARDS
from research.v688.pool import EnginePool

#: What definitions add, true and false.
DEFINED = (("is a kitten young", True), ("is a kitten old", False),
           ("can a cat roar", False), ("does a hammer have a handle", True),
           ("can a testicle secrete androgens", True),
           ("is a canoe pointed at both ends", True),
           ("can a penguin fly", False), ("does a fish have scales", True),
           ("can a dog fly", False), ("does a bird have feathers", True),
           ("is an apple red", True), ("does a car have four wheels", True),
           ("can a mudskipper move on land", True),
           ("can a bat fly", True), ("does a spoon have a handle", True))


def outcomes(store: Path, questions) -> dict:
    pool = EnginePool(store, workers=2)
    try:
        return {text: confidence.outcome_of(pool.ask_one(text).verdict)
                for text, _ in questions}
    finally:
        pool.close()


def main(argv=None) -> int:
    stores = [Path(one) for one in (argv or sys.argv[1:])]
    questions = list(GUARDS) + list(DEFINED)
    found = {store.name: outcomes(store, questions) for store in stores}
    print(f"{'question':42} {'truth':6} " + " ".join(
        f"{name[:24]:>24}" for name in found))
    for text, truth in questions:
        print(f"{text:42} {str(truth):6} " + " ".join(
            f"{found[name][text]:>24}" for name in found))
    for name, answers in found.items():
        wrong = sum(answers[text] == ("denied" if truth else "verified")
                    for text, truth in questions)
        right = sum(answers[text] == ("verified" if truth else "denied")
                    for text, truth in questions)
        print(f"{name}: {right} right, {wrong} wrong, "
              f"{len(questions) - right - wrong} unknown")
    Path("research/v688/audit-out/definition-guards.json").write_text(
        json.dumps(found, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

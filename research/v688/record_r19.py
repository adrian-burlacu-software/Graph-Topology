"""What R19 is actually asked, recorded from a live engine.

Condition 3 of AUDIT.md §17: §17 simulated R19's arithmetic with attribute
names on both sides, so it never tested `_hit`. This asks a real engine real
questions and records every (ancestor, term) that reaches `corroboration`,
which is what any distillation has to cover to be worth writing.

Nothing in v687 is edited: the method is wrapped on the instance.
"""
import collections
import json
import sys
from pathlib import Path

from research.v687.reasoning import ReasoningEngine
from research.v688 import audit

seen = collections.Counter()
sized = {}


def record(engine):
    profiles = engine.profiles
    original = profiles.corroboration

    def wrapped(ancestor, term):
        bearing, kinds = original(ancestor, term)
        seen[(ancestor, term)] += 1
        sized[(ancestor, term)] = (bearing, kinds)
        return bearing, kinds

    profiles.corroboration = wrapped
    return engine


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    engine = record(ReasoningEngine(audit.STORE))
    asked = sorted(audit.all_questions(limit, 0, "screened").items())
    print(f"[record] {len(asked)} questions", flush=True)
    for index, (_key, text) in enumerate(asked):
        try:
            engine.ask(text)
        except Exception as bad:                    # noqa: BLE001
            print(f"  ! {text}: {type(bad).__name__}: {bad}")
        if index and not index % 200:
            print(f"  {index}/{len(asked)}, {len(seen)} distinct calls",
                  flush=True)

    print(f"\nR19 consulted {sum(seen.values())} times, "
          f"{len(seen)} distinct (ancestor, term)")
    by_ancestor = collections.Counter()
    for (ancestor, _term), count in seen.items():
        by_ancestor[ancestor] += count
    print(f"\ndistinct ancestors: {len(by_ancestor)}")
    for ancestor, count in by_ancestor.most_common(20):
        kinds = max((sized[k][1] for k in sized if k[0] == ancestor),
                    default=0)
        print(f"  {ancestor:<34}{count:>6} calls, {kinds:>4} norm-covered kinds")
    spoke = [k for k, (_b, kinds) in sized.items() if kinds >= 8]
    print(f"\ncalls where R19 could speak (>= 8 kinds): "
          f"{len(spoke)} of {len(sized)}")
    Path("r19-calls.json").write_text(json.dumps(
        {"calls": [{"ancestor": a, "term": t, "hits": n,
                    "bearing": sized[(a, t)][0], "kinds": sized[(a, t)][1]}
                   for (a, t), n in seen.most_common()]}, indent=2),
        encoding="utf-8")
    print("written to r19-calls.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

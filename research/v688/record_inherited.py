"""Record the facts v687 actually inherits, and whether they were right.

`record_r19.py` captured what R19 is *asked*. This captures what inheritance
actually *uses*: the leading evidence of every answer that came from an
ancestor rather than from the concept itself, with the gold label attached.

It exists because `prune.py`'s first target was wrong. §19 aimed the
demotion at nodes with 1,000+ descendants -- `animal.n.01`, `person.n.01` --
on the reasoning that a wrong fact there reaches the most concepts. 2,985
facts were demoted and the audit did not move by a single decimal, because
**not one of them was ever the evidence for an answer.** Blast radius is not
the same as being read, and this is the difference: it lists the facts that
were read.

Nothing in v687 is edited; the method is wrapped on the instance.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from . import audit

OUT = Path(__file__).resolve().parent / "audit-out" / "inherited.json"


def run(limit: int = 120, gold: str = "screened",
        config: str = "crawl") -> dict:
    """`config` defaults to `crawl` and that matters.

    The shipped engine answers the 521 norm-covered concepts from the norms
    at distance zero, so it inherits on only 42 of 1,468 questions. `crawl`
    is the configuration that generalises -- it is what the other 44,678
    concepts get -- and it is where inheritance does the work and the damage.
    """
    engine = audit.engine_class(config)(audit.STORE)
    chosen = audit.pairs(limit, gold)
    held = {f"{pair.held}|{pair.prop}" for pair in chosen}
    foil = {f"{pair.foil}|{pair.prop}" for pair in chosen}
    asked = sorted(audit.all_questions(limit, 0, gold).items())

    used: dict = collections.defaultdict(
        lambda: {"right": 0, "wrong": 0, "unlabelled": 0})
    seen = 0
    for index, (key, text) in enumerate(asked):
        try:
            payload = engine.ask(text)
        except Exception:                           # noqa: BLE001
            continue
        if payload.get("verdict") != "VERIFIED":
            continue
        evidence = payload.get("evidence") or []
        if not evidence:
            continue
        lead = evidence[0]
        if not int(lead.get("distance") or 0):
            continue
        seen += 1
        row = used[(lead.get("concept") or "", lead.get("relation") or "",
                    str(lead.get("object") or ""))]
        row["source"] = lead.get("source") or ""
        row["distance"] = int(lead.get("distance") or 0)
        if key in held:
            row["right"] += 1
        elif key in foil:
            row["wrong"] += 1
        else:
            row["unlabelled"] += 1
        if index and not index % 400:
            print(f"[inherited] {index}/{len(asked)}, {len(used)} distinct",
                  flush=True)

    rows = [{"concept": c, "relation": r, "object": o, **one}
            for (c, r, o), one in used.items()]
    rows.sort(key=lambda one: (-one["wrong"], -one["right"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"facts": rows}, indent=1), encoding="utf-8")

    wrong = sum(one["wrong"] for one in rows)
    right = sum(one["right"] for one in rows)
    print(f"\ninherited answers: {seen}   distinct facts used: {len(rows)}")
    print(f"  right {right}   wrong {wrong}   "
          f"precision {right / max(right + wrong, 1):.1%}")
    carry = [one for one in rows if one["wrong"]]
    print(f"  facts that ever supplied a wrong answer: {len(carry)}")
    print(f"\n{'concept':<28}{'source':<11}{'d':>2}  {'wrong':>5}{'right':>6}"
          f"  object")
    for one in rows[:18]:
        print(f"  {one['concept'][:26]:<26}{one['source']:<11}"
              f"{one['distance']:>2}  {one['wrong']:>5}{one['right']:>6}"
              f"  {one['object'][:34]}")
    return {"facts": len(rows), "right": right, "wrong": wrong}


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
        config=sys.argv[2] if len(sys.argv) > 2 else "crawl")
    raise SystemExit(0)

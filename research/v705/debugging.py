"""Debugging (v705): where a run of a function goes, and where it goes
wrong -- in Python and TypeScript alike.

A plan's step asked the architecture's editor to "change reader.build so
that 'what is my dog called?' is read as ask_name": seven answers, no two
agreeing. Nothing said why build gave up on it, nor where. A person
debugging would run the case and watch.

**A run, step by step** (`trace`): each language's checker runs a function
on one case and keeps each step -- the line, the values changed there,
what is returned or raised (`pycheck.trace`, `tscheck.js` `trace`).
**Said** (`said`): the lines in order, a loop's repeats counted rather
than listed, each with its own code and what changed there, ending with
what it gave. **Where it goes wrong** (`suspects`): with cases that pass
and cases that fail, each line by how much more the failing runs take it
than the passing ones (Ochiai, over the lines each run takes) -- the
lines only failing runs reach first.
"""
from __future__ import annotations

import json
import math
from collections import Counter

#: the most steps said, and the most suspects named
SAID, NAMED = 24, 3


def trace(source: str, entry: str, case: list, language: str) -> dict:
    """One case run, each step kept: {ok, output, steps, cut} (`checker`)."""
    from research.v696.checker import checker
    return checker(language).trace(source, entry, case)


def said(found: dict, source: str, most: int = SAID) -> str:
    """A run in words: each line it took, with its code and what changed,
    a line taken again and again said once with how often; then what it
    gave or raised."""
    lines = source.splitlines()
    steps = found.get("steps") or []
    out, counted = [], Counter(one["line"] for one in steps
                               if "values" in one)
    told = set()
    for one in steps:
        line = one.get("line")
        code = lines[line - 1].strip() if line and 0 < line <= len(
            lines) else ""
        if "raises" in one:
            out.append(f"line {line} `{code}` raises {one['raises']}")
            continue
        if "returns" in one:
            out.append(f"line {line} `{code}` returns {one['returns']}")
            continue
        if line in told:
            continue
        told.add(line)
        changed = ", ".join(f"{name} = {value}" for name, value in
                            one["values"].items())
        again = f" ({counted[line]} times)" if counted[line] > 1 else ""
        out.append(f"line {line} `{code}`{again}"
                   + (f": {changed}" if changed else ""))
    if len(out) > most:
        out = out[:most // 2] + ["..."] + out[-(most - most // 2):]
    return "; ".join(out)


def taken(found: dict) -> set:
    """The lines a run took."""
    return {one["line"] for one in found.get("steps") or []
            if one.get("line")}


def suspects(runs: list, failed: list, most: int = NAMED) -> list:
    """The lines most suspected, by Ochiai over the lines each run took:
    failing runs taking a line, against all runs taking it and all runs
    failing. [(line, score)], most first; none where no run failed or no
    run passed."""
    if not any(failed) or all(failed):
        return []
    lines = set().union(*[taken(one) for one in runs])
    total_failed = sum(failed)
    scored = []
    for line in lines:
        by_failing = sum(1 for one, bad in zip(runs, failed)
                         if bad and line in taken(one))
        by_passing = sum(1 for one, bad in zip(runs, failed)
                         if not bad and line in taken(one))
        if not by_failing:
            continue
        score = by_failing / math.sqrt(total_failed *
                                       (by_failing + by_passing))
        scored.append((line, round(score, 3)))
    scored.sort(key=lambda one: (-one[1], one[0]))
    return scored[:most]


def wrong(output: dict, wanted) -> bool:
    return "value" not in output or json.dumps(
        output["value"], sort_keys=True) != json.dumps(wanted,
                                                        sort_keys=True)


def debug(source: str, entry: str, cases: list, wanted: list,
          language: str) -> dict:
    """Each case run and kept; those that fail said, and the lines most
    suspected where some pass: {runs, failed, said, suspects}."""
    runs = [trace(source, entry, case, language) for case in cases]
    failed = [wrong(one.get("output") or {}, want)
              for one, want in zip(runs, wanted)]
    first = next((at for at, bad in enumerate(failed) if bad), None)
    return {"runs": runs, "failed": failed,
            "said": said(runs[first], source) if first is not None else "",
            "case": cases[first] if first is not None else None,
            "suspects": suspects(runs, failed)}

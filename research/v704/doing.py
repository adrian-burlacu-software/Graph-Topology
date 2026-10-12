"""Carrying out a goal's plan (v704): each yes, the next step done -- not
only said.

A goal's plan was said, and a yes took one look (what fails now) and
dropped it: `look into the first one` then went to world facts ("One is
found far away from here"). Now the plan is **held** until done, and each
yes carries out its next step where the architecture can:

    the look     each case that fails now, **diagnosed** from the
                 conversation test's own trace: how each part's reader
                 reads it, which operators the executive tried and what
                 came of each, and where each lives in the code
                 (`selfmodel.places`); what most cases share is the
                 **finding** -- the place to change
    change,      a concrete request made from the finding (`in _ask_name,
    build        answer "what is my dog called?" after "my dog is called
                 Rex"`), said to itself as a change of the project when
                 the project is its own (the attachment loop, from inside)
    check        the cases kept as the goal's checks: it is done when they
                 are right
    measure      the checks run again (the conversation test, in the
                 background: the server says one turn at a time), told
                 on the next turn
    teach and    said, not done: what they would need that it cannot do
    the rest     itself yet (a GPU run, a default chosen, a document)

Every step says what it did, what came of it, and asks before the next.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter

from research.v704 import history, selfmodel

RUNS = selfmodel.RUNS
#: each part's reader of a message, as the shared encoder's heads
READERS = (("conversation", "acts"), ("code talk", "code_act"),
           ("data talk", "data_act"), ("shell", "shell_act"),
           ("mathematics", "math_act"))
#: what each reader reads as none of its own
NOTHING = ("none", "generic")
#: the cases a look diagnoses at most
MOST = 6


def readings(text: str) -> list:
    """How each part's reader reads a message: (part, label, chance) for
    each whose head the encoder has."""
    from research import encoder
    from research.v692.corpus import words
    if not encoder.enabled():
        return []
    heads = encoder.LOADED.get().heads
    asked = [(part, head) for part, head in READERS if head in heads]
    said = words(text)
    if not said or not asked:
        return []
    found = encoder.read(said, heads=tuple(head for _, head in asked))
    return [(part, found[head][0][0], round(found[head][0][1], 2))
            for part, head in asked]


def diagnose(case: dict, found_places: dict | None = None) -> dict:
    """A failing case taken apart: the turns before it, how each part's
    reader reads it, and each operator tried -- what came of it, and where
    it lives."""
    found_places = found_places if found_places is not None else \
        selfmodel.places()
    tried = []
    for one in case.get("fired") or []:
        place = selfmodel.place_of(one["operator"], one.get("rule", ""),
                                   found_places) or {}
        function = place.get("apply") or place.get("proposes") or {}
        tried.append({"operator": one["operator"],
                      "outcome": one["outcome"],
                      "function": function.get("function"),
                      "file": place.get("file")})
    turns = case.get("turns") or [case["said"]]
    return {"said": case["said"].split("\n")[0], "n": case.get("n"),
            "before": turns[:-1], "read": readings(turns[-1]),
            "tried": tried, "lost": _lost(case, tried, found_places),
            "answer": case.get("answer", "")}


def _lost(case: dict, tried: list, found_places: dict) -> dict | None:
    """A reading lost on its way to the executive: the conversation's
    reader read the case as an act an operator answers, the turn was taken
    as another, and that operator never tried -- the place it was lost,
    where a reading is built from what was read (v689 `reader.build`,
    found by its own code)."""
    acts = case.get("acts") or {}
    read, taken = acts.get("read") or [], acts.get("taken")
    if not read or not taken or read[0][0] == taken:
        return None
    act, chance = read[0]
    if any(step["operator"] == act for step in tried):
        return None
    place = selfmodel.place_of(act, "", found_places) or {}
    import inspect
    from pathlib import Path
    from research.v689 import reader
    return {"act": act, "chance": round(chance, 2), "taken": taken,
            "operator": act,
            "answers": (place.get("apply") or {}).get("function"),
            "function": reader.build.__name__,
            "file": Path(inspect.getsourcefile(reader.build)).resolve()
            .relative_to(selfmodel.ROOT).as_posix()}


def said_diagnosis(one: dict) -> str:
    """A diagnosis in a line."""
    line = f"\"{one['said']}\""
    if one["before"]:
        line += " (after " + "; ".join(f"\"{turn}\"" for turn in
                                       one["before"][-2:]) + ")"
    own = [f"{part} as {label} ({int(chance * 100)}%)"
           for part, label, chance in one["read"] if label not in NOTHING]
    line += ": read by " + (", ".join(own) if own else "no part as its own")
    lost = one.get("lost")
    if lost:
        line += (f"; taken as {lost['taken']}, so {lost['operator']}"
                 + (f" ({lost['answers']})" if lost["answers"] else "")
                 + " was never tried")
    steps = [f"{step['operator']}" + (
        f" ({step['function']}, {step['file']})" if step["function"]
        else "") + f" {step['outcome']}" for step in one["tried"]
        if step["outcome"] != "continue"]
    if steps:
        line += "; tried: " + " -> ".join(steps)
    return line


def finding(found: list) -> dict | None:
    """What most of the cases share: the operator that declined them most
    -- where an answer was looked for and not given -- else the one that
    answered them wrongly. Its function and file, and its first case."""
    declined: Counter = Counter()
    answered: Counter = Counter()
    lost: Counter = Counter()
    first: dict = {}
    for one in found:
        if one.get("lost"):
            key = (one["lost"]["operator"], one["lost"]["function"],
                   one["lost"]["file"])
            lost[key] += 1
            first.setdefault(key, one)
        for step in one["tried"]:
            if not step["function"]:
                continue
            key = (step["operator"], step["function"], step["file"])
            if step["outcome"] == "declined":
                declined[key] += 1
            elif step["outcome"] == "answered":
                answered[key] += 1
            first.setdefault(key, one)
    for counted, how in ((lost, "never tried"), (declined, "declined"),
                         (answered, "answered")):
        if counted:
            (operator, function, file), count = counted.most_common(1)[0]
            return {"operator": operator, "function": function,
                    "file": file, "how": how, "cases": count,
                    "case": first[(operator, function, file)]}
    return None


def request_for(found: dict | None) -> str | None:
    """The change the finding asks for, said as a person would say it to
    the architecture: in the function that declined its case (or answered
    it wrongly), answer the case -- after what was said before it."""
    if not found:
        return None
    case = found["case"]
    module = found["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    named = f"{module}.{found['function']}"
    if found["how"] == "never tried":
        lost = case["lost"]
        return (f"change {named} so that \"{case['said']}\" is read as "
                f"{lost['act']}, as the reader reads it")
    asked = f"change {named} so that it answers \"{case['said']}\""
    if case["before"]:
        asked += " after " + " and ".join(f"\"{turn}\"" for turn in
                                          case["before"][-2:])
    return asked


# -- measuring, in the background ---------------------------------------------

def measure(numbers: list, label: str) -> None:
    """The conversation test's own cases run again, apart from this turn
    (the server says one turn at a time: the run waits for this one)."""
    subprocess.Popen(
        [sys.executable, "-m", "research.v704.converse", "--label", label,
         "--only", *[str(one) for one in numbers]],
        cwd=str(history.ROOT), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


def measured(label: str) -> dict | None:
    """A measuring run's verdicts, once it is done: {n: verdict}."""
    if not (RUNS / f"{label}.summary.json").exists():
        return None
    path = RUNS / f"{label}.jsonl"
    with path.open(encoding="utf-8") as rows:
        return {one["n"]: one["verdict"] for one in map(json.loads, rows)}


# -- the plan, carried out ----------------------------------------------------

def carry(state: dict, change=None, held=None) -> list:
    """The next step of a held plan, done: its lines. `state` is the goal
    as held (`page.GOALS`), moved on; `change(text)` says a change to the
    architecture itself and returns what it said, where it can."""
    lines = _measured_lines(state)
    weak = state["weak"][0] if state.get("weak") else None
    steps = (state.get("plan") or {}).get("steps") or []
    at = state.get("at")
    if at is None:
        lines += _look(state, weak)
        state["at"] = 0
    elif at < len(steps):
        lines += _step(state, steps[at], change, held)
        state["at"] = at + 1
    at = state["at"]
    if at < len(steps):
        lines.append(f"Next: {at + 1}. {steps[at]['said']}. Shall I go on?")
    else:
        lines.append("That was the last step of the plan."
                     + (" I'll tell you what the measuring finds when you "
                        "ask." if state.get("measuring") else ""))
        state["done"] = True
    return lines


def _look(state: dict, weak: dict | None) -> list:
    if not weak:
        return ["I have no cases of my own for this goal to look at."]
    cases = (weak.get("cases") or weak.get("examples") or [])[:MOST]
    found_places = selfmodel.places()
    found = [diagnose(one, found_places) for one in cases]
    state["found"] = found
    state["finding"] = finding(found)
    lines = [f"First, a look at {weak['what']} -- what fails now, and "
             f"where:"]
    lines += [f"{at}. {said_diagnosis(one)}" for at, one in
              enumerate(found, 1)]
    shared = state["finding"]
    if shared and state.get("plan"):
        # the steps left, said again of what the look found
        history.ground(state["plan"], {**weak, "where": [{
            "operator": shared["operator"], "function": shared["function"],
            "file": shared["file"], "role": "found",
            "cases": shared["cases"], "case": shared["case"]["said"]}]})
    if shared:
        lines.append(f"Most telling: {shared['operator']} "
                     f"({shared['function']} in {shared['file']}) "
                     f"{shared['how']} {shared['cases']} of them.")
    return lines


def _step(state: dict, step: dict, change, held) -> list:
    kind = step["kind"]
    if kind in ("change", "build"):
        # of itself, the finding's request; of a project, the goal itself,
        # planned across the project as any change is (v703)
        asked = request_for(state.get("finding")) or (
            None if state.get("weak") else state.get("goal"))
        if asked is None:
            return [f"{step['said']}: I found no place to change, so I "
                    f"left it."]
        file = (state.get("finding") or {}).get("file")
        if change is None or held is None or (file and
                                              file not in held.files):
            return [f"{step['said']}: I would ask for \"{asked}\" -- but "
                    f"{file or 'the project'} is not one I hold, so I left "
                    f"it."]
        if asked in state.setdefault("asked", []):
            return [f"{step['said']}: asked already, in step "
                    f"{state['asked'].index(asked) + 1} -- left."]
        state["asked"].append(asked)
        shared = state.get("finding") or {}
        return [f"{step['said']}: I asked myself to {asked}.",
                change(asked, shared.get("function"), shared.get("file"))]
    if kind == "check":
        numbers = [one["n"] for one in state.get("found") or []
                   if one.get("n") is not None]
        state["checks"] = numbers
        return [f"{step['said']}: these {len(numbers)} cases are this "
                f"goal's checks now -- it is reached when they are right."]
    if kind == "measure":
        numbers = state.get("checks") or [
            one["n"] for one in state.get("found") or []
            if one.get("n") is not None]
        if not numbers:
            return [f"{step['said']}: there is nothing to measure yet."]
        label = f"goal-{time.strftime('%Y%m%d-%H%M%S')}"
        measure(numbers, label)
        state["measuring"] = {"label": label, "numbers": numbers}
        return [f"{step['said']}: I'm running its {len(numbers)} cases "
                f"again ({label}); I'll tell you on your next message."]
    needs = {"teach": "teaching a model takes data written for it and a "
                      "run on the GPU, which I can't start myself yet",
             "default": "choosing a default is yours, from what was "
                        "measured",
             "document": "what was done is in this conversation; writing "
                         "it down is yours to ask",
             "tidy": "what is left over is yours to delete"}
    return [f"{step['said']}: not done -- {needs.get(kind, 'I cannot do '
                                                        'this step myself')}."]


def _measured_lines(state: dict) -> list:
    pending = state.get("measuring")
    if not pending or pending.get("told"):
        return []
    found = measured(pending["label"])
    if found is None:
        return [f"(Still measuring {pending['label']}.)"]
    pending["told"] = True
    right = sum(found.get(one) == "right" for one in pending["numbers"])
    return [f"Measured: {right} of {len(pending['numbers'])} of the "
            f"goal's cases are right now."]

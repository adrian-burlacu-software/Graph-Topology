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
#: the functions a change is tried in: where the case gives up, then the
#: one that called it
UP = 2
#: rounds of subgoals a goal is looked at again for at most
ROUNDS = 3


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
            "part": case.get("part"),
            # the turns whole: said again, and found among the test's own
            # (`inside.cases_of`), with the code pasted under the first line
            # -- `how many times does this loop run?` alone was no case of
            # the test, and nothing of it could be checked
            "turns": list(turns),
            "before": turns[:-1], "read": readings(turns[-1]),
            "tried": tried, "lost": _lost(case, tried, found_places),
            "answer": case.get("answer", ""),
            # what a case found by a way of looking brings (v705
            # `evidence`): the check of it, and what it was made from
            **{key: case[key] for key in ("check", "original", "area",
                                          "guards", "expected", "idea")
               if key in case}}


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
    # a part of the project answered it: nothing was lost on its way, what
    # was said is wrong (`how many functions are there in tools.py?`, sent
    # to the conversation's reader, which no reading of would answer)
    if (case.get("part") or "").startswith("project"):
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


#: what a place did with a case, most telling first
HOW = ("never tried", "declined", "answered")


def findings(found: list) -> list:
    """Every place the cases fail at, each case under the one most telling
    of it (lost there, else declined there, else answered wrongly there):
    the goal's subgoals (v705), most telling and most cases first -- each
    as `finding` gives one."""
    places: dict = {}
    for one in found:
        keys = []
        if one.get("lost"):
            keys.append((("never tried", one["lost"]["function"],
                          one["lost"]["file"]), one["lost"]["operator"]))
        for step in one["tried"]:
            if step["function"] and step["outcome"] in ("declined",
                                                        "answered"):
                keys.append(((step["outcome"], step["function"],
                              step["file"]), step["operator"]))
        if not keys:
            continue
        key, operator = min(keys, key=lambda item: HOW.index(item[0][0]))
        places.setdefault(key, []).append((operator, one))
    out = []
    for (how, function, file), cases in sorted(
            places.items(), key=lambda item: (HOW.index(item[0][0]),
                                              -len(item[1]))):
        operators = list(dict.fromkeys(op for op, _ in cases))
        out.append({"operator": ", ".join(operators), "function": function,
                    "file": file, "how": how, "cases": len(cases),
                    "case": cases[0][1],
                    "all": [one["said"] for _, one in cases]})
    return out


def finding(found: list) -> dict | None:
    """What most of the cases share: the place in the code -- a function of
    a file -- that lost their readings most (read as one act, taken as
    another), else that declined them, else that answered them wrongly;
    counted by the place, whichever operator it was for (`ask_name` and
    `place located` are both lost in reader.build). The place, the
    operators, every case there and the first."""
    places: dict = {}
    for one in found:
        seen = set()
        if one.get("lost"):
            key = ("never tried", one["lost"]["function"],
                   one["lost"]["file"])
            places.setdefault(key, []).append((one["lost"]["operator"], one))
            seen.add(key)
        for step in one["tried"]:
            if not step["function"] or step["outcome"] not in (
                    "declined", "answered"):
                continue
            key = (step["outcome"], step["function"], step["file"])
            if key not in seen:
                places.setdefault(key, []).append((step["operator"], one))
                seen.add(key)
    for how in ("never tried", "declined", "answered"):
        mine = [(key, cases) for key, cases in places.items()
                if key[0] == how]
        if mine:
            (_, function, file), cases = max(mine,
                                             key=lambda item: len(item[1]))
            operators = list(dict.fromkeys(op for op, _ in cases))
            return {"operator": ", ".join(operators), "function": function,
                    "file": file, "how": how, "cases": len(cases),
                    "case": cases[0][1],
                    "all": [one["said"] for _, one in cases]}
    return None


def _inside(state: dict, debug) -> dict | None:
    """The finding's function watched while its case is said again (v705
    `inside`), where it is the architecture's own and can be: what the call
    that matters did, and the line it gave back at. Kept, looked once."""
    shared = state.get("finding") or {}
    if debug is None or not shared or not shared["file"].startswith(
            "research/"):
        return None
    if "inside" not in shared:
        case = shared["case"]
        lost = case.get("lost") or {}
        shared["inside"] = debug(list(case.get("turns") or
                                      list(case["before"]) + [case["said"]]),
                                 shared["file"], shared["function"],
                                 lost.get("act"))
    return shared["inside"]


def request_for(found: dict | None, looked: dict | None = None
                ) -> str | None:
    """The change the finding asks for, said as a person would say it to
    the architecture: in the function that declined its case (or answered
    it wrongly), answer the case -- after what was said before it."""
    if not found:
        return None
    case = found["case"]
    module = found["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    named = f"{module}.{found['function']}"
    if looked and looked.get("line"):
        # where it gave up, seen inside: the line, what it was given there
        lost = case.get("lost") or {}
        module = looked["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        named = f"{module}.{looked['function']}"
        return (f"change {named} so that it does not give up at line "
                f"{looked['line']} (`{looked['code']}`) for "
                f"\"{case['said']}\""
                + (f", read as {lost['act']}" if lost.get("act") else ""))
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

def carry(state: dict, change=None, held=None, debug=None) -> list:
    """The next step of a held plan, done: its lines. `state` is the goal
    as held (`page.GOALS`), moved on; `change(text)` says a change to the
    architecture itself and returns what it said, where it can."""
    lines = _measured_lines(state)
    weak = state["weak"][0] if state.get("weak") else None
    steps = (state.get("plan") or {}).get("steps") or []
    at = state.get("at")
    if at is None and state.get("subgoals") is not None:
        # a goal found by its own ways of looking (v705 `evidence`): its
        # subgoals are the places its cases fail
        lines += _look(state, weak)
        state["at"] = 0
        state["subgoals"] = findings(state.get("found") or [])
        state["sub"] = 0
        return lines + _next_subgoal(state)
    if state.get("subgoals") is not None:
        return lines + _subgoal(state, change, held, debug)
    if at is None:
        lines += _look(state, weak)
        state["at"] = 0
    elif at < len(steps):
        lines += _step(state, steps[at], change, held, debug)
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


def _said_subgoal(state: dict) -> str:
    subs, at = state["subgoals"], state["sub"]
    one = subs[at]
    return (f"subgoal {at + 1} of {len(subs)}: {one['function']} in "
            f"{one['file']}, where \"{one['case']['said']}\" goes wrong"
            + (f" ({one['cases']} cases)" if one["cases"] > 1 else ""))


def _next_subgoal(state: dict) -> list:
    if state["sub"] < len(state["subgoals"]):
        return [f"Next: {_said_subgoal(state)}. Shall I go on?"]
    return _again(state)


def _subgoal(state: dict, change, held, debug=None) -> list:
    """The next subgoal carried out: a change where its cases fail, looked
    inside, built and run again -- then the next, and when all are tried,
    a look again (`_again`)."""
    subs, at = state["subgoals"], state["sub"]
    if at >= len(subs):
        return _again(state)
    one = subs[at]
    state["finding"] = one
    lines = _step(state, {"kind": "change",
                          "said": _said_subgoal(state)[0].upper() +
                          _said_subgoal(state)[1:]}, change, held, debug)
    met = bool(state.get("changed"))
    state.setdefault("results", []).append(
        (one["function"], one["file"], met))
    # an idea's case: what came of it, for which ways of having ideas pay
    # (v705 `ideas.ledger`)
    idea = (one.get("case") or {}).get("idea")
    if idea:
        from research.v705 import ideas
        ideas.kept({**idea, "place": f"{one['function']} in {one['file']}",
                    "cases": [one["case"]]}, "fixed" if met else "not fixed")
    state["sub"] = at + 1
    return lines + _next_subgoal(state)


def _again(state: dict) -> list:
    """All subgoals tried: where any was met, the ways of looking are
    looked with again (`state["again"]`) -- what a change fixed is gone,
    what it uncovered is new -- and the places not tried yet are the next
    round; where none was, or nothing is left, the goal is done."""
    results = state.get("results") or []
    met = [one for one in results if one[2]]
    tried = {(one[0], one[1]) for one in results}
    said = (f"Of {len(results)} subgoals, {len(met)} met"
            + (f" ({', '.join(one[0] for one in met)})" if met else "")
            + ".")
    again = state.get("again")
    if met and again is not None and state.get("rounds", 1) < ROUNDS:
        cases = again()
        found_places = selfmodel.places()
        found = [diagnose(one, found_places) for one in cases[:MOST * 2]]
        fresh = [one for one in findings(found)
                 if (one["function"], one["file"]) not in tried]
        if fresh:
            state.update(subgoals=fresh, sub=0, found=found,
                         rounds=state.get("rounds", 1) + 1)
            return [said, f"Looked again: {len(cases)} cases still fail, "
                          f"{len(fresh)} at places not tried yet.",
                    f"Next: {_said_subgoal(state)}. Shall I go on?"]
        said += (f" Looked again: {len(cases)} cases still fail, all at "
                 f"places already tried." if cases else
                 " Looked again: nothing fails now.")
    if len(state.get("weak") or []) > 1 and state.get("subgoals") is not None:
        # the next of what it does worst, where this one is done: a goal
        # of itself is all of them (`make fewer mistakes` met nothing in
        # saying how to do something, and stopped there, the four others
        # it had said never looked at)
        done = state["weak"].pop(0)
        state.setdefault("weak done", []).append(done["what"])
        state.update(at=None, subgoals=[], sub=0, found=None, finding=None,
                     results=[], rounds=1)
        return [said, f"Next: {state['weak'][0]['what']} -- what fails "
                      f"there, and where. Shall I go on?"]
    state["done"] = True
    return [said + " That is as far as I can take this goal."]


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
        lines.append(f"Most telling: {shared['function']} in "
                     f"{shared['file']} -- {shared['how']} "
                     f"({shared['operator']}) for {shared['cases']} of "
                     f"them.")
    return lines


def _to_the_author(shared: dict, lines: list) -> None:
    """A finding that a part of the project answered wrongly, moved to the
    function that wrote the answer (v705 `inside.author`), where it is its
    own code."""
    case = shared.get("case") or {}
    if shared.get("how") != "answered" or "authored" in shared or \
            not (case.get("part") or "").startswith("project"):
        return
    from research.v705 import inside
    turns = list(case.get("turns") or list(case["before"]) + [case["said"]])
    try:
        found = inside.author(turns)
    except Exception:                               # noqa: BLE001
        found = None
    shared["authored"] = bool(found)
    if found:
        shared.update(function=found["function"], file=found["file"])
        lines.append(f"What \"{case['said']}\" was answered with was written "
                     f"by {found['function']} in {found['file']}.")


def _step(state: dict, step: dict, change, held, debug=None) -> list:
    kind = step["kind"]
    if kind in ("change", "build"):
        # of itself, the finding's request -- where it is lost, seen from
        # inside the function when it can be (v705) -- of a project, the
        # goal itself, planned across the project as any change is (v703)
        shared = state.get("finding") or {}
        file = shared.get("file")
        if change is None or held is None or (file and
                                              file not in held.files):
            asked = request_for(shared or None) or (
                None if state.get("weak") else state.get("goal"))
            return [f"{step['said']}: I would ask for \"{asked}\" -- but "
                    f"{file or 'the project'} is not one I hold, so I left "
                    f"it."]
        lines = []
        _to_the_author(shared, lines)
        file = shared.get("file")
        looked = _inside(state, debug)
        if looked:
            # each call followed to where the nothing came from, the last
            # said whole
            chain = []
            one = looked
            while one:
                chain.append(one)
                one = one.get("deeper")
            for upper, lower in zip(chain, chain[1:]):
                lines.append(f"{upper['function']} gives back "
                             f"{upper['returns']} at line {upper['line']} "
                             f"(`{upper['code']}`), from "
                             f"{lower['function']}.")
            looked = chain[-1]
            lines.append(f"Looking inside {looked['function']} with "
                         f"\"{shared['case']['said']}\": {looked['said']}")
        asked = request_for(shared or None, looked) or (
            None if state.get("weak") else state.get("goal"))
        if asked is None:
            return lines + [f"{step['said']}: I found no place to change, "
                            f"so I left it."]
        if asked in state.setdefault("asked", []):
            return lines + [f"{step['said']}: asked already, in step "
                            f"{state['asked'].index(asked) + 1} -- left."]
        state["asked"].append(asked)
        # made where the look inside found it, else where the look did --
        # and where no change there answers the case, in the function that
        # called it, which holds more of what the case needs (`_whose`
        # cannot make `my dog` a mention; `act`, which calls it, makes
        # mentions of the same words below it)
        targets = list(reversed(chain))[:UP] if looked else [shared]
        said, changed, target = "", False, targets[0]
        for at, target in enumerate(targets):
            if at:
                asked = request_for(shared or None, target)
                lines.append(f"{said} So, one call up:")
            try:
                said, changed = change(asked, target.get("function"),
                                       target.get("file"),
                                       (shared.get("case") or {}).get("said"))
            except (MemoryError, RuntimeError) as bad:
                # the card full (a torch OutOfMemoryError is a RuntimeError):
                # this place not met -- a look at every other still to come
                if not isinstance(bad, MemoryError) and \
                        "out of memory" not in str(bad).lower():
                    raise
                said, changed = (f"I ran out of memory changing "
                                 f"{target.get('function')}.", False)
            lines.append(f"{step['said'] if not at else 'Then'}: I asked "
                         f"myself to {asked}.")
            if changed:
                break
        # what it changed, for the person to answer (`_feedback`)
        state["changed"] = (target.get("function") or "the project") \
            if changed else None
        return lines + [said]
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


# -- what is said back to the plan --------------------------------------------

#: how near an option's words a phrase must be, and how much nearer than the
#: next option
LEAST_NEAR, LEAD = 0.3, 0.05


def options(state: dict) -> list:
    """The plan's options a reply can name: (kind, index, the words each is
    said in) -- each weakness by what it is and its name, each step by its
    number, kind and what it says, each case by its words."""
    out = []
    for at, one in enumerate(state.get("weak") or []):
        out.append(("weak", at, [one["what"], one.get("name") or
                                 one["what"]]))
    for at, one in enumerate((state.get("plan") or {}).get("steps") or []):
        out.append(("step", at, [one["said"], f"step {at + 1}",
                                 f"the {one['kind']} step"]))
    cases = state.get("found") or [
        {"said": one["said"].split("\n")[0]} for one in
        ((state.get("weak") or [{}])[0].get("cases") or [])[:MOST]]
    for at, one in enumerate(cases):
        out.append(("case", at, [one["said"], f"case {at + 1}"]))
    return out


def option_of(state: dict, phrase: str) -> tuple | None:
    """The option a phrase names: the nearest in meaning of every option's
    words, near enough and nearer than the next. (kind, index), else
    None."""
    from research.v701.likeness import nearest
    found = options(state)
    said = [(text, (kind, at)) for kind, at, texts in found
            for text in texts]
    ranked = nearest(phrase, [text for text, _ in said])
    best: dict = {}
    for score, text in ranked:
        key = next(option for one, option in said if one == text)
        best[key] = max(best.get(key, 0.0), score)
    order = sorted(best.items(), key=lambda item: -item[1])
    if not order or order[0][1] < LEAST_NEAR or (
            len(order) > 1 and order[0][1] - order[1][1] < LEAD):
        return None
    return order[0][0]


def talk(state: dict, act: str, phrase: str, change=None, held=None,
         undo=None, debug=None) -> list:
    """A reply to the plan, answered: an option picked is done, one asked
    about is said, and what it just did, answered -- a change it made put
    back where the person found it wanting."""
    if act == "feedback":
        return _feedback(state, phrase, undo)
    named = option_of(state, phrase) if phrase else None
    if named is None:
        listed = "; ".join(f"{texts[0]}" for _, _, texts in
                           options(state)[:8])
        return [f"I couldn't tell which of the plan's options \"{phrase}\" "
                f"is. They are: {listed}."]
    kind, at = named
    if act == "pick":
        return _picked(state, kind, at, change, held, debug)
    return _asked(state, kind, at)


def _picked(state: dict, kind: str, at: int, change, held,
            debug=None) -> list:
    if kind == "weak":
        weak = state["weak"]
        weak.insert(0, weak.pop(at))
        if state.get("plan"):
            history.ground(state["plan"], weak[0])
        state.update(at=None, found=None, finding=None)
        return carry(state, change, held, debug)
    if kind == "step":
        state["at"] = at
        return carry(state, change, held, debug)
    found = state.get("found")
    if not found:
        _look(state, state["weak"][0])
        state["at"] = 0
        found = state["found"]
    one = found[at]
    state["finding"] = finding([one]) or state.get("finding")
    lines = [f"Looking into \"{one['said']}\":", said_diagnosis(one)]
    asked = request_for(state["finding"])
    if asked:
        lines.append(f"What I'd change: {asked}. Shall I go on?")
    return lines


def _asked(state: dict, kind: str, at: int) -> list:
    if kind == "weak":
        one = state["weak"][at]
        cases = [f"\"{case['said'].splitlines()[0]}\"" for case in
                 (one.get("cases") or [])[:3]]
        return [f"{one['what']}: {one.get('how', '')}"
                + (f" -- e.g. {', '.join(cases)}" if cases else "") + "."]
    if kind == "step":
        step = state["plan"]["steps"][at]
        if step["kind"] in ("change", "build"):
            asked = request_for(state.get("finding"))
            does = (f"I'd {asked}" if asked else
                    "I'd need to look first, to know where")
        else:
            does = {"check": "I'd keep the failing cases as the goal's "
                             "checks",
                    "measure": "I'd run the checks again in the "
                               "background"}.get(step["kind"],
                                                 "I can't do this one "
                                                 "myself yet")
        return [f"Step {at + 1}, {step['said']}: {does}."]
    found = state.get("found") or [diagnose(one) for one in (
        state["weak"][0].get("cases") or [])[:MOST]]
    state["found"] = found
    lines = [said_diagnosis(found[at]) + "."]
    # and what it would do about it: the change its own finding asks for
    asked = request_for(finding([found[at]]))
    if asked:
        lines.append(f"What I'd change: {asked}.")
    return lines


def _feedback(state: dict, phrase: str, undo) -> list:
    last = state.get("changed")
    if last and undo is not None:
        put_back = undo()
        state["changed"] = None
        if put_back:
            return [f"I've put back my change to {last} -- "
                    f"{put_back}. Shall I try it another way, or go on?"]
    # what it said answered, not what it did: the finding looked at again,
    # with what it rests on
    if state.get("found"):
        state["finding"] = finding(state["found"])
        shared = state["finding"]
        if shared:
            return [f"Looking again: {shared['function']} in "
                    f"{shared['file']} -- {shared['how']} for "
                    f"{shared['cases']} of the {len(state['found'])} cases: "
                    + "; ".join(f"\"{one}\"" for one in shared["all"])
                    + ". Shall I go on from there?"]
    return [f"Noted about {phrase or 'that'}. Shall I try it another way, "
            f"or go on?"]

"""Where the architecture falls short of a goal, found by its own ways of
looking -- never asked of the person (v705).

A goal says what is wanted, not where it fails: told "make this code more
generalizable", the plan's steps asked the goal of itself again and did
nothing. Each **way of looking** here is said in words, and a goal takes
the ways nearest its own words (as plans are chosen, `v701.likeness`).
What they find are **cases**: the turns, the check that says whether an
answer is right, and how the run went (operators tried, acts read and
taken) -- what `doing.diagnose` takes apart into the place each fails,
each place a subgoal (`doing.findings`).
"""
from __future__ import annotations

#: each way of looking: (name, what it finds, said for a goal to be near)
WAYS = (
    ("failures", "what it gets wrong now",
     "what it gets wrong now: questions answered wrongly or not at all, "
     "things it was told and forgot, messages sent to the wrong part"),
    ("knowledge", "saying what it knows: facts its store holds, asked of it",
     "what it knows but cannot say: facts it holds about things -- what "
     "they are made of, used for, where they are found -- asked of it and "
     "answered wrongly or not at all; knowing, understanding, general "
     "knowledge"),
)
#: ways within this much of the nearest are taken too
LEAD = 0.05


def ways(goal: str) -> list:
    """The ways of looking a goal takes, nearest first: [(name, what)]."""
    from research.v701.likeness import nearest
    ranked = nearest(goal, [said for _, _, said in WAYS])
    best = ranked[0][0]
    by_said = {said: (name, what) for name, what, said in WAYS}
    return [by_said[said] for score, said in ranked
            if best - score <= LEAD]


def case_of(turn, turns: list, check, **more) -> dict:
    """A case as `doing.diagnose` takes it apart: what was said, how the
    run went, and the check that says whether it is right."""
    from research.v704 import converse
    found = turn.as_dict()
    return {"said": turns[-1], "turns": list(turns), "n": None,
            "fired": converse.fired_of(found),
            "acts": {"read": ((found.get("heard") or {}).get("acts")
                              or [])[:2],
                     "taken": (found.get("reading") or {}).get("act")},
            "answer": " ".join(((turn.answer or {}).get("text") or "")
                               .split())[:160],
            "part": next((one["operator"] for one in reversed(
                converse.fired_of(found)) if one.get("outcome") ==
                "answered"), None),
            "check": check, **more}


def failures(goal: str) -> list:
    """The conversation test's failed cases of the weaknesses near the
    goal (`selfmodel.weaknesses`), each with its check."""
    from research.v704 import selfmodel, suite
    model = selfmodel.held()
    if not model:
        return []
    checks = {tuple(item.turns): item.check for item in suite.SUITE
              if not item.project}
    out = []
    for weak in selfmodel.weaknesses(goal, model):
        for one in weak.get("cases") or []:
            check = checks.get(tuple(one.get("turns") or [one["said"]]))
            if check is not None:
                out.append({**one, "check": check, "area": weak["name"]})
    return out


def gather(goal: str, session=None) -> dict:
    """The goal's evidence: {ways: [(name, what)], cases}."""
    taken = ways(goal)
    cases = []
    for name, _ in taken:
        if name == "failures":
            cases += failures(goal)
        elif name == "knowledge" and session is not None:
            cases += known(session)["cases"]
    return {"ways": taken, "cases": cases}


def known(session) -> dict:
    """What its knowledge against what it can say found (`ideas`): the
    cases of each fault, each knowing the idea it came of; and the counts
    of all it was asked."""
    from research.v705 import ideas
    found = ideas.knowledge(session)
    cases, asked, right = [], 0, 0
    for idea in found:
        asked += idea["asked"]
        right += idea["right"]
        if idea["kind"] != "fault":
            continue
        for one in idea["cases"]:
            cases.append({**one, "idea": {key: value for key, value in
                                          idea.items() if key != "cases"}})
    return {"cases": cases, "asked": asked, "right": right,
            "ideas": found}


def weaknesses(goal: str, session, model: dict) -> list:
    """The goal's weaknesses, as a goal of itself shows them
    (`selfmodel.weaknesses`), by the ways of looking it takes."""
    from research.v704 import selfmodel
    out = []
    for name, what in ways(goal):
        if name == "failures":
            out += selfmodel.weaknesses(goal, model)
        elif name == "knowledge" and session is not None:
            found = known(session)
            if found["cases"]:
                wrong = found["asked"] - found["right"]
                out.append({"what": what, "name": name, "bad": wrong,
                            "right": found["right"], "of": found["asked"],
                            "wrong": wrong,
                            "how": f"{found['right']} right of "
                                   f"{found['asked']}, {wrong} wrong",
                            "examples": found["cases"][:2],
                            "cases": found["cases"], "where": []})
    return out

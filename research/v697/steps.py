"""A turn, step by step: v690's account, and what v697 adds to it.

    programmed  where the turn asked for a function: what was read of the
                request, its risks, what the writers wrote, what the search
                did and how the answer was chosen (`coding.py`)
"""
from __future__ import annotations

from research.v690 import steps as v690

STATUS = {
    "confirmed": "it meets the examples, and a second writer or the search "
                 "arrived at the same apart",
    "met": "it meets the examples; nothing apart confirms it beyond them",
    "unsolved": "nothing written or searched meets the examples",
    "unverified": "there were no examples to check it by",
    "agreed": "writers agree on what it does, with nothing given to check "
              "it by",
    "missing": "the request does not say how it is called",
    "failed": "solving it failed",
}


def programmed(turn: dict) -> dict | None:
    code = (turn.get("answer") or {}).get("code")
    if not code:
        return None
    answer = code.get("answer") or {}
    followup = code.get("followup")
    if followup:
        return {"step": "programmed",
                "line": f"A question about {answer.get('entry')} "
                        f"({followup['asked']}): answered from what was kept "
                        f"of writing it, and by running it.",
                "detail": code}
    written = sum(len(one["programs"]) for writer in code.get("writers", ())
                  for one in writer["rounds"])
    meeting = sum(sum(p["meets"] for p in one["programs"])
                  for writer in code.get("writers", ())
                  for one in writer["rounds"])
    search = code.get("search") or {}
    line = ("Taking it with what was asked before, " if code.get(
        "continued") else "") + (
            f"I wrote {written} program{'s' if written != 1 else ''} "
            f"({meeting} met the examples)"
            + (f", searched {search['evaluated']} candidates"
               if search else "")
            + f": {STATUS.get(answer.get('status'), answer.get('status'))}.")
    return {"step": "programmed", "line": line, "detail": code}


def steps_of(turn: dict, said: dict | None = None) -> list[dict]:
    found = v690.steps_of(turn, said)
    step = programmed(turn)
    if step is not None:
        at = next((n for n, one in enumerate(found)
                   if one["step"] == "answered"), len(found))
        found.insert(at, step)
    return found

"""A turn, step by step: v697's account, and the project.

    looked     where the turn asked about the project, or pasted code: what
               was looked up in the outline, the calls, the compiler
"""
from __future__ import annotations

from research.v697 import steps as v697

ASKED = {"overview": "what is in it", "files": "its files",
         "in file": "one file's functions", "where": "where a function is",
         "callers": "who calls a function", "calls": "what a function calls",
         "explain": "what a function is", "errors": "what the compiler finds",
         "pasted": "code pasted with a question"}


def looked(turn: dict) -> dict | None:
    code = (turn.get("answer") or {}).get("code") or {}
    asked = code.get("project")
    if not asked:
        return None
    return {"step": "looked",
            "line": f"I looked it up in the project ({ASKED.get(asked['asked'], asked['asked'])}"
                    + (f": {asked['name']}" if asked.get("name") else "")
                    + "), read by the compiler.",
            "detail": code}


def steps_of(turn: dict, said: dict | None = None) -> list[dict]:
    found = v697.steps_of(turn, said)
    step = looked(turn)
    if step is not None:
        at = next((n for n, one in enumerate(found)
                   if one["step"] == "answered"), len(found))
        found.insert(at, step)
    return found

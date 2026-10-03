"""A turn, step by step: v697's account, and v698's.

    understood  what the encoder read a message about code as: what it
                does, what it asks, of what, and how sure
    looked      where the turn asked about the project, or pasted code: what
                was looked up in the outline, the calls, the compiler
    kept        where it taught something: what was kept, long-term
"""
from __future__ import annotations

from research.v697 import steps as v697

ASKED = {"explain": "what it is", "files": "its files",
         "functions": "its functions", "where": "where it is",
         "size": "how big it is", "callers": "who calls it",
         "calls": "what it calls", "bugs": "its gaps (bugs)",
         "capabilities": "what can be asked of it",
         "unknown": "something it does not know",
         "run": "running it", "pasted": "code pasted with a question"}


def understood(turn: dict) -> dict | None:
    code = (turn.get("answer") or {}).get("code") or {}
    read = code.get("read")
    if not read:
        return None
    line = (f"The encoder read it as {read['act']} "
            f"({round(read['chance'] * 100)}% sure)"
            + (f", asking {read['aspect']}" if read["aspect"] != "none"
               else "")
            + (f" of {read['subject']}" if read["subject"] != "none" else "")
            + (f": “{read['spans']['SUBJ'][0]}”"
               if read["spans"].get("SUBJ") else "") + ".")
    return {"step": "understood", "line": line, "detail": code}


def looked(turn: dict) -> dict | None:
    code = (turn.get("answer") or {}).get("code") or {}
    asked = code.get("project")
    if not asked:
        return None
    return {"step": "looked",
            "line": f"I looked it up ({ASKED.get(asked['asked'], asked['asked'])}"
                    + (f": {asked['name']}" if asked.get("name") else "")
                    + "), read by the compiler.",
            "detail": code}


def kept(turn: dict) -> dict | None:
    code = (turn.get("answer") or {}).get("code") or {}
    found = code.get("knowledge")
    if not isinstance(found, dict) or not found.get("kept"):
        return None
    one = found["kept"]
    return {"step": "kept",
            "line": f"I kept, for every conversation: {one['concept']} is "
                    f"one of {', '.join(one['members'])}.",
            "detail": code}


def steps_of(turn: dict, said: dict | None = None) -> list[dict]:
    found = v697.steps_of(turn, said)
    at = next((n for n, one in enumerate(found)
               if one["step"] in ("programmed", "answered")), len(found))
    for step in (kept(turn), looked(turn), understood(turn)):
        if step is not None:
            found.insert(at, step)
    return found

"""Reaching a goal, measured by leaving ConceptNet's answer out.

ConceptNet says what some actions are done for: `climb MotivatedByGoal get
over fence`. For every such edge whose goal is a movement with a place --
a verb, the words that place it, a thing -- the edge is hidden and the
goal asked of `achieving`: does the action come back in the top three from
everything else it reads (who was seen doing it, what WordNet's glosses
say)? That is whether the way the answer is found generalises, not
whether an answer is looked up.

Two numbers, for the goals where anything is proposed at all:

    hit@3      the hidden action is among the first three proposed
    feasible   of those hits, how many the subject -- a person -- can do
               (`can.py`), which is what is said on the page

    python -m research.v695.held_out
"""
from __future__ import annotations

import sys

from research.v694 import knowing as K
from research.v695 import achieving as A
from research.v695 import mined

#: Places that make a goal a movement: `get over`, `get out of`.
PLACES = frozenset(A.ROLES)


def goals() -> list:
    """(goal, the action ConceptNet says is done for it)."""
    found = mined.connection()
    if found is None:
        return []
    out, seen = [], set()
    for head, tail in found.execute(
            "SELECT head_norm, tail_norm FROM edges WHERE relation = "
            "'MotivatedByGoal'"):
        words = tail.split()
        heads = head.split()
        if len(words) < 3 or not heads or words[1] not in PLACES:
            continue
        verb = mined._morphy(heads[0], True)
        if verb in A.LIGHT:
            continue
        # A movement to or past a thing: the goal's verb moves (or is
        # English's light `get`, `go`, `come`), and the thing is one --
        # `want to learn` and `believe in character` are not.
        if not (words[0] in ("get", "go", "come")
                or A._file(words[0]) == A.MOTION):
            continue
        if not K.is_a(words[-1], "physical_entity.n.01"):
            continue
        place = words[1:-1]
        goal = A.Goal(words[0], " ".join(place), words[-1])
        key = (goal.phrase, verb)
        if key in seen:
            continue
        seen.add(key)
        out.append((goal, verb))
    return out


def main() -> int:
    tried = proposed = hits = feasible = 0
    real = A._done_for

    def hidden(goal, found):
        # The goal's own edge is what is being guessed: everything else
        # ConceptNet says it is done for stays.
        real(goal, found)
        found.pop(HIDDEN[0], None)

    HIDDEN = [""]
    A._done_for = hidden
    try:
        for goal, verb in goals():
            HIDDEN[0] = verb
            tried += 1
            found = A.actions(goal)
            if not found:
                continue
            proposed += 1
            top = [one.verb for one in found[:3]]
            if verb in top:
                hits += 1
                one = next(one for one in found if one.verb == verb)
                feasible += bool(one.able)
    finally:
        A._done_for = real
    print(f"goals {tried}; something proposed for {proposed} "
          f"({proposed / max(tried, 1):.1%})")
    print(f"hit@3 {hits}/{proposed} ({hits / max(proposed, 1):.1%}) of "
          f"those; {hits / max(tried, 1):.1%} of all")
    print(f"feasible for a person, of the hits: {feasible}/{hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

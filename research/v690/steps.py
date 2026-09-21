"""A turn, step by step, as the page shows it.

    heard      the words, what a request asked, the words placed in normal
               form, who is new, what they say about when
    read       what the encoder read them as: the act and how sure, what a
               question asks, each word's role
    resolved   who each phrase means, and how that was found
    reasoned   the operators that fired, the rules a walk applied, what v688
               answered and how far it trusts it
    remembered what was written to memory: events, the trie, definitions
    planned    where the turn was an order: what was wanted, how many
               actions were possible, the goal stack the planner pushed and
               what it did (v691)
    answered   the answer in v689's own terms, and the message a reply to it
               has to carry
    said       the replies the decoder wrote, each read back, and the one said

Each step has a `line` -- one sentence of English, so the steps read as an
account -- and the details behind it, as they are.
"""
from __future__ import annotations

ROLE_NAMES = {"AUX": "auxiliary", "NEG": "denial", "B-SUBJ": "subject",
              "I-SUBJ": "subject", "B-OBJ": "object", "I-OBJ": "object",
              "REST": "said of it", "VERB": "verb", "B-KIND": "kind",
              "I-KIND": "kind", "SEQ": "sequence", "B-NAME": "name",
              "I-NAME": "name"}

ACT_NAMES = {
    "introduce": "puts someone new down", "tell": "tells me something",
    "teach": "teaches me about a kind", "name": "names someone",
    "compound": "says several things", "ask": "asks a yes or no",
    "generic": "asks about a kind", "what": "asks what something is",
    "define": "asks what a word means", "ask_name": "asks a name",
    "why": "asks why", "question": "asks about the conversation",
    "greet": "greets me", "farewell": "says goodbye", "thank": "thanks me",
    "wellbeing": "asks how I am", "identity": "asks who I am",
    "abilities": "asks what I can do", "affirm": "agrees",
    "deny": "disagrees", "apology": "apologises", "praise": "praises me"}

SOURCES = {"told": "what you told me", "taught": "what you taught me",
           "kind": "what its kind does", "tendency": "its kind's tendency",
           "conversation": "the conversation", "definition": "a definition",
           "learned": "what you taught me before",
           "world": "what I did in the world in front of me"}


def _describe(turn: dict, referent: str | None) -> str:
    from .message import referents_of

    found = referents_of(turn).get(referent or "") or {}
    return found.get("description") or referent or ""


def heard(turn: dict) -> dict:
    found = turn.get("heard") or {}
    bits = []
    if found.get("asked"):
        bits.append(f"the request asks “{found['asked']}”")
    if found.get("placed") and found.get("placed").lower() != (
            found.get("asked") or turn.get("said") or "").lower():
        bits.append(f"placed as “{found['placed']}”")
    if found.get("fresh"):
        bits.append("someone new: " + ", ".join(found["fresh"]))
    if found.get("time"):
        bits.append("when: " + " ".join(found["time"]))
    if found.get("anchor"):
        bits.append(f"{found.get('relation')} “{found['anchor']}”")
    line = f"You said “{turn.get('said')}”" + (
        "; " + "; ".join(bits) if bits else ".")
    return {"step": "heard", "line": line, "detail": found}


def read(turn: dict) -> dict:
    found = turn.get("heard") or {}
    reading = turn.get("reading") or {}
    acts = found.get("acts") or []
    act = reading.get("act") or turn.get("act") or ""
    goals = reading.get("goals") or []
    line = f"I read it as something that {ACT_NAMES.get(act, act)}"
    if acts:
        line += f" ({round(acts[0][1] * 100)}% sure)"
    if goals:
        line += "; it asks for " + " or ".join(
            f"the {goal['asked']} of {goal['relation']}" for goal in goals)
    roles = [[word, ROLE_NAMES.get(role, role)]
             for word, role in (found.get("roles") or []) if role != "O"]
    return {"step": "read", "line": line + ".",
            "detail": {"act": act, "acts": acts, "slots": found.get("slots"),
                       "goals": goals, "roles": roles,
                       "mention": reading.get("mention"),
                       "aux": reading.get("aux"), "rest": reading.get("rest"),
                       "holds": reading.get("holds"),
                       "object": reading.get("object")}}


def resolved(turn: dict) -> dict | None:
    lines, detail = [], []
    for key in ("resolution", "object"):
        found = turn.get(key)
        if not found:
            continue
        who = (_describe(turn, found.get("referent")) if found.get("referent")
               else "which one?" if found.get("ambiguous") else "nothing")
        lines.append(f"“{found.get('expression')}” is {who}")
        walked = found.get("identification") or {}
        detail.append({"expression": found.get("expression"), "means": who,
                       "how": found.get("how"),
                       "introduced": found.get("introduced"),
                       "candidates": [_describe(turn, one) for one in
                                      found.get("candidates") or []],
                       "wanted": walked.get("wanted"),
                       "visited": walked.get("visited")})
    if not lines:
        return None
    return {"step": "resolved", "line": "; ".join(lines) + ".",
            "detail": detail}


def reasoned(turn: dict) -> dict:
    fired = []
    for claim in turn.get("trace") or []:
        for one in claim.get("fired") or []:
            fired.append({"claim": claim.get("claim"), **one})
    answered = [one for one in fired if one["outcome"] == "answered"]
    walk = turn.get("walk") or {}
    run = turn.get("run") or {}
    parts = []
    if answered:
        parts.append("the " + ", then the ".join(
            one["operator"] for one in answered) + " operator answered")
    # An operator that went on (CONTINUE) did its work and let the cycle
    # carry on -- v691's `noting` writes down what was said about the world
    # and lets v689 answer. Calling that "had nothing" was wrong.
    went_on = list(dict.fromkeys(one["operator"] for one in fired
                                 if one["outcome"] == "continue"))
    if went_on:
        parts.append(", ".join(went_on) + " went first and let it go on")
    declined = list(dict.fromkeys(one["operator"] for one in fired
                                  if one["outcome"] == "declined"))
    if declined:
        parts.append("after " + ", ".join(declined) + " had nothing")
    rules = sorted({step.get("rule") for step in walk.get("steps") or []
                    if step.get("rule")})
    if rules:
        parts.append("the walk applied " + ", ".join(rules))
    if run:
        parts.append(f"the store's reasoning answered “{turn.get('asked')}” "
                     f"{run.get('outcome')}" + (f", {run.get('trust')}"
                                                if run.get('trust') else ""))
    line = ("; ".join(parts) or "nothing needed reasoning") + "."
    return {"step": "reasoned", "line": line[:1].upper() + line[1:],
            "detail": {"fired": fired, "walk": walk, "run": run,
                       "asked": turn.get("asked")}}


def planned(turn: dict) -> dict | None:
    """The plan, where a turn was an order and the agent worked one out.

    None on every other turn, so a conversation that never asks for
    anything to be done reads exactly as it did.
    """
    found = (turn.get("answer") or {}).get("planning") or {}
    if not found:
        return None
    wanted = ", ".join(found.get("goal") or []) or "nothing in particular"
    plan = found.get("plan") or []
    parts = [f"you asked for {wanted}"]
    parts.append(f"{found.get('offered', 0)} actions were possible, over "
                 f"{found.get('verbs', 0)} verbs")
    if found.get("solved"):
        parts.append(f"{len(plan)} action{'' if len(plan) == 1 else 's'} did "
                     f"it, off a goal stack {found.get('deep', 1)} deep "
                     f"({found.get('subgoals', 0)} subgoals, "
                     f"{found.get('fired', 0)} operators fired)")
    else:
        parts.append("no plan reached it")
    if found.get("surprises"):
        parts.append(f"{len(found['surprises'])} surprise"
                     f"{'' if len(found['surprises']) == 1 else 's'} on the "
                     f"way, and it planned again")
    line = "; ".join(parts) + "."
    return {"step": "planned", "line": line[:1].upper() + line[1:],
            "detail": found}


def remembered(turn: dict) -> dict | None:
    growth = turn.get("growth") or []
    learned = turn.get("learned") or []
    memory = turn.get("memory") or {}
    number = turn.get("number")
    events = [one for one in memory.get("log") or []
              if one.get("turn") == number
              and one.get("type") not in ("turn_began", "attended")]
    if not (growth or learned or events):
        return None
    bits = []
    if events:
        kinds: dict = {}
        for one in events:
            kinds[one["type"]] = kinds.get(one["type"], 0) + 1

        def counted(kind: str, count: int) -> str:
            words = kind.split("_")
            if count > 1 and len(words) > 1:
                words[0] += "s"                 # `referents added`
            return f"{count} {' '.join(words)}"

        bits.append(", ".join(counted(kind, count)
                              for kind, count in kinds.items()))
    if growth:
        grown = sum(one.get("allocated") or 0 for one in growth)
        bits.append(f"the trie grew by {grown} node"
                    f"{'' if abs(grown) == 1 else 's'}")
    if learned:
        bits.append("definitions read: " + ", ".join(
            one.get("concept") or "" for one in learned))
    return {"step": "remembered", "line": "I remembered: " + "; ".join(bits)
            + ".", "detail": {"events": events, "growth": growth,
                              "learned": learned}}


def answered(turn: dict, said: dict | None) -> dict:
    answer = turn.get("answer") or {}
    message = (said or {}).get("message") or {}
    stance = message.get("stance") or answer.get("outcome")
    source = SOURCES.get(answer.get("source"), answer.get("source") or "")
    if answer.get("outcome") == "acted":
        # Not a verdict about the world but something done in it, or said
        # about the scene the agent is keeping.
        return {"step": "answered",
                "line": f"Not a verdict: this came from {source}.",
                "detail": {"answer": answer, "message": message,
                           "prompt": (said or {}).get("prompt")}}
    line = f"The answer is {stance}" + (f", from {source}" if source else "")
    if message.get("rules"):
        line += " (" + ", ".join(message["rules"]) + ")"
    return {"step": "answered", "line": line + ".",
            "detail": {"answer": answer, "message": message,
                       "prompt": (said or {}).get("prompt")}}


def spoken(said: dict | None) -> dict | None:
    if not said:
        return None
    if said.get("source"):
        # v691's reply is written from the plan it carried out, and is said
        # as it stands: the decoder is trained to paraphrase v689's
        # verdicts, and given a story it was never shown it would
        # paraphrase it into one (`v690/server.py`, `_spoken`).
        return {"step": "said", "line": "I said what I did, in my own words "
                "-- written from the plan, not by the decoder.",
                "detail": said}
    candidates = said.get("candidates") or []
    written = [one for one in candidates if not one.get("shortened_from")]
    traced = [one for one in candidates if one["trace"]["traced"]]
    chosen = next((one for one in candidates
                   if one["text"] == said.get("text")), None)
    line = (f"I wrote {len(written)} repl{'y' if len(written) == 1 else 'ies'}; "
            + ("none read back until I left a sentence out, and I said that."
               if said.get("traced") and chosen and chosen.get("shortened_from")
               else f"{len(traced)} read back to the message, and I said the "
               f"first." if said.get("traced") else
               "none read back to the message, so I said the one with the "
               "least wrong, marked untraced."))
    return {"step": "said", "line": line, "detail": said}


def steps_of(turn: dict, said: dict | None = None) -> list[dict]:
    found = [heard(turn), read(turn), resolved(turn), reasoned(turn),
             planned(turn), remembered(turn), answered(turn, said),
             spoken(said)]
    return [one for one in found if one is not None]

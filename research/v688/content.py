"""What an answer that is not a yes or no actually says, in a line.

v687 answers `what can a dog do` with forty facts, `what kind of dog has
spots` with a dalmatian and `what do a dog and a cat have in common` with nine
shared properties -- and the summary headlines each as `LISTING — what can a
dog do`: a verdict word and the question back. A page that shows the summary
shows nothing. `digest` reads the payload the question came back with and says
what it held: a short text, the items it was made from, and whether that text
*is* the answer or sits beside a yes or no.

Every shape is v687's own, read where v687 put it:

    listing       evidence rows about the subject       what can a dog do
    backwards     `backwards.subjects`                  what has wings
    identified    `identification.candidates`           what kind of dog has spots
    profile       `profile.path`, the trie walk         tell me about dogs
    members       `profile.asked.members`               do all birds fly
    contrast      `contrast`, by mode                   what do a dog and a cat ...
    script        `causal.steps`, by phase              why does a dog bark
    explanation   `causal.hypotheses`                   what explains a fire
    kinds         `kinds.direct_kinds`                  how many kinds of dog
    definition    `definition`                          what is a kitten
    bridge        `bridge.route` and the evidence       what does a dog's owner need
"""
from __future__ import annotations

#: Items a line names before it says how many more there are.
SHOWN = 8

#: How a relation reads in front of its objects.
SAYS = {"capable_of": "can", "receives_action": "can be",
        "has_property": "is", "has_a": "has", "has_part": "has",
        "part_of": "has", "made_of": "is made of", "used_for": "is used for",
        "at_location": "is found in", "located_near": "is found near",
        "desires": "wants", "has_prerequisite": "needs", "causes": "causes",
        "has_subevent": "involves", "created_by": "is created by",
        "is_a": "is a kind of", "similar_to": "is similar to",
        "motivated_by_goal": "is done for", "not_capable_of": "cannot"}

#: The same relations, said of the things in front of an object.
PLURAL = {"capable_of": "can", "has_part": "have", "has_a": "have",
          "part_of": "have", "made_of": "are made of",
          "used_for": "are used for", "causes": "cause",
          "at_location": "are found in", "has_prerequisite": "need",
          "desires": "want", "has_property": "are",
          "receives_action": "can be"}

#: Verdicts a trie walk reports for a kind that does it, and one that does not.
DOES = ("HELD", "INHERITED")
DOES_NOT = ("DENIED",)


def name_of(concept) -> str:
    """`dalmatian.n.02` as `dalmatian`; anything else as it is."""
    text = str(concept or "")
    parts = text.rsplit(".", 2)
    if len(parts) == 3 and parts[2].isdigit() and len(parts[1]) == 1:
        return parts[0]
    return text


def listed(items, shown: int = SHOWN) -> str:
    """`a, b, c and 4 more`, each item once, in the order given."""
    seen = list(dict.fromkeys(str(item) for item in items if item))
    head = ", ".join(seen[:shown])
    more = len(seen) - shown
    return head + (f" and {more} more" if more > 0 else "")


def _found(kind: str, text: str, items, answers: bool = True) -> dict:
    return {"kind": kind, "text": text,
            "items": list(dict.fromkeys(str(item) for item in items if item)),
            "answers": answers}


def _definition(payload: dict):
    found = payload.get("definition")
    if not found or not found.get("gloss"):
        return None
    genus = name_of(found.get("genus"))
    text = f"{found.get('word') or name_of(found.get('sense'))}: “{found['gloss']}”"
    if genus:
        text += f" — a kind of {genus}"
    return _found("definition", text,
                  [name_of(one) for one in found.get("above") or []])


def _kinds(payload: dict):
    found = payload.get("kinds")
    if not found:
        return None
    names = [name_of(one) for one in found.get("direct_kinds") or []]
    text = (f"{found.get('total', 0):,} kinds of {found.get('word')}, "
            f"{found.get('direct', 0)} directly beneath it")
    if names:
        text += f": {listed(names)}"
    described = found.get("described_kinds") or []
    if described:
        text += f"; the norms describe {listed(described)}"
    return _found("kinds", text, names)


def _contrast(payload: dict):
    found = payload.get("contrast")
    if not found:
        return None
    mode = found.get("mode")
    if mode == "common":
        shared = found.get("shared") or []
        text = (f"{found.get('left')} and {found.get('right')} share "
                f"{found.get('shared_total', len(shared))}: {listed(shared)}"
                if shared else
                f"nothing recorded is shared by {found.get('left')} and "
                f"{found.get('right')}")
        return _found("contrast", text, shared)
    if mode == "difference":
        left, right = found.get("only_left") or [], found.get("only_right") or []
        text = (f"only {found.get('left')}: {listed(left, 5)}; only "
                f"{found.get('right')}: {listed(right, 5)} "
                f"({found.get('shared_total', 0)} shared)")
        return _found("contrast", text, left + right)
    if mode == "nearest":
        near = found.get("nearest") or []
        items = [f"{one['name']} ({round(one.get('jaccard', 0) * 100)}%)"
                 for one in near]
        if not items:
            return None
        return _found("contrast", f"nearest to {found.get('name')}: "
                                  f"{listed(items)}", items)
    if mode == "typicality":
        text = (f"{found.get('member')} has {found.get('score', 0):.0%} of "
                f"what a {found.get('klass')} typically has, "
                f"{found.get('rank')} of {found.get('of')}")
        if found.get("sound") is False:
            text += "; not a sound measure on these norms"
        return _found("typicality", text, found.get("missing") or [],
                      answers=False)
    return None


def _causal(payload: dict):
    found = payload.get("causal")
    if not found:
        return None
    if found.get("mode") == "abduction":
        names = [name_of(one.get("cause")) for one in
                 found.get("hypotheses") or []]
        if not names:
            return None
        return _found("explanation", f"best explanations of "
                                     f"{found.get('observation')}: "
                                     f"{listed(names, 6)}", names)
    phases: dict = {}
    for step in found.get("steps") or []:
        phases.setdefault(step.get("phase") or step.get("relation"),
                          []).append(step.get("object"))
    if not phases:
        return None
    parts = [f"{phase}: {listed(objects, 4)}"
             for phase, objects in phases.items()]
    return _found("script", f"{found.get('concept')} — " + "; ".join(parts),
                  [one for objects in phases.values() for one in objects])


def _bridge(payload: dict):
    found = payload.get("bridge")
    if not found or found.get("verdict") != "BRIDGED":
        return None
    hops = ((found.get("route") or {}).get("hops") or [])
    route = ", ".join(f"{name_of(hop.get('source'))} "
                      f"{SAYS.get(hop.get('relation'), hop.get('relation'))} "
                      f"{hop.get('text')}" for hop in hops)
    objects = [row.get("object") for row in payload.get("evidence") or []]
    relation = found.get("relation") or ""
    text = (f"{found.get('anchor_word')}'s {found.get('role_word')} "
            f"{SAYS.get(relation, relation.replace('_', ' '))}: "
            f"{listed(objects) or 'nothing recorded'}")
    if route:
        text += f" (through {route})"
    return _found("bridge", text, objects)


def _backwards(payload: dict):
    found = payload.get("backwards")
    if not found:
        return None
    names = [name_of(one.get("concept")) for one in
             found.get("subjects") or []]
    if not names:
        return None
    relation = found.get("relation") or ""
    says = PLURAL.get(relation, relation.replace("_", " "))
    return _found("backwards", f"{says} {found.get('phrase')}: "
                               f"{listed(names)}", names)


def _members(payload: dict):
    asked = (payload.get("profile") or {}).get("asked") or {}
    members = asked.get("members") or []
    if not members:
        return None
    yes = [one["name"] for one in members if one.get("verdict") in DOES]
    no = [one["name"] for one in members if one.get("verdict") in DOES_NOT]
    parts = []
    if no:
        parts.append(f"{len(no)} do not: {listed(no)}")
    if yes:
        parts.append(f"{len(yes)} do: {listed(yes, 5)}")
    if not parts:
        return None
    return _found("members", "; ".join(parts), no + yes, answers=False)


def _profile(payload: dict):
    found = payload.get("profile") or {}
    if payload.get("verdict") != "PROFILE" or not found.get("path"):
        return None
    predicates = [one.get("predicate") for one in found["path"]]
    return _found("profile", f"{found.get('name')}: {listed(predicates, 10)}",
                  predicates)


def _identified(payload: dict):
    found = payload.get("identification") or {}
    verdict = payload.get("verdict")
    candidates = found.get("candidates") or []
    if verdict not in ("IDENTIFIED", "AMBIGUOUS") or not candidates:
        return None
    names = [one.get("name") for one in candidates]
    if verdict == "IDENTIFIED" and len(candidates) == 1:
        matched = "; ".join(str(value) for value in
                            (candidates[0].get("matched") or {}).values())
        text = names[0] + (f" — {matched}" if matched else "")
    else:
        text = f"several fit: {listed(names)}"
    return _found("identified", text, names)


def _listing(payload: dict):
    rows = payload.get("evidence") or []
    if payload.get("verdict") != "LISTING" or not rows:
        return None
    subject = name_of(payload.get("concept")) or name_of(rows[0].get("concept"))
    by_relation: dict = {}
    for row in rows:
        by_relation.setdefault(row.get("relation") or "", []).append(
            row.get("object"))
    parts = [f"{SAYS.get(relation, relation.replace('_', ' '))} "
             f"{listed(objects, SHOWN if len(by_relation) == 1 else 4)}"
             for relation, objects in list(by_relation.items())[:3]]
    return _found("listing", f"{subject} " + "; ".join(parts),
                  [one for objects in by_relation.values() for one in objects])


#: Verdicts that decline the question, with a note that says why.
DECLINED = ("UNSUPPORTED", "UNKNOWN_WORD", "UNPARSED")


def _refused(payload: dict):
    """R18's refusal, or a word the ontology does not have: the reason is
    the answer. The page showed `UNSUPPORTED — should a dog eat chocolate`
    and nothing about why."""
    note = (payload.get("note") or "").strip()
    if payload.get("verdict") not in DECLINED or not note:
        return None
    if note.startswith("This asks for "):
        note = "this asks for " + note[len("This asks for "):]
    return _found("refused", f"not answerable here: {note}", [])


def _within(payload: dict):
    """`which birds cannot fly`: the kinds of the class on each side."""
    found = payload.get("within")
    if not found:
        return None
    norms, store = found.get("norms") or [], found.get("store") or []
    if not norms and not store:
        return None
    parts = []
    if norms:
        parts.append(f"{listed(norms)}")
    if store:
        parts.append(f"{'and ' if norms else ''}in the store beyond the "
                     f"norms: {listed(store)}")
    return _found("within", "; ".join(parts), norms + store)


def _attribute(payload: dict):
    """`what color is a banana`: the values of one dimension it carries."""
    found = payload.get("attribute")
    if not found:
        return None
    values = found.get("values") or []
    if not values:
        return _found("attribute", f"nothing recorded of {found.get('kind')} "
                                   f"says what its {found.get('dimension')} "
                                   f"is", [])
    text = listed(values)
    if found.get("measured"):
        text += (" — a word for it, never a measurement: there are no numbers "
                 "in this data")
    return _found("attribute", text, values)


def _above(payload: dict):
    """`what is a dog a kind of`: the taxonomy above it, nearest first."""
    found = payload.get("above")
    if not found or not found.get("chain"):
        return None
    return _found("above", f"{found.get('kind')} is a kind of "
                           f"{listed(found['chain'])}", found["chain"])


def _parts(payload: dict):
    found = payload.get("parts")
    if not found or not found.get("parts"):
        return None
    return _found("parts", f"{found.get('kind')} has "
                           f"{listed(found['parts'], 12)}", found["parts"])


def _choice(payload: dict):
    """`is a tomato a fruit or a vegetable`: the option the taxonomy holds."""
    found = payload.get("choice")
    if not found:
        return None
    holds = found.get("holds") or []
    options = [one.get("option") for one in found.get("options") or []]
    if holds:
        others = [one for one in options if one not in holds]
        text = (f"{found.get('subject')} is filed under "
                f"{' and '.join(holds)}"
                + (f", not recorded as {' or '.join(others)}" if others
                   else ""))
    else:
        text = (f"neither: {found.get('subject')} is not recorded as "
                f"{' or '.join(options)}")
    return _found("choice", text, holds)


#: Most specific first: the payloads of contrast, causal, kinds and the rest
#: all carry an identification tree too, drawn for the page.
READERS = (_within, _attribute, _above, _parts, _choice, _definition, _kinds,
           _contrast, _causal, _bridge, _backwards, _members, _profile,
           _identified, _listing, _refused)


def digest(payload: dict | None) -> dict | None:
    """`{kind, text, items, answers}` for what an answer holds, or None when
    all it holds is a verdict."""
    payload = payload or {}
    for reader in READERS:
        found = reader(payload)
        if found is not None:
            return found
    return None


# -- why ------------------------------------------------------------------

#: Verdicts that say it does, and that it does not.
YES = ("VERIFIED", "HELD", "INHERITED")
NO = ("CONTRADICTED", "DENIED")


def _plural(word: str) -> str:
    word = word or ""
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es" if not word.endswith("s") else word
    return word + "s"


def derivation(payload: dict) -> str:
    """What a yes or no rests on, from the answer that gave it.

    The walk says it: a fact stated of the kind itself (distance 0), or
    inherited from an ancestor so many levels up -- and the note beside it
    says what the trie and the corroborating kinds made of that."""
    note = (payload.get("note") or "").strip()
    rows = payload.get("evidence") or []
    subject = (name_of(payload.get("concept"))
               or (payload.get("parse") or {}).get("subject") or "it")
    if not rows:
        return note
    fact = rows[0]
    where = name_of(fact.get("concept"))
    relation = fact.get("relation") or ""
    recorded = (f"{where} {SAYS.get(relation, relation.replace('_', ' '))} "
                f"“{fact.get('object')}”"
                + (f" ({fact.get('source')})" if fact.get("source") else ""))
    distance = fact.get("distance") or 0
    text = (f"{subject} is a kind of {where}, {distance} "
            f"level{'s' if distance > 1 else ''} up, and {recorded}"
            if distance else f"it is recorded of {where} itself: {recorded}")
    return f"{text}. {note}" if note else text


def because(payload: dict | None, negative: bool = False,
            requirement: dict | None = None) -> dict | None:
    """The answer to a why: whether its premise holds, and what that rests
    on -- the derivation, and what the doers of it have in common.

    `negative` is the premise's polarity: `why can't a penguin fly` asks the
    grounds of a no. `requirement` is `{part, action, holders, doers, has,
    decisive}` from `graph.Requirements`: `fly` needs a wing, 31 of the 87
    things recorded as flying have one, and whether this subject does."""
    payload = payload or {}
    verdict = payload.get("verdict") or ""
    subject = ((payload.get("parse") or {}).get("subject")
               or name_of(payload.get("concept")) or "it")
    reason = derivation(payload)
    if reason[:1].isupper() and reason[1:2].islower():
        reason = reason[:1].lower() + reason[1:]      # `The norms state ...`
    if verdict not in YES + NO:
        text = ("nothing recorded settles whether it is so, so there is no "
                "why to give" + (f": {reason}" if reason else ""))
        return _found("why", text, [])
    holds = (verdict in YES) != negative
    if holds:
        text = f"because {reason}" if reason else f"v687 answers {verdict}"
    else:
        text = (f"the premise does not hold — v687 answers {verdict}"
                + (f": {reason}" if reason else ""))
    if requirement and requirement.get("part"):
        part = _plural(requirement["part"])
        share = (f"{requirement.get('holders')} of the "
                 f"{requirement.get('doers')} things recorded as able to "
                 f"{requirement.get('action')} have them")
        has = requirement.get("has")
        # Only a requirement that leads the field says what the doing needs:
        # `fly -> wing`. Swimmers have teeth too, and a dog having teeth is no
        # part of why it swims.
        if has is True and verdict in YES and requirement.get("decisive"):
            text += f"; and {subject} has {part}, which is what those that " \
                    f"{requirement.get('action')} have in common ({share})"
        elif has is True and requirement.get("decisive"):
            text += (f"; and it is not a matter of anatomy: {subject} has "
                     f"{part} ({share})")
        elif has is False:
            text += (f"; {subject} is not recorded as having {part} "
                     f"({share})")
    return _found("why", text, [])

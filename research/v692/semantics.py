"""Mathematics in v687's reasoner (R33): what is true of numbers and kinds.

Two kinds of question the store could never answer, because the answer is
not a fact anybody wrote down but one worked out:

**About an object.** *Is 91 prime? Is 84 divisible by 7? Is 2 + 2 = 5?* The
maths heads of the encoder read what is asked (`reading.py`) and it is done
exactly (`doing.py`). `VERIFIED` or `CONTRADICTED`, with the reason -- `91 =
7 times 13` -- as the note.

**About a kind.** *Are all primes odd? Can a prime be even? Is every
perfect square a natural number?* v687's own parse reads the subject and
the target, R20's quantifier reading says how many (`logic.parse`), and a
kind whose members can be listed -- the curriculum's number kinds -- is
checked on them: `no, 2 is a prime number and not an odd number`. The store
has `prime number has_property odd` from a crawl, and it is wrong about 2;
this is what corrects it, and says how far the check went, because checking
the first thousand is not a proof.

It is a layer of the reasoner (`reasoning.contributes`), tried before R18,
whose table refuses arithmetic by name. So v688's loop, which asks v687
through its pool, gets it without knowing it is there, and so does every
page.
"""
from __future__ import annotations

import sympy as S

from research.v692 import curriculum as C
from research.v692 import doing, reading

#: The rule, as the page names rules.
RULE = "R33"
#: How far a kind's members are checked: the integers from `LOW` to `HIGH`.
LOW, HIGH = -1000, 1000
#: Acts the reasoner answers: those whose answer is a yes or a no.
TRUTHS = frozenset({"is kind", "divides", "check", "subset"})


def answer(engine, question: str) -> dict | None:
    """R33, or None to let the next layer try."""
    found = _object(engine, question)
    if found is not None:
        return found
    return _kinds(engine, question)


def _object(engine, question: str) -> dict | None:
    read = reading.read(question)
    if read is None or read.act not in TRUTHS or read.trouble:
        return None
    result = doing.do(read.act, dict(read.parts, KIND=read.kind))
    if result.stance not in ("yes", "no"):
        return None
    verdict = "VERIFIED" if result.stance == "yes" else "CONTRADICTED"
    note = result.text + (f": {result.because}" if result.because else "")
    steps = [engine._step(0, RULE, result.about or question, "read",
                          f"read as {read.act}: "
                          + "; ".join(f"{role} {symbols}" for role, symbols
                                      in read.symbols.items())),
             engine._step(1, RULE, result.about or question, "worked out",
                          note, kind="match" if verdict == "VERIFIED"
                          else "stop")]
    payload = engine._shell(question, verdict, RULE,
                            concept=_concept(read), note=note, steps=steps,
                            extra={"mathematics": {
                                "act": read.act, "stance": result.stance,
                                "text": result.text,
                                "because": result.because}})
    payload["confidence"] = 1.0
    return payload


def _concept(read) -> str | None:
    kind = read.kind
    if kind is None:
        return None
    return kind.concept or kind.under or None


def members(kind: C.Kind) -> list:
    """What a kind of number has in the range checked, in order."""
    if kind.of != "number":
        return []
    return [S.Integer(value) for value in range(LOW, HIGH + 1)
            if kind.holds(S.Integer(value))]


def _kind_of(word: str | None) -> C.Kind | None:
    if not word:
        return None
    plain = word.replace("_", " ").split(".")[0]
    found = C.kind_said(plain)
    if found is not None:
        return found
    for kind in C.KINDS:
        if plain in (kind.concept.split(".")[0], kind.name):
            return kind
    return None


def _kinds(engine, question: str) -> dict | None:
    """A claim about a kind of number, checked on its members."""
    from research.v687 import logic
    try:
        parse = engine.parser.parse(question)
    except Exception:                                # noqa: BLE001
        return None
    subject = _kind_of(getattr(parse, "subject", None))
    target = _kind_of(getattr(parse, "target", None))
    if subject is None or target is None or subject.of != "number" \
            or target.of != "number":
        return None
    quantifier = logic.parse(question, frozenset()).quantifier
    if getattr(parse, "relation", "") == "capable_of" and quantifier is None:
        # `can a prime be even`: is there one.
        quantifier = "some"
    found = members(subject)
    if not found:
        return None
    both = [one for one in found if target.holds(one)]
    not_both = [one for one in found if not target.holds(one)]
    checked = (f"of the {len(found)} {subject.name}s from {LOW} to {HIGH}"
               f", {len(both)} are {target.said[0]}")
    if quantifier == "some":
        holds = bool(both)
        example = both[0] if both else None
        note = (f"Yes: {example} is {subject.said[0]} and {target.said[0]}"
                if holds else f"None I checked: {checked}")
        verdict = "VERIFIED" if holds else "UNKNOWN"
    elif quantifier == "none":
        holds = not both
        note = (f"None: {checked}" if holds else
                f"No: {both[0]} is {subject.said[0]} and {target.said[0]}")
        verdict = "VERIFIED" if holds else "CONTRADICTED"
    else:
        holds = not not_both
        if holds:
            # Every one checked: true as far as it went, which is not a
            # proof -- unless the store's own taxonomy says so, and then R1
            # would have answered first.
            note = (f"Every one I checked: {checked}. That is evidence, not "
                    f"a proof")
            verdict = "VERIFIED"
        else:
            counter = not_both[0] if len(not_both) < len(found) / 2 else None
            note = (f"No: {not_both[0]} is {subject.said[0]} and not "
                    f"{target.said[0]} -- {checked}")
            if counter is None:
                note = f"No: {checked}"
            verdict = "CONTRADICTED"
    steps = [engine._step(0, RULE, subject.concept or subject.under,
                          subject.name, f"the {subject.name}s from {LOW} to "
                                        f"{HIGH}: {len(found)}"),
             engine._step(1, RULE, target.concept or target.under,
                          target.name, note,
                          kind="match" if verdict == "VERIFIED" else "stop")]
    payload = engine._shell(question, verdict, RULE,
                            concept=subject.concept or subject.under,
                            note=note, steps=steps,
                            extra={"mathematics": {
                                "act": "kind claim", "quantifier": quantifier,
                                "checked": len(found), "holding": len(both)}})
    payload["confidence"] = 1.0 if verdict == "CONTRADICTED" else 0.9
    return payload


def layer(engine, question: str, when):
    """The operator R33 is, for `ReasoningEngine.layers`."""
    from research.v687.executive import Operator, attempt
    return Operator("mathematics",
                    attempt(lambda: answer(engine, question), slot="payload"),
                    rule=RULE, proposes=when)


def register() -> None:
    from research.v687 import reasoning
    reasoning.contributes(layer)


register()

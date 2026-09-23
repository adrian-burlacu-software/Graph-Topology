"""Design goals said in English, every word labelled: what the encoder
learns to read them from.

    python -m research.v693.stating            # llm/design-data

v692 taught the encoder mathematics by saying objects aloud; this says
**specifications** aloud. A goal made from an object (`generating.goal`)
is put as a person would ask for it -- *find a quadratic whose roots are 2
and minus 3 and whose value at 0 is 12* -- and every word carries its part
and the symbols it stands for:

    DKIND    the kind asked for; its label is the kind, or a kind with a
             degree (`quadratic`)
    CLAUSE   the word naming what is constrained; its label is the clause
             (`root`, `value`, `slope`, `term`, `start`, `through`, `monic`)
    AT       where: the 0 in *value at 0*, the 5 in *5th term*
    IS       what: the 12 in *is 12*, each root, each term a sequence
             starts with, each point it passes through
    MULT     how many times a root is one: *double*, *multiplicity 3*

Reading them back is `reading.spec_of`, which only pairs the parts up; what
each word is, the encoder says. A record is kept only when reading its own
labels back gives the specification it was said for -- the round trip, as
everywhere in this project.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as S

from research.v692.corpus import words
from research.v692.saying import DROP, Speaker, ordinal
from research.v693 import generating
from research.v693.spec import Spec

LLM = Path(__file__).resolve().parents[2] / "llm"
OUT = LLM / "design-data"
ACT = "design"
O = "O"

#: How a design goal opens, `{K}` the kind and `{C}` what it must satisfy.
OPENERS = ("find a {K} with {C}", "find a {K} {C}", "design a {K} with {C}",
           "give me a {K} with {C}", "i want a {K} {C}",
           "can you find a {K} with {C}", "make a {K} with {C}",
           "what {K} has {C}", "come up with a {K} with {C}",
           "construct a {K} {C}", "write down a {K} with {C}",
           "is there a {K} with {C}", "build a {K} with {C}")
#: Kinds as they are said, with the label each word carries.
KIND_WORDS = {"polynomial": (("polynomial", "polynomial"),),
              "sequence": (("sequence", "sequence"),
                           ("progression", "sequence")),
              "function": (("function", "function"),)}
DEGREE_WORDS = {1: "line", 2: "quadratic", 3: "cubic", 4: "quartic"}


def plain(text: str) -> list:
    return [(word, O, DROP) for word in words(text)]


def cue(text: str, clause: str, role: str = "CLAUSE") -> list:
    """A phrase naming a clause: its last word carries the label."""
    split = words(text)
    return [(word, role, clause if index == len(split) - 1 else DROP)
            for index, word in enumerate(split)]


def said(obj, role: str, speaker: Speaker) -> list:
    obj = S.sympify(obj)
    if obj.is_Rational and speaker.rng.random() < 0.3:
        # As it is typed: `-3`, `1/2` -- split as the page splits it.
        return typed(str(obj), role)
    found = speaker.say(obj)
    out = []
    for word, label in zip(found.words, found.labels):
        pieces = words(word)
        for piece in pieces:
            out.append((piece, role, label if piece == pieces[-1]
                        or label == "KEEP" else DROP))
    return out


def typed(text: str, role: str) -> list:
    """Something as it is typed, split as the page splits it (`words`):
    each piece its own symbols, a minus standing for the minus."""
    return [(piece, role, "-" if piece == "-" else "KEEP")
            for piece in words(text)]


def listed(items: list, role: str, speaker: Speaker, rng) -> list:
    out = []
    for index, one in enumerate(items):
        if index:
            out += plain("and" if index == len(items) - 1
                         else rng.choice((",", ",", "and")))
        out += said(one, role, speaker)
    return out


def _at(value, speaker, rng) -> list:
    return said(value, "AT", speaker)


# -- one phrase for each group of clauses ---------------------------------------

def roots_phrase(roots: list, speaker, rng) -> list:
    """Simple roots together (`roots 2 and -3`), a repeated one alone."""
    simple = [clause.args[0] for clause in roots
              if len(clause.args) == 1 or clause.args[1] == 1]
    out = []
    if simple:
        many = len(simple) > 1
        form = rng.random()
        if form < 0.4:
            out += cue("roots" if many else "a root", "root")
            if not many or rng.random() < 0.5:
                out += plain("at")
        elif form < 0.65:
            out += plain("whose") + cue("roots" if many else "root", "root")
            out += plain("are" if many else "is")
        elif form < 0.85:
            out += plain("a" if not many else "")
            out += cue("zeros" if many else "zero", "root") + plain("at")
        else:
            out += plain("that is") + cue("zero", "root") + plain("at")
        out += listed(simple, "IS", speaker, rng)
    for clause in roots:
        if len(clause.args) > 1 and clause.args[1] != 1:
            times = int(clause.args[1])
            if out:
                out += plain("and")
            if times == 2 and rng.random() < 0.6:
                out += [("a", O, DROP), ("double", "MULT", "2")]
                out += cue("root", "root") + plain("at")
            elif times == 3 and rng.random() < 0.5:
                out += [("a", O, DROP), ("triple", "MULT", "3")]
                out += cue("root", "root") + plain("at")
            else:
                out += plain("a") + cue("root", "root")
                out += plain("of multiplicity")
                out += said(S.Integer(times), "MULT", speaker)
                out += plain("at")
            out += said(clause.args[0], "IS", speaker)
    return out


def pair_phrase(clause, speaker, rng) -> list:
    """`value 12 at 0`, `whose value at 0 is 12`, `slope 3 at 1`."""
    kind = clause.kind
    at, is_ = clause.args
    name = {"value": ("value",), "slope": ("slope", "gradient",
                                           "derivative")}[kind]
    word = rng.choice(name)
    form = rng.random()
    if form < 0.4:
        return (cue(word, kind) + said(is_, "IS", speaker) + plain("at")
                + _at(at, speaker, rng))
    if form < 0.8:
        return (plain("whose") + cue(word, kind) + plain("at")
                + _at(at, speaker, rng) + plain("is")
                + said(is_, "IS", speaker))
    return (plain("a") + cue(word, kind) + plain("of")
            + said(is_, "IS", speaker) + plain("at")
            + _at(at, speaker, rng))


def through_phrase(clauses: list, speaker, rng) -> list:
    """`through (0, 12) and (1, 5)`: values as points."""
    points = [S.Tuple(one.args[0], one.args[1]) for one in clauses]
    word = rng.choice(("through", "passing through", "going through"))
    out = cue(word, "through")
    for index, point in enumerate(points):
        if index:
            out += plain("and" if index == len(points) - 1 else ",")
        if rng.random() < 0.5:
            # As it is typed, with a space: `(0,` and `-2)`, each word
            # its own symbols.
            out += typed(f"({point[0]}, {point[1]})", "IS")
            continue
        speaker_style = Speaker(speaker.rng, "written")
        out += said(point, "IS", speaker_style)
    return out


def derivative_phrase(clause, speaker, rng) -> list:
    """`whose derivative is 2x cos(x^2)`."""
    return (plain("whose") + cue("derivative", "derivative")
            + plain("is") + said(clause.args[0], "IS", speaker))


def rate_phrase(clause, speaker, rng) -> list | None:
    """`whose derivative is 3 times itself`, `whose second derivative is
    minus itself`: an equation said the way it is thought of. The number is
    how many times, so it is a MULT, and it stands next to the clause."""
    second, first, zeroth = (clause.args + (0,))[:3]
    if second == 0 and first == 1:
        label, opener = "rate", "whose derivative is"
    elif second == 1 and first == 0:
        label, opener = "second-rate", "whose second derivative is"
    else:
        return None
    times = -S.sympify(zeroth) / (first if second == 0 else second)
    out = plain("whose")
    words = opener.split()[1:]
    out += [(word, "CLAUSE", "DROP") for word in words[:-1]]
    out += plain(words[-1])
    if times == 1:
        out += [("equal", "CLAUSE", "DROP"), ("to", "CLAUSE", "DROP")]
        return out + [("itself", "CLAUSE", label)]
    if times == -1:
        return out + [("minus", "MULT", "-1"), ("itself", "CLAUSE", label)]
    out += said(times, "MULT", speaker)
    return out + [("times", "CLAUSE", "DROP"), ("itself", "CLAUSE", label)]


def stationary_phrase(clause, speaker, rng) -> list:
    word = rng.choice(("a turning point", "a stationary point",
                       "a turning point", "a flat point"))
    return cue(word, "stationary") + plain("at") + _at(clause.args[0],
                                                       speaker, rng)


def leading_phrase(clause, speaker, rng) -> list:
    if clause.args[0] == 1 and rng.random() < 0.4:
        return cue("monic", "monic") + plain("")
    if rng.random() < 0.5:
        return (cue("leading coefficient", "leading")
                + said(clause.args[0], "IS", speaker))
    return (plain("whose") + cue("leading coefficient", "leading")
            + plain("is") + said(clause.args[0], "IS", speaker))


def degree_phrase(clause, speaker, rng) -> list:
    return (plain("of") + cue("degree", "degree")
            + said(clause.args[0], "IS", speaker))


def integer_phrase(clause, speaker, rng) -> list:
    return cue(rng.choice(("whole-number coefficients",
                           "integer coefficients",
                           "whole number coefficients")), "integer")


def term_phrase(clause, speaker, rng) -> list:
    at, is_ = int(clause.args[0]), clause.args[1]
    place = [(ordinal(at, rng).words[-1], "AT", str(at))]
    form = rng.random()
    if form < 0.5:
        return (plain("whose") + place + cue("term", "term") + plain("is")
                + said(is_, "IS", speaker))
    if form < 0.8:
        return (plain("a") + place + cue("term", "term") + plain("of")
                + said(is_, "IS", speaker))
    return (cue("term", "term") + said(S.Integer(at), "AT", speaker)
            + plain("equal to") + said(is_, "IS", speaker))


def total_phrase(clause, speaker, rng) -> list:
    count, is_ = int(clause.args[0]), clause.args[1]
    if rng.random() < 0.5:
        return (plain("whose first") + said(S.Integer(count), "AT", speaker)
                + cue("terms add up to", "total")
                + said(is_, "IS", speaker))
    return (plain("where the") + cue("sum", "total") + plain("of the first")
            + said(S.Integer(count), "AT", speaker) + plain("terms is")
            + said(is_, "IS", speaker))


def start_phrase(clauses: list, speaker, rng) -> list:
    """`starting 3, 7, 11`: terms from the first on."""
    values = [one.args[1] for one in clauses]
    word = rng.choice(("starting", "beginning", "that starts",
                       "that begins", "going"))
    return cue(word, "start") + listed(values, "IS", speaker, rng)


# -- a whole goal -----------------------------------------------------------------

def _consecutive_from_one(terms: list) -> int:
    """How many of the terms are the 1st, 2nd, 3rd... in a row."""
    places = sorted(int(one.args[0]) for one in terms)
    count = 0
    for expected, place in zip(range(1, len(places) + 1), places):
        if place != expected:
            break
        count += 1
    return count


def stated(spec: Spec, rng: random.Random) -> tuple | None:
    """(the words with their parts and labels, the specification as said)
    -- which may differ from `spec` in form, not in meaning: a degree said
    as `quadratic` is still degree 2."""
    speaker = Speaker(rng, "mixed")
    clauses = list(spec.clauses)
    phrases = []
    kind_phrase = None
    degree = spec.of("degree")
    if spec.kind == "polynomial" and degree and int(
            degree[0].args[0]) in DEGREE_WORDS and rng.random() < 0.7:
        word = DEGREE_WORDS[int(degree[0].args[0])]
        kind_phrase = [(word, "DKIND", word)]
        clauses = [one for one in clauses if one.kind != "degree"]
    if kind_phrase is None:
        word, label = rng.choice(KIND_WORDS[spec.kind])
        kind_phrase = [(word, "DKIND", label)]
    roots = [one for one in clauses if one.kind == "root"]
    if roots:
        phrases.append(roots_phrase(roots, speaker, rng))
    values = [one for one in clauses if one.kind == "value"]
    if len(values) >= 2 and rng.random() < 0.4:
        phrases.append(through_phrase(values, speaker, rng))
        values = []
    for clause in values + [one for one in clauses if one.kind == "slope"]:
        phrases.append(pair_phrase(clause, speaker, rng))
    for clause in clauses:
        make = {"stationary": stationary_phrase, "leading": leading_phrase,
                "degree": degree_phrase, "integer": integer_phrase,
                "total": total_phrase, "derivative": derivative_phrase,
                "ode": rate_phrase}.get(clause.kind)
        if make is not None:
            phrase = make(clause, speaker, rng)
            if phrase is None:
                # Not something said in words: `f'' - 3f' + 2f = 0`.
                return None
            phrases.append(phrase)
    terms = [one for one in clauses if one.kind == "term"]
    run = _consecutive_from_one(terms)
    if run >= 2 and rng.random() < 0.7:
        ordered = sorted(terms, key=lambda one: int(one.args[0]))
        phrases.append(start_phrase(ordered[:run], speaker, rng))
        terms = ordered[run:]
    for clause in terms:
        phrases.append(term_phrase(clause, speaker, rng))
    rng.shuffle(phrases)
    body = []
    for index, phrase in enumerate(phrases):
        if index:
            body += plain("and" if index == len(phrases) - 1
                          else rng.choice((",", "and")))
        body += [one for one in phrase if one[0]]
    opener = rng.choice(OPENERS)
    if opener.startswith("i want") or opener.startswith("find a {K} {C}") \
            or opener.startswith("construct"):
        if body and body[0][1] == "IS":
            opener = opener.replace("{C}", "with {C}")
    out = []
    for piece in opener.replace("{K}", "\0K\0").replace(
            "{C}", "\0C\0").split("\0"):
        if piece == "K":
            out += kind_phrase
        elif piece == "C":
            out += body
        else:
            out += plain(piece)
    return out, spec


def canonical(spec: Spec) -> tuple:
    """What a specification says, in one form: for comparing what was
    said with what was read."""
    out = [spec.kind]
    for clause in spec.clauses:
        args = tuple(S.nsimplify(one) for one in clause.args)
        if clause.kind == "root" and len(args) > 1 and args[1] == 1:
            args = args[:1]
        out.append((clause.kind,) + tuple(str(one) for one in args))
    return tuple([out[0]] + sorted(out[1:]))


def record(rng: random.Random) -> dict | None:
    from research.v693.reading import spec_of
    spec = generating.goal(rng)
    if spec is None:
        return None
    made = stated(spec, rng)
    if made is None:
        return None
    tokens, spec = made
    if len(tokens) > 60:
        return None
    said_words = [word for word, _, _ in tokens]
    roles = [role for _, role, _ in tokens]
    labels = [label for _, _, label in tokens]
    try:
        back = spec_of(said_words, roles, labels)
    except Exception:                                # noqa: BLE001
        return None
    if back is None or canonical(back) != canonical(spec):
        return None
    return {"task": "math", "words": said_words, "act": ACT,
            "roles": roles, "symbols": labels,
            "tags": [""] * len(said_words), "deps": [""] * len(said_words),
            "names": [], "said": " ".join(said_words), "how": "design"}


#: Things said with a designer's words that ask for no mathematics at
#: all. **Only those**: in this task `none` means no mathematics, so a
#: maths question put here -- *solve x^2 - 5x + 6 = 0* -- teaches the
#: encoder that it is not one. That was done once, and v692's own bank
#: fell from 98 to 91; its acts are taught by v692's corpus, not here.
NEAR = ("design a house", "find my keys", "i want a pizza",
        "can you find a restaurant", "give me a hand", "make a cake",
        "build a bridge", "construct a sentence", "write down my name",
        "come up with a plan", "find a word that rhymes with cat",
        "i want a dog with brown fur", "find a book with a red cover",
        "is there a shop with bread", "what is a polynomial",
        "what is a sequence", "what is a root", "what is a quadratic",
        "the sequence of events was odd", "the house has a flat roof",
        "a turning point in history", "my degree is in physics",
        "find a function for the party",
        "i need a sequence of photos")


def build(count: int, seed: int = 696, out: Path = OUT) -> dict:
    rng = random.Random(seed)
    rows, tried = [], 0
    while len(rows) < count and tried < count * 4:
        tried += 1
        one = record(rng)
        if one is not None:
            rows.append(one)
    for _ in range(count // 10):
        text = rng.choice(NEAR)
        said_words = words(text)
        rows.append({"task": "math", "words": said_words, "act": "none",
                     "roles": [O] * len(said_words),
                     "symbols": [DROP] * len(said_words),
                     "tags": [""] * len(said_words),
                     "deps": [""] * len(said_words), "names": [],
                     "said": " ".join(said_words), "how": "design-near"})
    rng.shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train-design.jsonl", "w", encoding="utf-8") as train, \
            open(out / "valid-design.jsonl", "w", encoding="utf-8") as held:
        for index, one in enumerate(rows):
            (held if index % 10 == 0 else train).write(json.dumps(one)
                                                       + "\n")
    stats = {"records": len(rows), "kept of tried": f"{count}/{tried}"}
    (out / "stats.json").write_text(json.dumps(stats, indent=1),
                                    encoding="utf-8")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=20000)
    options = parser.parse_args(argv)
    print(json.dumps(build(options.count), indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())

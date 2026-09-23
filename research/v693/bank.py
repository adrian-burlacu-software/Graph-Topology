"""Design goals, written by hand before the designer was run on any of them.

    python -m research.v693.bank            # first run, then again with
                                            # what the first taught

Each goal is a specification (`spec.Spec`). There is no expected answer:
a design is right when the specification holds of it, checked exactly, and
a goal no object can meet (`possible=False`) is right only when the
designer says it cannot. Scored:

    designed    an object came back and the specification holds of it
    honest      designed, or said to be impossible -- never a wrong object
    first       the first design move tried was the one that worked
    tries       design moves tried, in all

    simplest    the design used the form with the fewest unknowns of
                those that fit (`designing.simplest`): 2·3ⁿ⁻¹, not a cubic

The bank is run in table order twice with one learner -- the second pass
is what the first taught, so *tries* going down is the designer getting
better with use -- and then with the learned proposer ordering the forms
(`proposer.py`), which was trained on generated goals and never on these.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from research.v693.spec import C, Spec


@dataclass
class Goal:
    spec: Spec
    possible: bool = True


def P(text, *clauses, possible=True) -> Goal:
    return Goal(Spec("polynomial", clauses, text), possible)


def Q(text, *clauses, possible=True) -> Goal:
    return Goal(Spec("sequence", clauses, text), possible)


def F(text, *clauses, possible=True) -> Goal:
    return Goal(Spec("function", clauses, text), possible)


BANK = [
    # -- polynomials ---------------------------------------------------------
    P("a polynomial with roots 2 and -3 whose value at 0 is 12",
      C("root", 2), C("root", -3), C("value", 0, 12)),
    P("a monic polynomial with roots 1, 2 and 3",
      C("root", 1), C("root", 2), C("root", 3), C("leading", 1)),
    P("a quadratic through (0, 1), (1, 3) and (2, 7)",
      C("degree", 2), C("value", 0, 1), C("value", 1, 3), C("value", 2, 7)),
    P("a polynomial through (1, 2), (2, 5) and (3, 10)",
      C("value", 1, 2), C("value", 2, 5), C("value", 3, 10)),
    P("a polynomial with a double root at 4 and value 16 at 0",
      C("root", 4, 2), C("value", 0, 16)),
    P("a monic quadratic with a turning point at 1 where it is 0",
      C("degree", 2), C("leading", 1), C("stationary", 1), C("value", 1, 0)),
    P("a polynomial with roots 0 and 5 whose slope at 0 is 10",
      C("root", 0), C("root", 5), C("slope", 0, 10)),
    P("a polynomial with a root at 1, value 2 at 0 and value 6 at 2",
      C("root", 1), C("value", 0, 2), C("value", 2, 6)),
    P("a cubic with roots 1 and -1, value 3 at 0 and leading coefficient 2",
      C("degree", 3), C("root", 1), C("root", -1), C("value", 0, 3),
      C("leading", 2)),
    P("a polynomial with whole-number coefficients and roots 1/2 and 3",
      C("root", "1/2"), C("root", 3), C("integer")),
    P("a polynomial through (1, 1), (2, 4), (3, 9) and (4, 16)",
      C("value", 1, 1), C("value", 2, 4), C("value", 3, 9),
      C("value", 4, 16)),
    P("a polynomial with slope 3 at 0, slope 5 at 1 and value 1 at 0",
      C("slope", 0, 3), C("slope", 1, 5), C("value", 0, 1)),
    P("a monic cubic with turning points at 2 and -2 and value 0 at 0",
      C("degree", 3), C("leading", 1), C("stationary", 2),
      C("stationary", -2), C("value", 0, 0)),
    P("a polynomial with roots 2 and 5 and value 8 at 3 and at 4",
      C("root", 2), C("root", 5), C("value", 3, 8), C("value", 4, 8)),
    P("a line with roots 1 and 2", C("degree", 1), C("root", 1),
      C("root", 2), possible=False),
    P("a quadratic with roots 1, 2 and 3", C("degree", 2), C("root", 1),
      C("root", 2), C("root", 3), possible=False),
    # -- sequences -------------------------------------------------------------
    Q("a sequence whose 5th term is 48 and 7th is 192",
      C("term", 5, 48), C("term", 7, 192)),
    Q("a sequence starting 3, 7, 11, 15",
      C("term", 1, 3), C("term", 2, 7), C("term", 3, 11), C("term", 4, 15)),
    Q("a sequence starting 2, 6, 18, 54",
      C("term", 1, 2), C("term", 2, 6), C("term", 3, 18), C("term", 4, 54)),
    Q("a sequence starting 1, 4, 9, 16",
      C("term", 1, 1), C("term", 2, 4), C("term", 3, 9), C("term", 4, 16)),
    Q("a sequence starting at 5 whose first 10 terms add to 275",
      C("term", 1, 5), C("total", 10, 275)),
    Q("a sequence whose 2nd term is -6 and 4th is -24",
      C("term", 2, -6), C("term", 4, -24)),
    Q("a sequence starting 2 whose 3rd term is -8",
      C("term", 1, 2), C("term", 3, -8)),
    Q("a sequence starting 1, 1, 2, 3, 5",
      C("term", 1, 1), C("term", 2, 1), C("term", 3, 2), C("term", 4, 3),
      C("term", 5, 5)),
    Q("a sequence starting 3, 6, 12 whose first 5 terms add to 93",
      C("term", 1, 3), C("term", 2, 6), C("term", 3, 12),
      C("total", 5, 93)),
    Q("a sequence starting 5, 15, 45, 135",
      C("term", 1, 5), C("term", 2, 15), C("term", 3, 45),
      C("term", 4, 135)),
    Q("a sequence starting 10, 7, 4, 1",
      C("term", 1, 10), C("term", 2, 7), C("term", 3, 4), C("term", 4, 1)),
    Q("a sequence whose 1st term is 2 and 1st term is 3",
      C("term", 1, 2), C("term", 1, 3), possible=False),
    # -- added with composed forms, before the proposer saw any ------------
    Q("a sequence starting 1, 3, 7, 15, 31",
      C("term", 1, 1), C("term", 2, 3), C("term", 3, 7), C("term", 4, 15),
      C("term", 5, 31)),
    Q("a sequence starting 3, 5, 9, 17, 33",
      C("term", 1, 3), C("term", 2, 5), C("term", 3, 9), C("term", 4, 17),
      C("term", 5, 33)),
    Q("a sequence starting 1, 1, 2, 3, 5, 8",
      C("term", 1, 1), C("term", 2, 1), C("term", 3, 2), C("term", 4, 3),
      C("term", 5, 5), C("term", 6, 8)),
    Q("a sequence starting 5, 13, 35, 97, 275",
      C("term", 1, 5), C("term", 2, 13), C("term", 3, 35),
      C("term", 4, 97), C("term", 5, 275)),
    Q("a sequence starting 2, 3, 5, 9, 17, 33",
      C("term", 1, 2), C("term", 2, 3), C("term", 3, 5), C("term", 4, 9),
      C("term", 5, 17), C("term", 6, 33)),
    # -- functions: added with the kind -----------------------------------
    F("a function whose derivative is 2x cos(x^2) and value 1 at 0",
      C("derivative", "2*x*cos(x**2)"), C("value", 0, 1)),
    F("a function with f'' = -f, f(0) = 1 and f'(0) = 0",
      C("ode", 1, 0, 1), C("value", 0, 1), C("slope", 0, 0)),
    F("a function with f' = 3f and f(0) = 2",
      C("ode", 0, 1, -3), C("value", 0, 2)),
    F("a function with f'' - 3f' + 2f = 0, f(0) = 2 and f'(0) = 3",
      C("ode", 1, -3, 2), C("value", 0, 2), C("slope", 0, 3)),
    F("a function with f'' = -4f, f(0) = 0 and f'(0) = 2",
      C("ode", 1, 0, 4), C("value", 0, 0), C("slope", 0, 2)),
    F("a function with value 1 at 0, 3 at 1 and slope 2 at 0",
      C("value", 0, 1), C("value", 1, 3), C("slope", 0, 2)),
    F("a function whose derivative is 1/x and value 0 at 1",
      C("derivative", "1/x"), C("value", 1, 0)),
    F("a function with f' = f and f' = 2f",
      C("ode", 0, 1, -1), C("ode", 0, 1, -2), possible=False),
]


# -- the same, asked in English ----------------------------------------------

#: (what is said, the specification meant, or None where it asks for no
#: design, whether it can be met). Written before the encoder was taught to
#: read design goals, some in phrasings the generator never uses (`i need`,
#: `that goes through`), so reading them is generalising, not recalling.
ENGLISH = [
    ("find a polynomial with roots 2 and -3 whose value at 0 is 12",
     P("", C("root", 2), C("root", -3), C("value", 0, 12))),
    ("give me a quadratic with roots 1 and 4",
     P("", C("degree", 2), C("root", 1), C("root", 4))),
    ("i need a cubic with roots 0, 1 and 2",
     P("", C("degree", 3), C("root", 0), C("root", 1), C("root", 2))),
    ("find a monic polynomial with roots 1, 2 and 3",
     P("", C("leading", 1), C("root", 1), C("root", 2), C("root", 3))),
    ("design a polynomial passing through (0, 1), (1, 3) and (2, 7)",
     P("", C("value", 0, 1), C("value", 1, 3), C("value", 2, 7))),
    ("find a quadratic through (1, 2), (2, 5) and (3, 10)",
     P("", C("degree", 2), C("value", 1, 2), C("value", 2, 5),
       C("value", 3, 10))),
    ("find a polynomial with a double root at 4 and value 16 at 0",
     P("", C("root", 4, 2), C("value", 0, 16))),
    ("find a quadratic with a turning point at 1 whose value at 1 is 0",
     P("", C("degree", 2), C("stationary", 1), C("value", 1, 0))),
    ("find a polynomial whose roots are 0 and 5 and whose slope at 0 is 10",
     P("", C("root", 0), C("root", 5), C("slope", 0, 10))),
    ("find a polynomial with integer coefficients and roots 1/2 and 3",
     P("", C("integer"), C("root", "1/2"), C("root", 3))),
    ("find a line through (1, 3) and (2, 5)",
     P("", C("degree", 1), C("value", 1, 3), C("value", 2, 5))),
    ("come up with a cubic with roots 1 and -1 and leading coefficient 2",
     P("", C("degree", 3), C("root", 1), C("root", -1), C("leading", 2))),
    ("find a polynomial with value 3 at 1 and slope 0 at 1",
     P("", C("value", 1, 3), C("slope", 1, 0))),
    ("make a polynomial with a triple root at 2",
     P("", C("root", 2, 3))),
    ("find a polynomial with zeros at -1 and 1 that goes through (0, -2)",
     P("", C("root", -1), C("root", 1), C("value", 0, -2))),
    ("is there a quadratic with roots 1, 2 and 3",
     P("", C("degree", 2), C("root", 1), C("root", 2), C("root", 3),
       possible=False)),
    ("find a sequence starting 2, 6, 18, 54",
     Q("", C("term", 1, 2), C("term", 2, 6), C("term", 3, 18),
       C("term", 4, 54))),
    ("find a sequence starting 3, 7, 11, 15",
     Q("", C("term", 1, 3), C("term", 2, 7), C("term", 3, 11),
       C("term", 4, 15))),
    ("find a sequence whose 5th term is 48 and whose 7th term is 192",
     Q("", C("term", 5, 48), C("term", 7, 192))),
    ("give me a sequence that starts 1, 4, 9, 16",
     Q("", C("term", 1, 1), C("term", 2, 4), C("term", 3, 9),
       C("term", 4, 16))),
    ("find a sequence starting 5 whose first 10 terms add up to 275",
     Q("", C("term", 1, 5), C("total", 10, 275))),
    ("design a sequence with a 2nd term of -6 and a 4th term of -24",
     Q("", C("term", 2, -6), C("term", 4, -24))),
    ("find a progression beginning 10, 7, 4, 1",
     Q("", C("term", 1, 10), C("term", 2, 7), C("term", 3, 4),
       C("term", 4, 1))),
    ("i want a sequence whose first term is 4 and whose third term is 36",
     Q("", C("term", 1, 4), C("term", 3, 36))),
    # functions: added with the kind, before the encoder that reads
    # them was taught
    ("find a function whose derivative is 2x cos(x^2) and whose value at "
     "0 is 1",
     F("", C("derivative", "2*x*cos(x**2)"), C("value", 0, 1))),
    ("find a function whose derivative is 3 times itself and whose value "
     "at 0 is 2",
     F("", C("ode", 0, 1, -3), C("value", 0, 2))),
    ("i need a function whose second derivative is minus 4 times itself, "
     "with value 0 at 0 and slope 2 at 0",
     F("", C("ode", 1, 0, 4), C("value", 0, 0), C("slope", 0, 2))),
    ("find a function whose derivative is e^x and whose value at 0 is 3",
     F("", C("derivative", "exp(x)"), C("value", 0, 3))),
    ("give me a function whose second derivative is minus itself with "
     "value 1 at 0 and slope 0 at 0",
     F("", C("ode", 1, 0, 1), C("value", 0, 1), C("slope", 0, 0))),
    # not design goals: another act's, or not mathematics at all
    ("find the roots of x^2 - 4", None),
    ("what is the next term in 2, 6, 18", None),
    ("design a logo for my shop", None),
    ("find a sequence of steps to bake bread", None),
    ("what is a quadratic", None),
]


def run_english(verbose: bool = False) -> dict:
    """Read each by the encoder, design what was read, and check the
    design against what was meant (not against what was read)."""
    from research.v692 import doing, reading
    from research.v693 import page  # noqa: F401 (registers `design`)
    from research.v693.stating import canonical
    totals = {"said": 0, "read": 0, "understood": 0, "right": 0}
    for text, goal in ENGLISH:
        totals["said"] += 1
        read = reading.read(text)
        act = read.act if read is not None else "none"
        wanted = "design" if goal is not None else None
        read_ok = (act == "design") == (goal is not None)
        totals["read"] += read_ok
        if goal is None:
            totals["understood"] += read_ok
            totals["right"] += read_ok
            if verbose or not read_ok:
                print(f"{'ok ' if read_ok else 'NO '}{text}  -> {act}")
            continue
        spec = (read.parts.get("SPEC") if read is not None
                and act == "design" else None)
        understood = spec is not None and canonical(spec) == canonical(
            goal.spec)
        totals["understood"] += understood
        result = doing.do("design", {"SPEC": spec}) if spec else None
        design = (result.value if result is not None
                  and result.stance == "value" else None)
        right = (goal.spec.holds(design) if goal.possible
                 else design is None and read_ok)
        totals["right"] += right
        if verbose or not right or not understood:
            print(f"{'ok ' if right else 'NO '}{text}\n      read "
                  f"{act}: {spec.said() if spec else '-'}\n      -> "
                  f"{result.text if result is not None else '(none)'}")
    print(f"english: act read {totals['read']}/{totals['said']}, "
          f"understood exactly {totals['understood']}/{totals['said']}, "
          f"right {totals['right']}/{totals['said']}")
    return totals


def score(goal: Goal, found) -> dict:
    from research.v693 import designing
    right = (goal.spec.holds(found.design) if goal.possible
             else found.design is None)
    best = designing.best_fit(goal.spec)
    return {"designed": found.design is not None and goal.spec.holds(
                found.design),
            "right": right,
            "honest": right or found.design is None,
            "first": bool(found.tried) and found.tried[0][1] is not None,
            "simplest": (found.form.name == best if found.form is not None
                         else best is None),
            "tries": len(found.tried)}


KEYS = ("right", "honest", "first", "simplest", "tries")


def run(learner=None, verbose=False, label="", proposer=None) -> dict:
    from research.v693 import designing
    totals = {"goals": 0, **{key: 0 for key in KEYS}}
    for goal in BANK:
        order = proposer.order(goal.spec) if proposer is not None else None
        found = designing.design(goal.spec, learner=learner, order=order)
        mark = score(goal, found)
        totals["goals"] += 1
        for key in KEYS:
            totals[key] += mark[key]
        if verbose or not mark["right"]:
            shown = ("(none)" if found.design is None
                     else designing.shown(found.design, goal.spec))
            tried = ", ".join(f"{name}{'' if got is not None else ' x'}"
                              for name, got in found.tried)
            print(f"{'ok ' if mark['right'] else 'NO '}{goal.spec.said()}\n"
                  f"      -> {shown}   [{tried}]")
    print(f"{label}right {totals['right']}/{totals['goals']}, honest "
          f"{totals['honest']}/{totals['goals']}, first move worked "
          f"{totals['first']}/{totals['goals']}, simplest design "
          f"{totals['simplest']}/{totals['goals']}, {totals['tries']} moves "
          f"tried")
    return totals


def stream(count: int, curiosity: float, seed: int = 695,
           proposer=None) -> dict:
    """A learner's life: `count` generated goals designed one after
    another with one learner, then every lesson it holds checked against
    what fitting every form to every goal showed. A lesson is **false**
    when some goal contradicts it: a block whose literal held while the
    form fitted, a requirement that was missing while it fitted."""
    from research.v691 import learned, lessons
    from research.v693 import designing
    from research.v693.forms import BY_NAME
    from research.v693.generating import labelled
    rows = labelled(count, seed)
    learner = lessons.Learner(learned.Learned(None))
    moves = right = explored = 0
    for spec, _ in rows:
        order = proposer.order(spec) if proposer is not None else None
        found = designing.design(spec, learner, order=order,
                                 curiosity=curiosity)
        moves += len(found.tried)
        right += spec.holds(found.design)
        explored += len(found.explored)
    false = []
    for table, held in (("blocks", learner.learned.blockings()),
                        ("requires", learner.learned.requirements())):
        for verb, literal, _ in held:
            feature = literal.split()[0]
            name = verb[len("fit-"):]
            for spec, _ in rows:
                there = feature in spec.features()
                if there == (table == "blocks") and name in [
                        one.name for one in designing.usable(spec)] and \
                        designing.fit(BY_NAME[name], spec) is not None:
                    false.append(f"{verb} {table} {literal}")
                    break
    lessons_held = (len(learner.learned.blockings())
                    + len(learner.learned.requirements()))
    return {"goals": count, "right": right, "moves": moves,
            "explored": explored, "lessons": lessons_held,
            "false": false}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--english", action="store_true",
                        help="only the goals asked in English (needs an "
                             "encoder taught to read them)")
    parser.add_argument("--stream", type=int, default=0,
                        help="also design this many generated goals with "
                             "one learner, with and without curiosity")
    options = parser.parse_args(argv)
    if options.english:
        run_english(options.verbose)
        return 0
    if options.stream:
        from research.v693.designing import CURIOSITY
        for curiosity in (0.0, CURIOSITY):
            found = stream(options.stream, curiosity)
            print(f"stream of {found['goals']}, curiosity {curiosity}: "
                  f"right {found['right']}, {found['moves']} moves "
                  f"({found['explored']} exploring), {found['lessons']} "
                  f"lessons held, {len(found['false'])} false: "
                  f"{found['false']}")
        return 0
    from research.v691 import learned, lessons
    from research.v693.proposer import Proposer
    learner = lessons.Learner(learned.Learned(None))
    run(learner, options.verbose, "table order, first pass:   ")
    run(learner, options.verbose, "table order, second pass:  ")
    proposer = Proposer.load()
    if proposer is None:
        print("no proposer (python -m research.v693.proposer train)")
    else:
        run(None, options.verbose, "proposer:                  ",
            proposer)
        both = lessons.Learner(learned.Learned(None))
        run(both, options.verbose, "proposer + learner, first: ",
            proposer)
        run(both, options.verbose, "proposer + learner, second:",
            proposer)
    for verb, literal, _ in learner.learned.requirements():
        print(f"  learned: {verb} requires {literal}")
    for verb, literal, _ in learner.learned.blockings():
        print(f"  learned: {verb} is blocked by {literal}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())

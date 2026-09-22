"""How many: amounts in a world, and what acting does to them.

A world's facts were all true or false: `with book john` holds or it does
not. *Mary has three apples* is not like that. Giving her two does not make
a fact true, it makes a number bigger, and a planner that can only add and
delete facts cannot say it -- it would need a fact for every count, and an
action for every count to every other. So an amount is a fact of its own
kind, and this module is what knows how to read one.

## An amount rides in the predicate

    with=3 apple mary          mary has exactly 3 apples
    with=2+ apple mary         mary has at least 2 -- the count was not said
    at=5 bird tree             there are 5 birds in the tree
    with>=5 apple mary         a condition: that she has at least 5
    with+=2 apple mary         an order: that she have 2 more than now

**The amount is part of the predicate, so the arguments are still things.**
Everything that reads a fact's things -- `split()[1:]`, all through the
scene, the learner and the page -- reads `apple mary` and nothing else, and
everything that asks what a fact's predicate is sees `with=3`, which no
verb brings about and no placement rule touches. Nothing had to learn to
step round a number.

`apple` here is not one apple, it is the apples: a pile. What VerbNet says
about moving *a* thing is lifted to moving *n* of them (`lifted`), which is
the one idea that makes every verb that moves things a verb that moves
amounts, without a list of those verbs anywhere.

## What is not known

**A count nobody said is not zero.** Mary's apples are unknown until
somebody says, and a condition on an unknown count does not hold -- which
is what makes the planner ask (`scene.missing`) rather than assume. What
*is* known is that counts are not negative, so giving Mary two apples when
nobody said how many she had leaves her with at least two (`2+`), and
that is enough to answer *does she have any*.

The operations themselves are `numbers.py`'s: exact, and never learned.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from fractions import Fraction

from research.v691 import numbers

#: A predicate carrying an amount: `with=3`, `with=2+`, `with>=5`, `with+=2`.
SHAPE = re.compile(r"^([a-z][a-z_-]*?)(\+=|-=|>=|<=|=|>|<)(-?\d+(?:/\d+)?)"
                   r"(\+?)$")
#: The comparisons a condition can make.
COMPARE = frozenset({">=", "<=", ">", "<", "="})
#: What an order asks to change by, rather than to come to.
RELATIVE = frozenset({"+=", "-="})


@dataclass(frozen=True)
class Amount:
    """One literal with an amount in it, taken apart."""

    predicate: str
    op: str
    value: Fraction
    #: `2+`: at least this many, the true count not said
    at_least: bool
    args: tuple

    @property
    def fluent(self) -> str:
        """What the amount is of: `with apple mary`."""
        return " ".join((self.predicate,) + self.args)

    @property
    def kind(self) -> str:
        return self.args[0] if self.args else ""

    @property
    def holder(self) -> str:
        return self.args[1] if len(self.args) > 1 else ""


@functools.lru_cache(maxsize=65536)
def read(literal: str) -> Amount | None:
    """A literal's amount, or None for a literal with none."""
    parts = str(literal).split()
    if not parts:
        return None
    found = SHAPE.match(parts[0])
    if found is None:
        return None
    predicate, op, value, least = found.groups()
    return Amount(predicate, op, Fraction(value), bool(least),
                  tuple(parts[1:]))


def _number(value) -> str:
    return str(Fraction(value))


def amount(fluent: str, value, at_least: bool = False) -> str:
    """What is known: `with=3 apple mary`, or `with=2+ apple mary`."""
    predicate, *args = fluent.split()
    return " ".join([f"{predicate}={_number(value)}{'+' if at_least else ''}"]
                    + args)


def condition(fluent: str, op: str, value) -> str:
    """What is wanted of one: `with>=5 apple mary`."""
    predicate, *args = fluent.split()
    return " ".join([f"{predicate}{op}{_number(value)}"] + args)


def is_amount(literal: str) -> bool:
    """Whether a literal is something known about a count."""
    found = read(literal)
    return found is not None and found.op == "="


def is_condition(literal: str) -> bool:
    """Whether a literal asks something of a count, rather than stating one:
    `with>=5 apple mary`. An exact `with=5 apple mary` is both, and is kept
    as what is known, so it is not one of these."""
    found = read(literal)
    return found is not None and found.op in COMPARE - {"="}


def is_relative(literal: str) -> bool:
    found = read(literal)
    return found is not None and found.op in RELATIVE


def plain(literal: str) -> str:
    """The literal without its amount: `with apple mary`. What VerbNet's
    verbs are asked about, since a verb that gives a thing gives some."""
    found = read(literal)
    return found.fluent if found is not None else literal


def known(facts, fluent: str) -> tuple | None:
    """(count, whether it is only a lower bound) for a fluent, or None when
    nobody said."""
    for fact in facts:
        found = read(fact)
        if found is not None and found.op == "=" and found.fluent == fluent:
            return found.value, found.at_least
    return None


def holds(literal: str, facts) -> bool:
    """Whether a literal holds: membership for an ordinary fact, and for a
    condition, the count that is known compared as asked. A lower bound
    answers `at least`, and cannot answer `at most`."""
    found = read(literal)
    if found is None or found.op in RELATIVE:
        return literal in facts
    have = known(facts, found.fluent)
    if have is None:
        return False
    count, least = have
    op = ">=" if found.op == "=" and found.at_least else found.op
    if op == ">=":
        return count >= found.value
    if op == ">":
        return count > found.value
    if least:
        return False
    if op == "=":
        return count == found.value
    if op == "<=":
        return count <= found.value
    return count < found.value


def satisfied(literals, facts) -> bool:
    """Whether every literal holds. Fast where none has an amount, which is
    every declared domain: a set comparison, as it always was."""
    facts = facts if isinstance(facts, (set, frozenset)) else set(facts)
    for literal in literals:
        if literal in facts:
            continue
        if read(literal) is None or not holds(literal, facts):
            return False
    return True


def changed(facts, changes) -> frozenset:
    """The facts after amounts change: `(fluent, delta)` pairs.

    Exact stays exact and a lower bound stays one. A count nobody said,
    added to, is at least what was added -- counts are not negative -- and
    taken from, is still not known.
    """
    facts = set(facts)
    for fluent, delta in changes:
        was = known(facts, fluent)
        if was is not None:
            facts.discard(amount(fluent, *was))
            count, least = was
            if least and count + delta <= 0:
                # At least none is not a count.
                continue
            facts.add(amount(fluent, count + delta, least))
        elif delta > 0:
            facts.add(amount(fluent, delta, True))
    return frozenset(facts)


def toward(delta, wanted: str) -> bool:
    """Whether a change of `delta` moves a count toward what a condition
    asks. An exact `=` can be come at from either side."""
    found = read(wanted)
    if found is None:
        return False
    if found.op in (">=", ">"):
        return delta > 0
    if found.op in ("<=", "<"):
        return delta < 0
    return found.op == "=" and delta != 0


def closes(delta, wanted: str, facts) -> bool:
    """Whether changing the count by `delta` from what is known now would
    make the condition hold -- what an action is taken to promise."""
    found = read(wanted)
    if found is None or not toward(delta, wanted):
        return False
    return holds(wanted, changed(facts, ((found.fluent, delta),)))


def conditions(literals) -> frozenset:
    """The literals that ask something of a count, exact ones included."""
    return frozenset(one for one in literals
                     if (found := read(one)) is not None
                     and found.op in COMPARE)


def resolved(literal: str, facts) -> str | None:
    """An order's `with+=2 apple mary` as what it comes to, given what is
    known now: `with=5 apple mary` if she has 3, `with>=2 apple mary` if
    nobody said how many she has. None where it cannot be said -- taking 2
    from a count nobody knows -- which is a question to ask, not a goal."""
    found = read(literal)
    if found is None or found.op not in RELATIVE:
        return literal
    have = known(facts, found.fluent)
    delta = found.value if found.op == "+=" else -found.value
    if have is None:
        return (condition(found.fluent, ">=", delta) if delta > 0 else None)
    count, least = have
    if least:
        return (condition(found.fluent, ">=", count + delta)
                if delta > 0 else None)
    return amount(found.fluent, count + delta)


# -- acting on amounts -----------------------------------------------------

#: The relations a count of a kind can be kept under. A holder's apples are
#: one count whether VerbNet puts them *with* her (give-13.1, possession)
#: or *at* her (take-10.5, location).
HOLDING = ("with", "at")


def kept(facts) -> dict:
    """(kind, holder) -> the relation the scene keeps that count under."""
    out = {}
    for fact in facts:
        found = read(fact)
        if found is not None and found.op == "=" and len(found.args) == 2:
            out[found.args] = found.predicate
    return out


def lifted(action, kind: str, n, canon: dict | None = None) -> object | None:
    """A ground action over one thing, lifted to `n` of a kind.

    VerbNet's give-13.1 says the Theme was with the Agent and ends up with
    the Recipient. Of apples, that is: the giver must have at least `n`,
    has `n` fewer after, and the recipient `n` more. So every literal about
    the kind that the action needs and takes away becomes a condition and a
    decrease, and every one it brings about an increase -- read off the
    action, whatever the verb.

    **Amounts are conserved when the verb does not say where they come
    from.** put-9.1 says where a Theme ends up and nothing about where it
    was, which for one book is harmless and for three apples is a planner
    that makes apples. So where nothing is taken from anyone, and the doer
    is not the one who gains, what is moved is the doer's: the axiom `to
    move a thing you must have it`, the pile form of `verbs.AXIOMS`. A doer
    who gains -- buying, finding, catching -- gains from outside the world,
    and that is what those verbs mean.
    """
    from research.v691.world import Action
    n = Fraction(n)
    if n <= 0:
        return None

    canon = canon or {}

    def about(literal: str) -> bool:
        parts = literal.split()
        return (len(parts) == 3 and parts[1] == kind and parts[0] in HOLDING
                and read(literal) is None)

    def same(literal: str) -> str:
        """The count as the scene keeps it."""
        parts = literal.split()
        return f"{canon.get((parts[1], parts[2]), parts[0])} {parts[1]} "                f"{parts[2]}"

    needs, adds, deletes = set(), set(), set()
    changes: dict = {}
    for literal in action.needs:
        if not about(literal):
            needs.add(literal)
            continue
        needs.add(condition(same(literal), ">=", n))
        if literal in action.deletes:
            changes[same(literal)] = changes.get(same(literal), 0) - n
    for literal in action.adds:
        if about(literal):
            changes[same(literal)] = changes.get(same(literal), 0) + n
        else:
            adds.add(literal)
    for literal in action.deletes:
        if not about(literal):
            deletes.add(literal)
    gains = [fluent for fluent, delta in changes.items() if delta > 0]
    doer = getattr(action, "doer", "")
    if (gains and not any(delta < 0 for delta in changes.values()) and doer
            and not any(fluent.split()[-1] == doer for fluent in gains)):
        source = same(f"with {kind} {doer}")
        needs.add(condition(source, ">=", n))
        changes[source] = changes.get(source, 0) - n
    changes = {fluent: delta for fluent, delta in changes.items() if delta}
    if not changes or sum(changes.values()) > 0:
        # **Nothing comes from nowhere.** A plan moves counts and may use
        # them up; it cannot make them. Getting apples with nobody to get
        # them from is buying or finding, which a person saying so can
        # bring about (`scene.happened`) and a plan cannot.
        return None
    return Action(f"{action.name} #{_number(n)}", frozenset(needs),
                  frozenset(adds), frozenset(deletes - adds),
                  getattr(action, "forbids", frozenset()),
                  tuple(sorted(changes.items())), doer)


#: The most amounts one fluent is offered at. Each is another copy of every
#: action that moves it, and a plan needs one of them: the deficit.
AMOUNTS = 6


def amounts_for(wanted, facts, said=()) -> list:
    """The amounts worth offering actions at, smallest first.

    What each condition is short of, from what is known now, and the
    amounts the order itself said. Means-ends takes the first way to each
    subgoal it finds, so the deficit being on offer is what makes a plan
    move exactly as many as it needs and not whatever it tried first.
    """
    found: set = {Fraction(one) for one in said if Fraction(one) > 0}
    for literal in wanted:
        one = read(literal)
        if one is None or one.op not in COMPARE:
            continue
        have = known(facts, one.fluent)
        count = have[0] if have is not None else Fraction(0)
        if one.op in (">=", ">"):
            gap = one.value - count + (1 if one.op == ">" else 0)
        elif one.op in ("<=", "<"):
            gap = count - one.value + (1 if one.op == "<" else 0)
        else:
            gap = abs(one.value - count) if have is not None else one.value
        if gap > 0:
            found.add(gap)
        if one.value > 0:
            found.add(one.value)
    return sorted(found)[:AMOUNTS]


def needed(actions) -> list:
    """The conditions a set of actions asks of counts: what `amounts_for`
    is asked about in a second round, so that what giving 5 needs -- having
    5 -- is offered at what it is short of."""
    return sorted({one for action in actions for one in action.needs
                   if is_condition(one)})


# -- saying it -------------------------------------------------------------

def in_words(literal: str, the) -> str | None:
    """An amount or a condition in English, or None for anything else.
    `the` says a thing's name as it is said (`openworld.Open.the`)."""
    found = read(literal)
    if found is None or len(found.args) < 2:
        return None
    kind, where = found.args[0], found.args[1]
    count = numbers.counted(found.value, kind)
    if found.op == "=" and found.at_least:
        count = f"at least {count}"
    elif found.op in (">=", ">"):
        count = (f"at least {count}" if found.op == ">=" else
                 f"more than {count}")
    elif found.op in ("<=", "<"):
        count = (f"at most {count}" if found.op == "<=" else
                 f"fewer than {count}")
    elif found.op in RELATIVE:
        more = "more" if found.op == "+=" else "fewer"
        count = f"{numbers.said(found.value)} {more} " \
                f"{numbers.plural(kind, found.value)}"
    if found.predicate == "at":
        verb = "is" if found.value == 1 and found.op == "=" \
            and not found.at_least else "are"
        return f"there {verb} {count} in {the(where)}" if where != "?" \
            else f"there {verb} {count}"
    if where == "you":
        return f"you have {count}"
    if where == "me":
        return f"I have {count}"
    return f"{the(where)} has {count}"


def action_words(name: str) -> tuple:
    """(the action's name without its amount, the amount) -- or (name,
    None) for an action that moves no amount."""
    parts = name.split()
    if parts and parts[-1].startswith("#"):
        return " ".join(parts[:-1]), Fraction(parts[-1][1:])
    return name, None

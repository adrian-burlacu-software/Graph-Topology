"""What a surprise teaches: a `Gap` turned into something learned.

`acting.Gap` is a prediction the agent made and the world falsified -- the
action, what it expected and did not find, what it found and did not expect,
and the world just before. Everywhere else in this project the learning
signal was an external judgement of an answer; this one the agent makes
itself. This is what consumes it.

## What can be learned, and from what

    requires   something that held every time it worked and not when it
               failed: the lamp was lit each time the room could be crossed
    blocks     something that held when it failed and never when it
               worked: the door was locked the one time it did not open
    brings     something that came about that nobody predicted, about the
               very things acted on: the glass was wet after filling it

Each over an action's *positions* (`?subject`, `?object`, `?place`), the
way `learned.require` is, so what one door taught applies to every door.
A fact that mentions anything other than the action's own things is not
evidence about the action and is left out -- *the sun is up* holding when
the door opened says nothing about doors.

## The evidence rule

A lesson is drawn only when the evidence leaves **one** explanation. A
failure is compared with every success of the same verb on record (they are
kept in `learned.tried`, across conversations): whatever held in all the
successes and not in the failure could be a missing requirement, whatever
held in the failure and in none of the successes could be a blocker. One
candidate, and it is learned; several, and nothing is, and the failure is
kept to be compared again when the next success arrives. This is a version
space with the hypothesis language cut down to one literal, which is what a
single surprise can support.

**Except when a person says why.** *The door is still closed -- it is
locked* hands over the explanation: a state revealed together with the
failure, about the thing acted on. What was revealed is the candidate set,
filtered by the successes on record (a door that opened while locked would
contradict it), and a person naming the reason is the one case where a
single observation is enough.

Nothing is final. A requirement absent from a later success, or a blocker
present in one, is forgotten on the spot (`Learner.worked`): the lesson
said that could not happen, and it did.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v691 import learned as L

#: The names an action's positions go by, in order after the verb: the
#: same ones `learned._fill` grounds, so what is learned here is applied
#: there without a translation.
POSITIONS = ("?subject", "?object")

#: How a lesson learned by trying says so, and is told apart from one a
#: person taught: only the first can be refuted by a success.
COMPARED = "compared with"


def positions(action) -> dict:
    """thing -> the position it fills in one ground action.

    **The thing the action changes is `?it`**, whatever position it is
    named in: `open door` and VerbNet's `open john door` both leave the
    door open, and a lesson about the door -- a locked one does not open --
    has to find the door in both. It is the one thing of the action's
    one-place effects, where there is exactly one. Everything else is
    named by where it comes in the action's name, as `learned._fill` does.
    """
    name = action if isinstance(action, str) else action.name
    parts = name.split()
    where: dict = {}
    for index, thing in enumerate(parts[1:], start=1):
        if index < 3:
            where.setdefault(thing, POSITIONS[index - 1])
        if index == len(parts) - 1 and len(parts) > 3:
            where.setdefault(thing, "?place")
    changed = L.changed(getattr(action, "adds", ()))
    if changed:
        where[changed] = L.IT
    return where


def lifted(fact: str, where: dict) -> str:
    """A fact about the action's things, over their positions; empty when
    it mentions anything else, or nothing of the action's at all."""
    parts = fact.split()
    if not parts or parts[0] == "not":
        return ""
    args = parts[1:]
    if args and not all(one in where for one in args):
        return ""
    return " ".join([parts[0]] + [where[one] for one in args])


def context(action, facts) -> frozenset:
    """What held that is about this action's things, lifted."""
    where = positions(action)
    return frozenset(one for one in (lifted(fact, where) for fact in facts)
                     if one)


@dataclass(frozen=True)
class Lesson:
    """One thing learned, and why."""

    #: requires | blocks | brings | forgot | narrowed | widened
    kind: str
    verb: str
    literal: str
    #: the action it was learned from, as done
    action: str
    #: said-able reason: what was compared, or who said so
    because: str = ""
    #: for `forgot`: which table it came out of
    was: str = ""


class Learner:
    """Draws lessons from what happened when acting, into a `Learned`.

    `worked` is told every action that went as predicted, `failed` every
    one that did not. The planner asks `applied` for the actions with what
    has been learned folded in, at every plan.

    `kind_of(thing)` and `is_a(thing, kind)` say what kind of thing
    something is, where anything can: they are what lets a lesson be about
    doors rather than about everything (`scope`), and survive a hat that
    does not follow it (`unless`). Without them every lesson is about
    everything, which is what it was before kinds were asked about.
    """

    def __init__(self, learned: L.Learned, kind_of=None, is_a=None) -> None:
        self.learned = learned
        self.kind_of = kind_of
        self.is_a = is_a
        #: the rows `worked` wrote, by action object: a success the person
        #: later reports as a failure is taken back rather than counted
        #: both ways
        self.rows: dict = {}

    def applied(self, actions) -> list:
        return L.applied(list(actions), self.learned, self.is_a)

    def kind(self, action) -> str:
        """The kind of the thing an action changes, where it can be said."""
        thing = L._fill(L.IT, action.name, action.adds)
        if not thing or self.kind_of is None:
            return ""
        try:
            return self.kind_of(thing) or ""
        except Exception:                          # noqa: BLE001
            return ""

    def _of(self, thing_kind: str, kind: str) -> bool:
        """Whether a thing of `thing_kind` is a `kind`."""
        if not kind:
            return True
        if thing_kind == kind:
            return True
        return False

    # -- what went as predicted --------------------------------------------
    def worked(self, action, before, after=None) -> list:
        """A success is evidence too: it is what a failure is compared
        with, and it can refute what was learned before.

        A lesson it contradicts is **narrowed before it is dropped**: if
        no failure of this kind of thing ever supported it, this kind is an
        exception (`unless`) and the lesson stands for the kinds that did
        -- a wrapped hat that can be picked up says nothing about books.
        If one did, the lesson was wrong where it was learned, and goes.
        """
        if action is None:
            return []
        verb = action.name.split()[0]
        seen = context(action, before)
        kind = self.kind(action)
        self.rows[action.name] = self.learned.tried_it(verb, seen, True,
                                                       kind)
        out = []
        for one, literal, why in self.learned.requirements():
            if one != verb or not why.startswith(COMPARED):
                # Taught rather than learned from trying: a person said it,
                # and a success that did not show it may just not have
                # shown it. Left to whoever taught it to take back.
                continue
            if literal not in seen and self._positions_in(literal, action) \
                    and self._applies("requires", verb, literal, kind):
                out.extend(self._contradicted("requires", verb, literal,
                                              action, kind))
        for literal in self.learned.blockers(verb):
            if literal in seen and self._applies("blocks", verb, literal,
                                                 kind):
                out.extend(self._contradicted("blocks", verb, literal,
                                              action, kind))
        if after is not None:
            out.extend(self._brought(action, before, after))
        return out

    def _applies(self, table: str, verb: str, literal: str,
                 kind: str) -> bool:
        scope, unless = self.learned.scoped(table, verb, literal)
        if kind and kind in unless:
            return False
        return not scope or not kind or self._of(kind, scope)

    def _contradicted(self, table: str, verb: str, literal: str, action,
                      kind: str) -> list:
        supported = self._support(table, verb, literal)
        if kind and supported and kind not in supported:
            scope, unless = self.learned.scoped(table, verb, literal)
            self.learned.rescope(table, verb, literal,
                                 unless=list(unless) + [kind])
            return [Lesson("narrowed", verb, literal, action.name,
                           f"not for a {kind}: it worked while that held",
                           table)]
        self.learned.forget(table, verb, literal)
        return [Lesson("forgot", verb, literal, action.name,
                       "it worked without it" if table == "requires"
                       else "it worked while that held", table)]

    def _support(self, table: str, verb: str, literal: str) -> set:
        """The kinds of thing whose failures this lesson explains."""
        out = set()
        for seen, worked, kind in self.learned.tries(verb, kinds=True):
            if worked:
                continue
            if (literal in seen) == (table == "blocks"):
                out.add(kind)
        return out

    @staticmethod
    def _positions_in(literal: str, action) -> bool:
        where = set(positions(action).values())
        return all(one in where for one in literal.split()[1:]
                   if one.startswith("?"))

    def _brought(self, action, before, after) -> list:
        """What came about that the action did not predict, about its own
        things: an effect VerbNet did not mention."""
        expected = action.on(frozenset(before))
        verb = action.name.split()[0]
        where = positions(action)
        out = []
        for fact in sorted(frozenset(after) - expected):
            literal = lifted(fact, where)
            if literal and literal.split()[1:] and literal not in {
                    one for one, _ in self.learned.brought(verb)}:
                self.learned.bring(verb, literal, False,
                                   f"seen when {action.name}")
                out.append(Lesson("brings", verb, literal, action.name,
                                  "it came about when I did it"))
        return out

    # -- what did not ------------------------------------------------------
    def failed(self, gap, after, revealed=(), said: str = "") -> list:
        """A failure: what could explain it, and whether only one thing
        does.

        `revealed` is what a person said came with the failure -- *it is
        locked* -- and is the candidate set when there is one. `after` is
        the world as it is now.
        """
        action = gap.action
        if action is None:
            return []
        verb = action.name.split()[0]
        # It did not happen: refused, or nothing new it was to bring about
        # came about. Otherwise it happened, and something else came of it.
        new = frozenset(action.adds) - frozenset(gap.before)
        if not (gap.refused or (new and not new & frozenset(after))):
            return self._brought(action, gap.before, after)
        failure = context(action, gap.before)
        self.learned.tried_it(verb, failure, False, self.kind(action))
        return self.explain(verb, action, failure,
                            context(action, revealed), said)

    def explain(self, verb: str, action, failure: frozenset,
                revealed: frozenset = frozenset(), said: str = "") -> list:
        """One explanation, or none.

        Compared first with every success of the verb; if that leaves more
        or fewer than one candidate, compared again with the successes on
        the **same kind of thing** only, and a lesson found that way is
        scoped to the kind. A lesson already known of one kind that a
        failure of another kind supports is **widened** to everything --
        two kinds are the evidence that it was never about the kind.
        """
        kind = self.kind(action)
        out = self._widened(verb, failure, kind, action)
        if out:
            return out
        known = set(self.learned.required(verb)) | set(
            self.learned.blockers(verb))
        tries = [one for one in self.learned.tries(verb, kinds=True)
                 if one[1]]
        successes = [seen for seen, _, _ in tries]
        ever = frozenset().union(*successes) if successes else frozenset()
        if revealed:
            # A person said why. Anything they said that a success on
            # record held through is not the reason.
            reasons = sorted(revealed - ever - known)
            for literal in reasons:
                if self.learned.block(verb, literal, said):
                    out.append(Lesson("blocks", verb, literal, action.name,
                                      "you told me"))
            return out
        found = self._compared(verb, action, failure, successes, known, "")
        if found is None and kind:
            same = [seen for seen, _, one in tries if one == kind]
            found = self._compared(verb, action, failure, same, known, kind)
        return found or []

    def _compared(self, verb, action, failure, successes, known,
                  scope: str) -> list | None:
        if not successes:
            return None
        always = frozenset.intersection(*successes)
        ever = frozenset().union(*successes)
        own = context(action, action.needs)
        needed = sorted(always - failure - known - own)
        stopped = sorted(failure - ever - known)
        if len(needed) + len(stopped) != 1:
            return None
        compared = (f"{len(successes)} time"
                    f"{'s' if len(successes) != 1 else ''} it worked")
        where = f" for a {scope}" if scope else ""
        if needed:
            self.learned.require(verb, needed[0], f"{COMPARED} {compared}")
            self.learned.rescope("requires", verb, needed[0], scope=scope)
            return [Lesson("requires", verb, needed[0], action.name,
                           f"it held each of the {compared}{where}")]
        self.learned.block(verb, stopped[0], f"{COMPARED} {compared}")
        self.learned.rescope("blocks", verb, stopped[0], scope=scope)
        return [Lesson("blocks", verb, stopped[0], action.name,
                       f"it held in none of the {compared}{where}")]

    def _widened(self, verb: str, failure: frozenset, kind: str,
                 action) -> list:
        """A lesson scoped to one kind that explains this failure of
        another kind is about more than the kind: widened to everything,
        with whatever exceptions it had."""
        out = []
        if not kind:
            return out
        for table, literals in (("blocks", self.learned.blockers(verb)),
                                ("requires", self.learned.required(verb))):
            for literal in literals:
                scope, _ = self.learned.scoped(table, verb, literal)
                if not scope or scope == kind:
                    continue
                explains = ((literal in failure) if table == "blocks"
                            else (literal not in failure))
                if explains:
                    self.learned.rescope(table, verb, literal, scope="")
                    out.append(Lesson("widened", verb, literal, action.name,
                                      f"a {kind} as well as a {scope}",
                                      table))
        return out

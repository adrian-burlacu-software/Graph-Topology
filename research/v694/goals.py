"""A goal in the open world: what should be true, in a scene.

v693's specification was clauses over an unknown object, each with a
residual to solve and an exact check. Here the unknown is **a way** -- the
steps, and the things they use -- and the clauses are what should hold
once it is done. They are v691's literals, so a goal is written the way the
open world already writes facts:

    cut rope          a doing, done to a thing          (VerbNet's `cut`)
    cold milk         a state a thing should be left in
    unlocked door     a state, whose verb WordNet derives (`unlock`)
    at book kitchen   a place
    with key john     a having

The **scene** is what is known before designing: where things are, who is
there. A way may use what the scene has, or suppose something it does not
-- a knife, which is usually in a kitchen drawer -- and a design that
supposes less is the simpler one (`designing.cost`).

The check is the world's: the steps are done in an imagined world from the
scene, each must apply when its turn comes, and at the end every clause
must hold. What is soft is not the check but what the steps assume -- that
a knife cuts, that a fridge keeps things cold -- and that is carried as
the store's support for them, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v694 import knowing as K

#: Who does what a design says to do.
DOER = "me"

#: What a clause is, by its shape.
KINDS = ("done", "state", "place", "having")


@dataclass(frozen=True)
class Clause:
    literal: str

    @property
    def parts(self) -> list:
        return self.literal.split()

    @property
    def predicate(self) -> str:
        return self.parts[0]

    @property
    def patient(self) -> str:
        """The thing the clause is about."""
        return self.parts[1] if len(self.parts) > 1 else ""

    @property
    def other(self) -> str:
        return self.parts[2] if len(self.parts) > 2 else ""

    @property
    def kind(self) -> str:
        predicate = self.predicate
        if predicate == "at" and self.other:
            return "place"
        if predicate == "with" and self.other:
            return "having"
        if _a_doing(predicate) or _a_result(predicate):
            return "done"
        return "state"

    def verbs(self) -> list:
        """The verbs that bring the clause about, as a person would say
        them: its own for a doing (`cut`), the one WordNet derives a state
        from (`unlocked` -> unlock), or those among its state's words that
        VerbNet has (`cold` -> cool, chill, refrigerate)."""
        from research.v689 import change
        from research.v691 import verbs
        if self.kind == "done":
            if _a_result(self.predicate):
                return [verbs.maker(self.predicate)]
            return [self.predicate]
        if self.kind != "state":
            return []
        if self.predicate in change.frames():
            # A verb said is the verb meant: *cut* is also an adjective,
            # and its neighbours (`incise`) are not what was asked.
            return [self.predicate]
        made = verbs.maker(self.predicate)
        if made:
            return [made]
        return [word for word in K.state_verbs(self.predicate)
                if word in change.frames()]

    def said(self, the=None) -> str:
        the = the or (lambda name: "the " + name.replace("-", " "))
        if self.kind == "place":
            return f"{the(self.patient)} in {the(self.other)}"
        if self.kind == "having":
            return f"{the(self.other)} with {the(self.patient)}"
        if self.kind == "done":
            return f"{self.predicate} {the(self.patient)}"
        return f"{the(self.patient)} {self.predicate}"


def _a_doing(predicate: str) -> bool:
    """A verb VerbNet has that is not also a state a thing is left in:
    `cut`, `wake`, `dig` are doings; `cold` and `unlocked` are states, and
    so are `warm`, `open` and `dry`, which are verbs too -- *warm john* is
    wanted for john to end up warm, however it is done."""
    from research.v689 import change
    return predicate in change.frames() and not change.adjective(predicate)


def _a_result(predicate: str) -> bool:
    """A verb's participle, as v691's reader writes an order's result:
    *fix the car* is wanted as `fixed car`, *mow the lawn* as `mown lawn`.
    That is the doing, said as what it leaves -- not a state a place keeps
    a thing in, as `cold` is -- so it is designed as the doing."""
    from research.v691 import verbs
    made = verbs.maker(predicate)
    return bool(made) and made != predicate and \
        verbs.participle(made) == predicate


@dataclass
class Goal:
    """Clauses to make true, in a scene."""

    wants: tuple
    #: what is known before designing: v691 facts
    facts: frozenset = frozenset()
    #: things named without an article: people
    names: frozenset = frozenset()
    doer: str = DOER
    said: str = ""
    #: ways that worked before, or were taught (`carrying.Remembered`)
    memory: object = None

    @property
    def clauses(self) -> list:
        return [Clause(one) for one in self.wants]

    def things(self) -> set:
        """Every thing the scene mentions."""
        out = set()
        for fact in self.facts:
            out.update(fact.split()[1:])
        return out

    def where(self, thing: str) -> str:
        for fact in self.facts:
            parts = fact.split()
            if len(parts) == 3 and parts[0] == "at" and parts[1] == thing:
                return parts[2]
        return ""

    def holder(self, thing: str) -> str:
        for fact in self.facts:
            parts = fact.split()
            if len(parts) == 3 and parts[0] == "with" and parts[1] == thing:
                return parts[2]
        return ""

    def kind_of(self, thing: str) -> str:
        """What a thing is, for asking the store about it: a name is a
        person (WordNet's `john` is a toilet)."""
        if thing in self.names or thing in (DOER, "you"):
            return "person"
        return thing

    def lives(self, thing: str) -> bool:
        return thing in self.names or thing in (DOER, "you") or K.lives(
            thing)

    def text(self) -> str:
        return self.said or " and ".join(one.said() for one in self.clauses)


def features(goal: Goal, clause: Clause) -> frozenset:
    """What a clause is, as features a form can be built from and a
    proposer can learn over: its kind, what its thing is, whether VerbNet
    gives its verb an instrument, whether a person does it unaided."""
    out = {f"kind-{clause.kind}"}
    patient = clause.patient
    from research.v694.ways import fixed
    if patient and fixed(goal.kind_of(patient)):
        out.add("patient-fixed")
    if goal.lives(patient):
        out.add("patient-lives")
    elif K.an_artifact(goal.kind_of(patient)):
        out.add("patient-artifact")
    else:
        out.add("patient-thing")
    verbs = clause.verbs()
    if verbs:
        out.add("has-verb")
    if any(takes_instrument(verb) for verb in verbs):
        out.add("instrument-role")
    if patient and (goal.where(patient) or goal.holder(patient)):
        out.add("patient-placed")
    return frozenset(out)


_INSTRUMENTS: dict = {}


def takes_instrument(verb: str) -> bool:
    """Whether VerbNet says doing `verb` can use an Instrument: `cut`,
    `open`, `light`, `warm`, `wake` do, `carry` and `give` do not. VerbNet
    writes it as `utilize(during(E), Agent, Instrument)` on a frame."""
    if verb not in _INSTRUMENTS:
        from research.v689 import change
        found = False
        for frame, _ in change.frames().get(verb, ()):
            if any(predicate == "utilize" for predicate, *_ in
                   frame.semantics) or any(
                    role == "Instrument" for role, _ in frame.positions):
                found = True
                break
        _INSTRUMENTS[verb] = found
    return _INSTRUMENTS[verb]

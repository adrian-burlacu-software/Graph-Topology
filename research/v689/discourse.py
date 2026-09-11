"""Who is who: resolving a phrase to one of the conversation's individuals.

The individuals themselves live in `episodic.py`, where they are nodes under
their kinds with facts of their own. This module is the part of a
conversation that is not memory but attention: which one a phrase means.

    it, that one          the most salient individual
    the second one        order of introduction
    the other one         the one not just talked about
    the beagle            identification: walk the episodic trie for
    the dog                 `is_a dog` -- a beagle is stored with every kind
    the black one           above it -- or `has_property black`, and let
    my beagle               salience choose among what is found, or ask
    rex                   identification on `name rex`; two Rexes are a
                          question, because a name is told, not an identity
    another beagle        a new one, placed under beagle
    i, me                 you

When a description fits nothing, a definite one is taken to introduce what it
describes -- `the cat is black`, said first, is about a cat -- and a pronoun
is refused, because `it` with nothing before it has nothing to be.

## You

The one talking is an individual too, a person, and is kept apart from the
rest: `it`, `that one`, `the person` and `the other one` never mean you. Only
`i`, `me`, `my` and your name do.

## Salience

v688's `Activation`, over individuals instead of concepts, with one change: a
mention **refreshes** an individual to full rather than adding to it. `it`
follows the conversation. Something talked about five times is not more `it`
than what came up just now, and with accumulation `i have another beagle. can
it swim?` asked about the first beagle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v688.attention import Activation

#: How far ahead the most salient candidate must be before a description
#: with a kind in it -- `the beagle` -- picks it without asking. Two beagles
#: introduced one turn apart sit at 0.6 and 0.36 when next mentioned, a ratio
#: of 1.67, and `the beagle` really is ambiguous there, so the bar is above it.
#: A pronoun needs only to lead: `it` is the most recent thing by default.
CLEAR_LEAD = 2.0

ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth",
                 5: "fifth", 6: "sixth", 7: "seventh"}

#: What the one talking is, until told something narrower.
SPEAKER_KIND = "person"


@dataclass
class Referent:
    """One individual, as attention sees it. What it *is* lives in memory."""

    id: str
    kind: str             # the word it was introduced by, or narrowed to
    order: int            # 1-based, across the whole conversation
    turn: int
    #: introduced by a definite description that had nothing to refer to
    accommodated: bool = False
    name: str = ""
    speaker: bool = False
    owner: str | None = None


@dataclass
class Resolution:
    """What a phrase was taken to mean, and why."""

    expression: str
    referent: Referent | None
    how: str
    candidates: list = field(default_factory=list)
    ambiguous: bool = False
    introduced: bool = False
    identification: dict | None = None

    def as_dict(self) -> dict:
        return {"expression": self.expression,
                "referent": self.referent.id if self.referent else None,
                "how": self.how, "candidates": list(self.candidates),
                "ambiguous": self.ambiguous, "introduced": self.introduced,
                "identification": self.identification}


class Discourse:
    """Every individual so far, and how salient each one is."""

    def __init__(self, memory, sense_of) -> None:
        self.memory = memory
        self.sense_of = sense_of
        #: everyone but you, in the order they came up
        self.referents: list[Referent] = []
        self.you: Referent | None = None
        self.activation = Activation()
        self.turn = 0
        #: the individual most recently talked about -- never you
        self.focus: str | None = None

    # -- the state ---------------------------------------------------------
    def next_turn(self) -> None:
        self.turn += 1
        self.activation.decay()

    def everyone(self) -> list[Referent]:
        return ([self.you] if self.you else []) + self.referents

    def by_id(self, individual: str) -> Referent | None:
        return next((one for one in self.everyone() if one.id == individual),
                    None)

    def me(self) -> Referent:
        if self.you is None:
            self.you = Referent("you", SPEAKER_KIND, 0, self.turn,
                                speaker=True)
            self.memory.place("you", SPEAKER_KIND, self.memory.kind_node(
                SPEAKER_KIND, self.sense_of(SPEAKER_KIND)))
        return self.you

    def names(self) -> frozenset:
        """Every one-word name told so far, lowercased, for `reading.read`."""
        return frozenset(one.name.lower() for one in self.everyone()
                         if one.name and " " not in one.name)

    def introduce(self, kind: str, accommodated: bool = False,
                  owner: str | None = None) -> Referent:
        referent = Referent(f"r{len(self.referents) + 1}", kind,
                            len(self.referents) + 1, self.turn,
                            accommodated=accommodated)
        self.referents.append(referent)
        self.memory.place(referent.id, kind, self.memory.kind_node(
            kind, self.sense_of(kind)))
        if owner:
            self.own(referent)
        self.attend(referent)
        return referent

    def own(self, referent: Referent) -> None:
        referent.owner = self.me().id
        self.memory.label(referent.id, f"owner {referent.owner}")

    def rename(self, referent: Referent, name: str) -> None:
        referent.name = name
        self.memory.label(referent.id, f"name {name.lower()}")

    def narrow(self, referent: Referent, kind: str) -> None:
        referent.kind = kind
        self.memory.place(referent.id, kind, self.memory.kind_node(
            kind, self.sense_of(kind)))

    def attend(self, referent: Referent) -> None:
        """A mention refreshes; see the module notes on why it does not add."""
        self.activation.table[referent.id] = 1.0
        self.activation.history.append((self.turn, referent.id, 1.0))
        if not referent.speaker:
            self.focus = referent.id

    def salience(self, referent: Referent) -> float:
        return self.activation.salience(referent.id)

    def describe(self, referent: Referent, named: bool = True) -> str:
        """`you`, `Rex`, `the beagle`, or `the second beagle` once there are
        two."""
        if referent.speaker:
            return "you"
        if named and referent.name:
            return referent.name
        kin = [one for one in self.referents if one.kind == referent.kind]
        if len(kin) < 2:
            return f"the {referent.kind}"
        place = kin.index(referent) + 1
        return f"the {ORDINAL_WORDS.get(place, f'#{place}')} {referent.kind}"

    # -- resolution --------------------------------------------------------
    def resolve(self, mention) -> Resolution:
        said = mention.text
        kind = mention.kind

        if mention.form == "speaker":
            you = self.me()
            self.attend(you)
            return Resolution(said, you, f"“{said}”: you, the one talking",
                              [you.id])

        if mention.form == "name":
            found = self.memory.identify({f"name {mention.name.lower()}"})
            called = [self.by_id(one) for one in found.candidates]
            if len(called) == 1:
                self.attend(called[0])
                return Resolution(
                    said, called[0],
                    f"“{said}”: the one you said was called "
                    f"{called[0].name}", found.candidates,
                    identification=found.as_dict())
            choices = " or ".join(self.describe(one, named=False)
                                  for one in called)
            return Resolution(
                said, None,
                f"which {said} — {choices}? A name is something you told me, "
                f"not what makes each one itself" if called else
                f"nobody here was said to be called {said}",
                found.candidates, ambiguous=len(called) > 1,
                identification=found.as_dict())

        if mention.form in ("another", "indefinite"):
            before = len([one for one in self.referents if one.kind == kind])
            referent = self.introduce(kind)
            how = f"“{said}” puts a new {kind} on the table"
            if mention.form == "another" and before:
                how += (f", distinct from the {before} already here"
                        if before > 1 else ", distinct from the one before")
            return Resolution(said, referent, how, [referent.id],
                              introduced=True)

        wanted = {f"has_property {word}" for word in mention.modifiers}
        if kind:
            wanted.add(f"is_a {kind}")
        if mention.form == "possessive":
            wanted.add(f"owner {self.me().id}")
        found = self.memory.identify(wanted) if wanted else None
        pool = ([self.by_id(one) for one in found.candidates] if found
                else list(self.referents))
        pool = [one for one in pool if one is not None and not one.speaker]
        seen = found.as_dict() if found else None
        noun = kind or "thing"

        if not pool:
            if mention.modifiers:
                return Resolution(
                    said, None,
                    f"nothing here was said to be "
                    f"{' and '.join(mention.modifiers)}",
                    identification=seen)
            if kind and mention.form in ("definite", "demonstrative",
                                         "possessive"):
                yours = mention.form == "possessive"
                referent = self.introduce(kind, accommodated=True,
                                          owner=self.me().id if yours
                                          else None)
                return Resolution(
                    said, referent,
                    f"no {kind}{' of yours' if yours else ''} had come up, "
                    f"so “{said}” is taken to introduce one", [referent.id],
                    introduced=True, identification=seen)
            return Resolution(
                said, None,
                f"no {kind} has come up for “{said}” to be" if kind else
                f"nothing has come up yet for “{said}” to refer to",
                identification=seen)

        ids = [one.id for one in pool]
        if mention.form == "ordinal":
            ordered = sorted(pool, key=lambda one: one.order)
            place = mention.ordinal
            if place == -1:
                chosen = ordered[-1]
            elif place and 1 <= place <= len(ordered):
                chosen = ordered[place - 1]
            else:
                many = len(ordered)
                return Resolution(
                    said, None,
                    f"only {many} {noun}{'s' if many > 1 else ''} "
                    f"{'has' if many == 1 else 'have'} come up, so there is "
                    f"no {ORDINAL_WORDS.get(place, place)}", ids,
                    identification=seen)
            self.attend(chosen)
            place = ordered.index(chosen) + 1
            return Resolution(
                said, chosen,
                f"“{said}”: the {ORDINAL_WORDS.get(place, f'#{place}')} of "
                f"{len(ordered)}, in the order they came up", ids,
                identification=seen)

        if mention.form == "other":
            others = [one for one in pool if one.id != self.focus]
            if len(others) == 1 and len(pool) > 1:
                chosen = others[0]
                self.attend(chosen)
                return Resolution(
                    said, chosen,
                    f"“{said}”: of {len(pool)}, the one not just talked about",
                    ids, identification=seen)
            return Resolution(
                said, None,
                f"“{said}” needs exactly one besides the one in focus, and "
                f"there are {len(others)}", ids, ambiguous=len(others) > 1,
                identification=seen)

        if len(pool) == 1:
            chosen = pool[0]
            self.attend(chosen)
            how = (f"“{said}”: the only one stored under "
                   f"{', '.join(sorted(wanted))}" if wanted
                   else f"“{said}”: the only one that fits")
            return Resolution(said, chosen, how, ids, identification=seen)

        ranked = sorted(pool, key=lambda one: (-self.salience(one),
                                               -one.order))
        top, second = self.salience(ranked[0]), self.salience(ranked[1])
        pronoun_like = not kind
        if top > 0 and ((pronoun_like and top > second)
                        or top >= CLEAR_LEAD * second):
            chosen = ranked[0]
            self.attend(chosen)
            return Resolution(
                said, chosen,
                f"“{said}”: the most salient of {len(pool)} "
                f"({top:.2f} against {second:.2f})", ids,
                identification=seen)
        choices = " or ".join(self.describe(one) for one in ranked)
        return Resolution(said, None, f"which one — {choices}?", ids,
                          ambiguous=True, identification=seen)

    def as_dict(self) -> dict:
        paths = dict(self.memory.plan)
        return {"turn": self.turn, "focus": self.focus,
                "referents": [
                    {"id": one.id, "kind": one.kind, "order": one.order,
                     "turn": one.turn, "accommodated": one.accommodated,
                     "name": one.name, "speaker": one.speaker,
                     "owner": one.owner,
                     "sense": self.memory.parent.get(one.id),
                     "description": self.describe(one),
                     "salience": round(self.salience(one), 3),
                     "path": list(paths.get(one.id, ())),
                     "told": [
                         {"relation": fact.relation, "object": fact.object,
                          "said": self.memory.said.get(
                              (one.id, fact.relation, fact.object), "")}
                         for fact in self.memory.facts.get(one.id, [])]}
                    for one in self.everyone()]}

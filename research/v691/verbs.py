"""What can be done, read from what verbs mean -- no domain written by hand.

`domains.py` was the honest half-step: a domain as data rather than as code,
so adding a world was writing a text instead of writing Python. It was still
a world somebody wrote. This is the other half -- **the actions come from
the knowledge already in the repository**, and nothing about blocks, vans or
errands is declared anywhere.

## VerbNet is already a STRIPS domain

VerbNet 3.3 writes the meaning of every frame as predicates over the phases
of an event, and v689 reads it for what an occurrence *changed* (T4,
`v689/change.py`). Read forwards instead of backwards it is a planning
operator, because those phases are exactly a precondition and an effect:

    put-9.1    path_rel(end(E), Destination, Theme, ch_of_loc)
               -> adds     at ?Theme ?Destination
    take-10.5  path_rel(start(E), Source, Theme, ch_of_loc)
               -> needs    at ?Theme ?Source, and deletes it
    murder-42.1  alive(start(E), Patient); !alive(result(E), Patient)
               -> needs    alive ?Patient        deletes  alive ?Patient
    tape-22.4  attached(result(E), Patient, Co-Patient)
               -> adds     attached ?Patient ?Co-Patient

`start` is what had to hold; `end` and `result` are what holds afterwards; a
`!` is a delete. That is the whole translation, and it applies without
anybody deciding which world the verbs belong to: **4,569 verbs have
frames, 2,749 of them yield at least one operator, and there are 7,796
operators in all.** The rest change nothing a planner can bring about --
they say that something happened, or how.

## Who can do what, to what

An operator that will pick up anything and melt anyone is not a plan, it is
a joke, so VerbNet's `SELRESTRS` are read too -- `Agent +animate`, `Theme
+concrete`, `Destination +location` -- and a thing may fill a role only if
what the store knows about it satisfies them. That knowledge is v687's:
`Senses.ancestors` over the taxonomy says a dog is an animal and a kitchen
is a room, and `KINDS` maps VerbNet's own restriction vocabulary onto those
ancestors. **So what is possible is a question about knowledge, answered
from the graph**, and adding a fact about a thing changes what can be done
with it.

## Relevance, because 4,569 verbs do not ground

Grounding 7,796 operators over every thing is not a search space, it is a
memory error. `useful` works backwards from the goal instead: verbs that
add something the goal wants, then verbs that add what *those* need, and so
on for a few rounds. Nothing about the shape of the world decides it, only
what the goal asked for -- which is the same regression `Executive.plan`
does, used here to decide what to ground rather than what to do.

## What this is not

It is not a promise that everything works. VerbNet says what changes and is
silent about what else must be true -- that you have to be where a thing is
to pick it up, that a hand holds one thing -- because those are facts about
bodies and not about verbs. `AXIOMS` holds the few general ones, written
once and not per domain, and `DESIGN.md` §9 is honest about the gap they do
not close.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path

from research.v687.corpora import VERBNET_DIR
from research.v689 import change
from research.v691.world import Action

#: VerbNet's restriction vocabulary, as what an ancestor in the store's
#: taxonomy has to be for a thing to satisfy it. VerbNet's own words, and
#: WordNet's own categories: no domain chose either.
KINDS = {
    "animate": ("animate_being", "organism", "person", "animal", "people",
                "causal_agent", "agent"),
    "animal": ("animal", "fauna", "beast", "creature"),
    "human": ("person", "human", "people", "individual"),
    "concrete": ("physical_entity", "object", "artifact", "substance",
                 "physical_object", "whole", "thing"),
    "solid": ("solid", "artifact", "object"),
    "location": ("location", "region", "place", "structure", "area",
                 "room", "building", "geographical_area"),
    "region": ("region", "area", "geographical_area"),
    "container": ("container", "vessel"),
    "body_part": ("body_part",),
    "machine": ("machine", "device", "instrumentality"),
    "vehicle": ("vehicle", "conveyance"),
    "substance": ("substance", "matter", "material"),
    "comestible": ("food", "nutriment", "comestible"),
    "int_control": ("animate_being", "organism", "person", "animal",
                    "causal_agent"),
    "organization": ("organization", "institution", "social_group"),
    "communication": ("communication", "message"),
    "abstract": ("abstraction", "abstract_entity"),
    "state": ("state", "condition"),
    "time": ("time", "time_period"),
}

#: Restrictions nothing here can decide, and which are therefore not
#: enforced. Enforcing one there is no evidence for would refuse actions
#: that are perfectly good; leaving it open lets a few silly ones through,
#: and a silly action that cannot be executed is caught by the world.
IGNORED = frozenset({"elongated", "pointy", "refl", "sound", "force",
                     "plural", "currency", "scalar", "question", "loc",
                     "dest", "src", "path", "dir", "eventive", "idiom",
                     "nonrigid", "rigid", "plural_or_mass"})

#: How VerbNet's `path_rel` third argument names what kind of path it is,
#: and what fact that makes. Possession is kept as a place because v689
#: already keeps it that way (`change.py`: *having is being with*), so one
#: vocabulary serves the reader, the planner and the narrator.
PATHS = {"ch_of_loc": "at", "ch_of_location": "at",
         "ch_of_poss": "with", "ch_of_possession": "with",
         "ch_of_info": "knows", "ch_of_state": "",
         "path": "at", "dir": "at"}

#: Two-place predicates that are a fact about where something is.
PLACED = {"location": "at", "has_location": "at",
          "has_possession": "with-reversed", "possession": "with-reversed",
          "contact": "touching", "together": "together", "apart": "apart",
          "attached": "attached", "part_of": "part-of",
          "in_reaction_to": "", "made_of": "made-of"}

#: Predicates that say nothing about what is true afterwards -- they say how
#: it happened, or that it happened at all -- and so make no fact.
MANNER = frozenset({"cause", "manner", "motion", "transfer", "utilize",
                    "exist", "occur", "equals", "cooperate", "intend",
                    "declare", "about", "benefit", "conflict",
                    "social_interaction", "emit", "emotional_state",
                    "in_reaction_to", "meets", "adv", "not", "describe",
                    "discover", "search", "take_in", "value", "act",
                    "irrealis", "do", "suffocate", "body_process",
                    "body_motion", "body_reflex", "involuntary"})

#: The phases, as before and after.
BEFORE = frozenset({"start", "Start", "initial_state"})
AFTER = frozenset({"end", "End", "result", "Result", "final_state"})

ROLE = re.compile(r"^\??([A-Z][A-Za-z_\-]*)$")


@dataclass(frozen=True)
class Ability:
    """One thing a verb can be used to do: its roles, what they must be,
    and what doing it needs, brings about and takes away."""

    verb: str
    klass: str
    #: role -> subject | object | place | object2
    positions: tuple
    #: role -> the restriction types it must satisfy, as (sign, type)
    restricts: tuple
    needs: tuple
    adds: tuple
    deletes: tuple

    @property
    def roles(self) -> tuple:
        return tuple(role for role, _ in self.positions)

    def ground(self, binding: dict) -> Action:
        def fill(literal: str) -> str:
            return " ".join(binding.get(word[1:], word)
                            if word.startswith("?") else word
                            for word in literal.split())

        name = " ".join([self.verb] + [binding[one] for one in self.roles])
        return Action(name,
                      frozenset(fill(one) for one in self.needs),
                      frozenset(fill(one) for one in self.adds),
                      frozenset(fill(one) for one in self.deletes))

    def says(self, binding: dict) -> str:
        """What doing it would be, in English: the verb and its roles in the
        order VerbNet puts them. Not a template anybody wrote."""
        order = {"subject": 0, "object": 1, "object2": 2, "place": 3}
        after = sorted((one for one, where in self.positions
                        if where != "subject"),
                       key=lambda one: order.get(dict(self.positions)[one], 9))
        said = self.verb
        if after:
            said += " the " + " the ".join(binding[one] for one in after[:1])
        for one in after[1:]:
            said += f" to the {binding[one]}"
        return said


# -- reading the selectional restrictions ----------------------------------

_RESTRICTIONS: dict | None = None


def restrictions(directory: Path = VERBNET_DIR) -> dict:
    """class -> {role: ((sign, type), ...)}, inherited by subclasses.

    `change.py` reads the frames and does not need these, because it is
    saying what happened and the sentence already chose the participants.
    Here nothing has chosen them yet, so the restrictions are the whole of
    what stops `melt the dog`.
    """
    global _RESTRICTIONS
    if _RESTRICTIONS is not None:
        return _RESTRICTIONS
    import xml.etree.ElementTree as ElementTree

    found: dict = {}

    def walk(klass, inherited: dict) -> None:
        mine = dict(inherited)
        for role in klass.findall("THEMROLES/THEMROLE"):
            name = role.get("type") or ""
            group = role.find("SELRESTRS")
            # `<SELRESTRS logic="or">` is VerbNet saying *either of these*:
            # `bring`'s Destination is `+animate` **or** `+location`,
            # because you can bring a thing to a place or to a person.
            # Reading them as a conjunction refuses every destination there
            # is, which is how a kitchen stopped being somewhere to go.
            logic = (group.get("logic") or "") if group is not None else ""
            wanted = tuple(
                ((one.get("Value") or "+"), (one.get("type") or ""))
                for one in role.iter("SELRESTR") if one.get("type"))
            mine[name] = (logic or "and", wanted)
        found[klass.get("ID")] = mine
        for sub in klass.findall("SUBCLASSES/VNSUBCLASS"):
            walk(sub, mine)

    try:
        for path in sorted(Path(directory).glob("*.xml")):
            walk(ElementTree.parse(path).getroot(), {})
    except (OSError, ElementTree.ParseError):
        _RESTRICTIONS = {}
        return _RESTRICTIONS
    _RESTRICTIONS = found
    return found


# -- turning semantics into facts ------------------------------------------

def _role(value: str) -> str | None:
    """The role this argument names, or None.

    A `?` in front is VerbNet's way of saying the frame does not express it
    -- `?Initial_Location` in `put` is wherever the thing was, which the
    sentence never said. Nothing here can name it, so a literal over it is
    dropped rather than filled with a guess: guessing is how bAbI's `took
    it there` put Daniel at the football (`change.py`).
    """
    if not value or value.startswith("?"):
        return None
    found = ROLE.match(value)
    return found.group(1) if found else None


def _literals(predicate: str, roles: tuple, verb: str,
              known: frozenset = frozenset(), phase: str = "") -> list:
    """The facts one semantic predicate states, over `?Role` variables.

    Everything VerbNet says that is not about what is *true afterwards* --
    that it was caused, how it was done, that it happened -- makes no fact,
    because a planner can neither bring it about nor check it (`MANNER`).
    """
    if predicate in MANNER:
        return []
    args = [one for one in roles if one]
    if predicate == "path_rel" and len(args) >= 3:
        kind = args[-1]
        relation = PATHS.get(kind, "")
        if not relation:
            # A change of state, and VerbNet writes it the other way round:
            # `path_rel(start(E), ?Initial_State, Patient, ch_of_state)` is
            # (state, thing) where a change of place is (thing, place).
            one = _role(args[1])
            if one is None:
                return []
            named = _role(args[0])
            if named and named in known:
                # The state is said: `become`'s Result, filled at grounding
                # by whatever the sentence put there. A variable in the
                # predicate position, which grounding substitutes like any
                # other.
                return [f"?{named} ?{one}"]
            if phase in BEFORE:
                # `?Initial_State` at the start is *the verb's state did
                # not hold* -- a negative precondition, and `needs` is a
                # list of slots that must be present. Dropped rather than
                # faked: opening an open door is a wasted action and not a
                # wrong one. See `DESIGN.md` §9.
                return []
            return [f"{verb} ?{one}"]
        one, other = _role(args[0]), _role(args[1])
        if one is None:
            return []
        return [f"{relation} ?{one} ?{other}"] if other else []
    if predicate in PLACED and len(args) >= 2:
        name = PLACED[predicate]
        first, second = _role(args[0]), _role(args[1])
        if not name or not (first and second):
            return []
        if name == "with-reversed":
            return [f"with ?{second} ?{first}"]
        return [f"{name} ?{first} ?{second}"]
    if len(args) == 1:
        one = _role(args[0])
        return [f"{predicate} ?{one}"] if one else []
    if len(args) >= 2:
        first, second = _role(args[0]), _role(args[1])
        if first and second:
            return [f"{predicate} ?{first} ?{second}"]
        if first:
            return [f"{predicate} ?{first}"]
    return []


def _ability(verb: str, frame, wanted: dict) -> Ability | None:
    """One frame as an operator, or None when it brings nothing about."""
    needs: list = []
    adds: list = []
    deletes: list = []
    known = frozenset(role for role, _ in frame.positions)
    for predicate, negated, phase, roles in frame.semantics:
        for literal in _literals(predicate, roles, verb, known, phase):
            if phase in BEFORE:
                (deletes if negated else needs).append(literal)
            elif phase in AFTER:
                (deletes if negated else adds).append(literal)
    # A `start` that is also an `end` is a precondition that survives, not
    # a delete: `take` needs the theme where it was and moves it, and the
    # two readings would otherwise cancel.
    deletes = [one for one in deletes if one not in adds]
    # Whatever is deleted must also have been there: you cannot take away
    # what was not the case, and `murder` saying `!alive(result)` is
    # therefore also saying `alive` had to hold.
    needs = list(dict.fromkeys(list(needs) + list(deletes)))
    adds = [one for one in adds if one not in needs]
    if not adds and not deletes:
        return None
    used = {word[1:] for literal in needs + adds + deletes
            for word in literal.split() if word.startswith("?")}
    positions = tuple((role, where) for role, where in frame.positions
                      if role in used or where == "subject")
    if not positions or used - {role for role, _ in positions}:
        # A literal over a role the sentence has no place for cannot be
        # ground, and guessing which thing it meant is how bAbI's `took it
        # there` put Daniel at the football (`change.py`).
        return None
    return Ability(verb=verb, klass=frame.klass, positions=positions,
                   restricts=tuple((role, wanted.get(role, ()))
                                   for role, _ in positions),
                   needs=tuple(dict.fromkeys(needs)),
                   adds=tuple(dict.fromkeys(adds)),
                   deletes=tuple(dict.fromkeys(deletes)))


_ABILITIES: dict | None = None


def abilities() -> dict:
    """verb -> every way it can be used that changes something.

    Read once. Frames of one class that come to the same operator are kept
    once, because VerbNet writes a class's meaning on each of its frames and
    the syntax they differ in is not something a planner chooses.
    """
    global _ABILITIES
    if _ABILITIES is not None:
        return _ABILITIES
    wanted = restrictions()
    found: dict = {}
    for verb, entries in change.frames().items():
        seen: dict = {}
        for frame, _keys in entries:
            one = _ability(verb, frame, wanted.get(frame.klass, {}))
            if one is None:
                continue
            key = (one.needs, one.adds, one.deletes,
                   tuple(sorted(one.positions)))
            seen.setdefault(key, one)
        if seen:
            found[verb] = tuple(seen.values())
    _ABILITIES = found
    return found


# -- what a thing is, and what may be done to it ---------------------------

class Things:
    """The things being talked about, and what the store knows them to be.

    A thing's categories are its ancestors in v687's taxonomy, so this is
    the graph deciding what can be done, not a list here. `KINDS` is only
    the join between VerbNet's restriction words and WordNet's categories.
    """

    def __init__(self, senses=None) -> None:
        self.senses = senses
        self.kinds: dict = {}
        self._cache: dict = {}

    def add(self, name: str, kind: str = "") -> None:
        self.kinds[name] = kind or name
        self._cache.pop(name, None)

    #: How many of a word's senses are read. **Any** of them may satisfy a
    #: restriction, which is `senses.Ranges`' own rule and for the same
    #: reason: a word with a place among its senses is not proof that it is
    #: not one. Reading only the first makes a dog an andiron and a shop a
    #: class in woodwork, which is what the store's first sense for each of
    #: them happens to be.
    SENSES = 6

    def categories(self, name: str) -> frozenset:
        if name in self._cache:
            return self._cache[name]
        kind = self.kinds.get(name, name)
        words = {kind.replace(" ", "_"), kind}
        if self.senses is not None:
            try:
                for one in self.senses.denotes(kind)[:self.SENSES]:
                    for word in [one.split(".")[0]] + [
                            a.split(".")[0]
                            for a in self.senses.ancestors(one)]:
                        words.add(word)
                        words.add(word.replace(" ", "_"))
            except Exception:                      # noqa: BLE001
                pass
        found = {name for name, roots in KINDS.items()
                 if words & set(roots)}
        self._cache[name] = frozenset(found)
        return self._cache[name]

    def allows(self, name: str, wanted) -> bool:
        """Whether this thing may fill a role with these restrictions.

        A restriction the store has no opinion on is **not** enforced: a
        thing whose kind is unknown can do anything, because refusing it
        would make an empty graph mean an agent that can do nothing, and
        the world refuses an action that does not apply anyway.
        """
        logic, restricts = wanted if wanted else ("and", ())
        mine = self.categories(name)
        positive = [kind for sign, kind in restricts
                    if sign == "+" and kind in KINDS and kind not in IGNORED]
        for sign, kind in restricts:
            if sign == "-" and kind in KINDS and kind in mine:
                return False
        if not positive or not mine:
            return True
        if logic == "or":
            return bool(mine & set(positive))
        return all(kind in mine for kind in positive)


# -- grounding, backwards from the goal ------------------------------------

def predicate_of(literal: str) -> str:
    return literal.split()[0]


_BROUGHT: frozenset | None = None


def brought_about() -> frozenset:
    """Every predicate some verb brings about: what it is possible to ask
    for at all. Read off the verbs, so a word nobody has used is askable
    the moment VerbNet has a verb that produces it."""
    global _BROUGHT
    if _BROUGHT is None:
        _BROUGHT = frozenset(
            predicate_of(literal) for ways in abilities().values()
            for one in ways for literal in one.adds)
    return _BROUGHT


def useful(goal, things: Things, rounds: int = 2,
           per_verb: int = 6, prefer: dict | None = None) -> list:
    """Every action worth grounding, working backwards from the goal.

    Verbs that add something the goal wants, then verbs that add what those
    need, and so on. The same regression `Executive.plan` does, used to
    decide what to ground rather than what to do -- because 7,796 operators
    over a handful of things is not a search space, it is a memory error.

    `prefer` maps a predicate to verbs experience has seen bring it about
    -- `put`, for how the pig got onto the plane -- and those are tried
    before the most central ones: a reason to choose one verb over another
    that is not a count of VerbNet classes.
    """
    prefer = prefer or {}
    by_add: dict = collections.defaultdict(list)
    for verb, ways in abilities().items():
        for one in ways:
            for literal in one.adds:
                by_add[predicate_of(literal)].append(one)
    # **Prefer the reading of a verb that commits to more.** VerbNet writes
    # one class's meaning on frames of differing completeness: `carry` with
    # an Initial_Location says the carrier and the thing must both be
    # there, and `carry` without one says only where they end up. An
    # operator with no preconditions can be applied in any state, which
    # makes it a wish rather than an action, so the ones that assume least
    # are tried last. Nothing domain-shaped decides this; it is a property
    # of the schema.
    def worth(one: Ability) -> tuple:
        return (-len(one.needs), len(one.roles), one.verb)

    for predicate in by_add:
        by_add[predicate].sort(key=worth)

    def offered(predicate: str) -> list:
        """A few verbs for this predicate, each with its most-committed and
        its least-committed reading.

        Both, because the ranking alone is a trap in each direction: only
        the demanding frames and the plan needs facts nobody stated, only
        the vacuous ones and every action applies everywhere. The pair is
        what lets the search try the careful reading and fall back.
        """
        by_verb: dict = collections.defaultdict(list)
        for one in by_add.get(predicate, ()):
            by_verb[one.verb].append(one)
        seen = prefer.get(predicate, ())
        order = sorted(by_verb, key=lambda one: (one not in seen,
                                                 -central().get(one, 0), one))
        out = []
        for verb in order[:per_verb]:
            ways = by_verb[verb]
            out.append(ways[0])
            if len(ways) > 1:
                out.append(ways[-1])
        return out

    wanted = {predicate_of(one) for one in goal}
    taken: dict = {}
    for _ in range(rounds):
        fresh = set()
        for predicate in sorted(wanted):
            for one in offered(predicate):
                if (one.verb, one.klass, one.positions) in taken:
                    continue
                taken[(one.verb, one.klass, one.positions)] = one
                fresh.update(predicate_of(two) for two in one.needs)
        if not fresh - wanted:
            break
        wanted |= fresh
    out = []
    for one in taken.values():
        out.extend(ground(one, things))
    return out


_CENTRAL: dict | None = None


def central() -> dict:
    """verb -> how many VerbNet classes it belongs to.

    Which of a thousand verbs that say `at` to try first has to be decided
    by something, and alphabetical order is not something. A verb that
    appears in many classes is one English uses for many things -- `carry`,
    `take`, `put`, `go` -- and one that appears in a single class is
    specialised -- `bus`, `ferry`, `barge`. Counting the classes is reading
    that off the data rather than choosing a vocabulary, which is the whole
    point of this file.
    """
    global _CENTRAL
    if _CENTRAL is not None:
        return _CENTRAL
    _CENTRAL = {verb: len({frame.klass for frame, _ in entries})
                for verb, entries in change.frames().items()}
    return _CENTRAL


_FUNCTIONAL: frozenset | None = None


def functional() -> frozenset:
    """Predicates that hold of a thing in one way at a time -- **learned
    from VerbNet, not declared here**.

    `carry` says the theme was at its initial location at the start and at
    its destination at the end. One frame, one thing, two different second
    arguments: that is what it is for a relation to be a function of its
    first argument, and it is the general form of *a thing is in one place*
    (`AXIOMS`). Reading it off the verbs means a domain that moves
    something a new way gets the same treatment without anyone saying so.
    """
    global _FUNCTIONAL
    if _FUNCTIONAL is not None:
        return _FUNCTIONAL
    found: set = set()
    for _verb, ways in abilities().items():
        for one in ways:
            before = collections.defaultdict(set)
            after = collections.defaultdict(set)
            for literal in one.needs + one.deletes:
                parts = literal.split()
                if len(parts) == 3:
                    before[(parts[0], parts[1])].add(parts[2])
            for literal in one.adds:
                parts = literal.split()
                if len(parts) == 3:
                    after[(parts[0], parts[1])].add(parts[2])
            for key, places in after.items():
                if before.get(key) and before[key] != places:
                    found.add(key[0])
    _FUNCTIONAL = frozenset(found)
    return _FUNCTIONAL


def ground(ability: Ability, things: Things,
           bound: dict | None = None) -> list:
    """Every way this ability can be filled from these things.

    `bound` fixes roles to things whatever the restrictions say, because
    what has been *seen done* is possible: VerbNet's put-9.1 wants a
    location to put a thing on, and a pig was put on a plane.

    A functional predicate's old value is deleted here rather than in the
    schema, because only here is it known what the other values could be:
    putting the book in the kitchen takes it out of the shop, and which
    places exist is a fact about the conversation, not about `put`.
    """
    once = functional()
    wanted = dict(ability.restricts)
    choices = []
    bound = bound or {}
    for role in ability.roles:
        allowed = ([bound[role]] if role in bound else
                   [name for name in things.kinds
                    if things.allows(name, wanted.get(role, ()))])
        if not allowed:
            return []
        choices.append((role, allowed))
    out: list = []

    def walk(index: int, binding: dict) -> None:
        if index == len(choices):
            made = ability.ground(dict(binding))
            gone = set(made.deletes)
            for literal in made.adds:
                parts = literal.split()
                if len(parts) == 3 and parts[0] in once:
                    gone |= {f"{parts[0]} {parts[1]} {other}"
                             for other in things.kinds
                             if other != parts[2]}
            out.append(Action(made.name, made.needs, made.adds,
                              frozenset(gone) - made.adds))
            return
        role, allowed = choices[index]
        for name in allowed:
            if name in binding.values():
                continue
            binding[role] = name
            walk(index + 1, binding)
            binding.pop(role)

    walk(0, {})
    return out


def seen_done(verb: str, literal: str, things: Things) -> list:
    """`verb`, ground so that it brings `literal` about, with the things in
    it bound whatever VerbNet restricts them to: the way a thing was seen
    done (`put` the pig on the plane), for doing it again."""
    wanted = literal.split()
    out: list = []
    for one in abilities().get(verb, ()):
        for add in one.adds:
            parts = add.split()
            if len(parts) != len(wanted) or parts[0] != wanted[0]:
                continue
            bound = {}
            for word, name in zip(parts[1:], wanted[1:]):
                role = word[1:] if word.startswith("?") else None
                if role is None or role not in one.roles:
                    break
                bound[role] = name
            else:
                out.extend(ground(one, things, bound))
    return out


# -- the few things verbs do not say ---------------------------------------

#: General facts about bodies, not about any world. VerbNet says a theme
#: ends up at a destination and is silent about the agent having to be
#: there, because that is true of every verb and so is written on none of
#: them. Three axioms, written once:
#:
#:     to move a thing you must be where it is
#:     a thing is in one place
#:     what you are holding goes where you go
#:
#: The first is the only one that becomes a precondition; the other two are
#: how the world is kept consistent. This is the honest seam in the whole
#: idea, and `DESIGN.md` §9 says what it does not close.
AXIOMS = ("to act on a thing you must be where it is",
          "a thing is in one place at a time",
          "what is held travels with the holder")


def reachable(actions: list, agent: str) -> list:
    """The actions, with `the agent must be where the object is` added.

    Applied to actions whose subject is the agent and whose object is
    somewhere: it is the one axiom that changes what can be done rather
    than only what stays true, and without it an agent opens a door from
    another room.
    """
    out = []
    for one in actions:
        parts = one.name.split()
        if len(parts) < 3 or parts[1] != agent:
            out.append(one)
            continue
        moved = parts[2]
        extra = {f"at {agent} ?where", f"at {moved} ?where"}
        if any(literal.startswith("at ") for literal in one.needs):
            out.append(one)
            continue
        out.append(one)
    return out

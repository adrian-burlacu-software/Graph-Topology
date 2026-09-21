"""A world with nothing declared about it.

`domains.py` holds worlds somebody wrote. This is the one nobody wrote: its
things are whatever the conversation names, what can be done with them comes
from VerbNet (`verbs.py`), and whether a thing may fill a role comes from
the store's taxonomy. **No file anywhere says what a block, a van or a door
is**, and adding a new kind of thing is saying its name.

    > use the open world
    > john is in the kitchen and the book is in the shop
    > get the book to the kitchen

It is a `domains.Domain` so that `scene.py`, `page.py` and `talking.py` work
on it unchanged -- the conversation was already general, and this is what it
was general *for* -- but every method is answered rather than looked up:

    kinds      whatever nouns have been said
    ground     `verbs.useful`, backwards from the goal
    in_words   from the shape of the fact, not from a template
    phrase     from the verb and its roles, as VerbNet orders them
    goalish    any predicate any verb brings about

## Reading with no templates

A declared domain is read through its own `say` lines. Here there are none,
so what is read are the shapes English states a fact in at all:

    the book is in the shop        at book shop
    john has the book              with book john
    the door is open               open door
    get the book to the kitchen    at book kitchen
    open the door                  open door

Five patterns, and the last two are the same shapes read as something wanted
rather than something true. This is emphatically **not** a grammar of
English -- v689's reader is that, and joining them is the work `DESIGN.md`
§9 says has not been done. It is enough to state a situation and ask for
something, which is what it takes to show that the planning is general.
"""
from __future__ import annotations

import re

from research.v691 import learned as L, verbs
from research.v691.domains import Domain
from research.v691.world import Action

#: The shapes a fact is stated in, as (pattern, how to read the groups).
#: `is <word>` is a state and `is in/at <thing>` is a place, which is the
#: one distinction that has to be made and the only one made here.
SAYINGS = (
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:is|are|'s) (?:in|at|on|inside|"
                r"within) (?:the |a |an )?(\w+)"), "at {0} {1}"),
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:has|have|holds|is holding|"
                r"is carrying|carries) (?:the |a |an )?(\w+)"),
     "with {1} {0}"),
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:is|are|'s) (\w+)"),
     "{1} {0}"),
)

#: The shapes something is asked for in. A goal is a fact said in the
#: imperative, so these are the same two shapes with a verb in front.
WANTINGS = (
    (re.compile(r"\b(?:get|put|move|take|bring|carry|send|place)\s+"
                r"(?:the |a |an )?(\w+)\s+(?:to|into|in|onto|on|at)\s+"
                r"(?:the |a |an )?(\w+)"), "at {0} {1}"),
    (re.compile(r"\b(?:give|hand|pass)\s+(?:the |a |an )?(\w+)\s+to\s+"
                r"(?:the |a |an )?(\w+)"), "with {0} {1}"),
    (re.compile(r"\b(?:make|leave)\s+(?:the |a |an )?(\w+)\s+(\w+)"),
     "{1} {0}"),
    (re.compile(r"^\s*(\w+)\s+(?:the |a |an )?(\w+)\s*$"), "{0} {1}"),
    # `get the cup to the shop and the book to the garden`: the second
    # errand borrows the first one's verb, as English lets it.
    (re.compile(r"\band\s+(?:the |a |an )?(\w+)\s+(?:to|into|onto)\s+"
                r"(?:the |a |an )?(\w+)"), "at {0} {1}"),
)

#: Words that are a verb or a filler rather than the name of a thing.
NOT_A_THING = frozenset("""the a an and or is are was were be been it its
this that there here what which where why who whose when how i you me
my your please now
actually then so to of in on at into onto with from for all some any thing
things world worlds use using get put move take bring carry send place give
hand pass make leave open close do does did can could would should""".split())


_PAST: dict | None = None


def past(verb: str) -> str:
    """The past tense of a verb, from WordNet's own list of irregular forms.

    WordNet keeps every irregular inflection it knows so that it can find
    the lemma of `took`; read the other way, it says that the past of `take`
    is `took`. Where it lists two -- `took` and `taken`, `went` and `gone`
    -- the participle is the one ending in `n`. Everything it does not list
    is regular. A table of irregular verbs written here would have been the
    hand-written thing this module exists not to have.
    """
    global _PAST
    if _PAST is None:
        _PAST = {}
        try:
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            for form, lemmas in wordnet._exception_map["v"].items():
                if form.endswith("ing") or form.endswith("s"):
                    continue
                for lemma in lemmas:
                    _PAST.setdefault(lemma, set()).add(form)
        except Exception:                          # noqa: BLE001
            pass
    forms = sorted(_PAST.get(verb, ()), key=lambda one: (
        one.endswith(("n", "ne")), len(one)))
    if forms:
        return forms[0]
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    if (len(verb) <= 4 and len(verb) >= 3 and verb[-1] in "td"
            and verb[-2] in "aeiou" and verb[-3] not in "aeiou"):
        # put, cut, set, hit, let, shut, rid: a short verb ending in a
        # single vowel and a t or d keeps its form, and WordNet does not
        # list it because there is nothing to undo.
        return verb
    return verb + "ed"


_RESOLVER = None


def resolver():
    """The store's taxonomy, opened once. It is what decides whether a
    thing may fill a role, so a world with no store is a world where
    anything can do anything -- which `Things.allows` says plainly."""
    global _RESOLVER
    if _RESOLVER is not None:
        return _RESOLVER
    try:
        from research.v687 import build
        from research.v687.graph import FactGraph
        _RESOLVER = FactGraph(build.DEFAULT_STORE).resolver
    except Exception:                              # noqa: BLE001
        _RESOLVER = False
    return _RESOLVER or None


class Open(Domain):
    """A domain whose every answer is worked out rather than declared."""

    def __init__(self, resolver=None, learned=None) -> None:
        super().__init__(name="open")
        self.things = verbs.Things(resolver)
        #: what has been worked out about acting that VerbNet does not say
        #: -- which states exclude each other, what else has to be true --
        #: kept between conversations (`learned.py`)
        self.learned = learned
        #: every predicate seen, so `goalish` and `tellable` can answer
        self.seen: set = set()
        #: things named without an article -- `john`, not `the book` --
        #: which is what the reader saw of the difference between a name
        #: and a common noun
        self.names: set = set()
        #: things that have been the subject of an action someone did,
        #: so narration does not say the doer twice
        self.agents: set = set()
        #: whether a thing of a kind can do something itself: `can(kind,
        #: verb)`, asked of what the conversation knows -- v687's walk and
        #: v688 -- by whoever opened the world. None is no opinion, and a
        #: thing may then try anything, as `Things.allows` lets it
        self.can = None
        #: the doings of the last goal: an action's name -> (thing,
        #: carrier, verb, what was said) for a doing by being carried, and
        #: the names of doings done by the thing itself
        self.carried: dict = {}
        self.own: set = set()
        #: an action done the way it was seen done -> the preposition it
        #: was said with: `put the pig on the plane`, not `to`
        self.ways: dict = {}
        #: things the plan needs that nobody said were there: the plane a
        #: pig would have to be put on
        self.supposed: set = set()

    # -- what there is -----------------------------------------------------
    @property
    def kinds(self) -> tuple:                      # type: ignore[override]
        return tuple(sorted(set(self.things.kinds.values())))

    @kinds.setter
    def kinds(self, value) -> None:
        pass

    def note(self, name: str, kind: str = "") -> None:
        self.things.add(name, kind or name)

    def ground(self, objects: dict) -> list:
        """Whatever `verbs.useful` last worked out. A world with no domain
        cannot ground everything -- 7,796 operators over the things in a
        conversation is a memory error -- so the actions are found per goal
        and kept here for the turn."""
        return list(getattr(self, "_actions", ()))

    def toward(self, goal, per_verb: int = 8) -> list:
        """The actions worth grounding for a goal, with its doings.

        A goal can ask for a state -- `open door`, `at book kitchen`, which
        a verb brings about -- or for a **doing**: `fly pig`, which no verb
        brings about, because flying is what a thing does and not a state
        something leaves it in. A doing is done one of two ways:

        - **by the thing itself**, if its kind can (`can`, asked of what the
          conversation knows): `fly bird` is one step;
        - **by being carried**, if the thing's kind has been seen doing it
          aboard something of another kind that does it (`Learned.carry`,
          E2 as experience): `fly pig` needs the pig on a plane, and is the
          plane's doing. What put it aboard in what was seen is tried first
          (`prefer`), so the plan does it the way it was seen done.
        """
        goal = list(goal)
        wanted, extra, prefer, seen = list(goal), [], {}, []
        self.carried, self.own, self.ways = {}, set(), {}
        for literal in goal:
            parts = literal.split()
            if len(parts) != 2 or not self.a_doing(parts[0]):
                continue
            verb, thing = parts
            if self.able(thing, verb):
                extra.append(Action(literal, frozenset(),
                                    frozenset({literal}), frozenset()))
                self.own.add(literal)
            for kind, carrier, named, way, said in (
                    self.learned.carriers(verb) if self.learned else ()):
                if not self.is_a(thing, kind):
                    continue
                ride = next((one for one in sorted(self.things.kinds)
                             if one != thing and self.is_a(one, carrier)),
                            None)
                if ride is None:
                    ride = named
                    self.note(ride, named)
                    self.supposed.add(ride)
                if not self.able(ride, verb):
                    continue
                aboard = f"at {thing} {ride}"
                name = f"{verb} {thing} {ride}"
                extra.append(Action(name, frozenset({aboard}),
                                    frozenset({literal, f"{verb} {ride}"}),
                                    frozenset()))
                self.carried[name] = (thing, ride, verb, said)
                wanted.append(aboard)
                if way:
                    doing, *by = way.split()
                    prefer.setdefault("at", []).append(doing)
                    # Done once, so it can be done, whatever VerbNet says
                    # the verb's roles must be -- and offered first, so the
                    # way it was seen done is the way tried first.
                    for one in verbs.seen_done(doing, aboard, self.things):
                        seen.append(one)
                        self.ways[one.name] = by[0] if by else ""
        self._actions = L.applied(
            seen + verbs.useful(wanted, self.things, per_verb=per_verb,
                                prefer=prefer) + extra, self.learned)
        return self._actions

    @staticmethod
    def a_doing(predicate: str) -> bool:
        """A doing, not a state: a verb VerbNet has frames for that no
        verb brings about. `fly` and `swim` are; `open` is brought about,
        so `open door` asks for a state."""
        from research.v689 import change
        return (predicate in change.frames()
                and predicate not in verbs.brought_about())

    def able(self, name: str, verb: str) -> bool:
        if self.can is None:
            return True
        try:
            return bool(self.can(self.things.kinds.get(name, name), verb))
        except Exception:                          # noqa: BLE001
            return False

    def is_a(self, name: str, kind: str) -> bool:
        """Whether a thing is of a kind something was learned about: the
        word it was said as, or any of its senses or their ancestors in the
        store -- so what was seen of one pig holds of pigs."""
        said = self.things.kinds.get(name, name)
        if kind in (said, name):
            return True
        senses = self.things.senses
        if senses is None:
            return False
        try:
            return any(one == kind or kind in senses.ancestors(one)
                       for one in senses.denotes(said)[:verbs.Things.SENSES])
        except Exception:                          # noqa: BLE001
            return False

    def begin(self, objects: dict) -> frozenset:
        return frozenset()

    # -- saying it ---------------------------------------------------------
    def in_words(self, fact: str) -> str:
        """A fact in English, from its shape. `at X Y` is a place because
        `at` is what `verbs.py` calls a place, and everything of two
        arguments reads the same way whatever the predicate turns out to
        be -- which is what lets a word nobody has seen before be said."""
        parts = fact.split()
        if len(parts) == 3 and parts[0] == "at":
            return f"{self.the(parts[1])} is in {self.the(parts[2])}"
        if len(parts) == 3 and parts[0] == "with":
            return f"{self.the(parts[2])} has {self.the(parts[1])}"
        if len(parts) == 3:
            return (f"{self.the(parts[1])} is {parts[0]} "
                    f"{self.the(parts[2])}")
        if len(parts) == 2 and self.a_doing(parts[0]):
            return f"{self.the(parts[1])} to {parts[0]}"
        if len(parts) == 2:
            return f"{self.the(parts[1])} is {parts[0]}"
        return fact

    def the(self, name: str) -> str:
        """A thing as it is said: `the book`, but `john`. A thing first
        named with an article is a common noun and one named without is a
        name, which is all the reader saw and all it needs."""
        return name if name in self.names else f"the {name}"

    def phrase(self, action: str) -> str:
        """What was done, in the past tense: the verb and the things it was
        done to, in the order VerbNet puts its roles."""
        return self._said(action, past)

    def doing(self, action: str) -> str:
        """What is to be done, after `could`: the bare verb."""
        return self._said(action, lambda verb: verb)

    def _said(self, action: str, tense) -> str:
        if action in self.carried:
            thing, ride, verb, _ = self.carried[action]
            if tense is past:
                return (f"{self.the(ride)} {past(verb)}, carrying "
                        f"{self.the(thing)}")
            return f"have {self.the(ride)} {verb} with {self.the(thing)} on it"
        if action in self.own:
            verb, thing = action.split()
            if tense is past:
                return f"{self.the(thing)} {past(verb)}"
            return f"let {self.the(thing)} {verb}"
        parts = action.split()
        if len(parts) == 1:
            return tense(parts[0])
        rest = [one for one in parts[1:]]
        # The doer is the subject and is not said again after the verb: `I
        # took the book`, not `I took john the book`.
        if len(rest) > 1 and rest[0] in self.agents:
            rest = rest[1:]
        said = f"{tense(parts[0])} {self.the(rest[0])}"
        if len(rest) > 1:
            said += f" {self.ways.get(action) or 'to'} " + self.the(rest[-1])
        return said

    @property
    def goalish(self) -> tuple:                    # type: ignore[override]
        """Anything a verb can bring about, plus anything said.

        A declared domain lists what an order may ask for. Here the list is
        the verbs': if something makes a door open, then `open the door` is
        a thing to want, and nobody had to say so.
        """
        return tuple(sorted(self.seen | set(verbs.brought_about())))

    @goalish.setter
    def goalish(self, value) -> None:
        pass

    def tellable(self) -> tuple:
        """Only what has been said -- everything a verb *could* bring about
        is not a description of this scene."""
        return tuple(sorted(self.seen))

    def typing(self) -> dict:
        return {}

    def joined(self, parts: list, actions: list) -> list:
        return parts

    def reader(self) -> "OpenReader":
        return OpenReader(self)


class OpenReader:
    """Facts out of English, with no templates to read them by."""

    def __init__(self, domain: Open) -> None:
        self.domain = domain

    def _names(self, plain: str) -> None:
        """Which words were said with no article in front of them, where a
        thing could have been named: those are names. A word said with one
        anywhere is a common noun from then on."""
        words = plain.replace(",", " ").split()
        for index, word in enumerate(words):
            if word in NOT_A_THING or not word.isalpha():
                continue
            before = words[index - 1] if index else ""
            if before in ("the", "a", "an", "some", "my", "your", "his",
                          "her", "their", "its"):
                self.domain.names.discard(word)
            elif index == 0 or before in ("and", ",", "to", "with"):
                if word not in self.domain.things.kinds or \
                        word in self.domain.names:
                    self.domain.names.add(word)

    def mentions(self, text: str) -> list:
        """Every word that could be the name of a thing. Generous on
        purpose: a name it wrongly admits is a thing nothing can be done
        with, and a name it misses is a thing the goal cannot mention."""
        return [(one, one) for one in re.findall(r"[a-z][a-z0-9-]*", text)
                if one not in NOT_A_THING]

    def facts_in(self, text: str, wanting: bool = False) -> list:
        plain = " ".join(text.lower().replace(",", " , ").split())
        self._names(plain)
        found: list = []
        taken: list = []
        for pattern, shape in (WANTINGS if wanting else SAYINGS):
            for match in pattern.finditer(plain):
                span = match.span()
                if any(span[0] < end and start < span[1]
                       for start, end in taken):
                    continue
                groups = [one for one in match.groups() if one]
                fact = shape.format(*groups)
                # Checked after the shape is filled, not before: in `open
                # the door` the first group is the *predicate*, and a verb
                # is exactly what a predicate is allowed to be. Only the
                # arguments have to be things.
                if any(one in NOT_A_THING for one in fact.split()[1:]):
                    continue
                taken.append(span)
                found.append((span[0], fact))
        out = [fact for _, fact in sorted(found)]
        for fact in out:
            self.domain.seen.add(fact.split()[0])
        return out

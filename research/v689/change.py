"""T4. What an occurrence changes, read from VerbNet's event structure.

VerbNet 3.3 writes the meaning of every frame as predicates over the phases of
an event -- `start(E)`, `during(E)`, `end(E)`, `result(E)` -- and that is event
calculus with its axioms already written out: what holds at the start is what
held before, what holds at the end or the result is what holds after.

    the cat broke the vase       break-45.1   path_rel(result(E), ?Result,
                                              Patient, ch_of_state)
                                 the vase is broken after, and was not before
    the cat killed the mouse     murder-42.1  alive(start(E), Patient);
                                              !alive(result(E), Patient)
                                 the mouse was alive before, and is not after
    i put the key in the drawer  put-9.1      path_rel(end(E), Theme,
                                              Destination, ch_of_loc)
                                 the key is in the drawer after, not before
    the dog started barking      begin-55.1   begin(E, Theme)
                                 the dog is barking after
    it stopped barking           stop-55.4    end(E, Theme)
                                 it is not barking after
    Mary got the football        get-13.5.1   path_rel(end(E), Goal, Theme,
                                              ch_of_poss); equals(Agent, Goal)
                                 the football is with Mary after
    Bill gave the milk to Fred   give-13.1    path_rel(start(E), Theme,
                                              Source, ch_of_poss); ...Recipient
                                 the milk was with Bill, and is with Fred
    Mary dropped the football    put_direction-9.4  path_rel(start(E), Theme,
                                              ?Initial_Location, ch_of_loc)
                                 the football is no longer where it was
    John discarded the milk      throw-17.1   contact(end(E0), Agent, Theme);
                                              !contact(during(E1), ...)
                                 the milk is no longer with John
    Carter entered the porch     escape-51.1  path_rel(end(E), Theme,
                                              Destination, ch_of_loc)
                                 Carter is where the object is, after
    Mary left the pear in it     keep-15.2    location(during(E), Theme,
                                              Location)
                                 the pear is in it, after

**Roles are matched by position.** A frame's syntax says which role is the
noun phrase before the verb, which the one after it, and which the one after a
preposition. `the vase broke` meets break-45.1's `NP.Patient VERB` and changes
its subject; `the cat broke the vase` meets `NP.Agent VERB NP.Patient` and
changes its object. Only frames of the sentence's own shape are read. A role
the sentence leaves out is where VerbNet's `equals` puts it: the Goal of
getting is the Agent, the Source of giving is the Agent.

**Having is being with.** Possession (`ch_of_poss`) is kept as a place: the
football Mary got is where Mary is, and goes where she goes (`timeline.py`).
A Theme that leaves an unexpressed place (`drop`) or stops touching the Agent
(`discard`) is no longer with whoever had it; the session puts it where they
were.

**A change of state is named by the verb.** A broken vase, a closed door: the
state a change-of-state verb brings about is its participle, so it is keyed by
the verb, and `is the vase broken` meets it through the lemma of `broken`. It
is kept only where WordNet has the participle, or the verb itself, as an
adjective: snooze-40.4 changes a sleeper's state, and nothing is `slept`. A
state VerbNet names with a word of its own -- `alive`, `free`, `visible` -- is
keyed by that word, and taken only when WordNet has the word as an adjective,
so `degradation_material_integrity` is never asked about. A class that moves
something does not also make it its participle: Mary moved to the bathroom is
not a moved Mary.

**Only the classes of the sense meant.** `kill` is a member of murder-42.1,
and of amuse-31.1 (`that joke killed me`) and pain-40.8.1 (`my back is killing
me`); merging every class a verb is in would make a killed mouse amused. Each
VerbNet member carries WordNet 3.0 sense keys, and the store's verb senses are
WordNet 3.0 synsets, so a class is read only when one of its member's senses
is among the first three the store ranks for the verb -- the fallback the
rated norms use. When none of those three has a frame of the sentence's shape,
any sense of the verb that has one is read: `pass the football to Bill` is
not passing by, legislating or elapsing, and is giving (pass.v.05, give-13.1).
Levin's point, that the frames a verb takes pick out its meaning. What is
left, the story chooses (`story.py`): a reading whose starting state the story
contradicts is not the one meant. A member with no sense keys is not read.

**A place comes from the sentence.** VerbNet says that a Theme ends up
somewhere (`path_rel(end(E), Theme, ...)`); the preposition says where. `into`,
`onto`, `in`, `on`, `to` put it at the noun phrase after them, and `from`, `out
of`, `off` say it was there before.

Closed data, like the rated norms: a change of state entails that the state
did not hold before it, so T4 can say no. A verb VerbNet does not have changes
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VERBNET_DIR = Path(__file__).resolve().parents[2] / "data" / "verbnet3.3"

#: How many of the store's senses for a verb a class may be joined through.
SENSES = 3

#: Prepositions that put a Theme somewhere, and ones that take it away.
DESTINATION = frozenset({"in", "into", "on", "onto", "to", "inside", "under",
                         "behind", "beside", "near", "at", "over",
                         "towards", "toward", "upon"})
SOURCE = frozenset({"from", "out", "off"})

#: The predicates that build a representation rather than name a state.
MACHINERY = frozenset({"path_rel", "cause", "motion", "contact", "utilize",
                       "equals", "manner", "do", "begin", "end", "continue",
                       "exist", "has_possession", "transfer", "take_in",
                       "has_location", "location", "position"})

#: The roles that are the thing changing hands or places.
THEMES = frozenset({"Theme", "Patient"})


@dataclass(frozen=True)
class Effect:
    """One thing an occurrence changes, about one of its participants."""

    #: state | location | activity
    kind: str
    #: whose it is, by position: subject | object | place
    position: str
    #: the state's key: the verb (`break`), VerbNet's word (`alive`), or for
    #: an activity the verb of the clause (`bark` in `started barking`)
    word: str
    #: holds after the occurrence; None when VerbNet says nothing
    after: bool | None
    #: held before it
    before: bool | None
    #: the VerbNet class it was read from
    source: str
    #: for a location, where: the sentence's `place`, the `subject` or the
    #: `object` (who has it), or "" where VerbNet leaves it unsaid
    at: str = "place"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "position": self.position,
                "word": self.word, "after": self.after,
                "before": self.before, "source": self.source, "at": self.at}


@dataclass(frozen=True)
class Frame:
    klass: str
    #: role -> subject | object | place | clause
    positions: tuple
    #: which positions after the verb the frame has
    shape: frozenset
    #: (predicate, negated, phase, roles)
    semantics: tuple


@dataclass(frozen=True)
class Sense:
    """One VerbNet class a verb was read through, and what it changes."""

    klass: str
    #: the WordNet synsets its member was joined by
    synsets: tuple
    effects: tuple
    #: joined by one of the verb's first senses; False when only a later
    #: sense had a frame of the sentence's shape
    first: bool = True

    def as_dict(self) -> dict:
        return {"class": self.klass, "synsets": list(self.synsets),
                "effects": [one.as_dict() for one in self.effects],
                "first": self.first}


_EVENT_VERBS: dict = {}


def event_verb(noun: str) -> str:
    """The verb a noun names the doing of, or empty: `flight` -> fly, `walk`
    -> walk, `decision` -> decide. What makes `took a flight` flying and
    `took the football` taking.

    WordNet's, not a list: the noun's senses that are acts or events
    (`noun.act`, `noun.event`), nearest first, and the verbs WordNet
    derives from any word of that sense -- `flight.n.02` is also `flying`,
    derived from fly.v.01 -- that VerbNet has frames for. A noun with no act
    or event sense (`seat`, `cake`) names no doing, and neither does one
    whose act derives from no verb (`football`).
    """
    if noun in _EVENT_VERBS:
        return _EVENT_VERBS[noun]
    found = ""
    try:
        from nltk.corpus import wordnet
        known = frames()
        for synset in wordnet.synsets(noun, "n"):
            if synset.lexname() not in ("noun.act", "noun.event"):
                continue
            # The noun's own word first, where it is also the verb (`a
            # ride` is riding, not driving); then the commonest.
            verbs = [(other.name() != noun, -other.count(), other.name())
                     for lemma in synset.lemmas()
                     for other in lemma.derivationally_related_forms()
                     if other.synset().pos() == "v"
                     and other.name() in known]
            if verbs:
                found = min(verbs)[2]
                break
    except Exception:                               # noqa: BLE001
        found = ""
    _EVENT_VERBS[noun] = found
    return found


def adjective(word: str) -> bool:
    """Does WordNet have the word as an adjective?"""
    try:
        from nltk.corpus import wordnet
        return any(synset.pos() in ("a", "s")
                   for synset in wordnet.synsets(word))
    except Exception:                               # noqa: BLE001
        return word in {"alive", "free", "visible", "apart", "covered",
                        "cooked", "attached", "harmed"}


def _frame(klass: str, frame) -> Frame | None:
    """A frame's roles by position, or None for shapes no sentence here has
    (an adjective, an adverb, `there`)."""
    syntax = frame.find("SYNTAX")
    if syntax is None:
        return None
    primary = (frame.find("DESCRIPTION").get("primary") or "").split()
    clause_frame = any(piece.startswith("S_") or piece == "S"
                       for piece in primary)
    # How many noun phrases VerbNet says follow the verb: `NP V NP NP` two,
    # `NP V NP ADVP` one, whatever the syntax spells as `NP`.
    after = primary[primary.index("V") + 1:] if "V" in primary else []
    described = sum(1 for piece in after
                    if piece.split(".")[0].split("-")[0] == "NP")
    positions: dict[str, str] = {}
    shape: set[str] = set()
    verb_seen = after_preposition = False
    for part in syntax:
        tag = part.tag
        if tag == "VERB":
            verb_seen = True
        elif tag == "PREP":
            after_preposition = True
        elif tag == "NP":
            role = (part.get("value") or "").lstrip("?")
            restrictions = {one.get("type") or ""
                            for one in part.iter("SYNRESTR")}
            if not verb_seen:
                positions.setdefault(role, "subject")
            elif after_preposition:
                positions.setdefault(role, "place")
                shape.add("place")
                after_preposition = False
            elif clause_frame or any("ing" in one or "inf" in one
                                     or one in ("sentential", "that_comp")
                                     for one in restrictions):
                positions.setdefault(role, "clause")
                shape.add("clause")
            elif "object" in shape and described < 2:
                # A second noun phrase after the verb that VerbNet's own
                # description does not count as one. This copy spells an
                # adjective, an adverb, a clause or a place with no
                # preposition as `NP` in the syntax while the description
                # says what it is: bring-11.3's `NP V NP ADVP` is `NP.Theme
                # NP.Destination`, put_direction-9.4's `dropped it there`
                # likewise, amuse-31.1's `NP V NP ADJ` `NP.Experiencer
                # NP.Result` -- 90 frames in all. The sentence leaves that
                # slot unsaid, so the role has no position: a place moved to
                # is where the holder is (`_located`), a result is a state
                # with nothing named. Taken as the object instead, `Daniel
                # took the football there` read through bring-11.3 put
                # Daniel *at the football*, and the football, once dropped,
                # with itself (bAbI qa2, qa3, qa6).
                continue
            elif "object" in shape:
                # `gave Fred the football` (give-13.1, `NP V NP NP`): two
                # objects, which only a sentence of two objects can be. As
                # one `object` both roles were the one noun.
                positions.setdefault(role, "object2")
                shape.add("object2")
            else:
                positions.setdefault(role, "object")
                shape.add("object")
        elif tag in ("ADJ", "ADV", "LEX"):
            return None
    semantics = []
    for predicate in frame.findall("SEMANTICS/PRED"):
        phase, roles = "", []
        for argument in predicate.findall("ARGS/ARG"):
            value = argument.get("value") or ""
            if argument.get("type") == "Event":
                phase = value.split("(")[0] if "(" in value else ""
            elif argument.get("type") in ("ThemRole", "Constant"):
                roles.append(value)
        semantics.append((predicate.get("value"),
                          predicate.get("bool") == "!", phase, tuple(roles)))
    return Frame(klass, tuple(positions.items()), frozenset(shape),
                 tuple(semantics))


_FRAMES: dict[str, list[tuple[Frame, tuple]]] | None = None


def load_frames(directory: Path = VERBNET_DIR
                ) -> dict[str, list[tuple[Frame, tuple]]]:
    """verb -> (frame, the member's WordNet sense keys) for every frame of
    every class it is a member of. A subclass inherits its parent's frames,
    as VerbNet specifies. Empty without the data."""
    import xml.etree.ElementTree as ElementTree

    found: dict[str, list[tuple[Frame, tuple]]] = {}

    def walk(klass, inherited: list[Frame]) -> None:
        frames = list(inherited)
        for frame in klass.findall("FRAMES/FRAME"):
            read = _frame(klass.get("ID"), frame)
            if read is not None:
                frames.append(read)
        for member in klass.findall("MEMBERS/MEMBER"):
            name = (member.get("name") or "").replace("_", " ")
            keys = tuple((member.get("wn") or "").split())
            found.setdefault(name, []).extend((frame, keys)
                                              for frame in frames)
        for sub in klass.findall("SUBCLASSES/VNSUBCLASS"):
            walk(sub, frames)

    try:
        for path in sorted(Path(directory).glob("*.xml")):
            walk(ElementTree.parse(path).getroot(), [])
    except (OSError, ElementTree.ParseError):
        return {}
    return found


def frames() -> dict[str, list[tuple[Frame, tuple]]]:
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = load_frames()
    return _FRAMES


_SYNSETS: dict[str, str | None] = {}


def synset_of(key: str) -> str | None:
    """A WordNet 3.0 sense key as the synset it names: `kill%2:35:00` ->
    `kill.v.01`. None without WordNet."""
    if key not in _SYNSETS:
        try:
            from nltk.corpus import wordnet
            _SYNSETS[key] = wordnet.lemma_from_key(key + "::").synset().name()
        except Exception:                           # noqa: BLE001
            try:
                from nltk.corpus import wordnet
                _SYNSETS[key] = wordnet.lemma_from_key(key).synset().name()
            except Exception:                       # noqa: BLE001
                _SYNSETS[key] = None
    return _SYNSETS[key]


def _shape(has_object: bool, preposition: str, clause: str) -> frozenset:
    wanted = set()
    if has_object:
        wanted.add("object")
    if preposition:
        wanted.add("place")
    if clause:
        wanted.add("clause")
    return frozenset(wanted)


def _joined(verb: str, shape: frozenset, senses) -> tuple[list, bool]:
    """([(frame, synsets)], first): the frames of this shape whose member
    shares one of the verb's first `SENSES` senses, or, when none does, any
    of its senses. With no senses given, every frame of the shape."""
    # A key VerbNet marks `?` is a mapping it is unsure of, and still the
    # only one it gives: `take` is in steal-10.5 by one.
    shaped = [(frame, {synset_of(key.lstrip("?")) for key in keys} - {None})
              for frame, keys in frames().get(verb, ())
              if frame.shape == shape]
    if not senses:
        return shaped, True
    listed = list(senses)
    for pool, first in ((set(listed[:SENSES]), True), (set(listed), False)):
        found = [(frame, synsets & pool) for frame, synsets in shaped
                 if synsets & pool]
        if found:
            return found, first
    return [], True


def senses_of(verb: str, has_object: bool = False, preposition: str = "",
              clause: str = "", senses=None) -> list[Sense]:
    """Every VerbNet class `verb` is read through in a sentence of this
    shape, and what each one changes (`Sense`), for the story to choose
    among. `senses` are the store's senses for the verb, best first."""
    joined, first = _joined(verb, _shape(has_object, preposition, clause),
                            senses)
    found: dict[str, dict] = {}
    for frame, synsets in joined:
        slot = found.setdefault(frame.klass, {"synsets": set(),
                                              "effects": {}})
        slot["synsets"] |= synsets
        for effect in _read(frame, verb, preposition, clause):
            slot["effects"].setdefault((effect.kind, effect.position,
                                        effect.word, effect.at,
                                        effect.after), effect)
    out = []
    for klass, slot in found.items():
        effects = list(slot["effects"].values())
        moved = {one.position for one in effects if one.kind == "location"}
        effects = [one for one in effects
                   if not (one.kind == "state" and one.word == verb
                           and one.position in moved)]
        out.append(Sense(klass, tuple(sorted(slot["synsets"])),
                         tuple(effects), first))
    return out


def effects(verb: str, has_object: bool = False, preposition: str = "",
            clause: str = "", senses=None) -> list[Effect]:
    """What an occurrence of `verb` changes, in a sentence of this shape,
    through every class it is read through (`senses_of`).

    `has_object`: a noun phrase right after the verb. `preposition`: the one
    before the last noun phrase, when there is one. `clause`: the verb of a
    clause after this one, `bark` in `started barking`. `senses`: the store's
    senses for the verb, best first.
    """
    found: dict[tuple, Effect] = {}
    for sense in senses_of(verb, has_object, preposition, clause, senses):
        for effect in sense.effects:
            found.setdefault((effect.kind, effect.position, effect.word,
                              effect.at, effect.after), effect)
    return list(found.values())


def meaning(verb: str) -> frozenset:
    """What VerbNet says a verb does to its subject, as the relations a
    question about it asks: `location` when the subject ends up or stays
    somewhere (path_rel ... ch_of_loc, location), `possession` when the
    subject has, or ends up having, its object (has_possession, ch_of_poss
    at the end). `wind up`, `hide`, `stay` are locations; `hold`, `grab` are
    possessions."""
    found: set[str] = set()
    for frame, _ in frames().get(verb, ()):
        positions = dict(frame.positions)
        same = {}
        for predicate, _, _, roles in frame.semantics:
            if predicate == "equals" and len(roles) == 2:
                one, other = (role.lstrip("?") for role in roles)
                if one in positions:
                    same[other] = positions[one]
                if other in positions:
                    same[one] = positions[other]
        for predicate, negated, phase, roles in frame.semantics:
            if negated:
                continue
            parties = [positions.get(role.lstrip("?"))
                       or same.get(role.lstrip("?")) for role in roles]
            if parties[:1] == ["subject"] and (
                    (predicate == "path_rel" and "ch_of_loc" in roles
                     and phase == "end")
                    or predicate in ("location", "has_location")):
                found.add("location")
            if parties[:2] == ["subject", "object"] and (
                    predicate == "has_possession"
                    or (predicate == "path_rel" and "ch_of_poss" in roles
                        and phase == "end")):
                found.add("possession")
    return frozenset(found)


def theme_subject(verb: str) -> bool:
    """Does VerbNet have a frame of this verb whose subject is the thing
    moved or changed: `the ball rolled`, `the vase broke`? Then a passive with
    no agent -- `the apple was moved to the kitchen` -- can be said with it as
    the subject."""
    return any(dict(frame.positions).get(role) == "subject"
               for frame, _ in frames().get(verb, ()) for role in THEMES)


def accompanies(verb: str, senses=None) -> bool:
    """Is being in the middle of this doing having its object with you?

    `carry` moves its Theme as its Agent moves (carry-11.4: `motion` of both,
    one event), `hold` is in contact with it (hold-15.1), `keep` and `have`
    possess it (keep-15.2, own-100.1). So `what is Mary carrying` asks what is
    with Mary, and `what is Mary eating` does not.
    """
    joined, _ = _joined(verb, frozenset({"object"}), senses)
    for frame, _ in joined:
        positions = dict(frame.positions)
        moving = set()
        for predicate, negated, phase, roles in frame.semantics:
            parties = [positions.get(role.lstrip("?")) for role in roles]
            if negated:
                continue
            if (predicate in ("has_possession", "contact")
                    and parties[:2] == ["subject", "object"]
                    and phase in ("during", "")):
                return True
            if predicate == "motion" and parties[:1] in (["subject"],
                                                         ["object"]):
                moving.add(parties[0])
        if moving == {"subject", "object"}:
            return True
    return False


def _read(frame: Frame, verb: str, preposition: str,
          clause: str) -> list[Effect]:
    positions = dict(frame.positions)
    # `equals(Agent, Goal)`: a role the sentence leaves out that is, all the
    # same, one it says -- who gets a thing is who got it.
    same: dict[str, str] = {}
    for predicate, _, _, roles in frame.semantics:
        if predicate == "equals" and len(roles) == 2:
            one, other = (role.lstrip("?") for role in roles)
            for said, unsaid in ((one, other), (other, one)):
                if said in positions and unsaid not in positions:
                    same[unsaid] = positions[said]

    def where(role: str) -> str:
        role = role.lstrip("?")
        return positions.get(role) or same.get(role, "")

    # `the dog chased the cat` moves the dog as well as the cat (chase-51.6):
    # the cat was never with the dog to be let go of.
    agent_moves = any(predicate == "motion" and roles
                      and where(roles[0]) == "subject"
                      for predicate, _, _, roles in frame.semantics)
    out: list[Effect] = []
    named: dict[tuple, dict] = {}
    for predicate, negated, phase, roles in frame.semantics:
        if predicate == "path_rel" and "ch_of_state" in roles:
            if phase not in ("result", "end"):
                continue
            entity = next((positions[role] for role in roles
                           if role in positions and "State" not in role
                           and "Result" not in role), None)
            if entity in ("subject", "object"):
                out.append(Effect("state", entity, verb, True, False,
                                  frame.klass))
        elif predicate == "path_rel" and "ch_of_poss" in roles:
            parties = [role for role in roles
                       if role not in ("ch_of_poss", "prep")]
            theme = next((role for role in parties
                          if role.lstrip("?") in THEMES), None)
            holder = next((role for role in parties if role != theme), None)
            if theme is None or holder is None:
                continue
            thing, holding = where(theme), where(holder)
            if (thing not in ("subject", "object", "place") or not holding
                    or holding == thing):
                continue
            if phase == "end":
                out.append(Effect("location", thing, "", True, None,
                                  frame.klass, at=holding))
            elif phase == "start":
                out.append(Effect("location", thing, "", False, True,
                                  frame.klass, at=holding))
        elif predicate == "path_rel" and "ch_of_loc" in roles:
            entity = positions.get(roles[0].lstrip("?")) if roles else None
            if entity not in ("subject", "object"):
                continue
            if preposition:
                if phase == "end" and preposition in DESTINATION:
                    out.append(Effect("location", entity, "", True, False,
                                      frame.klass))
                elif phase == "start" and preposition in SOURCE:
                    out.append(Effect("location", entity, "", False, True,
                                      frame.klass))
                continue
            # `Carter entered the porch`, `Mary reached the kitchen`: the
            # place it ends at is the object (escape-51.1, reach-51.8).
            if (phase == "end" and entity == "subject" and len(roles) > 1
                    and where(roles[1]) == "object"):
                out.append(Effect("location", "subject", "", True, False,
                                  frame.klass, at="object"))
                continue
            if phase != "start" or len(roles) < 2:
                continue
            # `John left the kitchen`: it was at the object. `Mary dropped
            # the football`: it was somewhere the sentence does not say.
            was = where(roles[1])
            if was and was != entity:
                out.append(Effect("location", entity, "", False, True,
                                  frame.klass, at=was))
            elif not was and entity == "object" and not agent_moves:
                out.append(Effect("location", entity, "", False, True,
                                  frame.klass, at=""))
        elif (predicate == "contact" and negated and len(roles) == 2
              and phase in ("during", "end")):
            # `John discarded the milk`: in contact with it, then not.
            if (where(roles[0]), where(roles[1])) == ("subject", "object"):
                out.append(Effect("location", "object", "", False, True,
                                  frame.klass, at="subject"))
        elif predicate in ("begin", "end") and clause:
            if any(positions.get(role) == "clause" for role in roles):
                began = predicate == "begin"
                out.append(Effect("activity", "subject", clause, began,
                                  not began, frame.klass))
        elif (predicate in ("location", "has_location") and not negated
              and phase in ("during", "result", "end") and len(roles) >= 2):
            # `Mary left the apple in the kitchen`, `she kept it in the
            # drawer` (keep-15.2: location(during(E), Theme, Location)):
            # where the Theme is kept is where it is, until something moves
            # it (T3).
            if (where(roles[0]) in ("subject", "object")
                    and where(roles[1]) == "place"
                    and preposition in DESTINATION):
                out.append(Effect("location", where(roles[0]), "", True,
                                  None, frame.klass))
        elif (predicate not in MACHINERY and "_" not in predicate
              and phase in ("start", "result", "end")):
            entity = next((positions[role] for role in roles
                           if role in positions), None)
            if entity not in ("subject", "object") or not adjective(
                    predicate):
                continue
            slot = named.setdefault((predicate, entity), {})
            slot["before" if phase == "start" else "after"] = not negated
    for (word, entity), phases in named.items():
        out.append(Effect("state", entity, word, phases.get("after"),
                          phases.get("before"), frame.klass))
    return out

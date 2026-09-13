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

**Roles are matched by position.** A frame's syntax says which role is the
noun phrase before the verb, which the one after it, and which the one after a
preposition. `the vase broke` meets break-45.1's `NP.Patient VERB` and changes
its subject; `the cat broke the vase` meets `NP.Agent VERB NP.Patient` and
changes its object. Only frames of the sentence's own shape are read.

**A change of state is named by the verb.** A broken vase, a closed door: the
state a change-of-state verb brings about is its participle, so it is keyed by
the verb, and `is the vase broken` meets it through the lemma of `broken`. It
is kept only where WordNet has the participle, or the verb itself, as an
adjective: snooze-40.4 changes a sleeper's state, and nothing is `slept`. A
state VerbNet names with a word of its own -- `alive`, `free`, `visible` -- is
keyed by that word, and taken only when WordNet has the word as an adjective,
so `degradation_material_integrity` is never asked about.

**Only the classes of the sense meant.** `kill` is a member of murder-42.1,
and of amuse-31.1 (`that joke killed me`) and pain-40.8.1 (`my back is killing
me`); merging every class a verb is in would make a killed mouse amused. Each
VerbNet member carries WordNet 3.0 sense keys, and the store's verb senses are
WordNet 3.0 synsets, so a class is read only when one of its member's senses
is among the first three the store ranks for the verb -- the fallback the
rated norms use. A member with no sense keys is not read.

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

    def as_dict(self) -> dict:
        return {"kind": self.kind, "position": self.position,
                "word": self.word, "after": self.after,
                "before": self.before, "source": self.source}


@dataclass(frozen=True)
class Frame:
    klass: str
    #: role -> subject | object | place | clause
    positions: tuple
    #: which positions after the verb the frame has
    shape: frozenset
    #: (predicate, negated, phase, roles)
    semantics: tuple


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


def effects(verb: str, has_object: bool = False, preposition: str = "",
            clause: str = "", senses=None) -> list[Effect]:
    """What an occurrence of `verb` changes, in a sentence of this shape.

    `has_object`: a noun phrase right after the verb. `preposition`: the one
    before the last noun phrase, when there is one. `clause`: the verb of a
    clause after this one, `bark` in `started barking`. `senses`: the store's
    senses for the verb, best first; only classes whose member shares one of
    the first `SENSES` are read, and with none given every class is.
    """
    wanted = set()
    if has_object:
        wanted.add("object")
    if preposition:
        wanted.add("place")
    if clause:
        wanted.add("clause")
    meant = set(list(senses)[:SENSES]) if senses else None
    found: dict[tuple, Effect] = {}
    for frame, keys in frames().get(verb, ()):
        if frame.shape != frozenset(wanted):
            continue
        if meant is not None and not meant & {synset_of(key) for key in keys}:
            continue
        for effect in _read(frame, verb, preposition, clause):
            key = (effect.kind, effect.position, effect.word)
            found.setdefault(key, effect)
    return list(found.values())


def _read(frame: Frame, verb: str, preposition: str,
          clause: str) -> list[Effect]:
    positions = dict(frame.positions)
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
        elif predicate == "path_rel" and "ch_of_loc" in roles:
            entity = positions.get(roles[0].lstrip("?")) if roles else None
            if entity not in ("subject", "object") or not preposition:
                continue
            if phase == "end" and preposition in DESTINATION:
                out.append(Effect("location", entity, "", True, False,
                                  frame.klass))
            elif phase == "start" and preposition in SOURCE:
                out.append(Effect("location", entity, "", False, True,
                                  frame.klass))
        elif predicate in ("begin", "end") and clause:
            if any(positions.get(role) == "clause" for role in roles):
                began = predicate == "begin"
                out.append(Effect("activity", "subject", clause, began,
                                  not began, frame.klass))
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

"""Whether a subject can do something, on its own: any subject.

    can("person", "jump")              yes: people jump (and a person is one)
    can("person", "fly")               not attested: people fly helicopters,
                                       with machines -- nothing on its own
    can("computer program", "jump")    no: VerbNet's jumper is animate or a
                                       machine, and a program is neither
    can("goat", "jump", "over fence")  yes: goats jump fences

Three things decide it, in order:

1. **Who the verb admits** (VerbNet). A class's roles say what may fill
   them: run-51.3.2's Theme is `+animate | +machine`, eat-39.1's Agent
   `+animate`. When no class of the verb admits the subject's kind -- read
   through WordNet: animate is an animal or a person, concrete a physical
   entity -- it cannot, whatever anything says. That is what rules the
   program out of jumping.
2. **What its kind is said to do, unaided.** Walking up the subject's kinds
   from the nearest, the store's `capable_of` rows and ConceptNet's
   `CapableOf` edges about each: the nearest kind with a row decides. A row
   counts only in the shape asked about: `jump`, `jump over puddle` and
   `climb tree` are doing it oneself; `fly helicopter` and `fly with
   machines` are doing it with something, and say nothing of flying
   unaided. `NotCapableOf` at a nearer kind says no -- weakly, since
   ConceptNet's says what some people do not like as often as what nobody
   can.
3. **Otherwise it is not attested**, which is not no: nothing said.

The subject is any kind WordNet has -- a person, a hen, a program -- so
the one talking, the agent and anyone named are asked the same way, by
what they are.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from research.v694 import knowing as K
from research.v695 import mined

#: VerbNet's restrictions, as the WordNet kinds that meet them.
RESTRICTIONS = {
    "animate": ("animal.n.01", "person.n.01"),
    "human": ("person.n.01",),
    "machine": ("machine.n.01", "device.n.01"),
    "vehicle": ("vehicle.n.01", "craft.n.02"),
    "concrete": ("physical_entity.n.01",),
    "organization": ("social_group.n.01",),
    "biotic": ("organism.n.01",),
    "body_part": ("body_part.n.01",),
    "int_control": ("animal.n.01", "person.n.01", "machine.n.01"),
}

#: Prepositions that say what it was done *with*: not unaided.
WITH = frozenset({"with", "using", "aboard", "via"})
#: Prepositions that are done in or on something -- `swim in water`, `fly
#: in a plane` -- which is unaided unless the something is operated.
CARRIED = frozenset({"in", "on", "by", "inside"})
#: Things operated or ridden, as WordNet has them: a plane, a machine.
OPERATED = ("instrumentality.n.03",)
#: The doer's role in a VerbNet class, in the order looked for.
DOERS = ("Agent", "Theme", "Experiencer", "Patient")

#: What a path is, as an object: something one goes over, up or through.
PATHS = ("structure.n.01", "way.n.06", "location.n.01",
         "natural_object.n.01", "geological_formation.n.01",
         "body_of_water.n.01", "plant.n.02", "vegetation.n.01")

#: The two senses a person is named by in the store and in ConceptNet,
#: which WordNet keeps apart (`knowing.PERSONS`).
PEOPLE = ("people", "human", "humans")

#: How far up the subject's kinds evidence is looked for: the kind and
#: the one above it. `a carnivore can climb trees` is true of cats, and
#: says nothing of dogs; what the kind above that can do, some of it can.
FARTHEST = 2


@dataclass
class Able:
    #: yes | no | unattested
    verdict: str
    why: str = ""
    #: the kind the deciding evidence was about, and how far up it is
    kind: str = ""
    distance: int = -1
    evidence: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.verdict == "yes"


# -- VerbNet ---------------------------------------------------------------

_CLASSES: dict | None = None


def _classes() -> dict:
    """{verb: [(class id, wn keys, {role: (restrictions, logic)}, subject
    roles)]}, with each subclass inheriting its parent's roles."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from research.v687.corpora import VERBNET_DIR
    out: dict = {}

    def walk(node, inherited, subjects_above):
        roles = dict(inherited)
        for role in node.findall("THEMROLES/THEMROLE"):
            restrictions = [(one.get("Value"), one.get("type"))
                            for one in role.iter("SELRESTR")]
            logic = role.find("SELRESTRS")
            roles[role.get("type")] = (
                restrictions,
                (logic.get("logic") if logic is not None else None) or "and")
        subjects = set(subjects_above[0])
        objects = set(subjects_above[1])
        for frame in node.findall("FRAMES/FRAME"):
            syntax = list(frame.find("SYNTAX") or [])
            for part in syntax:
                if part.tag == "VERB":
                    break
                if part.tag == "NP":
                    subjects.add((part.get("value") or "").lstrip("?"))
                    break
            tags = [part.tag for part in syntax]
            if "VERB" in tags:
                after = syntax[tags.index("VERB") + 1:]
                if after and after[0].tag == "NP":
                    objects.add((after[0].get("value") or "").lstrip("?"))
        for member in node.findall("MEMBERS/MEMBER"):
            keys = (member.get("wn") or "").replace("?", "").split()
            out.setdefault(member.get("name"), []).append(
                (node.get("ID"), keys, roles, frozenset(subjects),
                 frozenset(objects)))
        for sub in node.findall("SUBCLASSES/VNSUBCLASS"):
            walk(sub, roles, (subjects, objects))

    try:
        for path in sorted(VERBNET_DIR.glob("*.xml")):
            walk(ET.parse(path).getroot(), {}, (set(), set()))
    except (OSError, ET.ParseError):
        pass
    _CLASSES = out
    return out


def _sense_keys(verb: str) -> set:
    try:
        from nltk.corpus import wordnet
        # WordNet's keys end `::`; VerbNet's do not.
        return {lemma.key().rstrip(":") for synset in wordnet.synsets(
            verb, wordnet.VERB)[:3] for lemma in synset.lemmas()
            if lemma.name().lower() == verb}
    except Exception:                              # noqa: BLE001
        return set()


def _meets(kind: str, restrictions: list, logic: str) -> bool | None:
    """Whether a kind meets a role's restrictions; None where they say
    nothing WordNet can check."""
    results = []
    for value, name in restrictions:
        wanted = RESTRICTIONS.get(name)
        if wanted is None:
            continue
        found = K.is_a(kind, *wanted)
        results.append(found if value == "+" else not found)
    if not results:
        return None
    return any(results) if logic == "or" else all(results)


def admits(kind: str, verb: str) -> tuple:
    """(whether some class of the verb admits this kind as its subject, the
    classes that were read). True when VerbNet does not say."""
    classes = _classes().get(verb, [])
    keys = _sense_keys(verb)
    matched = [one for one in classes if keys & set(one[1])] or classes
    if not matched:
        return True, []
    verdicts = []
    for class_id, _, roles, subjects, _ in matched:
        # The doer: the Agent where the class has one said first, else the
        # one that moves or undergoes it.
        doer = next((role for role in DOERS if role in subjects
                     and role in roles), None)
        verdicts.append(_meets(kind, *roles[doer]) if doer else None)
    known = [one for one in verdicts if one is not None]
    return (any(known) if known else True), [one[0] for one in matched]


def object_roles(verb: str) -> set:
    """The roles a verb's direct object plays, across its classes: `jump
    the fence` is a Location crossed, `leave the room` an Initial_Location
    left, `enter the car` a Destination, `fly the kite` a Theme moved."""
    out = set()
    classes = _classes().get(verb, [])
    keys = _sense_keys(verb)
    # The classes of the verb's common senses: `leave` is also leaving a
    # book on a table, whose object is a Theme put at a Destination.
    for one in [one for one in classes if keys & set(one[1])] or classes:
        out.update(one[4])
    return out


# -- evidence --------------------------------------------------------------

def _kinds(kind: str) -> list:
    """(names, distance) for a kind and those above it, nearest first."""
    first = K.first_sense(kind)
    if first is None:
        return [((kind,), 0)]
    out, level, seen = [], [first], set()
    for distance in range(FARTHEST):
        names = []
        for synset in level:
            if synset.name() in seen:
                continue
            seen.add(synset.name())
            names += [one.lower().replace("_", " ")
                      for one in synset.lemma_names()]
            if synset.name() == "person.n.01":
                names += list(PEOPLE)
        if names:
            out.append((tuple(dict.fromkeys(names)), distance))
        level = [above for synset in level for above in synset.hypernyms()]
        if not level:
            break
    return out


def unaided(verb: str, said: str) -> bool:
    """Whether a row says the verb was done by the one alone, in the shape
    of doing it oneself: nothing after it, a place or a path (`jump over
    puddle`, `climb tree`) -- not a thing operated (`fly helicopter`) or
    what it was done with (`fly with machines`)."""
    words = [one for one in said.lower().split() if one not in
             mined.ARTICLES]
    if not words or mined._morphy(words[0], True) != verb:
        return False
    rest = words[1:]
    if not rest:
        return True
    if rest[0] in WITH:
        return False
    try:
        from nltk.corpus import wordnet
        head = wordnet.morphy(rest[-1], wordnet.NOUN) or rest[-1]
        if rest[0] in CARRIED:
            # `swim in water`, not `fall in love`: a place one is in, and
            # not a thing one operates.
            return (K.is_a(head, "physical_entity.n.01")
                    and not K.is_a(head, *OPERATED))
        preposition = rest[0] in {"over", "across", "through", "into",
                                  "onto", "up", "down", "out", "around",
                                  "from", "off", "to", "under", "along",
                                  "past", "for", "at", "about", "away",
                                  "back", "high", "far", "fast", "quickly"}
        if preposition:
            return True
        head = wordnet.morphy(rest[-1], wordnet.NOUN) or rest[-1]
    except Exception:                              # noqa: BLE001
        return False
    return K.is_a(head, *PATHS)


def _rows(names: tuple, verb: str) -> list:
    """(relation, said, weight, where) about any of these names that begin
    with the verb: the store's and ConceptNet's."""
    out = []
    forms = {verb} | set(K.lemmas(verb))
    for name in names:
        like = name.replace("-", " ")
        for relation, said, confidence in K.connection().execute(
                "SELECT relation, object, confidence FROM facts WHERE "
                "relation IN ('capable_of', 'not_capable_of') AND (concept "
                "= ? OR (concept >= ? AND concept < ?))",
                (like, like + ".n.", like + ".n/")):
            first = said.lower().split()[:1]
            if first and mined._morphy(first[0], True) in forms:
                out.append((relation, said, confidence, "the store"))
        for head, tail, weight in (
                mined.edges("CapableOf", head=like)
                + [(h, t, -w) for h, t, w in mined.edges("NotCapableOf",
                                                         head=like)]):
            first = tail.lower().split()[:1]
            if first and mined._morphy(first[0], True) in forms:
                relation = "capable_of" if weight > 0 else "not_capable_of"
                out.append((relation, tail, abs(weight), "ConceptNet"))
    return out


def can(kind: str, verb: str, rest: str = "") -> Able:
    """Whether one of `kind` can `verb` (`rest`), on its own."""
    verb = mined._morphy(verb.lower(), True)
    admitted, classes = admits(kind, verb)
    if not admitted:
        return Able("no", f"VerbNet's {verb} ({', '.join(classes[:2])}) "
                          f"needs a doer that {kind_word(kind)} is not",
                    kind, 0)
    for names, distance in _kinds(kind):
        rows = _rows(names, verb)
        yes = [one for one in rows if one[0] == "capable_of"
               and unaided(verb, one[1])]
        no = [one for one in rows if one[0] == "not_capable_of"
              and unaided(verb, one[1]) and (not rest or rest in one[1])]
        if no and not yes:
            return Able("no", f"{names[0]} cannot {no[0][1]} "
                              f"({no[0][3]})", names[0], distance, no)
        if yes:
            best = max(yes, key=lambda one: one[2])
            return Able("yes", f"{_a(names[0])} can {best[1]} ({best[3]})",
                        names[0], distance, yes)
    return Able("unattested", f"nothing says {_a(kind)} can {verb} on its "
                              f"own", kind)


def kind_word(kind: str) -> str:
    return _a(kind.replace("_", " "))


def _a(word: str) -> str:
    return ("an " if word[:1] in "aeiou" else "a ") + word

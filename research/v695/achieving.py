"""How a goal is reached: the actions that do it, that the one asking can do.

    how would I get over a fence       climb over it, or jump over it
    how can a dog get over a fence     jump over it
    how would a hen get over a fence   fly over it

A goal is a doing with where it goes: its verb, the words that place it
(`over`, `out of`, `down from`) and the thing (`fence`). Actions that reach
it are found three ways, none of them by name:

1. **Done for it** -- ConceptNet's `MotivatedByGoal` (`mined.py`): climbing
   is done to get over a fence.
2. **Seen doing it** -- any row, the store's or ConceptNet's, where
   something does a verb of the same kind as the goal's (WordNet's verb
   file: both are motion) to the same thing in the same place: deer, dogs
   and goats *jump over the fence*, hens *fly over the fence*. Who was seen
   is kept: it is evidence the action works, not that the asker can do it.
3. **What the language says** -- a verb whose WordNet gloss puts the
   goal's own place right before something the thing is a kind of: *vault:
   jump across or leap over (an obstacle)*, and a fence is an obstacle.
   (WordNet's phrasal verbs -- *climb down*, *file out* -- were tried as a
   fourth source and dropped: most are not ways to reach the place.)

Then **who can** (`can.py`): each action is kept only if the subject --
the one asking, the agent, a dog -- can do it on its own. A person does
not fly; a hen does.

The steps of the action, where ConceptNet has them for this very doing --
what comes first, what it needs -- are the set of actions it takes
(`steps`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v691 import hearing
from research.v694 import knowing as K
from research.v695 import can as C
from research.v695 import mined

#: Verbs that say nothing of how: they are the goal's, never a way to it.
LIGHT = frozenset({"get", "go", "be", "have", "do", "make", "come", "take",
                   "put", "use", "try", "want", "need", "move"})
#: Who a pronoun is, as a kind.
PRONOUNS = {"i": "person", "we": "person", "you": "person",
            "one": "person", "they": "person", "someone": "person"}
#: How many kinds up the thing's words are read for what a gloss names it.
ABOVE = 3
#: How many actions are said, and how well supported one must be, against
#: the best, to be said at all.
SAID = 3
WEAKEST = 0.3
#: What a place word says of the thing, as the VerbNet role a transitive
#: verb's object would play for the same goal: `jump the fence` crosses a
#: Location, `leave the hole` leaves an Initial_Location, `enter the car`
#: reaches a Destination. `enter the river` is not getting across it.
ROLES = {
    "over": {"Location", "Trajectory"}, "across": {"Location", "Trajectory"},
    "through": {"Location", "Trajectory"}, "past": {"Location", "Trajectory"},
    "around": {"Location", "Trajectory"}, "along": {"Location", "Trajectory"},
    "out": {"Initial_Location", "Source"}, "from": {"Initial_Location",
                                                   "Source"},
    "off": {"Initial_Location", "Source"}, "down": {"Initial_Location",
                                                   "Location"},
    "into": {"Destination", "Goal"}, "in": {"Destination", "Goal"},
    "onto": {"Destination", "Goal"}, "to": {"Destination", "Goal"},
    "up": {"Location", "Destination"},
}
#: Roles that say the opposite of a place: a verb whose object may be
#: either -- `leave the car` -- says nothing of getting into it.
OPPOSITE = {"Destination": {"Initial_Location", "Source"},
            "Goal": {"Initial_Location", "Source"},
            "Initial_Location": {"Destination", "Goal"},
            "Source": {"Destination", "Goal"}}
#: How many seen doing it make it evidence, unless one of them is the
#: subject's own kind: one automaton that turns into a car is not.
WITNESSES = 2
#: What a doing with a place is, whatever its verb alone is filed as: `get
#: across` is WordNet's communicating, and `get across a river` a motion.
MOTION = "verb.motion"


@dataclass
class Goal:
    verb: str
    #: where it goes: `over`, `out of`, `down from`
    place: str
    thing: str
    #: who is to do it, as a kind
    subject: str = "person"
    said: str = ""

    @property
    def phrase(self) -> str:
        return " ".join(one for one in (self.verb, self.place, self.thing)
                        if one)


@dataclass
class Action:
    verb: str
    #: as it would be said: `climb over the fence`
    said: str
    score: float = 0.0
    why: list = field(default_factory=list)
    #: who was seen doing it
    seen: list = field(default_factory=list)
    able: C.Able | None = None
    steps: list = field(default_factory=list)


# -- reading ---------------------------------------------------------------

def _subject(words, verb) -> str:
    for one in hearing.children(words, verb.index, {"nsubj"}):
        if one.text in PRONOUNS:
            return PRONOUNS[one.text]
        if one.tag in ("NN", "NNS"):
            return mined._morphy(one.text, False)
        return "person"
    return "person"


def read(text: str) -> Goal | None:
    """A how-question's goal, when it is a doing that goes somewhere."""
    words = hearing.parse(text)
    if not words:
        return None
    asking = words[0].text == "how" or any(
        one.lemma in ("way", "best") for one in words[:5])
    if not asking:
        return None
    verbs = [one for one in words if one.tag.startswith("VB")
             and one.lemma not in ("be", "do", "would", "can", "could",
                                   "should", "will")
             and one.dep in ("ROOT", "relcl", "xcomp", "acl", "advcl")]
    if not verbs:
        return None
    verb = verbs[0]
    place, thing = [], None
    for one in hearing.children(words, verb.index, {"prt"}):
        place.append(one.text)
    node = verb
    for _ in range(3):
        preps = hearing.children(words, node.index, {"prep"})
        if not preps:
            break
        prep = preps[0]
        place.append(prep.text)
        objects = hearing.children(words, prep.index, {"pobj"})
        if objects:
            thing = objects[0]
            break
        node = prep
    if thing is None or not place or thing.tag not in ("NN", "NNS"):
        return None
    if hearing.children(words, verb.index, {"dobj"}):
        return None                  # done to something: the designer's
    return Goal(verb.lemma, " ".join(place),
                mined._morphy(thing.text, False), _subject(words, verb),
                text)


# -- the actions -----------------------------------------------------------

def _file(verb: str, place: str = "") -> str:
    """WordNet's file for a verb, as a phrasal verb where it is one:
    `get over` is traverse, a motion; `get` alone is possession."""
    try:
        from nltk.corpus import wordnet
        for form in ([f"{verb}_{place.split()[0]}"] if place else []) + [
                verb]:
            found = wordnet.synsets(form, wordnet.VERB)
            if found:
                return found[0].lexname()
    except Exception:                              # noqa: BLE001
        pass
    return ""


def _obstacle_words(thing: str) -> set:
    """The thing's words and those of the kinds above it: a fence is a
    barrier, an obstruction, an obstacle."""
    out = {thing}
    first = K.first_sense(thing)
    level = [first] if first is not None else []
    for _ in range(ABOVE):
        above = []
        for synset in level:
            for one in synset.hypernyms():
                out.update(name.lower().replace("_", " ")
                           for name in one.lemma_names())
                above.append(one)
        level = above
    return out


def _add(found: dict, verb: str, said: str, score: float, why: str,
         seen: str = "") -> None:
    one = found.setdefault(verb, Action(verb, said))
    one.score += score
    if why and why not in one.why:
        one.why.append(why)
    if seen and seen not in one.seen:
        one.seen.append(seen)


def _done_for(goal: Goal, found: dict) -> None:
    for head, _, weight in mined.edges("MotivatedByGoal", tail=goal.phrase):
        words = head.lower().split()
        verb = mined._morphy(words[0], True) if words else ""
        if not verb or verb in LIGHT:
            continue
        _add(found, verb, f"{verb} {goal.place} the {goal.thing}",
             2.0 * weight,
             f"{_ing(verb)} is done to {goal.verb} {goal.place} "
             f"{_a(goal.thing)} (ConceptNet)")


def _seen(goal: Goal, found: dict) -> None:
    """Rows where something does a verb like the goal's to the thing, in
    the same place: `deer capable_of jump over the fence`."""
    kind = MOTION
    places = goal.place.split()
    rows = [(head, tail, weight) for head, relation, tail, weight in
            mined.mentioning(goal.thing, "tail", ("CapableOf",))]
    for row in K.index().mentioning(K.lemmas(goal.thing), ("capable_of",)):
        rows.append((K.word_of(row.concept), row.said, row.confidence))
    for head, tail, weight in rows:
        words = [one for one in tail.lower().split()
                 if one not in mined.ARTICLES]
        if len(words) < 2:
            continue
        verb = mined._morphy(words[0], True)
        if verb in LIGHT or verb == goal.verb:
            continue
        if mined._morphy(words[-1], False) != goal.thing:
            continue
        between = words[1:-1]
        if between and between != places[:len(between)]:
            continue
        if not between:
            wanted = ROLES.get(places[0], set())
            roles = C.object_roles(verb)
            against = set().union(*(OPPOSITE.get(one, set())
                                    for one in wanted))
            if not wanted & roles or (against & roles
                                      and not {"Location", "Trajectory"}
                                      & wanted):
                continue
        if _file(verb) != kind:
            continue
        # Seen doing it is evidence only from something that does things:
        # a path follows over a fence, and nobody learns from that how to.
        if not K.is_a(head.replace(" ", "_"), *C.RESTRICTIONS["animate"]):
            continue
        said = (f"{verb} {' '.join(between)} the {goal.thing}" if between
                else f"{verb} the {goal.thing}")
        _add(found, verb, said, min(float(weight), 1.0),
             "", mined.norm(head.replace("_", " ")))


_GLOSSES: list | None = None


def _glosses() -> list:
    global _GLOSSES
    if _GLOSSES is None:
        try:
            from nltk.corpus import wordnet
            _GLOSSES = [(synset.lemma_names()[0].lower(),
                         synset.definition().lower())
                        for synset in wordnet.all_synsets(wordnet.VERB)]
        except Exception:                          # noqa: BLE001
            _GLOSSES = []
    return _GLOSSES


def _named(goal: Goal, found: dict) -> None:
    """Verbs whose gloss is the goal's place and something the thing is:
    *vault -- jump across or leap over (an obstacle)*."""
    place = goal.place.split()[0]
    names = {name for name in _obstacle_words(goal.thing) if len(name) > 3}
    for verb, gloss in _glosses():
        if "_" in verb or verb in LIGHT or _file(verb) != MOTION:
            continue
        words = [one for one in gloss.replace("(", " ").replace(
            ")", " ").replace(",", " ").split() if one not in mined.ARTICLES]
        if not any(word == place and " ".join(words[index + 1:index + 3])
                   .startswith(tuple(names))
                   for index, word in enumerate(words)):
            continue
        _add(found, verb, f"{verb} {goal.place} the {goal.thing}", 0.5,
             f"to {verb} is to {gloss} (WordNet)")


def steps(goal: Goal, action: Action) -> list:
    """What comes first and what it needs, for this very doing: edges whose
    head is the action done to the thing (`climb fence`), not the verb in
    general -- climbing in general needs rope, a fence does not."""
    out = []
    for head in (f"{action.verb} {goal.thing}",
                 f"{action.verb} {goal.place} {goal.thing}"):
        for relation in ("HasPrerequisite", "HasFirstSubevent"):
            for _, tail, _ in mined.edges(relation, head=head):
                if tail not in out:
                    out.append(tail)
    return out


def actions(goal: Goal) -> list:
    """The actions that reach a goal, best supported first, each with
    whether the subject can do it."""
    found: dict = {}
    _done_for(goal, found)
    _seen(goal, found)
    _named(goal, found)
    for verb in list(found):
        one = found[verb]
        alone = not one.why and len(one.seen) < WITNESSES and not any(
            K.is_a(seen.replace(" ", "_"), goal.subject)
            or seen == goal.subject for seen in one.seen)
        if alone:
            del found[verb]
    for one in found.values():
        if one.seen:
            one.score += min(len(one.seen), 3) * 0.5
            one.why.append(f"{_listed(one.seen[:3])} "
                           f"{'is' if len(one.seen) == 1 else 'are'} seen "
                           f"to {one.said.replace(' the ', ' a ')}")
        one.able = C.can(goal.subject, one.verb)
        one.steps = steps(goal, one)
    return sorted(found.values(), key=lambda one: -one.score)


# -- saying it -------------------------------------------------------------

def _a(word: str) -> str:
    return ("an " if word[:1] in "aeiou" else "a ") + word


def _ing(verb: str) -> str:
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if (len(verb) > 2 and verb[-1] not in "aeiouwxy"
            and verb[-2] in "aeiou" and verb[-3] not in "aeiou"
            and len(verb) <= 4):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _listed(names) -> str:
    names = list(names)
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def answer(goal: Goal) -> str:
    """The actions the subject can do, and why each works."""
    found = actions(goal)
    doable = [one for one in found if one.able]
    if doable:
        doable = [one for one in doable
                  if one.score >= WEAKEST * doable[0].score]
    who = ("You" if goal.subject == "person"
           else _a(goal.subject).capitalize())
    if not doable:
        if not found:
            return ""
        others = [one for one in found if one.seen][:2]
        if not others:
            return ""
        return (f"I know how others do it -- {_listed(one.why[-1] for one in others)}"
                f" -- but nothing says {_a(goal.subject)} can "
                f"{_listed(one.verb for one in others)}.")
    best = doable[:SAID]
    ways = _listed(one.said for one in best)
    reasons = [one.why[0] for one in best if one.why]
    if goal.subject != "person":
        reasons += [one.able.why for one in best]
    reasons = [one for one in dict.fromkeys(reasons) if one]
    text = f"{who} could {ways}"
    text = text.replace(" and ", " or ") if len(best) > 1 else text
    first = best[0]
    if first.steps:
        text += f" (first: {_listed(first.steps[:2])})"
    return text + " -- " + "; ".join(reasons) + "."

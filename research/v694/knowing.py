"""What the store knows that a designer needs: what things are *for*.

v691 plans with what VerbNet says verbs do, over the things a conversation
names. That is enough to get a book to the kitchen and not enough to cut a
rope: VerbNet says cutting has an Instrument, and nothing about which thing
is one. The store does -- ConceptNet and Ascent++ were read into it as
`used_for`, `capable_of`, `has_property`, `at_location`, `made_of` -- and
this is where a designer asks it:

    used_for    refrigerator  keep food cold        a means to a state
    capable_of  scissors      cut paper             a means to a doing
    capable_of  people        open door             a doing needs no means
    has_property knife        sharp
    at_location knife         kitchen drawer        where one is found
    made_of     raft          log                   what one is made from

The rows are free text, so they are read the only way free text can be read
without a parser: **word by word, each word by its lemma** (WordNet's
`morphy`), and matched against the goal by what the words are. A row
serves a goal when its verb is the goal's verb (`cut paper` for *cut the
rope*), or when it keeps, makes or gets a thing in the goal's state (`keep
food cold` for *make the milk cold*). It serves it **better** when the
thing it names is the thing the goal is about, or a kind of it -- milk is a
food, so `keep food cold` is about milk and `cut hair` is not about rope.

Nothing here names a tool, a state or a verb. Every word comes from the
goal, and every candidate from the store.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

#: The relations read as *what a thing can be used to do*.
SERVING = ("used_for", "capable_of")

#: Verbs a row uses to say a thing is *left in* a state: `keep food cold`,
#: `make room warm`, `get warm`, `stay dry`. English's own light verbs, not a
#: list of states or tools.
KEEPING = frozenset({"keep", "make", "get", "stay", "remain", "add",
                     "provide", "give", "become"})

#: Words a row uses for *anything at all*: `keep things cold` is about milk
#: as much as about anything, and is not about some other thing.
ANYTHING = frozenset({"thing", "things", "stuff", "item", "object",
                      "something", "everything", "anything"})

#: Words about *who*, which say nothing of what a thing is used on.
PEOPLE = frozenset({"person", "people", "someone", "somebody", "you",
                    "one", "yourself", "human", "they", "them", "we",
                    "u", "i", "me"})

#: Small words that are not the thing a row is about.
FILLER = frozenset("""a an the of to for in on at with by from and or up
down out off into onto over under your my his her its their our this that
these those some all any it is are be when while as so very""".split())

#: The two senses of *people* that the store says can do things: whatever
#: they can do, a person can do with no means at all.
PERSONS = ("person.n.01", "people.n.01")


def _store_path():
    from research.v687 import build
    return build.DEFAULT_STORE


_CONNECTION = None


def connection():
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = sqlite3.connect(f"file:{_store_path()}?mode=ro",
                                      uri=True, check_same_thread=False)
    return _CONNECTION


# -- words -----------------------------------------------------------------

_LEMMAS: dict = {}


def lemmas(word: str) -> frozenset:
    """Every lemma a word could be: `cutting` is cut, `knives` knife,
    `cooling` cool. The word itself too, since `cold` is its own."""
    if word in _LEMMAS:
        return _LEMMAS[word]
    found = {word}
    try:
        from nltk.corpus import wordnet
        for pos in (wordnet.VERB, wordnet.NOUN, wordnet.ADJ):
            one = wordnet.morphy(word, pos)
            if one:
                found.add(one)
    except Exception:                              # noqa: BLE001
        pass
    _LEMMAS[word] = frozenset(found)
    return _LEMMAS[word]


def verb_of(word: str) -> str:
    """The verb a word is, if it is one: `cutting` is cut."""
    try:
        from nltk.corpus import wordnet
        return wordnet.morphy(word, wordnet.VERB) or ""
    except Exception:                              # noqa: BLE001
        return ""


def word_of(concept: str) -> str:
    """`carving knife.n.01` -> `carving knife`."""
    parts = concept.rsplit(".", 2)
    return parts[0] if len(parts) == 3 else concept


def name_of(concept_or_word: str) -> str:
    """A thing's name in a fact: one token, `carving-knife`."""
    return word_of(concept_or_word).replace(" ", "-").replace("_", "-")


_STATES: dict = {}


def state_words(state: str) -> frozenset:
    """The words that say a thing is in `state`, from WordNet's first
    adjective sense: its lemmas and the adjectives it lists as similar
    (`cold`: chilly, cool, frozen, icy), their nouns (`warmth`), and the
    verbs that leave a thing so (`cool`, `chill`, `freeze`, `refrigerate`).

    Only the first sense: `cold` is also *emotionless*, and a row that
    keeps a person emotionless does not keep milk cold. And only the
    state's own nouns and verbs, not its neighbours': *snappy* weather is
    cold, and snapping is not a way of making milk so; *baked* earth is
    dry, and baking is not how a shirt is dried.
    """
    if state in _STATES:
        return _STATES[state]
    found = {state}
    try:
        from nltk.corpus import wordnet
        senses = wordnet.synsets(state, wordnet.ADJ)
        if not senses:
            senses = wordnet.synsets(state)[:1]
        for synset in senses[:1]:
            for one in [synset] + synset.similar_tos():
                own = one is synset
                for lemma in one.lemmas():
                    name = lemma.name().lower()
                    if "_" in name or "-" in name:
                        continue
                    if not own and name.endswith("ing"):
                        continue
                    found.add(name)
                    if own:
                        for other in lemma.derivationally_related_forms():
                            if "_" not in other.name():
                                found.add(other.name().lower())
                    verb = wordnet.morphy(name, wordnet.VERB)
                    if verb and own:
                        found.add(verb)
    except Exception:                              # noqa: BLE001
        pass
    _STATES[state] = frozenset(found)
    return _STATES[state]


def state_verbs(state: str) -> list:
    """The verbs WordNet derives from a state's own word -- `bright` gives
    brighten, `dark` darken -- not its neighbours': *crisp* weather is cold,
    and crisping is not how milk is made so."""
    out: list = []
    try:
        from nltk.corpus import wordnet
        for synset in wordnet.synsets(state, wordnet.ADJ)[:1]:
            for lemma in synset.lemmas():
                if lemma.name() != state:
                    continue
                for other in lemma.derivationally_related_forms():
                    if other.synset().pos() == "v" and \
                            other.name() not in out:
                        out.append(other.name())
    except Exception:                              # noqa: BLE001
        pass
    return out


_KINDS: dict = {}


def kinds_of(word: str) -> frozenset:
    """What a thing is, as words: itself and everything above its first
    few noun senses. Milk is a dairy product, a food, a liquid, a
    substance."""
    if word in _KINDS:
        return _KINDS[word]
    found = {word, word.replace("-", " ")}
    try:
        from nltk.corpus import wordnet
        for synset in wordnet.synsets(word.replace("-", "_").replace(
                " ", "_"), wordnet.NOUN)[:3]:
            for one in [synset] + list(synset.closure(
                    lambda s: s.hypernyms())):
                if one is not synset and _general(one):
                    continue
                for lemma in one.lemma_names():
                    found.add(lemma.lower().replace("_", " "))
    except Exception:                              # noqa: BLE001
        pass
    _KINDS[word] = frozenset(found)
    return _KINDS[word]


#: Kinds too general to say what a row is about: a tooth is a whole and an
#: object, and so is everything a tractor pulls. WordNet's own top, where
#: depth does not already say so (`whole` is seven deep).
GENERAL = frozenset({"whole.n.02", "artifact.n.01", "instrumentality.n.03",
                     "living_thing.n.01", "organism.n.01",
                     "causal_agent.n.01", "matter.n.03", "part.n.01",
                     "unit.n.03"})


def _general(synset) -> bool:
    return synset.min_depth() <= 2 or synset.name() in GENERAL


#: How far up from each of two things their shared kind may be, for them to
#: be alike: rope and cord are both lines, one step up each; knife and
#: scissors both edge tools.
SIBLINGS = 2
#: How deep in WordNet that shared kind must be. Everything shares *entity*;
#: a shared *line* or *edge tool* says something.
SPECIFIC_KIND = 5

_SIMILAR: dict = {}

#: What a row about a close sibling of the goal's thing is worth (`about`):
#: between a row about nothing in particular (1) and one about a kind of
#: it (2).
SIBLING = 1.5


def similar(one: str, other: str) -> float:
    """How alike two things are, for carrying what worked for one over to
    the other: 1 for the same thing or a kind of it (a carving knife is a
    knife), a half for close siblings (rope and cord, knife and scissors,
    cloth and rag), 0 otherwise. From the first sense of each, as
    `kinds_of` reads them."""
    one, other = one.replace("-", " "), other.replace("-", " ")
    if one == other or one in kinds_of(other) or other in kinds_of(one):
        return 1.0
    key = (one, other) if one < other else (other, one)
    if key not in _SIMILAR:
        _SIMILAR[key] = 0.5 if _siblings(one, other) else 0.0
    return _SIMILAR[key]


def _up(synset, steps: int) -> dict:
    out, frontier = {synset: 0}, [synset]
    for depth in range(1, steps + 1):
        nxt = []
        for one in frontier:
            for above in one.hypernyms():
                if above not in out:
                    out[above] = depth
                    nxt.append(above)
        frontier = nxt
    return out


def _siblings(one: str, other: str) -> bool:
    first, second = first_sense(one), first_sense(other)
    if first is None or second is None:
        return False
    mine, theirs = _up(first, SIBLINGS), _up(second, SIBLINGS)
    return any(kind in theirs and kind.min_depth() >= SPECIFIC_KIND
               and not _general(kind) for kind in mine)


def similar_verb(one: str, other: str) -> float:
    """`_similar_verb` either way round: what worked for baking is worth
    trying for cooking, and the other way."""
    return max(_similar_verb(one, other), _similar_verb(other, one))


def _similar_verb(one: str, other: str) -> float:
    """How alike two doings are: 1 for the same verb or a synonym (*fix*
    and *repair* share a sense), a half where one is a way of doing the
    other (*slice*, *chop* and *saw* are ways to cut; *toast* is heating;
    *bake* is cooking), 0 otherwise. The first sense of the one, against
    the other's first few: baking is the third sense of *cook*."""
    if one == other:
        return 1.0
    try:
        from nltk.corpus import wordnet
        mine = wordnet.synsets(one, wordnet.VERB)[:3]
        theirs = wordnet.synsets(other, wordnet.VERB)[:3]
    except Exception:                              # noqa: BLE001
        return 0.0
    if not mine or not theirs:
        return 0.0
    if mine[0] == theirs[0]:
        return 1.0
    if any(one in mine[0].hypernyms() for one in theirs) or any(
            mine[0] in one.hypernyms() for one in theirs[:1]):
        return 0.5
    return 0.0


_FIRST: dict = {}


def first_sense(word: str):
    """A word's most common noun sense, or None."""
    if word not in _FIRST:
        try:
            from nltk.corpus import wordnet
            found = wordnet.synsets(word.replace("-", "_").replace(" ", "_"),
                                    wordnet.NOUN)
            _FIRST[word] = found[0] if found else None
        except Exception:                          # noqa: BLE001
            _FIRST[word] = None
    return _FIRST[word]


def is_a(word: str, *kinds: str) -> bool:
    """Whether a word's most common sense is one of these kinds of thing
    (WordNet synset names: `artifact.n.01`). Its most common sense, as
    `verbs.Things.acts` reads who acts: a knife is a tool and not the
    knife a surgeon is called."""
    first = first_sense(word)
    if first is None:
        return False
    above = {one.name() for one in first.closure(lambda s: s.hypernyms())}
    above.add(first.name())
    return any(kind in above for kind in kinds)


def lives(word: str) -> bool:
    return is_a(word, "living_thing.n.01", "person.n.01")


def an_artifact(word: str) -> bool:
    return is_a(word, "artifact.n.01")


# -- the store, read once --------------------------------------------------

@dataclass(frozen=True)
class Row:
    concept: str
    relation: str
    said: str
    confidence: float
    #: each word of `said` with its lemmas, in order
    words: tuple

    @property
    def thing(self) -> str:
        return word_of(self.concept)


class Index:
    """The rows a designer asks about, by every lemma of every word in
    them: a row saying `keeping food cold` is found from `keep`, `food`
    and `cold` alike."""

    def __init__(self, relations=SERVING + ("has_property",)) -> None:
        self.by_lemma: dict = defaultdict(list)
        self.relations = relations
        marks = ",".join("?" for _ in relations)
        for concept, relation, said, confidence in connection().execute(
                f"SELECT concept, relation, object, MAX(confidence) FROM "
                f"facts WHERE relation IN ({marks}) GROUP BY concept, "
                f"relation, object", relations):
            words = tuple((word, lemmas(word)) for word in
                          said.lower().replace(",", " ").split()[:8])
            if not words:
                continue
            row = Row(concept, relation, said.lower(), confidence, words)
            seen = set()
            for _, forms in words:
                for one in forms:
                    if one not in seen:
                        seen.add(one)
                        self.by_lemma[one].append(row)

    def mentioning(self, words, relations=None) -> list:
        out, seen = [], set()
        for word in words:
            for row in self.by_lemma.get(word, ()):
                if relations and row.relation not in relations:
                    continue
                key = (row.concept, row.relation, row.said)
                if key not in seen:
                    seen.add(key)
                    out.append(row)
        return out


_INDEX: Index | None = None


def index() -> Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX


def rows_about(concept_word: str, relation: str) -> list:
    """(object, confidence) the store says of a thing, by its word, best
    first: every sense of the word, since the store's senses are not
    WordNet's."""
    like = concept_word.replace("-", " ")
    return connection().execute(
        "SELECT object, MAX(confidence) FROM facts WHERE relation = ? AND "
        "(concept = ? OR concept LIKE ?) GROUP BY object ORDER BY "
        "MAX(confidence) DESC", (relation, like, like + ".n.%")).fetchall()


# -- reading one row against a goal ----------------------------------------

def _nouns(row: Row, skip: frozenset) -> list:
    """The words of a row that could name what it is used on: not its
    verb, not a filler, not a word of the goal's own."""
    out = []
    for index, (word, forms) in enumerate(row.words):
        if index == 0 and verb_of(word):
            continue
        if word in FILLER or forms & skip:
            continue
        if forms & ANYTHING:
            continue
        out.append(word)
    return out


def about(row: Row, patient: str, skip: frozenset = frozenset()) -> float:
    """How far a row is about the goal's thing: 3 when it names the thing
    itself (`kill flies`, for a fly), 2 when it names a kind of it (`keep
    food cold`, for milk; `wake people up`, for john), 1 when it names
    nothing (`cutting`, `keep things cold`), and a third when it names
    something else (`cut hair`, for a rope) -- still a way to cut, but not
    shown to be one for this."""
    if not patient:
        return 1.0
    kinds = kinds_of(patient)
    own = lemmas(patient.replace("-", " ")) | {patient.replace("-", " ")}
    named = _nouns(row, skip)
    if "person" in kinds and any(word in PEOPLE for word in named):
        return 2.0
    named = [word for word in named if word not in PEOPLE]
    if not named:
        return 1.0
    best = 1 / 3
    parts = meronyms(patient)
    for word in named:
        forms = lemmas(word)
        if forms & own:
            return 3.0
        if any(one in kinds for one in forms) or any(
                name_of(one) in parts for one in forms):
            # A kind of it, or a part of it: sweeping a floor is how a
            # kitchen is swept.
            best = 2.0
    # A pair of words may be the name: `cut paper towel`.
    joined = " ".join(named)
    if joined in own:
        return 3.0
    if joined in kinds:
        best = 2.0
    if best < 2.0 and any(similar(word, patient) >= 0.5 for word in named
                          if first_sense(word) is not None):
        # A close sibling: what the store says of toilets it nearly says
        # of sinks, and what it says of rope, of cord. Less than the thing
        # or a kind of it, more than something else.
        best = SIBLING
    return best


def does(row: Row, verb: str) -> bool:
    """Whether a row says doing `verb`: its first word is the verb, as
    itself or doing it (`cut`, `cuts`, `cutting`) -- not as what has been
    done to something: an axle *fixed to the vehicle* does not fix cars."""
    if not row.words or verb not in row.words[0][1]:
        return False
    word = row.words[0][0]
    return word in (verb, verb + "s", verb + "es") or word.endswith("ing")


def keeps(row: Row, state: str) -> bool:
    """Whether a row says leaving a thing in `state`: `keep food cold`,
    `cooling`, `add warmth`, or the state itself as a property."""
    words = state_words(state)
    if not row.words:
        return False
    first = row.words[0][1]
    if row.relation == "has_property":
        return len(row.words) == 1 and bool(first & words)
    if first & words:
        return True
    return bool(first & KEEPING) and any(forms & words
                                         for _, forms in row.words[1:])


# -- means -----------------------------------------------------------------

@dataclass
class Means:
    """A thing the store says serves a goal, and the rows that say so."""

    thing: str
    score: float = 0.0
    rows: list = field(default_factory=list)
    #: whether some row was about the goal's own thing, or a kind of it
    fits: bool = False
    #: the most any row was about the goal's thing (`about`): below 1, every
    #: row was about something else
    reach: float = 0.0
    #: why it is offered first, when it worked before or was taught
    remembered: str = ""

    @property
    def name(self) -> str:
        return name_of(self.thing)

    def why(self) -> str:
        """The row that counted most, as the store said it."""
        best = self.rows[0] if self.rows else None
        if best is None:
            return f"the word itself names {self.thing}"
        relation = {"used_for": "is used for", "capable_of": "can",
                    "has_property": "is"}.get(best.relation, best.relation)
        said = self.thing.replace("_", " ")
        article = "an" if said[:1] in "aeiou" else "a"
        return f"{article} {said} {relation} {best.said}"


#: How much a row may add to a means' score. The store says `refrigerator
#: used_for keep food cold` in ten phrasings; ten phrasings are more
#: evidence than one, and not ten times more.
MOST_ROWS = 6


#: How much each relation says a *made thing* serves. `used_for` is what a
#: tool is for; `capable_of` of an artifact says the thing does it itself
#: -- scissors cut paper, and also a microphone fixes cars -- which is
#: evidence, and half as much. So is a property: a basement *is* dry,
#: which is not the same as being what one dries a shirt in.
AS_TOOL = {"used_for": 1.0, "capable_of": 0.5, "has_property": 0.5}


def serving(verb: str = "", state: str = "", patient: str = "",
            relations=SERVING, keep=None, weights=None) -> list:
    """Things the store says serve a goal -- doing `verb`, or leaving a
    thing in `state` -- on `patient`, best supported first.

    `keep(word)` says which things may serve at all: artifacts for a tool,
    living things for someone to ask. A thing's score is its rows'
    confidences, each counted by how far it is about the patient
    (`about`) and by what its relation says (`weights`), at most
    `MOST_ROWS` of them.
    """
    if verb:
        words = lemmas(verb)
        rows = [row for row in index().mentioning(words, relations)
                if does(row, verb)]
        skip = words
    else:
        words = state_words(state)
        rows = [row for row in index().mentioning(words, relations)
                if keeps(row, state)]
        skip = frozenset(words) | KEEPING
    by: dict = {}
    for row in rows:
        thing = row.thing
        if patient and thing.replace(" ", "-") == patient:
            continue
        if keep is not None and not keep(thing):
            continue
        weight = about(row, patient, skip) * (
            weights.get(row.relation, 1.0) if weights else 1.0)
        found = by.setdefault(thing, [])
        found.append((weight * row.confidence, weight, row))
    out = []
    for thing, found in by.items():
        found.sort(key=lambda one: -one[0])
        kept = found[:MOST_ROWS]
        # The best row is the evidence; the rest are corroboration, and
        # count a quarter each. Summed evenly, a gun that kills animals in
        # six phrasings outscores the flyswatter that kills flies.
        means = Means(thing, kept[0][0] + sum(one[0] for one in kept[1:])
                      / 4, [one[2] for one in kept],
                      any(about(one[2], patient, skip) >= 2
                          for one in found))
        means.reach = max(about(one[2], patient, skip) for one in found)
        out.append(means)
    out.sort(key=lambda one: (-one.score, one.thing))
    return out


def person_can(verb: str, patient: str = "") -> float:
    """How well the store attests that a person does `verb` to `patient`
    with nothing but themselves: `people capable_of open door`, yes;
    nothing about cutting rope bare-handed, no. 0 when it says nothing."""
    best = 0.0
    for concept in PERSONS:
        for said, confidence in connection().execute(
                "SELECT object, confidence FROM facts WHERE concept = ? AND "
                "relation = 'capable_of'", (concept,)):
            words = tuple((word, lemmas(word))
                          for word in said.lower().split()[:8])
            row = Row(concept, "capable_of", said.lower(), confidence, words)
            if not does(row, verb):
                continue
            weight = min(about(row, patient, lemmas(verb)), 2.0)
            if weight < 1:
                continue
            best = max(best, confidence * weight / 2)
    return best


#: How many kinds of thing may be below a device before it is too general
#: to be one: `instrument` and `device` name thousands, a mower a handful.
SPECIFIC = 150


def _device(synset) -> bool:
    """A made thing one uses -- WordNet's instrumentality: a mower, a
    heater, a flatiron -- and a particular one, not `instrument`."""
    above = {one.name() for one in synset.closure(lambda s: s.hypernyms())}
    if "instrumentality.n.03" not in above:
        return False
    below = 0
    for _ in synset.closure(lambda s: s.hyponyms()):
        below += 1
        if below > SPECIFIC:
            return False
    return True


_NAMED: dict = {}


def named_tools(verb: str) -> list:
    """Tools the language itself names for a verb: the nouns WordNet
    derives it from or to (`mow` -> mower, `heat` -> heater, `shovel` ->
    shovel), and what its definition says it is done *with* (`iron`:
    *press and smooth with a heated iron*; `sweep`: *with a broom*). A
    made thing, either way. The verb's first two senses only."""
    if verb in _NAMED:
        return _NAMED[verb]
    out: list = []
    try:
        from nltk.corpus import wordnet
        for synset in wordnet.synsets(verb, wordnet.VERB)[:2]:
            for lemma in synset.lemmas():
                if lemma.name() != verb:
                    continue
                for other in lemma.derivationally_related_forms():
                    name = other.name().lower()
                    if other.synset().pos() == "n" and "_" not in name \
                            and _device(other.synset()) and name not in out:
                        out.append(name)
            words = synset.definition().lower().replace(",", " ").replace(
                ";", " ").split()
            for index, word in enumerate(words):
                if word != "with":
                    continue
                rest = [one for one in words[index + 1:index + 5]
                        if one not in ("a", "an", "the", "or", "as", "if")]
                for size in (2, 1):
                    for start in range(0, min(len(rest), 3)):
                        name = "_".join(rest[start:start + size])
                        if len(rest[start:start + size]) != size:
                            continue
                        base = wordnet.morphy(name, wordnet.NOUN) or name
                        if base not in out and any(
                                _device(one) for one in wordnet.synsets(
                                    base, wordnet.NOUN)[:4]):
                            out.append(base)
                            break
                    else:
                        continue
                    break
    except Exception:                              # noqa: BLE001
        pass
    _NAMED[verb] = [name_of(one) for one in out]
    return _NAMED[verb]


def found_in(word: str) -> str:
    """Where one of these is usually found, as a thing's name: the
    store's best `at_location` that is a place or a container -- a
    drawer, a kitchen -- not a backpack somebody once left one in."""
    for said, _ in rows_about(word, "at_location"):
        words = [one for one in said.lower().split() if one not in FILLER]
        if not words:
            continue
        place = words[-1]
        if len(words) >= 2 and first_sense("_".join(words[-2:])):
            place = "_".join(words[-2:])
        if is_a(place, "location.n.01", "structure.n.01", "container.n.01",
                "area.n.05", "furniture.n.01", "room.n.01"):
            return name_of(place)
    return ""


_KEPT: dict | None = None


def _kept() -> dict:
    """place word -> the words of things the store says are found there:
    `refrigerator` -> milk, beer, bacon, apple, ..."""
    global _KEPT
    if _KEPT is None:
        _KEPT = defaultdict(set)
        for concept, said in connection().execute(
                "SELECT concept, object FROM facts WHERE relation = "
                "'at_location'"):
            words = [one for one in said.lower().split()
                     if one not in FILLER]
            if not words:
                continue
            for place in {words[-1], "_".join(words[-2:])}:
                for form in lemmas(place):
                    _KEPT[form].add(word_of(concept))
    return _KEPT


def kept_at(thing: str, place: str) -> bool:
    """Whether the store says a thing like this one is found at `place`:
    milk in a refrigerator, a car in a garage. Things of a kind count for
    the kind, so what is kept in a fridge says bacon and beer and bread,
    and milk is a food like them only when the store says milk too, or
    food.

    The place's own word and its lemmas, so `refrigerator` finds what
    was said to be in one."""
    if not place:
        return False
    found = set()
    for form in lemmas(place.replace("-", "_")) | lemmas(
            place.replace("-", " ")):
        found |= _kept().get(form, set())
    if not found:
        return False
    kinds = kinds_of(thing)
    return any(one in kinds for one in found)


def a_place(word: str) -> int:
    """How many things the store says are found at `word`: how much of a
    place it is to put something."""
    total = 0
    for form in lemmas(word.replace("-", "_")):
        total = max(total, len(_kept().get(form, ())))
    return total


_MERONYMS: dict = {}


def meronyms(word: str) -> frozenset:
    """A thing's parts as WordNet lists them -- curated, unlike the
    store's `part_of` rows, which make a fire truck part of a roof. What
    decides whether a row is about the thing (`about`) has to be sure."""
    if word not in _MERONYMS:
        out = set()
        first = first_sense(word)
        if first is not None:
            for one in first.part_meronyms():
                out.update(name_of(lemma.lower())
                           for lemma in one.lemma_names())
        _MERONYMS[word] = frozenset(out)
    return _MERONYMS[word]


_PARTS: dict = {}


def has_parts(word: str) -> set:
    """The parts of a thing, as names: WordNet's part meronyms of its
    first sense and the store's `has_part` -- a door's lock and knob.
    Particular things only: the store also says a mammal is part of hair,
    and *part* is a part of everything."""
    if word not in _PARTS:
        _PARTS[word] = {one for one in _has_parts(word)
                        if not category(one) and not lives(one)}
    return _PARTS[word]


#: How many kinds of thing may be below a word before it names a category
#: rather than a thing: `kitchen utensil`, `part`, `tool` name dozens.
CATEGORY = 40

_CATEGORY: dict = {}


def category(word: str) -> bool:
    """Whether a word names a category -- dozens of kinds below it -- and
    not a thing one could get: *a kitchen utensil*, *a machine*."""
    if word not in _CATEGORY:
        first = first_sense(word)
        below = 0
        if first is not None:
            for _ in first.closure(lambda s: s.hyponyms()):
                below += 1
                if below > CATEGORY:
                    break
        _CATEGORY[word] = below > CATEGORY
    return _CATEGORY[word]


def _has_parts(word: str) -> set:
    out = set()
    first = first_sense(word)
    if first is not None:
        for one in first.part_meronyms():
            for lemma in one.lemma_names():
                out.add(name_of(lemma.lower()))
    for said, _ in rows_about(word, "has_part"):
        words = [one for one in said.lower().split() if one not in FILLER]
        if len(words) == 1:
            out.add(name_of(words[0]))
    # And the other way round: what the store says is *part of* one. A
    # knob is part of a door, and the store says it only that way.
    like = word.replace("-", " ")
    for concept, in connection().execute(
            "SELECT DISTINCT concept FROM facts WHERE relation = 'part_of' "
            "AND (object = ? OR object = ?)", (like, "a " + like)):
        out.add(name_of(concept))
    return out


def part_of_something_fixed(word: str) -> bool:
    """Whether WordNet says a thing is part of a room or a building -- a
    floor, a wall, a window: it is not carried anywhere."""
    first = first_sense(word)
    if first is None:
        return False
    for whole in first.part_holonyms():
        above = {one.name() for one in whole.closure(
            lambda s: s.hypernyms())} | {whole.name()}
        if above & {"structure.n.01", "room.n.01", "building.n.01",
                    "area.n.05"}:
            return True
    return False


def parts_of(word: str) -> list:
    """What one of these is made of, or has as parts, as names."""
    out = []
    for relation in ("made_of", "has_part"):
        for said, confidence in rows_about(word, relation):
            words = [one for one in said.lower().split()
                     if one not in FILLER]
            if len(words) == 1:
                out.append((name_of(words[0]), confidence, relation))
    return out

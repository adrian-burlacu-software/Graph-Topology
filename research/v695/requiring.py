"""What a goal requires of its means, and whether a thing has it.

Relatedness cannot choose between *make an ice pack with water* and *with
oil*: both are tied to ice packs, to freezing, to cold. What chooses is
what the goal needs of the thing, and what the thing is like:

    requirements   the properties of the goal's own things (an ice pack is
                   cold), what they are made of (ice, water), and the
                   properties shared by what the store says serves the
                   goal's doing (what crushes is hard and heavy)
    the thing      its properties, and itself if it is one of them
    against it     a property whose WordNet antonym is required: *hot*
                   water to soothe a sunburn that wants *cool*

A property is worth what the store is sure of, and nothing when it is
said of everything: `great`, `safe`, `useful` are evaluations, not what a
thing is like (`EVALUATIVE`). A requirement is what the means *agree* on
-- said of at least two of the things that serve the doing -- not
everything any one of them is: rarity alone was tried, and rewarded
`hermaphroditic` and `accusative`. Every score is bounded: a thing meets a
requirement or goes against it by at most the store's confidence.
"""
from __future__ import annotations

from research.v694 import knowing as K
from research.v695 import mined

#: How many of the means that serve a doing make its profile, and how
#: many of their properties are kept.
MEANS = 10
PROFILE = 15
#: How much a similar adjective counts against the same one.
SIMILAR = 0.5
#: A property said of more than this share of the store's concepts is an
#: evaluation: `great`, `safe`, `useful`.
EVALUATIVE = 0.01
#: How many of the means must share a property for it to be required.
AGREE = 2

_DF: dict | None = None
_CONCEPTS = 1


def _df() -> dict:
    """How many concepts each property is said of."""
    global _DF, _CONCEPTS
    if _DF is None:
        _DF = {}
        for said, count in K.connection().execute(
                "SELECT object, COUNT(DISTINCT concept) FROM facts WHERE "
                "relation = 'has_property' GROUP BY object"):
            _DF[said.lower()] = count
        _CONCEPTS = K.connection().execute(
            "SELECT COUNT(DISTINCT concept) FROM facts WHERE relation = "
            "'has_property'").fetchone()[0] or 1
    return _DF


def evaluative(prop: str) -> bool:
    """Said of so much that it says nothing of what a thing is like."""
    return _df().get(prop, 0) > EVALUATIVE * _CONCEPTS


def _adjective(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(word, wordnet.ADJ))
    except Exception:                              # noqa: BLE001
        return False


_PROPS: dict = {}


def properties(word: str) -> dict:
    """{property: confidence} the store says of a thing: one-word
    adjectives, not evaluations, confidence at most 1."""
    if word not in _PROPS:
        found: dict = {}
        for said, confidence in K.rows_about(word, "has_property"):
            said = said.lower().strip()
            if " " in said or not _adjective(said) or evaluative(said):
                continue
            found[said] = max(found.get(said, 0.0), min(confidence * 2, 1.0))
        _PROPS[word] = found
    return _PROPS[word]


def _made_of(word: str) -> set:
    out = set()
    for relation in ("made_of", "has_part"):
        for said, _ in K.rows_about(word, relation):
            words = [one for one in said.lower().split()
                     if one not in K.FILLER]
            if words:
                out.add(mined._morphy(words[-1], False))
    return out


_ANTONYMS: dict = {}


def antonyms(adjective: str) -> set:
    """WordNet's antonyms of an adjective, through the head of a satellite
    (*chilly* is similar to *cold*, whose antonym is *hot*)."""
    if adjective not in _ANTONYMS:
        out = set()
        try:
            from nltk.corpus import wordnet
            for synset in wordnet.synsets(adjective, wordnet.ADJ)[:3]:
                heads = [synset] + synset.similar_tos()
                for head in heads:
                    for lemma in head.lemmas():
                        for other in lemma.antonyms():
                            out.add(other.name().lower())
                            for near in other.synset().similar_tos():
                                out.update(name.lower() for name in
                                           near.lemma_names())
        except Exception:                          # noqa: BLE001
            pass
        out.discard(adjective)
        _ANTONYMS[adjective] = out
    return _ANTONYMS[adjective]


_SIMILAR: dict = {}


def similar(adjective: str) -> set:
    if adjective not in _SIMILAR:
        out = set()
        try:
            from nltk.corpus import wordnet
            for synset in wordnet.synsets(adjective, wordnet.ADJ)[:2]:
                for near in [synset] + synset.similar_tos():
                    out.update(name.lower() for name in near.lemma_names())
        except Exception:                          # noqa: BLE001
            pass
        out.discard(adjective)
        _SIMILAR[adjective] = out
    return _SIMILAR[adjective]


def _usable(word: str) -> bool:
    name = word.replace(" ", "_")
    return K.is_a(name, "physical_entity.n.01") and not K.lives(name)


_PROFILES: dict = {}


def _profile(verb: str, thing: str) -> dict:
    """The properties the means of doing `verb` to `thing` agree on: {prop:
    share of the means that have it, weighted by confidence}, for those at
    least `AGREE` of them have."""
    key = (verb, thing)
    if key not in _PROFILES:
        found: dict = {}
        count: dict = {}
        # What one uses: a made or natural thing, not someone -- people
        # attach things too, and what they are like is not what glue is.
        means = K.serving(verb=verb, patient=thing,
                          relations=("used_for", "capable_of"),
                          keep=_usable)[:MEANS]
        for one in means:
            for prop, value in properties(one.thing).items():
                found[prop] = found.get(prop, 0.0) + value
                count[prop] = count.get(prop, 0) + 1
        agreed = {prop: value / max(len(means), 1)
                  for prop, value in found.items() if count[prop] >= AGREE}
        top = max(agreed.values(), default=1.0)
        kept = sorted(agreed.items(), key=lambda item: -item[1])[:PROFILE]
        _PROFILES[key] = {prop: value / top for prop, value in kept}
    return _PROFILES[key]


def requirements(nouns, doings) -> tuple:
    """({property: weight}, {thing: weight}) a goal requires, each at most
    1: what its things are made of, and what its doings' means agree on.
    The goal's things' own properties are not requirements: a sunburn is
    painful, and what soothes it is not."""
    props: dict = {}
    things: dict = {}
    for noun in nouns:
        for part in _made_of(noun):
            things[part] = 1.0
    for verb, thing in doings:
        for prop, value in _profile(verb, thing).items():
            props[prop] = max(props.get(prop, 0.0), value)
    return props, things


def fit(word: str, props: dict, things: dict) -> float:
    """How well a thing (or an adjective said of the way) meets what is
    required, less what it has against it."""
    if word in things:
        return things[word]
    mine = dict(properties(word))
    if _adjective(word) and not evaluative(word):
        mine[word] = 1.0
    score = 0.0
    for prop, value in mine.items():
        if prop in props:
            score += min(value, props[prop])
        else:
            near = similar(prop) & props.keys()
            if near:
                score += SIMILAR * min(value, max(props[one] for one in
                                                  near))
        against = antonyms(prop) & props.keys()
        if against:
            score -= min(value, max(props[one] for one in against))
    # Bounded: however many properties a thing has, it meets or goes
    # against what is required by at most as much as one sure one.
    return max(-1.0, min(1.0, score))

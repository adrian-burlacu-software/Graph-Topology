"""PIQA as a benchmark of what the graph knows about doing things.

PIQA (Bisk et al., AAAI 2020) puts a goal and two ways to reach it, one of
which works: *to get over a fence* -- climb it, or tunnel under it with a
spoon. Humans choose right 95% of the time. Nothing here is trained on it:
the train split is where the scorer was looked at while it was built, and
the dev split is run once, at the end, as the number.

**How the graph chooses.** The two ways are nearly the same sentence, so
what decides is where they differ: the words only one of them has. Each is
asked how well the knowledge ties it to the goal and to what both ways
share -- *paper strips* to *bedding* and *cage*, *jeans* to nothing -- by

    store    the store's rows: a word's own rows mentioning a context
             word, or a context word's rows mentioning it
    actions  ConceptNet's doings, kept as said (`mined.py`): an edge with
             the word at one end and a context word at the other
    design   what the designer knows (`v694.knowing.serving`): how well
             the store says the word serves the doing the goal and the
             ways name -- a mortar and pestle crush, a ruler does not tie
    can      a way whose action nobody can do, as `can.py` reads it --
             a person does not fly -- counts against it
    require  what the goal requires of its means, and whether the word
             has it (`requiring.py`): water is what an ice pack is made
             of; hot is against the cool a sunburn wants

The side with more is chosen; with none on either, it abstains, and an
abstention counts half, as a coin would.

Summed, the sources drown each other: their scales are not the same, and
what a goal requires is a stronger claim than what a word is related to.
So they are also asked in turn (`--cascade`): the most specific that
chooses at all decides -- requirements, then serving, then relatedness.

    python -m research.v695.piqa --split train --limit 500
    python -m research.v695.piqa --split valid --sources store,actions,can
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from research.v694 import knowing as K
from research.v695 import mined

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "piqa"

SOURCES = ("store", "actions", "design", "can", "require")

#: Words that say nothing of what is done: English's small words, and the
#: ones PIQA's ways are phrased with.
STOP = K.FILLER | frozenset("""i you we they he she it me him them us my
your our its be is are was were been being do does did have has had can
could will would should may might must not no yes if then than so just
also into onto out up down over off about after before while until again
very more most much many few some any each every all both either neither
other another such own same only too really how what which who whom when
where why this that these those there here get got make made put take use
used using way want need go going one two""".split())

#: How much a word's rows may count toward one pair: many phrasings are
#: more evidence than one, and not many times more.
MOST = 3


@dataclass
class Item:
    goal: str
    sol1: str
    sol2: str
    label: int


def load(split: str, limit: int = 0) -> list:
    items = []
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle, \
            (DATA / f"{split}-labels.lst").open(encoding="utf-8") as labels:
        for line, label in zip(handle, labels):
            one = json.loads(line)
            items.append(Item(one["goal"], one["sol1"], one["sol2"],
                              int(label.strip())))
            if limit and len(items) >= limit:
                break
    return items


_WORD = re.compile(r"[a-z][a-z'-]*")


def words(text: str) -> list:
    """Content words, each as WordNet's lemma: `ripped paper strips` ->
    rip, paper, strip."""
    out = []
    for raw in _WORD.findall(text.lower()):
        raw = raw.strip("'-")
        if len(raw) < 3 or raw in STOP:
            continue
        lemma = mined._morphy(raw, False)
        verb = K.verb_of(raw)
        if verb and verb != lemma and not K.first_sense(lemma):
            lemma = verb
        if lemma not in STOP:
            out.append(lemma)
    return out


# -- what the knowledge says of a word -------------------------------------

_ROWS: dict = {}


def _store_rows(word: str) -> list:
    """(relation, lemmas of what it says, confidence) of the store's rows
    about a word, as a noun or a verb."""
    if word not in _ROWS:
        found = []
        for concept, relation, said, confidence in K.connection().execute(
                "SELECT concept, relation, object, confidence FROM facts "
                "WHERE concept = ? OR (concept >= ? AND concept < ?) OR "
                "(concept >= ? AND concept < ?)",
                (word, word + ".n.", word + ".n/", word + ".v.",
                 word + ".v/")):
            found.append((relation, frozenset(words(said)), confidence))
        _ROWS[word] = found
    return _ROWS[word]


_EDGES: dict = {}


def _action_edges(word: str) -> list:
    """(relation, lemmas at the other end, weight) of ConceptNet's doings
    with the word at one end."""
    if word not in _EDGES:
        found = []
        for side, other in (("head", 2), ("tail", 0)):
            for row in mined.mentioning(word, side):
                found.append((row[1], frozenset(words(row[other])),
                              min(float(row[3]), 2.0) / 2))
        _EDGES[word] = found
    return _EDGES[word]


def tie(word: str, context: set, sources) -> float:
    """How well the knowledge ties a word to the context: for each context
    word, the best row that joins them, summed."""
    best: dict = {}

    def seen(other: str, weight: float) -> None:
        best[other] = max(best.get(other, 0.0), weight)

    if "store" in sources:
        for _, said, confidence in _store_rows(word):
            for other in said & context:
                seen(other, confidence)
        for other in context:
            for _, said, confidence in _store_rows(other):
                if word in said:
                    seen(other, confidence)
    if "actions" in sources:
        for _, said, weight in _action_edges(word):
            for other in said & context:
                seen(other, weight)
    return sum(sorted(best.values(), reverse=True)[:MOST])


#: What meeting the goal's requirements counts, against relatedness.
REQUIRES = 1.0

#: What a word serving the doing counts, against a row that only ties it
#: to the context: serving is the stronger claim.
SERVES = 2.0

_DOINGS: dict = {}


def doings(text: str) -> list:
    """(verb, thing done to) for each doing a text names, off the parse:
    `to crush the petals of a flower` -> (crush, petal)."""
    if text not in _DOINGS:
        from research.v691 import hearing
        found = []
        parsed = hearing.parse(text)
        for word in parsed:
            if not word.tag.startswith("VB"):
                continue
            verb = word.lemma
            if verb in STOP or len(verb) < 3:
                continue
            thing = next((mined._morphy(one.text, False) for one in
                          hearing.children(parsed, word.index, {"dobj"})),
                         "")
            found.append((verb, thing))
        _DOINGS[text] = found
    return _DOINGS[text]


_SERVING: dict = {}


def _served(verb: str, thing: str) -> dict:
    """{thing's name: how well it serves doing `verb` to `thing`}."""
    key = (verb, thing)
    if key not in _SERVING:
        _SERVING[key] = {means.name: means.score for means in K.serving(
            verb=verb, patient=thing,
            relations=("used_for", "capable_of"))}
    return _SERVING[key]


def serves(word: str, doing: list, sources) -> float:
    """How well the store says a word serves any of these doings."""
    if "design" not in sources:
        return 0.0
    name = K.name_of(word)
    return max((_served(verb, thing).get(name, 0.0)
                for verb, thing in doing), default=0.0) * SERVES


def unable(text: str, sources) -> float:
    """How much a way asks of a person that nothing says a person can do:
    its verbs VerbNet admits no person for (`can.admits`)."""
    if "can" not in sources:
        return 0.0
    from research.v695 import can
    against = 0.0
    for raw in _WORD.findall(text.lower()):
        verb = K.verb_of(raw)
        if not verb or verb in STOP or K.first_sense(raw):
            continue
        if not can.admits("person", verb)[0]:
            against += 1.0
    return against


def required(item: Item, only_one, only_two, doing, sources) -> tuple:
    """(fit of sol1's own words, of sol2's) to what the goal requires."""
    if "require" not in sources:
        return 0.0, 0.0
    from research.v695 import requiring
    shared = set(words(item.sol1)) & set(words(item.sol2))
    nouns = [one for one in set(words(item.goal)) | shared
             if K.first_sense(one) is not None]
    props, things = requiring.requirements(nouns, doing)
    if not props and not things:
        return 0.0, 0.0
    return (sum(requiring.fit(one, props, things) for one in only_one),
            sum(requiring.fit(one, props, things) for one in only_two))


def score(item: Item, sources=SOURCES) -> tuple:
    """(choice, score of sol1, score of sol2): 0 or 1, or None to abstain."""
    one, two = words(item.sol1), words(item.sol2)
    only_one, only_two = set(one) - set(two), set(two) - set(one)
    context = set(words(item.goal)) | (set(one) & set(two))
    doing = doings(item.goal) + [
        one for one in doings(item.sol1) if one in doings(item.sol2)]
    first = sum(tie(word, context, sources) + serves(word, doing, sources)
                for word in only_one) - unable(item.sol1, sources)
    second = sum(tie(word, context, sources) + serves(word, doing, sources)
                 for word in only_two) - unable(item.sol2, sources)
    fit_one, fit_two = required(item, only_one, only_two, doing, sources)
    first += REQUIRES * fit_one
    second += REQUIRES * fit_two
    if abs(first - second) < 1e-9:
        return None, first, second
    return (0 if first > second else 1), first, second


@dataclass
class Result:
    total: int = 0
    answered: int = 0
    right: int = 0

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.right / self.answered if self.answered else 0.0

    @property
    def overall(self) -> float:
        """Right, with each abstention counted half."""
        return ((self.right + (self.total - self.answered) / 2) / self.total
                if self.total else 0.0)

    def line(self, name: str) -> str:
        return (f"{name:28} n={self.total:5}  answered {self.coverage:6.1%}"
                f"  right when answered {self.precision:6.1%}"
                f"  overall {self.overall:6.1%}")


#: The order sources are asked in, most specific first.
CASCADE = (("require",), ("design",), ("store", "actions"))


def cascade(item: Item, order=CASCADE) -> tuple:
    """The first source, in `order`, that chooses at all."""
    for sources in order:
        found = score(item, sources)
        if found[0] is not None:
            return found
    return None, 0.0, 0.0


def run(items: list, sources=SOURCES, cascaded: bool = False) -> Result:
    result = Result()
    for item in items:
        choice, _, _ = (cascade(item) if cascaded
                        else score(item, sources))
        result.total += 1
        if choice is None:
            continue
        result.answered += 1
        result.right += int(choice == item.label)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="train",
                        choices=("train", "valid"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--ablate", action="store_true",
                        help="each source alone, then all of them")
    parser.add_argument("--cascade", action="store_true",
                        help="the most specific source that chooses")
    options = parser.parse_args(argv)
    items = load(options.split, options.limit)
    if options.ablate:
        for sources in (("store",), ("actions",), ("design",),
                        ("require",), ("store", "actions"),
                        ("store", "actions", "design", "can"), SOURCES):
            print(run(items, sources).line("+".join(sources)), flush=True)
        print(run(items, cascaded=True).line("cascade"), flush=True)
        return 0
    if options.cascade:
        print(run(items, cascaded=True).line("cascade"))
        return 0
    sources = tuple(one for one in options.sources.split(",") if one)
    print(run(items, sources).line("+".join(sources)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

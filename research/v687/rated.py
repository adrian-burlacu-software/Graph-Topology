"""Rated norms: objects people scored, read as a yes, a no, or nothing.

XCSLB is a free listing, so its silence is what nobody happened to say. AwA2
scored every class on every attribute, so its zeros are judgements -- and it
covers 50 animals. Two sources shaped like AwA2 and far wider:

    THINGSplus   1,854 objects, each rated 1-7 by about 40 people on whether
                 it is something that lives, manmade, natural and heavy; its
                 real-world size, on a scale anchored from a grain of sand to
                 an aircraft carrier; and which of 53 everyday categories it
                 belongs to.
    NEWTON       777 household objects voted Low or High on softness,
                 sharpness, brittleness, elasticity, malleability, stiffness,
                 surface smoothness and surface hardness.

Every object was put the same question, so a low rating is an answer and not a
silence. That is what the crawl cannot give: `is a chair alive` and `does a
rock breathe` were UNKNOWN because nothing anywhere says a chair is not alive.

## Why these are not merged into the norms

`identify.stated` is what the trie is built from, and it is also what R19
counts. `Profiles.corroboration` takes every norm-covered kind beneath an
ancestor as its denominator -- `bird.n.01 capable_of fly` is 28 of 29 kinds.
Merging 2,000 rated objects would put THINGSplus's birds in that denominator,
and they were never asked about flying: fly would drop to 28 of ~50, under the
0.6 floor, and `do all birds fly` would list them as exceptions. Silence in a
source counts only for the questions that source asked.

So the ratings are their own layer, keyed by synset, and consulted only for the
predicates they rated -- beside AwA2's zeros in `Profiles.verify_one`, and for
subjects the norms do not cover in `reasoning._rated`.

## The join

THINGS names its objects with WordNet 3.0 synsets, and the store's ids are not
WordNet 3.0's (`badger.n.02` is the animal), so objects are joined by gloss:
1,790 of 1,854 have exactly one noun concept with the same definition. The 64
left have glosses from Google or Wikipedia and are dropped.

NEWTON names its objects with LVIS's categories, which carry WordNet 3.0
glosses: 484 join that way. The rest are joined to the one sense of the word
that is a physical thing, or to the first sense when that is one; 32 of 777
have neither and are dropped. `mouse` is `mouse.n.04`, the device, because
NEWTON's objects are household objects.

## The thresholds, and what they were checked against

The ratings are 1-7. A yes needs 5.5 and a no needs 2.5, measured against the
taxonomy on the 1,790 joined objects:

    lives <= 2.5 (not alive)    1,392 of 1,569 outside organism.n.01; 1 under
                                it (a honeypot, which is the ant)
    lives >= 5.5 (alive)        214 of 221 under organism.n.01; 51 outside,
                                and those are leaves, roots, fingers and fruit
    manmade <= 2.5              380 of 681 outside artifact.n.01; 6 under it
                                (a pearl, a stick, a ruby)

Size is AwA2's `big` in the other direction: its 20 big animals rate 281-376
and its 13 others 186-295, so `big` needs 320 and is denied at 240 or below,
and `small` needs 220 and is denied at 300 or above. NEWTON agrees with XCSLB
wherever both speak: of the 140 objects both cover, not one that XCSLB lists
as sharp, flexible or rigid was voted Low on it.
"""
from __future__ import annotations

import collections
import sqlite3
from dataclasses import asdict, dataclass

from . import corpora
from .identify import Identifier

#: A rating at or above the first is a yes; at or below the second, a no.
YES, NO = 5.5, 2.5

#: THINGSplus's 1-7 properties: (column, how the question was put, predicates).
SCALES = (
    ("lives", "as something that lives", ("is alive", "is living")),
    ("manmade", "as manmade", ("is man-made", "is manmade", "is man made")),
    ("natural", "as natural", ("is natural",)),
)

#: Heaviness has two ends, and `light` is only said of the light end.
HEAVY, LIGHT = 5.5, 2.0

#: The size scale. Where THINGSplus's own reference objects fall on it, so a
#: number can be read.
BIG, NOT_BIG = 320.0, 240.0
SMALL, NOT_SMALL = 220.0, 300.0
TINY = 150.0
ANCHORS = ("a grain of sand at 109, an egg at 181, a microwave at 260, a bed "
           "at 329 and an aircraft carrier at 421")

#: THINGSplus categories that are not said with an article.
MASS = frozenset("""
food clothing hardware seafood candy footwear headwear outerwear jewelry
lighting
""".split()) | frozenset({
    "home decor", "sports equipment", "construction equipment",
    "protective clothing", "safety equipment", "medical equipment",
    "scientific equipment", "women's clothing"})

#: NEWTON: attribute -> (held when High, denied when High, held when Low,
#: denied when Low), and what the High and Low options said.
NEWTON = {
    "softness": (("is soft",), ("is hard",), ("is hard",), ("is soft",),
                 "the majority of the object is soft", "object is hard"),
    "sharpness": (("is sharp",), ("is blunt",), ("is blunt",), ("is sharp",),
                  "object is designed to pierce",
                  "no sharp corners or edges, and not capable of piercing"),
    "brittleness": (("is fragile", "is brittle", "is breakable"), (), (),
                    ("is fragile", "is brittle", "is breakable"),
                    "shatters easily with impact force",
                    "can withstand most impact forces"),
    "elasticity": (("is elastic", "is springy"), (), (),
                   ("is elastic", "is springy"),
                   "recovers near perfectly to its original form",
                   "will not return to its original form"),
    "malleability": (("is malleable",), (), (), ("is malleable",),
                     "can be reshaped to most arbitrary forms",
                     "cannot be reshaped"),
    "stiffness": (("is sturdy",), ("is flimsy",), ("is flimsy",),
                  ("is sturdy",),
                  "can easily withstand more than a 10 kg weight",
                  "can support very minimal weight before deforming"),
    "surface smoothness": (("is smooth",), ("is rough",), ("is rough",),
                           ("is smooth",), "the majority of its surface is "
                           "smooth", "very rough, or has many bumps"),
    "surface hardness": (("is hard",), (), (), ("is hard",),
                         "does not get scratched or bruised easily",
                         "can be easily scratched or bruised"),
}

#: What a NEWTON object can be. Its objects are household objects, so a word
#: is joined to the sense of it that is one of these.
PHYSICAL = ("artifact.n.01", "food.n.01", "food.n.02", "plant part.n.01",
            "natural object.n.01", "substance.n.01", "solid.n.01")
ORGANISM = "organism.n.01"

#: How far apart two readings of one word may rate and still be one answer.
SIZE_TIE, HEAVY_TIE = 20.0, 1.0

#: How many of a word's senses a rating may be borrowed from when the sense
#: the engine chose was not rated.
FALLBACK = 3


def with_article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


@dataclass
class Rated:
    """One answer from the ratings: a yes or a no, and whose."""

    concept: str
    term: str
    verdict: str              # HELD | DENIED
    predicate: str
    source: str               # thingsplus | newton | verbnet
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


class Ratings:
    """Both sources joined to the store, as held and denied predicates.

    `empty` builds none of it, which is the ablation: every reader treats an
    empty `Ratings` as "nothing rated".
    """

    def __init__(self, reasoner, empty: bool = False) -> None:
        self.reasoner = reasoner
        #: concept -> predicate -> (source, what the rating was)
        self.held: dict[str, dict[str, tuple[str, str]]] = {}
        self.denied: dict[str, dict[str, tuple[str, str]]] = {}
        #: concept -> (mean, usual range start, usual range end)
        self.size: dict[str, tuple[float, float, float]] = {}
        self.heavy: dict[str, float] = {}
        #: concept -> category -> typicality, where THINGSplus measured it
        self.categories: dict[str, dict[str, float | None]] = {}
        #: concept -> the word it was rated under, for notes
        self.named: dict[str, str] = {}
        self.report: collections.Counter = collections.Counter()
        self._parents: dict[str, set[str]] | None = None
        self._above: dict[str, frozenset[str]] = {}
        self.animate_only: dict[str, tuple[str, ...]] = {}
        if empty:
            return
        self.animate_only = corpora.load_animate_only_verbs()
        try:
            glosses = self._glosses()
            self._read_things(glosses)
            self._read_newton(glosses)
        except sqlite3.Error as missing:
            # A store without glosses or a taxonomy -- the test fixtures build
            # nine concepts -- has nothing to join ratings to.
            self.held.clear()
            self.denied.clear()
            self.report[f"not joined: {missing}"] += 1
        self._settle()

    # -- building ------------------------------------------------------------
    def _glosses(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = collections.defaultdict(list)
        for row in self.reasoner.connection.execute(
                "SELECT id, definition FROM concepts WHERE pos = 'n'"):
            if row[1]:
                out[row[1].strip().lower()].append(row[0])
        return out

    def ancestors(self, concept: str) -> frozenset[str]:
        """The concept and everything above it, from the taxonomy table."""
        if self._parents is None:
            parents: dict[str, set[str]] = collections.defaultdict(set)
            for row in self.reasoner.connection.execute(
                    "SELECT child, parent FROM taxonomy"):
                parents[row[0]].add(row[1])
            self._parents = parents
        if concept not in self._above:
            seen, todo = {concept}, [concept]
            while todo:
                for parent in self._parents.get(todo.pop(), ()):
                    if parent not in seen:
                        seen.add(parent)
                        todo.append(parent)
            self._above[concept] = frozenset(seen)
        return self._above[concept]

    def living(self, concept: str) -> bool:
        return ORGANISM in self.ancestors(concept)

    def _say(self, concept: str, predicates, yes: bool, source: str,
             why: str) -> None:
        table = self.held if yes else self.denied
        for predicate in predicates:
            table.setdefault(concept, {}).setdefault(predicate, (source, why))

    def _read_things(self, glosses) -> None:
        for row in corpora.load_thingsplus():
            found = (glosses.get(row["definition"].lower(), [])
                     if row["definition"] else [])
            if len(found) != 1:
                self.report["thingsplus: no single gloss in the store"] += 1
                continue
            concept, word = found[0], row["word"]
            self.report["thingsplus: joined"] += 1
            self.named.setdefault(concept, word)
            said = f"THINGSplus: people rated “{word}”"
            for column, asked, predicates in SCALES:
                value = row[column]
                if value is None:
                    continue
                why = f"{said} {value:.1f} of 7 {asked}"
                if value >= YES:
                    self._say(concept, predicates, True, "thingsplus", why)
                elif value <= NO:
                    self._say(concept, predicates, False, "thingsplus", why)
            heavy = row["heavy"]
            if heavy is not None:
                self.heavy[concept] = heavy
                why = f"{said} {heavy:.1f} of 7 on how heavy it is"
                if heavy >= HEAVY:
                    self._say(concept, ("is heavy",), True, "thingsplus", why)
                    self._say(concept, ("is light", "is lightweight"), False,
                              "thingsplus", why)
                elif heavy <= LIGHT:
                    self._say(concept, ("is light", "is lightweight"), True,
                              "thingsplus", why)
                    self._say(concept, ("is heavy",), False, "thingsplus", why)
            size = row["size"]
            if size is not None:
                self.size[concept] = (size, row["size_start"] or size,
                                      row["size_end"] or size)
                why = (f"THINGSplus places “{word}” at {size:.0f} on a "
                       f"real-world size scale with {ANCHORS}")
                if size >= BIG:
                    self._say(concept, ("is big", "is large"), True,
                              "thingsplus", why)
                if size >= NOT_SMALL:
                    self._say(concept, ("is small", "is little", "is tiny"),
                              False, "thingsplus", why)
                if size <= SMALL:
                    self._say(concept, ("is small", "is little"), True,
                              "thingsplus", why)
                if size <= TINY:
                    self._say(concept, ("is tiny",), True, "thingsplus", why)
                if size <= NOT_BIG:
                    self._say(concept, ("is big", "is large", "is huge"),
                              False, "thingsplus", why)
            for category, typicality in sorted(row["categories"].items()):
                self.categories.setdefault(concept, {})[category] = typicality
                phrase = (f"is {category}" if category in MASS
                          else f"is {with_article(category)}")
                self._say(concept, (phrase,), True, "thingsplus",
                          f"THINGSplus: people sorted “{word}” under "
                          f"“{category}”")

    def _newton_concept(self, name: str, lvis, glosses) -> str | None:
        for gloss in lvis.get(name, []):
            found = glosses.get(gloss.strip().lower(), [])
            if len(found) == 1:
                self.report["newton: joined by LVIS gloss"] += 1
                return found[0]
        # The senses in the order the engine offers them, which is what "the
        # first sense" means everywhere else here.
        senses = [sense["id"] for sense in self.reasoner.senses_of(name)
                  if sense.get("pos") == "n"]
        physical = [sense for sense in senses
                    if self.ancestors(sense) & set(PHYSICAL)]
        if len(physical) == 1 or (physical and senses[0] == physical[0]):
            self.report["newton: joined by its physical sense"] += 1
            return physical[0]
        self.report["newton: no single physical sense"] += 1
        return None

    def _read_newton(self, glosses) -> None:
        lvis = corpora.load_lvis_categories()
        for name, votes in sorted(corpora.load_newton().items()):
            concept = self._newton_concept(name, lvis, glosses)
            if concept is None:
                continue
            self.named.setdefault(concept, name)
            for attribute, (majority, agreement) in sorted(votes.items()):
                entry = NEWTON.get(attribute)
                if entry is None:
                    continue
                high = majority == 3
                held, denied = ((entry[0], entry[1]) if high
                                else (entry[2], entry[3]))
                why = (f"NEWTON: annotators voted “{name}” "
                       f"{'High' if high else 'Low'} on {attribute} -- "
                       f"“{entry[4] if high else entry[5]}” -- with "
                       f"{agreement:.0%} agreement")
                self._say(concept, held, True, "newton", why)
                self._say(concept, denied, False, "newton", why)

    def _settle(self) -> None:
        """A predicate held and denied of one concept is neither."""
        for concept in sorted(set(self.held) & set(self.denied)):
            clash = set(self.held[concept]) & set(self.denied[concept])
            for predicate in clash:
                del self.held[concept][predicate]
                del self.denied[concept][predicate]
                self.report["held and denied by the sources, so neither"] += 1

    # -- reading -------------------------------------------------------------
    def answer(self, concept: str | None, term: str) -> Rated | None:
        """The rating that bears on one term, or None."""
        if not concept or not (term or "").strip():
            return None
        wanted = term.strip().lower()
        for verdict, table in (("HELD", self.held), ("DENIED", self.denied)):
            rated = table.get(concept)
            if not rated:
                continue
            hit = Identifier._hit(wanted, frozenset(rated))
            if hit is not None:
                source, why = rated[hit]
                return Rated(concept, term, verdict, hit, source, why)
        return None

    def settle(self, chosen: str, senses: list[str], term: str) -> Rated | None:
        """The rating of the sense chosen, unless another reading disagrees.

        THINGS rates `mouse` twice, the animal alive and the device not, and
        the store's first sense of `mouse` is the device. A no about the
        device read as `a mouse is not alive` is the answer about something
        else this codebase refuses everywhere, so when rated readings of one
        word disagree nothing is said.

        When the sense chosen was not rated, a rated one among the word's
        first few is taken -- `rock.n.01`, the material, for `rock.n.02`, the
        lump of it -- but never across the line between living things and the
        rest: the store's first `chicken` is the bird and THINGSplus's is the
        meat, and the meat is not alive.
        """
        found = self.answer(chosen, term)
        if found is None:
            side = self.living(chosen)
            for other in senses[:FALLBACK]:
                if other == chosen or self.living(other) != side:
                    continue
                found = self.answer(other, term)
                if found is not None:
                    break
        if found is None:
            return None
        for other in senses:
            if other == found.concept:
                continue
            rival = self.answer(other, term)
            if rival is not None and rival.verdict != found.verdict:
                return None
        return found

    def _verb(self, word: str) -> tuple[str, tuple[str, ...]]:
        forms = [word]
        if word.endswith("es"):
            forms.append(word[:-2])
        if word.endswith("s"):
            forms.append(word[:-1])
        for form in forms:
            if form in self.animate_only:
                return form, self.animate_only[form]
        return word, ()

    def asserted(self, concept: str, target: str) -> bool:
        """Does the store say this thing does exactly this?"""
        wanted = {Identifier.stem(word) for word in target.lower().split()}
        for row in self.reasoner.connection.execute(
                "SELECT object FROM facts WHERE concept = ? AND "
                "relation = 'capable_of'", (concept,)):
            if {Identifier.stem(word)
                    for word in (row[0] or "").lower().split()} == wanted:
                return True
        return False

    def unable(self, chosen: str, senses: list[str],
               target: str) -> Rated | None:
        """R32: something rated not alive, asked to do what only the living do.

        VerbNet says who can be the subject of a verb, and for `breathe`,
        `eat`, `drink` and `think` it is an animate doer in every class the
        verb is in. THINGSplus says whether a thing lives. Neither says a rock
        does not breathe, and together they do.

        It stands down in two places. When any rated reading of the word is
        alive, or the sense chosen is a living thing, the no could be about
        the wrong one -- `mouse` is the device first and the animal second.
        And when the store says this thing does exactly this, the test is the
        one AwA2's zeros face in `Profiles._zero_that_is_not_a_no`:
        disagreement, not overruling. `computer capable_of talk` and
        `automaton capable_of think` are in the store, so neither is denied.
        """
        words = (target or "").lower().split()
        # The verb alone. VerbNet restricts who does a verb, and an object can
        # choose a sense of it VerbNet does not list: `can a cup hold water`
        # is holding as containing, and `hold` is in VerbNet's grasping class
        # only. It also keeps R32 off a word the norms' conjunction split out
        # of a question about something else -- `can dogs eat chocolate` was
        # routed to chocolate and asked whether chocolate eats.
        if len(words) != 1 or not chosen:
            return None
        verb, classes = self._verb(words[0])
        if not classes or self.living(chosen):
            return None
        if any("is alive" in (self.held.get(sense) or {})
               for sense in [chosen, *senses]):
            return None
        rated = None
        for sense in [chosen, *senses[:FALLBACK]]:
            if not self.living(sense) and "is alive" in (
                    self.denied.get(sense) or {}):
                rated = sense
                break
        if rated is None:
            return None
        if self.asserted(chosen, target) or self.asserted(rated, target):
            return None
        denial = self.denied[rated]["is alive"]
        name = self.named.get(rated, rated.rsplit(".", 2)[0])
        shown = ", ".join(classes[:3])
        return Rated(
            rated, target, "DENIED", f"cannot {' '.join(words)}", "verbnet",
            f"VerbNet gives “{verb}” a living doer in every class it is in "
            f"({shown}), and {denial[1]}: {with_article(name)} does not live, "
            f"so it does not {verb}")

    def magnitude(self, dimension: str, chosen: str,
                  senses: list[str]) -> str | None:
        """The rated reading of a word on one scale.

        The sense chosen when it was rated, as every other answer here is
        about the sense chosen; otherwise a rated one among the word's first
        few, on the same side of alive. Every reading was once required to
        agree, and `horse` has a pommel horse and `cat` a Caterpillar tractor
        rated beside the animal, so `is a cat bigger than a horse` was refused.
        """
        table = self.size if dimension == "size" else self.heavy
        if chosen in table:
            return chosen
        side = self.living(chosen)
        for sense in senses[:FALLBACK]:
            if sense in table and self.living(sense) == side:
                return sense
        return None

    def summary(self) -> dict:
        return {
            "held": sum(len(one) for one in self.held.values()),
            "denied": sum(len(one) for one in self.denied.values()),
            "concepts": len(set(self.held) | set(self.denied)),
            "sized": len(self.size), "weighed": len(self.heavy),
            "categorised": len(self.categories),
            "animate_only_verbs": len(self.animate_only),
            **dict(self.report),
        }

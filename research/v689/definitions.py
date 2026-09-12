"""Definitions memory: what a WordNet gloss says about a kind, read as facts.

A gloss is not a sentence. It is a noun phrase, often with fragments after
semicolons:

    young domestic cat
    one of the two male reproductive glands that produce spermatozoa and ...
    a hand tool with a heavy rigid head and a handle; used to deliver ...
    found in tropical coastal regions of Africa and Asia; able to move on land

It names a broader kind -- the genus -- and says what sets this one apart.
This reads both from the dependency parse, one piece at a time and each piece
inside a sentence frame, because spaCy parses `it is a small and light boat`
correctly and, given the whole gloss bare, made `boat` the subject of
`pointed`.

    genus          the noun after `it is a`, through `one of`, `a kind of`,
                   `a breed of`, `female of`
    adjective      what modifies it            has_property young
    with / having  what it has                 has_a heavy rigid head
    that ...       a clause about it           capable_of produce spermatozoa
    used to ...                                used_for deliver an impulsive force
    able to ...                                capable_of move on land
    no ability to                              not_capable_of roar
    found in ...                               at_location tropical coastal regions
    a participle   with by, with or as         receives_action propelled with a paddle
                   without                     has_property pointed at both ends

Alternatives are not properties: `red or yellow or green skin` is skin.
Clauses are split before they are mapped, by the parse (`clauses.py`) and,
where the tagger calls a verb a noun -- `produce spermatozoa and secrete
androgens` -- by WordNet, at an `and` before a word that is only ever a verb.

**The genus is checked against the store's own taxonomy.** When a sense of
the word taken for the broader kind is one of the concept's ancestors, the
parse found the head it should have. When it is not -- `young domestic cat`
is WordNet's kitten, filed under young mammal and not under cat -- the facts
are still read, and marked, so that the teacher can be asked about them and
the audit can say whether the mark means anything.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from research.v687.reason import Fact

from . import clauses as coordination

#: The source column of a fact read out of a definition.
DEFINED = "defined"

#: What a defined fact is worth to the rules: curated text, read by a parser.
#: Below WordNet's own relations (0.95), because the parser can be wrong.
CONFIDENCE = 0.9

#: Heads that point through `of` to the real one: `one of the glands`.
PARTITIVES = frozenset({"one", "any", "member", "members", "piece", "kind",
                        "type", "sort", "breed", "variety", "form", "species",
                        "specimen", "individual", "unit"})

#: Heads that point through `of` and are a property besides: `female of
#: domestic cattle` is cattle, and female.
SEXES = frozenset({"female", "male"})

#: What `of` names when it is a taxon rather than a kind: `a member of the
#: genus Canis`. The broader kind is then not in the gloss at all.
TAXA = frozenset({"genus", "family", "order", "class", "phylum", "subfamily",
                  "suborder", "tribe", "division", "subclass", "superfamily"})

#: Heads that gather a kind: `a family of box-shaped musical instruments`.
GROUPS = frozenset({"group", "set", "collection", "range"})

#: Modifiers that say how many or how sure, not what it is like.
QUANTIFIERS = frozenset({"various", "numerous", "several", "many", "any",
                         "other", "certain", "some", "different", "two",
                         "three", "especially", "mostly", "usually", "often",
                         "typically", "generally", "chiefly", "sometimes",
                         "normally", "commonly", "esp", "etc"})

#: Dropped from a noun phrase. `no` is not one of them: `no ability` is a
#: denial, and R3 reads a denial written into the object.
DETERMINERS = frozenset({"a", "an", "the", "this", "these", "that", "those",
                         "its", "their", "his", "her"})

#: Participles that mean what follows is had: `characterized by feathers`.
HAVING = frozenset({"characterized", "characterised", "distinguished",
                    "marked"})

#: A fragment that says where the thing is.
LOCATED = frozenset({"found", "native", "living", "growing", "occurring",
                     "inhabiting"})

#: A fact longer than this is a sentence the reader did not understand.
LONGEST = 10

#: Verbs that say nothing a question could ask: `occurs in many breeds`.
EMPTY_VERBS = frozenset({"occur", "include", "exist", "belong", "comprise",
                         "resemble", "consist", "contain", "refer"})

#: Participles that report what someone thinks of it, not what it is:
#: `considered one of the great Spanish writers`.
REPORTED = frozenset({"considered", "called", "known", "regarded", "named",
                      "thought", "believed", "said", "reputed", "deemed"})

#: A fact with one of these in it is a condition the reader flattened:
#: `realized when the asset is sold`.
SUBORDINATE = frozenset({"when", "if", "because", "whereas", "while",
                         "although", "unless", "whenever"})

#: Openings of a clause that is not about the thing: `takes place as ...`.
EMPTY_OPENINGS = ("take place", "give rise")

#: What can be left dangling at either end of a phrase once a coordinated
#: part has been split off it: `hoisting in wells or`.
DANGLING = frozenset({"and", "or", "but", "nor", ",", "as", "of", "to", "in",
                      "for", "with", "by"})


@dataclass
class Defined:
    """One fact read out of a gloss."""

    relation: str
    object: str
    rule: str            # what in the gloss it was read from
    piece: str           # the part of the gloss it came from

    def as_dict(self) -> dict:
        return {"relation": self.relation, "object": self.object,
                "rule": self.rule, "piece": self.piece}


@dataclass
class Reading:
    """A gloss, read."""

    concept: str
    gloss: str
    genus: str = ""
    #: a sense of the genus is among the concept's ancestors; None unchecked
    agrees: bool | None = None
    facts: list = field(default_factory=list)
    #: pieces nothing could be read from
    unread: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"concept": self.concept, "gloss": self.gloss,
                "genus": self.genus, "agrees": self.agrees,
                "facts": [one.as_dict() for one in self.facts],
                "unread": list(self.unread)}


def pieces(gloss: str) -> list[str]:
    """The gloss without asides or examples, cut at its semicolons.

    `(probably descended from the common wolf)` is an aside, and everything
    after a colon is a list of examples: `...: domestic cats; wildcats`.
    """
    depth, kept = 0, []
    for character in gloss or "":
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif not depth:
            kept.append(character)
    text = "".join(kept).split(":")[0]
    return [" ".join(piece.split()).strip(" ,.")
            for piece in text.split(";") if piece.strip(" ,.")]


def _join(words) -> str:
    """Words back into text, with `short - legged` closed up again."""
    out = ""
    for word in words:
        text = word.text if hasattr(word, "text") else word
        if text == "-" or out.endswith("-"):
            out += text
        else:
            out += (" " if out else "") + text
    return out.strip()


def _children(words, index) -> list:
    return [word for word in words
            if word.head == index and word.index != index]


def _subtree(words, index, stop=frozenset()) -> list:
    """A word and everything under it, leaving out branches whose dependency
    is in `stop`."""
    out, todo = [], [index]
    while todo:
        at = todo.pop()
        out.append(words[at])
        todo.extend(child.index for child in _children(words, at)
                    if child.dep not in stop)
    return sorted(out, key=lambda word: word.index)


def _conjoined(words, index) -> list[int]:
    """A word and every word coordinated with it: `small and light`."""
    out, todo = [], [index]
    while todo:
        at = todo.pop()
        out.append(at)
        todo.extend(child.index for child in _children(words, at)
                    if child.dep == "conj")
    return sorted(out)


def _alternatives(words, index) -> bool:
    """Is this word one of several joined by `or`? `red or yellow`."""
    group = set(_conjoined(words, index))
    head = words[index]
    while head.dep == "conj" and head.head != head.index:
        head = words[head.head]
        group |= set(_conjoined(words, head.index))
    return any(child.dep == "cc" and child.text.lower() in ("or", "nor")
               for member in group for child in _children(words, member))


#: What a noun phrase does not carry into a fact: whatever hangs off it as a
#: clause or a further phrase. Articles are dropped by word, so `no` stays.
NOUN_STOP = frozenset({"prep", "acl", "relcl", "cc", "conj", "appos",
                       "punct", "advmod", "predet", "nummod"})


def noun_phrase(words, index) -> str:
    kept = [word for word in _subtree(words, index, NOUN_STOP)
            if word.text.lower() not in DETERMINERS
            and not (word.dep == "amod" and _alternatives(words, word.index))
            and word.text.lower() not in QUANTIFIERS]
    hyphens = [word for word in words if word.text == "-"
               and any(word.head == one.index for one in kept)]
    numbers = [word for word in _children(words, index)
               if word.dep == "nummod"]
    return _join(sorted(kept + hyphens + numbers,
                        key=lambda word: word.index)).lower()


def _short(text: str) -> bool:
    return bool(text) and len(text.split()) <= LONGEST


def _trim(text: str) -> str:
    """A phrase without what a split left hanging off either end."""
    words = (text or "").split()
    while words and words[-1] in DANGLING:
        words = words[:-1]
    while words and words[0] in ("and", "or", "but", "nor", ","):
        words = words[1:]
    return " ".join(words)


def _empty(fact: "Defined") -> bool:
    """A fact that says nothing a question about the kind could ask."""
    words = fact.object.split()
    if not words or not any(character.isalpha()
                            for character in fact.object):
        return True                                 # `-`, from `non-fat`
    if words[0] in REPORTED or any(word in SUBORDINATE for word in words):
        return True
    if fact.object.startswith(EMPTY_OPENINGS):
        return True
    # `giving`, from a gloss that is a gerund phrase and names no kind.
    return (fact.relation == "has_property" and len(words) == 1
            and words[0].endswith("ing") and "-" not in words[0])


def _adjective(words, index) -> str:
    """An adjective with its hyphenated parts -- `short-legged` -- and not
    the adjective beside it: `nocturnal mouselike` is two properties.

    spaCy often parses `non-fat milk` as three modifiers of `milk`, none the
    head of another, so the parts are joined across every neighbouring hyphen
    whatever the parse made of them. `fat` on its own is the opposite of what
    the gloss said."""
    kept = [words[index]]
    for child in _children(words, index):
        if child.dep == "punct" and child.text == "-":
            kept.append(child)
        elif (child.dep not in ("cc", "conj")
              and child.index + 1 < len(words)
              and words[child.index + 1].text == "-"):
            kept.extend(_subtree(words, child.index,
                                 {"cc", "conj", "advmod"}))
    spread = {word.index for word in kept}
    first = low = min(spread)
    last = high = max(spread)
    while low >= 2 and words[low - 1].text == "-":
        low -= 2
    while high + 2 < len(words) and words[high + 1].text == "-":
        high += 2
    spread |= set(range(low, first)) | set(range(last + 1, high + 1))
    return _join([words[one] for one in sorted(spread)]).lower()


def question_for(kind: str, relation: str, obj: str, rule: str = "") -> str:
    """A defined fact as the bare question the teacher is asked about it."""
    who = ("an " if kind[:1].lower() in "aeiou" else "a ") + kind
    if relation in ("capable_of", "not_capable_of"):
        return f"can {who} {obj}"
    if relation in ("has_a", "has_part"):
        return f"does {who} have {obj}"
    if relation == "used_for":
        return f"is {who} used {'to' if rule == 'used to' else 'for'} {obj}"
    if relation == "at_location":
        return f"is {who} found in {obj}"
    return f"is {who} {obj}"


#: A defined fact as it reads on the page.
SAYS = {"capable_of": "can {}", "not_capable_of": "cannot {}",
        "has_a": "has {}", "has_part": "has {}", "has_property": "is {}",
        "used_for": "is used for {}", "at_location": "is found in {}",
        "receives_action": "is {}"}


def says(relation: str, obj: str, rule: str = "") -> str:
    if relation == "used_for" and rule == "used to":
        return f"is used to {obj}"
    return SAYS.get(relation, relation + " {}").format(obj)


class GlossReader:
    """Reads glosses with an `Asker`'s parser, and checks their genus against
    its reasoner's taxonomy."""

    def __init__(self, asker) -> None:
        self.asker = asker

    # -- the whole gloss ----------------------------------------------------
    def read(self, concept: str, gloss: str, check: bool = True) -> Reading:
        reading = Reading(concept, gloss)
        if self._instance(concept):
            # `British statesman who bought controlling interest in the Suez
            # Canal`: a gloss of one person, place or event, not of a kind.
            reading.unread.append("an instance, not a kind")
            return reading
        for number, piece in enumerate(pieces(gloss)):
            if number == 0 and not self._fragment(piece):
                found = self._genus_piece(concept, piece, reading, check)
            else:
                found = self._fragment_piece(piece, piece)
            found = [kept for kept in map(self._normalise, found)
                     if kept is not None]
            if found:
                reading.facts.extend(found)
            elif not (number == 0 and reading.genus):
                reading.unread.append(piece)
        reading.facts = list({(one.relation, one.object): one
                              for one in reading.facts}.values())
        if reading.genus and check:
            reading.agrees = self.agrees(concept, reading.genus)
        return reading

    def _normalise(self, fact: Defined) -> Defined | None:
        """A fact as it should be kept, or None.

        Three shapes the teacher was going to be asked about, settled
        without it. `can have daisylike flowers` is had, not done. A can-fact
        whose first word WordNet has no verb for is a list read as a clause:
        `can fungus gnats`, from `mosquitoes; fungus gnats; crane flies`. And
        a fact with `is` inside it is two clauses the parse ran together:
        `list the alternatives is used in voting`.
        """
        fact.object = _trim(fact.object)
        words = fact.object.split()
        if not _short(fact.object) or _empty(fact):
            return None
        if any(word in ("is", "are", "was", "were") for word in words[1:]):
            return None
        if fact.relation in ("capable_of", "not_capable_of"):
            if (fact.relation == "capable_of" and len(words) > 1
                    and words[0] in ("have", "has")):
                return Defined("has_a", " ".join(words[1:]), fact.rule,
                               fact.piece)
            if not self._senses(self.asker.lemma(words[0]), "v"):
                return None
        return fact

    @staticmethod
    def _instance(concept: str) -> bool:
        """Is this synset an instance -- Disraeli, the Suez Canal -- rather
        than a kind? WordNet says, through its instance hypernyms."""
        try:
            from nltk.corpus import wordnet

            return bool(wordnet.synset(
                concept.replace(" ", "_")).instance_hypernyms())
        except Exception:                           # noqa: BLE001
            return False

    def agrees(self, concept: str, genus: str) -> bool:
        """Is a noun sense of the genus one of the concept's ancestors?"""
        senses = self._senses(genus, "n")
        if not senses:
            return False
        return any(node in senses for node, distance, _ in
                   self.asker.reasoner.ascend(concept) if distance)

    def _senses(self, word: str, pos: str) -> set:
        found = self.asker.reasoner.senses_of(word, pos) or []
        return {sense["id"] for sense in found if sense.get("pos") == pos}

    def _fragment(self, piece: str) -> bool:
        """Does this piece say something about the kind, rather than name
        one? `used to ...`, `found in ...`, `pointed at both ends`."""
        words = [word for word in piece.split()
                 if word.lower() not in QUANTIFIERS]
        if not words:
            return False
        first = words[0].lower()
        if (first in HAVING or first in LOCATED
                or first in ("used", "able", "has", "have", "having")):
            return True
        parsed = self.asker.words_of(" ".join(words[:3])) or []
        return (len(parsed) > 1 and parsed[0].tag in ("VBN", "VBD", "VBZ")
                and parsed[1].tag in ("IN", "TO", "RP"))

    # -- the first piece: the broader kind and what hangs off it ------------
    def _genus_piece(self, concept, piece, reading, check) -> list:
        first = piece.split()[0].lower()
        named = (piece if first in DETERMINERS or first in PARTITIVES
                 or first in SEXES else "a " + piece)
        words = self.asker.words_of("it is " + named) or []
        if len(words) < 3:
            return []
        head = next((word for word in _children(words, 1)
                     if word.dep in ("attr", "acomp") and word.index > 1),
                    None)
        if head is None:
            return []
        facts: list = []
        heads = [head]
        while head is not None and head.text.lower() in (
                PARTITIVES | SEXES | TAXA | GROUPS):
            of = next((child for child in _children(words, head.index)
                       if child.dep == "prep" and child.text.lower() == "of"),
                      None)
            target = (next((child for child in _children(words, of.index)
                            if child.dep == "pobj"), None) if of else None)
            if target is None:
                break
            if head.text.lower() in TAXA | GROUPS and (
                    target.tag in ("NNP", "NNPS")
                    or target.text[:1].isupper()):
                head = None                 # `the genus Quercus`: a taxon
                break
            if head.text.lower() in SEXES:
                facts.append(Defined("has_property", head.text.lower(),
                                     "adjective", piece))
            heads.append(target)
            head = None if target.text.lower() in TAXA else target
        genus = head if head is not None and (
            head.noun or head.tag == "JJ") and head.text.lower() not in (
                PARTITIVES | SEXES) else None
        # `genus of tropical American woody vines`: the concept is the genus,
        # and woody is what its members are. A taxon or a group is described
        # by its own modifiers only -- `large diverse order` -- not theirs,
        # though what they are still names its broader kind.
        group = heads[0].text.lower() in TAXA | GROUPS
        if genus is not None:
            kind_facts = self._genus(concept, words, genus, reading, piece,
                                     check)
            if not group:
                facts.extend(kind_facts)
        for one in heads[:1] if group else heads:
            facts.extend(self._modifiers(words, one, piece))
        # What the frame hung off `is` rather than off the noun: `having
        # webbed feet and wings`, after the regions the penguin lives in.
        for child in _children(words, 1):
            if child.dep in ("xcomp", "advcl") and child.index > head_index(
                    heads):
                facts.extend(self._fragment_piece(
                    _join(_subtree(words, child.index)), piece))
        return facts

    def _genus(self, concept, words, genus, reading, piece, check) -> list:
        """The broader kind's name, and compounds that were adjectives."""
        facts: list = []
        compounds = [child for child in _children(words, genus.index)
                     if child.dep == "compound"]
        name = genus.lemma or genus.text.lower()
        joined = _join([word.text.lower() for word in compounds] + [name])
        reading.genus = name
        if compounds and self.asker.known(joined) and not (
                check and not self.agrees(concept, joined)
                and self.agrees(concept, name)):
            reading.genus = joined
            return facts
        for word in compounds:
            # `flightless birds`, `pulpy fruit`: a compound that is not a
            # noun the ontology has is an adjective the tagger called one.
            # A name is not: `United States photographer`.
            if (not self.asker.known(word.text.lower())
                    and word.tag not in ("NNP", "NNPS")
                    and not word.text[:1].isupper()):
                facts.append(Defined("has_property", word.text.lower(),
                                     "adjective", piece))
        return facts

    def _modifiers(self, words, head, piece) -> list:
        facts: list = []
        for child in _children(words, head.index):
            if child.dep == "amod":
                if _alternatives(words, child.index):
                    continue
                for index in _conjoined(words, child.index):
                    if (words[index].tag in ("NNP", "NNPS")
                            or words[index].text[:1].isupper()):
                        continue            # `American`, `States`
                    text = _adjective(words, index)
                    if (text and text not in QUANTIFIERS
                            and words[index].tag != "JJS"):
                        facts.append(Defined("has_property", text,
                                             "adjective", piece))
            elif child.dep == "prep" and child.text.lower() == "with":
                facts.extend(self._had(words, child, piece, "with"))
            elif child.dep in ("acl", "relcl"):
                text = _join(_subtree(words, child.index))
                facts.extend(self._fragment_piece(text, piece))
        return facts

    def _had(self, words, preposition, piece, rule) -> list:
        """`with a heavy rigid head and a handle` -> each thing had."""
        target = next((child for child in _children(words, preposition.index)
                       if child.dep == "pobj"), None)
        if target is None:
            return []
        return [Defined("has_a", noun_phrase(words, index), rule, piece)
                for index in _conjoined(words, target.index)
                if words[index].noun and noun_phrase(words, index)]

    # -- a fragment, or a clause about the kind ------------------------------
    def _fragment_piece(self, text: str, piece: str) -> list:
        """Something said about the kind, read in a frame with `it` as its
        subject and split into its clauses."""
        words = (text or "").split()
        while words and (words[0].lower() in QUANTIFIERS
                         or words[0].lower() in ("that", "which", "who")):
            words = words[1:]
        if len(words) > 2 and words[0].lower() in ("this", "these"):
            words = words[2:]
        if not words:
            return []
        first = words[0].lower()
        second = words[1].lower() if len(words) > 1 else ""
        if first in ("has", "have") and second != "been":
            return self._has(" ".join(words[1:]), piece)
        if first == "having":
            return self._has(" ".join(words[1:]), piece)
        if first in ("is", "are") and second in DETERMINERS:
            return []                   # `that is the highest member of`
        if first in ("is", "are", "was", "were", "can") or (
                first in ("has", "have") and second == "been"):
            sentence = "it " + " ".join(words)
        elif (first in HAVING or first in LOCATED
              or first in ("used", "able")):
            sentence = "it is " + " ".join(words)
        else:
            lead = (self.asker.words_of(" ".join(words[:2])) or [None])[0]
            tag = lead.tag if lead is not None else ""
            if tag in ("VBN", "VBD", "JJ", "JJR"):
                sentence = "it is " + " ".join(words)
            else:
                sentence = ("it " if tag == "VBZ" else "they ") + " ".join(
                    words)
        parsed = self.asker.words_of(sentence) or []
        split = coordination.split([word.text for word in parsed],
                                   [(word.tag, word.dep, word.head)
                                    for word in parsed]) if parsed else None
        if not split:
            return []
        facts: list = []
        joined: list = []
        for clause in split:
            found = self._map(parsed, clause, piece)
            facts.extend(found)
            joined.extend([clause.joined_by] * len(found))
        # `produces or sells petroleum`: verbs joined by `or` share the object
        # only the last one carries.
        for index, fact in enumerate(facts):
            if len(fact.object.split()) != 1:
                continue
            later = next((facts[at] for at in range(index + 1, len(facts))
                          if facts[at].relation == fact.relation
                          and joined[at] == "or"
                          and len(facts[at].object.split()) > 1), None)
            if later is not None:
                fact.object = " ".join([fact.object]
                                       + later.object.split()[1:])
        return facts

    def _has(self, text: str, piece: str) -> list:
        """`a pointed blade with a sharp edge and a handle`, `scales and
        breathing through gills`: each thing had, or done."""
        facts: list = []
        for chunk in _chunks(text):
            words = chunk.split()
            if not words:
                continue
            lowered = [word.lower() for word in words]
            if "ability" in lowered[:2] and "to" in lowered[1:4]:
                verb = lowered[lowered.index("to") + 1:]
                relation = ("not_capable_of" if lowered[0] == "no"
                            else "capable_of")
                if verb:
                    facts.append(Defined(relation, " ".join(verb), "ability",
                                         piece))
                continue
            if lowered[0] in ("is", "are") or self._verb_only(lowered[0]):
                facts.extend(self._fragment_piece(chunk, piece))
                continue
            parsed = self.asker.words_of("it has " + chunk) or []
            if len(parsed) > 2 and parsed[2].tag == "VBG":
                verb = parsed[2].lemma or parsed[2].text.lower()
                rest = _join(parsed[3:]).lower()
                facts.append(Defined("capable_of",
                                     (verb + " " + rest).strip(), "clause",
                                     piece))
                continue
            # The head of what is had, not its first noun: `skin eruption`.
            noun = next((word for word in parsed[2:] if word.noun
                         and word.dep in ("dobj", "attr", "pobj", "conj",
                                          "ROOT")), None) or next(
                (word for word in parsed[2:] if word.noun), None)
            if noun is not None:
                phrase = noun_phrase(parsed, noun.index)
                if phrase:
                    facts.append(Defined("has_a", phrase, "has", piece))
        return facts

    def _map(self, parsed, clause, piece) -> list:
        """One clause of a fragment, as a fact by what its verb is."""
        rest = [parsed[word.index] for word in clause.rest]
        aux = [parsed[word.index].text.lower() for word in clause.aux]
        while rest and rest[0].text.lower() in QUANTIFIERS:
            rest = rest[1:]
        if not rest:
            return []
        head = rest[0]
        text = head.text.lower()
        if text in ("is", "are") and len(rest) > 1:
            aux.append(text)
            rest = rest[1:]
            head, text = rest[0], rest[0].text.lower()
        tail = rest[1:]
        after = _join(tail).lower()
        passive = any(verb in ("is", "are", "been", "be", "was", "were")
                      for verb in aux)

        if text == "used" and tail:
            return self._used(parsed, head, tail, piece)
        if text == "able" and len(tail) > 1 and tail[0].text.lower() == "to":
            relation = "not_capable_of" if clause.negated else "capable_of"
            return [Defined(relation, object_, "able", piece)
                    for object_ in self._verb_split(_join(tail[1:]).lower())]
        if text in LOCATED and tail and tail[0].tag == "IN":
            place = _join([word for word in tail[1:] if word.text.lower()
                           not in DETERMINERS]).lower()
            return [Defined("at_location", place, "found", piece)] if place \
                else []
        if text in HAVING and tail and tail[0].text.lower() in ("by", "with"):
            return self._has(_join(tail[1:]), piece)
        if head.tag.startswith("JJ"):
            return [Defined("has_property", _join(rest).lower(), "adjective",
                            piece)]
        if head.tag in ("VBN", "VBD") and (passive or head.tag == "VBN"):
            relation = ("receives_action"
                        if any(word.text.lower() in ("by", "with", "as")
                               for word in tail) else "has_property")
            return [Defined(relation, (text + " " + after).strip(),
                            "participle", piece)]
        if (head.verbal and text not in ("is", "are", "be", "has", "have")
                and (head.lemma or text) not in EMPTY_VERBS):
            relation = "not_capable_of" if clause.negated else "capable_of"
            verb = head.lemma or text
            return [Defined(relation, object_, "clause", piece)
                    for object_ in self._verb_split(
                        (verb + " " + after).strip())]
        return []

    def _used(self, parsed, used, tail, piece) -> list:
        """`used to stir or serve or take up food` -> stir food, serve food,
        take up food; `used as a cutting instrument` -> cutting instrument."""
        joiner = tail[0].text.lower()
        if joiner in ("as", "for", "in") and len(tail) > 1:
            target = _join([word for word in tail[1:]
                            if word.text.lower() not in DETERMINERS]).lower()
            return [Defined("used_for", target, "used as", piece)]
        if joiner != "to" or len(tail) < 2:
            return []
        verbs = [child for child in _children(parsed, used.index)
                 if child.dep == "xcomp"]
        if not verbs:
            return [Defined("used_for", object_, "used to", piece)
                    for object_ in self._verb_split(_join(tail[1:]).lower())]
        chain = _conjoined(parsed, verbs[0].index)
        phrases = []
        for index in chain:
            words = [word for word in _subtree(parsed, index,
                                               {"conj", "cc", "aux"})]
            phrases.append(words)
        # What the last verb does it to is what they all do it to: `stir or
        # serve or take up food`. Its object only, not its particle.
        last = phrases[-1] if phrases else []
        objects = [word for one in last if one.dep == "dobj"
                   for word in _subtree(parsed, one.index)]
        out = []
        for words in phrases:
            verb = words[0]
            if len(words) == 1 and objects:
                words = words + objects
            text = " ".join([verb.lemma or verb.text.lower()]
                            + [word.text.lower() for word in words[1:]
                               if word.text.lower() not in DETERMINERS])
            out.append(Defined("used_for", text, "used to", piece))
        return out

    def _verb_only(self, word: str) -> bool:
        """A word WordNet has as a verb and never as a noun."""
        lemma = self.asker.lemma(word)
        return bool(self._senses(lemma, "v")) and not self._senses(lemma, "n")

    def _verb_split(self, text: str) -> list[str]:
        """Split at an `and` or `or` before a word WordNet has only as a
        verb: `produce spermatozoa and secrete androgens`. The tagger calls
        `secrete` a noun there, and a word with no noun sense is not one."""
        words = text.split()
        out, current = [], []
        for index, word in enumerate(words):
            after = words[index + 1] if index + 1 < len(words) else ""
            if word in ("and", "or") and current and after:
                if self._verb_only(after):
                    out.append(" ".join(current))
                    current = []
                    continue
            current.append(word if current or index else
                           (self.asker.lemma(word) if word else word))
        if current:
            out.append(" ".join(current))
        return [one for one in out if one]


def head_index(heads) -> int:
    return max(one.index for one in heads) if heads else 0


def _chunks(text: str) -> list[str]:
    """What is had, one thing at a time: split at `and`, `or` and commas
    that are not inside the phrase they join."""
    out, current = [], []
    for word in (text or "").replace(",", " , ").split():
        if word in ("and", "or", ","):
            if current:
                out.append(" ".join(current))
            current = []
        else:
            current.append(word)
    if current:
        out.append(" ".join(current))
    return out


class DefinitionMemory:
    """Every definition read, and the facts read out of it, in sqlite.

    Part of long-term memory: shared by every conversation, filled as v688
    retrieves definitions and in bulk by `learn_definitions.py`, and never
    written into the store. A fact the teacher disputed is kept, so the
    audit can count it, and never read by the rules.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS glosses (
        concept TEXT PRIMARY KEY, gloss TEXT NOT NULL, genus TEXT,
        agrees INTEGER, unread TEXT, how TEXT NOT NULL, learned REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS defined (
        concept TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
        rule TEXT, piece TEXT, checked TEXT, confidence REAL,
        PRIMARY KEY (concept, relation, object));
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            str(self.path) if self.path else ":memory:",
            check_same_thread=False)
        self.connection.executescript(self.SCHEMA)
        self._facts: dict[str, list] = {}

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def has(self, concept: str) -> bool:
        with self.lock:
            return self.connection.execute(
                "SELECT 1 FROM glosses WHERE concept = ?",
                (concept,)).fetchone() is not None

    def keep(self, reading: Reading, how: str,
             checks: dict | None = None) -> None:
        """Write one reading; `checks` is (relation, object) -> (verdict,
        confidence) from the teacher, where it was asked."""
        self.keep_all([reading], how, {
            (reading.concept, relation, obj): found
            for (relation, obj), found in (checks or {}).items()})

    def keep_all(self, readings, how: str, checks: dict | None = None) -> None:
        """Write many readings in one transaction. `checks` is keyed by
        (concept, relation, object)."""
        checks = checks or {}
        now = time.time()
        with self.lock, self.connection:
            for reading in readings:
                self.connection.execute(
                    "INSERT OR REPLACE INTO glosses VALUES "
                    "(?, ?, ?, ?, ?, ?, ?)",
                    (reading.concept, reading.gloss, reading.genus,
                     None if reading.agrees is None else int(reading.agrees),
                     " | ".join(reading.unread), how, now))
                self.connection.execute(
                    "DELETE FROM defined WHERE concept = ?",
                    (reading.concept,))
                self.connection.executemany(
                    "INSERT OR REPLACE INTO defined VALUES "
                    "(?, ?, ?, ?, ?, ?, ?)",
                    [(reading.concept, fact.relation, fact.object, fact.rule,
                      fact.piece,
                      *(checks.get((reading.concept, fact.relation,
                                    fact.object)) or (None, None)))
                     for fact in reading.facts])
        for reading in readings:
            self._facts.pop(reading.concept, None)

    def check_all(self, updates) -> None:
        """(verdict, confidence, concept, relation, object) rows."""
        with self.lock, self.connection:
            self.connection.executemany(
                "UPDATE defined SET checked = ?, confidence = ? WHERE "
                "concept = ? AND relation = ? AND object = ?", list(updates))
        self._facts.clear()

    def facts(self, concept: str) -> list:
        """The facts the rules may read: everything not disputed."""
        if concept not in self._facts:
            with self.lock:
                rows = self.connection.execute(
                    "SELECT relation, object FROM defined WHERE concept = ? "
                    "AND (checked IS NULL OR checked != 'disputed')",
                    (concept,)).fetchall()
            self._facts[concept] = [
                Fact(concept, relation, obj, DEFINED, CONFIDENCE, False)
                for relation, obj in rows]
        return self._facts[concept]

    def entry(self, concept: str) -> dict | None:
        """The gloss, its genus, and every fact with how it was checked."""
        with self.lock:
            row = self.connection.execute(
                "SELECT gloss, genus, agrees, unread, how, learned FROM "
                "glosses WHERE concept = ?", (concept,)).fetchone()
            if row is None:
                return None
            facts = self.connection.execute(
                "SELECT relation, object, rule, checked, confidence FROM "
                "defined WHERE concept = ? ORDER BY rowid",
                (concept,)).fetchall()
        return {"concept": concept, "gloss": row[0], "genus": row[1],
                "agrees": None if row[2] is None else bool(row[2]),
                "unread": row[3], "how": row[4], "learned": row[5],
                "facts": [{"relation": relation, "object": obj, "rule": rule,
                           "checked": checked, "confidence": confidence}
                          for relation, obj, rule, checked, confidence
                          in facts]}

    def summary(self) -> dict:
        with self.lock:
            glosses = self.connection.execute(
                "SELECT count(*) FROM glosses").fetchone()[0]
            facts = self.connection.execute(
                "SELECT count(*) FROM defined").fetchone()[0]
        return {"definitions": glosses, "facts": facts}

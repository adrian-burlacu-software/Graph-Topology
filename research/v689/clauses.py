"""One claim at a time: a statement's coordinated clauses, split by the parse.

A statement can make several claims -- `testicles shrink in cold temperatures,
and they expand in warm ones` is two -- and everything below `read()` expects
one. Splitting on the word `and` is not enough, and was tried: `and they
expand` hides the verb behind a pronoun, `dogs eat meat and bones` is one
claim, and `beagles can swim and bark` shares its `can`. The dependency parse
already says which verbs are coordinated and what the subject of each one is,
so this reads that instead of guessing from the words.

Every clause after the first is filled in from the one before it, so it can be
read as a sentence of its own:

    and they expand in warm ones   ->  testicles expand in warm temperatures
    and bark                       ->  beagles can bark
    but they can run               ->  beagles can run
    and it can not swim            ->  it can not swim  (discourse resolves it)

- A clause with no subject of its own takes the one before it, and its
  auxiliary with it. Its negation only after `or` or `nor`: `can't swim or
  bark` denies both, `can't swim and bark` is not a claim about barking.
- A pronoun subject stands for the subject before it when that was a kind.
  When it was an individual the pronoun is left alone, because an individual
  is resolved by salience in `discourse.py`, and a kind is not an individual.
- `ones` is the noun at the same place in the clause before: `in warm ones`
  after `in cold temperatures`.

When the root of the parse is not a verb -- spaCy reads `dogs bark and cats
purr` as a noun phrase -- there is nothing to hang clauses on and `split`
returns None. `reading.py` then refuses what still looks like several claims
rather than storing it as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Dependencies that make a word, and everything under it, a subject.
SUBJECTS = frozenset({"nsubj", "nsubjpass", "expl", "csubj"})
AUXILIARIES = frozenset({"aux", "auxpass"})

#: What says a coordinated noun is a noun and not a second verb: `the bones`,
#: `big bones`. `bark` in `beagles can swim and bark` has none of them.
MODIFIERS = frozenset({"det", "amod", "poss", "nummod", "compound"})

#: Subject pronouns that can stand for the subject of the clause before.
PRONOUNS = frozenset({"it", "he", "she", "they"})

#: A noun left out and pointed back to.
ELIDED = frozenset({"ones", "one"})


@dataclass
class Word:
    index: int
    text: str
    tag: str
    dep: str
    head: int
    head_text: str = ""

    @property
    def verbal(self) -> bool:
        return self.tag.startswith("VB") or self.tag == "MD"

    @property
    def noun(self) -> bool:
        return self.tag.startswith("NN")


@dataclass
class Clause:
    """One claim: who, the auxiliaries and negation, and everything else."""

    subject: list[Word] = field(default_factory=list)
    aux: list[Word] = field(default_factory=list)
    negated: bool = False
    rest: list[Word] = field(default_factory=list)
    #: the word that joined it to the clause before: `and`, `but`, `or`
    joined_by: str = ""

    def words(self, typed: list[str]) -> list[str]:
        """The clause as words, spelled as they were typed."""
        return ([typed[word.index] for word in self.subject]
                + [typed[word.index] for word in self.aux]
                + (["not"] if self.negated else [])
                + [typed[word.index] for word in self.rest])


def parse(words: list[str], analysis) -> list[Word]:
    """`analysis` is (tag, dependency, head index) per word."""
    return [Word(index, text, tag, dep, head,
                 words[head] if 0 <= head < len(words) else "")
            for index, (text, (tag, dep, head))
            in enumerate(zip(words, analysis))]


def _children(parsed: list[Word], index: int, deps) -> list[Word]:
    return [word for word in parsed
            if word.head == index and word.index != index and word.dep in deps]


def predicates(parsed: list[Word]) -> list[int]:
    """The root and every verb coordinated with it, in order.

    A coordinated word counts when it is a verb, when it has a subject of its
    own, or when it hangs off a verb with nothing marking it as a noun.
    """
    root = next((word for word in parsed if word.dep == "ROOT"), None)
    if root is None or not root.verbal:
        return []
    found = {root.index}
    grew = True
    while grew:
        grew = False
        for word in parsed:
            if (word.index in found or word.dep != "conj"
                    or word.head not in found):
                continue
            if (word.verbal or _children(parsed, word.index, SUBJECTS)
                    or (parsed[word.head].verbal
                        and not _children(parsed, word.index, MODIFIERS))):
                found.add(word.index)
                grew = True
    return sorted(found)


def _owner(parsed: list[Word], index: int, found: set) -> int | None:
    """The predicate a word belongs to: the first one above it."""
    seen: set = set()
    while index not in found:
        if index in seen or parsed[index].head == index:
            return None
        seen.add(index)
        index = parsed[index].head
    return index


def _under(parsed: list[Word], index: int, top: int, deps) -> bool:
    """Is any word from this one up to its predicate one of `deps`?"""
    while index != top:
        if parsed[index].dep in deps:
            return True
        index = parsed[index].head
    return False


def split(words: list[str], analysis) -> list[Clause] | None:
    """The clauses of a statement in order, or None if the parse has no verb
    at its root to hang them on."""
    if not analysis or len(analysis) != len(words):
        return None
    parsed = parse(words, analysis)
    found = predicates(parsed)
    if not found:
        return None
    tops = set(found)
    clauses = {index: Clause() for index in found}
    for word in parsed:
        top = _owner(parsed, word.index, tops)
        if top is None:
            continue
        clause = clauses[top]
        if word.index != top:
            if word.dep == "cc" and word.head in tops:
                later = [index for index in found if index > word.index]
                if later:
                    clauses[later[0]].joined_by = word.text
                    continue
            if word.head == top and word.dep in AUXILIARIES:
                clause.aux.append(word)
                continue
            if word.head == top and word.dep == "neg":
                clause.negated = True
                continue
            if _under(parsed, word.index, top, SUBJECTS):
                clause.subject.append(word)
                continue
        clause.rest.append(word)
    return [clauses[index] for index in found]


def _elided(word: Word, before: Clause) -> Word:
    """`ones` as the noun it stands for in the clause before."""
    if word.text not in ELIDED:
        return word
    nouns = [one for one in before.rest
             if one.noun and one.text not in ELIDED]
    same = [one for one in nouns
            if one.dep == word.dep and one.head_text == word.head_text]
    chosen = (same or nouns)[-1:]
    if not chosen:
        return word
    return Word(chosen[0].index, chosen[0].text, chosen[0].tag, word.dep,
                word.head, word.head_text)


def standalone(clause: Clause, before: Clause, kind: bool) -> Clause:
    """This clause as a sentence of its own, filled in from the one before.

    `kind` says whether the clause before was a claim about a kind, which is
    when a pronoun subject may be replaced by that kind.
    """
    shared = not clause.subject
    pronoun = (len(clause.subject) == 1
               and clause.subject[0].text in PRONOUNS)
    subject = before.subject if shared or (kind and pronoun) else clause.subject
    aux = clause.aux if (clause.aux or not shared) else before.aux
    negated = clause.negated or (shared and not clause.aux
                                 and clause.joined_by in ("or", "nor")
                                 and before.negated)
    return Clause(list(subject), list(aux), negated,
                  [_elided(word, before) for word in clause.rest],
                  clause.joined_by)


def continues(clause: Clause, kind: bool) -> bool:
    """Does this clause go on talking about the same kind as the one before:
    no subject of its own, or a pronoun standing for it?"""
    return kind and (not clause.subject or (
        len(clause.subject) == 1 and clause.subject[0].text in PRONOUNS))

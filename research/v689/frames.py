"""A sentence read off its dependency parse, and said back in normal form.

`grammar.py` reads a question by the positions of its words, so a word it did
not expect breaks it: `where is Mary really`, `can you tell me where Mary
is`, `which room is Mary in` each ask what `where is Mary` asks and were
answered with a store listing. The parse already says which word is the
verb, which phrase its subject, which words hang off it as adjuncts and where
a question's gap is, whatever order they came in. This reads that, and says
the sentence back in the order the grammar and the statement reader read:

    where is Mary really                 where is mary          adjunct adverb
    where exactly is Mary                where is mary
    can you tell me where Mary is        where is mary          embedded question
    any idea where Mary is               where is mary
    which room is Mary in                where is mary          a place asked
    in which room is Mary                where is mary          by its kind
    so where is Mary                     where is mary          discourse word
    where is Mary at the moment          where is mary          time adjunct
    Mary quickly went to the kitchen     mary went to the kitchen
    in the end Mary went to the kitchen  mary went to the kitchen  fronted
    the apple was put in it by Mary      mary put the apple in it   passive

What counts as an adjunct is WordNet's, not a list: an adverb WordNet derives
from an adjective (a pertainym: `really` from real, `quickly` from quick) says
how or how surely, and never who, what or where; `back`, `then`, `now`,
`again`, `first`, `there` and `never` have none, and are kept. A phrase is
about time when its noun's first sense is in WordNet's `noun.time`. A passive
with no agent is said with its Theme as subject only where VerbNet has a frame
whose subject is the Theme (`the apple moved to the kitchen`, roll-51.3.1).

What is dropped is only what the frame's slots do not hold: the adjuncts are
kept on the frame (`Frame.adjuncts`), for whatever reads manner or certainty
later, and for saying the sentence back whole (v690).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .clauses import Word, parse

#: Question words, as the tagger marks them.
WH = frozenset({"where", "when", "who", "whom", "what", "which", "how", "why",
                "whose"})
WH_TAGS = frozenset({"WRB", "WP", "WDT", "WP$"})

#: A preposition that asks a place when its noun is a wh-word's:
#: `which room is Mary in`.
LOCATIVE = frozenset({"in", "inside", "at", "on"})

#: Dependencies that make a clause part of a bigger sentence: where an
#: embedded question hangs (`tell me where ...`, `any idea where ...`).
EMBEDDED = frozenset({"ccomp", "relcl", "advcl", "xcomp", "acl", "dobj",
                      "pobj", "csubj", "conj"})

SUBJECTS = frozenset({"nsubj", "nsubjpass", "expl", "csubj"})
AUXILIARIES = frozenset({"aux", "auxpass"})

#: The auxiliary a question asks a simple tense with.
DO = {"VBD": "did", "VBZ": "does", "VBP": "do", "VB": "do"}

#: The perfect's auxiliary, for a passive in the present: `the apple is
#: moved by Mary` is said `Mary has moved the apple`.
HAVE = {"VBZ": "has", "VBP": "have", "VB": "have"}


@lru_cache(maxsize=None)
def _irregular() -> dict[str, tuple]:
    """verb -> its irregular forms, from WordNet's exception list."""
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        forms: dict[str, list] = {}
        for form, lemmas in wordnet._exception_map["v"].items():
            for lemma in lemmas:
                forms.setdefault(lemma, []).append(form)
        return {verb: tuple(sorted(found)) for verb, found in forms.items()}
    except Exception:                               # noqa: BLE001
        return {}


def past_form(participle: str, verb: str, tags) -> str:
    """The simple past of a verb said as a participle: `moved` and `put` are
    their own; `given` is `gave`. A participle that is one of WordNet's
    irregular forms of its verb has its past among the others (`given`,
    `gave`), and the tagger picks the one it reads as a past tense after `it`
    -- asked of the participle alone, it reads `it given` as one too. Empty
    when none is."""
    forms = _irregular().get(verb, ())
    others = tuple(form for form in forms if form != participle)
    candidates = (others if participle in forms and others
                  else (participle,) + others)
    for form in candidates:
        found = tags(["it", form])
        if found and found[-1] == "VBD":
            return form
    return ""


@lru_cache(maxsize=None)
def derived_adverb(word: str) -> bool:
    """Does WordNet derive this adverb from an adjective (a pertainym)?"""
    try:
        from nltk.corpus import wordnet
        return any(lemma.name() == word and lemma.pertainyms()
                   for synset in wordnet.synsets(word, "r")
                   for lemma in synset.lemmas())
    except Exception:                               # noqa: BLE001
        return False


@lru_cache(maxsize=None)
def time_noun(word: str) -> bool:
    """Is this noun's first sense a time (`moment`, `day`)?"""
    try:
        from nltk.corpus import wordnet
        senses = wordnet.synsets(word, "n")
        return bool(senses) and senses[0].lexname() == "noun.time"
    except Exception:                               # noqa: BLE001
        return False


@dataclass
class Frame:
    """One clause: its verb, the words in the order said back, and what was
    read as an adjunct rather than a slot."""

    words: list[Word]
    head: int
    #: the indices said back, in order, with replacements
    said: list = field(default_factory=list)
    adjuncts: list[int] = field(default_factory=list)
    #: `where` for a place asked by its kind; the embedding it was lifted from
    asked: str = ""
    embedded: bool = False
    passive: bool = False

    def tokens(self, typed: list[str]) -> tuple[list[str], list[str]]:
        lower, shown = [], []
        for one in self.said:
            if isinstance(one, int):
                lower.append(self.words[one].text.lower())
                shown.append(typed[one])
            else:
                lower.append(one)
                shown.append(one)
        return lower, shown


def _children(words: list[Word], index: int, deps=None) -> list[Word]:
    return [word for word in words if word.head == index
            and word.index != index and (deps is None or word.dep in deps)]


def _subtree(words: list[Word], index: int) -> set[int]:
    found, grew = {index}, True
    while grew:
        grew = False
        for word in words:
            if word.index not in found and word.head in found \
                    and word.head != word.index:
                found.add(word.index)
                grew = True
    return found


def _verb_above(words: list[Word], index: int) -> int:
    seen = set()
    while not (words[index].verbal or words[index].dep == "ROOT"):
        if index in seen or words[index].head == index:
            break
        seen.add(index)
        index = words[index].head
    return index


#: The copula, whose last word is what is said of the subject.
COPULAS = frozenset({"is", "are", "was", "were", "am", "be", "been", "being"})


def _predicate(words: list[Word], word: Word, within: set[int]) -> bool:
    """Is this adverb what a copula says of its subject -- `is a rock hard`,
    where the tagger reads `hard` as an adverb -- rather than an adjunct? The
    last word after a copula is, unless a question word fills that place
    (`where is Mary really`)."""
    head = words[word.head] if 0 <= word.head < len(words) else word
    if head.text.lower() not in COPULAS or word.index != max(within):
        return False
    return not any(child.text.lower() in WH
                   for child in _children(words, head.index))


def _adjuncts(words: list[Word], within: set[int]) -> set[int]:
    """Adverbs WordNet derives from adjectives, hanging off a verb or off
    another adverb, and phrases about a time."""
    out: set[int] = set()
    for word in words:
        if word.index not in within:
            continue
        head = words[word.head] if 0 <= word.head < len(words) else word
        if (word.tag == "RB" and word.dep == "advmod"
                and word.text.lower() not in WH
                and (head.verbal or head.tag.startswith("RB")
                     or head.dep == "ROOT")
                and derived_adverb(word.text.lower())
                and not _predicate(words, word, within)):
            out |= {word.index}
        elif word.dep == "prep" and head.verbal:
            objects = _children(words, word.index, {"pobj"})
            if objects and time_noun(objects[0].text.lower()):
                out |= _subtree(words, word.index)
        elif word.dep == "npadvmod" and time_noun(word.text.lower()):
            out |= _subtree(words, word.index)
    return out


@lru_cache(maxsize=None)
def _verb_stem(word: str) -> str:
    """The verb a word is a form of, by WordNet's morphy: `hiding` -> hide."""
    try:
        from nltk.corpus import wordnet
        return wordnet.morphy(word, "v") or ""
    except Exception:                               # noqa: BLE001
        return ""


#: A tense that asks what holds now: `is holding`, `has got`, `has`.
STATIVE = frozenset({"VBZ", "VBP", "VBG"})


def _by_meaning(words, kept: list, first: Word, head: int, lemma,
                meaning) -> list | None:
    """A question asked with a verb the grammar has no shape for, said with
    the one its VerbNet meaning is (`change.meaning`): `where did Mary wind
    up`, `where is Mary hiding` -> `where is mary`; `who is holding the pear`,
    `who has got the pear` -> `who has the pear`. Having is asked only in a
    tense that asks what holds now: `what did Mary buy` is not what she has."""
    verb = words[head]
    word = lemma(verb.text.lower())
    # `where will Mary go` asks where she is going to be, not where she is.
    if word == "be" or not verb.verbal or any(
            child.tag == "MD" for child in _children(words, head, AUXILIARIES)):
        return None
    particle = next((child.text.lower() for child in _children(
        words, head, {"prt"})), "")
    # Lemmatised alone, `hiding` is a noun; as the verb it is here, WordNet
    # says `hide`.
    stems = [word, _verb_stem(verb.text.lower())]
    means = frozenset()
    for stem in dict.fromkeys(stem for stem in stems if stem):
        means = ((meaning(f"{stem} {particle}") if particle else frozenset())
                 or meaning(stem))
        if means:
            break
    if not means:
        return None
    subject = next((child for child in _children(words, head, SUBJECTS)
                    if child.index != first.index), None)
    obj = next((child for child in _children(words, head, {"dobj"})
                if child.index != first.index), None)

    def phrase(child):
        return [index for index in kept
                if index in _subtree(words, child.index)]

    asked = first.text.lower()
    present = verb.tag in STATIVE or any(
        child.text.lower() in ("is", "are", "has", "have")
        for child in _children(words, head, AUXILIARIES))
    if asked == "where" and "location" in means and subject is not None:
        return ["where", "is"] + phrase(subject)
    if "possession" in means and present:
        if asked == "who" and first.dep in SUBJECTS and obj is not None:
            return ["who", "has"] + phrase(obj)
        if asked == "what" and subject is not None and first.head == head:
            return ["what", "does"] + phrase(subject) + ["have"]
    return None


def read_question(words: list[Word], lemma, meaning=None) -> Frame | None:
    wh = [word for word in words
          if word.text.lower() in WH and word.tag in WH_TAGS]
    if not wh:
        return None
    first = wh[0]
    head = _verb_above(words, first.index)
    root = next((word.index for word in words if word.dep == "ROOT"), head)
    embedded = head != root and words[head].dep in EMBEDDED
    if embedded and not _asking(words, root):
        # `Mary went to the kitchen where she slept`: a clause of a
        # statement, not a question put inside a request.
        return None
    within = _subtree(words, head) if embedded else set(range(len(words)))
    frame = Frame(words, head, embedded=embedded)
    dropped = _adjuncts(words, within)
    # Before the question word, only a preposition it carries stays: `so
    # where is Mary` is `where is Mary`, `in which room` is a place asked.
    for word in words:
        if word.index < first.index and word.index in within and not (
                word.dep == "prep" and word.index == first.index - 1
                and first.dep == "det"):
            dropped.add(word.index)
    replaced: dict[int, str] = {}
    # `which room is Mary in`, `in which room is Mary`: a place, asked by the
    # kind of place it is.
    if first.dep == "det" and first.text.lower() in ("which", "what"):
        noun = first.head
        before = first.index - 1
        stranded = [word for word in words
                    if word.index in within and word.dep == "prep"
                    and word.text.lower() in LOCATIVE
                    and not _children(words, word.index, {"pobj"})]
        if (before >= 0 and words[before].text.lower() in LOCATIVE
                and words[before].dep == "prep"):
            dropped |= {before}
        elif stranded:
            dropped |= {stranded[-1].index}
        else:
            noun = None
        if noun is not None:
            dropped |= _subtree(words, noun) - {first.index}
            replaced[first.index] = "where"
            frame.asked = "where"
    kept = [index for index in sorted(within) if index not in dropped]
    by_meaning = (_by_meaning(words, kept, first, head, lemma, meaning)
                  if meaning is not None and not frame.asked else None)
    if by_meaning is not None:
        frame.said, frame.adjuncts = by_meaning, sorted(dropped)
        return frame
    if embedded:
        kept = _inverted(words, kept, head, first.index, lemma, replaced)
    frame.adjuncts = sorted(dropped)
    frame.said = [replaced.get(index, index) if isinstance(index, int)
                  else index for index in kept]
    return frame


#: Who a request or a wondering is between: the one asking and the one asked.
PARTIES = frozenset({"you", "i", "we"})


def _asking(words: list[Word], root: int) -> bool:
    """Is the sentence around an embedded question a request for it: an
    imperative (`tell me where`), something the speaker or the one spoken to
    does (`can you tell me`, `I wonder`), or no clause at all (`any idea
    where`)?"""
    if not words[root].verbal:
        return True
    subjects = _children(words, root, SUBJECTS)
    return not subjects or subjects[0].text.lower() in PARTIES


def _inverted(words, kept: list, head: int, wh: int, lemma,
              replaced: dict) -> list:
    """`where Mary is` as a question: `where is Mary`; `where Mary went`,
    `where did Mary go`."""
    wh_phrase = [index for index in kept if index <= wh
                 or (index in _subtree(words, words[wh].head)
                     and words[wh].dep == "det" and index < head)]
    subject = next((word for word in _children(words, head, SUBJECTS)), None)
    if subject is None or subject.index in wh_phrase:
        return kept
    subject_words = [index for index in kept
                     if index in _subtree(words, subject.index)]
    auxiliaries = [word.index for word in _children(words, head, AUXILIARIES)
                   if word.index in kept]
    rest = [index for index in kept if index not in wh_phrase
            and index not in subject_words and index not in auxiliaries
            and index != head]
    verb = words[head]
    if auxiliaries:
        return (wh_phrase + auxiliaries[:1] + subject_words
                + auxiliaries[1:] + [head] + rest)
    if lemma(verb.text.lower()) == "be":
        return wh_phrase + [head] + subject_words + rest
    return (wh_phrase + [DO.get(verb.tag, "did")] + subject_words
            + [lemma(verb.text.lower())] + rest)


def read_statement(words: list[Word], lemma, themes=None,
                   tags=None) -> Frame | None:
    root = next((word for word in words if word.dep == "ROOT"), None)
    if root is None or not root.verbal:
        return None
    frame = Frame(words, root.index)
    subjects = _children(words, root.index, SUBJECTS)
    dropped = _adjuncts(words, set(range(len(words))))
    if subjects:
        start = min(_subtree(words, subjects[0].index))
        for word in words:
            # Fronted before the subject: a phrase (`in the end`), not a word
            # that says whether (`maybe`), unless WordNet derives it.
            if word.index < start and word.head == root.index and (
                    word.dep == "prep" or (word.dep == "advmod"
                                           and derived_adverb(
                                               word.text.lower()))):
                dropped |= _subtree(words, word.index)
    order = [index for index in range(len(words)) if index not in dropped]
    passive = _children(words, root.index, {"nsubjpass"})
    auxpass = _children(words, root.index, {"auxpass"})
    if passive and auxpass and root.tag == "VBN":
        # Said in the tense it was told in: `was put` is `put`, never `had
        # put`, which would place it before what the story holds now.
        participle = root.text.lower()
        verb = (past_form(participle, lemma(participle), tags)
                if tags is not None and auxpass[0].tag == "VBD" else "")
        said_verb = ([verb] if verb else
                     [HAVE[auxpass[0].tag], root.index]
                     if auxpass[0].tag in HAVE else [])
        agent = _children(words, root.index, {"agent"})
        agent_object = (_children(words, agent[0].index, {"pobj"})
                        if agent else [])
        theme = [index for index in order
                 if index in _subtree(words, passive[0].index)]
        if agent_object and said_verb:
            doer = [index for index in order
                    if index in _subtree(words, agent_object[0].index)]
            rest = [index for index in order
                    if index not in theme and index not in doer
                    and index not in _subtree(words, agent[0].index)
                    and index != auxpass[0].index and index != root.index
                    and words[index].dep != "aux"]
            order = doer + said_verb + theme + rest
            frame.passive = True
        elif (verb and themes is not None
              and themes(lemma(root.text.lower()))):
            rest = [index for index in order
                    if index not in theme and index != auxpass[0].index
                    and index != root.index and words[index].dep != "aux"]
            order = theme + [verb] + rest
            frame.passive = True
    frame.adjuncts = sorted(dropped)
    frame.said = order
    return frame


def normal(tokens: list[str], typed: list[str], analysis, lemma,
           themes=None, tags=None,
           meaning=None) -> tuple[list[str], list[str]] | None:
    """The words said back in normal form, or None when the parse has
    nothing to change. `analysis` is (tag, dependency, head) per word, read
    over the words as typed; `lemma` a word's lemma; `themes(verb)` whether
    VerbNet has a frame with the verb's Theme as its subject; `tags(words)`
    the tagger's reading of words, for a passive's past tense."""
    if not analysis or len(analysis) != len(tokens):
        return None
    words = parse(typed, analysis)
    frame = read_question(words, lemma, meaning)
    if frame is None:
        frame = read_statement(words, lemma, themes, tags)
    if frame is None:
        return None
    said = frame.tokens(typed)
    return None if said[0] == tokens else said

"""What an utterance says about *which* thing, before anything is looked up.

v687 and v688 only ever hear about kinds: `does a beagle swim` is a question
about beagles. A conversation is about individuals -- *the* beagle, *that*
one, *the second* one, *it*, *me* -- and nothing in the words names which. This
module reads the words that pick one out, and leaves the picking to
`discourse.py`.

    there is a beagle          introduce    a new individual
    i have another beagle      introduce    a new one, and it is yours
    rex is a beagle            introduce    a new one, called Rex
    it can't swim              tell         something about one already here
    is the second one fast     ask          ... and a question about it
    my name is adrian          name         what you are called
    what is its name           ask_name
    who am i                   what         which kind it is
    does a beagle swim         generic      about beagles: v688's business

It reads a subject and, after the verb, at most one **object**: `does the cat
chase the dog` is about the cat and the dog, `it was in the plane` about it
and the plane. An object is resolved like a subject but never to the subject
itself -- `the pig is in it` does not put the pig inside the pig. An
indefinite object stays a kind -- `it chased a cat` is about cats -- except
after `in`, `on`, `inside` or `aboard`: `it was in an airplane` puts one
particular airplane on the table, because what carried it is a thing E2 has
to ask about.

## Names

A name is something you are *told* about an individual, like its colour. It is
never what makes it that individual: two beagles can both be Rex, and then
`does rex bark` asks which. Names are never looked up in the ontology --
WordNet's `adrian` is a physiologist.

The kind inside a phrase is found by v687's own parser (`lexicon.subject`),
which already knows that `a large dog` is about the dog and `fire truck` is
one concept. Everything else is string operations rather than regexes: three
patterns in this repository have had their escapes mangled in transit and
silently stopped matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v688.rephrase import rephrase

from . import clauses as coordination
from .relations import phrase as relation_phrase
from .tense import When, subordinate, take

#: Auxiliaries: what opens a yes/no question, and what a statement's verb
#: phrase can start with.
AUX = frozenset({"am", "is", "are", "was", "were", "can", "could", "does",
                 "do", "did", "has", "have", "had", "will", "would"})

#: Having: what a told `has` is read as, and what a perfect is made with.
HAVING = ("has", "have", "had")
COPULA = frozenset({"am", "is", "are", "was", "were"})

FIRST_PERSON = frozenset({"i", "me", "myself"})
#: The one being talked to: this program. Reported from the page -- `what is
#: your name` answered with the speaker's, because `your` was read as `my`.
SECOND_PERSON = frozenset({"you", "yourself"})
PRONOUNS = frozenset({"it", "he", "she", "him", "her"})
#: `they`: the individuals last talked about together (`discourse.py`), or,
#: when there are none, the kind last named (`session._they`)
PLURAL = frozenset({"they", "them"})
DEMONSTRATIVES = frozenset({"this", "that"})
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "last": -1}
RELATIVE = frozenset({"that", "who", "which"})
ARTICLES = frozenset({"a", "an", "the"})

#: Being inside or on something: what E2 reads as being carried, and the one
#: place an indefinite object is a new individual rather than a kind.
CARRYING = ("in", "on", "inside", "aboard")

#: Words that open a sentence and are never a name, however capitalised:
#: `The cat is black` and `There was a pig` are not about someone called The.
NOT_NAMES = (FIRST_PERSON | SECOND_PERSON | PRONOUNS | PLURAL
             | DEMONSTRATIVES | ARTICLES
             | frozenset({"another", "there", "here", "my", "your", "what",
                          "who", "which", "its", "his", "her", "why", "how",
                          "where", "when", "whose", "whom", "if",
                          "whether"}))

#: Question words. `why does a dog bark` taught a kind called `why` that it
#: `does a dog bark` -- WordNet has a noun `why`, so it was not refused as an
#: unknown word -- and `how many legs does a spider have` taught one called
#: `how many leg`. An utterance opening with one is a question, read whole.
QUESTION_WORDS = frozenset({"what", "who", "whom", "whose", "which", "why",
                            "how", "where", "when"})

#: Modal openers. `should a dog eat chocolate` is not a claim to store or a
#: question about one individual: it goes to v688 as said, where R18 names it.
MODALS = frozenset({"should", "must", "may", "might", "shall", "ought"})

#: `is it safe to ...`, `is it likely that ...`: an `it` that refers to
#: nothing, followed by one of these.
EXPLETIVE = frozenset({"to", "that", "for"})

#: Ways of putting a new individual on the table: the words before the
#: indefinite article in `there is a beagle` and `i have another beagle`.
INTRODUCERS = (("there", "is"), ("there", "was"), ("here", "is"),
               ("i", "have"), ("i", "see"), ("i", "saw"), ("i", "met"),
               ("i", "got"), ("i", "found"), ("we", "have"), ("meet",))

#: The introducers that make what they introduce yours: `i have a beagle`
#: is `my beagle` afterwards, `i saw a beagle` is not.
OWNING = frozenset({("i", "have"), ("i", "got"), ("we", "have")})

#: Contractions, expanded before anything is read, so negation has one
#: spelling everywhere below.
CONTRACTIONS = {"can't": ["can", "not"], "cannot": ["can", "not"],
                "won't": ["will", "not"], "doesn't": ["does", "not"],
                "couldn't": ["could", "not"], "wouldn't": ["would", "not"],
                "weren't": ["were", "not"],
                "don't": ["do", "not"], "didn't": ["did", "not"],
                "hadn't": ["had", "not"],
                "isn't": ["is", "not"], "aren't": ["are", "not"],
                "wasn't": ["was", "not"], "hasn't": ["has", "not"],
                "haven't": ["have", "not"], "there's": ["there", "is"],
                "here's": ["here", "is"], "it's": ["it", "is"],
                "that's": ["that", "is"], "i've": ["i", "have"],
                "i'm": ["i", "am"], "he's": ["he", "is"],
                "she's": ["she", "is"], "what's": ["what", "is"],
                "who's": ["who", "is"], "you're": ["you", "are"]}

#: The typographic apostrophe, spelled without an escape sequence.
CURLY_APOSTROPHE = chr(8217)

PUNCTUATION = ".,;:!?()" + chr(34)


def article(word: str) -> str:
    return "an" if (word or "")[:1].lower() in "aeiou" else "a"


def tokens_of(text: str) -> tuple[list[str], list[str]]:
    """(lowercased words, the same words as typed), contractions expanded.

    The typed form is kept only because a name is spelled the way it was
    said: `I am Adrian` is a name, `i am tired` is not.
    """
    lower: list[str] = []
    typed: list[str] = []
    for raw in (text or "").replace(CURLY_APOSTROPHE, "'").split():
        piece = raw.strip(PUNCTUATION)
        if not piece:
            continue
        expanded = CONTRACTIONS.get(piece.lower())
        if expanded:
            lower.extend(expanded)
            typed.extend(expanded)
        else:
            lower.append(piece.lower())
            typed.append(piece)
    return lower, typed


def words(text: str) -> list[str]:
    """Lowercased words, contractions expanded, punctuation dropped."""
    return tokens_of(text)[0]


def proper(pieces: list[str]) -> str:
    """A name as it will be shown: `adrian` -> `Adrian`."""
    name = " ".join(pieces).strip()
    return name[:1].upper() + name[1:]


@dataclass
class Mention:
    """A phrase that picks out an individual, or puts a new one down."""

    #: pronoun | demonstrative | definite | ordinal | other | another |
    #: indefinite | speaker | addressee | name | possessive | kind |
    #: group (`mary and daniel`) | plural (`they`) | individual (one already
    #: resolved, by id in `name`)
    form: str
    kind: str = ""                # "beagle"; empty for `it`, `the second one`
    ordinal: int | None = None    # 2 for `the second`, -1 for `the last`
    #: `black` in `the black one`: a description the referent must fit
    modifiers: list[str] = field(default_factory=list)
    text: str = ""
    end: int = 0                  # index of the first word after it
    name: str = ""                # `rex`, for a mention by name
    #: the mentions a group is made of: `mary` and `daniel`
    members: list = field(default_factory=list)
    #: a name nobody here has been called yet (`new_names`): someone new
    fresh: bool = False

    def as_dict(self) -> dict:
        return {"form": self.form, "kind": self.kind,
                "ordinal": self.ordinal, "modifiers": list(self.modifiers),
                "text": self.text, "name": self.name,
                "members": [one.as_dict() for one in self.members],
                "fresh": self.fresh}


@dataclass
class Reading:
    """What an utterance does, and to whom."""

    #: introduce | tell | ask | what | name | ask_name | teach | generic |
    #: define | compound, for several claims the parse could not tell apart
    act: str
    mention: Mention | None = None
    aux: str | None = None        # `can` in `it can't swim`; None for `it barks`
    rest: list[str] = field(default_factory=list)   # `swim`
    holds: bool = True            # False for `can't`, `has no`
    #: `there is a beagle that can't swim`: the clause, read as a statement
    relative: "Reading | None" = None
    said: str = ""
    name: str = ""                # what it is called, for `name` and `rex is a`
    owned: bool = False           # `i have a beagle`: the beagle is yours
    #: `the dog` in `it chased the dog`: the other individual, and the index
    #: in `rest` where its phrase starts
    obj: Mention | None = None
    obj_at: int = -1
    #: `and expand in warm ones`: the claims joined to this one, each read
    #: as its own statement
    more: list = field(default_factory=list)
    #: what it says about when (`tense.py`)
    when: When | None = None
    #: `how many objects is Mary carrying`: a count, not a list
    count: bool = False
    #: (asked, relation): the goal cell a question fills (`goals.py`), which
    #: the session answers by cell rather than by the act's name
    cell: tuple | None = None

    def as_dict(self) -> dict:
        return {"act": self.act,
                "mention": self.mention.as_dict() if self.mention else None,
                "aux": self.aux, "rest": " ".join(self.rest),
                "holds": self.holds, "name": self.name, "owned": self.owned,
                "object": self.obj.as_dict() if self.obj else None,
                "more": [one.as_dict() for one in self.more],
                "relative": (self.relative.as_dict() if self.relative
                             else None),
                "when": self.when.as_dict() if self.when else None}


def _stops(word: str) -> bool:
    """Where a noun phrase certainly ends: a verb, a negation, a clause."""
    return word in AUX or word == "not" or word in RELATIVE


def noun_phrase(tokens: list[str], start: int, lexicon, opener: str = "does",
                final_ok: bool = False):
    """(modifiers, kind, end) for the phrase beginning at `start`, or None.

    `one` closes a phrase with no kind in it -- `the second one`, `the black
    one`. Otherwise the kind is whatever v687's parser reads a question about
    this phrase as being about, which is how `black beagle` comes back as a
    beagle described as black and not as the colour.

    Unless `final_ok`, a phrase that runs to the end of the utterance is
    refused: `is that black` has nothing left to ask once `that black` is
    taken as a thing, so `that` has to be the thing.
    """
    for offset in range(min(3, len(tokens) - start)):
        word = tokens[start + offset]
        if _stops(word):
            break
        if word == "one":
            end = start + offset + 1
            if not final_ok and end >= len(tokens):
                return None
            return tokens[start:start + offset], "", end
    if start >= len(tokens):
        return None
    probe = f"{opener if opener in AUX else 'does'} a " \
            f"{' '.join(tokens[start:])}"
    subject = (lexicon.subject(probe) or "").lower()
    if not subject:
        return None
    first, length = subject.split()[0], len(subject.split())
    for index in range(start, min(start + 3, len(tokens))):
        word = tokens[index]
        if _stops(word):
            break
        if word == first or lexicon.lemma(word) == first:
            end = index + length
            # `the box of chocolates`: a thing of its own, not a box. `of`
            # and a plural, with nothing more of the phrase after it.
            if (tokens[end:end + 1] == ["of"] and end + 1 < len(tokens)
                    and tokens[end + 1] not in ARTICLES
                    and lexicon.lemma(tokens[end + 1]) != tokens[end + 1]
                    and tokens[end + 2:end + 3] != ["of"]):
                subject, end = f"{subject} of {tokens[end + 1]}", end + 2
            if not final_ok and end >= len(tokens):
                return None
            return tokens[start:index], subject, end
    return None


def read_mention(tokens: list[str], at: int, lexicon, opener: str = "does",
                 final_ok: bool = False,
                 names: frozenset = frozenset()) -> Mention | None:
    """The referring expression starting at `at`, if there is one."""
    if at >= len(tokens):
        return None
    word = tokens[at]
    after = tokens[at + 1] if at + 1 < len(tokens) else ""

    def mention(form: str, found, skip: int, ordinal=None) -> Mention:
        modifiers, kind, end = found or ([], "", at + skip)
        return Mention(form, kind, ordinal, list(modifiers),
                       " ".join(tokens[at:end]), end)

    if word in FIRST_PERSON:
        return Mention("speaker", text=word, end=at + 1)
    if word in SECOND_PERSON:
        return Mention("addressee", text=word, end=at + 1)
    if word in names:
        return Mention("name", text=word, end=at + 1, name=word)
    if word in PRONOUNS:
        return Mention("pronoun", text=word, end=at + 1)
    if word in PLURAL:
        return Mention("plural", text=word, end=at + 1)
    if word == "my":
        found = noun_phrase(tokens, at + 1, lexicon, opener, final_ok)
        return mention("possessive", found, 1) if found and found[1] else None
    if word in DEMONSTRATIVES:
        found = noun_phrase(tokens, at + 1, lexicon, opener, final_ok)
        if found:
            return mention("demonstrative", found, 1)
        return Mention("pronoun", text=word, end=at + 1)
    if word == "the":
        if after in ORDINALS:
            found = noun_phrase(tokens, at + 2, lexicon, opener, final_ok)
            return mention("ordinal", found, 2, ORDINALS[after])
        if after == "other":
            found = noun_phrase(tokens, at + 2, lexicon, opener, final_ok)
            return mention("other", found, 2)
        found = noun_phrase(tokens, at + 1, lexicon, opener, final_ok)
        return mention("definite", found, 1) if found else None
    if word in ("a", "an", "another"):
        found = noun_phrase(tokens, at + 1, lexicon, opener, final_ok)
        if found and found[1]:
            return mention("another" if word == "another" else "indefinite",
                           found, 1)
    return None


def clause(tokens: list[str]) -> Reading | None:
    """A verb phrase as a statement: `can not swim`, `has no tail`, `barks`."""
    if not tokens:
        return None
    aux, at = (tokens[0], 1) if tokens[0] in AUX else (None, 0)
    holds = True
    if at < len(tokens) and tokens[at] == "not":
        holds, at = False, at + 1
    elif aux in COPULA and tokens[at:at + 2] == ["no", "longer"]:
        # `it is no longer in the bedroom`: not there, as `not` says. That it
        # was is what the story already holds, or nothing does.
        holds, at = False, at + 2
    elif aux in HAVING and at < len(tokens) and tokens[at] == "no":
        holds, at = False, at + 1
    rest = tokens[at:]
    # `got the football there`: `there` is where the subject already is, and
    # the object is what closes the clause.
    if len(rest) > 1 and rest[-1] == "there":
        rest = rest[:-1]
    if not rest:
        return None
    return Reading("tell", aux=aux, rest=rest, holds=holds)


def object_of(aux: str | None, rest: list[str], lexicon,
              names: frozenset = frozenset()):
    """(mention, index) for the individual after the verb, or None.

    The phrase has to close the utterance and follow something: a verb
    (`chased the dog`), a preposition of carrying (`was in the plane`), or
    `has` itself (`has my hat`). After a bare copula nothing is an object --
    `it is a dog` says what it is. A bare `that` is not one either: `it can do
    that` points at a doing, not a thing.
    """
    first = 0 if aux in HAVING else 1
    for at in range(first, len(rest)):
        carried = at > 0 and rest[at - 1] in CARRYING
        # `the kitchen is north of the office`: the one a relation is to
        # (`relations.py`), after a copula as after a verb.
        opened = relation_phrase(rest[:at]) if at else None
        related = opened is not None and opened.length == at
        if aux in COPULA and not (carried or related):
            continue
        if rest[at] in DEMONSTRATIVES and at + 1 == len(rest):
            continue
        found = read_mention(rest, at, lexicon, "does", final_ok=True,
                             names=names)
        if found is None or found.end != len(rest):
            continue
        if found.form in ("indefinite", "another") and not carried:
            continue
        return found, at
    return None


def _with_object(reading: Reading | None, lexicon, names: frozenset):
    """The same reading, with its object found if it has one."""
    if reading is not None and reading.rest:
        found = object_of(reading.aux, reading.rest, lexicon, names)
        if found is not None:
            reading.obj, reading.obj_at = found
    return reading


def _whose(phrase: list[str], lexicon, names: frozenset) -> Mention | None:
    """Whose name: `my`, `its`, `the dog's`, `the second beagle's`."""
    if phrase == ["my"]:
        return Mention("speaker", text="my", end=1)
    if phrase == ["your"]:
        return Mention("addressee", text="your", end=1)
    if len(phrase) == 1 and phrase[0] in ("its", "his", "her"):
        return Mention("pronoun", text=phrase[0], end=1)
    if phrase and phrase[-1].endswith("'s"):
        stem = phrase[:-1] + [phrase[-1][:-2]]
        found = read_mention(stem, 0, lexicon, "is", final_ok=True,
                             names=names)
        if (found and found.end == len(stem)
                and found.form not in ("indefinite", "another")):
            found.text = " ".join(phrase)
            return found
    return None


def naming(tokens: list[str], typed: list[str], lexicon,
           names: frozenset) -> Reading | None:
    """`my name is adrian`, `call me adrian`, `rex is a beagle`, and asking.

    Read before anything else, because every one of them otherwise reads as
    something it is not: `my name is adrian` went to v688 as a claim about a
    kind and came back `is a my name Adrian`.
    """
    if (len(tokens) >= 4 and tokens[:2] == ["what", "is"]
            and tokens[-1] == "name"):
        whose = _whose(tokens[2:-1], lexicon, names)
        if whose:
            return Reading("ask_name", whose)

    if "name" in tokens:
        at = tokens.index("name")
        if (at + 2 < len(tokens) and tokens[at + 1] in COPULA
                and tokens[at + 2] != "not"):
            whose = _whose(tokens[:at], lexicon, names)
            if whose:
                return Reading("name", whose, name=proper(typed[at + 2:]))

    if tokens[:2] == ["call", "me"] and len(tokens) > 2:
        return Reading("name", Mention("speaker", text="me", end=2),
                       name=proper(typed[2:]))

    for verb in ("called", "named"):
        if verb not in tokens:
            continue
        at = tokens.index(verb)
        if at >= 2 and tokens[at - 1] in COPULA and at + 1 < len(tokens):
            who = read_mention(tokens[:at - 1], 0, lexicon, "is",
                               final_ok=True, names=names)
            if (who and who.end == at - 1
                    and who.form not in ("indefinite", "another")):
                return Reading("name", who, name=proper(typed[at + 1:]))

    # `I am Adrian`: a capitalised word after `i am`. Capitalisation is the
    # only evidence there is -- `i am tired` is not a name -- so a name typed
    # in lower case here is read as a quality, and `my name is` still works.
    if (tokens[:2] == ["i", "am"] and len(tokens) == 3
            and typed[2][:1].isupper()):
        return Reading("name", Mention("speaker", text="i", end=1),
                       name=proper(typed[2:]))

    # `Rex is a beagle`: a capitalised word nobody has used yet, said to be
    # a kind the ontology has. A new individual, told its name.
    if (len(tokens) >= 4 and tokens[1] in ("is", "was")
            and tokens[2] in ("a", "an") and typed[0][:1].isupper()
            and tokens[0] not in names and tokens[0] not in NOT_NAMES):
        kind = " ".join(tokens[3:])
        if lexicon.known(kind):
            return Reading(
                "introduce",
                Mention("indefinite", kind, text=" ".join(tokens[2:]),
                        end=len(tokens)),
                name=proper(typed[:1]))
    return None


def bare_kind(tokens: list[str], at: int, lexicon) -> Mention | None:
    """A kind named with nothing v687 can place: `can a wemble fly`, `do
    wembles fly`.

    v687's parser reads `can a wemble fly` as being about `fly`, because
    `wemble` is not a word it has. One word after an article, or a plural
    with none, is taken as the kind, and the rest is what is asked of it.
    """
    if at >= len(tokens) - 1:
        return None
    word = tokens[at]
    if word in ("a", "an"):
        if at + 2 >= len(tokens):
            return None
        return Mention("kind", tokens[at + 1],
                       text=" ".join(tokens[at:at + 2]), end=at + 2)
    if word in NOT_NAMES or word in AUX or word in ORDINALS:
        return None
    lemma = lexicon.lemma(word)
    if lemma == word and not lexicon.known(word):
        return None
    return Mention("kind", lemma, text=word, end=at + 1)


#: Penn tags for a verb said in the present tense: `dogs bark`, `a wemble
#: glows`. A past tense is something that happened -- `a dog chased me` -- and
#: not a claim about dogs.
PRESENT = frozenset({"VB", "VBP", "VBZ"})


def tags_of(tokens: list[str], lexicon) -> list[str] | None:
    """The tagger's reading of every word, in the context of the whole
    utterance, or None if the lexicon has no tagger.

    In context, because a word alone tells nothing: in `they ___ it` spaCy
    tags `bones`, `cold` and `temperatures` as verbs. In `dogs eat meat and
    bones` it knows `bones` is a noun.
    """
    tag = getattr(lexicon, "tags", None)
    found = tag(tokens) if tag is not None and tokens else None
    return list(found) if found and len(found) == len(tokens) else None


def generic_claim(tokens: list[str], lexicon,
                  names: frozenset = frozenset()) -> Reading | None:
    """`a wemble is a kind of animal`, `beagles can't swim`: a claim about a
    kind, for episodic memory, rather than a question for v688.

    The subject is a kind rather than an individual: an indefinite article,
    or a plural or a kind the ontology has with no determiner at all. A bare
    singular word the ontology does not have is refused -- `Adrian can swim`
    is about someone, not a kind of thing -- and so is a told name.
    """
    tags = tags_of(tokens, lexicon)
    found = [next((index for index, word in enumerate(tokens)
                   if index and word in AUX), None)]
    if tags is not None:
        # `testicles shrink in the cold`: no auxiliary, only a verb in the
        # present after at most three words of kind.
        found.append(next((index for index in range(1, min(4, len(tokens)))
                           if tags[index] in PRESENT), None))
    found = [index for index in found if index is not None]
    if (not found and len(tokens) >= 2 and tokens[0] not in NOT_NAMES
            and tokens[0] not in names and tokens[1] not in AUX
            and tokens[1] not in frozenset({"and", "or"}) | ARTICLES):
        # The tagger reads `dogs bark` as two proper nouns. v687's parser,
        # asked `does a dog bark`, finds the dog, and that is enough.
        lemma = lexicon.lemma(tokens[0])
        probe = f"does a {lemma} {' '.join(tokens[1:])}"
        if (lemma != tokens[0]
                and (lexicon.subject(probe) or "").lower() == lemma):
            found = [1]
    if not found:
        return None
    at = min(found)
    head = tokens[:at]
    led = head[0] in ("a", "an")
    if led:
        head = head[1:]
    if (not head or len(head) > 3 or head[0] in NOT_NAMES
            or head[0] in ORDINALS or head[0] in names
            or any(word.endswith("'s") for word in head)):
        return None
    kind = " ".join(head)
    if not led:
        lemma = lexicon.lemma(head[-1])
        kind = " ".join(head[:-1] + [lemma])
        if lemma == head[-1] and not lexicon.known(kind):
            return None
    body = clause(tokens[at:])
    if body is None:
        return None
    body.act = "teach"
    body.mention = Mention("kind", kind, text=" ".join(tokens[:at]), end=at)
    return body


def _analysis(tokens: list[str], lexicon):
    """(tag, dependency, head) per word, from the lexicon's parser, or None."""
    analyse = getattr(lexicon, "analyse", None)
    found = analyse(tokens) if analyse is not None and tokens else None
    return list(found) if found and len(found) == len(tokens) else None


def _looks_compound(tokens: list[str], lexicon) -> bool:
    """More than one claim, in words the parse could not split.

    A coordinator followed by something that can open a claim: a pronoun, or
    the plural of a kind the ontology has, with words after it. Only asked
    when the parse had no verb at its root; where it did, the parse decides.
    """
    for at in range(1, len(tokens) - 2):
        if tokens[at] not in ("and", "but", "or"):
            continue
        after = tokens[at + 1]
        if after in coordination.PRONOUNS | {"i", "we", "you"}:
            return True
        lemma = lexicon.lemma(after)
        if lemma != after and lexicon.known(lemma):
            return True
    return False


def _several(parts: list, typed: list[str], said: str, lexicon,
             names: frozenset) -> Reading:
    """Read each clause as its own statement; the rest ride on the first.

    A clause that goes on about the kind just taught -- `and expand in warm
    ones`, `but they can run` -- is built straight onto that kind, because
    `read` would have to find a kind in words that do not name it. Any other
    clause is filled in (`clauses.standalone`) and read from the start.
    """
    first = read(" ".join(parts[0].words(typed)), lexicon, names)
    readings = [first]
    before, reading_before = parts[0], first
    for part in parts[1:]:
        kind = (reading_before.act == "teach"
                and reading_before.mention is not None)
        whole = coordination.standalone(part, before, kind)
        if coordination.continues(part, kind):
            tail = [word.text for word in whole.aux] + (
                ["not"] if whole.negated else []) + [
                typed[word.index].lower() for word in whole.rest]
            one = clause(tail)
            if one is None:
                continue
            one.act, one.mention = "teach", reading_before.mention
        else:
            one = read(" ".join(whole.words(typed)), lexicon, names)
        readings.append(one)
        before, reading_before = whole, one
    for one in readings:
        one.said = said
    first.more = readings[1:]
    return first


#: What a count of individuals ends with: `how many dogs are there`.
COUNT_ENDS = (("are", "there"), ("is", "there"), ("there", "are"),
              ("have", "come", "up"), ("has", "come", "up"))

#: Words after `how many` that count kinds of a thing, which R25 answers.
KIND_WORDS = frozenset({"kind", "kinds", "type", "types", "sort", "sorts",
                        "breed", "breeds", "species"})

#: What a question about a quality toward something ends with: `what is
#: Gertrude afraid of`.
TOWARD = frozenset({"of", "to", "about", "with", "for", "at", "by"})

#: What a location question may end with: `what is the cat on`.
PLACES = frozenset({"on", "in", "inside", "under", "at", "near", "beside",
                    "behind", "above", "below"})

#: Words that pick one out of a sequence: `what did it do second`.
SEQUENCE = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "next": -1, "last": -1}


def _individual(found) -> bool:
    """A phrase naming one of this conversation's individuals, rather than a
    kind or a new one."""
    return found is not None and found.form not in ("indefinite", "another",
                                                    "kind", "plural", "group")


def _cell(reading: Reading, asked: str, relation: str) -> Reading:
    """The same reading, with the goal cell it fills: what is asked of which
    relation (`goals.py`)."""
    reading.cell = (asked, relation)
    return reading


def conversational(tokens: list[str], lexicon, names: frozenset,
                   said: str) -> Reading | None:
    """A wh-question about this conversation's individuals, or None.

        how many dogs are there        how_many   count them
        which dog is black             which      identify one by description
        who chased the cat             who        identify the doer
        where is the dog               where      what it was told to be in
        what is the cat on             where
        what did the dog chase         what_did   the object of what it did
        what can it do                 about      what was told, and its kind
        what do you know about it      about
        what happened                  happened   what was told, in order
        what did it do first           happened

    Each is answered from episodic memory, by the same identification that
    finds `the black one`: a walk down the trie for what the question names.
    A question about a kind -- `what can a dog do`, `how many legs does a
    spider have` -- is left for v688.
    """
    if len(tokens) < 2:
        return None
    first = tokens[0]

    # `when did the dog chase the cat`, `how many times did it bark`: an
    # occurrence found, and its time or its count read out (`timeline.py`).
    for opener, act, asked in ((["when"], "when", "time"),
                               (["how", "many", "times"], "how_many_times",
                                "times"),
                               (["how", "often"], "how_many_times", "times")):
        at = len(opener)
        if tokens[:at] != opener or len(tokens) < at + 3 \
                or tokens[at] not in AUX:
            continue
        found = read_mention(tokens, at + 1, lexicon, tokens[at],
                             names=names)
        if _individual(found) and found.end < len(tokens):
            return _cell(_with_object(Reading(act, found, tokens[at],
                                              tokens[found.end:], said=said),
                                      lexicon, names), asked, "occurrence")
        return None

    # `what was the dog doing`, `what is it doing`
    if (first == "what" and len(tokens) >= 4 and tokens[1] in COPULA
            and tokens[-1] == "doing"):
        found = read_mention(tokens[:-1], 2, lexicon, tokens[1],
                             final_ok=True, names=names)
        if _individual(found) and found.end == len(tokens) - 1:
            return Reading("doing", found, tokens[1], ["doing"], said=said)
        return None

    # `what is Mary carrying`, `how many objects is Mary carrying`: what is
    # with someone -- if the verb means that, which the session asks VerbNet
    # (`change.accompanies`).
    counting = tokens[:2] == ["how", "many"]
    at = 3 if counting else 1
    if ((first == "what" or counting) and len(tokens) >= at + 3
            and tokens[at] in COPULA and tokens[-1].endswith("ing")
            and hasattr(lexicon, "progressive")):
        found = read_mention(tokens[:-1], at + 1, lexicon, tokens[at],
                             final_ok=True, names=names)
        verb = lexicon.progressive(tokens[-1])
        if _individual(found) and found.end == len(tokens) - 1 and verb:
            return _cell(Reading("carrying", found, tokens[at], [verb],
                                 said=said, count=counting,
                                 obj=(Mention("kind", tokens[2],
                                              text=tokens[2])
                                      if counting else None)),
                         "count" if counting else "object", "holding")

    # `what is north of the office`, `what is the kitchen north of`: one side
    # of a relation asked, the other named (`relations.py`). `?` in `rest`
    # marks the side asked.
    if first == "what" and len(tokens) >= 4 and tokens[1] in COPULA:
        opened = relation_phrase(tokens[2:])
        if opened is not None:
            found = read_mention(tokens, 2 + opened.length, lexicon, "is",
                                 final_ok=True, names=names)
            if _individual(found) and found.end == len(tokens):
                return Reading("related", found, tokens[1],
                               ["?"] + tokens[2:2 + opened.length],
                               said=said)
        found = read_mention(tokens, 2, lexicon, "is", final_ok=True,
                             names=names)
        closed = relation_phrase(tokens[found.end:]) if found else None
        if (_individual(found) and closed is not None
                and found.end + closed.length == len(tokens)):
            return Reading("related", found, tokens[1],
                           tokens[found.end:] + ["?"], said=said)

    # `how do you go from the kitchen to the garden`: a way over the compass.
    if (first == "how" and tokens[1] in AUX and "from" in tokens
            and "to" in tokens[tokens.index("from"):]):
        start = tokens.index("from")
        there = read_mention(tokens, start + 1, lexicon, "is", final_ok=True,
                             names=names)
        if (_individual(there) and there.end < len(tokens)
                and tokens[there.end] == "to"):
            goal = read_mention(tokens, there.end + 1, lexicon, "is",
                                final_ok=True, names=names)
            if _individual(goal) and goal.end == len(tokens):
                asked = Reading("route", there, tokens[1],
                                tokens[3:start], said=said)
                asked.obj = goal
                return asked

    # `what is Gertrude afraid of`: what a quality toward something is toward.
    if (first == "what" and len(tokens) >= 5 and tokens[1] in COPULA
            and tokens[-1] in TOWARD):
        found = read_mention(tokens[:-2], 2, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens) - 2:
            return Reading("toward", found, tokens[1], tokens[-2:],
                           said=said)

    # `what color is Greg`: one of its attributes, the kind of value asked.
    if (first == "what" and len(tokens) >= 4 and tokens[2] in COPULA
            and tokens[1] not in AUX and tokens[1] not in QUESTION_WORDS):
        found = read_mention(tokens, 3, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens):
            return Reading("attribute", found, tokens[2], [tokens[1]],
                           said=said)

    # `where will Sumit go`: where someone is going, which nothing has told.
    if (first == "where" and len(tokens) == 4 and tokens[1] == "will"):
        found = read_mention(tokens, 2, lexicon, "will", final_ok=True,
                             names=names)
        if _individual(found) and found.end == 3:
            return Reading("where_going", found, "will", tokens[3:],
                           said=said)

    # `who did Fred give the football to`: the one it went to.
    if (first in ("who", "whom") and len(tokens) >= 5 and tokens[1] in AUX
            and tokens[-1] in ("to", "from")):
        found = read_mention(tokens, 2, lexicon, tokens[1], names=names)
        if _individual(found) and found.end < len(tokens) - 1:
            asked = _with_object(Reading("to_whom", found, tokens[1],
                                         tokens[found.end:-1], said=said),
                                 lexicon, names)
            asked.rest = asked.rest + [tokens[-1]]
            return _cell(asked, "recipient", "occurrence")

    if tokens[:2] == ["how", "many"]:
        end = next((len(one) for one in COUNT_ENDS
                    if tuple(tokens[-len(one):]) == one), 0)
        middle = tokens[2:len(tokens) - end] if end else []
        if not end or (middle and middle[0] in KIND_WORDS):
            return None
        kind = (" ".join(middle[:-1] + [lexicon.lemma(middle[-1])])
                if middle else "")
        return Reading("how_many", Mention("kind", kind, text=" ".join(middle),
                                           end=2 + len(middle)), said=said)

    if first == "which" and len(tokens) >= 3:
        at, kind = 1, ""
        if tokens[1] not in AUX:
            kind = "" if tokens[1] == "one" else lexicon.lemma(tokens[1])
            at = 2
        aux = tokens[at] if tokens[at] in AUX else None
        rest = tokens[at + 1:] if aux else tokens[at:]
        holds = rest[:1] != ["not"]
        rest = rest if holds else rest[1:]
        if not rest:
            return None
        return _with_object(
            Reading("which", Mention("kind", kind, text=" ".join(tokens[1:at]),
                                     end=at), aux, rest, holds=holds,
                    said=said), lexicon, names)

    if first in ("who", "whom") and tokens[1] not in COPULA:
        aux = tokens[1] if tokens[1] in AUX else None
        rest = tokens[2:] if aux else tokens[1:]
        holds = rest[:1] != ["not"]
        rest = rest if holds else rest[1:]
        if not rest:
            return None
        return _cell(_with_object(Reading("who", None, aux, rest, holds=holds,
                                          said=said), lexicon, names),
                     "subject", "occurrence")

    if first == "where" and tokens[1] in COPULA:
        found = read_mention(tokens, 2, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens):
            return _cell(Reading("where", found, said=said),
                         "place", "located")
        # `where was the football before the bathroom`: where it was just
        # before it came to be there (`story.where_around`).
        if (_individual(found) and len(tokens) > found.end + 1
                and tokens[found.end] in ("before", "after")):
            return _cell(Reading("where", found, tokens[1],
                                 tokens[found.end:], said=said),
                         "place", "located")
        return None

    if first == "what" and tokens[1] in COPULA and tokens[-1] in PLACES:
        found = read_mention(tokens[:-1], 2, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens) - 1:
            return _cell(Reading("where", found, said=said),
                         "place", "located")
        return None

    # `what will happen tomorrow`: what is to come.
    if tokens[:3] in (["what", "will", "happen"],
                      ["what", "is", "going"]) and (
            tokens[1] == "will" or tokens[3:5] == ["to", "happen"]):
        at = 3 if tokens[1] == "will" else 5
        return Reading("happened", rest=["future"] + [
            word for word in tokens[at:] if word in SEQUENCE], said=said)

    if tokens[:2] == ["what", "happened"] or tokens[:3] == ["what", "has",
                                                            "happened"]:
        at = 2 if tokens[1] == "happened" else 3
        # `what happened to the vase`: what it took part in.
        if tokens[at:at + 1] == ["to"]:
            found = read_mention(tokens, at + 1, lexicon, "is", final_ok=True,
                                 names=names)
            if _individual(found) and found.end == len(tokens):
                return Reading("happened", found, None, ["to"], said=said)
        return Reading("happened", rest=[word for word in tokens[at:]
                                         if word in SEQUENCE], said=said)

    # Before `what` + auxiliary, which would read `you` in `what do you
    # know about it` as the individual asked about.
    about = (5 if tokens[:5] == ["what", "do", "you", "know", "about"] else
             3 if tokens[:3] == ["tell", "me", "about"] else None)
    if about is not None:
        found = read_mention(tokens, about, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens):
            return Reading("about", found, said=said)
        return None

    # About the conversation itself: what was said, and how an answer was
    # reached. `what did i tell you` asked whether you told yourself things,
    # and `how do you know that` was refused as a question about method.
    for told in (["what", "did", "i", "tell", "you"],
                 ["what", "have", "i", "told", "you"],
                 ["what", "did", "i", "say"], ["what", "have", "i", "said"]):
        after = tokens[len(told):]
        if tokens[:len(told)] == told and (not after or (
                len(after) == 1 and after[0] in SEQUENCE)):
            # Telling time, not story time: the order it was said in.
            return Reading("happened", rest=["told"] + after, said=said)
    if tokens in (["how", "do", "you", "know"],
                  ["how", "do", "you", "know", "that"],
                  ["how", "sure", "are", "you"], ["are", "you", "sure"],
                  ["why", "do", "you", "think", "so"]):
        return Reading("meta", said=said)
    # `what about a cat`, `and a fish?`: the last question, of another kind.
    for opener in (["what", "about"], ["how", "about"], ["and"]):
        if tokens[:len(opener)] == opener and len(tokens) > len(opener):
            found = read_mention(tokens, len(opener), lexicon, "is",
                                 final_ok=True, names=names)
            if (found is not None and found.end == len(tokens)
                    and found.form in ("indefinite", "kind")):
                return Reading("ellipsis", found, said=said)

    if first == "what" and tokens[1] in AUX and tokens[1] not in COPULA:
        at, holds = (3, False) if tokens[2:3] == ["not"] else (2, True)
        found = read_mention(tokens, at, lexicon, tokens[1], names=names)
        # `what do you need to bake a cake`: `you` is anyone, not me.
        if not _individual(found) or found.form in ("speaker", "addressee"):
            return None
        rest = tokens[found.end:]
        if rest[:1] == ["do"] and rest[1:2] and rest[1] in SEQUENCE:
            return Reading("happened", found, tokens[1], rest[1:], said=said)
        # `what did the dog do`: what it did, not what it can do.
        if rest == ["do"] and tokens[1] == "did":
            return Reading("happened", found, tokens[1], [], said=said)
        if rest in (["do"], ["have"]):
            return Reading("about", found, tokens[1], rest, holds=holds,
                           said=said)
        if rest:
            return _cell(Reading("what_did", found, tokens[1], rest,
                                 holds=holds, said=said),
                         "object", "occurrence")
        return None

    if tokens[:3] == ["what", "kind", "of"] and "is" in tokens[3:]:
        at = tokens.index("is", 3) + 1
        found = read_mention(tokens, at, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens):
            return Reading("what", found, said=said)
    return None


#: Proper-noun tags: what a capitalised word is when it names someone.
PROPER = frozenset({"NNP"})


def new_names(tokens: list[str], typed: list[str], lexicon,
              names: frozenset = frozenset()) -> frozenset:
    """Names nobody here has been called yet, said as someone is talked about.

    `Mary moved to the bathroom`, `Bill gave the apple to Fred`: a capitalised
    word the tagger reads as a proper noun, or as a noun the ontology has no
    word for (`Sumit is tired`), names someone. Before this, `John went to
    the hallway` was about toilets, WordNet's first `john`. The words are
    tagged as typed, because capitalisation is the evidence.

    Only in a statement -- a question never puts anyone down -- and never in
    `Rex is a beagle`, which `naming` reads as someone new of that kind.
    """
    if (not tokens or tokens[0] in AUX or tokens[0] in QUESTION_WORDS
            or tokens[0] in MODALS):
        return frozenset()
    tags = tags_of(typed, lexicon)
    if tags is None:
        return frozenset()
    found = set()
    for at, (word, shown) in enumerate(zip(tokens, typed)):
        if (not shown[:1].isupper() or word in names or word in NOT_NAMES
                or word in AUX or word in QUESTION_WORDS):
            continue
        if (at == 0 and len(tokens) > 3 and tokens[1] in ("is", "was")
                and tokens[2] in ("a", "an")):
            continue
        if tags[at] in PROPER or (tags[at] == "NN"
                                  and not lexicon.known(word)):
            found.add(word)
    return frozenset(found)


def _mark_fresh(reading: "Reading", fresh: frozenset) -> None:
    """Say which of a reading's names are someone new."""
    waiting = [reading]
    while fresh and waiting:
        one = waiting.pop()
        if one is None:
            continue
        for mention in (one.mention, one.obj):
            for each in ([mention] + list(mention.members) if mention
                         else []):
                if each.form == "name" and each.name in fresh:
                    each.fresh = True
        waiting.extend([one.relative] + list(one.more))


def read(text: str, lexicon, names: frozenset = frozenset(),
         anchored: bool = True) -> Reading:
    """Read one utterance.

    `lexicon` needs `subject(question)`, `lemma(word)` and `known(phrase)`;
    v687's parser supplies all three. `names` is every name the conversation
    has been told, lowercased, so a mention of one is read as a mention.

    What it says about time is taken out first (`tense.py`) and kept on the
    reading as `when`. An anchor clause -- `after the dog chased the cat` --
    is only one if it reads as a statement about someone; otherwise the
    utterance is read whole.
    """
    said = (text or "").strip()
    # A request is read as the question inside it: `do you know if a dog can
    # swim` is `can a dog swim`, and `can't it swim` is `can it swim`.
    asked = rephrase(said)
    # Someone new can be named anywhere in a statement, and before a link or
    # a time word is taken off: `then Mary went to the kitchen`.
    fresh = new_names(*tokens_of(asked.text), lexicon, names)
    names = names | fresh
    split = subordinate(asked.text) if anchored else None
    if split is not None:
        anchor = read(split[2], lexicon, names, anchored=False)
        if anchor.act != "tell" or anchor.mention is None:
            split = None
    tokens, typed = tokens_of(split[0] if split else asked.text)
    tokens, typed, when = take(tokens, typed, names)
    if split is not None:
        when.relation, when.anchor = split[1], split[2]
    if split is not None or when.words:
        # Quoted without the words that placed it, except `again`, without
        # which three barks read as one said three times.
        when.main = " ".join(typed + list(when.again_words))
    found = _read(said, asked, tokens, typed, lexicon, names)
    _mark_fresh(found, fresh)
    found.when = when
    for one in found.more:
        if one.when is None or one.when.empty:
            one.when = When(frame=when.frame)
    return found


def _read(said: str, asked, tokens: list[str], typed: list[str], lexicon,
          names: frozenset) -> Reading:
    if not tokens:
        return Reading("generic", said=said)

    # An analogy, a condition, a modal question, or an `it` that refers to
    # nothing (`is it safe to eat a mushroom`): none is a claim to keep or a
    # question about an individual here. v688 reads it whole, and R18 names
    # what it cannot answer.
    if (asked.asking or tokens[0] in MODALS
            or (len(tokens) > 3 and tokens[:2] == ["is", "it"]
                and tokens[3] in EXPLETIVE)):
        return Reading("generic", said=said)

    # `why can't it fly`: the yes or no about one individual, asked so the
    # answer can say what it rests on. A bare `why` asks it of the last
    # answer. A why about a kind -- `why can't a penguin fly` -- was put as
    # its yes or no by `rephrase`, and v688 answers what it rests on.
    if tokens[0] == "why":
        rest = tokens[1:]
        if not rest or rest in (["not"], ["so"], ["is", "that"]):
            return Reading("why", said=said)
        if rest[0] in AUX:
            at = 3 if len(rest) > 1 and rest[1] == "not" else 2
            found = read_mention(tokens, at, lexicon, rest[0], names=names)
            if (found is not None
                    and found.form not in ("indefinite", "another", "kind")):
                return _with_object(
                    Reading("why", found, rest[0], tokens[found.end:],
                            holds=at == 2, said=said), lexicon, names)
        return Reading("generic", said=said)

    # `who chased the cat`, `where is the dog`, `how many dogs are there`:
    # about this conversation's individuals, answered from episodic memory.
    asked_here = conversational(tokens, lexicon, names, said)
    if asked_here is not None:
        return asked_here

    # A statement of several claims is read one claim at a time. Questions
    # are left whole: `can it swim and bark` asks one thing.
    if tokens[0] not in AUX and tokens[0] not in QUESTION_WORDS:
        analysis = _analysis(tokens, lexicon)
        if analysis is not None:
            parts = coordination.split(tokens, analysis)
            if parts is None and _looks_compound(tokens, lexicon):
                return Reading("compound", said=said)
            if parts is not None and len(parts) > 1:
                return _several(parts, typed, said, lexicon, names)

    named = naming(tokens, typed, lexicon, names)
    if named is not None:
        named.said = said
        return named

    if (tokens[0] in ("what", "who") and len(tokens) > 2
            and tokens[1] in ("is", "was", "am", "are")):
        tail = tokens[2:]
        if len(tail) > 1 and tail[0] == "the":
            # `what is the largest animal`, `what is the capital of france`:
            # a kind described, not an individual here. It was answered
            # `nothing here was said to be largest`; v688 reads it, and R18
            # names what it cannot answer.
            tags = tags_of(tokens, lexicon) or []
            after = tail[1:]
            superlative = any(
                word in ("most", "least")
                or (tags[index + 3] == "JJS" if len(tags) == len(tokens)
                    else word.endswith("est") and len(word) > 5)
                for index, word in enumerate(after))
            belongs = ("of" in after and after[-1] not in PRONOUNS
                       and after[after.index("of") + 1:][:1] != ["the"])
            if superlative or belongs:
                return Reading("generic", said=said)
        found = read_mention(tokens, 2, lexicon, "is", final_ok=True,
                             names=names)
        if (found and found.form not in ("indefinite", "another")
                and found.end == len(tokens)):
            return Reading("what", found, said=said)
        if (found and found.form == "indefinite" and found.kind
                and found.end == len(tokens)):
            # `what is a testicle`: a definition, which is retrieved once
            # and then answered from definitions memory.
            return Reading("define", found, said=said)
        return Reading("generic", said=said)

    for opener in INTRODUCERS:
        if tuple(tokens[:len(opener)]) != opener:
            continue
        found = read_mention(tokens, len(opener), lexicon, "is",
                             final_ok=True, names=names)
        start = len(opener)
        if (found is None and len(tokens) > start + 1
                and tokens[start] in ("a", "an", "another")):
            # A kind v687 has no word for: `there is a wemble`.
            found = Mention("another" if tokens[start] == "another"
                            else "indefinite", tokens[start + 1],
                            text=" ".join(tokens[start:start + 2]),
                            end=start + 2)
        if found and found.form in ("indefinite", "another"):
            tail = tokens[found.end:]
            relative = _with_object(
                clause(tail[1:]) if tail and tail[0] in RELATIVE else None,
                lexicon, names)
            return Reading("introduce", found, relative=relative, said=said,
                           owned=opener in OWNING)
        break

    if tokens[0] in AUX:
        found = read_mention(tokens, 1, lexicon, tokens[0], names=names)
        if found is None:
            found = bare_kind(tokens, 1, lexicon)
        if found is None:
            return Reading("generic", said=said)
        if found.form in ("indefinite", "another", "kind"):
            # About a kind. The kind travels with it now, so a kind this
            # conversation taught can be answered from episodic memory.
            return Reading("generic", found, tokens[0], tokens[found.end:],
                           said=said)
        return _with_object(Reading("ask", found, tokens[0],
                                    tokens[found.end:], said=said),
                            lexicon, names)

    taught = generic_claim(tokens, lexicon, names)
    if taught is not None:
        taught.said = said
        return taught

    found = read_mention(tokens, 0, lexicon, names=names)
    if found is None or found.form == "indefinite":
        return Reading("generic", said=said)
    # `Mary and Daniel went to the kitchen`: two named at once, and the claim
    # is told of each (`session._tell_each`).
    if (found.form == "name" and found.end + 1 < len(tokens)
            and tokens[found.end] == "and"):
        other = read_mention(tokens, found.end + 1, lexicon, names=names)
        if other is not None and other.form == "name":
            found = Mention("group", text=" ".join(tokens[:other.end]),
                            end=other.end, members=[found, other])
    body = _with_object(clause(tokens[found.end:]), lexicon, names)
    if found.form == "another":
        return Reading("introduce", found, relative=body, said=said)
    if body is None:
        return Reading("what", found, said=said)
    body.mention, body.said = found, said
    return body


# -- from an individual back to its kind ----------------------------------

def progressive(aux: str | None, rest: list[str], lexicon):
    """`was flying` -> (`does`, [`fly`]): an action in progress is something
    it does.

    Without this, `he was flying` was keyed as a quality -- `is fly`, like `is
    black` -- and `can the pig fly` never met it, so a pig you had watched fly
    was answered from what pigs do. `lexicon.progressive` decides whether the
    word is a verb in `it is ___`, so `is boring` stays a quality.
    """
    rest = list(rest)
    if aux in COPULA and rest and hasattr(lexicon, "progressive"):
        verb = lexicon.progressive(rest[0])
        if verb:
            return "does", [verb] + rest[1:]
    return aux, rest


def perfect(aux: str | None, rest: list[str], lexicon):
    """`had eaten` -> (`does`, [`eat`]): a perfect is something it did.

    Without this `it had eaten` was `has_part "eaten"`. `lexicon.participle`
    decides whether the word is a verb's participle, so `had a ball` and `has
    four legs` stay having.
    """
    rest = list(rest)
    if aux in HAVING and rest and hasattr(lexicon, "participle"):
        verb = lexicon.participle(rest[0])
        if verb:
            return "does", [verb] + rest[1:]
    return aux, rest


def mode_of(aux: str | None) -> str:
    """`can` asks what it is able to do; everything else, what it does."""
    return "can" if aux in ("can", "could") else "does"


def kind_question(aux: str | None, rest: list[str], kind: str,
                  lemma) -> str:
    """The question about the kind that a claim about one of them matches.

    `it can't swim` about a beagle is checked against `can a beagle swim`;
    `it barks` against `does a beagle bark`; `i am tired` against `is a person
    tired`.
    """
    who = f"{article(kind)} {kind}"
    tail = " ".join(rest)
    if aux in COPULA:
        return f"is {who} {tail}"
    if aux in ("can", "could"):
        return f"can {who} {tail}"
    if aux in HAVING:
        return f"does {who} have {tail}"
    if aux in ("does", "do", "did", "will", "would"):
        return f"does {who} {tail}"
    return " ".join(["does", who, lemma(rest[0])] + rest[1:])


def predicate_key(aux: str | None, rest: list[str], lemma) -> str:
    """One spelling for a predicate, so what was told can meet what is asked.

    `it swims`, `it can swim` and `can it swim` all key as `does swim`;
    `it has a tail` and `does it have a tail` as `have tail`. `can` and `does`
    share a key and are told apart by `mode_of` where it matters.
    """
    body = [word for word in rest if word not in ARTICLES]
    if aux in COPULA:
        head = "is"
    elif aux in HAVING:
        head = "have"
    else:
        head = "does"
    if head == "does" and body and body[0] in ("have", "has"):
        head, body = "have", body[1:]
    return " ".join([head] + [lemma(word) for word in body])

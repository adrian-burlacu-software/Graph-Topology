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

from . import clauses as coordination

#: Auxiliaries: what opens a yes/no question, and what a statement's verb
#: phrase can start with.
AUX = frozenset({"am", "is", "are", "was", "were", "can", "could", "does",
                 "do", "did", "has", "have", "will", "would"})
COPULA = frozenset({"am", "is", "are", "was", "were"})

FIRST_PERSON = frozenset({"i", "me", "myself"})
#: The one being talked to: this program. Reported from the page -- `what is
#: your name` answered with the speaker's, because `your` was read as `my`.
SECOND_PERSON = frozenset({"you", "yourself"})
PRONOUNS = frozenset({"it", "he", "she", "him", "her"})
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
NOT_NAMES = (FIRST_PERSON | SECOND_PERSON | PRONOUNS | DEMONSTRATIVES
             | ARTICLES
             | frozenset({"another", "there", "here", "my", "your", "what",
                          "who", "which", "its", "his", "her"}))

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
    #: indefinite | speaker | addressee | name | possessive | kind
    form: str
    kind: str = ""                # "beagle"; empty for `it`, `the second one`
    ordinal: int | None = None    # 2 for `the second`, -1 for `the last`
    #: `black` in `the black one`: a description the referent must fit
    modifiers: list[str] = field(default_factory=list)
    text: str = ""
    end: int = 0                  # index of the first word after it
    name: str = ""                # `rex`, for a mention by name

    def as_dict(self) -> dict:
        return {"form": self.form, "kind": self.kind,
                "ordinal": self.ordinal, "modifiers": list(self.modifiers),
                "text": self.text, "name": self.name}


@dataclass
class Reading:
    """What an utterance does, and to whom."""

    #: introduce | tell | ask | what | name | ask_name | teach | generic |
    #: compound, for several claims the parse could not tell apart
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

    def as_dict(self) -> dict:
        return {"act": self.act,
                "mention": self.mention.as_dict() if self.mention else None,
                "aux": self.aux, "rest": " ".join(self.rest),
                "holds": self.holds, "name": self.name, "owned": self.owned,
                "object": self.obj.as_dict() if self.obj else None,
                "more": [one.as_dict() for one in self.more],
                "relative": (self.relative.as_dict() if self.relative
                             else None)}


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
    elif aux in ("has", "have") and at < len(tokens) and tokens[at] == "no":
        holds, at = False, at + 1
    rest = tokens[at:]
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
    first = 0 if aux in ("has", "have") else 1
    for at in range(first, len(rest)):
        carried = at > 0 and rest[at - 1] in CARRYING
        if aux in COPULA and not carried:
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


def read(text: str, lexicon, names: frozenset = frozenset()) -> Reading:
    """Read one utterance.

    `lexicon` needs `subject(question)`, `lemma(word)` and `known(phrase)`;
    v687's parser supplies all three. `names` is every name the conversation
    has been told, lowercased, so a mention of one is read as a mention.
    """
    tokens, typed = tokens_of(text)
    said = (text or "").strip()
    if not tokens:
        return Reading("generic", said=said)

    # A statement of several claims is read one claim at a time. Questions
    # are left whole: `can it swim and bark` asks one thing.
    if tokens[0] not in AUX and tokens[0] not in ("what", "who"):
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
        found = read_mention(tokens, 2, lexicon, "is", final_ok=True,
                             names=names)
        if (found and found.form not in ("indefinite", "another")
                and found.end == len(tokens)):
            return Reading("what", found, said=said)
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
    if aux in ("has", "have"):
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
    elif aux in ("has", "have"):
        head = "have"
    else:
        head = "does"
    if head == "does" and body and body[0] in ("have", "has"):
        head, body = "have", body[1:]
    return " ".join([head] + [lemma(word) for word in body])

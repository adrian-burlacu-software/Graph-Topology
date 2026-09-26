"""What the two taking part have and know: told, and asked.

    you have a cat              a cat, introduced as mine
    do i have a beagle          yes, if you told me so -- or a kind of one
    do you have a dog           mine: I know what I have, so no is no
    what do you have            what you told me I have
    do you know john            whether anyone called John has come up
    you like cake               kept as mine
    do you like cake            yes if you told me so; I know my own mind
    have you ever seen a whale  I see nothing: I read what I am told

Read off the parse, not by the encoder: `you` or `i` is the subject, and
the verb's first sense is WordNet's possession (*have*, *own*) or it is
*know* with a name for its object. The encoder learned from `can you cut
bread with a knife` that `you` may be anyone, and read `do you have a dog`
as `do a dog have` -- a question about dogs.

**I know what I have; I do not know what you have.** What the program has
is what it was told it has, so anything it was not told of it does not
have. What the speaker has is only what they said, and anything else is
not known -- they may have a dog they never mentioned.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Who a pronoun is, as the one talking or the one talked to.
SPEAKER = frozenset({"i", "we"})
ADDRESSEE = frozenset({"you"})
#: What a quantity before the thing says: asked whether there are any.
ANY = frozenset({"any", "some"})
#: Articles that put down a new one: `you have a cat`.
INDEFINITE = frozenset({"a", "an"})
#: The program's own states other than having, as WordNet files verbs:
#: what it likes and what it perceives.
STATES = frozenset({"verb.emotion", "verb.perception"})


@dataclass
class Owning:
    #: speaker | addressee
    who: str
    #: have | know | a state's verb (`like`, `see`)
    verb: str
    #: the kind asked about (`dog`), or "" for `what do you have`
    kind: str = ""
    #: the phrase as said, without its determiner (`dog`, `pets`)
    said: str = ""
    #: a name known of, for `do you know john`
    name: str = ""
    #: what is said of it: `black` in `you have a black cat`
    modifiers: tuple = ()
    plural: bool = False
    #: for a state, the thing as said, with its determiner (`a whale`)
    thing: str = ""
    #: for a state: WordNet's file for its verb (`verb.emotion`)
    state: str = ""
    #: asked, not told
    asked: bool = False
    #: the verb as said (`saw`), for saying it back
    verb_said: str = ""

    def some(self) -> str:
        """`a cat`, `any pets`: the thing asked of, as a question puts it."""
        if self.plural:
            return f"any {self.said}"
        return ("an " if self.said[:1] in "aeiou" else "a ") + self.said


def lexname(lemma: str) -> str:
    """WordNet's file for a verb's first sense: `verb.possession`."""
    try:
        from nltk.corpus import wordnet
        found = wordnet.synsets(lemma, wordnet.VERB)
        return found[0].lexname() if found else ""
    except Exception:                              # noqa: BLE001
        return ""


def _possessing(lemma: str) -> bool:
    return lexname(lemma) == "verb.possession"


def _lemma(token) -> str:
    """A noun's singular, by WordNet's morphology: spaCy's lemma for
    *cookies* is `cooky`, and WordNet has *legs* as a noun of its own, so
    a plural takes the form that is not the word as said."""
    try:
        from nltk.corpus import wordnet
        forms = wordnet._morphy(token.text, wordnet.NOUN)
    except Exception:                              # noqa: BLE001
        return token.lemma_
    if token.tag_ in ("NNS", "NNPS"):
        other = [one for one in forms if one != token.text]
        if other:
            return other[0]
    return forms[0] if forms else token.lemma_


def parse(text: str):
    """(root, subject, object, asked) of an utterance whose subject is one
    of the two taking part, or None."""
    from research.v687 import language

    parser = language.load()
    if parser is None:
        return None
    doc = parser(" ".join(text.lower().replace("?", " ?").split()))
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is None or not root.tag_.startswith("VB"):
        return None
    subject = [child for child in root.children if child.dep_ == "nsubj"]
    if len(subject) != 1 or subject[0].text not in SPEAKER | ADDRESSEE:
        return None
    if any(child.dep_ in ("ccomp", "xcomp", "neg") for child in
           root.children):
        return None
    objects = [child for child in root.children if child.dep_ == "dobj"]
    if len(objects) != 1:
        return None
    if any(token.tag_ == "WRB" for token in doc) or any(
            child.dep_ == "aux" and child.tag_ == "MD"
            for child in root.children):
        # `where would you find a book`, `can you see me`: anyone's way of
        # doing it, or what I am able to do -- not what I have or like.
        return None
    asked = (text.rstrip().endswith("?") or doc[0].tag_ in (
        "MD", "VBP", "VBZ", "VBD", "WP", "WDT") or any(
        child.dep_ == "aux" and child.i < subject[0].i
        for child in root.children))
    return root, subject[0], objects[0], asked


def read(text: str):
    """(`Owning`, asked) for what the two taking part have or know, or
    None when the utterance is not about that."""
    found = _read(text)
    if found is not None:
        found[0].asked = found[1]
    return found


def _read(text: str):
    found = parse(text)
    if found is None:
        return None
    root, subject, thing, asked = found
    who = "speaker" if subject.text in SPEAKER else "addressee"
    verb = root.lemma_.lower()
    if thing.tag_ in ("WP", "WDT") and _possessing(verb):
        # `what do you have`
        return Owning(who, "have"), True
    if thing.tag_ not in ("NN", "NNS", "NNP", "NNPS"):
        return None
    determiners = [child.text for child in thing.children
                   if child.dep_ == "det"]
    said = " ".join(token.text for token in thing.subtree
                    if token.dep_ != "det")
    if verb == "know" and who == "addressee" and asked:
        if thing.tag_ in ("NNP", "NNPS") or not determiners:
            if not (set(determiners) & ANY):
                return Owning(who, "know", name=thing.text), True
        return None
    state = lexname(verb)
    if who == "addressee" and state in STATES:
        # `you like cake`, `have you ever seen a whale`: mine to keep, and
        # mine to answer.
        phrase = " ".join(token.text for token in thing.subtree)
        return Owning(who, verb, _lemma(thing), said, "", (),
                      thing.tag_ in ("NNS", "NNPS"), phrase, state,
                      verb_said=root.text), asked
    if not _possessing(verb):
        return None
    if thing.tag_ in ("NNP", "NNPS"):
        return None
    if not asked and (who != "addressee"
                      or not set(determiners) & INDEFINITE):
        # `i have a beagle` is the reader's already, relative clause and
        # all; `you have the book` is about a book already here.
        return None
    modifiers = tuple(child.text for child in thing.children
                      if child.dep_ == "amod")
    return Owning(who, "have", _lemma(thing), said, "", modifiers,
                  thing.tag_ in ("NNS", "NNPS")), asked

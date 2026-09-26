"""A request put as the question inside it, before anything is read.

    do you know if a dog can swim       can a dog swim
    tell me whether a cat has fur       does a cat have fur
    is it true that dogs bark           do dogs bark
    is it likely that a bird can fly    can a bird fly     (and says so)
    can't a dog swim                    can a dog swim     (and says so)
    why can't a penguin fly             can a penguin fly  (asked why)
    how come birds fly                  do birds fly       (asked why)
    describe a cat                      tell me about a cat
    define a hammer                     what is a hammer
    what does bark mean                 what is a bark
    name three birds                    what kinds of birds are there
    list animals that fly               which animals fly

Every one of these was answered about something else. `do you know if a dog
can swim` asked whether a computer program knows things, `doesn't a cat have
fur` became `does a doesn't a cat have fur`, and `describe a cat` was re-asked
as `what is a describe a cat`. The question was there all along, in words a
parser was never going to take apart, so they are taken apart first.

**Read by the encoder** (`research/encoder.py`, `rephrase`): what the request
was put as (`REQUESTS`: the note the answer carries, and whether it asks why),
and how each word is said back -- kept, left out, or another form of itself,
what is said after it, and which word comes next. The frames and templates
below (`requested`) are the encoder's teacher, asked offline
(`v689/teach_reader.py`) and never at run time.

**Why.** A why-question about a kind asks the yes or no inside it and what
that rests on, so it is asked as the yes or no and marked `why`; `negative`
says whether the premise was a denial (`why can't a penguin fly`), so the
answer can say whether the premise holds. Only a kind is taken this way --
`a penguin`, `birds`. `why does the dog bark` may be about one dog, and `why
does it rain` about no kind at all, so they are left for the conversation and
for v687's scripts.

Two shapes are left exactly as said and marked as questions, because a rule
downstream reads them whole: an analogy (`fins are to fish as what are to
birds`, R24) and a conditional (`if a dog had wings could it fly`, which R18
refuses by name).
"""
from __future__ import annotations

from dataclasses import dataclass

#: `do you know if ...`: the question is what follows.
EMBEDDING = ("do you know if ", "do you know whether ",
             "can you tell me if ", "can you tell me whether ",
             "could you tell me if ", "could you tell me whether ",
             "tell me if ", "tell me whether ", "i wonder if ",
             "i wonder whether ", "i want to know if ",
             "i want to know whether ", "please tell me if ",
             "please tell me whether ")

#: `is it true that ...`: the frame adds nothing to what follows.
TRUTH = ("is it true that ", "is it the case that ", "is it so that ")

#: `is it likely that ...`: asked as whether it holds, and said so.
LIKELY = ("is it likely that ", "is it possible that ",
          "is it probable that ", "is it unlikely that ")

#: What a statement puts after its subject and a question puts first.
AUX = frozenset({"can", "could", "is", "are", "was", "were", "has", "have",
                 "does", "do", "did", "will", "would", "should", "must",
                 "may", "might"})

#: The question words, in front of which nothing is moved.
WH = frozenset({"what", "who", "whom", "whose", "which", "why", "how",
                "where", "when"})

#: A negative question's opener, and the positive one it asks with.
NEGATIVE = {"can't": "can", "cannot": "can", "couldn't": "could",
            "isn't": "is", "aren't": "are", "wasn't": "was",
            "weren't": "were", "hasn't": "has", "haven't": "have",
            "doesn't": "does", "don't": "do", "didn't": "did",
            "won't": "will", "wouldn't": "would", "shouldn't": "should",
            "mustn't": "must"}

#: What `name three birds` may put before the kind.
COUNTING = frozenset({"me", "a", "an", "the", "some", "all", "any", "few",
                      "several", "many", "one", "two", "three", "four",
                      "five", "six", "seven", "eight", "nine", "ten"})

#: Words that end in `s` and are not a plural kind: `why does it`, `why is
#: this`.
NOT_KINDS = frozenset({"this", "these", "those", "us", "its", "his", "yes",
                       "is", "was", "as"})

NEGATIVE_NOTE = ("a negative question asks the same thing as the positive "
                 "one, so it is answered as that")
LIKELY_NOTE = ("asked as whether it holds: nothing here measures how likely "
               "it is")


@dataclass
class Rephrased:
    text: str
    note: str = ""
    #: A question already, whatever it opens with.
    asking: bool = False
    #: Asked why: the text is the yes or no whose grounds are wanted.
    why: bool = False
    #: The why's premise was a denial: `why can't a penguin fly`.
    negative: bool = False


def with_article(phrase: str) -> str:
    """`hammer` as `a hammer`; a phrase with a determiner as it is."""
    first = phrase.split(" ", 1)[0].lower()
    if first in ("a", "an", "the", "some", "any") or not phrase:
        return phrase
    return ("an " if first[:1] in "aeiou" else "a ") + phrase


def singular_verb(verb: str) -> str:
    """`barks` as `bark`, `flies` as `fly`, `watches` as `watch`."""
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


def polar(clause: str) -> str:
    """`a dog can swim` as `can a dog swim`; a question as it already is.

    A denial inside the statement is dropped with the rest of the frame:
    `a penguin can't fly` asks `can a penguin fly`."""
    words = clause.split()
    lower = [word.lower() for word in words]
    if not words or lower[0] in AUX or lower[0] in WH:
        return clause
    for index in range(1, min(len(words), 5)):
        aux = NEGATIVE.get(lower[index], lower[index])
        if aux not in AUX:
            continue
        subject = " ".join(words[:index])
        after = words[index + 1:]
        if lower[index] == aux and after[:1] and after[0].lower() == "not":
            after = after[1:]
        rest = " ".join(after)
        if aux in ("has", "have"):
            opener = "does" if aux == "has" else "do"
            return f"{opener} {subject} have {rest}".strip()
        return f"{aux} {subject} {rest}".strip()
    if len(words) >= 3 and lower[0] in ("a", "an", "the"):
        return (f"does {' '.join(words[:2])} {singular_verb(lower[2])} "
                f"{' '.join(words[3:])}").strip()
    return f"do {clause}"


def denies(clause: str) -> bool:
    """Is this yes-or-no, or statement, a denial?"""
    words = [word.lower() for word in clause.split()]
    return any(word in NEGATIVE or word in ("not", "never", "no")
               for word in words[:5])


def about_a_kind(words: list[str]) -> bool:
    """`a penguin ...`, `birds ...`: a kind, not one thing or none."""
    if not words:
        return False
    first = words[0]
    return first in ("a", "an") or (
        first.endswith("s") and not first.endswith("ss")
        and first not in NOT_KINDS)


def _why(stripped: str, lower: str) -> Rephrased | None:
    """`why can't a penguin fly`, `how come birds fly`."""
    for opener in ("why ", "how come "):
        if not lower.startswith(opener):
            continue
        body = stripped[len(opener):].strip()
        words = body.lower().split()
        if not words:
            return None
        if words[0] in AUX or words[0] in NEGATIVE:
            after = words[1:]
            if after[:1] == ["not"]:
                after = after[1:]
            if not about_a_kind(after):
                return None
            asked = requested(body).text if words[0] in NEGATIVE or (
                len(words) > 1 and words[1] == "not") else body
            return Rephrased(asked, why=True, negative=denies(body))
        if opener == "how come " and about_a_kind(words):
            return Rephrased(polar(body), why=True, negative=denies(body))
        return None
    # `how does a bird fly` asks what the doing rests on, which is what a why
    # answers -- and the part the doers have in common is the how of it. It
    # was answered as a bridge from birds to the insect called a fly.
    for opener in ("how does ", "how do ", "how can "):
        if not lower.startswith(opener):
            continue
        body = stripped[len("how "):].strip()
        words = body.lower().split()
        if len(words) >= 3 and about_a_kind(words[1:]):
            return Rephrased(body, "asked how: what doing it rests on, and "
                                   "what the things that do it have in common",
                             why=True)
    return None


#: Who `you` is when a question is about what anyone can do with a thing.
GENERIC = frozenset({"you", "i", "we", "one"})

#: What a thing is used with: `with a knife`, `using a spoon`.
TOOL = frozenset({"with", "using"})

#: Where a thing is used, for the verbs a place or a vessel is for: `sit on a
#: chair`, `drink from a cup`. `see in the dark` names no thing, and `see` is
#: not here.
PLACE = frozenset({"on", "in", "from", "into", "inside"})
PLACED_VERBS = frozenset("""
sit sleep lie stand drink eat cook bake boil fry write keep store carry pour
wash hang ride swim sail live
""".split())

#: The participle a `can ... be` question needs, for the verbs asked that way.
PARTICIPLE = {"eat": "eaten", "drink": "drunk"}

ARTICLES = ("a", "an", "the")

GENERIC_NOTE = ("asked of the thing: `you` here is anyone, not this program")


def gerund(verb: str) -> str:
    """`cut` as `cutting`, `write` as `writing`, `open` as `opening`."""
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith("ee") and len(verb) > 2:
        return verb[:-1] + "ing"
    vowels = "aeiou"
    if (len(verb) == 3 and verb[2] not in vowels + "wxy"
            and verb[1] in vowels and verb[0] not in vowels):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _anyone(words: list[str]) -> Rephrased | None:
    """`can you cut bread with a knife`: asked of the knife.

        can you cut bread with a knife    is a knife used for cutting bread
        can you drink from a cup          is a cup used for drinking
        can you eat an apple              can an apple be eaten
        what do you use to cut paper      what is used to cut paper

    `you` there is anyone, and the page answered about a computer program:
    `can a computer program cut bread with a knife`. Only a question that
    names the thing is taken this way, so `can you swim` and `can you see me`
    are still asked of the program.
    """
    if (len(words) > 5 and words[0] == "what" and words[1] in AUX
            and words[2] in GENERIC | {"people"} and words[3] == "use"
            and words[4] in ("to", "for")):
        return Rephrased(f"what is used {words[4]} {' '.join(words[5:])}",
                         GENERIC_NOTE)
    if (len(words) < 4 or words[0] not in ("can", "could")
            or words[1] not in GENERIC):
        return None
    verb, rest = words[2], words[3:]
    for index, word in enumerate(rest):
        if not (word in TOOL or (word in PLACE and verb in PLACED_VERBS)):
            continue
        if index + 2 < len(rest) and rest[index + 1] in ARTICLES:
            thing = " ".join(rest[index + 2:])
            doing = " ".join([gerund(verb), *rest[:index]])
            return Rephrased(f"is {with_article(thing)} used for {doing}",
                             GENERIC_NOTE)
        return None
    if verb in PARTICIPLE:
        if len(rest) > 1 and rest[0] in ARTICLES:
            return Rephrased(f"can {' '.join(rest)} be {PARTICIPLE[verb]}",
                             GENERIC_NOTE)
        if (len(rest) == 1 and rest[0].endswith("s")
                and rest[0] not in NOT_KINDS):
            return Rephrased(f"can {rest[0]} be {PARTICIPLE[verb]}",
                             GENERIC_NOTE)
    return None


def _placed(words: list[str]) -> Rephrased | None:
    """`is a fridge in a kitchen`: where a kind is found, as a yes or no.

    Read as a property it was `in a kitchen` asked of a fridge, and nothing
    has that property. `found in` is the relation's own cue.
    """
    if len(words) < 5 or words[0] not in ("is", "are"):
        return None
    if not about_a_kind(words[1:]) or "found" in words:
        return None
    for index in range(2, min(len(words) - 2, 6)):
        if words[index] in ("in", "on", "inside") and (
                words[index + 1] in ARTICLES):
            if index == 2 and words[1] in ("a", "an"):
                return None                      # `is a in a ...`: no noun
            return Rephrased(" ".join(words[:index] + ["found"]
                                      + words[index:]))
    return None


HOW_NOTE = ("asked how: what doing it rests on, and what the things that do "
            "it have in common")
WHICH_NOTE = "asked as which ones there are"
SUCH_NOTE = "asked as what it is: a kind the ontology has is one it knows"

#: What a request was put as, and what that says of the answer: (the note it
#: carries, whether it is a question already and read whole downstream,
#: whether it asks why, whether a why's premise was a denial).
REQUESTS = {
    "": ("", False, False, False),
    "asking": ("", True, False, False),
    "why": ("", False, True, False),
    "why denied": ("", False, True, True),
    "how": (HOW_NOTE, False, True, False),
    "likely": (LIKELY_NOTE, False, False, False),
    "negative": (NEGATIVE_NOTE, False, False, False),
    "anyone": (GENERIC_NOTE, False, False, False),
    "which ones": (WHICH_NOTE, False, False, False),
    "such": (SUCH_NOTE, False, False, False),
}

#: The heads the encoder reads a request with (`research/encoder.py`).
ASK_HEADS = ("request", "ask_op", "ask_insert", "ask_opening", "ask_next")

#: What surrounds a word and is not part of it.
PUNCTUATION = ".,;:!?()" + chr(34)


def asked_words(text: str) -> list[str]:
    """A request's words as the encoder reads them: as typed, without the
    punctuation around them."""
    found = []
    for raw in (text or "").replace(chr(8217), "'").split():
        word = raw.strip(PUNCTUATION)
        if word:
            found.append(word)
    return found


def saying(op: str, word: str, at: int = 0) -> str:
    """One word said back as the encoder says to (`encoder.OPS`)."""
    lower = word.lower()
    if op == "LOWER":
        return lower
    if op == "POSITIVE":
        return NEGATIVE.get(lower, word)
    if op == "SINGULAR":
        return singular_verb(lower)
    if op == "GERUND":
        return gerund(lower)
    if op == "PARTICIPLE":
        return PARTICIPLE.get(lower, lower)
    return word


def said_back(stripped: str, words: list[str], guess: dict) -> Rephrased:
    """The request as `guess` reads it (the encoder's reading, or a teacher's
    labels in the same shape): its words said back, or as it was typed when
    nothing about them changed."""
    from research.encoder import rewrite

    note, asking, why, negative = REQUESTS.get(guess["request"][0][0],
                                               REQUESTS[""])
    out = rewrite(words, guess["ask_op"], guess["ask_insert"],
                  guess["ask_opening"][0][0], guess["ask_next"], saying)
    text = stripped if out == words else " ".join(out)
    return Rephrased(text, note, asking, why, negative)


def request_of(found: Rephrased) -> str | None:
    """Which of `REQUESTS` a teacher's reading was put as."""
    shape = (found.note, found.asking, found.why, found.negative)
    return next((name for name, one in REQUESTS.items() if one == shape),
                None)


#: The addressee's own states, as WordNet files its verbs: what it has,
#: knows, likes or perceives. `do you have a dog` asks about the one
#: addressed; `can you cut bread with a knife` asks about anyone.
OWN_STATES = frozenset({"verb.possession", "verb.cognition",
                        "verb.emotion", "verb.perception"})

#: What says a noun phrase is a particular one: `the capital of france`.
DEFINITE = frozenset({"the", "my", "your", "his", "her", "its", "our",
                      "their"})

ADDRESSED_NOTE = "asked of me: what I have, know, like or perceive"
CONCEALED_NOTE = ("knowing a particular thing is knowing what it is, so it "
                  "is asked as that")


def _lexname(verb: str) -> str:
    try:
        from nltk.corpus import wordnet
        found = wordnet.synsets(verb, wordnet.VERB)
        return found[0].lexname() if found else ""
    except Exception:                              # noqa: BLE001
        return ""


def addressed(text: str) -> Rephrased | None:
    """A question that puts one of the addressee's own states to it, read
    off the parse: `you` is the subject, the verb's first sense is what one
    has, knows, likes or perceives, and its object is a noun phrase, not a
    clause.

        do you have a dog                   kept as said, asked of me
        have you ever seen a whale          kept as said, asked of me
        do you know the capital of france   what is the capital of france
        could you tell me the time          what is the time

    The encoder learned from the teacher's `can you cut bread with a knife`
    that `you` may be anyone and left out, and took `do you have a dog` as
    `do a dog have`. It is a rule about who takes part, not about a word:
    a state is had by someone, and here that someone is the one asked.
    Knowing a particular thing -- `the capital of france` -- is knowing
    what it is (a concealed question), and is asked as that. `do you know
    if a dog can swim` has a clause for its object and stays the
    encoder's."""
    from research.v687 import language

    parser = language.load()
    if parser is None:
        return None
    doc = parser(" ".join(text.lower().split()))
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is None or not root.tag_.startswith("VB"):
        return None
    subject = [child for child in root.children if child.dep_ == "nsubj"]
    if [child.text for child in subject] != ["you"]:
        return None
    if not any(child.dep_ in ("aux", "auxpass") and child.i < subject[0].i
               for child in root.children):
        return None                              # a statement, not asked
    if any(child.dep_ in ("ccomp", "xcomp") for child in root.children):
        return None
    objects = [child for child in root.children if child.dep_ == "dobj"]
    if not objects:
        return None
    kind = _lexname(root.lemma_.lower())
    told = kind == "verb.communication" and any(
        child.dep_ == "dative" and child.text == "me"
        for child in root.children)
    if any(token.tag_ == "WRB" for token in doc) or (not told and any(
            child.dep_ == "aux" and child.tag_ == "MD"
            for child in root.children)):
        # `where would you find a book`, `can you cut bread`: anyone's way
        # of doing it, or what I can do -- the encoder's, as before. `could
        # you tell me the time` is a request, and asks what the time is.
        return None
    if kind not in OWN_STATES and not told:
        return None
    thing = objects[0]
    determiners = [child.text for child in thing.children
                   if child.dep_ in ("det", "poss")]
    if (kind == "verb.cognition" or told) and determiners and (
            determiners[0] in DEFINITE):
        phrase = doc[thing.left_edge.i:thing.right_edge.i + 1].text
        be = "are" if thing.tag_ == "NNS" else "is"
        return Rephrased(f"what {be} {phrase}", CONCEALED_NOTE)
    if told:
        return None
    return Rephrased(" ".join(text.split()), ADDRESSED_NOTE)


def rephrase(text: str) -> Rephrased:
    """The question a request puts, as the encoder reads it -- except one
    put to the addressee about itself (`addressed`)."""
    from research import encoder

    said = " ".join((text or "").replace(chr(8217), "'").split())
    stripped = said.rstrip("?.! ")
    words = asked_words(stripped)
    if not words:
        return Rephrased(stripped)
    own = addressed(stripped)
    if own is not None:
        return own
    guess = encoder.read([word.lower() for word in words], heads=ASK_HEADS)
    return said_back(stripped, words, guess)


def requested(text: str) -> Rephrased:
    """The request put as its question by frames and templates: the
    encoder's teacher, never asked at run time."""
    said = " ".join((text or "").replace(chr(8217), "'").split())
    stripped = said.rstrip("?.! ")
    lower = stripped.lower()
    if lower.startswith("please "):
        stripped, lower = stripped[7:], lower[7:]
    if lower.endswith(" please"):
        stripped, lower = stripped[:-7].rstrip(", "), lower[:-7].rstrip(", ")
    words = lower.split()
    if not words:
        return Rephrased(stripped)

    asked_why = _why(stripped, lower)
    if asked_why is not None:
        return asked_why

    for frame in EMBEDDING + TRUTH:
        if lower.startswith(frame):
            return Rephrased(polar(stripped[len(frame):]))
    for frame in LIKELY:
        if lower.startswith(frame):
            return Rephrased(polar(stripped[len(frame):]), LIKELY_NOTE)

    if words[0] in NEGATIVE and len(words) > 1:
        return Rephrased(f"{NEGATIVE[words[0]]} {stripped.split(' ', 1)[1]}",
                         NEGATIVE_NOTE)
    if len(words) > 2 and words[0] in AUX and words[1] == "not":
        kept = stripped.split()
        return Rephrased(" ".join([kept[0].lower()] + kept[2:]),
                         NEGATIVE_NOTE)

    if words[0] == "describe" and len(words) > 1:
        return Rephrased("tell me about " + stripped.split(" ", 1)[1])
    if words[0] == "define" and len(words) > 1:
        return Rephrased("what is " + with_article(stripped.split(" ", 1)[1]))
    if (lower.startswith("what does ") and lower.endswith(" mean")
            and len(words) > 3):
        return Rephrased("what is " + with_article(stripped[10:-5].strip()))
    if lower.startswith("what is the meaning of ") and len(words) > 5:
        return Rephrased("what is " + with_article(stripped[23:].strip()))

    # `what is the purpose of a hammer` was a listing about `purpose`, and
    # `what is a hammer good for` one about a hammer's `good`.
    for frame in ("what is the purpose of ", "what's the purpose of ",
                  "what is the use of "):
        if lower.startswith(frame) and len(lower) > len(frame):
            return Rephrased("what is " + with_article(
                stripped[len(frame):].strip()) + " used for")
    if (lower.startswith(("what is a ", "what is an "))
            and lower.endswith(" good for") and len(words) > 5):
        return Rephrased(stripped[:-len(" good for")] + " used for")
    # `is there such a thing as a flying fish`: whether the ontology has the
    # kind, which is what asking what it is finds out.
    for frame in ("is there such a thing as ", "are there such things as "):
        if lower.startswith(frame) and len(lower) > len(frame):
            return Rephrased("what is " + with_article(
                stripped[len(frame):].strip()),
                "asked as what it is: a kind the ontology has is one it knows")

    if words[:2] == ["are", "there"] and len(words) > 3:
        # `are there birds that cannot fly`: which ones, and there are some
        # exactly when that list is not empty.
        body = words[2:]
        while body and body[0] in COUNTING:
            body = body[1:]
        for joiner in ("that", "which", "who"):
            if joiner in body[1:]:
                at = body.index(joiner, 1)
                return Rephrased(f"which {' '.join(body[:at])} "
                                 f"{' '.join(body[at + 1:])}".strip(),
                                 "asked as which ones there are")

    if words[0] in ("name", "list") and len(words) > 1:
        body = words[1:]
        while body and body[0] in COUNTING:
            body = body[1:]
        for joiner in ("that", "which", "who"):
            if joiner in body[1:]:
                at = body.index(joiner, 1)
                return Rephrased(f"which {' '.join(body[:at])} "
                                 f"{' '.join(body[at + 1:])}".strip())
        if body:
            return Rephrased(f"what kinds of {' '.join(body)} are there")

    anyone = _anyone(words)
    if anyone is not None:
        return anyone
    placed = _placed(words)
    if placed is not None:
        return placed

    padded = f" {lower} "
    if (" is to " in padded or " are to " in padded) and " as " in padded:
        return Rephrased(stripped, asking=True)
    if words[0] == "if":
        return Rephrased(stripped, asking=True)
    return Rephrased(stripped)

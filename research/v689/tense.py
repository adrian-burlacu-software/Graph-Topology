"""When: what an utterance says about time, read before anything else is.

    yesterday the dog chased the cat        frame yesterday
    the dog chased the cat yesterday        frame yesterday
    then it slept                           after the occurrence before it
    before that, it had barked              before the occurrence before it
    meanwhile the cat played                during it
    after the dog chased the cat, it slept  after that occurrence
    the dog barked after the cat ran        after that occurrence
    while the dog slept, the cat ate        during that occurrence
    it is empty now                         frame now
    it barked again                         one more of the same

Without this, a time word was read as part of what was said: `yesterday there
was a dog` went to v688 as a question about a kind, `it barked again` stored
`capable_of "bark again"`, and `earlier it had run` taught a kind called
`early`. So the words that place an utterance in time are taken out first, and
the rest is read as it always was.

## Frames, links and anchors

A **frame** names a time: yesterday, this morning, now, tomorrow. Frames sit
on one line of days, now at zero, so two framed episodes can be put in order
without anything else being said (`timeline.py`). `today`, `now` and `right
now` are one frame: the present.

A **link** places an utterance against the occurrence told before it: `then`,
`after that`, `before that`, `meanwhile`, `finally`. It is only a link at the
start of a clause, before something that can be its subject -- `then it slept`,
not `what happened then`.

An **anchor** is a clause the utterance is placed against, joined by `after`,
`before`, `while` or `when`. At the start of a sentence it needs its comma --
`after the dog chased the cat, it slept` -- and in the middle it needs two
words either side, so `the dog slept after dinner` is left alone for the
reader to refuse.

## Tense and aspect

Reichenbach's three times -- the time of an event, the time talked about, and
the time of speaking -- are what the auxiliary and the verb's form say:

    it barked                 past, simple         an occurrence
    it was barking            past, progressive    an occurrence, in progress
    it had barked             past, perfect        an occurrence, before then
    it will bark              future, simple       an occurrence, to come
    it is barking             present, progressive an occurrence, now
    it barks, it does bark    present, habit       what it does: no occurrence
    it can bark               present, ability     what it can do: no occurrence
    it was black              past, state          a state, then
    it is black               present, state       a state, now
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: (words, key, rank): a named time and its place on the line of days. The
#: longest match is taken, so `the day before yesterday` is not `yesterday`.
FRAMES = tuple(sorted((
    (("once", "upon", "a", "time"), "long ago", -6.0),
    (("a", "long", "time", "ago"), "long ago", -6.0),
    (("long", "ago"), "long ago", -6.0),
    (("last", "year"), "last year", -5.0),
    (("last", "month"), "last month", -4.0),
    (("last", "week"), "last week", -3.0),
    (("the", "day", "before", "yesterday"), "the day before yesterday", -2.5),
    (("yesterday", "morning"), "yesterday", -2.0),
    (("yesterday", "afternoon"), "yesterday", -2.0),
    (("yesterday", "evening"), "yesterday", -2.0),
    (("yesterday",), "yesterday", -2.0),
    (("last", "night"), "last night", -1.5),
    (("this", "morning"), "this morning", -1.0),
    (("earlier", "today"), "this morning", -1.0),
    (("today",), "now", 0.0),
    (("right", "now"), "now", 0.0),
    (("now",), "now", 0.0),
    (("at", "the", "moment"), "now", 0.0),
    (("currently",), "now", 0.0),
    (("this", "afternoon"), "now", 0.0),
    (("later", "today"), "later today", 1.0),
    (("this", "evening"), "tonight", 1.0),
    (("tonight",), "tonight", 1.0),
    (("tomorrow",), "tomorrow", 2.0),
    (("next", "week"), "next week", 3.0),
    (("next", "month"), "next month", 4.0),
    (("next", "year"), "next year", 5.0),
), key=lambda one: -len(one[0])))

#: What a frame is called when nothing more particular was said.
LABELS = {"now": "now", "then": "then", "later": "later"}

#: (words, relation): a clause placed against the occurrence before it.
LINKS = tuple(sorted((
    (("and", "then"), "after"), (("then",), "after"),
    (("after", "that"), "after"), (("afterwards",), "after"),
    (("following", "that"), "after"),
    (("afterward",), "after"), (("later", "on"), "after"),
    (("later",), "after"), (("next",), "after"), (("soon", "after"), "after"),
    (("after", "a", "while"), "after"), (("eventually",), "after"),
    (("second",), "after"), (("secondly",), "after"), (("third",), "after"),
    (("finally",), "last"), (("lastly",), "last"), (("in", "the", "end"), "last"),
    (("before", "that"), "before"), (("earlier",), "before"),
    (("previously",), "before"),
    (("meanwhile",), "during"), (("in", "the", "meantime"), "during"),
    (("at", "the", "same", "time"), "during"),
    (("first",), "first"), (("firstly",), "first"),
), key=lambda one: -len(one[0])))

#: The same occurrence again: `it barked again`.
AGAIN = (("once", "more"), ("once", "again"), ("again",))

#: A clause the main one is placed against, and how the main one stands to it.
SUBORDINATORS = {"after": "after", "before": "before", "while": "during",
                 "when": "during"}

#: What can follow a link and start its clause.
CLAUSE_STARTS = frozenset({"it", "he", "she", "they", "i", "we", "you", "the",
                           "a", "an", "my", "this", "that", "there", "his",
                           "her", "its", "our", "their"})


@dataclass(frozen=True)
class Frame:
    key: str
    rank: float
    #: the words it was named with
    label: str

    def as_dict(self) -> dict:
        return {"key": self.key, "rank": self.rank, "label": self.label}


@dataclass
class When:
    """What an utterance says about when."""

    frame: Frame | None = None
    #: after | before | during | first | last: against the occurrence before
    link: str = ""
    #: `again`, `once more`, and the words it was said with
    again: bool = False
    again_words: list = field(default_factory=list)
    #: the clause it is placed against, and how: after | before | during
    anchor: str = ""
    relation: str = ""
    #: the clause that is placed, without its anchor
    main: str = ""
    #: the words that were taken out, as said
    words: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.frame or self.link or self.again or self.anchor)

    def as_dict(self) -> dict:
        return {"frame": self.frame.as_dict() if self.frame else None,
                "link": self.link, "again": self.again,
                "anchor": self.anchor, "relation": self.relation,
                "words": list(self.words)}


def _match(tokens: list[str], at: int, table) -> tuple | None:
    for entry in table:
        words = entry[0]
        if tuple(tokens[at:at + len(words)]) == words:
            return entry
    return None


def _match_end(tokens: list[str], table) -> tuple | None:
    for entry in table:
        words = entry[0]
        if len(tokens) > len(words) and tuple(tokens[-len(words):]) == words:
            return entry
    return None


def take(tokens: list[str], typed: list[str],
         names: frozenset = frozenset()):
    """(tokens, typed, When): the time words taken off either end.

    Frames come off the start or the end; links only off the start, and only
    before a word that can begin a clause; `again` only off the end. What is
    left is never empty: `now` alone is not a time, it is an utterance.
    """
    tokens, typed = list(tokens), list(typed)
    when = When()
    starts = CLAUSE_STARTS | names

    def cut_start(length: int) -> None:
        when.words.extend(typed[:length])
        del tokens[:length], typed[:length]

    def cut_end(length: int) -> None:
        when.words.extend(typed[-length:])
        del tokens[-length:], typed[-length:]

    changed = True
    while changed and len(tokens) > 1:
        changed = False
        found = _match(tokens, 0, FRAMES)
        if (found and when.frame is None and len(tokens) - len(found[0]) >= 2
                and tokens[len(found[0])] not in ("is", "was", "are",
                                                  "were")):
            when.frame = Frame(found[1], found[2], " ".join(found[0]))
            cut_start(len(found[0]))
            changed = True
            continue
        found = _match(tokens, 0, LINKS)
        if (found and not when.link and len(tokens) > len(found[0])
                and tokens[len(found[0])] in starts):
            when.link = found[1]
            cut_start(len(found[0]))
            changed = True
            continue
    found = _match_end(tokens, FRAMES)
    if found and when.frame is None and len(tokens) > len(found[0]) + 1:
        when.frame = Frame(found[1], found[2], " ".join(found[0]))
        cut_end(len(found[0]))
    found = _match_end(tokens, [(words,) for words in AGAIN])
    if found and len(tokens) > len(found[0]) + 1:
        when.again = True
        when.again_words = list(typed[-len(found[0]):])
        cut_end(len(found[0]))
    return tokens, typed, when


def subordinate(text: str) -> tuple[str, str, str] | None:
    """(main clause, how it stands to the anchor, the anchor), or None.

    `after the dog chased the cat, it slept` -> (`it slept`, after, `the dog
    chased the cat`); `was the vase broken before the cat broke it` -> (`was
    the vase broken`, before, `the cat broke it`).
    """
    said = (text or "").strip().rstrip("?.!").strip()
    lowered = said.lower()
    # A word of a named time places nothing: `the day before yesterday it
    # rained` is one clause, said on a day. Joined, at the same length, the
    # phrase has no ` before ` to split on.
    for entry in FRAMES:
        phrase = " ".join(entry[0])
        if len(entry[0]) > 1 and SUBORDINATORS.keys() & set(entry[0]):
            lowered = lowered.replace(phrase, phrase.replace(" ", "_"))
    for word, relation in SUBORDINATORS.items():
        head = word + " "
        if lowered.startswith(head) and "," in lowered:
            cut = lowered.index(",")
            anchor, main = said[len(head):cut].strip(), said[cut + 1:].strip()
            if len(anchor.split()) >= 2 and main.split():
                return main, relation, anchor
    best = None
    for word, relation in SUBORDINATORS.items():
        at = lowered.find(f" {word} ")
        if at <= 0 or (best is not None and at >= best[0]):
            continue
        main, anchor = said[:at].strip(), said[at + len(word) + 2:].strip()
        if len(main.split()) >= 2 and len(anchor.split()) >= 2:
            best = (at, main, relation, anchor)
    return best[1:] if best else None


#: Tense and aspect of an auxiliary alone, before the verb is looked at.
_BY_AUX = {"did": ("past", "simple"), "will": ("future", "simple"),
           "can": ("present", "ability"), "could": ("past", "ability"),
           "does": ("present", "habit"), "do": ("present", "habit"),
           "would": ("present", "conditional")}


def tense_of(aux: str | None, rest: list[str],
             tags: list[str] | None = None) -> tuple[str, str]:
    """(tense, aspect) of a clause, from its auxiliary and its verb's form.

    `tags` are the Penn tags of `rest`, read in context; without them the
    form is guessed from the spelling, which is right for regular verbs.
    """
    first = rest[0] if rest else ""
    tag = tags[0] if tags else ""
    gerund = tag == "VBG" or (not tag and first.endswith("ing"))
    participle = tag in ("VBN", "VBD") or (not tag and first.endswith("ed"))
    if aux in ("was", "were"):
        return ("past", "progressive") if gerund else ("past", "state")
    if aux in ("is", "am", "are"):
        return ("present", "progressive") if gerund else ("present", "state")
    if aux == "had":
        return ("past", "perfect") if participle else ("past", "state")
    if aux in ("has", "have"):
        return ("present", "perfect") if participle else ("present", "state")
    if aux == "will" and first == "be":
        return "future", "progressive"
    if aux in _BY_AUX:
        return _BY_AUX[aux]
    if tag == "VBD" or (not tag and first.endswith("ed")):
        return "past", "simple"
    return "present", "habit"


#: The aspects that make a clause an occurrence rather than a state, a habit
#: or an ability.
OCCURRING = frozenset({"simple", "progressive", "perfect"})


def occurs(tense: str, aspect: str) -> bool:
    """Does a clause of this tense and aspect say something happened?"""
    return aspect in OCCURRING and not (tense == "present"
                                        and aspect == "simple")

"""Teaching the encoder reader (`reader.py`) what the grammar reads.

    python -m research.v689.teach_reader corpus [--variants 6] [--processes 6]
    python -m research.v689.teach_reader train [--base llm/MiniLM-L6-v2]

**Taking a reading apart.** The grammar and the statement reader read each
utterance of a corpus as they always have, over words that remember where
they were said (`At`), so every auxiliary, verb phrase and mention in the
reading points back at its words. `extract` turns the reading into what the
encoder is to say -- the act, the slots cell, who fills it, and each word's
role -- and `build` (`reader.py`) must put the very same reading back
together from that, or the utterance is not taught (`label`). Statements are
taken apart the same way, over their normal form (`reading.normal`), and a
statement of several claims claim by claim, each marked where it begins.
spaCy's tag and dependency of each word are kept with it, for the encoder to
read beside the word.

**What is read.** Every short string in v689's tests, bAbI's training lines,
the grammar's documented examples and its shapes, each also said with other
names, other nouns of the same kind and other verbs of the same VerbNet class
(`vary`), all read by the grammar.

**What the grammar never read.** Some wordings keep a reading's act and roles
by construction, so they are taught with the reading they were made from, each
word's role carried along (`TRANSPORTS`). Every word they add comes from a
resource, not a list written here:

    adjunct     an adverb WordNet derives from an adjective (a pertainym)
    timed       a noun WordNet files under noun.time
    opened      a word Brown tags as an interjection
    embedded    a question put inside a request, with a VerbNet verb of
                knowing or telling that takes a wh-clause (`NP V what S`)
    place kind  `which room is Mary in`: a hypernym of a story's rooms
    by location `where is Mary hiding`, `where did Mary turn up`: a verb
                VerbNet says leaves its subject somewhere
    by having   `who possesses the pear`: a verb VerbNet says has its object
                with its subject while it lasts (`change.accompanies`)
    by finding  `where can I find Mary`: WordNet's senses of finding where
                something is

A name the conversation knows is said to the encoder as one of a few names
(`reader.said_as`), so what is asked is learned apart from who it is asked of.
Questions worded like a cell's that ask something else -- `how old is Mary`,
`where did Mary grow up` -- are read by the grammar too (`_contrasts`).

**The rules before the grammar, and v687's cues.** The same encoder reads
what the rules read before the grammar ever did, and each is taught here from
the rule it replaced, over the same texts and more:

    ask     `rephrase.requested`: a request as its question
    place   `reading.new_names`, `tense.subordinate`, `tense.take` and
            `reading.normal`: who is new, the anchor, the time words, and
            the rest in normal form
    parse   `Parser.cued`: v687's relation, subject and target, over every
            text and every question v689's reading puts to v687 (`Probing`)

Where words are said back in another order, the rule's words are aligned to
the words said (`rewrite_labels`): each is kept, left out or said as another
form of itself, what the rule added is said after the word before it, and the
order is which word follows which. A record is kept only when the labels say
the rule's reading back exactly (`label_ask`, `label_place`, `label_parse`).

The rules and the grammar are only asked here, offline; what they taught is
kept in `llm/reader-data/`, and the model in `llm/reader/`.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import sys
import time
import zlib
from functools import lru_cache
from pathlib import Path

from research.encoder import LONGEST, Heads, features, gold, positions
from research.v687.language import taught_by_cues

from .reader import (ACTS, CLAUSES, ELIDED, LLM, MARKED, NAMED, NONE, ROLES,
                     SLOTS, STATED, STATEMENTS, WHO, analysed, build, first,
                     mark, said_as, spans)

#: Acts the grammar reads a denial in, which each such question is also
#: taught denied.
NEGATED = frozenset({"why", "subject occurrence", "object occurrence",
                     "which is_a", "facts any"})

REPOSITORY = Path(__file__).resolve().parents[2]
DATA = LLM / "reader-data"


class At(str):
    """A word that remembers where it was said."""

    def __new__(cls, text: str, at: int):
        word = super().__new__(cls, text)
        word.at = at
        return word


def where(word) -> int | None:
    return getattr(word, "at", None)


class Plain:
    """A lexicon asked in plain strings: spaCy and sqlite take no `At`."""

    def __init__(self, lexicon) -> None:
        self.lexicon = lexicon

    def __getattr__(self, name: str):
        found = getattr(self.lexicon, name)
        if not callable(found):
            return found

        def plain(value):
            if isinstance(value, (list, tuple)):
                return [str(one) for one in value]
            return str(value) if isinstance(value, str) else value

        def call(*args, **kwargs):
            return found(*[plain(one) for one in args],
                         **{key: plain(one) for key, one in kwargs.items()})
        return call


def _prepare(text: str, lexicon, names: frozenset):
    """What `reading.read` does before it reads: a request put as the
    question inside it, someone new named, an anchor and time words taken
    off, and a statement in normal form, all by the rules
    (`reading.taught_front`). (said, asked, tokens, typed, names)."""
    from .reading import taught_front

    front = taught_front(text, lexicon, names)
    return front.said, front.asked, front.tokens, front.typed, front.names


def prepared(text: str, lexicon, names: frozenset = frozenset()):
    """(tokens, names, said): the words the encoder reads."""
    said, _, tokens, _, names = _prepare(text, lexicon, names)
    return tokens, names, said


@taught_by_cues
def teacher(text: str, lexicon, names: frozenset = frozenset()):
    """(tokens, names, reading, typed): what the grammar and the statement
    reader read, each word keeping its place."""
    from .reading import grammar

    lexicon = Plain(lexicon)
    said, asked, tokens, typed, names = _prepare(text, lexicon, names)
    tokens = [At(word, at) for at, word in enumerate(tokens)]
    found = grammar(said, asked, tokens, typed, lexicon, names)
    return tokens, names, found, typed


# -- taking a reading apart ------------------------------------------------------
def _span(mention, tokens: list[str], after: int = 0):
    """Where a mention's words are: at its end when that is where they are,
    else the first place after `after`."""
    words = str(mention.text or "").lower().split()
    if not words:
        return None
    size = len(words)
    lower = [str(word).lower() for word in tokens]
    end = mention.end
    if end and size <= end <= len(tokens) and lower[end - size:end] == words:
        return end - size, end
    for start in range(after, len(tokens) - size + 1):
        if lower[start:start + size] == words:
            return start, start + size
    return None


def _reading_roles(found, name: str, tokens, roles: list[str],
                   lexicon) -> bool:
    from .reader import _progressive

    mention = found.mention
    subject_end = 0
    if mention is not None and mention.text:
        span = _span(mention, tokens)
        if span is None:
            return False
        mark(roles, span, "KIND" if name in ("count is_a", "which is_a")
             else "SUBJ")
        subject_end = span[1]

    def unplaced(word) -> int | None:
        """A word the shape put in as a literal (`will`, `to`): the first
        untagged word said like it."""
        return next((one for one in range(len(tokens))
                     if roles[one] == "O" and tokens[one] == word), None)

    at = where(found.aux)
    if at is None and found.aux:
        at = unplaced(found.aux)
    if at is not None:
        roles[at] = "AUX"
    rest = list(found.rest)
    if name in MARKED or name == "subject dimension":
        rest = rest[1:]
    elif name == "object dimension":
        rest = rest[:-1]
    for word in rest:
        at = where(word)
        if at is None and name not in ("object holding", "count holding"):
            at = unplaced(word)
        if at is not None:
            if roles[at] == "O":
                roles[at] = "REST"
            continue
        if name in ("object holding", "count holding"):
            verb = next((one for one in range(len(tokens))
                         if roles[one] == "O"
                         and _progressive(lexicon, tokens[one]) == word),
                        None)
            if verb is None:
                return False
            roles[verb] = "REST"
    if not found.holds and name != "ask":
        negation = next((at for at in range(len(tokens))
                         if roles[at] == "O" and tokens[at] in ("not", "no")),
                        None)
        if negation is not None:
            roles[negation] = "NEG"
    obj = found.obj
    if obj is not None:
        if name == "count holding":
            span = _span(obj, tokens)
            if span is None:
                return False
            mark(roles, span, "KIND")
        elif name == "path dimension":
            span = _span(obj, tokens, subject_end)
            if span is None:
                return False
            mark(roles, span, "OBJ")
        elif 0 <= found.obj_at < len(found.rest) and where(
                found.rest[found.obj_at]) is not None:
            start = where(found.rest[found.obj_at])
            mark(roles, (start, start + len(str(obj.text).split())), "OBJ")
    return True


def _goal_roles(goal, cell: str, tokens, roles: list[str], lexicon) -> str:
    """Tag a slots goal's words; the who label."""
    from .reader import _verb
    from .reading import AUX, COPULA, SEQUENCE

    asked, relation = cell.split()
    end = 0
    if goal.subject is not None:
        span = _span(goal.subject, tokens)
        if span is None:
            return ""
        mark(roles, span, "SUBJ")
        end = span[1]
    if goal.object is not None:
        span = _span(goal.object, tokens, end)
        if span is None:
            return ""
        mark(roles, span, "OBJ")
    clause = goal.clause
    if clause is not None:
        at = where(clause.aux)
        if at is not None:
            roles[at] = "AUX"
        for word in clause.rest:
            at = where(word)
            if at is not None and roles[at] == "O":
                roles[at] = "REST"
        if (goal.object is None and clause.obj is not None
                and 0 <= clause.obj_at < len(clause.rest)
                and where(clause.rest[clause.obj_at]) is not None):
            start = where(clause.rest[clause.obj_at])
            mark(roles, (start, start + len(str(clause.obj.text).split())),
                 "OBJ")
    if relation == "holding":
        verb = next((at for at in range(len(tokens)) if roles[at] == "O"
                     and (_verb(lexicon, tokens[at]) == goal.verb
                          or tokens[at] == goal.verb)), None)
        if verb is not None:
            roles[verb] = "VERB"
    if goal.sequence is not None:
        seq = [at for at in range(len(tokens)) if roles[at] == "O"
               and SEQUENCE.get(tokens[at]) == goal.sequence]
        if seq:
            roles[seq[-1]] = "SEQ"
    if asked == "count":
        if relation == "located":
            stop = next((at for at in range(3, len(tokens))
                         if tokens[at] in COPULA), 3)
        elif relation == "holding":
            stop = next((at for at in range(3, len(tokens))
                         if tokens[at] in AUX), 3)
        else:
            stop = 3
        mark(roles, (2, stop), "KIND")
        return "kind"
    return goal.who if goal.who in ("people", "things") else ""


def _act(found) -> str:
    return ("introduce owned" if found.act == "introduce" and found.owned
            else found.act)


def _name_span(name: str, typed) -> tuple[int, int] | None:
    """Where the words a name was said in are: running to the end, or
    opening the utterance (`Rex is a beagle`)."""
    from .reading import proper

    typed = [str(one) for one in typed]
    for start in range(len(typed)):
        if proper(typed[start:]) == name:
            return start, len(typed)
    for end in range(1, len(typed) + 1):
        if proper(typed[:end]) == name:
            return 0, end
    return None


def _statement_roles(found, name: str, tokens, typed, roles: list[str],
                     clauses: list[str], lexicon) -> bool:
    """A statement's relative clause and name, tagged."""
    relative = found.relative
    if relative is not None:
        places = [where(word) for word in [relative.aux] + list(relative.rest)
                  if where(word) is not None]
        if not places or not _reading_roles(relative, "tell", tokens, roles,
                                            lexicon):
            return False
        clauses[min(places)] = "B-relative"
    if found.name:
        span = _name_span(found.name, typed)
        if span is None:
            return False
        mark(roles, span, "NAME")
    return True


def _claims_record(tokens, found, analysis) -> dict | None:
    """A statement of several claims, claim by claim: the words of each
    tagged within its own stretch, and where each after the first begins,
    with what it does. The stretches are the parse's clauses
    (`clauses.split`), as `reading._several` read them."""
    from . import clauses as coordination

    words = [str(one) for one in tokens]
    parts = coordination.split(words, analysis) if analysis else None
    readings = [found] + list(found.more)
    if parts is None or len(parts) != len(readings):
        return None
    starts = []
    for part in parts:
        own = [one.index for one in part.subject + part.aux + part.rest]
        if not own:
            return None
        starts.append(min(own))
    starts[0] = 0
    if starts != sorted(set(starts)):
        return None
    bounds = starts + [len(words)]
    roles, clauses = ["O"] * len(words), ["O"] * len(words)
    for index, (part, one) in enumerate(zip(parts, readings)):
        lo, hi = bounds[index], bounds[index + 1]
        name = _act(one)
        if name not in ACTS or one.relative is not None or one.name:
            return None
        if index:
            clauses[lo] = f"B-{name}"
        # Who the claim is about: its own phrase where the words are in the
        # claim, else the parse's subject standing for the one before
        # (`they`), and nothing for a claim about no one.
        subject = sorted(word.index for word in part.subject)
        phrase = _span(one.mention, words[lo:hi]) \
            if one.mention is not None and one.mention.text else None
        if phrase is not None:
            mark(roles, (lo + phrase[0], lo + phrase[1]), "SUBJ")
        elif one.mention is not None and subject:
            if (subject != list(range(subject[0], subject[-1] + 1))
                    or subject[0] < lo or subject[-1] >= hi):
                return None
            mark(roles, (subject[0], subject[-1] + 1), "SUBJ")
        if one.aux is not None:
            at = next((word.index for word in part.aux
                       if words[word.index] == one.aux
                       and roles[word.index] == "O"), None)
            if at is not None:
                roles[at] = "AUX"
        pointer, places = lo, []
        for said in one.rest:
            at = next((at for at in range(pointer, hi) if roles[at] == "O"
                       and words[at] == said), None)
            if at is None:
                at = next((at for at in range(pointer, hi)
                           if roles[at] == "O" and words[at] in ELIDED), None)
            if at is None:
                return None
            roles[at] = "REST"
            places.append(at)
            pointer = at + 1
        if not one.holds:
            at = next((at for at in range(lo, hi) if roles[at] == "O"
                       and words[at] in ("not", "no")), None)
            if at is not None:
                roles[at] = "NEG"
        if one.obj is not None and 0 <= one.obj_at < len(places):
            start = places[one.obj_at]
            mark(roles, (start, start + len(str(one.obj.text).split())),
                 "OBJ")
    return {"words": words, "act": _act(found), "slots": NONE, "who": "",
            "stated": roles, "slotted": ["O"] * len(words),
            "clauses": clauses}


def extract(tokens, found, lexicon, typed=None,
            analysis=None) -> dict | None:
    """What the encoder is to say of a reading: its act, slots cell, who
    fills that, each word's role in each, and where claims begin. None when
    a part of the reading cannot be found among the words."""
    count = len(tokens)
    typed = list(typed) if typed is not None else [str(one) for one in tokens]
    if found.more:
        return _claims_record(tokens, found, analysis)
    own = next((one for one in found.goals if one.own), None)
    slots_goal = next((one for one in found.goals if not one.own), None)
    name = _act(found)
    if name == "question":
        if own is None:
            return None
        name = " ".join(own.cell)
    elif name not in ACTS:
        return None
    stated_roles, clauses = ["O"] * count, ["O"] * count
    if not _reading_roles(found, name, tokens, stated_roles, lexicon):
        return None
    slots, who, slotted_roles = NONE, "", ["O"] * count
    if name in STATEMENTS:
        if not _statement_roles(found, name, tokens, typed, stated_roles,
                                clauses, lexicon):
            return None
    elif slots_goal is not None:
        slots = " ".join(slots_goal.cell)
        who = _goal_roles(slots_goal, slots, tokens, slotted_roles, lexicon)
    return {"words": [str(word) for word in tokens], "act": name,
            "slots": slots, "who": who, "stated": stated_roles,
            "slotted": slotted_roles, "clauses": clauses}


def view(found) -> dict:
    """A reading as compared: everything it says, its goals and each claim
    joined to it; not when, which is read before, nor who is new."""
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(one) for key, one in value.items()
                    if key not in ("when", "fresh")}
        if isinstance(value, list):
            return [plain(one) for one in value]
        return value

    shown = found.as_dict()
    shown.update(obj_at=found.obj_at, count=found.count,
                 more=[view(one) for one in found.more])
    return plain(json.loads(json.dumps(shown, default=str)))


def _built(record: dict, lexicon, names: frozenset):
    return build(record["act"], record["slots"], record["who"],
                 record["stated"], record["slotted"], record["words"],
                 lexicon, names, record["said"], record["typed"],
                 record["clauses"], record["tags"], record.get("deps"))


@taught_by_cues
def label(text: str, lexicon, names: frozenset = frozenset()):
    """(record, why): the record to teach for a text, or None and why not."""
    from .reading import _analysis

    lexicon = Plain(lexicon)
    tokens, names, found, typed = teacher(text, lexicon, names)
    if not tokens:
        return None, "empty"
    words = [str(word) for word in tokens]
    analysis = _analysis(words, lexicon)
    record = extract(tokens, found, lexicon, typed, analysis)
    if record is None:
        return None, "unaligned"
    record.update(names=sorted(names), said=found.said,
                  typed=[str(one) for one in typed],
                  tags=[one[0] for one in analysis] if analysis
                  else [""] * len(words),
                  deps=[one[1] for one in analysis] if analysis
                  else [""] * len(words))
    built = _built(record, lexicon, names)
    if built is None:
        return None, "unbuilt"
    if view(built) != view(found):
        return None, "differs"
    return record, ""


@taught_by_cues
def explain(text: str, lexicon, names: frozenset = frozenset()) -> str:
    """Why a text does not round-trip: the two readings side by side."""
    from .reading import _analysis

    lexicon = Plain(lexicon)
    tokens, names, found, typed = teacher(text, lexicon, names)
    words = [str(word) for word in tokens]
    analysis = _analysis(words, lexicon)
    record = extract(tokens, found, lexicon, typed, analysis)
    lines = [f"> {text}", f"  act {found.act} cells {found.cells} "
                          f"more {len(found.more)}"]
    if record is None:
        return "\n".join(lines + ["  unaligned"])
    record.update(said=found.said, typed=[str(one) for one in typed],
                  tags=[one[0] for one in analysis] if analysis
                  else [""] * len(words))
    lines.append("  " + " ".join(
        f"{w}/{r}" + (f"/{c}" if c != "O" else "") for w, r, c in
        zip(record["words"], record["stated"], record["clauses"])))
    lines.append(f"  {record['act']} | {record['slots']} {record['who']} | "
                 + " ".join(f"{w}/{r}" for w, r in
                            zip(record["words"], record["slotted"])))
    built = _built(record, lexicon, names)
    if built is None:
        return "\n".join(lines + ["  unbuilt"])
    one, other = view(found), view(built)
    for key in sorted(set(one) | set(other)):
        if one.get(key) != other.get(key):
            lines.append(f"  {key}:\n    was {one.get(key)}\n    now "
                         f"{other.get(key)}")
    return "\n".join(lines)


# -- the rules before the grammar, and v687's cues --------------------------------
#: The kind of each head (`research/encoder.py`).
HEAD_KINDS = {
    "acts": "sentence", "slots": "sentence", "who": "sentence",
    "stated": "word", "slotted": "word", "clauses": "word",
    "request": "sentence", "ask_op": "word", "ask_insert": "word",
    "ask_opening": "sentence", "ask_next": "pointer",
    "new": "word", "when": "word", "anchor": "word", "place_op": "word",
    "place_insert": "word", "place_opening": "sentence",
    "place_next": "pointer",
    "relation": "sentence", "polar": "sentence", "parse": "word",
    "stance": "sentence", "part": "word"}

#: Which field of a record each head is taught from, for each reader. A
#: reply is read back by v690 (`v690/roundtrip.py`, `teach_decoder.py`).
FIELDS = {
    "read": {"acts": "act", "slots": "slots", "who": "who",
             "stated": "stated", "slotted": "slotted", "clauses": "clauses"},
    "ask": {"request": "request", "ask_op": "ops", "ask_insert": "inserts",
            "ask_opening": "opening", "ask_next": "order"},
    "place": {"new": "new", "when": "when", "anchor": "anchor",
              "place_op": "ops", "place_insert": "inserts",
              "place_opening": "opening", "place_next": "order"},
    "parse": {"relation": "relation", "polar": "polar", "parse": "parse"},
    "reply": {"stance": "stance", "part": "parts"}}


def task_of(record: dict) -> str:
    return record.get("task", "read")


#: The ops that say a word as it was, but for its case.
EXACT = ("KEEP", "LOWER")

#: The last reading each labeller could not say back: (the rules', the
#: labels'), to see why.
DIFFERENCES: dict = {}


def _op(word: str, said: str, transform, at: int, exact: bool):
    """Which op says `word` as `said`, if one does."""
    from research.encoder import OPS

    tried = EXACT if exact else [one for one in OPS
                                 if one not in EXACT and one != "DROP"]
    for op in tried:
        try:
            form = transform(op, word, at)
        except Exception:                               # noqa: BLE001
            form = None
        if form == said:
            return op
    return None


def _common(source: list[str], target: list, used: list[bool],
            aligned: list) -> list[tuple[int, int]]:
    """(target index, source index) of the longest run of target words said
    in the same order among the source words not yet used, as said but for
    case: `please tell me if a knife is a flag` keeps its two `a`s apart."""
    rows, columns = len(target), len(source)
    lower = [str(one).lower() for one in source]

    def same(i: int, j: int) -> bool:
        return (aligned[i] is None and not used[j]
                and str(target[i]).lower() == lower[j])

    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            table[i][j] = (table[i + 1][j + 1] + 1 if same(i, j)
                           else max(table[i + 1][j], table[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < rows and j < columns:
        if same(i, j) and table[i][j] == table[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def rewrite_labels(source: list[str], target: list, transform,
                   excluded=frozenset()):
    """(ops, inserts, opening, order): how `target` is said back from the
    words of `source`, as `encoder.rewrite` says words back. A target word
    that remembers where it was said (`At`) comes from there; the longest run
    of the rest said in the same order comes from where it was said
    (`_common`); any other from the first word not yet used that is said
    like it -- failing that, like a form of it (`transform(op, word, at)`)
    -- looking first after the word before it; and a word from nowhere is
    said after the word before it, or opens the words. `excluded` words say
    nothing."""
    count = len(source)
    used = [at in excluded for at in range(count)]
    aligned: list = [None] * len(target)
    for index, word in enumerate(target):
        at = where(word)
        if at is not None and 0 <= at < count and not used[at]:
            op = (_op(source[at], str(word), transform, at, True)
                  or _op(source[at], str(word), transform, at, False))
            if op:
                aligned[index] = (at, op)
                used[at] = True
    for index, at in _common(source, target, used, aligned):
        op = _op(source[at], str(target[index]), transform, at, True)
        if op:
            aligned[index] = (at, op)
            used[at] = True
    last = -1
    for index, word in enumerate(target):
        if aligned[index] is not None:
            last = aligned[index][0]
            continue
        free = [at for at in range(count) if not used[at]]
        ordered = ([at for at in free if at > last]
                   + [at for at in free if at <= last])
        found = None
        for exact in (True, False):
            for at in ordered:
                op = _op(source[at], str(word), transform, at, exact)
                if op:
                    found = (at, op)
                    break
            if found:
                break
        if found:
            aligned[index] = found
            used[found[0]] = True
            last = found[0]
    ops = ["DROP"] * count
    inserts: list[list[str]] = [[] for _ in range(count)]
    opening: list[str] = []
    order: list[int] = []
    for index, word in enumerate(target):
        if aligned[index] is None:
            (inserts[order[-1]] if order else opening).append(str(word))
            continue
        at, op = aligned[index]
        ops[at] = op
        order.append(at)
    return ops, [" ".join(one) for one in inserts], " ".join(opening), order


def label_ask(text: str):
    """(record, why): what the encoder is to read a request as, from what
    `rephrase.requested` puts it as; kept only when the labels say the same
    request back (`rephrase.said_back`)."""
    from research.v688.rephrase import (asked_words, request_of, requested,
                                        said_back, saying)

    said = " ".join((text or "").replace(chr(8217), "'").split())
    stripped = said.rstrip("?.! ")
    words = asked_words(stripped)
    if not words:
        return None, "empty"
    found = requested(said)
    request = request_of(found)
    if request is None:
        return None, "unrequested"
    ops, inserts, opening, order = rewrite_labels(
        words, asked_words(found.text), saying)
    record = {"task": "ask", "words": [word.lower() for word in words],
              "typed": words, "names": [], "tags": [""] * len(words),
              "deps": [""] * len(words), "request": request, "ops": ops,
              "inserts": inserts, "opening": opening, "order": order,
              "said": said}
    built = said_back(stripped, words,
                      gold(HEAD_KINDS, record, FIELDS["ask"]))

    def shown(one):
        return (asked_words(one.text), one.note, one.asking, one.why,
                one.negative)

    if shown(built) != shown(found):
        DIFFERENCES["ask"] = (shown(found), shown(built), record)
        return None, "differs"
    return record, ""


def _split_at(lower: list[str], split) -> tuple | None:
    """(the word placing it, the anchor's words, the main clause's words), as
    positions among `lower`, for what `tense.subordinate` split off."""
    from .reading import tokens_of
    from .tense import SUBORDINATORS

    main_text, relation, anchor_text = split
    main_words, anchor_words = tokens_of(main_text)[0], tokens_of(anchor_text)[0]
    count = len(lower)
    for sub in range(count):
        if SUBORDINATORS.get(lower[sub]) != relation:
            continue
        if sub == 0 and "," in lower:
            comma = lower.index(",")
            anchor = list(range(1, comma))
            main = [at for at in range(comma + 1, count) if lower[at] != ","]
        else:
            main = [at for at in range(sub) if lower[at] != ","]
            anchor = [at for at in range(sub + 1, count) if lower[at] != ","]
        if ([lower[at] for at in main] == main_words
                and [lower[at] for at in anchor] == anchor_words):
            return sub, anchor, main
    return None


def _when_tags(when, lower: list[str]) -> list[str] | None:
    """Each word's label for what `tense.take` took off, from where the
    words it took were said."""
    tags = ["O"] * len(lower)
    again = [where(word) for word in when.again_words]
    cut = [where(word) for word in when.words]
    if None in again or None in cut:
        return None
    for at in again:
        tags[at] = "AGAIN"
    rest = [at for at in cut if at not in again]
    if when.frame is not None:
        label = when.frame.label.split()
        for start in range(len(rest) - len(label) + 1):
            run = rest[start:start + len(label)]
            if ([lower[at] for at in run] == label
                    and run == list(range(run[0], run[0] + len(label)))):
                for at in run:
                    tags[at] = "FRAME"
                break
        else:
            return None
    for at in rest:
        if tags[at] == "O":
            if not when.link:
                return None
            tags[at] = f"LINK-{when.link}"
    return tags


def _when_view(when) -> tuple:
    return (when.frame.as_dict() if when.frame else None, when.link,
            when.again, [str(one) for one in when.again_words],
            [str(one) for one in when.words])


@taught_by_cues
def label_place(text: str, lexicon, names: frozenset = frozenset()):
    """(record, why): what the encoder is to place an utterance as -- as
    `rephrase.requested` asks it -- from what the rules take off it and the
    words they leave in normal form; kept only when the labels place it the
    same way (`reader.placed`)."""
    from research.v688.rephrase import requested

    from .reader import placed, placing
    from .reading import new_names, normal, pieces, taught, tokens_of
    from .tense import subordinate, take

    lexicon = Plain(lexicon)
    asked = requested((text or "").strip()).text
    lower, typed = pieces(asked)
    count = len(lower)
    if not count:
        return None, "empty"
    fresh = new_names(*tokens_of(asked), lexicon, names)
    known = names | fresh
    new = ["NAME" if lower[at] in fresh and typed[at][:1].isupper() else "O"
           for at in range(count)]
    anchor = ["O"] * count
    main = [at for at in range(count) if lower[at] != ","]
    relation = anchor_text = ""
    split = subordinate(asked)
    if split is not None:
        clause = taught(split[2], lexicon, known, anchored=False)
        if clause.act == "tell" and clause.mention is not None:
            located = _split_at(lower, split)
            if located is None:
                return None, "unsplit"
            sub, anchored, main = located
            relation, anchor_text = split[1], split[2]
            anchor[sub] = f"SUB-{relation}"
            for at in anchored:
                anchor[at] = "ANCHOR"
    tokens = [At(lower[at], at) for at in main]
    shown = [At(typed[at], at) for at in main]
    tokens, shown, when = take(tokens, shown, known)
    timed = _when_tags(when, lower)
    if timed is None:
        return None, "untimed"
    said_lower, said_typed = normal(list(tokens), list(shown), lexicon)
    left = {where(word) for word in shown}
    ops, inserts, opening, order = rewrite_labels(
        typed, list(said_typed), placing(lexicon),
        frozenset(at for at in range(count) if at not in left))
    tags, deps = analysed(typed, lexicon)
    record = {"task": "place", "words": lower, "typed": typed,
              "names": sorted(names), "tags": tags, "deps": deps,
              "new": new, "when": timed, "anchor": anchor, "ops": ops,
              "inserts": inserts, "opening": opening, "order": order,
              "said": asked}
    built = placed(lower, typed, gold(HEAD_KINDS, record, FIELDS["place"]),
                   lexicon, names)
    wanted = ([str(one) for one in said_lower],
              [str(one) for one in said_typed], _when_view(when), relation,
              tokens_of(anchor_text)[0], fresh, [str(one) for one in shown])
    got = (built.tokens, built.typed, _when_view(built.when), built.relation,
           tokens_of(built.anchor)[0], built.fresh, built.main)
    if got != wanted:
        DIFFERENCES["place"] = (wanted, got, record)
        return None, "differs"
    return record, ""


def _phrase(kept: list, phrase: str, texts_only: bool = False,
            after: int = 0) -> tuple[int, int] | None:
    """Where a phrase's words are among a question's tokens: said, or as
    their lemmas."""
    wanted = phrase.lower().split()
    size = len(wanted)
    for start in list(range(after, len(kept) - size + 1)) + list(
            range(0, min(after, len(kept) - size + 1))):
        group = kept[start:start + size]
        if ([token.text.lower() for token in group] == wanted
                or (not texts_only
                    and [token.lemma_.lower() for token in group] == wanted)):
            return start, start + size
    return None


def _target_span(doc, kept: list, target: str,
                 after: int) -> tuple[int, int] | None:
    """Where a target's words are among a question's tokens, as said:
    `can't mind` is two tokens and one stretch of text."""
    wanted = " ".join(target.lower().split())
    for start in list(range(after, len(kept))) + list(range(after)):
        for stop in range(len(kept), start, -1):
            said = doc[kept[start].i:kept[stop - 1].i + 1].text.lower()
            if " ".join(said.split()) == wanted:
                return start, stop
    return None


def _parse_view(parse) -> tuple:
    return (parse.subject, parse.relation, parse.target, parse.polar,
            parse.unknown, parse.hedged, parse.verb_slot, parse.note,
            [token["pos"] for token in parse.tokens])


def label_parse(question: str, parser):
    """(record, why): what the encoder is to read a question as for v687,
    from what the table of cues reads (`Parser.cued`); kept only when the
    labels parse it the same way (`Parser.parsed`)."""
    if parser.nlp is None:
        return None, "no parser"
    text = (question or "").strip().rstrip("?").strip()
    doc = parser.nlp(text)
    kept = [token for token in doc if not token.is_punct
            and not token.is_space]
    if not kept:
        return None, "empty"
    found = parser.cued(question)
    roles = ["O"] * len(kept)
    end = 0
    subject = found.subject or found.unknown
    if subject:
        span = _phrase(kept, subject)
        if span is None:
            return None, "unaligned subject"
        for at in range(*span):
            roles[at] = "SUBJ"
        end = span[1]
    if found.target:
        span = _target_span(doc, kept, found.target, end)
        if span is None:
            return None, "unaligned target"
        for at in range(*span):
            roles[at] = "TARGET"
    if found.verb_slot:
        order = list(range(end, len(kept))) + list(range(end))
        at = next((at for at in order if roles[at] in ("O", "TARGET")
                   and kept[at].lemma_.lower() == found.verb_slot), None)
        if at is None:
            at = next((at for at in order if roles[at] == "SUBJ"
                       and kept[at].lemma_.lower() == found.verb_slot), None)
        if at is None:
            return None, "unaligned verb"
        roles[at] = {"O": "VERB", "TARGET": "TARGET-VERB",
                     "SUBJ": "SUBJ-VERB"}[roles[at]]
    record = {"task": "parse", "words": [token.text.lower() for token in kept],
              "typed": [token.text for token in kept], "names": [],
              "tags": [token.tag_ for token in kept],
              "deps": [token.dep_ for token in kept],
              "relation": getattr(found, "cue", None) or "none",
              "polar": "yes" if found.polar else "no", "parse": roles,
              "said": question}
    built = parser.parsed(question, doc, kept,
                          gold(HEAD_KINDS, record, FIELDS["parse"]))
    if _parse_view(built) != _parse_view(found):
        DIFFERENCES["parse"] = (_parse_view(found), _parse_view(built),
                                record)
        return None, "differs"
    return record, ""


class Probing:
    """A lexicon that keeps every question v689's reading puts to v687's
    parser (`subject`), for the parse to be taught on the questions it is
    really asked."""

    def __init__(self, lexicon, asked: set) -> None:
        self.lexicon = lexicon
        self.asked = asked

    def __getattr__(self, name: str):
        return getattr(self.lexicon, name)

    def subject(self, question: str):
        self.asked.add(str(question))
        return self.lexicon.subject(question)


# -- what is read ----------------------------------------------------------------
#: The grammar's question shapes, each with its phrase left open: the ones
#: the reading equivalence compares, which variation starts from.
SHAPES = [
    "when did {x} chase the cat", "when did {x} go to the kitchen",
    "when was {x} in the kitchen", "how many times did {x} bark",
    "how often did {x} bark", "how many times has {x} barked",
    "who did {x} give the ball to", "who did {x} get the milk from",
    "whom did {x} give the ball to", "who gave {x} the ball",
    "who chased {x}", "who did not bark", "who barked", "who can swim",
    "who has {x}", "who is carrying {x}", "who owns {x}",
    "what did {x} chase", "what did {x} not eat", "what does {x} eat",
    "what did {x} give to John", "what can {x} eat",
    "where did {x} go", "where did {x} go first", "where did {x} go last",
    "where did {x} drop the ball", "where does {x} live",
    "where is {x}", "where was {x}", "where are {x}",
    "where was {x} before the garden", "where was {x} after the kitchen",
    "what is {x} on", "what is {x} in", "what was {x} under",
    "who is in {x}", "what is in {x}", "is anyone in {x}",
    "how many people are in {x}", "how many balls are in {x}",
    "what is {x} carrying", "what was {x} holding",
    "how many objects is {x} carrying", "how many things is {x} holding",
    "is anyone carrying {x}", "does anyone have {x}",
    "does {x} have the ball", "is {x} carrying the ball",
    "how many things does {x} have", "what does {x} have",
    "what did {x} have", "did anyone go to {x}",
    "how many people went to {x}", "what is {x} doing",
    "what was {x} doing", "what is north of {x}", "what is {x} north of",
    "what is {x} above", "what is {x} below", "what is {x} afraid of",
    "what color is {x}", "where will {x} go",
    "how do you go from {x} to the garden", "how many dogs are there",
    "how many beagles have come up", "which dog is black", "which one is {x}",
    "what happened", "what happened to {x}", "what happened first",
    "what will happen tomorrow", "what did i tell you", "what did i say",
    "what did {x} do", "what did {x} do first", "what can {x} do",
    "what can't {x} do", "what do you know about {x}", "tell me about {x}",
    "what about a cat", "and a fish", "what kind of dog is {x}",
    "who is {x}", "what is {x}", "who am i", "what is my name",
    "how do you know that", "why can {x} swim", "why not", "why", "why so",
    "why is that",
    "is {x} in the kitchen", "can {x} swim", "does {x} bark",
    "who invented the telephone", "what is the capital of france",
]

PHRASES = ["it", "he", "she", "the dog", "Mary", "the second beagle",
           "my beagle", "the dogs", "a dog", "the other one", "that one"]


def _test_strings(folders=("v689",)) -> list[str]:
    """Every short string in these versions' tests that reads as something
    said."""
    found = []
    paths = [path for folder in folders for path in
             sorted((REPOSITORY / "research" / folder).glob("test_*.py"))]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            text = node.value.strip()
            count = len(text.split())
            if (2 <= count <= 16 and "\n" not in text and len(text) < 120
                    and not text.startswith(("test", "research", "needs"))
                    and not any(mark in text for mark in "{}_=<>/\\|:")):
                found.append(text)
    return found


def _babi(questions: int = 6, statements: int = 2) -> list[str]:
    """bAbI's training lines, at most a few of each wording with its names
    and places left open."""
    import re

    from . import babi

    kept: dict[str, list[str]] = collections.defaultdict(list)
    for task in range(1, 21):
        for story in babi.stories(task, "train"):
            for line in story.lines:
                shape = re.sub(r"\b[A-Z][a-z]+\b", "N", line.text)
                limit = questions if line.question else statements
                if len(kept[shape]) < limit and line.text not in kept[shape]:
                    kept[shape].append(line.text)
    return [text for texts in kept.values() for text in texts]


#: A cell or act the last corpus taught fewer times than this is said in
#: `BOOST` times as many ways.
RARE = 400
BOOST = 4


def _contrasts() -> list[str]:
    """Questions worded like a cell's that ask something else, for the
    grammar to read: `how old is Mary` is not where she is, `where did Mary
    grow up` not where she went, `who did Mary call` not who she gave
    something to. Adjectives WordNet has as an attribute's values; verbs with
    a particle Brown tags as one that VerbNet does not say leave their subject
    anywhere; verbs VerbNet has with an object."""
    from nltk.corpus import wordnet

    from . import change

    counts, tagged = _brown(), _brown_tags()
    rng = random.Random(689)
    found = []
    adjectives = sorted({lemma.name() for synset in wordnet.all_synsets("a")
                         if synset.attributes() for lemma in synset.lemmas()
                         if lemma.name().isalpha()
                         and counts[lemma.name()] >= 50})
    found += [f"how {adjective} is {rng.choice(PHRASES)}"
              for adjective in adjectives]
    particles = {word for word, seen in tagged.items()
                 if seen["RP"] >= 0.3 * sum(seen.values())}
    for verb in sorted(change.frames()):
        parts = verb.split()
        if (len(parts) == 2 and parts[1] in particles
                and counts[parts[0]] >= 50
                and "location" not in change.meaning(verb)):
            found.append(f"where did {rng.choice(PHRASES)} {verb}")
    transitive = sorted({verb for verb, entries in change.frames().items()
                         if verb.isalpha() and counts[verb] >= 100
                         and any("object" in frame.shape
                                 for frame, _ in entries)})
    found += [f"who did {rng.choice(PHRASES)} {verb}"
              for verb in rng.sample(transitive, min(150, len(transitive)))]
    found += [f"what did {rng.choice(PHRASES)} {verb}"
              for verb in rng.sample(transitive, min(150, len(transitive)))]
    # `is it safe to eat a mushroom`: an `it` that refers to nothing, before
    # an adjective Brown has in `it is ___ to`.
    from nltk.corpus import brown
    said = list(brown.tagged_words())
    expletive = collections.Counter(
        said[at + 2][0].lower() for at in range(len(said) - 3)
        if said[at][0].lower() == "it"
        and said[at + 1][0].lower() in ("is", "was")
        and said[at + 2][1].startswith("JJ")
        and said[at + 3][0].lower() == "to")
    things = sorted({word for synset in wordnet.all_synsets("n")
                     if synset.lexname() in ("noun.food", "noun.artifact")
                     for word in synset.lemma_names()
                     if word.isalpha() and word.islower()
                     and counts[word] >= 30})
    # Only `is it`: the reader's expletive reading, and `was it best to`
    # is a yes or no about something.
    for adjective in sorted(expletive):
        if not adjective.isalpha():
            continue
        for _ in range(3):
            noun = rng.choice(things)
            found.append(f"is it {adjective} to {rng.choice(transitive)} "
                         f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}")
    # `testicles shrink in the cold, and they expand in warm ones`: claims
    # about a kind, joined. WordNet's animals, in their irregular plural
    # where WordNet lists one, and VerbNet's verbs with no object.
    irregular: dict = {}
    for form, lemmas in wordnet._exception_map["n"].items():
        for lemma in lemmas:
            irregular.setdefault(lemma, form)
    kinds = set()
    for synset in wordnet.all_synsets("n"):
        if synset.lexname() != "noun.animal":
            continue
        for word in synset.lemma_names():
            senses = wordnet.synsets(word, "n")
            if (not word.isalpha() or not word.islower() or counts[word] < 20
                    or not senses or senses[0].lexname() != "noun.animal"):
                continue
            if word in irregular:
                plural = irregular[word]
            elif word.endswith("y") and word[-2:-1] not in "aeiou":
                plural = word[:-1] + "ies"
            elif word.endswith(("s", "sh", "ch", "x", "z")):
                plural = word + "es"
            else:
                plural = word + "s"
            if wordnet.morphy(plural, "n") == word:
                kinds.add(plural)
    kinds = sorted(kinds)
    bare = sorted({verb for verb, entries in change.frames().items()
                   if verb.isalpha() and counts[verb] >= 100
                   and any(not frame.shape for frame, _ in entries)})
    for _ in range(120):
        kind = rng.choice(kinds)
        one, other = rng.sample(bare, 2)
        found += [f"{kind} can {one}, and they can {other}",
                  f"{kind} {one}, but they do not {other}",
                  f"{kind} can not {one} or {other}"]
    return found


def _statements(count: int = 30) -> list[str]:
    """Things said rather than asked, for the statement reader to read:
    someone new put down, named, told of, and kinds taught. The words are
    NLTK's names, the nouns WordNet files as animals and artifacts that Brown
    has, VerbNet's verbs with no object and WordNet's attribute adjectives;
    only the frames are written here, and what each says is the grammar's."""
    from nltk.corpus import wordnet

    from . import change

    counts = _brown()
    rng = random.Random(6890)
    nouns = sorted({word for synset in wordnet.all_synsets("n")
                    if synset.lexname() in ("noun.animal", "noun.artifact")
                    for word in synset.lemma_names()
                    if word.isalpha() and word.islower()
                    and counts[word] >= 40})
    verbs = sorted({verb for verb, entries in change.frames().items()
                    if verb.isalpha() and counts[verb] >= 100
                    and any(not frame.shape for frame, _ in entries)})
    adjectives = sorted({lemma.name() for synset in wordnet.all_synsets("a")
                         if synset.attributes() for lemma in synset.lemmas()
                         if lemma.name().isalpha()
                         and counts[lemma.name()] >= 50})
    names = [one for one in _people() if one.isalpha()] or ["Rex", "Ada"]
    frames = (
        "there is a {n}", "there was a {n} that can {v}", "here is a {n}",
        "i have a {n}", "i got a {n}", "we have a {n}", "i saw a {n}",
        "i met a {n}", "i found a {n}", "there is another {n}",
        "another {n} can {v}", "i have a {n} that can not {v}",
        "my name is {N}", "call me {N}", "i am {N}", "{N} is a {n}",
        "the {n} is called {N}", "its name is {N}", "my {n} is named {N}",
        "{N} can {v}", "it can not {v}", "the {n} is {a}", "{N} is {a}",
        "{N} has a {n}", "the {n} can {v}", "{N} and {M} can {v}",
        "{ns} can {v}", "{ns} do not {v}", "a {n} can {v}",
        "{ns} can {v}, and they can {w}", "{N} can {v} but it can not {w}")
    found = []
    for frame in frames:
        for _ in range(count):
            noun = rng.choice(nouns)
            one, other = rng.sample(verbs, 2)
            name, second = rng.sample(names, 2)
            found.append(frame.format(
                n=noun, ns=noun + "s", v=one, w=other, a=rng.choice(adjectives),
                N=name.capitalize(), M=second.capitalize()))
    return found


def _articled(text: str) -> str:
    """Each `a` or `an` as the sound of the word after it takes."""
    words = text.split()
    for at in range(len(words) - 1):
        if words[at] in ("a", "an"):
            words[at] = "an" if words[at + 1][:1] in "aeiou" else "a"
    return " ".join(words)


@lru_cache(maxsize=None)
def _plurals() -> dict:
    """WordNet's irregular plurals: `mouse` -> mice."""
    from nltk.corpus import wordnet

    found: dict = {}
    for form, lemmas in wordnet._exception_map["n"].items():
        for lemma in lemmas:
            found.setdefault(lemma, form)
    return found


def _plural(word: str) -> str:
    if word in _plurals():
        return _plurals()[word]
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"


@lru_cache(maxsize=None)
def _vocabulary() -> dict:
    """What the frames below are filled from, each word one Brown has: the
    nouns WordNet files as animals, artifacts, food and plants (first sense
    only), their parts, their kinds and what they are made of, rooms and
    buildings, VerbNet's verbs with and without an object, WordNet's
    attribute adjectives and pertainym adverbs, and NLTK's names."""
    from nltk.corpus import wordnet

    from . import change

    counts = _brown()

    def common(word: str, least: int) -> bool:
        return word.isalpha() and word.islower() and counts[word] >= least

    things, parts, kinds, stuff = set(), set(), set(), set()
    compounds = set()
    for synset in wordnet.all_synsets("n"):
        if synset.lexname() == "noun.substance":
            stuff.update(one for one in synset.lemma_names()
                         if common(one, 40))
        if synset.lexname() not in ("noun.animal", "noun.artifact",
                                    "noun.food", "noun.plant"):
            continue
        # `field mouse`, `fire truck`: a kind named in two words, both of
        # them words Brown has, as v688's family checks ask about them.
        compounds.update(
            one.replace("_", " ") for one in synset.lemma_names()
            if one.count("_") == 1 and one.islower()
            and all(common(part, 20) for part in one.split("_")))
        names = [one for one in synset.lemma_names() if common(one, 30)]
        if not names or wordnet.synsets(names[0], "n")[0] != synset:
            continue
        things.add(names[0])
        for part in synset.part_meronyms():
            parts.update(one for one in part.lemma_names() if common(one, 10))
        for above in synset.hypernyms():
            kinds.update(one for one in above.lemma_names()
                         if common(one, 10))
    places = set(ROOMS)
    for root in ("room.n.01", "building.n.01"):
        for below in wordnet.synset(root).closure(lambda one: one.hyponyms()):
            places.update(one for one in below.lemma_names()
                          if common(one, 10))
    frames = change.frames()
    bare = sorted(verb for verb, entries in frames.items()
                  if verb.isalpha() and counts[verb] >= 100
                  and any(not frame.shape for frame, _ in entries))
    transitive = sorted(verb for verb, entries in frames.items()
                        if verb.isalpha() and counts[verb] >= 100
                        and any("object" in frame.shape
                                for frame, _ in entries))
    adjectives = sorted({lemma.name() for synset in wordnet.all_synsets("a")
                         if synset.attributes() for lemma in synset.lemmas()
                         if lemma.name().isalpha()
                         and counts[lemma.name()] >= 50})
    adverbs = sorted({lemma.name() for synset in wordnet.all_synsets("r")
                      for lemma in synset.lemmas() if lemma.pertainyms()
                      and lemma.name().isalpha()
                      and lemma.name().endswith("ly")
                      and counts[lemma.name()] >= 5})
    names = [one for one in _people() if one.isalpha() and 3 <= len(one) <= 8
             and not wordnet.synsets(one.lower())]
    return {"things": sorted(things), "compounds": sorted(compounds),
            "parts": sorted(parts),
            "kinds": sorted(kinds), "stuff": sorted(stuff),
            "places": sorted(places), "bare": bare,
            "transitive": transitive, "adjectives": adjectives,
            "adverbs": adverbs, "names": names}


def _fill(frame: str, rng: random.Random) -> str | None:
    """A frame with its words drawn (`_vocabulary`), or None when a verb it
    needs has no regular form to say it in."""
    from research.v688.rephrase import gerund

    words = _vocabulary()
    thing, other, third = rng.sample(words["things"], 3)
    # A quarter of the kinds named in two words.
    if words["compounds"] and rng.random() < 0.25:
        thing = rng.choice(words["compounds"])
    if words["compounds"] and rng.random() < 0.25:
        other = rng.choice(words["compounds"])
    part = rng.choice(words["parts"])
    stuff, more = rng.sample(words["stuff"], 2)
    act = rng.choice(words["transitive"])
    name, second = rng.sample(words["names"], 2)
    values = {"n": thing, "ns": _plural(thing), "m": other,
              "ms": _plural(other), "o": third, "os": _plural(third),
              "v": rng.choice(words["bare"]), "t": act, "g": gerund(act),
              "vbn": inflect(act, "VBN"), "vbd": inflect(act, "VBD"),
              "a": rng.choice(words["adjectives"]), "p": part,
              "ps": _plural(part), "k": rng.choice(words["kinds"]),
              "s": stuff, "s2": more, "l": rng.choice(words["places"]),
              "r": rng.choice(words["adverbs"]), "N": name.capitalize(),
              "M": second.capitalize()}
    if any("{" + key + "}" in frame and not value
           for key, value in values.items()):
        return None
    return _articled(frame.format(**values))


#: The ways a question is put to v687 about a kind -- by v688's loop, by
#: v689 teaching a kind, and by whoever asks the page -- and the ways each
#: relation is named. Only the frames are written here.
KIND_FRAMES = (
    "can a {n} {v}", "can {ns} {v}", "does a {n} {v}", "do {ns} {v}",
    "can a {n} {t} a {m}", "does a {n} {t} {ms}", "is a {n} {a}",
    "are {ns} {a}", "is a {n} a {k}", "is a {n} {k}", "is a {n} a {m}",
    "does a {n} have a {p}", "does a {n} have {ps}", "do {ns} have {ps}",
    "what does a {n} have", "what is a {n} made of", "is a {n} made of {s}",
    "what is a {n} made from", "what is a {n} used for",
    "is a {n} used for {g}", "is a {n} used to {t} {ms}",
    "what is used to {t} {ms}", "where do you find a {n}",
    "where is a {n} found", "where does a {n} live", "where are {ns} kept",
    "is a {n} found in a {l}", "is a {n} located in a {l}",
    "is a {n} kept in a {l}", "is a {p} part of a {n}",
    "what is a {p} part of", "does a {n} want {ms}", "what does a {n} want",
    "does a {n} like {ms}", "what do {ns} desire", "what does a {n} need",
    "does a {n} need {s}", "what is needed to {t} a {m}",
    "does a {n} require {s}", "what does {s} cause", "does {s} cause {s2}",
    "can {s} lead to {s2}", "can a {n} be {vbn}", "can {ns} be {vbn}",
    "does a {n} get {vbn}", "what can a {n} do", "what does a {n} do",
    "which {ns} can {v}", "what kind of {k} is a {n}",
    "what kinds of {ns} are there", "what is a {n}", "why does a {n} {v}",
    "how does a {n} {v}", "what is the difference between a {n} and a {m}",
    "does a {n} have many {ps}", "is a {n} able to {v}",
    "is a {n} capable of {g}", "does a {n} contain {s}",
    "can a {n} {v} into a {m}", "can a {n} {t} {ms} with a {m}",
    "is a large {n} {a}", "can a small {n} {v}", "do all {ns} {v}",
    "is {s} {a}", "is a {n} a kind of {k}", "a {n} can {v}",
    "{ns} {v}", "a {n} has a {p}", "{ns} are {a}", "{N} is a {n}")


def _kind_questions(count: int = 40) -> list[str]:
    """Questions put to v687 about kinds (`KIND_FRAMES`), for its parse,
    and read by every other reader too."""
    rng = random.Random(6870)
    found = []
    for frame in KIND_FRAMES:
        for _ in range(count):
            text = _fill(frame, rng)
            if text:
                found.append(text)
    return found


#: What a request puts a question in or asks it as, beside the frames the
#: teacher strips (`rephrase.EMBEDDING`, `TRUTH`, `LIKELY`).
REQUEST_FRAMES = (
    "describe a {n}", "describe {ns}", "define {n}", "define a {n}",
    "what does {n} mean", "what is the meaning of {n}",
    "what is the purpose of a {n}", "what's the purpose of a {n}",
    "what is the use of a {n}", "what is a {n} good for",
    "is there such a thing as a {n}", "are there such things as {ns}",
    "are there {ns} that can {v}", "are there any {ns} that cannot {v}",
    "are there any {ns} which {v}", "are there two {ns}",
    "are there {ns} in the {l}", "name three {ns}", "name some {ns}",
    "list {ns} that {v}", "name {ns} that can {v}", "list {ns}",
    "list all the {ns} that {v}", "can you {t} {ms} with a {n}",
    "can you {t} a {m} using a {n}", "can you sit on a {n}",
    "can you drink from a {n}", "can you eat a {n}", "can you eat {ns}",
    "could you {t} {ms} with a {n}", "can you eat", "can you drink",
    "can you eat it", "can you drink it", "can you eat {s}",
    "can you drink {s}", "can you eat {ns}", "can you drink a {n}",
    "could you eat", "can you eat the {n}", "what do you use to {t} {ms}",
    "what do people use for {g}", "what do we use to {t} a {m}",
    "can you {v}", "can you see me", "can you {t} it",
    # A pronoun names no thing, so the program is still what is asked. One
    # frame of it against a hundred `can an apple be eaten` rewrites was
    # too few: a retrain said `can you eat it` back as `can you eaten it`.
    "could you {t} it", "can you {t} them", "could you {t} them",
    "can you {t} it for me",
    "can you {v} in the dark", "can you help me with my {n}",
    "is a {n} in a {l}", "are {ns} in the {l}", "is a {n} on a {l}",
    "is a {n} in danger", "is it in a {l}", "is a {n} found in a {l}",
    "why can a {n} {v}", "why can't a {n} {v}", "why do {ns} {v}",
    "why does a {n} {v}", "why don't {ns} {v}", "how come {ns} {v}",
    "how come a {n} can't {v}", "how does a {n} {v}", "how do {ns} {v}",
    "how can a {n} {v}", "how do you {t} a {m}", "why can't it {v}",
    "why does the {n} {v}", "why does it {v}",
    "{ns} are to {ms} as what are to {os}",
    "a {n} is to a {m} as a {o} is to what",
    "if a {n} had {ps} could it {v}", "if {ns} could {v} would they {t} {ms}",
    "can't a {n} {v}", "doesn't a {n} have {ps}", "isn't a {n} {a}",
    "does not a {n} {v}", "aren't {ns} {a}", "won't a {n} {v}",
    "please describe a {n}", "can a {n} {v} please", "please can a {n} {v}")

#: Claims and questions put inside a request.
PUT_INSIDE = ("a {n} can {v}", "{ns} {v}", "a {n} has {ps}", "{ns} are {a}",
              "a {n} is a {k}", "the {n} can {v}", "a {n} {v}s",
              "a {n} can't {v}", "can a {n} {v}", "{N} can {v}")


def _requests(count: int = 25) -> list[str]:
    """Requests around questions and claims about kinds: in the frames the
    teacher strips, and the templates it reads (`REQUEST_FRAMES`)."""
    from research.v688 import rephrase as rules

    rng = random.Random(6880)
    found = []
    frames = rules.EMBEDDING + rules.TRUTH + rules.LIKELY
    for _ in range(count * 12):
        text = _fill(rng.choice(frames) + rng.choice(PUT_INSIDE), rng)
        if text:
            found.append(text)
    for frame in REQUEST_FRAMES:
        for _ in range(count):
            text = _fill(frame, rng)
            if text:
                found.append(text)
    return found


def _times(count: int = 150) -> list[str]:
    """Things said at a time, one after another, again, or against something
    else said, around bAbI's lines and the statement frames': the frames and
    links `tense.py` takes off, and the words `subordinate` splits on --
    `tense.py`'s own tables, so they are what its rules take off."""
    from .reading import NOT_NAMES
    from .tense import AGAIN, FRAMES, LINKS, SUBORDINATORS

    rng = random.Random(6891)
    texts = _babi(questions=1, statements=3)
    statements = [one for one in texts if not one.rstrip().endswith("?")]
    statements += _statements(3)
    questions = [one for one in texts if one.rstrip().endswith("?")]

    def lowered(text: str) -> str:
        text = text.strip().rstrip(".?!")
        first_word = text.split()[0].lower() if text.split() else ""
        return (text[:1].lower() + text[1:] if first_word in NOT_NAMES
                else text)

    found = []
    for _ in range(count):
        claim, other = (lowered(one) for one in rng.sample(statements, 2))
        question = lowered(rng.choice(questions)) if questions else claim
        frame = " ".join(rng.choice(FRAMES)[0])
        link = " ".join(rng.choice(LINKS)[0])
        again = " ".join(rng.choice(AGAIN))
        word = rng.choice(sorted(SUBORDINATORS))
        found += [f"{frame} {claim}", f"{claim} {frame}", f"{link} {claim}",
                  f"{link}, {claim}", f"{claim} {again}",
                  f"{word} {other}, {claim}", f"{claim} {word} {other}",
                  f"{question} {frame}", f"{frame} {claim} {again}"]
    return found


#: Things said in the passive, and with an adverb among them or a phrase
#: before them: what normal form says in the active and without the adjunct.
PASSIVE_FRAMES = (
    "the {n} was {vbn} by {N}", "the {n} was {vbn} to {M} by {N}",
    "the {n} was {vbn} to the {l} by {N}", "the {n} was {vbn} to the {l}",
    "the {n} is {vbn} by {N}", "{N} {r} {vbd} the {n}",
    "{N} {vbd} the {n} {r}", "in the end {N} {vbd} the {n}",
    "{N} went to the {l} {r}", "{N} {r} went to the {l}")


def _passives(count: int = 60) -> list[str]:
    rng = random.Random(6892)
    found = []
    for frame in PASSIVE_FRAMES:
        for _ in range(count):
            text = _fill(frame, rng)
            if text:
                found.append(text)
    return found


#: Where question datasets are kept (`data/questions`, not committed):
#: WikiAnswers' first 20 thousand clusters and QA-SRL Bank 2.1. Both are
#: fetched by `python -m regenerate --only questions`, and both are read
#: only with `--external`. See `data/questions.SOURCE.md`.
QUESTIONS = REPOSITORY / "data" / "questions"

#: How QA-SRL says an argument it does not name.
UNNAMED = frozenset({"someone", "something"})


def _natural(limit: int = 30000) -> list[str]:
    """Questions people asked: WikiAnswers', read by the grammar as it
    stands. One is never read like another of its cluster: a WikiAnswers
    cluster is questions about one thing, not one question -- `where did
    Justin Bieber go to school` sits with `did Justin Bieber go
    snowboarding`.

    Quora's pairs were read here too and were dropped on 2026-09-16: the
    local export could not be traced to a source, several QQP releases carry
    the same `sentence1`/`sentence2` columns, and substituting one would
    have changed what this teaches with nothing to say so. Nothing shipped
    used it -- these datasets are `--external` only.
    """
    found: list[str] = []
    wiki = QUESTIONS / "wikianswers-20k.jsonl"
    if wiki.exists():
        with open(wiki, encoding="utf-8") as handle:
            for line in handle:
                found += json.loads(line)["set"]
    kept = [text for text in dict.fromkeys(one.strip() for one in found)
            if text.isascii() and 3 <= len(text.split()) <= 14
            and text.count("?") <= 1]
    return random.Random(689).sample(kept, min(limit, len(kept)))


def _asked_slot(slots: dict):
    """Which argument a QA-SRL question's wh-word asks: its subject, its
    object, or the object of its preposition."""
    if slots["wh"] not in ("who", "what"):
        return None
    if slots["subj"] == "_":
        return "subj"
    if slots["obj"] == "_" and (slots["prep"] == "_" or slots["obj2"] != "_"):
        return "obj"
    if slots["prep"] != "_" and slots["obj2"] == "_":
        return ("obj2", slots["prep"])
    return None


def _qasrl(limit: int = 30000) -> list[str]:
    """QA-SRL Bank 2.1's questions about who did what to whom, where and
    when, over the verbs of 44 thousand sentences, each argument it leaves
    unnamed said as the answer the question asking it gives: `what does
    someone study`, with `who studies something` answered `Geologists`, is
    `what does Geologists study`. Only questions every unnamed argument of
    which has an answer of at most four words, most annotators agreed on."""
    import gzip

    path = QUESTIONS / "qasrl" / "qasrl-v2_1" / "expanded" / "train.jsonl.gz"
    if not path.exists():
        return []
    found: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            sentence = json.loads(line)
            tokens = sentence["sentenceTokens"]
            for verb in sentence["verbEntries"].values():
                answers: dict = {}
                questions = []
                for question in verb["questionLabels"].values():
                    judged = question["answerJudgments"]
                    good = [one for one in judged
                            if one["isValid"] and one["spans"]]
                    if len(good) * 2 <= len(judged):
                        continue
                    start, end = collections.Counter(
                        tuple(one["spans"][0]) for one in good
                    ).most_common(1)[0][0]
                    slots = question["questionSlots"]
                    asked = _asked_slot(slots)
                    if asked is not None:
                        answers.setdefault(asked, " ".join(tokens[start:end]))
                    questions.append(question)
                for question in questions:
                    slots = question["questionSlots"]
                    if slots["obj2"] not in ("_",) and slots["obj2"] not in \
                            UNNAMED:
                        continue
                    fill = []
                    for slot in ("subj", "obj", "obj2"):
                        if slots[slot] in UNNAMED:
                            key = (slot if slot != "obj2"
                                   else ("obj2", slots["prep"]))
                            fill.append(answers.get(key))
                    if any(one is None or len(one.split()) > 4
                           or one.split()[0].lower() in ("how", "that",
                                                         "whether", "what")
                           for one in fill):
                        continue
                    words = []
                    for word in question["questionString"].rstrip("?").split():
                        words.append(fill.pop(0) if word in UNNAMED and fill
                                     else word)
                    found.append(" ".join(words) + "?")
    kept = list(dict.fromkeys(found))
    return random.Random(689).sample(kept, min(limit, len(kept)))


def documented() -> list[str]:
    """Every example quoted in the docstrings of the grammar's shapes, and
    the questions its module docstring lists as read into slots."""
    import re

    from . import grammar
    from .reading import QUESTION_WORDS

    found: list[str] = []
    shapes = {shape for table in grammar.SHAPES.values() for shape in table}
    for shape in sorted(shapes, key=lambda one: one.__name__):
        found += re.findall(r"`([^`]+)`", shape.__doc__ or "")
    doc = grammar.__doc__ or ""
    start = doc.find("Some shapes read a question straight into slots")
    block = doc[start:doc.find("That goal is tried first", start)]
    for line in block.splitlines()[1:]:
        found += [piece.strip() for piece in line.split(",") if piece.strip()]
    # `where did Mary go (first)`: the question, without its option.
    found = [" ".join(re.sub(r"\([^)]*\)", " ", one).split()) for one in found]
    return [one for one in dict.fromkeys(found)
            if one.split() and one.split()[0].lower() in QUESTION_WORDS]


#: Sources said in more ways, with other names, nouns and verbs (`vary`);
#: questions people asked are taught as they were asked.
VARIED = frozenset({"tests", "grammar", "babi", "contrast", "statement",
                    "time", "passive"})


def sources(external: bool = False) -> list[tuple[str, str]]:
    """(text, where it is from), each text once. The question datasets
    (`_natural`, `_qasrl`) only with `external`: read by this grammar they
    taught unfamiliar wording as generic -- WikiAnswers' `where is the X
    located` is about no one here -- and cost the held-out paraphrases."""
    texts = [(one, "tests") for one in _test_strings()]
    texts += [(one, "grammar") for one in documented()]
    for shape in SHAPES:
        texts += [(one, "grammar") for one in
                  ([shape.format(x=phrase) for phrase in PHRASES]
                   if "{x}" in shape else [shape])]
    texts += [(one, "babi") for one in _babi()]
    texts += [(one, "contrast") for one in _contrasts()]
    texts += [(one, "statement") for one in _statements()]
    # What v687 is asked and v688 is requested, for their readers: the
    # strings of their tests, questions about kinds, requests, and things
    # said in time and in the passive for placing.
    texts += [(one, "tests") for one in _test_strings(("v687", "v688"))]
    texts += [(one, "kind question") for one in _kind_questions()]
    texts += [(one, "request") for one in _requests()]
    texts += [(one, "time") for one in _times()]
    texts += [(one, "passive") for one in _passives()]
    if external:
        texts += [(one, "natural") for one in _natural()]
        texts += [(one, "qasrl") for one in _qasrl()]
    seen: set = set()
    kept = []
    for text, kind in texts:
        if text not in seen:
            seen.add(text)
            kept.append((text, kind))
    return kept


@lru_cache(maxsize=None)
def _people() -> tuple:
    from nltk.corpus import names
    return tuple(sorted(set(names.words())))


@lru_cache(maxsize=None)
def _brown() -> collections.Counter:
    from nltk.corpus import brown
    return collections.Counter(word.lower() for word in brown.words())


@lru_cache(maxsize=None)
def _brown_tags() -> dict:
    """word -> how often Brown tags it each way."""
    from nltk.corpus import brown
    found: dict = collections.defaultdict(collections.Counter)
    for word, tag in brown.tagged_words():
        found[word.lower()][tag] += 1
    return dict(found)


def names_in(text: str) -> frozenset:
    """The names a text uses: a capitalised word that is no word WordNet has,
    or one NLTK's names corpus lists, and that no reader list says opens a
    sentence (`The`, `Then`, `Where`)."""
    from nltk.corpus import wordnet

    from .reading import AUX, MODALS, NOT_NAMES, QUESTION_WORDS, tokens_of
    from .tense import FRAMES, LINKS

    opening = {entry[0][0] for entry in FRAMES + LINKS}
    people = {one.lower() for one in _people()}
    lower, typed = tokens_of(text)
    found = set()
    for word, shown in zip(lower, typed):
        if (not shown[:1].isupper() or not word.isalpha()
                or word in NOT_NAMES or word in AUX or word in QUESTION_WORDS
                or word in MODALS or word in opening):
            continue
        if word in people or not wordnet.synsets(word):
            found.add(word)
    return frozenset(found)


# -- resources --------------------------------------------------------------------
#: WordNet's noun files a story's things and places come from.
NOUN_FILES = frozenset({"noun.artifact", "noun.food", "noun.animal",
                        "noun.plant", "noun.object", "noun.substance",
                        "noun.location"})

#: The places a story's rooms are kinds of are found from these.
ROOMS = ("kitchen", "bathroom", "garden", "office", "hallway", "bedroom",
         "park", "school", "cinema")


def inflect(verb: str, tag: str, lexicon=None) -> str:
    """A verb in the form a Penn tag names, or empty: irregular forms from
    WordNet's exception list, told apart by the tagger; regular ones checked
    by WordNet's morphy."""
    from nltk.corpus import wordnet
    from research.v688.rephrase import gerund

    from .frames import _irregular

    if tag in ("VB", "VBP"):
        return verb
    if verb in ("have", "be"):
        return {("have", "VBZ"): "has", ("have", "VBD"): "had",
                ("have", "VBN"): "had", ("have", "VBG"): "having",
                ("be", "VBZ"): "is", ("be", "VBD"): "was",
                ("be", "VBN"): "been", ("be", "VBG"): "being"}.get(
                    (verb, tag), "")
    irregular = _irregular().get(verb, ())
    if tag == "VBG":
        form = gerund(verb)
    elif tag == "VBZ":
        if verb.endswith("y") and verb[-2:-1] not in "aeiou":
            form = verb[:-1] + "ies"
        elif verb.endswith(("s", "sh", "ch", "x", "z", "o")):
            form = verb + "es"
        else:
            form = verb + "s"
    elif tag in ("VBD", "VBN"):
        for form in irregular:
            probe = ["it", form] if tag == "VBD" else ["it", "has", form]
            tags = lexicon.tags(probe) if lexicon is not None else None
            if tags and tags[-1] == tag:
                return form
        if irregular:
            return ""
        if verb.endswith("e"):
            form = verb + "d"
        elif verb.endswith("y") and verb[-2:-1] not in "aeiou":
            form = verb[:-1] + "ied"
        else:
            form = verb + "ed"
    else:
        return ""
    return form if wordnet.morphy(form, "v") == verb else ""


class Pools:
    """What variation and the transports draw their words from, read out of
    WordNet, VerbNet, NLTK's names and the Brown corpus."""

    def __init__(self, lexicon) -> None:
        import xml.etree.ElementTree as ET

        from nltk.corpus import wordnet

        from . import change
        from .reading import QUESTION_WORDS

        counts = _brown()
        tagged = _brown_tags()

        def usually(word: str, *prefixes: str) -> bool:
            """Does Brown tag the word so more often than any other way?"""
            found = tagged.get(word)
            return bool(found) and found.most_common(1)[0][0].startswith(
                prefixes)

        self.names = [one for one in _people()
                      if 3 <= len(one) <= 8 and one.isalpha()
                      and not wordnet.synsets(one.lower())]
        # A story's rooms, in the sense that is a place, and the kinds of
        # place they are: what a thing is kept apart from, and what `which
        # room` asks by.
        rooms = [next((one for one in wordnet.synsets(word, "n")
                       if one.lexname() in ("noun.artifact", "noun.location")),
                      None) for word in ROOMS]
        rooms = [one for one in rooms if one is not None]
        kinds = {above for room in rooms for above in room.hypernyms()}
        self.kind_of: dict[str, str] = {}
        nouns = collections.defaultdict(list)
        for synset in wordnet.all_synsets("n"):
            if synset.lexname() not in NOUN_FILES:
                continue
            for word in synset.lemma_names():
                if (word in self.kind_of or not word.isalpha()
                        or not word.islower() or counts[word] < 3
                        or not usually(word, "NN")):
                    continue
                senses = wordnet.synsets(word, "n")
                if not senses or senses[0] != synset:
                    continue
                if not lexicon.known(word):
                    continue
                above = set(synset.closure(lambda one: one.hypernyms()))
                kind = ("place" if above & kinds or synset in kinds
                        else synset.lexname())
                self.kind_of[word] = kind
                nouns[kind].append(word)
        self.nouns = {kind: sorted(words) for kind, words in nouns.items()}
        self.place_kinds = sorted({
            word for synset in kinds for word in synset.lemma_names()
            if word.isalpha() and counts[word] >= 10})
        # Verbs by VerbNet class, the ones Brown has at all often.
        members = collections.defaultdict(set)
        self.classes: dict[str, set] = collections.defaultdict(set)
        for verb, entries in change.frames().items():
            for frame, _ in entries:
                members[frame.klass].add(verb)
                self.classes[verb].add(frame.klass)
        self.members = {klass: sorted(one for one in verbs
                                      if one.isalpha() and counts[one] >= 10)
                        for klass, verbs in members.items()}

        def lexname(verb: str) -> str:
            senses = wordnet.synsets(verb.replace(" ", "_"), "v")
            return senses[0].lexname() if senses else ""

        location = [verb for verb in change.frames()
                    if "location" in change.meaning(verb)]
        # Staying somewhere, for `where is Mary ___ing`: WordNet's stative
        # verbs VerbNet places their subject with.
        self.locating = sorted(
            verb for verb in location if verb.isalpha() and counts[verb] >= 5
            and lexname(verb) == "verb.stative" and usually(verb, "VB"))
        # Ending up somewhere with a particle, for `where did Mary turn up`:
        # a verb and a word Brown tags as a particle.
        particles = {word for word, found in tagged.items()
                     if found["RP"] >= 0.3 * sum(found.values())}
        # Not WordNet's synonyms of these: they bring `blow up` and `switch
        # off`, which leave no one anywhere.
        self.particles = sorted(
            verb for verb in location if len(verb.split()) == 2
            and verb.split()[1] in particles and counts[verb.split()[0]] >= 20)
        # Finding where someone is, for `where can I find Mary`: WordNet's
        # senses of finding and locating that are about a place or a search.
        self.finding = sorted(
            {word for verb in ("find", "locate")
             for synset in wordnet.synsets(verb, "v")
             if any(key in synset.definition()
                    for key in ("location", "place", "search"))
             for word in synset.lemma_names()
             if word.isalpha() and counts[word] >= 20})
        self.possessing = sorted(
            verb for verb in change.frames() if verb.isalpha()
            and "possession" in change.meaning(verb)
            and change.accompanies(verb)
            and lexname(verb) in ("verb.possession", "verb.stative"))
        # Verbs of knowing that take a wh-clause, of telling someone one,
        # and of wondering: the WordNet synonyms of VerbNet's asking verbs.
        wanted = {"NP V what S", "NP V how/whether S", "NP V what S_INF"}
        told = {"NP V NP what S"}
        knowing, telling, inquiring = set(), set(), set()

        def walk(klass, inherited: set) -> None:
            primaries = set(inherited) | {
                frame.find("DESCRIPTION").get("primary")
                for frame in klass.findall("FRAMES/FRAME")}
            for member in klass.findall("MEMBERS/MEMBER"):
                verb = member.get("name")
                if (klass.get("ID") or "").startswith("inquire"):
                    inquiring.add(verb)
                if (not verb.isalpha() or counts[verb] < 50
                        or not usually(verb, "VB")):
                    continue
                if primaries & wanted and lexname(verb) == "verb.cognition":
                    knowing.add(verb)
                if primaries & told and lexname(verb) in (
                        "verb.communication", "verb.cognition"):
                    telling.add(verb)
            for sub in klass.findall("SUBCLASSES/VNSUBCLASS"):
                walk(sub, primaries)

        for path in sorted(Path(change.VERBNET_DIR).glob("*.xml")):
            walk(ET.parse(path).getroot(), set())
        self.knowing, self.telling = sorted(knowing), sorted(telling)
        # `i wonder where`: a verb of knowing that is also one of asking's
        # WordNet synonyms.
        self.asking = sorted(
            {word for verb in inquiring for synset in wordnet.synsets(verb, "v")
             for word in synset.lemma_names()} & knowing)
        # Adverbs of manner and certainty WordNet derives from adjectives,
        # said with their `-ly`: `fastest` and `alone` say more than how.
        self.adverbs = sorted(
            {lemma.name() for synset in wordnet.all_synsets("r")
             for lemma in synset.lemmas() if lemma.pertainyms()
             and lemma.name().isalpha() and lemma.name().endswith("ly")
             and counts[lemma.name()] >= 5})
        self.times = sorted(
            {word for synset in wordnet.all_synsets("n")
             if synset.lexname() == "noun.time"
             for word in synset.lemma_names()
             if word.isalpha() and word.islower() and counts[word] >= 30
             and wordnet.synsets(word, "n")[0].lexname() == "noun.time"
             and usually(word, "NN")})
        self.openers = sorted(
            word for word, found in tagged.items()
            if word.isalpha() and sum(found.values()) >= 4
            and word not in QUESTION_WORDS and usually(word, "UH"))

    def as_dict(self) -> dict:
        return {key: (len(value) if isinstance(value, (list, dict))
                      else value)
                for key, value in vars(self).items()
                if key not in ("classes",)}

    def name(self, rng: random.Random) -> str:
        return rng.choice(self.names)

    def noun_like(self, word: str, rng: random.Random) -> str:
        kind = self.kind_of.get(word)
        return rng.choice(self.nouns[kind]) if kind else word

    def verb_like(self, verb: str, tag: str, rng: random.Random,
                  lexicon) -> str:
        classes = sorted(self.classes.get(verb, ()))
        if not classes:
            return ""
        others = [one for one in self.members.get(rng.choice(classes), ())
                  if one != verb]
        for other in rng.sample(others, min(4, len(others))):
            form = inflect(other, tag, lexicon)
            if form:
                return form
        return ""

    def time_phrase(self, rng: random.Random) -> list[str]:
        noun = rng.choice(self.times)
        return rng.choice([["at", "the", noun], ["this", noun],
                           ["that", noun], ["in", "the", noun],
                           ["during", "the", noun]])

    def request(self, rng: random.Random) -> list[str]:
        """Words a question can be put inside: `can you tell me`, `do you
        know`, `i wonder`."""
        choices = []
        if self.telling:
            verb = rng.choice(self.telling)
            choices += [["can", "you", verb, "me"], ["could", "you", verb, "me"],
                        [verb, "me"], ["would", "you", verb, "me"]]
        if self.knowing:
            verb = rng.choice(self.knowing)
            choices += [["do", "you", verb], ["can", "you", verb],
                        ["i", "want", "to", verb],
                        ["i", "would", "like", "to", verb]]
        if self.asking:
            choices.append(["i", rng.choice(self.asking)])
        return rng.choice(choices)


# -- saying it otherwise -------------------------------------------------------------
DETERMINERS = frozenset({"the", "a", "an", "my", "this", "that", "another",
                         "your", "his", "her"})


def vary(text: str, names: frozenset, pools: Pools, lexicon,
         rng: random.Random) -> tuple[str, frozenset]:
    """The text with other names, other nouns of the same kind and other
    verbs of the same VerbNet class: for the grammar to read again."""
    from .reading import AUX, PRONOUNS, tags_of, tokens_of

    lower, typed = tokens_of(text)
    tags = tags_of(typed, lexicon) or []
    mapping: dict[str, str] = {}
    out = []
    for at, (word, shown) in enumerate(zip(lower, typed)):
        if word in names:
            if word not in mapping:
                mapping[word] = (rng.choice(["he", "she", "it"])
                                 if rng.random() < 0.08 else pools.name(rng))
            out.append(mapping[word])
            continue
        before = lower[at - 1] if at else ""
        if before in DETERMINERS and word in pools.kind_of \
                and rng.random() < 0.6:
            out.append(pools.noun_like(word, rng))
            continue
        if (len(tags) == len(lower) and tags[at].startswith("VB")
                and word not in AUX and rng.random() < 0.35):
            other = pools.verb_like(lexicon.lemma(word), tags[at], rng,
                                    lexicon)
            if other:
                out.append(other)
                continue
        out.append(shown)
    new = frozenset(one.lower() for one in mapping.values()
                    if one not in PRONOUNS)
    ending = "?" if text.rstrip().endswith("?") else ""
    return " ".join(out) + ending, (names - set(mapping)) | new


def _copy(record: dict, words: list[str], stated: list[str],
          slotted: list[str], how: str,
          clauses: list[str] | None = None) -> dict:
    if clauses is None:
        clauses = (list(record["clauses"])
                   if len(words) == len(record["words"])
                   else ["O"] * len(words))
    return {**record, "words": words, "stated": stated, "slotted": slotted,
            "clauses": clauses, "how": how, "said": " ".join(words),
            "typed": words}


def _inside(record: dict, at: int) -> bool:
    return at < len(record["words"]) and (
        record["stated"][at].startswith("I-")
        or record["slotted"][at].startswith("I-"))


def _insert(record: dict, at: int, words: list[str], how: str) -> dict:
    size = len(words)
    return _copy(record, record["words"][:at] + words + record["words"][at:],
                 record["stated"][:at] + ["O"] * size + record["stated"][at:],
                 record["slotted"][:at] + ["O"] * size
                 + record["slotted"][at:], how,
                 record["clauses"][:at] + ["O"] * size
                 + record["clauses"][at:])


def adjunct(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    from .reading import QUESTION_WORDS

    words = record["words"]
    places = [len(words)]
    if words[0] in QUESTION_WORDS:
        places.append(1)
    verb = first(record["stated"], "REST")
    if verb:
        places.append(verb)
    at = rng.choice(places)
    if _inside(record, at):
        return None
    return _insert(record, at, [rng.choice(pools.adverbs)], "adjunct")


def timed(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    return _insert(record, len(record["words"]), pools.time_phrase(rng),
                   "timed")


def opened(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    return _insert(record, 0, [rng.choice(pools.openers)], "opened")


def _subject_after(record: dict, at: int) -> int | None:
    """The end of a subject phrase that starts at `at`, in either view."""
    ends = [end for start, end in spans(record["stated"], "SUBJ")
            + spans(record["slotted"], "SUBJ") if start == at]
    return max(ends) if ends else None


def embedded(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    """`where is Mary` as `can you tell me where Mary is`."""
    from .reading import AUX, QUESTION_WORDS

    words = record["words"]
    if (record["act"] in STATEMENTS or len(words) < 2
            or words[0] not in QUESTION_WORDS or words[0] == "why"):
        return None
    order: list[int] | None = None
    replaced: dict[int, str] = {}
    aux = next((at for at in range(1, len(words)) if words[at] in AUX), None)
    if aux is not None and all(
            record["stated"][at] in ("O", "B-KIND", "I-KIND")
            and record["slotted"][at] in ("O", "B-KIND", "I-KIND")
            for at in range(aux)):
        end = _subject_after(record, aux + 1)
        if end is not None and words[end:end + 1] != ["not"]:
            if words[aux] in ("do", "does", "did"):
                if end < len(words):
                    form = inflect(words[end], {"did": "VBD", "does": "VBZ",
                                                "do": "VBP"}[words[aux]],
                                   lexicon)
                    if form:
                        order = (list(range(aux)) + list(range(aux + 1, end))
                                 + list(range(end, len(words))))
                        replaced[end] = form
            else:
                order = (list(range(aux)) + list(range(aux + 1, end)) + [aux]
                         + list(range(end, len(words))))
    if order is None and (record["act"] == "subject occurrence"
                          or record["slots"].split()[:1] in (["subject"],
                                                             ["count"])):
        order = list(range(len(words)))
    if order is None:
        return None
    prefix = pools.request(rng)
    size = len(prefix)
    return _copy(record,
                 prefix + [replaced.get(at, words[at]) for at in order],
                 ["O"] * size + [record["stated"][at] for at in order],
                 ["O"] * size + [record["slotted"][at] for at in order],
                 "embedded")


def place_kind(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    """`where is Mary` as `which room is Mary in`, `in what area is Mary`."""
    from .reading import COPULA

    words = record["words"]
    if words[0] != "where" or len(words) < 3 or not pools.place_kinds:
        return None
    kind = rng.choice(pools.place_kinds)
    asked = rng.choice(["which", "what"])
    located = (record["act"] == "place located" and words[1] in COPULA
               and _subject_after(record, 2) == len(words))
    moved = (record["slots"] == "place occurrence"
             and words[1] in ("did", "does", "has", "had")
             and record["slotted"][-1] == "REST"
             and _subject_after(record, 2) == len(words) - 1)
    if not (located or moved):
        return None
    preposition = rng.choice(["in", "at"]) if located else "to"
    if rng.random() < 0.5:
        found = _copy(record, [asked, kind] + words[1:],
                      ["O", "O"] + record["stated"][1:],
                      ["O", "O"] + record["slotted"][1:], "place kind")
        return _insert(found, len(found["words"]), [preposition],
                       "place kind")
    return _copy(record, [preposition, asked, kind] + words[1:],
                 ["O", "O", "O"] + record["stated"][1:],
                 ["O", "O", "O"] + record["slotted"][1:], "place kind")


def by_location(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    """`where is Mary` as `where is Mary hiding`, `where did Mary turn up`."""
    words = record["words"]
    if (record["act"] != "place located" or record["slots"] != NONE
            or words[:1] != ["where"] or words[1:2] not in (["is"], ["was"])
            or _subject_after(record, 2) != len(words)):
        return None
    subject = list(range(2, len(words)))
    if rng.random() < 0.5 and pools.locating:
        form = inflect(rng.choice(pools.locating), "VBG", lexicon)
        return _insert(record, len(words), [form], "by location") \
            if form else None
    if not pools.particles:
        return None
    verb, particle = rng.choice(pools.particles).split(" ", 1)
    tense = rng.choice(["did", "has"])
    said = verb if tense == "did" else inflect(verb, "VBN", lexicon)
    if not said:
        return None
    added = [said] + particle.split()
    return _copy(record, ["where", tense] + [words[at] for at in subject]
                 + added,
                 ["O", "O"] + [record["stated"][at] for at in subject]
                 + ["O"] * len(added),
                 ["O", "O"] + [record["slotted"][at] for at in subject]
                 + ["O"] * len(added), "by location")


def by_having(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    """`who has the pear` as `who possesses the pear`."""
    if not record["slots"].endswith("holding") or not pools.possessing:
        return None
    at = first(record["slotted"], "VERB")
    if at is None or record["words"][at].endswith("ing"):
        return None
    tags = lexicon.tags(record["words"]) or []
    if len(tags) != len(record["words"]):
        return None
    form = inflect(rng.choice(pools.possessing), tags[at], lexicon)
    if not form:
        return None
    words = list(record["words"])
    words[at] = form
    return _copy(record, words, record["stated"], record["slotted"],
                 "by having")


def by_finding(record: dict, pools: Pools, lexicon, rng) -> dict | None:
    """`where is Mary` as `where can I find Mary`."""
    words = record["words"]
    if (record["act"] != "place located" or record["slots"] != NONE
            or words[:1] != ["where"] or words[1:2] not in (["is"], ["was"])
            or _subject_after(record, 2) != len(words) or not pools.finding):
        return None
    added = ["where", rng.choice(["can", "could", "would", "will", "might",
                                  "do"]),
             rng.choice(["i", "we", "you", "one"]), rng.choice(pools.finding)]
    size = len(added)
    return _copy(record, added + words[2:],
                 ["O"] * size + record["stated"][2:],
                 ["O"] * size + record["slotted"][2:], "by finding")


#: How many ways at most a record is said by a transport tried on nearly
#: every record it applies to: the scarce wordings.
SCARCE = 3

#: (transport, how often it is tried on a question, on a statement)
TRANSPORTS = ((adjunct, 0.4, 0.15), (timed, 0.15, 0.05), (opened, 0.15, 0.1),
              (embedded, 0.5, 0.0), (place_kind, 0.9, 0.0),
              (by_location, 0.9, 0.0), (by_having, 0.9, 0.0),
              (by_finding, 0.6, 0.0))


def transports(record: dict, pools: Pools, lexicon, rng) -> list[dict]:
    out = []
    asking = record["act"] not in STATEMENTS
    for transport, question, statement in TRANSPORTS:
        chance = question if asking else statement
        if rng.random() >= chance:
            continue
        # A wording few records can be said in (`which room is Mary in`) is
        # said up to `SCARCE` ways each, or it comes and goes between runs.
        said: set = set()
        for _ in range(SCARCE if chance >= 0.9 else 1):
            try:
                found = transport(record, pools, lexicon, rng)
                if found is not None and rng.random() < 0.3:
                    found = adjunct(found, pools, lexicon, rng) or found
            except Exception:                           # noqa: BLE001
                found = None
            if (found is not None and len(found["words"]) <= 40
                    and tuple(found["words"]) not in said):
                said.add(tuple(found["words"]))
                out.append(found)
    return out


# -- the corpus ----------------------------------------------------------------
_WORKER: dict = {}


def _start_worker(store: str) -> None:
    from . import babi
    from .session import Session

    babi._start(store, 1)
    lexicon = Session(babi._ASKER, conversation="teach",
                      example=True).lexicon()
    _WORKER.update(lexicon=lexicon, pools=Pools(lexicon),
                   parser=lexicon.asker.parser)


#: How many of the questions a job's reading put to v687 are parsed.
PROBES = 40


def _others(text: str, known: frozenset, lexicon, parser, seen: set,
            tasks=("ask", "place", "parse")):
    """(records, reasons): the same words asked, placed and parsed -- what
    `rephrase`, `reader.place` and `Parser.parse` are taught -- each once a
    job."""
    labellers = {"ask": lambda: label_ask(text),
                 "place": lambda: label_place(text, lexicon, known),
                 "parse": lambda: label_parse(text, parser)}
    records, reasons = [], collections.Counter()
    for task in tasks:
        key = (task, text, known if task == "place" else frozenset())
        if key in seen:
            continue
        seen.add(key)
        try:
            record, why = labellers[task]()
        except Exception as bad:                        # noqa: BLE001
            record, why = None, f"error {type(bad).__name__}"
        reasons[f"{task} {why or 'taught'}"] += 1
        if record is not None:
            records.append(record)
    return records, reasons


def _teach(job) -> tuple[list[dict], collections.Counter]:
    index, text, seed, variants, rare = job
    lexicon, pools = _WORKER["lexicon"], _WORKER["pools"]
    parser = _WORKER["parser"]
    asked: set = set()
    lexicon = Probing(lexicon, asked)
    rng = random.Random(seed)
    names = names_in(text)
    reasons: collections.Counter = collections.Counter()
    # What the last corpus had few of is said in more ways.
    try:
        source, _ = label(text, lexicon, names)
    except Exception:                                   # noqa: BLE001
        source = None
    if source is not None and (source["act"] in rare
                               or source["slots"] in rare):
        variants *= BOOST
    todo = [(text, names, "source")]
    # The same words when no one they name has been told of: `Winona is a
    # mouse` introduces her, and `where is Winona` asks of no one here.
    if names:
        todo.append((text, frozenset(), "unnamed"))
    # And with only the first of several told of: `Mary gave the football to
    # John` once Mary is known and John never was. Every name known or none
    # were the only two taught, so the mix was a guess, and a reader that
    # guessed John known answered `who has the football` with nobody.
    if len(names) > 1:
        from .reading import tokens_of
        said = [word for word in tokens_of(text)[0] if word in names]
        todo.append((text, frozenset(said[:1]), "partly"))
    # And denied, where the grammar reads a denial: `why can it not swim`.
    if source is not None and source["act"] in NEGATED \
            and "not" not in source["words"]:
        from .reading import AUX, tokens_of
        words = tokens_of(text)[0]
        at = next((index for index in range(1, len(words))
                   if words[index] in AUX), None)
        if at is not None:
            todo.append((" ".join(words[:at + 1] + ["not"]
                                  + words[at + 1:]), names, "negated"))
    for _ in range(variants):
        try:
            said, known = vary(text, names, pools, lexicon, rng)
            todo.append((said, known, "varied"))
        except Exception:                               # noqa: BLE001
            reasons["unvaried"] += 1
    records, seen, others = [], set(), set()

    def add(more, why, how) -> None:
        reasons.update(why)
        for one in more:
            one.update(source=index, how=how)
            records.append(one)

    for said, known, how in todo:
        # The same words with other names known are another reading.
        if (said, known) in seen:
            continue
        seen.add((said, known))
        try:
            record, why = label(said, lexicon, known)
        except Exception as bad:                        # noqa: BLE001
            record, why = None, f"error {type(bad).__name__}"
        reasons[why or "taught"] += 1
        add(*_others(said, known, lexicon, parser, others), how)
        if record is None:
            continue
        if how == "varied" and record["act"] in STATEMENTS \
                and rng.random() < 0.5:
            continue
        record.update(source=index, how=how, task="read")
        records.append(record)
        for moved in transports(record, pools, lexicon, rng):
            # Words were added: spaCy reads the new sentence again.
            moved["tags"], moved["deps"] = analysed(moved["words"], lexicon)
            records.append(moved)
            # And placed: a request, a time or an adverb said around it.
            add(*_others(" ".join(moved["words"]), frozenset(moved["names"]),
                         lexicon, parser, others, ("place",)), moved["how"])
    # What reading all that asked v687, parsed as v687 is asked it.
    for probe in sorted(asked)[:PROBES]:
        add(*_others(probe, frozenset(), lexicon, parser, others,
                     ("parse",)), "probe")
    return records, reasons


def corpus(variants: int, processes: int, limit: int = 0,
           external: bool = False) -> None:
    import multiprocessing

    from research.v687 import build as store

    texts = sources(external)
    if limit:
        texts = random.Random(689).sample(texts, min(limit, len(texts)))
    print(f"{len(texts)} sources, {variants} variants each of "
          f"{sorted(VARIED)}: "
          f"{dict(collections.Counter(kind for _, kind in texts))}",
          flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rare: frozenset = frozenset()
    if (DATA / "stats.json").exists():
        counts = json.loads((DATA / "stats.json").read_text(
            encoding="utf-8"))["counts"]
        rare = frozenset(key.split(" ", 1)[1] for key, value in counts.items()
                         if key.split(" ", 1)[0] in ("act", "slots")
                         and value < RARE)
        print(f"rare in the last corpus: {sorted(rare)}", flush=True)
    jobs = [(index, text, 689_000 + index,
             variants if kind in VARIED else 0, rare)
            for index, (text, kind) in enumerate(texts)]
    # A test's own strings are always taught: each is the one example of
    # its wording there is.
    held = {index for index, (text, kind) in enumerate(texts)
            if zlib.crc32(text.encode()) % 10 == 0 and kind != "tests"}
    placeholders = frozenset(NAMED)
    once: set = set()
    reasons: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    started = time.time()
    with (multiprocessing.get_context("spawn").Pool(
            processes, initializer=_start_worker,
            initargs=(str(store.DEFAULT_STORE),)) as pool,
          open(DATA / "train.jsonl", "w", encoding="utf-8") as train,
          open(DATA / "valid.jsonl", "w", encoding="utf-8") as valid):
        for done, (records, why) in enumerate(
                pool.imap_unordered(_teach, jobs, chunksize=4)):
            reasons.update(why)
            for record in records:
                task = task_of(record)
                known = set(record.get("names", ()))
                if task in ("read", "place") and any(
                        word in placeholders and word not in known
                        for word in record["words"]):
                    reasons["placeholder said"] += 1
                    continue
                if task != "read":
                    # Asked, placed or parsed once, whichever job it came
                    # from.
                    key = (task, tuple(record["words"]), tuple(sorted(known)))
                    if key in once:
                        continue
                    once.add(key)
                into = valid if record["source"] in held else train
                into.write(json.dumps(record) + "\n")
                counts[("valid" if into is valid else "train",
                        f"{task} {record['how']}")] += 1
                if task == "read":
                    counts[("act", record["act"])] += 1
                    counts[("slots", record["slots"])] += 1
                else:
                    counts[(task, " ".join(
                        str(one) for one in taught_as(record)[1:]))] += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} in {time.time() - started:.0f}s",
                      flush=True)
    stats = {"sources": len(texts), "variants": variants,
             "reasons": dict(reasons),
             "counts": {" ".join(key): value for key, value in
                        sorted(counts.items())}}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1),
                                     encoding="utf-8")
    print(json.dumps(stats, indent=1))


#: What a social phrase is said with around it (`social.py`).
SOCIAL_OPENINGS = ("", "", "oh ", "well ", "so ", "ok ")
SOCIAL_CLOSINGS = ("", "", " again", " there", " then")
SOCIAL_MARKS = ("", "", "!", ".", "?")


def social_texts(seed: int = 689) -> list[str]:
    """Every social phrase (`social.PHRASES`), as said and with the words and
    marks said around one, and two said together (`hi, how are you`)."""
    from .social import PHRASES

    rng = random.Random(seed)
    found: list[str] = []
    for act, phrases in PHRASES.items():
        for phrase in phrases:
            found.append(phrase)
            found.append(phrase.capitalize() + rng.choice(SOCIAL_MARKS))
            for _ in range(3):
                said = (rng.choice(SOCIAL_OPENINGS) + phrase
                        + rng.choice(SOCIAL_CLOSINGS))
                if rng.random() < 0.5:
                    said = said.capitalize()
                found.append(said + rng.choice(SOCIAL_MARKS))
    for _ in range(120):
        first, second = rng.sample(list(PHRASES), 2)
        found.append(f"{rng.choice(PHRASES[first])}, "
                     f"{rng.choice(PHRASES[second])}")
    # What says something of you or me in the words social phrases are said
    # in -- `i am happy`, `you are a teacher`, `that is heavy` -- is no social
    # act: taught beside them, as the grammar reads it. Without these,
    # `i am sorry` taught `i am happy` to be praise.
    words = _vocabulary()
    for _ in range(SOCIAL_CONTRASTS):
        frame = rng.choice(SOCIAL_CONTRAST_FRAMES)
        found.append(_articled(frame.format(
            a=rng.choice(words["adjectives"]), n=rng.choice(words["things"]))))
    return list(dict.fromkeys(found))


#: Statements said in the words of social phrases, and how many are taught.
SOCIAL_CONTRAST_FRAMES = (
    "i am {a}", "i am so {a}", "you are {a}", "you are very {a}",
    "i am a {n}", "you are a {n}", "i am not {a}", "you are not {a}",
    "that is {a}", "that is a {n}", "it is {a}", "i have a {n}",
    "i am a {a} {n}", "you are a {a} {n}")
SOCIAL_CONTRASTS = 240


def social(store: str) -> None:
    """Records for the social acts, read, asked and placed as the rules and
    the grammar read them, into `train-social.jsonl` and
    `valid-social.jsonl`: every tenth phrase held out."""
    _start_worker(store)
    lexicon, parser = _WORKER["lexicon"], _WORKER["parser"]
    reasons: collections.Counter = collections.Counter()
    seen: set = set()
    with open(DATA / "train-social.jsonl", "w", encoding="utf-8") as train, \
            open(DATA / "valid-social.jsonl", "w", encoding="utf-8") as valid:
        for index, text in enumerate(social_texts()):
            into = valid if zlib.crc32(text.encode()) % 10 == 0 else train
            record, why = label(text, lexicon, frozenset())
            reasons[why or "taught"] += 1
            if record is not None:
                record.update(source=f"social-{index}", how="social",
                              task="read")
                into.write(json.dumps(record) + "\n")
                reasons[f"act {record['act']}"] += 1
            more, why = _others(text, frozenset(), lexicon, parser, seen,
                                ("ask", "place"))
            reasons.update(why)
            for one in more:
                one.update(source=f"social-{index}", how="social")
                into.write(json.dumps(one) + "\n")
    print(json.dumps(dict(reasons), indent=1))


# -- training ---------------------------------------------------------------------
#: What share of each epoch each reader's records are.
#: Reading replies back took 16% at first, most of it from parsing, and a
#: word the parser's vocabulary lacks (`can a goldfish walk on land`) was
#: read half as its teacher reads it: v688 then answered a question
#: differently. The older readers keep close to the shares they were
#: taught at.
SHARES = {"read": 0.37, "ask": 0.13, "place": 0.18, "parse": 0.24,
          "reply": 0.08}

#: How much a head's loss counts, where not once.
WEIGHTS = {"who": 0.5}


def taught_as(record: dict) -> tuple:
    """What a record teaches, for drawing rarer teaching more often: a
    reading's act and cell; what a request was put as, and whether its words
    change; whether an utterance is placed against a clause, in time, names
    someone new, or changes; a question's relation and whether it is a yes
    or no."""
    task = task_of(record)
    if task == "read":
        return task, record["act"], record["slots"]
    if task == "reply":
        return task, record["stance"], "NEG" in record["parts"]
    same = (all(op == "KEEP" for op in record.get("ops", ()))
            and not record.get("opening") and not any(record.get("inserts", ()))
            and record.get("order") == sorted(record.get("order", ())))
    if task == "ask":
        return task, record["request"], same
    if task == "place":
        return (task, any(one != "O" for one in record["anchor"]),
                any(one != "O" for one in record["when"]),
                "NAME" in record["new"], same)
    return task, record["relation"], record["polar"]


def labels(rows=()) -> dict:
    """Every head, with its kind and every label it says -- the words a
    request or a normal form adds, as the corpus has them -- and the tags and
    dependencies spaCy gave the corpus (0 is none)."""
    from research.encoder import OPS
    from research.v687.language import PARSE_ROLES, POLARITY, RELATIONS
    from research.v688.rephrase import REQUESTS

    from .reader import ANCHOR, NEW, WHEN

    def phrases(task: str, field: str) -> list[str]:
        found: set = set()
        for row in rows:
            if task_of(row) != task:
                continue
            value = row[field]
            found.update(value if isinstance(value, list) else [value])
        return [""] + sorted(found - {""})

    tags = sorted({tag for row in rows for tag in row.get("tags", ()) if tag})
    deps = sorted({dep for row in rows for dep in row.get("deps", ()) if dep})
    said = {
        "acts": list(STATED) + list(ACTS), "slots": [NONE] + list(SLOTS),
        "who": list(WHO), "stated": list(ROLES), "slotted": list(ROLES),
        "clauses": list(CLAUSES), "request": list(REQUESTS),
        "ask_op": list(OPS), "ask_insert": phrases("ask", "inserts"),
        "ask_opening": phrases("ask", "opening"), "ask_next": [],
        "new": list(NEW), "when": list(WHEN), "anchor": list(ANCHOR),
        "place_op": list(OPS), "place_insert": phrases("place", "inserts"),
        "place_opening": phrases("place", "opening"), "place_next": [],
        "relation": list(RELATIONS), "polar": list(POLARITY),
        "parse": list(PARSE_ROLES)}
    if any(task_of(row) == "reply" for row in rows):
        from research.v690.message import STANCE_LABELS
        from research.v690.roundtrip import PARTS

        said.update(stance=list(STANCE_LABELS), part=list(PARTS))
    return {"heads": {name: {"kind": HEAD_KINDS[name], "labels": values}
                      for name, values in said.items()},
            "tags": [""] + tags, "deps": [""] + deps}


def _load(name: str) -> list[dict]:
    """The corpus's records for a split, and every other reader's taught
    beside it (`train-social.jsonl`, `train-reply.jsonl`)."""
    rows: list[dict] = []
    for path in [DATA / f"{name}.jsonl"] + sorted(DATA.glob(f"{name}-*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


#: How often a reading or a placing is taught with no parse beside its
#: words: a lexicon need not have spaCy (every test's has none), and reader11,
#: always taught with one, read `why can't it swim` as generic without it.
UNPARSED = 0.3


def _batch(tokenizer, chunk: list[dict], spec: dict, index: dict,
           device: str, rng: random.Random | None = None):
    """(inputs, targets, allowed): the words as read (a known name as one of
    `NAMED`, where names are read), and for every head what each record says
    or -100 where it says nothing. A pointer's target is, for the start and
    each word said back, where the next word is (or the end); `allowed` is
    which words it may point at. With `rng`, a reading or placing is read
    without its parse `UNPARSED` of the time."""
    import torch

    words = [said_as(one["words"], frozenset(one.get("names", ())))
             if task_of(one) in ("read", "place") else one["words"]
             for one in chunk]
    encoded = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=LONGEST, padding=True, return_tensors="pt")

    def parse_of(one: dict) -> tuple:
        if (rng is not None and task_of(one) in ("read", "place")
                and rng.random() < UNPARSED):
            return [""] * len(one["words"]), [""] * len(one["words"])
        return one["tags"], one["deps"]

    tag_ids, dep_ids = features(encoded, [parse_of(one) for one in chunk],
                                spec)
    shape = encoded["input_ids"].shape
    heads = spec["heads"]
    targets, allowed = {}, {}
    for name, head in heads.items():
        targets[name] = torch.full(
            (shape[0],) if head["kind"] == "sentence" else shape, -100)
        if head["kind"] == "pointer":
            allowed[name] = torch.zeros(shape, dtype=torch.bool)
    for row, one in enumerate(chunk):
        firsts, end = positions(encoded, row)
        for name, field in FIELDS[task_of(one)].items():
            kind = heads[name]["kind"]
            if kind == "sentence":
                if name == "who" and one["slots"] == NONE:
                    continue
                targets[name][row] = index[name][one[field]]
            elif kind == "word":
                for word, at in firsts.items():
                    targets[name][row, at] = index[name][one[field][word]]
            else:
                places = [firsts[at] for at in one[field] if at in firsts]
                for current, following in zip([0] + places, places + [end]):
                    targets[name][row, current] = following
                allowed[name][row, places + [end]] = True
    inputs = {"input_ids": encoded["input_ids"].to(device),
              "attention_mask": encoded["attention_mask"].to(device),
              "tag_ids": tag_ids.to(device), "dep_ids": dep_ids.to(device)}
    return (inputs, {name: one.to(device) for name, one in targets.items()},
            {name: one.to(device) for name, one in allowed.items()})


def _scores(out, name: str, kind: str, allowed: dict):
    scores = out[name].float()
    if kind == "pointer":
        scores = scores.masked_fill(~allowed[name][:, None, :], -1e4)
    return scores


def evaluate(net, tokenizer, rows: list[dict], spec: dict, index: dict,
             device: str, batch: int = 256) -> dict:
    """How often each head says what the records say, and how often every
    head says all of a record, for each reader."""
    import torch

    net.eval()
    right, seen = collections.Counter(), collections.Counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            inputs, targets, allowed = _batch(tokenizer, chunk, spec, index,
                                              device)
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                out = net(**inputs)
            whole = torch.ones(len(chunk), dtype=torch.bool, device=device)
            for name, head in spec["heads"].items():
                target = targets[name]
                guess = _scores(out, name, head["kind"], allowed).argmax(-1)
                if head["kind"] == "sentence":
                    present = target != -100
                    good = (guess == target) | ~present
                else:
                    present = (target != -100).any(-1)
                    good = ((guess == target) | (target == -100)).all(-1)
                right[name] += int((good & present).sum())
                seen[name] += int(present.sum())
                whole &= good
            for task in FIELDS:
                mine = torch.tensor([task_of(one) == task for one in chunk],
                                    device=device)
                right[f"{task} whole"] += int((whole & mine).sum())
                seen[f"{task} whole"] += int(mine.sum())
    net.train()
    return {key: round(right[key] / seen[key], 4) for key in seen
            if seen[key]}


def _weights(rows: list[dict]) -> list[float]:
    """How often each record is drawn: by the square root of how rare what
    it teaches is, a text as said three times as often as its variants, and
    each reader its share of the epoch."""
    kinds = [taught_as(row) for row in rows]
    counts = collections.Counter(kinds)
    raw = [counts[kind] ** -0.5
           * (3.0 if row.get("how") in ("source", "unnamed", "partly",
                                         "negated")
              else 1.0) for row, kind in zip(rows, kinds)]
    totals: collections.Counter = collections.Counter()
    for row, one in zip(rows, raw):
        totals[task_of(row)] += one
    return [one * SHARES[task_of(row)] / totals[task_of(row)]
            for row, one in zip(rows, raw)]


def train(base: Path, out: Path, epochs: int, batch: int, rate: float,
          seed: int = 689) -> None:
    import torch
    from transformers import (AutoModel, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows, held = _load("train"), _load("valid")
    weights = _weights(rows)
    spec = labels(rows + held)
    index = {name: {label: at for at, label in enumerate(head["labels"])}
             for name, head in spec["heads"].items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    encoder = AutoModel.from_pretrained(str(base))
    net = Heads(encoder, spec).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * 0.06),
                                               steps)
    loss_of = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # Where a claim begins is one word in many: missed, two claims are one.
    begins = torch.ones(len(spec["heads"]["clauses"]["labels"]))
    begins[1:] = 5.0
    losses = {"clauses": torch.nn.CrossEntropyLoss(weight=begins.to(device),
                                                   ignore_index=-100)}
    print(f"{len(rows)} train, {len(held)} valid, {steps} steps on {device}: "
          f"{dict(collections.Counter(task_of(row) for row in rows))}",
          flush=True)
    started = time.time()
    net.train()
    scores: dict = {}
    for epoch in range(epochs):
        drawn = rng.choices(rows, weights=weights, k=len(rows))
        total = 0.0
        for start in range(0, len(drawn), batch):
            inputs, targets, allowed = _batch(
                tokenizer, drawn[start:start + batch], spec, index, device,
                rng)
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                read = net(**inputs)
            loss = 0.0
            for name, head in spec["heads"].items():
                target = targets[name]
                # A batch with nothing for a head says nothing to it: a mean
                # over nothing is not a number.
                if not bool((target != -100).any()):
                    continue
                scores_of = _scores(read, name, head["kind"], allowed)
                loss = loss + WEIGHTS.get(name, 1.0) * losses.get(
                    name, loss_of)(scores_of.reshape(-1, scores_of.shape[-1]),
                                   target.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        scores = evaluate(net, tokenizer, held, spec, index, device)
        print(f"epoch {epoch + 1}: loss {total:.1f} valid {scores} "
              f"({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    torch.save({key: value for key, value in net.state_dict().items()
                if not key.startswith("encoder.")}, out / "heads.pt")
    (out / "labels.json").write_text(json.dumps({**spec, "base": str(base),
                                                 "valid": scores},
                                                indent=1), encoding="utf-8")
    print(f"saved to {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("corpus", "train", "pools",
                                         "sources", "social"))
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--external", action="store_true",
                        help="also read WikiAnswers and QA-SRL")
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base", default=str(LLM / "MiniLM-L6-v2"))
    parser.add_argument("--out", default=str(LLM / "reader"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--rate", type=float, default=5e-5)
    options = parser.parse_args(argv)
    if options.what == "corpus":
        corpus(options.variants, options.processes, options.limit,
               options.external)
    elif options.what == "train":
        train(Path(options.base), Path(options.out), options.epochs,
              options.batch, options.rate)
    elif options.what == "social":
        from research.v687 import build as store
        social(str(store.DEFAULT_STORE))
    elif options.what == "sources":
        texts = sources(external=True)
        print(dict(collections.Counter(kind for _, kind in texts)))
        rng = random.Random(1)
        for kind in ("natural", "qasrl"):
            chosen = [text for text, one in texts if one == kind]
            print(kind, rng.sample(chosen, min(30, len(chosen))))
    else:
        from research.v687 import build as store
        _start_worker(str(store.DEFAULT_STORE))
        pools = _WORKER["pools"]
        print(json.dumps(pools.as_dict(), indent=1))
        rng = random.Random(1)
        for key in ("place_kinds", "possessing", "particles", "knowing",
                    "telling", "asking", "openers"):
            print(key, getattr(pools, key)[:60])
        print("locating", rng.sample(pools.locating, 30))
        print("adverbs", rng.sample(pools.adverbs, 30))
        print("times", pools.times[:60])
        print({kind: rng.sample(words, min(8, len(words)))
               for kind, words in pools.nouns.items()})
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

"""What an utterance asks, read by an encoder rather than by patterns.

`grammar.py` reads a question by where its words are and which words they
are: `tokens[1] in COPULA`, `tokens[-1] in HAVE`. A word it did not expect
breaks it, so every wording has cost a shape, a list or a rule. Here a
sentence encoder does the recognising
and nothing is matched:

    where did Shanda wind up
      act     place located            what is asked, of which relation
      slots   -                        a goal read straight into slots
      roles   O O B-SUBJ O O           which words fill what

Two heads read the whole utterance -- the act (a cell of the grammar, or one
of `why`, `what`, `define`, `ask`, `ask_name`, `generic`, or a statement) and
the cell a question is read straight into slots as, with who may fill it --
and two tag every word with its role in each: `AUX`, `NEG`, `SUBJ`, `OBJ`,
`REST`, `VERB`, `KIND`, `SEQ` (`ROLES`).

**The encoder recognises; construction is kept.** What a cell's reading is
made of -- which mention, which auxiliary, what is left of the verb phrase,
the markers a relation's operator reads (`?` for the side asked) -- is the
grammar's own construction (`_own_goal`, `Goal`, `Reading`), built here from
the roles (`build`). Who a phrase picks out is still read by
`reading.read_mention`, and a reading whose phrase names nobody here is not
built: the next act the encoder ranks is tried instead, and likewise the
next cell for a goal read as slots. When the phrase names someone in fewer
words than were tagged -- `the second` of `the second beagle`, in a lexicon
without beagles -- the words left over are said of it, as the grammar reads
them.

**Taught by the grammar, offline.** `teach_reader.py` reads a corpus with the
grammar, takes each reading apart into its act, cells and roles, checks that
`build` puts the same reading back together, and fine-tunes the encoder on
that and on wordings of the same readings the grammar never read. At run time
the grammar is not asked.

A statement is read the same way: its act (`introduce`, `tell`, `teach`,
`name`, `compound`), the words of each claim, and where each claim after the
first begins (`CLAUSES`) -- `beagles can swim, and they bark` is two claims,
the second about beagles. What spaCy makes of each word (its tag and its
dependency) is read beside the word, so a word the corpus never had is still
a noun, a verb or an object.

**Placing, before reading** (`place`). The same encoder says who is someone
new, what the words say about when (a time named, a link to what was told
before, `again`), the clause an utterance is placed against (`after the dog
chased the cat, it slept`), and how the rest is said back in normal form --
each word kept, left out or said as its lemma or its past, what is said after
it, and which word comes next: `the apple was given to Fred by Bill` is `bill
gave the apple to fred`. These were `new_names`, `tense.take`,
`tense.subordinate` and `frames.normal`, and are the encoder's teachers now.

There is no other reader: without the model (`llm/reader`) nothing is read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.encoder import BASE, LLM, MODEL, enabled  # noqa: F401

NONE = "none"

#: What a statement does: puts someone new down (and whether they are yours),
#: says something of someone here, of a kind, or what someone is called; or
#: says several things the words do not tell apart.
STATEMENTS = ("introduce", "introduce owned", "tell", "teach", "name",
              "compound")

#: Acts other than a cell.
ACTS = ("why", "what", "define", "ask", "ask_name", "generic") + STATEMENTS

#: Where a claim begins, with what it does, and where the clause a new
#: individual is introduced with begins (`there is a beagle that can't swim`).
CLAUSES = ("O", "B-relative") + tuple(f"B-{one}" for one in ACTS)

#: The subjects that stand for the kind a claim before was about.
STANDING = frozenset({"it", "he", "she", "they"})

#: A noun left out and pointed back to: `in warm ones`.
ELIDED = frozenset({"ones", "one"})

#: The cells a question's own words state (the grammar's `_stated` shapes).
STATED = ("time occurrence", "times occurrence", "recipient occurrence",
          "subject occurrence", "events story", "facts any",
          "object occurrence", "verb occurrence", "place located",
          "object holding", "count holding", "subject dimension",
          "object dimension", "path dimension", "value attribute",
          "object attribute", "place motive", "count is_a", "which is_a",
          "kind is_a", "events future", "events told", "grounds answer",
          "again question")

#: The cells a question is read straight into as slots (`_slots` shapes).
SLOTS = ("subject located", "time located", "any located", "any holding",
         "whether holding", "any occurrence", "count located",
         "count holding", "count occurrence", "subject holding",
         "object holding", "place occurrence")

#: Who may fill a slot: people, things, or the kind the question counts.
WHO = ("", "people", "things", "kind")

ROLES = ("O", "AUX", "NEG", "B-SUBJ", "I-SUBJ", "B-OBJ", "I-OBJ", "REST",
         "VERB", "B-KIND", "I-KIND", "SEQ", "B-NAME", "I-NAME")

#: Cells whose verb phrase is searched for its object, as a told clause is.
WITH_OBJECT = frozenset({"time occurrence", "times occurrence",
                         "subject occurrence", "recipient occurrence",
                         "which is_a"})

#: Cells whose mention the grammar reads over the words before the ones that
#: close it: `what is the cat on`.
TRUNCATED = frozenset({"place located", "verb occurrence", "object holding",
                       "count holding", "object attribute"})

#: Cells and acts whose verb phrase is every word after the mention: words
#: of a tagged phrase that name no one are said of it.
SPILLED = frozenset({"time occurrence", "times occurrence",
                     "object occurrence", "recipient occurrence", "ask",
                     "why", "generic"})

#: Cells whose one-word markers the relation's operator reads.
MARKED = {"events future": "future", "events told": "told"}

#: Mention forms that are a kind, never one of this conversation's.
KINDS = frozenset({"indefinite", "another", "kind"})

#: How many goals-as-slots cells the encoder ranks are tried, and how likely
#: one must be.
SLOTS_TRIED = 2
SLOTS_FLOOR = 0.02

#: What a name this conversation knows is said as to the encoder, in the
#: order the names come: who is named is the discourse layer's to know
#: (`discourse.py`), and what is asked of someone is the same whoever they
#: are. Read as said, a name the corpus had few of was no subject at all:
#: `where is Inessa` was tagged O O O. None of these is ever read as a word
#: that names no one here (`teach_reader.corpus` drops a record that has one):
#: said as `mary`, every known name met the test string `mary put the apple
#: in the kitchen`, where no Mary had been told, and was read as a kind.
NAMED = ("ruth", "owen", "nina", "hugo", "iris", "leon", "vera", "omar")


#: What a kind this conversation taught is said as to the encoder: one the
#: store has, singular and plural. `what is a wemble` asks what a taught kind
#: is, as `what is a dog` does; said as `wemble`, a word the corpus only ever
#: had as nothing the store knows, it was read as nothing here.
TAUGHT = ("dog", "dogs")


def said_as(tokens: list[str], names, kinds=None, lemma=None) -> list[str]:
    """The words as the encoder reads them: each known name as one of
    `NAMED`, the same name as the same one, and each kind the conversation
    taught (`kinds`, by lemma) as `TAUGHT`."""
    order: dict[str, str] = {}
    out = []
    for word in tokens:
        if word in names:
            if word not in order:
                order[word] = NAMED[len(order) % len(NAMED)]
            out.append(order[word])
        elif kinds and word in kinds:
            out.append(TAUGHT[0])
        elif kinds and lemma is not None and lemma(word) in kinds:
            out.append(TAUGHT[1])
        else:
            out.append(word)
    return out


#: The heads an utterance's words are read into readings with
#: (`research/encoder.py`).
READ_HEADS = ("acts", "slots", "who", "stated", "slotted", "clauses")

#: Whether a word names someone nobody here has been called yet.
NEW = ("O", "NAME")

#: What a word says of when (`tense.py`): a time it names, how its clause
#: stands to the occurrence told before (`then`, `before that`, `meanwhile`,
#: `first`, `finally`), or that it happened again.
WHEN = ("O", "FRAME", "LINK-after", "LINK-before", "LINK-during",
        "LINK-first", "LINK-last", "AGAIN")

#: The clause an utterance is placed against, and the word that places it
#: there, with how: `after the dog chased the cat, it slept`.
ANCHOR = ("O", "ANCHOR", "SUB-after", "SUB-before", "SUB-during")

#: The heads an utterance is placed with.
PLACE_HEADS = ("new", "when", "anchor", "place_op", "place_insert",
               "place_opening", "place_next")


# -- roles --------------------------------------------------------------------
def spans(roles: list[str], name: str) -> list[tuple[int, int]]:
    """(start, end) of each run of `B-name I-name ...`."""
    out, start = [], None
    for at, role in enumerate(list(roles) + ["O"]):
        if role == f"I-{name}" and start is not None:
            continue
        if start is not None:
            out.append((start, at))
            start = None
        if role in (f"B-{name}", f"I-{name}"):
            start = at
    return out


def first(roles: list[str], name: str) -> int | None:
    return next((at for at, role in enumerate(roles) if role == name), None)


def mark(roles: list[str], span: tuple[int, int], name: str) -> None:
    start, end = span
    for at in range(start, end):
        roles[at] = ("B-" if at == start else "I-") + name


# -- building a reading from roles --------------------------------------------
def _opener(tokens: list[str], roles: list[str], start: int) -> str:
    from .reading import AUX
    aux = first(roles, "AUX")
    if aux is not None and aux < start:
        return tokens[aux]
    return tokens[start - 1] if start and tokens[start - 1] in AUX else "is"


def mention(tokens: list[str], span, lexicon, names: frozenset, opener: str,
            truncate: bool = False, shorter: bool = False):
    """The phrase over `span`, read as the grammar reads one: over every word
    or over the words up to its end, whichever the cell reads it over first.
    With `shorter`, a phrase that names someone in fewer words will do."""
    from .reading import read_mention
    start, end = span
    for words in ((tokens[:end], tokens) if truncate
                  else (tokens, tokens[:end])):
        found = read_mention(words, start, lexicon, opener, final_ok=True,
                             names=names)
        if found is not None and found.end == end:
            return found
    if shorter:
        found = read_mention(tokens, start, lexicon, opener, final_ok=True,
                             names=names)
        if found is not None and start < found.end < end:
            return found
    return None


def _individual(found) -> bool:
    from .grammar import NO_ONE
    return found is not None and found.form not in NO_ONE


def _progressive(lexicon, word: str) -> str:
    if not word.endswith("ing") or not hasattr(lexicon, "progressive"):
        return ""
    return lexicon.progressive(word) or ""


def _verb(lexicon, word: str) -> str:
    """A verb as a slot holds it: `carrying` -> carry, `has` -> have."""
    if word.endswith("ing"):
        return _progressive(lexicon, word)
    return lexicon.lemma(word)


def _rest(roles: list[str]) -> list[int]:
    return [at for at, role in enumerate(roles)
            if role in ("REST", "B-OBJ", "I-OBJ")]


def _spill(found, span, rest_at: list[int]) -> list[int]:
    """The verb phrase with the words of a tagged phrase its mention did not
    take."""
    if found is None or found.end >= span[1]:
        return rest_at
    return sorted(set(rest_at) | set(range(found.end, span[1])))


def _object(reading, roles, rest_at: list[int], tokens, lexicon,
            names: frozenset, upto: int | None = None) -> None:
    """The object tagged in a verb phrase, read as `reading.object_of`
    reads one: the phrase that closes it."""
    from .reading import read_mention
    found = spans(roles, "OBJ")
    if not found:
        return
    start, end = found[-1]
    if start not in rest_at:
        return
    at = rest_at.index(start)
    rest = reading.rest if upto is None else reading.rest[:upto]
    phrase = read_mention(rest, at, lexicon, "does", final_ok=True,
                          names=names)
    if phrase is not None and phrase.end == len(rest) \
            and phrase.end - at == end - start:
        reading.obj, reading.obj_at = phrase, at


def _kind(words: list[str], lexicon) -> str:
    return " ".join(words[:-1] + [lexicon.lemma(words[-1])]) if words else ""


def stated(cell: str, roles: list[str], tokens: list[str], lexicon,
           names: frozenset, said: str):
    """The reading a question's own words state, for one of `STATED`."""
    from .grammar import _own_goal
    from .reading import Mention, Reading

    asked, relation = cell.split()
    aux_at = first(roles, "AUX")
    aux = tokens[aux_at] if aux_at is not None else None
    rest_at = _rest(roles)
    holds = "NEG" not in roles
    subjects = spans(roles, "SUBJ")
    kinds = spans(roles, "KIND")
    found = None
    if subjects:
        start = subjects[0][0]
        found = mention(tokens, subjects[0], lexicon, names,
                        _opener(tokens, roles, start), cell in TRUNCATED,
                        shorter=cell in SPILLED)
        if cell == "again question":
            if found is None or found.form not in ("indefinite", "kind"):
                return None
        elif not _individual(found):
            return None
        elif (aux is not None and found.form in ("speaker", "addressee")
              and cell in ("object occurrence", "events story",
                           "facts any")):
            # `what do you need to bake a cake`: `you` is anyone.
            return None
        rest_at = _spill(found, subjects[0], rest_at)
    elif cell not in ("subject occurrence", "count is_a", "which is_a",
                      "events story", "events future", "events told",
                      "grounds answer"):
        return None
    rest = [tokens[at] for at in rest_at]
    count, obj = False, None
    if cell == "subject dimension":
        rest = ["?"] + rest
    elif cell == "object dimension":
        rest = rest + ["?"]
    elif cell in MARKED:
        rest = [MARKED[cell]] + rest
    elif cell in ("object holding", "count holding"):
        verb = _progressive(lexicon, rest[-1]) if rest else ""
        if not verb:
            return None
        rest = [verb]
        if asked == "count":
            if not kinds:
                return None
            word = tokens[kinds[0][0]]
            count, obj = True, Mention("kind", word, text=word)
    elif cell == "count is_a":
        words = [tokens[at] for at in range(*kinds[0])] if kinds else []
        end = kinds[0][1] if kinds else (aux_at or len(tokens))
        found = Mention("kind", _kind(words, lexicon), text=" ".join(words),
                        end=end)
    elif cell == "which is_a":
        words = [tokens[at] for at in range(*kinds[0])] if kinds else []
        end = (kinds[0][1] if kinds else aux_at if aux_at is not None
               else rest_at[0] if rest_at else 1)
        kind = "" if words in ([], ["one"]) else lexicon.lemma(words[-1])
        found = Mention("kind", kind, text=" ".join(words), end=end)
        if not rest:
            return None
    elif cell == "path dimension":
        goals = spans(roles, "OBJ")
        if not goals:
            return None
        start = goals[0][0]
        obj = mention(tokens, goals[0], lexicon, names, "is")
        if not _individual(obj):
            return None
        rest = [tokens[at] for at in rest_at if at < start and
                roles[at] == "REST"]
    reading = Reading("question", found, aux, rest, holds=holds, said=said,
                      count=count, obj=obj)
    if cell in WITH_OBJECT:
        upto = len(rest) - 1 if cell == "recipient occurrence" else None
        _object(reading, roles, rest_at, tokens, lexicon, names, upto)
    reading.goals = [_own_goal((asked, relation), reading, lexicon)]
    return reading


def act(name: str, roles: list[str], tokens: list[str], lexicon,
        names: frozenset, said: str):
    """A reading of one of `ACTS`, other than a statement."""
    from .reading import Reading, _whose, bare_kind

    subjects = spans(roles, "SUBJ")
    aux_at = first(roles, "AUX")
    aux = tokens[aux_at] if aux_at is not None else None
    rest_at = _rest(roles)
    if name == "generic" and not subjects:
        return Reading("generic", said=said)
    if name == "why" and not subjects:
        return Reading("why", said=said)
    if not subjects:
        return None
    span = subjects[0]
    if name == "ask_name":
        whose = _whose(list(tokens[span[0]:span[1]]), lexicon, names)
        return Reading("ask_name", whose, said=said) if whose else None
    opener = _opener(tokens, roles, span[0])
    found = mention(tokens, span, lexicon, names, opener,
                    shorter=name in SPILLED)
    if found is None and name == "what":
        found = _subject(tokens, span, lexicon, names)    # `Mary and John`
    if name == "generic":
        if found is None:
            found = bare_kind(tokens, span[0], lexicon)
            if found is None or not span[0] < found.end <= span[1]:
                return None
        if found.form not in KINDS:
            return None
        rest = [tokens[at] for at in _spill(found, span, rest_at)]
        return Reading("generic", found, aux, rest, said=said)
    if found is None:
        return None
    if name in ("what", "define"):
        wanted = found.form == "indefinite" and found.kind \
            if name == "define" else found.form not in ("indefinite",
                                                        "another")
        return Reading(name, found, said=said) if wanted else None
    if found.form in KINDS or aux is None:
        return None
    rest_at = _spill(found, span, rest_at)
    reading = Reading(name, found, aux, [tokens[at] for at in rest_at],
                      holds="NEG" not in roles, said=said)
    _object(reading, roles, rest_at, tokens, lexicon, names)
    return reading


def slotted(cell: str, who: str, roles: list[str], tokens: list[str],
            lexicon, names: frozenset, said: str):
    """The goal a question is read straight into, for one of `SLOTS`."""
    from .grammar import PERSONS, Goal
    from .reading import SEQUENCE, Reading

    asked, relation = cell.split()
    subject = object_ = None
    subjects, objects = spans(roles, "SUBJ"), spans(roles, "OBJ")
    clause_cell = relation == "occurrence"
    if subjects:
        start = subjects[0][0]
        subject = mention(tokens, subjects[0], lexicon, names,
                          _opener(tokens, roles, start),
                          cell in ("count holding", "object holding"))
        if not _individual(subject):
            return None
    if objects and not clause_cell:
        object_ = mention(tokens, objects[0], lexicon, names, "is")
        if not _individual(object_):
            return None
    needs_subject = cell in ("time located", "whether holding",
                             "count holding", "object holding",
                             "place occurrence")
    needs_object = cell in ("subject located", "time located", "any located",
                            "any holding", "whether holding",
                            "count located", "subject holding")
    if (needs_subject and subject is None) or (needs_object
                                               and object_ is None):
        return None
    verb = ""
    verb_at = first(roles, "VERB")
    if relation == "holding":
        if verb_at is None:
            return None
        verb = _verb(lexicon, tokens[verb_at])
        if not verb:
            return None
    if who == "kind":
        words = [tokens[at] for at in range(*spans(roles, "KIND")[0])] \
            if spans(roles, "KIND") else []
        if not words:
            return None
        filled = _kind(words, lexicon)
        filled = "people" if filled in PERSONS else filled
    else:
        filled = who
    sequence, clause = None, None
    if clause_cell:
        aux_at = first(roles, "AUX")
        rest_at = _rest(roles)
        if not rest_at:
            return None
        seq_at = first(roles, "SEQ")
        if seq_at is not None:
            sequence = SEQUENCE.get(tokens[seq_at])
        clause = Reading("question",
                         subject if cell == "place occurrence" else None,
                         tokens[aux_at] if aux_at is not None else None,
                         [tokens[at] for at in rest_at], said=said)
        _object(clause, roles, rest_at, tokens, lexicon, names)
        if cell == "place occurrence":
            if len(rest_at) > 1 and clause.obj is None:
                return None
            verb = lexicon.lemma(tokens[rest_at[0]])
    return Goal(asked, relation, filled, subject, object_, verb, sequence,
                clause, said)



# -- statements ---------------------------------------------------------------------
def _masked(roles: list[str], lo: int, hi: int) -> list[str]:
    """The roles of the words from `lo` up to `hi`; every other word O."""
    return [role if lo <= at < hi else "O" for at, role in enumerate(roles)]


def _body(roles: list[str], tokens: list[str], lexicon, names: frozenset,
          said: str = ""):
    """A verb phrase as a statement, as `reading.clause` puts one: its
    auxiliary, whether it is denied, what is said, and the object that closes
    it."""
    from .reading import Reading

    rest_at = _rest(roles)
    if not rest_at:
        return None
    aux_at = first(roles, "AUX")
    reading = Reading("tell",
                      aux=tokens[aux_at] if aux_at is not None else None,
                      rest=[tokens[at] for at in rest_at],
                      holds="NEG" not in roles, said=said)
    _object(reading, roles, rest_at, tokens, lexicon, names)
    return reading


def _subject(tokens: list[str], span, lexicon, names: frozenset):
    """Who a statement is about: one phrase, or two names (`Mary and
    Daniel`), told of each."""
    from .reading import Mention, read_mention

    found = mention(tokens, span, lexicon, names, "does")
    if found is not None:
        return found
    start, end = span
    one = read_mention(tokens, start, lexicon, names=names)
    if (one is None or one.form != "name" or one.end + 1 >= end
            or tokens[one.end] != "and"):
        return None
    other = read_mention(tokens, one.end + 1, lexicon, names=names)
    if other is None or other.form != "name" or other.end != end:
        return None
    return Mention("group", text=" ".join(tokens[start:end]), end=end,
                   members=[one, other])


def statement(name: str, roles: list[str], clauses: list[str],
              tokens: list[str], typed: list[str], lexicon, names: frozenset,
              said: str):
    """A reading of one of `STATEMENTS`, built from its roles."""
    from .reading import Mention, Reading, _whose, proper

    if name == "compound":
        return Reading("compound", said=said)
    subjects, named = spans(roles, "SUBJ"), spans(roles, "NAME")
    if not subjects:
        return None
    span = subjects[0]
    words = list(tokens[span[0]:span[1]])
    if name == "name":
        if not named:
            return None
        whose = _whose(words, lexicon, names) or mention(
            tokens, span, lexicon, names, "is")
        if whose is None or whose.form in ("indefinite", "another"):
            return None
        return Reading("name", whose, said=said,
                       name=proper(list(typed[named[0][0]:named[0][1]])))
    if name == "teach":
        from .reading import NOT_NAMES, ORDINALS

        led = words[0] in ("a", "an")
        head = words[1:] if led else words
        if (not head or head[0] in names or head[0] in NOT_NAMES
                or head[0] in ORDINALS):
            return None
        lemma = lexicon.lemma(head[-1])
        kind = " ".join(head) if led else " ".join(head[:-1] + [lemma])
        if not led and lemma == head[-1] and not lexicon.known(kind):
            # `Adrian can swim`: a bare singular the store has no word for
            # is someone, not a kind.
            return None
        body = _body(roles, tokens, lexicon, names, said)
        if body is None:
            return None
        body.act = "teach"
        body.mention = Mention("kind", kind, text=" ".join(words),
                               end=span[1])
        return body
    if name in ("introduce", "introduce owned"):
        if named:
            # `Rex is a beagle`: someone new, told their name and kind.
            if len(words) < 2 or words[0] not in ("a", "an"):
                return None
            found = Mention("indefinite", " ".join(words[1:]),
                            text=" ".join(words), end=span[1])
            return Reading("introduce", found, said=said, name=proper(
                list(typed[named[0][0]:named[0][1]])))
        found = mention(tokens, span, lexicon, names, "is")
        if (found is None and len(words) == 2
                and words[0] in ("a", "an", "another")):
            # A kind nothing here has a word for: `there is a wemble`.
            found = Mention("another" if words[0] == "another"
                            else "indefinite", words[1],
                            text=" ".join(words), end=span[1])
        if found is None or found.form not in ("indefinite", "another"):
            return None
        at = next((one for one, label in enumerate(clauses)
                   if label == "B-relative"), None)
        relative = (_body(_masked(roles, at, len(tokens)), tokens, lexicon,
                          names) if at is not None else None)
        return Reading("introduce", found, relative=relative, said=said,
                       owned=name == "introduce owned")
    if name == "tell":
        found = _subject(tokens, span, lexicon, names)
        if found is None or found.form in KINDS:
            return None
        body = _body(roles, tokens, lexicon, names)
        if body is None:
            return None
        body.mention, body.said = found, said
        return body
    return None


def _one(name: str, roles: list[str], clauses: list[str], tokens: list[str],
         typed: list[str], lexicon, names: frozenset, said: str):
    """The reading of one claim, or of a question, without goals as slots."""
    if name in STATED:
        return stated(name, roles, tokens, lexicon, names, said)
    if name in STATEMENTS:
        return statement(name, roles, clauses, tokens, typed, lexicon, names,
                         said)
    return act(name, roles, tokens, lexicon, names, said)


def _claims(markers: list[tuple[int, str]], name: str, roles: list[str],
            clauses: list[str], tokens: list[str], typed: list[str],
            tags: list[str], deps: list[str], lexicon, names: frozenset,
            said: str):
    """Several claims, each over its own words and each filled in from the
    one before, as `clauses.standalone` fills them: a claim with no subject
    of its own takes the subject before it, and its auxiliary, and its
    denial after `or`; a claim about the kind just taught goes on about it,
    through `they`; `ones` is the noun said before. The word the parse says
    joins two claims (`cc`) is said of neither."""
    roles = list(roles)
    for at, _ in markers:
        if deps and deps[at - 1] == "cc":
            roles[at - 1] = "O"
    bounds = [0] + [at for at, _ in markers] + [len(tokens)]
    acts = [name] + [label for _, label in markers]
    readings, prior, prior_roles = [], None, None
    for index, one_name in enumerate(acts):
        lo, hi = bounds[index], bounds[index + 1]
        own = _masked(roles, lo, hi)
        words = list(tokens)
        if prior is not None:
            subjects = spans(own, "SUBJ")
            subject = [tokens[at] for at in range(*subjects[0])] \
                if subjects else []
            shared = not subjects
            standing = len(subject) == 1 and subject[0] in STANDING
            nouns = [at for at in _rest(prior_roles)
                     if tags and tags[at].startswith("NN")
                     and tokens[at] not in ELIDED]
            for at in _rest(own):
                if tokens[at] in ELIDED and nouns:
                    words[at] = tokens[nouns[-1]]
            if shared and first(own, "AUX") is None:
                aux = first(prior_roles, "AUX")
                if aux is not None:
                    own[aux] = "AUX"
                if (lo and tokens[lo - 1] in ("or", "nor")
                        and "NEG" in prior_roles):
                    own[prior_roles.index("NEG")] = "NEG"
            if (one_name == "teach" and prior.act == "teach"
                    and prior.mention is not None and (shared or standing)):
                if standing:
                    own = ["O" if role.endswith("SUBJ") else role
                           for role in own]
                found = _body(own, words, lexicon, names, said)
                if found is None:
                    return None
                found.act, found.mention = "teach", prior.mention
                readings.append(found)
                prior, prior_roles = found, own
                continue
            if shared:
                for at in range(len(tokens)):
                    if prior_roles[at].endswith("SUBJ"):
                        own[at] = prior_roles[at]
        found = _one(one_name, own, clauses, words, typed, lexicon, names,
                     said)
        if found is None:
            return None
        found.said = said
        readings.append(found)
        prior, prior_roles = found, own
    readings[0].more = readings[1:]
    return readings[0]


@dataclass
class Guess:
    """What the encoder read: acts and slots cells best first, who fills
    the slots, a role per word for each, and where claims begin."""

    acts: list[tuple[str, float]]
    slots: list[tuple[str, float]]
    who: str
    stated: list[str]
    slotted: list[str]
    clauses: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def build(act_name: str, slots, who: str, stated_roles: list[str],
          slotted_roles: list[str], tokens: list[str], lexicon,
          names: frozenset, said: str, typed: list[str] | None = None,
          clauses: list[str] | None = None, tags: list[str] | None = None,
          deps: list[str] | None = None):
    """The reading these labels give, with its goals, or None when a phrase
    they need names nobody here. `slots` is a cell, or cells ranked with how
    likely each is: when the likeliest is a cell that names nobody here, the
    next is tried."""
    typed = list(typed) if typed is not None else list(tokens)
    clauses = list(clauses) if clauses else ["O"] * len(tokens)
    markers = [(at, label[2:]) for at, label in enumerate(clauses)
               if label.startswith("B-") and label != "B-relative" and at]
    if markers:
        return _claims(markers, act_name, stated_roles, clauses, tokens,
                       typed, tags or [], deps or [], lexicon, names, said)
    found = _one(act_name, stated_roles, clauses, tokens, typed, lexicon,
                 names, said)
    if found is None:
        return None
    if act_name in STATEMENTS:
        return found
    ranked = [(slots, 1.0)] if isinstance(slots, str) else list(slots)
    goal = None
    if len(tokens) >= 3 and ranked and ranked[0][0] in SLOTS:
        tried = [cell for cell, chance in ranked
                 if cell in SLOTS and chance >= SLOTS_FLOOR][:SLOTS_TRIED]
        for cell in tried:
            goal = slotted(cell, who, slotted_roles, tokens, lexicon, names,
                           said)
            if goal is not None:
                break
    found.goals = ([goal] if goal is not None else []) + list(found.goals)
    return found


# -- the encoder -----------------------------------------------------------------
def analysed(tokens: list[str], lexicon) -> tuple[list[str], list[str]]:
    """(tags, dependencies): spaCy's reading of each word in the sentence,
    or empty strings where the lexicon has no parser."""
    analyse = getattr(lexicon, "analyse", None)
    found = analyse([str(one) for one in tokens]) if analyse and tokens \
        else None
    if not found or len(found) != len(tokens):
        return [""] * len(tokens), [""] * len(tokens)
    return [one[0] for one in found], [one[1] for one in found]


def guessed(found: dict) -> Guess:
    """A `Guess` from what the encoder's heads read (`encoder.read`)."""
    return Guess(found["acts"], found["slots"], found["who"][0][0],
                 found["stated"], found["slotted"], found["clauses"])


#: How many of the acts the encoder ranks are tried before the reading is
#: left generic, and how likely one after the first must be.
TRIED = 3
ACT_FLOOR = 0.05


def reading(tokens: list[str], lexicon, names: frozenset, said: str,
            model=None, typed: list[str] | None = None):
    """(reading, guess): what the encoder reads the words as, built."""
    from research import encoder

    from .reading import Reading

    if not tokens:
        return Reading("generic", said=said), None
    tags, deps = analysed(tokens, lexicon)
    words = said_as(tokens, names, getattr(lexicon, "kinds", None),
                    getattr(lexicon, "lemma", None))
    if model is not None:
        found = model.guess([words], [(tags, deps)], READ_HEADS)[0]
    else:
        found = encoder.read(words, tags, deps, READ_HEADS)
    guess = guessed(found)
    for rank, (name, chance) in enumerate(guess.acts[:TRIED]):
        # An act the encoder hardly thinks likely is not read into a cell
        # because the likely one named nobody here.
        if rank and chance < ACT_FLOOR:
            break
        found = build(name, guess.slots, guess.who, guess.stated,
                      guess.slotted, tokens, lexicon, names, said, typed,
                      guess.clauses, tags, deps)
        if found is not None:
            return found, guess
    return Reading("generic", said=said), guess


# -- placing an utterance ---------------------------------------------------------
@dataclass
class Placed:
    """What placing an utterance takes off it, and the words left to read."""

    #: the words left, in normal form, lowercased and as typed
    tokens: list
    typed: list
    #: what it says about when (`tense.When`)
    when: object
    #: names nobody here has been called yet
    fresh: frozenset = frozenset()
    #: after | before | during: how it stands to the clause it is placed
    #: against, and that clause as typed
    relation: str = ""
    anchor: str = ""
    #: the words it is placed by, as typed, before normal form
    main: list = field(default_factory=list)
    #: every word but the time words, as typed: the utterance unplaced
    words: list = field(default_factory=list)

    def unanchored(self) -> tuple[list[str], list[str]]:
        """The words read whole, when the clause it was placed against is
        no statement about anyone."""
        return [one.lower() for one in self.words], list(self.words)


def place(text: str, lexicon, names: frozenset = frozenset(),
          anchored: bool = True) -> Placed:
    """An utterance placed, as the encoder reads it: who is someone new,
    what it says about when, the clause it is placed against, and the rest
    said back in normal form -- `can you tell me where Mary is` as `where is
    mary`, `the apple was given to Fred by Bill` as `bill gave the apple to
    fred`."""
    from research import encoder

    from .reading import pieces
    from .tense import When

    lower, typed = pieces(text)
    if not lower:
        return Placed([], [], When())
    tags, deps = analysed(typed, lexicon)
    guess = encoder.read(said_as(lower, names), tags, deps, PLACE_HEADS)
    return placed(lower, typed, guess, lexicon, names, anchored)


def placing(lexicon):
    """How a word is said back in normal form (`encoder.OPS`): as its
    lemma (`where did Mary go`), its simple past (`Bill gave`), or in lower
    case."""
    from .frames import past_form

    def saying(op: str, word: str, at: int = 0) -> str:
        if op == "LEMMA":
            return lexicon.lemma(word.lower())
        if op == "PAST":
            lemma = lexicon.lemma(word.lower())
            return past_form(word.lower(), lemma, lexicon.tags) or word
        if op == "LOWER":
            return word.lower()
        return word

    return saying


def placed(lower: list[str], typed: list[str], guess: dict, lexicon,
           names: frozenset = frozenset(), anchored: bool = True) -> Placed:
    """What `guess` (the encoder's reading, or a teacher's labels in the same
    shape) takes off an utterance and leaves of it."""
    from research.encoder import rewrite

    from .tense import FRAMES, Frame, When

    count = len(lower)
    fresh = frozenset(lower[at] for at, label in enumerate(guess["new"])
                      if label == "NAME" and lower[at] not in names
                      and lower[at] != ",")
    anchors = list(guess["anchor"]) if anchored else ["O"] * count
    sub = next((at for at, label in enumerate(anchors)
                if label.startswith("SUB-")), None)
    anchor_at = [at for at, label in enumerate(anchors)
                 if label == "ANCHOR" and lower[at] != ","]
    relation = anchor = ""
    split: set[int] = set()
    if sub is not None and anchor_at:
        relation = anchors[sub][len("SUB-"):]
        anchor = " ".join(typed[at] for at in anchor_at)
        split = set(anchor_at) | {sub}
    labels = guess["when"]
    timed = {at for at, label in enumerate(labels)
             if label != "O" and lower[at] != "," and at not in split}
    main = [at for at in range(count)
            if at not in timed and at not in split and lower[at] != ","]

    when = When()
    frame = [at for at in sorted(timed) if labels[at] == "FRAME"]
    entry = next((one for one in FRAMES
                  if one[0] == tuple(lower[at] for at in frame)), None)
    if frame and entry is not None:
        when.frame = Frame(entry[1], entry[2], " ".join(entry[0]))
    when.link = next((labels[at][len("LINK-"):] for at in sorted(timed)
                      if labels[at].startswith("LINK-")), "")
    again = [at for at in sorted(timed) if labels[at] == "AGAIN"]
    if again:
        when.again, when.again_words = True, [typed[at] for at in again]
    # In the order they were taken off: from the start, then a time named
    # at the end, then `again` before it.
    first = min(main) if main else count
    taken = ([at for at in sorted(timed) if at < first]
             + [at for at in frame if at >= first]
             + [at for at in again if at >= first])
    when.words = [typed[at] for at in taken]

    kept = set(main)
    ops = [op if at in kept else "DROP"
           for at, op in enumerate(guess["place_op"])]
    out = rewrite(typed, ops, guess["place_insert"],
                  guess["place_opening"][0][0], guess["place_next"],
                  placing(lexicon))
    return Placed([one.lower() for one in out], out, when, fresh, relation,
                  anchor, [typed[at] for at in main],
                  [typed[at] for at in range(count)
                   if at not in timed and lower[at] != ","])

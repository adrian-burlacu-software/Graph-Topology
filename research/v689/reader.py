"""What an utterance asks, read by an encoder rather than by patterns.

`grammar.py` reads a question by where its words are and which words they
are: `tokens[1] in COPULA`, `tokens[-1] in HAVE`. A word it did not expect
breaks it, so every wording has cost a shape, a list or a rule
(`frames.py`, `exemplars.py`). Here a sentence encoder does the recognising
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

A statement goes on to the statement reader (`reading._read`). Without the
model (`llm/reader`), or with `V689_READER=0`, the grammar reads as it did.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

LLM = Path(__file__).resolve().parents[2] / "llm"

#: The fine-tuned reader (another with `V689_READER_MODEL`), and the encoder
#: it starts from.
MODEL = Path(os.environ.get("V689_READER_MODEL") or LLM / "reader")
BASE = LLM / "MiniLM-L6-v2"

STATEMENT = "statement"
NONE = "none"

#: Acts other than a cell: `_read`'s, for a question.
ACTS = ("why", "what", "define", "ask", "ask_name", "generic", STATEMENT)

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
         "VERB", "B-KIND", "I-KIND", "SEQ")

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


def enabled() -> bool:
    return os.environ.get("V689_READER", "1") != "0" and (
        MODEL / "labels.json").exists()


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


@dataclass
class Guess:
    """What the encoder read: acts and slots cells best first, who fills
    the slots, and a role per word for each."""

    acts: list[tuple[str, float]]
    slots: list[tuple[str, float]]
    who: str
    stated: list[str]
    slotted: list[str]
    scores: dict = field(default_factory=dict)


def build(act_name: str, slots, who: str, stated_roles: list[str],
          slotted_roles: list[str], tokens: list[str], lexicon,
          names: frozenset, said: str):
    """The reading these labels give, with its goals, or None when a phrase
    they need names nobody here. `slots` is a cell, or cells ranked with how
    likely each is: when the likeliest is a cell that names nobody here, the
    next is tried."""
    if act_name == STATEMENT:
        return None
    if act_name in STATED:
        found = stated(act_name, stated_roles, tokens, lexicon, names, said)
    else:
        found = act(act_name, stated_roles, tokens, lexicon, names, said)
    if found is None:
        return None
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
class Model:
    """The encoder and its five heads, as trained (`teach_reader.py`)."""

    def __init__(self, path: Path = MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        meta = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        self.labels = meta
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        encoder = AutoModel.from_pretrained(str(path))
        self.net = Heads(encoder, meta)
        self.net.load_state_dict(torch.load(path / "heads.pt",
                                            map_location="cpu"), strict=False)
        self.net.to(self.device)
        if self.device == "cuda":
            self.net.half()
        self.net.eval()

    def guess(self, sentences: list[list[str]]) -> list[Guess]:
        torch = self.torch
        encoded = self.tokenizer(sentences, is_split_into_words=True,
                                 padding=True, truncation=True, max_length=64,
                                 return_tensors="pt")
        with torch.no_grad():
            out = self.net(encoded["input_ids"].to(self.device),
                           encoded["attention_mask"].to(self.device))
        acts, slots, who, stated_roles, slotted_roles = [
            one.float().cpu() for one in out]
        found = []
        labels = self.labels

        def ranked(scores, names):
            chances = scores.softmax(-1)
            return [(names[at], float(chances[at]))
                    for at in chances.argsort(descending=True).tolist()]

        for row, words in enumerate(sentences):
            firsts = {}
            for at, word in enumerate(encoded.word_ids(row)):
                if word is not None and word not in firsts:
                    firsts[word] = at
            found.append(Guess(
                ranked(acts[row], labels["acts"]),
                ranked(slots[row], labels["slots"]),
                labels["who"][int(who[row].argmax())],
                [labels["roles"][int(stated_roles[row, firsts[at]].argmax())]
                 if at in firsts else "O" for at in range(len(words))],
                [labels["roles"][int(slotted_roles[row, firsts[at]].argmax())]
                 if at in firsts else "O" for at in range(len(words))]))
        return found


def Heads(encoder, labels: dict):
    """The encoder with a head for each thing read."""
    import torch

    class _Heads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            size = encoder.config.hidden_size
            self.encoder = encoder
            self.acts = torch.nn.Linear(size, len(labels["acts"]))
            self.slots = torch.nn.Linear(size, len(labels["slots"]))
            self.who = torch.nn.Linear(size, len(labels["who"]))
            self.stated = torch.nn.Linear(size, len(labels["roles"]))
            self.slotted = torch.nn.Linear(size, len(labels["roles"]))

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask
                                  ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return (self.acts(pooled), self.slots(pooled), self.who(pooled),
                    self.stated(hidden), self.slotted(hidden))

    return _Heads()


class _Loaded:
    def __init__(self) -> None:
        self.model: Model | None = None
        self.tried = False
        self.lock = threading.Lock()

    def get(self) -> Model | None:
        with self.lock:
            if not self.tried:
                self.tried = True
                if enabled():
                    try:
                        self.model = Model()
                    except Exception:                   # noqa: BLE001
                        self.model = None
            return self.model


LOADED = _Loaded()

#: How many of the acts the encoder ranks are tried before the reading is
#: left generic, and how likely one after the first must be.
TRIED = 3
ACT_FLOOR = 0.05


def reading(tokens: list[str], lexicon, names: frozenset, said: str,
            model: Model | None = None):
    """(reading, guess): the reading the encoder gives, None for a statement,
    or (None, None) without the model."""
    model = model or LOADED.get()
    if model is None or not tokens:
        return None, None
    from .reading import Reading

    guess = model.guess([said_as(tokens, names,
                                 getattr(lexicon, "kinds", None),
                                 getattr(lexicon, "lemma", None))])[0]
    for rank, (name, chance) in enumerate(guess.acts[:TRIED]):
        # An act the encoder hardly thinks likely is not read into a cell
        # because the likely one named nobody here.
        if rank and chance < ACT_FLOOR:
            break
        if name == STATEMENT:
            return None, guess
        found = build(name, guess.slots, guess.who, guess.stated,
                      guess.slotted, tokens, lexicon, names, said)
        if found is not None:
            return found, guess
    return Reading("generic", said=said), guess

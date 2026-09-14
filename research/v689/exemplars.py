"""Questions read like one already understood: an exemplar memory.

The grammar (`grammar.py`) and the parse (`frames.py`) read what a question's
words and structure say. Neither knows that `where did Mary end up` asks what
`where did Mary go` asks, or `who has got the pear` what `who has the pear`
does -- that is what words mean, not how they are put together. A small
sentence encoder does know it, roughly: MiniLM-L6-v2 (22M parameters, run on
the GPU when there is one, `llm/MiniLM-L6-v2`) puts sentences that mean alike
near each other.

So a question nothing else read a goal from is compared with the questions
the grammar did read, and read *like* the nearest one:

    where did Shanda end up    ~ where did Mary go           place, occurrence
                               -> where did shanda go
    who has got the pear       ~ who has the football        subject, holding
                               -> who has the pear

**What is asked, not of whom.** Both sides are encoded with every phrase that
fills a slot said as `it` (`key`): `where did it end up` against `where did it
go`. Encoded as said, a rare name or a different thing outweighs the
question -- `where did Shanda end up` is 0.23 from `where did mary go`, and
`where did Mary end up` 0.91 -- because in a short sentence the words that
name things are most of what is pooled.

**The encoder proposes; the grammar decides.** The nearest exemplar says which
cell -- what is asked, of which relation -- and nothing else. Who and what are
the question's own phrases (its parse), put in place of the exemplar's own,
and the result is read again by the grammar: only a goal of the same cell is
taken. A proposal is made only above `SIMILARITY`, and only when the nearest
exemplar of any other cell is at least `MARGIN` further away. Only a
wh-question about this conversation's individuals is read this way: a yes or
no has its own reading (`is Mary in the kitchen` is not `who is in the
kitchen`), so has a why, and a question about a kind is v688's.

**Instance-based.** The exemplars start as the examples the grammar documents
of its own shapes (the ones it reads a goal from), and every question the
grammar reads a goal from while running is added. Nothing is trained: memory
grows by use, as ACT-R's instance-based learning does. It is kept per
process, not on disk.

Off with `V689_EXEMPLARS=0`, and silently absent without the model or torch.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .clauses import Word, parse
from .reading import QUESTION_WORDS

MODEL = Path(__file__).resolve().parents[2] / "llm" / "MiniLM-L6-v2"

#: The encoder fine-tuned on the cells (`train_exemplars.py`), used when
#: there is one.
TUNED = MODEL.parent / "MiniLM-L6-v2-cells"

#: How near the nearest exemplar must be, and how much nearer than the
#: nearest of another cell. Set on the development wordings.
SIMILARITY = 0.80
MARGIN = 0.03

#: How many exemplars of the chosen cell are tried as the frame to read in.
TRIED = 5

#: Most exemplars kept.
LIMIT = 20000

#: Question words: the reader's own.
WH = QUESTION_WORDS

#: Dependencies of a phrase that fills a slot.
ARGUMENTS = frozenset({"nsubj", "nsubjpass", "dobj", "pobj", "attr",
                       "dative", "poss", "npadvmod"})

#: The two people in the conversation, who fill no slot of a story.
PARTIES = frozenset({"i", "you", "me", "we", "us"})

#: What a slot is said as, in a key.
SLOT = "it"


def enabled() -> bool:
    return os.environ.get("V689_EXEMPLARS", "1") != "0" and MODEL.exists()


ENABLED = enabled()


@dataclass
class Exemplar:
    text: str
    cell: tuple
    documented: bool = False
    #: the names it was read with: `mary` in `where did mary go`
    names: frozenset = frozenset()
    #: the words as typed, for the parse
    typed: tuple = ()
    #: what is encoded: the text with its slots said as `it`
    key: str = ""


class Encoder:
    """MiniLM-L6-v2, mean-pooled and normalised, as sentence-transformers
    runs it; half precision on CUDA when there is a device, which measures
    the same as full precision on the CPU."""

    def __init__(self, path: Path | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = path or (TUNED if TUNED.exists() else MODEL)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path)).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def __call__(self, texts: list[str]):
        torch = self.torch
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=64, return_tensors="pt").to(
                                   self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=1)


# -- phrases, and keys ------------------------------------------------------
def phrases(words: list[Word]) -> list[tuple[int, int]]:
    """(start, end) of each noun phrase filling a slot, in order: a noun or
    a name that is a subject, an object or a preposition's object, with
    everything under it -- never a question word's, and never I or you."""
    spans = []
    for word in words:
        if word.dep not in ARGUMENTS or not (
                word.tag.startswith("NN") or word.tag == "PRP"):
            continue
        if word.text.lower() in PARTIES | WH:
            continue
        under = {word.index}
        grew = True
        while grew:
            grew = False
            for other in words:
                if other.index not in under and other.head in under \
                        and other.head != other.index:
                    under.add(other.index)
                    grew = True
        if any(words[index].text.lower() in WH for index in under):
            continue
        spans.append((min(under), max(under) + 1))
    spans.sort()
    # A phrase inside another (`Mary` in `Mary's location`) is the outer one.
    return [span for span in spans if not any(
        other != span and other[0] <= span[0] and span[1] <= other[1]
        for other in spans)]


def slots(typed, analyse) -> list[tuple[int, int]] | None:
    """The phrases of words as typed, by the parse; None without one."""
    found = analyse(list(typed)) if analyse is not None and typed else None
    if not found or len(found) != len(typed):
        return None
    return phrases(parse(list(typed), found))


def key(tokens: list[str], spans: list[tuple[int, int]]) -> str:
    """The words with each slot said as `it`: what is asked, not of whom."""
    out, at = [], 0
    for start, end in spans:
        out += tokens[at:start] + [SLOT]
        at = end
    return " ".join(out + tokens[at:])


# -- the exemplars the grammar documents -----------------------------------
def documented() -> list[str]:
    """Every example quoted in the docstrings of the grammar's shapes, and
    the questions its module docstring lists as read into slots."""
    from . import grammar

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
            if one.split() and one.split()[0].lower() in WH]


class Memory:
    """The exemplars and their encodings, built on first use."""

    def __init__(self) -> None:
        self.exemplars: list[Exemplar] = []
        self.vectors = None
        self.pending: list[Exemplar] = []
        self.encoder: Encoder | None = None
        self.analyse = None
        self.built = False
        self.lock = threading.Lock()
        self.texts: set[str] = set()
        self.keys: set[tuple] = set()

    def build(self, lexicon) -> bool:
        if self.built:
            return self.encoder is not None
        self.built = True
        if not ENABLED:
            return False
        try:
            self.encoder = Encoder()
        except Exception:                           # noqa: BLE001
            self.encoder = None
            return False
        self.analyse = getattr(lexicon, "analyse", None)
        from .grammar import propose
        from .reading import tokens_of

        for text in documented():
            lower, typed = tokens_of(text)
            names = frozenset(word.lower() for word in typed
                              if word[:1].isupper())
            goals = propose(lower, lexicon, names, text)
            if goals:
                self._add(lower, typed, goals[0].cell, True, names)
        return True

    def _add(self, tokens, typed, cell: tuple, documented: bool = False,
             names: frozenset = frozenset()) -> None:
        text = " ".join(tokens)
        if text in self.texts or len(self.texts) >= LIMIT:
            return
        self.texts.add(text)
        self.pending.append(Exemplar(text, cell, documented, frozenset(names),
                                     tuple(typed)))

    def learn(self, tokens: list[str], typed: list[str], goals: list,
              names: frozenset = frozenset()) -> None:
        """A question the grammar read a goal from, kept as an exemplar --
        as words until the encoder is first needed, which encodes them."""
        if not ENABLED or not goals or goals[0].like:
            return
        with self.lock:
            self._add(tokens, typed, goals[0].cell, names=names)

    def _flush(self) -> None:
        fresh = []
        for one in self.pending:
            spans = slots(one.typed, self.analyse)
            one.key = key(one.text.split(), spans or [])
            if (one.key, one.cell) not in self.keys:
                self.keys.add((one.key, one.cell))
                fresh.append(one)
        self.pending = []
        if not fresh:
            return
        encoded = self.encoder([one.key for one in fresh])
        self.vectors = (encoded if self.vectors is None
                        else self.encoder.torch.cat([self.vectors, encoded]))
        self.exemplars.extend(fresh)

    def nearest(self, text: str) -> list[tuple[float, Exemplar]]:
        """Exemplars by similarity of their keys to this key, best first."""
        with self.lock:
            self._flush()
            if self.vectors is None:
                return []
            scores = (self.vectors @ self.encoder([text]).T).squeeze(1)
            order = self.encoder.torch.argsort(scores, descending=True)
            return [(float(scores[index]), self.exemplars[int(index)])
                    for index in order[:200].tolist()]


MEMORY = Memory()


# -- reading a question like an exemplar -----------------------------------
def choose(ranked: list[tuple[float, Exemplar]]) -> tuple | None:
    """(cell, similarity, exemplars of that cell) when the nearest is near
    enough and clear of every other cell; else None."""
    if not ranked or ranked[0][0] < SIMILARITY:
        return None
    best, cell = ranked[0][0], ranked[0][1].cell
    rival = next((score for score, one in ranked if one.cell != cell), 0.0)
    if best - rival < MARGIN:
        return None
    return cell, best, [one for score, one in ranked
                        if one.cell == cell][:TRIED]


#: The tagger's question words: `where`, `who`, `which`, `whose`.
WH_TAGS = frozenset({"WRB", "WP", "WP$", "WDT"})

#: What a yes-or-no question opens with, by the tagger: `is`, `can`, `did`.
POLAR_TAGS = frozenset({"MD", "VBZ", "VBP", "VBD"})


def asking(tokens: list[str], analysis, said: str) -> bool:
    """Is this a question an exemplar may read: one the tagger finds a
    question word in (or that ends in a question mark), that does not open
    with its verb -- a yes or no, which has its own reading -- and is not a
    why, which has one too? The words come from the tagger; the reader's
    own list only adds what it does not tag (`whereabouts`)."""
    tags = [tag for tag, _, _ in analysis]
    wh = [at for at, tag in enumerate(tags) if tag in WH_TAGS
          or tokens[at] in WH]
    if not wh and not said.rstrip().endswith("?"):
        return False
    if tags and tags[0] in POLAR_TAGS and 0 not in wh:
        return False
    return not (wh and tokens[wh[0]] == "why")


def _mentions(goal) -> list[str]:
    found = []
    for mention in (goal.subject, goal.object,
                    goal.clause.mention if goal.clause else None,
                    goal.clause.obj if goal.clause else None):
        if mention is not None and mention.text and mention.text not in found:
            found.append(mention.text)
    return found


def _spans(tokens: list[str], texts: list[str]) -> list[tuple[int, int]] | None:
    spans = []
    for text in texts:
        words = text.lower().split()
        at = next((index for index in range(len(tokens) - len(words) + 1)
                   if tokens[index:index + len(words)] == words), None)
        if at is None:
            return None
        spans.append((at, at + len(words)))
    spans.sort()
    if any(one[1] > other[0] for one, other in zip(spans, spans[1:])):
        return None
    return spans


def like(tokens: list[str], typed: list[str], lexicon, names: frozenset,
         said: str):
    """(tokens, typed, goals) read like the nearest exemplar, or None.

    Only a question about this conversation's individuals: one naming at
    least one of them. `why can it swim` has a reading of its own, and `what
    can a dog do` is about a kind, which is v688's."""
    from .grammar import NO_ONE, propose
    from .reading import read_mention, tokens_of

    analyse = getattr(lexicon, "analyse", None)
    found = analyse(list(typed)) if analyse is not None and typed else None
    if not ENABLED or not found or len(found) != len(typed):
        return None
    if not asking(tokens, found, said):
        return None
    theirs = phrases(parse(list(typed), found))
    if not theirs:
        return None

    def individual(span: tuple) -> bool:
        mention = read_mention(tokens[:span[1]], span[0], lexicon,
                               final_ok=True, names=names)
        return mention is not None and mention.form not in NO_ONE

    if not any(individual(span) for span in theirs) \
            or not MEMORY.build(lexicon):
        return None
    chosen = choose(MEMORY.nearest(key(tokens, theirs)))
    if chosen is None:
        return None
    cell, similarity, candidates = chosen
    for exemplar in candidates:
        lower = tokens_of(exemplar.text)[0]
        read = propose(lower, lexicon, names | exemplar.names, exemplar.text)
        goal = next((one for one in read if one.cell == cell), None)
        spans = _spans(lower, _mentions(goal)) if goal else None
        if spans is None or len(spans) != len(theirs):
            continue
        new_tokens, new_typed, at = [], [], 0
        for (start, end), (their_start, their_end) in zip(spans, theirs):
            new_tokens += lower[at:start] + tokens[their_start:their_end]
            new_typed += lower[at:start] + typed[their_start:their_end]
            at = end
        new_tokens += lower[at:]
        new_typed += lower[at:]
        goals = propose(new_tokens, lexicon, names, said)
        if any(one.cell == cell for one in goals):
            for one in goals:
                one.like = f"{exemplar.text} ({similarity:.2f})"
            return new_tokens, new_typed, goals
    return None

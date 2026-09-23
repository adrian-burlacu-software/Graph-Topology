"""One encoder, read by every layer instead of rules.

    v687  parse   what a question asks of a kind: its subject, the relation
                  and the target (`v687/language.py`, `Parser.parse`)
    v688  ask     the question a request puts: `do you know if a dog can
                  swim` asks `can a dog swim` (`v688/rephrase.py`)
    v689  place   who is someone new, what the words say about when, the
                  clause an utterance is placed against, and the rest said
                  back in normal form (`v689/reader.py`, `place`)
    v689  read    what an utterance does, and of whom (`v689/reader.py`)

MiniLM-L6 with a head for each thing read. A **sentence** head reads the
whole utterance (the relation asked; what a request was put as), a **word**
head tags every word (a subject, a time word, how a word is said back), and a
**pointer** head says which word follows which when the words are said back in
another order: `can you tell me where Mary is` is `where is mary`, and nothing
says so but which word each word points at. What spaCy makes of each word --
its tag and its dependency -- is added to the word's embedding, so a word the
corpus never had is still a noun, a verb or an object.

Every head is taught offline by the rules it replaced (`v689/teach_reader.py`
reads a corpus with them, and a record is kept only when the labels build the
rules' reading back exactly). At run time no rule is asked, and without the
model (`llm/reader`) nothing is read.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

LLM = Path(__file__).resolve().parents[1] / "llm"

#: The fine-tuned encoder (another with `V689_READER_MODEL`), and the one it
#: starts from. The reader taught mathematics and design goals as well
#: (`research/v692`, `research/v693`, `regenerate.py`'s `reader-math`) is
#: read with where it exists; the readers before it are left as they were.
PREFERRED = ("reader-design4", "reader-maths2", "reader")
MODEL = Path(os.environ.get("V689_READER_MODEL") or next(
    (LLM / name for name in PREFERRED
     if (LLM / name / "labels.json").exists()), LLM / "reader"))
BASE = LLM / "MiniLM-L6-v2"

#: How wide a pointer's query and key are.
POINTING = 128

#: The longest utterance read, in word pieces: a reply read back runs longer
#: than anything said to it.
LONGEST = 96

#: How a word is said back: as it was, left out, or as another form of
#: itself -- its lemma, its past tense, in lower case, the positive of a
#: negative contraction (`can't` -> can), the plain form of a verb said with
#: `-s`, its gerund, its participle. The first is what a word the encoder
#: could not read (past `LONGEST`) is taken as.
OPS = ("KEEP", "DROP", "LEMMA", "PAST", "LOWER", "POSITIVE", "SINGULAR",
       "GERUND", "PARTICIPLE")


def enabled() -> bool:
    """Is there a model to read with?"""
    return (MODEL / "labels.json").exists()


def features(encoded, rows, labels: dict):
    """Each piece's word's tag and dependency, as indices (0 for none)."""
    import torch

    tags = {tag: at for at, tag in enumerate(labels["tags"])}
    deps = {dep: at for at, dep in enumerate(labels["deps"])}
    shape = encoded["input_ids"].shape
    tag_ids = torch.zeros(shape, dtype=torch.long)
    dep_ids = torch.zeros(shape, dtype=torch.long)
    for row, (row_tags, row_deps) in enumerate(rows):
        for at, word in enumerate(encoded.word_ids(row)):
            if word is not None and word < len(row_tags):
                tag_ids[row, at] = tags.get(row_tags[word], 0)
                dep_ids[row, at] = deps.get(row_deps[word], 0)
    return tag_ids, dep_ids


def Heads(encoder, labels: dict):
    """The encoder with a head for each thing read (`labels["heads"]`: a
    name, its kind -- sentence, word or pointer -- and what it says)."""
    import torch

    heads = labels["heads"]

    def of(kind: str) -> list[str]:
        return [name for name, spec in heads.items() if spec["kind"] == kind]

    class _Heads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            size = encoder.config.hidden_size
            self.encoder = encoder
            self.tags = torch.nn.Embedding(len(labels["tags"]), size)
            self.deps = torch.nn.Embedding(len(labels["deps"]), size)
            # From nothing: the encoder starts reading as it was trained.
            torch.nn.init.zeros_(self.tags.weight)
            torch.nn.init.zeros_(self.deps.weight)
            self.sentence = torch.nn.ModuleDict({
                name: torch.nn.Linear(size, len(heads[name]["labels"]))
                for name in of("sentence")})
            self.word = torch.nn.ModuleDict({
                name: torch.nn.Linear(size, len(heads[name]["labels"]))
                for name in of("word")})
            self.query = torch.nn.ModuleDict({
                name: torch.nn.Linear(size, POINTING)
                for name in of("pointer")})
            self.key = torch.nn.ModuleDict({
                name: torch.nn.Linear(size, POINTING)
                for name in of("pointer")})

        def forward(self, input_ids, attention_mask, tag_ids, dep_ids):
            embedded = (self.encoder.get_input_embeddings()(input_ids)
                        + self.tags(tag_ids) + self.deps(dep_ids))
            hidden = self.encoder(inputs_embeds=embedded,
                                  attention_mask=attention_mask
                                  ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out = {name: layer(pooled)
                   for name, layer in self.sentence.items()}
            out.update({name: layer(hidden)
                        for name, layer in self.word.items()})
            for name in self.query:
                out[name] = (self.query[name](hidden)
                             @ self.key[name](hidden).transpose(1, 2)
                             ) / POINTING ** 0.5
            return out

    return _Heads()


def positions(encoded, row: int) -> tuple[dict[int, int], int]:
    """({word: its first piece}, the piece after the last word): where each
    word is read, and where a pointer says the words end."""
    firsts: dict[int, int] = {}
    end = 1
    for at, word in enumerate(encoded.word_ids(row)):
        if word is None:
            continue
        firsts.setdefault(word, at)
        end = at + 1
    return firsts, end


class Model:
    """The encoder and its heads, as trained."""

    def __init__(self, path: Path = MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.labels = json.loads((path / "labels.json").read_text(
            encoding="utf-8"))
        self.heads = self.labels["heads"]
        self.device = device or os.environ.get("V689_READER_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        encoder = AutoModel.from_pretrained(str(path))
        self.net = Heads(encoder, self.labels)
        self.net.load_state_dict(torch.load(path / "heads.pt",
                                            map_location="cpu"), strict=False)
        self.net.to(self.device)
        if self.device == "cuda":
            self.net.half()
        self.net.eval()
        self.lock = threading.Lock()

    def guess(self, sentences: list[list[str]], analyses=None,
              wanted=None) -> list[dict]:
        """What each head reads each sentence as. A sentence head gives its
        labels best first, with how likely each is; a word head a label per
        word; a pointer head, for the start and each word, a score for each
        word being next (the last column is the end)."""
        torch = self.torch
        analyses = analyses or [None] * len(sentences)
        analyses = [one if one is not None else ([""] * len(words),
                                                 [""] * len(words))
                    for one, words in zip(analyses, sentences)]
        encoded = self.tokenizer(sentences, is_split_into_words=True,
                                 padding=True, truncation=True,
                                 max_length=LONGEST, return_tensors="pt")
        tag_ids, dep_ids = features(encoded, analyses, self.labels)
        with self.lock, torch.no_grad():
            out = self.net(encoded["input_ids"].to(self.device),
                           encoded["attention_mask"].to(self.device),
                           tag_ids.to(self.device), dep_ids.to(self.device))
        names = [name for name in self.heads
                 if wanted is None or name in wanted]
        out = {name: out[name].float().cpu() for name in names}
        found = []
        for row, words in enumerate(sentences):
            firsts, end = positions(encoded, row)
            read: dict = {}
            for name in names:
                spec, scores = self.heads[name], out[name][row]
                labels = spec["labels"] if spec["kind"] != "pointer" else ()
                if spec["kind"] == "sentence":
                    chances = scores.softmax(-1)
                    read[name] = [
                        (labels[at], float(chances[at]))
                        for at in chances.argsort(descending=True).tolist()]
                elif spec["kind"] == "word":
                    read[name] = [
                        labels[int(scores[firsts[at]].argmax())]
                        if at in firsts else labels[0]
                        for at in range(len(words))]
                else:
                    rows = [0] + [firsts.get(at, 0)
                                  for at in range(len(words))]
                    columns = [firsts.get(at, 0)
                               for at in range(len(words))] + [end]
                    read[name] = scores[rows][:, columns].tolist()
            found.append(read)
        return found


class _Loaded:
    def __init__(self) -> None:
        self.model: Model | None = None
        self.lock = threading.Lock()

    def get(self) -> Model:
        with self.lock:
            if self.model is None:
                if not enabled():
                    raise RuntimeError(
                        f"no reader at {MODEL}: nothing can be read without "
                        f"it (python -m research.v689.teach_reader corpus, "
                        f"then train)")
                self.model = Model()
            return self.model


LOADED = _Loaded()

#: What was read lately: v688 asks v687 the same question many times over.
_READ: OrderedDict = OrderedDict()
_READ_LOCK = threading.Lock()
_REMEMBERED = 8192


def read(words: list[str], tags=None, deps=None, heads=None) -> dict:
    """What the heads named in `heads` (all of them by default) read one
    sentence as, with spaCy's tag and dependency of each word if there are
    any (`Model.guess`)."""
    words = [str(one) for one in words]
    tags = list(tags) if tags else [""] * len(words)
    deps = list(deps) if deps else [""] * len(words)
    key = (tuple(words), tuple(tags), tuple(deps),
           tuple(heads) if heads else None)
    with _READ_LOCK:
        if key in _READ:
            _READ.move_to_end(key)
            return _READ[key]
    found = LOADED.get().guess([words], [(tags, deps)], heads)[0]
    with _READ_LOCK:
        _READ[key] = found
        while len(_READ) > _REMEMBERED:
            _READ.popitem(last=False)
    return found


def order(pointer, kept: list[int]) -> list[int]:
    """The words kept, in the order the pointer says them: from the start,
    each time the likeliest word not said yet."""
    said, current, left = [], 0, list(kept)
    while left:
        row = pointer[current]
        best = max(left, key=lambda at: row[at])
        said.append(best)
        left.remove(best)
        current = best + 1
    return said


def pointing(said: list[int], count: int) -> list[list[float]]:
    """A pointer that says exactly `said`: what a teacher's order is built
    back through, as the encoder's would be."""
    rows = [[0.0] * (count + 1) for _ in range(count + 1)]
    current = 0
    for at in said:
        rows[current][at] = 1.0
        current = at + 1
    rows[current][count] = 1.0
    return rows


def article(word: str) -> str:
    return "an" if (word or "")[:1].lower() in "aeiou" else "a"


def rewrite(typed: list[str], ops: list[str], inserts: list[str],
            opening: str, pointer, transform) -> list[str]:
    """The words said back: what opens them, then each word kept, in the
    pointer's order, as `transform(op, word, at)` says it, followed by what
    is said after it. An article said before a word is the one its sound
    takes."""
    kept = [at for at, op in enumerate(ops) if op != "DROP"]
    out: list[str] = []
    added: set[int] = set()

    def insert(phrase: str) -> None:
        for word in (phrase or "").split():
            added.add(len(out))
            out.append(word)

    insert(opening)
    for at in order(pointer, kept):
        said = transform(ops[at], typed[at], at)
        if said:
            out.append(said)
        insert(inserts[at] if at < len(inserts) else "")
    for at in sorted(added):
        if out[at] in ("a", "an") and at + 1 < len(out):
            out[at] = article(out[at + 1])
    return out


def gold(kinds: dict, record: dict, fields: dict) -> dict:
    """What a record's labels say, in the shape `Model.guess` gives: for
    building a teacher's reading back as the encoder's would be built.
    `kinds` is each head's kind, `fields` the record's field for each."""
    found: dict = {}
    count = len(record["words"])
    for name, field in fields.items():
        kind = kinds[name]
        if kind == "sentence":
            found[name] = [(record[field], 1.0)]
        elif kind == "word":
            found[name] = list(record[field])
        else:
            found[name] = pointing(record[field], count)
    return found

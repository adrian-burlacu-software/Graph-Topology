"""Teaching a reader where a request says an example of what the code
gives (2026-10-26, the core measured on Aider's editing benchmark: what is
asked, said by example).

    python -m research.v698.teach_examples fetch    data/mbpp/*.parquet
    python -m research.v698.teach_examples corpus   llm/example-data/*.jsonl
    python -m research.v698.teach_examples train    llm/example-reader

An exercise says what its code gives by example, in words: "Encoding
`test` gives `gvhg`", "brown-green should return 15". Said by the
person, an example is a test truer than any the code model guesses
(CanItEdit's own changes passed 14 of 123 of those) -- where it is read.
A reader of its own is taught it (`llm/example-reader`, MiniLM taught
whole to label words): each word of a sentence the input of an example
(`IN`, one span an argument), what it gives (`OUT`), or neither, read as
typed. (Taught as one head more on the shared reader, alone on its encoder
as it was, it read 0.75 of whole sentences right and missed "Encoding
`test` gives `gvhg`": its words are read for what else they are.)

**Labels by construction**: MBPP's train and validation splits (Austin et
al. 2021, google-research-datasets/mbpp, CC-BY 4.0) give calls and what
they give (`assert first_repeated_char("abcabc") == "a"`); each is said in
a sentence built around its values -- as Exercism and docs write them:
backquoted, quoted, bare, after a bullet, with an arrow -- the values'
words labelled where they were put. Sentences that give no example are the
rest: MBPP's own statements (`Write a function to ...`) and the first
sentences of CodeSearchNet's docstrings, numbers and all (`Returns the
first 10 rows.`), less any that read as an example. Nothing of Exercism or
CanItEdit is taught: they are what is measured.
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mbpp"
OUT = ROOT / "llm" / "example-data"
BASE = ROOT / "llm" / "MiniLM-L6-v2"
READER = ROOT / "llm" / "example-reader"
SEED = 698
#: the longest sentence read, in word pieces
LONGEST = 128
ROLES = ("O", "B-IN", "I-IN", "B-OUT", "I-OUT")
PARQUET = ("https://huggingface.co/api/datasets/google-research-datasets/"
           "mbpp/parquet/full/{split}/0.parquet")
#: sentences that give no example, of each split
NONE = {"train": 4000, "valid": 600}


def fetch() -> None:
    import urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        path = DATA / f"{split}.parquet"
        if path.exists():
            continue
        request = urllib.request.Request(PARQUET.format(split=split),
                                         headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=600) as reply:
            path.write_bytes(reply.read())
        print(f"{path.stat().st_size} bytes -> {path}")


def calls(split: str) -> list:
    """(the statement, the function's name, its arguments, what it gives)
    of each of a split's asserts that are a call of values."""
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / f"{split}.parquet").to_pylist()
    out = []
    for row in rows:
        for test in row["test_list"]:
            try:
                node = ast.parse(test).body[0]
            except SyntaxError:
                continue
            if not isinstance(node, ast.Assert) or not isinstance(
                    node.test, ast.Compare) or not isinstance(
                    node.test.ops[0], ast.Eq):
                continue
            call, wanted = node.test.left, node.test.comparators[0]
            if not isinstance(call, ast.Call) or not isinstance(
                    call.func, ast.Name) or call.keywords:
                continue
            try:
                args = [ast.literal_eval(one) for one in call.args]
                value = ast.literal_eval(wanted)
            except ValueError:
                continue
            if not args or len(args) > 3:
                continue
            out.append((row["text"], call.func.id, args, value))
    return out


def _said(value, rng) -> str:
    """A value as a request writes it: as Python does, in backquotes, or --
    a word or a number -- bare."""
    shown = repr(value)
    if isinstance(value, str) and rng.random() < 0.3:
        shown = '"' + value + '"'
    bare = isinstance(value, (int, float)) and not isinstance(value, bool) \
        or isinstance(value, str) and re.fullmatch(r"[\w-]+", value)
    style = rng.random()
    word = value if isinstance(value, str) else shown
    if bare and style < 0.3:
        return word
    if bare and style < 0.55:
        # in backquotes as it is (`test`, `15`): as Exercism writes it
        return f"`{word}`"
    if style < 0.75:
        return f"`{shown}`"
    return shown


def _named(name: str) -> str:
    """A function's name as words: `first_repeated_char` -> first
    repeated char."""
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
                    .replace("_", " ").lower().split())


#: how a sentence says one example: {in} its arguments, {out} what it
#: gives, {name} the function's name in words
ONE = ("For example, {in} should return {out}.",
       "{in} gives {out}.",
       "{in} -> {out}",
       "- {in} -> {out}",
       "{in} => {out}",
       "Given {in}, the result is {out}.",
       "Given {in}, return {out}.",
       "The {name} of {in} is {out}.",
       "{name} {in} gives {out}",
       "For {in}, the answer is {out}.",
       "If the input is {in}, the output should be {out}.",
       "With {in} the function returns {out}.",
       "Calling it with {in} gives {out}.",
       "{in} becomes {out}.",
       "{in} results in {out}.",
       "e.g. {in} is {out}",
       "So {in} should give {out}, as expected.",
       "An input of {in} produces {out}.",
       "Input: {in} Output: {out}",
       "{in} should be {out}.")
#: and the same, led by what the function does (`Encoding ...`), {lead}
LED = ("{lead} {in} gives {out}.",
       "{lead}: {in} -> {out}",
       "{lead} {in} returns {out}.",
       "{lead} of {in} is {out}.",
       "{lead} {in} should give {out}.")
#: and joining several arguments
JOINED = (", ", " and ", " with ")


def _gerund(verb: str) -> str:
    """`find` -> finding, `remove` -> removing, `get` -> getting."""
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if len(verb) <= 4 and re.fullmatch(r".*[^aeiou][aeiou][^aeiouwxy]",
                                       verb):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _lead(text: str) -> str | None:
    """What a statement asks the function to do, as a gerund: `Write a
    function to find the first repeated character in a given string` ->
    Finding the first repeated character."""
    found = re.search(r"\bto (\w+) (.+?)(?: in a given| from a given| of a "
                      r"given| for a given|[.,]|$)", text)
    if not found:
        return None
    return f"{_gerund(found.group(1).lower()).capitalize()} {found.group(2)}"


def example(call: tuple, rng) -> dict:
    """One example said, its words labelled by where each value was put."""
    text, name, args, value = call
    pieces = [_said(one, rng) for one in args]
    joint = rng.choice(JOINED)
    lead = _lead(text)
    carrier = rng.choice(LED) if lead and rng.random() < 0.35 else \
        rng.choice(ONE)
    if len(args) > 1 and carrier in ("The {name} of {in} is {out}.",
                                     "{lead} of {in} is {out}."):
        carrier = "{in} gives {out}."
    # the sentence as a list of (text, role) parts, then words
    parts = []
    for chunk in re.split(r"(\{in\}|\{out\}|\{name\}|\{lead\})",
                          carrier):
        if chunk == "{in}":
            for at, piece in enumerate(pieces):
                if at:
                    parts.append((joint, "O"))
                parts.append((piece, "IN"))
        elif chunk == "{out}":
            parts.append((_said(value, rng), "OUT"))
        elif chunk == "{name}":
            parts.append((_named(name), "O"))
        elif chunk == "{lead}":
            parts.append((lead, "O"))
        elif chunk:
            parts.append((chunk, "O"))
    sentence = "".join(one for one, _ in parts)
    if "{lead}" not in carrier and rng.random() < 0.4:
        # the statement first, as a request says what it wants, then the
        # example
        lead = text.strip().rstrip(".") + ". "
        parts.insert(0, (lead, "O"))
        sentence = lead + sentence
    return _labelled(parts, sentence)


def _labelled(parts: list, sentence: str) -> dict | None:
    from research.v692.corpus import words
    said, roles = [], []
    for text, role in parts:
        found = words(text)
        if not found:
            continue
        said += found
        roles += ["O"] * len(found) if role == "O" else \
            [f"B-{role}"] + [f"I-{role}"] * (len(found) - 1)
    if said != words(sentence):
        # a value joined to the words around it (`(1,` and `2)`): the
        # labels would not say where it is
        return None
    return {"task": "code", "words": said, "example": roles,
            "tags": [""] * len(said), "deps": [""] * len(said),
            "said": sentence, "source": "mbpp example"}


#: a sentence that reads as an example, left out of those that give none
SHOWS = re.compile(r"->|=>|>>>|\be\.g\b|\bexample|\bshould (return|give|be)"
                   r"|\breturns? `|==", re.I)


def none(split: str, count: int, rng) -> list:
    """Sentences that give no example: MBPP's statements and CodeSearchNet's
    docstrings' first sentences."""
    import pyarrow.parquet as pq
    texts = [row["text"] for row in pq.read_table(
        DATA / f"{split}.parquet").to_pylist()]
    language = ROOT / "data" / "codesearchnet" / (
        "python-train.parquet" if split == "train"
        else "python-validation.parquet")
    docs = pq.read_table(language, columns=[
        "func_documentation_string"]).to_pylist()
    rng.shuffle(docs)
    for one in docs:
        first = re.split(r"(?<=\.)\s|\n\s*\n", one[
            "func_documentation_string"].strip())[0]
        first = " ".join(first.split())
        if 4 <= len(first.split()) <= 40:
            texts.append(first)
        if len(texts) >= count * 2:
            break
    out = []
    from research.v692.corpus import words
    for text in texts:
        if SHOWS.search(text):
            continue
        said = words(text)
        if said:
            out.append({"task": "code", "words": said,
                        "example": ["O"] * len(said), "tags": [""] * len(said),
                        "deps": [""] * len(said), "said": text,
                        "source": "no example"})
    rng.shuffle(out)
    return out[:count]


def corpus() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, source in (("train", "train"), ("valid", "validation")):
        rows = []
        for call in calls(source):
            for _ in range(3):
                one = example(call, rng)
                if one is not None:
                    rows.append(one)
        given = len(rows)
        rows += none(source, NONE[split], rng)
        rng.shuffle(rows)
        path = OUT / f"{split}-examples.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(f"{split}: {given} examples, {len(rows) - given} none -> {path}")


def _typed(row: dict) -> list:
    """A record's words as typed (case kept): the reader reads them so."""
    from research.v698.reading import _typed
    return _typed(row["said"], row["words"])


def train(epochs: int = 4, batch: int = 32, rate: float = 5e-5) -> None:
    import torch
    from transformers import (AutoModelForTokenClassification,
                              AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in
            (OUT / "train-examples.jsonl").open(encoding="utf-8")]
    held = [json.loads(line) for line in
            (OUT / "valid-examples.jsonl").open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    net = AutoModelForTokenClassification.from_pretrained(
        str(BASE), num_labels=len(ROLES)).to("cuda")
    index = {role: at for at, role in enumerate(ROLES)}

    def encoded(chunk):
        found = tokenizer([_typed(one) for one in chunk],
                          is_split_into_words=True, truncation=True,
                          max_length=LONGEST, padding=True,
                          return_tensors="pt")
        labels = []
        for row, one in enumerate(chunk):
            seen, out = set(), []
            for word in found.word_ids(row):
                # each word labelled at its first piece
                out.append(-100 if word is None or word in seen
                           else index[one["example"][word]])
                seen.add(word)
            labels.append(out)
        return found.to("cuda"), torch.tensor(labels, device="cuda")

    def whole() -> float:
        net.eval()
        right = 0
        with torch.no_grad():
            for at in range(0, len(held), 64):
                chunk = held[at:at + 64]
                inputs, labels = encoded(chunk)
                guess = net(**inputs).logits.argmax(-1)
                kept = labels != -100
                right += sum(bool(((guess[row] == labels[row]) |
                                   ~kept[row]).all())
                             for row in range(len(chunk)))
        net.train()
        return right / len(held)

    optimiser = torch.optim.AdamW(net.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    schedule = get_linear_schedule_with_warmup(optimiser, steps // 16, steps)
    net.train()
    for epoch in range(epochs):
        rng.shuffle(rows)
        total = 0.0
        for at in range(0, len(rows), batch):
            inputs, labels = encoded(rows[at:at + batch])
            loss = net(**inputs, labels=labels).loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}: loss {total:.1f}, whole sentences right "
              f"{whole():.4f}", flush=True)
    READER.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(str(READER))
    tokenizer.save_pretrained(str(READER))
    (READER / "taught.json").write_text(json.dumps({
        "base": str(BASE), "roles": list(ROLES), "longest": LONGEST,
        "train": len(rows), "valid": len(held), "whole": whole()},
        indent=1), encoding="utf-8")
    print(f"saved to {READER}")


def main(argv=None) -> int:
    job = (argv or sys.argv[1:] or ["corpus"])[0]
    {"fetch": fetch, "corpus": corpus, "train": train}[job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Teaching a judge of changes: does this change do what was said?

    python -m research.v700.teach_judge train      llm/change-judge
    python -m research.v700.teach_judge measure    on the held commits

The editor (`teach_editor.py`) writes several changes for a statement;
the compiler says which still compile -- and adding `import math` compiles
as well as taking it out. Which of them does *what was said* is a reading
of the statement against the change, and here an encoder is taught it
(UniXcoder, `llm/unixcoder-base`, as v696's readers of meaning): the
statement and the change as a diff in, the chance it is the change said
out.

**What it is taught**, from the editor's commits (`edits.jsonl`): each
commit's message with its own change (yes), and with
- the change undone (`-` and `+` swapped): what was said, backwards --
  `remove the unused import` against an import put in;
- another commit's change, of the same language: a change of something
  else;
- a line of the part said twice, and an empty line put in: changes of
  nothing that was asked -- what the editor writes when it has nothing.

Held: the editor's `test` commits, never taught; measured as how often the
commit's own change is judged above each of its others.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import sys
import time
from pathlib import Path

from research.v700 import teach_editor as T

LLM = T.LLM
BASE = LLM / "unixcoder-base"
OUT = LLM / "change-judge"
#: the judge the architecture uses: taught also on the editor's own wrong
#: changes (`samples`), and on changes with more or less than was said
USED = LLM / "change-judge2"
SEED = 700
LONGEST = 384


def diff_of(part: str, made: str) -> str:
    """A change as people read one: the lines taken out and put in, with
    two around them."""
    return "".join(difflib.unified_diff(
        part.splitlines(True), made.splitlines(True), n=2))[0:4000]


def said(statement: str, change: str) -> tuple:
    return " ".join(statement.split()), change


def _others(row: dict, part: str, made: str, rows: list,
            rng: random.Random) -> list:
    out = [("reversed", diff_of(made, part))]
    other = rng.choice([one for one in rows
                        if one["language"] == row["language"]])
    other_made = T.applied(other["part"], other["target"])
    if other_made is not None and other is not row:
        out.append(("another", diff_of(other["part"], other_made)))
    lines = part.splitlines(True)
    if len(lines) > 2:
        at = rng.randrange(1, len(lines))
        out.append(("twice", diff_of(part, "".join(
            lines[:at] + [lines[at - 1]] + lines[at:]))))
        at = rng.randrange(1, len(lines))
        out.append(("blank", diff_of(part, "".join(
            lines[:at] + ["\n"] + lines[at:]))))
    return [(kind, one) for kind, one in out if one.strip()]


#: the editor's own answers to commits it was taught (`samples`): what it
#: writes when it is wrong is what the judge must tell from what was said
SAMPLES = T.OUT / "editor-samples.jsonl"


def _more(row: dict, part: str, made: str, rng: random.Random) -> list:
    """The change said, and something nobody asked for beside it (a line
    said twice elsewhere); and, of a change in several places, one of
    them alone -- not all that was said."""
    out = []
    lines = made.splitlines(True)
    changed = T._lines(made) != T._lines(part)
    if changed and len(lines) > 3:
        at = rng.randrange(1, len(lines))
        out.append(("extra", diff_of(part, "".join(
            lines[:at] + [lines[at - 1]] + lines[at:]))))
    blocks = T.blocks_said(row["target"])
    if len(blocks) > 1:
        one = T.applied(part, T.said([(old.splitlines(True),
                                       new.splitlines(True))
                                      for old, new in blocks[:1]]))
        if one is not None and one != made:
            out.append(("partial", diff_of(part, one)))
    return out


def _editor_wrong(rng: random.Random) -> dict:
    """commit -> the changes the editor made of it that are not the
    commit's (each of an answer's blocks alone too)."""
    out: dict = {}
    if not SAMPLES.exists():
        return out
    for line in SAMPLES.open(encoding="utf-8"):
        row = json.loads(line)
        want = T.applied(row["part"], row["target"])
        wrong = []
        for answer in row["answers"]:
            for made in T.made_each(row["part"], answer):
                if made != want and made != row["part"] and \
                        made not in wrong:
                    wrong.append(made)
        out[row["commit"]] = [diff_of(row["part"], one) for one in wrong]
    return out


def pairs(split: str, seed: int = SEED) -> list:
    """(statement, change, label, kind) for every commit of a split."""
    rng = random.Random(seed)
    rows = [json.loads(line) for line in T.EDITS.open(encoding="utf-8")]
    rows = [one for one in rows if one["split"] == split]
    wrong = _editor_wrong(random.Random(seed + 1))
    out = []
    for row in rows:
        made = T.applied(row["part"], row["target"])
        if made is None:
            continue
        out.append((row["statement"], diff_of(row["part"], made), 1.0,
                    "commit"))
        for kind, change in _others(row, row["part"], made, rows, rng) + \
                _more(row, row["part"], made, rng):
            out.append((row["statement"], change, 0.0, kind))
        for change in wrong.get(row["commit"], ())[:4]:
            out.append((row["statement"], change, 0.0, "editor"))
    return out


def samples(editor: Path, most: int = 1500, each: int = 4) -> None:
    """The editor's answers to commits it was taught, kept as it goes."""
    from research.v696.sketcher import Sketcher
    done = set()
    if SAMPLES.exists():
        done = {json.loads(line)["commit"]
                for line in SAMPLES.open(encoding="utf-8")}
    rows = [json.loads(line) for line in T.EDITS.open(encoding="utf-8")]
    rows = [one for one in rows if one["split"] == "train"]
    random.Random(SEED).shuffle(rows)
    rows = [one for one in rows[:most] if one["commit"] not in done]
    writer = Sketcher(editor)
    started = time.time()
    for at in range(0, len(rows), 4):
        chunk = rows[at:at + 4]
        written = writer.write([T._text(one) for one in chunk],
                               samples=each, longest=400)
        with SAMPLES.open("a", encoding="utf-8") as out:
            for row, answers in zip(chunk, written):
                out.write(json.dumps({
                    "commit": row["commit"], "part": row["part"],
                    "target": row["target"], "answers": answers}) + "\n")
        if at % 200 == 0:
            print(f"  {at + len(chunk)}/{len(rows)} "
                  f"({time.time() - started:.0f}s)", flush=True)


class Judge:
    """The judge, loaded: `chances(statement, [changes])`."""

    def __init__(self, path: Path = OUT) -> None:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path)).to("cuda").eval()

    def chances(self, statement: str, changes: list) -> list:
        torch = self.torch
        out = []
        for at in range(0, len(changes), 16):
            chunk = changes[at:at + 16]
            encoded = self.tokenizer(
                [" ".join(statement.split())] * len(chunk), chunk,
                truncation="only_second", max_length=LONGEST,
                padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda",
                                                 dtype=torch.bfloat16):
                logits = self.model(**encoded).logits[:, 0]
            out += torch.sigmoid(logits.float()).tolist()
        return out


_LOADED: dict = {}


def judge() -> Judge | None:
    if "one" not in _LOADED:
        _LOADED["one"] = Judge(USED) if (USED / "config.json").exists()             else None
    return _LOADED["one"]


def train(out: Path = OUT, epochs: int = 2, batch: int = 32,
          rate: float = 3e-5) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows = pairs("train")
    held = pairs("dev")[:3000]
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE), num_labels=1).to("cuda")

    def encoded(chunk):
        found = tokenizer([" ".join(one[0].split()) for one in chunk],
                          [one[1] for one in chunk], truncation="only_second",
                          max_length=LONGEST, padding=True,
                          return_tensors="pt").to("cuda")
        labels = torch.tensor([one[2] for one in chunk], device="cuda")
        return found, labels

    loss_of = torch.nn.BCEWithLogitsLoss()

    def evaluate() -> tuple:
        model.eval()
        right = total = 0
        with torch.no_grad():
            for at in range(0, len(held), 64):
                found, labels = encoded(held[at:at + 64])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**found).logits[:, 0].float()
                right += int(((logits > 0).float() == labels).sum())
                total += len(labels)
        model.train()
        return right / max(total, 1)

    optimiser = torch.optim.AdamW(model.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * math.ceil(len(rows) / batch)
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.06 * steps),
                                               steps)
    print(f"{len(rows)} pairs, {len(held)} held, {steps} steps; held "
          f"accuracy before {evaluate():.3f}", flush=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(rows)
        total = 0.0
        for step, at in enumerate(range(0, len(rows), batch), 1):
            found, labels = encoded(rows[at:at + batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**found).logits[:, 0].float()
            loss = loss_of(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            if step % 1000 == 0:
                print(f"  step {step}: loss {total / step:.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
        print(f"epoch {epoch + 1}: held accuracy {evaluate():.3f} "
              f"({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    (out / "judge.json").write_text(json.dumps({
        "base": BASE.name, "pairs": len(rows), "epochs": epochs,
        "longest": LONGEST}, indent=1), encoding="utf-8")
    print(f"-> {out}")


def measure(path: Path = OUT) -> dict:
    """On commits never taught: how often each kind of other change is
    judged below the commit's own, and the chance each gets."""
    found = Judge(path)
    rows = pairs("test")
    by_commit: dict = {}
    for statement, change, label, kind in rows:
        by_commit.setdefault(statement, []).append((change, label, kind))
    beaten, counted, chance = {}, {}, {}
    for statement, items in by_commit.items():
        chances = found.chances(statement, [one[0] for one in items])
        own = [p for p, one in zip(chances, items) if one[1] == 1.0]
        if not own:
            continue
        for p, (_, label, kind) in zip(chances, items):
            chance.setdefault(kind, []).append(p)
            if label == 1.0:
                continue
            counted[kind] = counted.get(kind, 0) + 1
            beaten[kind] = beaten.get(kind, 0) + (own[0] > p)
    got = {"below the commit's own": {kind: round(beaten[kind] /
                                                  counted[kind], 3)
                                      for kind in counted},
           "mean chance": {kind: round(sum(one) / len(one), 3)
                           for kind, one in chance.items()}}
    print(json.dumps(got))
    return got


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("samples", "train", "measure"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--editor", default="editor")
    parser.add_argument("--out", default=OUT.name)
    options = parser.parse_args(argv)
    if options.job == "samples":
        samples(LLM / options.editor)
    elif options.job == "train":
        train(LLM / options.out, epochs=options.epochs)
    else:
        measure(LLM / options.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

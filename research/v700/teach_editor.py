"""Teaching a writer to *change* code as it is asked: the editor.

    python -m research.v700.teach_editor fetch      data/commitpackft
    python -m research.v700.teach_editor corpus     llm/editor-data/edits.jsonl
    python -m research.v700.teach_editor train      llm/editor
    python -m research.v700.teach_editor measure    on the held commits

The writers of v696 were taught to write a function from a request; asked
to change one (`_turn reads the number from "n", the server sends
"number"`), each wrote it back as it was. What people say a change is, and
the change, is what a commit is: its message, and the file before and
after. CommitPackFT (`data/commitpackft`, bigcode: commits of permissively
licensed repositories whose message says what the change does) is taught
here, Python and TypeScript.

**What it is shown**: the message, the file's path, and the part of the
file the change is in -- in Python the function or method holding every
changed line (what the conversation names: `fix _turn ...`), in TypeScript
the lines around them. **What it answers**: each change as

    <<<
    the lines as they are
    ===
    the lines as they become
    >>>

-- the old lines found in what it was shown exactly once, so an answer is
put into the file where it says, or refused (`edits.applied`); a writer
asked to say the whole function again says it with changes nobody asked
for.

Held: 3% of commits (by hash) for the loss while teaching (`dev`), 3% never
taught (`test`), measured by `measure`: whether the code it makes is the
commit's.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "commitpackft"
LLM = ROOT / "llm"
OUT = LLM / "editor-data"
EDITS = OUT / "edits.jsonl"
BASE = LLM / "SmolLM2-360M-Instruct"
SEED = 700
SAYING = ("You change code as you are asked. Answer with each change as "
          "<<< the lines as they are === the lines as they become >>>, "
          "nothing else.")

#: what the editor answers where its part of a change is nothing (v703
#: `teach_parts`: a function a commit across files did not change)
NOTHING = "nothing to change here"
#: what is taught: a part of a file at most this long, changed this much
LONGEST_PART, LONGEST_CHANGE = 60, 24
LONGEST_MESSAGE = 300
#: lines around a TypeScript change, where no function is read
AROUND = 12


SOURCE = "https://huggingface.co/datasets/bigcode/commitpackft/resolve/main/" \
    "data/{}/data.jsonl"


def fetch() -> None:
    """CommitPackFT's Python, TypeScript, JSON, YAML and CSV commits, as
    published."""
    import urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    # and its data files' commits (v701): real JSON, YAML and CSV files,
    # what questions about data are made from
    for name in ("python", "typescript", "json", "yaml", "csv"):
        path = DATA / f"{name}.jsonl"
        if not path.exists():
            urllib.request.urlretrieve(SOURCE.format(name), path)
        print(f"{path}: {sum(1 for _ in path.open(encoding='utf-8'))} commits")


def split_of(commit: str) -> str:
    at = hashlib.sha1(commit.encode()).digest()[0]
    return "test" if at < 8 else "dev" if at < 16 else "train"


# -- the part of a file a change is in -------------------------------------------

def python_part(source: str, lines: list):
    """(start, end) 0-based, end exclusive: the smallest function or method
    holding every line in `lines`, decorators included; None where no one
    does or it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = min([node.lineno] + [one.lineno for one in
                                     node.decorator_list]) - 1
        end = node.end_lineno
        if all(start <= one < end for one in lines) and (
                best is None or end - start < best[1] - best[0]):
            best = (start, end)
    return best


def window_part(count: int, lines: list):
    return max(0, min(lines) - AROUND), min(count, max(lines) + 1 + AROUND)


# -- a change, said as blocks ----------------------------------------------------------

def _unique(part: list, start: int, end: int) -> bool:
    """Whether part[start:end] is found in `part` once only."""
    want = part[start:end]
    size = len(want)
    return sum(part[at:at + size] == want
               for at in range(len(part) - size + 1)) == 1


def blocks_of(old: list, new: list, offset: int, end: int):
    """The changes from `old` to `new` inside old's lines [offset, end), as
    (old lines, new lines) each found once in that part -- widened by the
    lines around it until it is -- or None where a change is outside it."""
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    groups = [one for one in matcher.get_opcodes() if one[0] != "equal"]
    if not groups:
        return None
    part = old[offset:end]
    out = []
    for tag, i1, i2, j1, j2 in groups:
        if i1 < offset or i2 > end:
            return None
        a, b = i1 - offset, i2 - offset
        before, after = 0, 0
        # an insertion is said with a line it goes beside
        if a == b:
            if a > 0:
                before = 1
            elif b < len(part):
                after = 1
            else:
                return None
        while not _unique(part, a - before, b + after) and (
                a - before > 0 or b + after < len(part)):
            if a - before > 0:
                before += 1
            if not _unique(part, a - before, b + after) and \
                    b + after < len(part):
                after += 1
        if not _unique(part, a - before, b + after):
            return None
        said_old = part[a - before:b + after]
        said_new = part[a - before:a] + new[j1:j2] + part[b:b + after]
        out.append((said_old, said_new))
    return out


def said(blocks) -> str:
    return "\n".join(f"<<<\n{''.join(old)}===\n{''.join(new)}>>>"
                     for old, new in blocks)


def prompt(statement: str, path: str, part: str) -> str:
    return f"{statement.strip()}\n\n{path}:\n{part.rstrip()}\n"


# -- the corpus ---------------------------------------------------------------------------

def _lines(text: str) -> list:
    out = text.splitlines(keepends=True)
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    return out


def row_of(record: dict) -> dict | None:
    old, new = record.get("old_contents") or "", record.get("new_contents")
    message = (record.get("message") or record.get("subject") or "").strip()
    if not old or new is None or old == new or len(old) > 200_000 or \
            not message or len(message) > LONGEST_MESSAGE or \
            len(message.split()) < 2 or record["old_file"] != \
            record["new_file"]:
        return None
    path = record["old_file"]
    python = path.endswith(".py")
    # and a data file's commits (v701): changed as text, the lines around
    data = bool(re.search(r"\.(json|ya?ml|csv|tsv)$", path, re.I))
    if not python and not data and not re.search(r"\.(ts|tsx)$", path):
        return None
    a, b = _lines(old), _lines(new)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = []
    size = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        size += (i2 - i1) + (j2 - j1)
        changed += list(range(i1, i2)) or [max(i1 - 1, 0)]
    if not changed or size > LONGEST_CHANGE:
        return None
    part = python_part(old, changed) if python else None
    if part is None or part[1] - part[0] > LONGEST_PART:
        # no one function holds it (a change to the module, or to two):
        # the lines around it, as TypeScript's
        part = window_part(len(a), changed)
        if part[1] - part[0] > LONGEST_PART:
            return None
    blocks = blocks_of(a, b, *part)
    if not blocks:
        return None
    shown = "".join(a[part[0]:part[1]])
    if applied(shown, said(blocks)) is None:
        return None
    return {"commit": record["commit"], "split": split_of(record["commit"]),
            "language": "python" if python else (
                path.rsplit(".", 1)[-1].lower() if data else "typescript"),
            "statement": message, "path": path, "part": shown,
            "start": part[0] + 1, "target": said(blocks)}


def corpus() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    seen, counts = set(), {}
    with EDITS.open("w", encoding="utf-8") as out:
        for name in ("python", "typescript"):
            for line in (DATA / f"{name}.jsonl").open(encoding="utf-8"):
                row = row_of(json.loads(line))
                if row is None:
                    continue
                key = (row["statement"], row["part"])
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(row) + "\n")
                counted = counts.setdefault(row["language"], {})
                counted[row["split"]] = counted.get(row["split"], 0) + 1
    print(json.dumps(counts))
    return counts


# -- the mix: what it is asked, more of it ------------------------------------

MIX = OUT / "edits-mix.jsonl"
#: commits taught again beside the faults, so what was taught is kept
KEPT = 12000


def _small_in_function(row: dict) -> bool:
    """A change of a few lines inside one Python function: what it is
    asked most (`fix _turn: ...`)."""
    if row["language"] != "python" or not re.match(
            r"\s*(@|(async\s+)?def\s)", row["part"]):
        return False
    changed = sum(len(old.splitlines()) + len(new.splitlines())
                  for old, new in blocks_said(row["target"]))
    return changed <= 4


def mix(seed: int = SEED) -> dict:
    """The faults (`teach_faults.py`), commits of a small change inside a
    function three times, and `KEPT` others: what the second editor is
    taught, from the first."""
    from research.v700.teach_faults import FAULTS
    rng = random.Random(seed)
    commits = [json.loads(line) for line in EDITS.open(encoding="utf-8")]
    faults = [json.loads(line) for line in FAULTS.open(encoding="utf-8")]
    small = [one for one in commits if _small_in_function(one)
             and one["split"] == "train"]
    others = [one for one in commits if not _small_in_function(one)
              and one["split"] == "train"]
    out = [one for one in commits if one["split"] != "train"]
    out += small * 3 + rng.sample(others, min(KEPT, len(others)))
    repeat = {"fault import": 1, "fault variable": 3, "fault variables": 5}
    for one in faults:
        out += [one] * (repeat.get(one["source"], 1)
                        if one["split"] == "train" else 1)
    rng.shuffle(out)
    with MIX.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    counts = {"small in a function": len(small), "others": min(KEPT,
              len(others)), "faults": len(faults),
              "train rows": sum(one["split"] == "train" for one in out)}
    print(json.dumps(counts))
    return counts


# -- an answer, put in -------------------------------------------------------------------

BLOCK = re.compile(r"<<<\n(.*?)===\n(.*?)>>>", re.S)


def blocks_said(answer: str) -> list:
    return [(old, new) for old, new in BLOCK.findall(answer)]


def made_each(part: str, answer: str) -> list:
    """What an answer could make of `part`: all its changes, and each of
    them alone -- an answer is often one change asked and others nobody
    asked for, and the one is what it is worth."""
    found = blocks_said(answer)
    out = []
    for chosen in ([found] if len(found) == 1 else
                   [found] + [[one] for one in found]):
        made = applied(part, said([(old.splitlines(True),
                                    new.splitlines(True))
                                   for old, new in chosen]))
        if made is not None and made not in out:
            out.append(made)
    return out


def applied(part: str, answer: str) -> str | None:
    """`part` with the answer's changes made, or None where the answer says
    none, or one whose old lines are not found in it once exactly."""
    found = blocks_said(answer)
    if not found:
        return None
    out = part if part.endswith("\n") else part + "\n"
    for old, new in found:
        if not old or out.count(old) != 1:
            return None
        out = out.replace(old, new, 1)
    return out


# -- teaching -----------------------------------------------------------------------------

def _encoded(tokenizer, text: str, target: str | None = None):
    from research.v696.sketcher import _encoded as encoded
    return encoded(tokenizer, text, target, saying=SAYING)


def _text(row: dict) -> str:
    return prompt(row["statement"], row["path"], row["part"])


def train(out: Path, epochs: int = 1, batch: int = 8, rate: float = 1e-4,
          longest: int = 1024, most: int | None = None,
          seed: int = SEED, base: Path = BASE,
          corpus: Path = EDITS) -> None:
    """Taught from `base`: SmolLM2 as it came, or an editor taught before,
    taught again (a second pass, its own seed for the order)."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = [json.loads(line) for line in corpus.open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # too long to read whole is left out, not cut
    rows = [one for one in rows if len(_encoded(
        tokenizer, _text(one), one["target"])[0]) <= longest]
    train_rows = [one for one in rows if one["split"] == "train"]
    rng.shuffle(train_rows)
    if most:
        train_rows = train_rows[:most]
    held = [one for one in rows if one["split"] == "dev"][:600]
    model = AutoModelForCausalLM.from_pretrained(str(base),
                                                 dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.get_input_embeddings().weight.requires_grad_(False)
    model.to("cuda")

    def batches(chunk):
        ids, labels = [], []
        for one in chunk:
            tokens, start = _encoded(tokenizer, _text(one), one["target"])
            ids.append(tokens)
            labels.append([-100] * min(start, len(tokens)) + tokens[start:])
        width = max(map(len, ids))
        pad = tokenizer.pad_token_id
        return (torch.tensor([one + [pad] * (width - len(one))
                              for one in ids], device="cuda"),
                torch.tensor([[1] * len(one) + [0] * (width - len(one))
                              for one in ids], device="cuda"),
                torch.tensor([one + [-100] * (width - len(one))
                              for one in labels], device="cuda"))

    def evaluate() -> float:
        model.eval()
        total = count = 0
        with torch.no_grad():
            for at in range(0, len(held), batch):
                ids, mask, labels = batches(held[at:at + batch])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    total += float(model(input_ids=ids, attention_mask=mask,
                                         labels=labels).loss) * len(ids)
                count += len(ids)
        model.train()
        return total / max(count, 1)

    trained = [one for one in model.parameters() if one.requires_grad]
    import bitsandbytes
    optimiser = bitsandbytes.optim.AdamW8bit(trained, lr=rate,
                                             weight_decay=0.01)
    steps = epochs * math.ceil(len(train_rows) / batch)
    schedule = get_cosine_schedule_with_warmup(optimiser, int(0.03 * steps),
                                               steps)
    print(f"{len(train_rows)} train, {len(held)} dev, {steps} steps; dev "
          f"loss before {evaluate():.3f}", flush=True)
    started = time.time()
    model.train()
    # by length, in buckets: a batch of like lengths pads little
    for epoch in range(epochs):
        rng.shuffle(train_rows)
        order = []
        for at in range(0, len(train_rows), batch * 50):
            bucket = sorted(train_rows[at:at + batch * 50],
                            key=lambda one: len(_text(one)) +
                            len(one["target"]))
            chunks = [bucket[k:k + batch]
                      for k in range(0, len(bucket), batch)]
            rng.shuffle(chunks)
            order += chunks
        total = 0.0
        for step, chunk in enumerate(order, 1):
            ids, mask, labels = batches(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=labels).loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            if step % 500 == 0:
                print(f"  step {step}/{len(order)}: train "
                      f"{total / step:.3f} ({time.time() - started:.0f}s)",
                      flush=True)
        print(f"epoch {epoch + 1}: train {total / len(order):.3f} dev "
              f"{evaluate():.3f} ({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    (out / "sketcher.json").write_text(json.dumps({
        "base": base.name, "saying": SAYING, "meaning": False,
        "functions": True, "edits": True, "corpus": corpus.name,
        "train": len(train_rows), "epochs": epochs}, indent=1),
        encoding="utf-8")
    print(f"-> {out}")


# -- measuring -----------------------------------------------------------------------------

def _same(a: str, b: str) -> bool:
    return [one.rstrip() for one in a.strip().splitlines()] == \
        [one.rstrip() for one in b.strip().splitlines()]


def measure(model: Path, most: int = 300, samples: int = 4,
            corpus: Path = EDITS) -> dict:
    """On commits (or faults) never taught: whether the greedy answer makes
    the commit's code, whether any of the samples does, and how often an
    answer could be put in at all."""
    from research.v696.sketcher import Sketcher
    rows = [json.loads(line) for line in corpus.open(encoding="utf-8")]
    rows = [one for one in rows if one["split"] == "test"]
    random.Random(SEED).shuffle(rows)
    rows = rows[:most]
    editor = Sketcher(model)
    got = {"rows": len(rows), "greedy": 0, "any": 0, "put in": 0}
    for at in range(0, len(rows), 4):
        chunk = rows[at:at + 4]
        written = editor.write([_text(one) for one in chunk],
                               samples=samples, longest=400)
        for row, answers in zip(chunk, written):
            want = applied(row["part"], row["target"])
            made = [applied(row["part"], one) for one in answers]
            got["put in"] += made[0] is not None
            got["greedy"] += made[0] is not None and _same(made[0], want)
            got["any"] += any(one is not None and _same(one, want)
                              for one in made)
    print(json.dumps(got))
    return got


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "corpus", "mix", "train",
                                        "measure"))
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default="editor")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--most", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base", default=None)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    options = parser.parse_args(argv)
    if options.job == "fetch":
        fetch()
    elif options.job == "corpus":
        corpus()
    elif options.job == "mix":
        mix()
    elif options.job == "train":
        train(LLM / options.out, epochs=options.epochs, most=options.most,
              batch=options.batch, rate=options.rate, seed=options.seed,
              base=LLM / options.base if options.base else BASE,
              corpus=OUT / options.corpus if options.corpus else EDITS)
    else:
        measure(LLM / options.out, most=options.most or 300,
                corpus=OUT / options.corpus if options.corpus else EDITS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

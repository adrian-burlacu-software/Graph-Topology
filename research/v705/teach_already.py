"""Whether a function already takes what a request asks it to (v705): read
by an encoder taught it, the request against each parameter as the code
says what it is.

    python -m research.v705.teach_already write    llm/already-data/written.jsonl
    python -m research.v705.teach_already check    llm/already-data/checked.jsonl
    python -m research.v705.teach_already train    llm/already
    python -m research.v705.teach_already probe    the requests below, read

`In natural.py, gained should also take the commit message and score each
window after it, so the edits that match what the change is for come
first`: `gained` takes it already, as `said` -- "what the change is for,
said before each window alike". Asked, a person says so; the change path
added `commit: str = ""` beside it. Neither the base sentence encoder
(0.46 for `said`, 0.38 for the next) nor the code model (it copies the
name asked: `commit_message=` likeliest) tells; the teacher does, two of
three times, and is too large to read at run time.

**What it is taught**, from this repository's own functions: a parameter
shown as its function's `def` line, what the code says of it (the
docstring's sentences naming it, the comments over where it is read) and
the lines that read it; and a request --
- asking for what the parameter is, by the teacher, never by its name:
  that parameter, yes; each other parameter of the function, no;
- asking for something the function does not take, by the teacher: no,
  for every parameter -- kept only where the teacher, asked which of the
  parameters it is with the choices in two orders, said none both times.

The labels are how a request was made, not the teacher's reading of it.
Split by the function.

    already(part, statement)    (parameter, chance) or None
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
DATA = LLM / "already-data"
WRITTEN = DATA / "written.jsonl"
CHECKED = DATA / "checked.jsonl"
BASE = LLM / "unixcoder-base"
OUT = LLM / "already"
SEED = 7055
LONGEST = 320
#: functions asked of, for what one of their parameters is and for
#: something new
FUNCTIONS = 1600
#: requests the teacher writes for each
EACH = 3
#: prompts written at once
BATCH = 24
VALID = 0.1
#: the chance from which a parameter is read as what was asked, and how
#: far above the next it must be
FLOOR, MARGIN = 0.5, 0.2


def described(function, source: str) -> dict:
    """{parameter: (what the code says of it, the lines reading it)}."""
    lines = source.splitlines()
    doc = ast.get_docstring(function) or ""
    out = {}
    for arg in function.args.posonlyargs + function.args.args + \
            function.args.kwonlyargs:
        name = arg.arg
        if name in ("self", "cls"):
            continue
        said = [one.strip() for one in re.split(r"(?<=[.:;])\s+", doc)
                if re.search(rf"\b{re.escape(name)}\b", one)]
        reads = []
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and node.id == name and \
                    isinstance(node.ctx, ast.Load):
                at = node.lineno - 2
                block = []
                while at >= 0 and lines[at].strip().startswith("#"):
                    block.insert(0, lines[at].strip().lstrip("#").strip())
                    at -= 1
                if block:
                    said.append(" ".join(block))
                reads.append(lines[node.lineno - 1].strip())
        out[name] = (" ".join(dict.fromkeys(said))[:400],
                     list(dict.fromkeys(reads))[:3])
    return out


def _header(function, source: str) -> str:
    lines = source.splitlines()
    end = function.body[0].lineno - 1
    if function.body and isinstance(function.body[0], ast.Expr) and \
            isinstance(function.body[0].value, ast.Constant):
        end = function.body[0].lineno - 1
    return "\n".join(line.strip() for line in lines[function.lineno - 1:end])


def shown(function, source: str, name: str, what=None) -> str:
    """A parameter as the encoder reads it."""
    what = what or described(function, source).get(name, ("", []))
    said, reads = what
    return (f"{_header(function, source)}\n{name}: {said}\n" +
            "\n".join(reads))


def parameters(source: str) -> list:
    """[(function node, function source, [parameters])] of a module's
    functions taking two or more."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [one.arg for one in node.args.posonlyargs + node.args.args +
                 node.args.kwonlyargs if one.arg not in ("self", "cls")]
        if len(names) >= 2:
            text = textwrap.dedent(ast.get_source_segment(source, node) or "")
            try:
                inner = ast.parse(text).body[0]
            except (SyntaxError, IndexError):
                continue
            out.append((inner, text, names))
    return out


def _functions() -> list:
    """This repository's functions, as (path, function, source, names)."""
    out = []
    for path in sorted((ROOT / "research").rglob("*.py")):
        if path.name.startswith("test"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for function, source, names in parameters(text):
            out.append((path.relative_to(ROOT).as_posix(), function, source,
                        names))
    return out


def split_of(key: str) -> str:
    return "valid" if hashlib.sha1(key.encode()).digest()[0] / 256 < VALID \
        else "train"


def _lines(reply: str) -> list:
    out = []
    for line in reply.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip(
            "\"'“”")
        if 8 <= len(line) <= 200:
            out.append(line)
    return out


def write() -> None:
    from research.v696.teach_meaning import Teacher
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    found = _functions()
    rng.shuffle(found)
    found = found[:FUNCTIONS]
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    jobs = []
    for path, function, source, names in found:
        said = described(function, source)
        # for what one parameter is: one the code says something of
        known = [one for one in names if said[one][0] or said[one][1]]
        if known:
            name = rng.choice(known)
            jobs.append(("is", path, function.name, name, (
                f"Here is a Python function's first line, and what its "
                f"parameter `{name}` is for:\n{_header(function, source)}\n"
                f"`{name}`: {said[name][0] or 'used as: ' + '; '.join(said[name][1])}"
                f"\n\nA programmer who does not know `{function.name}` already "
                f"has this parameter asks for it to take it. Write {EACH} "
                f"different short requests they might type, like "
                f"'{function.name} should also take ...', saying what it is "
                f"in their own words. Never write `{name}` or its words. One "
                f"per line, nothing else.")))
        jobs.append(("new", path, function.name, None, (
            f"Here is a Python function's first line:\n"
            f"{_header(function, source)}\n\nA programmer asks for it to "
            f"take something new that none of its parameters is "
            f"({', '.join(names)}). Write {EACH} different short requests "
            f"they might type, like '{function.name} should also take ...'. "
            f"One per line, nothing else.")))
    with WRITTEN.open("w", encoding="utf-8") as out:
        started = time.time()
        for at in range(0, len(jobs), BATCH):
            chunk = jobs[at:at + BATCH]
            replies = teacher.write([one[4] for one in chunk], longest=160)
            for (kind, path, function, name, _), reply in zip(chunk, replies):
                lines = _lines(reply[0])
                if name:
                    words = [one for one in re.split(r"_+", name.lower())
                             if len(one) >= 3] + [name.lower()]
                    lines = [one for one in lines if not any(
                        re.search(rf"\b{re.escape(word)}\b", one.lower())
                        for word in words)]
                out.write(json.dumps({"kind": kind, "file": path,
                                      "function": function, "name": name,
                                      "requests": lines[:EACH]}) + "\n")
            print(f"  {at + len(chunk)}/{len(jobs)} "
                  f"({time.time() - started:.0f}s)", flush=True)


def _sources() -> dict:
    return {(path, function.name): (function, source, names)
            for path, function, source, names in _functions()}


def check() -> dict:
    """Each request for something new asked of the teacher: which of the
    function's parameters it is, the choices in two orders -- kept where
    it said none both times."""
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    sources = _sources()
    todo = []
    for row in rows:
        if row["kind"] != "new" or (row["file"], row["function"]) not in \
                sources:
            continue
        _, source, names = sources[(row["file"], row["function"])]
        for request in row["requests"]:
            for order in (names, names[::-1]):
                options = list(order) + ["none of these: it is something new"]
                todo.append((row, request, options, (
                    f"```python\n{source[:2500]}\n```\nA programmer asks: "
                    f"\"{request}\"\nWhich parameter of {row['function']} "
                    f"already is what they ask it to take?\n" + "\n".join(
                        f"{letter}) {option}" for letter, option in
                        zip("ABCDEFGHIJKLMNOP", options)) +
                    "\nAnswer with the letter only.")))
    teacher = Teacher()
    said: dict = {}
    for at in range(0, len(todo), BATCH):
        chunk = todo[at:at + BATCH]
        replies = teacher.write([one[3] for one in chunk], longest=3)
        for (row, request, options, _), reply in zip(chunk, replies):
            letter = reply[0].strip()[:1].upper()
            letters = "ABCDEFGHIJKLMNOP"[:len(options)]
            none = letter in letters and letters.index(letter) == \
                len(options) - 1
            key = (row["file"], row["function"], request)
            said[key] = said.get(key, True) and none
    counts = {"kept": 0, "left": 0}
    with CHECKED.open("w", encoding="utf-8") as out:
        for row in rows:
            if row["kind"] == "new":
                kept = [one for one in row["requests"] if said.get(
                    (row["file"], row["function"], one))]
                counts["kept"] += len(kept)
                counts["left"] += len(row["requests"]) - len(kept)
                row = {**row, "requests": kept}
            out.write(json.dumps(row) + "\n")
    print(json.dumps(counts))
    return counts


def pairs(split: str) -> list:
    """[(request, a parameter as read, label)]."""
    sources = _sources()
    out = []
    for row in map(json.loads, CHECKED.open(encoding="utf-8")):
        key = (row["file"], row["function"])
        # the probes' functions never taught
        if key not in sources or split_of(f"{key[0]}#{key[1]}") != split \
                or key in {(one[0], one[1]) for one in PROBES}:
            continue
        function, source, names = sources[key]
        said = described(function, source)
        for request in row["requests"]:
            for name in names:
                out.append((request, shown(function, source, name,
                                           said[name]),
                            float(name == row["name"])))
    return out


def train(out: Path = OUT, epochs: int = 2, batch: int = 32,
          rate: float = 3e-5) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows = pairs("train")
    held = pairs("valid")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE), num_labels=1).to("cuda")
    positive = sum(one[2] for one in rows)
    # yes is one parameter in several: weighed as all the noes together
    weight = torch.tensor((len(rows) - positive) / max(positive, 1),
                          device="cuda")

    def encoded(chunk):
        found = tokenizer([one[0] for one in chunk], [one[1] for one in chunk],
                          truncation="only_second", max_length=LONGEST,
                          padding=True, return_tensors="pt").to("cuda")
        return found, torch.tensor([one[2] for one in chunk], device="cuda")

    loss_of = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    def evaluate() -> dict:
        model.eval()
        counts = {"yes": 0, "yes right": 0, "no": 0, "no right": 0}
        with torch.no_grad():
            for at in range(0, len(held), 64):
                found, labels = encoded(held[at:at + 64])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**found).logits[:, 0].float()
                for logit, label in zip(logits.tolist(), labels.tolist()):
                    kind = "yes" if label else "no"
                    counts[kind] += 1
                    counts[kind + " right"] += (logit > 0) == bool(label)
        model.train()
        return {"yes": round(counts["yes right"] / max(counts["yes"], 1), 3),
                "no": round(counts["no right"] / max(counts["no"], 1), 3)}

    optimiser = torch.optim.AdamW(model.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * math.ceil(len(rows) / batch)
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.06 * steps),
                                               steps)
    print(f"{len(rows)} pairs ({positive:.0f} yes), {len(held)} held, "
          f"{steps} steps", flush=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(rows)
        for at in range(0, len(rows), batch):
            found, labels = encoded(rows[at:at + batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**found).logits[:, 0].float()
            loss = loss_of(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
        print(f"epoch {epoch + 1}: held {evaluate()} "
              f"({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    (out / "already.json").write_text(json.dumps({
        "base": BASE.name, "pairs": len(rows), "epochs": epochs,
        "longest": LONGEST}, indent=1), encoding="utf-8")
    print(f"-> {out}")


_LOADED: dict = {}


def _model():
    if "one" not in _LOADED:
        _LOADED["one"] = None
        if (OUT / "config.json").exists():
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _LOADED["one"] = (torch, device,
                              AutoTokenizer.from_pretrained(str(OUT)),
                              AutoModelForSequenceClassification
                              .from_pretrained(str(OUT)).to(device).eval())
    return _LOADED["one"]


def chances(part: str, statement: str) -> dict:
    """{parameter: the chance it is what the request asks the function
    to take}; empty where the model is not there or the part is not one
    function."""
    loaded = _model()
    try:
        tree = ast.parse(textwrap.dedent(part))
    except SyntaxError:
        return {}
    if loaded is None or not tree.body or not isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    torch, device, tokenizer, model = loaded
    function, source = tree.body[0], textwrap.dedent(part)
    said = described(function, source)
    names = list(said)
    if not names:
        return {}
    found = tokenizer([statement] * len(names),
                      [shown(function, source, one, said[one])
                       for one in names], truncation="only_second",
                      max_length=LONGEST, padding=True,
                      return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**found).logits[:, 0].float()
    return dict(zip(names, torch.sigmoid(logits).tolist()))


def already(part: str, statement: str) -> tuple | None:
    """(the parameter, its chance) the function already takes that is
    what the request asks it to take -- read so above `FLOOR` and `MARGIN`
    above the next -- or None."""
    found = sorted(chances(part, statement).items(), key=lambda one: -one[1])
    if not found or found[0][1] < FLOOR or (
            len(found) > 1 and found[0][1] - found[1][1] < MARGIN):
        return None
    return found[0]


#: requests of this repository's code, by hand, and the parameter each is
#: (None: something new)
PROBES = [
    ("research/v705/natural.py", "gained",
     "In natural.py, gained should also take the commit message and score "
     "each window after it, so the edits that match what the change is for "
     "come first.", "said"),
    ("research/v705/natural.py", "gained",
     "gained should also take the commit message.", "said"),
    ("research/v705/natural.py", "gained",
     "gained should take a device to run on.", None),
    ("research/v705/natural.py", "gained",
     "gained should also take code to put ahead of each window.", "given"),
    ("research/v705/splice.py", "spliced",
     "spliced should take a tab width.", None),
    ("research/v705/shaped.py", "asked",
     "asked should take how many changes to make at most.", None),
    ("research/v705/shaped.py", "asked",
     "asked should also take the project's files.", "project"),
    ("research/v705/shaped.py", "asked",
     "asked should take what the request said.", "said"),
    ("research/v705/repair.py", "declared",
     "declared should also take the namespace to look the class up in.",
     "namespace"),
    ("research/v705/repair.py", "declared",
     "declared should take a cache.", None),
]


def probe() -> int:
    sources = _sources()
    right = 0
    for path, name, statement, want in PROBES:
        function, source, _ = sources[(path, name)]
        got = already(source, statement)
        right += (got[0] if got else None) == want
        print("OK " if (got[0] if got else None) == want else "-- ", name,
              got and (got[0], round(got[1], 2)), "want", want, "|",
              {k: round(v, 2) for k, v in chances(source, statement).items()},
              flush=True)
    print(f"{right}/{len(PROBES)}")
    return right


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "train", "probe"))
    parser.add_argument("--epochs", type=int, default=2)
    options = parser.parse_args(argv)
    if options.job == "write":
        write()
    elif options.job == "check":
        check()
    elif options.job == "train":
        train(epochs=options.epochs)
    else:
        probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())

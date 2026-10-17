"""Whether a function already takes what a request asks it to (v705): read
by an encoder taught it, the request against each parameter as the code
says what it is.

    python -m research.v705.teach_already write    llm/already-data/written.jsonl
    python -m research.v705.teach_already check    llm/already-data/checked.jsonl
    python -m research.v705.teach_already train    llm/already4
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
import sysconfig
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
DATA = LLM / "already-data"
WRITTEN = DATA / "written.jsonl"
CHECKED = DATA / "checked.jsonl"
BASE = LLM / "unixcoder-base"
#: (llm/already: taught on requests for parameters the code says nothing
#: of -- held 75% yes, 72% no, 3 of 10 probes; llm/already2: taught yes or
#: no of each, a parameter the code says something of read as asked for)
OUT = LLM / "already4"
SEED = 7055
LONGEST = 320
#: functions asked of, for what one of their parameters is and for
#: something new
FUNCTIONS = 1600
#: requests the teacher writes for each
EACH = 3
#: prompts written at once
BATCH = 24
#: what the code must say of a parameter for a request to be asked for it
#: (said nothing of, the teacher asked for what it made up: `towers` as "a
#: list of allowed moves"); and how many of a function's are asked for
DESCRIBED, ASKED_OF = 25, 2
#: requests for what a parameter is, written at most
ASKED = 3000
VALID = 0.1
#: the chance from which a parameter is read as what was asked (of its
#: parameters and none, together one)
FLOOR = 0.5


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


#: Python's own library, beside this repository's code: what its
#: docstrings say of its parameters (this repository says what few of its
#: are: 677 requests of a parameter, and `take a device` read as one the
#: function has) -- there wherever Python is, so it is rebuilt as it was
STDLIB = Path(sysconfig.get_paths()["stdlib"])
#: of it, what is not code a function of it is written as
NOT_STDLIB = ("test", "tests", "idlelib", "turtledemo", "site-packages",
              "lib2to3", "ensurepip", "pydoc_data", "__pycache__",
              "encodings", "__phello__")
WHERE = ("research", "stdlib")


def _files(where: tuple = WHERE) -> list:
    """[(the key a file is known by, its path)]."""
    out = []
    if "research" in where:
        out += [(path.relative_to(ROOT).as_posix(), path) for path in sorted(
            (ROOT / "research").rglob("*.py"))
            if not path.name.startswith("test")]
    if "stdlib" in where and STDLIB.exists():
        out += [("stdlib/" + path.relative_to(STDLIB).as_posix(), path)
                for path in sorted(STDLIB.rglob("*.py"))
                if not set(path.relative_to(STDLIB).parts[:-1]) & set(
                    NOT_STDLIB) and not path.name.startswith("test")]
    return out


def _functions(where: tuple = WHERE) -> list:
    """The functions of this repository and of Python's library, as (key,
    function, source, names)."""
    out = []
    for key, path in _files(where):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for function, source, names in parameters(text):
            out.append((key, function, source, names))
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


def write(kinds=("is", "new"), where: tuple = WHERE) -> None:
    """Requests for what a parameter is, of every function saying what one
    is (`ASKED` at most), and for something new, of `FUNCTIONS` of them;
    `kinds` only, or of `where` only, put after what is written."""
    from research.v696.teach_meaning import Teacher
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    found = _functions(where)
    rng.shuffle(found)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED + len(kinds) + len(where))
    jobs = []
    for number, (path, function, source, names) in enumerate(found):
        said = described(function, source)
        # for what a parameter is: one the code says what it is of
        known = [one for one in names if len(said.get(one, ("",))[0]) >=
                 DESCRIBED]
        rng.shuffle(known)
        if sum(one[0] == "is" for one in jobs) >= ASKED:
            known = []
        for name in known[:ASKED_OF] if "is" in kinds else ():
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
        if "new" not in kinds or number >= FUNCTIONS:
            continue
        jobs.append(("new", path, function.name, None, (
            f"Here is a Python function's first line:\n"
            f"{_header(function, source)}\n\nA programmer asks for it to "
            f"take something new: close to what it does, what one might "
            f"think it takes, but that none of its parameters is "
            f"({', '.join(names)}). Write {EACH} different short requests "
            f"they might type, like '{function.name} should also take ...', "
            f"saying what it is in their own words. One per line, nothing "
            f"else.")))
    with WRITTEN.open("w" if len(kinds) == 2 and where == WHERE else "a",
                      encoding="utf-8") as out:
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
    """Each request asked of the teacher: which of the function's
    parameters it is, the choices in two orders -- one for something new
    kept where it said none both times, one for what a parameter is where
    it said that one both times."""
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    sources = _sources()
    todo = []
    for row in rows:
        if (row["file"], row["function"]) not in sources:
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
            picked = options[letters.index(letter)] if letter in letters \
                else ""
            wanted = row["name"] if row["kind"] == "is" else options[-1]
            key = (row["file"], row["function"], request)
            said[key] = said.get(key, True) and picked == wanted
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for row in rows:
            kept = [one for one in row["requests"] if said.get(
                (row["file"], row["function"], one))]
            for kind, many in (("kept", len(kept)), ("left", len(
                    row["requests"]) - len(kept))):
                key = f"{row['kind']} {kind}"
                counts[key] = counts.get(key, 0) + many
            out.write(json.dumps({**row, "requests": kept}) + "\n")
    print(json.dumps(counts))
    return counts


#: what a function is shown as taking where it takes none of it: the
#: choice beside its parameters
NONE = "none: something it does not take now"


def none_shown(function, source: str) -> str:
    return f"{_header(function, source)}\n{NONE}"


def groups(split: str) -> list:
    """[(request, [each parameter as read, then none], the one it is)] --
    a choice among them, not a yes or no of each: taught yes or no, the
    encoder read a parameter the code says something of as asked for
    (`gained`'s `original` 0.93, `said` 0.98) and one it says nothing of
    as not (0.00), whatever was asked."""
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
        if row["kind"] == "is" and len(said.get(row["name"], ("",))[0]) < \
                DESCRIBED:
            continue
        names = [one for one in names if one in said]
        views = [shown(function, source, one, said[one]) for one in names] + [
            none_shown(function, source)]
        answer = names.index(row["name"]) if row["kind"] == "is" and \
            row["name"] in names else len(names)
        for request in row["requests"]:
            out.append((request, views, answer))
    return out


def train(out: Path = OUT, epochs: int = 3, batch: int = 8,
          rate: float = 3e-5) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows = groups("train")
    held = groups("valid")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE), num_labels=1).to("cuda")

    def scored(chunk):
        texts = [(one[0], view) for one in chunk for view in one[1]]
        found = tokenizer([one[0] for one in texts], [one[1] for one in texts],
                          truncation="only_second", max_length=LONGEST,
                          padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**found).logits[:, 0].float()
        out_, at = [], 0
        for one in chunk:
            out_.append(logits[at:at + len(one[1])])
            at += len(one[1])
        return out_

    def evaluate() -> dict:
        model.eval()
        counts = {"is": [0, 0], "none": [0, 0]}
        with torch.no_grad():
            for at in range(0, len(held), 16):
                chunk = held[at:at + 16]
                for one, logits in zip(chunk, scored(chunk)):
                    kind = "none" if one[2] == len(one[1]) - 1 else "is"
                    counts[kind][0] += int(logits.argmax()) == one[2]
                    counts[kind][1] += 1
        model.train()
        return {kind: round(right / max(total, 1), 3)
                for kind, (right, total) in counts.items()}

    optimiser = torch.optim.AdamW(model.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * math.ceil(len(rows) / batch)
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.06 * steps),
                                               steps)
    print(f"{len(rows)} requests ({sum(one[2] < len(one[1]) - 1 for one in rows)}"
          f" of a parameter), {len(held)} held, {steps} steps", flush=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(rows)
        for at in range(0, len(rows), batch):
            chunk = rows[at:at + batch]
            loss = sum(torch.nn.functional.cross_entropy(
                logits[None], torch.tensor([one[2]], device="cuda"))
                for one, logits in zip(chunk, scored(chunk))) / len(chunk)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
        print(f"epoch {epoch + 1}: held right {evaluate()} "
              f"({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    (out / "already.json").write_text(json.dumps({
        "base": BASE.name, "requests": len(rows), "epochs": epochs,
        "longest": LONGEST, "choice": True}, indent=1), encoding="utf-8")
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
    to take, `NONE`: that it is none of them}; empty where the model is
    not there or the part is not one function."""
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
    views = [shown(function, source, one, said[one]) for one in names] + [
        none_shown(function, source)]
    found = tokenizer([statement] * len(views), views,
                      truncation="only_second", max_length=LONGEST,
                      padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**found).logits[:, 0].float()
    return dict(zip(names + [NONE], torch.softmax(logits, 0).tolist()))


def already(part: str, statement: str) -> tuple | None:
    """(the parameter, its chance) the function already takes that is
    what the request asks it to take -- the likeliest choice, none below
    it, at `FLOOR` or above -- or None."""
    found = sorted(chances(part, statement).items(), key=lambda one: -one[1])
    if not found or found[0][0] == NONE or found[0][1] < FLOOR:
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
    parser.add_argument("--kinds", nargs="*", default=["is", "new"])
    parser.add_argument("--where", nargs="*", default=list(WHERE))
    options = parser.parse_args(argv)
    if options.job == "write":
        write(tuple(options.kinds), tuple(options.where))
    elif options.job == "check":
        check()
    elif options.job == "train":
        train(epochs=options.epochs)
    else:
        probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())

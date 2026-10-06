"""Teaching the reader of meaning: English and code in the same records.

Nothing here is read at run time. The learned reader (`reader.py`) reads
one sequence -- whatever of the English, the signature, the examples and a
body a record has -- into one `Meaning`; this builds what it learns from,
and every label comes from an exact channel (`meaning.py`): behaviour by
running, structure by the compiler.

    python -m research.v696.teach_meaning docs         lib.d.ts JSDoc
    python -m research.v696.teach_meaning solutions    MBPP-TS, solved by
                                                       SmolLM3, kept when
                                                       the task's tests pass
    python -m research.v696.teach_meaning described    generated programs,
                                                       said in English by
                                                       SmolLM3, kept when a
                                                       round trip agrees
    python -m research.v696.teach_meaning corpus       the records

SmolLM3 is the teacher only (never at run time), and nothing it writes is
kept unchecked: a solution by the task's own tests; a description by writing
the function again from the English alone and running it against the
program it describes.

HumanEval-TS is never taught from: it is the held test. MBPP-TS is split by
name into train and dev.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from research.v696 import generating as G
from research.v696 import meaning as M
from research.v696 import program as P
from research.v696.checker import CheckerError, checker

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "code-meaning"
TEACHER = ROOT / "llm" / "SmolLM3-3B"

DOCS = DATA / "docs.jsonl"
SOLUTIONS = DATA / "solutions.jsonl"
#: HumanEval solved the same way -- only to measure what the reader says a
#: held program is made of; never taught from.
SOLUTIONS_HELD = DATA / "solutions-held.jsonl"
DESCRIBED = DATA / "described.jsonl"
CORPUS = DATA / "corpus.jsonl"

#: The seeds the generated programs taught from are drawn from: apart from
#: the search's dev (0, 100000) and held (1000000) seeds.
TEACH = 5_000_000


def split(name: str) -> str:
    """MBPP by name: a fifth is dev. HumanEval is held."""
    if name.startswith("HumanEval"):
        return "held"
    digest = hashlib.sha1(name.encode()).digest()[0]
    return "dev" if digest % 5 == 0 else "train"


def _kept(path: Path, key: str = "name") -> set:
    if not path.exists():
        return set()
    return {json.loads(line)[key] for line in path.open(encoding="utf-8")}


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


# -- the teacher -------------------------------------------------------------

class Teacher:
    """SmolLM3 in 4 bits, writing in batches (left-padded)."""

    def __init__(self, path: Path = TEACHER) -> None:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), device_map="cuda", quantization_config=(
                BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)))
        self.model.eval()

    def write(self, prompts: list, longest: int, samples: int = 1,
              temperature: float = 0.7) -> list:
        """For each prompt, `samples` replies (greedy when one)."""
        texts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": one}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
            for one in prompts]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
        settings = (dict(do_sample=False) if samples == 1 else
                    dict(do_sample=True, temperature=temperature,
                         top_p=0.95, num_return_sequences=samples))
        with self.torch.no_grad():
            made = self.model.generate(
                **batch, max_new_tokens=longest,
                pad_token_id=self.tokenizer.pad_token_id, **settings)
        width = batch["input_ids"].shape[1]
        out = [[] for _ in prompts]
        for index, row in enumerate(made[:, width:]):
            out[index // samples].append(self.tokenizer.decode(
                row, skip_special_tokens=True).strip())
        return out


FENCE = re.compile(r"```(?:ts|typescript|javascript|js)?\s*\n(.*?)```", re.S)


def code_in(reply: str, entry: str) -> str | None:
    """The function named `entry` in a reply, fenced or not."""
    found = FENCE.findall(reply)
    text = found[0] if found else reply
    at = text.find(f"function {entry}")
    return text[at:].strip() + "\n" if at >= 0 else None


# -- lib.d.ts: each member's English beside its declaration ------------------

def docs() -> int:
    """Every operator and form of the library with its JSDoc, as a
    function over its declared types, its behaviour run on generated
    values (forms need a body, so theirs is unknown)."""
    lib = P.library()
    said = {}
    for one in checker().signatures(list(P.TYPES), list(P.GLOBALS)):
        if one.get("doc"):
            said.setdefault((one["receiver"], one["name"]), one["doc"])
    rng = random.Random(696)
    rows = []
    for op in lib.ops + lib.forms:
        if isinstance(op, P.Form):
            doc = said.get((op.receiver, op.name))
            takes = (op.receiver,)
        else:
            receiver = op.needs[0] if op.kind in ("method",
                                                   "property") else None
            doc = said.get((receiver, op.name))
            takes = op.needs
        if not doc:
            continue
        names = [f"{'abc'[at]}" for at in range(len(takes))] \
            if len(takes) <= 3 else None
        if names is None:
            continue
        pairs = None
        if isinstance(op, P.Form):
            body = None
            gives = op.gives
        else:
            body = op.said(names)
            gives = op.gives
            cases = [[G._value(kind, rng) for kind in takes]
                     for _ in range(8)]
            try:
                row = checker().values(names, cases, [body])[0]
            except CheckerError:
                continue
            pairs = [(case, one["value"]) for case, one in zip(cases, row)
                     if "error" not in one]
        signature = (f"function f({', '.join(f'{n}: {k}' for n, k in zip(names, takes))})"
                     f": {gives}")
        rows.append({
            "name": f"doc:{M.word(op)}:{','.join(takes)}",
            "source": "docs", "split": "train", "english": doc,
            "signature": signature,
            "examples": [] if pairs is None else pairs[:3],
            "body": None if body is None
            else f"{signature} {{\n  return {body};\n}}\n",
            "meaning": M.Meaning(
                tuple(takes), gives,
                None if pairs is None else M.behaviour(pairs),
                frozenset([M.word(op)]), M.word(op)).json()})
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    with DOCS.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} members with their English -> {DOCS}")
    return len(rows)


# -- MBPP-TS: solved by the teacher, kept by the tests -----------------------

SOLVE = ("Complete this TypeScript function. Reply with only the whole "
         "function in one ```ts block, no explanation.\n\n```ts\n{prompt}"
         "```")


def solutions(batch: int = 8, samples: int = 4, limit: int = 0,
              config: str = "mbpp-ts") -> None:
    from research.v696 import tasks
    path = SOLUTIONS_HELD if config == "humaneval-ts" else SOLUTIONS
    todo = [one for one in tasks.load(config)
            if one.name not in _kept(path)]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} MBPP-TS tasks to solve", flush=True)
    teacher = Teacher()
    solved = 0
    started = time.time()
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write([SOLVE.format(prompt=one.prompt)
                                 for one in chunk], longest=384,
                                samples=samples)
        for task, written in zip(chunk, replies):
            kept = None
            for reply in written:
                code = code_in(reply, task.entry)
                if not code:
                    continue
                try:
                    passed = checker().tests(code + "\n" + task.tests) \
                        is None
                except CheckerError:
                    # what it ran killed Node: not a solution
                    checker().restart()
                    passed = False
                if passed:
                    kept = code
                    break
            solved += kept is not None
            _append(path, {"name": task.name, "code": kept})
        print(f"  {at + len(chunk)}/{len(todo)}: {solved} solved "
              f"({time.time() - started:.0f}s)", flush=True)


# -- generated programs: said in English, kept by a round trip ---------------

DESCRIBE = ("Here is a TypeScript function:\n\n```ts\n{code}```\n\n"
            "Write the one-sentence request someone would make for this "
            "function, starting \"Write a function that\". Say what it "
            "computes from its inputs, not how: do not name the methods or "
            "operators it uses. Reply with the sentence only.")
REWRITE = ("{english}\n\nComplete this TypeScript function. Reply with only "
           "the whole function in one ```ts block.\n\n```ts\n{signature} {{\n"
           "```")


def _programs(count: int) -> list:
    """Generated programs at rungs 1 and 2, from the teaching seeds."""
    out, seed = [], TEACH
    while len(out) < count:
        rung = seed % 4
        found = (G.task2(seed, 1) if rung == 3
                 else G.task(seed, 1 + rung))
        if found is not None:
            out.append(found)
        seed += 1
    return out


def described(count: int = 3000, batch: int = 16, samples: int = 2) -> None:
    kept = _kept(DESCRIBED)
    todo = [one for one in _programs(count) if one.name not in kept]
    print(f"{len(todo)} generated programs to describe", flush=True)
    teacher = Teacher()
    agreed = 0
    started = time.time()
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        codes = [one.function(one.answer) for one in chunk]
        sentences = teacher.write([DESCRIBE.format(code=code)
                                   for code in codes], longest=60,
                                  samples=samples, temperature=0.8)
        asks, whose = [], []
        for index, (spec, said) in enumerate(zip(chunk, sentences)):
            for sentence in {one.split("\n")[0].strip() for one in said}:
                if sentence:
                    asks.append(REWRITE.format(english=sentence,
                                               signature=spec.signature()))
                    whose.append((index, sentence))
        rewritten = teacher.write(asks, longest=200) if asks else []
        good = {}
        for (index, sentence), reply in zip(whose, rewritten):
            spec = chunk[index]
            code = code_in(reply[0], spec.entry)
            if code is None or index in good:
                continue
            cases = [args for args, _ in spec.examples + spec.hidden]
            wanted = [out for _, out in spec.examples + spec.hidden]
            try:
                got = checker().run(code, spec.entry, cases)
            except CheckerError:
                checker().restart()
                continue
            if all("value" in one and json.dumps(one["value"])
                   == json.dumps(out) for one, out in zip(got, wanted)):
                good[index] = sentence
        for index, (spec, code) in enumerate(zip(chunk, codes)):
            agreed += index in good
            _append(DESCRIBED, {"name": spec.name, "code": code,
                                "english": good.get(index),
                                "said": sentences[index]})
        print(f"  {at + len(chunk)}/{len(todo)}: {agreed} agreed "
              f"({time.time() - started:.0f}s)", flush=True)


# -- the records -------------------------------------------------------------

COMMENT = re.compile(r"^\s*//\s?(.*)$")


def _english(prompt: str) -> str:
    """A MultiPL-E prompt's comment, without its examples -- or a Python
    prompt's docstring, without its doctests."""
    if "def " in prompt and ('"""' in prompt or "'''" in prompt):
        import ast
        try:
            tree = ast.parse(prompt + "\n    pass\n")
        except SyntaxError:
            tree = None
        function = next((one for one in getattr(tree, "body", [])
                         if isinstance(one, ast.FunctionDef)), None)
        doc = ast.get_docstring(function) if function is not None else None
        if doc:
            kept, skip = [], False
            for line in doc.splitlines():
                line = line.strip()
                if line.startswith(">>>"):
                    skip = True
                    continue
                if skip and line:
                    # a doctest's result, after its call
                    skip = False
                    continue
                skip = False
                if line.lower().startswith(("example", "for example")):
                    continue
                kept.append(line)
            return re.sub(r"\s+", " ", " ".join(kept)).strip()
    lines = []
    for line in prompt.splitlines():
        found = COMMENT.match(line)
        if not found:
            continue
        text = found.group(1)
        if text.startswith(">>>") or (lines and lines[-1].startswith(">>>")):
            lines.append(">>>" if text.startswith(">>>") else ">>>")
            continue
        lines.append(text)
    words = " ".join(one for one in lines if one != ">>>")
    words = re.sub(r"\btsthon\b", "TypeScript", words)
    return re.sub(r"\s+", " ", words).strip()


def _signature(prompt: str) -> str:
    found = re.search(r"function\s+\w+\s*\([^)]*\)\s*:\s*[^{]+", prompt, re.S)
    return re.sub(r"\s+", " ", found.group(0)).strip() if found else ""


def _task_records(config: str, solved: dict) -> list:
    from research.v696 import tasks
    rows = []
    for task in tasks.load(config):
        pairs = M.values_of(M.test_pairs(task.tests))
        if not pairs:
            continue
        code = solved.get(task.name)
        uses = root = None
        if code:
            uses, root = M.structure(code, task.entry)
        rows.append({
            "name": task.name, "source": config, "split": split(task.name),
            "english": _english(task.prompt),
            "signature": _signature(task.prompt),
            "examples": pairs[:3], "body": code,
            "meaning": M.Meaning(tuple(kind for _, kind in task.params),
                                 task.returns, M.behaviour(pairs), uses,
                                 root).json()})
    return rows


def _described_records() -> list:
    rows = []
    programs = {one.name: one for one in _programs(
        sum(1 for _ in DESCRIBED.open(encoding="utf-8")))} \
        if DESCRIBED.exists() else {}
    for line in DESCRIBED.open(encoding="utf-8") if DESCRIBED.exists() \
            else ():
        row = json.loads(line)
        spec = programs.get(row["name"])
        if spec is None:
            continue
        pairs = spec.examples + spec.hidden
        uses, root = M.structure(row["code"], spec.entry)
        rows.append({
            "name": row["name"], "source": "generated", "split": "train",
            "english": row["english"], "signature": spec.signature(),
            "examples": spec.examples[:3], "body": row["code"],
            "meaning": M.Meaning(tuple(kind for _, kind in spec.params),
                                 spec.returns, M.behaviour(pairs), uses,
                                 root).json()})
    return rows


def corpus() -> None:
    solved = {}
    for path in (SOLUTIONS, SOLUTIONS_HELD):
        if path.exists():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row["code"]:
                    solved[row["name"]] = row["code"]
    rows = []
    if DOCS.exists():
        rows += [json.loads(line) for line in DOCS.open(encoding="utf-8")]
    rows += _task_records("mbpp-ts", solved)
    rows += _task_records("humaneval-ts", solved)
    rows += _described_records()
    # and Python's, beside them: one reader for both (v699, `pycorpus.py`)
    from research.v696 import pycorpus
    rows += pycorpus.records()
    with CORPUS.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    count = {}
    for row in rows:
        key = (row["source"], row["split"], row["english"] is not None,
               row["body"] is not None)
        count[key] = count.get(key, 0) + 1
    for key in sorted(count, key=str):
        source, part, english, body = key
        print(f"  {source:13} {part:5} english={english!s:5} "
              f"body={body!s:5} {count[key]}")
    print(f"{len(rows)} records -> {CORPUS}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("docs", "solutions", "described",
                                        "corpus"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--held", action="store_true",
                        help="solutions: HumanEval, to measure with")
    args = parser.parse_args(argv)
    if args.job == "docs":
        docs()
    elif args.job == "solutions":
        solutions(batch=args.batch or 8, limit=args.limit,
                  config="humaneval-ts" if args.held else "mbpp-ts")
    elif args.job == "described":
        described(args.count, batch=args.batch or 16)
    else:
        corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())

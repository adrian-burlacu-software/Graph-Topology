"""Requests the decoder has not seen, written and solved by the teacher.

MBPP gives some 300 requests to teach the decoder from, and HumanEval may
not be taught from. The teacher (SmolLM3, offline -- never at run time)
writes new ones in MultiPL-E's own form -- a comment saying what is wanted,
examples as `// >>> f(...)` then the value, the signature -- each shown a
few MBPP train requests as the kind of thing wanted; then it solves them,
in samples of its own. Nothing it writes is kept unchecked. A request is
kept with a solution when:

    the solution meets the request's own examples -- two writings by the
        teacher, the request and the solution, agree on what it does --
        and its result on them is not the same for all of them
    it reads into the search's tree (`parse.py`): the decoder is taught
        only what the reader reads
    it is not one of HumanEval's or MBPP dev's: no entry name in common,
        no run of eight words in common with their English

    python -m research.v696.teach_requests write [--count N]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

from research.v696 import meaning as M
from research.v696 import tasks as T
from research.v696.checker import CheckerError, checker
from research.v696.parse import parse
from research.v696.teach_meaning import (DATA, Teacher, _append, _kept,
                                         code_in, split)

REQUESTS = DATA / "requests.jsonl"

INVENT = (
    "Here are some programming tasks. Each is a TypeScript function: a "
    "comment saying what it does, examples of calls and their results, and "
    "the signature.\n\n{seeds}\n\nWrite ONE new task of the same kind, "
    "different from these and on a different subject, and not trivial: one "
    "that takes a few steps to do. A comment of one to three sentences "
    "saying what the function does, then three examples, "
    "each as two comment lines `// >>> name(arguments)` and `// result`, "
    "then the signature line ending with `{{`. Use only numbers, strings, "
    "booleans and arrays of them. Reply with only the task in one ```ts "
    "block, no body.")

SOLVE = ("Complete this TypeScript function. Reply with only the whole "
         "function in one ```ts block, no explanation.\n\n```ts\n{prompt}"
         "\n```")

FENCE = re.compile(r"```(?:ts|typescript)?\s*\n(.*?)```", re.S)


def _seed(task: T.Task, pairs: list) -> str:
    """An MBPP train request in MultiPL-E's form, with two of its tests
    as its examples."""
    head, _, signature = task.prompt.rstrip().rpartition("\n")
    examples = "".join(f"// >>> {task.entry}({args})\n// {value}\n"
                       for args, value in pairs[:2])
    return f"{head}\n{examples}{signature}"


def _words(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def _grams(text: str, n: int = 8) -> set:
    words = _words(text)
    return {" ".join(words[at:at + n]) for at in range(len(words) - n + 1)}


def _withheld() -> tuple:
    """What may not be taught: HumanEval's and MBPP dev's names and
    English."""
    names, grams = set(), set()
    for config in T.CONFIGS:
        for task in T.load(config):
            if config == "humaneval-ts" or split(task.name) == "dev":
                names.add(task.entry.lower())
                grams |= _grams(task.prompt)
    return names, grams


def _request(reply: str) -> T.Task | None:
    found = FENCE.findall(reply)
    text = (found[0] if found else reply).strip()
    task = T.parse({"name": "", "prompt": text, "tests": ""}, "requests")
    if task is None or len(task.examples) < 2 or not task.params:
        return None
    return task


def _values(task: T.Task) -> list | None:
    """The request's examples as values, as Node makes them."""
    pairs = M.values_of([(args, value) for args, value in task.examples])
    return pairs if len(pairs) == len(task.examples) else None


def write(count: int = 3000, batch: int = 8, samples: int = 4,
          seed: int = 696) -> None:
    rng = random.Random(seed + len(_kept(REQUESTS, "prompt")))
    seeds = []
    for task in T.load("mbpp-ts"):
        if split(task.name) != "train":
            continue
        pairs = M.test_pairs(task.tests)
        if len(pairs) >= 2:
            seeds.append(_seed(task, pairs))
    names, grams = _withheld()
    # what has been asked already: a request is not taught twice
    asked = set()
    if REQUESTS.exists():
        for line in REQUESTS.open(encoding="utf-8"):
            row = json.loads(line)
            asked.add(row["entry"].lower())
            asked.add(" ".join(_words(row["prompt"])[:12]))
    teacher = Teacher()
    made = len(_kept(REQUESTS, "prompt"))
    kept = sum(1 for line in REQUESTS.open(encoding="utf-8")
               if json.loads(line)["code"]) if REQUESTS.exists() else 0
    started = time.time()
    while made < count:
        asks = [INVENT.format(seeds="\n\n".join(
            f"```ts\n{one}\n```" for one in rng.sample(seeds, 3)))
            for _ in range(batch)]
        found = []
        # sampled, not greedy: the same seeds would give the same request
        replies = [one for two in teacher.write(asks, longest=320, samples=2,
                                                temperature=1.0)
                   for one in two]
        for reply in replies:
            task = _request(reply)
            if task is None or task.entry.lower() in names \
                    or _grams(task.prompt) & grams:
                continue
            opening = " ".join(_words(task.prompt)[:12])
            if task.entry.lower() in asked or opening in asked:
                continue
            asked.update((task.entry.lower(), opening))
            pairs = _values(task)
            if pairs is None or len({json.dumps(out) for _, out in pairs}) < 2:
                continue
            found.append((task, pairs))
        if not found:
            continue
        replies = teacher.write([SOLVE.format(prompt=task.prompt)
                                 for task, _ in found], longest=384,
                                samples=samples)
        for (task, pairs), written in zip(found, replies):
            code = None
            for reply in written:
                one = code_in(reply, task.entry)
                if not one or parse(one, task.entry, task.params) is None:
                    continue
                try:
                    got = checker().run(one, task.entry,
                                        [args for args, _ in pairs])
                except CheckerError:
                    checker().restart()
                    continue
                if all("value" in row and json.dumps(row["value"],
                                                     sort_keys=True)
                       == json.dumps(out, sort_keys=True)
                       for row, (_, out) in zip(got, pairs)):
                    code = one
                    break
            made += 1
            kept += code is not None
            _append(REQUESTS, {
                "name": f"request_{made}_{task.entry}", "prompt": task.prompt,
                "entry": task.entry, "params": task.params,
                "returns": task.returns,
                "examples": [[args, out] for args, out in pairs],
                "code": code})
        print(f"  {made}/{count} requests, {kept} kept "
              f"({time.time() - started:.0f}s)", flush=True)


PEOPLE = DATA / "sketches-people.jsonl"


def corpus() -> int:
    """The decoder's records, in the form the pipeline asks (`reader.
    request`: the English, the signature, the examples) with no meaning
    line, each target a whole function as it was written: the teacher's
    requests kept here, and MBPP's verified solutions that read."""
    from research.v696 import reader as R
    from research.v696.teach_meaning import (SOLUTIONS, _english,
                                             _signature)
    rows = []

    def record(name, source, part, prompt, examples, code):
        english, said = R.said({"english": _english(prompt),
                                "signature": _signature(prompt),
                                "examples": examples},
                               R.VIEWS["english+signature+examples"])
        rows.append({"name": name, "source": source, "split": part,
                     "english": english, "code": said, "meaning": None,
                     "target": code.strip()})

    if REQUESTS.exists():
        for line in REQUESTS.open(encoding="utf-8"):
            row = json.loads(line)
            if row["code"]:
                record(row["name"], "requests", "train", row["prompt"],
                       row["examples"], row["code"])
    by = {one.name: one for one in T.load("mbpp-ts")}
    for line in SOLUTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        task = by.get(row["name"])
        if not row["code"] or task is None or parse(
                row["code"], task.entry, task.params) is None:
            continue
        pairs = M.values_of(M.test_pairs(task.tests))[:2]
        record(task.name, "mbpp-ts", split(task.name), task.prompt,
               [[list(args), out] for args, out in pairs], row["code"])
    with PEOPLE.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    counts = {}
    for row in rows:
        key = f"{row['source']} {row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(counts, f"-> {PEOPLE}")
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("write", "corpus"))
    parser.add_argument("--count", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.job == "write":
        write(args.count)
    else:
        corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Teaching the shared reader a change said in the words of what a program
does, naming none of its code, and a function asked for by what it does
(2026-10-21, the core made reliable).

    python -m research.v698.teach_change_said write    the teacher writes them
    python -m research.v698.teach_change_said check    and says what each asks
    python -m research.v698.teach_change_said corpus   the readers' records

The reader knew a change by the code it names (`make {SUBJ} take {NEW}`),
and one said as its users would say it not at all: `make loans last 21
days` was read as no code talk and written as a command (`geto -m 21`),
`let members hold 4 books` as a question of a class's fields -- and the
same with `instead of 14` after it as the change it is (0.99). Taught here:
what a constant or a function of a real project does, changed, said in
everyday words of what it does (act `change`, subject `described`, placed
afterwards by meaning: `asking._described`), some as said after another
request (`and ...`, `also ...`). And a function asked for, said as a bare
phrase or as the question it answers (`python function: is a year a leap
year?` was read as no code talk, and mathematics answered "6"): act `make`.
The same messages are no task of the shell and no question of data.

**What the teacher writes** (SmolLM3, offline): of a constant or a function
of this repository or Python's own library (`teach_code_ask.symbols`, the
conversation tests' projects never shown), its code shown, messages asking
that what it does change -- never its name nor a word of its code. **What
it checks**: each message put back to it as a choice -- a change of what the
program does, a question of the code, a command for the computer, something
else -- kept where it chose the change.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.v698 import teach_code_ask as A
from research.v698 import teach_code_talk as K

WRITTEN = K.DATA / "change-said-written.jsonl"
CHECKED = K.DATA / "change-said-checked.jsonl"
SHELL = K.LLM / "shell-data"
SEED = 7062
#: prompts per kind of what is changed, and samples of each
EACH, SAMPLES = 45, 2
#: each kept message taught this many times (beside 43 000 changes that
#: name their code)
REPEAT = 3

#: what the teacher is asked of each kind (its examples of no project the
#: conversation tests have)
#: (the teacher shown whole examples wrote them again for every setting --
#: `make the cache hold 300 entries` of one called GRACE: only their forms
#: are shown, what they are of is the code's)
JOBS = {
    "constant": ("asking that what this setting controls be different from "
                 "now on. First work out from its name, its comment and its "
                 "value what it does in the program; then say the change as "
                 "the program's users would, in everyday words about that, "
                 "with a new amount, as someone who has never read the code "
                 "(forms like 'let ... N at once', '... should last N ...', "
                 "'make ... N instead of M', 'allow up to N ...')"),
    "function": ("asking that what this function does change. First work "
                 "out what it does in the program; then say a change of "
                 "that as its users would, in everyday words about what it "
                 "does, as someone who has never read the code (forms like "
                 "'... should also ...', 'make ... ignore ...', '... must "
                 "refuse ...', 'don't ... when ...')"),
    # a function asked for, said by what it does: `python function: is a
    # year a leap year?` was read as no code talk (mathematics answered
    # "6"), `python function that returns the average of a list of
    # numbers` too
    "make": ("asking for a new function that does what this one does. "
             "First work out what it does; then ask for it by what it does, "
             "many as bare phrases or as the question it answers, naming "
             "the language now and then (forms like 'python function that "
             "returns ...', 'a function: is ...?', 'function to ...', "
             "'javascript: ...')"),
}
#: what each kind of job is taught as: (act, subject)
TAUGHT = {"constant": ("change", "described"),
          "function": ("change", "described"), "make": ("make", "none")}
#: a change pointing back at what was said before it (teaching only)
POINTED = re.compile(r"\b(it|that|this|those|these|them)\b", re.I)

def jobs() -> list:
    """(key, prompt, meta) for each kind of what is changed."""
    found = A.symbols()
    rng = random.Random(SEED)
    out = []
    for kind, asked in JOBS.items():
        act, subject = TAUGHT[kind]
        pool = [one for one in found["function" if kind == "make" else kind]
                if not A._held(one["name"])]
        # a setting said by what it controls: one whose comment says it
        # (`STEPS = 12` alone tells nothing of it)
        pool = [one for one in pool if kind != "constant" or (
            one["code"].startswith("#:") and re.search(
                r"=\s*-?[\d._]+\s*$", one["code"]))] or pool
        for at in range(EACH):
            one = rng.choice(pool)
            meta = {"act": act, "aspect": "none", "subject": subject,
                    "kind": kind, "name": one["name"]}
            prompt = (f"Here is part of a program, from {one['file']}:\n"
                      f"```python\n{one['code']}\n```\nWrite 12 different "
                      f"messages a programmer might type to a coding "
                      f"assistant {asked}. Never write the name "
                      f"{one['name']}, nor any word of the code. ")
            prompt += ("A few short, a few beginning with 'and' or 'also', "
                       "as said just after another request. Each asks for "
                       "the change; none says it is done. "
                       if act == "change" else
                       "Each asks for the function; none gives it. ")
            out.append((f"{kind}|{at}", prompt + K.STYLE, meta))
    return out


def _kept(lines: list, meta: dict) -> list:
    """Lines that never say the name, nor all of its words."""
    out = []
    name = meta["name"]
    parts = [one for one in re.split(r"[_.]", name.lower()) if len(one) >= 3]
    for line in dict.fromkeys(lines):
        lowered = line.lower()
        if not 6 <= len(line) <= 200 or name.lower() in lowered or (
                len(parts) > 1 and all(one in lowered for one in parts)) or \
                "`" in line or "(" in line:
            continue
        out.append(line)
    return out


def write() -> None:
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    todo = jobs()
    done = set()
    if WRITTEN.exists():
        done = {json.loads(line)["key"] for line in
                WRITTEN.open(encoding="utf-8")}
    todo = [one for one in todo if one[0] not in done]
    with WRITTEN.open("a", encoding="utf-8") as out:
        for at in range(0, len(todo), A.BATCH):
            chunk = todo[at:at + A.BATCH]
            replies = teacher.write([prompt for _, prompt, _ in chunk],
                                    longest=700, samples=SAMPLES,
                                    temperature=0.9)
            for (key, _, meta), written in zip(chunk, replies):
                lines = []
                for reply in written:
                    lines += K._lines(reply)
                out.write(json.dumps({"key": key, "meta": meta,
                                      "lines": _kept(lines, meta)}) + "\n")
                out.flush()
            print(f"{min(at + A.BATCH, len(todo))} of {len(todo)}",
                  flush=True)


#: what the check offers, in both orders: what a message is written as (a
#: change, a function asked for) is the job's; the check keeps the requests
#: and leaves questions and remarks. (SmolLM3 asked: a choice of five
#: letters kept 12%, leaving `function to test if it's a coroutine`;
#: yes-or-no said no to all; three letters in three orders agreed on 4 of
#: 15 hand-labelled. The 3B code model, two letters in two orders: 10 of
#: 15, one wrong -- `Let's talk about setting the value to 2700.`)
MEANING = {"do": "do something to code: write it, or change what it does",
           "other": "answer a question, or nothing: it only remarks"}
ORDERS = (("do", "other"), ("other", "do"))


def check(batch: int = 24) -> dict:
    """Each message put to the code model as what the person wants, in
    both orders of the choice; kept where both choose a request."""
    from research.v705 import coder
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(row["meta"], line) for row in rows for line in row["lines"]]
    judge = coder.big()
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            chosen = [[] for _ in chunk]
            for order in ORDERS:
                options = "\n".join(f"{letter}) {MEANING[one]}"
                                    for letter, one in zip("AB", order))
                prompts = [f"Message to a coding assistant: \"{line}\"\nThe "
                           f"person wants the assistant to:\n{options}\n"
                           f"Letter:" for _, line in chunk]
                for place, reply in enumerate(judge.write(prompts,
                                                          longest=3)):
                    reply = reply[0] if isinstance(reply, list) else reply
                    letter = str(reply).strip()[:1].upper()
                    chosen[place].append(order["AB".index(letter)]
                                         if letter and letter in "AB"
                                         else "?")
            for (meta, line), both in zip(chunk, chosen):
                kept = both == ["do", "do"]
                key = f"{meta['kind']} " + ("kept" if kept else "left")
                counts[key] = counts.get(key, 0) + 1
                out.write(json.dumps({"meta": meta, "line": line,
                                      "kept": kept}) + "\n")
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    return counts


def corpus() -> dict:
    """The kept messages as code-talk records (a change, described), and as
    no task of the shell and no question of data; the split by the name
    the message was written of (a held name never taught)."""
    from research.v701 import teach_data_talk as D
    from research.v702 import teach_shell as S
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    rng = random.Random(SEED)
    out = {"train": [], "valid": []}
    shell = {"train": [], "valid": []}
    data = {"train": [], "valid": []}
    for one in rows:
        if not one["kept"]:
            continue
        split = "valid" if A._held(one["meta"]["name"]) else "train"
        if one["meta"]["act"] == "change" and POINTED.search(one["line"]):
            # `can we make it 7?`: of what was talked of last, not said by
            # what it does
            continue
        said = K.words(one["line"])
        if not said:
            continue
        meta = one["meta"]
        for _ in range(REPEAT if split == "train" else 1):
            out[split].append(K.record(said, ["O"] * len(said), meta["act"],
                                       "none", meta["subject"],
                                       f"{meta['act']} said"))
        shell[split].append(S._record(said, ["O"] * len(said), "none",
                                      f"{meta['act']} said"))
        lowered = [str(word).lower() for word in said]
        data[split].append(D._tagged(D.record(
            lowered, ["O"] * len(lowered), "none", "none", "not data"), rng))
    counts = {}
    for split in out:
        rng.shuffle(out[split])
        for folder, name, records in (
                (K.DATA, f"{split}-change-said.jsonl", out[split]),
                (SHELL, f"{split}-shell-said.jsonl", shell[split]),
                (A.NOT_DATA, f"{split}-data-said.jsonl", data[split])):
            with (folder / name).open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            counts[f"{name}"] = len(records)
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "corpus", "jobs"))
    options = parser.parse_args(argv)
    if options.job == "jobs":
        made = jobs()
        print(len(made), "prompts")
        for key, prompt, _ in made[::20]:
            print("--", key, "\n", prompt[:900])
        return 0
    {"write": write, "check": check, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

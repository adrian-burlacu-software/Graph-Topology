"""Teaching the shared reader the questions a project's index answers
(2026-10-20, the core made reliable).

    python -m research.v698.teach_code_ask write    the teacher writes questions
    python -m research.v698.teach_code_ask check    and says what each asks
    python -m research.v698.teach_code_ask corpus   llm/code-talk-data/*-ask

The reader knew questions of a function by its name -- what it does, who
calls it, where it is -- and nothing else a project is asked: of 25
questions of the library project, 9 were read as no code talk at all
("what fields does a Book have?" was answered "a satchel, pony,
nightingale"), and the rest of what an index holds had no aspect to be
read as. Taught here, with the answers `asking` gives from
`symbols.Index`:

    value     what a constant is set to            (`what is LOAN_DAYS?`)
    fields    what a class holds: fields, methods  (`what fields does a Book have?`)
    tests     whether, and which, tests check it   (`is there a test for due_date?`)
    cases     what a function answers to: its commands, routes, inputs
              (`which routes does handle answer?`)
    outcome   what it gives or does in a situation said in words
              (`what does oldest return when there are no books?`)

and a subject of its own, **described**: what is asked of is said by what
it does, not named (`which function works out when a loan is due?`, `how
many days does a loan last by default?`) -- found afterwards in the
project by meaning (`asking._described`). And a setting said as it should
be (`loan_days should be 21 in the config`: read as a question, `list`)
taught as the change it asks.

**What the teacher writes** (SmolLM3, offline): questions about a real
symbol, its code shown -- this repository's and Python's own library's
functions, classes, constants -- naming it exactly so, or, described,
never naming it. **What it checks**: each question put back to it as a
choice among the aspects, kept where it chose the one it was written as.
**Labels by construction**: the name's words are found where it is written;
each named question said again with other names of its kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from research.v698 import teach_code_talk as K

ROOT = Path(__file__).resolve().parents[2]
DATA = K.DATA
WRITTEN = DATA / "ask-written.jsonl"
CHECKED = DATA / "ask-checked.jsonl"
#: what is no question of data, though it names data (`loan_days should be
#: 21 in the config`: read as `list`)
NOT_DATA = K.LLM / "data-talk-data"
SEED = 7061
#: prompts per job -- a described one more, said again with no other
#: name -- and samples of each
EACH, DESCRIBED, SETTINGS, SAMPLES = 12, 30, 30, 2
#: prompts the teacher writes at once (6 used half the card: 54 of 264 in
#: fifty minutes)
BATCH = 12
#: each named question said again with other names of its kind (six
#: fillings beside the reader's 450 000 records taught nothing: v705
#: `teach_change_talk.FILLED`)
SWAPS = 12

#: what each new aspect asks, as the teacher is told it and as the check
#: offers it
ASKS = {
    "value": "what value the constant {n} is set to, or what it is by "
             "default",
    "fields": "what fields, attributes or methods objects of the class {n} "
              "have, or what one holds",
    "tests": "whether there is a test for {n}, or which tests check it",
    "cases": "which commands, routes, paths, options or kinds of input "
             "the function {n} handles, accepts or answers",
    "outcome": "what the function {n} returns or does in one particular "
               "situation or for one particular input, the situation said "
               "in words (when a list is empty, for an unknown command, "
               "when the file is missing, for a negative number)",
    # the inverse: an output given, an input asked (CRUXEval's input
    # prediction -- `what input to f gives 12?` read as no code talk)
    "inputs": "which input or arguments make the function {n} return or "
              "give one particular result, that result written out (a "
              "number, a string, a list) -- the input asked for, not the "
              "output",
}
#: what the check offers, a letter each: the new aspects and the ones a
#: question of a symbol is most often confused with
CHOICES = {
    "value": "what value something is set to",
    "fields": "what fields, attributes or methods a class has",
    "tests": "whether there is a test for something",
    "cases": "which commands, routes or inputs a function handles",
    "outcome": "what a function returns or does in a particular situation",
    "inputs": "which input makes a function return a particular result",
    "where": "where something is, or which function or class does a thing",
    "callers": "what calls, uses or imports something",
    "explain": "what something is or does, in general",
    "none": "something else",
}

#: (aspect, subject, the kind of symbol, how the teacher is asked)
JOBS = [
    ("value", "named", "constant", None),
    ("fields", "named", "class", None),
    ("tests", "named", "function", None),
    ("cases", "named", "dispatch", None),
    ("outcome", "named", "function", None),
    ("inputs", "named", "function", None),
    # (25 kept of the first twelve prompts: said more ways)
    ("inputs", "named", "function",
     "what value to give the function {n} so that it returns a result "
     "you write out exactly (a number, a list, a string in quotes) -- "
     "asking for the argument, short questions like 'what input to {n} "
     "gives ...?', 'which x makes {n}(x) == ...?'"),
    ("inputs", "named", "function",
     "which arguments, called with, make the function {n} give back a "
     "particular value you name -- asking for the call that does it, as "
     "'find an input for which {n} returns ...', 'what should I pass to "
     "{n} to get ...?', '{n} gives ... for what?'"),
    ("where", "named", "constant",
     "where the constant {n} is defined, or which file has it"),
    ("where", "named", "class",
     "where the class {n} is defined, or which file has it"),
    ("where", "named", "function",
     "which file has the function {n}, or where it is defined"),
    ("callers", "named", "class",
     "which modules, files or functions import or use the class {n}"),
    ("callers", "named", "constant",
     "which functions use the constant {n}"),
    ("explain", "named", "constant", "what the constant {n} is for"),
    ("explain", "named", "class", "what the class {n} is or stands for"),
    # said by what it does, never named
    ("where", "described", "function",
     "which function in their project does what the one shown does -- "
     "describe what it does in plain words, as someone who has forgotten "
     "its name (like 'which function works out ...', 'what finds the ...', "
     "'where is the code that ...')"),
    ("where", "described", "class",
     "which class in their project stands for what the one shown stands "
     "for -- describe it in plain words, as someone who has forgotten its "
     "name (like 'what class holds ...', 'which type is a ...')"),
    ("value", "described", "constant",
     "what the setting shown is set to -- describe what it controls in "
     "plain words, as someone who has forgotten its name (like 'how many "
     "days does ... last by default?', 'what is the most ... allowed?', "
     "'what version does ... report?')"),
]
#: a setting said as it should be: the change it asks
SETTING_JOB = ("asking that the setting {n} in their file {file} be {v} "
               "from now on, said as people say it ('{n} should be {v} in "
               "{file}', 'make {n} {v}', 'I want {n} at {v} in the config')")


# -- what is named ---------------------------------------------------------------------

def _sources() -> list:
    """(path, text) of this repository's code and Python's own library."""
    out = []
    for path in sorted((ROOT / "research").rglob("*.py")):
        # the conversation tests' projects are what this is measured on
        if "test" not in path.name and "fixtures" not in path.parts:
            out.append(path)
    library = Path(sys.base_prefix) / "Lib"
    for path in sorted(library.glob("*.py")) + sorted(
            library.glob("*/*.py")):
        if "test" not in str(path).lower() and "site-packages" not in \
                str(path) and "idlelib" not in str(path):
            out.append(path)
    found = []
    for path in out:
        try:
            found.append((path.as_posix(), path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return found


def symbols() -> dict:
    """What is named, by kind: {kind: [{name, code, file, ...}]} --
    functions short enough to show, classes with fields, constants
    with a plain value, functions that answer to strings (`dispatch`), and
    settings of data files (`setting`)."""
    from research.v698.symbols import Index
    out = {"function": [], "class": [], "constant": [], "dispatch": []}
    for path, text in _sources():
        try:
            index = Index({"m.py": text})
        except (RecursionError, ValueError):
            continue
        lines = text.splitlines()
        base = path.rsplit("/", 1)[-1]
        for one in index.symbols.values():
            code = "\n".join(lines[one.start - 1:one.end])
            row = {"name": one.name, "code": code, "file": base}
            if one.kind == "function" and 3 <= one.end - one.start <= 30 \
                    and len(one.name) >= 4 and not one.name.startswith("_"):
                out["function"].append(row)
                if index.cases.get(f"m.py#{one.name}"):
                    out["dispatch"].append(row)
            elif one.kind == "class" and index.fields(one.name) and \
                    len(one.name) >= 3 and not one.name.startswith("_"):
                head = "\n".join(lines[one.start - 1:min(one.end,
                                                         one.start + 14)])
                out["class"].append({**row, "code": head})
            elif one.kind == "constant" and one.name.isupper() and \
                    len(one.name) >= 4 and len(one.value) <= 40 and \
                    re.fullmatch(r"[\d.\-'\"\w ,:/]+", one.value or ""):
                out["constant"].append({**row, "code": code
                                        if not one.doc else
                                        f"#: {one.doc}\n{code}"})
    for kind, rows in out.items():
        seen, kept = set(), []
        for row in rows:
            if row["name"] not in seen:
                seen.add(row["name"])
                kept.append(row)
        out[kind] = kept
    return out


def _held(name: str) -> bool:
    """Whether a name is held for validation: never taught."""
    return hashlib.sha1(name.lower().encode()).digest()[0] < 256 * K.VALID


def _settings(most: int = 400) -> list:
    """(setting, value now, file) of data files as projects have them:
    CommitPackFT's YAML and JSON (commits taught only,
    `teach_data_talk._files`)."""
    from research.v701 import datamodel
    from research.v701.teach_data_talk import _files
    out = []
    found = _files(random.Random(SEED))
    for path, text in found["yaml"][:most] + found["json"][:most]:
        model = datamodel.read(path, text)
        if model.error:
            continue
        for place in model.places.values():
            if "[]" in place.path or set(place.types) & {"object", "array"}                     or not place.examples:
                continue
            value = place.examples[0]
            if isinstance(value, bool) or not isinstance(
                    value, (int, float, str)) or len(str(value)) > 30:
                continue
            own = place.path.rsplit(".", 1)[-1]
            if re.fullmatch(r"[a-z][a-z_]{2,24}", own):
                out.append((own, value, path.rsplit("/", 1)[-1]))
    return sorted(set(out), key=str)


# -- what the teacher writes -------------------------------------------------------------

def jobs() -> list:
    """(key, prompt, meta) for each question asked of a symbol."""
    found = symbols()
    rng = random.Random(SEED)
    out = []
    times: dict = {}
    for aspect, subject, kind, asked in JOBS:
        # a job said again another way: its own keys (`inputs`, thrice)
        times[(aspect, subject, kind)] = times.get((aspect, subject,
                                                    kind), 0) + 1
        again = "" if times[(aspect, subject, kind)] == 1 else \
            f"|{times[(aspect, subject, kind)]}"
        pool = [one for one in found[kind] if not _held(one["name"])]
        for at in range(EACH if subject == "named" else DESCRIBED):
            if not pool:
                break
            one = rng.choice(pool)
            what = (asked or ASKS[aspect]).format(n=one["name"])
            meta = {"act": "ask", "aspect": aspect, "subject": subject,
                    "kind": kind, "name": one["name"]}
            prompt = (f"Here is part of a programmer's project, from "
                      f"{one['file']}:\n```python\n{one['code']}\n```\n"
                      f"Write 12 different messages the programmer might "
                      f"type to a coding assistant asking {what} -- each a "
                      f"question or a request, never the answer and never a "
                      f"guess at it. ")
            if subject == "named":
                prompt += (f"Write {one['name']} exactly so in every "
                           f"message. ")
            else:
                prompt += (f"Never write the name {one['name']}, nor any "
                           f"word of the code: say what it is in everyday "
                           f"words. ")
            out.append((f"{aspect}|{subject}|{kind}|{at}{again}",
                        prompt + K.STYLE, meta))
    settings = [one for one in _settings() if not _held(one[0])]
    for at in range(SETTINGS):
        if not settings:
            break
        name, value, file_ = rng.choice(settings)
        new = (value * 2 if isinstance(value, int) and not isinstance(
            value, bool) and value else value)
        if new == value or isinstance(value, str):
            new = rng.choice((21, 5, 9090, 30, 0.5, "debug", "true", 100))
        out.append((f"set|{at}", SETTING_JOB.format(n=name, v=new,
                                                    file=file_)
                    .join(("Write 12 different messages a programmer might "
                           "type to a coding assistant ", ". Write "
                           f"{name} and {new} exactly so in every message. "))
                    + K.STYLE,
                    {"act": "change", "aspect": "none", "subject": "named",
                     "kind": "setting", "name": name, "value": str(new),
                     "file": file_}))
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
        for at in range(0, len(todo), BATCH):
            chunk = todo[at:at + BATCH]
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
            print(f"{min(at + BATCH, len(todo))} of {len(todo)}", flush=True)


#: how a message that asks begins, where it ends with no question mark
#: (teaching only: the teacher wrote answers among its questions --
#: `By default, IPV6LENGTH is 128`)
ASKING = re.compile(r"^(what|which|where|who|whom|whose|how|why|when|is|are|"
                    r"was|were|does|do|did|can|could|would|will|should|has|"
                    r"have|tell|show|list|give|find|check|explain|describe|"
                    r"let|please|help|i need|i want|i'd like|i would|"
                    r"i'm (looking|trying|curious)|i am (looking|trying)|"
                    r"any|got|need|look|quick|hey|so|ok|okay|and|also|"
                    r"could|make|set|change|update|bump|put|use)\b", re.I)


def _asking(line: str) -> bool:
    return line.rstrip().endswith("?") or bool(ASKING.match(line.strip()))


def _kept(lines: list, meta: dict) -> list:
    """Lines that say the name where they must (and the value, of a
    setting) and never where they must not."""
    out = []
    name = meta["name"]
    for line in dict.fromkeys(lines):
        if not 6 <= len(line) <= 200 or not _asking(line):
            continue
        has = re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", line)
        if meta["subject"] == "described":
            parts = [one for one in re.split(r"[_.]", name.lower())
                     if len(one) >= 3]
            if has or name.lower() in line.lower() or (
                    len(parts) > 1 and all(one in line.lower()
                                           for one in parts)):
                continue
        elif not has:
            continue
        if meta.get("value") and meta["value"] not in line:
            continue
        out.append(line)
    return out


def check(batch: int = 24) -> dict:
    """Each question put back to the teacher as a choice of what it asks;
    kept where it chose what it was written as (a setting's change, kept
    as written: its check is that it names the setting and the value)."""
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(row["meta"], line) for row in rows for line in row["lines"]
            if _asking(line)]
    # each line judged once: what was checked before kept as it was (4253
    # lines took fifty minutes to judge again for one aspect added)
    before = {}
    if CHECKED.exists():
        for line in CHECKED.open(encoding="utf-8"):
            one = json.loads(line)
            before[(json.dumps(one["meta"], sort_keys=True), one["line"])] = \
                one["kept"]
    judged = [(meta, line, before[(json.dumps(meta, sort_keys=True), line)])
              for meta, line in todo
              if (json.dumps(meta, sort_keys=True), line) in before]
    todo = [(meta, line) for meta, line in todo
            if (json.dumps(meta, sort_keys=True), line) not in before]
    teacher = Teacher()
    options = list(CHOICES)
    letters = "ABCDEFGHIJ"
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for meta, line, kept in judged:
            key = f"{meta['aspect']}|{meta['subject']} " + (
                "kept" if kept else "left")
            counts[key] = counts.get(key, 0) + 1
            out.write(json.dumps({"meta": meta, "line": line,
                                  "kept": kept}) + "\n")
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            asked = [at for at, (meta, _) in enumerate(chunk)
                     if meta["act"] == "ask"]
            prompts = [f"A programmer typed this to a coding assistant "
                       f"about their project: \"{line}\"\nWhat does the "
                       f"person ask?\n" + "\n".join(
                           f"{letter}) {CHOICES[one]}" for letter, one in
                           zip(letters, options)) +
                       "\nAnswer with the letter only."
                       for _, line in (chunk[at] for at in asked)]
            replies = teacher.write(prompts, longest=3) if prompts else []
            # by place in the chunk (each reply is the list of its samples)
            chosen = {at: (reply[0] if isinstance(reply, list) else reply)
                      for at, reply in zip(asked, replies)}
            for at, (meta, line) in enumerate(chunk):
                if meta["act"] == "ask":
                    letter = str(chosen.get(at, "")).strip()[:1].upper()
                    kept = bool(letter) and letter in letters and \
                        options[letters.index(letter)] == meta["aspect"]
                else:
                    kept = True
                key = f"{meta['aspect']}|{meta['subject']} " + (
                    "kept" if kept else "left")
                counts[key] = counts.get(key, 0) + 1
                out.write(json.dumps({"meta": meta, "line": line,
                                      "kept": kept}) + "\n")
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    return counts


# -- the corpus ------------------------------------------------------------------------

def corpus() -> dict:
    """The kept questions as code-talk records -- act, aspect, subject, the
    name `SUBJ` where it is said -- each named one said again with other
    names of its kind; the split by the name (a held name never taught).
    And the settings' changes as no question of data."""
    from research.v701 import teach_data_talk as D
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    kept = [one for one in rows if one["kept"]]
    rng = random.Random(SEED)
    found = symbols()
    names = {kind: [one["name"] for one in rows_]
             for kind, rows_ in found.items()}
    settings = sorted({one[0] for one in _settings()})
    out = {"train": [], "valid": []}
    not_data = {"train": [], "valid": []}
    for one in kept:
        meta, line = one["meta"], one["line"]
        split = "valid" if _held(meta["name"]) else "train"
        sayings = [(line, meta["name"])]
        if meta["subject"] == "named":
            pool = settings if meta["kind"] == "setting" else \
                names.get(meta["kind"], ())
            others = [name for name in pool
                      if (_held(name) == (split == "valid")) and
                      name != meta["name"]]
            for name in rng.sample(others, min(SWAPS, len(others))):
                said = re.sub(rf"(?<![\w.]){re.escape(meta['name'])}(?!\w)",
                              lambda _, name=name: name, line)
                sayings.append((said, name))
        for said_text, name in sayings:
            said = K.words(said_text)
            roles = ["O"] * len(said)
            if meta["subject"] == "named":
                span = K._match(said, name)
                if span is None:
                    continue
                K._mark(roles, span, "SUBJ")
            made = K.record(said, roles, meta["act"], meta["aspect"],
                            meta["subject"], f"ask {meta['aspect']}")
            if meta["subject"] == "named" and rng.random() < K.KNOWN:
                made["tags"] = ["CODE" if role.endswith("SUBJ") else ""
                                for role in roles]
            out[split].append(made)
            if meta["act"] == "change":
                lowered = [str(word).lower() for word in said]
                not_data[split].append(D._tagged(D.record(
                    lowered, ["O"] * len(lowered), "none", "none",
                    "not data"), rng))
    counts = {}
    for split, records in out.items():
        rng.shuffle(records)
        with (DATA / f"{split}-ask.jsonl").open("w",
                                                encoding="utf-8") as stream:
            for one in records:
                stream.write(json.dumps(one) + "\n")
        counts[split] = len(records)
        with (NOT_DATA / f"{split}-data-settings.jsonl").open(
                "w", encoding="utf-8") as stream:
            for one in not_data[split]:
                stream.write(json.dumps(one) + "\n")
        counts[f"{split} not data"] = len(not_data[split])
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "corpus", "jobs"))
    options = parser.parse_args(argv)
    if options.job == "jobs":
        made = jobs()
        print(len(made), "prompts")
        for key, prompt, _ in made[::25]:
            print("--", key, "\n", prompt[:600])
        return 0
    {"write": write, "check": check, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

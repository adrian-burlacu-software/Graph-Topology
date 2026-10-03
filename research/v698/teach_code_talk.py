"""Teaching the shared reader code talk (`PLAN.md`, "Code talk read by the
encoder").

    python -m research.v698.teach_code_talk lists    the teacher's lists
    python -m research.v698.teach_code_talk write    the teacher writes
    python -m research.v698.teach_code_talk check    and checks each message
    python -m research.v698.teach_code_talk corpus   llm/code-talk-data

Nothing here reads a message at run time: this makes what the encoder is
taught, every label known by construction.

**What the teacher writes** (SmolLM3, offline, kept as it goes): messages a
programmer types to a coding assistant, for every act, aspect and kind of
subject -- each prompt naming a real function or file, written in the
message exactly (`about their function remove_kth_element`): a placeholder
is something a small teacher mangles, a name it copies. Questions about the
whole project and about the code just discussed name nothing. Requests to
write code are MBPP's own, said again in the teacher's words; corrections
are written against MBPP's requests; asking to run a function names it and
its input (from MBPP's tests); teaching a concept names it and each member.
What is *not* code talk: messages using software words in their everyday
senses, v689's reader corpus, v692's mathematics. HumanEval is held:
nothing of it is read here.

**What the teacher checks**: each message is put back to it as a choice --
what is this message doing (ask, make, change, run, teach, none of these),
and for a question, what it asks -- with every option in front of it; a
message is kept where its choice is what the message was written as. A
small model chooses well among options it can see, and badly says yes or
no to one described alone.

**Labels by construction**: a subject's (a concept's, each member's) words
are found in the message as they were given to be written; a message
without them is not kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
DATA = LLM / "code-talk-data"
#: the teacher's lists: project names, ways to name a project, concepts
LISTS = DATA / "written.jsonl"
WRITTEN = DATA / "written2.jsonl"
CHECKED = DATA / "checked2.jsonl"
SEED = 698
#: what share of messages (by what was written) is held for validation --
#: and of names, so that held names were never taught
VALID = 0.12
#: prompts each label is written from, each with a different name
EACH = 6
#: how often a name known to be code is marked so (`CODE`), and how often
#: a word of what is not code talk is
KNOWN, NOISE = 0.75, 0.15

ACTS = ("none", "ask", "make", "change", "run", "teach")
ASPECTS = ("none", "explain", "where", "size", "callers", "calls", "bugs",
           "sure", "why", "others", "risk", "files", "functions")
SUBJECTS = ("none", "project", "file", "named", "last")
ROLES = ("O", "B-SUBJ", "I-SUBJ", "B-CONCEPT", "I-CONCEPT", "B-MEMBER",
         "I-MEMBER")

#: what each aspect asks, as the teacher is told it
ASKS = {
    "explain": "what it is, what it does, or to describe or summarise it",
    "where": "where it is defined or located, or to open or find it",
    "size": "how big or long it is, how many lines or parts it has",
    "callers": "what calls it, uses it, imports it or depends on it",
    "calls": "what it calls, uses or depends on",
    "bugs": "whether it has bugs, errors, problems or mistakes, whether it "
            "compiles, or to review or check it",
    "sure": "how sure the assistant is that it is right, whether it can be "
            "trusted, whether it was verified",
    "why": "why the assistant chose, picked or wrote it that way",
    "others": "what other versions or alternatives were written or "
              "considered",
    "risk": "what risks it has, what could go wrong with it",
    "files": "which files it has",
    "functions": "which functions or methods it has",
}
#: which aspects each kind of subject is asked about (`asking.CAN`)
CAN = {"project": ("explain", "bugs", "files", "functions", "size"),
       "file": ("explain", "bugs", "functions", "size", "callers", "calls",
                "where"),
       "named": ("explain", "bugs", "where", "callers", "calls", "size",
                 "sure", "why", "others", "risk"),
       "last": ("explain", "bugs", "where", "size", "callers", "calls",
                "sure", "why", "others", "risk")}

STYLE = ("Vary the wording a lot: short and long, casual and formal, "
         "questions and requests, a few with small typos or no question "
         "mark. One message per line, nothing else: no numbering, no "
         "quotes, no explanations.")
#: Seed questions the teacher paraphrases, by (aspect, subject): the
#: teacher's input, as its prompts are -- nothing reads a message with
#: them. `{n}` is a real name.
SEEDS = {
    ("callers", "named"): ["who calls {n}?", "what uses {n}?",
                           "where is {n} called from?"],
    ("calls", "named"): ["what does {n} call?", "what does {n} use?",
                         "which functions does {n} depend on?"],
    ("bugs", "named"): ["is {n} buggy?", "any bugs in {n}?",
                        "does {n} have errors?"],
    ("explain", "named"): ["what does {n} do?", "explain {n}"],
    ("where", "named"): ["where is {n}?", "where is {n} defined?"],
    ("size", "named"): ["how long is {n}?"],
    ("sure", "named"): ["is {n} right?", "are you sure about {n}?"],
    ("why", "named"): ["why did you write {n} that way?"],
    ("others", "named"): ["what other versions of {n} were there?"],
    ("risk", "named"): ["what could go wrong with {n}?"],
    ("callers", "file"): ["who imports {n}?", "what uses {n}?"],
    ("calls", "file"): ["what does {n} import?"],
    ("bugs", "file"): ["any bugs in {n}?", "is {n} broken?"],
    ("explain", "file"): ["what is in {n}?"],
    ("functions", "file"): ["which functions are in {n}?"],
    ("where", "file"): ["open {n}"],
    ("size", "file"): ["how big is {n}?"],
    ("callers", "last"): ["who calls it?", "what uses it?"],
    ("calls", "last"): ["what does it call?"],
    ("bugs", "last"): ["is it buggy?", "any bugs in it?"],
    ("sure", "last"): ["is it right?", "is it confirmed?"],
    ("why", "last"): ["why that one?"],
    ("others", "last"): ["what else did you write?"],
    ("explain", "last"): ["what does it do?"],
    ("bugs", "project"): ["any bugs in the project?", "does it compile?"],
    ("explain", "project"): ["what is this project?", "what's in the repo?"],
    ("files", "project"): ["which files are there?"],
    ("functions", "project"): ["list the functions"],
    ("size", "project"): ["how big is the codebase?"],
}
#: everyday questions with a word that is also some function's name, to be
#: paraphrased as what is not code talk: (seed, the word)
EVERYDAY_SEEDS = [
    ("what is the main idea of the story?", "main"),
    ("who calls the shots around here?", "calls"),
    ("what does the label on the jar say?", "label"),
    ("who is going to render the painting?", "render"),
    ("where is the greeting card?", "greeting"),
    ("can you check the train schedule?", "check"),
    ("what is the total on the bill?", "total"),
    ("who will run the meeting?", "run"),
    ("is the filter in the coffee machine clean?", "filter"),
    ("where did you put the list of names?", "list"),
    ("what does the sign over the door mean?", "sign"),
    ("how long is the reverse gear warranty?", "reverse"),
]

#: the words the everyday messages are written around, for the teacher
EVERYDAY = ("bug", "error", "file", "function", "call", "run", "program",
            "project", "review", "code", "compile", "branch", "test",
            "script", "debug", "commit", "class", "method")


# -- what is real: names, requests, examples ----------------------------------------

def _held(text: str) -> bool:
    return hashlib.sha1(text.encode()).digest()[0] < 256 * VALID


def real() -> dict:
    """Function names, calls and examples, requests (MBPP's), file paths
    (v696's projects'), and the teacher's lists (`lists`)."""
    from research.v696 import tasks
    from research.v696.meaning import test_pairs
    from research.v696.teach_meaning import _english, split
    functions, calls, examples, requests = set(), [], [], []
    for task in tasks.load("mbpp-ts"):
        functions.add(task.entry)
        for args, value in test_pairs(task.tests)[:3]:
            calls.append((task.entry, args))
            examples.append(f"{task.entry}({args}) == {value}")
        if split(task.name) == "train":
            english = _english(task.prompt)
            if english:
                requests.append((task.entry, english))
    files = set()
    for part in ("dev", "held"):
        path = ROOT / "data" / "code-meaning" / f"projects-{part}.jsonl"
        if path.exists():
            for line in path.open(encoding="utf-8"):
                for one in json.loads(line)["files"]:
                    files.add(one.lstrip("/").replace("p/", "src/", 1))
    rng = random.Random(SEED)
    for name in sorted(functions):
        if rng.random() < 0.25:
            folder = rng.choice(("src", "lib", "src/utils", "app",
                                 "packages/core/src"))
            files.add(f"{folder}/{name}.{rng.choice(('ts', 'js', 'tsx'))}")
    lists = {"project": [], "names": [], "concepts": [], "functions": []}
    if LISTS.exists():
        for line in LISTS.open(encoding="utf-8"):
            row = json.loads(line)
            if "list" in row["meta"]:
                lists[row["meta"]["list"]] += row["lines"]
    concepts = []
    for line in lists["concepts"]:
        concept, _, members = line.partition(":")
        members = [one.strip() for one in members.split(",") if one.strip()]
        if concept.strip() and 2 <= len(members) <= 8 and \
                len(concept.split()) <= 3:
            concepts.append((concept.strip().lower(), members))
    # names as people write them: MBPP's in camelCase, and the teacher's
    # list -- a name that is an English word (`greeting`) is a name too
    written = sorted({_camel(one) for one in functions} | {
        one.strip() for one in lists["functions"]
        if re.fullmatch(r"[A-Za-z_$][\w$]{1,40}", one.strip())})
    return {"functions": sorted(functions), "files": sorted(files),
            "calls": calls, "examples": examples, "requests": requests,
            "concepts": concepts, "names": written}


# -- what the teacher is asked to write ------------------------------------------------

def jobs() -> list:
    """(key, prompt, meta): each label written from `EACH` prompts, each
    naming a different real function or file."""
    found = real()
    rng = random.Random(SEED)
    out = []
    for subject, aspects in CAN.items():
        for aspect in aspects:
            for at in range(EACH):
                meta = {"act": "ask", "aspect": aspect, "subject": subject}
                ask = (f"Write 12 different messages a programmer might type "
                       f"to a coding assistant asking {ASKS[aspect]}, ")
                if subject == "named":
                    name = rng.choice(found["functions"])
                    meta["name"] = name
                    ask += (f"about their function {name}. Write {name} "
                            f"exactly so in every message. ")
                elif subject == "file":
                    name = rng.choice(found["files"])
                    meta["name"] = name
                    ask += (f"about their source file {name}. Write {name} "
                            f"exactly so in every message. ")
                elif subject == "project":
                    ask += ("about their whole software project (the "
                            "repository, the codebase, the app). Name no "
                            "file or function. ")
                else:
                    ask += ("about the code they were just discussing with "
                            "the assistant. Do not name it: call it it, "
                            "that, this, the code, the function, that one. ")
                out.append((f"ask|{aspect}|{subject}|{at}", ask + STYLE, meta))
    for at, (entry, english) in enumerate(rng.sample(found["requests"],
                                                      EACH * 3)):
        out.append((f"make|{at}", f"Write 12 different ways a person might "
                    f"ask a coding assistant for this: \"{english}\" "
                    + STYLE, {"act": "make", "aspect": "none",
                              "subject": "none"}))
        out.append((f"change|{at}", f"A coding assistant has just written a "
                    f"function for this request: \"{english}\". Write 12 "
                    f"different messages the person might type next to "
                    f"correct it or add a requirement (it should also..., "
                    f"make it..., it must...), without naming the function. "
                    + STYLE, {"act": "change", "aspect": "none",
                              "subject": "last"}))
    for at, (entry, args) in enumerate(rng.sample(found["calls"], EACH * 2)):
        out.append((f"run|{at}", f"Write 12 different messages asking a "
                    f"coding assistant to run its function {entry} on the "
                    f"input {args} and say what it gives. Write {entry} and "
                    f"{args} exactly so in every message. " + STYLE,
                    {"act": "run", "aspect": "none", "subject": "named",
                     "name": entry, "args": args}))
    for at, (concept, members) in enumerate(
            rng.sample(found["concepts"], min(EACH * 3,
                                              len(found["concepts"])))):
        out.append((f"teach|{at}", f"Write 12 different messages a person "
                    f"might type to teach an assistant that a {concept} is "
                    f"one of: {', '.join(members)}. Write the word {concept} "
                    f"and every one of {', '.join(members)} exactly so in "
                    f"every message. " + STYLE,
                    {"act": "teach", "aspect": "none", "subject": "none",
                     "concept": concept, "members": members}))
    # and short: people type `who calls greeting`, `is label buggy`; the
    # teacher, asked for messages, writes sentences
    for subject, aspects in CAN.items():
        for aspect in aspects:
            meta = {"act": "ask", "aspect": aspect, "subject": subject,
                    "short": True}
            ask = (f"Write 20 very short messages (2 to 6 words, the way a "
                   f"busy programmer types) to a coding assistant asking "
                   f"{ASKS[aspect]}, ")
            if subject in ("named", "file"):
                name = rng.choice(found["functions"] if subject == "named"
                                  else found["files"])
                meta["name"] = name
                ask += (f"about {'their function' if subject == 'named' else 'their file'}"
                        f" {name}. Write {name} exactly so in every message. ")
            elif subject == "project":
                ask += "about their whole project. Name no file or function. "
            else:
                ask += ("about the code just discussed, calling it it, that "
                        "or this. ")
            out.append((f"short|{aspect}|{subject}", ask + STYLE, meta))
    # questions said again: seed questions -- short ones among them, `who
    # calls greeting?` -- paraphrased by the teacher, the name kept; a small
    # teacher paraphrases faithfully and writes `who calls` and `what does
    # it call` into one muddle when left to itself
    for (aspect, subject), seeds in SEEDS.items():
        for at, seed in enumerate(seeds):
            name = None
            if subject == "named":
                name = rng.choice(found["functions"])
            elif subject == "file":
                name = rng.choice(found["files"])
            said = seed.format(n=name or "")
            meta = {"act": "ask", "aspect": aspect, "subject": subject,
                    "paraphrase": True, "seed": said}
            if name:
                meta["name"] = name
            keep = f" Keep {name} exactly so in every one." if name else ""
            out.append((f"para|{aspect}|{subject}|{at}",
                        f"Say this question in 12 different ways, as a "
                        f"programmer would type it to a coding assistant, "
                        f"short ones too: \"{said}\".{keep} " + STYLE, meta))
    for at, (seed, word) in enumerate(EVERYDAY_SEEDS):
        out.append((f"para|none|{at}", f"Say this in 12 different ways, "
                    f"keeping its everyday meaning: \"{seed}\". Keep the word "
                    f"{word} in every one. " + STYLE,
                    {"act": "none", "aspect": "none", "subject": "none",
                     "paraphrase": True, "seed": seed, "word": word}))
    # teaching, said again: a small teacher paraphrases a sentence that
    # teaches far better than it writes one (its own read like `a luxurious
    # attribute is often sought after`); the words kept exactly are the check
    for at, (concept, members) in enumerate(found["concepts"]):
        listed = ", ".join(members[:-1]) + f" and {members[-1]}"
        out.append((f"teach2|{at}", f"Say this in 12 different ways, as a "
                    f"person teaching an assistant what the word means: "
                    f"\"a {concept} is one of {listed}\". Keep the word "
                    f"{concept} and each of {', '.join(members)} exactly so "
                    f"in every one. " + STYLE,
                    {"act": "teach", "aspect": "none", "subject": "none",
                     "concept": concept, "members": members,
                     "paraphrase": True}))
    for word in EVERYDAY:
        out.append((f"none|{word}", f"Write 12 different everyday messages "
                    f"that have nothing to do with software, using the word "
                    f"'{word}' in an ordinary, non-computing sense. " + STYLE,
                    {"act": "none", "aspect": "none", "subject": "none"}))
    return out


LIST_JOBS = (
    ("list|project-phrases", "List 40 different ways a programmer refers "
     "to the whole software project they are working in (like: this repo, "
     "the codebase, my project, our app). One per line, nothing else.",
     {"list": "project"}),
    ("list|project-names", "List 40 different plausible names of software "
     "repositories, in kebab-case or lowercase (like hello-world, stark-db, "
     "invoice-tool). One per line, nothing else.", {"list": "names"}),
    ("list|concepts", "List 40 small categories with their members, one per "
     "line, as: category: member, member, member (like 'vowel: a, e, i, o, "
     "u' or 'primary colour: red, yellow, blue'). Nothing else.",
     {"list": "concepts"}),
    ("list|function-names", "List 60 different function names as they are "
     "written in TypeScript and JavaScript projects: camelCase like "
     "parseConfig or getUserById, and single words like render, greeting, "
     "main, init, validate. One per line, nothing else.",
     {"list": "functions"}),
)


def _camel(name: str) -> str:
    """`remove_kth_element` as people write it in TypeScript:
    `removeKthElement`."""
    parts = [one for one in name.split("_") if one]
    return parts[0].lower() + "".join(one[:1].upper() + one[1:].lower()
                                      for one in parts[1:]) if parts else name


def _lines(reply: str) -> list:
    out = []
    for line in reply.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
        line = line.strip("\"'“”")
        if 3 <= len(line) <= 220 and not line.lower().startswith(
                ("here are", "sure", "certainly")):
            out.append(line)
    return out


def _write(todo: list, path: Path, samples: int, batch: int) -> None:
    from research.v696.teach_meaning import Teacher
    if not todo:
        return
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write([prompt for _, prompt, _ in chunk],
                                longest=700, samples=samples,
                                temperature=0.9)
        with path.open("a", encoding="utf-8") as out:
            for (key, _, meta), written in zip(chunk, replies):
                lines = [line for reply in written for line in _lines(reply)]
                out.write(json.dumps({"key": key, "meta": meta,
                                      "lines": lines}) + "\n")
                print(f"  {key}: {len(lines)} lines", flush=True)


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.open(encoding="utf-8")}


def lists() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    _write([one for one in LIST_JOBS if one[0] not in _done(LISTS)], LISTS,
           samples=4, batch=3)


def write(samples: int = 2, batch: int = 6) -> None:
    """Each job written `samples` times, kept as it goes: a run stopped
    part-way goes on from where it was."""
    DATA.mkdir(parents=True, exist_ok=True)
    todo = [one for one in jobs() if one[0] not in _done(WRITTEN)]
    print(f"{len(todo)} jobs to write", flush=True)
    _write(todo, WRITTEN, samples, batch)


# -- the teacher checks what it wrote, by choosing --------------------------------

ACT_CHOICES = (
    ("ask", "asking a question about existing code, a function, a file or "
            "a software project"),
    ("make", "asking for new code to be written"),
    ("change", "asking to change or correct code that was just written"),
    ("run", "asking to run a function on an input"),
    ("teach", "telling what a word means by listing what belongs to it"),
    ("none", "something not about software or code at all"),
)
LETTERS = "ABCDEFGHIJKLMNOP"


def _choice(line: str, options: list, what: str) -> str:
    listed = "\n".join(f"{LETTERS[at]}) {said}" for at, (_, said)
                       in enumerate(options))
    return (f'Someone typed this message: "{line}"\n{what} Choose one:\n'
            f"{listed}\nAnswer with the letter only.")


def _picked(reply: str, options: list) -> str | None:
    found = re.search(r"\b([A-P])\b", reply.strip().upper())
    if not found:
        return None
    at = LETTERS.index(found.group(1))
    return options[at][0] if at < len(options) else None


def check(batch: int = 24) -> dict:
    """Each message put back to the teacher as a choice of what it does --
    and, for a question, of what it asks; kept where both choices are what
    it was written as. Kept as it goes."""
    from research.v696.teach_meaning import Teacher
    done = set()
    if CHECKED.exists():
        done = {(row["key"], row["line"]) for row in
                map(json.loads, CHECKED.open(encoding="utf-8"))}
    todo = []
    for row in map(json.loads, WRITTEN.open(encoding="utf-8")):
        for line in dict.fromkeys(row["lines"]):
            if (row["key"], line) not in done:
                todo.append((row["key"], row["meta"], line))
    print(f"{len(todo)} messages to check", flush=True)
    if not todo:
        return {}
    teacher = Teacher()
    aspects = [(one, ASKS[one]) for one in ASPECTS if one != "none"]
    kept = total = 0
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        acts = teacher.write([_choice(line, list(ACT_CHOICES),
                                      "What is this message doing?")
                              for _, _, line in chunk], longest=3)
        asks = teacher.write([_choice(line, aspects + [
            ("none", "none of these")], "What is the person asking about?")
            for _, _, line in chunk], longest=3)
        with CHECKED.open("a", encoding="utf-8") as out:
            for (key, meta, line), act, ask in zip(chunk, acts, asks):
                act = _picked(act[0], list(ACT_CHOICES))
                ask = _picked(ask[0], aspects + [("none", "")])
                keep = act == meta["act"] and (
                    meta["act"] != "ask" or ask == meta["aspect"])
                kept += keep
                total += 1
                out.write(json.dumps({"key": key, "line": line, "act": act,
                                      "aspect": ask, "keep": keep}) + "\n")
        print(f"  {min(at + batch, len(todo))}/{len(todo)}: kept {kept} of "
              f"{total}", flush=True)
    return {"kept": kept, "checked": total}


# -- the corpus ---------------------------------------------------------------------

def words(text: str) -> list:
    from research.v692.corpus import words as said
    return said(text)


def _match(said: list, phrase: str, start: int = 0) -> tuple | None:
    """Where a phrase's words are in a message's words: (begin, end). A
    word written against the phrase is still it: `remove_kth_element()`,
    `greeting's`, `vowels`."""
    wanted = words(phrase)
    if not wanted:
        return None
    for at in range(start, len(said) - len(wanted) + 1):
        window = said[at:at + len(wanted)]
        if window[:-1] == wanted[:-1] and (
                window[-1] == wanted[-1]
                or (window[-1].startswith(wanted[-1]) and not
                    window[-1][len(wanted[-1]):][:1].isalnum())
                or window[-1] in (wanted[-1] + "s", wanted[-1] + "es")):
            return at, at + len(wanted)
    return None


def _mark(roles: list, span: tuple, name: str) -> None:
    roles[span[0]] = f"B-{name}"
    for at in range(span[0] + 1, span[1]):
        roles[at] = f"I-{name}"


def labelled(line: str, meta: dict) -> tuple | None:
    """(words, roles) of a written message, its spans found as they were
    given; None where one is not there."""
    said = words(line)
    if not said:
        return None
    roles = ["O"] * len(said)
    if meta.get("name"):
        span = _match(said, meta["name"])
        if span is None:
            return None
        _mark(roles, span, "SUBJ")
    if meta.get("concept"):
        span = _match(said, meta["concept"])
        if span is None:
            return None
        _mark(roles, span, "CONCEPT")
        # each member after the one before -- and the first after the
        # concept, or `a` would be the article of `a vowel is one of a, e`
        start = span[1] if _match(said, meta["members"][0], span[1]) \
            else 0
        for member in meta["members"]:
            where = _match(said, member, start)
            if where is None or any(roles[at] != "O" for at in
                                    range(*where)):
                return None
            _mark(roles, where, "MEMBER")
            start = where[1]
    return said, roles


def _swapped(said: list, roles: list, old: str, new: str) -> list | None:
    """The words with the function named swapped for another name: what is
    written against it kept (`rev("ab")` -> `render("ab")`). The labels are
    the same: the name is where it was."""
    if "B-SUBJ" not in roles:
        return None
    at = roles.index("B-SUBJ")
    if at + 1 < len(roles) and roles[at + 1] == "I-SUBJ":
        return None
    token, old = said[at], words(old)[0] if words(old) else old.lower()
    if not token.startswith(old):
        return None
    out = list(said)
    out[at] = new.lower() + token[len(old):]
    return out


def record(words_, roles, act, aspect, subject, source) -> dict:
    return {"task": "code", "words": words_, "roles": roles, "act": act,
            "aspect": aspect, "subject": subject,
            "tags": [""] * len(words_), "deps": [""] * len(words_),
            "names": [], "said": " ".join(words_), "source": source,
            "how": source}


def corpus(negatives: int = 4000) -> dict:
    """`train-code.jsonl` and `valid-code.jsonl`: every message the teacher
    wrote and kept, the real requests, examples and calls, and what is not
    code talk -- held by message and by name."""
    rng = random.Random(SEED)
    found = real()
    kept, picked = {}, {}
    if CHECKED.exists():
        for row in map(json.loads, CHECKED.open(encoding="utf-8")):
            kept[(row["key"], row["line"])] = row["keep"]
            picked[(row["key"], row["line"])] = row["act"]
    rows = {"train": [], "valid": []}

    def put(rec: dict, held: bool) -> None:
        # What is known of each word, beside it, as spaCy's tag is: a word
        # that names something of the project or the conversation's code is
        # `CODE` (`reading.read` looks it up). Taught most of the time, not
        # always -- a message is read without a project too -- and given to
        # a word of what is not code talk now and then, so that the tag
        # alone decides nothing.
        if rec["act"] != "none":
            if "B-SUBJ" in rec["roles"] and rec["subject"] in (
                    "named", "file") and rng.random() < KNOWN:
                rec["tags"] = ["CODE" if role.endswith("SUBJ") else ""
                               for role in rec["roles"]]
        else:
            # as at run time: a word of what is not code talk that is also
            # some function's name (`main`, `label`, `render`) is marked
            # where a project has it -- here, as often as names are
            names_here = lowered | set(rec.pop("known", ()))
            marked = [("CODE" if one in names_here and rng.random() < KNOWN
                       else "") for one in rec["words"]]
            if any(marked):
                rec["tags"] = marked
            elif rec["words"] and rng.random() < NOISE:
                at = rng.randrange(len(rec["words"]))
                rec["tags"] = ["CODE" if one == at else ""
                               for one in range(len(rec["words"]))]
        rows["valid" if held else "train"].append(rec)

    names = found["names"]
    lowered = {one.lower() for one in names}

    def swap(made, old: str, held: bool, act: str, aspect: str,
             source: str, times: int = 2) -> None:
        """The same message naming functions of other shapes -- a name in
        camelCase, a name that is an English word -- held where either
        name is."""
        said, roles = made
        for new in rng.sample(names, min(times, len(names))):
            swapped = _swapped(said, roles, old, new)
            if swapped is not None:
                put(record(swapped, roles, act, aspect, "named",
                           source + " renamed"),
                    held or _held("name|" + new))

    def keeps(key: str, line: str, meta: dict) -> bool:
        if meta.get("paraphrase") and meta.get("act") == "ask" \
                and (key, line) in kept and line != meta.get("seed"):
            # a question said again can drift (`what does it use` became
            # `function of positivecount`): where the teacher checked it,
            # what it chose it asks must be what it was said as
            return bool(kept[(key, line)])
        if meta.get("paraphrase"):
            # said again, its words kept exactly: `labelled` is the check
            return True
        if meta["act"] == "change":
            # the checking teacher takes a correction for a question about
            # the code as often as not; it is code talk either way, and the
            # message was written as a correction
            return picked.get((key, line)) in ("change", "ask")
        return bool(kept.get((key, line)))

    for row in map(json.loads, WRITTEN.open(encoding="utf-8")):
        meta = row["meta"]
        # a seed is taught beside what was said again of it
        lines = row["lines"] + ([meta["seed"]] if meta.get("seed") else [])
        for line in dict.fromkeys(lines):
            if not keeps(row["key"], line, meta):
                continue
            made = labelled(line, meta)
            if made is None:
                continue
            held = _held(line) or (meta.get("name") is not None
                                   and _held("name|" + meta["name"]))
            rec = record(*made, meta["act"], meta["aspect"], meta["subject"],
                         row["key"])
            if meta.get("word"):
                # the everyday word that is also a function's name: marked
                # as it would be where a project has that function
                rec["known"] = [meta["word"]]
            put(rec, held)
            if meta.get("subject") == "named" and meta.get("name"):
                swap(made, meta["name"], held, meta["act"], meta["aspect"],
                     row["key"])
    for entry, english in found["requests"]:
        said = words(english)
        put(record(said, ["O"] * len(said), "make", "none", "none",
                   "mbpp request"), _held(english))
    for text in rng.sample(found["examples"], min(900, len(found["examples"]))):
        said = words(text)
        roles = ["B-SUBJ"] + ["O"] * (len(said) - 1)
        put(record(said, roles, "change", "none", "named", "mbpp example"),
            _held(text))
        swap((said, roles), text.split("(")[0], _held(text), "change",
             "none", "mbpp example")
    for entry, args in rng.sample(found["calls"], min(500, len(found["calls"]))):
        said = words(f"{entry}({args})")
        roles = ["B-SUBJ"] + ["O"] * (len(said) - 1)
        put(record(said, roles, "run", "none", "named", "mbpp call"),
            _held(entry + args))
        swap((said, roles), entry, _held(entry + args), "run", "none",
             "mbpp call")
    corpus_rows = []
    for path in sorted((LLM / "reader-data").glob("train*.jsonl")):
        for line in path.open(encoding="utf-8"):
            one = json.loads(line)
            if one.get("words"):
                corpus_rows.append(one["words"])
    for path in sorted((LLM / "math-data").glob("train-math.jsonl")):
        for line in path.open(encoding="utf-8"):
            one = json.loads(line)
            if one.get("words") and rng.random() < 0.08:
                corpus_rows.append(one["words"])
    for said in rng.sample(corpus_rows, min(negatives, len(corpus_rows))):
        said = [str(one).lower() for one in said]
        put(record(said, ["O"] * len(said), "none", "none", "none",
                   "not code"), _held(" ".join(said)))
    DATA.mkdir(parents=True, exist_ok=True)
    stats = {}
    for part, made in rows.items():
        rng.shuffle(made)
        with (DATA / f"{part}-code.jsonl").open("w", encoding="utf-8") as out:
            for one in made:
                out.write(json.dumps(one) + "\n")
        count = {}
        for one in made:
            key = f"{one['act']}|{one['aspect']}|{one['subject']}"
            count[key] = count.get(key, 0) + 1
        stats[part] = {"records": len(made),
                       "by label": dict(sorted(count.items()))}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1),
                                     encoding="utf-8")
    print({part: one["records"] for part, one in stats.items()})
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("lists", "write", "check", "corpus"))
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)
    if args.job == "lists":
        lists()
    elif args.job == "write":
        write(samples=args.samples)
    elif args.job == "check":
        print(check())
    else:
        corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())

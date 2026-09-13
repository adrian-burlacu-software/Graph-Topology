"""bAbI: twenty toy tasks from a published source, put to v689 and to SmolLM3.

Weston et al.'s tasks (`data/babi.SOURCE.md`) are what v689 is for: short
stories about who went where, who carried what and what happened before
what, then a question with a one-word answer. They were written by someone
else, so a score on them can be set beside another system's.

    python -m research.v689.babi v689 --split test --processes 12
    python -m research.v689.babi smollm3 --split test
    python -m research.v689.babi report --split test

## How each system is asked

**v689** hears the story one line at a time, in a conversation of its own,
questions included, exactly as the page would get it: `Session.say`, with v688
behind it and no teacher, so the graph is measured alone. Each story is a
fresh conversation whose taught knowledge is its own, and nothing is kept on
disk.

**SmolLM3-3B** is given every statement before the question, the question,
and the form the answer takes (one word; yes or no; a count as a word; a list
with commas; directions as n, s, e, w). No examples, no thinking, greedy.

## How an answer is scored

Both replies go through the same reader, `predict`, so neither is scored on
friendlier terms. A reply is prose for v689 and nearly always one word for
the model; what is read out of it is:

    yes/no      the first word, if it is yes or no (or maybe, for qa10)
    count       the first number word
    list        every word of the answer space in the reply, as a set;
                `nothing` when none is and the reply says nothing or none
    path        every direction in order
    word        the one word of the answer space in the reply that is not
                in the question; none, or more than one, is no answer, and
                so is a listing (two commas or more). A short reply that
                only repeats a word of the question is that word, and wrong

The answer space is every answer in the task's training and validation
splits, never the test's. A reply with no answer in it is counted apart from
a wrong one: `right`, `wrong` and `none` add up to every question, and
accuracy is `right` over all of them.

Raw replies are kept (`babi-out/`, not committed) and `report` reads them
again, so a change to the scorer does not need a rerun.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA = REPOSITORY_ROOT / "data" / "babi" / "tasks_1-20_v1-2" / "en-valid"
OUT = Path(__file__).resolve().parent / "babi-out"

TASKS = {1: "single supporting fact", 2: "two supporting facts",
         3: "three supporting facts", 4: "two argument relations",
         5: "three argument relations", 6: "yes/no questions",
         7: "counting", 8: "lists/sets", 9: "simple negation",
         10: "indefinite knowledge", 11: "basic coreference",
         12: "conjunction", 13: "compound coreference",
         14: "time reasoning", 15: "basic deduction", 16: "basic induction",
         17: "positional reasoning", 18: "size reasoning",
         19: "path finding", 20: "agent's motivations"}

#: The form a task's answers take. Everything not named is one word.
KINDS = {6: "yesno", 9: "yesno", 17: "yesno", 18: "yesno", 10: "maybe",
         7: "count", 8: "list", 19: "path"}

#: What the model is told about the form, and nothing about the content.
FORMATS = {
    "yesno": "Answer yes or no, one word only.",
    "maybe": "Answer yes, no or maybe, one word only.",
    "count": "Answer with how many, written as a word (none, one, two, ...), "
             "one word only.",
    "list": "Answer with the things, separated by commas, or nothing if "
            "there are none. No other words.",
    "path": "Answer with the directions in order, separated by commas, using "
            "n, s, e and w. No other words.",
    "word": "Answer with one word only.",
}

PROMPT = "{story}\n\nQuestion: {question}\n{format}"

NUMBERS = {"none": "none", "zero": "none", "nothing": "none", "0": "none",
           "one": "one", "1": "one", "two": "two", "2": "two",
           "three": "three", "3": "three", "four": "four", "4": "four",
           "five": "five", "5": "five", "six": "six", "6": "six",
           "seven": "seven", "7": "seven", "eight": "eight", "8": "eight",
           "nine": "nine", "9": "nine", "ten": "ten", "10": "ten"}

DIRECTIONS = {"n": "n", "north": "n", "s": "s", "south": "s", "e": "e",
              "east": "e", "w": "w", "west": "w"}

#: `dog's` is one word, so its `s` is never a direction.
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Question:
    answer: str
    supports: tuple


@dataclass(frozen=True)
class Line:
    number: int
    text: str
    question: Question | None = None


@dataclass(frozen=True)
class Story:
    task: int
    index: int
    lines: tuple

    def before(self, at: int) -> str:
        """The statements told before line `at`, questions left out."""
        return "\n".join(line.text for line in self.lines[:at]
                         if line.question is None)


def kind_of(task: int) -> str:
    return KINDS.get(task, "word")


def parse(text: str, task: int = 0) -> list[Story]:
    """A bAbI file's stories. A line numbered 1 starts a new one."""
    stories: list[list[Line]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        number, _, rest = raw.partition(" ")
        if number == "1" or not stories:
            stories.append([])
        if "\t" in rest:
            said, answer, supports = (rest.split("\t") + ["", ""])[:3]
            question = Question(answer.strip(), tuple(
                int(one) for one in supports.split()))
            stories[-1].append(Line(int(number), said.strip(), question))
        else:
            stories[-1].append(Line(int(number), rest.strip()))
    return [Story(task, index, tuple(lines))
            for index, lines in enumerate(stories)]


@functools.lru_cache(maxsize=None)
def stories(task: int, split: str = "test",
            folder: Path = DATA) -> tuple[Story, ...]:
    path = folder / f"qa{task}_{split}.txt"
    return tuple(parse(path.read_text(encoding="utf-8"), task))


def words(text: str) -> list[str]:
    return WORD.findall((text or "").lower())


@functools.lru_cache(maxsize=None)
def base(word: str) -> str:
    """A noun's singular, so `wolves` answers `wolf`."""
    try:
        from nltk.corpus import wordnet
        found = wordnet.morphy(word, wordnet.NOUN)
    except Exception:                               # noqa: BLE001
        found = None
    return found or word


@functools.lru_cache(maxsize=None)
def vocabulary(task: int, folder: Path = DATA) -> frozenset:
    """Every answer the task's training and validation splits give."""
    found: set[str] = set()
    for split in ("train", "valid"):
        for story in stories(task, split, folder):
            for line in story.lines:
                if line.question:
                    found.update(base(one) for one in
                                 line.question.answer.lower().split(","))
    return frozenset(found)


def gold(kind: str, answer: str) -> str:
    answer = answer.strip().lower()
    if kind == "list":
        return ",".join(sorted(base(one) for one in answer.split(",")))
    return answer if kind in ("yesno", "maybe", "count", "path") \
        else base(answer)


def headline(reply: str) -> str:
    """What a reply answers, without what it rests on: v689 says the answer
    and then, after a dash, why -- `Fred — the last of two givings, “Bill
    gave the football to Fred”` -- and the quote names everyone in it. The
    model says the answer on its first line and sometimes goes on to explain
    it (`Garden\\n\\nStep 1: ... the bedroom`), so only the first line counts
    for either. A one-word reply is its own headline."""
    return (reply or "").strip().split("\n", 1)[0].split(" — ", 1)[0]


def predict(kind: str, reply: str, question: str,
            space: frozenset) -> str | None:
    """The answer read out of a reply's headline, or None when there is
    none in it."""
    reply = headline(reply)
    tokens = words(reply)
    if kind in ("yesno", "maybe"):
        allowed = ("yes", "no", "maybe") if kind == "maybe" else ("yes", "no")
        return tokens[0] if tokens and tokens[0] in allowed else None
    if kind == "count":
        return next((NUMBERS[one] for one in tokens if one in NUMBERS), None)
    if kind == "path":
        steps = [DIRECTIONS[one] for one in tokens if one in DIRECTIONS]
        return ",".join(steps) or None
    asked = {base(one) for one in words(question)}
    found: list[str] = []
    echoed: list[str] = []
    for token in tokens:
        one = base(token)
        if one in space:
            into = echoed if one in asked else found
            if one not in into:
                into.append(one)
    if kind == "list":
        found = [one for one in found if one != "nothing"]
        if found:
            return ",".join(sorted(found))
        return "nothing" if {"nothing", "none"} & set(tokens) else None
    # A listing is not an answer, whatever happens to be in it: `mary is
    # found in the hospital, prison, the kitchen, ...` does not say kitchen.
    if reply.count(",") >= 2:
        return None
    # A word from the question is an answer only when it is the only one
    # given: `bedroom` to `where was Julie before the bedroom` is wrong, not
    # silent.
    if not found and len(echoed) == 1 and len(tokens) <= 3:
        return echoed[0]
    return found[0] if len(found) == 1 else None


def outcome(predicted: str | None, expected: str) -> str:
    if predicted is None:
        return "none"
    return "right" if predicted == expected else "wrong"


def score(record: dict) -> dict:
    """A kept reply, read again with the scorer as it is now."""
    kind = kind_of(record["task"])
    predicted = predict(kind, record["reply"], record["question"],
                        vocabulary(record["task"]))
    return {**record, "predicted": predicted,
            "outcome": outcome(predicted, gold(kind, record["gold"]))}


def key(record: dict) -> str:
    return f"{record['task']}:{record['story']}:{record['line']}"


def prompt(story: Story, at: int) -> str:
    line = story.lines[at]
    return PROMPT.format(story=story.before(at), question=line.text,
                         format=FORMATS[kind_of(story.task)])


def _record(system: str, split: str, story: Story, at: int, reply: str,
            **extra) -> dict:
    line = story.lines[at]
    record = {"system": system, "split": split, "task": story.task,
              "story": story.index, "line": at, "question": line.text,
              "gold": line.question.answer, "reply": reply, **extra}
    return score(record)


def _kept(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    kept = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            record = json.loads(raw)
            kept[key(record)] = record
    return kept


def _tasks(option: str) -> list[int]:
    if not option:
        return list(TASKS)
    return [int(one) for one in option.split(",")]


# -- v689 ------------------------------------------------------------------
_ASKER = None


def _start(store: str, cycles: int) -> None:
    global _ASKER
    from research.v688 import server as v688
    from research.v689.server import StoreAsker

    service = v688.Service(Path(store), 1, cycles, teacher=False)
    _ASKER = StoreAsker(service)


def tell(asker, story: Story, split: str) -> list[dict]:
    """One story, said to a fresh conversation a line at a time."""
    from research.v689.session import Session

    session = Session(asker, conversation=f"babi-qa{story.task}-{story.index}",
                      example=True)
    records = []
    for at, line in enumerate(story.lines):
        started = time.time()
        try:
            turn = session.say(line.text)
            reply = (turn.answer or {}).get("text") or ""
            act = turn.act
        except Exception as bad:                    # noqa: BLE001
            reply, act = f"error: {type(bad).__name__}: {bad}", "error"
        if line.question is not None:
            records.append(_record("v689", split, story, at, reply, act=act,
                                   seconds=round(time.time() - started, 3)))
    return records


def _tell(job: tuple) -> list[dict]:
    task, split, index = job
    return tell(_ASKER, stories(task, split)[index], split)


def run_v689(split: str, tasks: list[int], processes: int, limit: int,
             store: Path, cycles: int) -> Path:
    import multiprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"v689-{split}.jsonl"
    kept = _kept(path)
    told = {(record["task"], record["story"]) for record in kept.values()}
    by_task = [[(task, split, story.index)
                for story in stories(task, split)[:limit or None]
                if (task, story.index) not in told] for task in tasks]
    # Round robin, so the long stories of qa3 do not all land at the end.
    jobs = [job for group in _interleave(by_task) for job in group]
    print(f"v689 on {split}: {len(jobs)} stories to tell, {len(told)} kept")
    started, done, right, asked = time.time(), 0, 0, 0
    with path.open("a", encoding="utf-8") as sink, multiprocessing.Pool(
            processes, initializer=_start,
            initargs=(str(store), cycles)) as pool:
        for records in pool.imap_unordered(_tell, jobs):
            for record in records:
                sink.write(json.dumps(record) + "\n")
                right += record["outcome"] == "right"
                asked += 1
            sink.flush()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} stories, {asked} questions, "
                      f"{right / max(asked, 1):.1%} right, "
                      f"{time.time() - started:.0f}s", flush=True)
    return path


def _interleave(groups: list[list]) -> list[list]:
    out: list[list] = []
    longest = max((len(group) for group in groups), default=0)
    for at in range(longest):
        out.append([group[at] for group in groups if at < len(group)])
    return out


# -- SmolLM3 ---------------------------------------------------------------
def run_smollm3(split: str, tasks: list[int], limit: int,
                batch_tokens: int, new_tokens: int) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from research.v688.teacher import MODEL

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"smollm3-{split}.jsonl"
    kept = _kept(path)
    items = [(story, at) for task in tasks
             for story in stories(task, split)[:limit or None]
             for at, line in enumerate(story.lines)
             if line.question is not None and f"{task}:{story.index}:{at}"
             not in kept]
    print(f"SmolLM3 on {split}: {len(items)} questions to ask, "
          f"{len(kept)} kept")
    if not items:
        return path
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float16, device_map="cuda")
    model.eval()
    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt(story, at)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
        for story, at in items]
    lengths = [len(ids) for ids in tokenizer(
        texts, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(items)), key=lambda one: lengths[one])
    batches, current = [], []
    for one in order:
        widest = max([lengths[one]] + [lengths[other] for other in current])
        if current and (len(current) + 1) * (widest + new_tokens) \
                > batch_tokens:
            batches.append(current)
            current = []
        current.append(one)
    if current:
        batches.append(current)
    started, asked, right = time.time(), 0, 0
    with path.open("a", encoding="utf-8") as sink:
        for number, batch in enumerate(batches, 1):
            encoded = tokenizer([texts[one] for one in batch],
                                add_special_tokens=False, padding=True,
                                return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated = model.generate(
                    **encoded, max_new_tokens=new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id)
            replies = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:],
                skip_special_tokens=True)
            for one, reply in zip(batch, replies):
                story, at = items[one]
                record = _record("smollm3", split, story, at, reply.strip(),
                                 tokens=lengths[one])
                sink.write(json.dumps(record) + "\n")
                right += record["outcome"] == "right"
                asked += 1
            sink.flush()
            if number % 25 == 0 or number == len(batches):
                print(f"  {asked}/{len(items)} questions, "
                      f"{right / max(asked, 1):.1%} right, "
                      f"{time.time() - started:.0f}s", flush=True)
    return path


# -- the report ------------------------------------------------------------
SYSTEMS = ("v689", "smollm3")


def report(split: str, systems=SYSTEMS, shared: bool = True) -> dict:
    """Per task and overall, each system's right / wrong / none, read again
    from the kept replies. With `shared`, only questions every system with
    results was asked, so a partial run is compared like with like."""
    results = {system: {name: score(record) for name, record in
                        _kept(OUT / f"{system}-{split}.jsonl").items()}
               for system in systems}
    results = {system: found for system, found in results.items() if found}
    keys = None
    for found in results.values():
        keys = set(found) if keys is None else keys & set(found)
    table: dict = {}
    for system, found in results.items():
        for name, record in found.items():
            if shared and name not in keys:
                continue
            row = table.setdefault(record["task"], {}).setdefault(
                system, {"right": 0, "wrong": 0, "none": 0})
            row[record["outcome"]] += 1
    return {"split": split, "systems": list(results), "tasks": table}


def render(summary: dict) -> str:
    systems = summary["systems"]
    head = "| task | questions | " + " | ".join(
        f"{system} right | {system} wrong | {system} none"
        for system in systems) + " |"
    lines = [head, "|" + "---|" * (2 + 3 * len(systems))]
    totals = {system: {"right": 0, "wrong": 0, "none": 0}
              for system in systems}
    for task in sorted(summary["tasks"]):
        row = summary["tasks"][task]
        count = sum(next(iter(row.values())).values())
        cells = []
        for system in systems:
            found = row.get(system, {"right": 0, "wrong": 0, "none": 0})
            asked = max(sum(found.values()), 1)
            for name in ("right", "wrong", "none"):
                totals[system][name] += found[name]
                cells.append(f"{found[name] / asked:.1%}")
        lines.append(f"| {task} {TASKS[task]} | {count} | "
                     + " | ".join(cells) + " |")
    cells = []
    for system in systems:
        asked = max(sum(totals[system].values()), 1)
        cells.extend(f"{totals[system][name] / asked:.1%}"
                     for name in ("right", "wrong", "none"))
    count = sum(totals[systems[0]].values()) if systems else 0
    lines.append(f"| **all** | {count} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("v689", "smollm3", "report"))
    parser.add_argument("--split", default="test",
                        choices=("train", "valid", "test"))
    parser.add_argument("--tasks", default="",
                        help="comma-separated task numbers; all by default")
    parser.add_argument("--stories", type=int, default=0,
                        help="only the first N stories of each task")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--batch-tokens", type=int, default=6000)
    parser.add_argument("--new-tokens", type=int, default=24)
    parser.add_argument("--all", action="store_true",
                        help="report every question each system was asked, "
                             "not only the shared ones")
    options = parser.parse_args(argv)
    tasks = _tasks(options.tasks)
    if options.what == "v689":
        from research.v687 import build

        run_v689(options.split, tasks, options.processes, options.stories,
                 options.store or build.DEFAULT_STORE, options.cycles)
    elif options.what == "smollm3":
        run_smollm3(options.split, tasks, options.stories,
                    options.batch_tokens, options.new_tokens)
    summary = report(options.split, shared=not options.all)
    print(render(summary))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    sys.exit(main())

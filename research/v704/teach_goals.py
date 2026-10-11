"""Teaching the shared reader that a goal is a goal (v704): something a
person wants to achieve or improve, said without what to change.

    python -m research.v704.teach_goals write    the teacher writes goals
    python -m research.v704.teach_goals check    and chooses what each is
    python -m research.v704.teach_goals corpus   llm/code-talk-data/*-goals

Told "I want to improve the ability for it to communicate in general",
the architecture read a change of code, found a function called `plan`
by the word, and wrote `plan.extend(request)` after its `return` -- it
compiled, two answers agreed. Asked to "make a plan to improve how you
communicate", it wrote a TypeScript function. Neither is a change or a
request for code: each says what is wanted, not what to do. Read as a
goal (the code-talk act `goal`), it is never acted on directly
(`page.py`): what it is is said, and what to do is asked or planned.

**What the teacher writes** (SmolLM3, offline): messages a person types to
an assistant that works on their software, each wanting something achieved
or improved -- of a project's qualities, of the assistant itself, or a
plan asked for -- without saying which code to change. **What it checks**:
each message put back to it as a choice among every code-talk act and
`goal` (`teach_code_talk.ACT_CHOICES`); kept where it chooses `goal`.
Labels by construction: act `goal`, no aspect, no subject, no names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys

from research.v698 import teach_code_talk as K

DATA = K.DATA
WRITTEN = DATA / "goals-written.jsonl"
CHECKED = DATA / "goals-checked.jsonl"
SEED = 704
#: what a goal is about: a project's qualities, the assistant itself, a plan
TOPICS = [
    "how fast the program runs", "how reliable the server is",
    "the test coverage", "the documentation", "the error messages",
    "security", "how easy the code is to read", "how the project is "
    "deployed", "onboarding new developers", "the logging",
    "memory use", "start-up time", "how the app looks", "accessibility",
    "the build", "the release process", "the API's design",
    "support for another operating system", "handling bad input",
    "how data is stored", "backups", "the command line tool",
    "monitoring in production", "the code's structure",
    "fewer bugs", "how quickly pages load", "offline use",
    "the search feature", "the user's privacy", "the configuration",
    "how the assistant communicates", "how the assistant understands "
    "requests", "how the assistant explains what it did",
    "how the assistant plans larger work", "what the assistant remembers",
    "how the assistant answers questions", "how the assistant learns from "
    "mistakes", "how careful the assistant is with changes",
    "how the assistant writes code", "how the assistant handles vague "
    "requests",
    "a plan for a new feature", "a plan for refactoring", "a plan for "
    "migrating the database", "a plan for adding authentication",
    "a plan for the next release", "a roadmap for the project",
    "a plan to learn the codebase", "a plan to reduce technical debt",
    "a plan for moving to Python 3.12", "a plan for a mobile version",
]
#: messages written for each topic, by each of two samples
EACH = 8
#: the act choices the teacher checks with: the code-talk acts and a goal
GOAL_CHOICE = ("goal", "saying something they want to achieve, improve or "
                       "plan, without saying which code to change or what "
                       "to write")
# the goal beside the acts, before `none` (not first: a small teacher
# leans to option A)
CHOICES = tuple(one for one in K.ACT_CHOICES if one[0] != "none") + \
    (GOAL_CHOICE,) + tuple(one for one in K.ACT_CHOICES if one[0] == "none")
VALID = K.VALID


def write(batch: int = 16) -> None:
    from research.v696.teach_meaning import Teacher
    DATA.mkdir(parents=True, exist_ok=True)
    done = set()
    if WRITTEN.exists():
        done = {json.loads(line)["topic"] for line in
                WRITTEN.open(encoding="utf-8")}
    todo = [one for one in TOPICS if one not in done]
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        prompts = [
            f"Write {EACH} different messages a person might type to an "
            f"assistant that works on their software, each saying what "
            f"they want to achieve or improve about {topic} -- an aim or a "
            f"wish, or asking for a plan, never which function or file to "
            f"change or what code to write. Vary them a lot: short and "
            f"long, casual and formal, 'I want', 'we should', 'can we', "
            f"'let's', 'help me', 'make a plan'. One per line, nothing "
            f"else." for topic in chunk]
        replies = teacher.write(prompts, longest=500, samples=2,
                                temperature=0.9)
        with WRITTEN.open("a", encoding="utf-8") as out:
            for topic, written in zip(chunk, replies):
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”")
                        if 8 <= len(line) <= 220 and line not in lines:
                            lines.append(line)
                out.write(json.dumps({"topic": topic, "lines": lines}) +
                          "\n")
        print(f"{at + len(chunk)} of {len(todo)} topics", flush=True)


def check(batch: int = 24) -> dict:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(one["topic"], line) for one in rows for line in one["lines"]]
    teacher = Teacher()
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            replies = teacher.write(
                [K._choice(line, list(CHOICES), "What is this message "
                           "doing?") for _, line in chunk],
                longest=3)
            for (topic, line), reply in zip(chunk, replies):
                chosen = K._picked(reply[0], list(CHOICES))
                counts[chosen] = counts.get(chosen, 0) + 1
                out.write(json.dumps({"topic": topic, "said": line,
                                      "chosen": chosen}) + "\n")
    print(json.dumps(counts))
    return counts


def split_of(text: str) -> str:
    at = hashlib.sha1(text.lower().encode()).digest()[0] / 256
    return "valid" if at < VALID else "train"


def corpus() -> dict:
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    kept = [one for one in rows if one["chosen"] == "goal"]
    rng = random.Random(SEED)
    rng.shuffle(kept)
    out = {"train": [], "valid": []}
    for one in kept:
        words = K.words(one["said"])
        if not words:
            continue
        out[split_of(one["said"])].append(K.record(
            words, ["O"] * len(words), "goal", "none", "none",
            "goal " + one["topic"]))
    counts = {}
    for split, records in out.items():
        with (DATA / f"{split}-goals.jsonl").open("w",
                                                  encoding="utf-8") as stream:
            for one in records:
                stream.write(json.dumps(one) + "\n")
        counts[split] = len(records)
    counts["written"] = len(rows)
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "corpus"))
    options = parser.parse_args(argv)
    {"write": write, "check": check, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

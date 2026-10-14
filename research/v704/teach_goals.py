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
#: a quality of the code as a whole, ordered (v706): "make this code more
#: generalizable" was read a change (1.0), placed nowhere, and asked which
#: function -- the goals taught were wishes ('I want', 'we should'), never
#: an order that names no part of the code
QUALITIES = [
    "more generalizable", "more general", "more robust", "faster",
    "easier to read", "easier to maintain", "more modular", "easier to "
    "extend", "better tested", "more reliable", "less hard-coded",
    "more consistent", "simpler", "more secure", "handle more cases",
    "better documented",
]
#: a quality's topic, as written and checked
ORDER = "the code: "
#: what the assistant does, ordered of it (v705): "improve your general
#: knowledge" was read a change of `Archive.knowledge`, and the editor
#: rewrote a query in it -- orders about its own abilities name no code
ABILITIES = [
    "general knowledge", "memory", "answers", "understanding of what "
    "people ask", "reasoning", "planning", "explanations", "code writing",
    "debugging", "reading of data files", "maths", "common sense",
    "sense of time", "conversation", "judgement of when it is unsure",
]
#: an ability's topic, as written and checked
YOURS = "your: "
#: what an order kept as a goal of its own abilities says (teaching only)
YOU = re.compile(r"\b(you|your|yourself)\b", re.I)
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
    todo = [one for one in TOPICS + [ORDER + quality
                                     for quality in QUALITIES] +
            [YOURS + ability for ability in ABILITIES]
            if one not in done]
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        prompts = [_prompt(topic) for topic in chunk]
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


def _prompt(topic: str) -> str:
    if topic.startswith(YOURS):
        return (f"Write {EACH} different messages a person might type to "
                f"an AI assistant, telling it to get better at its own "
                f"{topic[len(YOURS):]} -- an order to the assistant about "
                f"itself, never naming a function, file or line, nor "
                f"saying what code to write. Vary them a lot: short and "
                f"long, casual and formal, 'improve your', 'work on your', "
                f"'get better at', 'make your ... better', 'boost your'. "
                f"One per line, nothing else.")
    if topic.startswith(ORDER):
        return (f"Write {EACH} different messages a person might type to "
                f"an assistant that works on their software, telling it to "
                f"make the code {topic[len(ORDER):]} -- an order about the "
                f"code as a whole, never naming a function, file, class or "
                f"line, nor saying what to change or what code to write. "
                f"Vary them a lot: short and long, casual and formal, "
                f"'make this code', 'make the codebase', 'make it', "
                f"'improve', 'rework', 'clean up'. One per line, nothing "
                f"else.")
    return (f"Write {EACH} different messages a person might type to an "
            f"assistant that works on their software, each saying what "
            f"they want to achieve or improve about {topic} -- an aim or a "
            f"wish, or asking for a plan, never which function or file to "
            f"change or what code to write. Vary them a lot: short and "
            f"long, casual and formal, 'I want', 'we should', 'can we', "
            f"'let's', 'help me', 'make a plan'. One per line, nothing "
            f"else.")


#: what an order about the code is checked by: a yes or no -- a choice
#: among every act reads an order as the change it is worded as
NAMES_A_PART = ("A programmer typed this to an assistant that works on "
                "their code: \"{line}\"\nDoes the message name a "
                "particular function, file, class, variable or line to "
                "change, or say exactly what code to write? Answer yes "
                "or no only.")


#: what an order kept as a goal names: the code as a whole (teaching only)
WHOLE = re.compile(r"\b(code|codebase|code base|system|software|project|"
                   r"architecture|program|application|app)\b", re.I)


def check(batch: int = 24) -> dict:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    # checked once: checked again, the teacher's verdicts moved with how
    # the rows were batched (760 goals kept of 803 against 665)
    before = [json.loads(line) for line in CHECKED.open(encoding="utf-8")] \
        if CHECKED.exists() else []
    seen = {(one["topic"], one["said"]) for one in before}
    todo = [(one["topic"], line) for one in rows for line in one["lines"]
            if (one["topic"], line) not in seen]
    teacher = Teacher() if todo else None
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for one in before:
            out.write(json.dumps(one) + "\n")
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            replies = teacher.write(
                [NAMES_A_PART.format(line=line)
                 if topic.startswith((ORDER, YOURS))
                 else K._choice(line, list(CHOICES), "What is this "
                                "message doing?") for topic, line in chunk],
                longest=3)
            for (topic, line), reply in zip(chunk, replies):
                ordered = topic.startswith((ORDER, YOURS))
                chosen = (("change" if reply[0].strip().lower().startswith(
                    "yes") else "goal") if ordered else
                    K._picked(reply[0], list(CHOICES)))
                if chosen == "goal" and topic.startswith(YOURS) and \
                        not YOU.search(line):
                    chosen = "unnamed"
                if chosen == "goal" and topic.startswith(ORDER) and \
                        not WHOLE.search(line):
                    # `make it faster` is how a change of the last answer
                    # is asked (`make it recursive`): an order is taught a
                    # goal only where it says it is of the code whole --
                    # the teacher said no to all 253 (it never says yes)
                    chosen = "unnamed"
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

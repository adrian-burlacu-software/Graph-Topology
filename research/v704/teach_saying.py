"""Saying a goal's plan plainly (v704, step 4 of planning goals): the plan
as the architecture holds it, said in English by a decoder taught to.

    python -m research.v704.teach_saying messages   llm/plan-data/plan-messages.jsonl
    python -m research.v704.teach_saying write      the teacher says each
    python -m research.v704.teach_saying corpus     kept where faithful
    python -m research.v704.teach_saying train      llm/plan-sayer

A goal's reply was its parts joined -- "What I do worst now, by the
conversation test of 2026-10-11: 1) ... Goals like it here were reached
this way -- v698/vscode-harness (...): 1. Build ..." -- the parts right,
the saying a template's. The decoder (v690) says one or two sentences of
a verdict; a plan is more.

**The messages** are what the page holds for a goal (`message`): the
goal, its weaknesses if it is about the assistant itself, the past goals
it is like, its steps, each with what it was there. Made for every goal
the teacher wrote (`teach_goals`) and every goal this project reached.
**The teacher** (SmolLM3, offline) says each plainly. **Kept where
faithful, by construction** (`faithful`): every weakness's example said as
it is, every version the message cites and no other, as many steps as it
has, each with its own verb, and the first step asked about. **The
sayer** (SmolLM2-360M, `llm/plan-sayer`) is taught what is kept; at run
time its saying is checked the same way, and the parts joined where it
fails.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.v704 import history, selfmodel

DATA = history.HISTORY.parent
MESSAGES = DATA / "plan-messages.jsonl"
SAID = DATA / "plan-said.jsonl"
CORPUS = DATA / "plan-saying.jsonl"
SAYER = history.ROOT / "llm" / "plan-sayer"
SEED = 7041
SAYING = ("You say an assistant's plan for a goal to the person who set it, "
          "plainly, in its own words. Keep every quoted example, every "
          "version named and the numbered steps; add nothing.")
TEACHER_SAYS = (
    "Below is what an assistant holds about a goal the person set: maybe "
    "what it does worst now (with real examples), the past goals it is "
    "like, and its steps (each with what it was in a past goal). Write the "
    "assistant's reply to the person, in plain friendly English, as the "
    "assistant ('I'): one or two short sentences saying it has changed "
    "nothing and what it does worst (if given, quoting each example "
    "exactly as given), then the steps as a numbered list -- the same "
    "number of steps, in the same order, each starting with its verb "
    "(Build, Teach, Change, Add, Measure, Make, Write, Clear) and saying "
    "what it was in the past goal named (keep each version name such as "
    "v698/vscode-harness exactly) -- then ask whether to start on the "
    "first step. Add nothing that is not given. Reply with the reply "
    "only.\n\n")
#: the verbs a plan's steps start with (`history.SAYING`)
VERBS = {kind: said.split()[0] for kind, said in history.SAYING.items()}


def message(goal: str, weak: list, planned: dict, model: dict | None) -> str:
    """What the page holds for a goal, as the sayer is given it."""
    lines = [f"goal: {goal}"]
    if weak and model:
        lines.append(f"tested: {model['built']}")
        for one in weak:
            example = one["examples"][0] if one["examples"] else None
            lines.append(f"weak: {one['what']} | {one['how']}" + (
                f" | e.g. \"{example['said']}\" answered by "
                f"{example['part']}" if example else ""))
    lines.append("like: " + "; ".join(
        f"{one['version']} ({one['goal']})" for one in planned["like"]))
    for step in planned["steps"]:
        lines.append(f"step: {step['said']} | as in "
                     f"{step['as in']['version']}: "
                     f"{history._short(step['as in']['step'])}")
    return "\n".join(lines)


def faithful(message_: str, said: str) -> bool:
    """Whether a saying keeps what the message holds: each example quoted
    as it is, every version it cites and no other, its steps numbered, as
    many, each with its verb, in order, and the first asked about."""
    lowered = said.lower()
    for quoted in re.findall(r'e\.g\. "([^"]+)"', message_):
        if quoted.lower() not in lowered:
            return False
    versions = set(re.findall(r"\b(v\d{3}[\w/\-]*)", message_))
    if any(one.lower() not in lowered for one in versions):
        return False
    if set(re.findall(r"\b(v\d{3})", said)) - {one[:4] for one in versions}:
        return False
    steps = [line for line in message_.splitlines()
             if line.startswith("step: ")]
    numbered = re.findall(r"(?m)^\s*(\d+)[.)]\s+(\S+)", said)
    if len(numbered) != len(steps):
        return False
    for (number, verb), step in zip(numbered, steps):
        wanted = step[len("step: "):].split()[0].lower()
        if verb.lower().strip("*:,") != wanted:
            return False
    return "?" in said.rstrip()[-200:]


def messages() -> int:
    """A message for every goal the teacher wrote and this project reached:
    its plan, and -- of a goal about the assistant -- its weaknesses."""
    from research.v698.teach_code_talk import DATA as TALK
    goals = []
    for split in ("train", "valid"):
        path = TALK / f"{split}-goals.jsonl"
        if path.exists():
            goals += [(" ".join(json.loads(line)["words"]), split)
                      for line in path.open(encoding="utf-8")]
    goals += [(one["goal"], "train") for one in history._goals()]
    model = selfmodel.held()
    made = 0
    with MESSAGES.open("w", encoding="utf-8") as out:
        for goal, split in goals:
            planned = history.plan(goal)
            if not planned:
                continue
            weak = selfmodel.weaknesses(goal, model) \
                if model and selfmodel.about_itself(goal) else []
            out.write(json.dumps({"goal": goal, "split": split,
                                  "message": message(goal, weak, planned,
                                                     model)}) + "\n")
            made += 1
    print(f"{made} messages of {len(goals)} goals")
    return made


def write(batch: int = 12) -> None:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in MESSAGES.open(encoding="utf-8")]
    done = set()
    if SAID.exists():
        done = {json.loads(line)["message"] for line in
                SAID.open(encoding="utf-8")}
    todo = [one for one in rows if one["message"] not in done]
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write([TEACHER_SAYS + one["message"]
                                 for one in chunk], longest=450, samples=2,
                                temperature=0.8)
        with SAID.open("a", encoding="utf-8") as out:
            for one, said in zip(chunk, replies):
                out.write(json.dumps({**one, "said": said}) + "\n")
        print(f"{at + len(chunk)} of {len(todo)}", flush=True)


def corpus() -> dict:
    rows = [json.loads(line) for line in SAID.open(encoding="utf-8")]
    kept, total = [], 0
    for row in rows:
        for said in row["said"]:
            total += 1
            said = said.strip()
            if faithful(row["message"], said):
                kept.append({"split": row["split"], "statement":
                             row["message"], "target": said})
    random.Random(SEED).shuffle(kept)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in kept:
            out.write(json.dumps(one) + "\n")
    counts = {"said": total, "kept": len(kept),
              "train": sum(one["split"] == "train" for one in kept)}
    print(json.dumps(counts))
    return counts


def train(epochs: int = 2) -> None:
    """The sayer: SmolLM2-360M taught the kept sayings (the editor's
    trainer, told what it says)."""
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T._text = lambda row: row["statement"]
    # the editor's trainer holds a row's split as train / dev; the held
    # goals are dev here
    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    staged = CORPUS.with_name("plan-saying-staged.jsonl")
    with staged.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps({**one, "split": "train" if one["split"]
                                  == "train" else "dev"}) + "\n")
    T.train(SAYER, epochs=epochs, corpus=staged, longest=900)


_LOADED: dict = {}


def say(message_: str) -> str | None:
    """The plan said by the sayer, where it keeps what the message holds;
    None where there is no sayer or it does not."""
    if not (SAYER / "config.json").exists():
        return None
    if "sayer" not in _LOADED:
        from research.v696.sketcher import Sketcher
        _LOADED["sayer"] = Sketcher(SAYER)
    written = _LOADED["sayer"].write([message_], samples=2, longest=400,
                                     saying=SAYING)[0]
    return next((one.strip() for one in written
                 if faithful(message_, one.strip())), None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("messages", "write", "corpus",
                                        "train"))
    options = parser.parse_args(argv)
    {"messages": messages, "write": write, "corpus": corpus,
     "train": train}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

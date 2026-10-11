"""A goal planned from how this project reached its own goals (v704, step
2 of planning goals).

    python -m research.v704.history collect   llm/plan-data/goals-history.jsonl
    python -m research.v704.history plan "I want it to communicate better"

Read as a goal (`teach_goals`), "I want to improve the ability for it to
communicate in general" was said back with nothing done: the architecture
had nowhere to find what such a goal takes. This project has: each version
merged is a goal reached ("Bash, for basic computer use"), and its commits,
in order, the steps that reached it.

**Each step's kind, by construction** -- from what the commit touched, not
from its message: only tests (`check`), only documents (`document`), the
regeneration or ignore lists or what was deleted (`tidy`), a default model
set (`default`), a teaching script (`teach`), a measuring harness
(`measure`), a module added (`build`), else existing code changed
(`change`). The teacher (SmolLM3), asked each commit's kind as a choice
twice in two orders, agreed with itself on 68 of 256, `look` for a fix
among them: the messages are terse, and this project's own.

**A goal's plan** is chosen, not written (v703: planning is choosing): the
past goals nearest it in meaning (MiniLM, as pretrained: `v701.likeness`),
the kinds of step most of them took, in the order they took them, each
said with what it was in the nearest that took it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "llm" / "plan-data" / "goals-history.jsonl"
#: what a plan says of each kind of step
SAYING = {
    "build": "Build the part that is missing",
    "teach": "Teach the model that reads or writes it, from data made for "
             "it",
    "change": "Change the parts it touches, and fix what fails",
    "check": "Add the tests that hold it",
    "measure": "Measure it, on what was never taught",
    "default": "Make it the default where it measures better",
    "document": "Write down what was done and why",
    "tidy": "Clear what is left over",
}
#: the past goals a plan is chosen from, and how near the nearest must be
NEAREST, LEAST_LIKE = 3, 0.2
#: a kind of step is planned where this many of the nearest goals took it
TAKEN_BY = 2
DEFAULTS = re.compile(r"^\+\s*(PREFERRED|EDITOR|STEP_EDITOR|WRITER|MODEL|"
                      r"PICKER|USES|PROPOSERS\w*)\s*[=,]", re.M)
TEACHING = re.compile(r"(^|/)(teach|train)_?\w*\.py$")
MEASURING = re.compile(r"(attach|measure|bench|eval|harness|converse|audit|"
                       r"heldout|probe)\w*\.py$")


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, encoding="utf-8",
                          errors="replace").stdout or ""


def kind_of(commit: str) -> str:
    """A commit's kind of step, by what it touched."""
    files = [(line.split("\t")[0][:1], line.split("\t")[-1]) for line in
             _git("show", "--name-status", "--format=",
                  commit).splitlines() if "\t" in line]
    paths = [path for _, path in files]
    if not paths:
        return "tidy"
    if all(re.search(r"(^|/)test_\w*\.py$", path) for path in paths):
        return "check"
    if all(path.endswith((".md", ".txt")) for path in paths):
        return "document"
    if all(path in ("regenerate.py", ".gitignore") or status == "D"
           for status, path in files):
        return "tidy"
    if len(paths) <= 4 and DEFAULTS.search(
            _git("show", "--format=", "-U0", commit)):
        return "default"
    if any(TEACHING.search(path) for path in paths):
        return "teach"
    if any(MEASURING.search(path) for path in paths):
        return "measure"
    if any(status == "A" and path.endswith(".py") and "test_" not in path
           for status, path in files):
        return "build"
    return "change"


def collect() -> int:
    """Every version merged into master: the goal it reached, and its
    commits in order, each with its kind."""
    out = []
    for line in _git("log", "--merges", "--first-parent", "master",
                     "--format=%H %s").splitlines():
        commit, _, subject = line.partition(" ")
        found = re.match(r"Merge (?:branch )?'?([\w./\-]+)'?: (.+)",
                         subject) or re.match(
            r"Merge (the .+?|[\w./\- ]+?): (.+)", subject)
        if not found:
            continue
        steps = [{"commit": one[:12], "said": said, "kind": kind_of(one)}
                 for one, _, said in (row.partition(" ") for row in _git(
                     "log", "--no-merges", "--reverse", "--format=%H %s",
                     f"{commit}^1..{commit}^2").splitlines() if row.strip())]
        if steps:
            out.append({"version": found.group(1), "goal": found.group(2),
                        "steps": steps})
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    counts = Counter(step["kind"] for one in out for step in one["steps"])
    print(f"{len(out)} goals, {sum(counts.values())} steps: "
          f"{json.dumps(counts)}")
    return len(out)


def _goals() -> list:
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.open(encoding="utf-8")]


def plan(goal: str) -> dict | None:
    """The steps a goal takes, chosen from the past goals nearest it: the
    kinds of step most of them took, in the order they took them, each
    said with what it was in the nearest that took it. None where no past
    goal is near."""
    from research.v701.likeness import nearest
    goals = _goals()
    if not goals:
        return None
    ranked = nearest(goal, [one["goal"] for one in goals])
    by_goal = {one["goal"]: one for one in goals}
    near = [(score, by_goal[text]) for score, text in ranked[:NEAREST]]
    if not near or near[0][0] < LEAST_LIKE:
        return None
    taken: Counter = Counter()
    where: dict = {}
    example: dict = {}
    for _, one in near:
        steps = one["steps"]
        seen: set = set()
        for at, step in enumerate(steps):
            kind = step["kind"]
            if kind not in SAYING or kind in seen:
                continue
            # where in its goal a kind first came: its order in a plan
            where.setdefault(kind, []).append(at / max(len(steps) - 1, 1))
            seen.add(kind)
            example.setdefault(kind, (one["version"], step["said"]))
        taken.update(seen)
    least = min(TAKEN_BY, len(near))
    kinds = [kind for kind, count in taken.items() if count >= least]
    kinds.sort(key=lambda kind: sum(where[kind]) / len(where[kind]))
    steps = [{"kind": kind, "said": SAYING[kind],
              "as in": {"version": example[kind][0],
                        "step": example[kind][1]}} for kind in kinds]
    return {"steps": steps, "like": [
        {"version": one["version"], "goal": one["goal"],
         "likeness": round(score, 3)} for score, one in near]}


def said(found: dict) -> str:
    """A plan in words: the goals it is like, then each step."""
    like = "; ".join(f"{one['version']} ({one['goal']})"
                     for one in found["like"])
    lines = [f"Goals like it here were reached this way -- {like}:"]
    for at, step in enumerate(found["steps"], 1):
        example = step["as in"]
        lines.append(f"{at}. {step['said']} (as in {example['version']}: "
                     f"{_short(example['step'])})")
    return "\n".join(lines)


def _short(text: str, most: int = 90) -> str:
    text = re.sub(r"^v\d+[\w/\-]*:\s*", "", text)
    return text if len(text) <= most else text[:most].rsplit(" ", 1)[0] + \
        " ..."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("collect", "plan"))
    parser.add_argument("goal", nargs="?", default="")
    options = parser.parse_args(argv)
    if options.job == "collect":
        collect()
    else:
        found = plan(options.goal)
        print(said(found) if found else "no goal here is like it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

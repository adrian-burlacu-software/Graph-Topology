"""Teaching the architecture to plan a change across a project: a request,
the files it touches, and what to change in each.

    python -m research.v703.teach_plans commits    llm/plan-data/commits.jsonl
    python -m research.v703.teach_plans describe   the teacher says each file's part
    python -m research.v703.teach_plans corpus     llm/plan-data/plans.jsonl
    python -m research.v703.teach_plans train      llm/planner

Asked to make the architecture able to do larger subsystem changes, it
changed one function, chosen because a word of the request was its name
(`Open.the`); asked to report whether the shell asks in /api/health and
show it in the MCP's health tool, it wrote a curl command. It changes one
function at a time, and nothing chose which.

**What a plan is taught from**: commits that change several files -- this
project's own (how v700, v701, v702 were added: the nearest examples), and
CommitChronicle's (JetBrains Research, 2023: commits whole, every file's
diff) of Python projects. **The teacher** (SmolLM3, offline) reads the
commit's message and one file's diff and says what to change in that file,
as a developer tells it (`in page.py, add the shell act before code talk`):
each step is the change made, its file the file changed -- by
construction. **The planner** (SmolLM2-360M) is given a request and the
files that may be touched -- those the commit changed, among others of the
same project -- and writes one step per line, `path: what to change`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
CHRONICLE = ROOT / "data" / "commit-chronicle"
DATA = LLM / "plan-data"
COMMITS = DATA / "commits.jsonl"
DESCRIBED = DATA / "described.jsonl"
PLANS = DATA / "plans.jsonl"
PLANNER = LLM / "planner"
SEED = 703

#: a commit planned: as many files as this, no more; a file's diff shown
#: the teacher at most this many lines
LEAST_FILES, MOST_FILES, DIFF_LINES = 2, 6, 60
#: the files a planner is shown: the commit's, and others of the project
SHOWN = 12
CODE = re.compile(r"\.(py|ts|tsx|js|md|json|ya?ml|toml|cfg|txt|csv)$")
SAYING = ("You plan a change across a project. Given what is asked and its "
          "files, answer with one line per file to change: the path, a "
          "colon, what to change there. A file to make is said new path.")


def split_of(key: str) -> str:
    at = hashlib.sha1(key.encode()).digest()[0]
    return "test" if at < 8 else "dev" if at < 16 else "train"


# -- commits --------------------------------------------------------------------------

def _git(*args) -> str:
    """Git's output, read as UTF-8 (a message's quotes are not cp1252)."""
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, encoding="utf-8",
                          errors="replace").stdout or ""


def _ours() -> list:
    """This project's commits that change several files: (message, mods),
    each mod {path, change, diff}."""
    out = []
    for commit in _git("log", "--no-merges", "--format=%H").split():
        message = _git("log", "-1", "--format=%B", commit).strip()
        mods = []
        for line in _git("show", "--name-status", "--format=",
                         commit).splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not CODE.search(parts[-1]):
                continue
            path = parts[-1]
            mods.append({"path": path, "change": parts[0][:1],
                         "diff": _git("show", "--format=", commit, "--",
                                      path)})
        if LEAST_FILES <= len(mods) <= MOST_FILES and message:
            out.append({"key": f"ours:{commit}", "message": message,
                        "mods": mods, "project": "graph-topology",
                        "files": None})
    return out


def _chronicle(most: int) -> list:
    """CommitChronicle's commits of Python projects that change several
    files, as published (one shard)."""
    import pyarrow.parquet as pq
    out = []
    for shard in sorted(CHRONICLE.glob("*.parquet")):
        try:
            table = pq.read_table(shard, columns=["hash", "repo", "message",
                                                  "mods", "language"])
        except Exception as bad:                    # noqa: BLE001
            # a shard still coming (the download resumes): next time
            print(f"{shard.name} not read: {bad}", flush=True)
            continue
        for row in table.to_pylist():
            if row["language"] != "Python":
                continue
            mods = [{"path": one["new_path"] or one["old_path"],
                     "change": {"ADD": "A", "DELETE": "D"}.get(
                         one["change_type"], "M"),
                     "diff": one["diff"]}
                    for one in row["mods"] or []]
            mods = [one for one in mods if one["path"] and
                    CODE.search(one["path"])]
            words = len((row["message"] or "").split())
            if not (LEAST_FILES <= len(mods) <= MOST_FILES) or words < 4 \
                    or any(len(one["diff"]) > 20_000 for one in mods):
                continue
            out.append({"key": f"cc:{row['repo']}:{row['hash']}",
                        "message": row["message"].strip(), "mods": mods,
                        "project": row["repo"], "files": None})
            if len(out) >= most:
                return out
    return out


def commits(most: int = 6000) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    found = _ours() + _chronicle(most)
    with COMMITS.open("w", encoding="utf-8") as out:
        for one in found:
            out.write(json.dumps(one) + "\n")
    print(f"{len(found)} commits", flush=True)
    return len(found)


# -- the teacher says each file's part ------------------------------------------------

def _shown_diff(diff: str) -> str:
    lines = [one for one in diff.splitlines()
             if one.startswith(("+", "-", "@@")) and not
             one.startswith(("+++", "---"))]
    return "\n".join(lines[:DIFF_LINES])


def describe(batch: int = 16) -> None:
    """Each file's part of each commit, said as an instruction."""
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in COMMITS.open(encoding="utf-8")]
    done = set()
    if DESCRIBED.exists():
        done = {json.loads(line)["key"] for line in
                DESCRIBED.open(encoding="utf-8")}
    todo = [one for one in rows if one["key"] not in done]
    print(f"{len(todo)} commits to describe", flush=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    jobs = [(one, mod) for one in todo for mod in one["mods"]]
    said: dict = {}
    for at in range(0, len(jobs), batch):
        chunk = jobs[at:at + batch]
        prompts = [(
            f"A commit's message: \"{one['message'][:300]}\".\nIts change "
            f"to the file {mod['path']}:\n{_shown_diff(mod['diff'])}\n\n"
            f"In one short instruction, as a developer would tell a "
            f"colleague, say what to change in {mod['path']} -- name the "
            f"functions, classes or settings it touches. Do not name the "
            f"file. One line, nothing else.") for one, mod in chunk]
        replies = teacher.write(prompts, longest=60)
        for (one, mod), reply in zip(chunk, replies):
            line = reply[0].strip().splitlines()[0].strip().strip("\"'“”") \
                if reply[0].strip() else ""
            said.setdefault(one["key"], {})[mod["path"]] = line
        # each commit written once every file of it is said
        with DESCRIBED.open("a", encoding="utf-8") as out:
            for key in list(said):
                one = next(row for row in todo if row["key"] == key)
                if len(said[key]) == len(one["mods"]):
                    steps = [{"path": mod["path"], "change": mod["change"],
                              "said": said[key][mod["path"]]}
                             for mod in one["mods"]]
                    out.write(json.dumps({"key": key,
                                          "message": one["message"],
                                          "project": one["project"],
                                          "steps": steps}) + "\n")
                    del said[key]
        print(f"  {min(at + batch, len(jobs))}/{len(jobs)} files", flush=True)


# -- the corpus ---------------------------------------------------------------------------

def said_plan(steps: list) -> str:
    """A plan as the planner says it: one line per file."""
    return "\n".join(("new " if one["change"] == "A" else "")
                     + f"{one['path']}: {one['said']}" for one in steps)


def prompt(request: str, files: list) -> str:
    return (f"{request.strip()}\n\nfiles:\n" + "\n".join(files) + "\n")


def corpus() -> dict:
    """The planner's rows: a request and the files shown -- the commit's
    (but those it makes), among others of the same project -- and the
    plan."""
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in DESCRIBED.open(encoding="utf-8")]
    by_project: dict = {}
    for one in rows:
        by_project.setdefault(one["project"], set()).update(
            step["path"] for step in one["steps"])
    ours = _git("ls-files").split()
    ours = [one for one in ours if CODE.search(one)]
    every = sorted({path for paths in by_project.values() for path in paths})
    out = []
    for one in rows:
        steps = [step for step in one["steps"] if step["said"] and
                 len(step["said"].split()) >= 3]
        if len(steps) < LEAST_FILES:
            continue
        mine = [step["path"] for step in steps if step["change"] != "A"]
        pool = ours if one["project"] == "graph-topology" else sorted(
            by_project[one["project"]]) or every
        others = [path for path in pool if path not in mine]
        if len(others) < SHOWN - len(mine):
            others += rng.sample(every, min(len(every), SHOWN))
            others = [path for path in dict.fromkeys(others)
                      if path not in mine]
        shown = mine + rng.sample(others, max(0, min(len(others),
                                                     SHOWN - len(mine))))
        rng.shuffle(shown)
        request = one["message"].splitlines()[0][:300]
        out.append({"statement": request, "path": "plan",
                    "part": "\n".join(shown), "target": said_plan(steps),
                    "split": split_of(one["key"]), "source": one["key"]})
    rng.shuffle(out)
    with PLANS.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    count = {split: sum(one["split"] == split for one in out)
             for split in ("train", "dev", "test")}
    print(json.dumps(count))
    return count


def train(epochs: int = 2) -> None:
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T.train(PLANNER, epochs=epochs, rate=1e-4, seed=SEED, corpus=PLANS,
            longest=1024)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("commits", "describe", "corpus",
                                        "train"))
    parser.add_argument("--most", type=int, default=6000)
    options = parser.parse_args(argv)
    if options.job == "commits":
        commits(options.most)
    else:
        {"describe": describe, "corpus": corpus, "train": train}[
            options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

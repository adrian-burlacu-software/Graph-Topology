"""Saying a goal's plan plainly (v704, step 4 of planning goals): the plan
as the architecture holds it, said in English by a decoder taught to --
piece by piece.

    python -m research.v704.teach_saying messages   llm/plan-data/plan-pieces.jsonl
    python -m research.v704.teach_saying write      the teacher says each
    python -m research.v704.teach_saying corpus     kept where faithful
    python -m research.v704.teach_saying train      llm/plan-sayer

A goal's reply was its parts joined -- "What I do worst now, by the
conversation test of 2026-10-11: 1) ... Goals like it here were reached
this way -- v698/vscode-harness (...): 1. Build ..." -- the parts right,
the saying a template's.

**Said whole, the teacher failed** (SmolLM3, 0 of 24 faithful): it said
steps twice, dropped the versions, and made up that nothing changed when
the person tried them. **Said in pieces** it is asked one small thing at
a time: the **opening** -- nothing changed, and what the assistant does
worst now, each example quoted -- and each **step**, one sentence that
begins with its verb and says what it was in the version named. **Kept
where faithful, by construction** (`faithful`): every example quoted as
it is, every version the piece cites and no other, a step's verb first.
**The sayer** (SmolLM2-360M, `llm/plan-sayer`) is taught what is kept;
at run time each piece it says is checked the same way, and the piece
said as before where it fails.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.v704 import history, selfmodel

DATA = history.HISTORY.parent
PIECES = DATA / "plan-pieces.jsonl"
SAID = DATA / "plan-said.jsonl"
CORPUS = DATA / "plan-saying.jsonl"
SAYER = history.ROOT / "llm" / "plan-sayer"
SEED = 7041
SAYING = ("You say one piece of an assistant's plan to the person, plainly, "
          "in its own words: keep every quoted example and every version "
          "named, begin a step with its verb, and add nothing.")
OPENING_ASKS = (
    "An assistant was given a goal by a person. It has changed nothing "
    "yet. Below is the goal and, maybe, what the assistant does worst now "
    "with a real example of each. Write the assistant's first one or two "
    "sentences to the person, as 'I': that it has changed nothing, and "
    "what it does worst (if given), quoting each example exactly as "
    "written between the quotes. Do not list steps, do not add anything "
    "not given. Reply with the sentences only.\n\n")
STEP_ASKS = (
    "Rewrite this step of a plan as one plain sentence to the person. "
    "Begin it with the word '{verb}'. Say what the step was in "
    "{version}, keeping '{version}' exactly as written. Add nothing not "
    "given. Reply with the sentence only.\n\n")
#: the most a piece is said in
LONGEST_OPENING, LONGEST_STEP = 420, 260


def opening(goal: str, weak: list, model: dict | None) -> str:
    """The opening's piece: the goal, and its weaknesses if it is about the
    assistant itself."""
    lines = ["say: opening", f"goal: {goal}"]
    if weak and model:
        lines.append(f"tested: {model['built']}")
        for one in weak:
            example = one["examples"][0] if one["examples"] else None
            lines.append(f"weak: {one['what']} | {one['how']}" + (
                f" | e.g. \"{example['said']}\" answered by "
                f"{example['part']}" if example else ""))
    return "\n".join(lines)


def step(one: dict) -> str:
    """A step's piece: what to do, and what it was in the version named."""
    return (f"say: step\nstep: {one['said']} | as in "
            f"{one['as in']['version']}: "
            f"{history._short(one['as in']['step'])}")


FRAMING = DATA / "plan-framing.json"
#: a word is the openings' framing where this share of the teacher's
#: openings say it, whatever they open (`changed`, `anything`, `yet`)
FRAMES_IN = 0.15


def _content(text: str) -> set:
    return {one.rstrip("s") for one in re.findall(r"[a-z]{4,}", text.lower())}


def _framing() -> set | None:
    if not FRAMING.exists():
        return None
    return set(json.loads(FRAMING.read_text(encoding="utf-8")))


def faithful(piece: str, said: str, framing: set | None = None) -> bool:
    """Whether a saying keeps what its piece holds: each example quoted as
    it is, every version cited and no other; a step, one sentence that
    begins with its verb; an opening, no steps, and no word its piece has
    not but the openings' framing (v690's read-back: `I tend to focus too
    much on individual features` was said of a goal that said nothing of
    it)."""
    if framing is None:
        framing = _framing()
    said = said.strip()
    if not said or "\n" in said.strip() and piece.startswith("say: step"):
        return False
    lowered = said.lower()
    for quoted in re.findall(r'e\.g\. "([^"]+)"', piece):
        if quoted.lower() not in lowered:
            return False
    versions = set(re.findall(r"\b(v\d{3}[\w/\-]*)", piece))
    if any(one.lower() not in lowered for one in versions):
        return False
    if set(re.findall(r"\b(v\d{3})", said)) - {one[:4] for one in versions}:
        return False
    if piece.startswith("say: step"):
        verb = piece.split("step: ", 2)[-1].split()[0].lower()
        return said.split()[0].lower().strip("*:,") == verb and \
            len(said) <= LONGEST_STEP
    if re.search(r"(?m)^\s*\d+[.)]\s", said) or len(said) > LONGEST_OPENING:
        return False
    return framing is None or _content(said) <= _content(piece) | framing


def messages() -> int:
    """The pieces of every goal's plan: the teacher's goals and this
    project's -- each opening, and each step once."""
    from research.v698.teach_code_talk import DATA as TALK
    goals = []
    for split in ("train", "valid"):
        path = TALK / f"{split}-goals.jsonl"
        if path.exists():
            goals += [(" ".join(json.loads(line)["words"]), split)
                      for line in path.open(encoding="utf-8")]
    goals += [(one["goal"], "train") for one in history._goals()]
    model = selfmodel.held()
    pieces: dict = {}
    for goal, split in goals:
        planned = history.plan(goal)
        if not planned:
            continue
        weak = selfmodel.weaknesses(goal, model) \
            if model and selfmodel.about_itself(goal) else []
        pieces.setdefault(opening(goal, weak, model), split)
        for one in planned["steps"]:
            pieces.setdefault(step(one), split)
    with PIECES.open("w", encoding="utf-8") as out:
        for piece, split in pieces.items():
            out.write(json.dumps({"piece": piece, "split": split}) + "\n")
    print(f"{len(pieces)} pieces of {len(goals)} goals")
    return len(pieces)


def _asked(piece: str) -> str:
    if piece.startswith("say: step"):
        verb = piece.split("step: ", 2)[-1].split()[0]
        found = re.search(r"as in (\S+):", piece)
        return STEP_ASKS.format(verb=verb, version=found.group(1)
                                if found else "the past goal") + piece
    return OPENING_ASKS + piece


def write(batch: int = 16) -> None:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in PIECES.open(encoding="utf-8")]
    done = set()
    if SAID.exists():
        done = {json.loads(line)["piece"] for line in
                SAID.open(encoding="utf-8")}
    todo = [one for one in rows if one["piece"] not in done]
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write([_asked(one["piece"]) for one in chunk],
                                longest=160, samples=3, temperature=0.8)
        with SAID.open("a", encoding="utf-8") as out:
            for one, said in zip(chunk, replies):
                out.write(json.dumps({**one, "said": said}) + "\n")
        print(f"{at + len(chunk)} of {len(todo)}", flush=True)


def corpus() -> dict:
    rows = [json.loads(line) for line in SAID.open(encoding="utf-8")]
    # the openings' framing: the words a share of all openings say,
    # whatever their goal -- counted, not listed
    openings = [said.strip().strip('"') for row in rows
                if not row["piece"].startswith("say: step")
                for said in row["said"]]
    counted: dict = {}
    for said in openings:
        for word in _content(said):
            counted[word] = counted.get(word, 0) + 1
    framing = {word for word, count in counted.items()
               if count >= FRAMES_IN * max(len(openings), 1)}
    FRAMING.write_text(json.dumps(sorted(framing)), encoding="utf-8")
    kept, total, by = [], 0, {"opening": [0, 0], "step": [0, 0]}
    for row in rows:
        kind = "step" if row["piece"].startswith("say: step") else "opening"
        for said in row["said"]:
            said = said.strip().strip('"')
            total += 1
            by[kind][1] += 1
            if faithful(row["piece"], said, framing):
                by[kind][0] += 1
                kept.append({"split": row["split"], "statement":
                             row["piece"], "target": said})
    random.Random(SEED).shuffle(kept)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in kept:
            out.write(json.dumps(one) + "\n")
    counts = {"said": total, "kept": len(kept), "framing": sorted(framing),
              "kept of each": {kind: f"{one[0]} of {one[1]}"
                               for kind, one in by.items()}}
    print(json.dumps(counts))
    return counts


def train(epochs: int = 2) -> None:
    """The sayer: SmolLM2-360M taught the kept sayings (the editor's
    trainer, told what it says)."""
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T._text = lambda row: row["statement"]
    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    staged = CORPUS.with_name("plan-saying-staged.jsonl")
    with staged.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps({**one, "split": "train" if one["split"]
                                  == "train" else "dev"}) + "\n")
    T.train(SAYER, epochs=epochs, corpus=staged, longest=512)


_LOADED: dict = {}


def _sayer():
    if not (SAYER / "config.json").exists():
        return None
    if "sayer" not in _LOADED:
        from research.v696.sketcher import Sketcher
        _LOADED["sayer"] = Sketcher(SAYER)
    return _LOADED["sayer"]


def said_by_sayer(pieces: list) -> list:
    """Each piece said by the sayer where what it says keeps the piece,
    else None in its place; all None where there is no sayer."""
    sayer = _sayer()
    if sayer is None:
        return [None] * len(pieces)
    written = sayer.write(pieces, samples=2, longest=160, saying=SAYING)
    return [next((one.strip() for one in answers
                  if faithful(piece, one.strip())), None)
            for piece, answers in zip(pieces, written)]


def say(goal: str, weak: list, planned: dict, model: dict | None,
        joined_opening: str) -> str | None:
    """A goal's plan said plainly: the sayer's opening and steps, each where
    faithful, else as before (`joined_opening`, `history.SAYING`); None
    where there is no sayer."""
    if _sayer() is None:
        return None
    pieces = [opening(goal, weak, model)] + [step(one)
                                             for one in planned["steps"]]
    said = said_by_sayer(pieces)
    like = ", ".join(one["version"] for one in planned["like"])
    lines = [said[0] or joined_opening,
             f"Goals like it here ({like}) were reached this way:"]
    for at, (one, plainly) in enumerate(zip(planned["steps"], said[1:]), 1):
        example = one["as in"]
        lines.append(f"{at}. " + (plainly or (
            f"{one['said']} (as in {example['version']}: "
            f"{history._short(example['step'])})")))
    lines.append("Shall I start on the first step?")
    return "\n".join(lines)


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

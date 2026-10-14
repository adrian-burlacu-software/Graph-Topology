"""Teaching what a goal about its replies measures (v705): "make your
replies to goals more concise" is a check -- the length of what it says to
a goal, lower -- and the check is what makes a goal something a change can
be tried against (`measures`).

    python -m research.v705.teach_measures write    the teacher writes
    python -m research.v705.teach_measures check    and checks each
    python -m research.v705.teach_measures train    llm/measures-head

MiniLM alone takes the measure (`likeness`: `stop repeating yourself`,
repetition by 0.36) but not the way: `more concise` and `more detailed`
were within 0.04 of each other on every goal tried. **What the teacher
writes** (SmolLM3, offline): messages a person types to an assistant,
wanting its replies to have more or less of one measure (`measures.MEASURES`)
-- labelled by construction; and, for `none`, the goals it was taught that
are not about its replies (`goals-checked.jsonl`). **What it checks**: each
message put back to it as a yes or no -- does it ask for that? **What is
taught**: two heads over MiniLM's vectors as pretrained (the encoder is not
moved) -- which measure (or none), and which way.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.encoder import LLM
from research.v698 import teach_code_talk as K

DATA = K.DATA
WRITTEN = DATA / "measures-written.jsonl"
CHECKED = DATA / "measures-checked.jsonl"
HEAD = LLM / "measures-head"
SEED = 7055
EACH = 8
#: the ways, as the teacher is asked them
WAYS = {"min": "lower", "max": "higher"}
#: measures once written for, by the one each is taught as now
TAKEN_INTO = {"sentences": "length", "sentence length": "length"}
#: the goals not about its replies, at most, taught as `none`
NONE = 400


def _prompt(measure: str, way: str) -> str:
    return (f"Write {EACH} different messages a person might type to an AI "
            f"assistant, asking it to change how it replies so that its "
            f"replies are {WAYS[way]} in {measure}. Each message must be "
            f"about the assistant's replies themselves (its answers, its "
            f"responses to questions, to goals, to requests), never a "
            f"question or task of its own. Vary them a lot: short and long, "
            f"casual and formal, complaints ('your answers are too ...') "
            f"and wishes ('could you ...', 'make your replies ...', "
            f"'please ...'). One per line, nothing else.")


def write(batch: int = 10) -> None:
    from research.v696.teach_meaning import Teacher
    from research.v705 import measures
    DATA.mkdir(parents=True, exist_ok=True)
    done = set()
    if WRITTEN.exists():
        done = {(one["measure"], one["way"]) for one in
                map(json.loads, WRITTEN.open(encoding="utf-8"))}
    todo = [(name, way) for name in list(measures.MEASURES) + list(TAKEN_INTO)
            for way in WAYS
            if (name, way) not in done]
    if not todo:
        return
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write(
            [_prompt(name, way) for name, way in chunk],
            longest=500, samples=6, temperature=0.9)
        with WRITTEN.open("a", encoding="utf-8") as out:
            for (name, way), written in zip(chunk, replies):
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”")
                        if 8 <= len(line) <= 220 and line not in lines:
                            lines.append(line)
                out.write(json.dumps({"measure": name, "way": way,
                                      "lines": lines}) + "\n")
        print(f"{at + len(chunk)} of {len(todo)}", flush=True)


ASKS = ("A person typed this to an AI assistant: \"{line}\"\nDoes the "
        "message ask the assistant to give replies with {way} {said}? "
        "Answer yes or no only.")


def check(batch: int = 24) -> dict:
    from research.v696.teach_meaning import Teacher
    from research.v705 import measures
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(one["measure"], one["way"], line) for one in rows
            for line in one["lines"]]
    teacher = Teacher()
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            replies = teacher.write([ASKS.format(line=line, way=WAYS[way],
                                                 said=name)
                                     for name, way, line in chunk],
                                    longest=3)
            for (name, way, line), reply in zip(chunk, replies):
                kept = reply[0].strip().lower().startswith("yes")
                counts[kept] = counts.get(kept, 0) + 1
                out.write(json.dumps({"measure": name, "way": way,
                                      "said": line, "kept": kept}) + "\n")
    print(json.dumps({str(key): value for key, value in counts.items()}))
    return counts


def examples() -> list:
    """[(said, measure | "none", way | None)]: the checked messages, and
    goals it was taught that are about no measure of its replies."""
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    from research.v705 import measures
    # measures written for but no longer told apart are what they were
    # taken into: how many sentences and how long they run are a reply's
    # length (`more concise` was written under both)
    out = [(one["said"], one["measure"] if one["measure"] in
            measures.MEASURES else TAKEN_INTO[one["measure"]], one["way"])
           for one in rows if one["kept"]]
    goals = [json.loads(line) for line in
             (DATA / "goals-checked.jsonl").open(encoding="utf-8")]
    rng = random.Random(SEED)
    rng.shuffle(goals)
    out += [(one["said"], "none", None) for one in goals
            if one["chosen"] == "goal"][:NONE]
    return out


def train() -> dict:
    import torch
    from research.v701 import likeness
    from research.v705 import measures
    rows = examples()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    names = ["none"] + list(measures.MEASURES)
    vectors = torch.cat([likeness._vectors([one[0] for one in rows[at:at + 64]])
                         for at in range(0, len(rows), 64)])
    which = torch.tensor([names.index(one[1]) for one in rows])
    way = torch.tensor([1 if one[2] == "max" else 0 for one in rows])
    from research.v704.teach_goals import split_of
    held = [split_of(one[0]) == "valid" for one in rows]
    held = torch.tensor(held)
    heads = {"measure": torch.nn.Linear(vectors.shape[1], len(names)),
             "way": torch.nn.Linear(vectors.shape[1], 2)}
    out = {}
    for key, head in heads.items():
        target = which if key == "measure" else way
        mask = ~held if key == "measure" else ~held & (which != 0)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2,
                                      weight_decay=1e-3)
        for _ in range(400):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(vectors[mask]),
                                                     target[mask])
            loss.backward()
            optimizer.step()
        test = held if key == "measure" else held & (which != 0)
        with torch.no_grad():
            right = (head(vectors[test]).argmax(-1) == target[test])
        out[key] = {"held": int(test.sum()),
                    "right": round(float(right.float().mean()), 3)}
    HEAD.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "measure": heads["measure"].state_dict(),
                "way": heads["way"].state_dict()}, HEAD / "heads.pt")
    out["examples"] = len(rows)
    print(json.dumps(out))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "train"))
    options = parser.parse_args(argv)
    {"write": write, "check": check, "train": train}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

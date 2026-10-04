"""How well code talk is read: the encoder against the hand rules it
replaces, on held-out messages (`llm/code-talk-data/valid-code.jsonl`:
phrasings and names never taught).

    python -m research.v698.measure_reading [--model llm/reader-code7]

    act       none / ask / make / change / run / teach, as written
    aspect    of the questions (act ask)
    subject   project / file / named / last, of the questions
    span      the subject's words (SUBJ), exactly; the concept and every
              member of what is taught

The hand rules (`asking.route`, `coding.read`) read act and aspect only:
they have no spans and no `teach`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID = ROOT / "llm" / "code-talk-data" / "valid-code.jsonl"


def _hand(text: str) -> tuple:
    """(act, aspect) as the hand rules read a message, with no project and
    no code yet -- as a conversation's first message about code is."""
    from research.v697 import coding
    from research.v697.conversation import Workspace
    from research.v698 import asking
    from research.v698.project import Project
    held = Project("project")
    held.put({"src/index.ts": "export function main(): void {}\n"})
    if coding.read(text).get("asked"):
        return "make", "none"
    aspect, _ = asking.route(text, held, Workspace(), 1, "measure")
    if aspect is not None:
        return "ask", aspect
    return "none", "none"


def spans(words: list, roles: list) -> dict:
    out: dict = {}
    current = None
    for word, role in zip(words, roles):
        if role.startswith("B-"):
            current = role[2:]
            out.setdefault(current, []).append([word])
        elif role.startswith("I-") and current == role[2:]:
            out[current][-1].append(word)
        else:
            current = None
    return {name: [" ".join(one) for one in found]
            for name, found in out.items()}


def measure(limit: int = 0) -> dict:
    from research import encoder
    from research.v698.reading import HEADS
    rows = [json.loads(line) for line in VALID.open(encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    model = encoder.LOADED.get()
    right, seen = Counter(), Counter()
    confused = Counter()
    for at in range(0, len(rows), 64):
        chunk = rows[at:at + 64]
        read = model.guess([one["words"] for one in chunk],
                           [(one["tags"], one["deps"]) for one in chunk],
                           HEADS)
        for row, got in zip(chunk, read):
            act = got["code_act"][0][0]
            seen["act"] += 1
            right["act"] += act == row["act"]
            if act != row["act"]:
                confused[(row["act"], act)] += 1
            text = " ".join(row["words"])
            hand_act, hand_aspect = _hand(text)
            right["hand act"] += hand_act == row["act"] if row["act"] in (
                "none", "ask", "make") else 0
            seen["hand act"] += row["act"] in ("none", "ask", "make")
            if row["act"] == "ask":
                seen["aspect"] += 1
                right["aspect"] += got["code_aspect"][0][0] == row["aspect"]
                right["hand aspect"] += hand_aspect == row["aspect"]
                seen["hand aspect"] += 1
                seen["subject"] += 1
                right["subject"] += got["code_subject"][0][0] == \
                    row["subject"]
            wanted = spans(row["words"], row["roles"])
            found = spans(row["words"], got["code_role"])
            for name in ("SUBJ", "CONCEPT", "MEMBER"):
                if name in wanted:
                    seen[f"span {name}"] += 1
                    right[f"span {name}"] += found.get(name) == wanted[name]
    return {key: f"{right[key]}/{seen[key]} = "
                 f"{right[key] / max(seen[key], 1):.3f}"
            for key in seen} | {"most confused (as written, as read)": [
                f"{a}->{b}: {n}" for (a, b), n in confused.most_common(8)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.model:
        os.environ["V689_READER_MODEL"] = args.model
        import importlib
        from research import encoder
        importlib.reload(encoder)
    print(json.dumps(measure(args.limit), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

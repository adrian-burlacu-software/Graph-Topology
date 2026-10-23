"""WikiTableQuestions as conversations: a table a project holds, asked of in
words (2026-10-23, the core measured beyond its own test).

    python -m research.v704.wtq fetch       data/wtq/{train,test}.jsonl
    python -m research.v704.wtq projects    a project of each table asked
    python -m research.v704.converse --suite wtq ...   the architecture
    python -m research.v704.wtq alone       the code model alone

WikiTableQuestions (Pasupat & Liang 2015, "Compositional Semantic Parsing
on Semi-Structured Tables"; lighteval/wikitablequestions): questions of
Wikipedia tables -- superlatives, counts, comparisons, lookups -- each with
its answers. A question of the test split, SAID of them chosen once by a
seed, is a conversation of one turn (`data-ask`) about a project holding
its table as `data/table.csv`, right where every answer is said.

The mirror's splits are not the original's (its `test` holds 18 486
questions, its `train` 10): what is said is SAID of its `test`, chosen
once by the seed, and read as totals, as the garden fixture is; what may
be taught from is the rest of it, never a question said (`asked()` by id).

**The code model alone** (`alone`): the 3B code model given the table (as
Markdown) and the question, one answer (greedy).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "wtq"
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=lighteval/"
        "wikitablequestions&config=default&split={split}&offset={offset}"
        "&length=100")
#: questions of the test split said, chosen once
SAID, SEED = 200, 7041


def _page(url: str, tries: int = 6) -> dict:
    """A page of rows, asked again a few times, slower each time: the
    datasets server answers 502 now and then over hundreds of pages."""
    import time
    for at in range(tries):
        try:
            request = urllib.request.Request(url,
                                             headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=300) as reply:
                return json.load(reply)
        except (urllib.error.URLError, TimeoutError):
            if at == tries - 1:
                raise
            time.sleep(5 * 2 ** at)
    raise RuntimeError("unreachable")


def fetch() -> dict:
    counts = {}
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        rows, offset, total = [], 0, None
        while total is None or offset < total:
            found = _page(ROWS.format(split=split, offset=offset))
            total = found["num_rows_total"]
            rows += [one["row"] for one in found["rows"]]
            offset += 100
        with (DATA / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({key: row[key] for key in
                                      ("id", "question", "answers", "table")})
                          + "\n")
        counts[split] = len(rows)
    print(json.dumps(counts))
    return counts


def asked() -> list:
    """The test questions said, chosen once by the seed."""
    rows = [json.loads(line) for line in
            (DATA / "test.jsonl").open(encoding="utf-8")]
    return random.Random(SEED).sample(rows, min(SAID, len(rows)))


def _csv(table: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(table["header"])
    writer.writerows(table["rows"])
    return out.getvalue()


def projects() -> int:
    for row in asked():
        folder = DATA / "projects" / row["id"] / "data"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "table.csv").write_text(_csv(row["table"]),
                                          encoding="utf-8")
    print(f"{len(asked())} projects -> {DATA / 'projects'}")
    return len(asked())


def _norm(text: str) -> str:
    text = str(text).lower().replace(",", "")
    text = re.sub(r"(\d)\.0\b", r"\1", text)
    # a full stop is a number's only between digits (`2012-13.` ends a
    # sentence)
    text = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", text)
    return " ".join(re.sub(r"[^\w.%-]+", " ", text).split())


def _check(row: dict):
    answers = [_norm(one) for one in row["answers"]]

    def check(turn, root):
        from research.v704.converse import text_of
        said = f" {_norm(text_of(turn))} "
        return all(f" {one} " in said for one in answers)
    check.about = f"says {', '.join(row['answers'])[:60]}"
    return check


def suite() -> list:
    from research.v704.suite import Conversation
    return [Conversation("data-ask", [row["question"]], _check(row), True,
                         str(DATA / "projects" / row["id"]))
            for row in asked()]


ALONE = ("Here is a table:\n{table}\nQuestion: {question}\nAnswer with the "
         "answer alone.")


def _markdown(table: dict) -> str:
    lines = ["| " + " | ".join(table["header"]) + " |",
             "|" + "---|" * len(table["header"])]
    lines += ["| " + " | ".join(row) + " |" for row in table["rows"]]
    return "\n".join(lines)


def alone(label: str = "alone") -> dict:
    from research.v705 import coder
    model = coder.big()
    out = DATA / f"{label}.jsonl"
    done = {}
    if out.exists():
        done = {one["id"]: one for one in map(json.loads,
                                              out.open(encoding="utf-8"))}
    todo = [row for row in asked() if row["id"] not in done]
    with out.open("a", encoding="utf-8") as stream:
        for at in range(0, len(todo), 4):
            chunk = todo[at:at + 4]
            prompts = [ALONE.format(table=_markdown(row["table"])[:12000],
                                    question=row["question"])
                       for row in chunk]
            replies = model.write(prompts, longest=60)
            for row, reply in zip(chunk, replies):
                reply = reply[0] if isinstance(reply, list) else reply
                right = _check(row)({"reply": {"text": reply}}, None)
                one = {"id": row["id"], "verdict": "right" if right
                       else "wrong", "said": reply[:200]}
                done[row["id"]] = one
                stream.write(json.dumps(one) + "\n")
                stream.flush()
    right = sum(one["verdict"] == "right" for one in done.values())
    print(json.dumps({"right": right, "of": len(done)}))
    return {"right": right, "of": len(done)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "projects", "alone"))
    options = parser.parse_args(argv)
    {"fetch": fetch, "projects": projects, "alone": alone}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

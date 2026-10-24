"""SParC as conversations: questions of a database asked one after another,
each leaning on the last (`of these, which ...`, `what is its country?`)
(2026-10-24, the core measured beyond its own test: follow-ups over data).

    python -m research.v704.sparc fetch       data/sparc/dev.json, databases
    python -m research.v704.sparc projects    a project of each database
    python -m research.v704.converse --suite sparc ...   the architecture
    python -m research.v704.sparc alone       the code model alone, answering
    python -m research.v704.sparc alone-sql   the code model alone, as SParC
                                              asks it: SQL, run

SParC (Yu et al. 2019, "SParC: Cross-Domain Semantic Parsing in Context"):
4 298 sequences of questions over Spider's 200 databases, each turn with
the SQL that answers it. Its dev split (422 sequences over 20 databases) is
what is said, read as totals as the garden fixture is; train is what may be
taught from. Each sequence is a conversation (`follow-up`) about a project
holding its database's tables as `data/<table>.csv`; right where the last
turn says every value of what the last turn's SQL gives -- a sequence whose
last answer is empty, or more than `MOST` values, is left out (said in the
totals), as no reply says it whole.

**The code model alone**: the 3B code model given the tables (as Markdown,
cut at the same length WikiTableQuestions' were) and the questions, one
answer (greedy) -- and, as the benchmark is meant, given the schema and
asked for SQL, which is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sparc"
QUESTIONS = "https://huggingface.co/datasets/aherntech/sparc/resolve/main/{}"
DATABASES = ("https://huggingface.co/datasets/ashishkattamuri/"
             "multispider-processed/resolve/main/databases/{}.sqlite")
#: sequences said, chosen once; the most values a last answer may have
SAID, SEED, MOST = 200, 7051, 10


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=600) as reply:
        return reply.read()


def fetch() -> dict:
    (DATA / "databases").mkdir(parents=True, exist_ok=True)
    for split in ("dev", "train"):
        (DATA / f"{split}.json").write_bytes(_get(QUESTIONS.format(
            f"{split}.json")))
    names = sorted({one["db_id"] for one in _sequences()})
    for name in names:
        path = DATA / "databases" / f"{name}.sqlite"
        if not path.exists():
            path.write_bytes(_get(DATABASES.format(name)))
    counts = {"dev": len(_sequences()), "databases": len(names)}
    print(json.dumps(counts))
    return counts


def _sequences() -> list:
    return json.loads((DATA / "dev.json").read_text(encoding="utf-8"))


def _connect(name: str):
    connection = sqlite3.connect(DATA / "databases" / f"{name}.sqlite")
    connection.text_factory = lambda raw: raw.decode("utf-8", "replace")
    return connection


def _tables(connection) -> list:
    return [one[0] for one in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE "
        "'sqlite_%'")]


def projects() -> int:
    names = sorted({one["db_id"] for one in _sequences()})
    for name in names:
        folder = DATA / "projects" / name / "data"
        folder.mkdir(parents=True, exist_ok=True)
        with _connect(name) as connection:
            for table in _tables(connection):
                cursor = connection.execute(f'SELECT * FROM "{table}"')
                with (folder / f"{table.lower()}.csv").open(
                        "w", encoding="utf-8", newline="") as out:
                    writer = csv.writer(out, lineterminator="\n")
                    writer.writerow([one[0] for one in cursor.description])
                    writer.writerows(cursor)
    print(f"{len(names)} projects -> {DATA / 'projects'}")
    return len(names)


def _values(name: str, query: str) -> list | None:
    """Every distinct value the SQL gives, as said; None where it does not
    run."""
    try:
        with _connect(name) as connection:
            rows = connection.execute(query).fetchall()
    except sqlite3.Error:
        return None
    out = []
    for row in rows:
        for one in row:
            said = _norm(one)
            if said and said not in out:
                out.append(said)
    return out


def _norm(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value if value is not None else "").lower().replace(",", "")
    text = re.sub(r"(\d)\.0\b", r"\1", text)
    text = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", text)
    return " ".join(re.sub(r"[^\w.%-]+", " ", text).split())


def asked() -> tuple[list, dict]:
    """The sequences said, chosen once by the seed, each with what its last
    turn's SQL gives -- and how many were left out, and why."""
    kept, left = [], {"empty": 0, "too many": 0, "does not run": 0}
    for one in _sequences():
        values = _values(one["db_id"], one["final_query"])
        if values is None:
            left["does not run"] += 1
        elif not values:
            left["empty"] += 1
        elif len(values) > MOST:
            left["too many"] += 1
        else:
            kept.append({**one, "values": values})
    for at, one in enumerate(kept):
        one["n"] = at
    return random.Random(SEED).sample(kept, min(SAID, len(kept))), left


def _check(row: dict):
    def check(turn, root):
        from research.v704.converse import text_of
        said = f" {_norm(text_of(turn))} "
        return all(f" {one} " in said for one in row["values"])
    check.about = f"says {', '.join(row['values'])[:60]}"
    return check


def suite() -> list:
    from research.v704.suite import Conversation
    return [Conversation("follow-up", list(row["interaction_utterance"]),
                         _check(row), True,
                         str(DATA / "projects" / row["db_id"]))
            for row in asked()[0]]


#: what the code model alone is given
ALONE = ("Here are the tables of a database:\n{tables}\n{asked}\nAnswer the "
         "last question with the answer alone.")
ALONE_SQL = ("Here is a SQLite database's schema:\n{schema}\n{asked}\nWrite "
             "the SQLite query that answers the last question, in context. "
             "Answer with the query alone.")
#: as long as WikiTableQuestions' tables were cut
LONGEST = 12000


def _markdown(name: str) -> str:
    out = []
    with _connect(name) as connection:
        for table in _tables(connection):
            cursor = connection.execute(f'SELECT * FROM "{table}"')
            header = [one[0] for one in cursor.description]
            out.append(f"Table {table}:\n| " + " | ".join(header) + " |\n|" +
                       "---|" * len(header))
            out += ["| " + " | ".join(str(one) for one in row) + " |"
                    for row in cursor]
    return "\n".join(out)[:LONGEST]


def _schema(name: str) -> str:
    with _connect(name) as connection:
        return "\n".join(one[0] for one in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT "
            "NULL"))


def _asked(row: dict) -> str:
    return "\n".join(f"Question {at + 1}: {one}" for at, one in
                     enumerate(row["interaction_utterance"]))


def _query(reply: str) -> str:
    reply = re.sub(r"```(?:sql|sqlite)?", "", reply)
    found = re.search(r"\bSELECT\b.*", reply, flags=re.I | re.S)
    return (found.group(0) if found else reply).split(";")[0].strip()


def alone(sql: bool = False) -> dict:
    from research.v705 import coder
    model = coder.big()
    label = "alone-sql" if sql else "alone"
    out = DATA / f"{label}.jsonl"
    done = {}
    if out.exists():
        done = {one["n"]: one for one in map(json.loads,
                                             out.open(encoding="utf-8"))}
    todo = [row for row in asked()[0] if row["n"] not in done]
    with out.open("a", encoding="utf-8") as stream:
        for at in range(0, len(todo), 4):
            chunk = todo[at:at + 4]
            prompts = [ALONE_SQL.format(schema=_schema(row["db_id"]),
                                        asked=_asked(row)) if sql else
                       ALONE.format(tables=_markdown(row["db_id"]),
                                    asked=_asked(row)) for row in chunk]
            replies = model.write(prompts, longest=160 if sql else 60)
            for row, reply in zip(chunk, replies):
                reply = reply[0] if isinstance(reply, list) else reply
                said = reply
                if sql:
                    values = _values(row["db_id"], _query(reply))
                    said = " ".join(values or ())
                right = _check(row)({"reply": {"text": said}}, None)
                one = {"n": row["n"], "verdict": "right" if right
                       else "wrong", "said": reply[:200]}
                done[row["n"]] = one
                stream.write(json.dumps(one) + "\n")
                stream.flush()
    right = sum(one["verdict"] == "right" for one in done.values())
    print(json.dumps({"right": right, "of": len(done)}))
    return {"right": right, "of": len(done)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "projects", "alone",
                                        "alone-sql", "asked"))
    options = parser.parse_args(argv)
    if options.job == "asked":
        said, left = asked()
        print(json.dumps({"said": len(said), "left out": left}))
        return 0
    {"fetch": fetch, "projects": projects, "alone": alone,
     "alone-sql": lambda: alone(sql=True)}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

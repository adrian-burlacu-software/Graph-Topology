"""ConceptNet's actions, kept as the phrases they were said in.

The store maps every ConceptNet node to a noun sense, which suits things
and loses doings: `climb MotivatedByGoal get over fence` is kept there as
*ascent* motivated by the goal, and `climb HasPrerequisite get ladder` as
something about ascent. Here the edges between doings are kept as said,
from ConceptNet 5.7's own assertions, English at both ends:

    MotivatedByGoal     climb          -> get over fence   done for it
    HasPrerequisite     climb          -> get ladder        needed first
    HasFirstSubevent    climb          -> get good hand hold
    HasSubevent         climb          -> grab
    HasLastSubevent     climb          -> reach top
    CapableOf           goat           -> jump fence        who does it
    NotCapableOf        humans         -> lay eggs
    UsedFor             ladder         -> climb             what for

Each end is also kept normalised (`norm`): lowercased, articles dropped,
each word by WordNet's morphology -- the first as a verb where it is one --
so `jumping over the fences` and `jump over fence` meet.

    python -m research.v695.mined           build data/v695_actions.sqlite
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "conceptnet-assertions-5.7.0.csv.gz"
PATH = ROOT / "data" / "v695_actions.sqlite"

RELATIONS = ("MotivatedByGoal", "HasPrerequisite", "HasFirstSubevent",
             "HasSubevent", "HasLastSubevent", "CapableOf", "NotCapableOf",
             "UsedFor")

#: Words that are not part of what a phrase names.
ARTICLES = frozenset({"a", "an", "the", "your", "his", "her", "their",
                      "one's", "someone's", "my", "its"})

SCHEMA = """
CREATE TABLE edges (head TEXT, relation TEXT, tail TEXT, weight REAL,
                    head_norm TEXT, tail_norm TEXT);
CREATE INDEX edges_head ON edges(head_norm, relation);
CREATE INDEX edges_tail ON edges(tail_norm, relation);
CREATE TABLE words (word TEXT, edge INTEGER, side TEXT);
CREATE INDEX words_word ON words(word, side);
"""

_MORPHY: dict = {}


def _morphy(word: str, verb: bool) -> str:
    key = (word, verb)
    if key not in _MORPHY:
        try:
            from nltk.corpus import wordnet
            found = ""
            if verb:
                found = wordnet.morphy(word, wordnet.VERB) or ""
            found = found or wordnet.morphy(word, wordnet.NOUN) or word
        except Exception:                          # noqa: BLE001
            found = word
        _MORPHY[key] = found
    return _MORPHY[key]


def norm(phrase: str) -> str:
    """`jumping over the fences` -> `jump over fence`."""
    words = [one for one in phrase.lower().replace("_", " ").split()
             if one not in ARTICLES]
    return " ".join(_morphy(word, index == 0)
                    for index, word in enumerate(words))


def _node(uri: str) -> str:
    return uri.split("/")[3].replace("_", " ")


def build(source: Path = SOURCE, path: Path = PATH) -> int:
    """Read the English action edges out of ConceptNet into `path`."""
    rows = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split("\t")
            relation = parts[1][3:]
            if relation not in RELATIONS:
                continue
            if not (parts[2].startswith("/c/en/")
                    and parts[3].startswith("/c/en/")):
                continue
            head, tail = _node(parts[2]), _node(parts[3])
            try:
                weight = float(json.loads(parts[4]).get("weight", 1.0))
            except ValueError:
                weight = 1.0
            rows.append((head, relation, tail, weight, norm(head),
                         norm(tail)))
    fresh = path.with_suffix(".tmp")
    if fresh.exists():
        fresh.unlink()
    connection = sqlite3.connect(fresh)
    connection.executescript(SCHEMA)
    connection.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)",
                           rows)
    words = []
    for edge, (head_norm, tail_norm) in enumerate(
            connection.execute("SELECT head_norm, tail_norm FROM edges "
                               "ORDER BY rowid"), start=1):
        for side, said in (("head", head_norm), ("tail", tail_norm)):
            for word in set(said.split()):
                words.append((word, edge, side))
    connection.executemany("INSERT INTO words VALUES (?, ?, ?)", words)
    connection.commit()
    connection.close()
    if path.exists():
        path.unlink()
    fresh.rename(path)
    return len(rows)


_CONNECTION = None


def connection():
    global _CONNECTION
    if _CONNECTION is None:
        if not PATH.exists():
            return None
        _CONNECTION = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True,
                                      check_same_thread=False)
    return _CONNECTION


def edges(relation: str, head: str = "", tail: str = "") -> list:
    """(head, tail, weight) for edges whose normalised ends are these."""
    found = connection()
    if found is None:
        return []
    where, args = ["relation = ?"], [relation]
    if head:
        where.append("head_norm = ?")
        args.append(norm(head))
    if tail:
        where.append("tail_norm = ?")
        args.append(norm(tail))
    return found.execute(
        f"SELECT head, tail, weight FROM edges WHERE {' AND '.join(where)} "
        f"ORDER BY weight DESC", args).fetchall()


def mentioning(word: str, side: str, relations=RELATIONS) -> list:
    """(head, relation, tail, weight) for edges with this word on `side`."""
    found = connection()
    if found is None:
        return []
    marks = ",".join("?" for _ in relations)
    return found.execute(
        f"SELECT e.head, e.relation, e.tail, e.weight FROM words w JOIN "
        f"edges e ON e.rowid = w.edge WHERE w.word = ? AND w.side = ? AND "
        f"e.relation IN ({marks})",
        [_morphy(word, False), side, *relations]).fetchall()


def main() -> int:
    count = build()
    print(f"{count} edges -> {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

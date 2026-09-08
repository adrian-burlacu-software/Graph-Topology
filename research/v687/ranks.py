"""WordNet's own sense order, recorded per lemma.

Run: python -m research.v687.ranks

The store records, for each lemma, which senses it can mean and which one the
build's evidence chose. It never recorded **where in WordNet's own list for
that lemma** each sense sits, and that is a different thing from the number in
the synset id: `hog.n.03` is WordNet's *first* sense of the word "pig" and its
third of the word "hog". Without the per-lemma rank there is no way to ask
whether a reading is the obvious one.

It is not a usefulness ranking and this does not make it the default. The two
disagree in both directions:

    hammer   WordNet first is the part of a gunlock; the build chose the tool
    drink    WordNet first is the drink; the build chose beverage.n.01
    seal     WordNet first is sealing wax; the build chose the animal
    pig      WordNet first is domestic swine; the build chose a foundry mould
    mouse    WordNet first is the animal; the build chose the device

The build is right about the first three and wrong about the last two, and no
threshold on the rank separates those cases -- `seal` and `pig` are both the
build's fourth choice. So the rank is stored as evidence rather than used as
an override, and what reads it is v688, which says out loud when an answer
rests on a word's fifth reading.

This is a backfill, not a rebuild: WordNet is local and the mapping is exact,
so it costs a couple of minutes rather than an afternoon. `build.py` writes
the column too, so a fresh store has it without this being run.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

from . import build

#: What a lemma's rank is when WordNet does not list the sense under it at
#: all. Sorting puts these last without a special case.
UNRANKED = 99


def wordnet_order() -> dict[str, list[str]]:
    """Every lemma WordNet knows, and its senses in WordNet's own order."""
    from nltk.corpus import wordnet

    order: dict[str, list[str]] = {}
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ").lower()
            order.setdefault(word, [])
    for word in list(order):
        order[word] = [sense.name().replace("_", " ")
                       for sense in wordnet.synsets(word.replace(" ", "_"))]
    return order


def backfill(store: Path) -> dict[str, int]:
    """Add `sense_rank` to `lemmas` and fill it. Safe to run twice."""
    connection = sqlite3.connect(store)
    connection.row_factory = sqlite3.Row
    columns = {row["name"] for row in
               connection.execute("PRAGMA table_info(lemmas)")}
    if "sense_rank" not in columns:
        connection.execute("ALTER TABLE lemmas ADD COLUMN sense_rank "
                           f"INTEGER NOT NULL DEFAULT {UNRANKED}")

    order = wordnet_order()
    updates: list[tuple[int, str, str]] = []
    unknown = 0
    for row in connection.execute("SELECT lemma, concept FROM lemmas"):
        senses = order.get(row["lemma"])
        if not senses:
            unknown += 1
            continue
        try:
            rank = senses.index(row["concept"])
        except ValueError:
            unknown += 1
            continue
        updates.append((rank, row["lemma"], row["concept"]))

    connection.executemany(
        "UPDATE lemmas SET sense_rank = ? WHERE lemma = ? AND concept = ?",
        updates)
    connection.commit()
    counts = {"ranked": len(updates), "unranked": unknown}
    connection.close()
    return counts


def main() -> None:
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else build.DEFAULT_STORE
    if not store.exists():
        raise SystemExit(f"no store at {store}")
    started = time.time()
    counts = backfill(store)
    print(f"ranked {counts['ranked']:,} lemma/sense pairs, "
          f"{counts['unranked']:,} not listed by WordNet "
          f"[{time.time() - started:.0f}s]")


if __name__ == "__main__":
    main()

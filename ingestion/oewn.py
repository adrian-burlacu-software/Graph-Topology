"""Open English WordNet: the definitions it changed, read into definitions memory.

    python -m ingestion.oewn changes   # count and list what changed
    python -m ingestion.oewn read      # re-read the changed glosses

The store is Princeton WordNet 3.0 (117,659 synsets). Open English WordNet is
its maintained successor, and its 2025 edition renumbers every synset, so
the two are joined through sense keys: each OEWN noun sense carries one
(`dog%1:05:00::`), and WordNet 3.0 resolves it (`dog.n.01`). A synset takes
the 3.0 synset most of its sense keys resolve to.

Of 71,864 noun synsets, 67,223 have the same definition, 2,803 are new and
have no 3.0 sense -- nothing in the store to hang them on -- and 1,838 changed.
Many of those changes are punctuation: `etc` to `etc.`, backticks to curly
quotes. Only a change in the words is re-read, into the same definitions
memory the WordNet glosses went into, marked `oewn`.

Downloaded from the `2025-edition` release of
https://github.com/globalwordnet/english-wordnet (CC BY 4.0) into
`data/oewn/`.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import string
import time
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = REPOSITORY_ROOT / "data" / "oewn" / "english-wordnet-2025-json.zip"
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"
CHANGES = REPOSITORY_ROOT / "data" / "oewn" / "changed-noun-definitions.jsonl"


def _words(text: str) -> list[str]:
    """A definition as its words, without punctuation or quote styles."""
    table = str.maketrans({character: " " for character in
                           string.punctuation + "‘’“”"})
    return (text or "").lower().translate(table).split()


def changes() -> dict:
    from nltk.corpus import wordnet

    archive = zipfile.ZipFile(ARCHIVE)
    synsets: dict = {}
    for name in archive.namelist():
        if name.startswith("noun."):
            synsets.update(json.loads(archive.read(name)))
    votes: dict = collections.defaultdict(collections.Counter)
    for name in archive.namelist():
        if not name.startswith("entries"):
            continue
        for by_pos in json.loads(archive.read(name)).values():
            for sense in (by_pos.get("n") or {}).get("sense", []):
                try:
                    old = wordnet.lemma_from_key(sense["id"]).synset().name()
                except Exception:                   # noqa: BLE001
                    continue
                votes[sense["synset"]][old] += 1
    store = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    glosses = dict(store.execute(
        "SELECT id, definition FROM concepts WHERE pos = 'n'"))
    counts: collections.Counter = collections.Counter()
    rows = []
    for oewn_id, synset in synsets.items():
        definition = "; ".join(synset.get("definition") or [])
        if oewn_id not in votes:
            counts["new"] += 1
            continue
        old = votes[oewn_id].most_common(1)[0][0].replace("_", " ")
        if old not in glosses:
            counts["not in store"] += 1
            continue
        if _words(definition) == _words(glosses[old]):
            counts["same"] += 1
            continue
        counts["changed"] += 1
        rows.append({"synset": old, "oewn": oewn_id,
                     "wordnet": glosses[old], "definition": definition})
    CHANGES.write_text("".join(json.dumps(row) + "\n" for row in rows),
                       encoding="utf-8")
    return {"noun_synsets": len(synsets), **dict(counts)}


def read() -> dict:
    from research.v689.definitions import DefinitionMemory
    from research.v689.learn_definitions import MEMORY, _reader

    reader = _reader()
    memory = DefinitionMemory(MEMORY)
    rows = [json.loads(line) for line in
            CHANGES.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.time()
    before = after = 0
    readings = []
    for row in rows:
        entry = memory.entry(row["synset"])
        before += len(entry["facts"]) if entry else 0
        found = reader.read(row["synset"], row["definition"])
        after += len(found.facts)
        readings.append(found)
    memory.keep_all(readings, "oewn")
    return {"re_read": len(readings), "facts_before": before,
            "facts_after": after, "seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("changes", "read"))
    options = parser.parse_args(argv)
    result = changes() if options.stage == "changes" else read()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

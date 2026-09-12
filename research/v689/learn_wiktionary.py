"""Wiktionary's English noun senses, matched to the store's synsets and read
into a definitions memory of their own.

    python -m research.v689.learn_wiktionary read   [--workers 4] [--limit N]
    python -m research.v689.learn_wiktionary report [--sample 60]
    python -m ingestion.load --source definitions --into data/v684_wiktionary.sqlite \\
        --extra state/v689-wiktionary.sqlite:wiktionary

WordNet's glosses are written about a synset. Wiktionary's are written about
a word: `bass` has a fish, a voice and an instrument, in its own order and its
own splits, and nothing says which of the store's `bass` synsets a sense is.
A sense read onto the wrong synset is exactly the over-affirmation the audit
counts, so a sense is kept only when the taxonomy picks one synset for it:

- the gloss is read by `GlossReader`, which finds its broader kind: `A
  freshwater fish ...` -> `fish`;
- a candidate is a noun synset of the word in the store, not an instance;
- a candidate agrees when a noun sense of the broader kind is one of its
  ancestors, and that sense is not so general that half the taxonomy is under
  it (`organism`, `whole`, `abstraction`: more than `GENERAL` descendants);
- exactly one candidate agrees. None, or two, and the sense is left;
- and no other sense of the same word is matched to that synset. Two are two
  meanings Wiktionary tells apart and WordNet does not -- `homophobe` is
  someone prejudiced, someone who fears sameness and someone who fears men --
  and nothing says which one the synset is.

Senses marked figurative, slang, historical or offensive are left before
reading: they describe a use of the word, not the thing. What a matched sense
says is filtered as a Wikipedia lead's is (`articles.is_property`), and a fact
with a date or a lone initial in it is history or a species list. Facts
WordNet's own gloss already gave the synset are dropped, so what this memory
holds is what Wiktionary adds, and the audit measures that.

A run replaces the memory it writes; a `--limit` run should be given a
`--memory` of its own. The senses are `data/wiktionary/english-nouns.jsonl`,
English noun senses filtered from kaikki.org's extract of Wiktionary
(CC BY-SA 4.0).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .articles import is_property
from .definitions import DefinitionMemory, Reading
from .learn_definitions import reading_of

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"
SENSES = REPOSITORY_ROOT / "data" / "wiktionary" / "english-nouns.jsonl"
MEMORY = REPOSITORY_ROOT / "state" / "v689-wiktionary.sqlite"
DEFINITIONS = REPOSITORY_ROOT / "state" / "v689-definitions.sqlite"

#: A broader kind with more descendants than this is over too much of the
#: taxonomy to tell senses apart: `organism` has 19,447, `whole` 31,542.
#: `person` (10,296) and `artifact` (10,698) still do.
GENERAL = 12_000

#: Senses about a use of the word rather than the thing it names.
USES = frozenset({"figuratively", "broadly", "idiomatic", "humorous",
                  "rhetoric", "euphemistic", "metonymically", "by-extension",
                  "ironic", "sarcastic", "derogatory", "offensive", "slur",
                  "ethnic", "vulgar", "pejorative", "slang", "Internet",
                  "historical", "dated", "obsolete", "archaic", "nonstandard",
                  "neologism", "poetic"})

#: Glosses that point at another entry instead of defining one.
POINTER = re.compile(
    r"^(?:(?:alternative|archaic|obsolete|dated|rare|common|nonstandard|"
    r"informal|eye dialect)\s+)?(?:form|spelling|synonym|plural|abbreviation|"
    r"initialism|acronym|misspelling|clipping|ellipsis|contraction|"
    r"diminutive|short)\s+(?:form\s+)?(?:of|for)\b", re.IGNORECASE)

#: `proposed by Clark Kerr in the 1960s`: history, not a property.
DATED = re.compile(r"\b(?:1\d{3}|20\d{2})s?\b|\bcentur(?:y|ies)\b")

#: `P. dominica`: a species list the parse took for a part.
INITIAL = re.compile(r"(?:^|\s)[a-z]\.(?:\s|$)")

#: Senses a worker aligns between reports.
BATCH = 500


def cleaned(gloss: str) -> str:
    """A Wiktionary gloss in the shape WordNet writes one: no parentheses,
    no full stop, and a lower-case first word unless it is an acronym."""
    text = re.sub(r"\s*\([^()]*\)", "", gloss or "")
    text = " ".join(text.split()).strip().rstrip(".").strip()
    first = text.split(" ", 1)[0] if text else ""
    if first[:1].isupper() and first[1:] == first[1:].lower():
        text = text[:1].lower() + text[1:]
    return text


def usable(row: dict) -> str | None:
    """Why a sense is left before it is read, or None."""
    word = row.get("word") or ""
    if word != word.lower():
        return "proper noun"
    if USES & set(row.get("tags") or ()):
        return "a use of the word"
    gloss = cleaned(row.get("gloss", ""))
    if len(gloss.split()) < 2 or POINTER.match(gloss):
        return "no definition"
    return None


#: A lead files a thing -- `placed in the family`, `divided into` -- with
#: verbs a definition uses for what the thing does: `used to treat malaria`,
#: `divides in two`, `lists words`. Only naming is never a property.
NAMING = frozenset({"call", "name", "know", "refer", "abbreviate", "spell"})


def kept(reader, fact) -> bool:
    """Is this fact a property of the kind, rather than prose about it?"""
    return (is_property(reader, fact, NAMING)
            and not DATED.search(fact.object)
            and not INITIAL.search(fact.object))


def _descendants(reasoner, concept: str) -> int:
    row = reasoner.connection.execute(
        "SELECT descendants FROM concepts WHERE id = ?", (concept,)).fetchone()
    return row[0] if row else 0


def under(reader, concept: str, genus: str) -> bool:
    """Is a specific enough noun sense of `genus` above `concept`?"""
    senses = reader._senses(genus, "n")
    reasoner = reader.asker.reasoner
    return bool(senses) and any(
        node in senses and _descendants(reasoner, node) <= GENERAL
        for node, distance, _ in reasoner.ascend(concept) if distance)


def align(reader, word: str, gloss: str, candidates) -> dict:
    """Which one of `candidates` the gloss defines, and what it says."""
    kinds = [one for one in candidates if not reader._instance(one)]
    if not kinds:
        return {"status": "instances only"}
    found = reader.read(kinds[0], gloss, check=False)
    if not found.genus:
        return {"status": "no genus"}
    agreeing = [one for one in kinds if under(reader, one, found.genus)]
    if not agreeing:
        return {"status": "no sense agrees"}
    if len(agreeing) > 1:
        return {"status": "ambiguous"}
    found.concept, found.agrees = agreeing[0], True
    found.facts = [fact for fact in found.facts if kept(reader, fact)]
    return {"status": "aligned", "word": word, "reading": found.as_dict()}


def _align_batch(batch: list) -> list:
    from .learn_definitions import _reader

    reader = _reader()
    out = []
    for word, gloss, candidates in batch:
        try:
            out.append(align(reader, word, gloss, candidates))
        except Exception as bad:                    # noqa: BLE001
            out.append({"status": "error",
                        "error": f"{type(bad).__name__}: {bad}"})
    return out


def noun_senses(store: Path = STORE) -> dict:
    """word -> every noun synset the store files it under."""
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    found: dict = collections.defaultdict(list)
    for lemma, concept in connection.execute(
            "SELECT l.lemma, l.concept FROM lemmas l JOIN concepts c "
            "ON c.id = l.concept WHERE c.pos = 'n' ORDER BY 1, 2"):
        found[lemma].append(concept)
    connection.close()
    return found


def already_defined(path: Path = DEFINITIONS) -> set:
    """(concept, relation, object) WordNet's own glosses already gave."""
    if not Path(path).exists():
        return set()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = set(connection.execute(
        "SELECT concept, relation, object FROM defined"))
    connection.close()
    return rows


def several(aligned: list) -> set:
    """(word, synset) pairs two or more senses of the word were matched to."""
    pairs = collections.Counter((word, found.concept)
                                for word, found in aligned)
    return {pair for pair, count in pairs.items() if count > 1}


def merge(aligned: list, known: set) -> list:
    """One reading per synset from the (word, reading) pairs matched to it:
    none from a word that matched it twice, and without the facts the
    synset's WordNet gloss already gave."""
    doubled = several(aligned)
    by_concept: dict = {}
    for word, found in aligned:
        if (word, found.concept) in doubled:
            continue
        kept_reading = by_concept.get(found.concept)
        if kept_reading is None:
            kept_reading = by_concept[found.concept] = Reading(
                found.concept, found.gloss, found.genus, True, [], [])
        else:
            kept_reading.gloss += " | " + found.gloss
        seen = {(fact.relation, fact.object) for fact in kept_reading.facts}
        kept_reading.facts.extend(
            fact for fact in found.facts
            if (fact.relation, fact.object) not in seen
            and (found.concept, fact.relation, fact.object) not in known)
        kept_reading.unread.extend(found.unread)
    return list(by_concept.values())


def read(workers: int = 4, limit: int = 0, memory_path: Path = MEMORY,
         senses_path: Path = SENSES) -> dict:
    started = time.time()
    senses = noun_senses()
    known = already_defined()
    counts: collections.Counter = collections.Counter()
    todo = []
    with Path(senses_path).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["senses"] += 1
            why = usable(row)
            candidates = senses.get(row["word"]) if why is None else None
            if why is None and not candidates:
                why = "no noun in the store"
            if why:
                counts[why] += 1
                continue
            todo.append((row["word"], cleaned(row["gloss"]), candidates))
    if limit and limit < len(todo):
        stride = len(todo) / limit
        todo = [todo[int(index * stride)] for index in range(limit)]
    batches = [todo[index:index + BATCH]
               for index in range(0, len(todo), BATCH)]
    aligned, done, errors = [], 0, []
    reading_started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for results in pool.map(_align_batch, batches):
            for result in results:
                counts[result["status"]] += 1
                if "reading" in result:
                    aligned.append((result["word"],
                                    reading_of(result["reading"])))
                elif "error" in result and len(errors) < 5:
                    errors.append(result["error"])
            done += len(results)
            elapsed = time.time() - reading_started
            print(f"[wiktionary] {done:,}/{len(todo):,} senses, "
                  f"{counts['aligned']:,} aligned, "
                  f"{done / max(elapsed, 1e-9):.0f}/s", flush=True)
    doubled = several(aligned)
    counts["several senses, one synset"] = sum(
        (word, found.concept) in doubled for word, found in aligned)
    merged = merge(aligned, known)
    memory = DefinitionMemory(memory_path)
    with memory.lock, memory.connection:
        memory.connection.execute("DELETE FROM glosses")
        memory.connection.execute("DELETE FROM defined")
    memory.keep_all(merged, "wiktionary")
    return {**dict(counts), "read": len(todo), "synsets": len(merged),
            "facts_read": sum(len(found.facts) for _, found in aligned),
            "facts_new": sum(len(one.facts) for one in merged),
            "errors_seen": errors, "seconds": round(time.time() - started, 1),
            "workers": workers}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("read", "report"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--memory", type=Path, default=MEMORY)
    parser.add_argument("--senses", type=Path, default=SENSES)
    options = parser.parse_args(argv)
    if options.stage == "read":
        result = read(options.workers, options.limit, options.memory,
                      options.senses)
    else:
        from .learn_definitions import report

        result = report(options.sample, options.memory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

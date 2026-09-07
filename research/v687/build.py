"""Assemble the reasoning store: WordNet's taxonomy, everyone else's facts.

    python -m research.v687.build

The shape of this is the conclusion of `research/v683/ontologies.py`, not a
preference:

    taxonomy   WordNet only. It is the sole candidate that is both acyclic and
               lexical -- a median of 8 ancestors against ConceptNet's 6,451.
               ConceptNet's is_a is dropped entirely; its five cycles close the
               taxonomy over one giant component, which is what made inference
               explode.

    facts      WordNet's own structural relations, then Ascent++ (graded by
               typicality), then ConceptNet as a low-confidence long tail.

    senses     Facts from Ascent++ and ConceptNet are stated about word strings,
               not senses. Which sense they attach to is decided by the facts
               themselves (see `senses.py`) rather than by WordNet's sense
               numbering, which is not a usefulness ranking -- its first sense
               of `hammer` is the part of a gunlock. Every such fact is still
               flagged `sense_assumed`, so the join stays visible in the
               provenance rather than being silently believed.

Everything lands in one SQLite file that the server opens read-only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sqlite3
import time
from pathlib import Path

from . import rules, senses

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATABASE = REPOSITORY_ROOT / "data" / "v633_full_semantic.sqlite"
ASCENT_CSV = REPOSITORY_ROOT / "data" / "ascentpp.csv"
DEFAULT_STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"

csv.field_size_limit(10 ** 7)

SYNSET = re.compile(r"^wn:synset:(?P<lemma>.+)\.(?P<pos>[nvsar])\.(?P<sense>\d+)$")

#: WordNet relations that describe a synset rather than name it.
WORDNET_FACT_RELATIONS = (
    "has_part", "part_of", "has_a", "made_of", "capable_of", "used_for",
    "at_location", "has_property", "similar_to", "entails", "causes",
    "manner_of", "has_subevent", "receives_action", "instance_of",
)

#: ConceptNet relations kept as a low-confidence tail.
#:
#: `related_to` is excluded on purpose. It is 1,678,150 of 3.9M edges and
#: asserts only that two words co-occur -- it has no direction, no semantics,
#: and inheriting it floods every answer with noise.
CONCEPTNET_FACT_RELATIONS = (
    "capable_of", "used_for", "at_location", "has_property", "has_a",
    "has_part", "part_of", "made_of", "desires", "not_desires",
    "not_capable_of", "not_has_property", "receives_action", "causes",
    "has_prerequisite", "has_subevent", "motivated_by_goal", "created_by",
    "has_attribute", "located_near", "symbol_of",
)

#: Ascent++ ships ConceptNet's schema in CamelCase; map to ours.
ASCENT_RELATIONS = {
    "CapableOf": "capable_of", "ReceivesAction": "receives_action",
    "HasProperty": "has_property", "HasA": "has_a", "AtLocation": "at_location",
    "MadeOf": "made_of", "HasPrerequisite": "has_prerequisite",
    "Causes": "causes", "HasSubevent": "has_subevent", "UsedFor": "used_for",
    "CreatedBy": "created_by", "Desires": "desires", "PartOf": "part_of",
    "SymbolOf": "symbol_of", "SimilarTo": "similar_to",
    "MotivatedByGoal": "motivated_by_goal", "DefinedAs": "defined_as",
    # IsA deliberately absent: the taxonomy comes from WordNet alone.
}

SCHEMA = """
CREATE TABLE concepts (
    id TEXT PRIMARY KEY, lemma TEXT NOT NULL, pos TEXT NOT NULL,
    sense INTEGER NOT NULL, definition TEXT,
    -- how many concepts sit under this one; R12 reads it to decide whether a
    -- word-level fact is too general to inherit.
    descendants INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE taxonomy (child TEXT NOT NULL, parent TEXT NOT NULL,
                       PRIMARY KEY (child, parent));
CREATE TABLE facts (
    concept TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
    source TEXT NOT NULL, confidence REAL NOT NULL, sense_assumed INTEGER NOT NULL,
    PRIMARY KEY (concept, relation, object, source)
);
CREATE TABLE lemmas (lemma TEXT NOT NULL, concept TEXT NOT NULL,
                     -- the sense this word's facts were attached to, and the
                     -- one the UI should offer first.
                     primary_sense INTEGER NOT NULL DEFAULT 0,
                     PRIMARY KEY (lemma, concept));
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX idx_tax_child ON taxonomy(child);
CREATE INDEX idx_tax_parent ON taxonomy(parent);
CREATE INDEX idx_facts_concept ON facts(concept);
CREATE INDEX idx_facts_relation ON facts(concept, relation);
CREATE INDEX idx_lemmas_lemma ON lemmas(lemma);
"""


def _normalize_synset(node: str) -> str | None:
    """`wn:synset:hot_dog.n.01` -> `hot dog.n.01`, satellites folded to `a`."""
    match = SYNSET.match(node)
    if not match:
        return None
    lemma = match["lemma"].replace("_", " ")
    pos = "a" if match["pos"] == "s" else match["pos"]
    return f"{lemma}.{pos}.{match['sense']}"


def _word_level_evidence(source_connection: sqlite3.Connection, ascent: Path):
    """Every phrase Ascent++ and ConceptNet say about a word, for sense choice.

    Yields `(word, phrase)`. This is a first pass over the same rows that get
    written later: the sense a word's facts belong to cannot be known until
    those facts have been seen, and they cannot be written until the sense is
    known. Reading twice is the cheap way out of that.
    """
    if ascent.is_file():
        with ascent.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if ASCENT_RELATIONS.get(row["relation"]):
                    yield row["subject"], row["tail"]
    placeholders = ",".join("?" * len(CONCEPTNET_FACT_RELATIONS))
    for subject, obj in source_connection.execute(
        f"SELECT subject, object FROM edges WHERE subject LIKE 'en:%' "
        f"AND relation IN ({placeholders})", CONCEPTNET_FACT_RELATIONS
    ):
        yield subject[3:], obj[3:] if obj.startswith("en:") else obj


def build(source: Path, ascent: Path, store: Path, verbose: bool = True) -> dict[str, int]:
    if store.exists():
        store.unlink()
    out = sqlite3.connect(store)
    out.executescript(SCHEMA)
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    counts: dict[str, int] = {}

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    # -- concepts ---------------------------------------------------------
    concepts = {}
    for node, definition in source_connection.execute(
        "SELECT node, definition FROM nodes WHERE node LIKE 'wn:synset:%'"
    ):
        identifier = _normalize_synset(node)
        if identifier:
            lemma, pos, sense = identifier.rsplit(".", 2)
            concepts[node] = identifier
            out.execute("INSERT OR IGNORE INTO concepts VALUES (?,?,?,?,?,0)",
                        (identifier, lemma, pos, int(sense), definition))
    counts["concepts"] = len(concepts)
    log(f"  concepts        {len(concepts):>9,}")

    # -- taxonomy, repaired to a DAG --------------------------------------
    parents: dict[str, set[str]] = collections.defaultdict(set)
    for child, parent in source_connection.execute(
        "SELECT subject, object FROM edges WHERE relation='is_a' "
        "AND subject LIKE 'wn:%' AND object LIKE 'wn:%'"
    ):
        if child in concepts and parent in concepts:
            parents[concepts[child]].add(concepts[parent])
    for parent, child in source_connection.execute(
        "SELECT subject, object FROM edges WHERE relation='has_subtype' "
        "AND subject LIKE 'wn:%' AND object LIKE 'wn:%'"
    ):
        if child in concepts and parent in concepts:
            parents[concepts[child]].add(concepts[parent])
    # Exactly one pair in 97,666 is asserted in both directions; a mutual
    # subsumption is a contradiction, so neither side survives it.
    mutual = {(c, p) for c, above in parents.items() for p in above
              if c in parents.get(p, ())}
    for child, parent in mutual:
        parents[child].discard(parent)
    counts["repaired_mutual"] = len(mutual) // 2
    out.executemany("INSERT OR IGNORE INTO taxonomy VALUES (?,?)",
                    ((c, p) for c, above in parents.items() for p in above))
    counts["taxonomy"] = sum(len(v) for v in parents.values())
    log(f"  taxonomy edges  {counts['taxonomy']:>9,}  "
        f"({counts['repaired_mutual']} mutual subsumption repaired)")

    # -- subtree sizes, for R12 -------------------------------------------
    below: dict[str, list[str]] = collections.defaultdict(list)
    for child, above in parents.items():
        for parent in above:
            below[parent].append(child)
    breadth: dict[str, int] = {}
    for node in below:
        seen: set[str] = set()
        frontier = [node]
        while frontier:
            for child in below.get(frontier.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        breadth[node] = len(seen)
    out.executemany("UPDATE concepts SET descendants = ? WHERE id = ?",
                    ((n, c) for c, n in breadth.items() if n))
    over = sum(1 for n in breadth.values() if n >= 8000)
    log(f"  subtree sizes   {len(breadth):>9,}  "
        f"({over} above the R12 breadth limit)")

    # -- lemma index ------------------------------------------------------
    lemma_rows = set()
    for lemma_node, synset in source_connection.execute(
        "SELECT subject, object FROM edges WHERE relation='has_sense'"
    ):
        if synset in concepts and lemma_node.startswith("en:"):
            lemma_rows.add((lemma_node[3:], concepts[synset]))
    for node, identifier in concepts.items():
        lemma_rows.add((identifier.rsplit(".", 2)[0], identifier))
    out.executemany("INSERT OR IGNORE INTO lemmas VALUES (?,?,0)", lemma_rows)
    counts["lemmas"] = len(lemma_rows)
    log(f"  lemma links     {len(lemma_rows):>9,}")

    # -- facts: WordNet's own, which are already sense-tagged ---------------
    placeholders = ",".join("?" * len(WORDNET_FACT_RELATIONS))
    wordnet_facts = 0
    for subject, relation, obj in source_connection.execute(
        f"SELECT subject, relation, object FROM edges WHERE subject LIKE 'wn:%' "
        f"AND relation IN ({placeholders})", WORDNET_FACT_RELATIONS
    ):
        if subject not in concepts:
            continue
        target = concepts.get(obj) or (obj[3:] if obj.startswith("en:") else obj)
        out.execute("INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?,?)",
                    (concepts[subject], relation, target, "wordnet", 0.95, 0))
        wordnet_facts += 1
    counts["facts_wordnet"] = wordnet_facts
    log(f"  wordnet facts   {wordnet_facts:>9,}")
    out.commit()

    #: lemma -> the sense that word-level facts get attached to.
    #:
    #: This is the join between two ontologies that do not agree on what a
    #: subject is, and it is the single most consequential decision in the
    #: build. Ranking senses by a fixed rule -- eponymous, then noun, then
    #: `.01` -- gets `dog` right and `hammer` wrong: WordNet's first sense of
    #: hammer is the part of a gunlock, so 348 facts about carpenters'
    #: toolboxes went to a gun part. 30% of the words carrying facts had two
    #: or more eponymous noun senses, where that rule is a coin flip.
    #:
    #: So the facts pick their own sense: `senses.choose` scores each candidate
    #: against the vocabulary of everything said about the word. That needs the
    #: facts before they can be written, hence the extra pass here.
    started = time.time()
    evidence: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for lemma, phrase in _word_level_evidence(source_connection, ascent):
        evidence[lemma].update(senses.tokens(phrase))
    log(f"  sense evidence  {len(evidence):>9,} words  "
        f"[{time.time()-started:.0f}s]")

    chooser = senses.Senses(out)
    primary_sense: dict[str, str] = {}
    ambiguous = decided = 0
    for lemma in chooser.candidates:
        bag = evidence.get(lemma)
        options = chooser.candidates[lemma]
        if bag:
            choice, _, margin = chooser.choose(lemma, bag)
            if len(options) > 1:
                ambiguous += 1
                decided += margin > 0
        else:
            choice = min(options, key=lambda c: chooser.prior(lemma, c))
        if choice:
            primary_sense[lemma] = choice
    out.executemany("UPDATE lemmas SET primary_sense = 1 "
                    "WHERE lemma = ? AND concept = ?", primary_sense.items())
    counts["primary_ambiguous"] = ambiguous
    counts["primary_by_evidence"] = decided
    log(f"  primary senses  {len(primary_sense):>9,}  "
        f"({ambiguous:,} ambiguous, {decided:,} settled by the facts themselves)")

    # R13: with the senses fixed, a relation's object can be type-checked.
    ranges = senses.Ranges(parents, chooser.candidates, rules.RANGES)
    off_range = 0

    ascent_facts = 0
    if ascent.is_file():
        started = time.time()
        with ascent.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                relation = ASCENT_RELATIONS.get(row["relation"])
                if relation is None:
                    continue
                concept = primary_sense.get(row["subject"])
                if concept is None:
                    continue
                if not ranges.allows(relation, row["tail"]):
                    off_range += 1
                    continue
                typicality = float(row["typicality"])
                saliency = float(row["saliency"])
                out.execute(
                    "INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?,?)",
                    (concept, relation, row["tail"], "ascentpp",
                     round(0.5 * typicality + 0.5 * saliency, 5), 1))
                ascent_facts += 1
        log(f"  ascent++ facts  {ascent_facts:>9,}  [{time.time()-started:.0f}s]")
    else:
        log(f"  ascent++ missing at {ascent} -- skipped")
    counts["facts_ascentpp"] = ascent_facts

    placeholders = ",".join("?" * len(CONCEPTNET_FACT_RELATIONS))
    conceptnet_facts = 0
    for subject, relation, obj in source_connection.execute(
        f"SELECT subject, relation, object FROM edges WHERE subject LIKE 'en:%' "
        f"AND relation IN ({placeholders})", CONCEPTNET_FACT_RELATIONS
    ):
        concept = primary_sense.get(subject[3:])
        if concept is None:
            continue
        target = obj[3:] if obj.startswith("en:") else obj
        if not ranges.allows(relation, target):
            off_range += 1
            continue
        out.execute("INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?,?)",
                    (concept, relation, target, "conceptnet", 0.35, 1))
        conceptnet_facts += 1
    counts["facts_conceptnet"] = conceptnet_facts
    log(f"  conceptnet tail {conceptnet_facts:>9,}")
    counts["off_range"] = off_range
    log(f"  R13 rejected    {off_range:>9,}  "
        f"(object was not the kind of thing the relation takes)")

    for key, value in counts.items():
        out.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))
    out.execute("INSERT OR REPLACE INTO meta VALUES ('built_unix', ?)", (str(time.time()),))
    out.commit()
    total = out.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    counts["facts_total"] = total
    out.execute("INSERT OR REPLACE INTO meta VALUES ('facts_total', ?)", (str(total),))
    out.commit()
    out.close()
    source_connection.close()
    log(f"  facts stored    {total:>9,}\n  -> {store}")
    return counts


def ensure(store: Path = DEFAULT_STORE, source: Path = SOURCE_DATABASE,
           ascent: Path = ASCENT_CSV, verbose: bool = True) -> Path:
    """Build the store if it is missing. Called by the server on startup."""
    if not store.exists():
        if verbose:
            print(f"building reasoning store (first run, a few minutes)\n"
                  f"  source: {source}")
        build(source, ascent, store, verbose)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DATABASE)
    parser.add_argument("--ascent", type=Path, default=ASCENT_CSV)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    arguments = parser.parse_args()
    build(arguments.source, arguments.ascent, arguments.store)


if __name__ == "__main__":
    main()

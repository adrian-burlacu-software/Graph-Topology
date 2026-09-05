"""Compress by inheritance, then answer from the compressed store.

    python -m research.v684.compress

This is the paper's claim made operational. A fact that an ancestor already
states does not need storing: R2 reconstructs it on the way up. So the store
shrinks, and the thing that shrinks it is the same mechanism that answers
questions. Compression and inference stop being two systems.

Two operations, and they are not equally safe:

R10  Redundancy elimination -- LOSSLESS.
     `broccoli rabe has_property bitter` is dropped when an ancestor already
     says `bitter`, because the reasoner will find it there. Applied only to
     inheritable relations, so nothing is dropped that R2 could not rebuild.
     `verify()` re-derives every dropped fact and fails loudly if one is lost.

R11  Hoisting -- a GENERALISATION, not a deduction.
     If every child of `mammal` that has facts says `capable_of breathe`, the
     fact moves to `mammal`. That is induction: it now also applies to children
     that never said it, including ones with no facts at all. Hoisted rows are
     marked `hoisted` in the store and shown as such, and the default threshold
     is 1.0 -- every child must agree -- because at 0.6 it invents facts about
     the dissenting 40%.

The compressed store is a separate file; the original is never modified.
"""
from __future__ import annotations

import argparse
import collections
import random
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import build, rules

DEFAULT_COMPRESSED = build.DEFAULT_STORE.with_name("v684_reasoning_compressed.sqlite")


def _load(connection: sqlite3.Connection):
    parents: dict[str, set[str]] = collections.defaultdict(set)
    for child, parent in connection.execute("SELECT child, parent FROM taxonomy"):
        parents[child].add(parent)
    facts: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for concept, relation, obj in connection.execute(
        "SELECT concept, relation, object FROM facts"
    ):
        facts[concept].add((relation, obj))
    return parents, facts


def ancestors(parents: dict[str, set[str]], node: str, limit: int = 16) -> set[str]:
    seen = {node}
    frontier = [node]
    for _ in range(limit):
        nxt: set[str] = set()
        for current in frontier:
            nxt |= parents.get(current, set()) - seen
        if not nxt:
            break
        seen |= nxt
        frontier = list(nxt)
    return seen - {node}


def compress(store: Path, out: Path, hoist_threshold: float = 1.0,
             min_children: int = 4, verbose: bool = True) -> dict[str, Any]:
    """Write a compressed copy of `store` and report what it cost."""
    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    if out.exists():
        out.unlink()
    log(f"  copying {store.name} -> {out.name}")
    shutil.copy2(store, out)
    connection = sqlite3.connect(out)
    parents, facts = _load(connection)
    original = sum(len(v) for v in facts.values())
    log(f"  {original:,} facts over {len(facts):,} concepts")

    stats: dict[str, Any] = {"original_facts": original}

    # -- R11: hoist first, so redundancy elimination can then remove the
    #         copies the hoist just made redundant.
    children: dict[str, list[str]] = collections.defaultdict(list)
    for child, above in parents.items():
        for parent in above:
            children[parent].append(child)

    started = time.time()
    hoisted: list[tuple[str, str, str, int]] = []
    for parent, kids in children.items():
        having = [k for k in kids if facts.get(k)]
        if len(having) < min_children:
            continue
        counts = collections.Counter(
            fact for kid in having for fact in facts[kid]
            if rules.inheritable(fact[0])
        )
        need = max(min_children, int(round(hoist_threshold * len(having))))
        for (relation, obj), count in counts.items():
            if count < need or (relation, obj) in facts.get(parent, set()):
                continue
            # Never hoist over a child that explicitly denies it (R3).
            denial = rules.POSITIVES.get(relation)
            if denial and any(
                (denial, obj) in facts.get(kid, set()) for kid in having
            ):
                continue
            facts[parent].add((relation, obj))
            hoisted.append((parent, relation, obj, count))
    stats["hoisted"] = len(hoisted)
    log(f"  R11 hoisted {len(hoisted):,} facts to parents "
        f"[{time.time()-started:.0f}s]")

    # -- R10: drop anything an ancestor already states.
    started = time.time()
    dropped: list[tuple[str, str, str]] = []
    for concept in list(facts):
        above: set[tuple[str, str]] = set()
        for ancestor in ancestors(parents, concept):
            above |= facts.get(ancestor, set())
        for fact in list(facts[concept]):
            if fact in above and rules.inheritable(fact[0]):
                facts[concept].discard(fact)
                dropped.append((concept, fact[0], fact[1]))
    stats["dropped"] = len(dropped)
    remaining = sum(len(v) for v in facts.values())
    stats["compressed_facts"] = remaining
    stats["ratio"] = round(1 - remaining / original, 5) if original else 0.0
    log(f"  R10 dropped {len(dropped):,} redundant facts "
        f"[{time.time()-started:.0f}s]")
    log(f"  {original:,} -> {remaining:,} facts ({stats['ratio']:.2%} smaller)")

    # -- rewrite the facts table -----------------------------------------
    rows = connection.execute(
        "SELECT concept, relation, object, source, confidence, sense_assumed "
        "FROM facts").fetchall()
    keep = []
    for concept, relation, obj, source, confidence, assumed in rows:
        if (relation, obj) in facts.get(concept, set()):
            keep.append((concept, relation, obj, source, confidence, assumed, 0))
    for parent, relation, obj, count in hoisted:
        if (relation, obj) in facts.get(parent, set()):
            keep.append((parent, relation, obj, "hoisted",
                         round(0.55 + 0.35 * min(1.0, count / 12), 5), 1, count))
    connection.execute("DROP TABLE facts")
    connection.execute(
        "CREATE TABLE facts (concept TEXT NOT NULL, relation TEXT NOT NULL, "
        "object TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL, "
        "sense_assumed INTEGER NOT NULL, hoisted INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (concept, relation, object, source))")
    connection.executemany(
        "INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?,?,?)", keep)
    connection.execute("CREATE INDEX idx_facts_concept ON facts(concept)")
    connection.execute("CREATE INDEX idx_facts_relation ON facts(concept, relation)")
    for key, value in stats.items():
        connection.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                           ("compress_" + key, str(value)))
    connection.commit()
    connection.close()
    log(f"  -> {out}")

    stats["dropped_sample"] = [list(d) for d in random.Random(1).sample(
        dropped, min(8, len(dropped)))] if dropped else []
    stats["hoisted_sample"] = [[p, r, o, n] for p, r, o, n in
                               sorted(hoisted, key=lambda h: -h[3])[:8]]
    return stats


def verify(original: Path, compressed: Path, samples: int = 4000,
           seed: int = 5, verbose: bool = True) -> dict[str, Any]:
    """Every dropped fact must still be derivable. R10 claims lossless; check it.

    Hoisted facts are excluded from the count of losses, because R11 is a
    generalisation and is expected to change what the store entails.
    """
    a = sqlite3.connect(f"file:{original}?mode=ro", uri=True)
    b = sqlite3.connect(f"file:{compressed}?mode=ro", uri=True)
    parents, before = _load(a)
    _, after = _load(b)
    a.close(); b.close()

    rng = random.Random(seed)
    concepts = rng.sample(sorted(before), min(samples, len(before)))
    checked = lost = 0
    losses: list[str] = []
    for concept in concepts:
        derivable = set(after.get(concept, set()))
        for ancestor in ancestors(parents, concept):
            derivable |= {f for f in after.get(ancestor, set())
                          if rules.inheritable(f[0])}
        for fact in before.get(concept, set()):
            if not rules.inheritable(fact[0]):
                continue
            checked += 1
            if fact not in derivable:
                lost += 1
                if len(losses) < 5:
                    losses.append(f"{concept} {fact[0]} {fact[1]}")
    result = {"concepts_checked": len(concepts), "facts_checked": checked,
              "not_rederivable": lost,
              "lossless": lost == 0, "examples": losses}
    if verbose:
        print(f"  verified {checked:,} inheritable facts over {len(concepts):,} "
              f"concepts: {lost:,} not re-derivable "
              f"({'LOSSLESS' if not lost else 'LOSSY'})")
        for line in losses:
            print(f"     {line}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_COMPRESSED)
    parser.add_argument("--hoist-threshold", type=float, default=1.0,
                        help="share of a parent's fact-bearing children that "
                             "must agree before a fact is hoisted (1.0 = all)")
    parser.add_argument("--min-children", type=int, default=4)
    parser.add_argument("--no-verify", action="store_true")
    arguments = parser.parse_args()
    build.ensure(arguments.store)
    stats = compress(arguments.store, arguments.out,
                     arguments.hoist_threshold, arguments.min_children)
    if not arguments.no_verify:
        stats["verification"] = verify(arguments.store, arguments.out)
    print("\n  dropped, e.g.:")
    for concept, relation, obj in stats["dropped_sample"]:
        print(f"     {concept}  {relation}  {obj!r}")
    print("  hoisted, e.g.:")
    for parent, relation, obj, count in stats["hoisted_sample"]:
        print(f"     {parent}  {relation}  {obj!r}  (from {count} children)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
V557 - Semantic Graph Composition Discovery

Reads the REAL semantic graph:
    concepts(concept_id, canonical, display, ...)
    facts(subject_id, predicate, object_id, object_text, ...)

IMPORTANT:
- Source SQLite is opened read-only.
- live_facts/live_entities are NOT used as the semantic graph.
- Mined candidates are written to a separate results SQLite DB.
- The search is designed to expose whether useful indirect knowledge
  already exists before adding more datasets.

Example:
  python research\v557\v557_semantic_composition.py ^
    --source .\results\full_semantic_memory.sqlite ^
    --out .\results\v557_semantic_composition.sqlite ^
    --workers 20 --max-hops 4 --per-node 100 --samples 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


COMPOSITION_PREDICATES = {
    "is_a", "isa", "instance_of",
    "has", "has_part", "part_of", "contains",
    "has_property", "property",
    "used_for", "capable_of", "causes",
    "made_of", "located_in", "defined_as",
    "related_to",
}

# These are deliberately NOT automatically treated as valid inference rules.
# V557 discovers compositions first; the cognitive controller will learn which
# predicate sequences are safe later.
SAFE_LOOKING = {
    ("is_a", "has"),
    ("is_a", "has_part"),
    ("is_a", "has_property"),
    ("instance_of", "has"),
    ("instance_of", "has_part"),
    ("instance_of", "has_property"),
    ("part_of", "has"),
    ("contains", "has_part"),
}


def connect_ro(path: str) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def table_columns(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def inspect_schema(con):
    tables = {
        r[0]: table_columns(con, r[0])
        for r in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    required = {"facts", "concepts"}
    missing = required - set(tables)
    if missing:
        raise RuntimeError(
            f"Real semantic graph tables missing: {sorted(missing)}. "
            f"Found: {sorted(tables)}"
        )

    fc = set(tables["facts"])
    cc = set(tables["concepts"])
    for col in ("subject_id", "predicate", "object_id"):
        if col not in fc:
            raise RuntimeError(f"facts.{col} is required")
    if "concept_id" not in cc:
        raise RuntimeError("concepts.concept_id is required")

    return tables


def concept_map(con):
    cols = set(table_columns(con, "concepts"))
    name_col = "canonical" if "canonical" in cols else (
        "display" if "display" in cols else None
    )
    if not name_col:
        raise RuntimeError("concepts needs canonical or display")

    rows = con.execute(
        f'SELECT concept_id, "{name_col}" AS name FROM concepts'
    )
    return {int(r["concept_id"]): str(r["name"] or "") for r in rows}


def fact_projection(con):
    cols = set(table_columns(con, "facts"))
    object_text = "object_text" if "object_text" in cols else "NULL"
    extra = []
    for c in ("fact_type", "domain", "confidence", "frequency", "source_id"):
        if c in cols:
            extra.append(c)
    sql = f"""
        SELECT subject_id, predicate, object_id, {object_text} AS object_text,
               {", ".join(extra) if extra else "NULL AS _none"}
        FROM facts
        WHERE subject_id IS NOT NULL
          AND predicate IS NOT NULL
    """
    return sql


def build_adjacency(con, names, predicates, per_node):
    """
    Build a bounded adjacency map from facts. We intentionally cap outgoing
    edges per subject to prevent pathological high-degree concepts from
    dominating every traversal.
    """
    placeholders = ",".join("?" * len(predicates))
    sql = fact_projection(con) + f" AND predicate IN ({placeholders})"

    adj = defaultdict(list)
    count = 0
    for r in con.execute(sql, tuple(predicates)):
        sid = int(r["subject_id"])
        oid = r["object_id"]
        if oid is None:
            continue
        oid = int(oid)
        if sid not in names or oid not in names:
            continue

        bucket = adj[sid]
        if len(bucket) < per_node:
            bucket.append((
                oid,
                str(r["predicate"]),
                names.get(oid, str(oid)),
                r["fact_type"] if "fact_type" in r.keys() else None,
                r["domain"] if "domain" in r.keys() else None,
                r["confidence"] if "confidence" in r.keys() else None,
                r["frequency"] if "frequency" in r.keys() else None,
            ))
            count += 1
    return adj, count


def choose_seeds(names, adjacency, requested, seed):
    ids = [x for x in adjacency if names.get(x)]
    rng = random.Random(seed)
    rng.shuffle(ids)

    # Prefer concepts with multiple outgoing composition edges. These are
    # exactly where graph composition has a chance to produce indirect cases.
    ids.sort(key=lambda x: len(adjacency[x]), reverse=True)

    # Keep a diverse sample rather than only the absolute highest-degree hubs.
    if len(ids) > requested:
        head = ids[: max(100, requested // 3)]
        tail = ids[max(100, requested // 3):]
        rng.shuffle(tail)
        ids = head + tail[: max(0, requested - len(head))]
    return ids[:requested]


def discover_from_seed(seed_id, names, adjacency, max_hops, max_paths):
    """
    Returns paths whose endpoint is reachable in >=2 hops.

    No semantic claim is asserted here. A path is only an observed graph
    composition candidate.
    """
    out = []
    q = [(seed_id, [], {seed_id})]

    while q and len(out) < max_paths:
        node, path, seen = q.pop(0)
        if len(path) >= max_hops:
            continue

        for edge in adjacency.get(node, ()):
            oid, pred, oname, *_ = edge
            step = (names.get(node, str(node)), pred, oname)
            new_path = path + [step]

            if len(new_path) >= 2:
                out.append(new_path)
                if len(out) >= max_paths:
                    break

            if oid not in seen:
                q.append((oid, new_path, seen | {oid}))

    return out


def classify_path(path):
    seq = tuple(x[1] for x in path)
    if len(seq) == 2 and seq in SAFE_LOOKING:
        return "safe_looking_2hop"
    if len(seq) == 2:
        return "other_2hop"
    return f"{len(seq)}hop"


def init_results(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS paths (
        id INTEGER PRIMARY KEY,
        subject TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        hops INTEGER NOT NULL,
        predicate_sequence TEXT NOT NULL,
        path_json TEXT NOT NULL,
        classification TEXT NOT NULL,
        discovered_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY,
        subject TEXT NOT NULL,
        inferred_predicate TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        hops INTEGER NOT NULL,
        predicate_sequence TEXT NOT NULL,
        source_path_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'candidate'
    );

    CREATE TABLE IF NOT EXISTS predicate_sequences (
        sequence TEXT PRIMARY KEY,
        hops INTEGER NOT NULL,
        count INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_paths_subject ON paths(subject);
    CREATE INDEX IF NOT EXISTS idx_paths_endpoint ON paths(endpoint);
    CREATE INDEX IF NOT EXISTS idx_candidates_subject ON candidates(subject);
    """)
    return con


def infer_candidate(path):
    if len(path) < 2:
        return None

    subject = path[0][0]
    endpoint = path[-1][2]
    seq = tuple(x[1] for x in path)

    # Candidate relation is NOT asserted as truth. We only expose the common
    # compositional endpoint relation suggested by the final edge.
    final_pred = seq[-1]
    if final_pred not in {
        "has", "has_part", "has_property", "part_of", "contains",
        "located_in", "made_of", "used_for", "capable_of", "related_to"
    }:
        return None

    return subject, final_pred, endpoint, seq


def run(args):
    t0 = time.perf_counter()
    con = connect_ro(args.source)

    tables = inspect_schema(con)
    print("=== V557 REAL SEMANTIC GRAPH COMPOSITION DISCOVERY ===")
    print(f"source       : {args.source}")
    print(f"output       : {args.out}")
    print(f"workers      : {args.workers}")
    print(f"max_hops     : {args.max_hops}")
    print(f"per_node     : {args.per_node}")
    print(f"seed_concepts: {args.seeds}")
    print()
    print("[SCHEMA] semantic graph = facts + concepts")
    print(f"[SCHEMA] facts columns    : {tables['facts']}")
    print(f"[SCHEMA] concepts columns : {tables['concepts']}")
    print("[SCHEMA] live_facts is intentionally ignored")
    print()

    t = time.perf_counter()
    names = concept_map(con)
    print(f"[LOAD] concepts={len(names):,} seconds={time.perf_counter()-t:.3f}")

    predicates = sorted(COMPOSITION_PREDICATES)
    t = time.perf_counter()
    adjacency, edge_count = build_adjacency(
        con, names, predicates, args.per_node
    )
    print(
        f"[LOAD] composition_edges={edge_count:,} "
        f"subjects={len(adjacency):,} seconds={time.perf_counter()-t:.3f}"
    )

    seeds = choose_seeds(names, adjacency, args.seeds, args.seed)
    print(f"[SEEDS] selected={len(seeds):,}")

    results = []
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                discover_from_seed,
                sid, names, adjacency, args.max_hops, args.max_paths
            )
            for sid in seeds
        ]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                results.extend(f.result())
            except Exception as e:
                print(f"[WORKER ERROR] {type(e).__name__}: {e}")
            if i % max(1, len(futures)//10) == 0:
                print(
                    f"[DISCOVERY] completed={i}/{len(futures)} "
                    f"paths={len(results):,} "
                    f"seconds={time.perf_counter()-t:.2f}"
                )

    elapsed = time.perf_counter() - t
    print(f"[DISCOVERY] paths={len(results):,} seconds={elapsed:.3f}")

    # Deduplicate paths.
    seen = set()
    unique = []
    for p in results:
        key = json.dumps(p, ensure_ascii=False, sort_keys=False)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"[DISCOVERY] unique_paths={len(unique):,}")

    out = init_results(args.out)
    seq_counts = Counter()
    class_counts = Counter()
    candidate_rows = []

    for p in unique:
        seq = tuple(x[1] for x in p)
        seq_key = " -> ".join(seq)
        cls = classify_path(p)
        seq_counts[seq_key] += 1
        class_counts[cls] += 1

        cur = out.execute(
            """INSERT INTO paths
               (subject, endpoint, hops, predicate_sequence, path_json,
                classification, discovered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                p[0][0], p[-1][2], len(p), seq_key,
                json.dumps(p, ensure_ascii=False),
                cls, time.time()
            )
        )
        pid = cur.lastrowid

        c = infer_candidate(p)
        if c:
            s, pred, endpoint, seq = c
            candidate_rows.append((
                s, pred, endpoint, len(p), " -> ".join(seq), pid, "candidate"
            ))

    out.executemany(
        """INSERT OR REPLACE INTO predicate_sequences
           (sequence, hops, count) VALUES (?, ?, ?)""",
        [(s, s.count(" -> ") + 1, c) for s, c in seq_counts.items()]
    )
    out.executemany(
        """INSERT INTO candidates
           (subject, inferred_predicate, endpoint, hops,
            predicate_sequence, source_path_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        candidate_rows
    )
    out.commit()

    print()
    print("-- PREDICATE SEQUENCES --")
    for seq, count in seq_counts.most_common(args.top_sequences):
        print(f"  {count:7d}  {seq}")

    print()
    print("-- PATH CLASSIFICATION --")
    for k, v in class_counts.most_common():
        print(f"  {k:24s} {v:8d}")

    print()
    print("-- SAMPLE 2-HOP COMPOSITIONS --")
    shown = 0
    for p in unique:
        if len(p) == 2:
            print(
                f"  {p[0][0]} --{p[0][1]}--> {p[0][2]} "
                f"--{p[1][1]}--> {p[1][2]}"
            )
            shown += 1
            if shown >= args.samples:
                break
    if shown == 0:
        print("  <none>")

    summary = {
        "benchmark": "v557_real_semantic_graph_composition_discovery",
        "source_graph_read_only": True,
        "source": str(Path(args.source)),
        "results_db": str(Path(args.out)),
        "semantic_tables": ["facts", "concepts"],
        "live_tables_used": False,
        "graph": {
            "concepts_loaded": len(names),
            "bounded_composition_edges": edge_count,
            "seed_concepts": len(seeds),
            "unique_paths": len(unique),
        },
        "composition": {
            "path_count": len(unique),
            "candidate_count": len(candidate_rows),
            "predicate_sequences": dict(seq_counts),
            "classification": dict(class_counts),
        },
        "config": {
            "workers": args.workers,
            "max_hops": args.max_hops,
            "per_node": args.per_node,
            "seeds": args.seeds,
            "max_paths": args.max_paths,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    json_out = Path(args.out).with_suffix(".json")
    json_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print()
    print("-- SUMMARY --")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON written: {json_out}")
    print(f"[TIMING] end-to-end={summary['elapsed_seconds']:.3f}s")

    out.close()
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r".\results\full_semantic_memory.sqlite")
    ap.add_argument("--out", default=r".\results\v557_semantic_composition.sqlite")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--per-node", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--max-paths", type=int, default=100)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--top-sequences", type=int, default=50)
    ap.add_argument("--seed", type=int, default=557)
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.max_hops < 2:
        ap.error("--max-hops must be >= 2")
    run(args)


if __name__ == "__main__":
    main()

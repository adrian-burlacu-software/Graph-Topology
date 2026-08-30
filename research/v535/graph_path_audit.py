from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEMANTIC_RELATIONS = {
    "has", "has_a", "hasa", "has_part", "part_of", "is_a", "isa",
    "capable_of", "used_for", "causes", "causes_desire", "at_location",
    "located_in", "located_near", "made_of", "receives_action",
    "has_prerequisite", "has_first_subevent", "has_last_subevent",
    "motivated_by_goal", "has_property", "defined_as", "related_to",
    "similar_to", "synonym", "antonym",
}
LEXICAL_RELATIONS = {"hypernym", "hyponym", "synonym", "antonym"}
BLOCKED_RELATIONS = {
    "in_domain","domain","source","provenance","dataset","node_type","type","label",
    "nsubj","nsubjpass","obj","dobj","iobj","ccomp","xcomp","amod","advmod","nmod",
    "obl","oblique","root","dep","aux","auxpass","cop","det","case","mark","punct",
    "conj","cc","compound","appos","acl","advcl",
}


def norm(s: str | None) -> str:
    return str(s or "").strip().lower().replace("_", " ")


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-65536")
    return con


def batched(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def outgoing_batch(
    con: sqlite3.Connection,
    subjects: list[str],
    include_lexical: bool,
    per_node: int,
):
    """Fetch edges for many subjects in as few SQL calls as possible."""
    if not subjects:
        return {}

    out = defaultdict(list)
    # SQLite has a variable-number limit; keep batches conservative.
    for group in batched(subjects, 250):
        placeholders = ",".join("?" for _ in group)
        sql = f"""
            SELECT
                c.canonical AS subject,
                f.predicate,
                COALESCE(o.canonical, f.object_text) AS object_text,
                f.fact_type,
                f.confidence,
                f.frequency,
                f.answerable
            FROM facts f
            JOIN concepts c ON c.concept_id=f.subject_id
            LEFT JOIN concepts o ON o.concept_id=f.object_id
            WHERE lower(c.canonical) IN ({placeholders})
              AND f.answerable=1
            ORDER BY lower(c.canonical), f.confidence DESC, f.frequency DESC
        """
        rows = con.execute(sql, tuple(group)).fetchall()
        per_subject = defaultdict(int)
        for row in rows:
            subject = norm(row["subject"])
            pred = norm(row["predicate"])
            if pred in BLOCKED_RELATIONS:
                continue
            if not include_lexical and pred in LEXICAL_RELATIONS:
                continue
            obj = norm(row["object_text"])
            if not obj:
                continue
            if per_subject[subject] >= per_node:
                continue
            out[subject].append(dict(row))
            per_subject[subject] += 1
    return out


def expand_frontier(con, frontier, include_lexical, per_node, workers):
    if not frontier:
        return {}, 0.0
    start = time.perf_counter()
    frontier = list(frontier)
    if workers <= 1 or len(frontier) < 32:
        result = outgoing_batch(con, frontier, include_lexical, per_node)
        return result, time.perf_counter() - start

    # Separate read-only connections allow concurrent SQLite reads.
    chunks = [chunk for chunk in batched(frontier, max(32, len(frontier) // workers)) if chunk]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_expand_chunk, con.database if hasattr(con, "database") else None, chunk, include_lexical, per_node) for chunk in chunks]
        merged = defaultdict(list)
        for fut in as_completed(futures):
            part = fut.result()
            for key, rows in part.items():
                merged[key].extend(rows)
    elapsed = time.perf_counter() - start
    return dict(merged), elapsed


def _expand_chunk(memory_path, chunk, include_lexical, per_node):
    # `sqlite3.Connection.database` is not portable, so this helper is only used
    # when a path has been installed on the connection by our wrapper below.
    con = connect(memory_path)
    try:
        return outgoing_batch(con, chunk, include_lexical, per_node)
    finally:
        con.close()


def db_path_connection(path: str):
    con = connect(path)
    return con


def find_paths(con, memory_path, start, target, max_depth=4, include_lexical=False, max_paths=10, per_node=500, workers=1, label="search"):
    start, target = norm(start), norm(target)
    if not start or not target:
        return [], {"depths": [], "expanded": 0, "edges_examined": 0, "seconds": 0.0}
    if start == target:
        return [[{"subject": start, "predicate": "IDENTITY", "object_text": target}]], {"depths": [], "expanded": 0, "edges_examined": 0, "seconds": 0.0}

    started = time.perf_counter()
    frontier = {start}
    parents = {start: None}
    parent_edge = {}
    depth_seen = {start: 0}
    metrics = {"depths": [], "expanded": 0, "edges_examined": 0}

    for depth in range(max_depth):
        t0 = time.perf_counter()
        rows_by_subject = outgoing_batch(con, list(frontier), include_lexical, per_node)
        sql_seconds = time.perf_counter() - t0
        next_frontier = set()
        found_nodes = []
        edges_examined = 0

        for subject in frontier:
            for edge in rows_by_subject.get(subject, []):
                edges_examined += 1
                obj = norm(edge["object_text"])
                if not obj or obj in depth_seen:
                    continue
                depth_seen[obj] = depth + 1
                parents[obj] = subject
                parent_edge[obj] = edge
                if obj == target:
                    found_nodes.append(obj)
                elif depth + 1 < max_depth:
                    next_frontier.add(obj)

        metrics["expanded"] += len(frontier)
        metrics["edges_examined"] += edges_examined
        metrics["depths"].append({
            "depth": depth + 1,
            "frontier_nodes": len(frontier),
            "next_frontier_nodes": len(next_frontier),
            "edges_examined": edges_examined,
            "sql_seconds": sql_seconds,
            "layer_seconds": time.perf_counter() - t0,
        })
        print(f"[{label}] depth={depth+1} frontier={len(frontier)} edges={edges_examined} sql={sql_seconds:.3f}s total_layer={time.perf_counter()-t0:.3f}s", flush=True)

        if found_nodes:
            paths = []
            for node in found_nodes[:max_paths]:
                rev = []
                cur = node
                while cur != start:
                    edge = parent_edge[cur]
                    rev.append({
                        "subject": edge["subject"],
                        "predicate": edge["predicate"],
                        "object_text": edge["object_text"],
                        "fact_type": edge.get("fact_type"),
                        "confidence": edge.get("confidence"),
                    })
                    cur = parents[cur]
                paths.append(list(reversed(rev)))
            metrics["seconds"] = time.perf_counter() - started
            return paths, metrics
        frontier = next_frontier
        if not frontier:
            break

    metrics["seconds"] = time.perf_counter() - started
    return [], metrics


def summarize_paths(paths):
    if not paths:
        return None
    scored = []
    for path in paths:
        lexical = sum(1 for e in path if norm(e.get("predicate")) in LEXICAL_RELATIONS)
        scored.append((lexical, len(path), path))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0]


def audit_pair(memory_path, start, target, max_depth, workers, per_node):
    con = connect(memory_path)
    try:
        print(f"[TIMING] semantic search begin workers={workers} per_node={per_node}", flush=True)
        t0 = time.perf_counter()
        sem_paths, sem_metrics = find_paths(con, memory_path, start, target, max_depth, False, workers=workers, per_node=per_node, label="semantic")
        sem_time = time.perf_counter() - t0
        print(f"[TIMING] semantic search end seconds={sem_time:.3f}", flush=True)

        print(f"[TIMING] lexical-inclusive search begin", flush=True)
        t1 = time.perf_counter()
        all_paths, all_metrics = find_paths(con, memory_path, start, target, max_depth, True, workers=workers, per_node=per_node, label="all")
        all_time = time.perf_counter() - t1
        print(f"[TIMING] lexical-inclusive search end seconds={all_time:.3f}", flush=True)
    finally:
        con.close()

    best_sem = summarize_paths(sem_paths)
    best_all = summarize_paths(all_paths)
    if best_sem:
        classification = "semantic_path_exists"
    elif best_all:
        classification = "lexical_or_noisy_path_only"
    else:
        classification = "no_path"

    return {
        "start": norm(start),
        "target": norm(target),
        "max_depth": max_depth,
        "classification": classification,
        "semantic_path_count": len(sem_paths),
        "all_path_count": len(all_paths),
        "best_semantic_path": best_sem[2] if best_sem else None,
        "best_any_path": best_all[2] if best_all else None,
        "timing": {"semantic_seconds": sem_time, "all_seconds": all_time, "total_seconds": sem_time + all_time},
        "semantic_metrics": sem_metrics,
        "all_metrics": all_metrics,
    }


def print_path(path):
    for edge in path or []:
        print(f"  {edge['subject']} --{edge['predicate']}--> {edge['object_text']}")


def run(args):
    print("=== V535 GRAPH PATH / KNOWLEDGE AUDIT ===")
    print(f"memory      : {Path(args.memory).resolve()}")
    print(f"depth       : {args.depth}")
    print(f"workers     : {args.workers}")
    print(f"per_node    : {args.per_node}")
    print("LLM         : NOT USED")
    print("conversation: NOT USED")
    print("algorithm   : batched SQLite frontier traversal")
    print()

    if not (args.start and args.target):
        raise SystemExit("V535 currently audits --start + --target pairs.")

    overall = time.perf_counter()
    report = audit_pair(args.memory, args.start, args.target, args.depth, args.workers, args.per_node)

    print()
    print(f"CLASSIFICATION: {report['classification']}")
    print(f"TOTAL TIME    : {report['timing']['total_seconds']:.3f}s")
    print()
    print("-- BEST SEMANTIC PATH --")
    print_path(report["best_semantic_path"]) if report["best_semantic_path"] else print("  <none>")
    print()
    print("-- BEST PATH INCLUDING LEXICAL RELATIONS --")
    print_path(report["best_any_path"]) if report["best_any_path"] else print("  <none>")
    print()
    print("-- DEPTH METRICS: SEMANTIC --")
    for row in report["semantic_metrics"]["depths"]:
        print(" ", row)
    print("-- DEPTH METRICS: ALL --")
    for row in report["all_metrics"]["depths"]:
        print(" ", row)

    print()
    print("-- DIAGNOSIS --")
    if report["classification"] == "semantic_path_exists":
        print("  Useful semantic structure exists; retrieval/query planning is the next suspect.")
    elif report["classification"] == "lexical_or_noisy_path_only":
        print("  A connection exists only through lexical/noisy relations; normalization is likely needed.")
    else:
        print("  No path found at this depth; likely a knowledge-coverage problem.")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON written: {args.json}")
    print(f"[TIMING] end-to-end={time.perf_counter()-overall:.3f}s")


def main():
    ap = argparse.ArgumentParser(description="Instrumented SQLite semantic graph path audit")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1, help="Parallel read workers (batched traversal is used regardless)")
    ap.add_argument("--per-node", type=int, default=300, help="Outgoing edge cap per node")
    ap.add_argument("--json")
    args = ap.parse_args()
    if not 0 <= args.depth <= 8:
        ap.error("--depth must be between 0 and 8")
    if not 1 <= args.workers <= 16:
        ap.error("--workers must be between 1 and 16")
    run(args)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


def find_database(requested: str) -> Path:
    """
    Resolve a Windows path robustly.

    Important:
    - If an explicit path exists, use it.
    - If it does not, print nearby candidate SQLite databases so the user
      can immediately see the mismatch instead of getting a generic
      "unable to open database file".
    """
    p = Path(requested).expanduser()

    candidates = []

    # Try exactly as supplied.
    if p.exists() and p.is_file():
        return p

    # Try relative to current working directory explicitly.
    cwd_p = Path.cwd() / p
    if cwd_p.exists() and cwd_p.is_file():
        return cwd_p

    # Search the common results location for likely KG databases.
    results = Path.cwd() / "results"
    if results.exists():
        candidates = sorted(
            [
                x for x in results.glob("*.sqlite")
                if x.is_file()
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    msg = [
        "",
        "DATABASE OPEN FAILED BEFORE SQLITE ACCESS.",
        f"requested : {requested}",
        f"cwd       : {Path.cwd()}",
        f"resolved  : {p.resolve()}",
    ]

    if candidates:
        msg.append("")
        msg.append("SQLite files found in .\\results:")
        for c in candidates[:20]:
            try:
                size_gb = c.stat().st_size / (1024**3)
                msg.append(
                    f"  {c}  ({size_gb:.2f} GB)"
                )
            except OSError:
                msg.append(f"  {c}")

    raise FileNotFoundError("\n".join(msg))


def connect_ro(path: Path) -> sqlite3.Connection:
    """
    Windows-safe SQLite read-only connection.

    Path.as_uri() avoids the common Windows URI/backslash problem.

    We explicitly use mode=ro so the audit can never mutate the 50GB graph.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.stat().st_size == 0:
        raise RuntimeError(f"Database is empty: {path}")

    # Correct Windows file URI, e.g.
    # file:///C:/Users/.../foo.sqlite?mode=ro
    uri = path.as_uri() + "?mode=ro"

    try:
        con = sqlite3.connect(
            uri,
            uri=True,
            timeout=60.0,
        )
    except sqlite3.OperationalError as exc:
        # Fallback diagnostic with the exact URI.
        raise RuntimeError(
            "SQLite could not open the existing file read-only.\n"
            f"path : {path}\n"
            f"uri  : {uri}\n"
            f"size : {path.stat().st_size:,} bytes\n"
            f"error: {exc}"
        ) from exc

    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-65536")
    return con


def schema(con):
    tables = {
        r["name"]
        for r in con.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """
        )
    }

    if "edges" not in tables:
        raise RuntimeError(
            "Opened the SQLite file successfully, but it has no 'edges' "
            "table. This is probably not the V561 KG audit database."
        )

    cols = {
        r["name"]
        for r in con.execute("PRAGMA table_info(edges)")
    }

    required = {"subject", "relation", "object", "source"}
    missing = sorted(required - cols)

    if missing:
        raise RuntimeError(
            f"edges table is missing columns: {missing}"
        )

    indexes = [
        r["name"]
        for r in con.execute(
            "PRAGMA index_list(edges)"
        )
    ]

    return {
        "tables": sorted(tables),
        "edges_columns": sorted(cols),
        "indexes": indexes,
    }


def graph_stats(con):
    t = time.perf_counter()

    edges = con.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    relations = con.execute(
        "SELECT COUNT(DISTINCT relation) FROM edges"
    ).fetchone()[0]

    subjects = con.execute(
        "SELECT COUNT(DISTINCT subject) FROM edges"
    ).fetchone()[0]

    objects = con.execute(
        "SELECT COUNT(DISTINCT object) FROM edges"
    ).fetchone()[0]

    return {
        "edges": edges,
        "relations": relations,
        "subjects": subjects,
        "objects": objects,
        "seconds": time.perf_counter() - t,
    }


def relation_inventory(con, limit):
    t = time.perf_counter()

    rows = con.execute(
        """
        SELECT relation, COUNT(*) AS count
        FROM edges
        GROUP BY relation
        ORDER BY count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return {
        "top": [
            {
                "relation": r["relation"],
                "count": r["count"],
            }
            for r in rows
        ],
        "seconds": time.perf_counter() - t,
    }


def sample_subjects(
    con,
    count,
    windows,
    seed,
):
    rng = random.Random(seed)

    max_rowid = con.execute(
        "SELECT MAX(rowid) FROM edges"
    ).fetchone()[0]

    if not max_rowid:
        return []

    result = set()

    # Random rowid windows, not SELECT DISTINCT over 192M rows.
    for _ in range(windows):
        start = rng.randint(
            1,
            max(1, int(max_rowid) - 1000),
        )
        end = start + 999

        for row in con.execute(
            """
            SELECT subject
            FROM edges
            WHERE rowid BETWEEN ? AND ?
            """,
            (start, end),
        ):
            result.add(row["subject"])

            if len(result) >= count:
                return list(result)

    return list(result)


def load_adjacency(
    con,
    subjects,
    batch_size,
):
    adjacency = defaultdict(list)

    for i in range(0, len(subjects), batch_size):
        chunk = subjects[i:i + batch_size]

        if not chunk:
            continue

        placeholders = ",".join(
            "?" for _ in chunk
        )

        query = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(query, chunk):
            adjacency[row["subject"]].append(
                (
                    row["relation"],
                    row["object"],
                    row["source"],
                )
            )

    return adjacency


def composition_scan(
    adjacency,
    max_paths_per_subject,
    max_mid_edges,
):
    pair_stats = defaultdict(
        lambda: {
            "paths": 0,
            "confirmed": defaultdict(int),
            "subjects": set(),
            "endpoints": set(),
        }
    )

    # Build only for sampled subjects.
    direct = defaultdict(set)

    for subject, edges in adjacency.items():
        for relation, obj, source in edges:
            direct[(subject, obj)].add(relation)

    raw = 0
    capped = 0

    for subject, first_edges in adjacency.items():
        emitted = 0

        for r1, middle, source1 in first_edges:
            if emitted >= max_paths_per_subject:
                capped += 1
                break

            middle_edges = adjacency.get(middle)

            if not middle_edges:
                continue

            if len(middle_edges) > max_mid_edges:
                middle_edges = middle_edges[:max_mid_edges]

            for r2, endpoint, source2 in middle_edges:
                if endpoint in (subject, middle):
                    continue

                raw += 1
                emitted += 1

                st = pair_stats[(r1, r2)]

                st["paths"] += 1
                st["subjects"].add(subject)
                st["endpoints"].add(endpoint)

                for confirmation in direct.get(
                    (subject, endpoint),
                    (),
                ):
                    st["confirmed"][confirmation] += 1

    return pair_stats, raw, capped


def summarize(pair_stats, minimum_paths):
    rules = []

    for (r1, r2), st in pair_stats.items():
        if st["paths"] < minimum_paths:
            continue

        if st["confirmed"]:
            best_relation, best_count = max(
                st["confirmed"].items(),
                key=lambda kv: kv[1],
            )
        else:
            best_relation = None
            best_count = 0

        rate = (
            best_count / st["paths"]
            if st["paths"]
            else 0.0
        )

        rules.append(
            {
                "sequence": [r1, r2],
                "paths": st["paths"],
                "best_confirmation_relation": best_relation,
                "best_confirmation_count": best_count,
                "best_confirmation_rate": rate,
                "distinct_subjects": len(st["subjects"]),
                "distinct_endpoints": len(st["endpoints"]),
                "confirmation_relations": [
                    {
                        "relation": r,
                        "count": c,
                        "rate": c / st["paths"],
                    }
                    for r, c in sorted(
                        st["confirmed"].items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:10]
                ],
            }
        )

    rules.sort(
        key=lambda r: (
            r["best_confirmation_rate"],
            r["best_confirmation_count"],
            r["paths"],
        ),
        reverse=True,
    )

    return rules


def classify(rules):
    out = {
        "strong": [],
        "usable": [],
        "thin": [],
        "weak": [],
    }

    for r in rules:
        rate = r["best_confirmation_rate"]
        confirms = r["best_confirmation_count"]
        subjects = r["distinct_subjects"]
        endpoints = r["distinct_endpoints"]

        if (
            confirms >= 100
            and rate >= 0.05
            and subjects >= 50
            and endpoints >= 50
        ):
            out["strong"].append(r)

        elif (
            confirms >= 25
            and rate >= 0.02
            and subjects >= 20
            and endpoints >= 20
        ):
            out["usable"].append(r)

        elif confirms >= 5 and subjects >= 5:
            out["thin"].append(r)

        else:
            out["weak"].append(r)

    return out


def lookup_benchmark(
    con,
    subjects,
    sample_count,
    seed,
):
    """
    Measure the actual operation the cognitive architecture uses:
        subject -> outgoing edges

    Every lookup is an indexed equality lookup.
    """
    rng = random.Random(seed)

    if not subjects:
        return {
            "queries": 0,
            "rows": 0,
            "seconds": 0.0,
            "avg_ms": 0.0,
        }

    selected = [
        rng.choice(subjects)
        for _ in range(
            min(sample_count, len(subjects))
        )
    ]

    start = time.perf_counter()
    rows = 0

    for subject in selected:
        for _ in con.execute(
            """
            SELECT relation, object, source
            FROM edges
            WHERE subject = ?
            """,
            (subject,),
        ):
            rows += 1

    seconds = time.perf_counter() - start

    return {
        "queries": len(selected),
        "rows": rows,
        "seconds": seconds,
        "avg_ms": (
            1000 * seconds / len(selected)
            if selected
            else 0.0
        ),
    }


def run(args):
    started = time.perf_counter()

    print("=== V564 50GB SEMANTIC GRAPH REFERENTIAL AUDIT ===")
    print(f"requested DB : {args.database}")
    print(f"output       : {args.output}")
    print(f"subjects     : {args.sample_subjects:,}")
    print(f"windows      : {args.windows:,}")
    print(f"workers      : disk-backed SQLite; no shared in-memory graph")
    print("source       : READ-ONLY")
    print("LLM          : NOT USED")
    print()

    db = find_database(args.database)

    print(f"[DB] absolute path : {db.resolve()}")
    print(
        f"[DB] size          : "
        f"{db.stat().st_size / (1024**3):.2f} GB"
    )

    con = connect_ro(db)

    print("[DB] SQLite opened read-only")
    print()

    schema_info = schema(con)

    print(
        f"[SCHEMA] edges indexes: "
        f"{schema_info['indexes']}"
    )

    t = time.perf_counter()
    graph = graph_stats(con)

    print(
        f"[GRAPH] edges={graph['edges']:,} "
        f"subjects={graph['subjects']:,} "
        f"objects={graph['objects']:,} "
        f"relations={graph['relations']:,} "
        f"seconds={graph['seconds']:.2f}"
    )

    inventory = relation_inventory(
        con,
        args.top_relations,
    )

    print(
        f"[RELATIONS] inventory ready "
        f"seconds={inventory['seconds']:.2f}"
    )

    t = time.perf_counter()

    sampled = sample_subjects(
        con,
        args.sample_subjects,
        args.windows,
        args.seed,
    )

    print(
        f"[SAMPLE] subjects={len(sampled):,} "
        f"seconds={time.perf_counter()-t:.2f}"
    )

    t = time.perf_counter()

    adjacency = load_adjacency(
        con,
        sampled,
        args.batch_size,
    )

    adjacency_edges = sum(
        len(v) for v in adjacency.values()
    )

    print(
        f"[ADJACENCY] loaded_subjects={len(adjacency):,} "
        f"edges={adjacency_edges:,} "
        f"seconds={time.perf_counter()-t:.2f}"
    )

    t = time.perf_counter()

    pair_stats, raw_paths, capped = composition_scan(
        adjacency,
        args.max_paths_per_subject,
        args.max_mid_edges,
    )

    print(
        f"[COMPOSITION] raw_paths={raw_paths:,} "
        f"relation_pairs={len(pair_stats):,} "
        f"capped_subjects={capped:,} "
        f"seconds={time.perf_counter()-t:.2f}"
    )

    rules = summarize(
        pair_stats,
        args.min_paths,
    )

    buckets = classify(rules)

    print()
    print("=== COMPOSITION SUMMARY ===")
    print(f"eligible relation pairs: {len(rules):,}")
    for k in ("strong", "usable", "thin", "weak"):
        print(f"{k:8s}: {len(buckets[k]):,}")

    print()
    print("--- TOP COMPOSITION RULES ---")

    for r in rules[:args.top_rules]:
        print(
            f"{' -> '.join(r['sequence']):40s} "
            f"=> {str(r['best_confirmation_relation']):18s} "
            f"paths={r['paths']:8,} "
            f"confirm={r['best_confirmation_count']:7,} "
            f"rate={r['best_confirmation_rate']:.5f} "
            f"subjects={r['distinct_subjects']:6,}"
        )

    t = time.perf_counter()

    lookup = lookup_benchmark(
        con,
        sampled,
        args.lookup_queries,
        args.seed + 1,
    )

    print()
    print("=== REFERENTIAL LOOKUP BENCHMARK ===")
    print(
        f"queries={lookup['queries']:,} "
        f"rows={lookup['rows']:,} "
        f"seconds={lookup['seconds']:.3f} "
        f"avg_ms={lookup['avg_ms']:.3f}"
    )

    if len(buckets["strong"]) >= 10:
        verdict = "STRONG"
    elif (
        len(buckets["strong"]) >= 3
        or len(buckets["usable"]) >= 10
    ):
        verdict = "USABLE"
    elif (
        len(buckets["usable"]) >= 3
        or len(buckets["thin"]) >= 10
    ):
        verdict = "WEAK"
    else:
        verdict = "INSUFFICIENT"

    print()
    print("=== DATA VERDICT ===")
    print(f"VERDICT: {verdict}")

    report = {
        "benchmark": "v564_disk_backed_semantic_graph_audit",
        "database": str(db.resolve()),
        "database_size_bytes": db.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": graph,
        "relation_inventory": inventory,
        "sampling": {
            "subjects_requested": args.sample_subjects,
            "subjects_sampled": len(sampled),
            "windows": args.windows,
            "batch_size": args.batch_size,
            "max_paths_per_subject": args.max_paths_per_subject,
            "max_mid_edges": args.max_mid_edges,
        },
        "composition": {
            "raw_two_hop_paths": raw_paths,
            "relation_pairs": len(pair_stats),
            "eligible_pairs": len(rules),
            "rules": rules[:args.output_rules],
        },
        "buckets": {
            k: len(v)
            for k, v in buckets.items()
        },
        "lookup_benchmark": lookup,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    con.close()

    print()
    print(f"JSON: {output}")
    print(
        f"END-TO-END: "
        f"{report['elapsed_seconds']:.3f}s"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--database",
        default=r".\results\v561_kg_composition_audit.sqlite",
    )

    ap.add_argument(
        "--output",
        default=r".\results\v564_semantic_graph_audit.json",
    )

    ap.add_argument(
        "--sample-subjects",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--windows",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--max-paths-per-subject",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--max-mid-edges",
        type=int,
        default=250,
    )

    ap.add_argument(
        "--min-paths",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--top-relations",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--top-rules",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--output-rules",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--lookup-queries",
        type=int,
        default=1000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=564,
    )

    run(ap.parse_args())


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V568-20 — Fast Semantic Relation Induction / Vocabulary Probe

This version deliberately profiles ONLY the top 20 predicates by edge count.

Critical performance rule:
    NEVER scan an entire predicate's rows.

Sampling is done through random rowid probes against the already-indexed edges
table. For the current YAGO+DBpedia graph the top predicates occupy a large
fraction of the table, so bounded rowid-window sampling gives us hundreds of
examples per predicate without walking tens of millions of rows.

The source graph is read-only.

The goal is behavioral discovery, not semantic guessing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def resolve_db(path: str) -> Path:
    p = Path(path).expanduser()
    if p.exists() and p.is_file():
        return p.resolve()

    if not p.is_absolute():
        q = Path.cwd() / p
        if q.exists() and q.is_file():
            return q.resolve()

    results = Path.cwd() / "results"
    candidates = []
    if results.exists():
        candidates = sorted(
            results.glob("*.sqlite"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    lines = [
        "Could not find database.",
        f"requested: {path}",
        f"cwd: {Path.cwd()}",
    ]

    for c in candidates[:20]:
        try:
            lines.append(
                f"  {c} ({c.stat().st_size / 1024**3:.2f} GB)"
            )
        except OSError:
            lines.append(f"  {c}")

    raise FileNotFoundError("\n".join(lines))


def connect_ro(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    uri = path.as_uri() + "?mode=ro"

    con = sqlite3.connect(
        uri,
        uri=True,
        timeout=120.0,
        check_same_thread=False,
    )
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-32768")
    return con


def check_schema(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(edges)")}
    required = {"subject", "relation", "object", "source"}
    missing = required - cols

    if missing:
        raise RuntimeError(
            f"edges table missing: {sorted(missing)}"
        )

    return {
        "columns": sorted(cols),
        "indexes": [
            r[1]
            for r in con.execute("PRAGMA index_list(edges)")
        ],
    }


def exact_edge_count(con):
    return con.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]


def top_predicates(con, limit):
    """
    Uses idx_r. This is the one aggregate over relation we actually need.
    """
    rows = con.execute(
        """
        SELECT relation, COUNT(*) AS edge_count
        FROM edges
        GROUP BY relation
        ORDER BY edge_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return [
        {
            "rank": i + 1,
            "predicate": str(r["relation"]),
            "edge_count": int(r["edge_count"]),
        }
        for i, r in enumerate(rows)
    ]


def rowid_bounds(con):
    row = con.execute(
        "SELECT MIN(rowid), MAX(rowid) FROM edges"
    ).fetchone()
    return int(row[0]), int(row[1])


def bounded_relation_sample(
    con,
    predicate,
    sample_size,
    rowid_min,
    rowid_max,
    seed,
    window_size=5000,
    max_probes=5000,
):
    """
    Bounded random rowid windows.

    We intentionally do NOT:
        SELECT ... FROM edges WHERE relation = ?
    over the entire predicate.

    Instead, random rowid windows are checked until enough examples are
    collected.

    For dense top relations this is very fast. It also has an explicit probe
    ceiling so a sparse relation can never hang forever.
    """
    rng = random.Random(seed)

    found = {}
    probes = 0
    rows_examined = 0

    while (
        len(found) < sample_size
        and probes < max_probes
    ):
        start = rng.randint(
            rowid_min,
            max(
                rowid_min,
                rowid_max - window_size + 1,
            ),
        )
        end = min(
            rowid_max,
            start + window_size - 1,
        )

        for row in con.execute(
            """
            SELECT rowid, subject, object, source
            FROM edges
            WHERE rowid BETWEEN ? AND ?
              AND relation = ?
            """,
            (start, end, predicate),
        ):
            rows_examined += 1

            key = (
                row["subject"],
                row["object"],
            )

            found[key] = (
                row["subject"],
                row["object"],
                row["source"],
            )

            if len(found) >= sample_size:
                break

        probes += 1

    return (
        list(found.values())[:sample_size],
        probes,
        rows_examined,
    )


def load_outgoing(
    con,
    nodes,
    chunk_size=200,
    max_edges_per_node=200,
):
    out = defaultdict(list)
    nodes = list(nodes)

    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i:i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)

        q = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, chunk):
            bucket = out[row["subject"]]

            if len(bucket) < max_edges_per_node:
                bucket.append(
                    (
                        row["relation"],
                        row["object"],
                        row["source"],
                    )
                )

    return out


def endpoint_confirmations(
    con,
    pairs,
    chunk_size=200,
):
    by_subject = defaultdict(set)

    for s, o in pairs:
        by_subject[s].add(o)

    confirmed = defaultdict(Counter)

    subjects = list(by_subject)

    for i in range(0, len(subjects), chunk_size):
        chunk = subjects[i:i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        wanted = {
            s: by_subject[s]
            for s in chunk
        }

        for row in con.execute(q, chunk):
            if row["object"] in wanted.get(
                row["subject"],
                (),
            ):
                confirmed[
                    (row["subject"], row["object"])
                ][row["relation"]] += 1

    return confirmed


def entropy(counter):
    total = sum(counter.values())

    if total <= 0:
        return 0.0

    result = 0.0

    for n in counter.values():
        p = n / total
        result -= p * math.log2(p)

    return result


def lexical_hints(predicate):
    """
    Labels only as review hints. Never used for the behavioral cluster.
    """
    p = predicate.lower()

    patterns = {
        "part": "part-like",
        "parent": "parent-like",
        "father": "parent-like",
        "mother": "parent-like",
        "child": "child-like",
        "location": "location-like",
        "place": "location-like",
        "birth": "location-like",
        "death": "location-like",
        "material": "material-like",
        "color": "property-like",
        "colour": "property-like",
        "gender": "property-like",
        "language": "language-like",
        "member": "membership-like",
        "creator": "creation-like",
        "author": "creation-like",
        "owner": "ownership-like",
        "spouse": "social-like",
        "friend": "social-like",
    }

    return sorted({
        value
        for key, value in patterns.items()
        if key in p
    })


def profile_predicate(
    database,
    predicate,
    edge_count,
    sample_size,
    window_size,
    max_probes,
    max_second_edges,
    max_paths,
    seed,
):
    started = time.perf_counter()

    con = connect_ro(database)

    rowid_min, rowid_max = rowid_bounds(con)

    sample, probes, rows_examined = (
        bounded_relation_sample(
            con,
            predicate,
            sample_size,
            rowid_min,
            rowid_max,
            seed,
            window_size,
            max_probes,
        )
    )

    subjects = {
        s
        for s, _, _ in sample
    }

    objects = {
        o
        for _, o, _ in sample
    }

    adjacency = load_outgoing(
        con,
        objects,
        max_edges_per_node=max_second_edges,
    )

    second_relations = Counter()
    endpoint_pairs = set()

    paths = 0

    for subject, middle, _src1 in sample:
        outgoing = adjacency.get(middle, ())

        for r2, endpoint, _src2 in outgoing:
            if endpoint in {
                subject,
                middle,
            }:
                continue

            second_relations[r2] += 1
            endpoint_pairs.add(
                (subject, endpoint)
            )
            paths += 1

            if paths >= max_paths:
                break

        if paths >= max_paths:
            break

    confirmed = endpoint_confirmations(
        con,
        endpoint_pairs,
    )

    confirmation_relations = Counter()

    for counter in confirmed.values():
        for rel, n in counter.items():
            confirmation_relations[rel] += n

    intermediate_degrees = [
        len(adjacency.get(o, ()))
        for o in list(objects)[:1000]
    ]

    inverse_pairs = {
        (s, o)
        for s, o, _ in sample
        if s != o
    }

    inverse_hits = 0

    # This is bounded by the sample size and uses subject index.
    if inverse_pairs:
        wanted = defaultdict(set)

        for s, o in list(inverse_pairs)[:500]:
            wanted[o].add(s)

        nodes = list(wanted)

        for i in range(0, len(nodes), 200):
            chunk = nodes[i:i + 200]
            placeholders = ",".join("?" for _ in chunk)

            q = f"""
                SELECT subject, object
                FROM edges
                WHERE relation = ?
                  AND subject IN ({placeholders})
            """

            for row in con.execute(
                q,
                [predicate, *chunk],
            ):
                if row["object"] in wanted.get(
                    row["subject"],
                    (),
                ):
                    inverse_hits += 1

    inverse_overlap = (
        inverse_hits / max(
            1,
            min(500, len(inverse_pairs)),
        )
    )

    result = {
        "predicate": predicate,
        "edge_count": edge_count,
        "sampled_edges": len(sample),
        "sampling": {
            "rowid_probes": probes,
            "predicate_rows_examined": rows_examined,
            "probe_ceiling": max_probes,
            "window_size": window_size,
        },
        "sample_subjects": len(subjects),
        "sample_objects": len(objects),
        "behavior": {
            "two_hop_paths": paths,
            "second_relation_entropy_bits": entropy(
                second_relations
            ),
            "second_relation_profile": [
                {
                    "relation": rel,
                    "count": n,
                    "fraction": (
                        n / max(1, paths)
                    ),
                }
                for rel, n in
                second_relations.most_common(25)
            ],
            "endpoint_confirmation_total": sum(
                confirmation_relations.values()
            ),
            "endpoint_confirmation_profile": [
                {
                    "relation": rel,
                    "count": n,
                    "fraction": (
                        n / max(
                            1,
                            sum(
                                confirmation_relations.values()
                            ),
                        )
                    ),
                }
                for rel, n in
                confirmation_relations.most_common(25)
            ],
            "sampled_inverse_overlap": inverse_overlap,
            "mean_intermediate_outdegree": (
                statistics.mean(intermediate_degrees)
                if intermediate_degrees
                else 0.0
            ),
            "median_intermediate_outdegree": (
                statistics.median(intermediate_degrees)
                if intermediate_degrees
                else 0.0
            ),
        },
        "lexical_review_hints": lexical_hints(
            predicate
        ),
        "seconds": time.perf_counter() - started,
    }

    con.close()
    return result


def profile_map(items):
    return {
        x["relation"]: x["fraction"]
        for x in items
    }


def js_distance(a, b):
    keys = set(a) | set(b)

    if not keys:
        return 1.0

    pa = {
        k: a.get(k, 0.0)
        for k in keys
    }
    pb = {
        k: b.get(k, 0.0)
        for k in keys
    }

    total_a = sum(pa.values()) or 1.0
    total_b = sum(pb.values()) or 1.0

    pa = {
        k: v / total_a
        for k, v in pa.items()
    }
    pb = {
        k: v / total_b
        for k, v in pb.items()
    }

    m = {
        k: (pa[k] + pb[k]) / 2
        for k in keys
    }

    def kl(p):
        return sum(
            v * math.log2(
                v / m[k]
            )
            for k, v in p.items()
            if v > 0 and m[k] > 0
        )

    return math.sqrt(
        max(
            0.0,
            (kl(pa) + kl(pb)) / 2,
        )
    )


def behavior_distance(a, b):
    ac = a["behavior"]
    bc = b["behavior"]

    composition = js_distance(
        profile_map(
            ac["second_relation_profile"]
        ),
        profile_map(
            bc["second_relation_profile"]
        ),
    )

    confirmation = js_distance(
        profile_map(
            ac["endpoint_confirmation_profile"]
        ),
        profile_map(
            bc["endpoint_confirmation_profile"]
        ),
    )

    inverse = min(
        1.0,
        abs(
            ac["sampled_inverse_overlap"]
            - bc["sampled_inverse_overlap"]
        ),
    )

    degree = 1.0 - (
        min(
            math.log1p(
                ac["mean_intermediate_outdegree"]
            ),
            math.log1p(
                bc["mean_intermediate_outdegree"]
            ),
        )
        /
        max(
            math.log1p(
                ac["mean_intermediate_outdegree"]
            ),
            math.log1p(
                bc["mean_intermediate_outdegree"]
            ),
            1.0,
        )
    )

    return (
        0.50 * composition
        + 0.30 * confirmation
        + 0.10 * inverse
        + 0.10 * degree
    )


def nearest(signatures, k):
    names = list(signatures)
    result = []

    for name in names:
        pairs = []

        for other in names:
            if other == name:
                continue

            pairs.append(
                (
                    behavior_distance(
                        signatures[name],
                        signatures[other],
                    ),
                    other,
                )
            )

        pairs.sort()

        result.append(
            {
                "predicate": name,
                "neighbors": [
                    {
                        "predicate": other,
                        "distance": distance,
                    }
                    for distance, other in pairs[:k]
                ],
            }
        )

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--database",
        default=r".\results\v562_kg_composition_audit.sqlite",
    )
    ap.add_argument(
        "--output",
        default=r".\results\v568_20_relation_induction.json",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--top-predicates",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=600,
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=5000,
    )
    ap.add_argument(
        "--max-probes",
        type=int,
        default=5000,
    )
    ap.add_argument(
        "--max-second-edges",
        type=int,
        default=150,
    )
    ap.add_argument(
        "--max-paths",
        type=int,
        default=12000,
    )
    ap.add_argument(
        "--neighbors",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=56820,
    )

    args = ap.parse_args()

    start = time.perf_counter()

    db = resolve_db(args.database)

    print("=== V568-20 TOP PREDICATE BEHAVIORAL PROBE ===")
    print(f"database           : {db}")
    print(
        f"database size      : "
        f"{db.stat().st_size / 1024**3:.2f} GB"
    )
    print(f"workers            : {args.workers}")
    print(f"top predicates     : {args.top_predicates}")
    print(f"sample/predicate   : {args.sample_size}")
    print(f"window size        : {args.window_size}")
    print(f"probe ceiling      : {args.max_probes}")
    print("source             : READ-ONLY")
    print()

    con = connect_ro(db)
    schema_info = check_schema(con)

    print(
        f"[SCHEMA] indexes: "
        f"{schema_info['indexes']}"
    )

    edges = exact_edge_count(con)

    print(
        f"[GRAPH] edges={edges:,}"
    )

    print(
        "[INVENTORY] selecting top predicates by edge count...",
        flush=True,
    )

    inventory = top_predicates(
        con,
        args.top_predicates,
    )

    con.close()

    print()
    print("=== TOP PREDICATES ===")

    for item in inventory:
        print(
            f"{item['rank']:2d}. "
            f"{item['predicate']:32s} "
            f"{item['edge_count']:12,}"
        )

    results = []

    print()
    print(
        f"=== PROFILING {len(inventory)} "
        f"PREDICATES WITH "
        f"{min(args.workers, len(inventory))} WORKERS ==="
    )

    with ThreadPoolExecutor(
        max_workers=min(
            args.workers,
            len(inventory),
        )
    ) as pool:

        futures = {}

        for item in inventory:
            future = pool.submit(
                profile_predicate,
                db,
                item["predicate"],
                item["edge_count"],
                args.sample_size,
                args.window_size,
                args.max_probes,
                args.max_second_edges,
                args.max_paths,
                args.seed + item["rank"],
            )

            futures[future] = item

        completed = 0

        for future in as_completed(futures):
            item = futures[future]

            try:
                result = future.result()

            except Exception as exc:
                completed += 1
                print(
                    f"[{completed}/{len(inventory)}] "
                    f"{item['predicate']:32s} "
                    f"ERROR: {exc}",
                    flush=True,
                )
                continue

            completed += 1
            results.append(result)

            behavior = result["behavior"]

            print(
                f"[{completed}/{len(inventory)}] "
                f"{item['predicate']:32s} "
                f"sample={result['sampled_edges']:4,} "
                f"probes={result['sampling']['rowid_probes']:5,} "
                f"2hop={behavior['two_hop_paths']:6,} "
                f"confirm={behavior['endpoint_confirmation_total']:5,} "
                f"inv={behavior['sampled_inverse_overlap']:.3f} "
                f"time={result['seconds']:.2f}s",
                flush=True,
            )

    if not results:
        raise RuntimeError(
            "No predicates were profiled successfully."
        )

    results.sort(
        key=lambda x: next(
            item["rank"]
            for item in inventory
            if item["predicate"] == x["predicate"]
        )
    )

    signatures = {
        r["predicate"]: r
        for r in results
    }

    print()
    print(
        "=== BEHAVIORAL NEAREST NEIGHBORS ==="
    )

    neighbor_rows = nearest(
        signatures,
        args.neighbors,
    )

    for row in neighbor_rows:
        print()
        print(row["predicate"])

        for n in row["neighbors"]:
            print(
                f"  {n['predicate']:32s} "
                f"d={n['distance']:.4f}"
            )

    # Pairwise behavioral similarities.
    similar_pairs = []

    names = list(signatures)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            distance = behavior_distance(
                signatures[a],
                signatures[b],
            )

            similar_pairs.append(
                {
                    "predicate_a": a,
                    "predicate_b": b,
                    "distance": distance,
                }
            )

    similar_pairs.sort(
        key=lambda x: x["distance"]
    )

    # Useful candidates for the next cognitive validation stage.
    review_candidates = []

    for r in results:
        if (
            r["behavior"]["endpoint_confirmation_total"]
            >= 5
            or r["behavior"]["two_hop_paths"]
            >= 100
            or r["lexical_review_hints"]
        ):
            review_candidates.append(
                {
                    "predicate": r["predicate"],
                    "edge_count": r["edge_count"],
                    "lexical_hints": r[
                        "lexical_review_hints"
                    ],
                    "two_hop_paths": r[
                        "behavior"
                    ]["two_hop_paths"],
                    "confirmation_total": r[
                        "behavior"
                    ]["endpoint_confirmation_total"],
                    "inverse_overlap": r[
                        "behavior"
                    ]["sampled_inverse_overlap"],
                }
            )

    report = {
        "benchmark": "v568_top20_behavioral_relation_induction",
        "database": str(db),
        "database_size_bytes": db.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": {
            "edges": edges,
        },
        "inventory": inventory,
        "profiles": results,
        "nearest_neighbors": neighbor_rows,
        "similar_predicate_pairs": similar_pairs[:500],
        "review_candidates": review_candidates,
        "config": {
            "workers": args.workers,
            "top_predicates": args.top_predicates,
            "sample_size": args.sample_size,
            "window_size": args.window_size,
            "max_probes": args.max_probes,
            "max_second_edges": args.max_second_edges,
            "max_paths": args.max_paths,
            "neighbors": args.neighbors,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - start,
    }

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("V568-20 COMPLETE")
    print("=" * 72)
    print(
        f"predicates profiled : "
        f"{len(results)}"
    )
    print(
        f"similar pairs       : "
        f"{len(similar_pairs):,}"
    )
    print(
        f"JSON                : "
        f"{out}"
    )
    print(
        f"elapsed             : "
        f"{report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V565 — Targeted Composition Discovery / Coverage

Purpose
-------
Audit whether the large YAGO + DBpedia semantic graph contains enough
compositional evidence for the relation families that the cognitive
architecture actually cares about.

Unlike V564, this does NOT choose random subjects and hope useful relations
appear. It is relation-stratified.

For each target first-hop relation R1:
    sample actual R1 edges A -R1-> B
    retrieve B's outgoing edges R2
    produce A -R1-> B -R2-> C candidates
    check whether A -> C has a direct relation R3

This answers:
    "Given the semantic relation R1, what relations tend to compose after it?"

The source SQLite database is opened READ-ONLY.

No global edges×edges join is used.

Outputs:
    results/v565_targeted_composition.json

Key metrics:
    - first-hop coverage
    - sampled second-hop paths
    - composition sequence counts
    - direct endpoint confirmations
    - confirmation rate
    - subject diversity
    - endpoint diversity
    - source diversity
    - per-target-relational coverage
    - strongest candidate rules
    - trainability signal

Important:
A directly confirmed endpoint is evidence of graph consistency, not a proof
that a relation-composition rule is universally valid.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RELATIONS = [
    "is_a",
    "has",
    "has_part",
    "part_of",
    "contains",
    "made_of",
    "has_property",
    "capable_of",
    "used_for",
    "located_in",
    "causes",
]


def resolve_db(path: str) -> Path:
    p = Path(path).expanduser()

    if p.exists() and p.is_file():
        return p.resolve()

    if not p.is_absolute():
        alt = Path.cwd() / p
        if alt.exists() and alt.is_file():
            return alt.resolve()

    results = Path.cwd() / "results"
    nearby = []
    if results.exists():
        nearby = sorted(
            results.glob("*.sqlite"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    msg = [
        "Could not find the SQLite database.",
        f"requested: {path}",
        f"cwd: {Path.cwd()}",
    ]

    if nearby:
        msg.append("SQLite files in .\\results:")
        for x in nearby[:20]:
            try:
                msg.append(
                    f"  {x} ({x.stat().st_size / 1024**3:.2f} GB)"
                )
            except OSError:
                msg.append(f"  {x}")

    raise FileNotFoundError("\n".join(msg))


def connect_ro(path: Path):
    uri = path.as_uri() + "?mode=ro"
    con = sqlite3.connect(
        uri,
        uri=True,
        timeout=60.0,
    )
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-65536")
    return con


def check_schema(con):
    cols = {
        r[1]
        for r in con.execute("PRAGMA table_info(edges)")
    }

    missing = {
        "subject",
        "relation",
        "object",
        "source",
    } - cols

    if missing:
        raise RuntimeError(
            f"edges table missing columns: {sorted(missing)}"
        )

    indexes = [
        r[1]
        for r in con.execute("PRAGMA index_list(edges)")
    ]

    return {
        "columns": sorted(cols),
        "indexes": indexes,
    }


def graph_overview(con):
    t = time.perf_counter()

    row = con.execute(
        """
        SELECT
            COUNT(*) AS edges,
            COUNT(DISTINCT relation) AS relations,
            COUNT(DISTINCT subject) AS subjects,
            COUNT(DISTINCT object) AS objects
        FROM edges
        """
    ).fetchone()

    return {
        "edges": row["edges"],
        "relations": row["relations"],
        "subjects": row["subjects"],
        "objects": row["objects"],
        "seconds": time.perf_counter() - t,
    }


def relation_counts(con, relations):
    t = time.perf_counter()

    result = {}

    for relation in relations:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS edges,
                COUNT(DISTINCT subject) AS subjects,
                COUNT(DISTINCT object) AS objects
            FROM edges
            WHERE relation = ?
            """,
            (relation,),
        ).fetchone()

        result[relation] = {
            "edges": row["edges"],
            "subjects": row["subjects"],
            "objects": row["objects"],
        }

    return {
        "relations": result,
        "seconds": time.perf_counter() - t,
    }


def sample_relation_edges(
    con,
    relation,
    sample_size,
    seed,
):
    """
    Reservoir sample actual edges of one relation.

    This avoids scanning all rows into Python memory. SQLite still reads
    the relation index, which is exactly what this audit wants.
    """
    rng = random.Random(seed)

    reservoir = []
    seen = 0

    for row in con.execute(
        """
        SELECT subject, object, source
        FROM edges
        WHERE relation = ?
        """,
        (relation,),
    ):
        seen += 1

        item = (
            row["subject"],
            row["object"],
            row["source"],
        )

        if len(reservoir) < sample_size:
            reservoir.append(item)
        else:
            j = rng.randint(1, seen)

            if j <= sample_size:
                reservoir[j - 1] = item

    return reservoir, seen


def load_outgoing(con, nodes):
    """
    Load outgoing adjacency for a bounded set of intermediate nodes.

    Uses the subject index.
    """
    if not nodes:
        return {}

    result = defaultdict(list)

    # SQLite variable limit is often 999; keep chunks conservative.
    chunk_size = 400

    nodes = list(nodes)

    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i:i + chunk_size]

        placeholders = ",".join("?" for _ in chunk)

        query = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(query, chunk):
            result[row["subject"]].append(
                (
                    row["relation"],
                    row["object"],
                    row["source"],
                )
            )

    return result


def endpoint_confirmations(
    con,
    pairs,
    chunk_size=300,
):
    """
    For (subject, endpoint) pairs, retrieve direct relations.

    Uses subject+object index when available. The query is batched so we
    don't execute one SQL statement per path.
    """
    confirmed = defaultdict(Counter)

    # SQLite doesn't support portable tuple-IN efficiently for huge sets.
    # Use subject-grouped batched lookups. This still uses idx_s.
    by_subject = defaultdict(set)

    for subject, endpoint in pairs:
        by_subject[subject].add(endpoint)

    for start in range(0, len(by_subject), chunk_size):
        subjects = list(by_subject)[start:start + chunk_size]

        if not subjects:
            continue

        placeholders = ",".join("?" for _ in subjects)

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        wanted = {
            s: by_subject[s]
            for s in subjects
        }

        for row in con.execute(q, subjects):
            s = row["subject"]
            o = row["object"]

            if o in wanted.get(s, ()):
                confirmed[(s, o)][row["relation"]] += 1

    return confirmed


def discover_for_relation(
    con,
    relation,
    first_sample,
    max_second_edges,
    seed,
):
    first_edges, total_edges = sample_relation_edges(
        con,
        relation,
        first_sample,
        seed,
    )

    mids = {o for _, o, _ in first_edges}

    adjacency = load_outgoing(
        con,
        mids,
    )

    candidates = []

    for subject, middle, source1 in first_edges:
        outgoing = adjacency.get(middle, ())

        if len(outgoing) > max_second_edges:
            # deterministic truncation by iteration order
            outgoing = outgoing[:max_second_edges]

        for r2, endpoint, source2 in outgoing:
            if endpoint == subject or endpoint == middle:
                continue

            candidates.append(
                {
                    "subject": subject,
                    "middle": middle,
                    "endpoint": endpoint,
                    "r1": relation,
                    "r2": r2,
                    "source1": source1,
                    "source2": source2,
                }
            )

    # Deduplicate exact paths. The graph may contain multiple source rows.
    unique = {}
    for x in candidates:
        key = (
            x["subject"],
            x["r1"],
            x["middle"],
            x["r2"],
            x["endpoint"],
        )
        unique[key] = x

    candidates = list(unique.values())

    pairs = {
        (x["subject"], x["endpoint"])
        for x in candidates
    }

    confirmations = endpoint_confirmations(
        con,
        pairs,
    )

    stats = defaultdict(
        lambda: {
            "paths": 0,
            "subjects": set(),
            "endpoints": set(),
            "sources": set(),
            "confirmed": Counter(),
            "confirmed_subjects": defaultdict(set),
            "confirmed_endpoints": defaultdict(set),
        }
    )

    for x in candidates:
        key = (x["r1"], x["r2"])

        st = stats[key]

        st["paths"] += 1
        st["subjects"].add(x["subject"])
        st["endpoints"].add(x["endpoint"])
        st["sources"].add(x["source1"])
        st["sources"].add(x["source2"])

        direct = confirmations.get(
            (x["subject"], x["endpoint"]),
            {},
        )

        for rel, n in direct.items():
            st["confirmed"][rel] += 1
            st["confirmed_subjects"][rel].add(
                x["subject"]
            )
            st["confirmed_endpoints"][rel].add(
                x["endpoint"]
            )

    rows = []

    for (r1, r2), st in stats.items():
        best = None
        best_count = 0

        if st["confirmed"]:
            best, best_count = st["confirmed"].most_common(1)[0]

        rows.append(
            {
                "sequence": [r1, r2],
                "paths": st["paths"],
                "best_confirmation_relation": best,
                "best_confirmation_count": best_count,
                "best_confirmation_rate": (
                    best_count / st["paths"]
                    if st["paths"]
                    else 0.0
                ),
                "distinct_subjects": len(st["subjects"]),
                "distinct_endpoints": len(st["endpoints"]),
                "source_diversity": len(
                    st["sources"] - {None}
                ),
                "confirmation_relations": [
                    {
                        "relation": rel,
                        "count": count,
                        "rate": count / st["paths"],
                        "distinct_subjects": len(
                            st["confirmed_subjects"][rel]
                        ),
                        "distinct_endpoints": len(
                            st["confirmed_endpoints"][rel]
                        ),
                    }
                    for rel, count in st["confirmed"].most_common(10)
                ],
            }
        )

    rows.sort(
        key=lambda x: (
            x["best_confirmation_rate"],
            x["best_confirmation_count"],
            x["paths"],
        ),
        reverse=True,
    )

    return {
        "first_hop_relation": relation,
        "first_hop_total_edges": total_edges,
        "first_hop_sampled_edges": len(first_edges),
        "intermediate_nodes": len(mids),
        "raw_two_hop_candidates": len(candidates),
        "relation_pairs": len(rows),
        "rules": rows,
    }


def coverage_profile(all_results, min_paths):
    profiles = []

    for relation_result in all_results:
        relation = relation_result["first_hop_relation"]

        eligible = [
            r
            for r in relation_result["rules"]
            if r["paths"] >= min_paths
        ]

        strong = [
            r
            for r in eligible
            if (
                r["best_confirmation_count"] >= 50
                and r["best_confirmation_rate"] >= 0.05
                and r["distinct_subjects"] >= 20
            )
        ]

        usable = [
            r
            for r in eligible
            if (
                r["best_confirmation_count"] >= 20
                and r["best_confirmation_rate"] >= 0.02
                and r["distinct_subjects"] >= 10
            )
        ]

        gold = sum(
            r["best_confirmation_count"]
            for r in eligible
        )

        profiles.append(
            {
                "first_hop_relation": relation,
                "eligible_compositions": len(eligible),
                "strong_compositions": len(strong),
                "usable_compositions": len(usable),
                "confirmation_examples": gold,
                "max_confirmation_rate": max(
                    (
                        r["best_confirmation_rate"]
                        for r in eligible
                    ),
                    default=0.0,
                ),
                "top_rules": eligible[:20],
            }
        )

    return profiles


def main():
    ap = argparse.ArgumentParser(
        description="V565 targeted semantic composition audit"
    )

    ap.add_argument(
        "--database",
        default=r".\results\v562_kg_composition_audit.sqlite",
    )

    ap.add_argument(
        "--output",
        default=r".\results\v565_targeted_composition.json",
    )

    ap.add_argument(
        "--relations",
        nargs="+",
        default=DEFAULT_RELATIONS,
        help="First-hop relations to target.",
    )

    ap.add_argument(
        "--first-sample",
        type=int,
        default=1000,
        help="Actual edges sampled per first-hop relation.",
    )

    ap.add_argument(
        "--max-second-edges",
        type=int,
        default=200,
        help="Maximum outgoing edges followed from each intermediate.",
    )

    ap.add_argument(
        "--min-paths",
        type=int,
        default=20,
        help="Minimum two-hop paths for an eligible composition.",
    )

    ap.add_argument(
        "--top-rules",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=565,
    )

    args = ap.parse_args()

    start = time.perf_counter()

    db = resolve_db(args.database)

    print("=== V565 TARGETED COMPOSITION DISCOVERY / COVERAGE ===")
    print(f"database             : {db}")
    print(
        f"database size        : "
        f"{db.stat().st_size / (1024**3):.2f} GB"
    )
    print(f"target relations     : {len(args.relations)}")
    print(f"first-hop sample     : {args.first_sample:,}")
    print(f"max second-hop edges : {args.max_second_edges:,}")
    print(f"min paths            : {args.min_paths:,}")
    print("mode                 : disk-backed referential sampling")
    print("source               : READ-ONLY")
    print("LLM                  : NOT USED")
    print("conversation         : NOT USED")
    print()

    con = connect_ro(db)

    schema_info = check_schema(con)

    print(
        f"[SCHEMA] indexes={schema_info['indexes']}"
    )

    overview = graph_overview(con)

    print(
        f"[GRAPH] "
        f"edges={overview['edges']:,} "
        f"subjects={overview['subjects']:,} "
        f"relations={overview['relations']:,} "
        f"objects={overview['objects']:,} "
        f"seconds={overview['seconds']:.2f}"
    )

    counts = relation_counts(
        con,
        args.relations,
    )

    print()
    print("=== TARGET RELATION COVERAGE ===")

    for relation in args.relations:
        info = counts["relations"][relation]

        print(
            f"{relation:18s} "
            f"edges={info['edges']:12,} "
            f"subjects={info['subjects']:10,} "
            f"objects={info['objects']:10,}"
        )

    all_results = []

    print()
    print("=== TARGETED DISCOVERY ===")

    for i, relation in enumerate(
        args.relations,
        1,
    ):
        t = time.perf_counter()

        if counts["relations"][relation]["edges"] == 0:
            print(
                f"[{i}/{len(args.relations)}] "
                f"{relation}: NO EDGES"
            )
            continue

        result = discover_for_relation(
            con,
            relation,
            args.first_sample,
            args.max_second_edges,
            args.seed + i,
        )

        all_results.append(result)

        print(
            f"[{i}/{len(args.relations)}] "
            f"{relation:18s} "
            f"sampled={result['first_hop_sampled_edges']:5,} "
            f"2hop={result['raw_two_hop_candidates']:8,} "
            f"pairs={result['relation_pairs']:5,} "
            f"seconds={time.perf_counter()-t:.2f}"
        )

        for rule in result["rules"][:args.top_rules]:
            if rule["paths"] < args.min_paths:
                continue

            if rule["best_confirmation_count"] == 0:
                continue

            print(
                f"    {' -> '.join(rule['sequence']):32s} "
                f"=> {str(rule['best_confirmation_relation']):18s} "
                f"paths={rule['paths']:6,} "
                f"confirm={rule['best_confirmation_count']:5,} "
                f"rate={rule['best_confirmation_rate']:.3f}"
            )

    profiles = coverage_profile(
        all_results,
        args.min_paths,
    )

    total_strong = sum(
        x["strong_compositions"]
        for x in profiles
    )

    total_usable = sum(
        x["usable_compositions"]
        for x in profiles
    )

    total_confirmations = sum(
        x["confirmation_examples"]
        for x in profiles
    )

    if total_strong >= 10:
        verdict = "STRONG"
    elif total_strong >= 3 or total_usable >= 10:
        verdict = "USABLE"
    elif total_usable >= 3 or total_confirmations >= 100:
        verdict = "WEAK"
    else:
        verdict = "INSUFFICIENT"

    report = {
        "benchmark": "v565_targeted_composition_discovery",
        "database": str(db),
        "database_size_bytes": db.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": overview,
        "target_relations": args.relations,
        "relation_inventory": counts,
        "per_relation": all_results,
        "coverage": profiles,
        "summary": {
            "relations_targeted": len(args.relations),
            "relations_with_edges": sum(
                1
                for r in args.relations
                if counts["relations"][r]["edges"] > 0
            ),
            "strong_compositions": total_strong,
            "usable_compositions": total_usable,
            "confirmation_examples": total_confirmations,
            "verdict": verdict,
        },
        "config": {
            "first_sample": args.first_sample,
            "max_second_edges": args.max_second_edges,
            "min_paths": args.min_paths,
            "top_rules": args.top_rules,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - start,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("=" * 72)
    print("V565 RESULT")
    print("=" * 72)
    print(
        f"relations targeted  : "
        f"{len(args.relations)}"
    )
    print(
        f"strong compositions : "
        f"{total_strong:,}"
    )
    print(
        f"usable compositions : "
        f"{total_usable:,}"
    )
    print(
        f"confirmation cases  : "
        f"{total_confirmations:,}"
    )
    print(
        f"VERDICT              : "
        f"{verdict}"
    )
    print(
        f"JSON                 : "
        f"{output}"
    )
    print(
        f"elapsed              : "
        f"{time.perf_counter()-start:.2f}s"
    )


if __name__ == "__main__":
    main()

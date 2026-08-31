
#!/usr/bin/env python3
"""
V567 — Relation-Aware Targeted Composition Discovery

Fixes V566 in two important ways:

1. Worker SQLite connections receive a Path, so Windows file URI creation
   cannot fail with "'str' object has no attribute 'as_uri'".

2. The source graph is NOT assumed to contain canonical relation names such
   as has_part/capable_of/located_in. V567 inventories the actual predicate
   vocabulary and maps real predicates into semantic families using explicit
   aliases plus substring-safe rules.

The graph in the current experiment contains predicates such as:
    is_a
    schema:location
    yago:partOf
    schema:gender
    yago:hasFather
    schema:worksFor
etc.

The benchmark therefore works in two layers:

    raw predicate
        ↓
    semantic family
        ↓
    targeted first-hop discovery

This preserves the source predicate exactly in the results.

No writes are made to the source graph.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Semantic families
# ---------------------------------------------------------------------------

EXACT_FAMILY_ALIASES = {
    "is_a": "is_a",
    "isa": "is_a",
    "rdf:type": "is_a",
    "rdfs:subClassOf": "is_a",
    "schema:subClassOf": "is_a",

    "part_of": "part_of",
    "has_part": "has_part",
    "contains": "contains",
    "made_of": "made_of",

    "has_property": "has_property",
    "property": "has_property",

    "capable_of": "capable_of",
    "used_for": "used_for",
    "located_in": "located_in",
    "causes": "causes",
    "related_to": "related_to",
    "has": "has",
}


def semantic_family(predicate: str) -> str | None:
    """
    Conservative mapping.

    Do NOT map every predicate containing arbitrary words. Keep explicit
    high-confidence mappings and recognizable YAGO/DBpedia schema properties.
    """
    p = predicate.strip()

    if p in EXACT_FAMILY_ALIASES:
        return EXACT_FAMILY_ALIASES[p]

    low = p.lower()

    tail = low.rsplit(":", 1)[-1]
    tail = tail.rsplit("/", 1)[-1]

    if tail in {
        "type",
        "subclassof",
        "subclass",
        "isa",
        "is_a",
    }:
        return "is_a"

    if tail in {
        "partof",
        "part",
        "part_of",
    }:
        return "part_of"

    if tail in {
        "haspart",
        "has_part",
    }:
        return "has_part"

    if tail in {
        "contains",
        "contain",
        "hascontainer",
    }:
        return "contains"

    if tail in {
        "madeof",
        "material",
        "materialof",
    }:
        return "made_of"

    if tail in {
        "location",
        "locatedin",
        "place",
        "birthplace",
        "deathplace",
        "hometown",
    }:
        return "located_in"

    if tail in {
        "capableof",
        "capability",
    }:
        return "capable_of"

    if tail in {
        "usedfor",
        "used_for",
    }:
        return "used_for"

    if tail in {
        "causes",
        "cause",
    }:
        return "causes"

    if tail in {
        "property",
        "hasproperty",
    }:
        return "has_property"

    return None


# Target semantic families. We discover raw predicates belonging to each.
TARGET_FAMILIES = [
    "is_a",
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
        q = Path.cwd() / p
        if q.exists() and q.is_file():
            return q.resolve()

    results = Path.cwd() / "results"
    candidates = (
        sorted(
            results.glob("*.sqlite"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if results.exists()
        else []
    )

    lines = [
        "Could not find database.",
        f"requested: {path}",
        f"cwd: {Path.cwd()}",
    ]

    for x in candidates[:20]:
        lines.append(
            f"  {x} ({x.stat().st_size / 1024**3:.2f} GB)"
        )

    raise FileNotFoundError("\n".join(lines))


def connect_ro(path: Path) -> sqlite3.Connection:
    """
    Windows-safe, read-only SQLite connection.

    Note that Path is deliberate here. V566's worker bug passed a str and
    then called as_uri().
    """
    path = Path(path).resolve()

    uri = path.as_uri() + "?mode=ro"

    con = sqlite3.connect(
        uri,
        uri=True,
        timeout=180.0,
        check_same_thread=False,
    )

    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-32768")

    return con


def check_schema(con):
    cols = {
        r[1]
        for r in con.execute("PRAGMA table_info(edges)")
    }

    required = {
        "subject",
        "relation",
        "object",
        "source",
    }

    missing = required - cols

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


def exact_relation_inventory(con, min_edges: int):
    """
    This is one GROUP BY over relation only.

    Unlike COUNT(DISTINCT subject/object), this is cheap because idx_r exists.
    """
    t = time.perf_counter()

    rows = con.execute(
        """
        SELECT relation, COUNT(*) AS n
        FROM edges
        GROUP BY relation
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        """,
        (min_edges,),
    ).fetchall()

    inventory = []

    for row in rows:
        predicate = str(row["relation"])
        family = semantic_family(predicate)

        inventory.append(
            {
                "predicate": predicate,
                "count": row["n"],
                "semantic_family": family,
            }
        )

    return {
        "relations": inventory,
        "seconds": time.perf_counter() - t,
    }


def family_inventory(
    relation_rows,
    target_families,
):
    mapped = defaultdict(list)

    for row in relation_rows:
        fam = row["semantic_family"]

        if fam in target_families:
            mapped[fam].append(row)

    for fam in mapped:
        mapped[fam].sort(
            key=lambda x: x["count"],
            reverse=True,
        )

    return mapped


def relation_counts_by_predicate(
    con,
    predicates,
):
    result = {}

    for predicate in predicates:
        row = con.execute(
            """
            SELECT COUNT(*) AS edges
            FROM edges
            WHERE relation = ?
            """,
            (predicate,),
        ).fetchone()

        result[predicate] = int(row["edges"])

    return result


def reservoir_edges(
    con,
    predicate,
    n,
    seed,
):
    rng = random.Random(seed)

    reservoir = []
    seen = 0

    for row in con.execute(
        """
        SELECT subject, object, source
        FROM edges
        WHERE relation = ?
        """,
        (predicate,),
    ):
        seen += 1

        item = (
            row["subject"],
            row["object"],
            row["source"],
        )

        if len(reservoir) < n:
            reservoir.append(item)
            continue

        j = rng.randint(1, seen)

        if j <= n:
            reservoir[j - 1] = item

    return reservoir, seen


def load_outgoing(
    con,
    nodes,
    chunk,
):
    output = defaultdict(list)

    nodes = list(nodes)

    for i in range(0, len(nodes), chunk):
        part = nodes[i:i + chunk]

        if not part:
            continue

        placeholders = ",".join(
            "?" for _ in part
        )

        q = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, part):
            output[row["subject"]].append(
                (
                    row["relation"],
                    row["object"],
                    row["source"],
                )
            )

    return output


def endpoint_lookup(
    con,
    pairs,
    chunk,
):
    by_subject = defaultdict(set)

    for subject, endpoint in pairs:
        by_subject[subject].add(endpoint)

    result = defaultdict(set)

    subjects = list(by_subject)

    for i in range(0, len(subjects), chunk):
        part = subjects[i:i + chunk]

        placeholders = ",".join(
            "?" for _ in part
        )

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, part):
            subject = row["subject"]
            endpoint = row["object"]

            if endpoint in by_subject.get(subject, ()):
                result[(subject, endpoint)].add(
                    row["relation"]
                )

    return result


def audit_predicate(
    database: Path,
    predicate: str,
    family: str,
    first_sample: int,
    max_second_edges: int,
    max_paths: int,
    chunk: int,
    min_paths: int,
    seed: int,
):
    started = time.perf_counter()

    con = connect_ro(database)

    first_edges, total_edges = reservoir_edges(
        con,
        predicate,
        first_sample,
        seed,
    )

    mids = {
        obj
        for _, obj, _ in first_edges
    }

    adjacency = load_outgoing(
        con,
        mids,
        chunk,
    )

    candidates = []

    for subject, middle, source1 in first_edges:
        second = adjacency.get(
            middle,
            (),
        )

        if len(second) > max_second_edges:
            second = second[:max_second_edges]

        for r2, endpoint, source2 in second:
            if endpoint in {
                subject,
                middle,
            }:
                continue

            candidates.append(
                (
                    subject,
                    predicate,
                    family,
                    middle,
                    r2,
                    endpoint,
                    source1,
                    source2,
                )
            )

            if len(candidates) >= max_paths:
                break

        if len(candidates) >= max_paths:
            break

    unique = {}

    for row in candidates:
        key = (
            row[0],
            row[1],
            row[3],
            row[4],
            row[5],
        )
        unique[key] = row

    candidates = list(unique.values())

    endpoint_pairs = {
        (row[0], row[5])
        for row in candidates
    }

    direct = endpoint_lookup(
        con,
        endpoint_pairs,
        chunk,
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

    for row in candidates:
        (
            subject,
            r1,
            family1,
            middle,
            r2,
            endpoint,
            source1,
            source2,
        ) = row

        key = (r1, r2)

        st = stats[key]

        st["paths"] += 1
        st["subjects"].add(subject)
        st["endpoints"].add(endpoint)

        if source1 is not None:
            st["sources"].add(source1)

        if source2 is not None:
            st["sources"].add(source2)

        for confirmation in direct.get(
            (subject, endpoint),
            (),
        ):
            st["confirmed"][confirmation] += 1
            st["confirmed_subjects"][
                confirmation
            ].add(subject)
            st["confirmed_endpoints"][
                confirmation
            ].add(endpoint)

    rules = []

    for (r1, r2), st in stats.items():
        if st["paths"] < min_paths:
            continue

        if st["confirmed"]:
            best_relation, best_count = (
                st["confirmed"].most_common(1)[0]
            )
        else:
            best_relation, best_count = None, 0

        rules.append(
            {
                "first_hop_predicate": r1,
                "first_hop_family": family,
                "second_hop_predicate": r2,
                "second_hop_family": semantic_family(r2),
                "paths": st["paths"],
                "best_confirmation_relation": best_relation,
                "best_confirmation_family": (
                    semantic_family(best_relation)
                    if best_relation
                    else None
                ),
                "best_confirmation_count": best_count,
                "best_confirmation_rate": (
                    best_count / st["paths"]
                    if st["paths"]
                    else 0.0
                ),
                "distinct_subjects": len(
                    st["subjects"]
                ),
                "distinct_endpoints": len(
                    st["endpoints"]
                ),
                "source_diversity": len(
                    st["sources"]
                ),
                "confirmation_relations": [
                    {
                        "relation": rel,
                        "family": semantic_family(rel),
                        "count": count,
                        "rate": count / st["paths"],
                        "distinct_subjects": len(
                            st["confirmed_subjects"][rel]
                        ),
                        "distinct_endpoints": len(
                            st["confirmed_endpoints"][rel]
                        ),
                    }
                    for rel, count in
                    st["confirmed"].most_common(10)
                ],
            }
        )

    rules.sort(
        key=lambda x: (
            x["best_confirmation_rate"],
            x["best_confirmation_count"],
            x["paths"],
        ),
        reverse=True,
    )

    con.close()

    return {
        "first_hop_predicate": predicate,
        "first_hop_family": family,
        "first_hop_total_edges": total_edges,
        "first_hop_sampled_edges": len(first_edges),
        "intermediate_nodes": len(mids),
        "raw_two_hop_candidates": len(candidates),
        "relation_pairs": len(rules),
        "rules": rules,
        "seconds": time.perf_counter() - started,
    }


def classify(rules):
    buckets = {
        "strong": [],
        "usable": [],
        "thin": [],
        "weak": [],
    }

    for rule in rules:
        c = rule["best_confirmation_count"]
        rate = rule["best_confirmation_rate"]
        subjects = rule["distinct_subjects"]
        endpoints = rule["distinct_endpoints"]

        if (
            c >= 100
            and rate >= 0.05
            and subjects >= 50
            and endpoints >= 50
        ):
            buckets["strong"].append(rule)

        elif (
            c >= 25
            and rate >= 0.02
            and subjects >= 20
            and endpoints >= 20
        ):
            buckets["usable"].append(rule)

        elif c >= 5 and subjects >= 5:
            buckets["thin"].append(rule)

        else:
            buckets["weak"].append(rule)

    return buckets


def main():
    ap = argparse.ArgumentParser(
        description=(
            "V567 relation-aware targeted composition discovery"
        )
    )

    ap.add_argument(
        "--database",
        default=r".\results\v562_kg_composition_audit.sqlite",
    )

    ap.add_argument(
        "--output",
        default=r".\results\v567_targeted_composition.json",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--first-sample",
        type=int,
        default=1000,
    )

    ap.add_argument(
        "--max-second-edges",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--max-paths-per-predicate",
        type=int,
        default=50000,
    )

    ap.add_argument(
        "--in-chunk",
        type=int,
        default=300,
    )

    ap.add_argument(
        "--min-paths",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--min-predicate-edges",
        type=int,
        default=1000,
    )

    ap.add_argument(
        "--max-predicates-per-family",
        type=int,
        default=8,
        help=(
            "Cap raw predicates per semantic family. "
            "Highest-frequency predicates are chosen."
        ),
    )

    ap.add_argument(
        "--families",
        nargs="+",
        default=TARGET_FAMILIES,
    )

    ap.add_argument(
        "--top-rules",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=567,
    )

    args = ap.parse_args()

    start = time.perf_counter()

    db = resolve_db(args.database)

    print(
        "=== V567 RELATION-AWARE TARGETED COMPOSITION ==="
    )
    print(f"database              : {db}")
    print(
        f"database size         : "
        f"{db.stat().st_size / 1024**3:.2f} GB"
    )
    print(f"workers               : {args.workers}")
    print(
        f"families              : "
        f"{', '.join(args.families)}"
    )
    print(
        f"first-hop sample      : "
        f"{args.first_sample:,}"
    )
    print(
        f"max second-hop edges  : "
        f"{args.max_second_edges:,}"
    )
    print(
        f"max paths/predicate   : "
        f"{args.max_paths_per_predicate:,}"
    )
    print(
        f"min predicate edges   : "
        f"{args.min_predicate_edges:,}"
    )
    print("source                : READ-ONLY")
    print(
        "mode                  : "
        "parallel relation-stratified adjacency"
    )
    print("LLM                   : NOT USED")
    print()

    con = connect_ro(db)

    schema_info = check_schema(con)

    print(
        f"[SCHEMA] indexes: "
        f"{schema_info['indexes']}"
    )

    # Very cheap startup check.
    exact_edges = con.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    print(
        f"[GRAPH] edges={exact_edges:,}"
    )

    print()
    print(
        "[INVENTORY] reading actual predicate vocabulary..."
    )

    inventory = exact_relation_inventory(
        con,
        args.min_predicate_edges,
    )

    mapped = family_inventory(
        inventory["relations"],
        args.families,
    )

    print(
        f"[INVENTORY] predicates >= "
        f"{args.min_predicate_edges:,}: "
        f"{len(inventory['relations']):,}"
    )

    print()
    print("=== DISCOVERED TARGET PREDICATES ===")

    jobs = []

    for family in args.families:
        candidates = mapped.get(
            family,
            [],
        )

        selected = candidates[
            :args.max_predicates_per_family
        ]

        print(
            f"{family:18s}: "
            f"{len(candidates):3d} mapped, "
            f"{len(selected):2d} selected"
        )

        for row in selected:
            print(
                f"  {row['predicate']:32s} "
                f"edges={row['count']:,}"
            )

            jobs.append(
                (
                    family,
                    row["predicate"],
                )
            )

    con.close()

    if not jobs:
        raise RuntimeError(
            "No real predicates were discovered for the requested semantic "
            "families. Inspect relation inventory before continuing."
        )

    print()
    print(
        f"=== STARTING {len(jobs)} PREDICATE JOBS "
        f"WITH {min(args.workers, len(jobs))} WORKERS ==="
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=min(args.workers, len(jobs))
    ) as pool:

        futures = {}

        for i, (family, predicate) in enumerate(
            jobs,
            1,
        ):
            future = pool.submit(
                audit_predicate,
                db,                         # Path, not str
                predicate,
                family,
                args.first_sample,
                args.max_second_edges,
                args.max_paths_per_predicate,
                args.in_chunk,
                args.min_paths,
                args.seed + i,
            )

            futures[future] = (
                i,
                family,
                predicate,
            )

        completed = 0

        for future in as_completed(futures):
            i, family, predicate = futures[future]

            try:
                result = future.result()

            except Exception as exc:
                completed += 1

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"{family:18s} "
                    f"{predicate:32s} "
                    f"ERROR: {exc}",
                    flush=True,
                )
                continue

            completed += 1
            results.append(result)

            confirmations = [
                r
                for r in result["rules"]
                if r["best_confirmation_count"] > 0
            ]

            if confirmations:
                best = confirmations[0]

                best_text = (
                    f"{best['second_hop_predicate']} => "
                    f"{best['best_confirmation_relation']} "
                    f"rate="
                    f"{best['best_confirmation_rate']:.3f}"
                )
            else:
                best_text = "best=none"

            print(
                f"[{completed}/{len(jobs)}] "
                f"{family:18s} "
                f"{predicate:32s} "
                f"sample={result['first_hop_sampled_edges']:5,} "
                f"2hop={result['raw_two_hop_candidates']:7,} "
                f"rules={result['relation_pairs']:4,} "
                f"{best_text} "
                f"time={result['seconds']:.2f}s",
                flush=True,
            )

    results.sort(
        key=lambda x: (
            args.families.index(
                x["first_hop_family"]
            ),
            x["first_hop_predicate"],
        )
    )

    all_rules = [
        rule
        for result in results
        for rule in result["rules"]
    ]

    buckets = classify(all_rules)

    print()
    print(
        "=== COMPOSITION COVERAGE ==="
    )
    print(
        f"evaluated rules : "
        f"{len(all_rules):,}"
    )
    print(
        f"strong         : "
        f"{len(buckets['strong']):,}"
    )
    print(
        f"usable         : "
        f"{len(buckets['usable']):,}"
    )
    print(
        f"thin           : "
        f"{len(buckets['thin']):,}"
    )
    print(
        f"weak           : "
        f"{len(buckets['weak']):,}"
    )

    ranked = sorted(
        all_rules,
        key=lambda x: (
            x["best_confirmation_rate"],
            x["best_confirmation_count"],
            x["paths"],
        ),
        reverse=True,
    )

    print()
    print(
        "=== TOP COMPOSITION RULES ==="
    )

    for rule in ranked[:args.top_rules]:
        print(
            f"{rule['first_hop_predicate']:28s} "
            f"-> "
            f"{rule['second_hop_predicate']:28s} "
            f"=> "
            f"{str(rule['best_confirmation_relation']):28s} "
            f"paths={rule['paths']:7,} "
            f"confirm={rule['best_confirmation_count']:6,} "
            f"rate={rule['best_confirmation_rate']:.4f} "
            f"subjects={rule['distinct_subjects']:6,}"
        )

    # Family-level summary: this is more useful to the cognitive architecture
    # than a single global verdict.
    family_summary = {}

    for family in args.families:
        family_rules = [
            r
            for r in all_rules
            if r["first_hop_family"] == family
        ]

        family_summary[family] = {
            "predicates_evaluated": len(
                [
                    x
                    for x in results
                    if x["first_hop_family"] == family
                ]
            ),
            "composition_rules": len(
                family_rules
            ),
            "strong": sum(
                r in buckets["strong"]
                for r in family_rules
            ),
            "usable": sum(
                r in buckets["usable"]
                for r in family_rules
            ),
            "thin": sum(
                r in buckets["thin"]
                for r in family_rules
            ),
            "best_rule": (
                max(
                    family_rules,
                    key=lambda x: (
                        x["best_confirmation_rate"],
                        x["best_confirmation_count"],
                        x["paths"],
                    ),
                )
                if family_rules
                else None
            ),
        }

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

    report = {
        "benchmark": (
            "v567_relation_aware_targeted_composition"
        ),
        "database": str(db),
        "database_size_bytes": db.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": {
            "edges": exact_edges,
        },
        "relation_inventory": {
            "predicate_count": len(
                inventory["relations"]
            ),
            "min_predicate_edges": args.min_predicate_edges,
            "predicates": inventory["relations"],
        },
        "target_discovery": {
            "families_requested": args.families,
            "families_found": {
                family: len(
                    mapped.get(family, [])
                )
                for family in args.families
            },
            "selected_predicates": {
                family: [
                    x["predicate"]
                    for x in mapped.get(
                        family,
                        []
                    )[
                        :args.max_predicates_per_family
                    ]
                ]
                for family in args.families
            },
        },
        "workers": min(
            args.workers,
            len(jobs),
        ),
        "per_predicate": results,
        "family_summary": family_summary,
        "coverage": {
            "rules_evaluated": len(all_rules),
            "strong": len(buckets["strong"]),
            "usable": len(buckets["usable"]),
            "thin": len(buckets["thin"]),
            "weak": len(buckets["weak"]),
        },
        "top_rules": ranked[:args.top_rules],
        "verdict": verdict,
        "config": {
            "first_sample": args.first_sample,
            "max_second_edges": args.max_second_edges,
            "max_paths_per_predicate": (
                args.max_paths_per_predicate
            ),
            "in_chunk": args.in_chunk,
            "min_paths": args.min_paths,
            "min_predicate_edges": (
                args.min_predicate_edges
            ),
            "max_predicates_per_family": (
                args.max_predicates_per_family
            ),
            "seed": args.seed,
        },
        "elapsed_seconds": (
            time.perf_counter() - start
        ),
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

    print()
    print("=" * 72)
    print("V567 COMPLETE")
    print("=" * 72)
    print(f"VERDICT : {verdict}")
    print(f"JSON    : {output}")
    print(
        f"elapsed : "
        f"{report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V560 — Composition Data Sufficiency / Coverage Audit

Read-only audit of the real semantic graph.

Questions answered:
  1. Which predicate compositions R1 ∘ R2 occur?
  2. How many unique 2-hop/3-hop paths support each composition?
  3. How often is the composed endpoint directly confirmed by a candidate
     target relation?
  4. How many distinct subjects/endpoints/domains/sources support the pattern?
  5. How many hard negatives / near misses can be constructed?
  6. Which compositions pass a practical "trainability" gate?

This is NOT an inference engine and does NOT write inferred facts into the
source graph.

Important terminology:
  GOLD:
      A multi-hop path whose proposed composed relation is also directly
      present for the same (subject, endpoint). This is a graph self-
      confirmation, NOT universal truth.

  SILVER:
      A multi-hop path matching a candidate relation-preserving family but
      without direct confirmation.

  HARD_NEGATIVE:
      A path pattern that is common/plausible but whose endpoint is not
      confirmed for the proposed relation.

  DATA_GAP:
      A composition family has too few independent examples to judge.

The audit intentionally reports both raw path counts and "effective support":
many repeated paths from the same concepts/source do not count as independent
evidence.

Source graph:
  results/full_semantic_memory.sqlite
  tables: concepts, facts

Outputs:
  results/v560_composition_data_audit.json
  results/v560_composition_data_audit.sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FAMILY = {
    "is_a": "type",
    "isa": "type",
    "instance_of": "instance",
    "has": "possession",
    "has_part": "composition",
    "part_of": "composition",
    "contains": "composition",
    "made_of": "composition",
    "has_property": "property",
    "property": "property",
    "capable_of": "capability",
    "used_for": "purpose",
    "causes": "causality",
    "located_in": "location",
    "defined_as": "definition",
    "related_to": "association",
    "synonym": "lexical",
    "antonym": "lexical",
    "hypernym": "lexical",
    "hyponym": "lexical",
}

# Candidate target families. These are hypotheses used to measure data
# adequacy, not logical axioms.
COMPOSITION_TARGETS = {
    ("type", "possession"): {"possession", "composition"},
    ("type", "composition"): {"composition"},
    ("type", "property"): {"property"},
    ("type", "capability"): {"capability"},
    ("type", "purpose"): {"purpose"},
    ("type", "location"): {"location"},
    ("instance", "possession"): {"possession", "composition"},
    ("instance", "composition"): {"composition"},
    ("instance", "property"): {"property"},
    ("composition", "composition"): {"composition"},
    ("composition", "property"): {"property"},
    ("possession", "composition"): {"composition"},
    ("possession", "property"): {"property"},
    ("location", "property"): {"property"},
    ("capability", "property"): {"property"},
}

RELATION_TARGETS = defaultdict(set)
for rel, fam in FAMILY.items():
    RELATION_TARGETS[fam].add(rel)

NEGATIVE_PREFIXES = (
    "not", "without", "lacks", "lack", "cannot", "unable"
)


def connect_ro(path: str) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def columns(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def validate_schema(con):
    tables = {
        r[0]: columns(con, r[0])
        for r in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if "facts" not in tables or "concepts" not in tables:
        raise RuntimeError(
            "Expected facts and concepts tables. Found: " +
            ", ".join(sorted(tables))
        )
    required_facts = {"subject_id", "predicate", "object_id"}
    if not required_facts.issubset(tables["facts"]):
        raise RuntimeError(
            "facts missing: " +
            repr(sorted(required_facts - set(tables["facts"])))
        )
    if "concept_id" not in set(tables["concepts"]):
        raise RuntimeError("concepts.concept_id missing")
    return tables


def load_concept_metadata(con):
    cols = set(columns(con, "concepts"))
    label_col = "canonical" if "canonical" in cols else "display"

    names = {}
    ctypes = {}
    for r in con.execute(
        f'SELECT concept_id, "{label_col}" AS label, concept_type '
        f'FROM concepts'
    ):
        names[int(r["concept_id"])] = str(r["label"] or "")
        ctypes[int(r["concept_id"])] = str(r["concept_type"] or "")
    return names, ctypes


def load_graph(con, per_node, include_predicates):
    """
    Load only the semantic predicates we can evaluate. Keep the strongest
    per (subject,predicate,object) edge and bound outgoing degree.
    """
    placeholders = ",".join("?" for _ in include_predicates)
    sql = f"""
      SELECT subject_id, predicate, object_id,
             fact_type, domain, source_id, confidence, frequency
      FROM facts
      WHERE subject_id IS NOT NULL
        AND object_id IS NOT NULL
        AND predicate IN ({placeholders})
    """

    grouped = defaultdict(dict)
    source_map = defaultdict(set)
    domain_map = defaultdict(set)

    for r in con.execute(sql, sorted(include_predicates)):
        s = int(r["subject_id"])
        o = int(r["object_id"])
        p = str(r["predicate"]).lower()
        key = (p, o)
        score = (
            float(r["confidence"] or 0.0)
            * math.log1p(float(r["frequency"] or 1.0))
        )

        old = grouped[s].get(key)
        if old is None or score > old[-1]:
            grouped[s][key] = (
                o, p,
                str(r["fact_type"] or ""),
                str(r["domain"] or ""),
                r["source_id"],
                float(r["confidence"] or 0.0),
                float(r["frequency"] or 1.0),
                score,
            )

        source_map[(s, p, o)].add(
            int(r["source_id"]) if r["source_id"] is not None else -1
        )
        domain = str(r["domain"] or "")
        if domain:
            domain_map[(s, p, o)].add(domain)

    adj = {}
    for s, mapping in grouped.items():
        vals = sorted(
            mapping.values(),
            key=lambda x: (-x[7], x[1], x[0])
        )
        adj[s] = vals[:per_node]

    edge_set = set()
    edge_meta = {}

    for s, vals in adj.items():
        for edge in vals:
            o, p = edge[0], edge[1]
            edge_set.add((s, p, o))
            edge_meta[(s, p, o)] = {
                "sources": sorted(source_map.get((s, p, o), {-1})),
                "domains": sorted(domain_map.get((s, p, o), set())),
            }

    return adj, edge_set, edge_meta


def enumerate_2hop(adj, max_paths_per_subject):
    for subject, first_edges in adj.items():
        emitted = 0
        for first in first_edges:
            mid, p1 = first[0], first[1]
            for second in adj.get(mid, ()):
                endpoint, p2 = second[0], second[1]
                if endpoint == subject:
                    continue
                yield (subject, p1, mid, p2, endpoint)
                emitted += 1
                if emitted >= max_paths_per_subject:
                    break


def enumerate_3hop(adj, seed_subjects, max_paths_per_subject):
    # Only use on a sample of seeds because 3-hop combinations can explode.
    for subject in seed_subjects:
        emitted = 0
        for first in adj.get(subject, ()):
            n1, p1 = first[0], first[1]
            if n1 == subject:
                continue
            for second in adj.get(n1, ()):
                n2, p2 = second[0], second[1]
                if n2 in {subject, n1}:
                    continue
                for third in adj.get(n2, ()):
                    endpoint, p3 = third[0], third[1]
                    if endpoint in {subject, n1, n2}:
                        continue
                    yield (subject, p1, n1, p2, n2, p3, endpoint)
                    emitted += 1
                    if emitted >= max_paths_per_subject:
                        break
                if emitted >= max_paths_per_subject:
                    break
            if emitted >= max_paths_per_subject:
                break


def is_candidate_sequence(seq):
    fams = tuple(FAMILY.get(p, "other") for p in seq)
    return fams in COMPOSITION_TARGETS


def candidate_targets(seq):
    fams = tuple(FAMILY.get(p, "other") for p in seq)
    return COMPOSITION_TARGETS.get(fams, set())


def target_relations_for_families(fams):
    out = set()
    for f in fams:
        out.update(RELATION_TARGETS.get(f, set()))
    return out


def classify_paths(paths2, edge_set, edge_meta):
    """
    Aggregate 2-hop paths by predicate sequence and proposed target relation.
    For target relation selection we use relation-family hypotheses, then choose
    the endpoint's observed relation when it matches that family.

    This does NOT turn an unconfirmed path into a positive fact.
    """
    stats = defaultdict(lambda: {
        "paths": 0,
        "gold": 0,
        "silver": 0,
        "hard_negative": 0,
        "subjects": set(),
        "endpoints": set(),
        "source_ids": set(),
        "domains": set(),
        "gold_subjects": set(),
        "gold_endpoints": set(),
    })

    examples = []

    # Pre-index outgoing target relations for exact confirmation.
    outgoing_target = defaultdict(set)
    for s, p, o in edge_set:
        outgoing_target[(s, o)].add(p)

    for s, p1, mid, p2, endpoint in paths2:
        seq = (p1, p2)
        if not is_candidate_sequence(seq):
            continue

        fams = tuple(FAMILY.get(p, "other") for p in seq)
        target_fams = candidate_targets(seq)
        actual_targets = outgoing_target.get((s, endpoint), set())

        candidate_target_relations = [
            r for r in actual_targets
            if FAMILY.get(r) in target_fams
        ]

        if candidate_target_relations:
            # One endpoint may have multiple supported relations.
            for rel in candidate_target_relations:
                key = (seq, rel)
                st = stats[key]
                st["paths"] += 1
                st["gold"] += 1
                st["subjects"].add(s)
                st["endpoints"].add(endpoint)
                meta = edge_meta.get((s, rel, endpoint), {})
                st["source_ids"].update(meta.get("sources", []))
                st["domains"].update(meta.get("domains", []))
                st["gold_subjects"].add(s)
                st["gold_endpoints"].add(endpoint)

                if len(examples) < 10000:
                    examples.append({
                        "label": "GOLD",
                        "subject": s,
                        "relation": rel,
                        "object": endpoint,
                        "sequence": list(seq),
                        "path": [
                            [s, p1, mid],
                            [mid, p2, endpoint],
                        ],
                    })
        else:
            # No direct target relation. It is useful as SILVER evidence for a
            # candidate rule, but should not be called positive.
            for target_family in target_fams:
                rels = sorted(RELATION_TARGETS.get(target_family, set()))
                if not rels:
                    continue
                rel = rels[0]
                key = (seq, rel)
                st = stats[key]
                st["paths"] += 1
                st["silver"] += 1
                st["subjects"].add(s)
                st["endpoints"].add(endpoint)
                if len(examples) < 10000:
                    examples.append({
                        "label": "SILVER",
                        "subject": s,
                        "relation": rel,
                        "object": endpoint,
                        "sequence": list(seq),
                        "path": [
                            [s, p1, mid],
                            [mid, p2, endpoint],
                        ],
                    })
                break

    return stats, examples


def produce_negative_candidates(stats, edge_set, examples, seed, limit):
    """
    We do not invent categorical negatives. These are HARD_NEGATIVE CANDIDATES:
    path exists, but the proposed relation is not directly supported.

    For each retained candidate rule, sample the actual silver paths.
    """
    rng = random.Random(seed)
    silver_by_rule = defaultdict(list)

    for x in examples:
        if x["label"] == "SILVER":
            silver_by_rule[
                (tuple(x["sequence"]), x["relation"])
            ].append(x)

    out = []
    for key, rows in silver_by_rule.items():
        rng.shuffle(rows)
        for x in rows[: max(1, limit // max(1, len(silver_by_rule)))]:
            out.append({
                **x,
                "label": "HARD_NEGATIVE_CANDIDATE",
                "negative_reason": "no_direct_confirmation",
            })
            if len(out) >= limit:
                return out
    return out


def trainability(st):
    gold = st["gold"]
    subjects = len(st["gold_subjects"])
    sources = len(st["source_ids"] - {-1})
    endpoints = len(st["gold_endpoints"])
    total = st["paths"]

    precision = gold / total if total else 0.0

    # Practical engineering score. This is NOT a statistical confidence.
    diversity = (
        min(subjects, 100) / 100.0
        * min(max(sources, 1), 20) / 20.0
        * min(endpoints, 100) / 100.0
    )

    if gold >= 500 and subjects >= 100 and sources >= 5:
        gate = "TRAINABLE"
    elif gold >= 100 and subjects >= 30 and sources >= 3:
        gate = "PROMISING"
    elif gold >= 20 and subjects >= 10:
        gate = "THIN"
    else:
        gate = "DATA_POOR"

    return {
        "paths": total,
        "gold": gold,
        "silver": st["silver"],
        "gold_precision": precision,
        "distinct_subjects": len(st["subjects"]),
        "gold_subjects": subjects,
        "gold_endpoints": endpoints,
        "source_diversity": sources,
        "domain_diversity": len(st["domains"]),
        "diversity_score": diversity,
        "trainability_gate": gate,
    }


def save_shadow(path, rule_rows, gold_examples, silver_examples, negatives):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()

    con = sqlite3.connect(p)
    con.executescript("""
    DROP TABLE IF EXISTS examples;
    DROP TABLE IF EXISTS rules;

    CREATE TABLE rules (
        sequence TEXT NOT NULL,
        relation TEXT NOT NULL,
        paths INTEGER,
        gold INTEGER,
        silver INTEGER,
        gold_precision REAL,
        distinct_subjects INTEGER,
        gold_subjects INTEGER,
        gold_endpoints INTEGER,
        source_diversity INTEGER,
        domain_diversity INTEGER,
        diversity_score REAL,
        trainability_gate TEXT,
        PRIMARY KEY (sequence, relation)
    );

    CREATE TABLE examples (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        subject TEXT,
        relation TEXT,
        object TEXT,
        sequence TEXT,
        path_json TEXT
    );
    """)

    for r in rule_rows:
        con.execute(
            """INSERT INTO rules VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r["sequence"],
                r["relation"],
                r["paths"],
                r["gold"],
                r["silver"],
                r["gold_precision"],
                r["distinct_subjects"],
                r["gold_subjects"],
                r["gold_endpoints"],
                r["source_diversity"],
                r["domain_diversity"],
                r["diversity_score"],
                r["trainability_gate"],
            )
        )

    all_examples = gold_examples + silver_examples + negatives
    for i, x in enumerate(all_examples, 1):
        con.execute(
            """INSERT INTO examples
               VALUES (?,?,?,?,?,?,?)""",
            (
                i,
                x["label"],
                str(x["subject"]),
                x["relation"],
                str(x["object"]),
                " -> ".join(x["sequence"]),
                json.dumps(x["path"], ensure_ascii=False),
            )
        )

    con.commit()
    con.close()


def run(args):
    started = time.perf_counter()
    source = Path(args.source).resolve()

    print("=== V560 COMPOSITION DATA SUFFICIENCY / COVERAGE AUDIT ===")
    print(f"source       : {args.source}")
    print(f"shadow       : {args.shadow}")
    print(f"workers      : {args.workers}")
    print(f"max_hops     : {args.max_hops}")
    print(f"per_node     : {args.per_node}")
    print(f"seeds        : {args.seeds}")
    print(f"paths/seed   : {args.paths_per_seed}")
    print("source graph : READ-ONLY")
    print("LLM          : NOT USED")
    print()

    con = connect_ro(str(source))
    validate_schema(con)

    t = time.perf_counter()
    names, types = load_concept_metadata(con)
    print(
        f"[LOAD] concepts={len(names):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    include = set(FAMILY)
    t = time.perf_counter()
    adj, edge_set, edge_meta = load_graph(
        con, args.per_node, include
    )
    bounded_edges = sum(len(v) for v in adj.values())
    print(
        f"[LOAD] subjects={len(adj):,} "
        f"bounded_edges={bounded_edges:,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    # Seed selection deliberately mixes high-degree and random nodes.
    rng = random.Random(args.seed)
    seed_ids = list(adj)
    rng.shuffle(seed_ids)
    degree_sorted = sorted(seed_ids, key=lambda x: len(adj[x]), reverse=True)
    head_n = min(len(degree_sorted), max(100, args.seeds // 3))
    selected = degree_sorted[:head_n]
    rest = [x for x in seed_ids if x not in set(selected)]
    selected.extend(rest[: max(0, args.seeds - len(selected))])
    selected = selected[:args.seeds]

    # 2-hop mining over the selected seeds.
    t = time.perf_counter()
    paths2 = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        def worker(sid):
            return list(enumerate_2hop_for_subject(
                adj, sid, args.paths_per_seed
            ))
        futures = [ex.submit(worker, sid) for sid in selected]
        for f in futures:
            paths2.extend(f.result())

    print(
        f"[2-HOP] raw_paths={len(paths2):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    t = time.perf_counter()
    stats, examples = classify_paths(paths2, edge_set, edge_meta)
    print(
        f"[RULES] candidate_rules={len(stats):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    # 3-hop audit only for a fraction of seeds unless max_hops is 2.
    paths3 = []
    if args.max_hops >= 3 and args.seeds_3hop > 0:
        seeds3 = selected[:args.seeds_3hop]
        t = time.perf_counter()
        for p in enumerate_3hop(
            adj, seeds3, args.paths_per_seed_3hop
        ):
            paths3.append(p)
        print(
            f"[3-HOP] raw_paths={len(paths3):,} "
            f"seconds={time.perf_counter()-t:.3f}"
        )

    # The 3-hop inventory is reported, but the trainability gate is based on
    # 2-hop compositions in this first adequacy pass.
    rule_rows = []
    for (seq, rel), st in stats.items():
        r = trainability(st)
        r["sequence"] = " -> ".join(seq)
        r["relation"] = rel
        rule_rows.append(r)

    rule_rows.sort(
        key=lambda r: (
            {"TRAINABLE":0, "PROMISING":1, "THIN":2, "DATA_POOR":3}
            .get(r["trainability_gate"], 4),
            -r["gold"],
            -r["gold_precision"],
            -r["distinct_subjects"],
        )
    )

    # Hard-negative candidates are SILVER paths explicitly, never categorical
    # refutations.
    negatives = produce_negative_candidates(
        stats, edge_set, examples, args.seed, args.hard_negatives
    )

    gold_examples = [x for x in examples if x["label"] == "GOLD"]
    silver_examples = [x for x in examples if x["label"] == "SILVER"]

    gate_counts = Counter(r["trainability_gate"] for r in rule_rows)
    seq_counts = Counter(
        tuple(x["sequence"]) for x in examples
    )

    report = {
        "benchmark": "v560_composition_data_sufficiency",
        "source_graph_read_only": True,
        "source": str(source),
        "shadow_results_db": args.shadow,
        "graph": {
            "concepts": len(names),
            "subjects": len(adj),
            "bounded_edges": bounded_edges,
        },
        "paths": {
            "2hop_raw": len(paths2),
            "3hop_raw": len(paths3),
            "2hop_candidate_rule_paths": sum(
                r["paths"] for r in rule_rows
            ),
        },
        "composition_rules": {
            "candidate_rules": len(rule_rows),
            "gate_counts": dict(gate_counts),
            "top_rules": rule_rows[:args.top_rules],
            "sequence_counts": [
                {"sequence": list(s), "count": n}
                for s, n in seq_counts.most_common(args.top_sequences)
            ],
        },
        "dataset": {
            "gold": len(gold_examples),
            "silver": len(silver_examples),
            "hard_negative_candidates": len(negatives),
        },
        "adequacy": {
            "trainable_rules": sum(
                r["trainability_gate"] == "TRAINABLE"
                for r in rule_rows
            ),
            "promising_rules": sum(
                r["trainability_gate"] == "PROMISING"
                for r in rule_rows
            ),
            "thin_rules": sum(
                r["trainability_gate"] == "THIN"
                for r in rule_rows
            ),
            "data_poor_rules": sum(
                r["trainability_gate"] == "DATA_POOR"
                for r in rule_rows
            ),
            "recommendation": (
                "Enough composition data exists to train selected relation "
                "families; proceed with contrastive/controller training."
                if any(
                    r["trainability_gate"] == "TRAINABLE"
                    for r in rule_rows
                )
                else
                "Do not trust neural composition training yet. "
                "The graph contains candidates but insufficient independent "
                "gold support under the current confirmation criterion."
            ),
        },
        "config": {
            "workers": args.workers,
            "max_hops": args.max_hops,
            "per_node": args.per_node,
            "seeds": args.seeds,
            "paths_per_seed": args.paths_per_seed,
            "seeds_3hop": args.seeds_3hop,
            "paths_per_seed_3hop": args.paths_per_seed_3hop,
            "hard_negatives": args.hard_negatives,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    save_shadow(
        args.shadow,
        rule_rows,
        gold_examples,
        silver_examples,
        negatives,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== V560 RESULT ===")
    print(
        f"2-hop paths         : {len(paths2):,}"
    )
    print(
        f"3-hop paths         : {len(paths3):,}"
    )
    print(
        f"candidate rules     : {len(rule_rows):,}"
    )
    print(
        f"gold                : {len(gold_examples):,}"
    )
    print(
        f"silver              : {len(silver_examples):,}"
    )
    print(
        f"hard-negative cand. : {len(negatives):,}"
    )

    print()
    print("-- TRAINABILITY GATES --")
    for gate in ("TRAINABLE", "PROMISING", "THIN", "DATA_POOR"):
        print(f"  {gate:10s}: {gate_counts.get(gate, 0):,}")

    print()
    print("-- TOP RULES --")
    for r in rule_rows[: min(20, args.top_rules)]:
        print(
            f"  {r['trainability_gate']:10s} "
            f"{r['sequence']:32s} => {r['relation']:15s} "
            f"gold={r['gold']:5d} "
            f"precision={r['gold_precision']:.4f} "
            f"subjects={r['gold_subjects']:4d} "
            f"sources={r['source_diversity']:3d}"
        )

    print()
    print(
        "RECOMMENDATION : " +
        report["adequacy"]["recommendation"]
    )
    print(f"shadow DB      : {args.shadow}")
    print(f"JSON            : {output}")
    print(f"elapsed         : {report['elapsed_seconds']:.3f}s")

    con.close()


def enumerate_2hop_for_subject(adj, subject, max_paths):
    emitted = 0
    for first in adj.get(subject, ()):
        mid, p1 = first[0], first[1]
        if mid == subject:
            continue
        for second in adj.get(mid, ()):
            endpoint, p2 = second[0], second[1]
            if endpoint == subject:
                continue
            yield (subject, p1, mid, p2, endpoint)
            emitted += 1
            if emitted >= max_paths:
                return


def main():
    ap = argparse.ArgumentParser(
        description="V560 read-only semantic composition data audit"
    )
    ap.add_argument(
        "--source",
        default=r".\results\full_semantic_memory.sqlite"
    )
    ap.add_argument(
        "--shadow",
        default=r".\results\v560_composition_data_audit.sqlite"
    )
    ap.add_argument(
        "--output",
        default=r".\results\v560_composition_data_audit.json"
    )
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--per-node", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--paths-per-seed", type=int, default=100)
    ap.add_argument("--seeds-3hop", type=int, default=1000)
    ap.add_argument("--paths-per-seed-3hop", type=int, default=30)
    ap.add_argument("--hard-negatives", type=int, default=1000)
    ap.add_argument("--top-rules", type=int, default=50)
    ap.add_argument("--top-sequences", type=int, default=50)
    ap.add_argument("--seed", type=int, default=560)
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.per_node < 1:
        ap.error("--per-node must be >= 1")
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    run(args)


if __name__ == "__main__":
    main()

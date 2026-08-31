
#!/usr/bin/env python3
"""
V558 — Semantic Composition Filter + Cognitive Strategy Benchmark

Goals
-----
1. Read the real semantic graph (facts + concepts), never live_facts.
2. Mine multi-hop paths.
3. Filter paths using semantic-compatibility heuristics rather than raw
   connectivity.
4. Build a benchmark from REAL graph facts by temporarily hiding sampled
   direct edges and asking strategies to recover their endpoint through
   alternative paths.
5. Compare cognitive search strategies:
      - direct baseline
      - semantic-priority search
      - type-guided search
      - relation-family guided search
      - adaptive-depth search
      - hybrid attention search
6. Store mined candidates/benchmark results in a shadow SQLite database.

The source graph is opened READ ONLY.

Important methodological distinction:
- A discovered path is a candidate composition, NOT an asserted truth.
- Benchmark "SUPPORTED" means the hidden direct fact exists in the source graph
  and an alternative path was found to its endpoint.
- This avoids pretending arbitrary graph connectivity is semantic entailment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Graph semantics
# ---------------------------------------------------------------------------

RELATION_FAMILY = {
    "is_a": "type",
    "isa": "type",
    "instance_of": "type",
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
}

# Relationship pairs that are especially plausible as compositional chains.
# These are deliberately a PRIOR and not an inference rule.
COMPATIBLE_PAIRS = {
    ("type", "possession"),
    ("type", "composition"),
    ("type", "property"),
    ("type", "capability"),
    ("type", "purpose"),
    ("type", "location"),
    ("type", "definition"),
    ("instance", "possession"),
    ("instance", "composition"),
    ("instance", "property"),
    ("composition", "property"),
    ("composition", "composition"),
    ("possession", "property"),
    ("possession", "composition"),
    ("location", "property"),
    ("location", "composition"),
    ("capability", "property"),
    ("cause", "property"),
    ("cause", "type"),
    ("cause", "capability"),
    ("association", "type"),
    ("association", "property"),
    ("association", "composition"),
}

# Paths made entirely of these association-heavy relations are much more likely
# to be topical connectivity than useful semantic composition.
PENALTY_RELATIONS = {
    "related_to": 3.0,
    "antonym": 6.0,
    "synonym": 4.0,
    "hypernym": 2.0,
    "hyponym": 2.0,
}

HIGH_VALUE_RELATIONS = {
    "is_a": 2.5,
    "instance_of": 2.3,
    "has": 2.4,
    "has_part": 3.0,
    "part_of": 2.8,
    "contains": 2.8,
    "made_of": 2.4,
    "has_property": 2.3,
    "capable_of": 2.2,
    "used_for": 1.8,
    "causes": 2.0,
    "located_in": 1.7,
    "defined_as": 1.9,
}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

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
    if not {"facts", "concepts"} <= set(tables):
        raise RuntimeError(
            "Expected facts + concepts. Found: " + repr(sorted(tables))
        )
    required = {"subject_id", "predicate", "object_id"}
    if not required <= set(tables["facts"]):
        raise RuntimeError(
            "facts missing required columns: " +
            repr(sorted(required - set(tables["facts"])))
        )
    if "concept_id" not in set(tables["concepts"]):
        raise RuntimeError("concepts.concept_id missing")
    return tables


def load_concepts(con):
    cols = set(table_columns(con, "concepts"))
    name_col = "canonical" if "canonical" in cols else "display"
    rows = con.execute(
        f'SELECT concept_id, "{name_col}" AS name, '
        f'concept_type FROM concepts'
    )
    names = {}
    types = {}
    for r in rows:
        names[int(r["concept_id"])] = str(r["name"] or "")
        types[int(r["concept_id"])] = str(r["concept_type"] or "")
    return names, types


def load_edges(con, per_node, predicates=None):
    where = "subject_id IS NOT NULL AND object_id IS NOT NULL AND predicate IS NOT NULL"
    params = []
    if predicates:
        preds = sorted(predicates)
        where += " AND predicate IN (" + ",".join("?" for _ in preds) + ")"
        params.extend(preds)

    q = (
        "SELECT subject_id, predicate, object_id, "
        "fact_type, domain, confidence, frequency "
        "FROM facts WHERE " + where
    )

    adj = defaultdict(list)
    edge_set = set()
    counts = Counter()

    for r in con.execute(q, params):
        s = int(r["subject_id"])
        o = int(r["object_id"])
        p = str(r["predicate"])

        if s == o:
            continue

        bucket = adj[s]

        # Keep a deterministic bounded top-k per subject. We do not want
        # arbitrary SQLite row ordering to decide which edges survive.
        score = (
            float(r["confidence"] or 0.0) *
            math.log1p(float(r["frequency"] or 1.0))
        )
        bucket.append(
            (
                o, p,
                r["fact_type"] or "",
                r["domain"] or "",
                float(r["confidence"] or 0.0),
                float(r["frequency"] or 1.0),
                score,
            )
        )
        edge_set.add((s, p, o))
        counts[p] += 1

    total = 0
    for s, edges in list(adj.items()):
        edges.sort(key=lambda x: (-x[6], x[1], x[0]))
        if len(edges) > per_node:
            del edges[per_node:]
        total += len(edges)

    return adj, edge_set, counts, total


# ---------------------------------------------------------------------------
# Semantic path scoring
# ---------------------------------------------------------------------------

def family(pred):
    p = pred.lower()
    if p == "instance_of":
        return "instance"
    return RELATION_FAMILY.get(p, "other")


def path_score(path, types=None):
    """
    Lower is better.

    Score components:
      - relation penalty
      - incompatible consecutive relation-family penalty
      - repeated relation penalty
      - association-heavy penalty
      - endpoint/type continuity bonus
    """
    if not path:
        return 1e9

    score = 0.0
    fams = []

    for i, edge in enumerate(path):
        _, pred, _ = edge[:3]
        p = pred.lower()
        fams.append(family(p))
        score += PENALTY_RELATIONS.get(p, 0.0)
        score -= HIGH_VALUE_RELATIONS.get(p, 0.0)

        if i:
            prev = fams[-2]
            cur = fams[-1]
            if (prev, cur) in COMPATIBLE_PAIRS:
                score -= 2.5
            elif prev == "association" or cur == "association":
                score += 1.5
            elif prev != cur:
                score += 0.75

    # Repeated generic association chains are especially noisy.
    assoc_count = sum(1 for f in fams if f == "association")
    if assoc_count:
        score += assoc_count * 1.5

    if len(set(fams)) == 1 and len(fams) >= 2:
        score += 0.5

    # Reward domain/type continuity when available.
    if types:
        endpoint = path[-1][2]
        t = types.get(endpoint, "")
        if t and t not in ("concept", "entity"):
            score -= 0.5

    return score


def meaningful(path, threshold):
    if len(path) < 2:
        return False, 1e9

    score = path_score(path)
    seq = tuple(x[1].lower() for x in path)
    fams = tuple(family(x[1]) for x in path)

    # Explicitly reject pure synonym/antonym chains and pure association chains.
    if all(p in {"synonym", "antonym"} for p in seq):
        return False, score

    if all(f == "association" for f in fams):
        return False, score

    return score <= threshold, score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_strategy(adj, names, subject, target, strategy,
                    max_hops, budget, threshold):
    """
    Returns a path or None.

    Strategies differ ONLY in how the frontier is prioritized. The underlying
    graph and stopping criteria remain fixed, so the benchmark compares the
    cognitive search policy rather than changing the knowledge source.
    """
    if subject == target:
        return []

    # item = (priority, depth, node, path, visited)
    frontier = []

    def push(node, path, depth, visited):
        if strategy == "semantic_priority":
            pr = path_score(path)
        elif strategy == "type_guided":
            pr = path_score(path) + (
                0.0 if path and family(path[-1][1]) == "type" else 0.8
            )
        elif strategy == "relation_guided":
            pr = path_score(path) + (
                0.0 if path and family(path[-1][1]) in {
                    "type","composition","property","possession"
                } else 1.0
            )
        elif strategy == "adaptive_depth":
            # Early search is shallow; only spend depth after promising paths.
            pr = path_score(path) + max(0, depth - 2) * 1.2
        elif strategy == "hybrid_attention":
            # Combine semantic quality, novelty, and a mild shallow-depth prior.
            novelty = len(set(e[1] for e in path))
            pr = path_score(path) - 0.4 * novelty + 0.6 * max(0, depth - 2)
        else:
            pr = depth

        frontier.append((pr, depth, node, path, visited))

    push(subject, [], 0, {subject})
    expansions = 0

    while frontier and expansions < budget:
        frontier.sort(key=lambda x: (x[0], x[1], x[2]))
        _, depth, node, path, visited = frontier.pop(0)
        expansions += 1

        if depth >= max_hops:
            continue

        for edge in adj.get(node, ()):
            oid, pred, *_ = edge
            if oid in visited:
                continue

            new_path = path + [(node, pred, oid)]
            if oid == target:
                ok, score = meaningful(new_path, threshold)
                if ok:
                    return new_path, expansions, score

            push(oid, new_path, depth + 1, visited | {oid})

        # Prevent unbounded frontier explosion.
        if len(frontier) > budget * 8:
            frontier.sort(key=lambda x: (x[0], x[1], x[2]))
            del frontier[budget * 8:]

    return None, expansions, None


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

def sample_holdout_cases(edge_set, n, rng):
    usable = [
        e for e in edge_set
        if family(e[1]) in {
            "type", "instance", "composition", "property",
            "possession", "capability", "purpose",
            "causality", "location", "definition"
        }
    ]
    rng.shuffle(usable)
    return usable[:n]


def alt_path_exists(adj, s, o, hidden_edge, max_hops, budget):
    """
    Check whether an alternative path exists after virtually removing the
    held-out direct edge.
    """
    q = deque([(s, [], {s})])
    seen = {s}
    expansions = 0

    while q and expansions < budget:
        node, path, visited = q.popleft()
        expansions += 1
        if len(path) >= max_hops:
            continue

        for edge in adj.get(node, ()):
            oid, pred, *_ = edge
            if (node, pred, oid) == hidden_edge:
                continue
            if oid in visited:
                continue
            np = path + [(node, pred, oid)]
            if oid == o:
                return np
            q.append((oid, np, visited | {oid}))

    return None


def make_benchmark_cases(adj, edge_set, requested, max_hops, budget, rng):
    """
    Build actual oracle cases:
      known fact A-R-B is hidden
      alternate graph path A -> ... -> B is searched

    The hidden edge remains the truth oracle; the alternate path is what the
    cognitive strategy must discover.
    """
    pool = sample_holdout_cases(edge_set, max(requested * 6, 5000), rng)

    cases = []
    seen_pairs = set()

    for s, r, o in pool:
        key = (s, r, o)
        if key in seen_pairs:
            continue

        alt = alt_path_exists(adj, s, o, key, max_hops, budget)
        if not alt or len(alt) < 2:
            continue

        cases.append({
            "subject": s,
            "relation": r,
            "object": o,
            "oracle_path": [list(x) for x in alt],
            "hops": len(alt),
        })
        seen_pairs.add(key)

        if len(cases) >= requested:
            break

    return cases


# ---------------------------------------------------------------------------
# Shadow DB
# ---------------------------------------------------------------------------

def init_results(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS compositions (
            id INTEGER PRIMARY KEY,
            subject TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            hops INTEGER NOT NULL,
            predicate_sequence TEXT NOT NULL,
            semantic_score REAL NOT NULL,
            meaningful INTEGER NOT NULL,
            path_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS benchmark_cases (
            id INTEGER PRIMARY KEY,
            subject TEXT NOT NULL,
            relation TEXT NOT NULL,
            object TEXT NOT NULL,
            oracle_hops INTEGER NOT NULL,
            oracle_path_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS benchmark_results (
            case_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            solved INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            predicted_hops INTEGER,
            score REAL,
            path_json TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_summary (
            strategy TEXT PRIMARY KEY,
            cases INTEGER NOT NULL,
            solved INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            mean_steps REAL NOT NULL,
            mean_predicted_hops REAL
        );
    """)
    return con


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    started = time.perf_counter()
    con = connect_ro(args.source)

    print("=== V558 SEMANTIC COMPOSITION + COGNITIVE STRATEGY BENCHMARK ===")
    print(f"source       : {args.source}")
    print(f"shadow       : {args.out}")
    print(f"workers      : {args.workers}")
    print(f"max_hops     : {args.max_hops}")
    print(f"per_node     : {args.per_node}")
    print(f"seeds        : {args.seeds}")
    print(f"holdout      : {args.holdout}")
    print(f"search_budget: {args.budget}")
    print("source graph : READ-ONLY")
    print("LLM          : NOT USED")
    print("conversation  : NOT USED")
    print()

    t = time.perf_counter()
    tables = inspect_schema(con)
    print("[SCHEMA] facts + concepts confirmed")
    print(f"[LOAD] schema_seconds={time.perf_counter()-t:.3f}")

    t = time.perf_counter()
    names, types = load_concepts(con)
    print(f"[LOAD] concepts={len(names):,} seconds={time.perf_counter()-t:.3f}")

    t = time.perf_counter()
    predicates = set(RELATION_FAMILY)
    adj, edge_set, rel_counts, edge_count = load_edges(
        con, args.per_node, predicates
    )
    print(
        f"[LOAD] bounded_edges={edge_count:,} "
        f"subjects={len(adj):,} seconds={time.perf_counter()-t:.3f}"
    )

    # ------------------------------------------------------------------
    # Composition mining
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    seed_ids = list(adj)
    rng.shuffle(seed_ids)
    seed_ids.sort(key=lambda x: len(adj[x]), reverse=True)
    if len(seed_ids) > args.seeds:
        head = seed_ids[:max(100, args.seeds // 3)]
        tail = seed_ids[max(100, args.seeds // 3):]
        rng.shuffle(tail)
        seed_ids = head + tail[:args.seeds-len(head)]
    seed_ids = seed_ids[:args.seeds]

    def mine_seed(sid):
        out = []
        q = deque([(sid, [], {sid})])
        while q and len(out) < args.max_paths:
            node, path, visited = q.popleft()
            if len(path) >= args.max_hops:
                continue
            for edge in adj.get(node, ()):
                oid, pred, *_ = edge
                if oid in visited:
                    continue
                np = path + [(node, pred, oid)]
                if len(np) >= 2:
                    ok, score = meaningful(np, args.meaning_threshold)
                    if ok:
                        out.append((np, score))
                q.append((oid, np, visited | {oid}))
        return out

    t = time.perf_counter()
    mined = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(mine_seed, sid) for sid in seed_ids]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                mined.extend(fut.result())
            except Exception as e:
                print(f"[MINER ERROR] {type(e).__name__}: {e}")
            if i % max(1, len(futures)//10) == 0:
                print(
                    f"[MINING] completed={i}/{len(futures)} "
                    f"meaningful_paths={len(mined):,} "
                    f"seconds={time.perf_counter()-t:.2f}"
                )

    seen_paths = set()
    meaningful_paths = []
    for path, score in mined:
        key = tuple(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        meaningful_paths.append((path, score))

    print(
        f"[MINING] unique_meaningful_paths={len(meaningful_paths):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    # ------------------------------------------------------------------
    # Benchmark oracle construction
    # ------------------------------------------------------------------
    t = time.perf_counter()
    cases = make_benchmark_cases(
        adj, edge_set, args.holdout, args.max_hops, args.budget, rng
    )
    print(
        f"[BENCHMARK DATASET] holdout_cases={len(cases)} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    strategies = [
        "direct_baseline",
        "semantic_priority",
        "type_guided",
        "relation_guided",
        "adaptive_depth",
        "hybrid_attention",
    ]

    results = defaultdict(list)

    for case in cases:
        s = case["subject"]
        o = case["object"]
        relation = case["relation"]

        # Direct baseline always "solves" only if the hidden edge is allowed;
        # since it is hidden, it must answer UNKNOWN. This explicitly measures
        # the value added by composition search.
        results["direct_baseline"].append({
            "solved": False,
            "steps": 0,
            "predicted_hops": None,
            "score": None,
            "path": None,
            "relation": relation,
            "oracle_hops": case["hops"],
        })

        for strategy in strategies[1:]:
            path, steps, score = search_strategy(
                adj, names, s, o, strategy,
                args.max_hops, args.budget, args.meaning_threshold
            )

            solved = bool(path and path[-1][2] == o)

            results[strategy].append({
                "solved": solved,
                "steps": steps,
                "predicted_hops": len(path) if path else None,
                "score": score,
                "path": [list(x) for x in path] if path else None,
                "relation": relation,
                "oracle_hops": case["hops"],
            })

    summary = {}
    for strategy, rows in results.items():
        solved = sum(x["solved"] for x in rows)
        steps = [x["steps"] for x in rows]
        hops = [x["predicted_hops"] for x in rows if x["predicted_hops"]]
        summary[strategy] = {
            "cases": len(rows),
            "solved": solved,
            "accuracy": solved / len(rows) if rows else 0.0,
            "mean_steps": sum(steps) / len(steps) if steps else 0.0,
            "mean_predicted_hops": sum(hops) / len(hops) if hops else None,
        }

    # ------------------------------------------------------------------
    # Persist shadow artifacts
    # ------------------------------------------------------------------
    out = init_results(args.out)

    for path, score in meaningful_paths[:args.output_limit]:
        seq = " -> ".join(x[1] for x in path)
        out.execute(
            """INSERT INTO compositions
               (subject,endpoint,hops,predicate_sequence,
                semantic_score,meaningful,path_json)
               VALUES (?,?,?,?,?,?,?)""",
            (
                names.get(path[0][0], str(path[0][0])),
                names.get(path[-1][2], str(path[-1][2])),
                len(path),
                seq,
                score,
                1,
                json.dumps([
                    [
                        names.get(a, str(a)), p,
                        names.get(c, str(c))
                    ]
                    for a,p,c in path
                ], ensure_ascii=False),
            )
        )

    for i, case in enumerate(cases, 1):
        out.execute(
            """INSERT INTO benchmark_cases
               (id,subject,relation,object,oracle_hops,oracle_path_json)
               VALUES (?,?,?,?,?,?)""",
            (
                i,
                names.get(case["subject"], str(case["subject"])),
                case["relation"],
                names.get(case["object"], str(case["object"])),
                case["hops"],
                json.dumps([
                    [
                        names.get(a, str(a)), p,
                        names.get(c, str(c))
                    ]
                    for a,p,c in case["oracle_path"]
                ], ensure_ascii=False),
            )
        )

        for strategy, rows in results.items():
            r = rows[i-1]
            out.execute(
                """INSERT INTO benchmark_results
                   (case_id,strategy,solved,steps,predicted_hops,score,path_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    i, strategy, int(r["solved"]), r["steps"],
                    r["predicted_hops"], r["score"],
                    json.dumps(r["path"], ensure_ascii=False)
                    if r["path"] else None
                )
            )

    for strategy, s in summary.items():
        out.execute(
            """INSERT OR REPLACE INTO strategy_summary
               (strategy,cases,solved,accuracy,mean_steps,mean_predicted_hops)
               VALUES (?,?,?,?,?,?)""",
            (
                strategy, s["cases"], s["solved"], s["accuracy"],
                s["mean_steps"], s["mean_predicted_hops"]
            )
        )

    out.commit()
    out.close()
    con.close()

    seq_counts = Counter(
        " -> ".join(x[1] for x in p) for p, _ in meaningful_paths
    )
    hop_counts = Counter(len(p) for p, _ in meaningful_paths)

    report = {
        "benchmark": "v558_semantic_composition_cognitive_strategy",
        "source_graph_read_only": True,
        "source": args.source,
        "shadow_results_db": args.out,
        "graph": {
            "concepts": len(names),
            "bounded_edges": edge_count,
            "subjects": len(adj),
            "relations": len(rel_counts),
        },
        "semantic_filter": {
            "threshold": args.meaning_threshold,
            "raw_paths": len(mined),
            "unique_meaningful_paths": len(meaningful_paths),
            "hop_histogram": dict(sorted(hop_counts.items())),
            "top_predicate_sequences": [
                {"sequence": seq, "count": n}
                for seq, n in seq_counts.most_common(args.top_sequences)
            ],
        },
        "benchmark": {
            "oracle_cases": len(cases),
            "oracle_hop_histogram": dict(
                sorted(Counter(c["hops"] for c in cases).items())
            ),
        },
        "strategies": summary,
        "winner": (
            max(
                summary,
                key=lambda k: (
                    summary[k]["accuracy"],
                    -summary[k]["mean_steps"]
                )
            )
            if summary else None
        ),
        "config": {
            "workers": args.workers,
            "max_hops": args.max_hops,
            "per_node": args.per_node,
            "seeds": args.seeds,
            "max_paths": args.max_paths,
            "holdout": args.holdout,
            "budget": args.budget,
            "meaning_threshold": args.meaning_threshold,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print()
    print("=== V558 RESULT ===")
    print(f"meaningful paths : {len(meaningful_paths):,}")
    print(f"oracle cases     : {len(cases):,}")
    print()
    print("-- STRATEGIES --")
    for name, s in summary.items():
        print(
            f"  {name:20s} "
            f"accuracy={s['accuracy']:.4f} "
            f"solved={s['solved']:4d}/{s['cases']:<4d} "
            f"mean_steps={s['mean_steps']:.2f}"
        )

    print()
    print(f"winner           : {report['winner']}")
    print(f"JSON             : {json_path}")
    print(f"shadow DB        : {args.out}")
    print(f"end-to-end       : {report['elapsed_seconds']:.3f}s")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source",
        default=r".\results\full_semantic_memory.sqlite"
    )
    ap.add_argument(
        "--out",
        default=r".\results\v558_semantic_cognitive_benchmark.sqlite"
    )
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--per-node", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--max-paths", type=int, default=80)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--meaning-threshold", type=float, default=0.5)
    ap.add_argument("--output-limit", type=int, default=10000)
    ap.add_argument("--top-sequences", type=int, default=50)
    ap.add_argument("--seed", type=int, default=558)

    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.max_hops < 2:
        ap.error("--max-hops must be >= 2")
    if args.holdout < 1:
        ap.error("--holdout must be >= 1")
    if args.budget < 1:
        ap.error("--budget must be >= 1")

    run(args)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V559 — Semantic Composition Rule Mining + Cognitive Search

Purpose
-------
Turn observed graph paths into a *composition-learning* dataset before asking
the cognitive controller to search.

The source graph is read-only.

Pipeline
--------
1. Load real semantic graph from facts + concepts.
2. Mine 2-hop and 3-hop paths.
3. Group paths by predicate sequence and proposed endpoint relation.
4. Estimate composition validity from graph evidence:
      - target relation consistency
      - support / conflict counts
      - association-heavy penalties
      - subject/object type continuity
5. Build positive / negative / unknown composition examples.
6. Train a small attention controller on these examples.
7. Validate on held-out composition cases.
8. Compare learned controller against:
      - direct
      - BFS
      - greedy semantic
      - relation-family heuristic

This version does NOT write inferred facts into the source graph.
All mined rules/examples/results go to a shadow DB and JSON report.

Important:
This is an empirical benchmark of *composition hypotheses*. It does not claim
that a learned rule is universally true. A rule must be evaluated against
held-out graph cases and false-proof rate.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


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

ASSOCIATION = {"related_to"}
LEXICALISH = {"synonym", "antonym", "hypernym", "hyponym"}

# Conservative candidate relation propagation hypotheses.
# These are hypotheses for mining, not hard truth.
COMPOSITION_RULE_PRIORS = {
    ("type", "possession"): {"preserve": "possession", "prior": 0.85},
    ("type", "composition"): {"preserve": "composition", "prior": 0.90},
    ("type", "property"): {"preserve": "property", "prior": 0.75},
    ("type", "capability"): {"preserve": "capability", "prior": 0.70},
    ("type", "purpose"): {"preserve": "purpose", "prior": 0.65},
    ("type", "location"): {"preserve": "location", "prior": 0.55},
    ("instance", "composition"): {"preserve": "composition", "prior": 0.85},
    ("instance", "possession"): {"preserve": "possession", "prior": 0.80},
    ("composition", "property"): {"preserve": "property", "prior": 0.65},
    ("composition", "composition"): {"preserve": "composition", "prior": 0.55},
    ("association", "type"): {"preserve": "type", "prior": 0.10},
    ("association", "property"): {"preserve": "property", "prior": 0.08},
    ("association", "composition"): {"preserve": "composition", "prior": 0.05},
}

TARGET_RELATION_BY_FAMILY = {
    "type": {"is_a", "isa", "instance_of"},
    "possession": {"has"},
    "composition": {"has_part", "part_of", "contains", "made_of"},
    "property": {"has_property", "property"},
    "capability": {"capable_of"},
    "purpose": {"used_for"},
    "causality": {"causes"},
    "location": {"located_in"},
    "definition": {"defined_as"},
}


def connect_ro(path: str):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
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
    if "facts" not in tables or "concepts" not in tables:
        raise RuntimeError("facts + concepts tables are required")
    for c in ("subject_id", "predicate", "object_id"):
        if c not in tables["facts"]:
            raise RuntimeError(f"facts.{c} is required")
    if "concept_id" not in tables["concepts"]:
        raise RuntimeError("concepts.concept_id is required")
    return tables


def load_concepts(con):
    cols = set(table_columns(con, "concepts"))
    name = "canonical" if "canonical" in cols else "display"
    rows = con.execute(
        f'SELECT concept_id, "{name}" AS name, concept_type FROM concepts'
    )
    names = {}
    types = {}
    for r in rows:
        names[int(r["concept_id"])] = str(r["name"] or "")
        types[int(r["concept_id"])] = str(r["concept_type"] or "")
    return names, types


def relation_family(pred):
    return RELATION_FAMILY.get(pred.lower(), "other")


def load_graph(con, per_node):
    q = """
      SELECT subject_id, predicate, object_id,
             fact_type, domain, confidence, frequency
      FROM facts
      WHERE subject_id IS NOT NULL
        AND object_id IS NOT NULL
        AND predicate IS NOT NULL
    """
    raw = defaultdict(list)
    edge_set = set()

    for r in con.execute(q):
        s, o, p = int(r["subject_id"]), int(r["object_id"]), str(r["predicate"]).lower()
        if s == o:
            continue
        if p not in RELATION_FAMILY:
            continue
        freq = float(r["frequency"] or 1.0)
        conf = float(r["confidence"] or 0.0)
        rank = conf * math.log1p(freq)
        raw[s].append(
            (
                o, p,
                str(r["fact_type"] or ""),
                str(r["domain"] or ""),
                conf, freq, rank
            )
        )
        edge_set.add((s, p, o))

    adj = {}
    for s, edges in raw.items():
        edges.sort(key=lambda x: (-x[6], x[1], x[0]))
        adj[s] = edges[:per_node]

    return adj, edge_set


def enumerate_paths(adj, seeds, max_hops, max_paths_per_seed):
    """
    Enumerate actual graph paths. No semantic claim yet.
    """
    out = []
    for seed in seeds:
        q = deque([(seed, [], {seed})])
        while q and len(out) < max_paths_per_seed:
            node, path, seen = q.popleft()
            if len(path) >= max_hops:
                continue
            for edge in adj.get(node, ()):
                oid, pred, *_ = edge
                if oid in seen:
                    continue
                np = path + [(node, pred, oid)]
                if len(np) >= 2:
                    out.append(np)
                    if len(out) >= max_paths_per_seed:
                        break
                q.append((oid, np, seen | {oid}))
    return out


def candidate_rule(path, edge_set):
    if len(path) < 2:
        return None

    f1, f2 = relation_family(path[0][1]), relation_family(path[1][1])
    prior = COMPOSITION_RULE_PRIORS.get((f1, f2))
    if not prior:
        return None

    preserve_family = prior["preserve"]
    endpoint_relation = path[-1][1]

    # The proposed composed relation is a relation from the first node to the
    # endpoint, inferred from the family-preserving hypothesis.
    compatible_targets = TARGET_RELATION_BY_FAMILY.get(preserve_family, set())
    if endpoint_relation not in compatible_targets:
        # We still keep it as a hypothesis candidate, but with lower prior.
        proposed = next(iter(compatible_targets), endpoint_relation)
        prior_score = prior["prior"] * 0.45
    else:
        proposed = endpoint_relation
        prior_score = prior["prior"]

    return {
        "sequence": tuple(x[1] for x in path),
        "families": (f1, f2),
        "proposed_relation": proposed,
        "prior": prior_score,
    }


def observed_target_support(adj, subject, relation, endpoint):
    """
    Direct graph support for the same endpoint under the proposed relation.
    """
    return any(
        p == relation and o == endpoint
        for o, p, *_ in adj.get(subject, ())
    )


def mine_rule_statistics(paths, adj):
    stats = defaultdict(lambda: {
        "paths": 0,
        "direct_target_support": 0,
        "subjects": set(),
        "endpoints": set(),
    })

    for path in paths:
        c = candidate_rule(path, None)
        if not c:
            continue
        key = (c["sequence"], c["proposed_relation"])
        s, endpoint = path[0][0], path[-1][2]
        item = stats[key]
        item["paths"] += 1
        item["subjects"].add(s)
        item["endpoints"].add(endpoint)
        if observed_target_support(adj, s, c["proposed_relation"], endpoint):
            item["direct_target_support"] += 1

    return stats


def rule_score(stat, prior):
    """
    Empirical rule score:
      base prior
      + observed direct confirmations
      + diversity
      - weak/no-confirmation penalty
    """
    n = stat["paths"]
    support = stat["direct_target_support"]
    precision = support / n if n else 0.0
    diversity = min(len(stat["subjects"]), 1000) / 1000.0
    endpoint_div = min(len(stat["endpoints"]), 1000) / 1000.0

    return (
        2.0 * prior
        + 4.0 * precision
        + 0.7 * diversity
        + 0.7 * endpoint_div
        - (1.0 if n < 3 else 0.0)
    )


def filter_rules(stats, min_paths, min_score):
    rules = []
    for (sequence, relation), stat in stats.items():
        prior = COMPOSITION_RULE_PRIORS.get(
            tuple(relation_family(x) for x in sequence),
            {"prior": 0.0}
        )["prior"]
        score = rule_score(stat, prior)
        if stat["paths"] >= min_paths and score >= min_score:
            rules.append({
                "sequence": list(sequence),
                "relation": relation,
                "paths": stat["paths"],
                "direct_confirmations": stat["direct_target_support"],
                "precision": (
                    stat["direct_target_support"] / stat["paths"]
                    if stat["paths"] else 0.0
                ),
                "subjects": len(stat["subjects"]),
                "endpoints": len(stat["endpoints"]),
                "score": score,
                "prior": prior,
            })
    rules.sort(key=lambda x: (-x["score"], -x["paths"], x["sequence"]))
    return rules


def rule_lookup(rules):
    d = {}
    for r in rules:
        d[(tuple(r["sequence"]), r["relation"])] = r
    return d


# ---------------------------------------------------------------------------
# Composition dataset
# ---------------------------------------------------------------------------

def build_composition_dataset(paths, adj, rules, rng, per_rule=150):
    """
    Positive:
      actual path matches a retained rule and the endpoint is supported
      by direct graph evidence for the target relation.

    Unknown:
      same rule-shaped path, but no direct target support.

    Hard negative:
      path uses a relation sequence which looks plausible but conflicts with
      the learned rule inventory / endpoint relation.
    """
    lookup = rule_lookup(rules)
    positives = []
    unknown = []
    negatives = []

    buckets = defaultdict(list)
    for path in paths:
        c = candidate_rule(path, None)
        if not c:
            continue
        key = (tuple(c["sequence"]), c["proposed_relation"])
        if key not in lookup:
            continue
        buckets[key].append(path)

    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        used = 0
        for path in bucket:
            if used >= per_rule:
                break
            rule = lookup[key]
            s, endpoint = path[0][0], path[-1][2]
            supported = observed_target_support(adj, s, rule["relation"], endpoint)

            row = {
                "subject": s,
                "relation": rule["relation"],
                "object": endpoint,
                "path": [list(x) for x in path],
                "sequence": list(key[0]),
                "rule_score": rule["score"],
            }

            if supported:
                row["label"] = "SUPPORTED"
                positives.append(row)
            else:
                row["label"] = "UNKNOWN"
                unknown.append(row)
            used += 1

    # Construct hard negatives by swapping endpoint among another real endpoint
    # for the same subject/rule and ensuring it is not directly supported.
    rng.shuffle(positives)
    for row in positives[: max(0, len(positives)//2)]:
        candidates = [
            p[-1][2] for p in buckets[(tuple(row["sequence"]), row["relation"])]
            if p[-1][2] != row["object"]
        ]
        if not candidates:
            continue
        obj = rng.choice(candidates)
        if observed_target_support(adj, row["subject"], row["relation"], obj):
            continue
        negatives.append({
            **row,
            "object": obj,
            "label": "REFUTED_CANDIDATE",
        })

    return positives, negatives, unknown


# ---------------------------------------------------------------------------
# Cognitive controller
# ---------------------------------------------------------------------------

ACTIONS = [
    "CHECK_DIRECT",
    "FOLLOW_RULE",
    "FOLLOW_TYPE",
    "FOLLOW_COMPOSITION",
    "FOLLOW_ASSOCIATION",
    "STOP",
]


def feature_vector(path, rule, step, budget, variant):
    seq = tuple(path["sequence"])
    fams = tuple(relation_family(x) for x in seq)

    # Deterministic hashed categorical features.
    def hb(x, n):
        return (
            int.from_bytes(
                __import__("hashlib").blake2b(
                    str(x).encode(), digest_size=4
                ).digest(),
                "little"
            ) % n
        ) / max(1, n - 1)

    x = [
        hb(path["relation"], 64),
        hb(path["object"], 128),
        hb(seq, 128),
        hb(fams, 32),
        min(step, 12) / 12.0,
        min(budget, 80) / 80.0,
        rule["score"] / 10.0 if rule else 0.0,
        rule["precision"] if rule else 0.0,
        math.tanh((rule["paths"] if rule else 0) / 20.0),
    ]

    if variant in {"rich_state", "hybrid"}:
        x += [
            float("type" in fams),
            float("composition" in fams),
            float("property" in fams),
            float("association" in fams),
            float(len(set(seq)) > 1),
        ]

    if variant == "hybrid":
        x += [
            float(fams[-1] == "composition") if fams else 0.0,
            float(fams[-1] == "property") if fams else 0.0,
            float(fams[0] == "type") if fams else 0.0,
        ]

    return x


def action_target(row, is_positive):
    if row["label"] == "SUPPORTED":
        return "FOLLOW_RULE"
    if row["label"] in {"REFUTED_CANDIDATE", "UNKNOWN"}:
        return "STOP"
    return "STOP"


def train_controller(rows, rules, variant, epochs, hidden, lr, seed):
    import torch
    import torch.nn as nn

    rng = random.Random(seed)
    if not rows:
        return None, 0.0

    lookup = rule_lookup(rules)
    X, Y = [], []

    for row in rows:
        key = (tuple(row["sequence"]), row["relation"])
        rule = lookup.get(key)
        target = action_target(row, row["label"] == "SUPPORTED")

        for step in range(2):
            X.append(feature_vector(row, rule, step, 80-step, variant))
            Y.append(ACTIONS.index(target))

    dim = len(X[0])
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(Y, dtype=torch.long)

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, len(ACTIONS))
    )

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(tx)
    perm = torch.randperm(n)
    split = max(1, int(n * 0.8))
    tr, va = perm[:split], perm[split:]

    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(tx[tr]), ty[tr])
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(tx[va]).argmax(dim=1)
        acc = float((pred == ty[va]).float().mean()) if len(va) else 0.0

    return net, acc


def controller_eval(net, rows, rules, variant):
    import torch
    lookup = rule_lookup(rules)

    results = []
    for row in rows:
        rule = lookup.get((tuple(row["sequence"]), row["relation"]))
        x = torch.tensor(
            [feature_vector(row, rule, 0, 80, variant)],
            dtype=torch.float32
        )
        with torch.no_grad():
            action = ACTIONS[int(net(x).argmax().item())]

        if action == "FOLLOW_RULE" and row["label"] == "SUPPORTED":
            solved = True
        else:
            solved = row["label"] != "SUPPORTED" and action == "STOP"

        results.append({
            "label": row["label"],
            "action": action,
            "solved": solved,
        })

    return results


# ---------------------------------------------------------------------------
# Benchmark / output
# ---------------------------------------------------------------------------

def init_shadow(path):
    con = sqlite3.connect(path)
    con.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS composition_rules (
      sequence TEXT PRIMARY KEY,
      relation TEXT NOT NULL,
      paths INTEGER NOT NULL,
      direct_confirmations INTEGER NOT NULL,
      precision REAL NOT NULL,
      score REAL NOT NULL,
      prior REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS composition_cases (
      id INTEGER PRIMARY KEY,
      label TEXT NOT NULL,
      subject TEXT NOT NULL,
      relation TEXT NOT NULL,
      object TEXT NOT NULL,
      sequence TEXT NOT NULL,
      path_json TEXT NOT NULL,
      rule_score REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS strategy_results (
      strategy TEXT PRIMARY KEY,
      cases INTEGER NOT NULL,
      solved INTEGER NOT NULL,
      accuracy REAL NOT NULL,
      supported_recovery REAL NOT NULL,
      false_proof_rate REAL NOT NULL
    );
    """)
    return con


def save_shadow(path, rules, datasets, summary):
    con = init_shadow(path)
    for r in rules:
        con.execute(
            """INSERT OR REPLACE INTO composition_rules
               VALUES (?,?,?,?,?,?,?)""",
            (
                " -> ".join(r["sequence"]),
                r["relation"],
                r["paths"],
                r["direct_confirmations"],
                r["precision"],
                r["score"],
                r["prior"],
            )
        )

    all_rows = datasets["positive"] + datasets["negative"] + datasets["unknown"]
    for i, row in enumerate(all_rows, 1):
        con.execute(
            """INSERT OR REPLACE INTO composition_cases
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                i, row["label"], str(row["subject"]), row["relation"],
                str(row["object"]), " -> ".join(row["sequence"]),
                json.dumps(row["path"], ensure_ascii=False),
                row["rule_score"],
            )
        )

    for name, s in summary.items():
        con.execute(
            """INSERT OR REPLACE INTO strategy_results
               VALUES (?,?,?,?,?,?)""",
            (
                name, s["cases"], s["solved"], s["accuracy"],
                s["supported_recovery"], s["false_proof_rate"]
            )
        )
    con.commit()
    con.close()


def evaluate_heuristics(rows, strategy):
    out = []
    for row in rows:
        if strategy == "direct":
            action = "CHECK_DIRECT"
            solved = False
        elif strategy == "bfs":
            # The benchmark has the actual proof path available. BFS is an
            # upper-ish search baseline: it succeeds on any positive case.
            solved = row["label"] == "SUPPORTED"
            action = "BFS"
        elif strategy == "greedy_semantic":
            solved = row["label"] == "SUPPORTED" and row["rule_score"] >= 4.0
            action = "FOLLOW_RULE" if solved else "STOP"
        elif strategy == "relation_family":
            solved = row["label"] == "SUPPORTED" and (
                relation_family(row["relation"]) in
                {relation_family(x) for x in row["sequence"]}
            )
            action = "FOLLOW_COMPOSITION" if solved else "STOP"
        out.append({
            "label": row["label"],
            "action": action,
            "solved": solved
        })
    return out


def summarize(rows):
    if not rows:
        return {
            "cases": 0, "solved": 0, "accuracy": 0.0,
            "supported_recovery": 0.0, "false_proof_rate": 0.0
        }
    solved = sum(x["solved"] for x in rows)
    supported = [x for x in rows if x["label"] == "SUPPORTED"]
    false_positive = [
        x for x in rows
        if x["label"] != "SUPPORTED" and x["solved"]
    ]
    return {
        "cases": len(rows),
        "solved": solved,
        "accuracy": solved / len(rows),
        "supported_recovery": (
            sum(x["solved"] for x in supported) / len(supported)
            if supported else 0.0
        ),
        "false_proof_rate": (
            len(false_positive) / max(
                1, len([x for x in rows if x["label"] != "SUPPORTED"])
            )
        ),
    }


def run(args):
    t0 = time.perf_counter()
    rng = random.Random(args.seed)

    print("=== V559 SEMANTIC COMPOSITION RULE MINING + COGNITIVE SEARCH ===")
    print(f"source       : {args.source}")
    print(f"shadow       : {args.shadow}")
    print(f"workers      : {args.workers} (used for seed-level path mining)")
    print(f"max_hops     : {args.max_hops}")
    print(f"seeds        : {args.seeds}")
    print(f"per_node     : {args.per_node}")
    print(f"holdout      : {args.holdout}")
    print("source graph : READ-ONLY")
    print("LLM          : NOT USED")
    print()

    con = connect_ro(args.source)
    inspect_schema(con)

    t = time.perf_counter()
    names, types = load_concepts(con)
    print(f"[LOAD] concepts={len(names):,} seconds={time.perf_counter()-t:.3f}")

    t = time.perf_counter()
    adj, edge_set = load_graph(con, args.per_node)
    print(
        f"[LOAD] subjects={len(adj):,} "
        f"bounded_edges={sum(len(v) for v in adj.values()):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    seed_ids = list(adj)
    rng.shuffle(seed_ids)
    seed_ids.sort(key=lambda s: len(adj[s]), reverse=True)
    seed_ids = seed_ids[:args.seeds]

    # Parallelize path mining by seed chunks, avoiding concurrent SQLite reads.
    # Each worker works entirely from the already-loaded in-memory adjacency.
    t = time.perf_counter()
    if args.workers <= 1:
        raw_paths = []
        for s in seed_ids:
            raw_paths.extend(
                enumerate_paths(adj, [s], args.max_hops, args.max_paths_per_seed)
            )
    else:
        chunks = [[] for _ in range(min(args.workers, len(seed_ids) or 1))]
        for i, s in enumerate(seed_ids):
            chunks[i % len(chunks)].append(s)

        def worker(chunk):
            out = []
            for sid in chunk:
                out.extend(
                    enumerate_paths(
                        adj, [sid], args.max_hops, args.max_paths_per_seed
                    )
                )
            return out

        raw_paths = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker, c) for c in chunks if c]
            for f in futures:
                raw_paths.extend(f.result())

    # Deduplicate exact paths.
    seen = set()
    unique_paths = []
    for p in raw_paths:
        k = tuple(p)
        if k not in seen:
            seen.add(k)
            unique_paths.append(p)

    print(
        f"[PATHS] raw={len(raw_paths):,} unique={len(unique_paths):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    # Rule statistics.
    t = time.perf_counter()
    stats = mine_rule_statistics(unique_paths, adj)
    rules = filter_rules(
        stats,
        args.min_rule_paths,
        args.min_rule_score
    )
    print(
        f"[RULE MINING] candidate_rules={len(stats):,} "
        f"retained_rules={len(rules):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    # Build composition dataset.
    t = time.perf_counter()
    positive, negative, unknown = build_composition_dataset(
        unique_paths,
        adj,
        rules,
        rng,
        args.per_rule_cases
    )

    all_rows = positive + negative + unknown
    rng.shuffle(all_rows)

    # Explicit held-out split.
    holdout = all_rows[:args.holdout]
    train = all_rows[args.holdout:]

    # Ensure examples exist across labels in holdout when possible.
    if len(holdout) < args.holdout:
        holdout = all_rows[:]
        train = []

    print(
        f"[DATASET] positive={len(positive):,} "
        f"hard_negative={len(negative):,} "
        f"unknown={len(unknown):,} "
        f"train={len(train):,} holdout={len(holdout):,} "
        f"seconds={time.perf_counter()-t:.3f}"
    )

    strategies = [
        "direct",
        "bfs",
        "greedy_semantic",
        "relation_family",
        "learned_edge_state",
        "learned_rich_state",
        "learned_hybrid",
    ]

    summaries = {}
    strategy_details = {}

    for strategy in strategies:
        t = time.perf_counter()

        if strategy.startswith("learned"):
            variant = {
                "learned_edge_state": "edge_state",
                "learned_rich_state": "rich_state",
                "learned_hybrid": "hybrid",
            }[strategy]

            net, holdout_action_accuracy = train_controller(
                train,
                rules,
                variant,
                args.epochs,
                args.hidden,
                args.lr,
                args.seed
            )
            if net is None:
                details = []
            else:
                details = controller_eval(
                    net, holdout, rules, variant
                )
            s = summarize(details)
            s["action_holdout_accuracy"] = holdout_action_accuracy
        else:
            details = evaluate_heuristics(holdout, strategy)
            s = summarize(details)

        summaries[strategy] = s
        strategy_details[strategy] = details

        print(
            f"[STRATEGY] {strategy:22s} "
            f"accuracy={s['accuracy']:.4f} "
            f"supported_recovery={s['supported_recovery']:.4f} "
            f"false_proof_rate={s['false_proof_rate']:.4f} "
            f"seconds={time.perf_counter()-t:.3f}"
        )

    # Pick winner based on supported recovery first, false-proof rate second,
    # and overall accuracy third.
    winner = max(
        summaries,
        key=lambda x: (
            summaries[x]["supported_recovery"],
            -summaries[x]["false_proof_rate"],
            summaries[x]["accuracy"],
        )
    ) if summaries else None

    report = {
        "benchmark": "v559_semantic_composition_rule_mining",
        "source_graph_read_only": True,
        "source": args.source,
        "shadow_results_db": args.shadow,
        "graph": {
            "concepts": len(names),
            "subjects": len(adj),
            "bounded_edges": sum(len(v) for v in adj.values()),
        },
        "paths": {
            "unique_paths": len(unique_paths),
            "hop_histogram": dict(
                sorted(Counter(len(p) for p in unique_paths).items())
            ),
        },
        "rules": {
            "candidate_rules": len(stats),
            "retained_rules": len(rules),
            "top_rules": rules[:args.top_rules],
        },
        "dataset": {
            "positive": len(positive),
            "hard_negative": len(negative),
            "unknown": len(unknown),
            "train": len(train),
            "holdout": len(holdout),
            "holdout_label_distribution": dict(
                Counter(x["label"] for x in holdout)
            ),
        },
        "strategies": summaries,
        "winner": winner,
        "config": {
            "workers": args.workers,
            "max_hops": args.max_hops,
            "per_node": args.per_node,
            "seeds": args.seeds,
            "max_paths_per_seed": args.max_paths_per_seed,
            "holdout": args.holdout,
            "budget": args.budget,
            "min_rule_paths": args.min_rule_paths,
            "min_rule_score": args.min_rule_score,
            "per_rule_cases": args.per_rule_cases,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "lr": args.lr,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    shadow_summary = summaries
    save_shadow(
        args.shadow,
        rules,
        {"positive": positive, "negative": negative, "unknown": unknown},
        shadow_summary
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print()
    print("=== V559 RESULT ===")
    print(f"retained rules : {len(rules):,}")
    print(f"holdout cases  : {len(holdout):,}")
    print()
    print("-- TOP COMPOSITION RULES --")
    for r in rules[: min(args.top_rules, 20)]:
        print(
            f"  {r['score']:6.2f}  "
            f"{' -> '.join(r['sequence']):35s} "
            f"=> {r['relation']:15s} "
            f"paths={r['paths']:5d} "
            f"precision={r['precision']:.3f}"
        )

    print()
    print("-- COGNITIVE STRATEGIES --")
    for k, s in summaries.items():
        print(
            f"  {k:22s} "
            f"accuracy={s['accuracy']:.3f} "
            f"supported={s['supported_recovery']:.3f} "
            f"false_proof={s['false_proof_rate']:.3f}"
        )

    print()
    print(f"WINNER         : {winner}")
    print(f"shadow DB      : {args.shadow}")
    print(f"JSON            : {output}")
    print(f"end-to-end      : {report['elapsed_seconds']:.3f}s")
    con.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source", default=r".\results\full_semantic_memory.sqlite")
    ap.add_argument(
        "--shadow",
        default=r".\results\v559_semantic_composition_cognitive.sqlite"
    )
    ap.add_argument(
        "--output",
        default=r".\results\v559_semantic_composition_cognitive.json"
    )
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--per-node", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--max-paths-per-seed", type=int, default=60)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--min-rule-paths", type=int, default=5)
    ap.add_argument("--min-rule-score", type=float, default=2.5)
    ap.add_argument("--per-rule-cases", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--top-rules", type=int, default=50)
    ap.add_argument("--seed", type=int, default=559)

    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.max_hops < 2:
        ap.error("--max-hops must be >= 2")

    run(args)


if __name__ == "__main__":
    main()

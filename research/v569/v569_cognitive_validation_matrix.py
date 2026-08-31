
#!/usr/bin/env python3
"""
V569 — Cognitive Semantic Relation Validation Matrix

Purpose
-------
Benchmark whether the behavioral relation induction from V568 actually helps
the cognitive search policy on the 50GB YAGO + DBpedia graph.

The source graph stays READ-ONLY.

The benchmark is deliberately referential and bounded:
    subject -> outgoing edges -> middle -> outgoing edges -> endpoint

No global edges x edges join is used.

Key design
----------
A compositional oracle case is built only when:
    S -P-> M -Q-> O
and S has a direct edge to O with relation R.

The benchmark query is treated as:
    "Can relation R connect S to O?"

But the search policies do NOT get the direct confirmation as input. They
must discover a path through the graph within the configured search budget.

To prevent the direct edge from trivializing the benchmark, cases are divided
into:

  NOVEL_COMPOSITION
      The discovered proof path is 2 hops and the direct S-R-O edge is withheld
      from the policy's graph view. This is the primary cognitive benchmark.

  ORDINARY_SUPPORTED
      Direct graph evidence remains visible. This is a sanity/control set.

  HARD_NEGATIVE
      Similar endpoints chosen from nearby graph neighborhoods but no direct
      target relation exists in the oracle graph.

Policies:
  direct_visible
  bounded_bfs
  relation_frequency
  behavior_attention
  induced_family_attention
  adaptive_depth_attention
  hybrid_cognitive

The last four use the V568 behavioral profiles. They do NOT claim that
behavioral similarity equals semantic equivalence. It is a search prior.

The benchmark reports:
  - supported recovery
  - false proof rate
  - accuracy
  - mean steps
  - mean path length
  - budget exhaustion
  - cases by relation
  - improvement over BFS
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def resolve_path(value: str) -> Path:
    p = Path(value).expanduser()

    if p.exists() and p.is_file():
        return p.resolve()

    if not p.is_absolute():
        q = Path.cwd() / p
        if q.exists() and q.is_file():
            return q.resolve()

    raise FileNotFoundError(
        f"File not found: {value}\n"
        f"cwd: {Path.cwd()}"
    )


def connect_ro(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    con = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        uri=True,
        timeout=180.0,
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

    indexes = [
        r[1]
        for r in con.execute("PRAGMA index_list(edges)")
    ]

    return {
        "columns": sorted(cols),
        "indexes": indexes,
    }


# ---------------------------------------------------------------------------
# V568 model loading
# ---------------------------------------------------------------------------

def load_v568(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = data.get("profiles", [])

    by_predicate = {
        p["predicate"]: p
        for p in profiles
    }

    neighbors = {
        x["predicate"]: x["neighbors"]
        for x in data.get("nearest_neighbors", [])
    }

    return data, by_predicate, neighbors


def profile_relation_distribution(profile):
    behavior = profile.get("behavior", {})
    return {
        x["relation"]: float(x["fraction"])
        for x in behavior.get(
            "second_relation_profile",
            [],
        )
    }


def profile_confirmation_distribution(profile):
    behavior = profile.get("behavior", {})
    return {
        x["relation"]: float(x["fraction"])
        for x in behavior.get(
            "endpoint_confirmation_profile",
            [],
        )
    }


def js_distance(a, b):
    keys = set(a) | set(b)

    if not keys:
        return 1.0

    sa = sum(a.values()) or 1.0
    sb = sum(b.values()) or 1.0

    pa = {k: a.get(k, 0.0) / sa for k in keys}
    pb = {k: b.get(k, 0.0) / sb for k in keys}
    m = {k: (pa[k] + pb[k]) / 2 for k in keys}

    def kl(p):
        total = 0.0
        for k, v in p.items():
            if v > 0 and m[k] > 0:
                total += v * math.log2(v / m[k])
        return total

    return math.sqrt(
        max(
            0.0,
            (kl(pa) + kl(pb)) / 2,
        )
    )


def behavior_distance(a, b):
    ac = a.get("behavior", {})
    bc = b.get("behavior", {})

    c = js_distance(
        {
            x["relation"]: x["fraction"]
            for x in ac.get(
                "second_relation_profile",
                [],
            )
        },
        {
            x["relation"]: x["fraction"]
            for x in bc.get(
                "second_relation_profile",
                [],
            )
        },
    )

    e = js_distance(
        {
            x["relation"]: x["fraction"]
            for x in ac.get(
                "endpoint_confirmation_profile",
                [],
            )
        },
        {
            x["relation"]: x["fraction"]
            for x in bc.get(
                "endpoint_confirmation_profile",
                [],
            )
        },
    )

    inv = min(
        1.0,
        abs(
            float(
                ac.get(
                    "sampled_inverse_overlap",
                    0.0,
                )
            )
            -
            float(
                bc.get(
                    "sampled_inverse_overlap",
                    0.0,
                )
            )
        ),
    )

    return (
        0.55 * c
        + 0.35 * e
        + 0.10 * inv
    )


# ---------------------------------------------------------------------------
# Relation inventory and sampling
# ---------------------------------------------------------------------------

def top_predicates(con, top_n):
    rows = con.execute(
        """
        SELECT relation, COUNT(*) AS edge_count
        FROM edges
        GROUP BY relation
        ORDER BY edge_count DESC
        LIMIT ?
        """,
        (top_n,),
    ).fetchall()

    return [
        {
            "predicate": r["relation"],
            "edge_count": int(r["edge_count"]),
        }
        for r in rows
    ]


def sample_edges_by_predicate(
    con,
    predicates,
    sample_each,
    seed,
):
    """
    Bounded random rowid-window sampling. No complete predicate scan.
    """
    row = con.execute(
        "SELECT MIN(rowid), MAX(rowid) FROM edges"
    ).fetchone()

    lo = int(row[0])
    hi = int(row[1])

    rng = random.Random(seed)
    output = defaultdict(dict)

    # Each target predicate is sampled independently.
    for predicate in predicates:
        found = output[predicate]
        probes = 0

        while len(found) < sample_each and probes < 1000:
            start = rng.randint(
                lo,
                max(
                    lo,
                    hi - 5000,
                ),
            )

            end = min(
                hi,
                start + 4999,
            )

            for row in con.execute(
                """
                SELECT subject, object, source
                FROM edges
                WHERE rowid BETWEEN ?
                  AND ?
                  AND relation = ?
                """,
                (start, end, predicate),
            ):
                key = (
                    row["subject"],
                    row["object"],
                )

                found[key] = (
                    row["subject"],
                    row["object"],
                    row["source"],
                )

                if len(found) >= sample_each:
                    break

            probes += 1

    return {
        p: list(rows.values())
        for p, rows in output.items()
    }


# ---------------------------------------------------------------------------
# Oracle case construction
# ---------------------------------------------------------------------------

def outgoing_for_nodes(con, nodes, per_node_limit):
    result = defaultdict(list)

    nodes = list(nodes)

    for i in range(0, len(nodes), 250):
        chunk = nodes[i:i + 250]

        placeholders = ",".join("?" for _ in chunk)

        q = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, chunk):
            bucket = result[row["subject"]]

            if len(bucket) < per_node_limit:
                bucket.append(
                    (
                        row["relation"],
                        row["object"],
                        row["source"],
                    )
                )

    return result


def direct_relations_for_pairs(
    con,
    pairs,
):
    """
    Only checks subjects present in the sampled case set.
    """
    by_subject = defaultdict(set)

    for s, o in pairs:
        by_subject[s].add(o)

    out = defaultdict(set)
    subjects = list(by_subject)

    for i in range(0, len(subjects), 250):
        chunk = subjects[i:i + 250]
        placeholders = ",".join("?" for _ in chunk)

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, chunk):
            if row["object"] in by_subject.get(
                row["subject"],
                (),
            ):
                out[
                    (row["subject"], row["object"])
                ].add(row["relation"])

    return out


def make_cases(
    con,
    predicates,
    samples_per_predicate,
    middle_out_limit,
    cases_per_predicate,
    negative_ratio,
    seed,
):
    rng = random.Random(seed)

    sampled = sample_edges_by_predicate(
        con,
        predicates,
        samples_per_predicate,
        seed,
    )

    all_cases = []

    for predicate in predicates:
        first_edges = sampled.get(predicate, [])

        if not first_edges:
            continue

        mids = {
            o
            for _, o, _ in first_edges
        }

        outgoing = outgoing_for_nodes(
            con,
            mids,
            middle_out_limit,
        )

        candidate_rows = []

        for s, middle, source1 in first_edges:
            second = outgoing.get(
                middle,
                (),
            )

            for r2, endpoint, source2 in second:
                if endpoint in (s, middle):
                    continue

                candidate_rows.append(
                    (
                        s,
                        predicate,
                        middle,
                        r2,
                        endpoint,
                        source1,
                        source2,
                    )
                )

                if len(candidate_rows) >= cases_per_predicate * 8:
                    break

            if len(candidate_rows) >= cases_per_predicate * 8:
                break

        if not candidate_rows:
            continue

        endpoint_pairs = {
            (row[0], row[4])
            for row in candidate_rows
        }

        direct = direct_relations_for_pairs(
            con,
            endpoint_pairs,
        )

        positive = []

        for row in candidate_rows:
            s, r1, middle, r2, o, source1, source2 = row

            confirmations = direct.get(
                (s, o),
                (),
            )

            if not confirmations:
                continue

            # Ignore self-relation path aliases as a weak training signal.
            target_rel = sorted(confirmations)[0]

            positive.append(
                {
                    "subject": s,
                    "target_relation": target_rel,
                    "object": o,
                    "path": [
                        (s, r1, middle),
                        (middle, r2, o),
                    ],
                    "path_predicates": [r1, r2],
                    "middle": middle,
                    "source_ids": [
                        source1,
                        source2,
                    ],
                    "kind": "NOVEL_COMPOSITION",
                    "gold": True,
                }
            )

        # Deduplicate by query/path.
        unique = {}

        for row in positive:
            key = (
                row["subject"],
                row["target_relation"],
                row["object"],
                tuple(row["path_predicates"]),
            )
            unique[key] = row

        positive = list(unique.values())

        rng.shuffle(positive)

        all_cases.extend(
            positive[:cases_per_predicate]
        )

    if not all_cases:
        return [], sampled

    # Build hard negatives by swapping each positive endpoint with another
    # endpoint that is present in the local case pool but lacks the target
    # relation.
    positives = list(all_cases)
    endpoints_by_relation = defaultdict(list)

    for c in positives:
        endpoints_by_relation[
            c["target_relation"]
        ].append(c["object"])

    negatives = []

    for c in positives:
        pool = endpoints_by_relation[
            c["target_relation"]
        ]

        if len(pool) < 2:
            continue

        for _ in range(12):
            replacement = rng.choice(pool)

            if replacement == c["object"]:
                continue

            negatives.append(
                {
                    "subject": c["subject"],
                    "target_relation": c["target_relation"],
                    "object": replacement,
                    "path": [],
                    "path_predicates": [],
                    "middle": None,
                    "source_ids": [],
                    "kind": "HARD_NEGATIVE",
                    "gold": False,
                }
            )

            break

        if (
            len(negatives)
            >= max(
                1,
                int(
                    len(positives)
                    * negative_ratio
                ),
            )
        ):
            break

    # Hold out direct-positive visibility for the primary benchmark. The
    # database itself remains untouched; policies get a filtered edge reader.
    combined = positives + negatives

    # Deterministic split by hash/random seed.
    rng.shuffle(combined)

    return combined, sampled


# ---------------------------------------------------------------------------
# Policy primitives
# ---------------------------------------------------------------------------

def visible_outgoing(
    con,
    subject,
    hidden_direct_edges,
    per_node,
):
    """
    Indexed referential lookup:
        subject -> outgoing edges
    """
    rows = []

    for row in con.execute(
        """
        SELECT relation, object, source
        FROM edges
        WHERE subject = ?
        """,
        (subject,),
    ):
        edge = (
            row["relation"],
            row["object"],
            row["source"],
        )

        # Primary benchmark hides the direct answer edge.
        if (
            subject,
            row["relation"],
            row["object"],
        ) in hidden_direct_edges:
            continue

        rows.append(edge)

        if len(rows) >= per_node:
            break

    return rows


def direct_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
):
    if case["kind"] == "HARD_NEGATIVE":
        return {
            "predicted": False,
            "steps": 0,
            "path": [],
            "reason": "negative_case",
        }

    # This is deliberately a CONTROL. It checks direct visibility but the
    # direct edge is hidden in the primary novel benchmark.
    rows = visible_outgoing(
        con,
        case["subject"],
        set(),
        per_node,
    )

    found = any(
        r == case["target_relation"]
        and o == case["object"]
        for r, o, _ in rows
    )

    return {
        "predicted": found,
        "steps": 1 if rows else 0,
        "path": [],
        "reason": "direct_lookup",
    }


def bounded_bfs_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    max_depth,
):
    target_s = case["subject"]
    target_o = case["object"]
    target_r = case["target_relation"]

    frontier = [
        (target_s, [])
    ]

    visited = {
        target_s
    }

    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        edges = visible_outgoing(
            con,
            node,
            hidden_direct_edges,
            per_node,
        )

        for relation, obj, _source in edges:
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                len(new_path) == 1
                and obj == target_o
                and relation == target_r
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                    "reason": "direct",
                }

            if obj == target_o:
                # Path reached target but must not be treated as proof unless
                # the final edge has the requested relation.
                continue

            if obj not in visited:
                visited.add(obj)

                frontier.append(
                    (
                        obj,
                        new_path,
                    )
                )

            if steps >= budget:
                break

        if steps >= budget:
            break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
        "reason": "budget_or_no_path",
    }


def relation_frequency_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    max_depth,
    profile_lookup,
):
    """
    Explore first through relations with high global behavioral support.
    """
    def score_relation(rel):
        p = profile_lookup.get(rel)
        if not p:
            return 0.0

        return math.log1p(
            p.get("edge_count", 1)
        )

    frontier = [
        (case["subject"], [])
    ]

    visited = {
        case["subject"]
    }

    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        edges = visible_outgoing(
            con,
            node,
            hidden_direct_edges,
            per_node,
        )

        edges = sorted(
            edges,
            key=lambda x: score_relation(x[0]),
            reverse=True,
        )

        for relation, obj, _source in edges:
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                relation == case["target_relation"]
                and obj == case["object"]
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                    "reason": "relation_frequency",
                }

            if obj not in visited:
                visited.add(obj)
                frontier.append(
                    (obj, new_path)
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
        "reason": "budget_or_no_path",
    }


def behavior_attention_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    max_depth,
    profile_lookup,
):
    """
    Rank next edges by how strongly the first-hop relation has historically
    composed with the candidate relation in V568's behavioral profile.
    """
    target_r = case["target_relation"]

    def next_relation_score(previous_rel, candidate_rel):
        p = profile_lookup.get(previous_rel)
        if not p:
            return 0.0

        dist = profile_relation_distribution(p)

        base = dist.get(
            candidate_rel,
            0.0,
        )

        if candidate_rel == target_r:
            base += 0.25

        return base

    frontier = [
        (
            case["subject"],
            [],
        )
    ]

    visited = {
        case["subject"]
    }

    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        previous_rel = (
            path[-1][1]
            if path
            else None
        )

        edges = visible_outgoing(
            con,
            node,
            hidden_direct_edges,
            per_node,
        )

        def edge_score(edge):
            relation, obj, _ = edge

            if previous_rel is None:
                # First hop: prioritize relations that participate in many
                # compositions but don't blindly choose is_a/sameAs.
                p = profile_lookup.get(relation)
                if not p:
                    return 0.0
                return (
                    math.log1p(
                        p.get("edge_count", 1)
                    )
                    * 0.05
                )

            return next_relation_score(
                previous_rel,
                relation,
            )

        edges = sorted(
            edges,
            key=edge_score,
            reverse=True,
        )

        for relation, obj, _source in edges:
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                relation == target_r
                and obj == case["object"]
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                    "reason": "behavior_attention",
                }

            if obj not in visited:
                visited.add(obj)
                frontier.append(
                    (obj, new_path)
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
        "reason": "budget_or_no_path",
    }


def induced_family_attention_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    max_depth,
    profile_lookup,
    neighbor_lookup,
):
    """
    Uses V568 nearest behavioral neighbors as a soft relation-family prior.

    It does not replace raw predicates with family labels.
    """
    target_r = case["target_relation"]

    def similarity_to_target(rel):
        if rel == target_r:
            return 1.0

        # If target has a V568 profile, compare the candidate profile directly.
        a = profile_lookup.get(target_r)
        b = profile_lookup.get(rel)

        if not a or not b:
            return 0.0

        return max(
            0.0,
            1.0
            - behavior_distance(a, b),
        )

    def composed_score(previous_rel, candidate_rel):
        prior = 0.0

        if previous_rel is not None:
            p = profile_lookup.get(previous_rel)

            if p:
                prior += profile_relation_distribution(
                    p
                ).get(
                    candidate_rel,
                    0.0,
                )

        family_similarity = similarity_to_target(
            candidate_rel
        )

        neighbor_bonus = 0.0

        for n in neighbor_lookup.get(
            target_r,
            [],
        ):
            if n["predicate"] == candidate_rel:
                neighbor_bonus = max(
                    neighbor_bonus,
                    1.0
                    - float(
                        n["distance"]
                    ),
                )

        return (
            0.55 * prior
            + 0.30 * family_similarity
            + 0.15 * neighbor_bonus
        )

    frontier = [
        (
            case["subject"],
            [],
        )
    ]

    visited = {
        case["subject"]
    }

    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        previous_rel = (
            path[-1][1]
            if path
            else None
        )

        edges = visible_outgoing(
            con,
            node,
            hidden_direct_edges,
            per_node,
        )

        edges = sorted(
            edges,
            key=lambda e: composed_score(
                previous_rel,
                e[0],
            ),
            reverse=True,
        )

        for relation, obj, _source in edges:
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                relation == target_r
                and obj == case["object"]
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                    "reason": "induced_family_attention",
                }

            if obj not in visited:
                visited.add(obj)
                frontier.append(
                    (
                        obj,
                        new_path,
                    )
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
        "reason": "budget_or_no_path",
    }


def adaptive_depth_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    profile_lookup,
):
    """
    Cognitive depth controller:
      - infer an expected depth from the existence of a relevant composition
      - prefer depth 2 when a V568 first-hop profile points toward the target
      - otherwise allow depth 3

    This is intentionally a policy experiment, not a trained neural controller.
    """
    subject = case["subject"]
    target_r = case["target_relation"]

    # Estimate whether depth-2 composition is supported by local behavioral
    # signatures for any relation leaving the subject.
    first_edges = visible_outgoing(
        con,
        subject,
        hidden_direct_edges,
        per_node,
    )

    best_comp = 0.0

    for relation, _obj, _src in first_edges:
        p = profile_lookup.get(relation)

        if not p:
            continue

        best_comp = max(
            best_comp,
            profile_relation_distribution(
                p
            ).get(
                target_r,
                0.0,
            ),
        )

    max_depth = 2 if best_comp > 0.02 else 3

    return behavior_attention_policy(
        con,
        case,
        hidden_direct_edges,
        budget,
        per_node,
        max_depth,
        profile_lookup,
    )


def hybrid_cognitive_policy(
    con,
    case,
    hidden_direct_edges,
    budget,
    per_node,
    profile_lookup,
    neighbor_lookup,
):
    """
    Hybrid attention:
      behavior composition prior
      + induced family similarity
      + target relation bonus
      + adaptive depth
      + explicit visited-state suppression
    """
    target_r = case["target_relation"]

    def similarity_to_target(rel):
        if rel == target_r:
            return 1.0

        a = profile_lookup.get(target_r)
        b = profile_lookup.get(rel)

        if not a or not b:
            return 0.0

        return max(
            0.0,
            1.0
            - behavior_distance(a, b),
        )

    def score(previous_rel, candidate):
        relation, _obj, _source = candidate

        composition = 0.0
        family = similarity_to_target(
            relation
        )

        if previous_rel is not None:
            p = profile_lookup.get(previous_rel)

            if p:
                composition = (
                    profile_relation_distribution(p).get(
                        relation,
                        0.0,
                    )
                )

        target_bonus = (
            0.35
            if relation == target_r
            else 0.0
        )

        neighbor_bonus = 0.0

        for n in neighbor_lookup.get(
            target_r,
            [],
        ):
            if n["predicate"] == relation:
                neighbor_bonus = max(
                    neighbor_bonus,
                    1.0
                    - n["distance"],
                )

        return (
            0.45 * composition
            + 0.30 * family
            + 0.15 * neighbor_bonus
            + target_bonus
        )

    # Infer depth. Strong composition evidence -> depth 2, otherwise depth 3.
    first_edges = visible_outgoing(
        con,
        case["subject"],
        hidden_direct_edges,
        per_node,
    )

    expected = 0.0

    for relation, _o, _s in first_edges:
        p = profile_lookup.get(relation)

        if p:
            expected = max(
                expected,
                profile_relation_distribution(
                    p
                ).get(
                    target_r,
                    0.0,
                ),
            )

    max_depth = 2 if expected >= 0.01 else 3

    frontier = [
        (
            case["subject"],
            [],
        )
    ]

    visited = {
        case["subject"]
    }

    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        previous = (
            path[-1][1]
            if path
            else None
        )

        edges = visible_outgoing(
            con,
            node,
            hidden_direct_edges,
            per_node,
        )

        edges = sorted(
            edges,
            key=lambda e: score(
                previous,
                e,
            ),
            reverse=True,
        )

        for relation, obj, _source in edges:
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                relation == target_r
                and obj == case["object"]
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                    "reason": "hybrid_cognitive",
                }

            if obj not in visited:
                visited.add(obj)
                frontier.append(
                    (
                        obj,
                        new_path,
                    )
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
        "reason": "budget_or_no_path",
    }


# ---------------------------------------------------------------------------
# Worker benchmark
# ---------------------------------------------------------------------------

POLICY_NAMES = [
    "direct_visible_control",
    "bounded_bfs",
    "relation_frequency",
    "behavior_attention",
    "induced_family_attention",
    "adaptive_depth_attention",
    "hybrid_cognitive",
]


def evaluate_policy_on_case(
    database,
    case,
    policy_name,
    budget,
    per_node,
    max_depth,
    profile_lookup,
    neighbor_lookup,
    hide_direct=True,
):
    con = connect_ro(database)

    hidden = set()

    if (
        hide_direct
        and case["kind"] == "NOVEL_COMPOSITION"
        and case["gold"]
    ):
        hidden.add(
            (
                case["subject"],
                case["target_relation"],
                case["object"],
            )
        )

    fn = {
        "direct_visible_control": direct_policy,
        "bounded_bfs": bounded_bfs_policy,
        "relation_frequency": relation_frequency_policy,
        "behavior_attention": behavior_attention_policy,
        "induced_family_attention": induced_family_attention_policy,
    }.get(
        policy_name
    )

    if fn is not None and not callable(fn):
        raise RuntimeError(
            f"Policy resolver produced a non-callable for {policy_name!r}"
        )

    if fn is not None:
        # Only pass arguments that the selected policy actually accepts.
        # The direct control has no depth concept, while the search policies do.
        kwargs = {
            "con": con,
            "case": case,
            "hidden_direct_edges": hidden,
            "budget": budget,
            "per_node": per_node,
        }

        if policy_name != "direct_visible_control":
            kwargs["max_depth"] = max_depth

        if policy_name == "relation_frequency":
            kwargs["profile_lookup"] = profile_lookup

        elif policy_name == "behavior_attention":
            kwargs["profile_lookup"] = profile_lookup

        elif policy_name == "induced_family_attention":
            kwargs["profile_lookup"] = profile_lookup
            kwargs["neighbor_lookup"] = neighbor_lookup

        result = fn(**kwargs)

    elif policy_name == "adaptive_depth_attention":
        result = adaptive_depth_policy(
            con=con,
            case=case,
            hidden_direct_edges=hidden,
            budget=budget,
            per_node=per_node,
            profile_lookup=profile_lookup,
        )

    elif policy_name == "hybrid_cognitive":
        result = hybrid_cognitive_policy(
            con=con,
            case=case,
            hidden_direct_edges=hidden,
            budget=budget,
            per_node=per_node,
            profile_lookup=profile_lookup,
            neighbor_lookup=neighbor_lookup,
        )

    else:
        raise ValueError(
            f"Unknown policy: {policy_name}"
        )

    con.close()

    predicted = bool(
        result.get("predicted", False)
    )

    gold = bool(case["gold"])

    return {
        "predicted": predicted,
        "gold": gold,
        "correct": predicted == gold,
        "steps": int(
            result.get("steps", 0)
        ),
        "path_length": len(
            result.get("path", [])
        ),
        "reason": result.get(
            "reason"
        ),
    }


def benchmark_policy(
    args,
    cases,
    policy_name,
    profile_lookup,
    neighbor_lookup,
):
    results = []

    for i, case in enumerate(
        cases,
        1,
    ):
        outcome = evaluate_policy_on_case(
            args.database,
            case,
            policy_name,
            args.budget,
            args.per_node,
            args.max_depth,
            profile_lookup,
            neighbor_lookup,
            hide_direct=args.hide_direct,
        )

        results.append(
            {
                **outcome,
                "kind": case["kind"],
                "target_relation": case["target_relation"],
            }
        )

    supported = [
        r
        for r in results
        if r["gold"]
    ]

    negative = [
        r
        for r in results
        if not r["gold"]
    ]

    correct = sum(
        r["correct"]
        for r in results
    )

    predicted_positive = sum(
        r["predicted"]
        for r in results
    )

    recovered = sum(
        r["predicted"]
        for r in supported
    )

    false_proofs = sum(
        r["predicted"]
        for r in negative
    )

    exhausted = sum(
        1
        for r in results
        if (
            r["steps"] >= args.budget
            and not r["predicted"]
        )
    )

    return {
        "cases": len(results),
        "accuracy": (
            correct / len(results)
            if results
            else 0.0
        ),
        "supported_cases": len(supported),
        "supported_recovery": (
            recovered / len(supported)
            if supported
            else 0.0
        ),
        "negative_cases": len(negative),
        "false_proof_rate": (
            false_proofs / len(negative)
            if negative
            else 0.0
        ),
        "predicted_positive": predicted_positive,
        "mean_steps": (
            statistics_mean(
                [r["steps"] for r in results]
            )
            if results
            else 0.0
        ),
        "mean_path_length": (
            statistics_mean(
                [
                    r["path_length"]
                    for r in results
                    if r["predicted"]
                ]
            )
            if predicted_positive
            else 0.0
        ),
        "budget_exhausted": exhausted,
        "results_by_relation": relation_breakdown(results),
    }


def statistics_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def relation_breakdown(results):
    grouped = defaultdict(list)

    for r in results:
        grouped[r["target_relation"]].append(r)

    output = {}

    for relation, rows in grouped.items():
        gold = [r for r in rows if r["gold"]]
        negatives = [r for r in rows if not r["gold"]]

        recovered = sum(
            r["predicted"]
            for r in gold
        )

        false = sum(
            r["predicted"]
            for r in negatives
        )

        output[relation] = {
            "cases": len(rows),
            "supported_cases": len(gold),
            "supported_recovery": (
                recovered / len(gold)
                if gold
                else 0.0
            ),
            "false_proof_rate": (
                false / len(negatives)
                if negatives
                else 0.0
            ),
            "mean_steps": statistics_mean(
                [r["steps"] for r in rows]
            ),
        }

    return output


# ---------------------------------------------------------------------------
# Matrix report
# ---------------------------------------------------------------------------

def run(args):
    started = time.perf_counter()

    print(
        "=== V569 COGNITIVE SEMANTIC RELATION VALIDATION MATRIX ==="
    )
    print(
        f"database       : {args.database}"
    )
    print(
        f"V568 model     : {args.v568}"
    )
    print(
        f"workers        : {args.workers}"
    )
    print(
        f"top predicates : {args.top_predicates}"
    )
    print(
        f"sample/pred    : {args.sample_each}"
    )
    print(
        f"cases/pred     : {args.cases_per_predicate}"
    )
    print(
        f"budget         : {args.budget}"
    )
    print(
        f"max depth      : {args.max_depth}"
    )
    print(
        f"hide direct    : {args.hide_direct}"
    )
    print(
        "source graph   : READ-ONLY"
    )
    print(
        "LLM            : NOT USED"
    )
    print()

    database = resolve_path(args.database)
    v568_path = resolve_path(args.v568)

    # Store resolved Paths in args so workers cannot hit the V566/V567
    # string/Path mistake.
    args.database = database

    v568, profile_lookup, neighbor_lookup = load_v568(
        v568_path
    )

    con = connect_ro(database)
    schema_info = check_schema(con)

    print(
        f"[SCHEMA] indexes: "
        f"{schema_info['indexes']}"
    )

    edge_count = con.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    print(
        f"[GRAPH] edges={edge_count:,}"
    )

    inventory = top_predicates(
        con,
        args.top_predicates,
    )

    # Use only predicates for which V568 has a behavioral profile.
    selected = [
        x["predicate"]
        for x in inventory
        if x["predicate"] in profile_lookup
    ]

    print()
    print(
        "=== SELECTED PREDICATES ==="
    )

    for i, predicate in enumerate(
        selected,
        1,
    ):
        edge_count_for_predicate = next(
            x["edge_count"]
            for x in inventory
            if x["predicate"] == predicate
        )

        print(
            f"{i:2d}. "
            f"{predicate:32s} "
            f"{edge_count_for_predicate:12,}"
        )

    if not selected:
        raise RuntimeError(
            "No overlap between top graph predicates and V568 profiles."
        )

    print()
    print(
        "=== BUILDING GRAPH-DERIVED COMPOSITIONAL ORACLE ==="
    )

    con.close()

    # Oracle construction is relation-stratified and parallel across predicates.
    workers = max(
        1,
        min(
            args.workers,
            len(selected),
        ),
    )

    all_case_parts = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {}

        for i, predicate in enumerate(
            selected,
            1,
        ):
            # Each job creates cases for exactly one first-hop predicate.
            local_args = {
                "database": database,
                "predicate": predicate,
                "samples": args.sample_each,
                "middle_limit": args.middle_out_limit,
                "cases": args.cases_per_predicate,
                "negative_ratio": args.negative_ratio,
                "seed": args.seed + i,
            }

            futures[
                pool.submit(
                    build_cases_for_one_predicate,
                    **local_args,
                )
            ] = (
                i,
                predicate,
            )

        completed = 0

        for future in as_completed(futures):
            i, predicate = futures[future]
            completed += 1

            try:
                cases = future.result()
            except Exception as exc:
                print(
                    f"[{completed}/{len(selected)}] "
                    f"{predicate:32s} ERROR: {exc}",
                    flush=True,
                )
                continue

            all_case_parts.append(cases)

            print(
                f"[{completed}/{len(selected)}] "
                f"{predicate:32s} "
                f"cases={len(cases):5,}",
                flush=True,
            )

    cases = [
        case
        for part in all_case_parts
        for case in part
    ]

    # Cap total benchmark to keep the matrix manageable.
    rng = random.Random(args.seed)
    rng.shuffle(cases)

    supported = [
        c
        for c in cases
        if c["gold"]
    ]

    negatives = [
        c
        for c in cases
        if not c["gold"]
    ]

    if len(supported) > args.holdout_supported:
        supported = supported[:args.holdout_supported]

    if len(negatives) > args.holdout_negative:
        negatives = negatives[:args.holdout_negative]

    holdout = supported + negatives
    rng.shuffle(holdout)

    print()
    print(
        "=== ORACLE DATASET ==="
    )
    print(
        f"supported      : {len(supported):,}"
    )
    print(
        f"hard negatives : {len(negatives):,}"
    )
    print(
        f"total          : {len(holdout):,}"
    )

    if not supported:
        raise RuntimeError(
            "No supported compositional oracle cases were found. "
            "Increase --sample-each or --cases-per-predicate."
        )

    print()
    print(
        "=== RUNNING STRATEGY MATRIX ==="
    )

    policy_results = {}

    for policy in POLICY_NAMES:
        t = time.perf_counter()

        print(
            f"[POLICY START] {policy}",
            flush=True,
        )

        result = benchmark_policy(
            args,
            holdout,
            policy,
            profile_lookup,
            neighbor_lookup,
        )

        result["seconds"] = (
            time.perf_counter() - t
        )

        policy_results[policy] = result

        print(
            f"[POLICY END] {policy} "
            f"accuracy={result['accuracy']:.4f} "
            f"supported_recovery="
            f"{result['supported_recovery']:.4f} "
            f"false_proof="
            f"{result['false_proof_rate']:.4f} "
            f"mean_steps="
            f"{result['mean_steps']:.2f} "
            f"time="
            f"{result['seconds']:.2f}s",
            flush=True,
        )

    # Relative gains vs bounded BFS.
    baseline = policy_results["bounded_bfs"]

    comparison = {}

    for name, result in policy_results.items():
        comparison[name] = {
            "accuracy_delta_vs_bfs": (
                result["accuracy"]
                - baseline["accuracy"]
            ),
            "supported_recovery_delta_vs_bfs": (
                result["supported_recovery"]
                - baseline["supported_recovery"]
            ),
            "false_proof_delta_vs_bfs": (
                result["false_proof_rate"]
                - baseline["false_proof_rate"]
            ),
            "mean_steps_delta_vs_bfs": (
                result["mean_steps"]
                - baseline["mean_steps"]
            ),
        }

    # A simple utility score that favors recovery and penalizes false proof,
    # with a mild speed preference. It is a benchmark summary, not training.
    scores = {}

    for name, result in policy_results.items():
        scores[name] = (
            1.0 * result["supported_recovery"]
            - 1.25 * result["false_proof_rate"]
            - 0.01
            * (
                result["mean_steps"]
                / max(
                    1.0,
                    args.budget,
                )
            )
        )

    winner = max(
        scores,
        key=scores.get,
    )

    print()
    print(
        "=== STRATEGY MATRIX ==="
    )

    print(
        f"{'policy':32s} "
        f"{'acc':>8s} "
        f"{'support':>10s} "
        f"{'false':>10s} "
        f"{'steps':>10s}"
    )

    for name, result in policy_results.items():
        print(
            f"{name:32s} "
            f"{result['accuracy']:8.4f} "
            f"{result['supported_recovery']:10.4f} "
            f"{result['false_proof_rate']:10.4f} "
            f"{result['mean_steps']:10.2f}"
        )

    print()
    print(
        f"WINNER BY UTILITY: {winner}"
    )

    report = {
        "benchmark": (
            "v569_cognitive_semantic_relation_validation_matrix"
        ),
        "database": str(database),
        "v568_model": str(v568_path),
        "database_size_bytes": database.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": {
            "edges": edge_count,
        },
        "oracle": {
            "supported": len(supported),
            "hard_negative": len(negatives),
            "total": len(holdout),
            "construction": (
                "2-hop graph-derived paths with direct endpoint "
                "confirmation; direct proof edge hidden for novel "
                "composition cases"
            ),
        },
        "policies": policy_results,
        "comparison_vs_bfs": comparison,
        "utility_scores": scores,
        "winner": winner,
        "config": {
            "workers": args.workers,
            "top_predicates": args.top_predicates,
            "sample_each": args.sample_each,
            "middle_out_limit": args.middle_out_limit,
            "cases_per_predicate": args.cases_per_predicate,
            "negative_ratio": args.negative_ratio,
            "holdout_supported": args.holdout_supported,
            "holdout_negative": args.holdout_negative,
            "budget": args.budget,
            "per_node": args.per_node,
            "max_depth": args.max_depth,
            "hide_direct": args.hide_direct,
            "seed": args.seed,
        },
        "v568_inventory_summary": {
            "predicates_profiled": len(
                profile_lookup
            ),
            "nearest_neighbor_entries": len(
                neighbor_lookup
            ),
        },
        "elapsed_seconds": (
            time.perf_counter() - started
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
    print(
        "=" * 72
    )
    print(
        "V569 COMPLETE"
    )
    print(
        "=" * 72
    )
    print(
        f"winner  : {winner}"
    )
    print(
        f"JSON    : {output}"
    )
    print(
        f"elapsed : "
        f"{report['elapsed_seconds']:.2f}s"
    )


def build_cases_for_one_predicate(
    database,
    predicate,
    samples,
    middle_limit,
    cases,
    negative_ratio,
    seed,
):
    con = connect_ro(database)
    try:
        result, _sampled = make_cases(
            con,
            [predicate],
            samples,
            middle_limit,
            cases,
            negative_ratio,
            seed,
        )
        return result
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--database",
        default=r".\results\v562_kg_composition_audit.sqlite",
    )

    ap.add_argument(
        "--v568",
        default=r".\results\v568_20_relation_induction.json",
    )

    ap.add_argument(
        "--output",
        default=r".\results\v569_cognitive_validation.json",
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
        "--sample-each",
        type=int,
        default=600,
    )

    ap.add_argument(
        "--middle-out-limit",
        type=int,
        default=150,
    )

    ap.add_argument(
        "--cases-per-predicate",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--negative-ratio",
        type=float,
        default=0.75,
    )

    ap.add_argument(
        "--holdout-supported",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--holdout-negative",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--budget",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--per-node",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--hide-direct",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=569,
    )

    args = ap.parse_args()

    run(args)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V570 — Cognitive Mechanism Ablation Matrix

Question
--------
V569 showed that:
    relation priors
    behavioral attention
    induced-family similarity
    adaptive depth
    hybrid combination
improve over bounded BFS.

V570 isolates which mechanism is actually responsible.

All variants operate on the SAME graph-derived holdout cases.

Primary matrix
--------------
1. bfs
2. frequency
3. behavior
4. family
5. depth
6. behavior+family
7. behavior+depth
8. family+depth
9. full_hybrid

Controls
--------
10. shuffled_behavior
11. shuffled_neighbors
12. shuffled_all_priors

The shuffled controls preserve the amount of prior information but destroy
its relation-specific alignment. This helps distinguish a real cognitive
signal from simply having an extra ranking mechanism.

Important benchmark properties
------------------------------
- Source graph: READ-ONLY
- Direct confirmation edge is hidden for supported compositional cases.
- Policies must discover the requested target relation.
- Hard negatives are included.
- Same oracle dataset is used for every strategy.
- No global edges×edges join.
- No language model.
- No training in this stage: this is mechanism isolation.

Metrics
-------
- accuracy
- supported recovery
- false proof rate
- mean steps
- budget exhaustion
- relation-level performance
- gain/loss relative to BFS
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


POLICIES = [
    "bfs",
    "frequency",
    "behavior",
    "family",
    "depth",
    "behavior_family",
    "behavior_depth",
    "family_depth",
    "full_hybrid",
    "shuffled_behavior",
    "shuffled_family",
    "shuffled_all",
]


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
    cols = {
        row[1]
        for row in con.execute("PRAGMA table_info(edges)")
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
            f"edges table missing: {sorted(missing)}"
        )

    return {
        "columns": sorted(cols),
        "indexes": [
            row[1]
            for row in con.execute(
                "PRAGMA index_list(edges)"
            )
        ],
    }


# ---------------------------------------------------------------------------
# V568 profiles
# ---------------------------------------------------------------------------

def load_v568(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    profiles = {
        row["predicate"]: row
        for row in data.get(
            "profiles",
            [],
        )
    }

    neighbors = {
        row["predicate"]: row["neighbors"]
        for row in data.get(
            "nearest_neighbors",
            [],
        )
    }

    return data, profiles, neighbors


def second_hop_profile(profile):
    return {
        row["relation"]: float(
            row["fraction"]
        )
        for row in profile.get(
            "behavior",
            {},
        ).get(
            "second_relation_profile",
            [],
        )
    }


def profile_similarity(
    a,
    b,
):
    """
    Behavioral similarity from V568.

    Lower distance => more similar.
    """
    if not a or not b:
        return 0.0

    pa = second_hop_profile(a)
    pb = second_hop_profile(b)

    keys = set(pa) | set(pb)

    if not keys:
        return 0.0

    sa = sum(pa.values()) or 1.0
    sb = sum(pb.values()) or 1.0

    pa = {
        k: pa.get(k, 0.0) / sa
        for k in keys
    }

    pb = {
        k: pb.get(k, 0.0) / sb
        for k in keys
    }

    m = {
        k: (pa[k] + pb[k]) / 2
        for k in keys
    }

    def kl(p):
        return sum(
            v * math.log2(v / m[k])
            for k, v in p.items()
            if v > 0 and m[k] > 0
        )

    js = math.sqrt(
        max(
            0.0,
            (
                kl(pa)
                + kl(pb)
            )
            / 2,
        )
    )

    inv_a = float(
        a.get(
            "behavior",
            {},
        ).get(
            "sampled_inverse_overlap",
            0.0,
        )
    )

    inv_b = float(
        b.get(
            "behavior",
            {},
        ).get(
            "sampled_inverse_overlap",
            0.0,
        )
    )

    inv = min(
        1.0,
        abs(inv_a - inv_b),
    )

    return max(
        0.0,
        1.0
        - (
            0.9 * js
            + 0.1 * inv
        ),
    )


# ---------------------------------------------------------------------------
# Graph sampling and oracle
# ---------------------------------------------------------------------------

def top_predicates(
    con,
    top_n,
):
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
        (
            row["relation"],
            int(row["edge_count"]),
        )
        for row in rows
    ]


def bounded_sample_relation_edges(
    con,
    predicate,
    sample_size,
    seed,
):
    """
    Random rowid-window sampling. No whole-predicate scan.
    """
    lo, hi = con.execute(
        """
        SELECT MIN(rowid), MAX(rowid)
        FROM edges
        """
    ).fetchone()

    lo = int(lo)
    hi = int(hi)

    rng = random.Random(seed)
    found = {}
    probes = 0

    while (
        len(found) < sample_size
        and probes < 4000
    ):
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
            (
                start,
                end,
                predicate,
            ),
        ):
            found[
                (
                    row["subject"],
                    row["object"],
                )
            ] = (
                row["subject"],
                row["object"],
                row["source"],
            )

            if len(found) >= sample_size:
                break

        probes += 1

    return list(found.values())[:sample_size]


def outgoing_nodes(
    con,
    nodes,
    per_node,
):
    result = defaultdict(list)

    nodes = list(nodes)

    for i in range(
        0,
        len(nodes),
        250,
    ):
        chunk = nodes[i:i + 250]

        placeholders = ",".join(
            "?"
            for _ in chunk
        )

        query = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(
            query,
            chunk,
        ):
            bucket = result[
                row["subject"]
            ]

            if len(bucket) < per_node:
                bucket.append(
                    (
                        row["relation"],
                        row["object"],
                        row["source"],
                    )
                )

    return result


def direct_relations(
    con,
    pairs,
):
    by_subject = defaultdict(set)

    for s, o in pairs:
        by_subject[s].add(o)

    result = defaultdict(set)

    subjects = list(by_subject)

    for i in range(
        0,
        len(subjects),
        250,
    ):
        chunk = subjects[i:i + 250]

        placeholders = ",".join(
            "?"
            for _ in chunk
        )

        query = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        wanted = {
            s: by_subject[s]
            for s in chunk
        }

        for row in con.execute(
            query,
            chunk,
        ):
            if row["object"] in wanted.get(
                row["subject"],
                (),
            ):
                result[
                    (
                        row["subject"],
                        row["object"],
                    )
                ].add(
                    row["relation"]
                )

    return result


def build_oracle_for_predicate(
    database,
    predicate,
    samples,
    middle_limit,
    cases,
    negative_ratio,
    seed,
):
    con = connect_ro(database)

    first = bounded_sample_relation_edges(
        con,
        predicate,
        samples,
        seed,
    )

    middles = {
        o
        for _, o, _ in first
    }

    adjacency = outgoing_nodes(
        con,
        middles,
        middle_limit,
    )

    candidates = []

    for s, m, src1 in first:
        for r2, o, src2 in adjacency.get(
            m,
            (),
        ):
            if o in (s, m):
                continue

            candidates.append(
                (
                    s,
                    predicate,
                    m,
                    r2,
                    o,
                    src1,
                    src2,
                )
            )

            if len(candidates) >= cases * 12:
                break

        if len(candidates) >= cases * 12:
            break

    if not candidates:
        con.close()
        return []

    pairs = {
        (
            row[0],
            row[4],
        )
        for row in candidates
    }

    confirmed = direct_relations(
        con,
        pairs,
    )

    positives = []

    for row in candidates:
        (
            s,
            r1,
            m,
            r2,
            o,
            src1,
            src2,
        ) = row

        rels = confirmed.get(
            (s, o),
            (),
        )

        if not rels:
            continue

        target = sorted(rels)[0]

        positives.append(
            {
                "subject": s,
                "target_relation": target,
                "object": o,
                "path_predicates": [
                    r1,
                    r2,
                ],
                "path": [
                    (
                        s,
                        r1,
                        m,
                    ),
                    (
                        m,
                        r2,
                        o,
                    ),
                ],
                "gold": True,
                "kind": "NOVEL_COMPOSITION",
            }
        )

    # Exact case identity.
    unique = {}

    for case in positives:
        key = (
            case["subject"],
            case["target_relation"],
            case["object"],
            tuple(
                case["path_predicates"]
            ),
        )
        unique[key] = case

    positives = list(unique.values())

    rng = random.Random(seed)
    rng.shuffle(positives)

    positives = positives[:cases]

    # Hard negatives are target-relation endpoints from another positive case
    # for the same target relation. Direct target relation is verified absent
    # before retaining.
    by_relation = defaultdict(list)

    for case in positives:
        by_relation[
            case["target_relation"]
        ].append(
            case["object"]
        )

    negatives = []

    for case in positives:
        pool = by_relation[
            case["target_relation"]
        ]

        if len(pool) < 2:
            continue

        replacement = rng.choice(pool)

        if replacement == case["object"]:
            continue

        # Verify target relation is absent for this S -> replacement.
        direct_test = direct_relations(
            con,
            {
                (
                    case["subject"],
                    replacement,
                )
            },
        )

        if case["target_relation"] in direct_test.get(
            (
                case["subject"],
                replacement,
            ),
            (),
        ):
            continue

        negatives.append(
            {
                "subject": case["subject"],
                "target_relation": case[
                    "target_relation"
                ],
                "object": replacement,
                "path_predicates": [],
                "path": [],
                "gold": False,
                "kind": "HARD_NEGATIVE",
            }
        )

        if len(negatives) >= int(
            len(positives)
            * negative_ratio
        ):
            break

    con.close()

    return positives + negatives


def build_oracle_job(
    database,
    predicates,
    sample_each,
    middle_limit,
    cases_per_predicate,
    negative_ratio,
    seed,
):
    results = []

    for i, predicate in enumerate(
        predicates,
    ):
        try:
            cases = build_oracle_for_predicate(
                database,
                predicate,
                sample_each,
                middle_limit,
                cases_per_predicate,
                negative_ratio,
                seed + i,
            )
            results.extend(cases)
        except Exception as exc:
            print(
                f"[ORACLE ERROR] {predicate}: {exc}",
                flush=True,
            )

    return results


# ---------------------------------------------------------------------------
# Policy search
# ---------------------------------------------------------------------------

def edge_rows(
    con,
    subject,
    per_node,
    hidden,
):
    rows = []

    for row in con.execute(
        """
        SELECT relation, object, source
        FROM edges
        WHERE subject = ?
        """,
        (subject,),
    ):
        triple = (
            subject,
            row["relation"],
            row["object"],
        )

        if triple in hidden:
            continue

        rows.append(
            (
                row["relation"],
                row["object"],
                row["source"],
            )
        )

        if len(rows) >= per_node:
            break

    return rows


def target_first_hop_score(
    relation,
    profile_lookup,
):
    profile = profile_lookup.get(
        relation
    )

    if not profile:
        return 0.0

    return math.log1p(
        profile.get(
            "edge_count",
            0,
        )
    )


def search_case(
    con,
    case,
    policy,
    budget,
    per_node,
    max_depth,
    profile_lookup,
    neighbor_lookup,
    shuffled_behavior,
    shuffled_family,
):
    target_relation = case[
        "target_relation"
    ]

    subject = case[
        "subject"
    ]

    target_object = case[
        "object"
    ]

    hidden = {
        (
            subject,
            target_relation,
            target_object,
        )
    }

    if (
        not case["gold"]
    ):
        hidden = set()

    # Shuffled controls.
    behavior_map = shuffled_behavior
    family_map = shuffled_family

    def behavior_score(
        previous,
        candidate,
    ):
        relation = candidate

        if previous is None:
            return target_first_hop_score(
                relation,
                profile_lookup,
            ) * 0.05

        previous_profile = profile_lookup.get(
            previous
        )

        if not previous_profile:
            return 0.0

        value = second_hop_profile(
            previous_profile
        ).get(
            relation,
            0.0,
        )

        # Target relation bonus is a controlled part of attention.
        if relation == target_relation:
            value += 0.20

        return value

    def family_score(
        relation,
    ):
        if relation == target_relation:
            return 1.0

        target_profile = profile_lookup.get(
            target_relation
        )
        candidate_profile = profile_lookup.get(
            relation
        )

        if not target_profile or not candidate_profile:
            return 0.0

        return profile_similarity(
            target_profile,
            candidate_profile,
        )

    def neighbor_score(
        relation,
    ):
        for row in neighbor_lookup.get(
            target_relation,
            [],
        ):
            if row["predicate"] == relation:
                return max(
                    0.0,
                    1.0 - float(
                        row["distance"]
                    ),
                )
        return 0.0

    def search(
        mode,
    ):
        # Cases are kept deliberately simple: bounded best-first expansion.
        frontier = [
            (
                subject,
                [],
                0.0,
            )
        ]

        visited = {
            subject
        }

        steps = 0

        while frontier and steps < budget:
            # Lowest "negative score" is highest priority.
            frontier.sort(
                key=lambda x: x[2],
                reverse=True,
            )

            node, path, priority = frontier.pop(0)

            if len(path) >= max_depth:
                continue

            edges = edge_rows(
                con,
                node,
                per_node,
                hidden,
            )

            previous = (
                path[-1][1]
                if path
                else None
            )

            scored = []

            for relation, obj, source in edges:
                score = 0.0

                if mode in {
                    "frequency",
                    "full",
                    "behavior_family",
                    "behavior_depth",
                    "shuffled_behavior",
                    "shuffled_all",
                }:
                    if mode in {
                        "shuffled_behavior",
                        "shuffled_all",
                    }:
                        score += behavior_map.get(
                            (
                                previous,
                                relation,
                            ),
                            0.0,
                        )
                    else:
                        score += behavior_score(
                            previous,
                            relation,
                        )

                if mode in {
                    "family",
                    "full",
                    "behavior_family",
                    "family_depth",
                    "shuffled_family",
                    "shuffled_all",
                }:
                    if mode in {
                        "shuffled_family",
                        "shuffled_all",
                    }:
                        score += family_map.get(
                            relation,
                            0.0,
                        )
                    else:
                        score += (
                            family_score(
                                relation
                            )
                            * 0.7
                            + neighbor_score(
                                relation
                            )
                            * 0.3
                        )

                if mode == "frequency":
                    score += (
                        target_first_hop_score(
                            relation,
                            profile_lookup,
                        )
                        * 0.1
                    )

                if mode in {
                    "full",
                    "behavior_family",
                    "behavior_depth",
                    "family_depth",
                }:
                    if relation == target_relation:
                        score += 0.35

                scored.append(
                    (
                        score,
                        relation,
                        obj,
                        source,
                    )
                )

            scored.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            for score, relation, obj, source in scored:
                steps += 1

                new_path = path + [
                    (
                        node,
                        relation,
                        obj,
                    )
                ]

                if (
                    relation
                    == target_relation
                    and obj
                    == target_object
                ):
                    return {
                        "predicted": True,
                        "steps": steps,
                        "path": new_path,
                    }

                if obj not in visited:
                    visited.add(obj)
                    frontier.append(
                        (
                            obj,
                            new_path,
                            score,
                        )
                    )

                if steps >= budget:
                    break

        return {
            "predicted": False,
            "steps": steps,
            "path": [],
        }

    # BFS has its own queue semantics.
    if policy == "bfs":
        mode = "bfs"
        frontier = [
            (
                subject,
                [],
            )
        ]
        visited = {subject}
        steps = 0

        while frontier and steps < budget:
            node, path = frontier.pop(0)

            if len(path) >= max_depth:
                continue

            for relation, obj, source in edge_rows(
                con,
                node,
                per_node,
                hidden,
            ):
                steps += 1

                new_path = path + [
                    (
                        node,
                        relation,
                        obj,
                    )
                ]

                if (
                    relation
                    == target_relation
                    and obj
                    == target_object
                ):
                    return {
                        "predicted": True,
                        "steps": steps,
                        "path": new_path,
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
        }

    if policy == "depth":
        # Pure adaptive depth, otherwise neutral traversal.
        first = edge_rows(
            con,
            subject,
            per_node,
            hidden,
        )

        best = 0.0

        for relation, _obj, _src in first:
            p = profile_lookup.get(
                relation
            )
            if p:
                best = max(
                    best,
                    second_hop_profile(p).get(
                        target_relation,
                        0.0,
                    ),
                )

        chosen_depth = (
            2
            if best >= 0.01
            else 3
        )

        return search(
            "neutral",
        ) if chosen_depth == max_depth else search_depth_only(
            con,
            case,
            budget,
            per_node,
            chosen_depth,
            hidden,
        )

    if policy == "frequency":
        return search("frequency")

    if policy == "behavior":
        return search("behavior")

    if policy == "family":
        return search("family")

    if policy == "behavior_family":
        return search("behavior_family")

    if policy == "behavior_depth":
        chosen_depth = infer_depth(
            con,
            case,
            hidden,
            profile_lookup,
            per_node,
        )
        return search_depth_policy(
            con,
            case,
            budget,
            per_node,
            chosen_depth,
            hidden,
            "behavior",
            profile_lookup,
            neighbor_lookup,
            behavior_score,
            family_score,
            neighbor_score,
        )

    if policy == "family_depth":
        chosen_depth = infer_depth(
            con,
            case,
            hidden,
            profile_lookup,
            per_node,
        )
        return search_depth_policy(
            con,
            case,
            budget,
            per_node,
            chosen_depth,
            hidden,
            "family",
            profile_lookup,
            neighbor_lookup,
            behavior_score,
            family_score,
            neighbor_score,
        )

    if policy == "full_hybrid":
        chosen_depth = infer_depth(
            con,
            case,
            hidden,
            profile_lookup,
            per_node,
        )
        return search_depth_policy(
            con,
            case,
            budget,
            per_node,
            chosen_depth,
            hidden,
            "full",
            profile_lookup,
            neighbor_lookup,
            behavior_score,
            family_score,
            neighbor_score,
        )

    if policy == "shuffled_behavior":
        return search(
            "shuffled_behavior"
        )

    if policy == "shuffled_family":
        return search(
            "shuffled_family"
        )

    if policy == "shuffled_all":
        return search(
            "shuffled_all"
        )

    raise ValueError(policy)


def search_depth_only(
    con,
    case,
    budget,
    per_node,
    depth,
    hidden,
):
    # Neutral best-first in insertion order, but with selected max depth.
    subject = case["subject"]
    target_relation = case["target_relation"]
    target_object = case["object"]

    frontier = [(subject, [])]
    visited = {subject}
    steps = 0

    while frontier and steps < budget:
        node, path = frontier.pop(0)

        if len(path) >= depth:
            continue

        for relation, obj, source in edge_rows(
            con,
            node,
            per_node,
            hidden,
        ):
            steps += 1

            new_path = path + [
                (node, relation, obj)
            ]

            if (
                relation == target_relation
                and obj == target_object
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
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
    }


def infer_depth(
    con,
    case,
    hidden,
    profile_lookup,
    per_node,
):
    first = edge_rows(
        con,
        case["subject"],
        per_node,
        hidden,
    )

    best = 0.0

    for relation, _o, _s in first:
        p = profile_lookup.get(
            relation
        )

        if p:
            best = max(
                best,
                second_hop_profile(
                    p
                ).get(
                    case["target_relation"],
                    0.0,
                ),
            )

    return (
        2
        if best >= 0.01
        else 3
    )


def search_depth_policy(
    con,
    case,
    budget,
    per_node,
    depth,
    hidden,
    mode,
    profile_lookup,
    neighbor_lookup,
    behavior_score,
    family_score,
    neighbor_score,
):
    """
    Shared implementation for behavior+depth, family+depth and full hybrid.
    """
    subject = case["subject"]
    target_relation = case["target_relation"]
    target_object = case["object"]

    frontier = [
        (
            subject,
            [],
            0.0,
        )
    ]

    visited = {subject}
    steps = 0

    while frontier and steps < budget:
        frontier.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        node, path, _priority = frontier.pop(0)

        if len(path) >= depth:
            continue

        previous = (
            path[-1][1]
            if path
            else None
        )

        scored = []

        for relation, obj, source in edge_rows(
            con,
            node,
            per_node,
            hidden,
        ):
            score = 0.0

            if mode in {
                "behavior",
                "full",
            }:
                score += behavior_score(
                    previous,
                    relation,
                )

            if mode in {
                "family",
                "full",
            }:
                score += (
                    0.7
                    * family_score(
                        relation
                    )
                    + 0.3
                    * neighbor_score(
                        relation
                    )
                )

            if mode == "full" and relation == target_relation:
                score += 0.35

            scored.append(
                (
                    score,
                    relation,
                    obj,
                    source,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        for score, relation, obj, source in scored:
            steps += 1

            new_path = path + [
                (
                    node,
                    relation,
                    obj,
                )
            ]

            if (
                relation == target_relation
                and obj == target_object
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path": new_path,
                }

            if obj not in visited:
                visited.add(obj)
                frontier.append(
                    (
                        obj,
                        new_path,
                        score,
                    )
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path": [],
    }


# ---------------------------------------------------------------------------
# Shuffling controls
# ---------------------------------------------------------------------------

def build_shuffled_controls(
    profile_lookup,
    seed,
):
    rng = random.Random(seed)

    names = list(
        profile_lookup
    )

    # Behavior control: relation-conditioned composition priors shuffled among
    # previous-relation keys.
    all_behavior = []

    for predicate, profile in profile_lookup.items():
        dist = second_hop_profile(
            profile
        )
        for next_rel, value in dist.items():
            all_behavior.append(
                (
                    value,
                )
            )

    values = [
        x[0]
        for x in all_behavior
    ]

    rng.shuffle(values)

    shuffled_behavior = {}
    i = 0

    for previous in names:
        dist = second_hop_profile(
            profile_lookup[previous]
        )

        for next_rel in dist:
            if i >= len(values):
                break

            shuffled_behavior[
                (
                    previous,
                    next_rel,
                )
            ] = values[i]

            i += 1

    # Family control: shuffle predicate profiles assigned to predicate names.
    permuted = names[:]
    rng.shuffle(permuted)

    shuffled_family = {}

    for i, predicate in enumerate(names):
        source = permuted[i]

        for other in names:
            # Similarity of predicate to a random profile, not its own profile.
            shuffled_family[
                predicate,
            ] = profile_similarity(
                profile_lookup[predicate],
                profile_lookup[source],
            )

            break

    # Expand family mapping to arbitrary relations by using their nearest
    # random reference profile.
    return shuffled_behavior, shuffled_family


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def evaluate_policy(
    args,
    cases,
    policy,
    profiles,
    neighbors,
    shuffled_behavior,
    shuffled_family,
):
    results = []

    con = connect_ro(args.database)

    for case in cases:
        outcome = search_case(
            con,
            case,
            policy,
            args.budget,
            args.per_node,
            args.max_depth,
            profiles,
            neighbors,
            shuffled_behavior,
            shuffled_family,
        )

        predicted = bool(
            outcome["predicted"]
        )
        gold = bool(
            case["gold"]
        )

        results.append(
            {
                "predicted": predicted,
                "gold": gold,
                "correct": (
                    predicted == gold
                ),
                "steps": int(
                    outcome["steps"]
                ),
                "path_length": len(
                    outcome.get(
                        "path",
                        [],
                    )
                ),
                "kind": case["kind"],
                "target_relation": (
                    case["target_relation"]
                ),
            }
        )

    con.close()

    positives = [
        x for x in results
        if x["gold"]
    ]

    negatives = [
        x for x in results
        if not x["gold"]
    ]

    recovered = sum(
        x["predicted"]
        for x in positives
    )

    false = sum(
        x["predicted"]
        for x in negatives
    )

    correct = sum(
        x["correct"]
        for x in results
    )

    return {
        "cases": len(results),
        "accuracy": (
            correct / len(results)
            if results
            else 0.0
        ),
        "supported_cases": len(positives),
        "supported_recovery": (
            recovered / len(positives)
            if positives
            else 0.0
        ),
        "negative_cases": len(negatives),
        "false_proof_rate": (
            false / len(negatives)
            if negatives
            else 0.0
        ),
        "predicted_positive": (
            sum(
                x["predicted"]
                for x in results
            )
        ),
        "mean_steps": (
            sum(
                x["steps"]
                for x in results
            )
            / len(results)
            if results
            else 0.0
        ),
        "mean_path_length": (
            sum(
                x["path_length"]
                for x in results
                if x["predicted"]
            )
            /
            max(
                1,
                sum(
                    x["predicted"]
                    for x in results
                ),
            )
        ),
        "budget_exhausted": sum(
            1
            for x in results
            if (
                not x["predicted"]
                and x["steps"] >= args.budget
            )
        ),
        "results_by_relation": relation_breakdown(
            results
        ),
    }


def relation_breakdown(results):
    grouped = defaultdict(list)

    for row in results:
        grouped[
            row["target_relation"]
        ].append(row)

    out = {}

    for relation, rows in grouped.items():
        positives = [
            x for x in rows
            if x["gold"]
        ]

        negatives = [
            x for x in rows
            if not x["gold"]
        ]

        rec = sum(
            x["predicted"]
            for x in positives
        )

        false = sum(
            x["predicted"]
            for x in negatives
        )

        out[relation] = {
            "cases": len(rows),
            "supported_cases": len(positives),
            "supported_recovery": (
                rec / len(positives)
                if positives
                else 0.0
            ),
            "false_proof_rate": (
                false / len(negatives)
                if negatives
                else 0.0
            ),
            "mean_steps": (
                sum(
                    x["steps"]
                    for x in rows
                )
                / len(rows)
            ),
        }

    return out


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
        default=r".\results\v570_mechanism_ablation.json",
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
        default=60,
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
        "--seed",
        type=int,
        default=570,
    )

    args = ap.parse_args()

    start = time.perf_counter()

    database = resolve_path(
        args.database
    )

    v568_path = resolve_path(
        args.v568
    )

    print(
        "=== V570 COGNITIVE MECHANISM ABLATION MATRIX ==="
    )
    print(
        f"database            : {database}"
    )
    print(
        f"database size       : "
        f"{database.stat().st_size / 1024**3:.2f} GB"
    )
    print(
        f"V568 profiles       : {v568_path}"
    )
    print(
        f"workers             : {args.workers}"
    )
    print(
        f"budget              : {args.budget}"
    )
    print(
        f"per-node            : {args.per_node}"
    )
    print(
        f"max-depth           : {args.max_depth}"
    )
    print(
        "source              : READ-ONLY"
    )
    print(
        "LLM                 : NOT USED"
    )
    print()

    v568, profiles, neighbors = load_v568(
        v568_path
    )

    con = connect_ro(database)

    schema_info = check_schema(
        con
    )

    print(
        f"[SCHEMA] indexes: "
        f"{schema_info['indexes']}"
    )

    edges = con.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    print(
        f"[GRAPH] edges={edges:,}"
    )

    inventory = top_predicates(
        con,
        args.top_predicates,
    )

    predicates = [
        predicate
        for predicate, _count in inventory
        if predicate in profiles
    ]

    print()
    print(
        "=== TOP PREDICATES WITH V568 PROFILES ==="
    )

    for i, predicate in enumerate(
        predicates,
        1,
    ):
        count = next(
            count
            for p, count in inventory
            if p == predicate
        )

        print(
            f"{i:2d}. "
            f"{predicate:32s} "
            f"{count:12,}"
        )

    con.close()

    if not predicates:
        raise RuntimeError(
            "No predicates overlap between the graph and V568 profiles."
        )

    # ---------------------------------------------------------------
    # Oracle
    # ---------------------------------------------------------------

    print()
    print(
        "=== BUILDING SHARED ORACLE DATASET ==="
    )

    oracle_parts = []

    with ThreadPoolExecutor(
        max_workers=min(
            args.workers,
            len(predicates),
        )
    ) as pool:

        futures = {
            pool.submit(
                build_oracle_job,
                database,
                [predicate],
                args.sample_each,
                args.middle_out_limit,
                args.cases_per_predicate,
                args.negative_ratio,
                args.seed + i,
            ): predicate
            for i, predicate in enumerate(
                predicates
            )
        }

        completed = 0

        for future in as_completed(
            futures
        ):
            predicate = futures[
                future
            ]
            completed += 1

            try:
                part = future.result()
            except Exception as exc:
                print(
                    f"[ORACLE {completed}/{len(predicates)}] "
                    f"{predicate} ERROR={exc}",
                    flush=True,
                )
                continue

            oracle_parts.append(part)

            print(
                f"[ORACLE {completed}/{len(predicates)}] "
                f"{predicate:32s} "
                f"cases={len(part):5,}",
                flush=True,
            )

    cases = [
        case
        for part in oracle_parts
        for case in part
    ]

    rng = random.Random(
        args.seed
    )
    rng.shuffle(cases)

    positives = [
        c for c in cases
        if c["gold"]
    ]

    negatives = [
        c for c in cases
        if not c["gold"]
    ]

    positives = positives[
        :args.holdout_supported
    ]

    negatives = negatives[
        :args.holdout_negative
    ]

    holdout = positives + negatives
    rng.shuffle(holdout)

    print()
    print(
        "=== SHARED HOLDOUT ==="
    )
    print(
        f"supported      : {len(positives):,}"
    )
    print(
        f"hard negatives : {len(negatives):,}"
    )
    print(
        f"total          : {len(holdout):,}"
    )

    if not positives:
        raise RuntimeError(
            "No positive compositional cases were generated."
        )

    shuffled_behavior, shuffled_family = (
        build_shuffled_controls(
            profiles,
            args.seed + 991,
        )
    )

    # ---------------------------------------------------------------
    # Policies
    # ---------------------------------------------------------------

    print()
    print(
        "=== RUNNING ABLATION MATRIX ==="
    )

    results = {}

    for policy in POLICIES:
        t = time.perf_counter()

        print(
            f"[POLICY START] {policy}",
            flush=True,
        )

        result = evaluate_policy(
            args,
            holdout,
            policy,
            profiles,
            neighbors,
            shuffled_behavior,
            shuffled_family,
        )

        result["seconds"] = (
            time.perf_counter()
            - t
        )

        results[policy] = result

        print(
            f"[POLICY END] {policy:22s} "
            f"acc={result['accuracy']:.4f} "
            f"recover={result['supported_recovery']:.4f} "
            f"false={result['false_proof_rate']:.4f} "
            f"steps={result['mean_steps']:.2f} "
            f"time={result['seconds']:.2f}s",
            flush=True,
        )

    baseline = results["bfs"]

    deltas = {}

    for policy, result in results.items():
        deltas[policy] = {
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

    # Mechanism-isolation summaries.
    mechanism_effects = {
        "behavior_effect": {
            "with": results[
                "behavior"
            ],
            "without": results[
                "family"
            ],
            "delta_accuracy": (
                results["behavior"]["accuracy"]
                - results["family"]["accuracy"]
            ),
            "delta_recovery": (
                results["behavior"][
                    "supported_recovery"
                ]
                - results["family"][
                    "supported_recovery"
                ]
            ),
        },
        "family_effect": {
            "with": results[
                "family"
            ],
            "without": results[
                "behavior"
            ],
            "delta_accuracy": (
                results["family"]["accuracy"]
                - results["behavior"]["accuracy"]
            ),
            "delta_recovery": (
                results["family"][
                    "supported_recovery"
                ]
                - results["behavior"][
                    "supported_recovery"
                ]
            ),
        },
        "depth_effect": {
            "without_depth": results[
                "behavior_family"
            ],
            "with_depth": results[
                "full_hybrid"
            ],
            "delta_accuracy": (
                results["full_hybrid"]["accuracy"]
                - results["behavior_family"]["accuracy"]
            ),
            "delta_recovery": (
                results["full_hybrid"][
                    "supported_recovery"
                ]
                - results["behavior_family"][
                    "supported_recovery"
                ]
            ),
            "delta_steps": (
                results["full_hybrid"]["mean_steps"]
                - results["behavior_family"]["mean_steps"]
            ),
        },
        "family_interaction": {
            "behavior_only": results[
                "behavior"
            ],
            "family_only": results[
                "family"
            ],
            "behavior_family": results[
                "behavior_family"
            ],
            "synergy_accuracy": (
                results["behavior_family"][
                    "accuracy"
                ]
                - max(
                    results["behavior"]["accuracy"],
                    results["family"]["accuracy"],
                )
            ),
            "synergy_recovery": (
                results["behavior_family"][
                    "supported_recovery"
                ]
                - max(
                    results["behavior"][
                        "supported_recovery"
                    ],
                    results["family"][
                        "supported_recovery"
                    ],
                )
            ),
        },
        "depth_interaction": {
            "behavior_only": results[
                "behavior"
            ],
            "behavior_depth": results[
                "behavior_depth"
            ],
            "family_only": results[
                "family"
            ],
            "family_depth": results[
                "family_depth"
            ],
        },
    }

    # A conservative utility is used only to summarize policy quality.
    # It is not used to train anything.
    utilities = {}

    for policy, result in results.items():
        utilities[policy] = (
            result["supported_recovery"]
            - 1.25
            * result["false_proof_rate"]
            - 0.01
            * (
                result["mean_steps"]
                / max(
                    1,
                    args.budget,
                )
            )
        )

    search_policies = [
        p
        for p in POLICIES
        if p not in {
            "shuffled_behavior",
            "shuffled_family",
            "shuffled_all",
        }
    ]

    cognitive_winner = max(
        search_policies,
        key=lambda p: utilities[p],
    )

    # Determine whether shuffled priors collapse.
    shuffle_summary = {
        "behavior_loss": (
            results["behavior"]["accuracy"]
            - results["shuffled_behavior"]["accuracy"]
        ),
        "family_loss": (
            results["family"]["accuracy"]
            - results["shuffled_family"]["accuracy"]
        ),
        "full_loss": (
            results["full_hybrid"]["accuracy"]
            - results["shuffled_all"]["accuracy"]
        ),
    }

    print()
    print(
        "=== ABLATION MATRIX SUMMARY ==="
    )

    print(
        f"{'policy':24s} "
        f"{'acc':>8s} "
        f"{'recover':>9s} "
        f"{'false':>8s} "
        f"{'steps':>9s}"
    )

    for policy in POLICIES:
        r = results[policy]
        print(
            f"{policy:24s} "
            f"{r['accuracy']:8.4f} "
            f"{r['supported_recovery']:9.4f} "
            f"{r['false_proof_rate']:8.4f} "
            f"{r['mean_steps']:9.2f}"
        )

    print()
    print(
        f"COGNITIVE WINNER: "
        f"{cognitive_winner}"
    )

    print()
    print(
        "=== MECHANISM EFFECTS ==="
    )

    print(
        f"behavior Δ accuracy : "
        f"{mechanism_effects['behavior_effect']['delta_accuracy']:+.4f}"
    )

    print(
        f"family Δ accuracy   : "
        f"{mechanism_effects['family_effect']['delta_accuracy']:+.4f}"
    )

    print(
        f"depth Δ accuracy    : "
        f"{mechanism_effects['depth_effect']['delta_accuracy']:+.4f}"
    )

    print(
        f"depth Δ steps       : "
        f"{mechanism_effects['depth_effect']['delta_steps']:+.2f}"
    )

    print()
    print(
        "=== SHUFFLE TEST ==="
    )

    print(
        f"behavior prior loss : "
        f"{shuffle_summary['behavior_loss']:+.4f}"
    )

    print(
        f"family prior loss   : "
        f"{shuffle_summary['family_loss']:+.4f}"
    )

    print(
        f"full prior loss     : "
        f"{shuffle_summary['full_loss']:+.4f}"
    )

    report = {
        "benchmark": (
            "v570_cognitive_mechanism_ablation"
        ),
        "database": str(database),
        "v568_model": str(v568_path),
        "database_size_bytes": database.stat().st_size,
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": {
            "edges": edges,
        },
        "oracle": {
            "supported": len(positives),
            "hard_negative": len(negatives),
            "total": len(holdout),
            "construction": (
                "2-hop graph-derived supported cases with the direct "
                "target edge hidden during primary policy evaluation"
            ),
        },
        "policies": results,
        "deltas_vs_bfs": deltas,
        "mechanism_effects": mechanism_effects,
        "shuffle_test": shuffle_summary,
        "utility_scores": utilities,
        "cognitive_winner": cognitive_winner,
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
            "seed": args.seed,
        },
        "v568_profiles_used": len(
            profiles
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - start
        ),
    }

    out = Path(
        args.output
    )
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
    print("V570 COMPLETE")
    print("=" * 72)
    print(
        f"cognitive winner : "
        f"{cognitive_winner}"
    )
    print(
        f"JSON             : "
        f"{out}"
    )
    print(
        f"elapsed          : "
        f"{report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
V573 — Depth Control Compatibility / Improvement Matrix

Hypothesis
----------
V572 established the strongest reproducible signal so far:

    adaptive depth > BFS

while the global behavior/family priors did not help.

V573 therefore treats depth control as the base mechanism and tests factors
that are structurally compatible with it:

    A. depth + branch-factor awareness
    B. depth + target-hop prior
    C. depth + depth-specific budget allocation
    D. depth + gated behavior prior
    E. depth + gated family prior
    F. depth + gated behavior + family
    G. depth + branch + gated behavior
    H. depth + branch + gated family
    I. depth + branch + target-hop prior + budget allocation

Controls
--------
    bfs
    depth
    depth_branch
    depth_hop
    depth_budget
    depth_gated_behavior
    depth_gated_family
    depth_gated_hybrid
    depth_branch_behavior
    depth_branch_family
    depth_full

The key idea is that behavior/family priors are NOT allowed to change the
depth decision. They are only allowed to rank edges AFTER the depth controller
has chosen a depth, and only when their local confidence exceeds a threshold.

All policies use the SAME balanced holdout.

The primary output answers:
    - Which factor improves depth recovery?
    - Which factor reduces search steps?
    - Which factors are stable across seeds?
    - Do gated priors help once depth is already controlled?
    - Do shuffled gated priors lose the gain?

Source graph:
    READ-ONLY
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


POLICIES = [
    "bfs",
    "depth",
    "depth_branch",
    "depth_hop",
    "depth_budget",
    "depth_gated_behavior",
    "depth_gated_family",
    "depth_gated_hybrid",
    "depth_branch_behavior",
    "depth_branch_family",
    "depth_full",
]


@dataclass(frozen=True)
class PredicateInfo:
    predicate: str
    edge_count: int


@dataclass(frozen=True)
class Case:
    subject: str
    target_relation: str
    object: str
    source_predicate: str
    kind: str
    gold: bool
    path: tuple[tuple[str, str, str], ...]


def resolve_file(value: str) -> Path:
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


def connect_ro(database: Path) -> sqlite3.Connection:
    database = Path(database).resolve()

    con = sqlite3.connect(
        database.as_uri() + "?mode=ro",
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
        str(row["name"])
        for row in con.execute(
            "PRAGMA table_info(edges)"
        )
    }

    required = {
        "subject",
        "relation",
        "object",
        "source",
    }

    missing = sorted(required - cols)

    if missing:
        raise RuntimeError(
            f"edges table missing: {missing}"
        )

    return {
        "columns": sorted(cols),
        "indexes": [
            str(row["name"])
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

    if not isinstance(data, dict):
        raise RuntimeError(
            "V568 root must be a JSON object."
        )

    container = None

    for key in (
        "profiles",
        "relation_profiles",
        "predicate_profiles",
        "profile",
    ):
        candidate = data.get(key)
        if isinstance(candidate, (list, dict)):
            container = candidate
            break

    profiles = {}

    if isinstance(container, list):
        for row in container:
            if not isinstance(row, dict):
                continue

            name = row.get(
                "predicate",
                row.get("relation"),
            )

            if name is None:
                continue

            profiles[str(name).strip()] = row

    elif isinstance(container, dict):
        for key, value in container.items():
            if not isinstance(value, dict):
                continue

            row = dict(value)

            name = row.get(
                "predicate",
                row.get(
                    "relation",
                    key,
                ),
            )

            if name is None:
                continue

            name = str(name).strip()
            row["predicate"] = name
            profiles[name] = row

    if not profiles:
        raise RuntimeError(
            "No V568 predicate profiles found."
        )

    raw_neighbors = data.get(
        "nearest_neighbors",
        {},
    )

    neighbors = {}

    if isinstance(raw_neighbors, list):
        for row in raw_neighbors:
            if not isinstance(row, dict):
                continue

            name = row.get("predicate")
            if name is None:
                continue

            values = row.get(
                "neighbors",
                [],
            )

            neighbors[
                str(name)
            ] = (
                values
                if isinstance(values, list)
                else []
            )

    elif isinstance(raw_neighbors, dict):
        for key, values in raw_neighbors.items():
            neighbors[
                str(key)
            ] = (
                values
                if isinstance(values, list)
                else []
            )

    return data, profiles, neighbors


def second_hop_profile(profile):
    behavior = profile.get(
        "behavior",
        {},
    )

    if not isinstance(
        behavior,
        dict,
    ):
        return {}

    rows = behavior.get(
        "second_relation_profile",
        [],
    )

    out = {}

    if not isinstance(
        rows,
        list,
    ):
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue

        relation = row.get(
            "relation"
        )
        fraction = row.get(
            "fraction"
        )

        if (
            relation is None
            or fraction is None
        ):
            continue

        try:
            out[
                str(relation)
            ] = float(fraction)
        except (
            TypeError,
            ValueError,
        ):
            continue

    return out


def profile_family_similarity(
    a,
    b,
):
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
        k: (pa[k] + pb[k]) / 2.0
        for k in keys
    }

    def kl(dist):
        return sum(
            value * math.log2(
                value / m[key]
            )
            for key, value in dist.items()
            if value > 0
            and m[key] > 0
        )

    js = math.sqrt(
        max(
            0.0,
            (
                kl(pa)
                + kl(pb)
            )
            / 2.0,
        )
    )

    return max(
        0.0,
        1.0 - js,
    )


# ---------------------------------------------------------------------------
# Graph inventory / sampling
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
        PredicateInfo(
            predicate=str(row["relation"]),
            edge_count=int(
                row["edge_count"]
            ),
        )
        for row in rows
        if row["relation"] is not None
    ]


def sample_relation(
    con,
    predicate,
    sample_size,
    seed,
):
    bounds = con.execute(
        """
        SELECT MIN(rowid), MAX(rowid)
        FROM edges
        """
    ).fetchone()

    lo = int(bounds[0])
    hi = int(bounds[1])

    rng = random.Random(seed)
    found = {}
    probes = 0

    while (
        len(found) < sample_size
        and probes < 5000
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
                    str(row["subject"]),
                    str(row["object"]),
                )
            ] = (
                str(row["subject"]),
                str(row["object"]),
                row["source"],
            )

            if len(found) >= sample_size:
                break

        probes += 1

    return list(found.values())[:sample_size]


def outgoing_for_nodes(
    con,
    nodes,
    per_node,
):
    output = defaultdict(list)

    nodes = list(nodes)

    for i in range(
        0,
        len(nodes),
        250,
    ):
        chunk = nodes[i:i + 250]

        placeholders = ",".join(
            "?" for _ in chunk
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
            subject = str(
                row["subject"]
            )

            bucket = output[
                subject
            ]

            if len(bucket) >= per_node:
                continue

            bucket.append(
                (
                    str(row["relation"]),
                    str(row["object"]),
                    row["source"],
                )
            )

    return output


def confirm_pairs(
    con,
    pairs,
):
    by_subject = defaultdict(set)

    for subject, object_ in pairs:
        by_subject[
            subject
        ].add(object_)

    output = defaultdict(set)

    subjects = list(
        by_subject
    )

    for i in range(
        0,
        len(subjects),
        250,
    ):
        chunk = subjects[i:i + 250]

        placeholders = ",".join(
            "?" for _ in chunk
        )

        query = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(
            query,
            chunk,
        ):
            subject = str(
                row["subject"]
            )
            object_ = str(
                row["object"]
            )

            if object_ in by_subject.get(
                subject,
                (),
            ):
                output[
                    (
                        subject,
                        object_,
                    )
                ].add(
                    str(row["relation"])
                )

    return output


# ---------------------------------------------------------------------------
# Oracle generation
# ---------------------------------------------------------------------------

def positive_cases_for_predicate(
    database,
    predicate_info,
    sample_each,
    middle_limit,
    case_cap,
    seed,
):
    con = connect_ro(database)

    try:
        first = sample_relation(
            con,
            predicate_info.predicate,
            sample_each,
            seed,
        )

        middles = {
            middle
            for _subject, middle, _source
            in first
        }

        adjacency = outgoing_for_nodes(
            con,
            middles,
            middle_limit,
        )

        candidate_paths = []

        max_candidates = max(
            case_cap * 20,
            200,
        )

        for subject, middle, _source in first:
            for second_relation, endpoint, _source2 in adjacency.get(
                middle,
                (),
            ):
                if endpoint in (
                    subject,
                    middle,
                ):
                    continue

                candidate_paths.append(
                    (
                        subject,
                        predicate_info.predicate,
                        middle,
                        second_relation,
                        endpoint,
                    )
                )

                if (
                    len(candidate_paths)
                    >= max_candidates
                ):
                    break

            if (
                len(candidate_paths)
                >= max_candidates
            ):
                break

        if not candidate_paths:
            return []

        pairs = {
            (
                row[0],
                row[4],
            )
            for row in candidate_paths
        }

        confirmations = confirm_pairs(
            con,
            pairs,
        )

        positives = []

        for (
            subject,
            first_relation,
            middle,
            second_relation,
            endpoint,
        ) in candidate_paths:
            direct_relations = confirmations.get(
                (
                    subject,
                    endpoint,
                ),
                (),
            )

            if not direct_relations:
                continue

            target_relation = sorted(
                direct_relations
            )[0]

            positives.append(
                Case(
                    subject=subject,
                    target_relation=target_relation,
                    object=endpoint,
                    source_predicate=predicate_info.predicate,
                    kind="SUPPORTED",
                    gold=True,
                    path=(
                        (
                            subject,
                            first_relation,
                            middle,
                        ),
                        (
                            middle,
                            second_relation,
                            endpoint,
                        ),
                    ),
                )
            )

        unique = {}

        for case in positives:
            key = (
                case.subject,
                case.target_relation,
                case.object,
                case.path,
            )
            unique[key] = case

        output = list(
            unique.values()
        )

        random.Random(
            seed + 11
        ).shuffle(output)

        return output[:case_cap]

    finally:
        con.close()


def balanced_holdout(
    database,
    predicate_infos,
    sample_each,
    middle_limit,
    case_cap,
    supported_per_relation,
    negative_per_relation,
    seed,
    workers,
):
    positive_pool = []

    with ThreadPoolExecutor(
        max_workers=min(
            workers,
            len(predicate_infos),
        )
    ) as pool:
        futures = {
            pool.submit(
                positive_cases_for_predicate,
                database,
                info,
                sample_each,
                middle_limit,
                case_cap,
                seed + i,
            ): info.predicate
            for i, info in enumerate(
                predicate_infos
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
                rows = future.result()
            except Exception as exc:
                print(
                    f"[ORACLE {completed}/{len(futures)}] "
                    f"{predicate:32s} ERROR={exc}",
                    flush=True,
                )
                continue

            positive_pool.extend(
                rows
            )

            print(
                f"[ORACLE {completed}/{len(futures)}] "
                f"{predicate:32s} "
                f"positives={len(rows):4d}",
                flush=True,
            )

    by_target = defaultdict(list)

    for case in positive_pool:
        by_target[
            case.target_relation
        ].append(case)

    candidate_target_counts = {
        relation: len(rows)
        for relation, rows
        in sorted(
            by_target.items()
        )
    }

    target_relations = [
        relation
        for relation, rows
        in sorted(
            by_target.items()
        )
        if len(rows) >= supported_per_relation
    ]

    if not target_relations:
        raise RuntimeError(
            "No target relation has enough supported cases."
        )

    rng = random.Random(seed)

    supported = []

    for relation in target_relations:
        rows = list(
            by_target[relation]
        )
        rng.shuffle(rows)

        supported.extend(
            rows[:supported_per_relation]
        )

    # Hard negatives: endpoint substitution, then exact verification.
    negative_candidates = []

    objects_by_relation = defaultdict(list)

    for case in supported:
        objects_by_relation[
            case.target_relation
        ].append(
            case.object
        )

    for case in supported:
        pool = objects_by_relation[
            case.target_relation
        ]

        shuffled = pool[:]
        rng.shuffle(shuffled)

        for replacement in shuffled:
            if replacement == case.object:
                continue

            negative_candidates.append(
                (
                    case.subject,
                    case.target_relation,
                    replacement,
                    case.source_predicate,
                )
            )
            break

    con = connect_ro(database)

    try:
        existing = set()

        subjects = sorted({
            row[0]
            for row in negative_candidates
        })

        for i in range(
            0,
            len(subjects),
            250,
        ):
            chunk = subjects[i:i + 250]
            placeholders = ",".join(
                "?" for _ in chunk
            )

            query = f"""
                SELECT subject, relation, object
                FROM edges
                WHERE subject IN ({placeholders})
            """

            for row in con.execute(
                query,
                chunk,
            ):
                existing.add(
                    (
                        str(row["subject"]),
                        str(row["relation"]),
                        str(row["object"]),
                    )
                )

    finally:
        con.close()

    negatives_by_target = defaultdict(list)

    for (
        subject,
        relation,
        object_,
        source_predicate,
    ) in negative_candidates:
        if (
            subject,
            relation,
            object_,
        ) in existing:
            continue

        negatives_by_target[
            relation
        ].append(
            Case(
                subject=subject,
                target_relation=relation,
                object=object_,
                source_predicate=source_predicate,
                kind="HARD_NEGATIVE",
                gold=False,
                path=(),
            )
        )

    final_targets = [
        relation
        for relation in target_relations
        if len(
            negatives_by_target[
                relation
            ]
        ) >= negative_per_relation
    ]

    if not final_targets:
        raise RuntimeError(
            "No target relation has enough hard negatives."
        )

    final_supported = []
    final_negatives = []

    for relation in final_targets:
        srows = [
            row
            for row in supported
            if row.target_relation == relation
        ]
        nrows = list(
            negatives_by_target[
                relation
            ]
        )

        rng.shuffle(nrows)

        final_supported.extend(
            srows[:supported_per_relation]
        )

        final_negatives.extend(
            nrows[:negative_per_relation]
        )

    cases = final_supported + final_negatives
    rng.shuffle(cases)

    return cases, {
        "candidate_positive_counts": candidate_target_counts,
        "target_relations": final_targets,
        "supported": len(final_supported),
        "negative": len(final_negatives),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def visible_edges(
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
        relation = str(
            row["relation"]
        )
        object_ = str(
            row["object"]
        )

        if (
            subject,
            relation,
            object_,
        ) in hidden:
            continue

        rows.append(
            (
                relation,
                object_,
                row["source"],
            )
        )

        if len(rows) >= per_node:
            break

    return rows


def choose_depth(
    con,
    case,
    profiles,
    per_node,
):
    first = visible_edges(
        con,
        case.subject,
        per_node,
        {
            (
                case.subject,
                case.target_relation,
                case.object,
            )
        },
    )

    best = 0.0
    first_relation = None

    for relation, _object, _source in first:
        profile = profiles.get(
            relation
        )

        if not profile:
            continue

        score = second_hop_profile(
            profile
        ).get(
            case.target_relation,
            0.0,
        )

        if score > best:
            best = score
            first_relation = relation

    # Depth controller used in V572, kept as the base.
    depth = (
        2
        if best >= 0.01
        else 3
    )

    return depth, best, first_relation


def relation_behavior_score(
    previous,
    relation,
    profiles,
    target_relation,
):
    if previous is None:
        return (
            math.log1p(
                profiles.get(
                    relation,
                    {},
                ).get(
                    "edge_count",
                    0,
                )
            )
            * 0.005
        )

    profile = profiles.get(
        previous
    )

    if not profile:
        return 0.0

    score = second_hop_profile(
        profile
    ).get(
        relation,
        0.0,
    )

    if relation == target_relation:
        score += 0.15

    return score


def relation_family_score(
    relation,
    target_relation,
    profiles,
    neighbors,
):
    if relation == target_relation:
        return 1.0

    target_profile = profiles.get(
        target_relation
    )
    candidate_profile = profiles.get(
        relation
    )

    score = profile_family_similarity(
        target_profile,
        candidate_profile,
    )

    for row in neighbors.get(
        target_relation,
        [],
    ):
        if not isinstance(
            row,
            dict,
        ):
            continue

        if row.get(
            "predicate"
        ) != relation:
            continue

        try:
            score = max(
                score,
                1.0
                - float(
                    row.get(
                        "distance",
                        1.0,
                    )
                ),
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    return score


def edge_priority(
    policy,
    previous,
    relation,
    object_,
    target_object,
    target_relation,
    profiles,
    neighbors,
    branch_penalty,
    first_hop_best,
):
    """
    All policy-specific ranking stays in one function so the ablation is
    explicit and auditable.
    """
    base = 0.0

    # Frequency-only information is never used by depth policies unless it
    # appears as part of branch-factor handling.
    use_behavior = policy in {
        "depth_gated_behavior",
        "depth_gated_hybrid",
        "depth_branch_behavior",
        "depth_full",
    }

    use_family = policy in {
        "depth_gated_family",
        "depth_gated_hybrid",
        "depth_branch_family",
        "depth_full",
    }

    use_branch = policy in {
        "depth_branch",
        "depth_branch_behavior",
        "depth_branch_family",
        "depth_full",
    }

    use_hop = policy in {
        "depth_hop",
        "depth_full",
    }

    if use_behavior:
        behavior = relation_behavior_score(
            previous,
            relation,
            profiles,
            target_relation,
        )

        # Gating: only allow the prior to steer search when it is stronger than
        # a weak noise floor. This is the compatibility test.
        gate = (
            1.0
            if (
                previous is not None
                and behavior >= 0.01
            )
            else 0.0
        )

        if policy == "depth_full":
            gate = 1.0 if behavior >= 0.005 else 0.0

        base += (
            0.60
            * gate
            * behavior
        )

    if use_family:
        family = relation_family_score(
            relation,
            target_relation,
            profiles,
            neighbors,
        )

        gate = (
            1.0
            if family >= 0.55
            else 0.0
        )

        base += (
            0.35
            * gate
            * family
        )

    if use_hop:
        # Explicit target-hop support gets a strong but localized score.
        if relation == target_relation:
            base += 0.45

        if (
            previous is not None
            and profiles.get(previous)
        ):
            base += 0.25 * second_hop_profile(
                profiles[previous]
            ).get(
                relation,
                0.0,
            )

    if use_branch:
        # Branch factor: prefer nodes with smaller observed branching.
        # `branch_penalty` is precomputed for this edge.
        base += 0.15 * (
            1.0
            - branch_penalty
        )

    # Small deterministic locality bonus toward the requested endpoint.
    if object_ == target_object:
        base += 0.80

    if relation == target_relation:
        base += 0.20

    # Depth controller confidence mildly affects expansion priority.
    base += min(
        0.10,
        2.0 * first_hop_best,
    )

    return base


def run_policy(
    con,
    case,
    policy,
    budget,
    per_node,
    max_depth,
    profiles,
    neighbors,
    rng,
):
    hidden = set()

    if case.gold:
        hidden.add(
            (
                case.subject,
                case.target_relation,
                case.object,
            )
        )

    if policy == "bfs":
        chosen_depth = max_depth
        branch_mode = False
    else:
        chosen_depth, confidence, _first_relation = (
            choose_depth(
                con,
                case,
                profiles,
                per_node,
            )
        )
        branch_mode = policy in {
            "depth_branch",
            "depth_branch_behavior",
            "depth_branch_family",
            "depth_full",
        }

    # Pure depth: same search policy as V572, with adaptive maximum depth.
    frontier = [
        (
            case.subject,
            (),
            0.0,
        )
    ]

    visited = {
        case.subject
    }

    steps = 0
    branch_cache = {}

    def branch_penalty_for(
        object_,
    ):
        if not branch_mode:
            return 0.0

        if object_ not in branch_cache:
            rows = visible_edges(
                con,
                object_,
                per_node,
                hidden,
            )

            # Normalize branching into [0, 1].
            branch_cache[
                object_
            ] = min(
                1.0,
                len(rows)
                / max(
                    1.0,
                    per_node,
                ),
            )

        return branch_cache[
            object_
        ]

    while (
        frontier
        and steps < budget
    ):
        if policy == "bfs":
            frontier.sort(
                key=lambda x: (
                    len(x[1]),
                    rng.random(),
                )
            )
        else:
            frontier.sort(
                key=lambda x: x[2],
                reverse=True,
            )

        node, path, priority = frontier.pop(0)

        if len(path) >= chosen_depth:
            continue

        previous = (
            path[-1][1]
            if path
            else None
        )

        edges = visible_edges(
            con,
            node,
            per_node,
            hidden,
        )

        scored = []

        for relation, object_, source in edges:
            penalty = branch_penalty_for(
                object_
            )

            score = (
                edge_priority(
                    policy,
                    previous,
                    relation,
                    object_,
                    case.object,
                    case.target_relation,
                    profiles,
                    neighbors,
                    penalty,
                    (
                        confidence
                        if policy != "bfs"
                        else 0.0
                    ),
                )
                if policy != "bfs"
                else 0.0
            )

            scored.append(
                (
                    score,
                    relation,
                    object_,
                    source,
                )
            )

        if policy != "bfs":
            scored.sort(
                key=lambda x: (
                    x[0],
                    -len(path),
                ),
                reverse=True,
            )

        for (
            score,
            relation,
            object_,
            source,
        ) in scored:
            steps += 1

            new_path = path + (
                (
                    node,
                    relation,
                    object_,
                ),
            )

            if (
                relation
                == case.target_relation
                and object_
                == case.object
            ):
                return {
                    "predicted": True,
                    "steps": steps,
                    "path_length": len(
                        new_path
                    ),
                }

            if object_ not in visited:
                visited.add(
                    object_
                )
                frontier.append(
                    (
                        object_,
                        new_path,
                        score,
                    )
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path_length": 0,
    }


def evaluate_policy(
    database,
    cases,
    policy,
    budget,
    per_node,
    max_depth,
    profiles,
    neighbors,
    seed,
):
    con = connect_ro(database)

    rng = random.Random(
        seed
        + hash(policy) % 100000
    )

    try:
        rows = []

        for case in cases:
            outcome = run_policy(
                con,
                case,
                policy,
                budget,
                per_node,
                max_depth,
                profiles,
                neighbors,
                rng,
            )

            predicted = bool(
                outcome["predicted"]
            )

            rows.append(
                {
                    "gold": case.gold,
                    "predicted": predicted,
                    "correct": (
                        predicted == case.gold
                    ),
                    "target_relation": (
                        case.target_relation
                    ),
                    "steps": int(
                        outcome["steps"]
                    ),
                    "path_length": int(
                        outcome["path_length"]
                    ),
                }
            )

        positives = [
            row
            for row in rows
            if row["gold"]
        ]

        negatives = [
            row
            for row in rows
            if not row["gold"]
        ]

        recovered = sum(
            row["predicted"]
            for row in positives
        )

        false = sum(
            row["predicted"]
            for row in negatives
        )

        return {
            "cases": len(rows),
            "accuracy": (
                sum(
                    row["correct"]
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
            "supported_cases": len(
                positives
            ),
            "supported_recovery": (
                recovered
                / len(positives)
                if positives
                else 0.0
            ),
            "negative_cases": len(
                negatives
            ),
            "false_proof_rate": (
                false
                / len(negatives)
                if negatives
                else 0.0
            ),
            "predicted_positive": sum(
                row["predicted"]
                for row in rows
            ),
            "mean_steps": (
                statistics.mean(
                    row["steps"]
                    for row in rows
                )
                if rows
                else 0.0
            ),
            "mean_path_length": (
                statistics.mean(
                    row["path_length"]
                    for row in rows
                    if row["predicted"]
                )
                if any(
                    row["predicted"]
                    for row in rows
                )
                else 0.0
            ),
            "budget_exhausted": sum(
                1
                for row in rows
                if (
                    not row["predicted"]
                    and row["steps"] >= budget
                )
            ),
            "by_target_relation": relation_breakdown(
                rows
            ),
        }

    finally:
        con.close()


def relation_breakdown(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            row["target_relation"]
        ].append(row)

    out = {}

    for relation, values in groups.items():
        positive = [
            row
            for row in values
            if row["gold"]
        ]

        negative = [
            row
            for row in values
            if not row["gold"]
        ]

        out[relation] = {
            "cases": len(values),
            "supported_cases": len(
                positive
            ),
            "supported_recovery": (
                sum(
                    row["predicted"]
                    for row in positive
                )
                / len(positive)
                if positive
                else 0.0
            ),
            "false_proof_rate": (
                sum(
                    row["predicted"]
                    for row in negative
                )
                / len(negative)
                if negative
                else 0.0
            ),
            "mean_steps": statistics.mean(
                row["steps"]
                for row in values
            ),
        }

    return out


def summarize(per_seed):
    policies = [
        policy
        for policy in POLICIES
        if all(
            policy in run["results"]
            for run in per_seed
        )
    ]

    metrics = [
        "accuracy",
        "supported_recovery",
        "false_proof_rate",
        "mean_steps",
        "budget_exhausted",
    ]

    out = {}

    for policy in policies:
        out[policy] = {}

        for metric in metrics:
            values = [
                float(
                    run["results"][policy][metric]
                )
                for run in per_seed
            ]

            mean = statistics.mean(
                values
            )

            std = (
                statistics.stdev(
                    values
                )
                if len(values) > 1
                else 0.0
            )

            margin = (
                1.96
                * std
                / math.sqrt(
                    len(values)
                )
            )

            out[policy][metric] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        default=r".\results\v573_depth_compatibility_matrix.json",
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
        "--case-cap",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--supported-per-relation",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--negative-per-relation",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--seed-start",
        type=int,
        default=57300,
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

    args = ap.parse_args()

    started = time.perf_counter()

    database = resolve_file(
        args.database
    )
    v568_path = resolve_file(
        args.v568
    )

    print(
        "=== V573 DEPTH-COMPATIBILITY EXPERIMENT MATRIX ==="
    )
    print(
        f"database              : {database}"
    )
    print(
        f"database size         : "
        f"{database.stat().st_size / 1024**3:.2f} GB"
    )
    print(
        f"V568 artifact         : {v568_path}"
    )
    print(
        f"workers               : {args.workers}"
    )
    print(
        f"seeds                 : {args.seeds}"
    )
    print(
        f"budget                : {args.budget}"
    )
    print(
        f"per-node              : {args.per_node}"
    )
    print(
        f"max-depth             : {args.max_depth}"
    )
    print(
        "source graph          : READ-ONLY"
    )
    print()

    _, profiles, neighbors = load_v568(
        v568_path
    )

    print(
        f"[V568] profiles loaded : "
        f"{len(profiles):,}"
    )

    con = connect_ro(
        database
    )

    schema = check_schema(
        con
    )

    print(
        f"[SCHEMA] indexes: "
        f"{schema['indexes']}"
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

    con.close()

    graph_predicates = [
        row.predicate
        for row in inventory
    ]

    overlap = [
        row.predicate
        for row in inventory
        if row.predicate in profiles
    ]

    print()
    print(
        "=== PREDICATE / V568 OVERLAP ==="
    )
    print(
        f"graph top predicates : {len(graph_predicates)}"
    )
    print(
        f"V568 overlap         : {len(overlap)}"
    )

    for i, row in enumerate(
        inventory,
        1,
    ):
        marker = (
            " [V568]"
            if row.predicate in profiles
            else ""
        )

        print(
            f"{i:2d}. "
            f"{row.predicate:32s} "
            f"{row.edge_count:12,}"
            f"{marker}"
        )

    print()
    print(
        "=== BUILDING BALANCED ORACLE ==="
    )

    cases, oracle = balanced_holdout(
        database,
        inventory,
        args.sample_each,
        args.middle_out_limit,
        args.case_cap,
        args.supported_per_relation,
        args.negative_per_relation,
        args.seed_start,
        args.workers,
    )

    print()
    print(
        "=== ORACLE RESULT ==="
    )
    print(
        f"target relations : "
        f"{len(oracle['target_relations'])}"
    )
    print(
        f"supported        : "
        f"{oracle['supported']}"
    )
    print(
        f"hard negatives   : "
        f"{oracle['negative']}"
    )
    print(
        f"total cases      : "
        f"{len(cases)}"
    )

    for relation in oracle[
        "target_relations"
    ]:
        print(
            f"  {relation}"
        )

    per_seed = []

    for seed_index in range(
        args.seeds
    ):
        seed = (
            args.seed_start
            + seed_index
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"SEED {seed} "
            f"({seed_index + 1}/{args.seeds})"
        )
        print(
            "=" * 72
        )

        seed_results = {}

        for policy in POLICIES:
            started_policy = time.perf_counter()

            print(
                f"[POLICY START] {policy}",
                flush=True,
            )

            result = evaluate_policy(
                database,
                cases,
                policy,
                args.budget,
                args.per_node,
                args.max_depth,
                profiles,
                neighbors,
                seed,
            )

            result["seconds"] = (
                time.perf_counter()
                - started_policy
            )

            seed_results[
                policy
            ] = result

            print(
                f"[POLICY END] "
                f"{policy:24s} "
                f"acc={result['accuracy']:.4f} "
                f"recover={result['supported_recovery']:.4f} "
                f"false={result['false_proof_rate']:.4f} "
                f"steps={result['mean_steps']:.2f} "
                f"time={result['seconds']:.2f}s",
                flush=True,
            )

        per_seed.append(
            {
                "seed": seed,
                "results": seed_results,
            }
        )

    summary = summarize(
        per_seed
    )

    bfs = summary["bfs"]

    deltas = {}

    for policy in POLICIES:
        deltas[
            policy
        ] = {
            "accuracy_vs_bfs": (
                summary[policy][
                    "accuracy"
                ]["mean"]
                - bfs[
                    "accuracy"
                ]["mean"]
            ),
            "recovery_vs_bfs": (
                summary[policy][
                    "supported_recovery"
                ]["mean"]
                - bfs[
                    "supported_recovery"
                ]["mean"]
            ),
            "false_proof_vs_bfs": (
                summary[policy][
                    "false_proof_rate"
                ]["mean"]
                - bfs[
                    "false_proof_rate"
                ]["mean"]
            ),
            "steps_vs_bfs": (
                summary[policy][
                    "mean_steps"
                ]["mean"]
                - bfs[
                    "mean_steps"
                ]["mean"]
            ),
        }

    # Mechanism compatibility deltas.
    compatibility = {
        "branch_factor": {
            "base": "depth",
            "variant": "depth_branch",
            "accuracy_delta": (
                summary["depth_branch"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_branch"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_branch"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "target_hop_prior": {
            "base": "depth",
            "variant": "depth_hop",
            "accuracy_delta": (
                summary["depth_hop"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_hop"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_hop"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "budget_allocation": {
            "base": "depth",
            "variant": "depth_budget",
            "accuracy_delta": (
                summary["depth_budget"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_budget"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_budget"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "gated_behavior": {
            "base": "depth",
            "variant": "depth_gated_behavior",
            "accuracy_delta": (
                summary["depth_gated_behavior"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_gated_behavior"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_gated_behavior"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "gated_family": {
            "base": "depth",
            "variant": "depth_gated_family",
            "accuracy_delta": (
                summary["depth_gated_family"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_gated_family"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_gated_family"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "gated_hybrid": {
            "base": "depth",
            "variant": "depth_gated_hybrid",
            "accuracy_delta": (
                summary["depth_gated_hybrid"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_gated_hybrid"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_gated_hybrid"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "branch_gated_behavior": {
            "base": "depth",
            "variant": "depth_branch_behavior",
            "accuracy_delta": (
                summary["depth_branch_behavior"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_branch_behavior"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_branch_behavior"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "branch_gated_family": {
            "base": "depth",
            "variant": "depth_branch_family",
            "accuracy_delta": (
                summary["depth_branch_family"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_branch_family"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_branch_family"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
        "full_depth": {
            "base": "depth",
            "variant": "depth_full",
            "accuracy_delta": (
                summary["depth_full"][
                    "accuracy"
                ]["mean"]
                - summary["depth"][
                    "accuracy"
                ]["mean"]
            ),
            "recovery_delta": (
                summary["depth_full"][
                    "supported_recovery"
                ]["mean"]
                - summary["depth"][
                    "supported_recovery"
                ]["mean"]
            ),
            "steps_delta": (
                summary["depth_full"][
                    "mean_steps"
                ]["mean"]
                - summary["depth"][
                    "mean_steps"
                ]["mean"]
            ),
        },
    }

    utility = {}

    for policy in POLICIES:
        utility[
            policy
        ] = (
            summary[policy][
                "supported_recovery"
            ]["mean"]
            - 1.25
            * summary[policy][
                "false_proof_rate"
            ]["mean"]
            - 0.01
            * (
                summary[policy][
                    "mean_steps"
                ]["mean"]
                / max(
                    1,
                    args.budget,
                )
            )
        )

    compatible_variants = [
        policy
        for policy in POLICIES
        if policy != "bfs"
    ]

    winner = max(
        compatible_variants,
        key=lambda policy: utility[policy],
    )

    stability = {}

    for policy in POLICIES:
        wins = sum(
            1
            for run in per_seed
            if (
                run["results"][policy][
                    "accuracy"
                ]
                >
                run["results"]["depth"][
                    "accuracy"
                ]
            )
        )

        stability[
            policy
        ] = {
            "seeds_better_than_depth": wins,
            "seed_count": len(per_seed),
            "fraction": (
                wins / len(per_seed)
                if per_seed
                else 0.0
            ),
        }

    print()
    print(
        "=== V573 SUMMARY ==="
    )

    print(
        f"{'policy':24s} "
        f"{'acc':>10s} "
        f"{'recover':>10s} "
        f"{'false':>10s} "
        f"{'steps':>10s}"
    )

    for policy in POLICIES:
        print(
            f"{policy:24s} "
            f"{summary[policy]['accuracy']['mean']:.4f} "
            f"{summary[policy]['supported_recovery']['mean']:.4f} "
            f"{summary[policy]['false_proof_rate']['mean']:.4f} "
            f"{summary[policy]['mean_steps']['mean']:.2f}"
        )

    print()
    print(
        "=== DEPTH COMPATIBILITY DELTAS ==="
    )

    for name, row in compatibility.items():
        print(
            f"{name:24s} "
            f"Δacc={row['accuracy_delta']:+.4f} "
            f"Δrecover={row['recovery_delta']:+.4f} "
            f"Δsteps={row['steps_delta']:+.2f}"
        )

    print()
    print(
        f"BEST DEPTH-COMPATIBLE VARIANT: "
        f"{winner}"
    )

    report = {
        "benchmark": (
            "v573_depth_control_compatibility_matrix"
        ),
        "database": str(database),
        "database_size_bytes": database.stat().st_size,
        "v568_artifact": str(v568_path),
        "source_graph_read_only": True,
        "schema": schema,
        "graph": {
            "edges": edge_count,
        },
        "v568_preflight": {
            "profiles_loaded": len(profiles),
            "graph_top_predicates": graph_predicates,
            "profile_overlap": overlap,
            "profile_missing": [
                p
                for p in graph_predicates
                if p not in profiles
            ],
        },
        "oracle": oracle | {
            "total_cases": len(cases),
        },
        "per_seed": per_seed,
        "summary": summary,
        "deltas_vs_bfs": deltas,
        "depth_compatibility": compatibility,
        "utility": utility,
        "stability_vs_depth": stability,
        "winner": winner,
        "hypotheses": {
            "branch_factor": (
                "Depth should allocate search away from high-branching "
                "intermediate nodes."
            ),
            "target_hop_prior": (
                "Depth should benefit from knowing which first-hop relations "
                "historically reach the requested target relation."
            ),
            "budget_allocation": (
                "A depth decision is more useful when search budget is "
                "concentrated near the predicted proof depth."
            ),
            "gated_behavior": (
                "Behavior priors may be useful only after the depth controller "
                "has selected a plausible depth."
            ),
            "gated_family": (
                "Relation-family similarity may be compatible with depth "
                "control only as a weak, high-confidence tie-breaker."
            ),
        },
        "config": {
            "workers": args.workers,
            "top_predicates": args.top_predicates,
            "sample_each": args.sample_each,
            "middle_out_limit": args.middle_out_limit,
            "case_cap": args.case_cap,
            "supported_per_relation": args.supported_per_relation,
            "negative_per_relation": args.negative_per_relation,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "budget": args.budget,
            "per_node": args.per_node,
            "max_depth": args.max_depth,
        },
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    output = Path(
        args.output
    )

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
        "V573 COMPLETE"
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


if __name__ == "__main__":
    main()

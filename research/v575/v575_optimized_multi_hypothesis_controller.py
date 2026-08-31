
#!/usr/bin/env python3
"""
V575.3 — Optimized Multi-Hypothesis Cognitive Attention Controller

Performance rebuild of V575.2 for the ~45 GB SQLite semantic graph.

Key fixes
---------
* Batch topology reads for whole frontiers rather than querying one node at a time.
* Bounded LRU cache for outgoing adjacency with explicit memory cap.
* Branching cost is computed from already-fetched topology.
* Per-case progress, policy throughput, ETA, and probe counts.
* Optional --skip-bfs to avoid re-running a fixed baseline.
* --max-probes-per-case guard to prevent pathological cases from monopolizing
  a policy.
* --max-case-seconds guard to abort a pathological single case.
* Same balanced oracle across all policies and seeds.

The graph remains READ-ONLY.

This is still an experimental controller, not a claim of biological realism.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import time
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path


ALL_POLICIES = [
    "bfs",
    "depth_branch",
    "multi_depth",
    "multi_depth_branch_value",
    "multi_depth_backtrack",
    "multi_depth_frontier_value",
    "multi_depth_counterfactual",
    "multi_depth_learned_state",
    "full_cognitive_controller",
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


@dataclass
class H:
    hid: int
    depth_limit: int
    node: str
    path: tuple[tuple[str, str, str], ...]
    score: float
    expansions: int = 0
    failures: int = 0
    alive: bool = True


@dataclass
class Stats:
    hypotheses_created: int = 0
    hypothesis_promotions: int = 0
    hypothesis_abandons: int = 0
    backtracks: int = 0
    branch_switches: int = 0
    allocation_rounds: int = 0
    counterfactual_switches: int = 0
    information_gain: float = 0.0
    max_live_hypotheses: int = 0
    topology_batch_queries: int = 0
    topology_nodes_fetched: int = 0
    topology_cache_hits: int = 0
    topology_cache_misses: int = 0
    cases_aborted: int = 0


class TopologyCache:
    """
    Bounded LRU cache.

    Important: values are plain Python tuples, not sqlite Row objects.
    That avoids retaining database-layer objects and keeps the cache compact.
    """

    def __init__(
        self,
        con: sqlite3.Connection,
        max_entries: int,
    ):
        self.con = con
        self.max_entries = max(
            1,
            max_entries,
        )
        self.data = OrderedDict()

    def get_many(
        self,
        nodes,
        limit: int,
        hidden: set[tuple[str, str, str]],
        stats: Stats,
    ):
        wanted = list(dict.fromkeys(nodes))
        result = {}

        misses = []

        for node in wanted:
            key = (node, limit)

            if key in self.data:
                value = self.data.pop(key)
                self.data[key] = value

                if hidden:
                    result[node] = [
                        edge
                        for edge in value
                        if (
                            node,
                            edge[0],
                            edge[1],
                        ) not in hidden
                    ]
                else:
                    result[node] = value

                stats.topology_cache_hits += 1
            else:
                misses.append(node)
                stats.topology_cache_misses += 1

        for i in range(
            0,
            len(misses),
            150,
        ):
            chunk = misses[i:i + 150]
            placeholders = ",".join(
                "?" * len(chunk)
            )

            query = f"""
                SELECT subject, relation, object, source
                FROM edges
                WHERE subject IN ({placeholders})
            """

            stats.topology_batch_queries += 1

            grouped = defaultdict(list)

            for row in self.con.execute(
                query,
                chunk,
            ):
                subject = str(row["subject"])

                bucket = grouped[
                    subject
                ]

                if len(bucket) >= limit:
                    continue

                bucket.append(
                    (
                        str(row["relation"]),
                        str(row["object"]),
                        row["source"],
                    )
                )

            for node in chunk:
                value = grouped.get(
                    node,
                    [],
                )

                key = (node, limit)

                self.data[key] = value

                while len(self.data) > self.max_entries:
                    self.data.popitem(
                        last=False
                    )

                if hidden:
                    result[node] = [
                        edge
                        for edge in value
                        if (
                            node,
                            edge[0],
                            edge[1],
                        ) not in hidden
                    ]
                else:
                    result[node] = value

                stats.topology_nodes_fetched += 1

        return result

    def get_one(
        self,
        node,
        limit,
        hidden,
        stats,
    ):
        return self.get_many(
            [node],
            limit,
            hidden,
            stats,
        ).get(
            node,
            [],
        )

    def __len__(self):
        return len(self.data)


def resolve_file(value: str) -> Path:
    p = Path(value).expanduser()

    if p.exists() and p.is_file():
        return p.resolve()

    raise FileNotFoundError(
        f"File not found: {value}\n"
        f"cwd: {Path.cwd()}"
    )


def connect_ro(database: Path):
    con = sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=180,
        check_same_thread=False,
    )

    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-32768")

    return con


def check_schema(con):
    cols = {
        str(r["name"])
        for r in con.execute(
            "PRAGMA table_info(edges)"
        )
    }

    required = {
        "subject",
        "relation",
        "object",
        "source",
    }

    missing = sorted(
        required - cols
    )

    if missing:
        raise RuntimeError(
            f"edges table missing: {missing}"
        )

    return {
        "columns": sorted(cols),
        "indexes": [
            str(r["name"])
            for r in con.execute(
                "PRAGMA index_list(edges)"
            )
        ],
    }


def load_v568(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "V568 root must be an object"
        )

    container = None

    for key in (
        "profiles",
        "relation_profiles",
        "predicate_profiles",
        "profile",
    ):
        if isinstance(
            data.get(key),
            (list, dict),
        ):
            container = data[key]
            break

    profiles = {}

    if isinstance(
        container,
        list,
    ):
        for row in container:
            if not isinstance(
                row,
                dict,
            ):
                continue

            name = row.get(
                "predicate",
                row.get("relation"),
            )

            if name is not None:
                profiles[
                    str(name).strip()
                ] = row

    elif isinstance(
        container,
        dict,
    ):
        for key, value in container.items():
            if not isinstance(
                value,
                dict,
            ):
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

            row["predicate"] = str(
                name
            ).strip()

            profiles[
                row["predicate"]
            ] = row

    if not profiles:
        raise RuntimeError(
            "No recognizable V568 predicate profiles."
        )

    raw_neighbors = data.get(
        "nearest_neighbors",
        {},
    )

    neighbors = {}

    if isinstance(
        raw_neighbors,
        list,
    ):
        for row in raw_neighbors:
            if not isinstance(
                row,
                dict,
            ):
                continue

            name = row.get(
                "predicate"
            )

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
                if isinstance(
                    values,
                    list,
                )
                else []
            )

    elif isinstance(
        raw_neighbors,
        dict,
    ):
        neighbors = {
            str(k): (
                v
                if isinstance(v, list)
                else []
            )
            for k, v in raw_neighbors.items()
        }

    return profiles, neighbors


def second_profile(profile):
    behavior = (
        profile.get("behavior", {})
        if isinstance(profile, dict)
        else {}
    )

    rows = (
        behavior.get(
            "second_relation_profile",
            [],
        )
        if isinstance(behavior, dict)
        else []
    )

    output = {}

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            relation = row.get(
                "relation"
            )

            if relation is None:
                continue

            try:
                output[
                    str(relation)
                ] = float(
                    row.get(
                        "fraction",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    return output


def family_similarity(a, b):
    if not a or not b:
        return 0.0

    pa = second_profile(a)
    pb = second_profile(b)

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

    return max(
        0.0,
        1.0 - math.sqrt(
            max(
                0.0,
                (
                    kl(pa)
                    + kl(pb)
                )
                / 2.0,
            )
        ),
    )


def top_predicates(con, n):
    rows = con.execute(
        """
        SELECT relation, COUNT(*) AS edge_count
        FROM edges
        GROUP BY relation
        ORDER BY edge_count DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()

    return [
        PredicateInfo(
            predicate=str(
                row["relation"]
            ),
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
    lo, hi = con.execute(
        """
        SELECT MIN(rowid), MAX(rowid)
        FROM edges
        """
    ).fetchone()

    lo = int(lo)
    hi = int(hi)

    rng = random.Random(
        seed
    )

    found = {}

    for _ in range(5000):
        if len(found) >= sample_size:
            break

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

    return list(
        found.values()
    )[:sample_size]


def confirm_pairs(
    con,
    pairs,
):
    wanted = defaultdict(set)

    for subject, object_ in pairs:
        wanted[
            subject
        ].add(object_)

    result = defaultdict(set)

    subjects = list(
        wanted
    )

    for i in range(
        0,
        len(subjects),
        250,
    ):
        chunk = subjects[i:i + 250]

        placeholders = ",".join(
            "?" * len(chunk)
        )

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(
            q,
            chunk,
        ):
            s = str(
                row["subject"]
            )
            o = str(
                row["object"]
            )

            if o in wanted.get(
                s,
                (),
            ):
                result[
                    (s, o)
                ].add(
                    str(
                        row["relation"]
                    )
                )

    return result


def build_positive_cases(
    database,
    info,
    sample_each,
    middle_limit,
    cap,
    seed,
):
    con = connect_ro(
        database
    )

    try:
        first = sample_relation(
            con,
            info.predicate,
            sample_each,
            seed,
        )

        mids = {
            middle
            for _s, middle, _src in first
        }

        adjacency = defaultdict(list)

        values = list(mids)

        for i in range(
            0,
            len(values),
            250,
        ):
            chunk = values[
                i:i + 250
            ]

            placeholders = ",".join(
                "?" * len(chunk)
            )

            q = f"""
                SELECT subject, relation, object, source
                FROM edges
                WHERE subject IN ({placeholders})
            """

            for row in con.execute(
                q,
                chunk,
            ):
                s = str(
                    row["subject"]
                )

                if len(
                    adjacency[s]
                ) < middle_limit:
                    adjacency[s].append(
                        (
                            str(
                                row["relation"]
                            ),
                            str(
                                row["object"]
                            ),
                            row["source"],
                        )
                    )

        candidates = []

        for s, middle, _src in first:
            for r2, endpoint, _src2 in adjacency.get(
                middle,
                (),
            ):
                if endpoint in (
                    s,
                    middle,
                ):
                    continue

                candidates.append(
                    (
                        s,
                        info.predicate,
                        middle,
                        r2,
                        endpoint,
                    )
                )

                if len(candidates) >= cap * 20:
                    break

            if len(candidates) >= cap * 20:
                break

        confirmations = confirm_pairs(
            con,
            {
                (
                    row[0],
                    row[4],
                )
                for row in candidates
            },
        )

        unique = {}

        for (
            s,
            r1,
            middle,
            r2,
            endpoint,
        ) in candidates:
            rels = confirmations.get(
                (
                    s,
                    endpoint,
                ),
                (),
            )

            if not rels:
                continue

            target = sorted(
                rels
            )[0]

            case = Case(
                subject=s,
                target_relation=target,
                object=endpoint,
                source_predicate=info.predicate,
                kind="SUPPORTED",
                gold=True,
                path=(
                    (
                        s,
                        r1,
                        middle,
                    ),
                    (
                        middle,
                        r2,
                        endpoint,
                    ),
                ),
            )

            unique[
                (
                    case.subject,
                    case.target_relation,
                    case.object,
                    case.path,
                )
            ] = case

        rows = list(
            unique.values()
        )

        random.Random(
            seed + 71
        ).shuffle(rows)

        return rows[:cap]

    finally:
        con.close()


def build_negatives(
    database,
    positives,
    per_relation,
    seed,
):
    by_relation = defaultdict(list)

    for case in positives:
        by_relation[
            case.target_relation
        ].append(case)

    rng = random.Random(
        seed
    )

    candidates = []

    for relation, rows in by_relation.items():
        objects = list({
            case.object
            for case in rows
        })

        if len(objects) < 2:
            continue

        for case in rows:
            pool = objects[:]
            rng.shuffle(pool)

            for replacement in pool:
                if replacement == case.object:
                    continue

                candidates.append(
                    (
                        case.subject,
                        relation,
                        replacement,
                        case.source_predicate,
                    )
                )
                break

    existing = set()

    subjects = sorted({
        row[0]
        for row in candidates
    })

    con = connect_ro(
        database
    )

    try:
        for i in range(
            0,
            len(subjects),
            250,
        ):
            chunk = subjects[
                i:i + 250
            ]

            placeholders = ",".join(
                "?" * len(chunk)
            )

            q = f"""
                SELECT subject, relation, object
                FROM edges
                WHERE subject IN ({placeholders})
            """

            for row in con.execute(
                q,
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

    grouped = defaultdict(list)

    for (
        subject,
        relation,
        object_,
        source_predicate,
    ) in candidates:
        if (
            subject,
            relation,
            object_,
        ) in existing:
            continue

        grouped[
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

    result = []

    for relation, rows in grouped.items():
        rng.shuffle(rows)
        result.extend(
            rows[
                :per_relation
            ]
        )

    return result


def build_balanced_oracle(
    database,
    inventory,
    sample_each,
    middle_limit,
    cap,
    supported_per_relation,
    negative_per_relation,
    seed,
    workers,
):
    positives = []

    with ThreadPoolExecutor(
        max_workers=min(
            workers,
            len(inventory),
        )
    ) as pool:
        futures = {
            pool.submit(
                build_positive_cases,
                database,
                info,
                sample_each,
                middle_limit,
                cap,
                seed + i,
            ): info.predicate
            for i, info in enumerate(
                inventory
            )
        }

        done = 0

        for future in as_completed(
            futures
        ):
            done += 1
            predicate = futures[
                future
            ]

            try:
                rows = future.result()
            except Exception as exc:
                print(
                    f"[ORACLE {done}/{len(futures)}] "
                    f"{predicate:30s} ERROR={exc}",
                    flush=True,
                )
                continue

            positives.extend(rows)

            print(
                f"[ORACLE {done}/{len(futures)}] "
                f"{predicate:30s} positives={len(rows):4d}",
                flush=True,
            )

    by_relation = defaultdict(list)

    for case in positives:
        by_relation[
            case.target_relation
        ].append(case)

    available = {
        relation: len(rows)
        for relation, rows
        in sorted(
            by_relation.items()
        )
    }

    usable = [
        relation
        for relation, rows
        in sorted(
            by_relation.items()
        )
        if len(rows) >= supported_per_relation
    ]

    if not usable:
        raise RuntimeError(
            "No target relation has enough supported cases."
        )

    rng = random.Random(
        seed
    )

    supported = []

    for relation in usable:
        rows = list(
            by_relation[relation]
        )

        rng.shuffle(rows)

        supported.extend(
            rows[
                :supported_per_relation
            ]
        )

    negatives = build_negatives(
        database,
        supported,
        negative_per_relation,
        seed + 991,
    )

    neg_by_relation = defaultdict(list)

    for case in negatives:
        neg_by_relation[
            case.target_relation
        ].append(case)

    final_relations = [
        relation
        for relation in usable
        if len(
            neg_by_relation[
                relation
            ]
        ) >= negative_per_relation
    ]

    if not final_relations:
        raise RuntimeError(
            "No target relation has enough verified negatives."
        )

    final_supported = []
    final_negative = []

    for relation in final_relations:
        final_supported.extend(
            [
                case
                for case in supported
                if case.target_relation == relation
            ][:supported_per_relation]
        )

        neg = list(
            neg_by_relation[
                relation
            ]
        )

        rng.shuffle(neg)

        final_negative.extend(
            neg[
                :negative_per_relation
            ]
        )

    cases = (
        final_supported
        + final_negative
    )

    rng.shuffle(cases)

    return cases, {
        "candidate_positive_counts": available,
        "target_relations": final_relations,
        "supported": len(final_supported),
        "negative": len(final_negative),
    }


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def transition_support(
    previous_relation,
    target_relation,
    profiles,
):
    if previous_relation is None:
        return 0.0

    profile = profiles.get(
        previous_relation
    )

    if not profile:
        return 0.0

    return second_profile(
        profile
    ).get(
        target_relation,
        0.0,
    )


def target_family(
    relation,
    target_relation,
    profiles,
    neighbors,
):
    if relation == target_relation:
        return 1.0

    score = family_similarity(
        profiles.get(
            relation
        ),
        profiles.get(
            target_relation
        ),
    )

    for row in neighbors.get(
        target_relation,
        [],
    ):
        if (
            isinstance(row, dict)
            and row.get(
                "predicate"
            ) == relation
        ):
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


def choose_initial_depth(
    cache,
    case,
    profiles,
    per_node,
    stats,
):
    hidden = {
        (
            case.subject,
            case.target_relation,
            case.object,
        )
    } if case.gold else set()

    first = cache.get_one(
        case.subject,
        per_node,
        hidden,
        stats,
    )

    best = 0.0

    for relation, _object, _source in first:
        best = max(
            best,
            transition_support(
                relation,
                case.target_relation,
                profiles,
            ),
        )

    return (
        2 if best >= 0.01 else 3,
        best,
    )


# ---------------------------------------------------------------------------
# Main search engines
# ---------------------------------------------------------------------------

def run_bfs(
    cache,
    case,
    budget,
    per_node,
    max_depth,
    stats,
    seed,
):
    hidden = {
        (
            case.subject,
            case.target_relation,
            case.object,
        )
    } if case.gold else set()

    frontier = [
        (
            case.subject,
            (),
        )
    ]

    visited = {
        case.subject
    }

    steps = 0

    while (
        frontier
        and steps < budget
    ):
        node, path = frontier.pop(0)

        if len(path) >= max_depth:
            continue

        edges = cache.get_one(
            node,
            per_node,
            hidden,
            stats,
        )

        for relation, object_, _source in edges:
            steps += 1

            new_path = path + (
                (
                    node,
                    relation,
                    object_,
                ),
            )

            if (
                relation == case.target_relation
                and object_ == case.object
            ):
                return (
                    True,
                    steps,
                    len(new_path),
                    stats,
                )

            if object_ not in visited:
                visited.add(
                    object_
                )

                frontier.append(
                    (
                        object_,
                        new_path,
                    )
                )

            if steps >= budget:
                break

    return (
        False,
        steps,
        0,
        stats,
    )


def run_depth_branch(
    cache,
    case,
    budget,
    per_node,
    max_depth,
    profiles,
    neighbors,
    stats,
):
    chosen_depth, confidence = (
        choose_initial_depth(
            cache,
            case,
            profiles,
            per_node,
            stats,
        )
    )

    hidden = {
        (
            case.subject,
            case.target_relation,
            case.object,
        )
    } if case.gold else set()

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

    while (
        frontier
        and steps < budget
    ):
        frontier.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        node, path, _priority = frontier.pop(0)

        if len(path) >= chosen_depth:
            continue

        edges = cache.get_one(
            node,
            per_node,
            hidden,
            stats,
        )

        candidate_objects = [
            object_
            for _relation, object_, _source
            in edges
        ]

        object_topology = cache.get_many(
            candidate_objects,
            per_node,
            hidden,
            stats,
        ) if candidate_objects else {}

        scored = []

        previous = (
            path[-1][1]
            if path
            else None
        )

        for relation, object_, source in edges:
            degree = min(
                1.0,
                len(
                    object_topology.get(
                        object_,
                        [],
                    )
                )
                / max(
                    1,
                    per_node,
                ),
            )

            score = (
                0.30 * (
                    1.0
                    - degree
                )
                + 0.75 * (
                    1.0
                    if object_
                    == case.object
                    else 0.0
                )
                + 0.20 * (
                    1.0
                    if relation
                    == case.target_relation
                    else 0.0
                )
                + min(
                    0.10,
                    confidence * 2.0,
                )
            )

            scored.append(
                (
                    score,
                    relation,
                    object_,
                    source,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        for score, relation, object_, source in scored:
            steps += 1

            new_path = path + (
                (
                    node,
                    relation,
                    object_,
                ),
            )

            if (
                relation == case.target_relation
                and object_ == case.object
            ):
                return (
                    True,
                    steps,
                    len(new_path),
                    stats,
                )

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

    return (
        False,
        steps,
        0,
        stats,
    )


def run_multi_hypothesis(
    cache,
    case,
    policy,
    budget,
    per_node,
    max_depth,
    profiles,
    neighbors,
    stats,
    seed,
    max_probes,
    case_deadline,
):
    start = time.perf_counter()

    initial_depth, confidence = (
        choose_initial_depth(
            cache,
            case,
            profiles,
            per_node,
            stats,
        )
    )

    hidden = {
        (
            case.subject,
            case.target_relation,
            case.object,
        )
    } if case.gold else set()

    next_id = 1

    hypotheses = [
        H(
            hid=next_id,
            depth_limit=initial_depth,
            node=case.subject,
            path=(),
            score=1.0,
        )
    ]

    next_id += 1

    alternate_depth = min(
        max_depth,
        initial_depth + 1,
    )

    if alternate_depth != initial_depth:
        hypotheses.append(
            H(
                hid=next_id,
                depth_limit=alternate_depth,
                node=case.subject,
                path=(),
                score=0.85,
            )
        )
        next_id += 1

    stats.hypotheses_created = (
        len(hypotheses)
    )

    stats.max_live_hypotheses = len(
        hypotheses
    )

    visited = {
        case.subject
    }

    steps = 0

    # The controller works in attention quanta.
    quantum = 5

    while (
        hypotheses
        and steps < budget
    ):
        if (
            time.perf_counter()
            - start
            > case_deadline
        ):
            stats.cases_aborted += 1
            return (
                False,
                steps,
                0,
                stats,
            )

        if steps >= max_probes:
            stats.cases_aborted += 1
            return (
                False,
                steps,
                0,
                stats,
            )

        live = [
            h
            for h in hypotheses
            if h.alive
        ]

        if not live:
            break

        stats.allocation_rounds += 1

        state_rows = []

        # Batch-fetch the current frontier first.
        frontier_nodes = [
            h.node
            for h in live
        ]

        frontier_edges = cache.get_many(
            frontier_nodes,
            per_node,
            hidden,
            stats,
        )

        for h in live:
            edges = frontier_edges.get(
                h.node,
                [],
            )

            previous = (
                h.path[-1][1]
                if h.path
                else None
            )

            transition = transition_support(
                previous,
                case.target_relation,
                profiles,
            )

            branch = min(
                1.0,
                len(edges)
                / max(
                    1,
                    per_node,
                ),
            )

            novelty = 1.0 / (
                1.0
                + h.expansions
            )

            remaining = max(
                0,
                h.depth_limit
                - len(h.path),
            )

            depth_fit = (
                1.0
                / max(
                    1,
                    remaining,
                )
            )

            base_value = (
                0.55 * transition
                + 0.20 * novelty
                + 0.20 * depth_fit
                - 0.30 * branch
                - 0.05 * h.failures
            )

            if policy in {
                "multi_depth",
            }:
                value = (
                    0.7
                    * depth_fit
                    + 0.3
                    * (
                        1.0
                        - branch
                    )
                )

            elif policy == "multi_depth_branch_value":
                value = base_value

            elif policy == "multi_depth_backtrack":
                value = base_value

            elif policy == "multi_depth_frontier_value":
                value = (
                    base_value
                    + 0.20
                    * min(
                        1.0,
                        len(edges)
                        / 15.0,
                    )
                )

            elif policy == "multi_depth_counterfactual":
                value = base_value

            elif policy == "multi_depth_learned_state":
                value = (
                    base_value
                    + 0.30 * novelty
                    - 0.08 * h.failures
                )

            elif policy == "full_cognitive_controller":
                fam = 0.0

                for relation, _o, _s in edges[:12]:
                    fam = max(
                        fam,
                        target_family(
                            relation,
                            case.target_relation,
                            profiles,
                            neighbors,
                        ),
                    )

                value = (
                    base_value
                    + 0.15 * fam
                    + 0.20 * novelty
                    - 0.08 * h.failures
                )

            else:
                value = base_value

            state_rows.append(
                (
                    value,
                    h,
                )
            )

        state_rows.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        chosen_value, chosen = state_rows[0]

        if policy in {
            "multi_depth_counterfactual",
            "full_cognitive_controller",
        } and len(state_rows) > 1:
            alternate_value, alternate = state_rows[1]

            if (
                alternate_value
                >= chosen_value * 0.95
                and alternate_value
                > chosen_value
            ):
                chosen = alternate
                chosen_value = alternate_value
                stats.counterfactual_switches += 1
                stats.branch_switches += 1

        edges = frontier_edges.get(
            chosen.node,
            [],
        )

        if not edges:
            chosen.failures += 1

            if (
                policy
                in {
                    "multi_depth_backtrack",
                    "multi_depth_counterfactual",
                    "multi_depth_learned_state",
                    "full_cognitive_controller",
                }
                and chosen.failures >= 2
            ):
                chosen.alive = False
                stats.hypothesis_abandons += 1
                stats.backtracks += 1

            continue

        previous = (
            chosen.path[-1][1]
            if chosen.path
            else None
        )

        edge_nodes = [
            object_
            for _relation, object_, _source
            in edges
        ]

        # BATCH child topology for this entire attention quantum.
        child_topology = cache.get_many(
            edge_nodes,
            per_node,
            hidden,
            stats,
        )

        ranked = []

        for relation, object_, source in edges:
            degree = min(
                1.0,
                len(
                    child_topology.get(
                        object_,
                        [],
                    )
                )
                / max(
                    1,
                    per_node,
                ),
            )

            transition = transition_support(
                previous,
                case.target_relation,
                profiles,
            )

            family = target_family(
                relation,
                case.target_relation,
                profiles,
                neighbors,
            )

            endpoint = (
                1.0
                if object_ == case.object
                else 0.0
            )

            target = (
                1.0
                if relation
                == case.target_relation
                else 0.0
            )

            score = (
                0.45 * endpoint
                + 0.20 * target
                + 0.20 * transition
                - 0.30 * degree
            )

            if policy in {
                "multi_depth_branch_value",
                "multi_depth_frontier_value",
                "multi_depth_counterfactual",
                "multi_depth_learned_state",
                "full_cognitive_controller",
            }:
                score += (
                    0.10
                    * family
                )

            # In uncertainty-aware/full modes, retain a small alternative
            # rather than fully collapsing the ranking.
            if policy in {
                "multi_depth_learned_state",
                "full_cognitive_controller",
            }:
                score += (
                    0.02
                    * random.Random(
                        chosen.hid
                        * 1009
                        + steps
                        + hash(object_)
                    ).random()
                )

            ranked.append(
                (
                    score,
                    relation,
                    object_,
                    source,
                )
            )

        ranked.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        yielded = 0

        for (
            score,
            relation,
            object_,
            source,
        ) in ranked[:quantum]:
            if steps >= budget:
                break

            steps += 1
            chosen.expansions += 1

            new_path = chosen.path + (
                (
                    chosen.node,
                    relation,
                    object_,
                ),
            )

            if (
                relation == case.target_relation
                and object_ == case.object
            ):
                return (
                    True,
                    steps,
                    len(new_path),
                    stats,
                )

            if object_ not in visited:
                visited.add(
                    object_
                )

                child = H(
                    hid=next_id,
                    depth_limit=chosen.depth_limit,
                    node=object_,
                    path=new_path,
                    score=score,
                )

                next_id += 1

                hypotheses.append(
                    child
                )

                stats.hypotheses_created += 1
                yielded += 1
                stats.information_gain += 1.0

                if len(hypotheses) > 24:
                    live2 = [
                        h
                        for h in hypotheses
                        if h.alive
                    ]

                    live2.sort(
                        key=lambda h: h.score,
                        reverse=True,
                    )

                    keep = {
                        h.hid
                        for h in live2[:12]
                    }

                    for h in live2[12:]:
                        if h.hid not in keep:
                            h.alive = False
                            stats.hypothesis_abandons += 1

        if (
            yielded == 0
        ):
            chosen.failures += 1
        else:
            chosen.failures = 0

        if (
            policy
            in {
                "multi_depth_backtrack",
                "multi_depth_counterfactual",
                "multi_depth_learned_state",
                "full_cognitive_controller",
            }
            and chosen.failures >= 2
        ):
            chosen.alive = False
            stats.hypothesis_abandons += 1
            stats.backtracks += 1

            for _value, alternate in state_rows[1:]:
                if alternate.alive:
                    alternate.score += 0.15
                    stats.hypothesis_promotions += 1
                    stats.branch_switches += 1
                    break

        stats.max_live_hypotheses = max(
            stats.max_live_hypotheses,
            sum(
                1
                for h in hypotheses
                if h.alive
            ),
        )

    return (
        False,
        steps,
        0,
        stats,
    )


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

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
    cache_entries,
    progress_every,
    max_probes,
    max_case_seconds,
):
    con = connect_ro(
        database
    )

    cache = TopologyCache(
        con,
        cache_entries,
    )

    started = time.perf_counter()

    try:
        rows = []
        totals = Stats()

        for index, case in enumerate(
            cases
        ):
            case_started = time.perf_counter()

            local = Stats()

            try:
                if policy == "bfs":
                    pred, steps, path_length, local = run_bfs(
                        cache,
                        case,
                        budget,
                        per_node,
                        max_depth,
                        local,
                        seed + index,
                    )
                elif policy == "depth_branch":
                    pred, steps, path_length, local = run_depth_branch(
                        cache,
                        case,
                        budget,
                        per_node,
                        max_depth,
                        profiles,
                        neighbors,
                        local,
                    )
                else:
                    pred, steps, path_length, local = run_multi_hypothesis(
                        cache,
                        case,
                        policy,
                        budget,
                        per_node,
                        max_depth,
                        profiles,
                        neighbors,
                        local,
                        seed + index,
                        max_probes,
                        max_case_seconds,
                    )
            except Exception:
                local.cases_aborted += 1
                pred = False
                steps = min(
                    budget,
                    max_probes,
                )
                path_length = 0

            elapsed_case = (
                time.perf_counter()
                - case_started
            )

            # Copy controller metrics into aggregate.
            for field_name in (
                "hypotheses_created",
                "hypothesis_promotions",
                "hypothesis_abandons",
                "backtracks",
                "branch_switches",
                "allocation_rounds",
                "counterfactual_switches",
                "information_gain",
                "max_live_hypotheses",
                "topology_batch_queries",
                "topology_nodes_fetched",
                "topology_cache_hits",
                "topology_cache_misses",
                "cases_aborted",
            ):
                setattr(
                    totals,
                    field_name,
                    getattr(
                        totals,
                        field_name,
                    )
                    + getattr(
                        local,
                        field_name,
                    ),
                )

            rows.append(
                (
                    case,
                    bool(pred),
                    int(steps),
                    int(path_length),
                )
            )

            done = index + 1

            if (
                done == 1
                or done % progress_every == 0
                or done == len(cases)
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )
                rate = (
                    done / elapsed
                    if elapsed > 0
                    else 0.0
                )
                eta = (
                    (len(cases) - done)
                    / rate
                    if rate > 0
                    else 0.0
                )

                print(
                    f"    [CASE {done}/{len(cases)}] "
                    f"{policy:32s} "
                    f"rate={rate:.2f}/s "
                    f"elapsed={elapsed:.1f}s "
                    f"eta={eta:.1f}s "
                    f"case={elapsed_case:.2f}s "
                    f"cache={len(cache):,} "
                    f"batchq={totals.topology_batch_queries:,}",
                    flush=True,
                )

        positives = [
            row
            for row in rows
            if row[0].gold
        ]

        negatives = [
            row
            for row in rows
            if not row[0].gold
        ]

        recovered = sum(
            row[1]
            for row in positives
        )

        false = sum(
            row[1]
            for row in negatives
        )

        result = {
            "cases": len(rows),
            "accuracy": (
                sum(
                    row[1] == row[0].gold
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
                row[1]
                for row in rows
            ),
            "mean_steps": (
                statistics.mean(
                    row[2]
                    for row in rows
                )
                if rows
                else 0.0
            ),
            "mean_path_length": (
                statistics.mean(
                    row[3]
                    for row in rows
                    if row[1]
                )
                if any(
                    row[1]
                    for row in rows
                )
                else 0.0
            ),
            "budget_exhausted": sum(
                1
                for row in rows
                if (
                    not row[1]
                    and row[2] >= budget
                )
            ),
            "controller_stats": asdict(
                totals
            ),
            "topology_cache_entries": len(
                cache
            ),
            "policy_seconds": (
                time.perf_counter()
                - started
            ),
            "by_target_relation": relation_metrics(
                rows
            ),
        }

        return result

    finally:
        con.close()


def relation_metrics(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row[0].target_relation
        ].append(row)

    out = {}

    for relation, values in grouped.items():
        positives = [
            row
            for row in values
            if row[0].gold
        ]

        negatives = [
            row
            for row in values
            if not row[0].gold
        ]

        out[
            relation
        ] = {
            "cases": len(values),
            "supported_cases": len(
                positives
            ),
            "supported_recovery": (
                sum(
                    row[1]
                    for row in positives
                )
                / len(positives)
                if positives
                else 0.0
            ),
            "false_proof_rate": (
                sum(
                    row[1]
                    for row in negatives
                )
                / len(negatives)
                if negatives
                else 0.0
            ),
            "mean_steps": statistics.mean(
                row[2]
                for row in values
            ),
        }

    return out


def mean_ci(values):
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    mean = statistics.mean(
        values
    )

    std = (
        statistics.stdev(values)
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

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


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
        default=r".\results\v575_3_optimized.json",
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
        "--middle-limit",
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
        default=57530,
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
        "--cache-entries",
        type=int,
        default=50000,
    )

    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--max-probes-per-case",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--max-case-seconds",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--skip-bfs",
        action="store_true",
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
        "=== V575.3 OPTIMIZED MULTI-HYPOTHESIS CONTROLLER ==="
    )
    print(
        f"database             : {database}"
    )
    print(
        f"database size        : "
        f"{database.stat().st_size / 1024**3:.2f} GB"
    )
    print(
        f"V568                 : {v568_path}"
    )
    print(
        f"workers              : {args.workers}"
    )
    print(
        f"seeds                : {args.seeds}"
    )
    print(
        f"budget               : {args.budget}"
    )
    print(
        f"per-node             : {args.per_node}"
    )
    print(
        f"max-depth            : {args.max_depth}"
    )
    print(
        f"cache entries        : {args.cache_entries:,}"
    )
    print(
        f"progress every       : {args.progress_every}"
    )
    print(
        f"max probes/case      : {args.max_probes_per_case}"
    )
    print(
        f"max seconds/case     : {args.max_case_seconds}"
    )
    print(
        "source graph         : READ-ONLY"
    )
    print(
        "topology mode        : BATCHED + LRU"
    )
    print()

    profiles, neighbors = load_v568(
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

    print()
    print(
        "=== GRAPH PREDICATES ==="
    )

    for i, info in enumerate(
        inventory,
        1,
    ):
        marker = (
            " [V568]"
            if info.predicate in profiles
            else ""
        )

        print(
            f"{i:2d}. "
            f"{info.predicate:32s} "
            f"{info.edge_count:12,}"
            f"{marker}"
        )

    print()
    print(
        "=== BUILDING SHARED BALANCED ORACLE ==="
    )

    cases, oracle_data = (
        build_balanced_oracle(
            database,
            inventory,
            args.sample_each,
            args.middle_limit,
            args.case_cap,
            args.supported_per_relation,
            args.negative_per_relation,
            args.seed_start,
            args.workers,
        )
    )

    print()
    print(
        "=== ORACLE ==="
    )
    print(
        f"target relations : "
        f"{len(oracle_data['target_relations'])}"
    )
    print(
        f"supported        : "
        f"{oracle_data['supported']}"
    )
    print(
        f"hard negatives   : "
        f"{oracle_data['negative']}"
    )
    print(
        f"total             : "
        f"{len(cases)}"
    )

    for relation in oracle_data[
        "target_relations"
    ]:
        print(
            f"  {relation}"
        )

    policies = [
        p
        for p in ALL_POLICIES
        if not (
            args.skip_bfs
            and p == "bfs"
        )
    ]

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

        results = {}

        for policy in policies:
            print()
            print(
                f"[POLICY START] {policy}",
                flush=True,
            )

            t = time.perf_counter()

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
                args.cache_entries,
                args.progress_every,
                args.max_probes_per_case,
                args.max_case_seconds,
            )

            result["seconds"] = (
                time.perf_counter()
                - t
            )

            results[
                policy
            ] = result

            cs = result[
                "controller_stats"
            ]

            print(
                f"[POLICY END] "
                f"{policy:32s} "
                f"acc={result['accuracy']:.4f} "
                f"recover={result['supported_recovery']:.4f} "
                f"false={result['false_proof_rate']:.4f} "
                f"steps={result['mean_steps']:.2f} "
                f"cache={result['topology_cache_entries']:,} "
                f"batchq={cs['topology_batch_queries']:,} "
                f"aborts={cs['cases_aborted']:,} "
                f"time={result['seconds']:.2f}s",
                flush=True,
            )

        per_seed.append(
            {
                "seed": seed,
                "results": results,
            }
        )

    summary = {}

    for policy in policies:
        summary[
            policy
        ] = {}

        for metric in (
            "accuracy",
            "supported_recovery",
            "false_proof_rate",
            "mean_steps",
            "budget_exhausted",
        ):
            summary[
                policy
            ][metric] = mean_ci(
                [
                    float(
                        run["results"][policy][metric]
                    )
                    for run in per_seed
                ]
            )

        for metric in (
            "hypotheses_created",
            "hypothesis_promotions",
            "hypothesis_abandons",
            "backtracks",
            "branch_switches",
            "allocation_rounds",
            "counterfactual_switches",
            "information_gain",
            "max_live_hypotheses",
            "topology_batch_queries",
            "topology_nodes_fetched",
            "topology_cache_hits",
            "topology_cache_misses",
            "cases_aborted",
        ):
            summary[
                policy
            ][
                "controller_" + metric
            ] = mean_ci(
                [
                    float(
                        run["results"][policy][
                            "controller_stats"
                        ].get(
                            metric,
                            0,
                        )
                    )
                    for run in per_seed
                ]
            )

    # Use the validated V573 base when present.
    baseline_name = (
        "depth_branch"
        if "depth_branch" in summary
        else "bfs"
    )

    baseline = summary[
        baseline_name
    ]

    deltas = {}

    for policy in policies:
        deltas[
            policy
        ] = {
            "accuracy_vs_base": (
                summary[policy][
                    "accuracy"
                ]["mean"]
                - baseline[
                    "accuracy"
                ]["mean"]
            ),
            "recovery_vs_base": (
                summary[policy][
                    "supported_recovery"
                ]["mean"]
                - baseline[
                    "supported_recovery"
                ]["mean"]
            ),
            "false_proof_vs_base": (
                summary[policy][
                    "false_proof_rate"
                ]["mean"]
                - baseline[
                    "false_proof_rate"
                ]["mean"]
            ),
            "steps_vs_base": (
                summary[policy][
                    "mean_steps"
                ]["mean"]
                - baseline[
                    "mean_steps"
                ]["mean"]
            ),
        }

    utility = {}

    for policy in policies:
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

    candidates = [
        p
        for p in policies
        if p != "bfs"
    ]

    winner = max(
        candidates,
        key=lambda p: utility[p],
    )

    stability = {}

    for policy in policies:
        better = sum(
            1
            for run in per_seed
            if run["results"][policy][
                "accuracy"
            ]
            >
            run["results"][
                baseline_name
            ]["accuracy"]
        )

        stability[
            policy
        ] = {
            "better_than_base_seeds": better,
            "seed_count": len(
                per_seed
            ),
        }

    print()
    print(
        "=== V575.3 SUMMARY ==="
    )

    print(
        f"{'policy':32s} "
        f"{'acc':>9s} "
        f"{'recover':>10s} "
        f"{'false':>9s} "
        f"{'steps':>10s}"
    )

    for policy in policies:
        print(
            f"{policy:32s} "
            f"{summary[policy]['accuracy']['mean']:.4f} "
            f"{summary[policy]['supported_recovery']['mean']:.4f} "
            f"{summary[policy]['false_proof_rate']['mean']:.4f} "
            f"{summary[policy]['mean_steps']['mean']:.2f}"
        )

    print()
    print(
        f"=== DELTAS VS {baseline_name.upper()} ==="
    )

    for policy in policies:
        if policy == baseline_name:
            continue

        d = deltas[
            policy
        ]

        print(
            f"{policy:32s} "
            f"Δacc={d['accuracy_vs_base']:+.4f} "
            f"Δrecover={d['recovery_vs_base']:+.4f} "
            f"Δfalse={d['false_proof_vs_base']:+.4f} "
            f"Δsteps={d['steps_vs_base']:+.2f}"
        )

    print()
    print(
        "=== WINNER ACTIVITY ==="
    )

    for key in (
        "hypotheses_created",
        "hypothesis_promotions",
        "hypothesis_abandons",
        "backtracks",
        "branch_switches",
        "allocation_rounds",
        "counterfactual_switches",
        "information_gain",
        "max_live_hypotheses",
        "topology_batch_queries",
        "topology_cache_hits",
        "topology_cache_misses",
        "cases_aborted",
    ):
        print(
            f"{key:30s} "
            f"mean="
            f"{summary[winner]['controller_' + key]['mean']:.2f}"
        )

    report = {
        "benchmark": (
            "v575_3_optimized_multi_hypothesis_controller"
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
            "graph_top_predicates": [
                info.predicate
                for info in inventory
            ],
            "profile_overlap": [
                info.predicate
                for info in inventory
                if info.predicate in profiles
            ],
        },
        "oracle": {
            **oracle_data,
            "total_cases": len(cases),
            "direct_target_edge_hidden": True,
        },
        "policies": policies,
        "baseline_name": baseline_name,
        "per_seed": per_seed,
        "summary": summary,
        "deltas_vs_base": deltas,
        "utility": utility,
        "stability_vs_base": stability,
        "winner": winner,
        "performance": {
            "topology_cache": True,
            "cache_entries": args.cache_entries,
            "batched_subject_queries": True,
            "progress_reporting": True,
            "max_probes_per_case": args.max_probes_per_case,
            "max_case_seconds": args.max_case_seconds,
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
        "V575.3 COMPLETE"
    )
    print(
        f"baseline : {baseline_name}"
    )
    print(
        f"winner   : {winner}"
    )
    print(
        f"JSON     : {output}"
    )
    print(
        f"elapsed  : "
        f"{report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()

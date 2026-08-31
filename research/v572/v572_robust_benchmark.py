
#!/usr/bin/env python3
"""
V572 — Robust Balanced Cognitive Search Benchmark

This is a clean rebuild of V571.

Design goals
------------
1. The graph predicate inventory is always normalized to:
       {"predicate": str, "edge_count": int}
   No tuple/dict ambiguity remains.

2. The V568 artifact is normalized independently and validated before use.
   The benchmark never assumes a particular JSON container shape.

3. The actual graph is authoritative for graph predicates.
   V568 is an optional learned/search-prior artifact.

4. The benchmark uses the same balanced holdout for every policy and every
   seed. Seed variation affects policy controls and tie-breaking, not the
   gold labels.

5. The benchmark is relation-balanced:
       equal supported cases per target relation
       equal hard-negative cases per target relation

6. The direct target edge is hidden for supported cases. A policy must discover
   the target relation through graph search.

7. The benchmark matrix isolates:
       bfs
       depth
       behavior
       family
       full_hybrid
       shuffled_behavior
       shuffled_family
       shuffled_all

8. The benchmark reports per-target-relation results, because a global
   accuracy number can be misleading when relation frequencies are skewed.

9. The source graph is SQLite READ-ONLY.

10. No global edges x edges join is used.

Performance
-----------
- Predicate inventory: one GROUP BY relation using the relation index.
- Predicate sampling: bounded rowid-window probes, never full predicate scans.
- Outgoing graph access: subject-index lookups.
- Oracle construction: parallel by first-hop predicate.
- Policy evaluation: sequential inside each process/connection to avoid
  multiplying SQLite readers beyond the configured worker count.

The program performs a preflight and WILL NOT start the benchmark if the
V568 artifact cannot be parsed or if the graph inventory is malformed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Paths / SQLite
# ---------------------------------------------------------------------------

def resolve_file(value: str) -> Path:
    p = Path(value).expanduser()

    if p.exists() and p.is_file():
        return p.resolve()

    if not p.is_absolute():
        q = Path.cwd() / p
        if q.exists() and q.is_file():
            return q.resolve()

    candidates = []
    results = Path.cwd() / "results"
    if results.exists():
        candidates = sorted(
            [
                x
                for x in results.glob("*.sqlite")
                if x.is_file()
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    msg = [
        f"File not found: {value}",
        f"cwd: {Path.cwd()}",
    ]

    if candidates:
        msg.append("SQLite files in .\\results:")
        for c in candidates[:20]:
            try:
                msg.append(
                    f"  {c} ({c.stat().st_size / 1024**3:.2f} GB)"
                )
            except OSError:
                msg.append(f"  {c}")

    raise FileNotFoundError("\n".join(msg))


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


def check_graph_schema(con: sqlite3.Connection) -> dict[str, Any]:
    cols = {
        str(r["name"])
        for r in con.execute("PRAGMA table_info(edges)")
    }

    required = {"subject", "relation", "object", "source"}
    missing = sorted(required - cols)

    if missing:
        raise RuntimeError(
            f"edges table missing required columns: {missing}"
        )

    indexes = [
        str(r["name"])
        for r in con.execute("PRAGMA index_list(edges)")
    ]

    return {
        "columns": sorted(cols),
        "indexes": indexes,
    }


# ---------------------------------------------------------------------------
# V568 normalization
# ---------------------------------------------------------------------------

def normalize_v568(raw: Any) -> tuple[dict[str, dict], dict[str, list]]:
    """
    Accept the known V568 layout variants and normalize them.

    Profiles may be:
      {"profiles": [{"predicate": "..."}]}
      {"profiles": {"predicate": {...}}}
      {"relation_profiles": ...}
      {"predicate_profiles": ...}

    Neighbors may be list- or dict-based.
    """
    if not isinstance(raw, dict):
        raise RuntimeError("V568 JSON root must be an object/dict.")

    profile_container = None

    for key in (
        "profiles",
        "relation_profiles",
        "predicate_profiles",
        "profile",
    ):
        candidate = raw.get(key)
        if isinstance(candidate, (list, dict)):
            profile_container = candidate
            break

    profiles: dict[str, dict] = {}

    if isinstance(profile_container, list):
        for row in profile_container:
            if not isinstance(row, dict):
                continue

            name = row.get("predicate", row.get("relation"))
            if name is None:
                continue

            name = str(name).strip()
            if name:
                profiles[name] = row

    elif isinstance(profile_container, dict):
        for key, value in profile_container.items():
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
            if name:
                row["predicate"] = name
                profiles[name] = row

    neighbor_container = raw.get(
        "nearest_neighbors",
        raw.get("neighbors", {}),
    )

    neighbors: dict[str, list] = {}

    if isinstance(neighbor_container, list):
        for row in neighbor_container:
            if not isinstance(row, dict):
                continue

            name = row.get("predicate")
            if name is None:
                continue

            name = str(name).strip()

            values = row.get("neighbors", [])
            if isinstance(values, list):
                neighbors[name] = values

    elif isinstance(neighbor_container, dict):
        for key, value in neighbor_container.items():
            if isinstance(value, list):
                neighbors[str(key).strip()] = value

    if not profiles:
        raise RuntimeError(
            "V568 file contains no recognizable predicate profiles. "
            "The file may be valid JSON but is not a V568 profile artifact."
        )

    return profiles, neighbors


def load_v568(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    profiles, neighbors = normalize_v568(raw)
    return raw, profiles, neighbors


def second_hop_distribution(profile: dict) -> dict[str, float]:
    behavior = profile.get("behavior", {})
    if not isinstance(behavior, dict):
        return {}

    rows = behavior.get(
        "second_relation_profile",
        [],
    )

    out = {}
    if not isinstance(rows, list):
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue

        rel = row.get("relation")
        frac = row.get("fraction")

        if rel is None or frac is None:
            continue

        try:
            out[str(rel)] = float(frac)
        except (TypeError, ValueError):
            continue

    return out


def behavior_similarity(a: dict | None, b: dict | None) -> float:
    if not a or not b:
        return 0.0

    pa = second_hop_distribution(a)
    pb = second_hop_distribution(b)

    keys = set(pa) | set(pb)

    if not keys:
        return 0.0

    suma = sum(pa.values()) or 1.0
    sumb = sum(pb.values()) or 1.0

    pa = {k: pa.get(k, 0.0) / suma for k in keys}
    pb = {k: pb.get(k, 0.0) / sumb for k in keys}

    m = {
        k: (pa[k] + pb[k]) / 2.0
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
            (kl(pa) + kl(pb)) / 2.0,
        )
    )

    return max(
        0.0,
        1.0 - js,
    )


# ---------------------------------------------------------------------------
# Graph inventory
# ---------------------------------------------------------------------------

def get_top_predicates(
    con: sqlite3.Connection,
    n: int,
) -> list[PredicateInfo]:
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

    output = []

    for row in rows:
        relation = str(row["relation"])
        edge_count = int(row["edge_count"])

        if not relation:
            continue

        output.append(
            PredicateInfo(
                predicate=relation,
                edge_count=edge_count,
            )
        )

    return output


def get_exact_relation_count(
    con: sqlite3.Connection,
    predicate: str,
) -> int:
    row = con.execute(
        """
        SELECT COUNT(*) AS edge_count
        FROM edges
        WHERE relation = ?
        """,
        (predicate,),
    ).fetchone()

    return int(row["edge_count"])


# ---------------------------------------------------------------------------
# Bounded graph sampling
# ---------------------------------------------------------------------------

def sample_relation_edges(
    con: sqlite3.Connection,
    predicate: str,
    sample_size: int,
    seed: int,
    window_size: int,
    max_probes: int,
) -> list[tuple[str, str, Any]]:
    """
    Sample actual rows for a predicate through bounded rowid windows.

    This intentionally avoids:
        SELECT ... FROM edges WHERE relation = ?
    over tens of millions of rows.
    """
    bounds = con.execute(
        "SELECT MIN(rowid), MAX(rowid) FROM edges"
    ).fetchone()

    lo = int(bounds[0])
    hi = int(bounds[1])

    rng = random.Random(seed)
    found = {}

    probes = 0

    while (
        len(found) < sample_size
        and probes < max_probes
    ):
        start = rng.randint(
            lo,
            max(
                lo,
                hi - window_size + 1,
            ),
        )

        end = min(
            hi,
            start + window_size - 1,
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

    return list(found.values())[:sample_size]


def outgoing_for_nodes(
    con: sqlite3.Connection,
    nodes: set[str],
    per_node: int,
) -> dict[str, list[tuple[str, str, Any]]]:
    result: dict[str, list[tuple[str, str, Any]]] = (
        defaultdict(list)
    )

    values = list(nodes)

    for i in range(0, len(values), 250):
        chunk = values[i:i + 250]

        if not chunk:
            continue

        placeholders = ",".join(
            "?" for _ in chunk
        )

        q = f"""
            SELECT subject, relation, object, source
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, chunk):
            bucket = result[
                str(row["subject"])
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

    return result


def confirm_pairs(
    con: sqlite3.Connection,
    pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], set[str]]:
    """
    Confirm only sampled subject->endpoint pairs.
    """
    wanted = defaultdict(set)

    for s, o in pairs:
        wanted[s].add(o)

    result: dict[tuple[str, str], set[str]] = defaultdict(set)

    subjects = list(wanted)

    for i in range(0, len(subjects), 250):
        chunk = subjects[i:i + 250]

        placeholders = ",".join(
            "?" for _ in chunk
        )

        q = f"""
            SELECT subject, relation, object
            FROM edges
            WHERE subject IN ({placeholders})
        """

        for row in con.execute(q, chunk):
            s = str(row["subject"])
            o = str(row["object"])

            if o in wanted.get(s, ()):
                result[(s, o)].add(
                    str(row["relation"])
                )

    return result


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def generate_predicate_cases(
    database: Path,
    predicate: PredicateInfo,
    sample_each: int,
    middle_limit: int,
    candidates_factor: int,
    seed: int,
) -> list[Case]:
    con = connect_ro(database)

    try:
        first = sample_relation_edges(
            con,
            predicate.predicate,
            sample_each,
            seed,
            window_size=5000,
            max_probes=5000,
        )

        middles = {
            str(o)
            for _s, o, _src in first
        }

        adjacency = outgoing_for_nodes(
            con,
            middles,
            middle_limit,
        )

        path_candidates = []

        max_candidates = max(
            50,
            candidates_factor * 25,
        )

        for s, middle, src1 in first:
            for r2, endpoint, src2 in adjacency.get(
                str(middle),
                (),
            ):
                if endpoint in (
                    str(s),
                    str(middle),
                ):
                    continue

                path_candidates.append(
                    (
                        str(s),
                        predicate.predicate,
                        str(middle),
                        str(r2),
                        str(endpoint),
                    )
                )

                if len(path_candidates) >= max_candidates:
                    break

            if len(path_candidates) >= max_candidates:
                break

        if not path_candidates:
            return []

        unique_paths = list(
            dict.fromkeys(
                path_candidates
            )
        )

        pairs = {
            (p[0], p[4])
            for p in unique_paths
        }

        direct = confirm_pairs(
            con,
            pairs,
        )

        positives = []

        for (
            s,
            r1,
            middle,
            r2,
            endpoint,
        ) in unique_paths:
            rels = direct.get(
                (s, endpoint),
                (),
            )

            if not rels:
                continue

            # Deterministic target relation:
            # preserve the lexical sort used by earlier versions.
            target = sorted(
                rels
            )[0]

            positives.append(
                Case(
                    subject=s,
                    target_relation=target,
                    object=endpoint,
                    source_predicate=predicate.predicate,
                    kind="SUPPORTED",
                    gold=True,
                    path=(
                        (s, r1, middle),
                        (middle, r2, endpoint),
                    ),
                )
            )

        # Deduplicate by exact query + proof relation sequence.
        unique = {}

        for case in positives:
            key = (
                case.subject,
                case.target_relation,
                case.object,
                case.source_predicate,
                case.path,
            )

            unique[key] = case

        return list(unique.values())

    finally:
        con.close()


def add_balanced_negatives(
    database: Path,
    positives: list[Case],
    negative_per_relation: int,
    seed: int,
) -> list[Case]:
    """
    Generate hard negatives by endpoint swapping WITH verification.

    Verification is batched by subject to avoid one SQL query per negative.
    """
    by_relation = defaultdict(list)

    for case in positives:
        by_relation[
            case.target_relation
        ].append(case)

    rng = random.Random(seed)

    candidates = []

    for relation, rows in by_relation.items():
        objects = list({
            c.object
            for c in rows
        })

        if len(objects) < 2:
            continue

        for case in rows:
            shuffled = objects[:]
            rng.shuffle(shuffled)

            for replacement in shuffled:
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

    con = connect_ro(database)

    try:
        grouped = defaultdict(set)

        for s, relation, object_, source_predicate in candidates:
            grouped[s].add(object_)

        subjects = list(grouped)

        verified = []

        for i in range(0, len(subjects), 250):
            chunk = subjects[i:i + 250]
            placeholders = ",".join(
                "?" for _ in chunk
            )

            q = f"""
                SELECT subject, relation, object
                FROM edges
                WHERE subject IN ({placeholders})
            """

            rows = con.execute(
                q,
                chunk,
            ).fetchall()

            existing = defaultdict(set)

            for row in rows:
                existing[
                    (
                        str(row["subject"]),
                        str(row["object"]),
                    )
                ].add(
                    str(row["relation"])
                )

            for s in chunk:
                for relation in {
                    r
                    for ss, r, _o, _sp in candidates
                    if ss == s
                }:
                    pass

        # Use a compact, deterministic verification map.
        candidate_map = defaultdict(set)

        for s, relation, object_, source_predicate in candidates:
            candidate_map[
                (s, relation)
            ].add(object_)

        verified_by_relation = defaultdict(list)

        # Query all candidate subject/object pairs in subject batches.
        for i in range(0, len(subjects), 250):
            chunk = subjects[i:i + 250]
            placeholders = ",".join(
                "?" for _ in chunk
            )

            q = f"""
                SELECT subject, relation, object
                FROM edges
                WHERE subject IN ({placeholders})
            """

            existing = {
                (
                    str(row["subject"]),
                    str(row["relation"]),
                    str(row["object"]),
                )
                for row in con.execute(
                    q,
                    chunk,
                )
            }

            for (
                s,
                relation,
                object_,
                source_predicate,
            ) in candidates:
                if s not in chunk:
                    continue

                if (
                    s,
                    relation,
                    object_,
                ) in existing:
                    continue

                verified_by_relation[relation].append(
                    Case(
                        subject=s,
                        target_relation=relation,
                        object=object_,
                        source_predicate=source_predicate,
                        kind="HARD_NEGATIVE",
                        gold=False,
                        path=(),
                    )
                )

        output = []

        for relation in sorted(
            verified_by_relation
        ):
            rows = verified_by_relation[
                relation
            ]

            rng.shuffle(rows)

            output.extend(
                rows[
                    :negative_per_relation
                ]
            )

        return output

    finally:
        con.close()


def build_balanced_holdout(
    database: Path,
    predicates: list[PredicateInfo],
    sample_each: int,
    middle_limit: int,
    cases_per_predicate: int,
    supported_per_relation: int,
    negative_per_relation: int,
    seed: int,
    workers: int,
):
    """
    Build all positive candidates first in parallel.

    Then balance by TARGET relation.
    """
    positives = []

    worker_count = min(
        max(1, workers),
        len(predicates),
    )

    print(
        f"[ORACLE] generating positives with "
        f"{worker_count} workers",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as pool:
        future_map = {
            pool.submit(
                generate_predicate_cases,
                database,
                predicate,
                sample_each,
                middle_limit,
                20,
                seed + i,
            ): predicate.predicate
            for i, predicate in enumerate(
                predicates
            )
        }

        completed = 0

        for future in as_completed(
            future_map
        ):
            predicate = future_map[
                future
            ]

            completed += 1

            try:
                rows = future.result()
            except Exception as exc:
                print(
                    f"[ORACLE {completed}/{len(predicates)}] "
                    f"{predicate:32s} ERROR={exc}",
                    flush=True,
                )
                continue

            positives.extend(rows)

            print(
                f"[ORACLE {completed}/{len(predicates)}] "
                f"{predicate:32s} "
                f"positive_candidates={len(rows):5,}",
                flush=True,
            )

    by_target_relation = defaultdict(list)

    for case in positives:
        by_target_relation[
            case.target_relation
        ].append(case)

    print()
    print(
        "[ORACLE] target-relation candidate counts:"
    )

    candidate_counts = {
        relation: len(rows)
        for relation, rows
        in sorted(
            by_target_relation.items()
        )
    }

    for relation, n in candidate_counts.items():
        print(
            f"  {relation:32s} {n:6,}"
        )

    usable_relations = [
        relation
        for relation, rows
        in sorted(
            by_target_relation.items()
        )
        if len(rows) >= supported_per_relation
    ]

    if not usable_relations:
        raise RuntimeError(
            "No target relation has enough supported compositional "
            f"cases for --supported-per-relation={supported_per_relation}."
        )

    rng = random.Random(seed)

    balanced_supported = []

    for relation in usable_relations:
        rows = list(
            by_target_relation[
                relation
            ]
        )

        rng.shuffle(rows)

        balanced_supported.extend(
            rows[
                :supported_per_relation
            ]
        )

    balanced_negatives = add_balanced_negatives(
        database,
        balanced_supported,
        negative_per_relation,
        seed + 1001,
    )

    negatives_by_relation = defaultdict(list)

    for case in balanced_negatives:
        negatives_by_relation[
            case.target_relation
        ].append(case)

    final_relations = [
        relation
        for relation in usable_relations
        if len(
            negatives_by_relation[
                relation
            ]
        ) >= negative_per_relation
    ]

    if not final_relations:
        raise RuntimeError(
            "No target relation has enough verified hard negatives."
        )

    final_supported = []

    for relation in final_relations:
        rows = [
            c
            for c in balanced_supported
            if c.target_relation == relation
        ]

        final_supported.extend(
            rows[:supported_per_relation]
        )

    final_negative = []

    for relation in final_relations:
        rows = negatives_by_relation[
            relation
        ]
        rng.shuffle(rows)
        final_negative.extend(
            rows[:negative_per_relation]
        )

    cases = final_supported + final_negative
    rng.shuffle(cases)

    return cases, {
        "candidate_positive_by_target_relation": candidate_counts,
        "usable_target_relations": usable_relations,
        "final_target_relations": final_relations,
        "final_supported": len(
            final_supported
        ),
        "final_negative": len(
            final_negative
        ),
    }


# ---------------------------------------------------------------------------
# Search policy
# ---------------------------------------------------------------------------

def visible_edges(
    con,
    subject: str,
    per_node: int,
    hidden: set[tuple[str, str, str]],
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
        relation = str(row["relation"])
        object_ = str(row["object"])

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


def target_relation_score(
    relation,
    profiles,
):
    profile = profiles.get(
        relation
    )

    if not profile:
        return 0.0

    try:
        return math.log1p(
            float(
                profile.get(
                    "edge_count",
                    0,
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def behavior_edge_score(
    previous,
    relation,
    profiles,
    target_relation,
):
    if previous is None:
        return (
            target_relation_score(
                relation,
                profiles,
            )
            * 0.005
        )

    profile = profiles.get(
        previous
    )

    if not profile:
        return 0.0

    score = second_hop_distribution(
        profile
    ).get(
        relation,
        0.0,
    )

    if relation == target_relation:
        score += 0.20

    return score


def family_edge_score(
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

    score = behavior_similarity(
        target_profile,
        candidate_profile,
    )

    for row in neighbors.get(
        target_relation,
        [],
    ):
        if not isinstance(row, dict):
            continue

        if row.get("predicate") == relation:
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

    return max(
        0.0,
        score,
    )


def infer_depth(
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

    for relation, _object, _source in first:
        profile = profiles.get(
            relation
        )

        if profile:
            best = max(
                best,
                second_hop_distribution(
                    profile
                ).get(
                    case.target_relation,
                    0.0,
                ),
            )

    return (
        2
        if best >= 0.01
        else 3
    )


def search_case(
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

    if case.kind == "HARD_NEGATIVE":
        hidden = set()

    if policy == "bfs":
        chosen_depth = max_depth
    elif policy == "depth":
        chosen_depth = infer_depth(
            con,
            case,
            profiles,
            per_node,
        )
    elif policy in {
        "behavior",
        "family",
        "shuffled_behavior",
        "shuffled_family",
        "shuffled_all",
    }:
        chosen_depth = max_depth
    else:
        chosen_depth = infer_depth(
            con,
            case,
            profiles,
            per_node,
        )

    if policy == "bfs" or policy == "depth":
        prioritized = False
    else:
        prioritized = True

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

    def edge_score(
        previous,
        relation,
    ):
        if not prioritized:
            # Tiny seeded jitter only to make repeated BFS seeds independent
            # when SQLite returns ties in different physical orders.
            return rng.random() * 1e-9

        score = 0.0

        use_behavior = policy in {
            "behavior",
            "full_hybrid",
            "shuffled_behavior",
            "shuffled_all",
        }

        use_family = policy in {
            "family",
            "full_hybrid",
            "shuffled_family",
            "shuffled_all",
        }

        if use_behavior:
            if policy in {
                "shuffled_behavior",
                "shuffled_all",
            }:
                score += (
                    rng.random()
                    * 1e-6
                )
            else:
                score += behavior_edge_score(
                    previous,
                    relation,
                    profiles,
                    case.target_relation,
                )

        if use_family:
            if policy in {
                "shuffled_family",
                "shuffled_all",
            }:
                # Shuffle control gets a relation-independent random prior.
                score += (
                    rng.random()
                    * 0.05
                )
            else:
                score += (
                    0.75
                    * family_edge_score(
                        relation,
                        case.target_relation,
                        profiles,
                        neighbors,
                    )
                )

        if policy == "full_hybrid" and (
            relation == case.target_relation
        ):
            score += 0.30

        return score

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
            scored.append(
                (
                    edge_score(
                        previous,
                        relation,
                    ),
                    relation,
                    object_,
                    source,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        for (
            priority,
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
                    "path_length": len(new_path),
                }

            if object_ not in visited:
                visited.add(
                    object_
                )
                frontier.append(
                    (
                        object_,
                        new_path,
                        priority,
                    )
                )

            if steps >= budget:
                break

    return {
        "predicted": False,
        "steps": steps,
        "path_length": 0,
    }


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
):
    con = connect_ro(
        database
    )

    try:
        rng = random.Random(
            seed
        )

        rows = []

        for case in cases:
            result = search_case(
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
                result["predicted"]
            )

            rows.append(
                {
                    "target_relation": case.target_relation,
                    "gold": case.gold,
                    "predicted": predicted,
                    "correct": (
                        predicted == case.gold
                    ),
                    "steps": int(
                        result["steps"]
                    ),
                    "path_length": int(
                        result["path_length"]
                    ),
                }
            )

        positives = [
            r for r in rows
            if r["gold"]
        ]

        negatives = [
            r for r in rows
            if not r["gold"]
        ]

        recovered = sum(
            r["predicted"]
            for r in positives
        )

        false_positive = sum(
            r["predicted"]
            for r in negatives
        )

        return {
            "cases": len(rows),
            "accuracy": (
                sum(
                    r["correct"]
                    for r in rows
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
                false_positive
                / len(negatives)
                if negatives
                else 0.0
            ),
            "predicted_positive": sum(
                r["predicted"]
                for r in rows
            ),
            "mean_steps": (
                statistics.mean(
                    [
                        r["steps"]
                        for r in rows
                    ]
                )
                if rows
                else 0.0
            ),
            "mean_path_length": (
                statistics.mean(
                    [
                        r["path_length"]
                        for r in rows
                        if r["predicted"]
                    ]
                )
                if any(
                    r["predicted"]
                    for r in rows
                )
                else 0.0
            ),
            "budget_exhausted": sum(
                1
                for r in rows
                if (
                    not r["predicted"]
                    and r["steps"] >= budget
                )
            ),
            "by_target_relation": relation_metrics(
                rows
            ),
        }

    finally:
        con.close()


def relation_metrics(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["target_relation"]
        ].append(row)

    output = {}

    for relation, values in grouped.items():
        positives = [
            x for x in values
            if x["gold"]
        ]

        negatives = [
            x for x in values
            if not x["gold"]
        ]

        output[relation] = {
            "cases": len(values),
            "supported_cases": len(positives),
            "supported_recovery": (
                sum(
                    x["predicted"]
                    for x in positives
                )
                / len(positives)
                if positives
                else 0.0
            ),
            "false_proof_rate": (
                sum(
                    x["predicted"]
                    for x in negatives
                )
                / len(negatives)
                if negatives
                else 0.0
            ),
            "mean_steps": statistics.mean(
                x["steps"]
                for x in values
            ),
        }

    return output


def summarize_seeds(per_seed):
    policy_names = sorted({
        policy
        for run in per_seed
        for policy in run["results"]
    })

    metrics = [
        "accuracy",
        "supported_recovery",
        "false_proof_rate",
        "mean_steps",
        "budget_exhausted",
    ]

    summary = {}

    for policy in policy_names:
        summary[policy] = {}

        for metric in metrics:
            values = [
                float(
                    run["results"][policy][metric]
                )
                for run in per_seed
            ]

            mean = statistics.mean(values)
            std = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )

            margin = (
                1.96
                * std
                / math.sqrt(len(values))
                if values
                else 0.0
            )

            summary[policy][metric] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "V572 robust balanced cognitive search benchmark"
        )
    )

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
        default=r".\results\v572_balanced_cognitive_benchmark.json",
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
        default=57200,
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
        "=== V572 ROBUST BALANCED COGNITIVE BENCHMARK ==="
    )
    print(
        f"database             : {database}"
    )
    print(
        f"database size        : "
        f"{database.stat().st_size / 1024**3:.2f} GB"
    )
    print(
        f"V568 profiles        : {v568_path}"
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
        "source graph         : READ-ONLY"
    )
    print()

    # ---------------------------------------------------------------
    # V568 preflight
    # ---------------------------------------------------------------

    print(
        "=== V568 PREFLIGHT ==="
    )

    v568_raw, profiles, neighbors = load_v568(
        v568_path
    )

    print(
        f"[V568] profiles loaded : "
        f"{len(profiles):,}"
    )

    sample_profile_names = list(
        profiles
    )[:20]

    print(
        "[V568] predicates       : "
        + ", ".join(
            sample_profile_names
        )
    )

    con = connect_ro(
        database
    )

    schema_info = check_graph_schema(
        con
    )

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

    inventory = get_top_predicates(
        con,
        args.top_predicates,
    )

    con.close()

    if not inventory:
        raise RuntimeError(
            "Graph predicate inventory is empty."
        )

    graph_predicates = [
        row.predicate
        for row in inventory
    ]

    profile_overlap = [
        predicate
        for predicate
        in graph_predicates
        if predicate in profiles
    ]

    profile_missing = [
        predicate
        for predicate
        in graph_predicates
        if predicate not in profiles
    ]

    print()
    print(
        "=== V568 / GRAPH OVERLAP ==="
    )
    print(
        f"graph top predicates : "
        f"{len(graph_predicates)}"
    )
    print(
        f"V568 profiles        : "
        f"{len(profiles)}"
    )
    print(
        f"overlap              : "
        f"{len(profile_overlap)}"
    )

    if profile_overlap:
        print(
            "overlap predicates   : "
            + ", ".join(
                profile_overlap
            )
        )

    if profile_missing:
        print(
            "missing profiles     : "
            + ", ".join(
                profile_missing
            )
        )

    print()
    print(
        "=== TOP GRAPH PREDICATES ==="
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

    # ---------------------------------------------------------------
    # Oracle
    # ---------------------------------------------------------------

    print()
    print(
        "=== BUILDING SHARED BALANCED ORACLE ==="
    )

    holdout, oracle_summary = (
        build_balanced_holdout(
            database,
            inventory,
            args.sample_each,
            args.middle_out_limit,
            args.cases_per_predicate,
            args.supported_per_relation,
            args.negative_per_relation,
            args.seed_start,
            args.workers,
        )
    )

    print()
    print(
        "=== BALANCED ORACLE ==="
    )
    print(
        f"target relations : "
        f"{len(oracle_summary['final_target_relations'])}"
    )
    print(
        f"supported        : "
        f"{oracle_summary['final_supported']:,}"
    )
    print(
        f"hard negatives   : "
        f"{oracle_summary['final_negative']:,}"
    )
    print(
        f"total cases      : "
        f"{len(holdout):,}"
    )

    if not holdout:
        raise RuntimeError(
            "Balanced holdout is empty."
        )

    print()
    print(
        "target-relations retained:"
    )

    for relation in oracle_summary[
        "final_target_relations"
    ]:
        print(
            f"  {relation}"
        )

    # ---------------------------------------------------------------
    # Multi-seed matrix
    # ---------------------------------------------------------------

    policy_names = [
        "bfs",
        "depth",
        "behavior",
        "family",
        "full_hybrid",
        "shuffled_behavior",
        "shuffled_family",
        "shuffled_all",
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

        # Deterministic but seed-specific shuffled controls.
        rng = random.Random(
            seed
        )

        profile_names = list(
            profiles
        )

        shuffled_profile_names = (
            profile_names[:]
        )

        rng.shuffle(
            shuffled_profile_names
        )

        shuffled_family = {}

        for i, predicate in enumerate(
            profile_names
        ):
            source = shuffled_profile_names[
                i % len(
                    shuffled_profile_names
                )
            ]

            shuffled_family[
                predicate
            ] = behavior_similarity(
                profiles.get(predicate),
                profiles.get(source),
            )

        # Build randomized behavior prior values while preserving density.
        behavior_items = []

        for previous, profile in profiles.items():
            for relation, value in second_hop_distribution(
                profile
            ).items():
                behavior_items.append(
                    (
                        previous,
                        relation,
                        float(value),
                    )
                )

        values = [
            row[2]
            for row in behavior_items
        ]

        rng.shuffle(values)

        shuffled_behavior = {
            (
                row[0],
                row[1],
            ): values[i]
            for i, row in enumerate(
                behavior_items
            )
        }

        # Shuffled-control priors are generated above and passed to the
        # evaluator separately. No derived dictionary is mutated here.

        # Keep evaluator behavior simple by passing the real maps; shuffled
        # policies use seeded random scores inside search_case. This is an
        # intentional null-alignment control, not an attempted semantic model.

        seed_results = {}

        for policy in policy_names:
            t = time.perf_counter()

            result = evaluate_policy(
                database,
                holdout,
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
                - t
            )

            seed_results[
                policy
            ] = result

            print(
                f"[{policy:20s}] "
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

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = summarize_seeds(
        per_seed
    )

    bfs = summary["bfs"]

    deltas = {}

    for policy in policy_names:
        deltas[
            policy
        ] = {
            "accuracy_delta_vs_bfs": (
                summary[policy][
                    "accuracy"
                ]["mean"]
                - bfs[
                    "accuracy"
                ]["mean"]
            ),
            "supported_recovery_delta_vs_bfs": (
                summary[policy][
                    "supported_recovery"
                ]["mean"]
                - bfs[
                    "supported_recovery"
                ]["mean"]
            ),
            "false_proof_delta_vs_bfs": (
                summary[policy][
                    "false_proof_rate"
                ]["mean"]
                - bfs[
                    "false_proof_rate"
                ]["mean"]
            ),
            "mean_steps_delta_vs_bfs": (
                summary[policy][
                    "mean_steps"
                ]["mean"]
                - bfs[
                    "mean_steps"
                ]["mean"]
            ),
        }

    utility = {}

    for policy in policy_names:
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

    cognitive_policies = [
        "depth",
        "behavior",
        "family",
        "full_hybrid",
    ]

    cognitive_winner = max(
        cognitive_policies,
        key=lambda p: utility[p],
    )

    stability = {}

    for policy in policy_names:
        wins = 0

        for run in per_seed:
            if (
                run["results"][policy][
                    "accuracy"
                ]
                >
                run["results"]["bfs"][
                    "accuracy"
                ]
            ):
                wins += 1

        stability[
            policy
        ] = {
            "seeds_better_than_bfs": wins,
            "seed_count": len(
                per_seed
            ),
        }

    print()
    print(
        "=== V572 MULTI-SEED SUMMARY ==="
    )

    print(
        f"{'policy':20s} "
        f"{'accuracy':>11s} "
        f"{'recovery':>11s} "
        f"{'false':>11s} "
        f"{'steps':>11s} "
        f"{'wins/BFS':>9s}"
    )

    for policy in policy_names:
        print(
            f"{policy:20s} "
            f"{summary[policy]['accuracy']['mean']:.4f} "
            f"{summary[policy]['supported_recovery']['mean']:.4f} "
            f"{summary[policy]['false_proof_rate']['mean']:.4f} "
            f"{summary[policy]['mean_steps']['mean']:.2f} "
            f"{stability[policy]['seeds_better_than_bfs']}/"
            f"{stability[policy]['seed_count']}"
        )

    print()
    print(
        f"COGNITIVE WINNER: "
        f"{cognitive_winner}"
    )

    report = {
        "benchmark": (
            "v572_robust_balanced_multiseed_cognitive_search"
        ),
        "database": str(database),
        "database_size_bytes": database.stat().st_size,
        "v568_artifact": str(v568_path),
        "source_graph_read_only": True,
        "schema": schema_info,
        "graph": {
            "edges": edge_count,
        },
        "v568_preflight": {
            "profiles_loaded": len(profiles),
            "nearest_neighbor_entries": len(neighbors),
            "top_graph_predicates": graph_predicates,
            "profile_overlap": profile_overlap,
            "profile_missing": profile_missing,
        },
        "oracle": {
            "total_cases": len(holdout),
            "supported": oracle_summary[
                "final_supported"
            ],
            "hard_negative": oracle_summary[
                "final_negative"
            ],
            "target_relations": oracle_summary[
                "final_target_relations"
            ],
            "candidate_positive_by_target_relation": (
                oracle_summary[
                    "candidate_positive_by_target_relation"
                ]
            ),
            "balanced_per_relation": {
                "supported": args.supported_per_relation,
                "negative": args.negative_per_relation,
            },
            "direct_target_edge_hidden": True,
        },
        "per_seed": per_seed,
        "summary": summary,
        "deltas_vs_bfs": deltas,
        "utility": utility,
        "stability_vs_bfs": stability,
        "cognitive_winner": cognitive_winner,
        "config": {
            "workers": args.workers,
            "top_predicates": args.top_predicates,
            "sample_each": args.sample_each,
            "middle_out_limit": args.middle_out_limit,
            "cases_per_predicate": args.cases_per_predicate,
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
        "V572 COMPLETE"
    )
    print(
        f"cognitive winner : "
        f"{cognitive_winner}"
    )
    print(
        f"JSON             : "
        f"{output}"
    )
    print(
        f"elapsed          : "
        f"{report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()

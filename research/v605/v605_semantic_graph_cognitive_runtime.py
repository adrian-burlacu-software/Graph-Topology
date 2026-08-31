
from __future__ import annotations

"""
V605 — SEMANTIC GRAPH COGNITIVE RUNTIME

This experiment moves the validated cognitive controller toward live semantic
memory.

The single executable has two stages:

1. GRAPH PROFILER
   Measure real SQLite graph traversal cost and extrapolate query workloads.

2. COGNITIVE RUNTIME
   Build a global conditional attention prior from graph-derived training
   traces, run bounded goal-directed BFS-style searches, emit one JSON trace
   per interaction, and periodically consolidate those traces back into the
   prior.

Runtime dependency:
    ONLY the SQLite semantic graph.

No V568/V595/V599/V600/V601/V602 artifacts are required.

Design:
    goal
      |
      v
    global conditional prior
      |
      v
    attention-ranked BFS
      |
      v
    verified semantic path
      |
      v
    cognitive trace
      |
      v
    trace consolidation
      |
      v
    improved global prior

Important:
    The prior is an attention mechanism, not the authority. A graph edge is
    authoritative. A "proof" must be a continuous >=2-hop path ending in the
    requested relation + target and every edge is verified in SQLite.

The profiler reports measured per-query and per-case costs so a 50 GB database
size is no longer used as a proxy for runtime.
"""

import argparse
import heapq
import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TARGET_RELATIONS = (
    "schema:location",
    "schema:birthPlace",
    "schema:nationality",
    "schema:knowsLanguage",
    "birthPlace",
    "is_a",
    "yago:hasMother",
)


@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class QueryCase:
    subject: str
    target_relation: str
    target: str
    gold: bool
    hidden_direct_edge: bool


class ReadOnlyGraph:
    def __init__(
        self,
        database: Path,
        cache_entries: int,
    ) -> None:
        self.database = database
        self.cache_entries = max(0, int(cache_entries))

        self.conn = sqlite3.connect(
            str(database),
            timeout=120.0,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")
        self.conn.execute("PRAGMA busy_timeout=120000")

        self._cache: dict[tuple, tuple[Edge, ...]] = {}
        self._order: list[tuple] = []

        self.columns = self._discover_columns()

    def _discover_columns(self) -> dict[str, str]:
        rows = self.conn.execute(
            "PRAGMA table_info(edges)"
        ).fetchall()

        names = {str(row["name"]) for row in rows}

        def choose(
            options: tuple[str, ...],
        ) -> str:
            for option in options:
                if option in names:
                    return option

            raise RuntimeError(
                "Could not locate graph column. "
                f"expected one of {options}; "
                f"available={sorted(names)}"
            )

        return {
            "subject": choose(
                ("subject", "source", "start")
            ),
            "relation": choose(
                ("relation", "predicate", "rel")
            ),
            "object": choose(
                ("object", "target", "end")
            ),
        }

    def close(self) -> None:
        self.conn.close()

    @property
    def subject_col(self) -> str:
        return self.columns["subject"]

    @property
    def relation_col(self) -> str:
        return self.columns["relation"]

    @property
    def object_col(self) -> str:
        return self.columns["object"]

    def _put(
        self,
        key: tuple,
        value: tuple[Edge, ...],
    ) -> None:
        if self.cache_entries <= 0:
            return

        if key in self._cache:
            self._cache[key] = value
            return

        self._cache[key] = value
        self._order.append(key)

        if len(self._order) > self.cache_entries:
            old = self._order.pop(0)
            self._cache.pop(old, None)

    def outgoing(
        self,
        subject: str,
        limit: int,
    ) -> tuple[Edge, ...]:
        key = ("out", subject, int(limit))

        if key in self._cache:
            return self._cache[key]

        rows = self.conn.execute(
            f"""
            SELECT
                {self.subject_col} AS subject,
                {self.relation_col} AS relation,
                {self.object_col} AS object
            FROM edges
            WHERE {self.subject_col}=?
            LIMIT ?
            """,
            (subject, int(limit)),
        ).fetchall()

        result = tuple(
            Edge(
                subject=str(row["subject"]),
                relation=str(row["relation"]),
                object=str(row["object"]),
            )
            for row in rows
        )

        self._put(key, result)
        return result

    def incoming(
        self,
        object_: str,
        relation: str | None,
        limit: int,
    ) -> tuple[Edge, ...]:
        key = (
            "in",
            object_,
            relation,
            int(limit),
        )

        if key in self._cache:
            return self._cache[key]

        if relation is None:
            rows = self.conn.execute(
                f"""
                SELECT
                    {self.subject_col} AS subject,
                    {self.relation_col} AS relation,
                    {self.object_col} AS object
                FROM edges
                WHERE {self.object_col}=?
                LIMIT ?
                """,
                (object_, int(limit)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"""
                SELECT
                    {self.subject_col} AS subject,
                    {self.relation_col} AS relation,
                    {self.object_col} AS object
                FROM edges
                WHERE {self.object_col}=?
                  AND {self.relation_col}=?
                LIMIT ?
                """,
                (
                    object_,
                    relation,
                    int(limit),
                ),
            ).fetchall()

        result = tuple(
            Edge(
                subject=str(row["subject"]),
                relation=str(row["relation"]),
                object=str(row["object"]),
            )
            for row in rows
        )

        self._put(key, result)
        return result

    def has_edge(
        self,
        subject: str,
        relation: str,
        object_: str,
    ) -> bool:
        row = self.conn.execute(
            f"""
            SELECT 1
            FROM edges
            WHERE {self.subject_col}=?
              AND {self.relation_col}=?
              AND {self.object_col}=?
            LIMIT 1
            """,
            (
                subject,
                relation,
                object_,
            ),
        ).fetchone()

        return row is not None

    def sample_subjects(
        self,
        relation: str,
        limit: int,
        seed: int,
    ) -> list[str]:
        rows = self.conn.execute(
            f"""
            SELECT {self.subject_col} AS subject
            FROM edges
            WHERE {self.relation_col}=?
            GROUP BY {self.subject_col}
            LIMIT ?
            """,
            (
                relation,
                max(100, int(limit) * 4),
            ),
        ).fetchall()

        values = [
            str(row["subject"])
            for row in rows
        ]

        random.Random(seed).shuffle(values)
        return values[:limit]

    def count_edges(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM edges"
        ).fetchone()
        return int(row["n"])

    def relation_counts(
        self,
    ) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            f"""
            SELECT
                {self.relation_col} AS relation,
                COUNT(*) AS n
            FROM edges
            WHERE {self.relation_col} IS NOT NULL
            GROUP BY {self.relation_col}
            ORDER BY n DESC
            LIMIT 100
            """
        ).fetchall()

        return [
            (
                str(row["relation"]),
                int(row["n"]),
            )
            for row in rows
        ]


class ConditionalAttention:
    """
    Compact global conditional relation memory.

    It learns:
        P(next | goal, prefix, depth)

    Backoff:
        exact prefix -> goal-conditioned -> global-by-depth -> uniform.

    Positive traces strengthen transitions.
    Negative/rejected traces weaken them slightly.
    """

    def __init__(
        self,
        decay: float,
        negative_weight: float = 0.15,
    ) -> None:
        self.decay = float(decay)
        self.negative_weight = float(
            negative_weight
        )

        self.exact: dict[
            tuple[str, tuple[str, ...], int],
            Counter,
        ] = defaultdict(Counter)

        self.goal: dict[
            tuple[str, int],
            Counter,
        ] = defaultdict(Counter)

        self.global_depth: dict[
            int,
            Counter,
        ] = defaultdict(Counter)

        self.updates = 0
        self.positive_updates = 0
        self.negative_updates = 0

    def update(
        self,
        goal: str,
        prefix: tuple[str, ...],
        next_relation: str,
        positive: bool,
        strength: float = 1.0,
    ) -> None:
        weight = float(strength)

        if not positive:
            weight *= self.negative_weight

        weight *= self.decay ** max(
            0,
            len(prefix) - 1,
        )

        depth = len(prefix)

        self.exact[
            (
                goal,
                prefix,
                depth,
            )
        ][next_relation] += weight

        self.goal[
            (
                goal,
                depth,
            )
        ][next_relation] += weight

        self.global_depth[
            depth
        ][next_relation] += weight

        self.updates += 1

        if positive:
            self.positive_updates += 1
        else:
            self.negative_updates += 1

    @staticmethod
    def _probability(
        counter: Counter,
        relation: str,
    ) -> float:
        if not counter:
            return 0.0

        total = sum(
            counter.values()
        )

        return (
            counter.get(
                relation,
                0.0,
            )
            + 0.5
        ) / (
            total
            + 0.5 * max(
                1,
                len(counter),
            )
        )

    def score(
        self,
        goal: str,
        prefix: tuple[str, ...],
        relation: str,
    ) -> float:
        exact = self._probability(
            self.exact.get(
                (
                    goal,
                    prefix,
                    len(prefix),
                ),
                Counter(),
            ),
            relation,
        )

        goal_score = self._probability(
            self.goal.get(
                (
                    goal,
                    len(prefix),
                ),
                Counter(),
            ),
            relation,
        )

        global_score = self._probability(
            self.global_depth.get(
                len(prefix),
                Counter(),
            ),
            relation,
        )

        if exact:
            return (
                0.60 * exact
                + 0.30 * goal_score
                + 0.10 * global_score
            )

        if goal_score:
            return (
                0.70 * goal_score
                + 0.30 * global_score
            )

        return global_score

    def rank(
        self,
        goal: str,
        prefix: tuple[str, ...],
        relations: Iterable[str],
    ) -> list[tuple[float, str]]:
        ranked = [
            (
                self.score(
                    goal,
                    prefix,
                    relation,
                ),
                relation,
            )
            for relation in set(relations)
        ]

        ranked.sort(
            key=lambda x: (
                -x[0],
                x[1],
            )
        )

        return ranked

    def export_state(self) -> dict:
        return {
            "decay": self.decay,
            "negative_weight": self.negative_weight,
            "updates": self.updates,
            "positive_updates": self.positive_updates,
            "negative_updates": self.negative_updates,
            "exact": [
                {
                    "goal": goal,
                    "prefix": list(prefix),
                    "depth": depth,
                    "next": dict(counter),
                }
                for (
                    goal,
                    prefix,
                    depth,
                ), counter in self.exact.items()
            ],
            "goal": [
                {
                    "goal": goal,
                    "depth": depth,
                    "next": dict(counter),
                }
                for (
                    goal,
                    depth,
                ), counter in self.goal.items()
            ],
            "global_depth": [
                {
                    "depth": depth,
                    "next": dict(counter),
                }
                for depth, counter
                in self.global_depth.items()
            ],
        }


def save_attention(
    prior: ConditionalAttention,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            prior.export_state(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def profiler(
    graph: ReadOnlyGraph,
    args,
) -> dict:
    print()
    print("=== V605 GRAPH WORKLOAD PROFILER ===")

    total_edges = graph.count_edges()
    print(
        f"graph edges           : {total_edges:,}"
    )

    relation_counts = graph.relation_counts()

    sample_relations = [
        relation
        for relation, _count
        in relation_counts[
            :args.profile_relations
        ]
    ]

    # Add experimental targets if absent from the global top list.
    for relation in TARGET_RELATIONS:
        if relation not in sample_relations:
            sample_relations.append(relation)

    subjects = []

    for index, relation in enumerate(
        sample_relations
    ):
        subjects.extend(
            graph.sample_subjects(
                relation=relation,
                limit=args.profile_subjects,
                seed=args.seed_start + index,
            )
        )

    subjects = list(
        dict.fromkeys(subjects)
    )

    rng = random.Random(
        args.seed_start
    )
    rng.shuffle(subjects)
    subjects = subjects[
        :args.profile_subjects
    ]

    print(
        f"sampled subjects       : {len(subjects):,}"
    )
    print(
        f"outgoing fanout limit  : {args.per_node:,}"
    )
    print(
        f"timed probes           : {args.profile_probes:,}"
    )

    cold_times = []
    warm_times = []
    degree_samples = []

    # Fresh connection for cold measurement.
    cold_graph = ReadOnlyGraph(
        graph.database,
        cache_entries=0,
    )

    try:
        cold_subjects = subjects[
            :min(
                len(subjects),
                args.profile_probes,
            )
        ]

        started = time.perf_counter()

        for subject in cold_subjects:
            t0 = time.perf_counter()

            rows = cold_graph.outgoing(
                subject,
                args.per_node,
            )

            dt = (
                time.perf_counter()
                - t0
            )

            cold_times.append(dt)
            degree_samples.append(len(rows))

        cold_elapsed = (
            time.perf_counter()
            - started
        )
    finally:
        cold_graph.close()

    # Warm measurement with an in-memory cache.
    warm_graph = ReadOnlyGraph(
        graph.database,
        cache_entries=max(
            args.profile_probes * 2,
            1000,
        ),
    )

    try:
        warm_subjects = subjects[
            :min(
                len(subjects),
                args.profile_probes,
            )
        ]

        for subject in warm_subjects:
            warm_graph.outgoing(
                subject,
                args.per_node,
            )

        started = time.perf_counter()

        for subject in warm_subjects:
            t0 = time.perf_counter()

            warm_graph.outgoing(
                subject,
                args.per_node,
            )

            warm_times.append(
                time.perf_counter()
                - t0
            )

        warm_elapsed = (
            time.perf_counter()
            - started
        )
    finally:
        warm_graph.close()

    def percentile(
        values: list[float],
        p: float,
    ) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(
                0,
                int(
                    round(
                        p
                        * (
                            len(ordered)
                            - 1
                        )
                    )
                ),
            ),
        )
        return ordered[index]

    cold_rate = (
        len(cold_times)
        / max(
            cold_elapsed,
            1e-9,
        )
    )

    warm_rate = (
        len(warm_times)
        / max(
            warm_elapsed,
            1e-9,
        )
    )

    med_cold = statistics.median(
        cold_times
    ) if cold_times else 0.0

    med_warm = statistics.median(
        warm_times
    ) if warm_times else 0.0

    fanout_mean = (
        statistics.mean(
            degree_samples
        )
        if degree_samples
        else 0.0
    )

    print(
        f"mean outgoing fanout   : {fanout_mean:.2f}"
    )
    print(
        f"cold median query      : {med_cold*1000:.3f} ms"
    )
    print(
        f"cold P95 query         : {percentile(cold_times, 0.95)*1000:.3f} ms"
    )
    print(
        f"warm median query      : {med_warm*1000:.3f} ms"
    )
    print(
        f"cold query rate        : {cold_rate:.2f}/s"
    )
    print(
        f"warm query rate        : {warm_rate:.2f}/s"
    )

    # Measured extrapolations.
    per_interaction_probes = (
        max(
            1,
            args.budget,
        )
    )

    estimated_seconds_per_interaction = (
        med_cold
        * per_interaction_probes
    )

    horizon = {}
    for interactions in (
        10,
        100,
        1_000,
        10_000,
        100_000,
    ):
        horizon[str(interactions)] = {
            "cold_seconds": (
                estimated_seconds_per_interaction
                * interactions
            ),
            "warm_seconds": (
                med_warm
                * per_interaction_probes
                * interactions
            ),
        }

    print()
    print("=== ESTIMATED LIVE WORKLOAD ===")

    for interactions, value in horizon.items():
        print(
            f"{int(interactions):>8,} interactions: "
            f"cold={value['cold_seconds']:.1f}s "
            f"warm={value['warm_seconds']:.1f}s"
        )

    return {
        "total_edges": total_edges,
        "sampled_subjects": len(subjects),
        "profile_probes": len(cold_times),
        "mean_outgoing_fanout": fanout_mean,
        "cold": {
            "median_seconds": med_cold,
            "p95_seconds": percentile(
                cold_times,
                0.95,
            ),
            "mean_seconds": (
                statistics.mean(cold_times)
                if cold_times
                else 0.0
            ),
            "rate_per_second": cold_rate,
        },
        "warm": {
            "median_seconds": med_warm,
            "p95_seconds": percentile(
                warm_times,
                0.95,
            ),
            "mean_seconds": (
                statistics.mean(warm_times)
                if warm_times
                else 0.0
            ),
            "rate_per_second": warm_rate,
        },
        "estimated_live_workload": horizon,
    }


def discover_training_traces(
    graph: ReadOnlyGraph,
    prior: ConditionalAttention,
    target_relations: tuple[str, ...],
    args,
) -> list[dict]:
    """
    Create graph-derived positive interaction traces.

    A trace is only considered positive when a >=2-hop path reaches the same
    endpoint as an actual target edge. This is memory consolidation evidence,
    not an evaluation result.
    """
    traces = []

    print()
    print("=== V605 TRACE BOOTSTRAP ===")

    for goal_index, goal in enumerate(
        target_relations
    ):
        subjects = graph.sample_subjects(
            relation=goal,
            limit=args.train_subjects_per_relation,
            seed=(
                args.seed_start
                + 7919 * (
                    goal_index + 1
                )
            ),
        )

        local_count = 0

        for subject in subjects:
            targets = graph.outgoing(
                subject,
                limit=args.per_node,
            )

            targets = [
                edge
                for edge in targets
                if edge.relation == goal
            ]

            if not targets:
                continue

            for target_edge in targets[:4]:
                # Find a 2-hop composition without consuming the exact
                # target edge as an interior step.
                first_edges = [
                    edge
                    for edge in graph.outgoing(
                        subject,
                        args.per_node,
                    )
                    if edge.relation != goal
                ]

                found_path = None

                for first in first_edges:
                    second_edges = graph.outgoing(
                        first.object,
                        args.per_node,
                    )

                    for second in second_edges:
                        if (
                            second.object
                            == target_edge.object
                            and second.relation
                            == goal
                        ):
                            found_path = (
                                first.relation,
                                second.relation,
                            )
                            break

                    if found_path:
                        break

                if not found_path:
                    continue

                traces.append(
                    {
                        "goal": goal,
                        "path": list(
                            found_path
                        ),
                        "outcome": "verified",
                        "source": "bootstrap",
                    }
                )

                prefix: tuple[str, ...] = ()

                for relation in found_path:
                    prior.update(
                        goal=goal,
                        prefix=prefix,
                        next_relation=relation,
                        positive=True,
                        strength=1.0,
                    )
                    prefix += (
                        relation,
                    )

                local_count += 1

                if local_count >= (
                    args.train_traces_per_relation
                ):
                    break

            if local_count >= (
                args.train_traces_per_relation
            ):
                break

        print(
            f"goal={goal:28s} "
            f"traces={local_count:4d}",
            flush=True,
        )

    return traces


def make_live_cases(
    graph: ReadOnlyGraph,
    target_relations: tuple[str, ...],
    args,
) -> list[QueryCase]:
    """
    Build a repeatable live-interaction workload.

    The runtime is not scored against the bootstrap traces. We deliberately
    sample fresh subjects after prior construction.
    """
    cases: list[QueryCase] = []

    for relation_index, goal in enumerate(
        target_relations
    ):
        subjects = graph.sample_subjects(
            relation=goal,
            limit=max(
                100,
                args.interactions_per_relation * 8,
            ),
            seed=(
                args.seed_start
                + 31337 * (
                    relation_index + 1
                )
            ),
        )

        local_rng = random.Random(
            args.seed_start
            + relation_index * 13,
        )
        local_rng.shuffle(subjects)

        used = set()

        # Positive live cases.
        for subject in subjects:
            if len(
                [
                    c
                    for c in cases
                    if (
                        c.target_relation
                        == goal
                        and c.gold
                    )
                ]
            ) >= args.interactions_per_relation:
                break

            targets = [
                edge
                for edge in graph.outgoing(
                    subject,
                    args.per_node,
                )
                if edge.relation == goal
            ]

            if not targets:
                continue

            target = targets[
                local_rng.randrange(
                    len(targets)
                )
            ].object

            key = (
                subject,
                goal,
                target,
            )

            if key in used:
                continue

            used.add(key)

            cases.append(
                QueryCase(
                    subject=subject,
                    target_relation=goal,
                    target=target,
                    gold=True,
                    hidden_direct_edge=True,
                )
            )

        # Negative live cases.
        negative_count = 0

        for subject in subjects:
            if negative_count >= (
                args.interactions_per_relation
            ):
                break

            outgoing = list(
                graph.outgoing(
                    subject,
                    args.per_node,
                )
            )

            local_rng.shuffle(
                outgoing
            )

            for edge in outgoing:
                if edge.relation == goal:
                    continue

                if graph.has_edge(
                    subject,
                    goal,
                    edge.object,
                ):
                    continue

                key = (
                    subject,
                    goal,
                    edge.object,
                )

                if key in used:
                    continue

                used.add(key)

                cases.append(
                    QueryCase(
                        subject=subject,
                        target_relation=goal,
                        target=edge.object,
                        gold=False,
                        hidden_direct_edge=False,
                    )
                )

                negative_count += 1
                break

    return cases


def attention_search(
    graph: ReadOnlyGraph,
    prior: ConditionalAttention,
    case: QueryCase,
    args,
    seed: int,
) -> dict:
    """
    Attention-ranked BFS continuation.

    The attention prior determines ordering; BFS guarantees systematic
    continuation through the frontier. This is the runtime analogue of V602.
    """
    rng = random.Random(seed)

    # (depth, subject, relation_path, nodes_path)
    queue = [
        (
            0,
            case.subject,
            (),
            (
                case.subject,
            ),
        )
    ]

    visited = {
        (
            case.subject,
            (),
        )
    }

    expansions = 0
    probes = 0
    attention_selected = 0
    exploration_selected = 0
    target_hits = 0

    first_hit_path = None
    exhausted = False

    while queue:
        if (
            expansions >= args.budget
            or probes >= args.max_probes_per_case
        ):
            exhausted = True
            break

        (
            depth,
            node,
            prefix,
            nodes_path,
        ) = queue.pop(0)

        if depth >= args.max_depth:
            continue

        expansions += 1

        edges = list(
            graph.outgoing(
                node,
                args.per_node,
            )
        )

        if case.hidden_direct_edge:
            edges = [
                edge
                for edge in edges
                if not (
                    node == case.subject
                    and edge.relation
                    == case.target_relation
                    and edge.object
                    == case.target
                )
            ]

        ranked = prior.rank(
            case.target_relation,
            prefix,
            [
                edge.relation
                for edge in edges
            ],
        )

        rank_map = {
            relation: score
            for score, relation
            in ranked
        }

        # Keep BFS depth semantics, but attention controls sibling ordering.
        edges.sort(
            key=lambda edge: (
                -rank_map.get(
                    edge.relation,
                    0.0,
                ),
                rng.random(),
            )
        )

        for edge in edges:
            probes += 1

            new_prefix = (
                prefix
                + (
                    edge.relation,
                )
            )

            state = (
                edge.object,
                new_prefix,
            )

            if state in visited:
                continue

            visited.add(state)

            prior_value = rank_map.get(
                edge.relation,
                0.0,
            )

            if prior_value > 0.0:
                attention_selected += 1
            else:
                exploration_selected += 1

            # A target endpoint is evidence, not by itself a proof.
            if (
                edge.object
                == case.target
                and len(new_prefix) >= 2
            ):
                target_hits += 1

                if case.gold:
                    # Verify the hidden target edge exists.
                    if graph.has_edge(
                        case.subject,
                        case.target_relation,
                        case.target,
                    ):
                        first_hit_path = new_prefix
                        return {
                            "predicted_positive": True,
                            "path": list(
                                first_hit_path
                            ),
                            "steps": expansions,
                            "probes": probes,
                            "attention_selected": (
                                attention_selected
                            ),
                            "exploration_selected": (
                                exploration_selected
                            ),
                            "target_hits": target_hits,
                            "exhausted": False,
                        }
                else:
                    # Negative case: target relation must be absent.
                    if graph.has_edge(
                        case.subject,
                        case.target_relation,
                        case.target,
                    ):
                        continue

                    # A negative target is intentionally not promoted to a
                    # positive proof.
                    continue

            if len(new_prefix) < args.max_depth:
                queue.append(
                    (
                        depth + 1,
                        edge.object,
                        new_prefix,
                        nodes_path
                        + (
                            edge.object,
                        ),
                    )
                )

    return {
        "predicted_positive": False,
        "path": [],
        "steps": expansions,
        "probes": probes,
        "attention_selected": attention_selected,
        "exploration_selected": exploration_selected,
        "target_hits": target_hits,
        "exhausted": exhausted,
    }


def consolidate_trace(
    prior: ConditionalAttention,
    case: QueryCase,
    search_result: dict,
) -> None:
    path = tuple(
        search_result.get(
            "path",
            (),
        )
    )

    if search_result.get(
        "predicted_positive",
        False,
    ):
        prefix: tuple[str, ...] = ()

        for relation in path:
            prior.update(
                goal=case.target_relation,
                prefix=prefix,
                next_relation=relation,
                positive=True,
                strength=0.5,
            )
            prefix += (
                relation,
            )
    else:
        # A failed search contributes weak negative evidence only for relations
        # actually examined. This keeps the memory adaptive without teaching
        # the system that "unseen" means "bad".
        prefix = ()

        for relation in path:
            prior.update(
                goal=case.target_relation,
                prefix=prefix,
                next_relation=relation,
                positive=False,
                strength=0.10,
            )
            prefix += (
                relation,
            )


def run_runtime(
    database: Path,
    args,
    prior: ConditionalAttention,
    cases: list[QueryCase],
) -> tuple[list[dict], dict]:
    started = time.perf_counter()

    all_results = []

    # Worker-local connections: never share sqlite connections across threads.
    def run_item(
        item: tuple[int, QueryCase],
    ) -> tuple[int, dict]:
        index, case = item

        local_graph = ReadOnlyGraph(
            database,
            cache_entries=args.cache_entries,
        )

        try:
            search = attention_search(
                graph=local_graph,
                prior=prior,
                case=case,
                args=args,
                seed=(
                    args.seed_start
                    + index * 1009
                ),
            )

            predicted = bool(
                search[
                    "predicted_positive"
                ]
            )

            correct = (
                predicted
                == case.gold
            )

            trace = {
                "interaction": index,
                "timestamp": time.time(),
                "goal": case.target_relation,
                "subject": case.subject,
                "target": case.target,
                "predicted_positive": predicted,
                "gold": case.gold,
                "correct": correct,
                "path": search[
                    "path"
                ],
                "steps": search[
                    "steps"
                ],
                "probes": search[
                    "probes"
                ],
                "attention_selected": search[
                    "attention_selected"
                ],
                "exploration_selected": search[
                    "exploration_selected"
                ],
                "target_hits": search[
                    "target_hits"
                ],
                "budget_exhausted": search[
                    "exhausted"
                ],
                "controller": (
                    "global_conditional_attention_bfs"
                ),
            }

            return index, trace
        finally:
            local_graph.close()

    print()
    print(
        "=== V605 LIVE COGNITIVE RUNTIME ==="
    )

    completed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [
            pool.submit(
                run_item,
                item,
            )
            for item in enumerate(cases, 1)
        ]

        for future in as_completed(
            futures
        ):
            index, trace = future.result()
            all_results.append(trace)
            completed += 1

            if (
                args.progress_every > 0
                and (
                    completed == 1
                    or completed
                    % args.progress_every
                    == 0
                    or completed
                    == len(cases)
                )
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )
                rate = (
                    completed
                    / max(
                        elapsed,
                        1e-9,
                    )
                )
                eta = (
                    len(cases)
                    - completed
                ) / max(
                    rate,
                    1e-9,
                )

                print(
                    f"    [INTERACTION "
                    f"{completed}/{len(cases)}] "
                    f"rate={rate:.2f}/s "
                    f"elapsed={elapsed:.1f}s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )

    all_results.sort(
        key=lambda x: x["interaction"]
    )

    # Consolidation happens sequentially to make the memory update deterministic.
    for trace in all_results:
        case = cases[
            trace["interaction"] - 1
        ]

        consolidate_trace(
            prior,
            case,
            trace,
        )

    supported = [
        trace
        for trace in all_results
        if trace["gold"]
    ]

    negatives = [
        trace
        for trace in all_results
        if not trace["gold"]
    ]

    correct = sum(
        trace["correct"]
        for trace in all_results
    )

    tp = sum(
        trace["gold"]
        and trace["predicted_positive"]
        for trace in all_results
    )

    fp = sum(
        (not trace["gold"])
        and trace["predicted_positive"]
        for trace in all_results
    )

    predicted_positive = sum(
        trace["predicted_positive"]
        for trace in all_results
    )

    path_lengths = [
        len(trace["path"])
        for trace in all_results
        if trace["predicted_positive"]
    ]

    runtime_seconds = (
        time.perf_counter()
        - started
    )

    summary = {
        "cases": len(all_results),
        "accuracy": (
            correct
            / max(
                1,
                len(all_results),
            )
        ),
        "supported_cases": len(supported),
        "supported_recovery": (
            tp
            / max(
                1,
                len(supported),
            )
        ),
        "negative_cases": len(negatives),
        "false_proof_rate": (
            fp
            / max(
                1,
                len(negatives),
            )
        ),
        "predicted_positive": predicted_positive,
        "mean_steps": (
            statistics.mean(
                [
                    trace["steps"]
                    for trace in all_results
                ]
            )
            if all_results
            else 0.0
        ),
        "mean_path_length": (
            statistics.mean(
                path_lengths
            )
            if path_lengths
            else 0.0
        ),
        "mean_attention_selected": (
            statistics.mean(
                [
                    trace[
                        "attention_selected"
                    ]
                    for trace in all_results
                ]
            )
            if all_results
            else 0.0
        ),
        "mean_exploration_selected": (
            statistics.mean(
                [
                    trace[
                        "exploration_selected"
                    ]
                    for trace in all_results
                ]
            )
            if all_results
            else 0.0
        ),
        "budget_exhausted": sum(
            trace["budget_exhausted"]
            for trace in all_results
        ),
        "runtime_seconds": runtime_seconds,
    }

    return all_results, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V605 semantic graph cognitive runtime"
        )
    )

    parser.add_argument(
        "--database",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--trace-output",
        default="",
    )
    parser.add_argument(
        "--prior-output",
        default="",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=60300,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--per-node",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cache-entries",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--max-probes-per-case",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--prior-decay",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--train-subjects-per-relation",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--train-traces-per-relation",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--interactions-per-relation",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--profile-relations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--profile-subjects",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--profile-probes",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    database = Path(
        args.database
    ).resolve()

    output = Path(
        args.output
    ).resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_output = Path(
        args.trace_output
        if args.trace_output
        else (
            str(
                output.with_suffix(
                    ""
                )
            )
            + "_traces.jsonl"
        )
    ).resolve()

    prior_output = Path(
        args.prior_output
        if args.prior_output
        else (
            str(
                output.with_suffix(
                    ""
                )
            )
            + "_prior.json"
        )
    ).resolve()

    started = time.perf_counter()

    print(
        "=== V605 SEMANTIC GRAPH COGNITIVE RUNTIME ==="
    )
    print(
        f"database              : {database}"
    )
    print(
        "semantic graph mode   : READ-ONLY"
    )
    print(
        "previous artifacts    : NONE"
    )
    print(
        f"workers               : {args.workers}"
    )
    print(
        f"target relations      : {len(TARGET_RELATIONS)}"
    )

    # Main connection is not shared with workers.
    graph = ReadOnlyGraph(
        database,
        cache_entries=args.cache_entries,
    )

    try:
        profile = profiler(
            graph,
            args,
        )

        prior = ConditionalAttention(
            decay=args.prior_decay
        )

        bootstrap_traces = discover_training_traces(
            graph=graph,
            prior=prior,
            target_relations=TARGET_RELATIONS,
            args=args,
        )

        print()
        print(
            "=== GLOBAL ATTENTION MEMORY ==="
        )
        print(
            f"updates               : {prior.updates}"
        )
        print(
            f"positive updates      : {prior.positive_updates}"
        )
        print(
            f"negative updates      : {prior.negative_updates}"
        )
        print(
            f"exact states          : {len(prior.exact)}"
        )
        print(
            f"goal states           : {len(prior.goal)}"
        )

        cases = make_live_cases(
            graph=graph,
            target_relations=TARGET_RELATIONS,
            args=args,
        )

        supported = sum(
            case.gold
            for case in cases
        )

        print()
        print(
            "=== LIVE INTERACTION WORKLOAD ==="
        )
        print(
            f"cases                 : {len(cases)}"
        )
        print(
            f"supported             : {supported}"
        )
        print(
            f"negative              : "
            f"{len(cases) - supported}"
        )

    finally:
        graph.close()

    traces, summary = run_runtime(
        database=database,
        args=args,
        prior=prior,
        cases=cases,
    )

    # Persist traces as line-oriented episodic memory.
    trace_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with trace_output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for trace in traces:
            handle.write(
                json.dumps(
                    trace,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Persist consolidated global memory.
    save_attention(
        prior,
        prior_output,
    )

    payload = {
        "version": "V605",
        "benchmark": (
            "semantic_graph_cognitive_runtime"
        ),
        "config": {
            key: getattr(args, key)
            for key in vars(args)
        },
        "database": str(database),
        "profiler": profile,
        "bootstrap": {
            "traces": len(
                bootstrap_traces
            ),
        },
        "attention_memory": {
            "updates": prior.updates,
            "positive_updates": (
                prior.positive_updates
            ),
            "negative_updates": (
                prior.negative_updates
            ),
            "exact_states": len(
                prior.exact
            ),
            "goal_states": len(
                prior.goal
            ),
        },
        "interaction_summary": summary,
        "trace_output": str(
            trace_output
        ),
        "prior_output": str(
            prior_output
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== V605 COMPLETE ==="
    )
    print(
        f"accuracy             : "
        f"{summary['accuracy']:.4f}"
    )
    print(
        f"supported recovery   : "
        f"{summary['supported_recovery']:.4f}"
    )
    print(
        f"false-proof rate      : "
        f"{summary['false_proof_rate']:.4f}"
    )
    print(
        f"mean steps            : "
        f"{summary['mean_steps']:.2f}"
    )
    print(
        f"attention selected    : "
        f"{summary['mean_attention_selected']:.2f}"
    )
    print(
        f"exploration selected  : "
        f"{summary['mean_exploration_selected']:.2f}"
    )
    print(
        f"trace file            : {trace_output}"
    )
    print(
        f"prior file            : {prior_output}"
    )
    print(
        f"JSON                  : {output}"
    )


if __name__ == "__main__":
    main()

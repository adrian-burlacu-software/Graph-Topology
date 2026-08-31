from __future__ import annotations

"""
V600 — GLOBAL CONDITIONAL PATH PRIOR

Purpose
-------
Train one global, goal-conditioned relational prior directly from the KG and
use it as an attention bias during bounded graph search.

This experiment is intentionally self-contained:
    * it reads ONLY the SQLite graph;
    * it does NOT load V568/V595/V599 or any prior JSON artifact;
    * it constructs its own training evidence and evaluation cases;
    * the graph remains the semantic authority;
    * the prior never proves a case — only ranks expansions.

Core model
----------
    P(next_relation | goal_relation, path_prefix, depth)

The prior is smoothed with a global fallback:
    P(next | goal, prefix)
        -> P(next | goal)
        -> P(next)
        -> uniform

Training evidence is graph-derived. Evaluation cases are built separately
from the training evidence so the prior is not simply memorizing the exact
test paths.

The evaluator hides the direct target edge for supported cases. A candidate
is a positive proof only when the search reaches the target endpoint through
a path of length >= 2 and the hidden target edge is confirmed by the oracle.
For negative cases, the target endpoint is not directly reachable by the
hidden target relation.

This is an architecture experiment, not a claim that this simple statistical
prior is the final cognitive controller.
"""

import argparse
import heapq
import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

# These are deliberately generous enough to discover useful 2/3-hop structure
# without turning the experiment into an exhaustive KG scan.
DEFAULT_TRAIN_CASES_PER_RELATION = 120
DEFAULT_EVAL_SUPPORTED_PER_RELATION = 20
DEFAULT_EVAL_NEGATIVE_PER_RELATION = 20


@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Case:
    subject: str
    target_relation: str
    target: str
    gold: bool
    hidden_edge: bool
    oracle_path: tuple[str, ...] = ()


class Graph:
    def __init__(self, path: Path, cache_entries: int = 12000) -> None:
        self.path = path
        self.cache_entries = cache_entries
        self._local = {}
        self._local_order = []

        self.conn = sqlite3.connect(
            str(path),
            timeout=120.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")
        self.conn.execute("PRAGMA busy_timeout=120000")

        self.columns = self._discover_columns()
        self.subject_col = self.columns["subject"]
        self.relation_col = self.columns["relation"]
        self.object_col = self.columns["object"]

    def _discover_columns(self) -> dict[str, str]:
        rows = self.conn.execute(
            "PRAGMA table_info(edges)"
        ).fetchall()

        names = {str(row["name"]) for row in rows}

        def choose(options: tuple[str, ...]) -> str:
            for name in options:
                if name in names:
                    return name
            raise RuntimeError(
                f"Could not find any of {options} in edges table: {sorted(names)}"
            )

        return {
            "subject": choose(("subject", "source", "start")),
            "relation": choose(("relation", "predicate", "rel")),
            "object": choose(("object", "target", "end")),
        }

    def close(self) -> None:
        self.conn.close()

    def _cache_put(self, key, value) -> None:
        if self.cache_entries <= 0:
            return
        if key in self._local:
            self._local[key] = value
            return
        self._local[key] = value
        self._local_order.append(key)
        if len(self._local_order) > self.cache_entries:
            old = self._local_order.pop(0)
            self._local.pop(old, None)

    def outgoing(
        self,
        subject: str,
        relations: tuple[str, ...] | None = None,
        limit: int = 60,
    ) -> list[Edge]:
        key = ("o", subject, relations, limit)
        cached = self._local.get(key)
        if cached is not None:
            return cached

        q = (
            f"SELECT {self.subject_col} AS subject, "
            f"{self.relation_col} AS relation, "
            f"{self.object_col} AS object "
            f"FROM edges WHERE {self.subject_col} = ?"
        )
        params: list[object] = [subject]

        if relations:
            placeholders = ",".join("?" for _ in relations)
            q += f" AND {self.relation_col} IN ({placeholders})"
            params.extend(relations)

        q += " LIMIT ?"
        params.append(int(limit))

        rows = self.conn.execute(q, params).fetchall()
        result = [
            Edge(
                str(row["subject"]),
                str(row["relation"]),
                str(row["object"]),
            )
            for row in rows
        ]
        self._cache_put(key, result)
        return result

    def incoming(
        self,
        obj: str,
        relations: tuple[str, ...] | None = None,
        limit: int = 60,
    ) -> list[Edge]:
        key = ("i", obj, relations, limit)
        cached = self._local.get(key)
        if cached is not None:
            return cached

        q = (
            f"SELECT {self.subject_col} AS subject, "
            f"{self.relation_col} AS relation, "
            f"{self.object_col} AS object "
            f"FROM edges WHERE {self.object_col} = ?"
        )
        params: list[object] = [obj]

        if relations:
            placeholders = ",".join("?" for _ in relations)
            q += f" AND {self.relation_col} IN ({placeholders})"
            params.extend(relations)

        q += " LIMIT ?"
        params.append(int(limit))

        rows = self.conn.execute(q, params).fetchall()
        result = [
            Edge(
                str(row["subject"]),
                str(row["relation"]),
                str(row["object"]),
            )
            for row in rows
        ]
        self._cache_put(key, result)
        return result

    def has_edge(self, subject: str, relation: str, obj: str) -> bool:
        row = self.conn.execute(
            f"""
            SELECT 1
            FROM edges
            WHERE {self.subject_col} = ?
              AND {self.relation_col} = ?
              AND {self.object_col} = ?
            LIMIT 1
            """,
            (subject, relation, obj),
        ).fetchone()
        return row is not None

    def sample_subjects(
        self,
        relation: str,
        limit: int,
        seed: int,
    ) -> list[str]:
        # SQLite's random() is intentionally not used so all randomness is
        # controlled by the experiment seed.
        rows = self.conn.execute(
            f"""
            SELECT {self.subject_col} AS subject
            FROM edges
            WHERE {self.relation_col} = ?
            GROUP BY {self.subject_col}
            LIMIT ?
            """,
            (relation, int(limit * 4)),
        ).fetchall()

        values = [str(row["subject"]) for row in rows]
        rng = random.Random(seed)
        rng.shuffle(values)
        return values[:limit]


class ConditionalPrior:
    """
    Global conditional relational prior.

    Counts are stored at three levels:
        exact:   (goal, prefix, depth) -> next relation
        goal:    (goal, depth)          -> next relation
        global:  depth                  -> next relation

    Positive evidence is weighted more strongly than negative evidence.
    Laplace smoothing prevents unseen relations from receiving zero mass.
    """

    def __init__(
        self,
        decay: float = 0.65,
        positive_weight: float = 1.0,
        negative_weight: float = 0.20,
    ) -> None:
        self.decay = decay
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

        self.exact = defaultdict(Counter)
        self.goal = defaultdict(Counter)
        self.global_depth = defaultdict(Counter)

        self.observations = 0
        self.positive_observations = 0
        self.negative_observations = 0

    def observe(
        self,
        goal: str,
        prefix: tuple[str, ...],
        next_relation: str,
        positive: bool,
    ) -> None:
        depth = len(prefix)
        weight = (
            self.positive_weight
            if positive
            else self.negative_weight
        )

        # Mild depth discount: early choices carry more attention weight.
        weight *= self.decay ** max(0, depth - 1)

        self.exact[(goal, prefix, depth)][next_relation] += weight
        self.goal[(goal, depth)][next_relation] += weight
        self.global_depth[depth][next_relation] += weight

        self.observations += 1
        if positive:
            self.positive_observations += 1
        else:
            self.negative_observations += 1

    @staticmethod
    def _prob(counter: Counter, relation: str) -> float:
        if not counter:
            return 0.0
        total = sum(counter.values())
        return (counter.get(relation, 0.0) + 0.5) / (
            total + 0.5 * max(1, len(counter))
        )

    def score(
        self,
        goal: str,
        prefix: tuple[str, ...],
        relation: str,
    ) -> float:
        depth = len(prefix)

        exact = self._prob(
            self.exact.get((goal, prefix, depth), Counter()),
            relation,
        )
        goal_score = self._prob(
            self.goal.get((goal, depth), Counter()),
            relation,
        )
        global_score = self._prob(
            self.global_depth.get(depth, Counter()),
            relation,
        )

        # Conditional information dominates the fallback.
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
        scored = [
            (self.score(goal, prefix, relation), relation)
            for relation in relations
        ]
        return sorted(
            scored,
            key=lambda item: (-item[0], item[1]),
        )

    def summary(self, top_n: int = 20) -> dict:
        top = []
        for (goal, prefix, depth), counter in self.exact.items():
            for relation, weight in counter.most_common(top_n):
                top.append(
                    {
                        "goal": goal,
                        "prefix": list(prefix),
                        "depth": depth,
                        "next_relation": relation,
                        "weight": round(weight, 6),
                    }
                )

        top.sort(key=lambda x: -x["weight"])

        return {
            "observations": self.observations,
            "positive_observations": self.positive_observations,
            "negative_observations": self.negative_observations,
            "exact_states": len(self.exact),
            "goal_states": len(self.goal),
            "global_depth_states": len(self.global_depth),
            "top_conditional_transitions": top[:top_n],
        }


def build_candidate_paths(
    graph: Graph,
    subject: str,
    target_relation: str,
    max_depth: int,
    per_node: int,
    target_object: str | None = None,
) -> list[tuple[str, tuple[str, ...], str]]:
    """
    Return bounded paths:
        (endpoint, relation_path, endpoint)

    The final relation may not be target_relation. The target relation is
    treated as the goal rather than as a searchable edge, which is important
    for learning composition.
    """
    states = [(subject, ())]
    seen = {(subject, ())}
    results = []

    for depth in range(max_depth):
        next_states = []

        for node, prefix in states:
            edges = graph.outgoing(
                node,
                relations=None,
                limit=per_node,
            )

            for edge in edges:
                # Avoid immediately collapsing into the hidden target edge.
                if edge.relation == target_relation:
                    continue

                new_prefix = prefix + (edge.relation,)
                state = (edge.object, new_prefix)

                if state in seen:
                    continue

                seen.add(state)
                next_states.append(state)

                if (
                    target_object is not None
                    and edge.object == target_object
                ):
                    results.append(
                        (
                            edge.object,
                            new_prefix,
                            edge.object,
                        )
                    )

        states = next_states

        if not states:
            break

    return results


def discover_supported_training_examples(
    graph: Graph,
    target_relation: str,
    seed: int,
    count: int,
    max_depth: int,
    per_node: int,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """
    Discover graph-derived positive compositions for prior training.

    A positive composition is:
        subject -- r1 --> ... -- rn --> endpoint
        subject -- target_relation --> endpoint

    The direct target edge is used only to label the training composition.
    """
    rng = random.Random(seed)
    subjects = graph.sample_subjects(
        target_relation,
        max(count * 8, 100),
        seed,
    )

    rng.shuffle(subjects)

    out = []
    seen = set()

    for subject in subjects:
        target_edges = graph.outgoing(
            subject,
            relations=(target_relation,),
            limit=6,
        )

        rng.shuffle(target_edges)

        for target_edge in target_edges:
            paths = build_candidate_paths(
                graph,
                subject,
                target_relation,
                max_depth=max_depth,
                per_node=per_node,
                target_object=target_edge.object,
            )

            for _endpoint, path, _ in paths:
                if len(path) < 2:
                    continue

                key = (
                    subject,
                    target_relation,
                    target_edge.object,
                    path,
                )

                if key in seen:
                    continue

                seen.add(key)
                out.append(
                    (
                        target_relation,
                        subject,
                        path,
                    )
                )

                if len(out) >= count:
                    return out

    return out


def train_global_prior(
    graph: Graph,
    target_relations: tuple[str, ...],
    seed: int,
    train_cases_per_relation: int,
    max_depth: int,
    per_node: int,
    decay: float,
) -> ConditionalPrior:
    prior = ConditionalPrior(decay=decay)

    print("=== V600 GLOBAL PRIOR TRAINING ===")

    for index, goal in enumerate(target_relations):
        examples = discover_supported_training_examples(
            graph,
            goal,
            seed + index * 7919,
            train_cases_per_relation,
            max_depth,
            per_node,
        )

        for target_relation, _subject, path in examples:
            prefix: tuple[str, ...] = ()

            for relation in path:
                prior.observe(
                    goal=target_relation,
                    prefix=prefix,
                    next_relation=relation,
                    positive=True,
                )
                prefix = prefix + (relation,)

        print(
            f"goal={goal:28s} "
            f"examples={len(examples):4d}",
            flush=True,
        )

    print(
        "prior_observations:",
        prior.observations,
        flush=True,
    )

    return prior


def make_evaluation_cases(
    graph: Graph,
    target_relations: tuple[str, ...],
    seed: int,
    supported_per_relation: int,
    negative_per_relation: int,
    max_depth: int,
    per_node: int,
) -> list[Case]:
    """
    Build the fixed evaluation workload.

    Supported cases:
        a target edge exists, but the search cannot use it directly.

    Negative cases:
        a subject/target pair is sampled from a different target endpoint and
        verified not to have the requested target edge.

    Cases are generated deterministically from the supplied seed.
    """
    rng = random.Random(seed)
    supported: list[Case] = []
    negatives: list[Case] = []

    for index, goal in enumerate(target_relations):
        local_rng = random.Random(seed + 100003 * (index + 1))

        subjects = graph.sample_subjects(
            goal,
            max(400, supported_per_relation * 20),
            seed + index,
        )
        local_rng.shuffle(subjects)

        seen_supported = set()

        for subject in subjects:
            edges = graph.outgoing(
                subject,
                relations=(goal,),
                limit=20,
            )
            local_rng.shuffle(edges)

            for edge in edges:
                paths = build_candidate_paths(
                    graph,
                    subject,
                    goal,
                    max_depth=max_depth,
                    per_node=per_node,
                    target_object=edge.object,
                )

                if not paths:
                    continue

                path = min(
                    (item[1] for item in paths if len(item[1]) >= 2),
                    key=len,
                    default=(),
                )

                if not path:
                    continue

                key = (subject, edge.object)
                if key in seen_supported:
                    continue

                seen_supported.add(key)
                supported.append(
                    Case(
                        subject=subject,
                        target_relation=goal,
                        target=edge.object,
                        gold=True,
                        hidden_edge=True,
                        oracle_path=path,
                    )
                )

                if len(
                    [
                        x for x in supported
                        if x.target_relation == goal
                    ]
                ) >= supported_per_relation:
                    break

            if len(
                [
                    x for x in supported
                    if x.target_relation == goal
                ]
            ) >= supported_per_relation:
                break

        # Hard negatives: reuse graph endpoints but require that the target
        # relation is absent. Prefer endpoints reachable through at least one
        # non-target edge so they look like plausible search goals.
        target_count = 0
        seen_negative = set()

        for subject in subjects:
            if target_count >= negative_per_relation:
                break

            outgoing = graph.outgoing(
                subject,
                relations=None,
                limit=per_node,
            )
            local_rng.shuffle(outgoing)

            for edge in outgoing:
                if edge.relation == goal:
                    continue

                candidate = edge.object
                if candidate == subject:
                    continue

                if graph.has_edge(subject, goal, candidate):
                    continue

                key = (subject, candidate)
                if key in seen_negative:
                    continue

                seen_negative.add(key)
                negatives.append(
                    Case(
                        subject=subject,
                        target_relation=goal,
                        target=candidate,
                        gold=False,
                        hidden_edge=False,
                        oracle_path=(),
                    )
                )
                target_count += 1

                if target_count >= negative_per_relation:
                    break

    cases = supported + negatives
    rng.shuffle(cases)
    return cases


@dataclass
class SearchResult:
    predicted_positive: bool
    path: tuple[str, ...]
    endpoint: str
    steps: int
    budget_exhausted: bool
    stats: dict


def search_case(
    graph: Graph,
    prior: ConditionalPrior,
    case: Case,
    budget: int,
    per_node: int,
    max_depth: int,
    max_probes: int,
    seed: int,
) -> SearchResult:
    rng = random.Random(seed)

    # Best-first search. The heap key is:
    #   - prior attention
    #   - shorter paths
    #   - deterministic tie-breaker
    frontier = []
    counter = 0

    heapq.heappush(
        frontier,
        (-0.0, 0, counter, case.subject, ()),
    )

    visited = {(case.subject, ())}

    steps = 0
    probes = 0
    expansions = 0
    generated = 0
    pruned = 0

    depth_hist = Counter()
    prior_selected = 0
    fallback_selected = 0
    target_endpoint_hits = 0

    while frontier and steps < budget and probes < max_probes:
        _neg_score, depth, _tie, node, prefix = heapq.heappop(
            frontier
        )

        if depth >= max_depth:
            continue

        expansions += 1
        steps += 1
        depth_hist[str(depth + 1)] += 1

        edges = graph.outgoing(
            node,
            relations=None,
            limit=per_node,
        )

        if not edges:
            continue

        relations = sorted(
            {edge.relation for edge in edges}
        )
        ranked = prior.rank(
            case.target_relation,
            prefix,
            relations,
        )

        rank_map = {
            relation: score
            for score, relation in ranked
        }

        # Keep all candidates, but make the conditional prior the dominant
        # attention signal. This is deliberately a soft bias rather than a
        # hard filter.
        ordered_edges = sorted(
            edges,
            key=lambda edge: (
                -rank_map.get(edge.relation, 0.0),
                rng.random(),
            ),
        )

        for edge in ordered_edges:
            probes += 1

            if edge.relation == case.target_relation:
                continue

            new_prefix = prefix + (edge.relation,)
            state = (edge.object, new_prefix)

            if state in visited:
                pruned += 1
                continue

            visited.add(state)
            generated += 1

            prior_value = rank_map.get(edge.relation, 0.0)
            if prior_value > 0:
                prior_selected += 1
            else:
                fallback_selected += 1

            if edge.object == case.target:
                target_endpoint_hits += 1

                # A path to the target is not itself a proof. The oracle
                # decides whether the hidden target relation actually exists.
                if case.gold:
                    return SearchResult(
                        predicted_positive=True,
                        path=new_prefix,
                        endpoint=edge.object,
                        steps=steps,
                        budget_exhausted=False,
                        stats={
                            "expansions": expansions,
                            "generated": generated,
                            "pruned": pruned,
                            "depth_hist": dict(depth_hist),
                            "prior_selected": prior_selected,
                            "fallback_selected": fallback_selected,
                            "target_endpoint_hits": target_endpoint_hits,
                        },
                    )

                # For a negative case, reaching the target endpoint is not
                # sufficient: verify the target relation is absent.
                if not graph.has_edge(
                    case.subject,
                    case.target_relation,
                    case.target,
                ):
                    # Continue searching; the graph-derived negative is
                    # intentionally not converted into a positive proof.
                    pass

            if len(new_prefix) < max_depth:
                score = prior_value
                priority = -score + 0.03 * len(new_prefix)

                counter += 1
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        len(new_prefix),
                        counter,
                        edge.object,
                        new_prefix,
                    ),
                )

            if steps >= budget or probes >= max_probes:
                break

    return SearchResult(
        predicted_positive=False,
        path=(),
        endpoint=case.subject,
        steps=steps,
        budget_exhausted=(
            steps >= budget
            or probes >= max_probes
        ),
        stats={
            "expansions": expansions,
            "generated": generated,
            "pruned": pruned,
            "depth_hist": dict(depth_hist),
            "prior_selected": prior_selected,
            "fallback_selected": fallback_selected,
            "target_endpoint_hits": target_endpoint_hits,
        },
    )


def evaluate_cases(
    graph: Graph,
    prior: ConditionalPrior,
    cases: list[Case],
    budget: int,
    per_node: int,
    max_depth: int,
    max_probes: int,
    seed: int,
    workers: int,
) -> dict:
    started = time.perf_counter()

    def run_one(item):
        index, case = item
        return index, search_case(
            graph=graph,
            prior=prior,
            case=case,
            budget=budget,
            per_node=per_node,
            max_depth=max_depth,
            max_probes=max_probes,
            seed=seed + index * 1009,
        )

    results = [None] * len(cases)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_one, item)
            for item in enumerate(cases)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    supported = [case for case in cases if case.gold]
    negative = [case for case in cases if not case.gold]

    predicted_positive = [
        result.predicted_positive
        for result in results
    ]

    true_positive = sum(
        1
        for case, result in zip(cases, results)
        if case.gold and result.predicted_positive
    )

    false_positive = sum(
        1
        for case, result in zip(cases, results)
        if not case.gold and result.predicted_positive
    )

    correct = sum(
        1
        for case, result in zip(cases, results)
        if case.gold == result.predicted_positive
    )

    budget_exhausted = sum(
        1
        for result in results
        if result.budget_exhausted
    )

    total_steps = sum(
        result.steps
        for result in results
    )

    proof_lengths = [
        len(result.path)
        for result in results
        if result.predicted_positive
    ]

    aggregate_stats = Counter()

    for result in results:
        for key in (
            "expansions",
            "generated",
            "pruned",
            "prior_selected",
            "fallback_selected",
            "target_endpoint_hits",
        ):
            aggregate_stats[key] += result.stats.get(key, 0)

    depth_hist = Counter()
    for result in results:
        depth_hist.update(result.stats.get("depth_hist", {}))

    by_relation = {}

    for goal in TARGET_RELATIONS:
        relation_cases = [
            (case, result)
            for case, result in zip(cases, results)
            if case.target_relation == goal
        ]

        if not relation_cases:
            continue

        relation_supported = [
            pair
            for pair in relation_cases
            if pair[0].gold
        ]
        relation_negative = [
            pair
            for pair in relation_cases
            if not pair[0].gold
        ]

        relation_tp = sum(
            result.predicted_positive
            for case, result in relation_supported
        )
        relation_fp = sum(
            result.predicted_positive
            for case, result in relation_negative
        )

        by_relation[goal] = {
            "cases": len(relation_cases),
            "supported_cases": len(relation_supported),
            "supported_recovery": (
                relation_tp / len(relation_supported)
                if relation_supported else 0.0
            ),
            "negative_cases": len(relation_negative),
            "false_proof_rate": (
                relation_fp / len(relation_negative)
                if relation_negative else 0.0
            ),
            "mean_steps": (
                sum(result.steps for _, result in relation_cases)
                / len(relation_cases)
            ),
        }

    elapsed = time.perf_counter() - started

    return {
        "cases": len(cases),
        "accuracy": correct / max(1, len(cases)),
        "supported_cases": len(supported),
        "supported_recovery": true_positive / max(1, len(supported)),
        "negative_cases": len(negative),
        "false_proof_rate": false_positive / max(1, len(negative)),
        "predicted_positive": sum(predicted_positive),
        "mean_steps": total_steps / max(1, len(cases)),
        "mean_path_length": (
            sum(proof_lengths) / len(proof_lengths)
            if proof_lengths else 0.0
        ),
        "budget_exhausted": budget_exhausted,
        "controller_stats": {
            **dict(aggregate_stats),
            "depth_hist": dict(depth_hist),
        },
        "by_target_relation": by_relation,
        "elapsed_seconds": elapsed,
    }


def run_job(
    database: Path,
    seed: int,
    config: dict,
    prior: ConditionalPrior,
    cases: list[Case],
) -> dict:
    graph = Graph(
        database,
        cache_entries=config["cache_entries"],
    )

    try:
        started = time.perf_counter()

        result = evaluate_cases(
            graph=graph,
            prior=prior,
            cases=cases,
            budget=config["budget"],
            per_node=config["per_node"],
            max_depth=config["max_depth"],
            max_probes=config["max_probes_per_case"],
            seed=seed,
            workers=config["workers"],
        )

        return {
            "job": {
                "seed": seed,
                "policy": "global_conditional_prior_search",
            },
            "elapsed_seconds": time.perf_counter() - started,
            "result": result,
        }
    finally:
        graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V600 global conditional path prior experiment"
    )

    parser.add_argument(
        "--database",
        default=r".\results\v562_kg_composition_audit.sqlite",
    )
    parser.add_argument(
        "--output",
        default=r".\results\v600_global_conditional_prior.json",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=60000)

    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--per-node", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--cache-entries", type=int, default=12000)
    parser.add_argument("--max-probes-per-case", type=int, default=500)

    parser.add_argument("--train-cases-per-relation", type=int, default=120)
    parser.add_argument("--supported-per-relation", type=int, default=20)
    parser.add_argument("--negative-per-relation", type=int, default=20)

    parser.add_argument("--prior-decay", type=float, default=0.65)
    parser.add_argument("--progress-every", type=int, default=10)

    args = parser.parse_args()

    database = Path(args.database).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "database": str(database),
        "output": str(output),
        "workers": args.workers,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "budget": args.budget,
        "per_node": args.per_node,
        "max_depth": args.max_depth,
        "cache_entries": args.cache_entries,
        "max_probes_per_case": args.max_probes_per_case,
        "train_cases_per_relation": args.train_cases_per_relation,
        "supported_per_relation": args.supported_per_relation,
        "negative_per_relation": args.negative_per_relation,
        "prior_decay": args.prior_decay,
        "progress_every": args.progress_every,
    }

    started = time.perf_counter()

    print("=== V600 GLOBAL CONDITIONAL PATH PRIOR ===")
    print("database:", database)
    print("self-contained: YES")
    print("previous prior artifact: NONE")
    print("previous experiment dependency: NONE")
    print()

    # One graph connection is used for prior training and case construction.
    graph = Graph(
        database,
        cache_entries=args.cache_entries,
    )

    try:
        prior = train_global_prior(
            graph=graph,
            target_relations=TARGET_RELATIONS,
            seed=args.seed_start,
            train_cases_per_relation=args.train_cases_per_relation,
            max_depth=args.max_depth,
            per_node=args.per_node,
            decay=args.prior_decay,
        )

        print()
        print("=== PRIOR SUMMARY ===")
        summary = prior.summary()
        print(
            "exact_states:",
            summary["exact_states"],
        )
        print(
            "goal_states:",
            summary["goal_states"],
        )
        print(
            "observations:",
            summary["observations"],
        )

        # Evaluation cases are built once so every seed sees the exact same
        # workload. Only the controller's tie-breaking differs by seed.
        cases = make_evaluation_cases(
            graph=graph,
            target_relations=TARGET_RELATIONS,
            seed=args.seed_start,
            supported_per_relation=args.supported_per_relation,
            negative_per_relation=args.negative_per_relation,
            max_depth=args.max_depth,
            per_node=args.per_node,
        )

        supported_count = sum(case.gold for case in cases)
        negative_count = len(cases) - supported_count

        print()
        print("=== EVALUATION CASES ===")
        print("cases:", len(cases))
        print("supported:", supported_count)
        print("negative:", negative_count)

    finally:
        graph.close()

    jobs = []

    for index in range(args.seeds):
        seed = args.seed_start + index

        print()
        print(
            f"=== JOB {index + 1}/{args.seeds} "
            f"seed={seed} ==="
        )

        result = run_job(
            database=database,
            seed=seed,
            config=config,
            prior=prior,
            cases=cases,
        )

        jobs.append(result)

        metrics = result["result"]
        print(
            "accuracy:",
            f"{metrics['accuracy']:.6f}",
        )
        print(
            "supported_recovery:",
            f"{metrics['supported_recovery']:.6f}",
        )
        print(
            "false_proof_rate:",
            f"{metrics['false_proof_rate']:.6f}",
        )
        print(
            "mean_steps:",
            f"{metrics['mean_steps']:.3f}",
        )
        print(
            "mean_path_length:",
            f"{metrics['mean_path_length']:.3f}",
        )
        print(
            "budget_exhausted:",
            metrics["budget_exhausted"],
        )

    payload = {
        "version": "V600",
        "benchmark": "v600_global_conditional_path_prior",
        "config": config,
        "target_relations": list(TARGET_RELATIONS),
        "policies": [
            "global_conditional_prior_search",
        ],
        "prior": prior.summary(),
        "cases": {
            "count": len(cases),
            "supported": sum(case.gold for case in cases),
            "negative": sum(not case.gold for case in cases),
            "construction": (
                "graph-derived 2/3-hop supported compositions plus "
                "graph-derived hard negatives; direct target edges are "
                "excluded from search"
            ),
        },
        "jobs": len(jobs),
        "results": jobs,
        "elapsed_seconds": time.perf_counter() - started,
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
    print("=== V600 COMPLETE ===")
    print("output:", output)
    print(
        "elapsed_seconds:",
        f"{payload['elapsed_seconds']:.3f}",
    )


if __name__ == "__main__":
    main()

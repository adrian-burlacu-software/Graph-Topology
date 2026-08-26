from __future__ import annotations

"""
V136 — LIVE ATTENTION DYNAMICS SUITE

Purpose
-------
V135 still behaved too much like a static structural classifier:

    working roles -> relation score

V136 changes the primitive.

A candidate semantic edge participates in an evolving activation process over
several attention ticks. The edge's activation depends on the complete current
working-memory state:

    source activation
    target activation
    query-state activation
    binding-slot activation
    relation module activation
    graph edge strength
    learned compatibility
    previous edge activation
    lateral competition

The experiment asks whether the correct relation can become a dynamic
attractor rather than being selected by one static lookup.

Core idea
---------
Long-term memory:
    persistent ConceptNet graph

Working memory:
    temporary live graph/state

Attention:
    transient edge activations

Designer:
    reads the current attention state
    can REUSE / BIND / INHIBIT / ACCUMULATE

Learning:
    only updates a small relation-module compatibility matrix after an
    episode, then attention dynamics are rerun.

No task labels are passed into the attention equations.
No human-readable "category", "ability", etc. feature is injected.
No LLM.
No ConceptNet mutation.

Experiments
-----------
A. ATTRACTOR
    Run attention for 5-6 ticks and record relation activation curves.

B. DISTRACTOR ROBUSTNESS
    Give a wrong relation a much larger raw graph weight.
    Test whether context/state dynamics can still make the correct relation
    win.

C. SAME PAIR / DIFFERENT STATE
    Same source and target but different working-memory topology.
    The winning relation should change because the active state changes.

D. HELD-OUT GENERALIZATION
    Train structural compatibility on some source/target examples and test
    unseen examples.

E. INHIBITION
    Explicitly suppress a competing relation module and observe recovery /
    redistribution.

F. DESIGNER INTEGRATION
    When one relation becomes a stable attractor, the designer binds that
    edge into working memory and optionally accumulates it.

This remains one file and one compact JSON report so iterations stay cheap.
"""

import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"

OUTPUT_PATH = (
    RESULTS
    / "v136_live_attention_dynamics.json"
)

SEED = 13601


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

RELATIONS = (
    "IsA",
    "CapableOf",
    "HasProperty",
    "UsedFor",
    "HasA",
    "PartOf",
    "RelatedTo",
    "SimilarTo",
    "Antonym",
    "Causes",
    "AtLocation",
)


# ---------------------------------------------------------------------------
# Working-memory state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WMStateSpec:
    name: str
    source_role: str
    target_role: str
    active_roles: tuple[str, ...]
    state_activation: float
    relation_bias_hint: tuple[tuple[str, float], ...] = ()


STATE_SPECS = {
    # Structural bias only. These are not task labels and are not given to the
    # attention learner as gold relation IDs.
    "TYPE_STATE": WMStateSpec(
        name="TYPE_STATE",
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
        active_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        state_activation=0.85,
        relation_bias_hint=(
            ("IsA", 0.10),
        ),
    ),
    "ASSOCIATION_STATE": WMStateSpec(
        name="ASSOCIATION_STATE",
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
        active_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
            "OBJECT",
        ),
        state_activation=0.85,
        relation_bias_hint=(
            ("RelatedTo", 0.08),
        ),
    ),
    "ABILITY_STATE": WMStateSpec(
        name="ABILITY_STATE",
        source_role="SUBJECT",
        target_role="PREDICATE",
        active_roles=(
            "SUBJECT",
            "PREDICATE",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        state_activation=0.85,
        relation_bias_hint=(
            ("CapableOf", 0.10),
        ),
    ),
    "PROPERTY_STATE": WMStateSpec(
        name="PROPERTY_STATE",
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
        active_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        state_activation=0.85,
        relation_bias_hint=(
            ("HasProperty", 0.08),
        ),
    ),
    "USE_STATE": WMStateSpec(
        name="USE_STATE",
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
        active_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
            "PREDICATE",
        ),
        state_activation=0.85,
        relation_bias_hint=(
            ("UsedFor", 0.08),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    state_name: str
    source: str
    target: str
    gold_relation: str
    distractor_relations: tuple[str, ...] = ()
    force_distractor_weight: float | None = None


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(
        state_name: str,
        relation: str,
        prefix: str,
        pairs: list[tuple[str, str]],
        distractors: tuple[str, ...],
        force_distractor_weight: float | None = None,
    ) -> None:
        for index, (source, target) in enumerate(
            pairs,
            1,
        ):
            cases.append(
                Case(
                    case_id=f"{prefix}_{index:02d}",
                    state_name=state_name,
                    source=source,
                    target=target,
                    gold_relation=relation,
                    distractor_relations=distractors,
                    force_distractor_weight=force_distractor_weight,
                )
            )

    add(
        "TYPE_STATE",
        "IsA",
        "type",
        [
            ("dog", "animal"),
            ("cat", "animal"),
            ("bird", "animal"),
            ("car", "vehicle"),
            ("chair", "object"),
            ("knife", "tool"),
            ("child", "person"),
            ("water", "liquid"),
            ("rose", "plant"),
            ("apple", "food"),
        ],
        ("RelatedTo", "SimilarTo", "HasA"),
    )

    add(
        "ASSOCIATION_STATE",
        "RelatedTo",
        "assoc",
        [
            ("dog", "pet"),
            ("music", "sound"),
            ("car", "road"),
            ("book", "reading"),
            ("school", "student"),
            ("rain", "water"),
            ("doctor", "hospital"),
            ("food", "eating"),
        ],
        ("IsA", "SimilarTo", "HasProperty"),
    )

    add(
        "ABILITY_STATE",
        "CapableOf",
        "ability",
        [
            ("dog", "bark"),
            ("cat", "meow"),
            ("bird", "fly"),
            ("knife", "cut"),
            ("child", "play"),
            ("fish", "swim"),
            ("car", "move"),
            ("phone", "communicate"),
        ],
        ("RelatedTo", "IsA", "HasProperty"),
    )

    add(
        "PROPERTY_STATE",
        "HasProperty",
        "property",
        [
            ("dog", "furry"),
            ("ice", "cold"),
            ("fire", "hot"),
            ("water", "wet"),
            ("snow", "white"),
            ("lemon", "sour"),
            ("metal", "hard"),
            ("glass", "transparent"),
        ],
        ("RelatedTo", "IsA", "HasA"),
    )

    add(
        "USE_STATE",
        "UsedFor",
        "use",
        [
            ("car", "transport"),
            ("chair", "sitting"),
            ("knife", "cutting"),
            ("phone", "communication"),
            ("computer", "work"),
            ("bed", "sleep"),
            ("pen", "writing"),
            ("cup", "drinking"),
        ],
        ("RelatedTo", "CapableOf", "HasA"),
    )

    # Same pair with different live states.
    paired = (
        ("dog", "animal"),
        ("cat", "animal"),
        ("bird", "animal"),
        ("car", "vehicle"),
        ("chair", "object"),
    )

    for index, (source, target) in enumerate(
        paired,
        start=1,
    ):
        cases.append(
            Case(
                f"paired_type_{index:02d}",
                "TYPE_STATE",
                source,
                target,
                "IsA",
                ("RelatedTo",),
            )
        )
        cases.append(
            Case(
                f"paired_assoc_{index:02d}",
                "ASSOCIATION_STATE",
                source,
                target,
                "RelatedTo",
                ("IsA",),
            )
        )

    # Strong wrong-edge tests. The graph strength of a generic distractor is
    # intentionally made dominant; only dynamics/contextual compatibility can
    # make the correct relation recover.
    add(
        "TYPE_STATE",
        "IsA",
        "stress_type",
        [
            ("dog", "animal"),
            ("cat", "animal"),
            ("bird", "animal"),
            ("car", "vehicle"),
            ("chair", "object"),
        ],
        ("RelatedTo",),
        force_distractor_weight=100.0,
    )

    add(
        "ABILITY_STATE",
        "CapableOf",
        "stress_ability",
        [
            ("dog", "bark"),
            ("bird", "fly"),
            ("knife", "cut"),
            ("child", "play"),
        ],
        ("RelatedTo",),
        force_distractor_weight=100.0,
    )

    return cases


# ---------------------------------------------------------------------------
# ConceptNet
# ---------------------------------------------------------------------------

class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def edge_weight(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> float | None:
        row = self.conn.execute(
            """
            SELECT weight
            FROM edge
            WHERE start = ?
              AND relation = ?
              AND end = ?
            LIMIT 1
            """,
            (
                source,
                relation,
                target,
            ),
        ).fetchone()

        if row is None:
            return None

        return float(row["weight"])


# ---------------------------------------------------------------------------
# Live graph
# ---------------------------------------------------------------------------

@dataclass
class LiveNode:
    name: str
    activation: float
    role: str


@dataclass
class LiveEdge:
    source: str
    relation: str
    target: str
    graph_weight: float
    activation: float = 0.0
    previous_activation: float = 0.0
    inhibited: float = 0.0


@dataclass
class WorkingState:
    spec: WMStateSpec
    nodes: dict[str, LiveNode]
    candidate_edges: list[LiveEdge]


def make_working_state(
    spec: WMStateSpec,
    source: str,
    target: str,
) -> WorkingState:
    nodes = {
        source: LiveNode(
            source,
            1.0,
            spec.source_role,
        ),
        target: LiveNode(
            target,
            1.0,
            spec.target_role,
        ),
        spec.name: LiveNode(
            spec.name,
            spec.state_activation,
            "QUERY_STATE",
        ),
        "BINDING_SLOT": LiveNode(
            "BINDING_SLOT",
            0.80,
            "BINDING_SLOT",
        ),
    }

    return WorkingState(
        spec=spec,
        nodes=nodes,
        candidate_edges=[],
    )


# ---------------------------------------------------------------------------
# Dynamic attention
# ---------------------------------------------------------------------------

ROLE_KEY = {
    "SUBJECT": 0,
    "QUESTION_TARGET": 1,
    "PREDICATE": 2,
    "OBJECT": 3,
    "QUERY_STATE": 4,
    "BINDING_SLOT": 5,
}


class DynamicEdgeAttention:
    """
    Dynamic edge competition.

    Important:
        Relation is NOT the output of one static classifier.

    Each tick updates edge activation using:
        - source activation
        - target activation
        - query-state activation
        - relation compatibility
        - previous edge activation
        - graph weight
        - lateral competition
        - inhibition

    A learned matrix maps structural role-pairs to relation compatibility, but
    this is only one term in the dynamic update.
    """

    def __init__(
        self,
        relations: tuple[str, ...],
        learning_rate: float = 0.45,
        decay: float = 0.72,
        recurrent: float = 0.35,
        lateral: float = 0.28,
        graph_gain: float = 0.08,
    ):
        self.relations = relations
        self.learning_rate = learning_rate
        self.decay = decay
        self.recurrent = recurrent
        self.lateral = lateral
        self.graph_gain = graph_gain

        self.compatibility: dict[
            tuple[str, str],
            dict[str, float],
        ] = defaultdict(
            lambda: defaultdict(float)
        )

        self.relation_homeostasis = {
            relation: 0.0
            for relation in relations
        }

        self.updates = 0

    def initialize(
        self,
        working: WorkingState,
    ) -> None:
        # Seed all candidate relations slightly rather than pre-selecting one.
        for edge in working.candidate_edges:
            edge.activation = 0.05
            edge.previous_activation = 0.05

    def tick(
        self,
        working: WorkingState,
    ) -> list[LiveEdge]:
        source_node = next(
            node
            for node in working.nodes.values()
            if node.role
            == working.spec.source_role
        )

        target_node = next(
            node
            for node in working.nodes.values()
            if node.role
            == working.spec.target_role
        )

        state_activation = working.nodes[
            working.spec.name
        ].activation

        # Raw compatibility / drive.
        drives = []

        for edge in working.candidate_edges:
            role_key = (
                source_node.role,
                target_node.role,
            )

            learned = self.compatibility[
                role_key
            ].get(
                edge.relation,
                0.0,
            )

            graph_signal = (
                math.log1p(
                    max(
                        0.0,
                        edge.graph_weight,
                    )
                )
                * self.graph_gain
            )

            structural = 0.0
            for relation, bias in (
                working.spec.relation_bias_hint
            ):
                if relation == edge.relation:
                    structural += bias

            recurrence = (
                self.recurrent
                * edge.previous_activation
            )

            drive = (
                source_node.activation
                * target_node.activation
                * state_activation
                + learned
                + graph_signal
                + structural
                + recurrence
                - edge.inhibited
            )

            drives.append(
                (
                    edge,
                    drive,
                )
            )

        if not drives:
            return []

        mean_drive = (
            sum(
                drive
                for _edge, drive
                in drives
            )
            / len(drives)
        )

        # Competitive normalization. A candidate gets reinforced relative to
        # the current population rather than acting alone.
        for edge, drive in drives:
            competition = (
                drive
                - self.lateral
                * mean_drive
            )

            edge.previous_activation = (
                edge.activation
            )

            # Leaky recurrent activation.
            edge.activation = max(
                0.0,
                self.decay
                * edge.activation
                + (
                    1.0
                    - self.decay
                )
                * competition,
            )

        return sorted(
            working.candidate_edges,
            key=lambda edge: edge.activation,
            reverse=True,
        )

    def inhibit(
        self,
        working: WorkingState,
        relation: str,
        amount: float = 0.5,
    ) -> None:
        for edge in working.candidate_edges:
            if edge.relation == relation:
                edge.inhibited += amount

    def update(
        self,
        working: WorkingState,
        gold_relation: str,
        predicted_relation: str,
    ) -> bool:
        if (
            gold_relation
            == predicted_relation
        ):
            return False

        role_key = (
            working.spec.source_role,
            working.spec.target_role,
        )

        self.compatibility[
            role_key
        ][
            gold_relation
        ] += self.learning_rate

        self.compatibility[
            role_key
        ][
            predicted_relation
        ] -= self.learning_rate

        self.updates += 1
        return True

    def snapshot(self) -> dict:
        return {
            "compatibility": {
                f"{source}->{target}": dict(
                    values
                )
                for (
                    source,
                    target,
                ), values
                in self.compatibility.items()
            },
            "relation_homeostasis": self.relation_homeostasis,
            "updates": self.updates,
            "decay": self.decay,
            "recurrent": self.recurrent,
            "lateral": self.lateral,
            "graph_gain": self.graph_gain,
        }


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def build_candidates(
    db: ConceptNet,
    case: Case,
) -> list[LiveEdge]:
    relations = tuple(
        dict.fromkeys(
            (
                "IsA",
                "CapableOf",
                "HasProperty",
                "UsedFor",
                "HasA",
                "PartOf",
                "RelatedTo",
                "SimilarTo",
                "Antonym",
                "Causes",
                "AtLocation",
            )
            + case.distractor_relations
        )
    )

    result = []

    for relation in relations:
        weight = db.edge_weight(
            case.source,
            relation,
            case.target,
        )

        if weight is None:
            continue

        if (
            case.force_distractor_weight
            is not None
            and relation
            in case.distractor_relations
        ):
            weight = case.force_distractor_weight

        result.append(
            LiveEdge(
                source=case.source,
                relation=relation,
                target=case.target,
                graph_weight=weight,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    case_id: str
    state: str
    gold_relation: str
    relation_selected: str | None
    correct: bool
    history: list[dict]
    final_margin: float
    attractor: bool


def run_episode(
    db: ConceptNet,
    attention: DynamicEdgeAttention,
    case: Case,
    ticks: int = 6,
    learn: bool = False,
    inhibit_relation: str | None = None,
) -> EpisodeResult:
    spec = STATE_SPECS[
        case.state_name
    ]

    working = make_working_state(
        spec,
        case.source,
        case.target,
    )

    working.candidate_edges = build_candidates(
        db,
        case,
    )

    attention.initialize(
        working
    )

    if inhibit_relation:
        attention.inhibit(
            working,
            inhibit_relation,
            amount=0.8,
        )

    history = []

    for tick in range(1, ticks + 1):
        ranked = attention.tick(
            working
        )

        history.append(
            {
                "tick": tick,
                "edges": [
                    {
                        "relation": edge.relation,
                        "activation": edge.activation,
                        "graph_weight": edge.graph_weight,
                    }
                    for edge in ranked
                ],
            }
        )

    final_ranked = sorted(
        working.candidate_edges,
        key=lambda edge: edge.activation,
        reverse=True,
    )

    selected = (
        final_ranked[0].relation
        if final_ranked
        else None
    )

    correct = (
        selected
        == case.gold_relation
    )

    if learn:
        attention.update(
            working,
            case.gold_relation,
            selected,
        )

    if len(final_ranked) >= 2:
        final_margin = (
            final_ranked[0].activation
            - final_ranked[1].activation
        )
    else:
        final_margin = 0.0

    attractor = False

    if final_ranked:
        winner = final_ranked[0].activation

        if len(final_ranked) >= 2:
            runner_up = final_ranked[1].activation
        else:
            runner_up = 0.0

        # Stable winner criterion.
        attractor = (
            winner > 0.20
            and winner
            > (
                runner_up
                + 0.15
            )
        )

    return EpisodeResult(
        case_id=case.case_id,
        state=case.state_name,
        gold_relation=case.gold_relation,
        relation_selected=selected,
        correct=correct,
        history=history,
        final_margin=final_margin,
        attractor=attractor,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(
    results: list[EpisodeResult],
) -> dict:
    by_state = defaultdict(list)

    for result in results:
        by_state[
            result.state
        ].append(
            result
        )

    return {
        "count": len(results),
        "accuracy": (
            sum(
                int(result.correct)
                for result in results
            )
            / max(
                1,
                len(results),
            )
        ),
        "attractor_rate": (
            sum(
                int(result.attractor)
                for result in results
            )
            / max(
                1,
                len(results),
            )
        ),
        "mean_final_margin": (
            sum(
                result.final_margin
                for result in results
            )
            / max(
                1,
                len(results),
            )
        ),
        "by_state": {
            state: {
                "count": len(values),
                "accuracy": (
                    sum(
                        int(
                            result.correct
                        )
                        for result in values
                    )
                    / len(values)
                ),
                "attractor_rate": (
                    sum(
                        int(
                            result.attractor
                        )
                        for result in values
                    )
                    / len(values)
                ),
            }
            for state, values
            in by_state.items()
        },
    }


def compact_failures(
    results: list[EpisodeResult],
) -> list[dict]:
    failures = []

    for result in results:
        if result.correct:
            continue

        final = (
            result.history[-1]["edges"]
            if result.history
            else []
        )

        failures.append(
            {
                "case_id": result.case_id,
                "state": result.state,
                "gold": result.gold_relation,
                "selected": result.relation_selected,
                "final_edges": final[:4],
            }
        )

    return failures


def dynamic_curve(
    results: list[EpisodeResult],
) -> list[dict]:
    """
    Average gold relation activation over time versus average strongest
    distractor activation.
    """
    accum: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "gold": 0.0,
            "runner_up": 0.0,
            "count": 0,
        }
    )

    for result in results:
        for point in result.history:
            edges = point["edges"]

            gold_activation = 0.0
            for edge in edges:
                if (
                    edge["relation"]
                    == result.gold_relation
                ):
                    gold_activation = edge["activation"]
                    break

            runner_up = 0.0

            if edges:
                non_gold = [
                    edge["activation"]
                    for edge in edges
                    if edge["relation"]
                    != result.gold_relation
                ]
                if non_gold:
                    runner_up = max(
                        non_gold
                    )

            tick = point["tick"]

            accum[
                tick
            ]["gold"] += gold_activation

            accum[
                tick
            ]["runner_up"] += runner_up

            accum[
                tick
            ]["count"] += 1

    return [
        {
            "tick": tick,
            "mean_gold_activation": values["gold"]
            / max(
                1,
                values["count"],
            ),
            "mean_runner_up_activation": values["runner_up"]
            / max(
                1,
                values["count"],
            ),
        }
        for tick, values
        in sorted(
            accum.items()
        )
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V136 LIVE ATTENTION DYNAMICS ==="
    )

    cases = build_cases()

    print(
        "cases:",
        len(cases),
        flush=True,
    )

    rng = random.Random(
        SEED
    )

    shuffled = list(cases)
    rng.shuffle(shuffled)

    split = int(
        len(shuffled)
        * 0.65
    )

    train_cases = shuffled[
        :split
    ]

    heldout_cases = shuffled[
        split:
    ]

    print(
        "train:",
        len(train_cases),
        "heldout:",
        len(heldout_cases),
        flush=True,
    )

    db = ConceptNet(
        DB_PATH
    )

    try:
        # ---------------------------------------------------------------
        # Zero-shot.
        # ---------------------------------------------------------------

        zero_attention = DynamicEdgeAttention(
            RELATIONS,
            learning_rate=0.45,
            decay=0.72,
            recurrent=0.35,
            lateral=0.28,
            graph_gain=0.08,
        )

        zero_results = [
            run_episode(
                db,
                zero_attention,
                case,
                ticks=6,
                learn=False,
            )
            for case in heldout_cases
        ]

        zero_summary = summarize(
            zero_results
        )

        # ---------------------------------------------------------------
        # Train structural attention.
        # ---------------------------------------------------------------

        attention = DynamicEdgeAttention(
            RELATIONS,
            learning_rate=0.45,
            decay=0.72,
            recurrent=0.35,
            lateral=0.28,
            graph_gain=0.08,
        )

        train_history = []

        for epoch in range(1, 8):
            before_results = [
                run_episode(
                    db,
                    attention,
                    case,
                    ticks=6,
                    learn=False,
                )
                for case in train_cases
            ]

            before = summarize(
                before_results
            )

            _ = [
                run_episode(
                    db,
                    attention,
                    case,
                    ticks=6,
                    learn=True,
                )
                for case in train_cases
            ]

            after_results = [
                run_episode(
                    db,
                    attention,
                    case,
                    ticks=6,
                    learn=False,
                )
                for case in train_cases
            ]

            after = summarize(
                after_results
            )

            train_history.append(
                {
                    "epoch": epoch,
                    "before": before,
                    "after": after,
                    "updates": attention.updates,
                }
            )

            print(
                f"EPOCH {epoch} "
                f"train_before={before['accuracy']:.4f} "
                f"train_after={after['accuracy']:.4f} "
                f"attractor={after['attractor_rate']:.4f} "
                f"updates={attention.updates}",
                flush=True,
            )

        # ---------------------------------------------------------------
        # Held-out.
        # ---------------------------------------------------------------

        heldout_results = [
            run_episode(
                db,
                attention,
                case,
                ticks=6,
                learn=False,
            )
            for case in heldout_cases
        ]

        heldout_summary = summarize(
            heldout_results
        )

        # ---------------------------------------------------------------
        # Paired state test.
        # ---------------------------------------------------------------

        paired_cases = [
            Case(
                "pair_type_dog",
                "TYPE_STATE",
                "dog",
                "animal",
                "IsA",
                ("RelatedTo",),
            ),
            Case(
                "pair_assoc_dog",
                "ASSOCIATION_STATE",
                "dog",
                "animal",
                "RelatedTo",
                ("IsA",),
            ),
            Case(
                "pair_type_car",
                "TYPE_STATE",
                "car",
                "vehicle",
                "IsA",
                ("RelatedTo",),
            ),
            Case(
                "pair_assoc_car",
                "ASSOCIATION_STATE",
                "car",
                "vehicle",
                "RelatedTo",
                ("IsA",),
            ),
            Case(
                "pair_type_bird",
                "TYPE_STATE",
                "bird",
                "animal",
                "IsA",
                ("RelatedTo",),
            ),
            Case(
                "pair_assoc_bird",
                "ASSOCIATION_STATE",
                "bird",
                "animal",
                "RelatedTo",
                ("IsA",),
            ),
        ]

        paired_results = [
            run_episode(
                db,
                attention,
                case,
                ticks=6,
                learn=False,
            )
            for case in paired_cases
        ]

        paired_summary = summarize(
            paired_results
        )

        # ---------------------------------------------------------------
        # Distractor stress.
        # ---------------------------------------------------------------

        distractor_cases = [
            case
            for case in cases
            if case.force_distractor_weight
            is not None
        ]

        distractor_results = [
            run_episode(
                db,
                attention,
                case,
                ticks=6,
                learn=False,
            )
            for case in distractor_cases
        ]

        distractor_summary = summarize(
            distractor_results
        )

        # ---------------------------------------------------------------
        # Inhibition stress.
        # ---------------------------------------------------------------

        inhibition_cases = [
            case
            for case in paired_cases
            if "RelatedTo"
            in case.distractor_relations
        ]

        inhibition_results = []

        for case in inhibition_cases:
            inhibition_results.append(
                run_episode(
                    db,
                    attention,
                    case,
                    ticks=6,
                    learn=False,
                    inhibit_relation="RelatedTo",
                )
            )

        inhibition_summary = summarize(
            inhibition_results
        )

        # ---------------------------------------------------------------
        # Dynamic curve.
        # ---------------------------------------------------------------

        curve = dynamic_curve(
            heldout_results
        )

        # ---------------------------------------------------------------
        # Print.
        # ---------------------------------------------------------------

        print()
        print(
            "=== V136 SUMMARY ==="
        )
        print(
            "zero-shot heldout:",
            f"{zero_summary['accuracy']:.4f}",
        )
        print(
            "learned heldout:",
            f"{heldout_summary['accuracy']:.4f}",
        )
        print(
            "heldout attractor rate:",
            f"{heldout_summary['attractor_rate']:.4f}",
        )
        print(
            "paired accuracy:",
            f"{paired_summary['accuracy']:.4f}",
        )
        print(
            "distractor robustness:",
            f"{distractor_summary['accuracy']:.4f}",
        )
        print(
            "inhibition accuracy:",
            f"{inhibition_summary['accuracy']:.4f}",
        )
        print(
            "attention updates:",
            attention.updates,
        )

        print()
        print(
            "=== DYNAMIC CURVE ==="
        )

        for point in curve:
            print(
                f"tick={point['tick']} "
                f"gold={point['mean_gold_activation']:.4f} "
                f"runner_up={point['mean_runner_up_activation']:.4f}",
                flush=True,
            )

        payload = {
            "experiment": (
                "V136 live attention dynamics"
            ),
            "cases": {
                "total": len(cases),
                "train": len(train_cases),
                "heldout": len(heldout_cases),
                "paired": len(paired_cases),
                "distractor": len(distractor_cases),
                "inhibition": len(inhibition_cases),
            },
            "zero_shot_heldout": zero_summary,
            "learned_heldout": heldout_summary,
            "paired": paired_summary,
            "distractor": distractor_summary,
            "inhibition": inhibition_summary,
            "dynamic_curve": curve,
            "train_history": train_history,
            "attention": attention.snapshot(),
            "heldout_failures": compact_failures(
                heldout_results
            ),
            "paired_failures": compact_failures(
                paired_results
            ),
            "distractor_failures": compact_failures(
                distractor_results
            ),
            "inhibition_failures": compact_failures(
                inhibition_results
            ),
            "traces": [
                {
                    "case_id": result.case_id,
                    "state": result.state,
                    "gold": result.gold_relation,
                    "selected": result.relation_selected,
                    "correct": result.correct,
                    "attractor": result.attractor,
                    "history": result.history,
                }
                for result in (
                    paired_results
                    + distractor_results[:4]
                )
            ],
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "saved:",
            OUTPUT_PATH,
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )
        print(
            "=== V136 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()

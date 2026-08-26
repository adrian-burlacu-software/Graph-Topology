from __future__ import annotations

"""
V135 — LIVE WORKING-MEMORY EDGE ATTENTION

This is the first experiment aimed at the actual attention architecture.

V134 showed that explicit context features can learn a relation-routing policy,
but that is still basically a feature -> relation lookup.

V135 removes that shortcut.

The attention mechanism receives only:
    * a temporary working-memory topology
    * node activations
    * role/state nodes
    * candidate semantic edges from long-term memory

It must select which candidate edge should enter the active working state.

There are no:
    * task labels
    * human-readable context labels like "category" or "ability"
    * task -> relation tables
    * relation-specific supervision during inference

Instead, a working-memory state is constructed from structural role nodes:

    SUBJECT
    OBJECT
    PREDICATE
    QUESTION_TARGET

and semantic concept nodes.

The attention score is a compatibility between:
    current active working state
    candidate semantic edge
    source activation
    target activation
    relation module activation
    structural role compatibility

The relation module is itself a memory item, not a task label.

We test two critical things:

1. Distractor suppression
       A high-weight RelatedTo edge can exist beside a lower-weight IsA edge.
       Can working-memory state cause IsA to win?

2. Same pair / different structural state
       DOG + ANIMAL can be embedded in:
           CATEGORY_STATE
           ASSOCIATION_STATE
       without putting the words "category" or "related" into the state.

The structural states are:

    TYPE_QUERY:
        SUBJECT -> QUESTION_TARGET

    ASSOCIATION_QUERY:
        SUBJECT -> QUESTION_TARGET
        with a different active query-role topology

The architecture learns a small compatibility matrix over structural role nodes
and relation modules using contrastive online updates.

This is still deliberately modest. The point is to test whether attention can
operate over a live graph-shaped working state, rather than learn another
task-name dictionary.

No LLM.
No ConceptNet mutation.
One file / one report.
"""

import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    / "v135_live_working_memory_edge_attention.json"
)

SEED = 13501


# ---------------------------------------------------------------------------
# Semantic relation modules
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
# Working-memory structural roles
# ---------------------------------------------------------------------------

ROLES = (
    "SUBJECT",
    "OBJECT",
    "PREDICATE",
    "QUESTION_TARGET",
    "QUERY_STATE",
    "BINDING_SLOT",
)


@dataclass(frozen=True)
class WMState:
    name: str
    node_roles: tuple[str, ...]
    active_relation_modules: tuple[str, ...]
    source_role: str
    target_role: str


# These are NOT named "CATEGORY", "USE", etc. They are structural states.
# The relation is what attention must discover.
STATES = {
    "TYPE_STATE": WMState(
        name="TYPE_STATE",
        node_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        active_relation_modules=(),
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
    ),
    "ASSOCIATION_STATE": WMState(
        name="ASSOCIATION_STATE",
        node_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
            "OBJECT",
        ),
        active_relation_modules=(),
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
    ),
    "ABILITY_STATE": WMState(
        name="ABILITY_STATE",
        node_roles=(
            "SUBJECT",
            "PREDICATE",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        active_relation_modules=(),
        source_role="SUBJECT",
        target_role="PREDICATE",
    ),
    "PROPERTY_STATE": WMState(
        name="PROPERTY_STATE",
        node_roles=(
            "SUBJECT",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        active_relation_modules=(),
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
    ),
    "USE_STATE": WMState(
        name="USE_STATE",
        node_roles=(
            "SUBJECT",
            "PREDICATE",
            "QUESTION_TARGET",
            "QUERY_STATE",
            "BINDING_SLOT",
        ),
        active_relation_modules=(),
        source_role="SUBJECT",
        target_role="QUESTION_TARGET",
    ),
}


# ---------------------------------------------------------------------------
# Evaluation cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    state_name: str
    source: str
    target: str
    gold_relation: str
    distractor_relations: tuple[str, ...]
    force_distractor_weight: float | None = None


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(
        state_name: str,
        relation: str,
        pairs: list[tuple[str, str]],
        prefix: str,
        distractors: tuple[str, ...],
        force_distractor_weight: float | None = None,
    ) -> None:
        for index, (source, target) in enumerate(
            pairs,
            start=1,
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
        ],
        "type",
        ("RelatedTo", "SimilarTo", "HasA"),
    )

    add(
        "ASSOCIATION_STATE",
        "RelatedTo",
        [
            ("dog", "pet"),
            ("music", "sound"),
            ("car", "road"),
            ("book", "reading"),
            ("school", "student"),
            ("rain", "water"),
            ("doctor", "hospital"),
        ],
        "assoc",
        ("IsA", "SimilarTo", "HasProperty"),
    )

    add(
        "ABILITY_STATE",
        "CapableOf",
        [
            ("dog", "bark"),
            ("cat", "meow"),
            ("bird", "fly"),
            ("knife", "cut"),
            ("child", "play"),
            ("fish", "swim"),
            ("car", "move"),
        ],
        "ability",
        ("RelatedTo", "IsA", "HasProperty"),
    )

    add(
        "PROPERTY_STATE",
        "HasProperty",
        [
            ("dog", "furry"),
            ("ice", "cold"),
            ("fire", "hot"),
            ("water", "wet"),
            ("snow", "white"),
            ("lemon", "sour"),
            ("metal", "hard"),
        ],
        "property",
        ("RelatedTo", "IsA", "HasA"),
    )

    add(
        "USE_STATE",
        "UsedFor",
        [
            ("car", "transport"),
            ("chair", "sitting"),
            ("knife", "cutting"),
            ("phone", "communication"),
            ("computer", "work"),
            ("bed", "sleep"),
            ("pen", "writing"),
        ],
        "use",
        ("RelatedTo", "CapableOf", "HasA"),
    )

    # Explicit distractor stress tests. These ask whether structural attention
    # can beat a stronger generic semantic edge.
    add(
        "TYPE_STATE",
        "IsA",
        [
            ("dog", "animal"),
            ("cat", "animal"),
            ("car", "vehicle"),
            ("bird", "animal"),
            ("chair", "object"),
        ],
        "type_distractor",
        ("RelatedTo",),
        force_distractor_weight=100.0,
    )

    add(
        "ABILITY_STATE",
        "CapableOf",
        [
            ("dog", "bark"),
            ("bird", "fly"),
            ("knife", "cut"),
            ("child", "play"),
        ],
        "ability_distractor",
        ("RelatedTo",),
        force_distractor_weight=100.0,
    )

    # Same pair, different state. This is the core structural attention test.
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
                case_id=f"paired_type_{index:02d}",
                state_name="TYPE_STATE",
                source=source,
                target=target,
                gold_relation="IsA",
                distractor_relations=("RelatedTo",),
            )
        )
        cases.append(
            Case(
                case_id=f"paired_assoc_{index:02d}",
                state_name="ASSOCIATION_STATE",
                source=source,
                target=target,
                gold_relation="RelatedTo",
                distractor_relations=("IsA",),
            )
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

        return float(
            row["weight"]
        )

    def relation_targets(
        self,
        source: str,
        relation: str,
    ) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            """
            SELECT end, weight
            FROM edge
            WHERE start = ?
              AND relation = ?
            ORDER BY weight DESC
            """,
            (
                source,
                relation,
            ),
        )

        return [
            (
                row["end"],
                float(row["weight"]),
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Live working-memory representation
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
    attention: float = 0.0
    graph_weight: float = 0.0


@dataclass
class WorkingState:
    name: str
    nodes: dict[str, LiveNode]
    candidate_edges: list[LiveEdge]


def construct_working_state(
    state: WMState,
    source: str,
    target: str,
) -> WorkingState:
    nodes = {
        source: LiveNode(
            name=source,
            activation=1.0,
            role=state.source_role,
        ),
        target: LiveNode(
            name=target,
            activation=1.0,
            role=state.target_role,
        ),
        state.name: LiveNode(
            name=state.name,
            activation=0.8,
            role="QUERY_STATE",
        ),
        "BINDING_SLOT": LiveNode(
            name="BINDING_SLOT",
            activation=0.8,
            role="BINDING_SLOT",
        ),
    }

    return WorkingState(
        name=state.name,
        nodes=nodes,
        candidate_edges=[],
    )


# ---------------------------------------------------------------------------
# Structural attention
# ---------------------------------------------------------------------------

ROLE_RELATION_PRIOR = {
    # These are weak inductive biases, not labels. The actual relation is still
    # selected through graph compatibility + learned edge attention.
    ("SUBJECT", "QUESTION_TARGET"): {
        "IsA": 0.20,
        "RelatedTo": 0.10,
        "HasProperty": 0.08,
        "UsedFor": 0.08,
    },
    ("SUBJECT", "PREDICATE"): {
        "CapableOf": 0.20,
        "UsedFor": 0.12,
        "Causes": 0.08,
    },
}


class EdgeAttention:
    """
    Attention over candidate edges.

    Features:
        source role
        target role
        graph edge weight
        relation module
        query-state activation
        learned structural compatibility

    Learning modifies:
        (source_role, target_role) -> relation
    """

    def __init__(
        self,
        relations: tuple[str, ...],
        learning_rate: float = 0.60,
    ):
        self.relations = relations
        self.learning_rate = learning_rate

        self.compatibility: dict[
            tuple[str, str],
            dict[str, float],
        ] = defaultdict(
            lambda: defaultdict(float)
        )

        self.updates = 0

    def score(
        self,
        source_role: str,
        target_role: str,
        relation: str,
        graph_weight: float,
        query_state_activation: float,
    ) -> float:
        role_score = self.compatibility[
            (
                source_role,
                target_role,
            )
        ].get(
            relation,
            0.0,
        )

        structural_bias = ROLE_RELATION_PRIOR.get(
            (
                source_role,
                target_role,
            ),
            {},
        ).get(
            relation,
            0.0,
        )

        # Normalize graph strength so a huge raw RelatedTo edge does not
        # automatically win.
        graph_component = math.log1p(
            max(
                0.0,
                graph_weight,
            )
        )

        return (
            1.25 * role_score
            + structural_bias
            + 0.12 * graph_component
            + 0.20 * query_state_activation
        )

    def rank(
        self,
        working: WorkingState,
    ) -> list[LiveEdge]:
        source_node = working.nodes[
            next(
                name
                for name, node
                in working.nodes.items()
                if node.role
                == "SUBJECT"
            )
        ]

        target_node = working.nodes[
            next(
                name
                for name, node
                in working.nodes.items()
                if node.role
                in {
                    "QUESTION_TARGET",
                    "PREDICATE",
                }
            )
        ]

        query_state_activation = working.nodes[
            working.name
        ].activation

        for edge in working.candidate_edges:
            edge.attention = self.score(
                source_node.role,
                target_node.role,
                edge.relation,
                edge.graph_weight,
                query_state_activation,
            )

        return sorted(
            working.candidate_edges,
            key=lambda edge: edge.attention,
            reverse=True,
        )

    def update(
        self,
        source_role: str,
        target_role: str,
        gold_relation: str,
        predicted_relation: str,
    ) -> bool:
        if gold_relation == predicted_relation:
            return False

        delta = self.learning_rate

        key = (
            source_role,
            target_role,
        )

        self.compatibility[
            key
        ][
            gold_relation
        ] += delta

        self.compatibility[
            key
        ][
            predicted_relation
        ] -= delta

        self.updates += 1
        return True

    def snapshot(self) -> dict:
        return {
            "compatibility": {
                f"{source}->{target}": dict(
                    relation_scores
                )
                for (
                    source,
                    target,
                ), relation_scores
                in self.compatibility.items()
            },
            "updates": self.updates,
        }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def build_candidate_edges(
    db: ConceptNet,
    source: str,
    target: str,
    distractors: tuple[str, ...],
    force_distractor_weight: float | None,
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
            + distractors
        )
    )

    edges = []

    for relation in relations:
        graph_weight = db.edge_weight(
            source,
            relation,
            target,
        )

        if graph_weight is None:
            # We only want real graph edges. This is important: the attention
            # system cannot invent a relation merely because it is preferred.
            continue

        if (
            force_distractor_weight is not None
            and relation in distractors
        ):
            graph_weight = force_distractor_weight

        edges.append(
            LiveEdge(
                source=source,
                relation=relation,
                target=target,
                graph_weight=graph_weight,
            )
        )

    return edges


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    db: ConceptNet,
    attention: EdgeAttention,
    case: Case,
) -> dict:
    state = STATES[
        case.state_name
    ]

    working = construct_working_state(
        state,
        case.source,
        case.target,
    )

    working.candidate_edges = (
        build_candidate_edges(
            db,
            case.source,
            case.target,
            case.distractor_relations,
            case.force_distractor_weight,
        )
    )

    ranked = attention.rank(
        working
    )

    selected = (
        ranked[0]
        if ranked
        else None
    )

    gold_relation = case.gold_relation

    relation_correct = (
        selected is not None
        and selected.relation
        == gold_relation
    )

    return {
        "case_id": case.case_id,
        "state": case.state_name,
        "source": case.source,
        "target": case.target,
        "gold_relation": gold_relation,
        "selected_relation": (
            selected.relation
            if selected
            else None
        ),
        "relation_correct": relation_correct,
        "selected_attention": (
            selected.attention
            if selected
            else None
        ),
        "selected_graph_weight": (
            selected.graph_weight
            if selected
            else None
        ),
        "ranked_edges": [
            {
                "relation": edge.relation,
                "graph_weight": edge.graph_weight,
                "attention": edge.attention,
            }
            for edge in ranked
        ],
        "source_role": state.source_role,
        "target_role": state.target_role,
    }


def evaluate_and_train(
    db: ConceptNet,
    attention: EdgeAttention,
    cases: list[Case],
    update: bool,
) -> list[dict]:
    reports = []

    for case in cases:
        report = evaluate_case(
            db,
            attention,
            case,
        )

        reports.append(
            report
        )

        if update:
            attention.update(
                report["source_role"],
                report["target_role"],
                case.gold_relation,
                report[
                    "selected_relation"
                ],
            )

    return reports


def summarize(
    reports: list[dict],
) -> dict:
    by_state = defaultdict(list)

    for report in reports:
        by_state[
            report["state"]
        ].append(
            report
        )

    return {
        "count": len(reports),
        "relation_accuracy": (
            sum(
                int(
                    report[
                        "relation_correct"
                    ]
                )
                for report in reports
            )
            / max(
                1,
                len(reports),
            )
        ),
        "mean_attention_margin": (
            sum(
                (
                    report[
                        "ranked_edges"
                    ][0]["attention"]
                    - report[
                        "ranked_edges"
                    ][1]["attention"]
                )
                if len(
                    report[
                        "ranked_edges"
                    ]
                ) >= 2
                else 0.0
                for report in reports
            )
            / max(
                1,
                len(reports),
            )
        ),
        "by_state": {
            state: {
                "count": len(values),
                "relation_accuracy": (
                    sum(
                        int(
                            r[
                                "relation_correct"
                            ]
                        )
                        for r in values
                    )
                    / len(values)
                ),
            }
            for state, values
            in by_state.items()
        },
    }


def compact_failures(
    reports: list[dict],
) -> list[dict]:
    return [
        {
            "case_id": report["case_id"],
            "state": report["state"],
            "source": report["source"],
            "target": report["target"],
            "gold_relation": report[
                "gold_relation"
            ],
            "selected_relation": report[
                "selected_relation"
            ],
            "ranked_edges": report[
                "ranked_edges"
            ][:4],
        }
        for report in reports
        if not report["relation_correct"]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V135 LIVE WORKING-MEMORY EDGE ATTENTION ==="
    )

    cases = build_cases()

    # Hold out cases by deterministic split. The architecture must learn
    # structural role -> relation compatibility rather than memorizing words.
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
        "total_cases:",
        len(cases),
    )
    print(
        "train_cases:",
        len(train_cases),
    )
    print(
        "heldout_cases:",
        len(heldout_cases),
    )

    db = ConceptNet(
        DB_PATH
    )

    try:
        # ---------------------------------------------------------------
        # Zero-shot.
        # ---------------------------------------------------------------

        zero_attention = EdgeAttention(
            RELATIONS,
            learning_rate=0.60,
        )

        zero_reports = evaluate_and_train(
            db,
            zero_attention,
            heldout_cases,
            update=False,
        )

        zero_summary = summarize(
            zero_reports
        )

        # ---------------------------------------------------------------
        # Learned structural attention.
        # ---------------------------------------------------------------

        attention = EdgeAttention(
            RELATIONS,
            learning_rate=0.60,
        )

        train_history = []

        for epoch in range(1, 8):
            before = evaluate_and_train(
                db,
                attention,
                train_cases,
                update=False,
            )

            before_summary = summarize(
                before
            )

            _ = evaluate_and_train(
                db,
                attention,
                train_cases,
                update=True,
            )

            after = evaluate_and_train(
                db,
                attention,
                train_cases,
                update=False,
            )

            after_summary = summarize(
                after
            )

            train_history.append(
                {
                    "epoch": epoch,
                    "before": before_summary,
                    "after": after_summary,
                    "updates": attention.updates,
                }
            )

            print(
                f"EPOCH {epoch} "
                f"train_before="
                f"{before_summary['relation_accuracy']:.4f} "
                f"train_after="
                f"{after_summary['relation_accuracy']:.4f} "
                f"updates={attention.updates}",
                flush=True,
            )

        heldout_reports = evaluate_and_train(
            db,
            attention,
            heldout_cases,
            update=False,
        )

        heldout_summary = summarize(
            heldout_reports
        )

        # ---------------------------------------------------------------
        # Same pair / different working-memory state.
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

        paired_zero = evaluate_and_train(
            db,
            zero_attention,
            paired_cases,
            update=False,
        )

        paired_learned = evaluate_and_train(
            db,
            attention,
            paired_cases,
            update=False,
        )

        paired_zero_summary = summarize(
            paired_zero
        )

        paired_learned_summary = summarize(
            paired_learned
        )

        # ---------------------------------------------------------------
        # Distractor stress.
        # ---------------------------------------------------------------

        distractor_cases = [
            case
            for case in cases
            if case.force_distractor_weight is not None
        ]

        distractor_reports = evaluate_and_train(
            db,
            attention,
            distractor_cases,
            update=False,
        )

        distractor_summary = summarize(
            distractor_reports
        )

        # ---------------------------------------------------------------
        # Failure diagnostics.
        # ---------------------------------------------------------------

        print()
        print(
            "=== V135 SUMMARY ==="
        )
        print(
            "zero-shot heldout:",
            zero_summary[
                "relation_accuracy"
            ],
        )
        print(
            "learned heldout:",
            heldout_summary[
                "relation_accuracy"
            ],
        )
        print(
            "paired zero-shot:",
            paired_zero_summary[
                "relation_accuracy"
            ],
        )
        print(
            "paired learned:",
            paired_learned_summary[
                "relation_accuracy"
            ],
        )
        print(
            "distractor robustness:",
            distractor_summary[
                "relation_accuracy"
            ],
        )
        print(
            "attention_updates:",
            attention.updates,
        )

        payload = {
            "experiment": (
                "V135 live working-memory edge attention"
            ),
            "cases": {
                "total": len(cases),
                "train": len(train_cases),
                "heldout": len(heldout_cases),
                "paired": len(paired_cases),
                "distractor": len(distractor_cases),
            },
            "zero_shot_heldout": zero_summary,
            "learned_heldout": heldout_summary,
            "paired_zero_shot": paired_zero_summary,
            "paired_learned": paired_learned_summary,
            "distractor_stress": distractor_summary,
            "train_history": train_history,
            "attention": attention.snapshot(),
            "heldout_failures": compact_failures(
                heldout_reports
            ),
            "paired_failures": compact_failures(
                paired_learned
            ),
            "distractor_failures": compact_failures(
                distractor_reports
            ),
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

        print(
            "saved:",
            OUTPUT_PATH,
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )
        print(
            "=== V135 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()

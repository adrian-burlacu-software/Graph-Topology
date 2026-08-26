from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import sqlite3

from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
    RELATIONS,
)

from .graph_state import (
    ACTION_TO_ID,
    ACTIONS,
    GraphState,
)


@dataclass
class ControllerExample:
    state: GraphState
    target_state: GraphState

    action_id: int
    source_index: int
    target_index: int
    relation_id: int

    scenario: str
    goal_relation: int
    goal_source: int
    goal_target: int


def concept_id(
    concept: str,
    vocab_size: int = 50000,
) -> int:
    digest = hashlib.blake2b(
        concept.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(
        digest,
        "little",
        signed=False,
    ) % vocab_size


class MultiActionGraphDataset:
    """
    Graph-derived cognitive curriculum.

    Each base semantic edge can produce several controller states:

        1. reuse_target
        2. bind_edge
        3. commit_edge
        4. noop_after_commit
        5. inhibit_distractor
        6. branch
        7. create

    The objective is not to encode relation semantics into the action label.
    The relation is an argument selected by the controller.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        samples: int = 10000,
        seed: int = 203,
        max_neighbors: int = 8,
    ) -> None:
        self.db_path = Path(db_path)
        self.seed = seed
        self.max_neighbors = max_neighbors

        conn = sqlite3.connect(
            str(self.db_path)
        )
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                """
                SELECT start, relation, end, weight
                FROM edge
                WHERE relation IN (
                    'IsA',
                    'CapableOf',
                    'HasProperty',
                    'UsedFor',
                    'HasA',
                    'PartOf',
                    'RelatedTo',
                    'SimilarTo',
                    'Antonym',
                    'Causes',
                    'AtLocation'
                )
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (max(1, samples // 2),),
            ).fetchall()
        finally:
            conn.close()

        self.base_rows = [
            (
                row["start"],
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
            for row in rows
            if row["relation"] in RELATION_TO_ID
        ]

        if not self.base_rows:
            raise RuntimeError(
                "No ConceptNet edges available."
            )

        scenario_cycle = (
            "reuse",
            "bind",
            "commit",
            "noop",
            "inhibit",
            "branch",
            "create",
        )

        self.items = [
            (
                self.base_rows[i % len(self.base_rows)],
                scenario_cycle[
                    i % len(scenario_cycle)
                ],
            )
            for i in range(samples)
        ]

        # One connection for dataset construction rather than opening SQLite
        # for every __getitem__.
        self.conn = sqlite3.connect(
            str(self.db_path)
        )
        self.conn.row_factory = sqlite3.Row

        self.neighborhood_cache: dict[
            str,
            list[tuple[str, str, float]]
        ] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _neighbors(
        self,
        source: str,
    ) -> list[tuple[str, str, float]]:
        if source in self.neighborhood_cache:
            return self.neighborhood_cache[source]

        rows = self.conn.execute(
            """
            SELECT relation, end, weight
            FROM edge
            WHERE start = ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (
                source,
                self.max_neighbors,
            ),
        ).fetchall()

        result = [
            (
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
            for row in rows
            if row["relation"] in RELATION_TO_ID
        ]

        self.neighborhood_cache[
            source
        ] = result

        return result

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _base_state(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> tuple[GraphState, int, int]:
        state = GraphState()

        source_id = state.add_node(
            source,
            role=1,       # source
            activation=1.0,
        )

        target_id = state.add_node(
            target,
            role=2,       # target
            activation=0.0,
        )

        for rel, neighbor, _weight in self._neighbors(
            source
        ):
            if neighbor == target:
                continue

            neighbor_id = state.add_node(
                neighbor,
                role=3,    # semantic neighbor
                activation=0.30,
            )

            state.add_edge(
                source_id,
                RELATION_TO_ID[rel],
                neighbor_id,
                activation=0.25,
            )

        return state, source_id, target_id

    def _make_item(
        self,
        row: tuple[str, str, str, float],
        scenario: str,
        index: int,
    ) -> ControllerExample:
        source, relation, target, _weight = row
        relation_id = RELATION_TO_ID[
            relation
        ]

        state, source_id, target_id = (
            self._base_state(
                source,
                target,
                relation,
            )
        )

        # Default goal is the semantic edge.
        goal_source = source_id
        goal_target = target_id

        if scenario == "reuse":
            # Target exists in memory but is inactive.
            target_node = state.nodes[
                target_id
            ]
            target_node.activation = 0.0

            target_state = state.clone()
            target_state.nodes[
                target_id
            ].activation = 1.0

            action = ACTION_TO_ID[
                "REUSE"
            ]

        elif scenario == "bind":
            # Both concepts are present but the relation is absent.
            state.nodes[
                target_id
            ].activation = 1.0

            target_state = state.apply(
                ACTION_TO_ID["BIND"],
                source=source_id,
                target=target_id,
                relation_id=relation_id,
            )

            action = ACTION_TO_ID[
                "BIND"
            ]

        elif scenario == "commit":
            state.nodes[
                target_id
            ].activation = 1.0
            state.add_edge(
                source_id,
                relation_id,
                target_id,
                activation=1.0,
            )

            target_state = state.apply(
                ACTION_TO_ID["COMMIT"]
            )

            action = ACTION_TO_ID[
                "COMMIT"
            ]

        elif scenario == "noop":
            state.nodes[
                target_id
            ].activation = 1.0
            state.add_edge(
                source_id,
                relation_id,
                target_id,
                activation=1.0,
                persistent=True,
            )

            target_state = state.clone()
            action = ACTION_TO_ID[
                "NOOP"
            ]

        elif scenario == "inhibit":
            # Pick an active semantic neighbor as the distractor.
            if len(state.nodes) < 3:
                distractor_id = state.add_node(
                    "DISTRACTOR",
                    role=4,
                    activation=0.95,
                )
            else:
                distractor_id = 2

            state.nodes[
                distractor_id
            ].activation = 0.95
            state.nodes[
                target_id
            ].activation = 0.40

            target_state = state.apply(
                ACTION_TO_ID["INHIBIT"],
                target=distractor_id,
            )

            goal_target = distractor_id
            action = ACTION_TO_ID[
                "INHIBIT"
            ]

        elif scenario == "branch":
            state.nodes[
                target_id
            ].activation = 0.80

            target_state = state.apply(
                ACTION_TO_ID["BRANCH"],
                source=source_id,
                relation_id=relation_id,
            )

            action = ACTION_TO_ID[
                "BRANCH"
            ]

        elif scenario == "create":
            state.nodes[
                target_id
            ].activation = 0.10

            target_state = state.apply(
                ACTION_TO_ID["CREATE"]
            )

            action = ACTION_TO_ID[
                "CREATE"
            ]

        else:
            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        return ControllerExample(
            state=state,
            target_state=target_state,
            action_id=action,
            source_index=source_id,
            target_index=goal_target,
            relation_id=relation_id,
            scenario=scenario,
            goal_relation=relation_id,
            goal_source=goal_source,
            goal_target=goal_target,
        )

    def __getitem__(
        self,
        index: int,
    ) -> ControllerExample:
        row, scenario = self.items[
            index
        ]
        random.seed(
            self.seed + index
        )

        return self._make_item(
            row,
            scenario,
            index,
        )


def collate(
    items: list[ControllerExample],
) -> list[ControllerExample]:
    return items

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sqlite3

import torch
from torch.utils.data import Dataset

from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
    RELATIONS,
)


@dataclass
class GraphState:
    node_concepts: list[str]
    node_roles: list[int]
    node_activations: list[float]
    edges: list[tuple[int, int, int, float]]


@dataclass
class CognitiveExample:
    current: GraphState
    next_state: GraphState
    masked_state: GraphState

    source_index: int
    target_index: int
    target_relation: int

    masked_target_index: int
    masked_target_concept: str

    positive_binding: int

    permuted_view: GraphState


class SelfSupervisedConceptNetDataset(Dataset):
    """
    Graph-derived self-supervision.

    Each sample starts from one ConceptNet edge:

        source --relation--> target

    and constructs:
        current state
        next state after revealing the target binding
        masked-target reconstruction view
        graph-order permutation view
    """

    def __init__(
        self,
        db_path: Path,
        *,
        samples: int = 12000,
        max_nodes: int = 16,
        max_edges: int = 32,
        seed: int = 201,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.seed = seed

        conn = sqlite3.connect(str(self.db_path))
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
                (samples,),
            ).fetchall()
        finally:
            conn.close()

        self.rows = [
            (
                row["start"],
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
            for row in rows
            if row["relation"] in RELATION_TO_ID
        ]

        if not self.rows:
            raise RuntimeError("No ConceptNet rows found.")

    def __len__(self) -> int:
        return len(self.rows)

    def _neighborhood(
        self,
        source: str,
        target: str,
    ) -> tuple[list[str], list[tuple[str, str, str, float]]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT start, relation, end, weight
                FROM edge
                WHERE start = ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (source, self.max_edges),
            ).fetchall()
        finally:
            conn.close()

        concepts = [source]
        if target != source:
            concepts.append(target)

        edges: list[tuple[str, str, str, float]] = []

        for row in rows:
            start = row["start"]
            relation = row["relation"]
            end = row["end"]

            if relation not in RELATION_TO_ID:
                continue

            if start not in concepts and len(concepts) < self.max_nodes:
                concepts.append(start)

            if end not in concepts and len(concepts) < self.max_nodes:
                concepts.append(end)

            if start in concepts and end in concepts:
                edges.append(
                    (
                        start,
                        relation,
                        end,
                        min(
                            5.0,
                            max(
                                0.01,
                                float(row["weight"]),
                            ),
                        ),
                    )
                )

            if len(edges) >= self.max_edges:
                break

        return concepts, edges

    @staticmethod
    def _state(
        concepts: list[str],
        edge_rows: list[tuple[str, str, str, float]],
        active: set[str],
        *,
        masked_index: int | None = None,
    ) -> GraphState:
        concept_to_id = {
            concept: index
            for index, concept in enumerate(concepts)
        }

        node_roles = []
        node_activations = []

        for index, concept in enumerate(concepts):
            if masked_index == index:
                node_roles.append(3)  # MASK
                node_activations.append(0.25)
                continue

            if index == 0:
                node_roles.append(1)  # SOURCE
            elif index == 1:
                node_roles.append(2)  # TARGET
            else:
                node_roles.append(0)

            node_activations.append(
                1.0 if concept in active else 0.25
            )

        edge_list = []

        for start, relation, end, weight in edge_rows:
            if (
                start not in concept_to_id
                or end not in concept_to_id
            ):
                continue

            edge_list.append(
                (
                    concept_to_id[start],
                    concept_to_id[end],
                    RELATION_TO_ID[relation],
                    weight,
                )
            )

        return GraphState(
            node_concepts=concepts,
            node_roles=node_roles,
            node_activations=node_activations,
            edges=edge_list,
        )

    def __getitem__(self, index: int) -> CognitiveExample:
        random.seed(self.seed + index)

        source, relation, target, _weight = self.rows[index]

        concepts, edge_rows = self._neighborhood(
            source,
            target,
        )

        if target not in concepts:
            concepts.insert(
                min(1, len(concepts)),
                target,
            )

        source_id = concepts.index(source)
        target_id = concepts.index(target)

        # Current cognitive state: source is active, target is not yet active.
        current = self._state(
            concepts,
            edge_rows,
            active={source},
        )

        # Next state: target becomes active and the target relation is part of
        # the active working state.
        next_state = self._state(
            concepts,
            edge_rows,
            active={source, target},
        )

        # Mask target identity.
        masked_concepts = list(concepts)
        masked_concepts[target_id] = "[MASK]"

        masked_state = self._state(
            masked_concepts,
            edge_rows,
            active={source},
            masked_index=target_id,
        )

        # Permutation view: preserve graph structure but shuffle node ordering.
        permutation = list(range(len(concepts)))
        random.shuffle(permutation)

        old_to_new = {
            old: new
            for new, old in enumerate(permutation)
        }

        permuted_concepts = [
            concepts[old]
            for old in permutation
        ]

        permuted_roles = [
            current.node_roles[old]
            for old in permutation
        ]

        permuted_activations = [
            current.node_activations[old]
            for old in permutation
        ]

        permuted_edges = [
            (
                old_to_new[start],
                old_to_new[target_node],
                relation_id,
                weight,
            )
            for (
                start,
                target_node,
                relation_id,
                weight,
            ) in current.edges
        ]

        permuted_view = GraphState(
            node_concepts=permuted_concepts,
            node_roles=permuted_roles,
            node_activations=permuted_activations,
            edges=permuted_edges,
        )

        return CognitiveExample(
            current=current,
            next_state=next_state,
            masked_state=masked_state,
            source_index=source_id,
            target_index=target_id,
            target_relation=RELATION_TO_ID[relation],
            masked_target_index=target_id,
            masked_target_concept=target,
            positive_binding=1,
            permuted_view=permuted_view,
        )


def collate_identity(
    items: list[CognitiveExample],
) -> list[CognitiveExample]:
    return items

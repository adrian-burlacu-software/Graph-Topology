from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sqlite3

import torch
from torch.utils.data import Dataset

from .long_term_memory import (
    RELATION_TO_ID,
    RELATIONS,
    ConceptNetMemory,
)


@dataclass
class TrainingExample:
    node_concepts: list[str]
    node_roles: list[int]
    node_activations: list[float]
    edges: list[tuple[int, int, int, float]]
    target_edge_index: int
    target_relation: int
    target_source: int
    target_target: int


class ConceptNetEdgeDataset(Dataset):
    """
    Builds a simple learnable graph task:

        Given a local graph containing a masked/unknown edge type,
        predict the relation of that edge.

    This is intentionally a generic graph-learning task. The model is not told
    that a relation corresponds to a named cognitive task.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        samples: int = 10000,
        max_nodes: int = 18,
        max_edges: int = 40,
        seed: int = 1,
    ) -> None:
        self.db_path = Path(db_path)
        self.samples = samples
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        memory = ConceptNetMemory(
            self.db_path
        )

        try:
            rows = memory.conn.execute(
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

            self.rows = [
                (
                    row["start"],
                    row["relation"],
                    row["end"],
                    float(row["weight"]),
                )
                for row in rows
            ]
        finally:
            memory.close()

        if not self.rows:
            raise RuntimeError(
                "No ConceptNet training edges found."
            )

        self.seed = seed

    def __len__(self) -> int:
        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ) -> TrainingExample:
        random.seed(
            self.seed + index
        )

        source, relation, target, weight = (
            self.rows[index]
        )

        # The smallest meaningful graph consists of source and target plus
        # distractor context nodes. We fetch a bounded neighborhood from SQL.
        conn = sqlite3.connect(
            str(self.db_path)
        )
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
                (
                    source,
                    self.max_edges,
                ),
            ).fetchall()
        finally:
            conn.close()

        concepts = [source]
        if target != source:
            concepts.append(target)

        for row in rows:
            for concept in (
                row["start"],
                row["end"],
            ):
                if concept not in concepts:
                    concepts.append(concept)
                if len(concepts) >= self.max_nodes:
                    break
            if len(concepts) >= self.max_nodes:
                break

        concept_to_id = {
            concept: i
            for i, concept in enumerate(
                concepts
            )
        }

        edges = []
        target_edge_index = -1

        for row in rows:
            if (
                row["start"]
                not in concept_to_id
                or row["end"]
                not in concept_to_id
            ):
                continue

            src_id = concept_to_id[
                row["start"]
            ]
            dst_id = concept_to_id[
                row["end"]
            ]

            rel_id = RELATION_TO_ID.get(
                row["relation"]
            )

            if rel_id is None:
                continue

            edge_number = len(
                edges
            )

            edges.append(
                (
                    src_id,
                    dst_id,
                    rel_id,
                    min(
                        5.0,
                        max(
                            0.01,
                            float(
                                row["weight"]
                            ),
                        ),
                    ),
                )
            )

            if (
                row["start"] == source
                and row["end"] == target
                and row["relation"] == relation
            ):
                target_edge_index = edge_number

        if target_edge_index < 0:
            # Fallback: synthesize a target edge so every example has a label.
            src_id = concept_to_id[
                source
            ]
            dst_id = concept_to_id[
                target
            ]

            target_edge_index = len(
                edges
            )
            edges.append(
                (
                    src_id,
                    dst_id,
                    RELATION_TO_ID[
                        relation
                    ],
                    min(
                        5.0,
                        max(
                            0.01,
                            weight,
                        ),
                    ),
                )
            )

        # Roles are intentionally generic: source and target have distinct
        # working-memory roles; neighbors are ordinary memory nodes.
        node_roles = [
            1 if concept == source
            else 2 if concept == target
            else 0
            for concept in concepts
        ]

        activations = [
            1.0 if concept in {source, target}
            else 0.25
            for concept in concepts
        ]

        return TrainingExample(
            node_concepts=concepts,
            node_roles=node_roles,
            node_activations=activations,
            edges=edges,
            target_edge_index=target_edge_index,
            target_relation=RELATION_TO_ID[
                relation
            ],
            target_source=concept_to_id[
                source
            ],
            target_target=concept_to_id[
                target
            ],
        )


def collate_examples(
    examples: list[TrainingExample],
) -> list[TrainingExample]:
    return examples

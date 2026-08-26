from __future__ import annotations

import json
from pathlib import Path
import random

try:
    from .state import (
        ACTIONS,
        State,
        Node,
        Edge,
    )
except ImportError:
    import sys
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from state import (
        ACTIONS,
        State,
        Node,
        Edge,
    )


def state_from_json(
    payload: dict,
) -> State:
    return State(
        nodes=[
            Node(
                concept=str(node["concept"]),
                activation=float(
                    node.get("activation", 0.0)
                ),
                role=int(
                    node.get("role", 0)
                ),
                persistent=bool(
                    node.get("persistent", False)
                ),
            )
            for node in payload.get(
                "nodes",
                [],
            )
        ],
        edges=[
            Edge(
                source=str(edge["source"]),
                relation=str(edge["relation"]),
                target=str(edge["target"]),
                activation=float(
                    edge.get("activation", 0.0)
                ),
                persistent=bool(
                    edge.get("persistent", False)
                ),
            )
            for edge in payload.get(
                "edges",
                [],
            )
        ],
    )


class TeacherDataset:
    """
    Reads only accepted teacher trajectories.

    The LLM is never called here.
    """

    def __init__(
        self,
        path: Path,
        *,
        seed: int = 209,
    ) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                if not record.get(
                    "final_state"
                ):
                    continue

                final_action = record.get(
                    "final_action"
                )

                if not isinstance(
                    final_action,
                    dict,
                ):
                    continue

                if not final_action.get(
                    "action"
                ):
                    continue

                goal = record.get(
                    "goal"
                )

                if not isinstance(
                    goal,
                    dict,
                ):
                    continue

                self.rows.append(
                    record
                )

        if not self.rows:
            raise RuntimeError(
                f"No usable teacher trajectories found in {self.path}"
            )

        random.Random(seed).shuffle(
            self.rows
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        record = self.rows[index]

        state = state_from_json(
            record["initial_state"]
        )

        next_state = state_from_json(
            record["final_state"]
        )

        action = record[
            "final_action"
        ]

        source = action.get("source")
        target = action.get("target")
        relation = action.get("relation")

        source_index = -1
        target_index = -1

        if source is not None:
            for i, node in enumerate(
                state.nodes
            ):
                if node.concept == source:
                    source_index = i
                    break

        if target is not None:
            for i, node in enumerate(
                state.nodes
            ):
                if node.concept == target:
                    target_index = i
                    break

        goal = record[
            "goal"
        ]

        goal_source = goal.get(
            "source"
        )
        goal_target = goal.get(
            "target"
        )
        goal_relation = goal.get(
            "relation"
        )

        goal_source_index = -1
        goal_target_index = -1

        for i, node in enumerate(
            state.nodes
        ):
            if node.concept == goal_source:
                goal_source_index = i
            if node.concept == goal_target:
                goal_target_index = i

        return {
            "state": state,
            "next_state": next_state,
            "action_id": ACTIONS.index(
                action["action"]
            ),
            "source_index": source_index,
            "target_index": target_index,
            "relation": relation,
            "goal_source_index": goal_source_index,
            "goal_target_index": goal_target_index,
            "goal_relation": goal_relation,
            "teacher_confidence": float(
                (
                    record.get(
                        "teacher_turn2",
                        {}
                    ).get(
                        "confidence"
                    )
                    or record.get(
                        "teacher_turn1",
                        {}
                    ).get(
                        "confidence"
                    )
                    or 1.0
                )
            ),
            "corrected": bool(
                record.get(
                    "teacher_turn2",
                    {}
                ).get(
                    "corrected",
                    False
                )
            ),
            "case_id": record.get(
                "case_id",
                f"sample_{index}",
            ),
        }

    def split(
        self,
        valid_fraction: float = 0.15,
        seed: int = 209,
    ) -> tuple[list[int], list[int]]:
        indices = list(
            range(len(self))
        )

        random.Random(seed).shuffle(
            indices
        )

        valid_size = max(
            1,
            int(
                len(indices)
                * valid_fraction
            ),
        )

        valid = indices[
            :valid_size
        ]
        train = indices[
            valid_size:
        ]

        return train, valid

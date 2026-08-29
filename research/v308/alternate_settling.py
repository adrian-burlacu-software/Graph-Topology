
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlternateStateConfig:
    name: str
    query_bias: float
    query_exploration: float
    settle_steps: int
    settle_gain: float
    settle_leak: float
    branch_separation: float
    branch_decay: float
    recombine_mode: str
    counterfactual_weight: float


class ActiveQueryDualSettling:
    """
    V308 composition:

        active query
             ↓
        construct two branches
             ├── actual
             └── alternate
             ↓
        settle both independently
             ↓
        preserve both states
             ↓
        recombine according to task-independent confidence

    The alternate branch is a genuine internal state, not an answer label.
    """

    def __init__(self, config: AlternateStateConfig):
        self.config=config
        self.selected=0
        self.actual_state=0.0
        self.alternate_state=0.0
        self.actual_confidence=0.0
        self.alternate_confidence=0.0
        self.count=0

    def inject_state(self, graph):
        graph.add_node(
            "query_state",
            "query_state",
            value=float(self.selected),
            persistent=True,
        )
        graph.add_node(
            "actual_state",
            "actual_state",
            value=self.actual_state,
            persistent=True,
        )
        graph.add_node(
            "alternate_state",
            "alternate_state",
            value=self.alternate_state,
            persistent=True,
        )

    def _memory(self, graph) -> int:
        node=graph.nodes.get("memory")
        return int(
            node is not None
            and node.value >= 0
        )

    def _cues(self, graph):
        values=[]
        for role in ("cue1","cue2","cue3"):
            node=next(
                (
                    n for n in graph.nodes.values()
                    if n.role==role
                ),
                None,
            )
            values.append(
                int(
                    node is not None
                    and node.value >= 0.5
                )
            )
        return values

    def choose_query(self, graph):
        cues=self._cues(graph)

        scores=[
            self.config.query_bias
            +self.config.query_exploration*(i+1)
            +0.10*bit
            for i,bit in enumerate(cues)
        ]

        self.selected=max(
            range(3),
            key=lambda i:scores[i],
        )

        return self.selected,cues

    def settle_branch(
        self,
        memory_value: int,
        cue_value: int,
        initial: float,
    ) -> float:
        x=initial

        for _ in range(
            self.config.settle_steps
        ):
            drive=(
                self.config.settle_gain
                *(
                    float(memory_value)
                    +float(cue_value)
                )
                -self.config.settle_leak*x
            )

            target=1.0 if drive>=0.5 else 0.0

            x=(
                0.65*x
                +0.35*target
            )

        return x

    def run(self, graph, episode):
        selected,cues=self.choose_query(graph)

        memory=self._memory(graph)
        cue=cues[selected]

        # Actual branch sees the selected cue.
        actual=self.settle_branch(
            memory,
            cue,
            0.5*float(memory)+0.5*float(cue),
        )

        # Alternate branch sees the opposite cue state, while preserving the
        # same memory. It is never discarded.
        alternate=self.settle_branch(
            memory,
            1-cue,
            0.5*float(memory)+0.5*float(1-cue),
        )

        # Explicit branch separation discourages collapse to the same state.
        if abs(actual-alternate)<self.config.branch_separation:
            alternate=max(
                0.0,
                min(
                    1.0,
                    alternate
                    +self.config.branch_separation,
                ),
            )

        actual*=self.config.branch_decay
        alternate*=self.config.branch_decay

        self.actual_state=actual
        self.alternate_state=alternate

        self.actual_confidence=max(
            actual,
            1.0-actual,
        )
        self.alternate_confidence=max(
            alternate,
            1.0-alternate,
        )

        if self.config.recombine_mode=="actual_first":
            bound=actual

        elif self.config.recombine_mode=="contrastive":
            # Preserve both hypotheses, then choose the branch with stronger
            # internal confidence. Ties favor actual.
            if (
                self.alternate_confidence
                > self.actual_confidence
            ):
                bound=alternate
            else:
                bound=actual

        elif self.config.recombine_mode=="blend":
            bound=(
                (1-self.config.counterfactual_weight)*actual
                +self.config.counterfactual_weight*alternate
            )

        else:
            raise ValueError(
                self.config.recombine_mode
            )

        self.count+=1

        return int(bound>=0.5)


CONFIGS={
    "dual_balanced":AlternateStateConfig(
        "dual_balanced",
        query_bias=0.60,
        query_exploration=0.10,
        settle_steps=4,
        settle_gain=0.80,
        settle_leak=0.25,
        branch_separation=0.20,
        branch_decay=0.98,
        recombine_mode="actual_first",
        counterfactual_weight=0.30,
    ),
    "dual_contrastive":AlternateStateConfig(
        "dual_contrastive",
        query_bias=0.60,
        query_exploration=0.10,
        settle_steps=5,
        settle_gain=0.80,
        settle_leak=0.25,
        branch_separation=0.15,
        branch_decay=0.98,
        recombine_mode="contrastive",
        counterfactual_weight=0.50,
    ),
    "dual_blend":AlternateStateConfig(
        "dual_blend",
        query_bias=0.60,
        query_exploration=0.10,
        settle_steps=5,
        settle_gain=0.80,
        settle_leak=0.25,
        branch_separation=0.15,
        branch_decay=0.98,
        recombine_mode="blend",
        counterfactual_weight=0.50,
    ),
    "dual_deep":AlternateStateConfig(
        "dual_deep",
        query_bias=0.65,
        query_exploration=0.05,
        settle_steps=8,
        settle_gain=0.80,
        settle_leak=0.20,
        branch_separation=0.20,
        branch_decay=0.98,
        recombine_mode="contrastive",
        counterfactual_weight=0.50,
    ),
}

assert len(CONFIGS)==4

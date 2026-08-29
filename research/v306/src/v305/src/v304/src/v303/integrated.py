
from __future__ import annotations

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
    MemoryReadout,
    BindingPlanner,
)
from hypothesis import (
    HypothesisRevision,
    CONFIGS as HYPOTHESIS_CONFIGS,
)
from selective import (
    SelectiveRepresentation,
    CONFIGS as SELECTIVE_CONFIGS,
)


class IntegratedSystem:
    """
    V303 architecture:

        persistent memory
        + transform dynamics
        + selective representation
        + memory readout
        + binding planner
        + hypothesis/revision
    """

    def __init__(
        self,
        selective: SelectiveRepresentation,
        hypothesis: HypothesisRevision,
    ):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.readout=MemoryReadout()
        self.planner=BindingPlanner()
        self.selective=selective
        self.hypothesis=hypothesis

    def run(
        self,
        episode,
        learn=True,
    ):
        graph=episode.graph.clone()

        self.selective.inject_state(graph)
        self.hypothesis.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient=next(
            (
                name
                for name in list(graph.nodes)
                if graph.nodes[name].role=="initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(transient)

        for step in range(
            1,
            episode.decision_step+1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
            )

            # Selection operates throughout the temporal computation, not
            # only as a post-hoc readout trick.
            self.selective.apply(graph)

        recalled,path=self.readout.read(
            graph,
        )

        decision=self.planner.plan(
            graph,
            recalled,
        )

        decision=self.hypothesis.transform_decision(
            graph,
            decision,
            episode,
        )

        correct=(
            decision==episode.answer_bit
        )

        if learn:
            self.hypothesis.feedback(
                graph,
                decision,
                episode.answer_bit,
                episode,
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
        }

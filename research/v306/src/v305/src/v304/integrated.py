
from __future__ import annotations

from competition import GoalCompetition
from hypothesis import HypothesisRevision, CONFIGS as HYP_CONFIGS
from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
    MemoryReadout,
    BindingPlanner,
)


class IntegratedSystem:
    """
    V304:
        persistent memory
        + transform dynamics
        + goal-conditioned competition
        + memory readout
        + binding planner
        + hypothesis/revision
    """

    def __init__(
        self,
        competition,
        hypothesis,
    ):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.readout=MemoryReadout()
        self.planner=BindingPlanner()
        self.competition=competition
        self.hypothesis=hypothesis

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

        self.competition.inject_state(graph)
        self.hypothesis.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient=next(
            (
                name for name in list(graph.nodes)
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
            self.competition.apply(
                graph,
                episode,
            )

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

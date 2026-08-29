
from __future__ import annotations

from richer_cognition import PersistentMemory, TransformDynamics
from hypothesis import HypothesisRevision, CONFIGS as HYP_CONFIGS
from candidate_binding import (
    CompetitiveHypothesisBinding,
)

class IntegratedSystem:
    def __init__(self,binding,hypothesis):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.binding=binding
        self.hypothesis=hypothesis

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

        self.binding.inject_state(graph)
        self.hypothesis.inject_state(graph)

        self.memory.observe(graph,episode.query)

        transient=next(
            (
                name for name in list(graph.nodes)
                if graph.nodes[name].role=="initial_fact"
            ),
            None,
        )
        if transient is not None:
            graph.nodes.pop(transient)

        for step in range(1,episode.decision_step+1):
            self.dynamics.step(graph,step)
            self.memory.maintain(graph)

        bound,working=self.binding.compete(graph)

        decision=int(bound)
        decision=self.hypothesis.transform_decision(
            graph,decision,episode
        )

        correct=(decision==episode.answer_bit)

        if learn:
            self.hypothesis.feedback(
                graph,decision,episode.answer_bit,episode
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
            "working_set":[
                {
                    "kind":c.kind,
                    "value":c.value,
                    "support":c.support,
                }
                for c in working
            ],
            "confidence":self.binding.confidence,
            "collisions":self.binding.collisions,
        }

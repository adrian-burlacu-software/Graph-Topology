
from __future__ import annotations
from richer_cognition import PersistentMemory,TransformDynamics
from decomposition import ExplicitAnswerCognition


class IntegratedSystem:
    def __init__(self,mode):
        breadth={
            "criterion_balanced":8,
            "criterion_narrow":4,
            "criterion_broad":16,
            "criterion_strict":8,
        }[mode]

        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.cognition=ExplicitAnswerCognition(
            breadth
        )
        self.count=0

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

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
            self.dynamics.step(graph,step)
            self.memory.maintain(graph)

        (
            state,
            criterion,
            semantics,
            solved,
            state_eval,
            answer_eval,
            decision,
        )=self.cognition.run(
            graph,
            episode,
            self.memory,
        )

        self.count+=1

        return {
            "correct":decision==episode.answer_bit,
            "decision":decision,
            "answer":episode.answer_bit,
            "state_valid":state_eval.satisfied,
            "answer_valid":answer_eval.valid,
            "state_confidence":state_eval.confidence,
            "answer_confidence":answer_eval.confidence,
            "goal_type":criterion.name,
            "subgoals":len(solved.plan.subgoals),
            "achieved":len(solved.achieved),
            "trace":solved.trace,
            "registers":solved.registers,
            "answer_unmet":answer_eval.unmet,
        }

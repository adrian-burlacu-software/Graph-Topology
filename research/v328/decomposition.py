
from __future__ import annotations
from dataclasses import dataclass
from goal_decomposition import decompose_goal
from explicit_goal import build_goal_semantics
from planning_state import PlanningFrame
from semantic_planner import SemanticPlanner
from answer_criterion import build_answer_criterion
from answer_evaluator import AnswerEvaluator


@dataclass(frozen=True)
class CognitiveState:
    memory:int
    cues:tuple[int,int,int]
    context:str
    source:str
    target:str
    relation_a:str
    relation_b:str
    goal_text:str


class StateEstimator:
    def build(self,graph,episode,memory_module):
        node=graph.nodes.get("memory")
        memory=int(node is not None and node.value>=0)

        cues=[]
        for role in ("cue1","cue2","cue3"):
            node=next(
                (
                    n for n in graph.nodes.values()
                    if n.role==role
                ),
                None,
            )
            cues.append(
                int(node is not None and node.value>=0.5)
            )

        return CognitiveState(
            memory=memory,
            cues=(cues[0],cues[1],cues[2]),
            context=episode.task,
            source=str(episode.query.source),
            target=str(episode.query.target),
            relation_a=str(episode.query.relation_a),
            relation_b=str(episode.query.relation_b),
            goal_text=str(episode.query),
        )


class ExplicitAnswerCognition:
    def __init__(self,breadth):
        self.estimator=StateEstimator()
        self.planner=SemanticPlanner(breadth)
        self.count=0

    def run(self,graph,episode,memory_module):
        state=self.estimator.build(
            graph,
            episode,
            memory_module,
        )
        plan=decompose_goal(episode)
        semantics=build_goal_semantics(episode)
        criterion=build_answer_criterion(episode)

        frame=PlanningFrame(
            state=state,
            plan=plan,
            registers={
                "memory":state.memory,
                "cue1":state.cues[0],
                "cue2":state.cues[1],
                "cue3":state.cues[2],
            },
        )

        solved=self.planner.solve(
            frame,
            max_steps=20,
        )

        state_eval=__import__(
            "semantic_goal_verifier"
        ).SemanticGoalVerifier().evaluate(
            semantics,
            solved,
        )

        answer_eval=AnswerEvaluator().evaluate(
            criterion,
            solved,
        )

        decision=int(
            solved.registers.get(
                "result",
                state.memory,
            )
        )

        self.count+=1

        return (
            state,
            criterion,
            semantics,
            solved,
            state_eval,
            answer_eval,
            decision,
        )


from __future__ import annotations
from richer_cognition import TASKS,make_sequence
from answer_criterion import AnswerCriterion,build_answer_criterion
from answer_evaluator import AnswerEvaluator
from planning_state import CognitiveState,PlanningFrame
from goal_decomposition import decompose_goal
from explicit_goal import build_goal_semantics
from semantic_goal_verifier import SemanticGoalVerifier
from integrated import IntegratedSystem


def main():
    assert len(TASKS)==6

    seq=make_sequence(
        328,
        "counterfactual",
        1,
        9,
    )
    ep=seq.episodes[0]

    criterion=build_answer_criterion(ep)
    semantics=build_goal_semantics(ep)

    assert isinstance(criterion,AnswerCriterion)
    assert criterion.name=="counterfactual_consequence"

    state=CognitiveState(
        memory=1,
        cues=(0,1,0),
        context=ep.task,
        source=str(ep.query.source),
        target=str(ep.query.target),
        relation_a=str(ep.query.relation_a),
        relation_b=str(ep.query.relation_b),
        goal_text=str(ep.query),
    )

    plan=decompose_goal(ep)

    frame=PlanningFrame(
        state=state,
        plan=plan,
        registers={
            "memory":1,
            "cue1":0,
            "cue2":1,
            "cue3":0,
            "actual":1,
            "alternate":0,
            "result":1,
            "result_source":"actual",
        },
    )

    # State can be structurally complete but fail answer semantics.
    state_eval=SemanticGoalVerifier().evaluate(
        semantics,
        frame,
    )
    assert state_eval.satisfied

    answer=AnswerEvaluator().evaluate(
        criterion,
        frame,
    )
    assert not answer.valid

    frame.registers["result"]=0
    frame.registers["result_source"]="alternate"

    answer=AnswerEvaluator().evaluate(
        criterion,
        frame,
    )
    assert answer.valid

    # End-to-end task paths.
    for mode in (
        "criterion_balanced",
        "criterion_narrow",
        "criterion_broad",
        "criterion_strict",
    ):
        for task in TASKS:
            seq=make_sequence(
                329,
                task,
                4,
                9,
            )
            system=IntegratedSystem(mode)
            rows=[
                system.run(ep,True)
                for ep in seq.episodes
            ]
            assert len(rows)==4
            assert system.count==4

    print("V328 validation: PASS")
    print("explicit AnswerCriterion: PASS")
    print("state semantics != answer semantics: PASS")
    print("answer evaluator: PASS")
    print("alternate-world distinction: PASS")
    print("all mode/task paths: PASS")


if __name__=="__main__":
    main()

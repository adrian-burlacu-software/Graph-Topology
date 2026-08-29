
from __future__ import annotations
from planning_state import GoalPlan,Subgoal


def decompose_goal(episode):
    task=episode.task

    if task=="delayed_memory":
        subs=(
            Subgoal("retrieve","retrieve_memory","memory"),
            Subgoal("decide","produce_answer","result"),
        )
        q="retrieve"
    elif task=="sequence_binding":
        subs=(
            Subgoal("retrieve","retrieve_memory","memory"),
            Subgoal("retrieve","retrieve_cue1","cue1"),
            Subgoal("retrieve","retrieve_cue2","cue2"),
            Subgoal("transform","combine_sequence","memory,cue1,cue2"),
            Subgoal("decide","produce_answer","result"),
        )
        q="bind_sequence"
    elif task=="interference":
        subs=(
            Subgoal("retrieve","retrieve_memory","memory"),
            Subgoal("retrieve","retrieve_relevant_cue","cue1"),
            Subgoal("transform","compute_relevance","memory,cue1"),
            Subgoal("decide","produce_answer","result"),
        )
        q="select_relevant"
    elif task=="rule_change":
        subs=(
            Subgoal("retrieve","collect_memory","memory"),
            Subgoal("retrieve","collect_cues","cue1,cue2,cue3"),
            Subgoal("transform","derive_rule_evidence","memory,cue1"),
            Subgoal("revise","revise_rule","hypothesis"),
            Subgoal("decide","produce_answer","result"),
        )
        q="infer_rule"
    elif task=="planning":
        subs=(
            Subgoal("retrieve","collect_memory","memory"),
            Subgoal("retrieve","collect_cues","cue1,cue2,cue3"),
            Subgoal("transform","construct_plan_state","memory,cue1,cue2,cue3"),
            Subgoal("decide","produce_answer","result"),
        )
        q="plan"
    elif task=="counterfactual":
        subs=(
            Subgoal("retrieve","collect_actual_state","memory"),
            Subgoal("retrieve","collect_alternatives","cue1,cue2,cue3"),
            Subgoal("transform","construct_actual_state","memory,cue1"),
            Subgoal("transform","construct_alternate_state","memory,cue2"),
            Subgoal("transform","evaluate_alternate","alternate"),
            Subgoal("decide","produce_answer","result"),
        )
        q="counterfactual"
    else:
        subs=(
            Subgoal("retrieve","retrieve_memory","memory"),
            Subgoal("decide","produce_answer","result"),
        )
        q="generic"

    return GoalPlan(
        task=task,
        query_type=q,
        subgoals=subs,
        requires_revision=(task=="rule_change"),
        allow_alternative=(task=="counterfactual"),
    )

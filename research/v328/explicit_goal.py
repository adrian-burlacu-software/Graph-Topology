
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GoalPredicate:
    name:str
    operator:str
    subject:str
    value:object=None


@dataclass(frozen=True)
class GoalSemantics:
    task:str
    query_type:str
    predicates:Tuple[GoalPredicate,...]


def build_goal_semantics(episode):
    task=episode.task

    if task=="counterfactual":
        preds=(
            GoalPredicate("actual","exists","actual"),
            GoalPredicate("alternate","exists","alternate"),
            GoalPredicate("result","exists","result"),
        )
        q="counterfactual"
    elif task=="interference":
        preds=(
            GoalPredicate("memory","exists","memory"),
            GoalPredicate("cue1","exists","cue1"),
            GoalPredicate("result","exists","result"),
        )
        q="select_relevant"
    elif task=="rule_change":
        preds=(
            GoalPredicate("memory","exists","memory"),
            GoalPredicate("cue1","exists","cue1"),
            GoalPredicate("hypothesis","exists","hypothesis"),
            GoalPredicate("result","exists","result"),
        )
        q="infer_rule"
    elif task=="planning":
        preds=tuple(
            GoalPredicate(x,"exists",x)
            for x in ("memory","cue1","cue2","cue3","result")
        )
        q="plan"
    elif task=="sequence_binding":
        preds=tuple(
            GoalPredicate(x,"exists",x)
            for x in ("memory","cue1","cue2","result")
        )
        q="bind_sequence"
    else:
        preds=(GoalPredicate("result","exists","result"),)
        q="retrieve"

    return GoalSemantics(
        task=task,
        query_type=q,
        predicates=preds,
    )

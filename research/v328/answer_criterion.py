
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AnswerCriterion:
    """
    Explicit specification of what a correct answer means.

    This is intentionally separate from GoalSemantics (state validity).
    """
    name: str
    source: str
    transform: str
    comparison: str
    operands: Tuple[str, ...]
    truth_condition: str


def build_answer_criterion(episode) -> AnswerCriterion:
    task=episode.task

    if task=="delayed_memory":
        return AnswerCriterion(
            "memory_identity",
            "memory",
            "identity",
            "none",
            ("memory",),
            "result_equals_memory",
        )

    if task=="sequence_binding":
        return AnswerCriterion(
            "sequence_result",
            "result",
            "identity",
            "none",
            ("memory","cue1","cue2"),
            "result_is_sequence_computation",
        )

    if task=="interference":
        return AnswerCriterion(
            "relevant_signal",
            "result",
            "identity",
            "relevance",
            ("memory","cue1"),
            "result_is_relevance_computation",
        )

    if task=="rule_change":
        return AnswerCriterion(
            "revised_rule",
            "result",
            "identity",
            "rule_change",
            ("memory","cue1","rule_phase"),
            "result_is_post_change_rule",
        )

    if task=="planning":
        return AnswerCriterion(
            "plan_result",
            "result",
            "identity",
            "planning",
            ("memory","cue1","cue2","cue3"),
            "result_is_plan_computation",
        )

    if task=="counterfactual":
        return AnswerCriterion(
            "counterfactual_consequence",
            "result",
            "identity",
            "actual_vs_alternate",
            ("actual","alternate"),
            "result_is_alternate_consequence",
        )

    return AnswerCriterion(
        "generic_result",
        "result",
        "identity",
        "none",
        ("result",),
        "result_exists",
    )

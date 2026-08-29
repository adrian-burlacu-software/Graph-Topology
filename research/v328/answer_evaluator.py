
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerEvaluation:
    criterion: object
    valid: bool
    confidence: float
    evidence: tuple[str,...]
    unmet: tuple[str,...]


class AnswerEvaluator:
    """
    Semantic answer evaluator.

    It evaluates an explicit AnswerCriterion against the simulated state.
    No hidden answer is used here.
    """

    def evaluate(self,criterion,frame):
        missing=[
            x for x in criterion.operands
            if x not in frame.registers
        ]
        if missing:
            return AnswerEvaluation(
                criterion,
                False,
                0.0,
                (),
                tuple(f"missing:{x}" for x in missing),
            )

        result=frame.registers.get("result")

        if criterion.truth_condition=="result_equals_memory":
            ok=result==frame.registers["memory"]
            return self._make(
                criterion,
                ok,
                "result_equals_memory",
            )

        if criterion.truth_condition=="result_is_sequence_computation":
            expected=(
                frame.registers["memory"]
                ^ frame.registers["cue1"]
                ^ frame.registers["cue2"]
            )
            ok=result==expected
            return self._make(
                criterion,
                ok,
                "sequence_formula_matches",
            )

        if criterion.truth_condition=="result_is_relevance_computation":
            expected=(
                frame.registers["memory"]
                ^ frame.registers["cue1"]
            )
            ok=(
                result==expected
                and frame.registers.get(
                    "result_source"
                )=="memory,cue1"
            )
            return self._make(
                criterion,
                ok,
                "relevance_computation_matches",
            )

        if criterion.truth_condition=="result_is_post_change_rule":
            ok=(
                frame.registers.get("rule_phase")==1
                and frame.registers.get(
                    "result_source"
                )=="memory,cue1"
            )
            return self._make(
                criterion,
                ok,
                "post_change_rule_is_explicit",
            )

        if criterion.truth_condition=="result_is_plan_computation":
            expected=(
                frame.registers["memory"]
                ^ frame.registers["cue1"]
                ^ frame.registers["cue2"]
                ^ frame.registers["cue3"]
            )
            ok=result==expected
            return self._make(
                criterion,
                ok,
                "plan_computation_matches",
            )

        if criterion.truth_condition=="result_is_alternate_consequence":
            alternate=frame.registers["alternate"]
            ok=(
                result==alternate
                and frame.registers.get(
                    "result_source"
                )=="alternate"
            )
            return self._make(
                criterion,
                ok,
                "alternate_world_is_answer_source",
            )

        if criterion.truth_condition=="result_exists":
            return self._make(
                criterion,
                "result" in frame.registers,
                "result_present",
            )

        return AnswerEvaluation(
            criterion,
            False,
            0.0,
            (),
            ("unknown_truth_condition",),
        )

    @staticmethod
    def _make(criterion,ok,reason):
        return AnswerEvaluation(
            criterion=criterion,
            valid=bool(ok),
            confidence=1.0 if ok else 0.0,
            evidence=(reason,) if ok else (),
            unmet=() if ok else (reason,),
        )

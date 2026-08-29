
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PredicateResult:
    name:str
    satisfied:bool


@dataclass(frozen=True)
class GoalEvaluation:
    satisfied:bool
    confidence:float
    predicates:tuple[PredicateResult,...]


class SemanticGoalVerifier:
    def evaluate(self,goal,frame):
        results=[]

        for p in goal.predicates:
            ok=(
                p.operator=="exists"
                and (
                    p.subject in frame.registers
                    or p.subject in frame.objects
                )
            )
            results.append(
                PredicateResult(
                    p.name,
                    ok,
                )
            )

        return GoalEvaluation(
            satisfied=all(x.satisfied for x in results),
            confidence=sum(int(x.satisfied) for x in results)/max(1,len(results)),
            predicates=tuple(results),
        )

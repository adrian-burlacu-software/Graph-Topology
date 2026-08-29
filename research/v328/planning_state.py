
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CognitiveState:
    memory: int
    cues: tuple[int,int,int]
    context: str
    source: str
    target: str
    relation_a: str
    relation_b: str
    goal_text: str


@dataclass(frozen=True)
class Subgoal:
    kind: str
    name: str
    operand: str


@dataclass(frozen=True)
class GoalPlan:
    task: str
    query_type: str
    subgoals: tuple[Subgoal,...]
    requires_revision: bool
    allow_alternative: bool


@dataclass
class PlanningFrame:
    state: CognitiveState
    plan: GoalPlan
    registers: Dict[str,object]=field(default_factory=dict)
    objects: Dict[str,str]=field(default_factory=dict)
    achieved: List[str]=field(default_factory=list)
    trace: List[str]=field(default_factory=list)

    def clone(self):
        return PlanningFrame(
            state=self.state,
            plan=self.plan,
            registers=dict(self.registers),
            objects=dict(self.objects),
            achieved=list(self.achieved),
            trace=list(self.trace),
        )

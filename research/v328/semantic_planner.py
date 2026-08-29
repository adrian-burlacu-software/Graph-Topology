
from __future__ import annotations
from dataclasses import dataclass
from answer_evaluator import AnswerEvaluator
from explicit_goal import build_goal_semantics
from grounded_operators import OPERATORS
from planning_state import PlanningFrame


@dataclass(frozen=True)
class Candidate:
    operator: object
    frame: PlanningFrame
    score: float


class SemanticPlanner:
    def __init__(self,breadth=8):
        self.breadth=breadth
        self.answer=AnswerEvaluator()

    def _needed(self,frame):
        for s in frame.plan.subgoals:
            if s.name not in frame.achieved:
                return s
        return None

    def _ops(self,frame,s):
        mapping={
            "retrieve_memory":["read_memory"],
            "collect_memory":["read_memory"],
            "retrieve_cue1":["read_cue1"],
            "retrieve_cue2":["read_cue2"],
            "collect_cues":["read_cue1","read_cue2","read_cue3"],
            "retrieve_relevant_cue":["read_cue1"],
            "combine_sequence":["xor_sequence"],
            "compute_relevance":["compute_relevance"],
            "derive_rule_evidence":["derive_rule_evidence"],
            "revise_rule":["revise_hypothesis"],
            "construct_plan_state":["plan_result"],
            "collect_actual_state":["read_memory"],
            "collect_alternatives":["read_cue1","read_cue2","read_cue3"],
            "construct_actual_state":["set_actual"],
            "construct_alternate_state":["set_alternate"],
            "evaluate_alternate":["evaluate_alternate"],
            "produce_answer":["emit_result"],
        }
        wanted=mapping.get(s.name,[])
        return [x for x in OPERATORS if x.name in wanted]

    def _subgoal_ok(self,frame,s):
        r=frame.registers

        if s.name in (
            "retrieve_memory","collect_memory"
        ):
            return "memory" in r
        if s.name=="retrieve_cue1":
            return "cue1" in r
        if s.name=="retrieve_cue2":
            return "cue2" in r
        if s.name=="retrieve_cue3":
            return "cue3" in r
        if s.name=="collect_cues":
            return all(x in r for x in ("cue1","cue2","cue3"))
        if s.name=="retrieve_relevant_cue":
            return "cue1" in r
        if s.name=="combine_sequence":
            return (
                "result" in r
                and all(x in r for x in ("memory","cue1","cue2"))
            )
        if s.name=="compute_relevance":
            return (
                "result" in r
                and r.get("result_source")=="memory,cue1"
            )
        if s.name=="derive_rule_evidence":
            return (
                "result" in r
                and r.get("rule_phase")==1
            )
        if s.name=="revise_rule":
            return "hypothesis" in r
        if s.name=="construct_plan_state":
            return (
                "result" in r
                and all(
                    x in r
                    for x in ("memory","cue1","cue2","cue3")
                )
            )
        if s.name=="collect_actual_state":
            return "memory" in r
        if s.name=="collect_alternatives":
            return all(x in r for x in ("cue1","cue2","cue3"))
        if s.name=="construct_actual_state":
            return "actual" in r
        if s.name=="construct_alternate_state":
            return "alternate" in r
        if s.name=="evaluate_alternate":
            return (
                "result" in r
                and r.get("result_source")=="alternate"
            )
        if s.name=="produce_answer":
            return "result" in r
        return False

    def expand(self,frame):
        s=self._needed(frame)
        if s is None:
            return []

        candidates=[]
        for op in self._ops(frame,s):
            simulated=op.execute(frame,s)

            before_ok=self._subgoal_ok(frame,s)
            after_ok=self._subgoal_ok(simulated,s)

            score=(10.0 if after_ok and not before_ok else 0.0)
            score += 0.2*len(simulated.achieved)

            candidates.append(
                Candidate(
                    op,
                    simulated,
                    score,
                )
            )

        candidates.sort(
            key=lambda x:x.score,
            reverse=True,
        )
        return candidates[:self.breadth]

    def solve(self,frame,max_steps=20):
        current=frame

        for _ in range(max_steps):
            s=self._needed(current)
            if s is None:
                break

            candidates=self.expand(current)
            if not candidates:
                break

            best=candidates[0]

            if self._subgoal_ok(best.frame,s):
                current=best.frame
                current.achieved.append(s.name)
            else:
                break

        return current

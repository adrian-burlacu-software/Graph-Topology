
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    goal: object
    selected_content: str
    selected_source: str
    selected_score: float
    candidate_count: int
    confidence: float


class GoalPlanner:
    """
    Architecture-level planner.

    The participant proposes. Knowledge supplies evidence. The planner scores
    and selects. The realizer only renders the selected content.
    """

    def __init__(
        self,
        evaluator,
        participant,
        realizer,
        memory,
        trace=True,
    ):
        self.evaluator=evaluator
        self.participant=participant
        self.realizer=realizer
        self.memory=memory
        self.trace=trace

    def _trace_candidate(self,c):
        if not self.trace:
            return
        print(
            f"  [CANDIDATE] source={c.source} "
            f"score={c.total:.2f}",
            flush=True,
        )
        print(
            f"    goal={c.goal_score:.2f} "
            f"context={c.context_score:.2f} "
            f"evidence={c.evidence_score:.2f} "
            f"progress={c.progress_score:.2f} "
            f"natural={c.naturalness_score:.2f} "
            f"brevity={c.brevity_score:.2f}",
            flush=True,
        )
        print(
            f"    content={c.content!r}",
            flush=True,
        )

    def select(
        self,
        goal,
        user_text,
        knowledge_relations,
        participant_content,
    ):
        candidates=[]

        has_knowledge=bool(knowledge_relations)

        # Knowledge candidate only exists if actual semantic content exists.
        if has_knowledge:
            statements=[
                f"{a} {b.replace('_',' ')} {c}"
                for a,b,c in knowledge_relations[:2]
            ]
            if statements:
                candidates.append(
                    self.evaluator.evaluate(
                        goal,
                        user_text,
                        " ".join(statements)+".",
                        "knowledge",
                        self.memory,
                        True,
                    )[0]
                )

        if participant_content:
            candidates.append(
                self.evaluator.evaluate(
                    goal,
                    user_text,
                    participant_content,
                    "participant",
                    self.memory,
                    has_knowledge,
                )[0]
            )

        fallback=self.realizer.fallback(goal)

        candidates.append(
            self.evaluator.evaluate(
                goal,
                user_text,
                fallback,
                "fallback",
                self.memory,
                has_knowledge,
            )[0]
        )

        for candidate in candidates:
            self._trace_candidate(candidate)

        self.memory.inc(
            "candidate_count",
            len(candidates),
        )

        candidates.sort(
            key=lambda c:c.total,
            reverse=True,
        )

        winner=candidates[0]
        self.memory.inc("candidate_selected")

        if self.trace:
            margin=(
                winner.total
                -candidates[1].total
                if len(candidates)>1
                else winner.total
            )

            print(
                f"  [PLANNER] winner={winner.source} "
                f"score={winner.total:.2f} "
                f"margin={margin:.2f}",
                flush=True,
            )

        return winner

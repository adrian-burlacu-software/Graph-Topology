
from __future__ import annotations

from dataclasses import dataclass

from model_utils import clean_generated,looks_meta
from response_firewall import is_internal
from realization_guard import verify_realization
from llm_evaluator import assess,ProposalAssessment


@dataclass
class ParticipantProposal:
    raw: str
    assessment: ProposalAssessment


class Participant:
    def __init__(self,llm,trace=True):
        self.llm=llm
        self.trace=trace

    def propose(self,goal,user_text,context,facts):
        evidence="\n".join(
            f"- {f['subject']} {f['predicate']} {f['object_text']}"
            for f in facts[:8]
        )

        raw=self.llm.generate(
            (
                "You are an internal cognitive participant. "
                "Suggest exactly one useful conversational move or proposition "
                "for the architecture to evaluate. "
                "Do not answer with analysis. "
                "Do not mention the architecture, participant, candidate, "
                "goal, or this instruction. "
                "For factual goals, use only the supplied evidence. "
                "If the evidence does not answer the user's exact question, "
                "propose a concise statement of uncertainty instead. "
                "Do not answer a different question using related facts."
            ),
            (
                f"GOAL: {goal.name}\n"
                f"USER: {user_text}\n"
                f"STATE:\n{context}\n"
                f"EVIDENCE:\n{evidence}\n"
                "Return only the proposed content."
            ),
        )

        text=clean_generated(raw)

        prefixes=(
            "a useful candidate proposition is:",
            "a useful candidate proposition could be:",
            "a useful proposition is:",
            "one useful candidate is:",
            "proposal:",
            "answer:",
            "response:",
        )
        low=text.lower()
        for prefix in prefixes:
            if low.startswith(prefix):
                text=text[len(prefix):].strip()
                break

        assessment=assess(
            goal,
            user_text,
            context,
            facts,
            text,
        )

        if self.trace:
            print(
                f"  [PARTICIPANT] raw={text!r}",
                flush=True,
            )
            print(
                f"  [PARTICIPANT SCORE] "
                f"goal={assessment.goal_score:.2f} "
                f"context={assessment.context_score:.2f} "
                f"evidence={assessment.evidence_score:.2f} "
                f"unsupported={assessment.unsupported_score:.2f} "
                f"natural={assessment.natural_score:.2f} "
                f"brief={assessment.brevity_score:.2f} "
                f"contribution={assessment.contribution:.2f} "
                f"valid={assessment.valid} "
                f"reason={assessment.reason}",
                flush=True,
            )

        if not assessment.accepted:
            return None

        return ParticipantProposal(
            raw=text,
            assessment=assessment,
        )

    def realize(self,goal,selected,context,facts=()):
        raw=self.llm.generate(
            "You are only a language realizer. "
            "Express the selected content naturally. "
            "Preserve its meaning exactly. "
            "Do not introduce any new facts, entities, quantities, "
            "examples, explanations, or questions. "
            "Return only the final reply.",
            (
                f"GOAL: {goal.name}\n"
                f"SELECTED CONTENT: {selected}\n"
                f"CONTEXT:\n{context}"
            ),
        )

        text=clean_generated(raw)

        if not text or looks_meta(text) or is_internal(text):
            return None

        ok,overlap,reason=verify_realization(
            selected,
            text,
            facts=facts,
            goal_name=goal.name,
        )

        if self.trace:
            print(
                f"  [REALIZER CHECK] accepted={ok} "
                f"grounding={overlap:.2f} reason={reason}",
                flush=True,
            )

        return text if ok else None

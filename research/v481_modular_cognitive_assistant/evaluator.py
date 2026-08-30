
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Candidate:
    source: str
    action: str
    content: str
    goal_score: float
    context_score: float
    evidence_score: float
    progress_score: float
    naturalness_score: float
    brevity_score: float

    @property
    def total(self):
        return (
            self.goal_score
            +self.context_score
            +self.evidence_score
            +self.progress_score
            +self.naturalness_score
            +self.brevity_score
        )


def words(text):
    return set(
        x
        for x in re.findall(
            r"[a-z0-9']+",
            text.lower(),
        )
        if len(x)>1
    )


class CandidateEvaluator:
    """
    The evaluator is deliberately independent from the LLM.

    The LLM can propose something. It cannot assign its own score.
    """

    def _goal_score(self,goal,content):
        n=len(content.split())

        if goal.name.startswith("social_"):
            return 9.0 if n>=1 else 1.0
        if goal.name=="request_explanation":
            return 8.0 if n>=5 else 2.0
        if goal.name=="request_generation":
            return 8.5 if n>=4 else 2.0
        if goal.name=="challenge_claim":
            return 8.5 if n>=3 else 2.0
        if goal.name=="request_opinion":
            return 8.0 if n>=3 else 2.0
        if goal.name=="explore_assistant":
            return 8.0 if n>=3 else 2.0
        if goal.name=="request_information":
            return 8.0 if n>=2 else 2.0
        return 6.0 if n>=2 else 2.0

    def _context_score(self,content,memory):
        state=memory.state()
        score=0.0
        c=words(content)

        if state.topic and state.topic.lower() in c:
            score+=4.0

        if state.last_assistant_text:
            score+=1.0

        return score

    def _evidence_score(self,source,has_knowledge):
        if source=="knowledge":
            return 7.0 if has_knowledge else 0.0
        if source=="participant":
            return 3.0
        return 2.0

    def _progress_score(self,goal,content):
        low=content.lower()

        if goal.name=="challenge_claim":
            if any(
                x in low
                for x in (
                    "yes","no","however","actually",
                    "not necessarily","that's",
                )
            ):
                return 3.0
            return 1.0

        if goal.name=="explore_assistant":
            if (
                "i'm" in low
                or "i am" in low
                or "i'd" in low
                or "i would" in low
            ):
                return 3.0

        if goal.name=="request_information":
            if len(words(content))>=3:
                return 2.0

        return 1.0

    def _naturalness(self,content):
        score=0.0
        if content.endswith((".","!","?")):
            score+=2.0
        if len(content.split())>=3:
            score+=2.0
        return score

    def _brevity(self,content):
        n=len(content.split())
        if n<=8:
            return 2.0
        if n<=16:
            return 1.5
        if n<=30:
            return 1.0
        return 0.0

    def _parrot_penalty(self,user_text,content):
        u=words(user_text)
        c=words(content)
        if not u or not c:
            return 0.0
        if u==c:
            return 8.0
        overlap=len(u&c)/max(1,len(c))
        if overlap>=.85 and len(c-u)<=3:
            return 6.0
        return 0.0

    def evaluate(
        self,
        goal,
        user_text,
        content,
        source,
        memory,
        has_knowledge,
    ):
        goal_score=self._goal_score(goal,content)
        context_score=self._context_score(content,memory)
        evidence_score=self._evidence_score(
            source,
            has_knowledge,
        )
        progress_score=self._progress_score(
            goal,
            content,
        )
        naturalness_score=self._naturalness(content)
        brevity_score=self._brevity(content)

        total=(
            goal_score
            +context_score
            +evidence_score
            +progress_score
            +naturalness_score
            +brevity_score
            -self._parrot_penalty(
                user_text,
                content,
            )
        )

        return Candidate(
            source=source,
            action="answer",
            content=content,
            goal_score=goal_score,
            context_score=context_score,
            evidence_score=evidence_score,
            progress_score=progress_score,
            naturalness_score=naturalness_score,
            brevity_score=brevity_score,
        ),total

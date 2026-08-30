
from __future__ import annotations

from dataclasses import dataclass
import re

from model_utils import token_set as model_token_set,clean_generated,looks_meta
from response_firewall import is_internal


STOPWORDS={
    "the","a","an","is","are","am","be","to","of","and","or","in","on",
    "for","with","that","this","it","i","you","me","my","your","we","they",
    "he","she","do","does","did","what","how","why","when","where","who",
    "can","could","would","should","will","may","might","please","tell",
    "give","show","about","just","really","very","one","thing",
}


@dataclass
class ProposalAssessment:
    text: str
    valid: bool
    goal_score: float
    context_score: float
    evidence_score: float
    unsupported_score: float
    natural_score: float
    brevity_score: float
    contribution: float
    reason: str
    tokens: set[str]

    @property
    def accepted(self):
        return self.valid and self.contribution >= 0.55


def content_tokens(text):
    return {
        t for t in model_token_set(text)
        if t not in STOPWORDS and len(t)>2
    }


def _goal_terms(goal):
    terms=content_tokens(
        f"{goal.name.replace('_',' ')} {goal.description}"
    )
    return terms


def _user_terms(user_text):
    return content_tokens(user_text)


def _context_terms(context):
    return content_tokens(context)


def assess(goal,user_text,context,facts,text):
    text=clean_generated(text)
    toks=content_tokens(text)

    if not text:
        return ProposalAssessment(
            text,False,0,0,0,1,0,0,0,"empty",toks
        )

    if looks_meta(text) or is_internal(text):
        return ProposalAssessment(
            text,False,0,0,0,1,0,0,0,"internal_or_meta",toks
        )

    if "\n" in text or len(text.split())>80:
        brevity=0.2
    elif len(text.split())<=8:
        brevity=1.0
    elif len(text.split())<=20:
        brevity=0.8
    elif len(text.split())<=40:
        brevity=0.5
    else:
        brevity=0.25

    goal_terms=_goal_terms(goal)
    user_terms=_user_terms(user_text)
    context_terms=_context_terms(context)
    fact_terms=set()
    for f in facts[:12]:
        fact_terms |= content_tokens(
            f"{f.get('subject','')} "
            f"{f.get('predicate','')} "
            f"{f.get('object_text','')}"
        )

    goal_overlap=len(toks & goal_terms)/max(1,len(toks))
    user_overlap=len(toks & user_terms)/max(1,len(toks))
    context_overlap=len(toks & context_terms)/max(1,len(toks))
    evidence_overlap=len(toks & fact_terms)/max(1,len(toks))

    # For conversational goals, the user turn itself is the principal grounding.
    if goal.mode in {"social","share","continue"}:
        goal_score=max(goal_overlap,user_overlap*0.9)
    else:
        goal_score=max(goal_overlap,user_overlap*0.7,evidence_overlap*0.9)

    context_score=context_overlap
    evidence_score=evidence_overlap

    # Unsupported-claim detection is deliberately conservative:
    # if a factual request has no evidence and the proposal introduces concrete
    # nouns/numbers not present in the user/context, treat it as unsupported.
    concrete=toks-user_terms-context_terms
    unsupported=0.0
    factual_goal=goal.name in {
        "request_information",
        "request_explanation",
        "request_opinion",
        "challenge_claim",
    }

    if goal.name=="request_generation":
        # Generation is explicitly authorized to create novel content of the
        # requested type. The constraint is task fulfillment, not grounding.
        if len(text.split())>=3 and not text.endswith("?"):
            goal_score=1.0
        else:
            goal_score=0.2
        unsupported=0.0

    if factual_goal:
        if not facts and concrete:
            unsupported=1.0
        elif facts and concrete:
            support_ratio=len(concrete & fact_terms)/max(1,len(concrete))
            unsupported=1.0-support_ratio

    natural=1.0
    if text.endswith((".","!","?")):
        natural=1.0
    if re.search(
        r"\b(candidate|architecture|participant|realizer|goal description)\b",
        text.lower()
    ):
        natural=0.0

    # "Hello" is a valid social move even when lexical overlap with
    # "social_greeting" is low. Recognize basic act-fit markers.
    act_fit=0.0
    low=text.lower()
    if goal.name=="social_greeting" and re.search(
        r"\b(hello|hi|hey|greetings|nice to meet)\b",low
    ):
        act_fit=1.0
    elif goal.name=="social_thanks" and re.search(
        r"\b(welcome|pleasure|glad|no problem|you're welcome)\b",low
    ):
        act_fit=1.0
    elif goal.name=="social_goodbye" and re.search(
        r"\b(bye|goodbye|later|take care|see you)\b",low
    ):
        act_fit=1.0
    elif goal.name=="social_affection" and re.search(
        r"\b(glad|nice|appreciate|like|love|thank)\b",low
    ):
        act_fit=1.0

    goal_score=max(goal_score,act_fit)

    # Contribution is not a pure weighted sum: unsupported factual content
    # and zero goal contribution are gating failures.
    contribution=(
        0.40*goal_score
        +0.15*context_score
        +0.20*evidence_score
        +0.10*natural
        +0.05*brevity
        +0.10*(1.0-unsupported)
    )

    if goal.mode in {"social","share","continue"}:
        # Social turns don't need external evidence.
        contribution=(
            0.65*goal_score
            +0.15*context_score
            +0.10*natural
            +0.10*brevity
        )

    if goal.name=="request_generation":
        contribution=(
            0.75*goal_score
            +0.10*natural
            +0.10*brevity
            +0.05*(1.0-unsupported)
        )

    if unsupported>=0.95 and factual_goal:
        contribution=min(contribution,0.20)

    if goal_score<0.20:
        contribution=min(contribution,0.25)

    reason="accepted"
    valid=True
    if goal_score<0.20:
        reason="no_goal_contribution"
        valid=False if factual_goal or goal.name=="request_generation" else True
    elif factual_goal and unsupported>=0.95 and not facts:
        reason="unsupported_without_evidence"
        valid=False
    elif unsupported>0.75:
        reason="mostly_unsupported"
        valid=False

    return ProposalAssessment(
        text=text,
        valid=valid,
        goal_score=goal_score,
        context_score=context_score,
        evidence_score=evidence_score,
        unsupported_score=unsupported,
        natural_score=natural,
        brevity_score=brevity,
        contribution=contribution,
        reason=reason,
        tokens=toks,
    )

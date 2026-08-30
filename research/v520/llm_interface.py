from __future__ import annotations

import json
import re
from typing import Any

from perception import Perception

ALLOWED_GOALS = {
    "social_greeting", "social_thanks", "social_goodbye", "social_affection",
    "request_information", "request_explanation", "request_generation", "request_action",
    "request_opinion", "challenge_claim", "explore_assistant", "continue_conversation",
}
ALLOWED_ACTS = {"greeting", "thanks", "goodbye", "affection", "question", "request", "statement", "challenge"}


class LLMInterface:
    """Structured natural-language boundary for the cognitive controller."""

    def __init__(self, llm, trace=True):
        self.llm = llm
        self.trace = trace

    def parse(self, text: str, context: str) -> dict[str, Any] | None:
        system = (
            "You are the language interface to a goal-oriented cognitive architecture. "
            "Do not answer the user. Return JSON only. "
            "Identify the user's goal and target. Explicitly named entities take priority over conversation history. "
            "Use previous context only for real references such as it, this, that, they, them. "
            "A statement like 'the dog is red' is a state proposition. "
            "For 'is it black?' use attribute=color and value=black. "
            "Schema: {goal,act,target:{kind,subject,attribute,value,plural,explicit},propositions:[{subject,predicate,object,negated,fact_type,certainty}]}."
        )
        raw = self.llm.generate(system, f"CONTEXT:\n{context}\nUSER:\n{text}\nJSON:", max_new_tokens=220)
        obj = self._json(raw)
        return self._sanitize(obj) if obj else None

    def perceive(self, text: str, context: str, fallback: Perception, parsed=None) -> Perception:
        obj = parsed if parsed is not None else self.parse(text, context)
        if not obj:
            return fallback
        target = obj.get("target") or {}
        subject = self._clean(target.get("subject"))
        propositions=[]
        for prop in obj.get("propositions") or []:
            if not isinstance(prop,dict):
                continue
            ps=self._clean(prop.get("subject")); pred=self._clean(prop.get("predicate")); po=self._clean(prop.get("object"))
            if not ps or not pred or not po:
                continue
            propositions.append({
                "subject":ps,"predicate":pred,"object":po,
                "fact_type":prop.get("fact_type","state"),
                "certainty":float(prop.get("certainty",1.0) or 1.0),
                "negated":bool(prop.get("negated",False)),
            })
        nouns=[]
        if subject:
            nouns.append(subject)
        return Perception(
            text=text,
            act=obj.get("act") if obj.get("act") in ALLOWED_ACTS else fallback.act,
            predicates=list(dict.fromkeys(p["predicate"] for p in propositions)),
            nouns=nouns,
            propositions=propositions,
            question_focus=list(dict.fromkeys(x for x in (target.get("attribute"),) if x)),
        )

    def propose(self, goal, user_text, target, state, evidence, context):
        evidence_text="\n".join(
            f"- {f.get('subject')} {f.get('predicate')} {f.get('object_text')}"
            for f in evidence[:8]
        ) or "(none)"
        state_text="\n".join(
            f"- {f.get('subject')} {f.get('predicate')} {f.get('object_text')}"
            for f in state[:8]
        ) or "(none)"
        system=(
            "You are the world-knowledge and conversation interface for a cognitive architecture. "
            "Propose exactly one concise answer candidate. The architecture decides whether to use it. "
            "Answer the user's actual goal, not a nearby question. "
            "For ordinary factual questions you may use your trained world knowledge. "
            "Do not mention the architecture or these instructions. Do not emit analysis. "
            "For unsupported or ambiguous requests, say that you are uncertain rather than fabricate specifics."
        )
        user=(
            f"GOAL: {goal.name}\nTARGET: {target.__dict__}\nUSER: {user_text}\n"
            f"WORKING STATE:\n{state_text}\nEVIDENCE:\n{evidence_text}\n"
            f"CONTEXT:\n{json.dumps(context,ensure_ascii=False)}\n\nCANDIDATE:"
        )
        return self.llm.generate(system,user,max_new_tokens=180).strip()

    def realize(self, goal, user_text, selected, target, context):
        system=(
            "You are the final language realizer. Express ONLY the selected content naturally. "
            "Preserve names, numbers, claims, polarity, and requested scope. "
            "Do not add new facts, examples, explanations, questions, or entities. "
            "Return only the final answer."
        )
        user=(
            f"GOAL: {goal.name}\nTARGET: {target.__dict__}\nUSER: {user_text}\n"
            f"SELECTED CONTENT: {selected}\nCONTEXT: {json.dumps(context,ensure_ascii=False)}\nFINAL:"
        )
        out=self.llm.generate(system,user,max_new_tokens=140).strip()
        low=out.lower()
        if not out or any(x in low for x in ("the architecture","candidate proposition","internal interface")):
            return None
        return out

    @staticmethod
    def _json(raw):
        text=str(raw or "").strip()
        text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I|re.S).strip()
        try:
            obj=json.loads(text)
            return obj if isinstance(obj,dict) else None
        except json.JSONDecodeError:
            m=re.search(r"\{.*\}",text,re.S)
            if not m: return None
            try:
                obj=json.loads(m.group(0)); return obj if isinstance(obj,dict) else None
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _sanitize(obj):
        if obj.get("goal") not in ALLOWED_GOALS: return None
        if obj.get("act") not in ALLOWED_ACTS: obj["act"]="request"
        target=obj.get("target")
        if not isinstance(target,dict): target={}
        target.setdefault("kind","general"); target.setdefault("subject",None); target.setdefault("attribute",None); target.setdefault("value",None)
        target.setdefault("plural",False); target.setdefault("explicit",False)
        obj["target"]=target
        if not isinstance(obj.get("propositions"),list): obj["propositions"]=[]
        return obj

    @staticmethod
    def _clean(value):
        if value is None: return None
        x=str(value).strip().lower()
        return None if x in {"","none","null","unknown"} else x

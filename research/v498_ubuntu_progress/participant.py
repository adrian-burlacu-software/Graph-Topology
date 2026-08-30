
from __future__ import annotations

import re

from model_utils import clean_generated,looks_meta


class Participant:
    def __init__(self,llm):
        self.llm=llm

    def propose(self,goal,user_text,context,facts):
        evidence="\n".join(
            f"- {f['subject']} {f['predicate']} {f['object_text']}"
            for f in facts[:8]
        )

        raw=self.llm.generate(
            "You are an internal cognitive participant. "
            "Propose useful content for the architecture. "
            "Return only the content itself. "
            "Never give instructions to the user. "
            "Never describe the architecture or your role.",
            (
                f"GOAL: {goal.name}\n"
                f"USER: {user_text}\n"
                f"STATE:\n{context}\n"
                f"RELEVANT EVIDENCE:\n{evidence}\n"
                "Return one concise proposition or conversational move."
            ),
        )

        text=clean_generated(raw)

        # Small-model protocol normalization.
        prefixes=(
            "a useful candidate proposition is:",
            "a useful candidate proposition could be:",
            "one useful candidate is:",
            "proposal:",
            "answer:",
            "response:",
        )
        low=text.lower()
        for p in prefixes:
            if low.startswith(p):
                text=text[len(p):].strip()
                break

        if looks_meta(text):
            return None

        if re.match(
            r"^(you can|you could|you should|start by|try |say |tell )",
            text.lower(),
        ):
            return None

        if goal.name in {
            "request_information","request_explanation",
            "request_opinion","challenge_claim",
        } and text.endswith("?"):
            return None

        return text or None

    def realize(self,goal,selected,context):
        raw=self.llm.generate(
            "You are only a language realizer. "
            "Preserve the selected content exactly in meaning. "
            "Do not add facts, claims, or questions. "
            "Return only the final natural-language reply.",
            (
                f"GOAL: {goal.name}\n"
                f"SELECTED CONTENT: {selected}\n"
                f"CONTEXT:\n{context}"
            ),
        )
        text=clean_generated(raw)
        if not text or looks_meta(text):
            return None

        # Require lexical grounding in the selected content.
        a=set(re.findall(r"[a-z0-9']+",selected.lower()))
        b=set(re.findall(r"[a-z0-9']+",text.lower()))
        overlap=len(a&b)/max(1,len(a))
        if overlap < 0.35:
            return None
        return text

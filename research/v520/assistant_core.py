from __future__ import annotations

import re

from goals import GOALS, infer_goal
from relevance import rank_static_facts
from operators import OperatorEngine
from planner import Planner
from response_firewall import final_text, is_internal
from query_target import QueryTarget, infer_target
from evidence_selector import select_evidence, combine_relevant_facts


class CognitiveAssistant:
    """Goal-oriented controller with an explicit LLM interface boundary.

    The architecture owns:
      - goal selection
      - target/state tracking
      - deterministic operators
      - memory scope
      - evidence selection
      - final candidate selection

    The LLM interface may:
      - parse the natural-language request into structured intent
      - propose world/conversational content
      - realize architecture-selected content

    It does not get to choose the next cognitive action or mutate memory by
    itself.
    """

    def __init__(self, memory, perceiver, llm=None, trace=True, interface=None):
        self.memory = memory
        self.perceiver = perceiver
        self.llm = llm
        self.interface = interface
        self.operators = OperatorEngine()
        self.planner = Planner(memory, trace=trace)
        self.trace = trace
        self.last_interface = None

    def reset(self):
        self.memory.reset_session()
        self.last_interface = None

    def _topic(self, p):
        for prop in reversed(p.propositions):
            if prop.get("subject"):
                return prop["subject"]
        concrete = [
            n for n in p.nouns
            if n not in {"color", "colour", "thing", "question", "answer", "domain", "user", "assistant"}
        ]
        if concrete:
            return concrete[-1]
        return self.memory.topic

    def _reference(self, text):
        low = text.lower()
        if not re.search(r"\b(it|this|that|they|them)\b", low):
            return None
        return self.memory.topic

    def _llm_context(self, goal, text):
        return {
            "goal": goal.name,
            "topic": self.memory.topic,
            "previous_user": self.memory.last_user,
            "previous_assistant": self.memory.last_assistant,
            "facts": self.memory.facts()[:12],
            "knowledge_frozen": self.memory.knowledge_is_frozen(),
        }

    def _is_assertive_proposition(self, prop, p, text):
        subject = str(prop.get("subject", "")).strip().lower()
        predicate = str(prop.get("predicate", "")).strip().lower()
        obj = str(prop.get("object", "")).strip().lower()
        low = text.strip().lower()
        if not subject or not predicate or not obj or low.endswith("?"):
            return False
        if low.startswith((
            "how ", "what ", "which ", "who ", "where ", "when ", "why ",
            "is ", "are ", "do ", "does ", "did ", "can ", "could ",
            "would ", "should ", "will ", "have ", "has ", "what if ",
        )):
            return False
        if re.search(r"\b(?:there is|there are|there's|i am|i'm|it is|it's|this is|that is)\b", low):
            return True
        return p.act in {"statement", "assertion", "declaration"}

    @staticmethod
    def _subject_is_explicit(text, subject):
        if not subject:
            return False
        return bool(re.search(rf"\b{re.escape(subject)}(?:s)?\b", text.lower()))

    @staticmethod
    def _safe_interface_candidate(text):
        low = str(text or "").strip().lower()
        if not low or is_internal(low):
            return False
        if re.search(r"\[[A-Z][A-Z _→←-]*\]", text):
            return False
        return True

    def respond(self, text):
        fallback_p = self.perceiver.perceive(text)
        interface_parse = None
        if self.interface is not None:
            try:
                interface_parse = self.interface.parse(text, self.memory.context(limit=10))
            except Exception as exc:
                if self.trace:
                    print(f"  [LLM INTERFACE] parse failed: {exc}", flush=True)

        if interface_parse:
            p = self.interface.perceive(text, self.memory.context(limit=10), fallback_p, parsed=interface_parse)
        else:
            p = fallback_p
        self.last_interface = interface_parse

        goal_name = (interface_parse or {}).get("goal")
        goal = GOALS.get(goal_name) if goal_name in GOALS else infer_goal(
            text,
            p.act,
            previous_assistant=self.memory.last_assistant,
            previous_goal=self.memory.goal,
        )

        topic = self._topic(p)
        reference = self._reference(text)
        target = infer_target(text, topic)

        if interface_parse:
            raw = interface_parse.get("target") or {}
            if isinstance(raw, dict):
                target = QueryTarget(
                    kind=raw.get("kind") or target.kind,
                    subject=raw.get("subject") or target.subject,
                    attribute=raw.get("attribute") or target.attribute,
                    value=raw.get("value") if raw.get("value") is not None else target.value,
                    plural=bool(raw.get("plural", target.plural)),
                    qualifier=raw.get("qualifier"),
                    explicit=bool(raw.get("explicit", target.explicit)),
                )

        self.memory.set_context(text, goal.name, target.subject or topic)
        self.memory.add_live_turn("user", text)

        for prop in p.propositions:
            if not self._is_assertive_proposition(prop, p, text):
                continue
            self.memory.add_live_fact(prop)

        if self.trace:
            print(f"[FRAME] goal={goal.name} act={p.act} target={target.__dict__}", flush=True)
            print(f"[KNOWLEDGE POLICY] frozen={self.memory.knowledge_is_frozen()}", flush=True)
            if interface_parse:
                print("[LLM INTERFACE] structured request accepted", flush=True)

        # 1. Architecture-native deterministic answer always wins.
        logical = self.operators.answer(text, self.memory, target=target)
        if logical is not None:
            architecture_content = getattr(logical, "answer", logical)
            source = getattr(logical, "kind", "architecture")
        else:
            architecture_content = None
            source = None

        # 2. Long-term semantic memory is optional. A frozen memory means the
        # graph is not the world-knowledge oracle for this turn; the LLM can be.
        targeted = []
        if not self.memory.knowledge_is_frozen() and target.subject:
            terms = [target.subject]
            static = self.memory.static_facts(terms, goal.name, domains=None, limit=32)
            static = rank_static_facts(static, goal.name, terms, topic=self.memory.topic, max_items=24)
            targeted = select_evidence(static, target, max_items=6)

        knowledge_content = combine_relevant_facts(targeted) if targeted else None
        if not architecture_content and knowledge_content and goal.name in {"request_information", "request_explanation"}:
            architecture_content = knowledge_content
            source = "knowledge"

        # 3. LLM is now the world/conversation interface. It proposes content;
        # the architecture accepts it only after checking the cognitive target.
        interface_candidate = None
        if self.interface is not None and not architecture_content:
            try:
                interface_candidate = self.interface.propose(
                    goal=goal,
                    user_text=text,
                    target=target,
                    state=self.memory.facts()[:12],
                    evidence=targeted,
                    context=self._llm_context(goal, text),
                )
            except Exception as exc:
                if self.trace:
                    print(f"[LLM INTERFACE] proposal failed: {exc}", flush=True)

            if interface_candidate and self._safe_interface_candidate(interface_candidate):
                # Explicit entity requests must be reflected in the proposal.
                if target.subject and target.explicit:
                    if target.subject.lower() not in interface_candidate.lower():
                        interface_candidate = None
                if target.value and goal.name == "challenge_claim":
                    if target.value.lower() not in interface_candidate.lower():
                        interface_candidate = None

        selected = architecture_content or interface_candidate
        if architecture_content:
            source = source or "architecture"
        elif interface_candidate:
            source = "llm_interface"
        else:
            source = "fallback"
            selected = {
                "social_greeting": "Hello!",
                "social_thanks": "You're welcome!",
                "social_goodbye": "Bye!",
                "social_affection": "That's nice to hear.",
                "explore_assistant": "I'm thinking about our conversation.",
                "request_information": "I'm not sure yet.",
                "request_explanation": "I'm not sure yet.",
                "request_generation": "Sure.",
                "request_action": "Sure. What would you like me to do?",
                "request_opinion": "I'm not sure yet.",
                "challenge_claim": "I'm not certain yet.",
                "continue_conversation": "Tell me more.",
            }.get(goal.name, "I'm not sure yet.")

        final = selected
        if self.interface is not None and source in {"llm_interface", "knowledge"}:
            realized = self.interface.realize(
                goal=goal,
                user_text=text,
                selected=selected,
                target=target,
                context=self._llm_context(goal, text),
            )
            if realized:
                final = realized
                if self.trace:
                    print("[LLM REALIZER] accepted", flush=True)
            elif self.trace:
                print("[LLM REALIZER] rejected; selected content retained", flush=True)

        fallback = selected if source != "fallback" else {
            "social_greeting": "Hello!",
            "social_thanks": "You're welcome!",
            "social_goodbye": "Bye!",
            "social_affection": "That's nice to hear.",
            "explore_assistant": "I'm thinking about our conversation.",
            "request_information": "I'm not sure yet.",
            "request_explanation": "I'm not sure yet.",
            "request_generation": "Sure.",
            "request_action": "Sure. What would you like me to do?",
            "request_opinion": "I'm not sure yet.",
            "challenge_claim": "I'm not certain yet.",
            "continue_conversation": "Tell me more.",
        }.get(goal.name, "I'm not sure yet.")

        final = final_text(final, fallback)
        self.memory.add_assistant_turn(final)
        self.memory.remember_answer(source, final)
        self.memory.turn_index += 1

        if self.trace:
            print(
                f"[DECISION] source={source} target={target.kind} subject={target.subject}"
            )

        return {
            "response": final,
            "source": source,
            "goal": goal.name,
            "target": target.kind,
            "target_subject": target.subject,
            "target_attribute": target.attribute,
            "target_value": target.value,
            "knowledge_frozen": self.memory.knowledge_is_frozen(),
            "logic_kind": getattr(logical, "kind", "architecture" if logical else None),
        }

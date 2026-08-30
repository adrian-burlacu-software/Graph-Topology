from __future__ import annotations

import re
from dataclasses import replace

from goals import GOALS, infer_goal
from relevance import rank_static_facts
from operators import OperatorEngine
from planner import Planner
from response_firewall import final_text, is_internal
from query_target import QueryTarget, infer_target
from evidence_selector import select_evidence, compact_evidence, combine_relevant_facts


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

    @staticmethod
    def _needs_reference_context(text):
        return bool(re.search(
            r"\b(?:it|this|that|they|them|he|she|there|here|also|too|again|more|same|previous|earlier|before|still)\b",
            text.lower(),
        ))

    @staticmethod
    def _subject_mentioned(text, subject):
        if not subject:
            return False
        base = re.escape(str(subject).strip().lower())
        return bool(re.search(rf"\b{base}(?:s|es)?\b", text.lower()))

    @staticmethod
    def _candidate_mentions_subject(candidate, subject):
        return CognitiveAssistant._subject_mentioned(candidate, subject)

    def respond(self, text):
        fallback_p = self.perceiver.perceive(text)
        p = fallback_p
        self.last_interface = None

        goal = infer_goal(
            text,
            p.act,
            previous_assistant=self.memory.last_assistant,
            previous_goal=self.memory.goal,
        )

        topic = self._topic(p)
        reference = self._reference(text)
        if reference:
            topic = reference
        target = infer_target(text, topic)

        self.memory.set_context(text, goal.name, target.subject or topic)
        self.memory.add_live_turn("user", text)

        for prop in p.propositions:
            if not self._is_assertive_proposition(prop, p, text):
                continue
            self.memory.add_live_fact(prop)

        if self.trace:
            print(f"[FRAME] goal={goal.name} act={p.act} target={target.__dict__}", flush=True)
            print(f"[KNOWLEDGE POLICY] frozen={self.memory.knowledge_is_frozen()}", flush=True)

        # Architecture-native deterministic logic is authoritative. Its output becomes
        # structured semantic answer data for the LLM renderer.
        logical = self.operators.answer(text, self.memory, target=target)
        architecture_content = getattr(logical, "answer", logical) if logical is not None else None
        logic_kind = getattr(logical, "kind", None) if logical is not None else None

        # ------------------------------------------------------------------
        # Query-scoped knowledge retrieval.
        # Only evidence that can answer the current typed target is allowed to
        # cross the LLM boundary. Conversation state and broad topic facts are
        # deliberately excluded here.
        # ------------------------------------------------------------------
        evidence = []
        retrieval_terms = []
        for term in [target.subject, target.qualifier]:
            term = str(term or "").strip().lower()
            if term and term not in retrieval_terms:
                retrieval_terms.append(term)

        if retrieval_terms and self.memory.knowledge_is_frozen():
            try:
                static = self.memory.static_facts(
                    retrieval_terms, goal.name, domains=None, limit=48
                )
                # Ranking may consider query terms, but the selector is the
                # final authority on whether a fact is actually admissible.
                static = rank_static_facts(
                    static, goal.name, retrieval_terms,
                    topic=None, max_items=48
                )
                evidence = select_evidence(
                    static, target, max_items=8,
                    context_subject=target.qualifier,
                )
            except Exception as exc:
                if self.trace:
                    print(f"[KNOWLEDGE RETRIEVAL] failed: {exc}", flush=True)

        # De-duplicate evidence before presenting it to the model/debugger.
        unique_evidence = []
        seen_evidence = set()
        for fact in evidence:
            key=(
                fact.get("subject"),
                fact.get("predicate"),
                fact.get("object_text"),
                fact.get("fact_type"),
            )
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            unique_evidence.append(fact)
        evidence = unique_evidence[:8]
        llm_evidence = compact_evidence(evidence)

        fallback = {
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

        structured_request = {
            "version": "v531",
            "goal": goal.name,
            "act": p.act,
            "target": target.__dict__,
            "answer": architecture_content,
            "answer_kind": logic_kind,
            "evidence": llm_evidence,
            "state": [],
            "policy": {
                "architecture_is_semantic_authority": True,
                "render_only": True,
                "do_not_invent_supported_content": True,
                "answer_required_for_direct_claim": True,
                "knowledge_frozen": self.memory.knowledge_is_frozen(),
            },
        }

        if self.trace:
            print(f"[ARCHITECTURE EVIDENCE] count={len(evidence)}", flush=True)
            for idx, fact in enumerate(evidence, 1):
                print(f"  [{idx}] {fact}", flush=True)
            print(f"[ARCHITECTURE ANSWER] kind={logic_kind} value={architecture_content!r}", flush=True)

        rendered = None
        renderer_allowed = (
            self.interface is not None
            and (
                architecture_content is not None
                or bool(llm_evidence)
                or goal.name in {"social_greeting", "social_thanks", "social_goodbye", "social_affection", "explore_assistant", "continue_conversation"}
            )
        )
        if renderer_allowed:
            rendered = self.interface.render(structured_request)

        if rendered and not is_internal(rendered):
            final = final_text(rendered, "I don't know.")
            source = "llm_renderer"
        elif architecture_content is not None:
            final = final_text(str(architecture_content), fallback)
            source = logic_kind or "architecture"
        else:
            if goal.name in {"request_information", "request_explanation", "challenge_claim", "request_opinion"}:
                final = "I don't know."
            else:
                final = fallback
            source = "fallback"

        self.memory.add_assistant_turn(final)
        self.memory.remember_answer(source, final)
        self.memory.turn_index += 1

        if self.trace:
            print(f"[DECISION] source={source} target={target.kind} subject={target.subject}", flush=True)

        return {
            "response": final,
            "source": source,
            "goal": goal.name,
            "target": target.kind,
            "target_subject": target.subject,
            "target_attribute": target.attribute,
            "target_value": target.value,
            "knowledge_frozen": self.memory.knowledge_is_frozen(),
            "logic_kind": logic_kind,
        }

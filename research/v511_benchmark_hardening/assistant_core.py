
from __future__ import annotations

import re

from goals import infer_goal
from relevance import rank_static_facts,relation_text
from operators import OperatorEngine
from planner import Planner
from surface import fact_to_sentence
from response_firewall import final_text,is_internal
from participant import Participant,ParticipantProposal
from query_target import infer_target
from evidence_selector import select_evidence,combine_relevant_facts


class CognitiveAssistant:
    def __init__(
        self,
        memory,
        perceiver,
        llm=None,
        trace=True,
    ):
        self.memory=memory
        self.perceiver=perceiver
        self.llm=llm
        self.participant=Participant(llm) if llm else None
        self.operators=OperatorEngine()
        self.planner=Planner(memory,trace=trace)
        self.trace=trace
        self._last_attention={}

    def reset(self):
        self.memory.reset_session()
        self._last_attention={}

    def _topic(self,p):
        # Prefer concrete proposition subjects over adjective/abstract nouns.
        for prop in reversed(p.propositions):
            if prop.get("subject"):
                return prop["subject"]
        concrete=[
            n for n in p.nouns
            if n not in {
                "color","colour","thing","question","answer",
                "domain","user","assistant",
            }
        ]
        if concrete:
            return concrete[-1]
        return self.memory.topic

    def _reference(self,text):
        low=text.lower()

        if not re.search(r"\b(it|this|that|they|them)\b",low):
            return None

        # A pronoun in "what color is it?" should resolve to the prior
        # concrete topic, not the word "color".
        if self.memory.topic:
            return self.memory.topic

        return None

    def _llm_context(self,goal,text):
        if goal.name=="request_generation":
            return {
                "goal":goal.name,
                "request":text,
            }

        return {
            "goal":self.memory.goal,
            "topic":self.memory.topic,
            "previous_user":self.memory.last_user,
            "previous_assistant":self.memory.last_assistant,
            "facts":self.memory.facts()[:12],
        }

    def _is_assertive_proposition(self,prop,p,text):
        subject=str(prop.get("subject","")).strip().lower()
        predicate=str(prop.get("predicate","")).strip().lower()
        obj=str(prop.get("object","")).strip().lower()
        low=text.strip().lower()

        if not subject or not predicate or not obj:
            return False

        if low.endswith("?"):
            return False

        if low.startswith((
            "how many ","what ","which ","who ","where ","when ","why ",
            "how ","is ","are ","do ","does ","did ","can ","could ",
            "would ","should ","will ","have ","has ","isn't ","aren't ",
        )):
            return False

        if re.search(
            r"\b(?:there is|there are|there's|i am|i'm|it is|it's|"
            r"this is|that is)\b",
            low,
        ):
            return True

        if predicate in {"be","do","have"}:
            return False

        return p.act in {"statement","assertion","declaration"}

    def respond(self,text):
        p=self.perceiver.perceive(text)
        goal=infer_goal(
            text,
            p.act,
            previous_assistant=self.memory.last_assistant,
            # The current stored goal belongs to the previous completed turn.
            # Use it for elliptical follow-ups like "I'm good and you?".
            previous_goal=self.memory.goal,
        )
        topic=self._topic(p)
        reference=self._reference(text)
        target=infer_target(text,topic)

        previous_attention=self._last_attention or None
        target_signature={
            "kind": target.kind,
            "subject": target.subject,
            "attribute": target.attribute,
            "value": target.value,
        }
        target_changed = bool(
            previous_attention
            and previous_attention.get("target") != target_signature
        )

        # Track explicit new entities instead of treating every property fact
        # as a new object.
        if (
            any(
                self._is_assertive_proposition(prop,p,text)
                for prop in p.propositions
            )
            and target.subject
            and target.subject not in {"question","thing"}
        ):
            low=text.lower()
            is_new=bool(
                re.search(
                    rf"\b(?:another|an additional|a different)\s+{re.escape(target.subject)}\b",
                    low,
                )
            )
            self.memory.mention_entity(target.subject,is_new)

        self.memory.set_context(text,goal.name,topic)
        self.memory.add_live_turn("user",text)

        for prop in p.propositions:
            if not self._is_assertive_proposition(prop,p,text):
                if self.trace:
                    print(
                        f"  [STATE] rejected "
                        f"{prop['subject']} --{prop['predicate']}--> "
                        f"{prop.get('object')}",
                        flush=True,
                    )
                continue

            self.memory.add_live_fact(prop)
            if self.trace:
                sign="not " if prop.get("negated") else ""
                print(
                    f"  [LIVE STATE] {prop['subject']} "
                    f"--{prop['predicate']}--> "
                    f"{sign}{prop['object']}",
                    flush=True,
                )

        if self.trace:
            print(
                f"  [KNOWLEDGE POLICY] frozen={self.memory.knowledge_is_frozen()}",
                flush=True,
            )
            print(
                f"  [PERCEPTION] act={p.act} "
                f"predicates={p.predicates} nouns={p.nouns}",
                flush=True,
            )
            print(
                f"  [GOAL] {goal.name}: {goal.description}",
                flush=True,
            )
            if reference:
                print(f"  [REFERENCE] {reference}",flush=True)

        # Architecture-level conversational state answers must outrank static
        # lexical knowledge. "I'm good and you?" is about the assistant state,
        # not the WordNet meaning of "be".
        state_answer=self.operators.answer(text,self.memory,target=target)

        if goal.name == "explore_assistant":
            low=text.lower()

            if (
                "and you" in low
                and self.memory.goal in {"social_greeting","explore_assistant"}
            ):
                # Reciprocal elliptical turn:
                #   Assistant: Hello!
                #   User: I'm good and you?
                state_answer = "I'm doing well. How about you?"
            elif "how are you" in low or "how's it going" in low:
                state_answer = "I'm doing well. How about you?"
            elif "what are you curious" in low:
                state_answer = "I'm curious about where our conversation might go."
            elif "what are you thinking" in low or "what's on your mind" in low:
                state_answer = "I'm thinking about what we should explore next."
            elif "what do you want to do" in low:
                state_answer = "I'd like to keep chatting and see where this goes."

        static_terms=list(dict.fromkeys(
            p.nouns+
            [
                x for x in p.predicates
                if x not in {"be","have","do"}
            ]+
            ([reference] if reference else [])+
            ([target.subject] if target.subject else [])
        ))

        if self.memory.knowledge_is_frozen() or goal.name=="request_generation":
            static=[]
        else:
            static=self.memory.static_facts(
                static_terms,
                goal.name,
                domains=None,
                limit=32,
            )
        static=rank_static_facts(
            static,
            goal.name,
            static_terms,
            topic=self.memory.topic,
            max_items=24,
        )

        targeted=select_evidence(
            static,
            target,
            max_items=6,
        )

        if self.trace:
            print(
                f"  [TARGET] kind={target.kind} "
                f"subject={target.subject} "
                f"attribute={target.attribute} "
                f"value={target.value}",
                flush=True,
            )
            print(
                f"  [RELEVANCE] static_raw={len(static)} "
                f"targeted={len(targeted)}",
                flush=True,
            )
            for f in targeted[:6]:
                print(
                    f"    TARGET-MATCH {f['subject']} "
                    f"--{f['predicate']}--> {f['object_text']} "
                    f"type={f['fact_type']} "
                    f"score={f['_target_score']:.2f}",
                    flush=True,
                )

        state_contents=[]
        if state_answer:
            state_contents.append(state_answer)

        if not state_contents and not targeted:
            if target.kind=="property" and target.attribute=="size":
                state_contents.append(
                    f"I don't know how large {target.subject or 'it'} is yet."
                )
            elif target.kind=="property" and target.attribute=="color":
                state_contents.append(
                    f"I don't know what color {target.subject or 'it'} is yet."
                )
            elif target.kind=="property" and target.attribute=="shape":
                state_contents.append(
                    f"I don't know what shape {target.subject or 'it'} is yet."
                )

        knowledge_contents=[]
        if targeted:
            combined=combine_relevant_facts(targeted)
            if combined:
                knowledge_contents.append(combined)

        # Conversational goals do not consume static lexical evidence.
        if goal.mode in {"social","share","continue"} or goal.name=="explore_assistant":
            knowledge_contents=[]

        participant_proposal=None
        context=self._llm_context(goal,text)

        should_consult_participant=(
            self.participant is not None
            and (
                goal.mode in {"social","share","continue","generate"}
                or bool(targeted)
                or goal.name in {"request_action","request_opinion"}
            )
            and not (
                self.memory.knowledge_is_frozen()
                and goal.mode=="answer"
                and not state_contents
                and not targeted
            )
        )

        if should_consult_participant:
            if self.trace:
                print("  [ARCHITECTURE → PARTICIPANT]",flush=True)
            participant_proposal=self.participant.propose(
                goal,text,context,targeted
            )
            if self.trace:
                print(
                    f"  [PARTICIPANT] "
                    f"{participant_proposal!r}",
                    flush=True,
                )

        winner=self.planner.choose(
            goal,
            text,
            state_contents,
            knowledge_contents,
            participant_proposal,
        )

        selected=winner.content
        final=selected

        # The LLM realizer is ONLY allowed to rewrite content that originated
        # from an accepted participant proposal. Architecture-owned fallback,
        # state, and knowledge content must not be handed to a free generator.
        if self.participant and winner.source in {"participant","knowledge"}:
            if self.trace:
                print("  [ARCHITECTURE → REALIZER]",flush=True)

            grounded_selected_facts=targeted if winner.source=="knowledge" else targeted

            realized=self.participant.realize(
                goal,
                selected,
                context,
                facts=grounded_selected_facts,
            )

            if realized and not is_internal(realized):
                final=realized
                if self.trace:
                    print(
                        f"  [REALIZER] accepted {realized!r}",
                        flush=True,
                    )
            elif self.trace:
                print(
                    "  [REALIZER] rejected; architecture-selected "
                    "content retained",
                    flush=True,
                )
        elif self.trace and self.participant:
            print(
                f"  [REALIZER] skipped source={winner.source}",
                flush=True,
            )

        # HARD PUBLIC-OUTPUT FIREWALL.
        # Nothing internal can cross this boundary.
        public_fallback={
            "social_greeting":"Hello!",
            "social_thanks":"You're welcome!",
            "social_goodbye":"Bye!",
            "social_affection":"That's nice to hear.",
            "explore_assistant":"I'm thinking about our conversation.",
            "request_information":"I'm not sure yet.",
            "request_explanation":"I'm not sure yet.",
            "request_generation":"Sure.",
            "request_action":"Sure. What would you like me to do?",
            "request_opinion":"I'm not sure yet.",
            "challenge_claim":"I'm not certain yet.",
            "continue_conversation":"Tell me more.",
        }.get(goal.name,"I'm not sure yet.")

        final=final_text(final,public_fallback)

        self.memory.add_assistant_turn(final)
        self.memory.remember_answer(winner.source,final)
        self.memory.turn_index+=1

        if self.trace:
            print(
                f"  [DECISION] source={winner.source} score={winner.score:.2f}",
                flush=True,
            )

        planner_trace=dict(self.planner.last_trace or {})
        attention={
            "turn_index": self.memory.turn_index,
            "goal": goal.name,
            "topic": self.memory.topic,
            "target": target_signature,
            "target_changed": target_changed,
            "previous": previous_attention,
            "planner": planner_trace,
            "selected_source": winner.source,
            "selected_content": winner.content,
            "selected_score": winner.score,
            "switch_cost_proxy": (
                1 if target_changed and winner.source != "state" else 0
            ),
        }
        self._last_attention=attention

        return {
            "response":final,
            "source":winner.source,
            "score":winner.score,
            "goal":goal.name,
            "target":target.kind,
            "target_signature":target_signature,
            "target_changed":target_changed,
            "attention":attention,
        }

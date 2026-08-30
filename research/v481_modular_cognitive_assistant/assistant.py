
from __future__ import annotations

import re
import time

from goals import infer_goal
from memory import digest
from evaluator import CandidateEvaluator
from knowledge import KnowledgeStore
from realizer import Realizer
from planner import GoalPlanner


class CognitiveAssistant:
    def __init__(
        self,
        con,
        perceiver,
        memory,
        knowledge,
        participant,
        trace=True,
    ):
        self.con=con
        self.perceiver=perceiver
        self.memory=memory
        self.knowledge=knowledge
        self.participant=participant
        self.evaluator=CandidateEvaluator()
        self.realizer=Realizer()
        self.planner=GoalPlanner(
            self.evaluator,
            participant,
            self.realizer,
            memory,
            trace=trace,
        )
        self.trace=trace
        self.turn_index=0

    def _context(self):
        state=self.memory.state()
        recent=self.memory.recent_dialogue()
        facts=self.memory.facts()

        lines=[]

        if state.topic:
            lines.append(f"CURRENT TOPIC: {state.topic}")
        if state.goal:
            lines.append(f"CURRENT GOAL: {state.goal}")

        if state.last_user_text:
            lines.append(
                f"LAST USER: {state.last_user_text}"
            )

        if state.last_assistant_text:
            lines.append(
                f"LAST ASSISTANT: {state.last_assistant_text}"
            )

        if facts:
            rendered=[]
            for kind,subject,predicate,value,salience in facts[:8]:
                rendered.append(
                    f"{subject} {predicate} {value}"
                )
            if rendered:
                lines.append(
                    "MEMORY: "+"; ".join(rendered)
                )

        if recent:
            lines.append(
                "RECENT: "
                +" | ".join(
                    f"{speaker}: {text}"
                    for speaker,text in recent[-8:]
                )
            )

        return "\n".join(lines)

    def _store_final_turn(
        self,
        user_text,
        perception,
        goal,
        response,
        winner,
    ):
        turn_id=digest(
            "user_turn",
            self.memory.session_id,
            self.turn_index,
            user_text,
        )

        self.memory.store_turn(
            turn_id,
            self.turn_index,
            "user",
            user_text,
            {
                "perception":perception.__dict__,
                "goal":goal.__dict__,
            },
            "goal_directed",
            winner.total,
        )

        self.memory.store_turn(
            digest("assistant_turn",turn_id),
            self.turn_index,
            "assistant",
            response,
            {
                "source":winner.source,
                "selected_content":winner.content,
            },
            winner.action,
            winner.total,
        )

        self.memory.update_assistant(
            response,
            winner.action,
        )

        self.turn_index+=1

    def respond(self,text):
        perception=self.perceiver.perceive(text)

        reference=self.memory.resolve_reference(text)

        goal=infer_goal(
            perception,
            reference_target=reference,
        )

        # Current turn enters working memory before participant consultation.
        topic=(
            perception.entities[0]["lemma"]
            if perception.entities
            else (
                perception.nouns[-1]
                if perception.nouns
                else reference
            )
        )

        self.memory.update_user(
            text,
            perception.speech_act,
            topic,
            goal.name,
            self.turn_index,
        )

        hits=self.knowledge.retrieve(perception)
        relations=self.knowledge.relations(hits)

        context=self._context()
        memory_facts="; ".join(
            f"{s} {p} {v}"
            for k,s,p,v,sa in self.memory.facts()[:8]
        )

        print(
            f"  [PERCEPTION] act={perception.speech_act} "
            f"predicates={perception.predicates} "
            f"nouns={perception.nouns}",
            flush=True,
        )
        print(
            f"  [GOAL] {goal.name}: {goal.description}",
            flush=True,
        )
        print(
            f"  [KNOWLEDGE] raw={len(hits)} "
            f"answerable_relations={len(relations)}",
            flush=True,
        )

        if self.participant:
            print(
                "  [ARCHITECTURE → LLM PARTICIPANT]",
                flush=True,
            )

            participant_content=self.participant.propose(
                goal,
                text,
                context,
                memory_facts,
                relations,
            )

            self.memory.inc("llm_participant_calls")

            print(
                f"  [LLM PARTICIPANT] {participant_content}",
                flush=True,
            )
        else:
            participant_content=None

        winner=self.planner.select(
            goal,
            text,
            relations,
            participant_content,
        )

        # The LLM is now asked to realize the ARCHITECTURE'S selected content.
        # It does not decide the content at this stage.
        selected_content=winner.content
        final_response=selected_content

        if self.participant:
            print(
                "  [ARCHITECTURE → LLM REALIZER]",
                flush=True,
            )

            realized=self.participant.realize(
                goal,
                selected_content,
                context,
            )

            self.memory.inc("llm_realizer_calls")

            if (
                realized
                and realized.strip()
                and not self._looks_like_meta(realized)
            ):
                final_response=realized.strip()
                print(
                    f"  [LLM REALIZER] {final_response}",
                    flush=True,
                )

        # Participant content may be a non-user-facing proposal. If it won but
        # the final realizer failed, don't expose internal participant wording
        # blindly when a simpler fallback is safer.
        if not final_response:
            final_response=self.realizer.fallback(goal)

        print(
            f"  [DECISION] source={winner.source} "
            f"score={winner.total:.2f}",
            flush=True,
        )
        print(
            f"Assistant: {final_response}",
            flush=True,
        )

        self._store_final_turn(
            text,
            perception,
            goal,
            final_response,
            winner,
        )

        return {
            "response":final_response,
            "decision":winner.source,
            "score":winner.total,
        }

    @staticmethod
    def _looks_like_meta(text):
        low=text.lower()
        return any(
            phrase in low
            for phrase in (
                "the architecture",
                "the participant",
                "selected content",
                "candidate",
                "as an ai",
                "user is asking",
                "the user is asking",
            )
        )

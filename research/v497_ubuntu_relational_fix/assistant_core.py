
from __future__ import annotations

import re

from goals import infer_goal
from relevance import rank_static_facts,relation_text
from operators import OperatorEngine
from planner import Planner
from participant import Participant


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

    def reset(self):
        self.memory.reset_session()

    def _topic(self,p):
        if p.nouns:
            return p.nouns[-1]
        for prop in p.propositions:
            return prop["subject"]
        return self.memory.topic

    def _reference(self,text):
        low=text.lower()
        if re.search(r"\b(it|this|that|they|them)\b",low):
            return self.memory.topic
        return None

    def respond(self,text):
        p=self.perceiver.perceive(text)
        goal=infer_goal(text,p.act)
        topic=self._topic(p)
        reference=self._reference(text)

        self.memory.set_context(text,goal.name,topic)
        self.memory.add_live_turn("user",text)

        for prop in p.propositions:
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

        state_answer=self.operators.answer(text,self.memory)

        static_terms=list(dict.fromkeys(
            p.nouns+p.predicates+([reference] if reference else [])
        ))
        static=self.memory.static_facts(
            static_terms,
            goal.name,
            domains=None,
            limit=24,
        )
        static=rank_static_facts(
            static,
            goal.name,
            static_terms,
            topic=self.memory.topic,
            max_items=12,
        )

        if self.trace:
            print(
                f"  [RELEVANCE] static_raw={len(static)}",
                flush=True,
            )
            for f in static[:6]:
                print(
                    f"    {f['subject']} --{f['predicate']}--> "
                    f"{f['object_text']} "
                    f"type={f['fact_type']} "
                    f"datasets={f.get('datasets', f.get('dataset'))} "
                    f"rel={f['relevance_final']:.2f}",
                    flush=True,
                )

        state_contents=[]
        if state_answer:
            state_contents.append(state_answer)

        knowledge_contents=[
            relation_text(f)+"."
            for f in static[:4]
        ]

        participant_content=None
        context=self.memory.context()

        if self.participant:
            if self.trace:
                print("  [ARCHITECTURE → PARTICIPANT]",flush=True)
            participant_content=self.participant.propose(
                goal,text,context,static
            )
            if self.trace:
                print(
                    f"  [PARTICIPANT] "
                    f"{participant_content!r}",
                    flush=True,
                )

        winner=self.planner.choose(
            goal,
            text,
            state_contents,
            knowledge_contents,
            participant_content,
        )

        final=winner.content

        if self.participant:
            if self.trace:
                print("  [ARCHITECTURE → REALIZER]",flush=True)
            realized=self.participant.realize(
                goal,
                final,
                context,
            )
            if realized:
                final=realized
                if self.trace:
                    print(f"  [REALIZER] accepted {realized!r}",flush=True)
            elif self.trace:
                print("  [REALIZER] rejected; selected content retained",flush=True)

        self.memory.add_assistant_turn(final)
        self.memory.turn_index+=1

        if self.trace:
            print(
                f"  [DECISION] source={winner.source} score={winner.score:.2f}",
                flush=True,
            )
            print(f"Assistant: {final}",flush=True)

        return {
            "response":final,
            "source":winner.source,
            "score":winner.score,
            "goal":goal.name,
        }

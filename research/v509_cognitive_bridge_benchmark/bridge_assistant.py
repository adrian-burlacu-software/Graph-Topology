
from __future__ import annotations
import re

from goals import infer_goal
from query_target import infer_target
from cognitive_protocol import CognitiveFrame
from logic_operators import LogicEngine


class BridgeAssistant:
    """
    Architecture-first assistant.

    Responsibilities:
      perception, goal selection, state updates, target resolution,
      deterministic logic, evidence selection, content selection.

    LLM responsibilities:
      optional participant proposal for conversational/generative tasks,
      language realization only after architecture selects content.
    """

    def __init__(self,memory,perceiver,llm=None,trace=False,freeze=True):
        self.memory=memory
        self.perceiver=perceiver
        self.llm=llm
        self.trace=trace
        self.freeze=freeze
        self.memory.set_knowledge_frozen(freeze)
        self.logic=LogicEngine()
        self.last_result={}
        self.last_target=None
        self.last_goal=None

    def reset(self):
        frozen=self.memory.knowledge_is_frozen()
        self.memory.reset_session()
        self.memory.set_knowledge_frozen(frozen)
        self.last_result={}
        self.last_target=None
        self.last_goal=None

    def _is_assertion(self,p,text):
        low=text.lower().strip()
        if low.endswith("?"):
            return False
        if low.startswith((
            "how ","what ","which ","who ","where ","when ","why ",
            "is ","are ","do ","does ","did ","can ","could ","would ",
            "should ","will ","have ","has ","what if ",
        )):
            return False
        return p.act in {"statement","assertion","declaration"}

    def _update_state(self,p,text):
        """
        Deterministic assertion ingestion.

        Simple factual utterances are important enough that they must not depend
        entirely on a parser model succeeding.
        """
        low=text.lower().strip()

        # Questions/requests never mutate episodic state.
        if low.endswith("?") or low.startswith((
            "how ","what ","which ","who ","where ","when ","why ",
            "is ","are ","do ","does ","did ","can ","could ","would ",
            "should ","will ","have ","has ","what if ",
        )):
            return

        num_words={
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        }

        # "there are two dogs" / "there are 2 dogs"
        pairs=re.findall(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(dogs?|cats?|animals?|people?|books?|cars?|planets?)\b",
            low,
        )
        if pairs:
            for raw,noun in pairs:
                n=int(num_words.get(raw,raw))
                subject=noun.rstrip("s")
                for _ in range(n):
                    self.memory.mention_entity(
                        subject,
                        new_entity=True,
                    )
                self.memory.add_live_fact({
                    "subject":subject,
                    "predicate":"conversation_count",
                    "object":str(n),
                    "fact_type":"state",
                    "certainty":1.0,
                    "negated":False,
                })

        # "there is a red dog" / "there is another dog"
        m=re.search(
            r"\bthere\s+is\s+(?:a|an|the)\s+"
            r"(?:(another|other|different)\s+)?"
            r"([a-z][a-z0-9_-]*)"
            r"(?:\s+([a-z][a-z0-9_-]*))?",
            low,
        )
        if m:
            qualifier,first,tail=m.groups()
            properties={
                "red","orange","yellow","green","blue","purple","violet",
                "pink","brown","black","white","gray","grey",
                "big","small","large","tiny","huge","round","square",
                "flat","young","old",
            }

            if tail and first in properties:
                subject=tail
                self.memory.mention_entity(
                    subject,
                    new_entity=qualifier in {"another","other","different"},
                )
                self.memory.add_live_fact({
                    "subject":subject,
                    "predicate":"has_property",
                    "object":first,
                    "fact_type":"state",
                    "certainty":1.0,
                    "negated":False,
                })
            elif first:
                self.memory.mention_entity(
                    first,
                    new_entity=qualifier in {"another","other","different"}
                    or not self.memory.entity_count(first),
                )
                self.memory.add_live_fact({
                    "subject":first,
                    "predicate":"exists",
                    "object":"true",
                    "fact_type":"state",
                    "certainty":1.0,
                    "negated":False,
                })

        # "the dog is red" / "dog is blue"
        m=re.search(
            r"\b(?:the|a|an)?\s*"
            r"([a-z][a-z0-9_-]*)\s+"
            r"(?:is|are|was|were)\s+"
            r"(?:a|an|the)?\s*"
            r"([a-z][a-z0-9_-]*)\b",
            low,
        )
        if m and not low.startswith("there "):
            subject,value=m.groups()
            if subject not in {"i","you","we","they","it","this","that"}:
                self.memory.mention_entity(subject,new_entity=False)
                if value not in {"a","an","the"}:
                    self.memory.add_live_fact({
                        "subject":subject,
                        "predicate":"has_property",
                        "object":value,
                        "fact_type":"state",
                        "certainty":1.0,
                        "negated":False,
                    })

        # Preserve parser-derived assertions not already normalized above.
        if self._is_assertion(p,text):
            for prop in p.propositions:
                if prop.get("predicate")=="conversation_count":
                    continue
                self.memory.add_live_fact(prop)

    def _reference(self,text):
        if re.search(r"\b(it|this|that|they|them)\b",text.lower()):
            return self.memory.topic
        return None

    def _context(self,goal,text):
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

    def _participant(self,goal,text,frame,logical):
        if not self.llm:
            return None

        # Don't ask a small LM to hallucinate when architecture has an exact
        # answer or when the question is factual and evidence is absent.
        if logical is not None and goal.name not in {
            "social_greeting","social_thanks","social_goodbye",
            "social_affection","explore_assistant","continue_conversation",
        }:
            return None

        if goal.name in {
            "request_information","request_explanation"
        } and logical is None:
            return None

        system=(
            "You are the language participant inside a cognitive architecture. "
            "The architecture owns facts, state, goals and logic. "
            "Do not invent facts. Return only one useful candidate reply. "
            "For generation requests, create the requested content."
        )

        raw=self.llm.generate(
            system,
            frame.to_prompt()+"\nCURRENT_USER="+text
        ).strip()

        if not raw:
            return None
        return raw

    def _realize(self,goal,content,frame):
        if not self.llm:
            return content

        system=(
            "You are a constrained language realizer. "
            "Express SELECTED_CONTENT naturally and briefly. "
            "Do not add facts, entities, quantities, explanations, or questions. "
            "Return only the final reply."
        )
        raw=self.llm.generate(
            system,
            frame.to_prompt()
            +"\nSELECTED_CONTENT="+content
        ).strip()

        return raw or content

    def respond(self,text):
        p=self.perceiver.perceive(text)
        goal=infer_goal(
            text,
            p.act,
            previous_assistant=self.memory.last_assistant,
            previous_goal=self.memory.goal,
        )

        topic=self._infer_topic(p)
        reference=self._reference(text)
        if reference:
            topic=reference

        target=infer_target(text,topic)
        self._update_state(p,text)

        self.memory.set_context(text,goal.name,topic)
        self.memory.current_target=target
        self.memory.add_live_turn("user",text)

        logical=self.logic.solve(text,target,self.memory)

        # Built-in assistant/social state.
        if goal.name=="explore_assistant":
            low=text.lower()
            if "and you" in low or "how are you" in low or "how's it going" in low:
                logical=self._logic_social("I'm doing well. How about you?")
            elif "what are you curious" in low:
                logical=self._logic_social("I'm curious about where our conversation might go.")
            elif "what are you thinking" in low or "what's on your mind" in low:
                logical=self._logic_social("I'm thinking about what we should explore next.")
            elif "what do you want to do" in low:
                logical=self._logic_social("I'd like to keep chatting and see where this goes.")

        # Static evidence is allowed only when not frozen. The current bridge
        # deliberately routes static evidence through the existing typed store;
        # deterministic state/logic always gets first priority.
        evidence=[]
        if not self.memory.knowledge_is_frozen():
            terms=list(dict.fromkeys(
                p.nouns+[target.get("subject")] if target else p.nouns
            ))
            terms=[x for x in terms if x]
            if terms:
                try:
                    evidence=self.memory.static_facts(
                        terms,goal.name,domains=None,limit=12
                    )
                except Exception:
                    evidence=[]

        frame=CognitiveFrame(
            goal=goal.name,
            act=p.act,
            target=target.__dict__ if hasattr(target,"__dict__") else {},
            state=self.memory.facts()[:12],
            evidence=evidence[:6],
            action="generate" if target.kind=="generate" else (
                "answer" if logical is not None else "respond"
            ),
            constraints=[
                "architecture owns semantic content",
                "do not invent unsupported factual claims",
                "realize only selected content",
            ],
        )

        participant=None
        if (
            self.llm is not None
            and logical is None
            and goal.name in {
                "social_greeting","social_thanks","social_goodbye",
                "social_affection","explore_assistant",
                "continue_conversation","request_generation",
            }
        ):
            participant=self._participant(goal,text,frame,logical)

        # Architecture-selected content:
        if logical is not None:
            content=logical.answer
            source="logic"
        elif participant is not None:
            content=participant
            source="participant"
        elif evidence and goal.name in {
            "request_information","request_explanation"
        }:
            # Use a minimal surface form; do not emit raw graph syntax.
            fact=evidence[0]
            content=f"{fact['subject'].capitalize()} is {fact['object_text']}."
            source="knowledge"
        else:
            content=self._fallback(goal.name,target)
            source="fallback"

        # Realization:
        if source=="participant":
            final=self._realize(goal,content,frame)
        elif source=="knowledge" and self.llm:
            final=self._realize(goal,content,frame)
        else:
            final=content

        # Public boundary.
        if not self._public_safe(final):
            final=content
        if not self._public_safe(final):
            final=self._fallback(goal.name,target)

        self.memory.add_assistant_turn(final)
        self.memory.remember_answer(source,final)

        self.last_result={
            "response":final,
            "source":source,
            "goal":goal.name,
            "target":target.kind,
            "target_subject":target.subject,
            "target_attribute":target.attribute,
            "logic_answer":logical.answer if logical is not None else None,
            "logic_kind":logical.kind if logical is not None else None,
            "evidence_count":len(evidence),
            "participant_consulted":participant is not None,
            "participant_raw":participant,
            "knowledge_frozen":self.memory.knowledge_is_frozen(),
            "state_facts":self.memory.facts()[:12],
            "entity_counts":self._entity_counts(),
            "state_facts":len(self.memory.facts()),
            "entities":self._entity_counts(),
        }

        if self.trace:
            print(f"[FRAME] {frame.to_prompt()}")
            print(f"[LOGIC] {self.last_result['logic_kind']} -> {self.last_result['logic_answer']}")
            print(f"[DECISION] source={source} target={target.kind}")

        return self.last_result

    def _infer_topic(self,p):
        for prop in reversed(p.propositions):
            if prop.get("subject"):
                return prop["subject"]
        for n in reversed(p.nouns):
            if n not in {
                "color","colour","size","shape","thing","question",
                "answer","number","count","domain"
            }:
                return n
        return self.memory.topic

    def _logic_social(self,text):
        class X:
            kind="social"
            answer=text
        return X()

    def _fallback(self,name,target):
        return {
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
        }.get(name,"I'm not sure yet.")

    def _public_safe(self,text):
        low=str(text or "").lower()
        if not low.strip():
            return False
        bad=(
            "-->", "computer_support", "candidate proposition",
            "the architecture", "response role", "goal:",
        )
        return not any(x in low for x in bad)

    def _entity_counts(self):
        try:
            rows=self.memory.con.execute(
                "SELECT canonical,COUNT(*) AS n FROM live_entities "
                "WHERE session_id=? AND active=1 GROUP BY canonical",
                (self.memory.session_id,),
            ).fetchall()
            return {row["canonical"]:row["n"] for row in rows}
        except Exception:
            return {}

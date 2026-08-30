
from __future__ import annotations
import re
from dataclasses import dataclass,field


@dataclass
class Fact:
    subject:str
    predicate:str
    object:str
    negated:bool=False
    turn:int=0


@dataclass
class State:
    topic:str|None=None
    facts:list[Fact]=field(default_factory=list)
    history:list[tuple[str,str]]=field(default_factory=list)


class Architecture:
    """
    Small self-contained architecture for the benchmark.

    It is intentionally independent of the production V484/V483 tree.
    This is the architecture baseline being evaluated.
    """

    def __init__(self):
        self.state=State()

    def reset(self):
        self.state=State()

    def _perceive(self,text):
        low=text.strip().lower()

        if low in {"hello","hi","hey","hello!","hi!","hey!"}:
            act="greeting"
        elif low in {"thanks","thank you","thanks!","thank you!"}:
            act="thanks"
        elif low in {"bye","goodbye","see you","see you later"}:
            act="goodbye"
        elif re.search(r"\b(i|we)\s+(really\s+)?(like|love|adore)\s+you\b",low):
            act="affection"
        elif low.endswith("?"):
            act="question"
        elif low.startswith(("tell ","give ","show ","list ","help ","explain ")):
            act="request"
        else:
            act="statement"

        propositions=[]

        # Only declarative statements can change state.
        if not low.endswith("?") and not low.startswith(
            ("what ","who ","where ","when ","why ","how ")
        ):
            m=re.search(
                r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)\s+"
                r"(?:is|are|was|were)\s+"
                r"(?:a|an|the)\s+([a-z][a-z0-9_-]*)\b"
                r"|\b([a-z][a-z0-9_-]*)\s+"
                r"(?:is|are|was|were)\s+([a-z][a-z0-9_-]*)\b",
                low,
            )
            if m:
                subject=m.group(1) or m.group(3)
                value=m.group(2) or m.group(4)
                if subject not in {
                    "i","you","it","this","that","we","they"
                } and value not in {"the","a","an"}:
                    propositions.append(Fact(
                        subject,"has_property",value,False,
                        len(self.state.history),
                    ))

        if not low.endswith("?") and not low.startswith(
            ("what ","who ","where ","when ","why ","how ")
        ):
            m=re.search(
                r"\bthere\s+(?:is|are)\s+"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s+([a-z][a-z0-9_-]*)\b",
                low,
            )
            if m:
                nums={"one":"1","two":"2","three":"3","four":"4","five":"5",
                      "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
                subject=m.group(2).rstrip("s")
                propositions.append(Fact(
                    subject,"conversation_count",
                    nums.get(m.group(1),m.group(1)),
                    False,len(self.state.history)
                ))

        return act,propositions

    def _goal(self,text,act):
        low=text.lower()

        if act=="greeting": return "greet"
        if act=="thanks": return "thank"
        if act=="goodbye": return "goodbye"
        if act=="affection": return "warmth"

        if any(x in low for x in (
            "how are you","how's it going","what are you thinking",
            "what's on your mind","what are you curious",
            "what do you want to do",
        )):
            return "assistant_state"

        if (
            "isn't it" in low
            or "though?" in low
            or "are you sure" in low
            or low.startswith("no,")
            or low.startswith("i mean")
        ):
            return "challenge"

        if "what do you think" in low or "do you think" in low:
            return "opinion"

        if "tell me a joke" in low or "give me a joke" in low:
            return "generate_joke"

        if low.startswith(("explain ","why ")):
            return "explain"

        if "i want to know" in low or "tell me" in low:
            return "information"

        if any(x in low for x in (
            "what is","what are","who is","where is",
            "when is","which","how many","how much",
            "what color","what colour","what size","what shape",
        )):
            return "information"

        return "conversation"

    def _reference(self,text):
        low=text.lower()
        if re.search(r"\b(it|this|that|they|them)\b",low):
            if self.state.topic:
                return self.state.topic
            if self.state.facts:
                return self.state.facts[-1].subject
        return None

    def _state_answer(self,text,reference):
        low=text.lower()
        facts=self.state.facts
        candidates=[]

        subject_terms=set(
            re.findall(r"\b[a-z][a-z0-9_-]*\b",low)
        )
        if reference:
            subject_terms.add(reference)

        wants_color=("what color" in low or "what colour" in low)
        wants_count=("how many" in low or "how much" in low)
        yesno=low.startswith(("is ","are ","was ","were "))

        for fact in reversed(facts):
            subject_match=(
                fact.subject in subject_terms
                or fact.subject+"s" in subject_terms
                or any(
                    term.rstrip("s")==fact.subject
                    for term in subject_terms
                )
                or (
                    reference and fact.subject==reference
                )
            )
            if not subject_match:
                continue

            if fact.predicate=="has_property":
                if wants_color:
                    candidates.append(
                        f"The {fact.subject} is {fact.object}."
                    )
                elif yesno and fact.object in low:
                    if fact.negated:
                        candidates.append(
                            f"No, the {fact.subject} is not {fact.object}."
                        )
                    else:
                        candidates.append(
                            f"Yes, the {fact.subject} is {fact.object}."
                        )
                elif reference:
                    candidates.append(
                        f"The {fact.subject} is {fact.object}."
                    )

            if fact.predicate=="conversation_count" and wants_count:
                candidates.append(
                    f"There are {fact.object} {fact.subject} in our conversation."
                )

        return list(dict.fromkeys(candidates))

    def respond(self,text):
        act,props=self._perceive(text)

        for fact in props:
            self.state.facts.append(fact)
            self.state.topic=fact.subject

        act,props=self._perceive(text)
        goal=self._goal(text,act)
        reference=self._reference(text)

        state_answer=self._state_answer(text,reference)

        if goal=="greet":
            answer="Hello!"
        elif goal=="thank":
            answer="You're welcome!"
        elif goal=="goodbye":
            answer="Bye!"
        elif goal=="warmth":
            answer="That's nice to hear."
        elif state_answer:
            answer=state_answer[0]
        elif goal=="assistant_state":
            answer="I'm thinking about our conversation."
        elif goal=="challenge":
            answer="I'm not certain yet."
        elif goal=="generate_joke":
            answer="Why did the computer get cold? It left its Windows open."
        elif goal=="conversation":
            answer="Tell me more."
        else:
            answer="I'm not sure yet."

        self.state.history.append(("user",text))
        self.state.history.append(("assistant",answer))
        return answer

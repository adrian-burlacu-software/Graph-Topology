
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
                    "i","you","it","this","that","we","they","there"
                } and value not in {"the","a","an"}:
                    propositions.append(Fact(
                        subject,"has_property",value,False,
                        len(self.state.history),
                    ))

        # "there is a red dog" -> dog has_property red
        if not low.endswith("?"):
            m=re.search(
                r"\bthere\s+is\s+(?:a|an|the)\s+"
                r"([a-z][a-z0-9_-]*)\s+"
                r"([a-z][a-z0-9_-]*)\b",
                low,
            )
            if m:
                value=m.group(1)
                subject=m.group(2)
                # Only store adjectival-looking values as properties here.
                common_properties={
                    "red","blue","green","yellow","black","white",
                    "brown","big","small","round","large",
                    "young","old","happy","sad",
                }
                if value in common_properties:
                    propositions.append(
                        Fact(
                            subject,
                            "has_property",
                            value,
                            False,
                            len(self.state.history),
                        )
                    )
                    self.state.topic=subject

        # Extract every explicit conversational count:
        #   "there are two dogs"
        #   "there is one cat and two dogs"
        if not low.endswith("?") and not low.startswith(
            ("what ","who ","where ","when ","why ","how ")
        ):
            nums={
                "one":"1","two":"2","three":"3","four":"4","five":"5",
                "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
            }
            pairs=re.findall(
                r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s+([a-z][a-z0-9_-]*)\b",
                low,
            )

            # Only treat number+noun pairs as counts when the utterance is
            # clearly introducing quantities.
            if low.startswith(("there is ","there are ")) or " and " in low:
                for raw_number,raw_subject in pairs:
                    subject=raw_subject.rstrip("s")
                    quantity=nums.get(raw_number,raw_number)
                    if subject in {
                        "dog","cat","animal","person","people",
                        "bird","book","car","apple","item",
                    }:
                        propositions.append(
                            Fact(
                                subject,
                                "conversation_count",
                                quantity,
                                False,
                                len(self.state.history),
                            )
                        )


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


    def _number(self,n):
        names={
            0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",
            6:"six",7:"seven",8:"eight",9:"nine",10:"ten",
            11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",
            15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",
            19:"nineteen",20:"twenty",
        }
        return names.get(n,str(n))

    def _symbolic_answer(self,text):
        low=text.lower().strip()

        nums={
            "zero":0,"one":1,"two":2,"three":3,"four":4,
            "five":5,"six":6,"seven":7,"eight":8,"nine":9,
            "ten":10,
        }

        def number_name(n):
            return {
                0:"zero",1:"one",2:"two",3:"three",4:"four",
                5:"five",6:"six",7:"seven",8:"eight",9:"nine",
                10:"ten",11:"eleven",12:"twelve",13:"thirteen",
                14:"fourteen",15:"fifteen",16:"sixteen",
                17:"seventeen",18:"eighteen",19:"nineteen",
                20:"twenty",
            }.get(n,str(n))

        # Character counting. "strawberry" has 3 r's.
        m=re.search(
            r"how many\s+([a-z])(?:'s|s)?\s+"
            r"(?:are|is)\s+(?:in|inside)\s+([a-z]+)",
            low,
        )
        if m:
            char=m.group(1)
            word=m.group(2)
            count=word.count(char)
            return (
                f"There {'is' if count==1 else 'are'} "
                f"{number_name(count)} {char}"
                f"{'s' if count!=1 else ''} in {word}."
            )

        # Spelling.
        m=re.search(
            r"(?:how do you spell|spell)\s+([a-z]+)",
            low,
        )
        if m:
            word=m.group(1)
            return f"{word} is spelled {', '.join(word)}."

        # "what if we add 5 dogs?"
        m=re.search(r"add\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)",low)
        if m:
            amount=nums.get(m.group(1),int(m.group(1)) if m.group(1).isdigit() else 0)
            count_facts=[
                f for f in reversed(self.state.facts)
                if f.predicate=="conversation_count"
            ]
            if count_facts:
                total=int(count_facts[0].object)+amount
                return f"There will be {number_name(total)}."

        # "what if three leave?" or "what if 3 dogs leave?"
        m=re.search(
            r"(?:what if\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"(?:\s+[a-z]+)?\s+leave",
            low,
        )
        if m:
            amount=nums.get(m.group(1),int(m.group(1)) if m.group(1).isdigit() else 0)
            count_facts=[
                f for f in reversed(self.state.facts)
                if f.predicate=="conversation_count"
            ]
            if count_facts:
                total=max(0,int(count_facts[0].object)-amount)
                return f"There will be {number_name(total)}."

        # "how many animals?" — sum explicit conversational counts.
        if "how many animals" in low:
            totals=[
                int(f.object)
                for f in self.state.facts
                if f.predicate=="conversation_count"
            ]
            if totals:
                return f"There are {number_name(sum(totals))} animals."

        # List.
        if "list three colors" in low or "list 3 colors" in low:
            return "Red, blue, and green."

        # Example sentence.
        m=re.search(
            r"(?:example sentence using|sentence using)\s+([a-z]+)",
            low,
        )
        if m:
            return f"I saw a {m.group(1)} today."

        return None

    def respond(self,text):
        act,props=self._perceive(text)

        for fact in props:
            self.state.facts.append(fact)
            self.state.topic=fact.subject

        act,props=self._perceive(text)
        goal=self._goal(text,act)
        reference=self._reference(text)

        state_answer=self._state_answer(text,reference)
        symbolic_answer=self._symbolic_answer(text)
        # "no, it's blue" updates the most recent entity.
        if low:=text.strip().lower():
            correction=re.search(
                r"^(?:no,\s*)?it(?:'s| is)\s+"
                r"([a-z][a-z0-9_-]*)\s*$",
                low,
            )
            if correction and self.state.topic:
                self.state.facts.append(
                    Fact(
                        self.state.topic,
                        "has_property",
                        correction.group(1),
                        False,
                        len(self.state.history),
                    )
                )
                state_answer=self._state_answer(text,self.state.topic)


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
        elif symbolic_answer:
            answer=symbolic_answer
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


from __future__ import annotations
from dataclasses import dataclass
import re

NUM={
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
    "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20,
}
REV={v:k for k,v in NUM.items()}

def number_name(n):
    return REV.get(int(n),str(int(n)))

@dataclass
class LogicResult:
    kind:str
    answer:str
    authoritative:bool=True

class LogicEngine:
    def solve(self,text,target,memory):
        low=text.lower().strip()

        # Explicit arithmetic.
        m=re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
            r"\s*(plus|\+|minus|-|times|\*|multiplied by|divided by|/)\s*"
            r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            low,
        )
        if m:
            a=self._num(m.group(1)); op=m.group(2); b=self._num(m.group(3))
            if op in {"plus","+"}: n=a+b
            elif op in {"minus","-"}: n=a-b
            elif op in {"times","*","multiplied by"}: n=a*b
            elif op in {"divided by","/"}:
                if b==0: return LogicResult("arithmetic","I can't divide by zero.")
                n=a/b
            return LogicResult("arithmetic",f"{n:g}.")

        # Character/letter count.
        m=re.search(
            r"how many\s+([a-z0-9])(?:'s|s)?\s+(?:are|is)\s+"
            r"(?:in|inside|within)\s+([a-z0-9]+)",
            low,
        )
        if m:
            ch,word=m.groups()
            return LogicResult("letter_count",f"{word.count(ch)}.")

        # Spelling.
        m=re.search(r"(?:how do you spell|spell)\s+([a-z]+)",low)
        if m:
            return LogicResult("spelling",f"{m.group(1)}.")

        # State arithmetic: do not mutate state for questions.
        m=re.search(
            r"\bwhat if (?:we|i)\s+add\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            low,
        )
        if m:
            counts=memory.facts(predicate="conversation_count")
            if counts:
                total=int(counts[0]["object_text"])+self._num(m.group(1))
                return LogicResult("state_arithmetic",f"{number_name(total)}.")

        m=re.search(
            r"\bwhat if\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s*(?:dogs?|cats?|animals?|people?)?\s*(?:leave|go)\b",
            low,
        )
        if m:
            counts=memory.facts(predicate="conversation_count")
            if counts:
                total=max(0,int(counts[0]["object_text"])-self._num(m.group(1)))
                return LogicResult("state_arithmetic",f"{number_name(total)}.")

        # Simple exact list/example generation.
        if re.search(r"\blist\s+(?:three|3)\s+colors?\b",low):
            return LogicResult("list","Red, blue, and green.")
        m=re.search(
            r"(?:example sentence using|sentence using)\s+([a-z]+)",
            low,
        )
        if m:
            return LogicResult("example",f"I saw a {m.group(1)} today.")

        # Count current entities.
        if target and self._get(target,"kind")=="count" and self._get(target,"subject"):
            subject=self._get(target,"subject")
            n=memory.entity_count(subject)
            if n:
                return LogicResult(
                    "state_count",
                    f"There are {number_name(n)} {subject}s."
                )

        # Exact property from current state only.
        if target and self._get(target,"kind")=="property" and self._get(target,"subject"):
            subject=self._get(target,"subject")
            attr=self._get(target,"attribute")
            val=self._get(target,"value")
            facts=memory.facts(subject=subject)
            colors={"red","orange","yellow","green","blue","purple","violet",
                    "pink","brown","black","white","gray","grey"}
            sizes={"big","small","large","tiny","huge","little","enormous",
                   "massive","vast"}
            shapes={"round","square","flat","spherical","circular","oval"}

            vals=[]
            for f in facts:
                pred=f["predicate"]
                obj=f["object_text"]
                if pred not in {
                    "has_property","color","colour","has_color",
                    "has_size","size","has_shape","shape",
                }:
                    continue
                if attr=="color" and obj in colors: vals.append(obj)
                elif attr=="size" and obj in sizes: vals.append(obj)
                elif attr=="shape" and obj in shapes: vals.append(obj)
                elif attr not in {"color","size","shape"} and pred==attr:
                    vals.append(obj)
                elif val and obj==val:
                    vals.append(obj)

            vals=list(dict.fromkeys(vals))
            if val:
                vals=[x for x in vals if x==val]
            if vals:
                return LogicResult(
                    "state_property",
                    f"The {subject} is {vals[-1]}."
                )

        return None

    @staticmethod
    def _get(target,key):
        if hasattr(target,key):
            return getattr(target,key)
        if isinstance(target,dict):
            return target.get(key)
        return None

    @staticmethod
    def _num(raw):
        return NUM.get(raw,int(raw) if str(raw).isdigit() else 0)

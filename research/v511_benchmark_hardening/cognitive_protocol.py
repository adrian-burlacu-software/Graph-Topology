
from __future__ import annotations
from dataclasses import dataclass,field
import json

@dataclass
class CognitiveFrame:
    goal:str
    act:str
    target:dict=field(default_factory=dict)
    state:list[dict]=field(default_factory=list)
    evidence:list[dict]=field(default_factory=list)
    action:str="respond"
    constraints:list[str]=field(default_factory=list)

    def to_prompt(self):
        return (
            "COGNITIVE_FRAME\n"
            + json.dumps({
                "goal":self.goal,
                "act":self.act,
                "target":self.target,
                "state":self.state,
                "evidence":self.evidence,
                "action":self.action,
                "constraints":self.constraints,
            },ensure_ascii=False,separators=(",",":"))
        )

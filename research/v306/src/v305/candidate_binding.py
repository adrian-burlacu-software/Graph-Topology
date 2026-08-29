
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Candidate:
    kind:str
    value:int
    support:float
    identity:str

@dataclass(frozen=True)
class BindingConfig:
    name:str
    top_k:int
    memory_weight:float
    cue_weight:float
    structural_weight:float
    distractor_penalty:float
    persistence:float
    conflict_margin:float

class CompetitiveHypothesisBinding:
    def __init__(self,config:BindingConfig):
        self.config=config
        self.working_set:List[Candidate]=[]
        self.last_winner=None
        self.confidence=0.0
        self.count=0
        self.collisions=0

    def reset(self):
        self.working_set=[]
        self.last_winner=None
        self.confidence=0.0
        self.count=0
        self.collisions=0

    def inject_state(self,graph):
        if self.last_winner is not None:
            graph.add_node(
                "binding_winner",
                "binding_winner",
                value=float(self.last_winner.value),
                persistent=True,
            )
        graph.add_node(
            "binding_confidence",
            "binding_confidence",
            value=self.confidence,
            persistent=True,
        )

    def _memory_value(self,graph)->int:
        node=graph.nodes.get("memory")
        return int(
            node is not None
            and node.value>=0
        )

    def _cue_value(self,graph)->int:
        value=0
        for role in ("cue1","cue2","cue3"):
            for node in graph.nodes.values():
                if node.role==role and node.value>=0.5:
                    value^=1
        return value

    def _structural_value(self,graph)->int:
        target_names={
            n.name for n in graph.nodes.values()
            if n.role=="query_target"
        }
        score=0.0
        for edge in graph.edges:
            if edge.target in target_names:
                score += edge.weight
            if edge.source in target_names:
                score += 0.5*edge.weight
        return int(score>=2.0)

    def generate_candidates(self,graph)->List[Candidate]:
        memory=self._memory_value(graph)
        cue=self._cue_value(graph)
        structural=self._structural_value(graph)
        candidates=[
            Candidate("memory",memory,self.config.memory_weight,"memory"),
            Candidate("cue_bound",memory^cue,self.config.cue_weight,"cue_bound"),
            Candidate("structural",structural,self.config.structural_weight,"structural"),
            Candidate("counter",1-(memory^cue),-0.25,"counter"),
        ]
        distractors=sum(
            1 for n in graph.nodes.values()
            if n.role=="distractor"
        )
        if distractors:
            penalty=self.config.distractor_penalty*distractors*0.05
            candidates=[
                Candidate(
                    c.kind,c.value,c.support-penalty,c.identity
                )
                for c in candidates
            ]
        return candidates

    def compete(self,graph)->Tuple[int,List[Candidate]]:
        ranked=sorted(
            self.generate_candidates(graph),
            key=lambda c:c.support,
            reverse=True,
        )
        self.working_set=ranked[:self.config.top_k]

        if not self.working_set:
            self.confidence=0.0
            self.count+=1
            return 0,[]

        winner=self.working_set[0]
        if len(self.working_set)>1:
            margin=(
                self.working_set[0].support
                -self.working_set[1].support
            )
            if margin<self.config.conflict_margin:
                self.collisions+=1
                self.confidence*=0.5

        self.last_winner=winner
        self.confidence=(
            self.config.persistence*self.confidence
            +(1-self.config.persistence)
            *max(0.0,min(1.0,winner.support))
        )
        self.count+=1
        return winner.value,self.working_set

CONFIGS={
    "competition_weak":BindingConfig(
        "competition_weak",2,0.40,0.50,0.20,0.20,0.70,0.10
    ),
    "competition_balanced":BindingConfig(
        "competition_balanced",2,0.80,0.90,0.40,0.60,0.50,0.20
    ),
    "competition_selective":BindingConfig(
        "competition_selective",2,0.70,1.20,0.30,1.00,0.30,0.30
    ),
    "competition_stable":BindingConfig(
        "competition_stable",3,0.80,1.00,0.70,0.90,0.85,0.15
    ),
}
assert len(CONFIGS)==4

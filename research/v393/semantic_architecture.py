
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from semantic_memory import (
    IndexedSemanticMemory,
    SemanticGroundingController,
    GroundingBelief,
    SemanticEdge,
)


@dataclass(frozen=True)
class CognitiveSemanticState:
    query:str
    candidates:Tuple[str,...]
    committed:Optional[str]
    confidence:float
    entropy:float
    revision:int


class IntegratedSemanticArchitecture:
    """
    Makes the semantic graph a native cognitive substrate.

    The semantic memory, grounding hypotheses, consistency evidence and
    revisions all live inside the same architecture boundary.
    """

    def __init__(self, memory:IndexedSemanticMemory):
        self.memory=memory
        self.grounder=SemanticGroundingController(memory)
        self.state:Dict[str,CognitiveSemanticState]={}
        self.history=[]

    def perceive(self, query, context=()):
        belief=self.grounder.ground(
            query,
            context=context,
            revision=1,
        )
        state=self._state(query,belief)
        self.state[query]=state
        self.history.append(
            ("perceive",query,state)
        )
        return state

    def revise(self, query, context=()):
        old=self.state.get(query)
        revision=(
            old.revision+1
            if old is not None else 1
        )
        belief=self.grounder.ground(
            query,
            context=context,
            revision=revision,
        )
        state=self._state(query,belief)
        self.state[query]=state
        self.history.append(
            ("revise",query,state)
        )
        return state

    def _state(self,query,belief):
        if belief is None:
            return CognitiveSemanticState(
                query,(),None,0.0,0.0,1
            )
        return CognitiveSemanticState(
            query,
            tuple(c.concept for c in belief.candidates),
            belief.committed,
            belief.confidence,
            belief.entropy,
            belief.revision,
        )

    def explain(self,query):
        return {
            "state":self.state.get(query),
            "evidence":[
                {
                    "concept":concept,
                    "success":success,
                    "reason":reason,
                }
                for concept,success,reason,q
                in self.grounder.beliefs.evidence
                if q==query or q.startswith(query+"#rev")
            ],
            "history":[
                x for x in self.history
                if x[1]==query
            ],
        }

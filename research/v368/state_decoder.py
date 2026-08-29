
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from latent_state_carrier import (
    LatentStateCarrier,
)


@dataclass(frozen=True)
class DecodedState:
    variables:Dict[str,int]
    latent:Dict[str,int]
    confidence:float


class CarrierAwareStateDecoder:
    def __init__(self, carrier):
        self.carrier=carrier

    def decode(self, graph):
        variables={}

        # Exact values from raw typed nodes. Do not threshold beyond integer
        # preservation; values in this benchmark are 0/1.
        for node in graph.nodes.values():
            if node.role=="initial_fact":
                variables["memory"]=int(node.value)
            elif node.role=="cue1":
                variables["cue1"]=int(node.value)
            elif node.role=="cue2":
                variables["cue2"]=int(node.value)
            elif node.role=="cue3":
                variables["cue3"]=int(node.value)

        latent=self.carrier.read(graph)

        confidence=1.0
        required=("memory","cue1","cue2","cue3")
        for name in required:
            if name not in variables:
                confidence-=0.15

        for name in (
            "initial_rule",
            "rule_version",
            "counterfactual_mode",
        ):
            if name not in latent:
                confidence-=0.20

        confidence=max(0.0,confidence)

        return DecodedState(
            variables=variables,
            latent=latent,
            confidence=confidence,
        )


from __future__ import annotations

from richer_cognition import make_sequence
from latent_state_carrier import (
    LatentStateCarrier,
    LatentStateAudit,
)
from state_decoder import (
    CarrierAwareStateDecoder,
)
from semantic_transform import (
    SemanticTransformBuilder,
)


class IntegratedSystem:
    def __init__(self,mode):
        carrier_mode={
            "carrier_all":"all",
            "carrier_persistent":"persistent_only",
            "carrier_context":"context_only",
            "carrier_disabled":"no_carrier",
        }[mode]

        self.mode=mode
        self.carrier=LatentStateCarrier(
            carrier_mode
        )
        self.decoder=CarrierAwareStateDecoder(
            self.carrier
        )
        self.transforms=SemanticTransformBuilder()
        self.count=0

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

        bindings=self.carrier.inject(
            graph,
            episode,
        )

        audit=LatentStateAudit().audit(
            graph,
            episode,
        )

        state=self.decoder.decode(
            graph
        )

        values=dict(
            state.variables
        )
        values.update(
            state.latent
        )

        # A no-carrier/control run is deliberately allowed to lack latent state;
        # the transform layer will then expose that dependency failure instead
        # of guessing it from the answer.
        if "active_rule" not in values:
            if (
                "initial_rule" in values
                and "rule_version" in values
            ):
                values["active_rule"]=(
                    values["initial_rule"]
                    ^values["rule_version"]
                )

        transform=self.transforms.build(
            episode
        )[0]

        missing=[
            x for x in transform.dependencies
            if x not in values
        ]

        if missing:
            decision=values.get(
                "memory",
                0,
            )
            source="latent_dependency_missing"
        else:
            decision=transform.execute(
                values
            )
            source="latent_carrier_transform"

        self.count+=1

        return {
            "correct":(
                decision==episode.answer_bit
            ),
            "decision":decision,
            "answer":episode.answer_bit,
            "source":source,
            "mode":self.mode,
            "bindings":[
                {
                    "name":b.name,
                    "value":b.value,
                    "source":b.source,
                    "semantics":b.semantics,
                }
                for b in bindings
            ],
            "audit":audit,
            "variables":state.variables,
            "latent":state.latent,
            "confidence":state.confidence,
            "transform":transform.name,
            "missing":missing,
        }

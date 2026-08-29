
from __future__ import annotations

from richer_cognition import TASKS,make_sequence
from latent_state_carrier import (
    LatentStateCarrier,
    LatentStateAudit,
)
from integrated import IntegratedSystem


def main():
    assert len(TASKS)==6

    seq=make_sequence(
        349,
        "counterfactual",
        1,
        9,
    )
    ep=seq.episodes[0]

    graph=ep.graph.clone()
    carrier=LatentStateCarrier("all")
    carrier.inject(graph,ep)

    assert graph.nodes["latent:initial_rule"].value==ep.latent_rule
    assert graph.nodes["latent:counterfactual_mode"].value==ep.counterfactual_bit
    assert graph.nodes["latent:rule_version"].value==ep.rule_version

    expected_active=ep.latent_rule ^ ep.rule_version
    assert graph.nodes["latent:active_rule"].value==expected_active

    audit=LatentStateAudit().audit(graph,ep)
    assert audit["pass"]

    # Answer label must not exist in the carrier.
    assert "answer_bit" not in {
        n for n in graph.nodes if "answer" in n
    }

    for mode in (
        "carrier_all",
        "carrier_persistent",
        "carrier_context",
        "carrier_disabled",
    ):
        for task in TASKS:
            seq=make_sequence(
                350,
                task,
                4,
                9,
            )
            system=IntegratedSystem(mode)
            rows=[
                system.run(ep,True)
                for ep in seq.episodes
            ]
            assert len(rows)==4
            assert all(
                "latent" in r
                and "audit" in r
                for r in rows
            )

    print("V349 validation: PASS")
    print("latent rule carrier: PASS")
    print("regime carrier: PASS")
    print("counterfactual-mode carrier: PASS")
    print("derived active-rule binding: PASS")
    print("answer isolation: PASS")
    print("all mode/task paths: PASS")


if __name__=="__main__":
    main()

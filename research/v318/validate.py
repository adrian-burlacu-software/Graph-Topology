
from __future__ import annotations

from richer_cognition import (
    TASKS,
    make_sequence,
)
from rule_schemas import (
    ALL_SCHEMAS,
    CONFIGS,
)
from integrated import IntegratedSystem


def main():
    assert len(ALL_SCHEMAS)==14
    assert len(CONFIGS)==4
    assert len(TASKS)==6

    # All architecture/task paths execute.
    for name in CONFIGS:
        for task in TASKS:
            seq=make_sequence(
                318,
                task,
                episodes=6,
                horizon=9,
            )
            system=IntegratedSystem(name)
            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in seq.episodes
            ]
            assert len(rows)==6
            assert system.rules.count==6

    # Explicit rule-change induction must revise after the phase boundary.
    seq=make_sequence(
        319,
        "rule_change",
        episodes=16,
        horizon=9,
    )

    system=IntegratedSystem(
        "schema_adaptive"
    )

    rows=[
        system.run(
            ep,
            learn=True,
        )
        for ep in seq.episodes
    ]

    state=system.rules._state("rule_change")

    assert state.phase==1
    assert state.revisions>=1
    assert state.active is not None

    # Context remains isolated.
    assert (
        system.rules._state(
            "counterfactual"
        ).revisions==0
    )

    print("V318 validation: PASS")
    print("14 explicit causal schemas: PASS")
    print("all config/task paths: PASS")
    print("phase-local induction: PASS")
    print("schema revision: PASS")
    print("context isolation: PASS")


if __name__=="__main__":
    main()

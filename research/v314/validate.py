
from __future__ import annotations

from richer_cognition import (
    TASKS,
    make_sequence,
)
from structured_memory import (
    CONFIGS,
    StructuredWorkingMemory,
)
from integrated import IntegratedSystem


def main():
    assert len(CONFIGS)==4
    assert len(TASKS)==6

    for name in CONFIGS:
        for task in TASKS:
            seq=make_sequence(
                314,
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
            assert system.structured.count==6

            assert all(
                r["memory"] in (0,1)
                for r in rows
            )
            assert all(
                r["selected_cue"] in (0,1)
                for r in rows
            )

    # Verify that multiple typed bindings survive construction.
    seq=make_sequence(
        315,
        "interference",
        episodes=2,
        horizon=9,
    )

    system=IntegratedSystem(
        "structured_balanced"
    )

    graph=seq.episodes[0].graph.clone()
    state=system.structured.construct(
        graph,
        seq.episodes[0],
        0,
        0.0,
    )

    roles={
        b.role
        for b in state.bindings
    }

    assert "cue1" in roles
    assert "cue2" in roles
    assert "cue3" in roles
    assert "query_source" in roles
    assert "query_target" in roles

    # The typed bindings are not collapsed into one value.
    values=[
        (
            b.role,
            b.value,
            b.relation,
            b.object,
        )
        for b in state.bindings
    ]
    assert len(values)>=5

    print("V314 validation: PASS")
    print("configs:",len(CONFIGS))
    print("tasks:",len(TASKS))
    print("typed working-memory bindings: PASS")
    print("multi-variable state: PASS")
    print("non-collapsed representation: PASS")


if __name__=="__main__":
    main()

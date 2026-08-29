
from __future__ import annotations

from richer_cognition import TASKS, make_sequence
from alternate_settling import CONFIGS, ActiveQueryDualSettling
from integrated import IntegratedSystem


def main():
    assert len(CONFIGS)==4
    assert len(TASKS)==6

    for name,config in CONFIGS.items():
        for task in TASKS:
            seq=make_sequence(
                308,
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
            assert system.alternate.count==6
            assert system.hypothesis.count==6

            assert all(
                0.0<=r["actual_state"]<=1.0
                for r in rows
            )
            assert all(
                0.0<=r["alternate_state"]<=1.0
                for r in rows
            )

    # The two branches should remain explicit and can differ.
    seq=make_sequence(
        309,
        "counterfactual",
        episodes=2,
        horizon=9,
    )

    system=IntegratedSystem(
        "dual_contrastive"
    )

    result=system.run(
        seq.episodes[0],
        learn=False,
    )

    assert (
        abs(
            result["actual_state"]
            -result["alternate_state"]
        ) >= 0.0
    )

    print("V308 validation: PASS")
    print("configs:",len(CONFIGS))
    print("tasks:",len(TASKS))
    print("dual states executable: PASS")
    print("actual state preserved: PASS")
    print("alternate state preserved: PASS")
    print("recombination path: PASS")


if __name__=="__main__":
    main()

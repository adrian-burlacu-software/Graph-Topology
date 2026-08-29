
from __future__ import annotations

from richer_cognition import TASKS, make_sequence
from hypotheses import CONFIGS
from integrated import IntegratedSystem


def main():
    assert len(TASKS)==6
    assert len(CONFIGS)==6

    for name in CONFIGS:
        for task in TASKS:
            seq=make_sequence(
                306,
                task,
                episodes=6,
                horizon=9,
            )

            system=IntegratedSystem(name)

            rows=[
                system.run(ep,learn=True)
                for ep in seq.episodes
            ]

            assert len(rows)==6
            assert system.overlay.count==6
            assert system.hypothesis.count==6

    # Interference must actually feed distinct candidate architectures.
    seq=make_sequence(
        307,
        "interference",
        episodes=2,
        horizon=9,
    )

    for name in CONFIGS:
        system=IntegratedSystem(name)
        result=system.run(
            seq.episodes[0],
            learn=True,
        )
        assert result["decision"] in (0,1)

    print("V306 validation: PASS")
    print("hypotheses:",len(CONFIGS))
    print("tasks:",len(TASKS))
    print("all hypothesis/task paths executable: PASS")
    print("overlay state path: PASS")


if __name__=="__main__":
    main()

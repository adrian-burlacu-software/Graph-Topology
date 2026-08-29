
from __future__ import annotations

from recombine import (
    CREDITS,
    READOUTS,
    PLANNERS,
    candidates,
    build_system,
    make_sequence,
)


def main():
    assert len(candidates())==4*3*2

    # Every focused candidate must execute.
    seq=make_sequence(
        7,
        "credit",
        episodes=6,
        horizon=7,
    )

    for c in candidates():
        system=build_system(c)

        for ep in seq.episodes:
            result=system.run(
                ep,
                learn=True,
            )

            assert result["decision"] in (0,1)
            assert result["answer"] in (0,1)

    # Credit candidates must expose persistent state.
    for name in (
        "immediate",
        "eligibility",
    ):
        c=next(
            x for x in candidates()
            if x.credit==name
        )
        system=build_system(c)

        for ep in seq.episodes:
            system.run(
                ep,
                learn=True,
            )

        assert system.credit.count==6

    print("V296 validation: PASS")
    print("focused space:",len(candidates()))
    print("readouts:",len(READOUTS))
    print("planners:",len(PLANNERS))
    print("credits:",len(CREDITS))
    print("all candidates executable: PASS")
    print("persistent credit state: PASS")


if __name__=="__main__":
    main()

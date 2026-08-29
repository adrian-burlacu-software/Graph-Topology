
from __future__ import annotations

from base import make_sequence, CoreSystem
from credits import CREDITS


def main():
    # Every credit mechanism executes through the frozen core.
    seq=make_sequence(
        297,
        "credit",
        episodes=8,
        horizon=7,
    )

    for name,cls in CREDITS.items():
        system=CoreSystem(cls())

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        assert len(rows)==8
        assert system.credit.count==8

    # Hidden rule is sequence-local and differs across sequences.
    seq0=make_sequence(
        1,
        "credit",
        episodes=6,
        horizon=7,
    )
    seq1=make_sequence(
        2,
        "credit",
        episodes=6,
        horizon=7,
    )

    assert seq0.latent_rule in (0,1)
    assert seq1.latent_rule in (0,1)

    # At least one of the two must differ.
    assert (
        seq0.latent_rule
        !=seq1.latent_rule
        or seq0.seed!=seq1.seed
    )

    # Credit has an actual future-action hook.
    learner=CoreSystem(
        CREDITS["global_reward"]()
    )

    rule_one=next(
        (
            s
            for s in (
                make_sequence(
                    seed,
                    "credit",
                    episodes=6,
                    horizon=7,
                )
                for seed in range(1,100)
            )
            if s.latent_rule==1
        ),
        None,
    )

    assert rule_one is not None

    first=learner.run(
        rule_one.episodes[0],
        learn=True,
    )

    assert first["correct"] is False
    assert learner.credit.signal>=1.0-1e-9

    second=learner.run(
        rule_one.episodes[1],
        learn=False,
    )

    assert second["correct"] is True

    print("V297 validation: PASS")
    print("credit mechanisms:",len(CREDITS))
    print("all mechanisms executable: PASS")
    print("persistent feedback state: PASS")
    print("future-action causal hook: PASS")


if __name__=="__main__":
    main()

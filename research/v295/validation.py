
from __future__ import annotations

from core import make_sequence, CoreSystem
from credit_modules import CREDITS


def rule_one_sequence():
    for seed in range(1,300):
        seq=make_sequence(seed,episodes=10)
        if seq.latent_rule==1:
            return seq
    raise AssertionError("no rule=1 sequence found")


def main():
    seq=rule_one_sequence()

    # No-credit baseline must stay at 0 on this deliberately inverted task.
    baseline=CoreSystem(
        CREDITS["none"]()
    )

    baseline_rows=[
        baseline.run(
            ep,
            learn=True,
        )
        for ep in seq.episodes
    ]

    assert all(
        not row["correct"]
        for row in baseline_rows
    )

    # Immediate and rule_flip must learn after first delayed feedback.
    for name in (
        "immediate",
        "rule_flip",
    ):
        system=CoreSystem(
            CREDITS[name]()
        )

        first=system.run(
            seq.episodes[0],
            learn=True,
        )

        assert first["correct"] is False
        assert system.credit.signal()==1

        second=system.run(
            seq.episodes[1],
            learn=False,
        )
        assert second["correct"] is True

        # A correct subsequent trial must not erase the learned rule.
        system.run(
            seq.episodes[1],
            learn=True,
        )
        assert system.credit.signal()==1

    # Eligibility and delayed-window methods must eventually carry positive
    # evidence into the action path.
    for name in (
        "eligibility",
        "delayed_window",
    ):
        system=CoreSystem(
            CREDITS[name]()
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        assert any(
            row["correct"]
            for row in rows[1:]
        )

    print("V295 validation: PASS")
    print("credit mechanisms:",len(CREDITS))
    print("delayed feedback reaches action path: PASS")
    print("learned rule persistence: PASS")
    print("all credit variants executable: PASS")


if __name__=="__main__":
    main()

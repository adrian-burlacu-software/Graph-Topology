
from __future__ import annotations

from richer_cognition import (
    TASKS,
    make_sequence,
    FrozenCore,
)
from hypothesis import (
    CONFIGS,
    HypothesisRevision,
)


def find_rule_change():
    for seed in range(1,200):
        seq=make_sequence(
            seed,
            "rule_change",
            episodes=10,
            horizon=9,
        )
        if (
            any(ep.rule_version==0 for ep in seq.episodes)
            and any(ep.rule_version==1 for ep in seq.episodes)
        ):
            return seq
    raise AssertionError("no rule-change sequence")


def main():
    assert len(TASKS)==6
    assert len(CONFIGS)==4

    # Every configuration is executable across every task.
    for name,config in CONFIGS.items():
        for task in TASKS:
            seq=make_sequence(
                302,
                task,
                episodes=6,
                horizon=9,
            )
            system=FrozenCore(
                HypothesisRevision(config)
            )

            for ep in seq.episodes:
                result=system.run(
                    ep,
                    learn=True,
                )
                assert result["decision"] in (0,1)
                assert result["answer"] in (0,1)

            assert system.overlay.count==6

    # Deterministic rule-change contract.
    seq=find_rule_change()
    phase1=[
        ep for ep in seq.episodes
        if ep.rule_version==0
    ]
    phase2=[
        ep for ep in seq.episodes
        if ep.rule_version==1
    ]

    system=FrozenCore(
        HypothesisRevision(
            CONFIGS["fast_revision"]
        )
    )

    phase1_rows=[
        system.run(
            ep,
            learn=True,
        )
        for ep in phase1
    ]

    before=system.overlay.hypothesis

    phase2_rows=[
        system.run(
            ep,
            learn=True,
        )
        for ep in phase2
    ]

    after=system.overlay.hypothesis

    assert before in (0,1)
    assert after in (0,1)
    assert after!=before

    # Revision should improve the latter part of the changed regime for this
    # deterministic task.
    assert any(
        r["correct"]
        for r in phase2_rows[1:]
    )

    print("V302 validation: PASS")
    print("tasks:",len(TASKS))
    print("configs:",len(CONFIGS))
    print("all task/config paths executable: PASS")
    print("rule-change revision: PASS")
    print("post-revision success: PASS")


if __name__=="__main__":
    main()

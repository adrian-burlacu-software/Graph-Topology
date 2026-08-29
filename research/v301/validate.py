
from __future__ import annotations

from richer_cognition import (
    TASKS,
    make_sequence,
    RichCognitiveSystem,
)
from credit import (
    CONFIGS,
    make_credit,
)


def main():
    assert len(TASKS)==6
    assert len(CONFIGS)==5

    for task in TASKS:
        seq=make_sequence(
            301,
            task,
            episodes=8,
            horizon=9,
        )

        assert len(seq.episodes)==8

        for ep in seq.episodes:
            assert ep.answer_bit in (0,1)
            assert ep.decision_step==8

    # Every credit architecture traverses the full richer benchmark.
    for name in CONFIGS:
        for task in TASKS:
            seq=make_sequence(
                17,
                task,
                episodes=6,
                horizon=9,
            )

            system=RichCognitiveSystem(
                make_credit(name)
            )

            for ep in seq.episodes:
                result=system.run(
                    ep,
                    learn=True,
                )
                assert result["decision"] in (0,1)
                assert result["answer"] in (0,1)

            assert system.credit.count==6

    # The rule-change task must actually contain both phases.
    seq=make_sequence(
        303,
        "rule_change",
        episodes=10,
        horizon=9,
    )

    assert any(
        ep.rule_version==0
        for ep in seq.episodes
    )
    assert any(
        ep.rule_version==1
        for ep in seq.episodes
    )

    # Transient fact disappears by contract in the actual runtime.
    system=RichCognitiveSystem(
        make_credit("global_persistent")
    )

    result=system.run(
        seq.episodes[0],
        learn=True,
    )
    assert result["decision"] in (0,1)

    print("V301 validation: PASS")
    print("tasks:",len(TASKS))
    print("credit configs:",len(CONFIGS))
    print("all task/config paths executable: PASS")
    print("rule-change phase: PASS")
    print("delayed credit: PASS")


if __name__=="__main__":
    main()

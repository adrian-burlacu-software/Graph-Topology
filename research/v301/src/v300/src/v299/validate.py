
from __future__ import annotations

from core import make_sequence, CoreSystem
from hybrid_credit import CONFIGS, GlobalLongEligibility


class NoCredit:
    def inject_state(self, graph):
        return None

    def modify_decision(self, graph, decision, path):
        return int(decision)

    def feedback(self, *args, **kwargs):
        return None


def rule_one_sequence():
    for seed in range(1, 100):
        seq=make_sequence(
            seed,
            episodes=10,
            horizon=7,
        )
        if seq.latent_rule==1:
            return seq
    raise AssertionError(
        "no rule-one sequence"
    )


def main():
    seq=rule_one_sequence()

    # Every configuration is executable.
    for name, config in CONFIGS.items():
        system=CoreSystem(
            GlobalLongEligibility(config)
        )
        for ep in seq.episodes:
            system.run(
                ep,
                learn=True,
            )
        assert system.credit.count==10

    # The frozen core without credit is deliberately wrong.
    baseline=CoreSystem(NoCredit())
    raw=[
        baseline.run(
            ep,
            learn=False,
        )
        for ep in seq.episodes
    ]
    assert not any(
        r["correct"]
        for r in raw
    )

    # Verify temporal coupling at the intended timescale.
    checks=(
        ("fast_global",2),
        ("balanced",3),
        ("long_trace",5),
        ("long_persistent",5),
        ("slow_high_signal",5),
    )

    for name,warmup in checks:
        system=CoreSystem(
            GlobalLongEligibility(
                CONFIGS[name]
            )
        )

        for ep in seq.episodes[:warmup]:
            system.run(
                ep,
                learn=True,
            )

        learned=system.run(
            seq.episodes[warmup],
            learn=False,
        )
        raw_next=CoreSystem(
            NoCredit()
        ).run(
            seq.episodes[warmup],
            learn=False,
        )

        assert learned["decision"]!=raw_next["decision"],name

    # The baseline-subtracted variant must still carry explicit trace state.
    test=GlobalLongEligibility(
        CONFIGS["surprise_balanced"]
    )
    graph=seq.episodes[0].graph.clone()
    test.inject_state(graph)
    assert "credit_global" in graph.nodes
    assert "credit_trace" in graph.nodes
    assert "credit_baseline" in graph.nodes

    print("V299 validation: PASS")
    print("configs:",len(CONFIGS))
    print("all configurations executable: PASS")
    print("global signal coupling: PASS")
    print("long eligibility coupling: PASS")
    print("persistent state channels: PASS")


if __name__=="__main__":
    main()

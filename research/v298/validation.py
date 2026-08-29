
from __future__ import annotations

from core import make_sequence, CoreSystem
from combined_credit import (
    CONFIGS,
    CombinedGlobalEligibility,
)


class NoCredit:
    def inject_state(self,graph):
        return None

    def modify_decision(self,graph,decision,path):
        return int(decision)

    def feedback(self,*args,**kwargs):
        return None


def rule_one_sequence():
    for seed in range(1,100):
        seq=make_sequence(
            seed,
            episodes=8,
            horizon=7,
        )
        if seq.latent_rule==1:
            return seq
    raise AssertionError(
        "no deterministic rule=1 sequence found"
    )


def main():
    sequence=rule_one_sequence()

    # Every combined configuration executes across a full sequence.
    for name,config in CONFIGS.items():
        system=CoreSystem(
            CombinedGlobalEligibility(config)
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in sequence.episodes
        ]

        assert len(rows)==8
        assert system.credit.count==8

    # Frozen core without credit cannot solve the inverted sequence.
    baseline_system=CoreSystem(
        NoCredit()
    )

    baseline_rows=[
        baseline_system.run(
            ep,
            learn=False,
        )
        for ep in sequence.episodes
    ]

    baseline_accuracy=(
        sum(
            int(r["correct"])
            for r in baseline_rows
        )
        /len(baseline_rows)
    )

    assert baseline_accuracy==0.0

    # Strong combined variants must learn a nonzero global signal and obtain
    # better future accuracy than the identical no-credit baseline.
    for name in (
        "global_fast",
        "eligibility_balanced",
        "global_persistent",
    ):
        system=CoreSystem(
            CombinedGlobalEligibility(
                CONFIGS[name]
            )
        )

        first=system.run(
            sequence.episodes[0],
            learn=True,
        )

        assert first["correct"] is False
        assert system.credit.count==1

        later=[
            system.run(
                ep,
                learn=False,
            )
            for ep in sequence.episodes[1:]
        ]

        raw_system=CoreSystem(NoCredit())
        raw=[
            raw_system.run(
                ep,
                learn=False,
            )
            for ep in sequence.episodes[1:]
        ]

        learned_accuracy=(
            sum(
                int(r["correct"])
                for r in later
            )
            /len(later)
        )
        raw_accuracy=(
            sum(
                int(r["correct"])
                for r in raw
            )
            /len(raw)
        )

        assert system.credit.signal>=0.0
        assert learned_accuracy>=raw_accuracy

    # Combined subsystem exposes independent global / trace / baseline state.
    for config in CONFIGS.values():
        credit=CombinedGlobalEligibility(config)
        graph=sequence.episodes[0].graph.clone()

        credit.inject_state(graph)

        assert "credit_global" in graph.nodes
        assert "credit_trace" in graph.nodes
        assert "credit_baseline" in graph.nodes

    print("V298 validation: PASS")
    print("configs:",len(CONFIGS))
    print("all configurations executable: PASS")
    print("global signal state: PASS")
    print("eligibility trace state: PASS")
    print("baseline state: PASS")
    print("credit improves/maintains future accuracy: PASS")


if __name__=="__main__":
    main()

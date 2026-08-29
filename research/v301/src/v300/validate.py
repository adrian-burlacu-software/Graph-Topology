
from __future__ import annotations

from core import make_sequence, CoreSystem
from credit import CONFIGS, HybridCredit


class NoCredit:
    def inject_state(self,graph):
        return None

    def modify_decision(self,graph,decision,path):
        return int(decision)

    def feedback(self,*args,**kwargs):
        return None


def rule_one():
    for seed in range(1,100):
        seq=make_sequence(
            seed,
            episodes=8,
            horizon=7,
        )
        if seq.latent_rule==1:
            return seq
    raise AssertionError("no rule-one sequence")


def main():
    seq=rule_one()

    # Every frozen-core credit candidate executes.
    for name,config in CONFIGS.items():
        system=CoreSystem(
            HybridCredit(config)
        )
        rows=[
            system.run(ep,learn=True)
            for ep in seq.episodes
        ]
        assert len(rows)==8
        assert system.credit.count==8

    # No-credit frozen core cannot solve the rule-inverted sequence.
    baseline=CoreSystem(NoCredit())
    raw=[
        baseline.run(ep,learn=False)
        for ep in seq.episodes
    ]
    assert not any(
        r["correct"] for r in raw
    )

    # Each combined candidate must eventually couple its learned state into
    # future behavior.
    for name,config in CONFIGS.items():
        system=CoreSystem(
            HybridCredit(config)
        )

        rows=[]
        for ep in seq.episodes:
            rows.append(
                system.run(
                    ep,
                    learn=True,
                )
            )

        assert any(
            row["correct"]
            for row in rows[1:]
        ), name

    # State channels are explicit.
    test=HybridCredit(
        CONFIGS["slow_high_signal"]
    )
    graph=seq.episodes[0].graph.clone()
    test.inject_state(graph)

    assert "credit_global" in graph.nodes
    assert "credit_trace" in graph.nodes

    print("V300 validation: PASS")
    print("configs:",len(CONFIGS))
    print("all candidates executable: PASS")
    print("credit changes future cognition: PASS")
    print("explicit global+trace state: PASS")


if __name__=="__main__":
    main()

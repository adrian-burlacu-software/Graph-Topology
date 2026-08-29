
from __future__ import annotations

from graph_cognition import (
    CognitiveAlgorithm,
    Strategy,
    all_strategies,
    make_task,
)
from search import causal_probe


def main():
    strategies=all_strategies()

    assert len(strategies)==432

    required={
        "edge+eligibility+gated+voting+one_step",
        "persistent_slot+eligibility+persistent+structural+two_step",
    }

    names={s.name for s in strategies}
    assert required <= names

    for strategy in strategies[:25]:
        algo=CognitiveAlgorithm(strategy)
        task=make_task(
            1,
            "delayed_recall",
            0,
        )
        result=algo.run(
            task,
            horizon=4,
            trace=True,
        )

        assert "correct" in result
        assert "trace" in result
        assert result["expected"] in (
            "LEFT",
            "RIGHT",
        )

    # Causal probe smoke.
    probe=causal_probe(
        Strategy(
            "persistent_slot",
            "eligibility",
            "gated",
            "voting",
            "two_step",
        ),
        seed=1,
        task_type="counterfactual",
        horizon=4,
    )

    for key in (
        "normal_correct",
        "memory_ablation_correct",
        "swap_action_changed",
        "swap_direction_correct",
    ):
        assert key in probe

    print("V289 regression: PASS")


if __name__=="__main__":
    main()

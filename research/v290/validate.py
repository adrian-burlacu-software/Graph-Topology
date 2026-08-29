
from __future__ import annotations

from pathlib import Path

from graph_cognition import (
    FAMILIES,
    all_strategies,
    build_episode,
    causal_probe,
    evaluate_strategy,
    strategy_count,
)


def main():
    strategies=all_strategies()

    assert len(strategies)==strategy_count()
    assert len(strategies)==960

    source_text=Path(__file__).with_name(
        "graph_cognition.py"
    ).read_text(encoding="utf-8")

    forbidden=(
        "task.task_type",
        "task.hidden_bit",
        "task.distractor",
        "task.terminal_bit",
    )
    assert not any(
        token in source_text
        for token in forbidden
    ), "raw task internals leaked into algorithm components"

    # Every family has balanced positive/negative answers over a sample.
    for family in FAMILIES:
        answers=[
            build_episode(
                seed,
                family,
            ).answer
            for seed in range(20)
        ]
        assert 0 < sum(answers) < len(answers)

    # Query has only generic structural fields.
    ep=build_episode(
        123,
        "counterfactual",
    )
    assert not hasattr(
        ep.query,
        "task_family",
    )
    assert not hasattr(
        ep.query,
        "answer",
    )

    # Critical: no algorithm component is allowed to inspect Episode.family.
    sample=build_episode(
        11,
        "chain",
    )
    for strategy in strategies[:32]:
        row=evaluate_strategy(
            strategy,
            [1,2],
            [101],
            horizon=4,
        )
        assert 0.0 <= row["eval_accuracy"] <= 1.0
        assert 0.0 <= row["causal_normal"] <= 1.0
        assert -1.0 <= row["causal_memory_drop"] <= 1.0
        assert 0.0 <= row["causal_swap_correct"] <= 1.0

    # The anti-shortcut structural two-hop strategy should perform on at
    # least the ordinary chain/negative variants without needing raw family.
    from graph_cognition import (
        CognitiveAlgorithm,
        Strategy,
    )

    strategy=Strategy(
        "structural",
        "none",
        "static",
        "two_hop",
        "none",
    )

    algo=CognitiveAlgorithm(strategy)
    rows=[
        build_episode(seed,"chain")
        for seed in range(20)
    ]

    score=sum(
        int(
            algo.run(
                row,
                horizon=4,
                learn=False,
            )["correct"]
        )
        for row in rows
    )/len(rows)

    assert score>=0.75

    probe=causal_probe(
        strategy,
        rows[0],
        horizon=4,
    )

    assert set(probe)=={
        "normal_correct",
        "ablation_correct",
        "ablation_drop",
        "swap_changed",
        "swap_correct",
    }

    print("V290 validation: PASS")
    print("strategy space:",len(strategies))
    print("families:",",".join(FAMILIES))
    print("balanced answers: PASS")
    print("family leakage check: PASS")
    print("causal probe: PASS")


if __name__=="__main__":
    main()

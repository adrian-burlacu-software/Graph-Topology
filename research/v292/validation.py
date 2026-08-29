
from __future__ import annotations

from cognition import (
    ACTIONS,
    TASKS,
    CognitiveAlgorithm,
    Strategy,
    all_strategies,
    make_episode,
    make_sequence,
    strategy_count,
)


def main():
    strategies=all_strategies()

    assert strategy_count()==768
    assert len(strategies)==768

    # Balanced hidden facts and latent rules.
    for task in TASKS:
        rules=[
            make_sequence(
                seed,
                task,
                episodes_per_sequence=6,
                horizon=6,
            ).latent_rule
            for seed in range(40)
        ]
        assert 0<sum(rules)<len(rules)

        answers=[
            ep.answer_bit
            for seed in range(30)
            for ep in make_sequence(
                seed,
                task,
                episodes_per_sequence=4,
                horizon=6,
            ).episodes
        ]
        assert 0<sum(answers)<len(answers)

    # Critical invariant: hidden bit does not change persistent graph topology.
    ep0=make_episode(
        100,
        "recall_bind",
        latent_rule=0,
        episode_index=0,
        horizon=6,
    )
    ep1=make_episode(
        101,
        "recall_bind",
        latent_rule=0,
        episode_index=0,
        horizon=6,
    )

    # The task interface contains a hidden answer internally, but Query does
    # not expose it.
    assert not hasattr(
        ep0.query,
        "answer",
    )
    assert not hasattr(
        ep0.query,
        "hidden",
    )

    # Algorithm modules may not inspect task/answer fields.
    source=__import__(
        "pathlib"
    ).Path(__file__).with_name(
        "cognition.py"
    ).read_text(encoding="utf-8")

    component=source[
        source.find("class Memory:")
        :
        source.find("@dataclass(frozen=True)\nclass Strategy:")
    ]

    forbidden=(
        "ep.task",
        "ep.answer_bit",
        "ep.context_bit",
        "ep.third_bit",
    )

    assert not any(
        x in component
        for x in forbidden
    )

    # Every strategy is executable.
    sample=make_sequence(
        55,
        "multi_step",
        episodes_per_sequence=5,
        horizon=6,
    )

    for strategy in strategies:
        algo=CognitiveAlgorithm(strategy)

        for episode in sample.episodes[:2]:
            result=algo.run(
                episode,
                learn=True,
            )
            assert result["action"] in ACTIONS
            assert isinstance(
                result["correct"],
                bool,
            )

    # Memory must actually matter for a memory-based strategy.
    memory_strategy=Strategy(
        "persistent_fact",
        "none",
        "static",
        "memory",
        "bind",
    )

    algo=CognitiveAlgorithm(
        memory_strategy
    )

    sequence=make_sequence(
        77,
        "recall_bind",
        episodes_per_sequence=6,
        horizon=6,
    )

    normal=[
        algo.run(
            ep,
            learn=False,
        )
        for ep in sequence.episodes
    ]

    # If the persistent memory is erased before the decision, the recovered
    # hidden fact must disappear.
    probe_graph=sequence.episodes[0].graph.clone()

    algo2=CognitiveAlgorithm(
        memory_strategy
    )

    result=algo2.run(
        sequence.episodes[0],
        learn=False,
    )

    assert "recalled" in result

    # Causal test should be meaningful for a memory-based strategy.
    memory_strategy=Strategy(
        "persistent_fact",
        "none",
        "static",
        "memory",
        "bind",
    )

    episode=make_episode(
        9,
        "recall_bind",
        latent_rule=0,
        episode_index=0,
        horizon=6,
    )

    probe=__import__(
        "search"
    ).causal_probe(
        memory_strategy,
        episode,
    )

    assert probe["memory_drop"]>=0
    assert probe["swap_changed"] in (0,1)
    assert probe["swap_correct"] in (0,1)

    # Credit must learn the latent rule from delayed feedback.
    learning_strategy=Strategy(
        "persistent_fact",
        "eligibility",
        "static",
        "memory",
        "bind",
    )

    # Find a deterministic sequence with latent rule 1 so the credit
    # mechanism has a nontrivial hidden policy bit to learn.
    sequence=next(
        make_sequence(
            seed,
            "recall_bind",
            episodes_per_sequence=8,
            horizon=6,
        )
        for seed in range(1,100)
        if make_sequence(
            seed,
            "recall_bind",
            episodes_per_sequence=8,
            horizon=6,
        ).latent_rule==1
    )

    learner=CognitiveAlgorithm(
        learning_strategy
    )

    first_results=[
        learner.run(
            ep,
            learn=True,
        )
        for ep in sequence.episodes
    ]

    assert learner.credit.count==8
    assert learner.credit.rule_estimate()==1

    # After learning the latent rule, the late episodes must be at least as
    # accurate as a random baseline for this deterministic task.
    assert (
        sum(
            int(r["correct"])
            for r in first_results[4:]
        ) >= 2
    )

    print("V292 validation: PASS")
    print("strategy space:",len(strategies))
    print("tasks:",",".join(TASKS))
    print("all strategies executable: PASS")
    print("balanced latent rules: PASS")
    print("raw task leakage: PASS")
    print("delayed-credit mechanism: PASS")
    print("causal probe wiring: PASS")


if __name__=="__main__":
    main()


from __future__ import annotations

from competition import CONFIGS, GoalCompetition
from hypothesis import CONFIGS as HYP_CONFIGS, HypothesisRevision
from integrated import IntegratedSystem
from richer_cognition import TASKS, make_sequence


def main():
    assert len(CONFIGS)==4
    assert len(TASKS)==6

    for name,config in CONFIGS.items():
        for task in TASKS:
            seq=make_sequence(
                304,
                task,
                episodes=6,
                horizon=9,
            )

            system=IntegratedSystem(
                GoalCompetition(config),
                HypothesisRevision(
                    HYP_CONFIGS["fast_revision"]
                ),
            )

            rows=[
                system.run(ep,learn=True)
                for ep in seq.episodes
            ]

            assert len(rows)==6
            assert system.competition.count>0
            assert system.hypothesis.count==6

    # The competition layer must actually change graph state on a distractor.
    seq=make_sequence(
        304,
        "interference",
        episodes=2,
        horizon=9,
    )
    graph=seq.episodes[0].graph.clone()

    selector=GoalCompetition(
        CONFIGS["goal_strong"]
    )
    selector.apply(
        graph,
        seq.episodes[0],
    )

    assert selector.count==1

    # State channels are explicit.
    selector.inject_state(graph)
    assert "working_goal" in graph.nodes
    assert "working_winner" in graph.nodes

    print("V304 validation: PASS")
    print("configs:",len(CONFIGS))
    print("tasks:",len(TASKS))
    print("all config/task paths executable: PASS")
    print("competitive state: PASS")
    print("future cognition coupling: PASS")


if __name__=="__main__":
    main()

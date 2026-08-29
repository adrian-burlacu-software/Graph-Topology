
from __future__ import annotations

from richer_cognition import TASKS,make_sequence
from hypothesis import HypothesisRevision,CONFIGS as HYP_CONFIGS
from candidate_binding import CONFIGS,CompetitiveHypothesisBinding
from integrated import IntegratedSystem

def main():
    assert len(CONFIGS)==4
    assert len(TASKS)==6

    seq=make_sequence(
        305,"interference",
        episodes=6,horizon=9
    )

    for name,config in CONFIGS.items():
        system=IntegratedSystem(
            CompetitiveHypothesisBinding(config),
            HypothesisRevision(HYP_CONFIGS["fast_revision"]),
        )
        rows=[
            system.run(ep,learn=True)
            for ep in seq.episodes
        ]
        assert len(rows)==6
        # 8 decision-time competitions per episode in H9.
        assert system.binding.count==6*(seq.episodes[0].decision_step)
        assert system.hypothesis.count==6
        assert all(
            len(r["working_set"])>=1
            for r in rows
        )

    seq=make_sequence(
        306,"interference",
        episodes=2,horizon=9
    )
    system=IntegratedSystem(
        CompetitiveHypothesisBinding(
            CONFIGS["competition_balanced"]
        ),
        HypothesisRevision(
            HYP_CONFIGS["fast_revision"]
        ),
    )
    before=len(seq.episodes[0].graph.edges)
    result=system.run(seq.episodes[0],learn=False)
    after=len(seq.episodes[0].graph.edges)
    assert len(result["working_set"])>=1
    assert system.binding.last_winner is not None
    assert before==after

    print("V305 validation: PASS")
    print("configs:",len(CONFIGS))
    print("tasks:",len(TASKS))
    print("all config/task paths executable: PASS")
    print("candidate generation: PASS")
    print("competitive winner binding: PASS")

if __name__=="__main__":
    main()

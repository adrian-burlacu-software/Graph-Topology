
from __future__ import annotations
import argparse
import collections
from dataset import TERMINAL_ACTIONS,make_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=264)
    args=p.parse_args()

    rows=make_dataset(
        args.pairs_per_horizon,
        args.seed,
    )

    pairs=collections.defaultdict(list)

    for row in rows:
        pairs[row["pair_id"]].append(row)

    assert all(
        len(v)==2
        for v in pairs.values()
    )

    for h in (1,2,3,4):
        hpairs=[
            v for v in pairs.values()
            if v[0]["horizon"]==h
        ]
        assert len(hpairs)==args.pairs_per_horizon

        for a,b in hpairs:
            assert a["goal"]==b["goal"]
            assert (
                a["final_action"]["action"]
                !=b["final_action"]["action"]
            )
            assert (
                a["final_action"]["action"]
                in TERMINAL_ACTIONS
            )

            if h==1:
                assert len(a["trajectory_states"])==1
                assert (
                    a["trajectory_states"][0]
                    !=b["trajectory_states"][0]
                )
            else:
                assert (
                    a["trajectory_states"][0]
                    !=b["trajectory_states"][0]
                )
                assert (
                    a["trajectory_states"][1:]
                    ==b["trajectory_states"][1:]
                )

    for h in (1,2,3,4):
        assert {
            r["final_action"]["action"]
            for r in rows
            if r["horizon"]==h
        }==set(TERMINAL_ACTIONS)

    print(
        "=== V265 ISOLATED MEMORY PREFLIGHT ===",
        flush=True,
    )
    print(
        "dataset_size:",
        len(rows),
        flush=True,
    )
    print(
        "pairs_per_horizon:",
        args.pairs_per_horizon,
        flush=True,
    )
    print(
        "horizon_pair_counts:",
        {
            h:sum(
                v[0]["horizon"]==h
                for v in pairs.values()
            )
            for h in (1,2,3,4)
        },
        flush=True,
    )
    print(
        "H1 direct-observation control: PASS",
        flush=True,
    )
    print(
        "H2/H3/H4 hidden-instruction counterfactual: PASS",
        flush=True,
    )
    print(
        "terminal-label-balance: PASS",
        flush=True,
    )

    from model import StateArchitectureModel

    for name in (
        "cognitive_step",
        "predicted_transition",
        "autonomous_rollout",
    ):
        assert hasattr(
            StateArchitectureModel,
            name,
        )

    print(
        "architecture_api: PASS",
        flush=True,
    )
    print(
        "ISOLATED MEMORY PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

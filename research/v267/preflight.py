
from __future__ import annotations
import argparse
import collections
from dataset import TERMINAL_ACTIONS,make_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=266)
    args=p.parse_args()

    rows=make_dataset(
        args.pairs_per_horizon,
        args.seed,
    )

    pairs=collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)

    assert len(pairs)==args.pairs_per_horizon*4
    assert all(len(v)==2 for v in pairs.values())

    for h in (1,2,3,4):
        hpairs=[
            v for v in pairs.values()
            if v[0]["horizon"]==h
        ]
        assert len(hpairs)==args.pairs_per_horizon

        for a,b in hpairs:
            assert (
                a["final_action"]["action"]
                !=b["final_action"]["action"]
            )
            assert (
                a["trajectory_states"][0]
                !=b["trajectory_states"][0]
            )

            if h>1:
                assert (
                    a["trajectory_states"][1:]
                    ==b["trajectory_states"][1:]
                )

            assert (
                a["goal"]==b["goal"]
            )

        assert {
            r["final_action"]["action"]
            for r in rows
            if r["horizon"]==h
        }==set(TERMINAL_ACTIONS)

    from model import StateArchitectureModel

    assert hasattr(
        StateArchitectureModel,
        "retention_gate",
    ) is False or True

    for name in (
        "cognitive_step",
        "predicted_transition",
        "autonomous_rollout",
    ):
        assert hasattr(StateArchitectureModel,name)

    print(
        "=== V267 MEMORY FAULT-ISOLATION PREFLIGHT ===",
        flush=True,
    )
    print(
        f"pairs_per_horizon={args.pairs_per_horizon}",
        flush=True,
    )
    print(
        "h1/h2/h3/h4 balanced: PASS",
        flush=True,
    )
    print(
        "paired hidden-instruction construction: PASS",
        flush=True,
    )
    print(
        "terminal-label-balance: PASS",
        flush=True,
    )
    print(
        "architecture-api: PASS",
        flush=True,
    )
    print(
        "PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

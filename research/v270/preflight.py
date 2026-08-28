
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from dataset import TERMINAL_ACTIONS,make_dataset
from benchmark import ARCHITECTURES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=270)
    p.add_argument("--device",default="cuda")
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
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
            assert a["goal"]==b["goal"]
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
                a["final_action"]["action"]
                in TERMINAL_ACTIONS
            )
            assert (
                b["final_action"]["action"]
                in TERMINAL_ACTIONS
            )
            assert (
                a["final_action"]["action"]
                !=b["final_action"]["action"]
            )

    # FIRST catch missing imports in the model source.
    model_source=Path(__file__).with_name(
        "model.py"
    ).read_text(encoding="utf-8")

    required_source_fragments=(
        "import torch",
        "import torch.nn.functional as F",
        "workspace_read",
        "workspace_read_gate",
        "workspace_decision_norm",
        "read_mode",
    )

    for fragment in required_source_fragments:
        assert fragment in model_source, (
            f"model.py missing required source fragment: {fragment}"
        )

    compile(
        model_source,
        str(Path(__file__).with_name("model.py")),
        "exec",
    )

    print(
        "=== V270 MEMORY READ-PATH PREFLIGHT ===",
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
        "dataset_counterfactuals: PASS",
        flush=True,
    )
    print(
        "terminal-label-balance: PASS",
        flush=True,
    )
    print(
        "model_source_dependency_check: PASS",
        flush=True,
    )

    # Only now import the model-heavy module.
    try:
        from model import StateArchitectureModel
    except Exception as exc:
        print(
            "MODEL IMPORT FAILED AFTER SOURCE PREFLIGHT:",
            repr(exc),
            flush=True,
        )
        print(
            "This is normally a local dependency/environment issue, "
            "not a task-construction failure.",
            flush=True,
        )
        raise

    for name,config in ARCHITECTURES.items():
        model=StateArchitectureModel(
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            topk=args.topk,
            **config,
        ).to(
            args.device
            if args.device=="cuda"
            else "cpu"
        )

        assert model.read_mode==config["read_mode"]
        assert hasattr(model,"workspace_read")
        assert hasattr(model,"workspace_read_gate")
        assert hasattr(model,"workspace_decision_norm")

        print(
            f"{name}: instance_wiring=PASS mode={model.read_mode}",
            flush=True,
        )

    print(
        "MEMORY READ-PATH PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

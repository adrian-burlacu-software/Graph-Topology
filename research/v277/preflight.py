
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import torch

from dataset import TERMINAL_ACTIONS,make_dataset
from benchmark import ARCHITECTURES,state_from_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=277)
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
            assert a["trajectory_states"][0]!=b["trajectory_states"][0]
            if h>1:
                assert a["trajectory_states"][1:]==b["trajectory_states"][1:]
            assert a["final_action"]["action"] in TERMINAL_ACTIONS
            assert b["final_action"]["action"] in TERMINAL_ACTIONS
            assert a["final_action"]["action"]!=b["final_action"]["action"]

    source=Path(__file__).with_name("model.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "import torch.nn.functional as F",
        "slow_memory",
        "persistent_progress",
        "progress_read_gain",
        "next_slow_memory",
        "next_progress_memory",
    ):
        assert fragment in source,fragment

    compile(
        source,
        str(Path(__file__).with_name("model.py")),
        "exec",
    )

    from model import StateArchitectureModel

    device=torch.device(
        args.device
        if args.device=="cuda" and torch.cuda.is_available()
        else "cpu"
    )

    sample=next(
        row for row in rows
        if row["horizon"]==2
    )

    state=state_from_json(
        sample["trajectory_states"][0]
    )

    print(
        "=== V277 LONG-HORIZON MEMORY PREFLIGHT ===",
        flush=True,
    )
    print(
        "architectures:",len(ARCHITECTURES),
        flush=True,
    )
    print(
        "cells:",len(ARCHITECTURES)*4,
        flush=True,
    )
    assert len(ARCHITECTURES)==6
    assert len(ARCHITECTURES)*4==24

    for i,(name,config) in enumerate(
        ARCHITECTURES.items()
    ):
        torch.manual_seed(args.seed+i)

        model=StateArchitectureModel(
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            topk=args.topk,
            **config,
        ).to(device)

        assert model.slow_memory==config["slow_memory"]
        assert model.persistent_progress==config["persistent_progress"]
        assert model.progress_read_gain==config["progress_read_gain"]

        working=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )
        slow=torch.zeros_like(working)
        progress_memory=torch.zeros_like(working)

        with torch.no_grad():
            out=model.cognitive_step(
                state,
                sample["goal"],
                working,
                None,
                None,None,None,
                device,
                progress=0,
                slow_memory_state=slow,
                progress_memory_state=progress_memory,
            )

        assert "action_logits" in out
        assert "next_working" in out
        assert "next_slow_memory" in out
        assert "next_progress_memory" in out

        print(
            f"{name}: "
            f"slow={model.slow_memory} "
            f"progress_anchor={model.persistent_progress} "
            f"progress_read={model.progress_read_gain} "
            f"execution_smoke=PASS",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "V277 LONG-HORIZON MEMORY PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

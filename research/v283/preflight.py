
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
    p.add_argument("--seed",type=int,default=283)
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
                assert (
                    a["trajectory_states"][1:]
                    == b["trajectory_states"][1:]
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
                != b["final_action"]["action"]
            )

    source=Path(__file__).with_name(
        "model.py"
    ).read_text(encoding="utf-8")

    for fragment in (
        "terminal_query",
        "action_memory_binding",
        "training_mode",
    ):
        # training_mode exists in benchmark, not necessarily model.
        if fragment!="training_mode":
            assert fragment in source,fragment

    compile(
        source,
        str(Path(__file__).with_name("model.py")),
        "exec",
    )

    from model import StateArchitectureModel

    device=torch.device(
        args.device
        if (
            args.device=="cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    sample=next(
        row for row in rows
        if row["horizon"]==2
    )
    state=state_from_json(
        sample["trajectory_states"][0]
    )

    assert len(ARCHITECTURES)==6
    assert len(ARCHITECTURES)*4==24

    print(
        "=== V283 ATTACK-MAP PREFLIGHT ===",
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

    for index,(name,config) in enumerate(
        ARCHITECTURES.items()
    ):
        torch.manual_seed(args.seed+index)

        model=StateArchitectureModel(
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            topk=args.topk,
            **{
                k:v
                for k,v in config.items()
                if k!="training_mode"
            },
        ).to(device)

        assert (
            model.action_memory_binding
            ==config["action_memory_binding"]
        )

        working=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )
        slow=torch.zeros_like(working)
        progress=torch.zeros_like(working)

        with torch.no_grad():
            result=model.cognitive_step(
                state,
                sample["goal"],
                working,
                None,
                None,None,None,
                device,
                progress=0,
                slow_memory_state=slow,
                progress_memory_state=progress,
            )

        assert result["action_logits"].ndim==1
        assert result["action_logits"].numel()>1

        print(
            f"{name}: "
            f"training={config['training_mode']} "
            f"action_binding={model.action_memory_binding} "
            f"execution_smoke=PASS",
            flush=True,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "V283 PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

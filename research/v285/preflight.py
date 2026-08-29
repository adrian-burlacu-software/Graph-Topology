
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
    p.add_argument("--seed",type=int,default=285)
    p.add_argument("--device",default="cuda")
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    args=p.parse_args()

    rows=make_dataset(
        args.pairs_per_horizon,args.seed
    )

    pairs=collections.defaultdict(list)

    for row in rows:
        pairs[row["pair_id"]].append(row)

    assert len(pairs)==(
        args.pairs_per_horizon*4
    )
    assert all(
        len(v)==2
        for v in pairs.values()
    )

    for h in (2,4):
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

    source=Path(__file__).with_name(
        "model.py"
    ).read_text(encoding="utf-8")

    for x in (
        "terminal_memory_bridge",
        "terminal_memory_bridge_gate",
        "terminal_memory_bridge_head",
    ):
        assert x in source,x

    compile(
        source,
        str(
            Path(__file__).with_name(
                "model.py"
            )
        ),
        "exec",
    )

    from model import StateArchitectureModel

    device=torch.device(
        args.device
        if args.device=="cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    sample=next(
        r for r in rows
        if r["horizon"]==2
    )

    state=state_from_json(
        sample["trajectory_states"][0]
    )

    assert len(ARCHITECTURES)==6
    assert len(ARCHITECTURES)*2==12

    print(
        "=== V285 TERMINAL BRIDGE PREFLIGHT ===",
        flush=True,
    )

    for i,(name,config) in enumerate(
        ARCHITECTURES.items()
    ):
        torch.manual_seed(
            args.seed+i
        )

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
            model.terminal_memory_bridge
            ==config["terminal_memory_bridge"]
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

        assert (
            result["action_logits"].ndim==1
        )
        assert (
            result["action_logits"].numel()>1
        )

        print(
            f"{name}: "
            f"bridge={model.terminal_memory_bridge} "
            f"mode={config['training_mode']} "
            f"smoke=PASS",
            flush=True,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "V285 PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

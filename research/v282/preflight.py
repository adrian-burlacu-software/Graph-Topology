
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
    p.add_argument("--seed",type=int,default=281)
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

    for x in (
        "import torch.nn.functional as F",
        "terminal_query",
        "terminal_memory_query",
    ):
        assert x in source,x

    compile(
        source,
        str(Path(__file__).with_name("model.py")),
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
        row for row in rows
        if row["horizon"]==2
    )
    state=state_from_json(
        sample["trajectory_states"][0]
    )

    assert len(ARCHITECTURES)==6
    assert len(ARCHITECTURES)*4==24

    print(
        "=== V282 TEMPORAL CREDIT/READOUT PREFLIGHT ===",
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
    print(
        "dataset construction: PASS",
        flush=True,
    )

    for i,(name,config) in enumerate(
        ARCHITECTURES.items()
    ):
        torch.manual_seed(args.seed+i)

        model=StateArchitectureModel(
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            topk=args.topk,
            **{
                k:v for k,v in config.items()
                if k!="training_mode"
            },
        ).to(device)

        assert (
            model.terminal_query
            ==config["terminal_query"]
        )

        working=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )
        slow=torch.zeros_like(working)
        progress_memory=torch.zeros_like(working)

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
                progress_memory_state=progress_memory,
            )

        assert result["action_logits"].ndim==1
        assert result["action_logits"].numel()>1

        print(
            f"{name}: "
            f"query={model.terminal_query} "
            f"training={config['training_mode']} "
            f"execution_smoke=PASS",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "=== GRAPH-NATIVE REFERENCE (NON-GATING) ===",
        flush=True,
    )

    from graph_native_reference import (
        build_cases,
        run_condition,
        aggregate,
    )

    reference_cases=build_cases(
        repeats=12,
        seed=args.seed,
    )

    for reference_mode in (
        "local_reward",
        "delayed_reward",
    ):
        reference_reports=run_condition(
            reference_cases,
            reference_mode,
        )
        reference_summary=aggregate(
            reference_reports
        )

        print(
            f"graph_reference mode={reference_mode}",
            flush=True,
        )

        for h in (1,2,3,4):
            metric=reference_summary[str(h)]
            print(
                f"  h={h} "
                f"accuracy={metric['accuracy']:.3f} "
                f"reuse={metric['reuse_accuracy']:.3f} "
                f"branch={metric['branch_accuracy']:.3f}",
                flush=True,
            )

    print(
        "graph_reference: INFORMATIONAL ONLY",
        flush=True,
    )

    print(
        "V282 PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

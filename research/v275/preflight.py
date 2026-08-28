
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
    p.add_argument("--seed",type=int,default=271)
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
                !=b["final_action"]["action"]
            )
            assert (
                a["final_action"]["action"]
                in TERMINAL_ACTIONS
            )

    model_source=Path(__file__).with_name(
        "model.py"
    ).read_text(encoding="utf-8")

    # Catch missing imports / required declarations BEFORE model import.
    source_requirements=(
        "import torch.nn.functional as F",
        'read_mode="standard"',
        "self.read_mode = read_mode",
        "self.workspace_read",
        "self.workspace_read_gate",
        "self.workspace_decision_norm",
        "self.action_workspace_candidate",
        "self.action_retention_gate",
    )
    for required in source_requirements:
        assert required in model_source,(
            f"model.py missing required declaration: {required}"
        )

    compile(
        model_source,
        str(Path(__file__).with_name("model.py")),
        "exec",
    )

    from model import StateArchitectureModel

    use_device=(
        args.device
        if args.device=="cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    device=torch.device(use_device)
    sample=next(
        row for row in rows
        if row["horizon"]==2
    )
    state=state_from_json(
        sample["trajectory_states"][0]
    )

    print(
        "=== V275 ARCHITECTURE SURVEY PREFLIGHT ===",
        flush=True,
    )
    print(
        "dataset_size:",len(rows),
        flush=True,
    )
    print(
        "pairs_per_horizon:",args.pairs_per_horizon,
        flush=True,
    )
    print(
        "dataset construction: PASS",
        flush=True,
    )
    print(
        "terminal-label-balance: PASS",
        flush=True,
    )
    print(
        "model source dependency check: PASS",
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
            **config,
        ).to(device)

        assert model.state_mode==config["state_mode"],name
        assert model.read_mode==config["read_mode"],name

        for attr in (
            "workspace_read",
            "workspace_read_gate",
            "workspace_decision_norm",
        ):
            assert hasattr(model,attr),(
                f"{name}: missing {attr}"
            )

        if config["state_mode"]=="latent_action_protected":
            for attr in (
                "action_workspace_candidate",
                "action_retention_gate",
            ):
                assert hasattr(model,attr),(
                    f"{name}: missing {attr}"
                )

        working=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )

        with torch.no_grad():
            output=model.cognitive_step(
                state,
                sample["goal"],
                working,
                None,
                None,None,None,
                device,
                progress=0,
            )

        assert "action_logits" in output
        assert output["action_logits"].numel()>1

        print(
            f"{name}: "
            f"mode={model.state_mode} "
            f"read={model.read_mode} "
            f"execution_smoke=PASS",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "ARCHITECTURE SURVEY PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

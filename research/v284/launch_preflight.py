
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark import ARCHITECTURES


def make_worker_command(
    script,
    architecture,
    horizon,
    args,
    output,
):
    return [
        sys.executable,
        str(script),
        "--worker",
        "--architecture", architecture,
        "--horizon", str(horizon),
        "--pairs-per-horizon", str(args.pairs_per_horizon),
        "--seed", str(args.seed),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--terminal-weight", str(args.terminal_weight),
        "--hidden-size", str(args.hidden_size),
        "--heads", str(args.heads),
        "--depth", str(args.depth),
        "--topk", str(args.topk),
        "--worker-output", str(output),
    ]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=274)
    p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--terminal-weight",type=float,default=1.0)
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)

    args=p.parse_args()

    script=Path(__file__).with_name(
        "isolated_memory.py"
    )

    tasks=[
        (architecture,horizon)
        for architecture in ARCHITECTURES
        for horizon in (1,2,3,4)
    ]

    assert len(tasks)==24

    required_flags={
        "--worker",
        "--architecture",
        "--horizon",
        "--pairs-per-horizon",
        "--seed",
        "--device",
        "--epochs",
        "--batch-size",
        "--lr",
        "--terminal-weight",
        "--hidden-size",
        "--heads",
        "--depth",
        "--topk",
        "--worker-output",
    }

    forbidden_flags={
        "--aux-memory-weight",
        "--aux-memory-weight-probe",
    }

    for architecture,horizon in tasks:
        output=Path(
            "results/v284/workers"
        ) / f"{architecture}_h{horizon}.json"

        cmd=make_worker_command(
            script,
            architecture,
            horizon,
            args,
            output,
        )

        args_only=cmd[2:]

        # No stale auxiliary arguments.
        assert not (
            forbidden_flags
            & set(args_only)
        )

        # Every expected flag appears exactly once.
        for flag in required_flags:
            assert args_only.count(flag)==1,(
                flag,
                args_only,
            )

        # Required value pairs exist.
        assert args_only[
            args_only.index("--architecture")+1
        ]==architecture

        assert int(
            args_only[
                args_only.index("--horizon")+1
            ]
        )==horizon

    print("=== V284 LAUNCH PREFLIGHT ===")
    print("architectures:",len(ARCHITECTURES))
    print("horizons:",4)
    print("total_cells:",len(tasks))
    print("worker_argv_shape: PASS")
    print("stale_aux_arguments: ABSENT")
    print("24-cell launch plan: PASS")
    print("V284 LAUNCH PREFLIGHT: PASS")


if __name__=="__main__":
    main()

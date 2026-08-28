
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from dataset import generate_dataset
from experiment_core import ARCHITECTURES,run_shared_cell,split_exact


def worker(args):
    import torch

    rows=generate_dataset(
        args.samples,
        args.seed,
    )
    train_ids,eval_ids=split_exact(
        rows,
        args.seed,
    )

    device=torch.device(args.device)

    print(
        f"WORKER_START architecture={args.architecture} "
        f"pid={os.getpid()}",
        flush=True,
    )

    result=run_shared_cell(
        args.architecture,
        args,
        rows,
        train_ids,
        eval_ids,
        device,
        args.seed+list(ARCHITECTURES).index(args.architecture),
        args.preflight_steps,
        log=args.worker_log,
    )

    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )

    ev=result["evaluation"]

    print(
        f"WORKER_DONE architecture={args.architecture} "
        f"ratio={result['train']['loss_ratio']:.3f} "
        f"terminal={ev['terminal_accuracy']:.3f}",
        flush=True,
    )


def coordinator(args):
    result_dir=Path(
        "results/v258_preflight_workers"
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending=list(ARCHITECTURES)
    active={}
    results={}

    parallelism=min(
        args.parallelism,
        len(ARCHITECTURES),
    )

    print(
        "=== V258 PARALLEL SHARED-ENGINE PREFLIGHT ===",
        flush=True,
    )
    print("device:",args.device,flush=True)
    print("samples:",args.samples,flush=True)
    print("preflight_steps:",args.preflight_steps,flush=True)
    print("parallelism:",parallelism,flush=True)

    while pending or active:
        while pending and len(active)<parallelism:
            name=pending.pop(0)
            output=result_dir/f"{name}.json"

            cmd=[
                sys.executable,
                str(HERE/"architecture_preflight.py"),
                "--worker",
                "--architecture",name,
                "--samples",str(args.samples),
                "--seed",str(args.seed),
                "--device",args.device,
                "--preflight-steps",str(args.preflight_steps),
                "--lr",str(args.lr),
                "--terminal-weight",str(args.terminal_weight),
                "--output",str(output),
            ]

            if args.worker_log:
                cmd.append("--worker-log")

            proc=subprocess.Popen(
                cmd,
                cwd=HERE,
            )
            active[proc]=(name,output)

            print(
                f"LAUNCH {name} active={len(active)}/{parallelism}",
                flush=True,
            )

        finished=[]

        for proc,(name,output) in list(active.items()):
            code=proc.poll()
            if code is None:
                continue

            finished.append(proc)

            if code!=0:
                raise RuntimeError(
                    f"Worker failed: architecture={name} exit_code={code}"
                )

            if not output.exists():
                raise RuntimeError(
                    f"Worker {name} completed without result."
                )

            results[name]=json.loads(
                output.read_text(encoding="utf-8")
            )

        for proc in finished:
            active.pop(proc)

        if not finished:
            time.sleep(0.10)

    stateful=[
        r for n,r in results.items()
        if n!="baseline_graph"
    ]

    learners=[
        r for r in stateful
        if (
            r["train"]["loss_ratio"]<0.80
            and r["evaluation"]["terminal_accuracy"]>=0.50
        )
    ]

    print(
        "\n"+"="*78,
        flush=True,
    )
    print("V258 PREFLIGHT SUMMARY",flush=True)
    print("="*78,flush=True)

    for name in ARCHITECTURES:
        r=results[name]
        ev=r["evaluation"]
        ab=r["memory_ablation"]

        print(
            f"{name:25s} "
            f"loss={r['train']['initial_loss']:.5f}"
            f"->{r['train']['best_loss']:.5f} "
            f"ratio={r['train']['loss_ratio']:.3f} "
            f"terminal={ev['terminal_accuracy']:.3f} "
            f"memory={ev['task_terminal_accuracy'].get('memory',0):.3f} "
            f"progress={ev['task_terminal_accuracy'].get('progress',0):.3f} "
            f"zero_mem={ab['zero_workspace_terminal_accuracy']}",
            flush=True,
        )

    assert learners,(
        "No stateful architecture learned the randomized task."
    )

    summary=Path("results/v258_preflight.json")
    summary.write_text(
        json.dumps(results,indent=2),
        encoding="utf-8",
    )

    print(
        "SHARED-ENGINE ARCHITECTURE PREFLIGHT: PASS",
        flush=True,
    )
    print(
        "diagnostics_saved:",
        summary,
        flush=True,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--architecture",choices=list(ARCHITECTURES))
    p.add_argument("--output",type=Path,default=Path("worker.json"))
    p.add_argument("--worker-log",action="store_true")
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--seed",type=int,default=255)
    p.add_argument(
        "--device",
        default="cuda"
        if __import__("torch").cuda.is_available()
        else "cpu",
    )
    p.add_argument("--preflight-steps",type=int,default=80)
    p.add_argument("--lr",type=float,default=2e-3)
    p.add_argument("--terminal-weight",type=float,default=2.0)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--parallelism",type=int,default=6)
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    args=p.parse_args()

    if args.worker:
        worker(args)
    else:
        coordinator(args)


if __name__=="__main__":
    main()

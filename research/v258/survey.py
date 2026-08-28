
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_core import (
    ARCHITECTURES,
    run_shared_cell,
    split_exact,
)


HERE=Path(__file__).resolve().parent


def worker_main(args):
    import torch
    from dataset import generate_dataset

    rows=generate_dataset(
        args.samples,
        args.seed,
    )

    train_ids,valid_ids=split_exact(
        rows,
        args.seed,
    )

    train_ids=[
        i for i in train_ids
        if rows[i]["horizon"]==args.horizon
    ]
    valid_ids=[
        i for i in valid_ids
        if rows[i]["horizon"]==args.horizon
    ]

    device=torch.device(args.device)

    print(
        f"WORKER_START architecture={args.architecture} "
        f"horizon={args.horizon} pid={os.getpid()} device={device}",
        flush=True,
    )

    result=run_shared_cell(
        args.architecture,
        args,
        rows,
        train_ids,
        valid_ids,
        device,
        args.seed+list(ARCHITECTURES).index(
            args.architecture
        ),
        args.epochs,
        log=True,
    )

    result["experiment"] = (
        f"v258_{args.architecture}_steps{args.horizon}"
    )
    result["horizon"] = args.horizon

    output=Path(args.worker_output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )

    ev=result["evaluation"]
    ab=result["memory_ablation"]

    print(
        f"WORKER_DONE architecture={args.architecture} "
        f"horizon={args.horizon} "
        f"terminal={ev['terminal_accuracy']:.4f} "
        f"memory={ev['task_terminal_accuracy'].get('memory',0):.4f} "
        f"progress={ev['task_terminal_accuracy'].get('progress',0):.4f} "
        f"zero_mem={ab['zero_workspace_terminal_accuracy']}",
        flush=True,
    )


def coordinator(args):
    from dataset import generate_dataset,save_dataset

    rows=generate_dataset(
        args.samples,
        args.seed,
    )
    args.dataset_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_dataset(
        rows,
        args.dataset_output,
    )

    split_exact(
        rows,
        args.seed,
    )

    tasks=[
        (architecture,horizon)
        for architecture in args.architectures
        for horizon in args.horizons
    ]

    parallelism=max(
        1,
        min(
            args.parallelism,
            len(tasks),
        ),
    )

    worker_dir=Path(
        "results/v258_workers"
    )
    worker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending=list(tasks)
    active={}
    results={}

    print(
        "=== V258 MEMORY-SAFE PARALLEL ARCHITECTURAL SURVEY ===",
        flush=True,
    )
    print("device:",args.device,flush=True)
    print("samples:",args.samples,flush=True)
    print("epochs:",args.epochs,flush=True)
    print("parallelism:",parallelism,flush=True)
    print("matrix_cells:",len(tasks),flush=True)

    while pending or active:
        while pending and len(active)<parallelism:
            architecture,horizon=pending.pop(0)

            output=worker_dir/f"{architecture}_steps{horizon}.json"

            cmd=[
                sys.executable,
                str(HERE/"survey.py"),
                "--worker",
                "--architecture",architecture,
                "--horizon",str(horizon),
                "--samples",str(args.samples),
                "--seed",str(args.seed),
                "--device",args.device,
                "--epochs",str(args.epochs),
                "--lr",str(args.lr),
                "--terminal-weight",str(args.terminal_weight),
                "--batch-size",str(args.batch_size),
                "--log-every",str(args.log_every),
                "--hidden-size",str(args.hidden_size),
                "--heads",str(args.heads),
                "--depth",str(args.depth),
                "--topk",str(args.topk),
                "--worker-output",str(output),
            ]

            proc=subprocess.Popen(
                cmd,
                cwd=HERE,
            )
            active[proc]=(architecture,horizon,output)

            print(
                f"LAUNCH {architecture} steps={horizon} "
                f"active={len(active)}/{parallelism}",
                flush=True,
            )

        finished=[]

        for proc,(architecture,horizon,output) in list(
            active.items()
        ):
            code=proc.poll()

            if code is None:
                continue

            finished.append(proc)

            if code!=0:
                for other in active:
                    if other is not proc and other.poll() is None:
                        other.terminate()
                raise RuntimeError(
                    f"Worker failed: "
                    f"architecture={architecture} "
                    f"horizon={horizon} "
                    f"exit_code={code}"
                )

            if not output.exists():
                raise RuntimeError(
                    f"Worker completed without result: "
                    f"{architecture} steps={horizon}"
                )

            results[(architecture,horizon)]=json.loads(
                output.read_text(
                    encoding="utf-8"
                )
            )

        for proc in finished:
            active.pop(proc)

        if not finished:
            time.sleep(0.10)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered=[
        results[key]
        for key in tasks
    ]

    for result in ordered:
        (args.output_dir/f"{result['experiment']}.json").write_text(
            json.dumps(result,indent=2),
            encoding="utf-8",
        )

    summary=args.output_dir/"v258_summary.json"
    summary.write_text(
        json.dumps(ordered,indent=2),
        encoding="utf-8",
    )

    print(
        "\n"+"="*78,
        flush=True,
    )
    print("V258 SUMMARY",flush=True)
    print("="*78,flush=True)

    for result in ordered:
        ev=result["evaluation"]
        ab=result["memory_ablation"]

        print(
            f"{result['architecture']:25s} "
            f"steps={result['horizon']} "
            f"terminal={ev['terminal_accuracy']:.4f} "
            f"memory_terminal={ev['task_terminal_accuracy'].get('memory',0):.4f} "
            f"progress_terminal={ev['task_terminal_accuracy'].get('progress',0):.4f} "
            f"seq={ev['sequence_exact']:.4f} "
            f"zero_mem={ab['zero_workspace_terminal_accuracy']}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


def main():
    p=argparse.ArgumentParser()

    p.add_argument("--worker",action="store_true")
    p.add_argument("--architecture",choices=list(ARCHITECTURES))
    p.add_argument("--horizon",type=int)
    p.add_argument(
        "--worker-output",
        type=Path,
        default=Path("worker.json"),
    )

    p.add_argument("--samples",type=int,default=300)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--seed",type=int,default=257)

    p.add_argument(
        "--device",
        default=(
            "cuda"
            if __import__("torch").cuda.is_available()
            else "cpu"
        ),
    )

    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--terminal-weight",type=float,default=2.0)
    p.add_argument("--batch-size",type=int,default=8, help="Cases per optimizer update; default 8.")
    p.add_argument("--log-every",type=int,default=1)

    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)

    p.add_argument(
        "--parallelism",
        type=int,
        default=2,
        help="Concurrent survey cells; default is 2.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v258"),
    )

    p.add_argument(
        "--dataset-output",
        type=Path,
        default=Path(
            "results/v258_random_trace_dataset.jsonl"
        ),
    )

    p.add_argument(
        "--architectures",
        nargs="+",
        choices=list(ARCHITECTURES),
        default=list(ARCHITECTURES),
    )

    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[2,4],
    )

    args=p.parse_args()

    if args.worker:
        if (
            args.architecture is None
            or args.horizon not in (2,4)
        ):
            raise SystemExit(
                "--worker requires --architecture and "
                "--horizon 2|4"
            )
        worker_main(args)
        return

    if set(args.horizons)!={2,4}:
        raise ValueError(
            "V258 supports horizons 2 and 4."
        )

    coordinator(args)


if __name__=="__main__":
    main()

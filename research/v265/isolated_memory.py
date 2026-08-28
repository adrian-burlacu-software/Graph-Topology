
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dataset import make_dataset,save_dataset
from benchmark import (
    ARCHITECTURES,
    prepare,
    split_pairs,
    make_model,
    train,
    paired_metrics,
)


HERE=Path(__file__).resolve().parent


def worker(args):
    import torch

    rows=make_dataset(
        args.pairs_per_horizon,
        args.seed,
    )

    train_ids,valid_ids=split_pairs(
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

    data=prepare(rows)
    device=torch.device(args.device)

    print(
        f"WORKER_START architecture={args.architecture} "
        f"horizon={args.horizon} pid={os.getpid()} "
        f"train_pairs={len(train_ids)//2} "
        f"valid_pairs={len(valid_ids)//2}",
        flush=True,
    )

    model=make_model(
        args.architecture,
        args,
        device,
        args.seed,
    )

    tr=train(
        model,
        [data[i] for i in train_ids],
        device,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    metrics=paired_metrics(
        model,
        data,
        valid_ids,
        device,
    )

    result={
        "architecture":args.architecture,
        "horizon":args.horizon,
        "train":tr,
        "metrics":metrics,
    }

    output=Path(args.worker_output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )

    m=metrics

    print(
        f"WORKER_DONE architecture={args.architecture} "
        f"horizon={args.horizon} "
        f"pair={m['both_correct_rate']:.3f} "
        f"disc={m['discrimination_rate']:.3f} "
        f"normal={m['normal_accuracy']:.3f} "
        f"zero_mem={m['zero_workspace_accuracy']:.3f} "
        f"drop={m['ablation_drop']:.3f}",
        flush=True,
    )

    for step,diag in m["step_diagnostics"].items():
        print(
            f"TRACE h={args.horizon} t={step} "
            f"work_delta={diag['mean_working_delta']:.4e} "
            f"logit_delta={diag['mean_action_logit_delta']:.4e} "
            f"retain={diag.get('retention_vs_t1','n/a')}",
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

    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=264)
    p.add_argument(
        "--device",
        default="cuda"
        if __import__("torch").cuda.is_available()
        else "cpu",
    )
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    p.add_argument("--parallelism",type=int,default=2)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v265"),
    )
    p.add_argument(
        "--dataset-output",
        type=Path,
        default=Path(
            "results/v265_memory_diagnostic_dataset.jsonl"
        ),
    )

    args=p.parse_args()

    if args.worker:
        if (
            args.architecture is None
            or args.horizon not in (1,2,3,4)
        ):
            raise SystemExit(
                "--worker requires --architecture and "
                "--horizon 1|2|3|4"
            )
        worker(args)
        return

    rows=make_dataset(
        args.pairs_per_horizon,
        args.seed,
    )
    save_dataset(
        rows,
        args.dataset_output,
    )

    split_pairs(
        rows,
        args.seed,
    )

    tasks=[
        (arch,h)
        for arch in ARCHITECTURES
        for h in (1,2,3,4)
    ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    worker_dir=args.output_dir/"workers"
    worker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending=list(tasks)
    active={}
    results={}

    max_parallel=max(
        1,
        min(
            args.parallelism,
            len(tasks),
        ),
    )

    print(
        "=== V265 ISOLATED MEMORY PERSISTENCE DIAGNOSTIC ===",
        flush=True,
    )
    print("device:",args.device,flush=True)
    print(
        "pairs_per_horizon:",
        args.pairs_per_horizon,
        flush=True,
    )
    print("epochs:",args.epochs,flush=True)
    print("batch_size:",args.batch_size,flush=True)
    print("horizons: 1 2 3 4",flush=True)
    print("parallelism:",max_parallel,flush=True)

    while pending or active:
        while (
            pending
            and len(active)<max_parallel
        ):
            arch,h=pending.pop(0)
            output=worker_dir/f"{arch}_h{h}.json"

            proc=subprocess.Popen(
                [
                    sys.executable,
                    str(HERE/"isolated_memory.py"),
                    "--worker",
                    "--architecture",arch,
                    "--horizon",str(h),
                    "--pairs-per-horizon",
                    str(args.pairs_per_horizon),
                    "--seed",str(args.seed),
                    "--device",args.device,
                    "--epochs",str(args.epochs),
                    "--batch-size",str(args.batch_size),
                    "--lr",str(args.lr),
                    "--hidden-size",str(args.hidden_size),
                    "--heads",str(args.heads),
                    "--depth",str(args.depth),
                    "--topk",str(args.topk),
                    "--worker-output",str(output),
                ],
                cwd=HERE,
            )

            active[proc]=(arch,h,output)

            print(
                f"LAUNCH {arch} horizon={h} "
                f"active={len(active)}/{max_parallel}",
                flush=True,
            )

        finished=[]

        for proc,(arch,h,output) in list(active.items()):
            code=proc.poll()

            if code is None:
                continue

            finished.append(proc)

            if code!=0:
                for other in active:
                    if other is not proc and other.poll() is None:
                        other.terminate()

                raise RuntimeError(
                    f"worker failed {arch} h={h} code={code}"
                )

            if not output.exists():
                raise RuntimeError(
                    f"worker returned no result {arch} h={h}"
                )

            results[(arch,h)]=json.loads(
                output.read_text(encoding="utf-8")
            )

        for proc in finished:
            active.pop(proc)

        if not finished:
            time.sleep(0.10)

    ordered=[
        results[k]
        for k in tasks
    ]

    summary=args.output_dir/"v265_summary.json"
    summary.write_text(
        json.dumps(
            ordered,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"+"="*84,
        flush=True,
    )
    print(
        "V265 ISOLATED MEMORY SUMMARY",
        flush=True,
    )
    print(
        "="*84,
        flush=True,
    )

    for result in ordered:
        m=result["metrics"]

        trace=m["step_diagnostics"]

        retention=(
            trace.get(str(result["horizon"]-1),{})
            .get("retention_vs_t1")
            if result["horizon"]>1
            else None
        )

        print(
            f"{result['architecture']:20s} "
            f"h={result['horizon']} "
            f"pair={m['both_correct_rate']:.3f} "
            f"disc={m['discrimination_rate']:.3f} "
            f"normal={m['normal_accuracy']:.3f} "
            f"zero_mem={m['zero_workspace_accuracy']:.3f} "
            f"drop={m['ablation_drop']:.3f} "
            f"retain_end={retention}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


if __name__=="__main__":
    main()

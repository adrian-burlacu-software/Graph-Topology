
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dataset import make_dataset,save_dataset
from benchmark_core import ARCHITECTURES,prepare,split_pairs

# Top-level worker implementation is below; import-safe on Windows.
HERE=Path(__file__).resolve().parent


def worker_main(args):
    import torch
    from dataset import make_dataset
    from benchmark_core import (
        split_pairs,
        make_model,
        train,
        evaluate,
        paired_counterfactual_score,
        memory_ablation,
    )

    rows=make_dataset(
        args.pairs_per_probe,
        args.seed,
    )
    train_ids,valid_ids=split_pairs(
        rows,
        args.seed,
    )

    # This cell trains on ALL probes for its horizon, because the question is
    # which architecture generalizes over the causal task family.
    train_ids=[
        i for i in train_ids
        if rows[i]["horizon"]==args.horizon
    ]
    valid_ids=[
        i for i in valid_ids
        if rows[i]["horizon"]==args.horizon
    ]

    data=prepare(rows)
    model=make_model(
        args.architecture,
        args,
        torch.device(args.device),
        args.seed,
    )

    tr=train(
        model,
        [data[i] for i in train_ids],
        torch.device(args.device),
        args.epochs,
        args.batch_size,
        args.lr,
        args.terminal_weight,
    )

    ev=evaluate(
        model,
        data,
        valid_ids,
        torch.device(args.device),
    )

    cf=paired_counterfactual_score(
        model,
        data,
        valid_ids,
        torch.device(args.device),
    )

    abl=memory_ablation(
        model,
        data,
        valid_ids,
        torch.device(args.device),
    )

    result={
        "architecture":args.architecture,
        "horizon":args.horizon,
        "train":tr,
        "evaluation":ev,
        "counterfactual":cf,
        "memory_ablation":abl,
    }

    output=Path(args.worker_output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )

    print(
        f"WORKER_DONE architecture={args.architecture} "
        f"horizon={args.horizon} "
        f"memory_cf={cf.get('P1_memory',{}).get('both_correct_rate',0):.3f} "
        f"progress_cf={cf.get('P4_progress',{}).get('both_correct_rate',0):.3f} "
        f"loss_ratio={tr['loss_ratio']:.3f}",
        flush=True,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--architecture",choices=list(ARCHITECTURES))
    p.add_argument("--horizon",type=int)
    p.add_argument("--worker-output",type=Path,default=Path("worker.json"))

    p.add_argument("--pairs-per-probe",type=int,default=24)
    p.add_argument("--seed",type=int,default=259)
    p.add_argument("--device",default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--terminal-weight",type=float,default=4.0)
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    p.add_argument("--parallelism",type=int,default=2)
    p.add_argument("--output-dir",type=Path,default=Path("results/v261"))
    p.add_argument("--dataset-output",type=Path,default=Path("results/v261_counterfactual_dataset.jsonl"))
    args=p.parse_args()

    if args.worker:
        worker_main(args)
        return

    rows=make_dataset(
        args.pairs_per_probe,
        args.seed,
    )
    save_dataset(
        rows,
        args.dataset_output,
    )

    train_ids,valid_ids=split_pairs(
        rows,
        args.seed,
    )

    tasks=[
        (a,h)
        for a in ARCHITECTURES
        for h in (2,4)
    ]

    output_dir=args.output_dir
    output_dir.mkdir(parents=True,exist_ok=True)

    worker_dir=output_dir/"workers"
    worker_dir.mkdir(parents=True,exist_ok=True)

    pending=list(tasks)
    active={}
    results={}

    pmax=max(1,min(args.parallelism,len(tasks)))

    print(
        "=== V261 COMPACT CAUSAL ARCHITECTURE BENCHMARK ===",
        flush=True,
    )
    print("device:",args.device,flush=True)
    print("pairs_per_probe:",args.pairs_per_probe,flush=True)
    print("epochs:",args.epochs,flush=True)
    print("batch_size:",args.batch_size,flush=True)
    print("parallelism:",pmax,flush=True)
    print("matrix_cells:",len(tasks),flush=True)

    while pending or active:
        while pending and len(active)<pmax:
            arch,h=pending.pop(0)
            output=worker_dir/f"{arch}_h{h}.json"

            cmd=[
                sys.executable,
                str(HERE/"survey.py"),
                "--worker",
                "--architecture",arch,
                "--horizon",str(h),
                "--pairs-per-probe",str(args.pairs_per_probe),
                "--seed",str(args.seed),
                "--device",args.device,
                "--epochs",str(args.epochs),
                "--batch-size",str(args.batch_size),
                "--lr",str(args.lr),
                "--terminal-weight",str(args.terminal_weight),
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
            active[proc]=(arch,h,output)

            print(
                f"LAUNCH {arch} horizon={h} "
                f"active={len(active)}/{pmax}",
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
                    f"Worker failed architecture={arch} horizon={h} code={code}"
                )

            if not output.exists():
                raise RuntimeError(
                    f"Worker returned no result: {arch} h={h}"
                )

            results[(arch,h)]=json.loads(
                output.read_text(encoding="utf-8")
            )

        for proc in finished:
            active.pop(proc)

        if not finished:
            time.sleep(0.10)

    ordered=[
        results[key]
        for key in tasks
    ]

    for result in ordered:
        name=f"{result['architecture']}_h{result['horizon']}.json"
        (output_dir/name).write_text(
            json.dumps(result,indent=2),
            encoding="utf-8",
        )

    summary=output_dir/"v261_summary.json"
    summary.write_text(
        json.dumps(ordered,indent=2),
        encoding="utf-8",
    )

    print(
        "\n"+"="*80,
        flush=True,
    )
    print("V261 SUMMARY",flush=True)
    print("="*80,flush=True)

    for result in ordered:
        ev=result["evaluation"]
        cf=result["counterfactual"]
        abl=result["memory_ablation"]

        print(
            f"{result['architecture']:25s} "
            f"h={result['horizon']} "
            f"memory={ev.get('P1_memory',{}).get('terminal_accuracy',0):.3f} "
            f"progress={ev.get('P4_progress',{}).get('terminal_accuracy',0):.3f} "
            f"memory_pair={cf.get('P1_memory',{}).get('both_correct_rate',0):.3f} "
            f"progress_pair={cf.get('P4_progress',{}).get('both_correct_rate',0):.3f} "
            f"zero_mem={abl['zero_workspace_terminal_accuracy']}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


if __name__=="__main__":
    main()

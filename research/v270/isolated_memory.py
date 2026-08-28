
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

from dataset import make_dataset,save_dataset
from benchmark import (
    ARCHITECTURES,
    prepare,split_pairs,make_model,train,
    intervention_metrics,
    swap_state_decoder_probe,
    read_path_sensitivity,
)

HERE=Path(__file__).resolve().parent


def worker(args):
    import torch

    rows=make_dataset(
        args.pairs_per_horizon,
        args.seed,
    )
    train_ids,valid_ids=split_pairs(
        rows,args.seed
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
        f"horizon={args.horizon} "
        f"aux={args.aux_memory_weight} "
        f"train_pairs={len(train_ids)//2} "
        f"valid_pairs={len(valid_ids)//2}",
        flush=True,
    )

    model=make_model(
        args.architecture,args,device,args.seed
    )

    tr=train(
        model,
        [data[i] for i in train_ids],
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.aux_memory_weight,
    )

    metrics=intervention_metrics(
        model,data,valid_ids,device
    )
    swap=swap_state_decoder_probe(
        model,data,valid_ids,device
    )
    read=read_path_sensitivity(
        model,data,valid_ids,device
    )

    result={
        "architecture":args.architecture,
        "horizon":args.horizon,
        "aux_memory_weight":args.aux_memory_weight,
        "train":tr,
        "metrics":metrics,
        "swap_probe":swap,
        "read_path":read,
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
        f"normal={metrics['normal_accuracy']:.3f} "
        f"zero={metrics['zero_workspace_accuracy']:.3f} "
        f"drop={metrics['normal_vs_zero_drop']:.3f} "
        f"disc={metrics['pair_discrimination_rate']:.3f} "
        f"swap={swap['workspace_swap_changes_decision_rate']:.3f} "
        f"read0={read['mean_normal_vs_zero_logit_delta']:.3e} "
        f"loss={tr['loss_ratio']:.3f}",
        flush=True,
    )

    for t,d in metrics["step_diagnostics"].items():
        print(
            f"TRACE h={args.horizon} t={t} "
            f"work={d['mean_work_delta']:.4e} "
            f"logit={d['mean_action_logit_delta']:.4e} "
            f"work_ret={d.get('work_retention_vs_t1','n/a')} "
            f"logit_ret={d.get('logit_retention_vs_t1','n/a')}",
            flush=True,
        )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--architecture",choices=list(ARCHITECTURES))
    p.add_argument("--horizon",type=int)
    p.add_argument("--worker-output",type=Path,default=Path("worker.json"))
    p.add_argument("--pairs-per-horizon",type=int,default=24)
    p.add_argument("--seed",type=int,default=268)
    p.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available()
        else "cpu",
    )
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--terminal-weight",type=float,default=1.0)
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    p.add_argument("--parallelism",type=int,default=2)
    p.add_argument(
        "--aux-memory-weight",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v270"),
    )
    p.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("results/v270_memory_read_dataset.jsonl"),
    )
    args=p.parse_args()

    if args.worker:
        if args.architecture is None or args.horizon not in (1,2,3,4):
            raise SystemExit(
                "--worker requires --architecture and --horizon 1|2|3|4"
            )
        worker(args)
        return

    rows=make_dataset(
        args.pairs_per_horizon,args.seed
    )
    save_dataset(rows,args.dataset_output)
    split_pairs(rows,args.seed)

    # All read-path candidates at all horizons.
    tasks=[
        (architecture,horizon,0.0)
        for architecture in ARCHITECTURES
        for horizon in (1,2,3,4)
    ]

    # Auxiliary diagnostic only for the read architectures at longer horizons.
    for architecture in (
        "direct_read",
        "normalized_read",
        "gated_read",
        "protected_read",
    ):
        for horizon in (3,4):
            tasks.append(
                (
                    architecture,
                    horizon,
                    args.aux_memory_weight,
                )
            )

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

    pmax=max(
        1,
        min(args.parallelism,len(tasks))
    )

    print(
        "=== V270 MEMORY READ-PATH BENCHMARK ===",
        flush=True,
    )
    print(
        "architectures:",
        " ".join(ARCHITECTURES),
        flush=True,
    )
    print(
        "horizons: 1 2 3 4",
        flush=True,
    )
    print(
        "parallelism:",
        pmax,
        flush=True,
    )
    print(
        "total_cells:",
        len(tasks),
        flush=True,
    )

    while pending or active:
        while pending and len(active)<pmax:
            arch,h,aux=pending.pop(0)

            tag=(
                f"{arch}_h{h}_aux"
                f"{str(aux).replace('.','p')}"
            )
            output=worker_dir/f"{tag}.json"

            proc=subprocess.Popen(
                [
                    sys.executable,
                    str(HERE/"isolated_memory.py"),
                    "--worker",
                    "--architecture",arch,
                    "--horizon",str(h),
                    "--pairs-per-horizon",str(args.pairs_per_horizon),
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
                    "--aux-memory-weight",str(aux),
                    "--worker-output",str(output),
                ],
                cwd=HERE,
            )
            active[proc]=(arch,h,aux,output)

            print(
                f"LAUNCH {arch} h={h} aux={aux} "
                f"active={len(active)}/{pmax}",
                flush=True,
            )

        finished=[]

        for proc,(arch,h,aux,output) in list(active.items()):
            code=proc.poll()
            if code is None:
                continue

            finished.append(proc)

            if code!=0:
                for other in active:
                    if other is not proc and other.poll() is None:
                        other.terminate()
                raise RuntimeError(
                    f"worker failed {arch} h={h} aux={aux} code={code}"
                )

            if not output.exists():
                raise RuntimeError(
                    f"worker missing result {arch} h={h} aux={aux}"
                )

            results[(arch,h,aux)]=json.loads(
                output.read_text(encoding="utf-8")
            )

        for proc in finished:
            active.pop(proc)

        if not finished:
            time.sleep(0.10)

    ordered=[results[key] for key in tasks]

    summary=args.output_dir/"v270_summary.json"
    summary.write_text(
        json.dumps(ordered,indent=2),
        encoding="utf-8",
    )

    print(
        "\n"+"="*104,
        flush=True,
    )
    print(
        "V270 MEMORY READ-PATH SUMMARY",
        flush=True,
    )
    print(
        "="*104,
        flush=True,
    )

    for r in ordered:
        m=r["metrics"]
        sw=r["swap_probe"]
        rd=r["read_path"]

        print(
            f"{r['architecture']:20s} "
            f"h={r['horizon']} "
            f"aux={r['aux_memory_weight']} "
            f"normal={m['normal_accuracy']:.3f} "
            f"zero={m['zero_workspace_accuracy']:.3f} "
            f"drop={m['normal_vs_zero_drop']:.3f} "
            f"disc={m['pair_discrimination_rate']:.3f} "
            f"swap={sw['workspace_swap_changes_decision_rate']:.3f} "
            f"read0={rd['mean_normal_vs_zero_logit_delta']:.3e} "
            f"loss={r['train']['loss_ratio']:.3f}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


if __name__=="__main__":
    main()

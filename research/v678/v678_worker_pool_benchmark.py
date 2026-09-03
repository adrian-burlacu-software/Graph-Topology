from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

from v678_memory import SharedCheckpoint
from v678_offline_learning import GENERAL_LANES, worker_main
from v678_worker_summary import worker_summary


def worker_entry(args, worker_id, stop_event):
    worker_main(args, worker_id, stop_event)


def parse_worker_counts(value):
    counts = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not counts or any(count < 10 or count > 15 for count in counts):
        raise argparse.ArgumentTypeError(
            "worker counts must be comma-separated values from 10 through 15"
        )
    return counts


def run_configuration(args, worker_count, root):
    run_dir = root / f"workers_{worker_count:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shared_memory = run_dir / "shared.sqlite"
    log_dir = run_dir / "workers"
    total_workers = worker_count + 1
    bootstrap = SharedCheckpoint(
        shared_memory,
        worker_count,
        total_workers,
        args.checkpoint_seconds,
    )
    bootstrap.close()
    child_args = argparse.Namespace(
        database=args.database,
        shared_memory=str(shared_memory),
        worker_log_dir=str(log_dir),
        total_workers=total_workers,
        checkpoint_seconds=args.checkpoint_seconds,
        seed=args.seed,
        batch_sleep=0.0,
        duration_seconds=args.duration_seconds,
        composition_fanout=args.composition_fanout,
        composition_max=args.composition_max,
        worker_query_batch_subjects=args.worker_query_batch_subjects,
        task_poll_seconds=args.task_poll_seconds,
    )
    stop_event = mp.Event()
    started = time.perf_counter()
    processes = [
        mp.Process(
            target=worker_entry,
            args=(child_args, worker_id, stop_event),
            name=f"v678-benchmark-{worker_id:02d}",
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=args.duration_seconds + 15)
    stop_event.set()
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
    wall_seconds = time.perf_counter() - started
    workers, totals = worker_summary(log_dir)
    cpu_seconds = float(totals.get("cpu_seconds", 0.0))
    learned = int(totals.get("learned", 0))
    available_cpus = os.cpu_count() or 1
    return {
        "record_type": "worker_pool_benchmark",
        "worker_count": worker_count,
        "lane_count": len(GENERAL_LANES),
        "lanes": list(GENERAL_LANES),
        "available_cpus": available_cpus,
        "duration_seconds": args.duration_seconds,
        "wall_seconds": wall_seconds,
        "aggregate_cpu_seconds": cpu_seconds,
        "worker_utilization": cpu_seconds / max(wall_seconds * worker_count, 1e-9),
        "host_cpu_utilization": cpu_seconds / max(wall_seconds * available_cpus, 1e-9),
        "learned_per_second": learned / max(wall_seconds, 1e-9),
        "totals": totals,
        "workers": workers,
        "exit_codes": {process.name: process.exitcode for process in processes},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark V678 general-lane worker-pool sizes on this machine."
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", default="./results/v678/worker_pool_benchmark.jsonl")
    parser.add_argument("--worker-counts", type=parse_worker_counts, default=(10, 12, 15))
    parser.add_argument("--duration-seconds", type=int, default=30)
    parser.add_argument("--checkpoint-seconds", type=int, choices=(60, 300), default=60)
    parser.add_argument("--seed", type=int, default=67800)
    parser.add_argument("--composition-fanout", type=int, default=4)
    parser.add_argument("--composition-max", type=int, default=2000)
    parser.add_argument("--worker-query-batch-subjects", type=int, default=128)
    parser.add_argument("--task-poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / f"{output.stem}_runs"
    records = [
        run_configuration(args, worker_count, scratch)
        for worker_count in args.worker_counts
    ]
    best = max(
        records,
        key=lambda record: (
            record["learned_per_second"],
            record["host_cpu_utilization"],
        ),
    )
    records.append(
        {
            "record_type": "worker_pool_recommendation",
            "recommended_worker_count": best["worker_count"],
            "selection_metric": "learned_per_second, then host_cpu_utilization",
            "available_cpus": os.cpu_count() or 1,
        }
    )
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"Recommended --worker-count {best['worker_count']} "
        f"({best['learned_per_second']:.1f} learned/s, "
        f"{best['host_cpu_utilization']:.1%} host CPU utilization)."
    )
    print(f"Wrote {len(records)} benchmark records to {output.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

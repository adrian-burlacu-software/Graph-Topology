from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

from v671_offline_learning import worker_main
from v671_memory import SharedCheckpoint


def build_parser():
    ap = argparse.ArgumentParser(description="V671 19 offline workers + 1 online semantic chat worker")
    ap.add_argument("--database", required=True)
    ap.add_argument("--output", default="./results/v671_chat.json")
    ap.add_argument("--trace-output", default="./results/v671_chat_traces.jsonl")
    ap.add_argument("--memory-output", default="./results/v671_memory.json")
    ap.add_argument("--worker-log-dir", default="./results/v671_workers")
    ap.add_argument("--shared-memory", default="./results/v671_shared_memory.sqlite")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--mode", choices=("chat", "smoke"), default="chat")
    ap.add_argument("--max-hypotheses", type=int, default=12)
    ap.add_argument("--goal-budget", type=int, default=40)
    ap.add_argument("--per-node", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache-entries", type=int, default=12000)
    ap.add_argument("--checkpoint-seconds", type=int, choices=(60,300), default=300)
    ap.add_argument("--seed", type=int, default=67100)
    ap.add_argument("--batch-sleep", type=float, default=0.20)
    ap.add_argument("--duration-seconds", type=int, default=0, help="0 means until chat exits")
    return ap


def launch_offline(args, stop_event):
    child_args = argparse.Namespace(**vars(args))
    child_args.total_workers = 20
    processes = []
    for worker_id in range(19):
        p = mp.Process(target=offline_process_entry, args=(child_args, worker_id, stop_event), name=f"v671-offline-{worker_id:02d}")
        p.daemon = True
        p.start()
        processes.append(p)
    return processes


def offline_process_entry(args, worker_id, stop_event):
    try:
        # The offline worker is intentionally cooperative; the parent also terminates
        # daemonic children when chat exits. This loop lets a stop event end cleanly.
        import threading
        thread = threading.Thread(target=lambda: None)
        thread.start(); thread.join()
        worker_main(args, worker_id, stop_event)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[V671 offline worker {worker_id}] FATAL: {exc!r}", flush=True)


def run_chat(args, stop_event):
    from v671_semantic_chat_gateway import run_chat_worker
    chat_args = argparse.Namespace(**vars(args))
    chat_args.worker_id = 19
    chat_args.total_workers = 20
    chat_args.stop_event = stop_event
    return run_chat_worker(chat_args)


def main():
    args = build_parser().parse_args()
    Path(args.worker_log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.shared_memory).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace_output).parent.mkdir(parents=True, exist_ok=True)
    stop_event = mp.Event()

    # Create the shared checkpoint schema once before fan-out. This avoids a
    # startup DDL stampede among 20 processes.
    bootstrap = SharedCheckpoint(args.shared_memory, 19, 20, args.checkpoint_seconds)
    bootstrap.close()

    print("=== V671 COGNITIVE RUNTIME ===", flush=True)
    print("workers : 19 offline learners + 1 online chat worker", flush=True)
    print(f"shared  : {Path(args.shared_memory).resolve()}", flush=True)
    print(f"sync    : modulus-staggered every {args.checkpoint_seconds}s", flush=True)

    offline = launch_offline(args, stop_event)
    print("offline workers started:", ", ".join(f"{i:02d}" for i in range(19)), flush=True)

    try:
        run_chat(args, stop_event)
    finally:
        stop_event.set()
        deadline = time.time() + 8.0
        for p in offline:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=min(1.5, remaining))
        # Hard-kill only a genuinely stuck worker. Normal shutdown should yield
        # exitcode 0 instead of the previous SIGTERM -15 for every worker.
        for p in offline:
            if p.is_alive():
                p.terminate()
        for p in offline:
            p.join(timeout=1.0)
        print("=== V671 RUNTIME COMPLETE ===", flush=True)
        print("offline exit codes:", {p.name: p.exitcode for p in offline}, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()

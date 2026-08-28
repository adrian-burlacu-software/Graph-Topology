
from __future__ import annotations

import argparse
import collections
import json
import math

from dataset import ACTIONS,INSTRUCTION_TOKEN,generate_dataset


def compact(x):
    return json.dumps(
        x,
        sort_keys=True,
        separators=(",",":"),
    )


def entropy(counter):
    total=sum(counter.values())
    if total<=0:
        return 0.0

    return -sum(
        (n/total)*math.log2(n/total)
        for n in counter.values()
        if n
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--seed",type=int,default=255)
    args=p.parse_args()

    rows=generate_dataset(
        args.samples,
        args.seed,
    )

    counts=collections.Counter(
        (r["task_type"],r["horizon"])
        for r in rows
    )

    assert {
        ("memory",2),
        ("memory",4),
        ("progress",2),
        ("progress",4),
    } <= set(counts)

    for row in rows:
        assert row["version"]=="v258"
        h=row["horizon"]
        assert len(row["trajectory_states"])==h
        assert len(row["trajectory_actions"])==h
        assert len(row["trajectory_attention"])==h
        assert len(row["trace_program"])==h

    inverse={
        token:action
        for action,token in INSTRUCTION_TOKEN.items()
    }

    pairs=collections.defaultdict(list)

    for row in rows:
        if row["task_type"]=="memory":
            pairs[row["pair_id"]].append(row)

    assert pairs

    memory_counts=collections.Counter()

    for pair in pairs.values():
        assert len(pair)==2
        a,b=pair

        assert a["goal"]==b["goal"]
        assert compact(a["trajectory_states"][1:])==compact(
            b["trajectory_states"][1:]
        )
        assert a["instruction_token"]!=b["instruction_token"]

        for row in pair:
            assert inverse[row["instruction_token"]]==row["memory_action"]
            assert row["memory_action"]==row["final_action"]["action"]
            memory_counts[row["final_action"]["action"]]+=1

            for state in row["trajectory_states"][1:]:
                assert row["instruction_token"] not in compact(state)

    assert all(
        memory_counts[a]>0
        for a in ACTIONS
    )

    progress=[
        row for row in rows
        if row["task_type"]=="progress"
    ]

    progress_counts=collections.Counter(
        row["final_action"]["action"]
        for row in progress
    )

    assert all(
        progress_counts[a]>0
        for a in ACTIONS
    )

    max_entropy=math.log2(len(ACTIONS))
    progress_entropy=entropy(progress_counts)

    assert progress_entropy>=max_entropy*0.85

    variants={
        h:len({
            tuple(r["trace_program"])
            for r in progress
            if r["horizon"]==h
        })
        for h in (2,4)
    }

    assert all(
        variants[h]>=2
        for h in (2,4)
    )

    print("=== V258 RANDOM TRACE PREFLIGHT ===",flush=True)
    print("dataset_size:",len(rows),flush=True)
    print("task_horizon_counts:",dict(counts),flush=True)
    print("memory_terminal_counts:",dict(memory_counts),flush=True)
    print("progress_terminal_counts:",dict(progress_counts),flush=True)
    print(
        f"progress_terminal_entropy:"
        f" {progress_entropy:.3f}/{max_entropy:.3f}",
        flush=True,
    )
    print(
        "progress_program_variants:",
        variants,
        flush=True,
    )
    print(
        "stateless_memory_final_ceiling: 0.5000",
        flush=True,
    )
    print(
        "TASK / RANDOM TRACE PREFLIGHT: PASS",
        flush=True,
    )


if __name__=="__main__":
    main()

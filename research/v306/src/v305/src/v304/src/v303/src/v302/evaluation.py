
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import TASKS, make_sequence, FrozenCore
from hypothesis import CONFIGS, HypothesisRevision


def evaluate_config(
    name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        seq_scores=[]
        first_scores=[]
        second_scores=[]

        for seed in seeds:
            sequence=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            system=FrozenCore(
                HypothesisRevision(
                    CONFIGS[name]
                )
            )

            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in sequence.episodes
            ]

            half=len(rows)//2

            seq_scores.append(
                sum(
                    int(r["correct"])
                    for r in rows
                )/len(rows)
            )
            first_scores.append(
                sum(
                    int(r["correct"])
                    for r in rows[:half]
                )/half
            )
            second_scores.append(
                sum(
                    int(r["correct"])
                    for r in rows[half:]
                )/(len(rows)-half)
            )

        task_rows.append(
            {
                "task":task,
                "accuracy":sum(seq_scores)/len(seq_scores),
                "first":sum(first_scores)/len(first_scores),
                "second":sum(second_scores)/len(second_scores),
            }
        )

    return task_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=9)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v302_hypothesis_revision.json"
        ),
    )
    args=p.parse_args()

    seeds=list(
        range(
            302,
            302+args.seeds,
        )
    )

    started=time.perf_counter()

    results=[]

    for name in CONFIGS:
        tasks=evaluate_config(
            name,
            seeds,
            args.episodes,
            args.horizon,
        )

        overall=sum(
            row["accuracy"]
            for row in tasks
        )/len(tasks)

        first=sum(
            row["first"]
            for row in tasks
        )/len(tasks)

        second=sum(
            row["second"]
            for row in tasks
        )/len(tasks)

        results.append(
            {
                "config":name,
                "eval_accuracy":overall,
                "first_half":first,
                "second_half":second,
                "learning_gain":second-first,
                "tasks":tasks,
            }
        )

    elapsed=time.perf_counter()-started

    results.sort(
        key=lambda r:(
            r["eval_accuracy"],
            r["learning_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v302",
        "benchmark":"hypothesis_revision",
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "config_count":len(CONFIGS),
        "wall_time_seconds":elapsed,
        "results":results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== V302 HYPOTHESIS + REVISION ==="
    )
    print(
        f"configs={len(CONFIGS)} "
        f"tasks={len(TASKS)} "
        f"seeds={len(seeds)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon} "
        f"wall={elapsed:.3f}s"
    )

    for i,row in enumerate(
        results,
        1,
    ):
        print(
            f"{i}. {row['config']:24s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"first={row['first_half']:.3f} "
            f"second={row['second_half']:.3f} "
            f"gain={row['learning_gain']:+.3f}"
        )

        print(
            "   "
            + " ".join(
                f"{x['task']}={x['accuracy']:.2f}"
                for x in row["tasks"]
            )
        )


if __name__=="__main__":
    main()

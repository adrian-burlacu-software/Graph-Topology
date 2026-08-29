
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import TASKS, make_sequence
from alternate_settling import CONFIGS
from integrated import IntegratedSystem


def evaluate(
    name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        acc=[]
        first=[]
        second=[]

        for seed in seeds:
            seq=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            system=IntegratedSystem(name)

            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in seq.episodes
            ]

            half=len(rows)//2

            acc.append(
                sum(
                    int(r["correct"])
                    for r in rows
                )/len(rows)
            )
            first.append(
                sum(
                    int(r["correct"])
                    for r in rows[:half]
                )/half
            )
            second.append(
                sum(
                    int(r["correct"])
                    for r in rows[half:]
                )/len(rows[half:])
            )

        task_rows.append({
            "task":task,
            "accuracy":sum(acc)/len(acc),
            "first":sum(first)/len(first),
            "second":sum(second)/len(second),
        })

    overall=sum(x["accuracy"] for x in task_rows)/len(task_rows)
    fh=sum(x["first"] for x in task_rows)/len(task_rows)
    sh=sum(x["second"] for x in task_rows)/len(task_rows)

    return {
        "eval_accuracy":overall,
        "first_half":fh,
        "second_half":sh,
        "learning_gain":sh-fh,
        "tasks":task_rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=9)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        /"results"
        /"v308_dual_state.json",
    )
    args=p.parse_args()

    seeds=list(
        range(
            308,
            308+args.seeds,
        )
    )

    started=time.perf_counter()
    results=[]

    for name in CONFIGS:
        results.append({
            "configuration":name,
            **evaluate(
                name,
                seeds,
                args.episodes,
                args.horizon,
            ),
        })

    elapsed=time.perf_counter()-started

    results.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["learning_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v308",
        "benchmark":"alternate_state_settling",
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "configurations":list(CONFIGS),
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
        "=== V308 ALTERNATE STATE SETTLING ==="
    )
    print(
        f"configs={len(CONFIGS)} "
        f"tasks={len(TASKS)} "
        f"seeds={len(seeds)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon} "
        f"wall={elapsed:.3f}s"
    )

    for row in results:
        t={
            x["task"]:x
            for x in row["tasks"]
        }

        print(
            f"{row['configuration']:22s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"interference={t['interference']['accuracy']:.3f} "
            f"rule_change={t['rule_change']['accuracy']:.3f} "
            f"counterfactual={t['counterfactual']['accuracy']:.3f}"
        )


if __name__=="__main__":
    main()

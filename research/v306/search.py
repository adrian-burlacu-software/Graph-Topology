
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import (
    TASKS,
    make_sequence,
)
from hypotheses import CONFIGS
from integrated import IntegratedSystem


def evaluate(
    name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        accuracies=[]
        firsts=[]
        seconds=[]

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

            accuracies.append(
                sum(int(r["correct"]) for r in rows)
                /len(rows)
            )
            firsts.append(
                sum(int(r["correct"]) for r in rows[:half])
                /half
            )
            seconds.append(
                sum(int(r["correct"]) for r in rows[half:])
                /len(rows[half:])
            )

        task_rows.append({
            "task":task,
            "accuracy":sum(accuracies)/len(accuracies),
            "first":sum(firsts)/len(firsts),
            "second":sum(seconds)/len(seconds),
        })

    overall=sum(x["accuracy"] for x in task_rows)/len(task_rows)
    first=sum(x["first"] for x in task_rows)/len(task_rows)
    second=sum(x["second"] for x in task_rows)/len(task_rows)

    return {
        "eval_accuracy":overall,
        "first_half":first,
        "second_half":second,
        "learning_gain":second-first,
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
        /"v306_beyond_binding.json",
    )
    args=p.parse_args()

    seeds=list(
        range(
            306,
            306+args.seeds,
        )
    )

    started=time.perf_counter()
    results=[]

    for name in CONFIGS:
        results.append({
            "hypothesis":name,
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
        "version":"v306",
        "benchmark":"beyond_binding",
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "hypotheses":list(CONFIGS),
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

    print("=== V306 BEYOND COMPOSITIONAL BINDING ===")
    print(
        f"hypotheses={len(CONFIGS)} "
        f"tasks={len(TASKS)} "
        f"seeds={len(seeds)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon} "
        f"wall={elapsed:.3f}s"
    )

    for row in results:
        t={x["task"]:x for x in row["tasks"]}
        print(
            f"{row['hypothesis']:24s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"interference={t['interference']['accuracy']:.3f} "
            f"rule_change={t['rule_change']['accuracy']:.3f} "
            f"counterfactual={t['counterfactual']['accuracy']:.3f}"
        )


if __name__=="__main__":
    main()


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import (
    TASKS,
    make_sequence,
    RichCognitiveSystem,
)
from credit import CONFIGS, make_credit


def evaluate_config(
    config_name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        accuracies=[]
        firsts=[]
        lasts=[]

        for seed in seeds:
            sequence=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            system=RichCognitiveSystem(
                make_credit(config_name)
            )

            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in sequence.episodes
            ]

            half=len(rows)//2

            accuracies.append(
                sum(
                    int(r["correct"])
                    for r in rows
                )/len(rows)
            )
            firsts.append(
                sum(
                    int(r["correct"])
                    for r in rows[:half]
                )/half
            )
            lasts.append(
                sum(
                    int(r["correct"])
                    for r in rows[half:]
                )/len(rows[half:])
            )

        task_rows.append(
            {
                "task":task,
                "accuracy":sum(accuracies)/len(accuracies),
                "first":sum(firsts)/len(firsts),
                "second":sum(lasts)/len(lasts),
            }
        )

    return task_rows


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v301_richer_cognition.json"
        ),
    )

    args=parser.parse_args()

    seeds=list(
        range(
            301,
            301+args.seeds,
        )
    )

    results=[]
    start=time.perf_counter()

    for name in CONFIGS:
        tasks=evaluate_config(
            name,
            seeds,
            args.episodes,
            args.horizon,
        )

        overall=sum(
            r["accuracy"]
            for r in tasks
        )/len(tasks)

        first=sum(
            r["first"]
            for r in tasks
        )/len(tasks)

        second=sum(
            r["second"]
            for r in tasks
        )/len(tasks)

        results.append(
            {
                "credit":name,
                "eval_accuracy":overall,
                "first_half":first,
                "second_half":second,
                "learning_gain":second-first,
                "tasks":tasks,
            }
        )

    elapsed=time.perf_counter()-start

    results.sort(
        key=lambda r:(
            r["eval_accuracy"],
            r["learning_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v301",
        "benchmark":"richer_cognition",
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
        "=== V301 RICHER COGNITION ==="
    )
    print(
        f"tasks={len(TASKS)} "
        f"configs={len(CONFIGS)} "
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
            f"{i}. {row['credit']:24s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"first={row['first_half']:.3f} "
            f"second={row['second_half']:.3f} "
            f"gain={row['learning_gain']:+.3f}"
        )

        print(
            "   "
            + " ".join(
                f"{t['task']}={t['accuracy']:.2f}"
                for t in row["tasks"]
            )
        )


if __name__=="__main__":
    main()

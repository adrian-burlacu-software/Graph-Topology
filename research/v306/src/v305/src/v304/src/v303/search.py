
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import (
    TASKS,
    make_sequence,
)
from hypothesis import (
    CONFIGS as HYPOTHESIS_CONFIGS,
    HypothesisRevision,
)
from selective import CONFIGS as SELECTIVE_CONFIGS
from integrated import IntegratedSystem
from selective import SelectiveRepresentation


def evaluate(
    selective_name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        scores=[]
        first=[]
        second=[]

        for seed in seeds:
            seq=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            # Use best V302 hypothesis/revision regime as fixed controller.
            hypothesis_config=HYPOTHESIS_CONFIGS["fast_revision"]

            system=IntegratedSystem(
                SelectiveRepresentation(
                    SELECTIVE_CONFIGS[
                        selective_name
                    ]
                ),
                HypothesisRevision(
                    hypothesis_config
                ),
            )

            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in seq.episodes
            ]

            half=len(rows)//2

            scores.append(
                sum(int(r["correct"]) for r in rows)
                /len(rows)
            )

            first.append(
                sum(int(r["correct"]) for r in rows[:half])
                /half
            )

            second.append(
                sum(int(r["correct"]) for r in rows[half:])
                /len(rows[half:])
            )

        task_rows.append({
            "task":task,
            "accuracy":sum(scores)/len(scores),
            "first":sum(first)/len(first),
            "second":sum(second)/len(second),
        })

    overall=sum(
        row["accuracy"]
        for row in task_rows
    )/len(task_rows)

    first_mean=sum(
        row["first"]
        for row in task_rows
    )/len(task_rows)

    second_mean=sum(
        row["second"]
        for row in task_rows
    )/len(task_rows)

    return {
        "eval_accuracy":overall,
        "first_half":first_mean,
        "second_half":second_mean,
        "learning_gain":second_mean-first_mean,
        "tasks":task_rows,
    }


def main():
    p=argparse.ArgumentParser()

    p.add_argument(
        "--seeds",
        type=int,
        default=12,
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=16,
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=9,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            __file__
        ).resolve().parent / "results" / "v303_selective.json",
    )

    args=p.parse_args()

    seeds=list(
        range(
            303,
            303+args.seeds,
        )
    )

    started=time.perf_counter()
    results=[]

    for name in SELECTIVE_CONFIGS:
        metrics=evaluate(
            name,
            seeds,
            args.episodes,
            args.horizon,
        )

        results.append({
            "selective":name,
            **metrics,
        })

    results.sort(
        key=lambda row:(
            row["eval_accuracy"],
            row["learning_gain"],
        ),
        reverse=True,
    )

    elapsed=time.perf_counter()-started

    payload={
        "version":"v303",
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "fixed_architecture":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
            "hypothesis_revision":"fast_revision",
        },
        "selective_configs":list(
            SELECTIVE_CONFIGS
        ),
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
        "=== V303 SELECTIVE REPRESENTATION ==="
    )
    print(
        f"configs={len(SELECTIVE_CONFIGS)} "
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
            f"{i}. {row['selective']:22s} "
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

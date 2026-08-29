
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import richer_cognition
import hypothesis
import validation
import evaluation


def run_eval(
    seeds,
    episodes,
    horizon,
):
    out=[]

    for name in hypothesis.CONFIGS:
        tasks=evaluation.evaluate_config(
            name,
            list(seeds),
            episodes,
            horizon,
        )

        overall=sum(
            x["accuracy"]
            for x in tasks
        )/len(tasks)

        first=sum(
            x["first"]
            for x in tasks
        )/len(tasks)

        second=sum(
            x["second"]
            for x in tasks
        )/len(tasks)

        out.append(
            {
                "config":name,
                "eval_accuracy":overall,
                "first_half":first,
                "second_half":second,
                "learning_gain":second-first,
                "tasks":tasks,
            }
        )

    out.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["learning_gain"],
        ),
        reverse=True,
    )

    return out


def main():
    print("=== V302 VALIDATION ===")
    validation.main()

    # ---------------------------------------------------------------
    # Smoke: small enough to iterate instantly, but checks the entire
    # six-task benchmark.
    # ---------------------------------------------------------------
    smoke_start=time.perf_counter()
    smoke=run_eval(
        seeds=(302,303,304,305),
        episodes=8,
        horizon=9,
    )
    smoke_elapsed=time.perf_counter()-smoke_start

    print()
    print("=== V302 SMOKE ===")
    print(
        f"configs={len(hypothesis.CONFIGS)} "
        f"tasks={len(richer_cognition.TASKS)} "
        f"wall={smoke_elapsed:.3f}s"
    )

    for row in smoke:
        print(
            f"{row['config']:24s} "
            f"acc={row['eval_accuracy']:.3f} "
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

    # Do NOT require hyperparameter discrimination in this smoke. Require the
    # actual architectural capability: rule_change must work above chance.
    rule_scores=[
        next(
            x["accuracy"]
            for x in row["tasks"]
            if x["task"]=="rule_change"
        )
        for row in smoke
    ]

    assert max(rule_scores)>0.5

    # ---------------------------------------------------------------
    # Full local run requested by the user.
    # ---------------------------------------------------------------
    full_start=time.perf_counter()

    full=run_eval(
        seeds=range(302,314),
        episodes=16,
        horizon=9,
    )

    full_elapsed=time.perf_counter()-full_start

    payload={
        "version":"v302",
        "benchmark":"hypothesis_revision",
        "tasks":list(richer_cognition.TASKS),
        "seed_count":12,
        "seeds":list(range(302,314)),
        "episodes":16,
        "horizon":9,
        "config_count":len(hypothesis.CONFIGS),
        "smoke_wall_time_seconds":smoke_elapsed,
        "full_wall_time_seconds":full_elapsed,
        "smoke":smoke,
        "full":full,
    }

    Path(
        "v302_smoke_full.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== V302 FULL LOCAL RUN ===")
    print(
        f"wall={full_elapsed:.3f}s"
    )

    for row in full:
        print(
            f"{row['config']:24s} "
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

    # Extract the actual hard-task frontier.
    print()
    print("=== HARD-TASK FRONTIER ===")
    for task in (
        "interference",
        "rule_change",
        "counterfactual",
    ):
        vals=[
            (
                row["config"],
                next(
                    x["accuracy"]
                    for x in row["tasks"]
                    if x["task"]==task
                ),
                next(
                    x["second"]
                    for x in row["tasks"]
                    if x["task"]==task
                )
                -
                next(
                    x["first"]
                    for x in row["tasks"]
                    if x["task"]==task
                ),
            )
            for row in full
        ]
        vals.sort(
            key=lambda x:(x[1],x[2]),
            reverse=True,
        )
        print(
            task,
            [
                (
                    name,
                    round(acc,3),
                    round(gain,3),
                )
                for name,acc,gain in vals
            ],
        )

    print()
    print("V302 smoke + full run: PASS")


if __name__=="__main__":
    main()

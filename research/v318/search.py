
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from richer_cognition import (
    TASKS,
    make_sequence,
)
from rule_schemas import CONFIGS
from integrated import IntegratedSystem


def evaluate(
    name,
    seeds,
    episodes,
    horizon,
):
    task_rows = []

    for task in TASKS:
        accuracy = []
        first = []
        second = []
        revisions = []
        final_schemas = []

        for seed in seeds:
            seq = make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            system = IntegratedSystem(name)

            rows = [
                system.run(
                    ep,
                    learn=True,
                )
                for ep in seq.episodes
            ]

            half = len(rows) // 2

            accuracy.append(
                sum(
                    int(r["correct"])
                    for r in rows
                ) / len(rows)
            )

            first.append(
                sum(
                    int(r["correct"])
                    for r in rows[:half]
                ) / half
            )

            second.append(
                sum(
                    int(r["correct"])
                    for r in rows[half:]
                ) / len(rows[half:])
            )

            state = system.rules._state(task)

            revisions.append(
                state.revisions
            )

            final_schemas.append(
                state.active.name
                if state.active
                else None
            )

        task_rows.append({
            "task": task,
            "accuracy": sum(accuracy) / len(accuracy),
            "first": sum(first) / len(first),
            "second": sum(second) / len(second),
            "revisions": sum(revisions) / len(revisions),
            "final_schemas": final_schemas,
        })

    overall = sum(
        row["accuracy"]
        for row in task_rows
    ) / len(task_rows)

    first_mean = sum(
        row["first"]
        for row in task_rows
    ) / len(task_rows)

    second_mean = sum(
        row["second"]
        for row in task_rows
    ) / len(task_rows)

    return {
        "eval_accuracy": overall,
        "first_half": first_mean,
        "second_half": second_mean,
        "learning_gain": second_mean - first_mean,
        "tasks": task_rows,
    }


def main():
    parser = argparse.ArgumentParser()

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
        default=Path(__file__).resolve().parent
        / "results"
        / "v318_rule_schemas.json",
    )

    args = parser.parse_args()

    seeds = list(
        range(
            318,
            318 + args.seeds,
        )
    )

    started = time.perf_counter()

    results = []

    for name in CONFIGS:
        results.append({
            "rule_schema": name,
            **evaluate(
                name,
                seeds,
                args.episodes,
                args.horizon,
            ),
        })

    elapsed = time.perf_counter() - started

    results.sort(
        key=lambda row:(
            row["eval_accuracy"],
            row["learning_gain"],
        ),
        reverse=True,
    )

    payload = {
        "version": "v318",
        "benchmark": "explicit_rule_schemas",
        "schema_count": 14,
        "tasks": list(TASKS),
        "seeds": seeds,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "wall_time_seconds": elapsed,
        "results": results,
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

    print("=== V318 EXPLICIT RULE SCHEMAS ===")
    print(
        f"schemas=14 "
        f"configs={len(CONFIGS)} "
        f"tasks={len(TASKS)} "
        f"seeds={len(seeds)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon} "
        f"wall={elapsed:.3f}s"
    )

    for row in results:
        t = {
            x["task"]: x
            for x in row["tasks"]
        }

        print(
            f"{row['rule_schema']:22s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"interference={t['interference']['accuracy']:.3f} "
            f"rule_change={t['rule_change']['accuracy']:.3f} "
            f"counterfactual={t['counterfactual']['accuracy']:.3f} "
            f"revs={t['rule_change']['revisions']:.2f}"
        )


if __name__=="__main__":
    main()

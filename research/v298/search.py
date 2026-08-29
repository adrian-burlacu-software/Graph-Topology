
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import make_sequence, CoreSystem
from combined_credit import (
    CONFIGS,
    CombinedGlobalEligibility,
)


def evaluate_config(
    name,
    train_seeds,
    eval_seeds,
    episodes,
    horizon,
):
    train_scores=[]

    for seed in train_seeds:
        sequence=make_sequence(
            seed,
            episodes,
            horizon,
        )

        system=CoreSystem(
            CombinedGlobalEligibility(
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

        train_scores.append(
            sum(
                int(r["correct"])
                for r in rows
            )/len(rows)
        )

    eval_metrics=[]

    for seed in eval_seeds:
        sequence=make_sequence(
            seed,
            episodes,
            horizon,
        )

        system=CoreSystem(
            CombinedGlobalEligibility(
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

        eval_metrics.append({
            "accuracy":(
                sum(
                    int(r["correct"])
                    for r in rows
                )/len(rows)
            ),
            "first":(
                sum(
                    int(r["correct"])
                    for r in rows[:half]
                )/half
            ),
            "second":(
                sum(
                    int(r["correct"])
                    for r in rows[half:]
                )/(len(rows)-half)
            ),
        })

    return {
        "train":sum(train_scores)/len(train_scores),
        "eval":sum(
            x["accuracy"]
            for x in eval_metrics
        )/len(eval_metrics),
        "first":sum(
            x["first"]
            for x in eval_metrics
        )/len(eval_metrics),
        "second":sum(
            x["second"]
            for x in eval_metrics
        )/len(eval_metrics),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-seeds",type=int,default=12)
    p.add_argument("--eval-seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=7)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v298_combined_credit.json"
        ),
    )
    args=p.parse_args()

    train=list(
        range(
            298,
            298+args.train_seeds,
        )
    )
    evaluation=list(
        range(
            10298,
            10298+args.eval_seeds,
        )
    )

    results=[]

    for name in CONFIGS:
        row=evaluate_config(
            name,
            train,
            evaluation,
            args.episodes,
            args.horizon,
        )
        row["config"]=name
        row["online_gain"]=(
            row["second"]
            -row["first"]
        )
        results.append(row)

    results.sort(
        key=lambda x:(
            x["eval"],
            x["online_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v298",
        "frozen_core":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
        },
        "configs":list(CONFIGS),
        "results":results,
        "train_seeds":train,
        "eval_seeds":evaluation,
        "episodes":args.episodes,
        "horizon":args.horizon,
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
        "=== V298 COMBINED GLOBAL + ELIGIBILITY ==="
    )

    for i,row in enumerate(results,1):
        print(
            f"{i}. {row['config']:24s} "
            f"eval={row['eval']:.3f} "
            f"first={row['first']:.3f} "
            f"second={row['second']:.3f} "
            f"gain={row['online_gain']:+.3f}"
        )


if __name__=="__main__":
    main()

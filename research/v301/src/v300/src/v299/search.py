
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import make_sequence, CoreSystem
from hybrid_credit import CONFIGS, GlobalLongEligibility


def evaluate(
    name,
    train_seeds,
    eval_seeds,
    episodes,
    horizon,
):
    config=CONFIGS[name]

    train_scores=[]
    eval_rows=[]

    for seed in train_seeds:
        seq=make_sequence(
            seed,
            episodes,
            horizon,
        )
        system=CoreSystem(
            GlobalLongEligibility(config)
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        train_scores.append(
            sum(
                int(r["correct"])
                for r in rows
            )/len(rows)
        )

    for seed in eval_seeds:
        seq=make_sequence(
            seed,
            episodes,
            horizon,
        )
        system=CoreSystem(
            GlobalLongEligibility(config)
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        half=len(rows)//2

        eval_rows.append({
            "accuracy":(
                sum(int(r["correct"]) for r in rows)
                /len(rows)
            ),
            "first":(
                sum(int(r["correct"]) for r in rows[:half])
                /half
            ),
            "second":(
                sum(int(r["correct"]) for r in rows[half:])
                /(len(rows)-half)
            ),
        })

    return {
        "train":sum(train_scores)/len(train_scores),
        "eval":sum(
            r["accuracy"] for r in eval_rows
        )/len(eval_rows),
        "first":sum(
            r["first"] for r in eval_rows
        )/len(eval_rows),
        "second":sum(
            r["second"] for r in eval_rows
        )/len(eval_rows),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-seeds",type=int,default=12)
    p.add_argument("--eval-seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=7)
    p.add_argument("--topk",type=int,default=6)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v299_hybrid_credit.json"
        ),
    )
    args=p.parse_args()

    train=list(
        range(
            299,
            299+args.train_seeds,
        )
    )
    evaluation=list(
        range(
            10299,
            10299+args.eval_seeds,
        )
    )

    rows=[]

    for name in CONFIGS:
        metrics=evaluate(
            name,
            train,
            evaluation,
            args.episodes,
            args.horizon,
        )

        rows.append({
            "config":name,
            **metrics,
            "online_gain":(
                metrics["second"]
                -metrics["first"]
            ),
        })

    rows.sort(
        key=lambda x:(
            x["eval"],
            x["online_gain"],
            x["second"],
        ),
        reverse=True,
    )

    payload={
        "version":"v299",
        "mechanism":"global_long_eligibility",
        "frozen_core":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
        },
        "configs":list(CONFIGS),
        "results":rows,
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

    print("=== V299 GLOBAL + LONG ELIGIBILITY ===")
    for i,row in enumerate(rows,1):
        print(
            f"{i:2d}. {row['config']:22s} "
            f"eval={row['eval']:.3f} "
            f"first={row['first']:.3f} "
            f"second={row['second']:.3f} "
            f"gain={row['online_gain']:+.3f}"
        )


if __name__=="__main__":
    main()

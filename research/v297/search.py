
from __future__ import annotations

import argparse
import json
from pathlib import Path

from base import make_sequence, TASKS, CoreSystem
from credits import CREDITS


def evaluate_credit(
    name:str,
    train_seeds,
    eval_seeds,
    episodes:int,
    horizon:int,
):
    train_scores=[]
    eval_scores=[]

    for seed in train_seeds:
        sequence=make_sequence(
            seed,
            "credit",
            episodes,
            horizon,
        )

        system=CoreSystem(
            CREDITS[name]()
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

    for seed in eval_seeds:
        # Fresh system: must learn the new sequence-local rule.
        sequence=make_sequence(
            seed,
            "credit",
            episodes,
            horizon,
        )

        system=CoreSystem(
            CREDITS[name]()
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in sequence.episodes
        ]

        half=len(rows)//2

        eval_scores.append({
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
        "train":(
            sum(train_scores)/len(train_scores)
        ),
        "eval":(
            sum(x["accuracy"] for x in eval_scores)
            /len(eval_scores)
        ),
        "first":(
            sum(x["first"] for x in eval_scores)
            /len(eval_scores)
        ),
        "second":(
            sum(x["second"] for x in eval_scores)
            /len(eval_scores)
        ),
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
            "results/v297_credit_architecture.json"
        ),
    )
    args=p.parse_args()

    train=list(
        range(
            297,
            297+args.train_seeds,
        )
    )
    evaluation=list(
        range(
            10297,
            10297+args.eval_seeds,
        )
    )

    print(
        "=== V297 CREDIT ARCHITECTURE SEARCH ==="
    )
    print(
        "mechanisms=",
        len(CREDITS),
        "train=",
        len(train),
        "eval=",
        len(evaluation),
        "episodes=",
        args.episodes,
    )

    results=[]

    for name in CREDITS:
        row=evaluate_credit(
            name,
            train,
            evaluation,
            args.episodes,
            args.horizon,
        )

        row["credit"]=name
        row["online_gain"]=(
            row["second"]
            -row["first"]
        )

        results.append(row)

    results.sort(
        key=lambda x:(
            x["eval"],
            x["online_gain"],
            x["second"],
        ),
        reverse=True,
    )

    payload={
        "version":"v297",
        "frozen_core":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
        },
        "credits":list(CREDITS),
        "train_seeds":train,
        "eval_seeds":evaluation,
        "episodes":args.episodes,
        "horizon":args.horizon,
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

    print()
    for i,row in enumerate(results,1):
        print(
            f"{i}. {row['credit']:16s} "
            f"eval={row['eval']:.3f} "
            f"first={row['first']:.3f} "
            f"second={row['second']:.3f} "
            f"gain={row['online_gain']:+.3f}"
        )


if __name__=="__main__":
    main()

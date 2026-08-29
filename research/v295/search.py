
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import make_sequence, CoreSystem
from credit_modules import CREDITS


def run_credit(
    credit_name:str,
    train_seeds,
    eval_seeds,
    episodes:int,
):
    train_scores=[]
    eval_scores=[]

    # Training: credit has a persistent state within each sequence.
    for seed in train_seeds:
        sequence=make_sequence(
            seed,
            episodes,
        )
        system=CoreSystem(
            CREDITS[credit_name]()
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
                int(x["correct"])
                for x in rows
            )/len(rows)
        )

    # Evaluation: a fresh credit module must learn the new hidden rule from
    # feedback, so this tests actual online credit rather than memorization.
    for seed in eval_seeds:
        sequence=make_sequence(
            seed,
            episodes,
        )
        system=CoreSystem(
            CREDITS[credit_name]()
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
                    int(x["correct"])
                    for x in rows
                )/len(rows)
            ),
            "first":(
                sum(
                    int(x["correct"])
                    for x in rows[:half]
                )/half
            ),
            "second":(
                sum(
                    int(x["correct"])
                    for x in rows[half:]
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
    p.add_argument("--seed-offset",type=int,default=295)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v295_credit_search.json"
        ),
    )
    args=p.parse_args()

    train=list(
        range(
            args.seed_offset,
            args.seed_offset+args.train_seeds,
        )
    )
    evaluation=list(
        range(
            args.seed_offset+10000,
            args.seed_offset+10000+args.eval_seeds,
        )
    )

    print(
        "=== V295 CREDIT ASSIGNMENT SEARCH ==="
    )
    print(
        f"credits={len(CREDITS)} "
        f"train={len(train)} "
        f"eval={len(evaluation)} "
        f"episodes={args.episodes}"
    )

    results=[]

    for name in CREDITS:
        row=run_credit(
            name,
            train,
            evaluation,
            args.episodes,
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
        "version":"v295",
        "credits":list(CREDITS),
        "train_seeds":train,
        "eval_seeds":evaluation,
        "episodes":args.episodes,
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

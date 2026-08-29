
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from core import make_sequence, CoreSystem
from credit import CONFIGS, HybridCredit


def evaluate(
    config_name,
    train_seeds,
    eval_seeds,
    episodes,
    horizon,
):
    config=CONFIGS[config_name]

    train_scores=[]
    eval_rows=[]

    for seed in train_seeds:
        seq=make_sequence(
            seed,
            episodes,
            horizon,
        )
        system=CoreSystem(
            HybridCredit(config)
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
            HybridCredit(config)
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        half=len(rows)//2

        eval_rows.append(
            {
                "accuracy":sum(
                    int(r["correct"])
                    for r in rows
                )/len(rows),
                "first":sum(
                    int(r["correct"])
                    for r in rows[:half]
                )/half,
                "second":sum(
                    int(r["correct"])
                    for r in rows[half:]
                )/(len(rows)-half),
            }
        )

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
    p.add_argument("--topk",type=int,default=2)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v300_core_credit.json"
        ),
    )
    args=p.parse_args()

    train=list(
        range(
            300,
            300+args.train_seeds,
        )
    )
    evaluation=list(
        range(
            10300,
            10300+args.eval_seeds,
        )
    )

    start=time.perf_counter()
    results=[]

    for name in CONFIGS:
        metrics=evaluate(
            name,
            train,
            evaluation,
            args.episodes,
            args.horizon,
        )

        results.append(
            {
                "credit":name,
                **metrics,
                "online_gain":(
                    metrics["second"]
                    -metrics["first"]
                ),
            }
        )

    elapsed=time.perf_counter()-start

    results.sort(
        key=lambda x:(
            x["eval"],
            x["online_gain"],
        ),
        reverse=True,
    )

    total_episodes=(
        (
            len(train)
            +len(evaluation)
        )
        *args.episodes
        *len(CONFIGS)
    )
    total_steps=total_episodes*(
        args.horizon-1
    )

    payload={
        "version":"v300",
        "frozen_core":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
        },
        "configs":list(CONFIGS),
        "train_seeds":train,
        "eval_seeds":evaluation,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "total_episode_runs":total_episodes,
        "total_state_steps":total_steps,
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

    print("=== V300 FROZEN CORE + CREDIT COMPARISON ===")
    print(
        f"configs={len(CONFIGS)} "
        f"train={len(train)} "
        f"eval={len(evaluation)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon}"
    )
    print(
        f"episode_runs={total_episodes} "
        f"state_steps={total_steps} "
        f"wall_time={elapsed:.3f}s"
    )

    for i,row in enumerate(results,1):
        print(
            f"{i}. {row['credit']:20s} "
            f"eval={row['eval']:.3f} "
            f"first={row['first']:.3f} "
            f"second={row['second']:.3f} "
            f"gain={row['online_gain']:+.3f}"
        )


if __name__=="__main__":
    main()

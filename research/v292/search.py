
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognition import (
    TASKS,
    CognitiveAlgorithm,
    Strategy,
    all_strategies,
    causal_probe,
    make_sequences,
    strategy_count,
    evaluate_sequence,
)


def score_candidate(
    strategy:Strategy,
    train_sequences,
    eval_sequences,
):
    train_scores=[]
    eval_scores=[]
    first=[]
    second=[]

    for sequence in train_sequences:
        row=evaluate_sequence(
            strategy,
            sequence,
            learn=True,
        )
        train_scores.append(
            row["accuracy"]
        )

    # Fresh algorithm for held-out sequences. No train-state leakage.
    for sequence in eval_sequences:
        row=evaluate_sequence(
            strategy,
            sequence,
            learn=True,
        )
        eval_scores.append(
            row["accuracy"]
        )
        first.append(
            row["first_half_accuracy"]
        )
        second.append(
            row["second_half_accuracy"]
        )

    return {
        "train_accuracy":(
            sum(train_scores)
            /len(train_scores)
        ),
        "eval_accuracy":(
            sum(eval_scores)
            /len(eval_scores)
        ),
        "eval_first_half":(
            sum(first)
            /len(first)
        ),
        "eval_second_half":(
            sum(second)
            /len(second)
        ),
    }


def main():
    p=argparse.ArgumentParser(
        description="Graph-native full cognition search",
    )
    p.add_argument("--train-seeds",type=int,default=8)
    p.add_argument("--eval-seeds",type=int,default=8)
    p.add_argument("--episodes-per-sequence",type=int,default=8)
    p.add_argument("--horizon",type=int,default=6)
    p.add_argument("--seed-offset",type=int,default=292)
    p.add_argument("--topk",type=int,default=20)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v292_search.json"
        ),
    )
    args=p.parse_args()

    assert args.episodes_per_sequence>=4
    assert args.horizon>=4

    strategies=all_strategies()

    assert len(strategies)==strategy_count()
    assert len(strategies)==768

    train_seed_values=list(
        range(
            args.seed_offset,
            args.seed_offset+args.train_seeds,
        )
    )
    eval_seed_values=list(
        range(
            args.seed_offset+10000,
            args.seed_offset+10000+args.eval_seeds,
        )
    )

    train_sequences=make_sequences(
        train_seed_values,
        args.episodes_per_sequence,
        args.horizon,
    )

    eval_sequences=make_sequences(
        eval_seed_values,
        args.episodes_per_sequence,
        args.horizon,
    )

    print(
        "=== V292 FULL GRAPH COGNITION SEARCH ===",
        flush=True,
    )
    print(
        f"strategies={len(strategies)} "
        f"train_sequences={len(train_sequences)} "
        f"eval_sequences={len(eval_sequences)} "
        f"episodes_per_sequence={args.episodes_per_sequence} "
        f"horizon={args.horizon}",
        flush=True,
    )
    print(
        "tasks="+",".join(TASKS),
        flush=True,
    )

    rows=[]

    for index,strategy in enumerate(
        strategies,
        start=1,
    ):
        metrics=score_candidate(
            strategy,
            train_sequences,
            eval_sequences,
        )

        rows.append(
            {
                "strategy":strategy,
                **metrics,
            }
        )

        if index%100==0:
            print(
                f"evaluated={index}/{len(strategies)}",
                flush=True,
            )

    rows.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["eval_second_half"],
            x["eval_second_half"]
            -x["eval_first_half"],
        ),
        reverse=True,
    )

    top=[]

    # Causal tests use fresh held-out episodes.
    heldout_episodes=[
        sequence.episodes[0]
        for sequence in eval_sequences
    ]

    for row in rows[:args.topk]:
        probes=[
            causal_probe(
                row["strategy"],
                episode,
            )
            for episode in heldout_episodes
        ]

        n=len(probes)

        top.append(
            {
                "name":row["strategy"].name,
                "strategy":{
                    "memory":row["strategy"].memory,
                    "credit":row["strategy"].credit,
                    "dynamics":row["strategy"].dynamics,
                    "readout":row["strategy"].readout,
                    "planner":row["strategy"].planner,
                },
                "train_accuracy":row["train_accuracy"],
                "eval_accuracy":row["eval_accuracy"],
                "eval_first_half":row["eval_first_half"],
                "eval_second_half":row["eval_second_half"],
                "online_learning_gain":(
                    row["eval_second_half"]
                    -row["eval_first_half"]
                ),
                "memory_drop":(
                    sum(
                        p["memory_drop"]
                        for p in probes
                    )/n
                ),
            }
        )

    payload={
        "version":"v292",
        "space_size":len(strategies),
        "tasks":list(TASKS),
        "horizon":args.horizon,
        "episodes_per_sequence":args.episodes_per_sequence,
        "train_seed_count":args.train_seeds,
        "eval_seed_count":args.eval_seeds,
        "top":top,
        "all_results":[
            {
                "name":r["strategy"].name,
                "strategy":{
                    "memory":r["strategy"].memory,
                    "credit":r["strategy"].credit,
                    "dynamics":r["strategy"].dynamics,
                    "readout":r["strategy"].readout,
                    "planner":r["strategy"].planner,
                },
                "train_accuracy":r["train_accuracy"],
                "eval_accuracy":r["eval_accuracy"],
                "eval_first_half":r["eval_first_half"],
                "eval_second_half":r["eval_second_half"],
            }
            for r in rows
        ],
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
    print("=== TOP STRATEGIES ===")

    for i,row in enumerate(top,1):
        print(
            f"{i:2d}. "
            f"{row['name']:72s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"learn_gain={row['online_learning_gain']:+.3f} "
            f"memdrop={row['memory_drop']:.3f}"
        )


if __name__=="__main__":
    main()

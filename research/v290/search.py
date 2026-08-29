
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from graph_cognition import (
    all_strategies,
    evaluate_strategy,
    result_to_json,
    strategy_count,
)


def main():
    p=argparse.ArgumentParser(
        description="Anti-shortcut graph-native cognitive search",
    )
    p.add_argument(
        "--train-seeds",
        type=int,
        default=8,
    )
    p.add_argument(
        "--eval-seeds",
        type=int,
        default=8,
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=290,
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=4,
    )
    p.add_argument(
        "--topk",
        type=int,
        default=20,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v290_strategy_search.json"
        ),
    )
    args=p.parse_args()

    train_seeds=list(
        range(
            args.seed_offset,
            args.seed_offset+args.train_seeds,
        )
    )
    eval_seeds=list(
        range(
            args.seed_offset+1000,
            args.seed_offset+1000+args.eval_seeds,
        )
    )

    strategies=all_strategies()

    assert len(strategies)==strategy_count()
    assert len(strategies)==3*5*4*4*4

    print(
        "=== V290 GRAPH-NATIVE ANTI-SHORTCUT SEARCH ===",
        flush=True,
    )
    print(
        f"strategy_space={len(strategies)}",
        flush=True,
    )
    print(
        f"train_seeds={train_seeds}",
        flush=True,
    )
    print(
        f"eval_seeds={eval_seeds}",
        flush=True,
    )
    print(
        f"horizon={args.horizon}",
        flush=True,
    )

    rows=[]

    for index,strategy in enumerate(
        strategies,
        start=1,
    ):
        row=evaluate_strategy(
            strategy,
            train_seeds,
            eval_seeds,
            args.horizon,
        )
        rows.append(row)

        if index%100==0:
            print(
                f"evaluated={index}/{len(strategies)}",
                flush=True,
            )

    # Composite ranking: generalization first, then causal dependence.
    rows.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["causal_memory_drop"],
            x["causal_swap_correct"],
            x["causal_swap_change"],
            x["train_accuracy"],
        ),
        reverse=True,
    )

    top=[
        result_to_json(x)
        for x in rows[:args.topk]
    ]

    payload={
        "version":"v290",
        "space_size":len(strategies),
        "train_seed_count":args.train_seeds,
        "eval_seed_count":args.eval_seeds,
        "horizon":args.horizon,
        "anti_shortcut":True,
        "ranking":"eval_accuracy_then_causal_metrics",
        "top":top,
        "all_results":[
            result_to_json(x)
            for x in rows
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

    for rank,row in enumerate(top,1):
        print(
            f"{rank:2d}. "
            f"{row['name']:78s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"drop={row['causal_memory_drop']:.3f} "
            f"swap={row['causal_swap_correct']:.3f}"
        )

    print(
        "saved:",
        args.output,
        flush=True,
    )


if __name__=="__main__":
    main()

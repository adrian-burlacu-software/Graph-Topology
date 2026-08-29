
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognition import (
    TASKS,
    CognitiveSystem,
    all_architectures,
    architecture_count,
    causal_profile,
    make_sequence,
    evaluate_sequence,
)


def main():
    p=argparse.ArgumentParser(
        description="Fully discriminated graph-native architecture search"
    )

    p.add_argument("--train-seeds",type=int,default=6)
    p.add_argument("--eval-seeds",type=int,default=6)
    p.add_argument("--episodes-per-sequence",type=int,default=10)
    p.add_argument("--horizon",type=int,default=7)
    p.add_argument("--seed-offset",type=int,default=294)
    p.add_argument("--topk",type=int,default=20)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v294_architecture_search.json"
        ),
    )

    args=p.parse_args()

    architectures=all_architectures()

    assert len(architectures)==architecture_count()
    assert len(architectures)==architecture_count()

    train=[
        make_sequence(
            seed,
            task,
            args.episodes_per_sequence,
            args.horizon,
        )
        for seed in range(
            args.seed_offset,
            args.seed_offset+args.train_seeds,
        )
        for task in TASKS
    ]

    evaluation=[
        make_sequence(
            seed,
            task,
            args.episodes_per_sequence,
            args.horizon,
        )
        for seed in range(
            args.seed_offset+10000,
            args.seed_offset+10000+args.eval_seeds,
        )
        for task in TASKS
    ]

    print(
        "=== V294 FULLY DISCRIMINATED SEARCH ===",
        flush=True,
    )
    print(
        f"architectures={len(architectures)} "
        f"train_sequences={len(train)} "
        f"eval_sequences={len(evaluation)} "
        f"episodes={args.episodes_per_sequence} "
        f"horizon={args.horizon}",
        flush=True,
    )

    rows=[]

    for i,architecture in enumerate(
        architectures,
        1,
    ):
        train_metrics=[
            evaluate_sequence(
                architecture,
                sequence,
            )
            for sequence in train
        ]

        eval_metrics=[
            evaluate_sequence(
                architecture,
                sequence,
            )
            for sequence in evaluation
        ]

        eval_accuracy=sum(
            m["accuracy"]
            for m in eval_metrics
        )/len(eval_metrics)

        first=sum(
            m["first_half"]
            for m in eval_metrics
        )/len(eval_metrics)

        second=sum(
            m["second_half"]
            for m in eval_metrics
        )/len(eval_metrics)

        train_accuracy=sum(
            m["accuracy"]
            for m in train_metrics
        )/len(train_metrics)

        rows.append({
            "architecture":architecture,
            "train_accuracy":train_accuracy,
            "eval_accuracy":eval_accuracy,
            "first_half":first,
            "second_half":second,
            "learning_gain":second-first,
        })

        if i%100==0:
            print(
                f"evaluated={i}/{len(architectures)}",
                flush=True,
            )

    rows.sort(
        key=lambda r:(
            r["eval_accuracy"],
            r["learning_gain"],
        ),
        reverse=True,
    )

    causal_episodes=[
        sequence.episodes[0]
        for sequence in evaluation
    ]

    finalists=[]

    for row in rows[:args.topk]:
        profile=causal_profile(
            row["architecture"],
            causal_episodes,
        )

        positive=[
            max(0.0,float(v))
            for v in profile["drops"].values()
        ]

        active=sum(
            value not in (
                "none",
                "static",
                "null",
                "direct",
            )
            for value in (
                row["architecture"].memory,
                row["architecture"].dynamics,
                row["architecture"].readout,
                row["architecture"].planner,
                row["architecture"].credit,
            )
        )

        synergy=(
            sum(positive)
            +0.20*sum(
                int(v>0.10)
                for v in positive
            )
            +0.05*active
        )

        finalists.append({
            "name":row["architecture"].name,
            "architecture":{
                "memory":row["architecture"].memory,
                "dynamics":row["architecture"].dynamics,
                "readout":row["architecture"].readout,
                "planner":row["architecture"].planner,
                "credit":row["architecture"].credit,
            },
            "train_accuracy":row["train_accuracy"],
            "eval_accuracy":row["eval_accuracy"],
            "first_half":row["first_half"],
            "second_half":row["second_half"],
            "learning_gain":row["learning_gain"],
            "causal_normal":profile["normal"],
            "ablation_drop":profile["drops"],
            "architecture_synergy":synergy,
        })

    finalists.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["architecture_synergy"],
            x["learning_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v294",
        "space_size":len(architectures),
        "tasks":list(TASKS),
        "horizon":args.horizon,
        "episodes_per_sequence":args.episodes_per_sequence,
        "top":finalists,
        "all_results":[
            {
                "name":r["architecture"].name,
                "architecture":{
                    "memory":r["architecture"].memory,
                    "dynamics":r["architecture"].dynamics,
                    "readout":r["architecture"].readout,
                    "planner":r["architecture"].planner,
                    "credit":r["architecture"].credit,
                },
                "train_accuracy":r["train_accuracy"],
                "eval_accuracy":r["eval_accuracy"],
                "first_half":r["first_half"],
                "second_half":r["second_half"],
                "learning_gain":r["learning_gain"],
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
    print("=== TOP DISCRIMINATED CANDIDATES ===")

    for i,row in enumerate(
        finalists,
        1,
    ):
        print(
            f"{i:2d}. "
            f"{row['name']:72s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"synergy={row['architecture_synergy']:.3f}"
        )


if __name__=="__main__":
    main()

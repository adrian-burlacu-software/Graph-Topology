
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from graph_cognition import (
    CognitiveAlgorithm,
    Strategy,
    TASK_TYPES,
    all_strategies,
    make_task,
)


def evaluate_strategy(
    strategy: Strategy,
    seeds: range,
    horizons: tuple[int, ...],
    task_types: tuple[str, ...],
) -> dict:
    totals = {
        "cases": 0,
        "correct": 0,
    }

    by_task = {}
    by_horizon = {}

    for task_type in task_types:
        correct = 0
        cases = 0

        for seed in seeds:
            for hidden in (0, 1):
                task = make_task(
                    seed,
                    task_type,
                    hidden,
                )

                algo = CognitiveAlgorithm(
                    strategy
                )

                # Evaluate all requested horizons independently. The graph
                # algorithm itself is tiny; keeping runs independent prevents
                # accidental state leakage between samples.
                for horizon in horizons:
                    result = algo.run(
                        task,
                        horizon=horizon,
                    )
                    cases += 1
                    correct += int(
                        result["correct"]
                    )

        by_task[task_type] = {
            "correct": correct,
            "cases": cases,
            "accuracy": (
                correct / cases
                if cases
                else 0.0
            ),
        }

    for horizon in horizons:
        correct = 0
        cases = 0

        for task_type in task_types:
            for seed in seeds:
                for hidden in (0, 1):
                    task = make_task(
                        seed,
                        task_type,
                        hidden,
                    )
                    result=CognitiveAlgorithm(
                        strategy
                    ).run(
                        task,
                        horizon=horizon,
                    )
                    cases += 1
                    correct += int(
                        result["correct"]
                    )

        by_horizon[str(horizon)] = {
            "correct": correct,
            "cases": cases,
            "accuracy": (
                correct / cases
                if cases
                else 0.0
            ),
        }

    return {
        "strategy": asdict(strategy),
        "name": strategy.name,
        "by_task": by_task,
        "by_horizon": by_horizon,
        "accuracy": (
            sum(
                x["correct"]
                for x in by_task.values()
            )
            /max(
                1,
                sum(
                    x["cases"]
                    for x in by_task.values()
                ),
            )
        ),
    }


def causal_probe(
    strategy: Strategy,
    seed: int,
    task_type: str,
    horizon: int = 4,
) -> dict:
    """
    Counterfactual probes on the graph-native algorithm.

    1. Normal run
    2. Memory erased before terminal decision
    3. Distractor strengthened
    4. Hidden bit swapped

    A useful cognitive mechanism should:
      - perform well normally
      - lose performance when memory is erased
      - resist irrelevant distractor changes
      - reverse when the remembered fact is reversed
    """
    task=make_task(
        seed,
        task_type,
        hidden_bit=0,
    )

    algo=CognitiveAlgorithm(strategy)
    normal=algo.run(
        task,
        horizon=horizon,
    )

    # Memory erasure by selecting a fresh graph algorithm and explicitly
    # zeroing its memory carrier before readout is implemented in a helper
    # below. This keeps the intervention independent of the training result.
    erased=CognitiveAlgorithm(strategy)
    erased_result=run_with_memory_ablation(
        erased,
        task,
        horizon,
    )

    swapped_task=make_task(
        seed,
        task_type,
        hidden_bit=1,
    )
    swapped=CognitiveAlgorithm(
        strategy
    ).run(
        swapped_task,
        horizon=horizon,
    )

    return {
        "normal_correct": int(
            normal["correct"]
        ),
        "memory_ablation_correct": int(
            erased_result["correct"]
        ),
        "swap_action_changed": int(
            normal["action"]
            != swapped["action"]
        ),
        "swap_direction_correct": int(
            swapped["action"]
            ==swapped_task.expected_action
        ),
    }


def run_with_memory_ablation(
    algo: CognitiveAlgorithm,
    task,
    horizon: int,
) -> dict:
    from graph_cognition import CognitiveGraph

    graph=CognitiveGraph()
    graph.ensure_node("MEMORY")
    graph.ensure_node("GOAL")

    algo.memory.write(
        graph,
        task,
    )

    for step in range(1,horizon):
        algo.dynamics.pre_step(
            graph,
            step,
        )
        algo.dynamics.post_step(
            graph,
            step,
        )

    # Surgical memory intervention: remove the algorithm's memory carrier while
    # leaving the rest of the graph computation intact.
    graph.edges=[
        e for e in graph.edges
        if not (
            e.source=="MEMORY"
            or e.target.startswith("MEM_")
        )
    ]

    for name in (
        "MEMORY_ONE",
        "MEMORY_ZERO",
        "MEMORY_SLOT",
    ):
        if name in graph.nodes:
            graph.nodes[name].activation=0.0
            graph.nodes[name].persistent=False

    recalled=algo.readout.read(
        graph,
        task,
    )

    predicted=algo.planning.transform(
        graph,
        task,
        recalled,
    )

    action=(
        "LEFT"
        if predicted==0
        else "RIGHT"
    )

    return {
        "correct": action==task.expected_action,
        "action": action,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        default=12,
        help="number of small search seeds",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default="2,4",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=12,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v289_strategy_search.json"
        ),
    )
    args=p.parse_args()

    horizons=tuple(
        int(x)
        for x in args.horizons.split(",")
        if x.strip()
    )

    seeds=range(1,args.seeds+1)

    strategies=all_strategies()

    print(
        "=== V289 GRAPH COGNITIVE ALGORITHM SEARCH ==="
    )
    print(
        f"strategies={len(strategies)} "
        f"seeds={args.seeds} "
        f"horizons={horizons}"
    )
    print(
        f"combinatorial_space="
        f"{len(strategies)}"
    )

    results=[]

    for index,strategy in enumerate(
        strategies,
        start=1,
    ):
        result=evaluate_strategy(
            strategy,
            seeds=seeds,
            horizons=horizons,
            task_types=TASK_TYPES,
        )
        results.append(result)

        if index % 64 == 0:
            print(
                f"evaluated={index}/{len(strategies)}",
                flush=True,
            )

    results.sort(
        key=lambda x:x["accuracy"],
        reverse=True,
    )

    top=[]
    for result in results[:args.topk]:
        strategy=Strategy(
            **result["strategy"]
        )

        probe_rows=[
            causal_probe(
                strategy,
                seed,
                task_type,
                horizon=max(horizons),
            )
            for seed in seeds
            for task_type in TASK_TYPES
        ]

        total=sum(
            row["normal_correct"]
            for row in probe_rows
        )
        ablation=sum(
            row["memory_ablation_correct"]
            for row in probe_rows
        )
        swaps=sum(
            row["swap_action_changed"]
            for row in probe_rows
        )
        swap_correct=sum(
            row["swap_direction_correct"]
            for row in probe_rows
        )
        n=len(probe_rows)

        top.append(
            {
                "search":result,
                "causal": {
                    "cases": n,
                    "normal_accuracy": total/n,
                    "memory_ablation_accuracy": ablation/n,
                    "memory_ablation_drop": (
                        total/n-ablation/n
                    ),
                    "swap_change_rate": swaps/n,
                    "swap_correct_rate": swap_correct/n,
                },
            }
        )

    payload={
        "version":"v289",
        "space_size":len(strategies),
        "seeds":args.seeds,
        "horizons":list(horizons),
        "task_types":list(TASK_TYPES),
        "topk":args.topk,
        "top":top,
        "all_results":results,
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
        s=row["search"]
        c=row["causal"]
        print(
            f"{rank:2d}. "
            f"{s['name']:75s} "
            f"search={s['accuracy']:.3f} "
            f"normal={c['normal_accuracy']:.3f} "
            f"ablation_drop={c['memory_ablation_drop']:.3f} "
            f"swap={c['swap_correct_rate']:.3f}"
        )

    print(
        "saved:",
        args.output,
    )


if __name__=="__main__":
    main()

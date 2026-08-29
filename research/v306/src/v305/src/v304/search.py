
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from competition import CONFIGS as COMP_CONFIGS, GoalCompetition
from hypothesis import CONFIGS as HYP_CONFIGS, HypothesisRevision
from integrated import IntegratedSystem
from richer_cognition import TASKS, make_sequence


def evaluate(
    competition_name,
    seeds,
    episodes,
    horizon,
):
    task_rows=[]

    for task in TASKS:
        acc=[]
        first=[]
        second=[]

        for seed in seeds:
            seq=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )

            system=IntegratedSystem(
                GoalCompetition(
                    COMP_CONFIGS[competition_name]
                ),
                HypothesisRevision(
                    HYP_CONFIGS["fast_revision"]
                ),
            )

            rows=[
                system.run(
                    ep,
                    learn=True,
                )
                for ep in seq.episodes
            ]

            half=len(rows)//2

            acc.append(
                sum(int(r["correct"]) for r in rows)/len(rows)
            )
            first.append(
                sum(int(r["correct"]) for r in rows[:half])/half
            )
            second.append(
                sum(int(r["correct"]) for r in rows[half:])/len(rows[half:])
            )

        task_rows.append({
            "task":task,
            "accuracy":sum(acc)/len(acc),
            "first":sum(first)/len(first),
            "second":sum(second)/len(second),
        })

    overall=sum(x["accuracy"] for x in task_rows)/len(task_rows)
    fh=sum(x["first"] for x in task_rows)/len(task_rows)
    sh=sum(x["second"] for x in task_rows)/len(task_rows)

    return {
        "eval_accuracy":overall,
        "first_half":fh,
        "second_half":sh,
        "learning_gain":sh-fh,
        "tasks":task_rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=9)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent/"results"/"v304_goal_competition.json",
    )
    args=p.parse_args()

    seeds=list(
        range(
            304,
            304+args.seeds,
        )
    )

    started=time.perf_counter()
    rows=[]

    for name in COMP_CONFIGS:
        rows.append({
            "competition":name,
            **evaluate(
                name,
                seeds,
                args.episodes,
                args.horizon,
            ),
        })

    elapsed=time.perf_counter()-started

    rows.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["learning_gain"],
        ),
        reverse=True,
    )

    payload={
        "version":"v304",
        "benchmark":"goal_competition",
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":args.episodes,
        "horizon":args.horizon,
        "config_count":len(COMP_CONFIGS),
        "fixed_architecture":{
            "memory":"persistent",
            "dynamics":"transform",
            "readout":"memory",
            "planner":"binding",
            "hypothesis":"fast_revision",
        },
        "wall_time_seconds":elapsed,
        "results":rows,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload,indent=2),
        encoding="utf-8",
    )

    print("=== V304 GOAL-COMPETITION ===")
    print(
        f"configs={len(COMP_CONFIGS)} "
        f"tasks={len(TASKS)} "
        f"seeds={len(seeds)} "
        f"episodes={args.episodes} "
        f"horizon={args.horizon} "
        f"wall={elapsed:.3f}s"
    )

    for row in rows:
        task_map={x["task"]:x for x in row["tasks"]}
        print(
            f"{row['competition']:20s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"interference={task_map['interference']['accuracy']:.3f} "
            f"rule_change={task_map['rule_change']['accuracy']:.3f} "
            f"counterfactual={task_map['counterfactual']['accuracy']:.3f}"
        )


if __name__=="__main__":
    main()


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import copy
import json
from pathlib import Path

# Import the V294 graph-native implementation under a stable namespace.
import sys
ROOT=Path(__file__).resolve().parent
V294=ROOT/"v294_source"
sys.path.insert(0,str(V294))

import cognition as base


# The discovered V294 core is frozen to:
#   persistent memory
#   transform dynamics
#
# We focus on:
#   readout × planner × credit
#
# Credit is restricted to the two meaningful candidates from V295.
FIXED_MEMORY="persistent"
FIXED_DYNAMICS="transform"
READOUTS=(
    "memory",
    "relational",
    "integrative",
    "state",
)
PLANNERS=(
    "binding",
    "control",
    "rollout",
)
CREDITS=(
    "immediate",
    "eligibility",
)

TASKS=(
    "memory",
    "binding",
    "dynamics",
    "credit",
    "planning",
)


@dataclass(frozen=True)
class Candidate:
    readout:str
    planner:str
    credit:str

    @property
    def name(self)->str:
        return (
            f"{FIXED_MEMORY}+"
            f"{FIXED_DYNAMICS}+"
            f"{self.readout}+"
            f"{self.planner}+"
            f"{self.credit}"
        )


def candidates()->List[Candidate]:
    return [
        Candidate(r,p,c)
        for r in READOUTS
        for p in PLANNERS
        for c in CREDITS
    ]


def make_episode(
    seed:int,
    task:str,
    index:int,
    latent_rule:int=0,
    horizon:int=7,
):
    return base.make_episode(
        seed,
        task,
        index,
        latent_rule,
        horizon,
    )


def make_sequence(
    seed:int,
    task:str,
    episodes:int=10,
    horizon:int=7,
):
    return base.make_sequence(
        seed,
        task,
        episodes,
        horizon,
    )


def build_system(c:Candidate):
    return base.CognitiveSystem(
        base.Architecture(
            FIXED_MEMORY,
            FIXED_DYNAMICS,
            c.readout,
            c.planner,
            c.credit,
        )
    )


def evaluate_sequence(
    c:Candidate,
    sequence,
)->dict:
    system=build_system(c)

    rows=[
        system.run(
            ep,
            learn=True,
        )
        for ep in sequence.episodes
    ]

    half=len(rows)//2

    return {
        "accuracy":(
            sum(
                int(r["correct"])
                for r in rows
            )/len(rows)
        ),
        "first_half":(
            sum(
                int(r["correct"])
                for r in rows[:half]
            )/half
        ),
        "second_half":(
            sum(
                int(r["correct"])
                for r in rows[half:]
            )/(len(rows)-half)
        ),
    }


def component_ablation(
    c:Candidate,
    sequences,
    component:str,
)->float:
    rows=[]

    for seq in sequences:
        parts={
            "memory":FIXED_MEMORY,
            "dynamics":FIXED_DYNAMICS,
            "readout":c.readout,
            "planner":c.planner,
            "credit":c.credit,
        }

        if component=="readout":
            replacement="null"
        elif component=="planner":
            replacement="direct"
        elif component=="credit":
            replacement="none"
        else:
            raise ValueError(component)

        parts[component]=replacement

        system=base.CognitiveSystem(
            base.Architecture(
                parts["memory"],
                parts["dynamics"],
                parts["readout"],
                parts["planner"],
                parts["credit"],
            )
        )

        run_rows=[
            system.run(
                ep,
                learn=False,
            )
            for ep in seq.episodes
        ]

        rows.extend(run_rows)

    return (
        sum(int(r["correct"]) for r in rows)
        /len(rows)
    )


def evaluate(
    c:Candidate,
    train_sequences,
    eval_sequences,
)->dict:
    train=[
        evaluate_sequence(c,seq)
        for seq in train_sequences
    ]
    evaluation=[
        evaluate_sequence(c,seq)
        for seq in eval_sequences
    ]

    eval_accuracy=sum(
        x["accuracy"]
        for x in evaluation
    )/len(evaluation)

    first=sum(
        x["first_half"]
        for x in evaluation
    )/len(evaluation)

    second=sum(
        x["second_half"]
        for x in evaluation
    )/len(evaluation)

    return {
        "train_accuracy":sum(
            x["accuracy"]
            for x in train
        )/len(train),
        "eval_accuracy":eval_accuracy,
        "first_half":first,
        "second_half":second,
        "learning_gain":second-first,
    }


def focused_causal_profile(
    c:Candidate,
    eval_sequences,
)->dict:
    episodes=[
        seq.episodes[0]
        for seq in eval_sequences
    ]

    normal_system=build_system(c)
    normal=[
        normal_system.run(
            ep,
            learn=False,
        )
        for ep in episodes
    ]

    normal_acc=sum(
        int(r["correct"])
        for r in normal
    )/len(normal)

    drops={}

    for component in (
        "readout",
        "planner",
        "credit",
    ):
        ablated=component_ablation(
            c,
            eval_sequences,
            component,
        )
        drops[component]=normal_acc-ablated

    return {
        "normal":normal_acc,
        "drops":drops,
    }


def run_search(
    train_seeds:List[int],
    eval_seeds:List[int],
    episodes:int,
    horizon:int,
    topk:int,
)->dict:
    train=[
        make_sequence(
            seed,
            task,
            episodes,
            horizon,
        )
        for seed in train_seeds
        for task in TASKS
    ]

    evaluation=[
        make_sequence(
            seed,
            task,
            episodes,
            horizon,
        )
        for seed in eval_seeds
        for task in TASKS
    ]

    rows=[]

    for c in candidates():
        metrics=evaluate(
            c,
            train,
            evaluation,
        )
        rows.append({
            "candidate":c,
            **metrics,
        })

    rows.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["learning_gain"],
            x["second_half"],
        ),
        reverse=True,
    )

    top=[]

    for row in rows[:topk]:
        profile=focused_causal_profile(
            row["candidate"],
            evaluation,
        )

        synergy=sum(
            max(0.0,float(v))
            for v in profile["drops"].values()
        )

        c=row["candidate"]

        top.append({
            "name":c.name,
            "readout":c.readout,
            "planner":c.planner,
            "credit":c.credit,
            "train_accuracy":row["train_accuracy"],
            "eval_accuracy":row["eval_accuracy"],
            "first_half":row["first_half"],
            "second_half":row["second_half"],
            "learning_gain":row["learning_gain"],
            "causal_normal":profile["normal"],
            "ablation_drop":profile["drops"],
            "synergy":synergy,
        })

    top.sort(
        key=lambda x:(
            x["eval_accuracy"],
            x["synergy"],
            x["learning_gain"],
        ),
        reverse=True,
    )

    return {
        "version":"v296",
        "fixed_core":{
            "memory":FIXED_MEMORY,
            "dynamics":FIXED_DYNAMICS,
        },
        "search_space":{
            "readout":list(READOUTS),
            "planner":list(PLANNERS),
            "credit":list(CREDITS),
            "count":len(candidates()),
        },
        "tasks":list(TASKS),
        "train_seeds":train_seeds,
        "eval_seeds":eval_seeds,
        "episodes":episodes,
        "horizon":horizon,
        "top":top,
        "all_results":[
            {
                "name":r["candidate"].name,
                "readout":r["candidate"].readout,
                "planner":r["candidate"].planner,
                "credit":r["candidate"].credit,
                "train_accuracy":r["train_accuracy"],
                "eval_accuracy":r["eval_accuracy"],
                "first_half":r["first_half"],
                "second_half":r["second_half"],
                "learning_gain":r["learning_gain"],
            }
            for r in rows
        ],
    }


if __name__=="__main__":
    import argparse

    p=argparse.ArgumentParser()
    p.add_argument("--train-seeds",type=int,default=6)
    p.add_argument("--eval-seeds",type=int,default=6)
    p.add_argument("--episodes",type=int,default=10)
    p.add_argument("--horizon",type=int,default=7)
    p.add_argument("--topk",type=int,default=12)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v296_recombination.json"
        ),
    )
    args=p.parse_args()

    payload=run_search(
        list(
            range(
                296,
                296+args.train_seeds,
            )
        ),
        list(
            range(
                10296,
                10296+args.eval_seeds,
            )
        ),
        args.episodes,
        args.horizon,
        args.topk,
    )

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

    print("=== V296 FOCUSED RECOMBINATION ===")
    print(
        "fixed:",
        payload["fixed_core"],
    )
    print(
        "space:",
        payload["search_space"]["count"],
    )

    for i,row in enumerate(
        payload["top"],
        1,
    ):
        print(
            f"{i:2d}. {row['name']:65s} "
            f"eval={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"synergy={row['synergy']:.3f}"
        )

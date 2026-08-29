
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from richer_cognition import TASKS,make_sequence
from integrated import IntegratedSystem


MODES=(
    "carrier_all",
    "carrier_persistent",
    "carrier_context",
    "carrier_disabled",
)


def evaluate(mode,seeds,episodes,horizon):
    rows=[]
    overall=[]
    latent_pass=[]
    missing=Counter()

    for task in TASKS:
        acc=[]
        passes=[]

        for seed in seeds:
            seq=make_sequence(
                seed,
                task,
                episodes,
                horizon,
            )
            system=IntegratedSystem(mode)

            result=[
                system.run(ep,True)
                for ep in seq.episodes
            ]

            acc.append(
                sum(
                    int(x["correct"])
                    for x in result
                )/len(result)
            )

            passes.append(
                sum(
                    int(x["audit"]["pass"])
                    for x in result
                )/len(result)
            )

            for x in result:
                for code in x["audit"]["missing"]:
                    missing[f"missing:{code}"]+=1
                for code in x["audit"]["mismatched"]:
                    missing[f"mismatch:{code}"]+=1
                if x["audit"]["active_rule_mismatch"]:
                    missing["active_rule_mismatch"]+=1

        value=sum(acc)/len(acc)
        rows.append({
            "task":task,
            "accuracy":value,
            "latent_binding_valid_rate":(
                sum(passes)/len(passes)
            ),
        })
        overall.append(value)
        latent_pass.extend(passes)

    return {
        "eval_accuracy":sum(overall)/len(overall),
        "latent_binding_valid_rate":(
            sum(latent_pass)/len(latent_pass)
        ),
        "failure_counts":dict(
            missing.most_common()
        ),
        "tasks":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--episodes",type=int,default=16)
    p.add_argument("--horizon",type=int,default=9)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        /"results"/"v349.json",
    )
    a=p.parse_args()

    seeds=list(range(349,349+a.seeds))
    start=time.perf_counter()

    results=[
        {
            "architecture":m,
            **evaluate(
                m,
                seeds,
                a.episodes,
                a.horizon,
            ),
        }
        for m in MODES
    ]

    elapsed=time.perf_counter()-start
    results.sort(
        key=lambda x:x["eval_accuracy"],
        reverse=True,
    )

    payload={
        "version":"v349",
        "benchmark":"explicit_latent_state_carrier",
        "modes":list(MODES),
        "tasks":list(TASKS),
        "seeds":seeds,
        "episodes":a.episodes,
        "horizon":a.horizon,
        "wall_time_seconds":elapsed,
        "results":results,
    }

    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        json.dumps(payload,indent=2),
        encoding="utf-8",
    )

    print("=== V349 LATENT STATE CARRIER ===")
    print(f"wall={elapsed:.3f}s")
    for row in results:
        print(
            f"{row['architecture']:20s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"latent_valid={row['latent_binding_valid_rate']:.3f}"
        )
        print(" failures=",row["failure_counts"])
        for task in row["tasks"]:
            print(
                "  ",
                task["task"],
                "acc=",round(task["accuracy"],3),
                "latent=",round(
                    task["latent_binding_valid_rate"],
                    3,
                ),
            )


if __name__=="__main__":
    main()


from __future__ import annotations
import argparse,json,time
from pathlib import Path
from richer_cognition import TASKS,make_sequence
from integrated import IntegratedSystem


MODES=(
    "criterion_balanced",
    "criterion_narrow",
    "criterion_broad",
    "criterion_strict",
)


def evaluate(mode,seeds,episodes,horizon):
    rows=[]

    for task in TASKS:
        accuracy=[]
        first=[]
        second=[]
        state_valid=[]
        answer_valid=[]

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

            h=len(result)//2

            accuracy.append(
                sum(int(x["correct"]) for x in result)/len(result)
            )
            first.append(
                sum(int(x["correct"]) for x in result[:h])/h
            )
            second.append(
                sum(int(x["correct"]) for x in result[h:])/len(result[h:])
            )
            state_valid.append(
                sum(int(x["state_valid"]) for x in result)/len(result)
            )
            answer_valid.append(
                sum(int(x["answer_valid"]) for x in result)/len(result)
            )

        rows.append({
            "task":task,
            "accuracy":sum(accuracy)/len(accuracy),
            "first":sum(first)/len(first),
            "second":sum(second)/len(second),
            "state_valid":sum(state_valid)/len(state_valid),
            "answer_valid":sum(answer_valid)/len(answer_valid),
        })

    f=sum(x["first"] for x in rows)/len(rows)
    s=sum(x["second"] for x in rows)/len(rows)

    return {
        "eval_accuracy":sum(x["accuracy"] for x in rows)/len(rows),
        "first_half":f,
        "second_half":s,
        "learning_gain":s-f,
        "tasks":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=4)
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--horizon",type=int,default=9)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent/"results"/"v328_answer_criterion.json",
    )
    a=p.parse_args()

    seeds=list(range(328,328+a.seeds))

    start=time.perf_counter()
    results=[
        {
            "architecture":mode,
            **evaluate(
                mode,
                seeds,
                a.episodes,
                a.horizon,
            ),
        }
        for mode in MODES
    ]
    elapsed=time.perf_counter()-start

    results.sort(
        key=lambda x:(x["eval_accuracy"],x["learning_gain"]),
        reverse=True,
    )

    payload={
        "version":"v328",
        "benchmark":"explicit_answer_criterion",
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

    print("=== V328 EXPLICIT ANSWER CRITERION ===")
    print(
        f"modes={len(MODES)} tasks={len(TASKS)} seeds={len(seeds)} "
        f"episodes={a.episodes} horizon={a.horizon} wall={elapsed:.3f}s"
    )

    for row in results:
        t={x["task"]:x for x in row["tasks"]}
        print(
            f"{row['architecture']:24s} "
            f"overall={row['eval_accuracy']:.3f} "
            f"gain={row['learning_gain']:+.3f} "
            f"interference={t['interference']['accuracy']:.3f} "
            f"rule_change={t['rule_change']['accuracy']:.3f} "
            f"counterfactual={t['counterfactual']['accuracy']:.3f} "
            f"state_valid={sum(x['state_valid'] for x in row['tasks'])/len(row['tasks']):.3f} "
            f"answer_valid={sum(x['answer_valid'] for x in row['tasks'])/len(row['tasks']):.3f}"
        )


if __name__=="__main__":
    main()

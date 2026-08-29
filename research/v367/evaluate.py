
from __future__ import annotations
import argparse,json,time
from pathlib import Path

from benchmark import curriculum
from architecture import VerifiedOperatorArchitecture


MODES=(
    "verified_balanced",
    "verified_sparse",
    "verified_exploratory",
    "verified_conservative",
)


def run_mode(mode,seeds):
    overall=[]
    task_values={}
    novel=[]
    recovery=[]
    transfer=[]
    forgetting=[]
    commits=[]
    schema_support=[]
    operator_support=[]
    diagnostics=[]

    for seed in seeds:
        arch=VerifiedOperatorArchitecture(mode)
        rows=[
            (p,t,arch.run(ep,True))
            for p,t,ep in curriculum(seed)
        ]

        overall.append(
            sum(int(r["correct"]) for _,_,r in rows)
            /len(rows)
        )

        for _,task,r in rows:
            task_values.setdefault(task,[]).append(
                int(r["correct"])
            )

        phase={p:r for p,_,r in rows}
        novel.append(int(phase["R2"]["correct"]))
        recovery.append(int(phase["R0"]["correct"]))
        forgetting.append(int(phase["R1"]["correct"]))
        transfer.append(int(phase["TRANSFER"]["correct"]))

        d=arch.diagnostics()
        diagnostics.append(d)

        commits.append(
            d["committed_beliefs"]/max(1,d["beliefs"])
        )
        schema_support.append(
            d["schema_events"]
        )
        operator_support.append(
            d["persistent_models"]["total_models"]
        )

    return {
        "mode":mode,
        "accuracy":sum(overall)/len(overall),
        "tasks":{
            k:sum(v)/len(v)
            for k,v in task_values.items()
        },
        "properties":{
            "novel_regime_induction":sum(novel)/len(novel),
            "regime_recovery":sum(recovery)/len(recovery),
            "representation_transfer":sum(transfer)/len(transfer),
            "forgetting_resistance":sum(forgetting)/len(forgetting),
            "posterior_commitment":sum(commits)/len(commits),
            "learned_schema_generation":min(
                1.0,
                sum(schema_support)/len(schema_support)
            ),
            "verified_operator_generation":min(
                1.0,
                sum(operator_support)/max(1,len(operator_support))
            ),
        },
        "diagnostics":{
            "avg_learned_schemas":sum(schema_support)/len(schema_support),
            "avg_persistent_models":sum(operator_support)/len(operator_support),
            "avg_interventions":sum(
                d["epistemic_interventions"] for d in diagnostics
            )/len(diagnostics),
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent/"results"/"v367.json",
    )
    a=p.parse_args()

    seeds=list(range(367,367+a.seeds))
    start=time.perf_counter()
    results=[
        run_mode(m,seeds)
        for m in MODES
    ]
    elapsed=time.perf_counter()-start

    payload={
        "version":"v367",
        "benchmark":"verified_operator_induction",
        "curriculum":[
            "R0","R1","R0","R2","R1",
            "M0","M1","M0",
            "TRANSFER","COMPOSE",
        ],
        "seeds":seeds,
        "wall_time_seconds":elapsed,
        "results":results,
    }

    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        json.dumps(payload,indent=2,default=str),
        encoding="utf-8",
    )

    print("=== V367 ===")
    print("wall=",round(elapsed,3))
    for r in results:
        print(
            r["mode"],
            "acc=",round(r["accuracy"],3),
            "tasks=",r["tasks"],
            "props=",r["properties"],
            "diag=",r["diagnostics"],
        )


if __name__=="__main__":
    main()

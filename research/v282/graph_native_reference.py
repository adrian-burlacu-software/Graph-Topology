
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from graph_simulator import BRANCH, REUSE, Config, Network


def build_cases(repeats:int,seed:int)->list[dict]:
    rng=random.Random(seed)
    return [
        {
            "case_id":f"h{h}_{i:04d}",
            "horizon":h,
            "label":REUSE if i%2==0 else BRANCH,
            "seed":rng.randrange(2**31),
        }
        for h in (1,2,3,4)
        for i in range(repeats)
    ]


def run_condition(cases:list[dict],mode:str)->list[dict]:
    if mode not in ("local_reward","delayed_reward"):
        raise ValueError(mode)

    reports=[]

    for h in (1,2,3,4):
        hcases=[c for c in cases if c["horizon"]==h]

        net=Network(
            Config(
                designer_learning_rate=0.05,
                vocabulary_learning_rate=0.05,
            )
        )

        roots={}
        for label in (REUSE,BRANCH):
            root=net.create_vocabulary_cell(
                f"MEM_{label}",None,0
            )
            roots[label]=root
            if label==REUSE:
                net.create_vocabulary_cell(
                    "TOKEN",root,1
                )

        for case in hcases:
            root=roots[case["label"]]

            for step in range(h-1):
                symbol=f"NEUTRAL_{step}"
                net.spike_designer(root,symbol)
                net.designer_signal(root,symbol)

            expected=case["label"]

            net._reset_designer_input()
            action=net.designer_signal(
                root,
                "TOKEN",
            )

            correct=action==expected
            reward=1.0 if correct else -1.0

            # Both paths use the native designer and its graph state.
            # "delayed_reward" means no reward signal precedes the terminal
            # query. Feedback is applied only after the decision.
            net.learn_designer(
                action,
                expected,
                reward,
            )

            reports.append(
                {
                    "case_id":case["case_id"],
                    "horizon":h,
                    "expected":expected,
                    "action":action,
                    "correct":bool(correct),
                }
            )

    return reports


def aggregate(reports:list[dict])->dict:
    out={}

    for h in (1,2,3,4):
        rows=[r for r in reports if r["horizon"]==h]

        def rate(label):
            subset=[r for r in rows if r["expected"]==label]
            return (
                sum(int(r["correct"]) for r in subset)
                /max(1,len(subset))
            )

        out[str(h)]={
            "cases":len(rows),
            "accuracy":(
                sum(int(r["correct"]) for r in rows)
                /max(1,len(rows))
            ),
            "reuse_accuracy":rate(REUSE),
            "branch_accuracy":rate(BRANCH),
        }

    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repeats",type=int,default=12)
    p.add_argument("--seed",type=int,default=282)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v282_graph_native_reference.json"
        ),
    )
    args=p.parse_args()

    cases=build_cases(
        args.repeats,
        args.seed,
    )

    print(
        "=== V282 GRAPH-NATIVE REFERENCE (NON-GATING) ===",
        flush=True,
    )
    print(
        "binary_action_space=REUSE/BRANCH",
        flush=True,
    )
    print(
        "diagnostic_only=True",
        flush=True,
    )

    payload={
        "version":"v282",
        "reference":"graph_native_non_gating",
        "diagnostic_only":True,
        "binary_action_space":[REUSE,BRANCH],
        "results":{},
    }

    for mode in ("local_reward","delayed_reward"):
        summary=aggregate(
            run_condition(cases,mode)
        )
        payload["results"][mode]=summary

        print(
            f"mode={mode}",
            flush=True,
        )

        for h in (1,2,3,4):
            m=summary[str(h)]
            print(
                f"  h={h} "
                f"accuracy={m['accuracy']:.3f} "
                f"reuse={m['reuse_accuracy']:.3f} "
                f"branch={m['branch_accuracy']:.3f}",
                flush=True,
            )

    args.output.parent.mkdir(
        parents=True,exist_ok=True
    )
    args.output.write_text(
        json.dumps(payload,indent=2),
        encoding="utf-8",
    )

    print(
        "graph_reference: INFORMATIONAL ONLY",
        flush=True,
    )
    print(
        "reference_saved:",
        args.output,
        flush=True,
    )


if __name__=="__main__":
    main()

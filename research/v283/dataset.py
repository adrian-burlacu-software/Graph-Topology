
from __future__ import annotations
import json
import random
from pathlib import Path
from state import ACTIONS,Edge,Node,State

TERMINAL_ACTIONS=(
    "REUSE","CREATE","BRANCH","INHIBIT","BIND","COMMIT",
)

def base_state():
    return State(
        [
            Node("alpha",0.9,2),
            Node("beta",0.7,1),
            Node("gamma",0.5,1),
            Node("delta",0.2,1),
        ],
        [
            Edge("alpha","IsA","beta",0.9),
            Edge("beta","RelatedTo","gamma",0.85),
            Edge("gamma","PartOf","delta",0.7),
        ],
    )

def instruction_state(action):
    s=base_state()
    s.add_node(f"instruction_{action}",2.0,7)
    return s

def action_record(action):
    return {
        "action":action,
        "source":None,
        "target":None,
        "relation":None,
    }

def make_case(case_id,horizon,states,actions,pair_id,remembered_action):
    return {
        "version":"v283",
        "case_id":case_id,
        "pair_id":pair_id,
        "task_type":"P1_memory",
        "horizon":horizon,
        "trajectory_states":[s.signature() for s in states],
        "trajectory_actions":[action_record(a) for a in actions],
        "goal":{
            "task":"memory",
            "source":"alpha",
            "target":"gamma",
            "relation":"RelatedTo",
            "depth":horizon,
        },
        "final_action":action_record(actions[-1]),
        # This is ground-truth only. It is used by the optional auxiliary
        # memory/readout loss and is never exposed to the normal model.
        "remembered_action":remembered_action,
    }

def make_dataset(pairs_per_horizon=24,seed=266):
    rng=random.Random(seed)
    rows=[]

    for horizon in (1,2,3,4):
        for i in range(pairs_per_horizon):
            a=TERMINAL_ACTIONS[(2*i)%len(TERMINAL_ACTIONS)]
            b=TERMINAL_ACTIONS[(2*i+1)%len(TERMINAL_ACTIONS)]
            if a==b:
                b=TERMINAL_ACTIONS[(2*i+2)%len(TERMINAL_ACTIONS)]

            visible=base_state()

            states_a=[instruction_state(a)]
            states_b=[instruction_state(b)]

            if horizon>1:
                states_a += [
                    visible.clone()
                    for _ in range(horizon-1)
                ]
                states_b += [
                    visible.clone()
                    for _ in range(horizon-1)
                ]

            actions_a=["NOOP"]*(horizon-1)+[a]
            actions_b=["NOOP"]*(horizon-1)+[b]

            pid=f"P1_{i:04d}_h{horizon}"

            rows.append(
                make_case(
                    f"{pid}_A",
                    horizon,
                    states_a,
                    actions_a,
                    pid,
                    a,
                )
            )
            rows.append(
                make_case(
                    f"{pid}_B",
                    horizon,
                    states_b,
                    actions_b,
                    pid,
                    b,
                )
            )

    rng.shuffle(rows)
    return rows

def save_dataset(rows,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")

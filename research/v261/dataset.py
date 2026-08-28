
from __future__ import annotations

import json
import random
from pathlib import Path

from state import ACTIONS,Edge,Node,State

TERMINAL_ACTIONS=(
    "REUSE",
    "CREATE",
    "BRANCH",
    "INHIBIT",
    "BIND",
    "COMMIT",
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


def token_state(token):
    s=base_state()
    s.add_node(token,2.0,7)
    return s


def dual_fact_state():
    s=base_state()
    s.add_node("fact_CREATE",2.0,7)
    s.add_node("fact_REUSE",1.9,7)
    return s


def action_record(action):
    return {
        "action":action,
        "source":None,
        "target":None,
        "relation":None,
    }


def goal(kind,horizon,**extra):
    g={
        "task":kind,
        "source":"alpha",
        "target":"gamma",
        "relation":"RelatedTo",
        "depth":horizon,
    }
    g.update(extra)
    return g


def make_case(
    case_id,
    task_type,
    horizon,
    states,
    actions,
    goal_payload,
    pair_id,
    progress_values,
):
    assert len(states)==horizon
    assert len(actions)==horizon
    assert len(progress_values)==horizon

    return {
        "version":"v261",
        "case_id":case_id,
        "pair_id":pair_id,
        "task_type":task_type,
        "horizon":horizon,
        "trajectory_states":[
            s.signature()
            for s in states
        ],
        "trajectory_actions":[
            action_record(a)
            for a in actions
        ],
        "trajectory_progress":[
            float(x)
            for x in progress_values
        ],
        "trace_program":list(actions),
        "goal":goal_payload,
        "final_action":action_record(actions[-1]),
    }


def p1_memory(i,h,a,b):
    visible=base_state()

    first_a=token_state(
        f"instruction_{a}"
    )
    first_b=token_state(
        f"instruction_{b}"
    )

    progress=[
        float(t)/(h-1)
        if h>1 else 0.0
        for t in range(h)
    ]

    pid=f"P1_{i:04d}_h{h}"

    return [
        make_case(
            f"{pid}_A",
            "P1_memory",
            h,
            [first_a]+[
                visible.clone()
                for _ in range(h-1)
            ],
            ["NOOP"]*(h-1)+[a],
            goal("memory",h),
            pid,
            progress,
        ),
        make_case(
            f"{pid}_B",
            "P1_memory",
            h,
            [first_b]+[
                visible.clone()
                for _ in range(h-1)
            ],
            ["NOOP"]*(h-1)+[b],
            goal("memory",h),
            pid,
            progress,
        ),
    ]


def p2_action_context(i,h,prev_a,prev_b,a,b):
    state=base_state()
    progress=[
        float(t)/(h-1)
        if h>1 else 0.0
        for t in range(h)
    ]

    if h==2:
        states_a=[state.clone(),state.clone()]
        states_b=[state.clone(),state.clone()]
        actions_a=[prev_a,a]
        actions_b=[prev_b,b]
    else:
        states_a=[state.clone() for _ in range(h)]
        states_b=[state.clone() for _ in range(h)]
        actions_a=["NOOP",prev_a,"NOOP",a]
        actions_b=["NOOP",prev_b,"NOOP",b]

    pid=f"P2_{i:04d}_h{h}"

    return [
        make_case(
            f"{pid}_A",
            "P2_action_context",
            h,states_a,actions_a,
            goal("action_context",h),
            pid,progress,
        ),
        make_case(
            f"{pid}_B",
            "P2_action_context",
            h,states_b,actions_b,
            goal("action_context",h),
            pid,progress,
        ),
    ]


def p3_attention(i,h,a,b):
    state=dual_fact_state()
    states=[state.clone() for _ in range(h)]
    progress=[
        float(t)/(h-1)
        if h>1 else 0.0
        for t in range(h)
    ]

    ga=goal(
        "attention",
        h,
        focus="CREATE",
    )
    gb=goal(
        "attention",
        h,
        focus="REUSE",
    )

    pid=f"P3_{i:04d}_h{h}"

    return [
        make_case(
            f"{pid}_A",
            "P3_attention",
            h,
            states,
            ["NOOP"]*(h-1)+[a],
            ga,
            pid,
            progress,
        ),
        make_case(
            f"{pid}_B",
            "P3_attention",
            h,
            states,
            ["NOOP"]*(h-1)+[b],
            gb,
            pid,
            progress,
        ),
    ]


def p4_progress(i,h,a,b):
    """
    Same graph + same goal + no differing action history.
    Only the actual progress/cursor input differs.
    """
    state=base_state()
    states_a=[state.clone() for _ in range(h)]
    states_b=[state.clone() for _ in range(h)]

    progress_a=[
        0.20 for _ in range(h)
    ]
    progress_b=[
        0.80 for _ in range(h)
    ]

    pid=f"P4_{i:04d}_h{h}"

    return [
        make_case(
            f"{pid}_A",
            "P4_progress",
            h,
            states_a,
            ["NOOP"]*(h-1)+[a],
            goal("progress",h),
            pid,
            progress_a,
        ),
        make_case(
            f"{pid}_B",
            "P4_progress",
            h,
            states_b,
            ["NOOP"]*(h-1)+[b],
            goal("progress",h),
            pid,
            progress_b,
        ),
    ]


def p5_action_progress(i,h,prev_a,prev_b,a,b):
    """
    Same explicit progress signal and same graph, but previous action differs.
    """
    state=base_state()
    states_a=[state.clone() for _ in range(h)]
    states_b=[state.clone() for _ in range(h)]

    progress=[
        0.50 for _ in range(h)
    ]

    if h==2:
        actions_a=[prev_a,a]
        actions_b=[prev_b,b]
    else:
        actions_a=["NOOP",prev_a,"NOOP",a]
        actions_b=["NOOP",prev_b,"NOOP",b]

    pid=f"P5_{i:04d}_h{h}"

    return [
        make_case(
            f"{pid}_A",
            "P5_action_progress",
            h,
            states_a,actions_a,
            goal("action_progress",h),
            pid,
            progress,
        ),
        make_case(
            f"{pid}_B",
            "P5_action_progress",
            h,
            states_b,actions_b,
            goal("action_progress",h),
            pid,
            progress,
        ),
    ]


def make_dataset(pairs_per_probe=12,seed=261):
    rng=random.Random(seed)
    rows=[]

    for h in (2,4):
        for i in range(pairs_per_probe):
            a=TERMINAL_ACTIONS[
                (2*i)%len(TERMINAL_ACTIONS)
            ]
            b=TERMINAL_ACTIONS[
                (2*i+1)%len(TERMINAL_ACTIONS)
            ]

            if a==b:
                b=TERMINAL_ACTIONS[
                    (2*i+2)%len(TERMINAL_ACTIONS)
                ]

            prev_a=TERMINAL_ACTIONS[
                i%len(TERMINAL_ACTIONS)
            ]
            prev_b=TERMINAL_ACTIONS[
                (i+1)%len(TERMINAL_ACTIONS)
            ]

            target_a=TERMINAL_ACTIONS[
                (i+2)%len(TERMINAL_ACTIONS)
            ]
            target_b=TERMINAL_ACTIONS[
                (i+3)%len(TERMINAL_ACTIONS)
            ]

            if target_a==target_b:
                target_b=TERMINAL_ACTIONS[
                    (i+4)%len(TERMINAL_ACTIONS)
                ]

            rows.extend(
                p1_memory(i,h,a,b)
            )
            rows.extend(
                p2_action_context(
                    i,h,
                    prev_a,prev_b,
                    target_a,target_b,
                )
            )
            rows.extend(
                p3_attention(i,h,a,b)
            )
            rows.extend(
                p4_progress(i,h,a,b)
            )
            rows.extend(
                p5_action_progress(
                    i,h,
                    prev_a,prev_b,
                    target_a,target_b,
                )
            )

    rng.shuffle(rows)
    return rows


def save_dataset(rows,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(row)+"\n"
            )

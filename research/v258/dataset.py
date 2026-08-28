
from __future__ import annotations

import json
import random
from pathlib import Path

from state import ACTIONS, Edge, Node, State


INSTRUCTION_TOKEN={
    "NOOP":"instruction_noop",
    "REUSE":"instruction_reuse",
    "CREATE":"instruction_create",
    "BRANCH":"instruction_branch",
    "INHIBIT":"instruction_inhibit",
    "BIND":"instruction_bind",
    "COMMIT":"instruction_commit",
}

COMMON_STATE=State(
    [
        Node("alpha",0.90,2),
        Node("beta",0.70,1),
        Node("gamma",0.50,1),
        Node("delta",0.20,1),
    ],
    [
        Edge("alpha","IsA","beta",0.90),
        Edge("beta","RelatedTo","gamma",0.85),
        Edge("gamma","PartOf","delta",0.70),
    ],
)


def clean_state():
    return COMMON_STATE.clone()


def instruction_state(action):
    s=COMMON_STATE.clone()
    s.add_node(
        INSTRUCTION_TOKEN[action],
        2.0,
        7,
    )
    return s


def goal(horizon):
    return {
        "source":"alpha",
        "target":"gamma",
        "relation":"RelatedTo",
        "depth":horizon,
    }


def action_record(action):
    return {
        "action":action,
        "source":None,
        "target":None,
        "relation":None,
    }


def sample_trace(rng,horizon,terminal):
    pool=[
        action
        for action in ACTIONS
        if action!=terminal
    ]
    return [
        *[
            rng.choice(pool)
            for _ in range(horizon-1)
        ],
        terminal,
    ]


def memory_pair(action_a,action_b,pair_index,horizon,rng):
    first_a=instruction_state(action_a)
    first_b=instruction_state(action_b)
    final=clean_state()

    # IMPORTANT: the randomized intermediate trace is identical between the
    # pair. Only the hidden instruction and required terminal answer differ.
    shared_trace=sample_trace(
        rng,
        horizon,
        action_a,
    )

    # Ensure the pair differs at the answer while the shared visible trace
    # remains identical after t=0.
    shared_trace[-1]=action_a

    trace_a=list(shared_trace)
    trace_b=list(shared_trace)
    trace_b[-1]=action_b

    final_sig=final.signature()
    pair_id=f"mem_{pair_index:05d}_h{horizon}"

    def row(label,action,first,trace):
        actions=[
            action_record(x)
            for x in trace
        ]

        return {
            "version":"v258",
            "case_id":f"v258_{pair_id}_{label}",
            "pair_id":pair_id,
            "task_type":"memory",
            "memory_action":action,
            "instruction_token":INSTRUCTION_TOKEN[action],
            "horizon":horizon,
            "initial_state":first.signature(),
            "goal":goal(horizon),
            "trajectory_states":[
                first.signature(),
                *[
                    final_sig
                    for _ in range(horizon-1)
                ],
            ],
            "trajectory_attention":[
                [INSTRUCTION_TOKEN[action]],
                *[[] for _ in range(horizon-1)],
            ],
            "trajectory_actions":actions,
            "final_action":actions[-1],
            "trace_program":trace,
        }

    return [
        row("A",action_a,first_a,trace_a),
        row("B",action_b,first_b,trace_b),
    ]


def progress_trace(rng,index,horizon,terminal):
    state=clean_state()
    sig=state.signature()

    trace=sample_trace(
        rng,
        horizon,
        terminal,
    )

    actions=[
        action_record(x)
        for x in trace
    ]

    return {
        "version":"v258",
        "case_id":f"v258_progress_{index:05d}_h{horizon}",
        "pair_id":None,
        "task_type":"progress",
        "memory_action":None,
        "instruction_token":None,
        "horizon":horizon,
        "initial_state":sig,
        "goal":goal(horizon),
        "trajectory_states":[
            sig for _ in range(horizon)
        ],
        "trajectory_attention":[
            [] for _ in range(horizon)
        ],
        "trajectory_actions":actions,
        "final_action":actions[-1],
        "trace_program":trace,
    }


def generate_dataset(samples=500,seed=253):
    if samples<56:
        raise ValueError("V258 requires at least 56 samples.")

    rng=random.Random(seed)
    actions=list(ACTIONS)

    memory_total=((samples//2)//2)*2
    progress_total=samples-memory_total

    memory_2=memory_total//2
    memory_4=memory_total-memory_2
    progress_2=progress_total//2
    progress_4=progress_total-progress_2

    if memory_2%2:
        memory_2-=1
        memory_4+=1

    if min(
        memory_2,
        memory_4,
        progress_2,
        progress_4,
    )<=0:
        raise ValueError("Cannot populate all task×horizon buckets.")

    rows=[]
    pair_index=0

    # Memory pairs: balanced action labels.
    pair_specs=[]
    total_pairs=(memory_2+memory_4)//2

    for i in range(total_pairs):
        a=actions[(2*i)%len(actions)]
        b=actions[(2*i+1)%len(actions)]
        if a==b:
            b=actions[(2*i+2)%len(actions)]
        pair_specs.append((a,b))

    rng.shuffle(pair_specs)

    cursor=0
    for count,horizon in ((memory_2,2),(memory_4,4)):
        for _ in range(count//2):
            a,b=pair_specs[cursor]
            rows.extend(
                memory_pair(
                    a,b,pair_index,horizon,rng
                )
            )
            pair_index+=1
            cursor+=1

    # Progress traces: balanced terminal classes.
    progress_specs=[]
    for i in range(progress_total):
        progress_specs.append(
            actions[i%len(actions)]
        )
    rng.shuffle(progress_specs)

    progress_index=0
    cursor=0

    for count,horizon in (
        (progress_2,2),
        (progress_4,4),
    ):
        for _ in range(count):
            terminal=progress_specs[cursor]
            rows.append(
                progress_trace(
                    rng,
                    progress_index,
                    horizon,
                    terminal,
                )
            )
            progress_index+=1
            cursor+=1

    if len(rows)!=samples:
        raise AssertionError(
            f"generated {len(rows)} rows, expected {samples}"
        )

    rng.shuffle(rows)
    return rows


def save_dataset(rows,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )+"\n"
            )

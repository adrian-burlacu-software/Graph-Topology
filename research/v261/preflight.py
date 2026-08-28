from __future__ import annotations

import argparse
import collections

from dataset import ACTIONS,TERMINAL_ACTIONS,make_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-probe",type=int,default=12)
    p.add_argument("--seed",type=int,default=261)
    p.add_argument("--device",default="cuda")
    p.add_argument("--hidden-size",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth",type=int,default=8)
    p.add_argument("--topk",type=int,default=5)
    args=p.parse_args()

    rows=make_dataset(
        args.pairs_per_probe,
        args.seed,
    )

    required={
        "P1_memory",
        "P2_action_context",
        "P3_attention",
        "P4_progress",
        "P5_action_progress",
    }

    counts=collections.Counter(
        r["task_type"] for r in rows
    )
    assert required <= set(counts)

    pairs=collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)

    assert pairs
    assert all(len(v)==2 for v in pairs.values())

    for a,b in pairs.values():
        task=a["task_type"]
        h=a["horizon"]

        assert h in (2,4)
        assert a["final_action"]["action"] in TERMINAL_ACTIONS
        assert b["final_action"]["action"] in TERMINAL_ACTIONS
        assert a["final_action"]["action"] != b["final_action"]["action"]

        if task=="P1_memory":
            assert a["goal"]==b["goal"]
            assert a["trajectory_states"][0] != b["trajectory_states"][0]
            assert a["trajectory_states"][1:] == b["trajectory_states"][1:]

        elif task=="P2_action_context":
            assert a["trajectory_states"] == b["trajectory_states"]
            idx=0 if h==2 else 1
            assert a["trajectory_actions"][idx] != b["trajectory_actions"][idx]

        elif task=="P3_attention":
            assert a["trajectory_states"] == b["trajectory_states"]
            assert a["goal"]["focus"] != b["goal"]["focus"]

        elif task=="P4_progress":
            assert a["trajectory_states"] == b["trajectory_states"]
            assert a["trajectory_progress"] != b["trajectory_progress"]
            assert a["goal"] == b["goal"]

        elif task=="P5_action_progress":
            assert a["trajectory_states"] == b["trajectory_states"]
            assert a["trajectory_progress"] == b["trajectory_progress"]
            idx=0 if h==2 else 1
            assert a["trajectory_actions"][idx] != b["trajectory_actions"][idx]

    for task in sorted(required):
        for h in (2,4):
            labels={
                r["final_action"]["action"]
                for r in rows
                if r["task_type"]==task and r["horizon"]==h
            }
            assert labels==set(TERMINAL_ACTIONS)

    from model import StateArchitectureModel
    from benchmark_core import ARCHITECTURES,prepare

    import torch
    data=prepare(rows)
    device=torch.device(
        args.device
        if args.device=="cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    def pick(task,pair_id=None):
        for x in data:
            if x["task_type"]==task and (
                pair_id is None or x["pair_id"]==pair_id
            ):
                return x
        raise AssertionError("probe case not found")

    probes={}
    for task in sorted(required):
        a=pick(task)
        probes[task]=(
            a,
            pick(task,a["pair_id"]),
        )

    print("=== V261 CAUSAL TASK PREFLIGHT ===",flush=True)
    print("dataset_size:",len(rows),flush=True)
    print("task_counts:",dict(counts),flush=True)
    print("terminal_label_balance: PASS",flush=True)
    print("structural_counterfactuals: PASS",flush=True)

    for index,name in enumerate(ARCHITECTURES):
        torch.manual_seed(args.seed+index)

        model=StateArchitectureModel(
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            topk=args.topk,
            **ARCHITECTURES[name],
        ).to(device)

        def zeros():
            return torch.zeros(
                (1,model.hidden_size),
                device=device,
            )

        p1,p1b=probes["P1_memory"]
        o1=model.cognitive_step(
            p1["states"][0],p1["goal"],zeros(),None,
            None,None,None,device,
            progress=p1["progress_values"][0],
        )
        o1b=model.cognitive_step(
            p1b["states"][0],p1b["goal"],zeros(),None,
            None,None,None,device,
            progress=p1b["progress_values"][0],
        )
        d1=float(
            (o1["action_logits"]-o1b["action_logits"]).abs().max().item()
        )

        p2,p2b=probes["P2_action_context"]
        i2=0 if p2["horizon"]==2 else 1
        prev2=p2["actions"][i2-1]
        prev2b=p2b["actions"][i2-1]
        o2=model.cognitive_step(
            p2["states"][i2],p2["goal"],zeros(),prev2,
            None,None,None,device,
            progress=p2["progress_values"][i2],
        )
        o2b=model.cognitive_step(
            p2b["states"][i2],p2b["goal"],zeros(),prev2b,
            None,None,None,device,
            progress=p2b["progress_values"][i2],
        )
        d2=float(
            (o2["action_logits"]-o2b["action_logits"]).abs().max().item()
        )

        p3,p3b=probes["P3_attention"]
        o3=model.cognitive_step(
            p3["states"][-1],p3["goal"],zeros(),None,
            None,None,None,device,
            progress=p3["progress_values"][-1],
        )
        o3b=model.cognitive_step(
            p3b["states"][-1],p3b["goal"],zeros(),None,
            None,None,None,device,
            progress=p3b["progress_values"][-1],
        )
        d3=float(
            (o3["action_logits"]-o3b["action_logits"]).abs().max().item()
        )

        p4,p4b=probes["P4_progress"]
        o4=model.cognitive_step(
            p4["states"][-1],p4["goal"],zeros(),None,
            None,None,None,device,
            progress=p4["progress_values"][-1],
        )
        o4b=model.cognitive_step(
            p4b["states"][-1],p4b["goal"],zeros(),None,
            None,None,None,device,
            progress=p4b["progress_values"][-1],
        )
        d4=float(
            (o4["action_logits"]-o4b["action_logits"]).abs().max().item()
        )

        p5,p5b=probes["P5_action_progress"]
        i5=0 if p5["horizon"]==2 else 1
        prev5=p5["actions"][i5-1]
        prev5b=p5b["actions"][i5-1]
        o5=model.cognitive_step(
            p5["states"][i5],p5["goal"],zeros(),prev5,
            None,None,None,device,
            progress=p5["progress_values"][i5],
        )
        o5b=model.cognitive_step(
            p5b["states"][i5],p5b["goal"],zeros(),prev5b,
            None,None,None,device,
            progress=p5["progress_values"][i5],
        )
        d5=float(
            (o5["action_logits"]-o5b["action_logits"]).abs().max().item()
        )

        print(
            f"{name:25s} "
            f"P1={d1:.3e} P2={d2:.3e} P3={d3:.3e} "
            f"P4={d4:.3e} P5={d5:.3e}",
            flush=True,
        )

        # All architectures should consume goal focus. Only latent_action should
        # be required to consume previous-action information for the P2/P5 claim.
        assert d3>1e-8, f"{name}: goal focus is not visible to decision"
        if name=="latent_action":
            assert d2>1e-8, f"{name}: previous action not wired into decision"
            assert d5>1e-8, f"{name}: previous action not wired into decision"

        if name in (
            "latent_workspace",
            "latent_action",
            "workspace_attention",
            "workspace_progress",
            "workspace_action_progress",
        ):
            assert d1>1e-8, f"{name}: latent state does not react to P1"

        if name in (
            "workspace_progress",
            "workspace_action_progress",
        ):
            assert d4>1e-8, f"{name}: progress input does not reach decision"

    print("P1 memory wiring: PASS",flush=True)
    print("P2 action wiring: PASS",flush=True)
    print("P3 goal-focus wiring: PASS",flush=True)
    print("P4 progress wiring: PASS",flush=True)
    print("P5 action+progress wiring: PASS",flush=True)
    print("TASK / CAUSAL INPUT PREFLIGHT: PASS",flush=True)


if __name__=="__main__":
    main()

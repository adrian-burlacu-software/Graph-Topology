
from __future__ import annotations

import copy
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset import ACTIONS


ARCHITECTURES={
    "baseline_graph":dict(
        state_mode="stateless",
        attention_workspace=False,
        explicit_progress=False,
        direct_goal_to_workspace=False,
    ),
    "latent_workspace":dict(
        state_mode="latent",
        attention_workspace=False,
        explicit_progress=False,
        direct_goal_to_workspace=False,
    ),
    "latent_action":dict(
        state_mode="latent_action",
        attention_workspace=False,
        explicit_progress=False,
        direct_goal_to_workspace=False,
    ),
    "workspace_attention":dict(
        state_mode="latent",
        attention_workspace=True,
        explicit_progress=False,
        direct_goal_to_workspace=True,
    ),
    "workspace_progress":dict(
        state_mode="latent",
        attention_workspace=True,
        explicit_progress=True,
        direct_goal_to_workspace=True,
    ),
    "workspace_action_progress":dict(
        state_mode="latent_action",
        attention_workspace=True,
        explicit_progress=True,
        direct_goal_to_workspace=True,
    ),
}


def state_from_json(payload):
    from state import Edge,Node,State
    return State(
        [
            Node(
                str(n["concept"]),
                float(n["activation"]),
                int(n["role"]),
                bool(n.get("persistent",False)),
            )
            for n in payload["nodes"]
        ],
        [
            Edge(
                str(e["source"]),
                str(e["relation"]),
                str(e["target"]),
                float(e["activation"]),
                bool(e.get("persistent",False)),
            )
            for e in payload["edges"]
        ],
    )


def prepare(rows):
    return [
        {
            "case_id":r["case_id"],
            "pair_id":r["pair_id"],
            "task_type":r["task_type"],
            "horizon":r["horizon"],
            "goal":r["goal"],
            "states":[state_from_json(x) for x in r["trajectory_states"]],
            "actions":[ACTIONS.index(x["action"]) for x in r["trajectory_actions"]],
            "progress_values":[float(x) for x in r["trajectory_progress"]],
        }
        for r in rows
    ]


def make_model(name,args,device,seed):
    from model import StateArchitectureModel
    torch.manual_seed(seed)
    return StateArchitectureModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=args.depth,
        topk=args.topk,
        **ARCHITECTURES[name],
    ).to(device)


def split_pairs(rows,seed,valid_fraction=0.25):
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        gid=row["pair_id"] or row["case_id"]
        groups[
            (row["task_type"],row["horizon"],gid)
        ].append(i)

    buckets=defaultdict(list)
    for (task,h,gid),ids in groups.items():
        buckets[(task,h)].append((gid,ids))

    rng=random.Random(seed)
    train=[]; valid=[]

    for bucket,items in sorted(buckets.items()):
        items=list(items)
        rng.shuffle(items)
        n=max(1,min(len(items)-1,int(round(len(items)*valid_fraction))))
        for _,ids in items[n:]:
            train.extend(ids)
        for _,ids in items[:n]:
            valid.extend(ids)

    assert not (set(train)&set(valid))

    return train,valid


def step(model,item,t,working,previous,device):
    progress_value=(
        item["progress_values"][t]
        if "progress_values" in item
        else t
    )
    return model.cognitive_step(
        item["states"][t],
        item["goal"],
        working,
        previous,
        None,None,None,
        device,
        progress=progress_value,
    )


def rollout(model,item,device):
    work=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    previous=None
    outputs=[]

    for t in range(item["horizon"]):
        out=step(
            model,
            item,
            t,
            work,
            previous,
            device,
        )
        outputs.append(out)
        work=out["next_working"]
        previous=item["actions"][t]

    return outputs


def loss_for_case(
    model,
    item,
    device,
    terminal_weight=1.0,
):
    outputs=rollout(
        model,
        item,
        device,
    )
    terminal=outputs[-1]
    target=torch.tensor(
        [item["actions"][-1]],
        dtype=torch.long,
        device=device,
    )
    return F.cross_entropy(
        terminal["action_logits"][None],
        target,
    )*terminal_weight


def train(
    model,
    train_data,
    device,
    epochs,
    batch_size,
    lr,
    terminal_weight,
):
    if not train_data:
        raise RuntimeError("Empty training data.")

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    initial=None
    best=float("inf")
    best_state=copy.deepcopy(model.state_dict())

    # Deterministic case ordering is important for reproducibility.
    train_data=sorted(
        train_data,
        key=lambda x:x["case_id"],
    )

    for epoch in range(epochs):
        model.train()

        if initial is None:
            with torch.no_grad():
                total=0.0
                for item in train_data:
                    total+=float(
                        loss_for_case(
                            model,item,device,terminal_weight
                        ).item()
                    )
                initial=total/len(train_data)

        optimizer.zero_grad(set_to_none=True)
        running=0.0
        batch_count=0

        for item in train_data:
            case_loss=loss_for_case(
                model,item,device,terminal_weight
            )
            running+=float(case_loss.detach().item())

            (
                case_loss/batch_size
            ).backward()

            del case_loss
            batch_count+=1

            if (
                batch_count==batch_size
                or item is train_data[-1]
            ):
                # Correct only the final partial batch.
                remainder=batch_count%batch_size
                if remainder:
                    scale=batch_size/remainder
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batch_count=0

        mean=running/len(train_data)

        if mean<best:
            best=mean
            best_state=copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    return {
        "initial_loss":initial,
        "best_loss":best,
        "loss_ratio":best/initial if initial else 0.0,
    }


@torch.no_grad()
def evaluate(model,data,indices,device):
    model.eval()

    by_task=defaultdict(
        lambda:{
            "cases":0,
            "terminal":0,
            "sequence":0,
        }
    )

    for i in indices:
        item=data[i]
        outputs=rollout(model,item,device)

        pred=[
            int(o["action_logits"].argmax().item())
            for o in outputs
        ]
        truth=item["actions"]

        g=by_task[item["task_type"]]
        g["cases"]+=1
        g["terminal"]+=int(pred[-1]==truth[-1])
        g["sequence"]+=int(pred==truth)

    return {
        task:{
            "cases":v["cases"],
            "terminal_accuracy":v["terminal"]/v["cases"],
            "sequence_exact":v["sequence"]/v["cases"],
        }
        for task,v in by_task.items()
    }


@torch.no_grad()
def paired_counterfactual_score(model,data,indices,device):
    """
    Scores pair discrimination rather than just independent accuracy.

    A successful pair requires:
      - both members' terminal labels are correct
      - the model predicts different terminal actions for the pair
    """
    pairs=defaultdict(list)

    for i in indices:
        item=data[i]
        if item["pair_id"]:
            pairs[item["pair_id"]].append(item)

    out=defaultdict(
        lambda:{
            "pairs":0,
            "both_correct":0,
            "discriminated":0,
        }
    )

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        pa=int(
            rollout(model,a,device)[-1]["action_logits"]
            .argmax().item()
        )
        pb=int(
            rollout(model,b,device)[-1]["action_logits"]
            .argmax().item()
        )

        assert a["task_type"]==b["task_type"]

        g=out[a["task_type"]]
        g["pairs"]+=1
        g["both_correct"]+=int(
            pa==a["actions"][-1]
            and pb==b["actions"][-1]
        )
        g["discriminated"]+=int(
            pa!=pb
        )

    for g in out.values():
        n=g["pairs"]
        g["both_correct_rate"]=g["both_correct"]/n if n else 0.0
        g["discrimination_rate"]=g["discriminated"]/n if n else 0.0

    return dict(out)


@torch.no_grad()
def memory_ablation(model,data,indices,device):
    normal=[]
    zero=[]

    for i in indices:
        item=data[i]

        if item["task_type"]!="P1_memory":
            continue

        work=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )

        first=step(
            model,item,0,work,None,device
        )

        normal_out=step(
            model,
            item,
            item["horizon"]-1,
            first["next_working"],
            item["actions"][0],
            device,
        )

        zero_out=step(
            model,
            item,
            item["horizon"]-1,
            torch.zeros_like(
                first["next_working"]
            ),
            item["actions"][0],
            device,
        )

        target=item["actions"][-1]

        normal.append(
            int(
                normal_out["action_logits"].argmax().item()
                ==target
            )
        )
        zero.append(
            int(
                zero_out["action_logits"].argmax().item()
                ==target
            )
        )

    if not normal:
        return {
            "normal_terminal_accuracy":None,
            "zero_workspace_terminal_accuracy":None,
            "drop":None,
            "cases":0,
        }

    n=sum(normal)/len(normal)
    z=sum(zero)/len(zero)

    return {
        "normal_terminal_accuracy":n,
        "zero_workspace_terminal_accuracy":z,
        "drop":n-z,
        "cases":len(normal),
    }


@torch.no_grad()
def actual_input_probes(model, data, device):
    """
    Probe the actual cognitive_step interface BEFORE training.

    We verify:
      P1: changing initial graph changes the first encoded decision input.
      P2: previous-action argument changes the next decision input.
      P3: goal focus changes the produced decision when the model's goal path
          consumes that field. If it does not, the benchmark fails explicitly.
      P4: stage graph token changes the decision input.
      P5: previous action changes the decision input while stage stays fixed.
    """
    out={}

    def initial():
        return torch.zeros(
            (1,model.hidden_size),
            device=device,
        )

    # P1: graph difference.
    a=data[0]
    b=data[1]
    wa=initial()
    wb=initial()

    xa=model.cognitive_step(
        a["states"][0],
        a["goal"],
        wa,
        None,
        None,None,None,
        device,
        progress=0,
    )
    xb=model.cognitive_step(
        b["states"][0],
        b["goal"],
        wb,
        None,
        None,None,None,
        device,
        progress=0,
    )

    out["P1_state_delta"] = float(
        (
            xa["action_logits"]
            -xb["action_logits"]
        ).abs().max().item()
    )

    # P2: use dedicated cases.
    p2_a=next(x for x in data if x["task_type"]=="P2_action_context")
    p2_b=next(x for x in data if x["task_type"]=="P2_action_context" and x["pair_id"]==p2_a["pair_id"] and x is not p2_a)

    ta=model.cognitive_step(
        p2_a["states"][1],
        p2_a["goal"],
        initial(),
        p2_a["actions"][0],
        None,None,None,
        device,
        progress=1,
    )
    tb=model.cognitive_step(
        p2_b["states"][1],
        p2_b["goal"],
        initial(),
        p2_b["actions"][0],
        None,None,None,
        device,
        progress=1,
    )

    out["P2_previous_action_delta"] = float(
        (
            ta["action_logits"]
            -tb["action_logits"]
        ).abs().max().item()
    )

    # P3: current goal differs only by focus.
    p3_a=next(x for x in data if x["task_type"]=="P3_attention")
    p3_b=next(x for x in data if x["task_type"]=="P3_attention" and x["pair_id"]==p3_a["pair_id"] and x is not p3_a)

    ga=model.cognitive_step(
        p3_a["states"][-1],
        p3_a["goal"],
        initial(),
        None,
        None,None,None,
        device,
        progress=p3_a["horizon"]-1,
    )
    gb=model.cognitive_step(
        p3_b["states"][-1],
        p3_b["goal"],
        initial(),
        None,
        None,None,None,
        device,
        progress=p3_b["horizon"]-1,
    )

    out["P3_goal_delta"] = float(
        (
            ga["action_logits"]
            -gb["action_logits"]
        ).abs().max().item()
    )

    # P4: stage graph difference.
    p4_a=next(x for x in data if x["task_type"]=="P4_progress")
    p4_b=next(x for x in data if x["task_type"]=="P4_progress" and x["pair_id"]==p4_a["pair_id"] and x is not p4_a)

    ca=model.cognitive_step(
        p4_a["states"][-1],
        p4_a["goal"],
        initial(),
        None,
        None,None,None,
        device,
        progress=p4_a["horizon"]-1,
    )
    cb=model.cognitive_step(
        p4_b["states"][-1],
        p4_b["goal"],
        initial(),
        None,
        None,None,None,
        device,
        progress=p4_b["horizon"]-1,
    )

    out["P4_stage_delta"] = float(
        (
            ca["action_logits"]
            -cb["action_logits"]
        ).abs().max().item()
    )

    # P5: stage same, previous action different.
    p5_a=next(x for x in data if x["task_type"]=="P5_action_progress")
    p5_b=next(x for x in data if x["task_type"]=="P5_action_progress" and x["pair_id"]==p5_a["pair_id"] and x is not p5_a)

    da=model.cognitive_step(
        p5_a["states"][1],
        p5_a["goal"],
        initial(),
        p5_a["actions"][0],
        None,None,None,
        device,
        progress=1,
    )
    db=model.cognitive_step(
        p5_b["states"][1],
        p5_b["goal"],
        initial(),
        p5_b["actions"][0],
        None,None,None,
        device,
        progress=1,
    )

    out["P5_previous_action_delta"] = float(
        (
            da["action_logits"]
            -db["action_logits"]
        ).abs().max().item()
    )

    return out

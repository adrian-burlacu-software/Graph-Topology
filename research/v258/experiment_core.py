
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
            "states":[
                state_from_json(x)
                for x in r["trajectory_states"]
            ],
            "actions":[
                ACTIONS.index(x["action"])
                for x in r["trajectory_actions"]
            ],
            "trace_program":list(r["trace_program"]),
        }
        for r in rows
    ]


def split_exact(rows,seed,valid_fraction=0.20):
    rng=random.Random(seed)
    groups=defaultdict(list)

    for i,row in enumerate(rows):
        gid=row["pair_id"] or row["case_id"]
        groups[
            (row["task_type"],row["horizon"],gid)
        ].append(i)

    buckets=defaultdict(list)
    for (task,horizon,gid),ids in groups.items():
        buckets[(task,horizon)].append((gid,ids))

    train=[]
    valid=[]

    for bucket in sorted(buckets):
        items=list(buckets[bucket])
        rng.shuffle(items)

        if len(items)<2:
            raise AssertionError(
                f"Bucket {bucket} has too few groups."
            )

        n=max(
            1,
            int(round(len(items)*valid_fraction)),
        )
        n=min(n,len(items)-1)

        for _,ids in items[n:]:
            train.extend(ids)
        for _,ids in items[:n]:
            valid.extend(ids)

    if set(train)&set(valid):
        raise AssertionError(
            "Train/validation overlap."
        )

    for task in ("memory","progress"):
        for horizon in (2,4):
            tr=sum(
                rows[i]["task_type"]==task
                and rows[i]["horizon"]==horizon
                for i in train
            )
            va=sum(
                rows[i]["task_type"]==task
                and rows[i]["horizon"]==horizon
                for i in valid
            )
            if tr==0 or va==0:
                raise AssertionError(
                    f"Missing {task} horizon={horizon}"
                )

    return train,valid


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


def model_step(model,item,t,working,previous,device):
    return model.cognitive_step(
        item["states"][t],
        item["goal"],
        working,
        previous,
        None,None,None,
        device,
        progress=t,
    )


def rollout(model,item,device):
    working=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    previous=None
    outputs=[]

    for t in range(item["horizon"]):
        out=model_step(
            model,
            item,
            t,
            working,
            previous,
            device,
        )
        outputs.append(out)
        working=out["next_working"]
        previous=item["actions"][t]

    return outputs


def sequence_loss(
    model,
    item,
    device,
    terminal_weight=2.0,
):
    outputs=rollout(model,item,device)
    total=torch.zeros((),device=device)

    for t,out in enumerate(outputs):
        target=torch.tensor(
            [item["actions"][t]],
            dtype=torch.long,
            device=device,
        )

        weight=(
            terminal_weight
            if t==item["horizon"]-1
            else 1.0
        )

        total=total+weight*F.cross_entropy(
            out["action_logits"][None],
            target,
        )

    return total/(item["horizon"]+terminal_weight-1.0)


def train_model(
    model,
    train_data,
    device,
    epochs,
    lr,
    terminal_weight=2.0,
    log_every=None,
    batch_size=8,
):
    """
    Memory-safe mini-batch training.

    Each case is backpropagated immediately, so its autograd graph can be
    released before the next case is evaluated. Gradients accumulate in model
    parameters for at most `batch_size` cases, then the optimizer updates.

    This preserves bounded peak memory while retaining many optimizer updates
    per epoch.
    """
    if not train_data:
        raise RuntimeError("Empty training set.")

    batch_size=max(
        1,
        min(batch_size,len(train_data)),
    )

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    @torch.no_grad()
    def measure_loss():
        model.eval()
        total=0.0

        for item in train_data:
            total += float(
                sequence_loss(
                    model,
                    item,
                    device,
                    terminal_weight,
                ).item()
            )

        model.train()
        return total/len(train_data)

    initial=measure_loss()
    best=initial
    best_state=copy.deepcopy(model.state_dict())

    updates_per_epoch=(
        len(train_data)+batch_size-1
    )//batch_size

    total_updates=epochs*updates_per_epoch

    for epoch in range(1,epochs+1):
        model.train()
        running=0.0
        optimizer.zero_grad(set_to_none=True)

        for case_index,item in enumerate(
            train_data,
            start=1,
        ):
            case_loss=sequence_loss(
                model,
                item,
                device,
                terminal_weight,
            )

            running += float(
                case_loss.detach().item()
            )

            (
                case_loss/batch_size
            ).backward()

            del case_loss

            is_batch_end=(
                case_index % batch_size == 0
                or case_index == len(train_data)
            )

            if is_batch_end:
                # Correct the final partial batch normalization.
                remainder=case_index % batch_size
                if remainder:
                    # Existing gradients were scaled by batch_size rather than
                    # remainder. Correct them before stepping.
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

        mean=running/len(train_data)

        if mean<best:
            best=mean
            best_state=copy.deepcopy(
                model.state_dict()
            )

        if log_every and (
            epoch==1
            or epoch%log_every==0
            or epoch==epochs
        ):
            print(
                f"epoch={epoch}/{epochs} "
                f"train_loss={mean:.5f} "
                f"updates={epoch*updates_per_epoch}/{total_updates}",
                flush=True,
            )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    return {
        "initial_loss":initial,
        "best_loss":best,
        "loss_ratio":best/initial if initial else 0.0,
        "epochs":epochs,
        "batch_size":batch_size,
        "updates_per_epoch":updates_per_epoch,
        "total_updates":total_updates,
    }


@torch.no_grad()
def evaluate(model,data,indices,device):
    model.eval()
    total=0
    terminal=0
    exact=0

    by_task=defaultdict(
        lambda:{
            "cases":0,
            "terminal":0,
            "exact":0,
        }
    )

    for i in indices:
        item=data[i]
        outputs=rollout(model,item,device)

        predicted=[
            int(
                out["action_logits"].argmax().item()
            )
            for out in outputs
        ]
        truth=item["actions"]

        is_terminal=predicted[-1]==truth[-1]
        is_exact=predicted==truth

        total+=1
        terminal+=int(is_terminal)
        exact+=int(is_exact)

        task=by_task[item["task_type"]]
        task["cases"]+=1
        task["terminal"]+=int(is_terminal)
        task["exact"]+=int(is_exact)

    return {
        "terminal_accuracy":terminal/total,
        "sequence_exact":exact/total,
        "task_terminal_accuracy":{
            k:v["terminal"]/v["cases"]
            for k,v in by_task.items()
        },
        "task_sequence_exact":{
            k:v["exact"]/v["cases"]
            for k,v in by_task.items()
        },
        "task_cases":{
            k:v["cases"]
            for k,v in by_task.items()
        },
    }


@torch.no_grad()
def memory_ablation(model,data,indices,device):
    normal=[]
    zero=[]

    for i in indices:
        item=data[i]
        if item["task_type"]!="memory":
            continue

        working=torch.zeros(
            (1,model.hidden_size),
            device=device,
        )

        first=model_step(
            model,item,0,working,None,device
        )

        normal_out=model_step(
            model,
            item,
            item["horizon"]-1,
            first["next_working"],
            item["actions"][0],
            device,
        )

        zero_out=model_step(
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

    n=sum(normal)/len(normal) if normal else None
    z=sum(zero)/len(zero) if zero else None

    return {
        "normal_terminal_accuracy":n,
        "zero_workspace_terminal_accuracy":z,
        "ablation_drop":n-z if n is not None and z is not None else None,
        "cases":len(normal),
    }


def run_shared_cell(
    name,
    args,
    rows,
    train_indices,
    eval_indices,
    device,
    seed,
    steps,
    log=False,
):
    data=prepare(rows)

    model=make_model(
        name,
        args,
        device,
        seed,
    )

    train_data=[
        data[i]
        for i in train_indices
    ]

    train=train_model(
        model,
        train_data,
        device,
        steps,
        args.lr,
        args.terminal_weight,
        args.log_every if log else None,
        args.batch_size,
    )

    evaluation=evaluate(
        model,
        data,
        eval_indices,
        device,
    )

    ablation=memory_ablation(
        model,
        data,
        eval_indices,
        device,
    )

    return {
        "architecture":name,
        "train":train,
        "evaluation":evaluation,
        "memory_ablation":ablation,
    }

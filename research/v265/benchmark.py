
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
            "actions":[
                ACTIONS.index(x["action"])
                for x in r["trajectory_actions"]
            ],
        }
        for r in rows
    ]


def split_pairs(rows,seed,valid_fraction=0.25):
    groups=defaultdict(list)

    for i,row in enumerate(rows):
        groups[row["pair_id"]].append(i)

    by_horizon=defaultdict(list)

    for pair_id,ids in groups.items():
        hs={rows[i]["horizon"] for i in ids}
        assert len(hs)==1
        by_horizon[next(iter(hs))].append((pair_id,ids))

    rng=random.Random(seed)
    train=[]
    valid=[]

    for horizon in (1,2,3,4):
        items=list(by_horizon[horizon])
        if len(items)<2:
            raise AssertionError(
                f"Too few pairs at horizon={horizon}"
            )
        rng.shuffle(items)

        n_valid=max(
            1,
            min(
                len(items)-1,
                int(round(
                    len(items)*valid_fraction
                )),
            ),
        )

        for _,ids in items[n_valid:]:
            train.extend(ids)

        for _,ids in items[:n_valid]:
            valid.extend(ids)

    assert not(set(train)&set(valid))
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


def step(model,item,t,working,previous,device):
    return model.cognitive_step(
        item["states"][t],
        item["goal"],
        working,
        previous,
        None,None,None,
        device,
        progress=t,
    )


def rollout_with_trace(model,item,device):
    working=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    previous=None

    outputs=[]
    work_before=[]
    work_after=[]

    for t in range(item["horizon"]):
        work_before.append(
            working.detach().clone()
        )

        out=step(
            model,
            item,
            t,
            working,
            previous,
            device,
        )

        outputs.append(out)

        working=out["next_working"]

        work_after.append(
            working.detach().clone()
        )

        previous=item["actions"][t]

    return outputs,work_before,work_after


def terminal_loss(model,item,device):
    out=rollout_with_trace(
        model,item,device
    )[0][-1]

    target=torch.tensor(
        [item["actions"][-1]],
        dtype=torch.long,
        device=device,
    )

    return F.cross_entropy(
        out["action_logits"][None],
        target,
    )


def train(model,train_data,device,epochs,batch_size,lr):
    if not train_data:
        raise RuntimeError("Empty training set.")

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    base=sorted(
        train_data,
        key=lambda x:x["case_id"],
    )

    @torch.no_grad()
    def measure():
        model.eval()
        total=0.0
        for item in base:
            total+=float(
                terminal_loss(
                    model,item,device
                ).item()
            )
        model.train()
        return total/len(base)

    initial=measure()
    best=initial
    best_state=copy.deepcopy(
        model.state_dict()
    )

    updates_per_epoch=(
        len(base)+batch_size-1
    )//batch_size

    for epoch in range(1,epochs+1):
        model.train()
        epoch_data=list(base)
        random.Random(
            seed := (epoch*100003)+len(base)
        ).shuffle(epoch_data)

        optimizer.zero_grad(set_to_none=True)
        running=0.0
        count=0

        for j,item in enumerate(epoch_data):
            loss=terminal_loss(
                model,item,device
            )
            running+=float(
                loss.detach().item()
            )
            count+=1

            (loss/batch_size).backward()
            del loss

            if count==batch_size or j==len(epoch_data)-1:
                if count<batch_size:
                    scale=batch_size/count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                count=0

        mean=running/len(epoch_data)

        if mean<best:
            best=mean
            best_state=copy.deepcopy(
                model.state_dict()
            )

        print(
            f"epoch={epoch}/{epochs} "
            f"loss={mean:.5f} "
            f"updates={updates_per_epoch}",
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
        "updates_per_epoch":updates_per_epoch,
        "total_updates":epochs*updates_per_epoch,
    }


@torch.no_grad()
def paired_metrics(model,data,indices,device):
    model.eval()

    pairs=defaultdict(list)
    for i in indices:
        pairs[data[i]["pair_id"]].append(data[i])

    total=0
    both=0
    discrim=0

    normal=[]
    zero_accuracy=[]

    per_step_work=defaultdict(list)
    per_step_logits=defaultdict(list)
    per_step_norm=defaultdict(list)

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        oa,wba,waa=rollout_with_trace(
            model,a,device
        )
        ob,wbb,wab=rollout_with_trace(
            model,b,device
        )

        pa=int(
            oa[-1]["action_logits"].argmax().item()
        )
        pb=int(
            ob[-1]["action_logits"].argmax().item()
        )

        ta=a["actions"][-1]
        tb=b["actions"][-1]

        total+=1
        both+=int(
            pa==ta and pb==tb
        )
        discrim+=int(
            pa!=pb
        )

        normal += [
            int(pa==ta),
            int(pb==tb),
        ]

        # State persistence diagnostic.
        for t,(x,y) in enumerate(
            zip(wba,wbb)
        ):
            wd=float(
                (x-y).norm().item()
            )
            per_step_work[t].append(wd)

            per_step_norm[t].append(
                0.5*(
                    float(x.norm().item())
                    +float(y.norm().item())
                )
            )

            ld=float(
                (
                    oa[t]["action_logits"]
                    -ob[t]["action_logits"]
                ).abs().max().item()
            )
            per_step_logits[t].append(ld)

        # Terminal zero-state ablation, separately per pair member.
        for item in (a,b):
            outputs,work_before,work_after=rollout_with_trace(
                model,item,device
            )

            terminal_normal=outputs[-1]

            zero_workspace=torch.zeros_like(
                work_before[-1]
            )

            zero_terminal=step(
                model,
                item,
                item["horizon"]-1,
                zero_workspace,
                item["actions"][-2]
                if item["horizon"]>1
                else None,
                device,
            )

            target=item["actions"][-1]

            normal.append(
                int(
                    terminal_normal["action_logits"]
                    .argmax().item()
                    ==target
                )
            )

            zero_accuracy.append(
                int(
                    zero_terminal["action_logits"]
                    .argmax().item()
                    ==target
                )
            )

    n_normal=sum(normal)/len(normal) if normal else 0.0
    n_zero=sum(zero_accuracy)/len(zero_accuracy) if zero_accuracy else 0.0

    step_diag={}

    for t in sorted(per_step_work):
        values=per_step_work[t]
        logs=per_step_logits[t]
        norms=per_step_norm[t]

        step_diag[str(t)]={
            "pairs_measured":len(values),
            "mean_working_delta":sum(values)/len(values),
            "max_working_delta":max(values),
            "mean_working_norm":sum(norms)/len(norms),
            "mean_action_logit_delta":sum(logs)/len(logs),
            "max_action_logit_delta":max(logs),
        }

    # Retention ratio: state-pair separation at the current timestep relative
    # to the separation immediately after the first instruction-bearing step.
    for t in step_diag:
        idx=int(t)
        if "1" in step_diag and idx>0:
            base=step_diag["1"]["mean_working_delta"]
            step_diag[t]["retention_vs_t1"]=(
                step_diag[t]["mean_working_delta"]/base
                if base>1e-12 else 0.0
            )

    return {
        "pairs":total,
        "both_correct_rate":both/total if total else 0.0,
        "discrimination_rate":discrim/total if total else 0.0,
        "normal_accuracy":n_normal,
        "zero_workspace_accuracy":n_zero,
        "ablation_drop":n_normal-n_zero,
        "step_diagnostics":step_diag,
    }

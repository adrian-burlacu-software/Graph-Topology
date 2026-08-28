
from __future__ import annotations

import copy
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset import ACTIONS,TERMINAL_ACTIONS


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
            Node(str(n["concept"]),float(n["activation"]),
                 int(n["role"]),bool(n.get("persistent",False)))
            for n in payload["nodes"]
        ],
        [
            Edge(str(e["source"]),str(e["relation"]),
                 str(e["target"]),float(e["activation"]),
                 bool(e.get("persistent",False)))
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
            "remembered_action":ACTIONS.index(r["remembered_action"]),
        }
        for r in rows
    ]


def split_pairs(rows,seed,valid_fraction=0.25):
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[row["pair_id"]].append(i)

    by_h=defaultdict(list)
    for pid,ids in groups.items():
        hs={rows[i]["horizon"] for i in ids}
        assert len(hs)==1
        by_h[next(iter(hs))].append((pid,ids))

    rng=random.Random(seed)
    train=[]
    valid=[]

    for h in (1,2,3,4):
        items=list(by_h[h])
        rng.shuffle(items)
        n=max(
            1,
            min(
                len(items)-1,
                int(round(len(items)*valid_fraction))
            )
        )
        for _,ids in items[n:]:
            train.extend(ids)
        for _,ids in items[:n]:
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


def rollout(model,item,device,mode="carry"):
    """
    mode:
      carry  = normal recurrent state
      freeze = preserve state produced at t=1
      zero   = wipe state before terminal
    """
    working=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    previous=None
    outputs=[]
    states_before=[]
    states_after=[]

    for t in range(item["horizon"]):
        states_before.append(
            working.detach().clone()
        )

        out=step(
            model,item,t,working,previous,device
        )
        outputs.append(out)

        working=out["next_working"]

        if mode=="freeze" and t>=0:
            # Capture the first meaningful memory write and keep it.
            if t==0:
                frozen=working.detach().clone()

            if t>=1:
                working=frozen.clone()

        previous=item["actions"][t]

        states_after.append(
            working.detach().clone()
        )

    if mode=="zero" and item["horizon"]>1:
        # Re-run terminal decision with zeroed state.
        out=step(
            model,
            item,
            item["horizon"]-1,
            torch.zeros_like(states_before[-1]),
            item["actions"][-2],
            device,
        )
        outputs[-1]=out
        states_before[-1]=torch.zeros_like(states_before[-1])
        states_after[-1]=out["next_working"].detach().clone()

    return outputs,states_before,states_after


def terminal_loss(model,item,device):
    out=rollout(
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


def memory_auxiliary_loss(model,item,device):
    """
    Diagnostic auxiliary objective.

    At every post-instruction timestep, force the latent workspace to expose the
    remembered action through the action controller itself. This is NOT part of
    the original architecture claim; it diagnoses whether the long-horizon
    failure is fundamentally a credit-assignment/readout problem.
    """
    if item["horizon"]<=1:
        return torch.zeros((),device=device)

    working=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    previous=None
    total=torch.zeros((),device=device)

    target=torch.tensor(
        [item["remembered_action"]],
        dtype=torch.long,
        device=device,
    )

    for t in range(item["horizon"]):
        out=step(
            model,item,t,working,previous,device
        )
        working=out["next_working"]
        previous=item["actions"][t]

        if t>=1 and model.state_mode!="stateless":
            total=total+F.cross_entropy(
                out["action_logits"][None],
                target,
            )

    return total/max(1,item["horizon"]-1)


def train(
    model,
    train_data,
    device,
    epochs,
    batch_size,
    lr,
    aux_memory_weight=0.0,
):
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    base=sorted(
        train_data,
        key=lambda x:x["case_id"],
    )

    def measure():
        model.eval()
        total=0.0
        with torch.no_grad():
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
            9176*epoch+len(base)
        ).shuffle(epoch_data)

        optimizer.zero_grad(
            set_to_none=True
        )
        running=0.0
        count=0

        for j,item in enumerate(epoch_data):
            terminal=terminal_loss(
                model,item,device
            )

            total=terminal

            if aux_memory_weight>0:
                total=total+(
                    aux_memory_weight
                    *memory_auxiliary_loss(
                        model,item,device
                    )
                )

            running+=float(
                terminal.detach().item()
            )

            count+=1

            (total/batch_size).backward()

            del terminal
            del total

            if (
                count==batch_size
                or j==len(epoch_data)-1
            ):
                if count<batch_size:
                    scale=batch_size/count
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),1.0
                )
                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
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
        "aux_memory_weight":aux_memory_weight,
    }


@torch.no_grad()
def intervention_metrics(model,data,indices,device):
    pairs=defaultdict(list)

    for i in indices:
        pairs[data[i]["pair_id"]].append(data[i])

    normal=[]
    zero=[]
    freeze=[]
    discrimination=[]

    step_diag=defaultdict(
        lambda:{
            "work_delta":[],
            "work_norm":[],
            "logit_delta":[],
            "freeze_logit_delta":[],
        }
    )

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        oa,wba,waa=rollout(
            model,a,device,"carry"
        )
        ob,wbb,wab=rollout(
            model,b,device,"carry"
        )

        fa=rollout(
            model,a,device,"freeze"
        )[0][-1]
        fb=rollout(
            model,b,device,"freeze"
        )[0][-1]

        za=rollout(
            model,a,device,"zero"
        )[0][-1]
        zb=rollout(
            model,b,device,"zero"
        )[0][-1]

        pa=int(
            oa[-1]["action_logits"].argmax().item()
        )
        pb=int(
            ob[-1]["action_logits"].argmax().item()
        )

        pfa=int(
            fa["action_logits"].argmax().item()
        )
        pfb=int(
            fb["action_logits"].argmax().item()
        )

        pza=int(
            za["action_logits"].argmax().item()
        )
        pzb=int(
            zb["action_logits"].argmax().item()
        )

        ta=a["actions"][-1]
        tb=b["actions"][-1]

        normal.extend([
            int(pa==ta),
            int(pb==tb),
        ])
        freeze.extend([
            int(pfa==ta),
            int(pfb==tb),
        ])
        zero.extend([
            int(pza==ta),
            int(pzb==tb),
        ])

        discrimination.append(
            int(pa!=pb)
        )

        # Pairwise state/decision separation over the entire trajectory.
        for t,(xa,xb) in enumerate(zip(wba,wbb)):
            wd=float(
                (xa-xb).norm().item()
            )
            wn=0.5*(
                float(xa.norm().item())
                +float(xb.norm().item())
            )

            ld=float(
                (
                    oa[t]["action_logits"]
                    -ob[t]["action_logits"]
                ).abs().max().item()
            )

            fld=float(
                (
                    (
                        fa["action_logits"]
                        -fb["action_logits"]
                    ).abs().max().item()
                )
            )

            step_diag[t]["work_delta"].append(wd)
            step_diag[t]["work_norm"].append(wn)
            step_diag[t]["logit_delta"].append(ld)
            step_diag[t]["freeze_logit_delta"].append(fld)

    n=len(normal)

    metrics={
        "pairs":len(normal)//2,
        "normal_accuracy":sum(normal)/n if n else 0.0,
        "freeze_accuracy":sum(freeze)/n if n else 0.0,
        "zero_workspace_accuracy":sum(zero)/n if n else 0.0,
        "normal_vs_zero_drop":(
            (sum(normal)-sum(zero))/n
            if n else 0.0
        ),
        "normal_vs_freeze_drop":(
            (sum(normal)-sum(freeze))/n
            if n else 0.0
        ),
        "pair_discrimination_rate":(
            sum(discrimination)/len(discrimination)
            if discrimination else 0.0
        ),
        "step_diagnostics":{},
    }

    for t in sorted(step_diag):
        d=step_diag[t]
        metrics["step_diagnostics"][str(t)]={
            "mean_work_delta":sum(d["work_delta"])/len(d["work_delta"]),
            "max_work_delta":max(d["work_delta"]),
            "mean_work_norm":sum(d["work_norm"])/len(d["work_norm"]),
            "mean_action_logit_delta":sum(d["logit_delta"])/len(d["logit_delta"]),
        }

    # Persistence ratios relative to the first write (t=1).
    if "1" in metrics["step_diagnostics"]:
        base=metrics["step_diagnostics"]["1"]["mean_work_delta"]
        base_logit=metrics["step_diagnostics"]["1"]["mean_action_logit_delta"]

        for key,d in metrics["step_diagnostics"].items():
            t=int(key)
            if t>=1:
                d["work_retention_vs_t1"]=(
                    d["mean_work_delta"]/base
                    if base>1e-12 else 0.0
                )
                d["logit_retention_vs_t1"]=(
                    d["mean_action_logit_delta"]/base_logit
                    if base_logit>1e-12 else 0.0
                )

    return metrics


@torch.no_grad()
def swap_state_decoder_probe(model,data,indices,device):
    """
    For each pair at terminal time:
      feed A's carried workspace with B's visible graph/goal, and vice versa.

    If swapped state swaps the decision, the controller reads workspace.
    """
    pairs=defaultdict(list)
    for i in indices:
        pairs[data[i]["pair_id"]].append(data[i])

    total=0
    swapped_correct=0
    swap_changed=0

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        oa,_,waa=rollout(
            model,a,device,"carry"
        )
        ob,_,wbb=rollout(
            model,b,device,"carry"
        )

        terminal_a=step(
            model,
            a,
            a["horizon"]-1,
            wbb[-1],
            a["actions"][-2] if a["horizon"]>1 else None,
            device,
        )
        terminal_b=step(
            model,
            b,
            b["horizon"]-1,
            waa[-1],
            b["actions"][-2] if b["horizon"]>1 else None,
            device,
        )

        pa=int(
            oa[-1]["action_logits"].argmax().item()
        )
        pb=int(
            ob[-1]["action_logits"].argmax().item()
        )
        psa=int(
            terminal_a["action_logits"].argmax().item()
        )
        psb=int(
            terminal_b["action_logits"].argmax().item()
        )

        swapped_correct+=int(
            psa==b["actions"][-1]
        )
        swapped_correct+=int(
            psb==a["actions"][-1]
        )

        swap_changed+=int(
            psa!=pa
        )
        swap_changed+=int(
            psb!=pb
        )
        total+=2

    return {
        "swapped_workspace_correct_rate":(
            swapped_correct/total
            if total else 0.0
        ),
        "workspace_swap_changes_decision_rate":(
            swap_changed/total
            if total else 0.0
        ),
        "cases":total,
    }

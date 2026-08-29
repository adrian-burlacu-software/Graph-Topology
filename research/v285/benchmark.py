
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
        read_mode="standard",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=False,
        training_mode="terminal",
    ),

    "protected_read_progress":dict(
        state_mode="gated_latent",
        attention_workspace=False,
        explicit_progress=True,
        direct_goal_to_workspace=False,
        read_mode="protected_read",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=False,
        training_mode="terminal",
    ),

    "eligibility_trace":dict(
        state_mode="gated_latent",
        attention_workspace=False,
        explicit_progress=True,
        direct_goal_to_workspace=False,
        read_mode="protected_read",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=False,
        training_mode="eligibility",
    ),

    "transition_supervision":dict(
        state_mode="gated_latent",
        attention_workspace=False,
        explicit_progress=True,
        direct_goal_to_workspace=False,
        read_mode="protected_read",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=False,
        training_mode="transition",
    ),

    # Strongest discovered credit combination + explicit decision bridge.
    "bridge_eligibility_transition":dict(
        state_mode="gated_latent",
        attention_workspace=False,
        explicit_progress=True,
        direct_goal_to_workspace=False,
        read_mode="protected_read",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=True,
        training_mode="eligibility_transition",
    ),

    # Ultimate readout stress test: explicit query + explicit bridge + best
    # discovered credit mechanism.
    "bridge_query_eligibility_transition":dict(
        state_mode="gated_latent",
        attention_workspace=False,
        explicit_progress=True,
        direct_goal_to_workspace=False,
        read_mode="protected_read",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=True,
        action_memory_binding=False,
        terminal_memory_bridge=True,
        training_mode="eligibility_transition",
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
    config=dict(ARCHITECTURES[name])
    config.pop("training_mode",None)

    return StateArchitectureModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=args.depth,
        topk=args.topk,
        **config,
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
    working=torch.zeros(
        (1,model.hidden_size),
        device=device,
    )
    slow_memory_state=torch.zeros_like(working)
    progress_memory_state=torch.zeros_like(working)
    previous=None

    outputs=[]
    states_before=[]
    states_after=[]
    slow_states=[]
    progress_states=[]

    for t in range(item["horizon"]):
        states_before.append(
            working.detach().clone()
        )

        out=model.cognitive_step(
            item["states"][t],
            item["goal"],
            working,
            previous,
            None,None,None,
            device,
            progress=t,
            slow_memory_state=slow_memory_state,
            progress_memory_state=progress_memory_state,
        )

        outputs.append(out)

        working=out["next_working"]
        slow_memory_state=out["next_slow_memory"]
        progress_memory_state=out["next_progress_memory"]

        if mode=="freeze":
            if t==0:
                frozen=working.detach().clone()
            else:
                working=frozen.clone()

        states_after.append(
            working.detach().clone()
        )
        slow_states.append(
            slow_memory_state.detach().clone()
        )
        progress_states.append(
            progress_memory_state.detach().clone()
        )

        previous=item["actions"][t]

    if mode=="zero" and item["horizon"]>1:
        outputs[-1]=model.cognitive_step(
            item["states"][-1],
            item["goal"],
            torch.zeros_like(working),
            item["actions"][-2],
            None,None,None,
            device,
            progress=item["horizon"]-1,
            slow_memory_state=slow_memory_state,
            progress_memory_state=progress_memory_state,
        )

    return (
        outputs,
        states_before,
        states_after,
        slow_states,
        progress_states,
    )



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


def deep_supervision_loss(model,item,device):
    if item["horizon"]<=1:
        return torch.zeros((),device=device)

    outputs,_,_,_,_=rollout(
        model,item,device,"carry"
    )

    target=torch.tensor(
        [item["actions"][-1]],
        dtype=torch.long,
        device=device,
    )

    values=[
        F.cross_entropy(
            outputs[t]["action_logits"][None],
            target,
        )
        for t in range(1,item["horizon"])
    ]

    return (
        torch.stack(values).mean()
        if values
        else torch.zeros((),device=device)
    )


def contrastive_memory_loss(model,pair_items,device,margin=0.75):
    by_pair=defaultdict(list)
    for item in pair_items:
        by_pair[item["pair_id"]].append(item)

    values=[]

    for pair in by_pair.values():
        if len(pair)!=2:
            continue

        a,b=pair

        _,_,wa,_,_=rollout(
            model,a,device,"carry"
        )
        _,_,wb,_,_=rollout(
            model,b,device,"carry"
        )

        for xa,xb in zip(wa[1:],wb[1:]):
            xa=F.normalize(
                xa,p=2,dim=-1,eps=1e-8
            )
            xb=F.normalize(
                xb,p=2,dim=-1,eps=1e-8
            )

            distance=(xa-xb).norm(dim=-1).mean()

            values.append(
                F.relu(margin-distance)
            )

    return (
        torch.stack(values).mean()
        if values
        else torch.zeros((),device=device)
    )


def training_objective(
    model,item,epoch_data,device,training_mode
):
    terminal=terminal_loss(
        model,item,device
    )

    if training_mode=="terminal":
        return terminal

    if training_mode=="deep_supervision":
        return terminal+deep_supervision_loss(
            model,item,device
        )

    if training_mode=="contrastive":
        return terminal+0.25*contrastive_memory_loss(
            model,epoch_data,device
        )

    raise ValueError(
        f"unknown training_mode={training_mode}"
    )


def final_action_target(item,device):
    return torch.tensor(
        [item["actions"][-1]],
        dtype=torch.long,
        device=device,
    )


def terminal_action_loss(model,item,device):
    outputs,_,_,_,_=rollout(
        model,item,device,"carry"
    )
    return F.cross_entropy(
        outputs[-1]["action_logits"][None],
        final_action_target(item,device),
    )


def eligibility_loss(
    model,
    item,
    device,
    decay=0.80,
):
    if item["horizon"]<=1:
        return torch.zeros((),device=device)

    outputs,_,_,_,_=rollout(
        model,item,device,"carry"
    )

    target=final_action_target(item,device)
    horizon=item["horizon"]
    values=[]

    for t in range(1,horizon):
        distance=(horizon-1)-t
        weight=decay**distance

        values.append(
            weight*F.cross_entropy(
                outputs[t]["action_logits"][None],
                target,
            )
        )

    return (
        torch.stack(values).mean()
        if values
        else torch.zeros((),device=device)
    )


def transition_supervision_loss(
    model,
    item,
    device,
):
    if item["horizon"]<=1:
        return torch.zeros((),device=device)

    outputs,before,after,_,_=rollout(
        model,item,device,"carry"
    )

    target=final_action_target(item,device)
    values=[]

    for t in range(1,item["horizon"]):
        delta=after[t]-before[t]

        norm=delta.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        normalized_delta=delta/norm

        action_loss=F.cross_entropy(
            outputs[t]["action_logits"][None],
            target,
        )

        transition_reg=(
            normalized_delta.norm(dim=-1)-1.0
        ).abs().mean()

        values.append(
            action_loss
            +0.05*transition_reg
        )

    return torch.stack(values).mean()


def bridge_supervision_loss(
    model,
    item,
    device,
):
    if not item.get("horizon",0)>1:
        return torch.zeros((),device=device)

    outputs,_,_,_,_=rollout(
        model,item,device,"carry"
    )

    target=final_action_target(item,device)

    values=[
        F.cross_entropy(
            outputs[t]["action_logits"][None],
            target,
        )
        for t in range(1,item["horizon"])
    ]

    return (
        torch.stack(values).mean()
        if values
        else torch.zeros((),device=device)
    )


def combined_credit_loss(model,item,device):
    return (
        eligibility_loss(
            model,item,device
        )
        + transition_supervision_loss(
            model,item,device
        )
    )


def bridge_credit_loss(model,item,device):
    return (
        combined_credit_loss(
            model,item,device
        )
        +0.50*bridge_supervision_loss(
            model,item,device
        )
    )


def training_objective(
    model,
    item,
    epoch_data,
    device,
    training_mode,
):
    terminal=terminal_action_loss(
        model,item,device
    )

    if training_mode=="terminal":
        return terminal

    if training_mode=="eligibility":
        return (
            terminal
            +eligibility_loss(
                model,item,device
            )
        )

    if training_mode=="transition":
        return (
            terminal
            +transition_supervision_loss(
                model,item,device
            )
        )

    if training_mode=="eligibility_transition":
        return (
            terminal
            +combined_credit_loss(
                model,item,device
            )
        )

    raise ValueError(
        f"unknown training_mode={training_mode}"
    )


def train(
    model,
    train_data,
    device,
    epochs,
    batch_size,
    lr,
    training_mode="terminal",
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
                    terminal_action_loss(
                        model,item,device
                    ).item()
                )

        model.train()

        return total/max(
            1,len(base)
        )

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
            total=training_objective(
                model,
                item,
                epoch_data,
                device,
                training_mode,
            )

            terminal=terminal_action_loss(
                model,
                item,
                device,
            )

            running+=float(
                terminal.detach().item()
            )

            count+=1

            (total/batch_size).backward()

            del total
            del terminal

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
                    model.parameters(),
                    1.0,
                )

                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

                count=0

        mean=running/max(
            1,len(epoch_data)
        )

        if mean<best:
            best=mean
            best_state=copy.deepcopy(
                model.state_dict()
            )

        print(
            f"epoch={epoch}/{epochs} "
            f"loss={mean:.5f} "
            f"updates={updates_per_epoch} "
            f"mode={training_mode}",
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
        "training_mode":training_mode,
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
            "slow_delta":[],
            "progress_delta":[],
        }
    )

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        oa,wba,waa,slow_a,progress_a=rollout(
            model,a,device,"carry"
        )
        ob,wbb,wab,slow_b,progress_b=rollout(
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
            sd=float(
                (slow_a[t]-slow_b[t]).norm().item()
            )
            pd=float(
                (progress_a[t]-progress_b[t]).norm().item()
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
            step_diag[t]["slow_delta"].append(sd)
            step_diag[t]["progress_delta"].append(pd)

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
            "mean_slow_memory_delta":sum(d["slow_delta"])/len(d["slow_delta"]),
            "mean_progress_memory_delta":sum(d["progress_delta"])/len(d["progress_delta"]),
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
def swap_state_decoder_probe(
    model,data,indices,device
):
    pairs=defaultdict(list)
    for i in indices:
        pairs[data[i]["pair_id"]].append(data[i])

    normal_correct=0
    swapped_correct=0
    changed=0
    directional=0
    total=0

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        a,b=pair

        oa,_,waa,_,_=rollout(
            model,a,device,"carry"
        )
        ob,_,wbb,_,_=rollout(
            model,b,device,"carry"
        )

        a_base=int(
            oa[-1]["action_logits"].argmax().item()
        )
        b_base=int(
            ob[-1]["action_logits"].argmax().item()
        )

        a_from_b=step(
            model,b,b["horizon"]-1,waa[-1],
            b["actions"][-2] if b["horizon"]>1 else None,
            device,
        )
        b_from_a=step(
            model,a,a["horizon"]-1,wbb[-1],
            a["actions"][-2] if a["horizon"]>1 else None,
            device,
        )

        pa=int(a_from_b["action_logits"].argmax().item())
        pb=int(b_from_a["action_logits"].argmax().item())

        ta=a["actions"][-1]
        tb=b["actions"][-1]

        normal_correct+=int(a_base==ta)+int(b_base==tb)
        swapped_correct+=int(pa==ta)+int(pb==tb)

        changed+=int(pa!=b_base)+int(pb!=a_base)
        directional+=int(pa==ta)+int(pb==tb)
        total+=2

    return {
        "normal_correct_rate":normal_correct/total if total else 0.0,
        "swapped_correct_rate":swapped_correct/total if total else 0.0,
        "workspace_swap_changes_decision_rate":changed/total if total else 0.0,
        "workspace_swap_directional_rate":directional/total if total else 0.0,
        "cases":total,
    }


@torch.no_grad()
def read_path_sensitivity(model,data,indices,device):
    pairs=defaultdict(list)
    for i in indices:
        pairs[data[i]["pair_id"]].append(data[i])

    half_deltas=[]
    zero_deltas=[]

    for pair in pairs.values():
        if len(pair)!=2:
            continue

        for item in pair:
            if item["horizon"]<=1:
                continue

            outputs,_,work_after,_,_=rollout(
                model,item,device,"carry"
            )

            t=item["horizon"]-1
            prev=item["actions"][-2]

            normal=step(
                model,item,t,work_after[-1],prev,device
            )
            half=step(
                model,item,t,0.5*work_after[-1],prev,device
            )
            zero=step(
                model,item,t,
                torch.zeros_like(work_after[-1]),
                prev,
                device,
            )

            half_deltas.append(
                float(
                    (
                        normal["action_logits"]
                        -half["action_logits"]
                    ).abs().max().item()
                )
            )
            zero_deltas.append(
                float(
                    (
                        normal["action_logits"]
                        -zero["action_logits"]
                    ).abs().max().item()
                )
            )

    return {
        "mean_normal_vs_half_logit_delta":sum(half_deltas)/len(half_deltas) if half_deltas else 0.0,
        "mean_normal_vs_zero_logit_delta":sum(zero_deltas)/len(zero_deltas) if zero_deltas else 0.0,
        "max_normal_vs_zero_logit_delta":max(zero_deltas) if zero_deltas else 0.0,
    }

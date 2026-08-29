
from __future__ import annotations

import torch
from torch import nn

from benchmark import (
    terminal_action_loss,
    eligibility_loss,
    transition_supervision_loss,
    bridge_supervision_loss,
    combined_credit_loss,
    bridge_credit_loss,
    training_objective,
)


class FakeModel(nn.Module):
    hidden_size=4
    state_mode="gated_latent"

    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(
            torch.tensor([1.0,0.0,0.0])
        )

    def cognitive_step(
        self,
        state,goal,working,previous_action_id,
        previous_source,previous_target,
        previous_relation,device,
        progress=0,
        slow_memory_state=None,
        progress_memory_state=None,
    ):
        if slow_memory_state is None:
            slow_memory_state=torch.zeros_like(working)

        if progress_memory_state is None:
            progress_memory_state=torch.zeros_like(working)

        logits=self.bias+working[0,:3]

        return {
            "action_logits":logits,
            "next_working":working+0.5,
            "next_slow_memory":slow_memory_state,
            "next_progress_memory":progress_memory_state,
        }


def main():
    model=FakeModel()
    device=torch.device("cpu")

    item={
        "pair_id":1,
        "case_id":0,
        "horizon":4,
        "actions":[0,1,1,2],
        "states":[{}, {}, {}, {}],
        "goal":{},
    }

    for fn in (
        terminal_action_loss,
        eligibility_loss,
        transition_supervision_loss,
        bridge_supervision_loss,
        combined_credit_loss,
        bridge_credit_loss,
    ):
        value=fn(model,item,device)
        assert torch.isfinite(value),fn.__name__
        assert value.requires_grad,fn.__name__
        value.backward()
        model.zero_grad(set_to_none=True)

    for mode in (
        "terminal",
        "eligibility",
        "transition",
        "eligibility_transition",
    ):
        value=training_objective(
            model,
            item,
            [item],
            device,
            mode,
        )
        assert torch.isfinite(value)
        assert value.requires_grad

    print(
        "V288 bridge regression: PASS"
    )


if __name__=="__main__":
    main()

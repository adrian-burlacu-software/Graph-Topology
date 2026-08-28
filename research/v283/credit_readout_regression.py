
from __future__ import annotations

import torch
from torch import nn

from benchmark import (
    deep_supervision_loss,
    contrastive_memory_loss,
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
        state,
        goal,
        working,
        previous_action_id,
        previous_source,
        previous_target,
        previous_relation,
        device,
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

    items=[
        {
            "pair_id":1,
            "case_id":0,
            "horizon":3,
            "actions":[0,1,2],
            "states":[{}, {}, {}],
            "goal":{},
        },
        {
            "pair_id":1,
            "case_id":1,
            "horizon":3,
            "actions":[1,1,2],
            "states":[{}, {}, {}],
            "goal":{},
        },
    ]

    for mode in (
        "terminal",
        "deep_supervision",
        "contrastive",
    ):
        value=training_objective(
            model,
            items[0],
            items,
            device,
            mode,
        )
        assert torch.isfinite(value)
        assert value.requires_grad
        value.backward()
        model.zero_grad(set_to_none=True)

    direct=deep_supervision_loss(
        model,
        items[0],
        device,
    )
    assert torch.isfinite(direct)

    contrast=contrastive_memory_loss(
        model,
        items,
        device,
    )
    assert torch.isfinite(contrast)

    print("V283 credit/readout regression: PASS")


if __name__=="__main__":
    main()

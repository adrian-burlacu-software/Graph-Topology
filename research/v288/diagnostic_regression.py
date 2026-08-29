
from __future__ import annotations

import torch
from benchmark import rollout


class FakeModel:
    hidden_size=4
    state_mode="gated_latent"
    read_mode="protected_read"

    def cognitive_step(
        self,
        state,goal,working,previous_action_id,
        previous_source,previous_target,previous_relation,
        device,progress=0,
        slow_memory_state=None,
        progress_memory_state=None,
    ):
        if slow_memory_state is None:
            slow_memory_state=torch.zeros_like(working)
        if progress_memory_state is None:
            progress_memory_state=torch.zeros_like(working)
        return {
            "action_logits":torch.tensor([1.0,0.0,0.0],device=device),
            "next_working":working+1,
            "next_slow_memory":slow_memory_state+2,
            "next_progress_memory":progress_memory_state+3,
        }


def main():
    model=FakeModel()
    item={
        "horizon":4,
        "states":[None]*4,
        "goal":{},
        "actions":[0,0,0,0],
    }
    outputs,work_before,work_after,slow_states,progress_states=rollout(
        model,item,torch.device("cpu"),"carry"
    )
    assert len(outputs)==4
    assert len(slow_states)==4
    assert len(progress_states)==4
    assert all(isinstance(x,torch.Tensor) for x in slow_states)
    assert all(isinstance(x,torch.Tensor) for x in progress_states)
    assert float(slow_states[-1][0,0])==8.0
    assert float(progress_states[-1][0,0])==12.0
    print("V288 diagnostic trace regression: PASS")


if __name__=="__main__":
    main()

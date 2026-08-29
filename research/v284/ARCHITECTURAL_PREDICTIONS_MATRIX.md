# Architectural Predictions Matrix — V260

## Research question

The earlier surveys mostly asked:

> Which architecture gets the highest score?

V260 instead asks:

> **What observable behavioral difference should each architectural decision cause, and can a minimal counterfactual task falsify that prediction?**

The benchmark is deliberately compact. It does not try to estimate every capability of
the system. It tests the causal claims that distinguish the architectures.

| ID | Architectural contrast | Minimal causal task | Prediction | Control / falsification |
|---|---|---|---|---|
| P1 | Stateless vs persistent latent workspace | Same final graph + goal, different hidden instruction history, different required answer | Persistent-state models outperform stateless because the answer is absent from the final observation | If stateless matches stateful **and** state ablation causes no loss, persistent state is not functionally necessary |
| P2 | Latent workspace vs latent + action | Same current graph + goal + memory, but previous action changes the correct next action | Action-conditioned state should separate cases that latent state alone conflates | If latent workspace matches latent-action, previous action is not adding useful state |
| P3 | Plain workspace vs attention | Two facts are stored; the goal changes which fact is relevant | Attention should improve counterfactual retrieval/selectivity, not merely raw accuracy | If attention gives no selective advantage over plain workspace, the attention mechanism is not buying anything |
| P4 | Generic latent state vs explicit progress | Same graph, goal, and action history summary except stage differs; correct action changes with stage | Explicit progress should improve stage discrimination, especially at horizon 4 | If generic latent state is equal or better, an explicit progress variable is unnecessary |
| P5 | Progress vs action + progress | Same stage but different immediately preceding operation changes the answer | Action+progress should outperform progress-only | If progress-only is equal, action identity is not needed for the task |
| P6 | State as useful mechanism vs state as decorative capacity | Counterfactual state ablation on exactly the same trained model | Removing state should selectively damage the state-dependent task, not unrelated controls | Zero/near-zero ablation drop falsifies the “state is being used” claim |
| P7 | Short vs longer temporal dependence | Same causal rule at horizon 2 and horizon 4 | A genuinely recurrent mechanism should degrade gracefully rather than collapse at the longer horizon | Collapse only at h=4 suggests state retention/update is the bottleneck |
| P8 | Architecture effect vs dataset shortcut | Balanced terminal labels + randomized intermediate traces + paired counterfactuals | No single constant action should solve the task | High constant-action ceiling or identical paired inputs with contradictory labels invalidates the task |

## Decision rule

A result is considered architecturally meaningful only when the predicted contrast appears
in the **specific counterfactual task assigned to that contrast**.

We do not require every architecture to win globally.

The benchmark is successful if it produces a pattern such as:

```text
P1  persistent state wins          ✓
P2  action-conditioned state wins  ✓
P3  attention wins selectively     ✓
P4  explicit progress wins         ✓
P5  action + progress wins         ✓
P6  state ablation is causal       ✓
```

A mixed pattern is also useful because it tells us which architectural additions are
not justified.

## What this benchmark intentionally does NOT claim

Passing P1 does not prove general intelligence.

Passing P4 does not prove a human-like notion of progress.

The benchmark only tests whether the proposed internal mechanisms are **functionally
necessary or useful for carefully constructed causal counterfactuals**.

## Experimental budget

V260 uses:

```text
6 architectures
5 core task contrasts
2 horizons
```

but each contrast is a small paired/counterfactual probe rather than a large random
task matrix.

The objective is to maximize **information per training minute**, not benchmark volume.

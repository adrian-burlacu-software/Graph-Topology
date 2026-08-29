
# V295 — Focused Credit Assignment Search

V294 identified a promising graph-native core:

```text
persistent memory
+
transforming dynamics
+
memory/state readout
+
binding/planning
```

Credit was the only slot that remained flat.

V295 therefore does NOT search the whole architecture again.

It freezes the core and searches the credit mechanism directly.

## Credit mechanisms

```text
none
immediate
eligibility
delayed_window
rule_flip
```

## Why this benchmark is different

Each sequence gets a hidden policy rule:

```text
answer = observation XOR cue XOR latent_rule
```

The latent rule changes between sequences.

It is not present in the visible query or persistent graph.

The initial observation disappears before decision time.

Therefore:

```text
episode 1
→ action
→ feedback
→ credit update
→ episode 2
→ action
...
```

is the only route to learning the sequence-specific rule.

## Measurements

```text
train accuracy
held-out evaluation accuracy
first-half accuracy
second-half accuracy
online learning gain
```

The critical metric is:

```text
second_half - first_half
```

A useful credit mechanism must alter the later action path after receiving delayed
feedback; otherwise it is rejected as a decorative metric.

## Smoke

```powershell
python .\research\v295\validation.py
python .\research\v295\search.py --train-seeds 3 --eval-seeds 3 --episodes 8
```

## Full

```powershell
python .\research\v295\search.py --train-seeds 12 --eval-seeds 12 --episodes 16
```

This is intentionally a focused experiment rather than another 1,000-way
search.

If one credit mechanism clearly separates here, freeze it and reinsert it into
the full graph architecture search.

Only after that should we expand credit itself into a larger algorithm family.

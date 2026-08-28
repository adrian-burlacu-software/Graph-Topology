
# V282 — Graph-Native Non-Gating Reference

V282 preserves the V281 transformer experiment and adds the original graph
designer as an independent diagnostic reference.

The reference uses the real graph-native `Network` implementation.

## Why binary?

The original designer's native learned action space is:

```text
REUSE
BRANCH
```

The V281 transformer task has six terminal actions, so forcing the graph
designer into the six-way label space would be a fake comparison.

Instead we measure an analogous graph-native memory task:

```text
encode REUSE/BRANCH in graph structure
        ↓
wait H-1 neutral graph steps
        ↓
query the stored structure
        ↓
native designer chooses REUSE/BRANCH
```

## Reference outputs

For each H1-H4:

```text
accuracy
reuse_accuracy
branch_accuracy
```

Two diagnostic modes are reported:

```text
local_reward
delayed_reward
```

The reference is **non-gating**. It cannot make transformer preflight pass or
fail.

## Run standalone

```powershell
python .\research\v282\graph_native_reference.py --repeats 12
```

## Run with transformer preflight

```powershell
python .\research\v282\preflight.py --pairs-per-horizon 24
```

The preflight prints the graph-native reference after transformer execution
checks.

## What to do with the result

```text
graph high local + high delayed
    → native graph state is inherently robust to delay

graph high local + low delayed
    → feedback / credit assignment matters

graph low both
    → underlying delayed binary problem is difficult natively too
```

Do not turn this into another architecture score.

## Further directions

The next high-value experiments are:

```text
eligibility traces
terminal credit assigned to memory-write events
temporal-difference/value targets
action-specific memory binding
graph event tokens
graph-constrained candidate selection
distilling graph transitions rather than final labels
```

The most interesting discriminator is whether an eligibility-trace style
credit mechanism lets the transformer approach the graph-native delayed
behavior without adding another memory subsystem.

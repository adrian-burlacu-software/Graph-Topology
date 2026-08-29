
# V287 — H4-Only Finalist Check

We have learned enough from H2.

V287 intentionally burns compute only on the stress case.

## Experiment

```text
eligibility_transition
bridge_query_eligibility_transition

× H4
× 3 seeds
= 6 cells
```

Each seed uses:

```text
30 paired H4 cases
parallelism=2
```

## Question

Does the apparent advantage of:

```text
bridge_query_eligibility_transition
```

survive three seeds when we look only at H4?

## Decision metrics

Priority:

```text
1. pair_discrimination_rate
2. workspace_swap_directional_rate
3. normal_accuracy
4. normal_vs_zero_drop
```

The H4 causal metrics matter more than a lucky accuracy score on six pairs.

## Commands

Preflight:

```powershell
python .\research\v287\preflight.py --pairs-per-horizon 30
```

Run:

```powershell
python .\research\v287\run_memory_diagnostic.py --pairs-per-horizon 30 --epochs 10 --batch-size 2 --parallelism 2 --seeds 287,288,289
```

Analyze:

```powershell
python .\research\v287\analyze_h4.py .\results\v287_summary.json
```

Exactly six worker cells.
No H2.
No H1/H3.
No auxiliary runs.

## Interpretation

```text
query+bridge wins H4 repeatedly and causally
    → promote it

query+bridge wins accuracy but not causal metrics
    → interesting but not decisive

eligibility_transition wins
    → keep the simpler architecture

both remain weak
    → move to graph/state dynamics rather than another readout trick
```

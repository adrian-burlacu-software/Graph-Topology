
# V281 — Temporal Credit / Readout Diagnostic

V280 showed that the latent memory can remain distinguishable while terminal
decision sensitivity decays across time.

V281 attacks the temporal credit/readout mechanism directly.

## Hypotheses

**Deep supervision**

Train the final action target at every post-instruction step.

```text
memory write
    ↓
intermediate action loss
    ↓
shorter credit path
```

**Terminal memory query**

Add a dedicated:

```text
workspace + goal + progress
        ↓
memory query head
        ↓
action logits
```

**Query + deep supervision**

Tests whether readout and temporal credit are complementary.

**Contrastive memory**

Keep the two counterfactual workspace trajectories separated with an explicit
pairwise margin.

## Exactly 24 cells

```text
baseline_graph
protected_read_progress
deep_supervision
terminal_query
query_deep_supervision
contrastive_memory

× H1/H2/H3/H4
= 24 cells
```

`baseline_graph` is the stateless control. `protected_read_progress` is the
current validated memory baseline.

## Run

```powershell
python .\research\v281\preflight.py --pairs-per-horizon 24
```

Then:

```powershell
python .\research\v281\credit_readout_regression.py
```

Then:

```powershell
python .\research\v281\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

There are no auxiliary duplicate runs.

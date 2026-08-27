# V224 — Parallel Recurrent Matrix — Attention Alignment Fix

V224 fixes the V223 failure where attention metrics still compared logits from
the model-generated current state against a cached oracle-sized target.

The single `aligned_attention_target()` function is now used by both the loss
and metrics.

Invariant:

```text
len(attention_logits)
==
len(attention_target)
==
len(current_state.nodes)
```

A preflight tests variable state sizes (1, 2, 3, 7, 20, 21, 22, 23 nodes)
before expensive training.

Matrix:

```text
                2 steps   4 steps   8 steps
teacher            ✓         ✓         ✓
scheduled          ✓         ✓         ✓
free               ✓         ✓         ✓
```

9 experiments, depth 8, parallelism 2.

Smoke test:

```powershell
python .\research\v224\run_all.py --samples 50 --epochs 2
```

Expected:

```text
attention_alignment_preflight: PASS
dataset_size: 50
manifest_samples: 50
manifest_valid_size: 7
```

Then:

```powershell
python .\research\v224\run_all.py
```

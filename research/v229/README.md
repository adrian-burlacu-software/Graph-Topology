
# V229 — State Architecture Matrix

Clean package: one `v229/` directory.

This is the state-architecture experiment after V224/V219. It compares:

```text
stateless
latent
latent_action
```

at:

```text
2, 4, 8 cognitive steps
```

with the default training regime:

```text
free
```

Use all regimes with:

```powershell
python .\research\v229\run_all.py --regimes teacher scheduled free
```

## Critical preflight

Before launching workers V229 runs the same weights on the same graph/goal
through all three state modes and prints:

```text
V229 ARCHITECTURE EFFECT PREFLIGHT
stateless_next_state_norm = 0.000000
latent_next_state_norm = ...
latent_vs_stateless = ...
latent_action_vs_latent = ...
architecture_effect_preflight: PASS
```

The run aborts if any of the architectural distinctions collapse.

## Matrix

Default:

```text
3 state modes × 1 regime × 3 horizons = 9 cells
```

Default parallelism:

```text
2
```

The 6-step experiment is deliberately removed.

## Run

Smoke:

```powershell
python .\research\v229\run_all.py --samples 50 --epochs 2
```

Full default:

```powershell
python .\research\v229\run_all.py
```

All regimes:

```powershell
python .\research\v229\run_all.py --regimes teacher scheduled free
```

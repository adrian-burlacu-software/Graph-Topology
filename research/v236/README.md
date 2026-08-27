
# V236 — Fixed-Horizon Architectural Decision Survey

This release fixes the V235 error where legacy short trajectories reached the
survey, producing:

```text
AssertionError: Training case v224_... does not support horizon=2
```

V236 contains a fresh dataset generator. Every trajectory has exactly:

```text
8 reasoning/cursor decisions + 1 terminal decision = 9 decisions
```

No `v224_` data is accepted.

Before training, the survey prints the actual generated trajectory lengths and
eligible cases:

```text
V236 DATASET HORIZON PREFLIGHT
trajectory_length_min=9 max=9 unique=[9]
horizon=2 eligible_train=... eligible_valid=...
horizon=8 eligible_train=... eligible_valid=...
dataset_horizon_preflight: PASS
```

The autonomous evaluator also refuses to evaluate a case unless the requested
horizon is completely present.

## Run

```powershell
python .\research\v236\run_survey.py --samples 50 --epochs 1
```

Then the full survey:

```powershell
python .\research\v236\run_survey.py
```

Battery → dataset horizon preflight → survey.

This is the Architectural Decision Survey, not another recurrence sweep.

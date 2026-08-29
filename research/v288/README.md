
# V288 — H4 Scheduler Fix

This is a bug-fix release for the H4-only finalist run.

The experiment remains:

```text
2 finalists
× H4
× 3 seeds
= 6 cells
parallelism=2
```

The V287 scheduler had two bookkeeping errors:

1. `pending` stored `(architecture, horizon, seed)` but the scheduler consumed
   it as a two-tuple.
2. Worker output filenames omitted the seed, allowing one seed to overwrite
   another.
3. Results were keyed without the seed, so final ordering could not resolve
   the full task list.

V288 fixes all three.

## Run

```powershell
python .\research\v288\preflight.py --pairs-per-horizon 30
```

```powershell
python .\research\v288\run_memory_diagnostic.py --pairs-per-horizon 30 --epochs 10 --batch-size 2 --parallelism 2 --seeds 287,288,289
```

The worker files now look like:

```text
eligibility_transition_h4_s287.json
eligibility_transition_h4_s288.json
eligibility_transition_h4_s289.json
...
```

and the result dictionary uses:

```text
(architecture, horizon, seed)
```

as its key.

Run the scheduler regression before CUDA:

```powershell
python .\research\v288\scheduler_regression.py
```

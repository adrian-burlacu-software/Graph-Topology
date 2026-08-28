
# V258 — Memory-Safe Mini-Batch Parallel Survey

V257 fixed the GPU crash by freeing each case's autograd graph after backward.

However, V257 accidentally made:

```text
1 epoch = 1 optimizer update
```

because gradients were accumulated over the entire dataset before stepping.

V258 keeps the memory fix but adds mini-batch optimizer updates:

```text
case 1 → backward
case 2 → backward
...
case 8 → backward
             ↓
       optimizer.step()

case 9 → backward
...
```

Default:

```text
batch_size = 8
```

So 300 samples and 2 epochs now produce roughly:

```text
38 optimizer updates / epoch
76 optimizer updates total
```

instead of 2 optimizer updates total.

## Survey parallelism

Actual survey parallelism remains:

```text
2
```

by default.

Two cells run concurrently; each cell is memory-safe internally.

## Recommended fast test

```powershell
python .\research\v258\run_survey.py --samples 100 --epochs 1 --batch-size 8 --parallelism 2 --skip-architecture-preflight
```

For the decisive run after timing the test:

```powershell
python .\research\v258\run_survey.py --samples 500 --epochs 3 --batch-size 8 --parallelism 2 --skip-architecture-preflight
```

Or the conservative version:

```powershell
python .\research\v258\run_survey.py --samples 300 --epochs 2 --batch-size 8 --parallelism 2 --skip-architecture-preflight
```

`--steps` remains a preflight compatibility option and does not control survey
training.

# V201 CUDA-safe fix

This fixes the V201 CUDA device-side assert by validating every index before
GPU indexing and by using a stable vocabulary hash.

It also:
- allocates the full relation vocabulary;
- correctly uses the masked-state graph for the masked-node objective;
- fixes repo-relative paths for execution from `research`;
- keeps the same self-supervised objectives.

Replace these three files in:

```text
research/v201_self_supervised_cognitive/
```

Then run from `research`:

```powershell
python .\v201_self_supervised_cognitive\train_self_supervised.py --epochs 6 --samples 12000 --batch-size 32
```

If another invalid index appears, the script now raises a descriptive Python
error before CUDA gets an asynchronous device-side assertion.

# PIQA

Physical Interaction: Question Answering. A goal and two ways to reach it,
one of which works; humans choose right about 95% of the time.

- Paper: Bisk, Zellers, Le Bras, Gao & Choi, *PIQA: Reasoning about
  Physical Commonsense in Natural Language*, AAAI 2020. arXiv:1911.11641
- Home: https://yonatanbisk.com/piqa/
- Downloaded from: https://yonatanbisk.com/piqa/data/ (`train.jsonl`,
  `train-labels.lst`, `valid.jsonl`, `valid-labels.lst`)
- Licence: **not checked** at download -- see the home page before any
  use beyond research. Kept under `data/`, which is not committed.
- Retrieved: 2026-09-26

## Files

    piqa/train.jsonl, train-labels.lst    16,113 goals, sol1, sol2; label 0|1
    piqa/valid.jsonl, valid-labels.lst     1,838 of the same: the dev split

The test split's labels are not public and are not fetched.

## Use

`research/v695/piqa.py` scores the graph on it. The train split is where
the scorer was looked at while it was built; the dev split is run once,
as the number. Nothing is trained on either.

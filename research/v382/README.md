
# V382 — BabyLM Accounting Fix

The prior run exposed a bookkeeping mismatch: the corpus contained 10,000
training records, but only 4,194 produced at least one grammar hypothesis.

V382 distinguishes:

```text
corpus_sentences_seen
grammar_observations
empty_hypothesis_sentences
```

A sentence that produces no grammar candidate is still counted as corpus input;
it is simply not counted as a grammar observation.

## Smoke

```powershell
python .\research\v380\run_babylm_grammar.py --smoke
```

## Accounting regression

```powershell
python .\research\v382\accounting_regression_test.py
```

## Real run

From the Graph-Topology root:

```powershell
python .\research\v382\run_babylm_grammar.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --progress-every 100 `
  --checkpoint-every 500
```

The result is still written to:

```text
.\results\v380_babylm_grammar.json
```

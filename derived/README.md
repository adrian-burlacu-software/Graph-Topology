# Derived artifacts

Files a model produced that the code then reads. **Tracked, unlike `data/`**,
and the distinction is not tidiness: `data/` holds corpora somebody else
published and this holds things this repository computed. Two of these change
what v687 answers, and a repository whose shipped behaviour depends on an
untracked file cannot be reproduced or reviewed.

| file | read by | changes answers? |
| --- | --- | --- |
| `distilled_norms.json` | `corpora.load_distilled` → `Profiles.corroboration` | **yes** — R19's evidence |
| `demoted_facts.json` | `corpora.load_demoted` → `reason.py` R30 | in principle; measured inert (§19) |
| `r19-calls.json` | `densify.observed` | no — it decides what to *ask* |
| `inherited.json` | nothing; a measurement | no |

Every one is optional. Delete any of them and the code treats it as "nothing
distilled", which is also how each ablation is run.

## Rebuilding, in order

The order matters: each step reads the one before it.

```
# 1. what R19 is actually asked          ~13 min, one engine, no GPU
python -m research.v688.record_r19 120

# 2. fill the cells it reads             ~15 min GPU (free from cache)
python -m research.v688.densify --build

# 3. demote class facts that are not     ~18 min GPU (free from cache)
#    claims about the class
python -m research.v688.prune --build

# 4. what inheritance actually uses      ~13 min, one engine, no GPU
python -m research.v688.record_inherited 120 crawl
```

Step 1 has to come first and is the one people forget: without
`r19-calls.json`, `densify.py` has no idea which cells are worth asking about
and falls back to every inheritable term on every ancestor — 824,431 cells
and eleven GPU-hours instead of fifteen minutes.

## What is *not* tracked, and what that costs

- **`llm/adjudications.json`** — 99,000 judgements, about **1.4 GPU-hours**.
  Every step above is free when this is warm and needs a GPU when it is not.
  Left untracked because it grows without bound, is specific to one model
  checkpoint, and git would keep every version of a 9.5 MB file. **Copy it
  rather than regenerate it** when moving machines.
- **`data/xcslb/comps_screened.jsonl`** — 7.3 MB, the screened benchmark
  (`AUDIT.md` §16). Rebuild with `python -m research.v688.screen --build`,
  free from a warm cache. `audit.py` warns and falls back to `--gold base`
  when it is missing, so the audit still runs without it — it just measures
  the wrong thing, which is the whole point of §16.
- **The stores in `data/*.sqlite`** — built by `research/v684`, far too large
  to track, and unchanged by any of this. Nothing here writes to a store.

## Provenance

These are SmolLM3-3B judgements, greedy-decoded with the `careful` system
prompt in `research/v688/teacher.py`. They are **not** elicited from people
and are deliberately kept out of `identify.stated`, which holds the norms
that are: `AUDIT.md` §17's first condition, and a test in
`research/v688/test_distil.py` asserts `identify.py` never merges them.

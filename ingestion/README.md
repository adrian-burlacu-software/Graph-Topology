# ingestion

Sources the store is built from, and the code that turns each one into facts.
Beside `research/` rather than inside it because a source outlives the version
of the reasoner that reads it.

```
python -m ingestion.genericskb --fetch          # download it (39 MB)
python -m ingestion.genericskb --report         # what the transform reaches
python -m ingestion.genericskb --sample 20      # see it on real rows
python -m ingestion.load --source genericskb    # into a copy of the store
```

## Rebuilding

There are two paths and they are for different things.

**`research/v687/build.py` assembles the store from nothing** — WordNet's
taxonomy, then Ascent++, then ConceptNet. It is the durable path, and it is
where a source belongs once it has earned its place.

**`ingestion/load.py` copies an existing store and adds one source**, which is
what an experiment needs. The only difference between the two files is the
source under test, so `audit.py --store` can compare them and the comparison
means something. It never writes to the store the server reads.

## Sources here

| module | what it is | rows | provenance |
| --- | --- | --- | --- |
| `genericskb.py` | AI2's GenericsKB-Best: generic sentences about kinds | 1,020,868 in, 570,717 facts out | `data/genericskb.SOURCE.md` |

Already in the store and not yet moved here: WordNet, ConceptNet and Ascent++,
all read by `research/v687/build.py`. Moving them is a refactor, not a fix.

## What a source module owes

Each one is expected to carry, in its own docstring and checkable from its
`--report`:

- **What it is worth, measured before it was built.** GenericsKB was measured
  against the gold positives the crawl misses *before* the adapter existed:
  95% concept coverage, 15% of misses reachable, 9% bare enough for R28. If a
  source cannot be shown to be worth adding, it should not be added.
- **What the transform drops, counted.** GenericsKB's subject strips cleanly
  from 89% of sentences; the other 11% are dropped rather than guessed at.
- **What it double-counts.** GenericsKB collects from ConceptNet and
  WordNet3.0, both already in the store under their own names, so 135k rows
  are skipped. A fact that corroborates itself under a second name is what
  R19 exists to prevent.
- **Its confidence distribution**, so `research/v688/confidence.py` can
  calibrate it by percentile rather than trusting a raw number. Two of the
  three existing sources write a constant on every row, which is why the raw
  number is never comparable across sources.

## Known limitations

- **`are animals` becomes `has_property`, not `is_a`.** A determiner
  separates `is a mammal` from `is thick`, and a bare plural noun has none.
  Fixing it needs a lexicon lookup per row; taxonomic features are already
  the best-covered category (50%, from WordNet), so it is recorded rather
  than fixed.
- **Terms are joined to their primary sense.** The same assumption Ascent++
  and ConceptNet rows carry, flagged `sense_assumed` on every row so it stays
  visible in the provenance. 206,777 GenericsKB terms have no lemma in the
  store at all and are dropped.

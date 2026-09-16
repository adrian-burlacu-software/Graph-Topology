# Buchanan — English semantic feature production norms

What people list when asked to describe a concept, with **production
frequency**: how many participants named that feature. Used by
`research/v687` as the largest and only frequency-carrying Appendix 3 corpus.

- Paper: Buchanan, Valentine & Maxwell, *English semantic feature production
  norms: An extended database of 4,436 concepts*, Behavior Research Methods
  (2019). doi:10.3758/s13428-019-01243-z, PMID 31044359
- Home: http://wordnorms.com/ (searchable portal)
- Repository: https://github.com/doomlab/Word-Norms-2
- Downloaded from:
  https://raw.githubusercontent.com/doomlab/Word-Norms-2/master/3%20parsed/top%20to%20final.csv
  (1.4 MB, the `3 parsed/` stage)
- Retrieved: 2026-09-16

## The filename trap

Upstream the file is **`3 parsed/top to final.csv`** — with spaces, in a
directory whose name also has a space. It is stored here as
`buchanan/top_to_final.csv`, renamed with underscores. Searching the repo or
the web for `top_to_final.csv` finds nothing; that rename is why this file
had no provenance for so long and could not be recovered when `data/` was
destroyed.

## Files

    buchanan/top_to_final.csv    where, cue, feature, translated,
                                 frequency_feature

`load_buchanan` reads four of those columns. `frequency_feature` must parse as
a digit or the row is skipped, so `min_frequency` is a real knob rather than a
guess. `translated` collapses morphological variants (`leaving` and `leave`
are one predicate); `root_forms=False` reads `feature` instead, which measures
spelling rather than compression.

## Numbers this repository depends on

Quoted by `research/v687/corpora.py` and asserted by
`research/v687/test_norms.py`:

    concepts (min_frequency=1, root forms)    3,722
    features                                 10,850
    test_buchanan_scales_the_norms_up        > 3,500 concepts

The paper's headline is 4,436 concepts; 3,722 is what survives this loader's
filtering, so the two numbers are not in conflict.

## Licence

See the repository. Cite Buchanan, Valentine & Maxwell (2019).

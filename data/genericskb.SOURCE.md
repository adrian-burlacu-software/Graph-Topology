# GenericsKB

**GenericsKB-Best**, 1,020,868 generic sentences over 128,828 terms.

- Paper: Bhakthavatsalam, Anastasiades & Clark, *GenericsKB: A Knowledge Base
  of Generic Statements* (AI2, 2020). arXiv:2005.00660
- Home: https://allenai.org/data/genericskb
- Downloaded from: https://huggingface.co/datasets/community-datasets/generics_kb
  (`generics_kb_best/train-00000-of-00001.parquet`, 39 MB)
- Licence: CC BY 4.0
- Retrieved: 2026-09-09

`python -m ingestion.genericskb --fetch` re-downloads it.

## Files

    generics_kb_best.parquet    the curated ~1M subset

Three larger configurations exist and are not here: `generics_kb` (3.4M),
`generics_kb_waterloo` (much larger, nine shards) and
`generics_kb_simplewiki`. Best is the curated one; the others trade quality
for size and there is no reason to take that trade before the curated set has
been shown to help.

## Columns

    term                  the subject concept       "aardvark", "aa battery"
    generic_sentence      a generic claim about it  "Aardvarks dig to get food."
    source                where GenericsKB got it   Waterloo | ARC | TupleKB |
                                                    WordNet3.0 | ConceptNet |
                                                    SimpleWikipedia
    score                 0.234 .. 1.000
    quantifier_frequency  usually empty
    quantifier_number     usually empty

## Numbers this repository depends on

Measured, and quoted by `ingestion/genericskb.py` and `research/v688/AUDIT.md`:

    rows                              1,020,868
    terms                               128,828
    subject strips cleanly                  89%
    score p5/p25/p50/p75/p90    0.256 0.369 0.575 0.838 1.000
    at exactly 1.000                        ~14%

    source breakdown    Waterloo         568,301
                        ARC              198,857
                        TupleKB          111,317
                        WordNet3.0        71,919   already in the store
                        ConceptNet        63,011   already in the store
                        SimpleWikipedia    7,463

## The trap

**A "generic" is not a bare claim.** The word describes the genericity of the
*subject* — "Dogs bark" rather than "that dog barked" — and says nothing
about qualifiers. `Most cheese is made from cow's milk, although some is made
from goat, sheep and water buffalo milk.` is a generic sentence.

This matters because the reason for adding GenericsKB was partly to relieve
R28, which refuses a qualified fact as support for a bare claim. Measured
against the gold positives the crawl misses: 15% have a GenericsKB sentence
covering the property, and only 9% are bare enough for R28 to take. It helps;
it does not help for the reason it first appeared to.

**63,011 of its rows come from ConceptNet and 71,919 from WordNet3.0**, both
already in the store under their own names. `ingestion/genericskb.py` drops
them, because a fact corroborating itself under a second name is what R19
exists to prevent.

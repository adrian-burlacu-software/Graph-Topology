# XCSLB — extended CSLB concept property norms

Human-elicited semantic properties: what people say when asked to describe a
concept. Used by `research/v687` as an Appendix 3 corpus.

## Where it came from

    https://github.com/kanishkamisra/comps/tree/main/data

Misra, Rayz & Ettinger (2023), *COMPS: Conceptual Minimal Pair Sentences for
testing Robust Property Knowledge and its Inheritance in Pre-trained Language
Models*, which extends the Cambridge CSLB norms (Devereux et al. 2014) from
638 concepts to 521 concepts × 3,643 properties with hand-added coverage.

The original CSLB site (`cslb.psychol.cam.ac.uk/propnorms`) returned 404 when
this was fetched on 2026-09-06 and was registration-gated before that. The
COMPS release is the practical route to the same data.

## Files pulled in

| file | what it is |
| --- | --- |
| `xcslb/comps_base.jsonl` | 14 MB. One row per (property, concept-that-has-it, concept-that-does-not). **This is what v687 reads.** |
| `xcslb/feature_lexicon.csv` | property → type (visual perceptual / functional / encyclopedic / taxonomic / other perceptual), plus its negation and plural |
| `xcslb/concept_senses.csv` | concept → WordNet sense key and category. This is the join to v687's synsets. |
| `xcslb/concept_matrix.txt` | 521 × 3,643 binary matrix. **Not used** — see below. |

## Why the matrix is not used

`concept_matrix.txt` ships without column labels, and its column order is not
the order of `feature_lexicon.csv`. Lining them up in file order gives
`budgie` the property `can be covered in lip balm`.

Recovering the labels by matching each column's concept set against the COMPS
pairs identified only **31.2%** of columns uniquely: hundreds of rare
properties are held by exactly one concept, so they share a column signature
and cannot be told apart.

The pair file needs none of that — every row names a concept and a property it
has — and carries 3,592 of the 3,643 properties (98.6%), missing only those
COMPS could not build a minimal pair from.

## What it does not supply

No production frequency (the original CSLB has it, this extension does not),
no per-property distinctiveness or cue validity, and no images. For cue
validity see McRae et al. 2005; for perceptual attributes see `awa2`.

## Licence

The COMPS repository is MIT-licensed. The underlying CSLB norms are the
property of the Centre for Speech, Language and the Brain, Cambridge, and are
for non-commercial research use. Cite both.

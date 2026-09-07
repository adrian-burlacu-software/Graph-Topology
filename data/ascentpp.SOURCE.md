# Ascent++ commonsense knowledge base

    https://www.mpi-inf.mpg.de/fileadmin/inf/d5/research/ascentpp/ascentpp.csv.tar.gz

65 MB gzipped, 421 MB as CSV. CC-BY 4.0. Nguyen, Razniewski, Weikum,
"Refined Commonsense Knowledge from Large-Scale Web Contents" (arXiv:2112.04596).
Project page: https://commonsense.scads.ai/ascentpp/

2,054,882 assertions over 8,071 primary concepts (18,888 subjects including
subgroups and aspects), extracted from the C4 web crawl.

## Columns

    primary_subject  the base concept, e.g. "dog"
    subject_type     primary | subgroup | aspect
    subject          the actual subject, e.g. "guide dog" for a subgroup
    head/relation/tail   ConceptNet's 19-relation schema
    subject/predicate/object   the OpenIE form
    saliency         how central the assertion is to the concept
    typicality       how generally true it is
    facets           JSON qualifiers: LOCATION, TEMPORAL, TRANSITIVE-OBJECT

## Why it is here

It supplies what v633's ConceptNet layer does not: a per-assertion graded
quality signal. Every gate tried in `research/v687/diagnose.py` failed for lack
of one.

Subjects join to what we already have: 99.9% of primary concepts are WordNet
lemmas, 100% are `en:` nodes in v633_full_semantic.sqlite. The relation
vocabulary is ConceptNet's, so `research/v687/normalize.py` applies with a
loader change only.

## What it does not supply

A taxonomy. Only 24,912 of the 2M assertions are `IsA`. It is a property layer,
not a replacement for WordNet's hierarchy.

Linked objects. Tails are free-text phrases -- 816,346 distinct, only 2.6% of
which are WordNet lemmas. Good for reading, poor for predicate sharing, which
is why it compresses badly under the V683 trie (1.89% reuse) even though its
assertions are of higher quality.

Ground truth. `typicality` is a model score, not a human label.

Subgroups are lexical specialisations ("guide dog", "snap trap"), not word
senses. They do not separate hammer-the-tool from hammer-the-throwing-implement.

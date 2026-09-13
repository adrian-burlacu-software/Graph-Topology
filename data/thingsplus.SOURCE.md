# THINGSplus

Norms and metadata for the THINGS database of 1,854 object concepts: property
ratings, real-world size, 53 higher-level categories and typicality.

- Papers: Hebart et al., *THINGS: A database of 1,854 object concepts and more
  than 26,000 naturalistic object images*, PLoS ONE 14(10) (2019); Stoinski,
  Perkuhn & Hebart, *THINGSplus: New norms and metadata for the THINGS database
  of 1854 object concepts and 26,107 natural object images*, Behavior Research
  Methods 56, 1583-1603 (2024).
- Home: https://osf.io/jum2f/ (THINGS object concept and object image database)
- Licence: CC BY for the features and metadata (`thingsplus/LICENSE.txt`).
  The images are not downloaded.
- Retrieved: 2026-09-13

## Files

    thingsplus/concepts-metadata_things.tsv   word, uniqueID, WordNet 3.0 synset
                                              and gloss, categories   osf.io/download/5uvc2
    thingsplus/property-ratings.tsv           per-object means        osf.io/download/7cz69
    thingsplus/category53_long-format.tsv     category membership     osf.io/download/vehr3
    thingsplus/typicality53_mean-ratings.tsv  typicality per member   osf.io/download/qj7ec
    thingsplus/object-level_description.txt   what every column is    osf.io/download/kdyq6
    thingsplus/category-level_description.txt                         osf.io/download/bpxye

## What was asked

About 40 raters per object, on a 1-7 scale, with one image: manmade, precious,
"something that lives", heavy, natural, "something that moves", how easy to
grasp, to hold, to move, how pleasant, how arousing. Size is a two-step task on
a 520-unit scale anchored by a grain of sand, a marble, a chicken egg, a
grapefruit, a microwave oven, a washing machine, a king-size bed, an ambulance
and an aircraft carrier, about 45 raters per object. Where those objects are
themselves rated: grain 109, marble 142, egg 181, grapefruit 217, microwave
260, washing machine 299, bed 329, ambulance 379, aircraft carrier 421.

## Numbers this repository depends on

Measured, and quoted by `research/v687/rated.py`:

    objects                                   1,854
    joined to the store by gloss              1,790   64 have non-WordNet glosses
    lives <= 2.5 and outside organism.n.01    1,392 of 1,569   (1 inside: honeypot)
    lives >= 5.5 and under organism.n.01        214 of 221     (51 outside: leaves, fruit)
    manmade <= 2.5 and outside artifact.n.01    380 of 681     (6 inside: pearl, stick, ruby)
    AwA2 big=1 animals, size                  281-376 (median 321), n=20
    AwA2 big=0 animals, size                  186-295 (median 241), n=13

## The trap

**A word is not an object.** THINGS rates `mouse1` (the animal, lives 6.8) and
`mouse2` (the device, lives 1.6) under the one word `mouse`, and `bat1`/`bat2`
likewise; its `chicken` is the meat. The store's first sense of `mouse` is the
device. The join therefore goes through the gloss and never through the word,
and `Ratings.settle` says nothing when rated readings of a word disagree.

**Categories are not denials.** Membership was assigned, not scored against
every category, so a whale not being in any `fish` category (there is none) is
not a no. Only membership is read.

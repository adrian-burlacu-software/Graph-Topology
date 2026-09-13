# NEWTON

Physical attributes of household objects, annotated by people: the
"confident" track of NEWTON's benchmark.

- Paper: Wang, Duan, Fox & Srinivasa, *NEWTON: Are Large Language Models
  Capable of Physical Reasoning?*, Findings of EMNLP 2023. arXiv:2310.07018
- Home: https://newtonreasoning.github.io/
- Downloaded from: https://github.com/NewtonReasoning/Newton
  (`data/confident_questions.csv`, 847 KB)
- Licence: **none stated** in the repository at the time of download. Kept
  under `data/`, which is not committed.
- Retrieved: 2026-09-13

## Files

    newton/confident_questions.csv    2,891 rows: attribute, object, question,
                                      the three options, majority, agreement
    newton/lvis_v1_categories.json    LVIS v1's 1,203 categories: name,
                                      synonyms, WordNet 3.0 synset and gloss.
                                      Extracted from detectron2's
                                      `detectron2/data/datasets/lvis_v1_categories.py`
    newton/LICENSE.detectron2.txt     Apache 2.0, for the category list

## What was asked

777 objects from Objaverse, YCB, Google Scanned Objects and Amazon Berkeley
Objects, each asked about some of eight attributes with three options (Low,
Moderate, High). This file is the confident track: only rows whose majority was
Low or High, with 75% or 100% agreement.

    softness 517   sharpness 503   elasticity 493   malleability 482
    brittleness 345   surface smoothness 220   stiffness 203   surface hardness 128

## Numbers this repository depends on

    objects                                    777
    joined by LVIS gloss                       484
    joined to the one physical sense of the word, or the first when it is one   261
    dropped: no single physical sense           32
    objects XCSLB also covers                  140; not one XCSLB lists as sharp,
                                               flexible or rigid is voted Low on it

## The trap

The objects are household objects, so `mouse` is the device and `orange` the
fruit. A word is joined to the sense of it that is an artifact, a food, a plant
part, a natural object or a substance, never to its first sense blindly.

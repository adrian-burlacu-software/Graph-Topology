# AwA2 — Animals with Attributes 2, class/attribute table only

50 animal classes scored on 85 attributes. Used by `research/v686` as the
dense Appendix 3 corpus and as the source of fine-grained identification
("what kind of dog has spots").

## Where it came from

    https://cvml.ista.ac.at/AwA2/AwA2-base.zip      (32 KB)

Xian, Lampert, Schiele & Akata (2019), *Zero-Shot Learning — A Comprehensive
Evaluation of the Good, the Bad and the Ugly*. The attribute matrix descends
from Osherson's classical animal/feature matrix.

Only the **base** archive is pulled in: it is 32 KB and holds the class,
attribute and matrix files. The images (13 GB) and the ResNet features (13 GB)
are not needed — nothing here looks at a pixel.

## Files

| file | what it is |
| --- | --- |
| `awa2/Animals_with_Attributes2/classes.txt` | 50 class names, `1  antelope` |
| `awa2/Animals_with_Attributes2/predicates.txt` | 85 attribute names, `1  black` |
| `awa2/Animals_with_Attributes2/predicate-matrix-binary.txt` | 50 × 85 of 0/1 |
| `awa2/Animals_with_Attributes2/predicate-matrix-continuous.txt` | the same, graded |

`load_awa2()` reads the binary matrix. A zero contributes no predicate: the
trie's alphabet is what an individual *carries*, so absence is absence rather
than a negative predicate.

## Why it is here alongside XCSLB

It is the opposite shape. XCSLB is 521 concepts over 3,592 sparsely-shared
properties, median 23 each; AwA2 is 50 classes over 85 attributes that every
class is scored on. Dense and closed is the case Appendix 3 assumes, and it
compresses far better — 43.5% against 12.5%. Having both is what shows the
compression tracks the shape of the data rather than the dataset.

It also carries the breed-level distinctions XCSLB lacks: `dalmatian`,
`collie`, `german shepherd`, `chihuahua`, `persian cat`, `siamese cat`.
XCSLB has `dog` and `cat` and stops there.

## Licence

Free for research use; the archive ships its own licence files. Cite Xian et
al. 2019.

"""Feature norms as Appendix 3 corpora: individuals carrying predicate sets.

V683 measured the paper's trie on a scraped ontology and got a real but modest
result, because a scraped ontology is not what Appendix 3 describes. Its
predicates are whatever a crawler happened to record, so two dogs rarely carry
the same predicate set and there is little prefix to share.

These two datasets are what the paper assumes. Both are *elicited*: a fixed
question was put to people about every concept, so the predicate vocabulary is
closed and every individual is described over the same alphabet.

    XCSLB   521 concepts x 3,643 properties, from the Cambridge CSLB property
            norms extended by Misra et al. Human-produced features -- `has
            keys`, `is a musical instrument`, `can make music` -- typed as
            visual perceptual, functional, encyclopedic or taxonomic, and each
            with its own negation. 530 concepts carry a WordNet sense key,
            which is the join v684 and v685 spent their time guessing at.

    AwA2    50 animal classes x 85 attributes, the Osherson matrix. Much
            smaller and completely dense: every class is scored on every
            attribute, so it is the cleanest possible test of the mechanism
            and the one place where the exhaustive `optimal` ordering is
            almost reachable.

Nothing here is rebuilt or re-derived: both are read exactly as published.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..v683.substrate import Corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
XCSLB_DIR = REPOSITORY_ROOT / "data" / "xcslb"
AWA2_DIR = REPOSITORY_ROOT / "data" / "awa2" / "Animals_with_Attributes2"

#: The feature taxonomy XCSLB ships. `visual perceptual` is the one that
#: matters for identifying a thing by looking at it.
FEATURE_TYPES = ("visual perceptual", "other perceptual", "functional",
                 "encyclopedic", "taxonomic")


def _xcslb_features() -> list[dict[str, str]]:
    with (XCSLB_DIR / "feature_lexicon.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_types() -> dict[str, str]:
    """property -> its type, for slicing the corpus by kind of knowledge."""
    return {row["feature"]: row["feature_type"] for row in _xcslb_features()}


def negations() -> dict[str, str]:
    """property -> how a speaker denies it. XCSLB gives this for every one."""
    return {row["feature"]: row["negation"] for row in _xcslb_features()}


def senses() -> dict[str, str]:
    """concept -> WordNet sense key, and concept -> category."""
    with (XCSLB_DIR / "concept_senses.csv").open(encoding="utf-8") as handle:
        return {row["concept"]: row["sensekey"] for row in csv.DictReader(handle)}


def categories() -> dict[str, str]:
    with (XCSLB_DIR / "concept_senses.csv").open(encoding="utf-8") as handle:
        return {row["concept"]: row["category"] for row in csv.DictReader(handle)}


def load_xcslb(kinds: tuple[str, ...] | None = None,
               category: str | None = None,
               name: str | None = None) -> Corpus:
    """XCSLB as a corpus, optionally sliced by feature type or category.

    Read from the COMPS pair file rather than `concept_matrix.txt`. The matrix
    ships without column labels and its column order is not the lexicon's:
    lining them up in file order gave `budgie` the property `can be covered in
    lip balm`. Trying to recover the labels by matching each column's concept
    set against COMPS identified only 31.2% of columns uniquely, because
    hundreds of rare properties are held by exactly one concept and so share a
    column signature.

    The pair file needs none of that. Every row states one concept and one
    property it has, which is the corpus this experiment wants, and it carries
    3,592 of the matrix's 3,643 properties -- 98.6%, missing only those COMPS
    could not build a minimal pair from.
    """
    pairs: dict[str, set[str]] = {}
    with (XCSLB_DIR / "comps_base.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pairs.setdefault(row["acceptable_concept"], set()).add(row["property"])

    kept_types = feature_types()
    in_category = categories()
    items = []
    for concept, properties in sorted(pairs.items()):
        if category and in_category.get(concept) != category:
            continue
        chosen = frozenset(
            p for p in properties
            if kinds is None or kept_types.get(p) in kinds)
        if chosen:
            items.append((concept, chosen))

    label = name or "xcslb"
    if kinds:
        label += "/" + "+".join(k.split()[0] for k in kinds)
    if category:
        label += f"/{category}"
    return Corpus(label, tuple(items))


def load_awa2(binary: bool = True, name: str = "awa2") -> Corpus:
    """AwA2's 50 classes over 85 attributes.

    The binary matrix is the published thresholding of the continuous one.
    Attributes a class does *not* have are simply absent: the trie's alphabet
    is what an individual carries, so a zero contributes no predicate.
    """
    def read(path: Path) -> list[str]:
        return [line.split("\t")[-1].strip() if "\t" in line else line.split()[-1]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    classes = read(AWA2_DIR / "classes.txt")
    predicates = read(AWA2_DIR / "predicates.txt")
    matrix_file = ("predicate-matrix-binary.txt" if binary
                   else "predicate-matrix-continuous.txt")
    rows = [line.split() for line in
            (AWA2_DIR / matrix_file).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    items = []
    for animal, flags in zip(classes, rows):
        carried = frozenset(
            predicates[index] for index, value in enumerate(flags)
            if float(value) > 0)
        if carried:
            items.append((animal.replace("+", " "), carried))
    return Corpus(name, tuple(items))

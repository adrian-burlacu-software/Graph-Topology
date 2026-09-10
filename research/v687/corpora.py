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

from .substrate import Corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
XCSLB_DIR = REPOSITORY_ROOT / "data" / "xcslb"
AWA2_DIR = REPOSITORY_ROOT / "data" / "awa2" / "Animals_with_Attributes2"
BUCHANAN = REPOSITORY_ROOT / "data" / "buchanan" / "top_to_final.csv"

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


def load_buchanan(min_frequency: int = 1, root_forms: bool = True,
                  name: str | None = None) -> Corpus:
    """Buchanan et al. (2019): 3,722 concepts over 10,850 features.

    Seven times XCSLB's concept count, and the only one of the three that
    reports **production frequency** -- how many of the participants listed
    that feature. `min_frequency` is therefore a real knob rather than a
    guess: at 1 every feature anyone mentioned is kept, and raising it trades
    coverage for agreement.

    `root_forms` collapses the morphological variants the norms record
    separately -- `leaving` and `leave` are one predicate, not two -- which is
    the difference between measuring compression and measuring spelling.
    """
    pairs: dict[str, set[str]] = {}
    with BUCHANAN.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            frequency = row.get("frequency_feature", "").strip()
            if not frequency.isdigit() or int(frequency) < min_frequency:
                continue
            feature = (row["translated"] if root_forms else row["feature"]).strip()
            if feature:
                pairs.setdefault(row["cue"].strip(), set()).add(feature)

    label = name or "buchanan"
    if min_frequency > 1:
        label += f">={min_frequency}"
    if not root_forms:
        label += "/surface"
    return Corpus(label, tuple(
        (cue, frozenset(features)) for cue, features in sorted(pairs.items())
        if features))


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


def denied_awa2() -> dict[str, frozenset[str]]:
    """class -> the attributes the matrix records as *false* for it.

    AwA2 is closed over its 85 attributes: every class was scored on every
    one, so a zero is a denial and not a silence. `load_awa2` drops the zeros
    because the trie's alphabet is what an individual carries; this keeps them,
    because "is a blue whale furry" is answerable only from the zeros.
    """
    def read(path: Path) -> list[str]:
        return [line.split("\t")[-1].strip() if "\t" in line else line.split()[-1]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    classes = read(AWA2_DIR / "classes.txt")
    predicates = read(AWA2_DIR / "predicates.txt")
    rows = [line.split() for line in
            (AWA2_DIR / "predicate-matrix-binary.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    return {animal.replace("+", " "): frozenset(
                predicates[index] for index, value in enumerate(flags)
                if float(value) <= 0)
            for animal, flags in zip(classes, rows)}


def denied_xcslb() -> dict[str, frozenset[str]]:
    """concept -> the properties COMPS picked it as a *foil* for.

    **Not denials, and not a denial source.** This docstring used to call the
    unacceptable half "the only place in any of this data where absence is
    stated rather than merely observed". It is the opposite:
    `concept_matrix.txt` is 521 x 3,644 binary and 1.58% dense -- 30,009 ones
    in 1.9M cells -- so it is a free listing, and a zero is what no
    participant happened to mention. COMPS samples its foils out of those
    zeros, which is why they do not read as denials:

        stocking  NOT absorbs sweat     (taxonomic)
        potato    NOT absorbs water     (co-occurrence)

    `Profiles` merged this into the denials it answers DENIED from until
    2026-09-09, and `is a violin made of wood` was CONTRADICTED because a
    violin is COMPS' foil for `can be made of ivory`.

    Kept because the foils are real data and `negative_sample_type` orders
    them by nearness, which is what `research/v688/audit.py` scores against.
    Do not wire it back into an answer.
    """
    denied: dict[str, set[str]] = {}
    with (XCSLB_DIR / "comps_base.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            denied.setdefault(row["unacceptable_concept"], set()).add(row["property"])
    return {concept: frozenset(properties)
            for concept, properties in sorted(denied.items())}


#: Written by `research/v688/densify.py`. **Tracked**, unlike the corpora in
#: `data/`, because it changes what v687 answers: a repository whose shipped
#: behaviour depends on an untracked file cannot be reproduced or reviewed.
#: `derived/README.md` says how to rebuild it. Every caller still treats a
#: missing file as "no distilled norms", so deleting it is a valid ablation.
DISTILLED = REPOSITORY_ROOT / "derived" / "distilled_norms.json"


def load_distilled(path: Path | None = None) -> dict[str, frozenset[str]]:
    """concept -> properties a model asserted, for R19's evidence only.

    **Kept apart from `load_xcslb` and `load_awa2` on purpose.** Those are
    elicited from people; this is distilled from SmolLM3, and `AUDIT.md` §17
    only has evidence for using it in one place -- `Profiles.corroboration`,
    where a wrong property is one vote of eight or more in a ratio that has
    to clear a third. `identify.stated` therefore does not merge it, so the
    predicate trie, the profile display and `_from_below` are unaffected and
    the `shipped` control in the audit still means what it says.

    The reason it exists: `corroboration` counts a kind as not bearing a term
    out when the norms never asked, and XCSLB averages 23.7 properties per
    concept out of a 3,592-feature lexicon -- 0.66% dense. On the 30 concepts
    XCSLB and AwA2 share, every probe measured flips to REFUSED on sparsity
    alone: 26 of 30 animals have four legs and 4 free-listers said so.
    """
    target = Path(path) if path else DISTILLED
    try:
        rows = json.loads(target.read_text(encoding="utf-8"))
    except Exception:                               # noqa: BLE001
        return {}
    return {concept: frozenset(properties)
            for concept, properties in rows.items() if properties}


#: Written by `research/v688/prune.py`. Tracked for the same reason as
#: `DISTILLED`, though `AUDIT.md` §19 measured this particular artifact as
#: inert -- not one demoted fact was ever the evidence for an answer. A
#: missing file means "demote nothing".
DEMOTED = REPOSITORY_ROOT / "derived" / "demoted_facts.json"


def load_demoted(path: Path | None = None) -> frozenset[tuple[str, str, str]]:
    """(concept, relation, object) a model says is not a claim about the class.

    `AUDIT.md` §19: a fact stated of a concept answers about that concept at
    87.6% precision, and the same fact carried down to a member answers at
    72.2%. The gap is not the crawl being wrong -- `an animal can be used in
    research` is true -- it is the crawl attaching a sentence-level
    observation to a class node, from which R1 hands it to all 4,016
    descendants.

    So this is a demotion and not a deletion. The fact stays, answers about
    the concept it was recorded of, and is skipped only when `distance` is
    non-zero. The store is not modified and the file can be thrown away.
    """
    target = Path(path) if path else DEMOTED
    try:
        rows = json.loads(target.read_text(encoding="utf-8"))
    except Exception:                               # noqa: BLE001
        return frozenset()
    return frozenset((row[0], row[1], row[2]) for row in rows)

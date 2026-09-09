"""GenericsKB as facts the store can hold.

    python -m ingestion.genericskb --fetch      # download it (39 MB)
    python -m ingestion.genericskb --report     # what this extracts, and what it drops
    python -m ingestion.genericskb --sample 20  # see the transform on real rows

GenericsKB-Best is 1,020,868 generic sentences about 128,828 terms, from AI2.
It is here because `research/v688/AUDIT.md` measured where this project's
ceiling is: the crawl reaches 19.1% of what people list about everyday
concepts, and 92% of that gap is data the store does not hold.

## What it is worth, measured before it was built

Against the 971 gold positives the crawl misses (`audit.py`'s sample):

    concepts GenericsKB covers                              95%  (444/466)
    a sentence covers >=60% of the property's words          15%
    ...and is bare enough that R28 would take it              9%

So roughly **7 to 12 coverage points**, which is more than any rule change
available -- R28 turned off entirely is worth 6.3, and it costs three points
of accuracy.

**A "generic" is not a bare claim.** The word describes the genericity of the
*subject* -- "Dogs bark" rather than "that dog barked" -- and says nothing
about qualifiers. `Most cheese is made from cow's milk, although some is made
from goat...` is a generic. That is why 15% becomes 9%: R28 refuses the
difference, and it is right to.

## The transform

Every sentence opens with its term, usually inflected or determined, so the
predicate is what is left after the subject is stripped. **89% strip cleanly**
and the remainder are sentences whose subject is not spelled the way the term
is; they are dropped rather than guessed at.

The leading word of the predicate then chooses the relation, and the shapes
are few because the corpus is regular:

    is/are          392,539     a determiner means a kind, otherwise a property
    have/has        100,213     has_a
    can              40,671     capable_of
    isa               8,202     is_a, and WordNet already holds these
    live/found in     8,586     at_location
    everything else              capable_of, a bare verb phrase

The determiner is the signal separating `is a mammal` from `is thick`, which
is the same categorical test `engine.py` uses to tell `is a mouse an animal`
from `is a television modern`. It is not a tagger and it does not need to be.

## Two things that would double-count

GenericsKB collects from six sources, and two of them are already in the
store: **ConceptNet (63,011 rows) and WordNet3.0 (71,919)**. `facts()` drops
them by default -- they are not new evidence, and letting a fact corroborate
itself under a second name is exactly what R19 exists to prevent.

## Confidence

`score` is a real distribution, unlike ConceptNet's constant 0.35:

    p5 0.256  p25 0.369  p50 0.575  p75 0.838  p90 1.000

About 14% of rows sit at exactly 1.000, so the top of the range carries less
information than the middle. `research/v688/confidence.py` calibrates a
source by its percentile rather than its raw number, and these are the edges
it would need.
"""
from __future__ import annotations

import argparse
import collections
import re
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = REPOSITORY_ROOT / "data" / "genericskb"
PARQUET = DIRECTORY / "generics_kb_best.parquet"
URL = ("https://huggingface.co/datasets/community-datasets/generics_kb/"
       "resolve/main/generics_kb_best/train-00000-of-00001.parquet")

#: Sources inside GenericsKB the store already holds under their own name.
ALREADY_HELD = frozenset({"ConceptNet", "WordNet3.0"})

#: Measured over all 1,020,868 rows, for `confidence.percentile`.
SCORE = ((0.256, 0.05), (0.280, 0.10), (0.369, 0.25), (0.575, 0.50),
         (0.838, 0.75), (1.000, 0.90))

#: Openers that name where a thing is rather than what it does.
LOCATIVE = ("live in", "lives in", "live on", "lives on", "are found in",
            "is found in", "are found on", "is found on", "live near",
            "grow in", "grows in", "grow on", "grows on")

#: Openers that name what a thing is for.
PURPOSIVE = ("are used for", "is used for", "are used to", "is used to",
             "are used as", "is used as", "are used by", "is used by",
             "used for", "used to", "used as", "used by")

#: Adverbs a generic opens with that say nothing about the claim. Left in,
#: they become part of the object -- `also dig to get food` -- and R28 then
#: refuses the fact for carrying a surplus that is not there.
HEDGES = ("also", "often", "usually", "generally", "typically", "sometimes",
          "always", "mainly", "mostly", "commonly", "normally", "frequently",
          "occasionally", "rarely", "still", "then", "thus", "therefore",
          "however", "actually", "probably", "perhaps", "really", "even")

DETERMINERS = ("a ", "an ", "the ", "one ", "any ")


def fetch(force: bool = False) -> Path:
    """Download GenericsKB-Best if it is not already here. 39 MB."""
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if PARQUET.exists() and not force:
        return PARQUET
    urllib.request.urlretrieve(URL, PARQUET)
    return PARQUET


def rows():
    """(term, sentence, source, score), straight off the parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(fetch())
    return zip(table.column("term").to_pylist(),
               table.column("generic_sentence").to_pylist(),
               table.column("source").to_pylist(),
               table.column("score").to_pylist())


def predicate(term: str, sentence: str) -> str | None:
    """The sentence with its subject removed, or None if it does not open
    with one.

    A term appears as itself, pluralised, or under a determiner or a
    quantifier. Nothing here guesses: a sentence whose subject is spelled
    differently from its term is dropped, and `--report` counts how many.
    """
    text = (sentence or "").strip().rstrip(".").strip()
    head = (term or "").strip().lower()
    if not text or not head:
        return None
    low = text.lower()
    forms = [head + "s", head + "es", head]
    for lead in ("a ", "an ", "the ", "some ", "most ", "all ", "many ",
                 "several ", "two "):
        forms += [lead + head + "s", lead + head + "es", lead + head]
    for form in sorted(forms, key=len, reverse=True):
        if low.startswith(form + " "):
            return text[len(form) + 1:].strip()
    return None


def relation_of(tail: str) -> tuple | None:
    """(relation, object) for a predicate phrase, or None to drop it.

    The leading word decides, because the corpus is regular enough that it
    can: 392,539 of the predicates open with `is` or `are`.
    """
    if not tail:
        return None
    parts = tail.split()
    while parts and parts[0].lower().strip(",") in HEDGES:
        parts = parts[1:]
    tail = " ".join(parts)
    if not tail:
        return None
    low = tail.lower()
    for phrase in PURPOSIVE:
        if low.startswith(phrase + " "):
            return "used_for", tail[len(phrase) + 1:].strip()
    for phrase in LOCATIVE:
        if low.startswith(phrase + " "):
            return "at_location", tail[len(phrase) + 1:].strip()

    words = tail.split()
    while words and words[0].lower().strip(",") in HEDGES:
        words = words[1:]
    if not words:
        return None
    tail = " ".join(words)
    head, rest = words[0].lower(), " ".join(words[1:]).strip()
    if head == "isa":
        return ("is_a", rest) if rest else None
    if head in ("is", "are", "was", "were"):
        if not rest:
            return None
        # A determiner means a kind, its absence a property. The same
        # categorical test `engine.py` uses to tell `is a mouse an animal`
        # from `is a television modern`, and for the same reason: no tagger
        # separates them and the grammar already does.
        low_rest = rest.lower()
        if low_rest.startswith(("made of ", "made from ", "made out of ")):
            return "made_of", rest.split(" ", 2)[-1].strip()
        if low_rest.startswith(DETERMINERS):
            return "is_a", rest.split(" ", 1)[-1].strip()
        return "has_property", rest
    if head in ("have", "has"):
        return ("has_a", rest) if rest else None
    if head in ("can", "could", "may", "might"):
        return ("capable_of", rest) if rest else None
    if head in ("contain", "contains"):
        return ("has_a", rest) if rest else None
    if not head.isalpha():
        return None
    # A bare verb phrase. `Aardvarks dig to get food` is something aardvarks
    # do, which is what `capable_of` means in this store.
    return "capable_of", tail


def facts(skip_held: bool = True, floor: float = 0.0):
    """(term, relation, object, source, confidence) for every usable row.

    `skip_held` drops the ConceptNet and WordNet rows GenericsKB collected,
    because the store already holds both under their own names and a fact
    that corroborates itself is not corroboration.
    """
    for term, sentence, source, score in rows():
        if skip_held and source in ALREADY_HELD:
            continue
        if score is not None and score < floor:
            continue
        tail = predicate(term, sentence)
        if tail is None:
            continue
        found = relation_of(tail)
        if found is None:
            continue
        relation, obj = found
        obj = obj.strip().rstrip(".").strip()
        if not obj or len(obj) > 120:
            continue
        yield (term.strip().lower(), relation, obj,
               "genericskb", float(score or 0.0))


def report(limit: int = 0) -> dict:
    """What the transform reaches and what it drops.

    Every count here is one this file's docstring quotes, so the docstring
    can be checked rather than believed.
    """
    seen = kept = no_subject = no_relation = held = 0
    relations: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    for index, (term, sentence, source, _score) in enumerate(rows()):
        if limit and index >= limit:
            break
        seen += 1
        sources[source] += 1
        if source in ALREADY_HELD:
            held += 1
            continue
        tail = predicate(term, sentence)
        if tail is None:
            no_subject += 1
            continue
        found = relation_of(tail)
        if found is None:
            no_relation += 1
            continue
        kept += 1
        relations[found[0]] += 1
    return {"rows": seen, "kept": kept,
            "already_held": held,
            "subject_not_matched": no_subject,
            "no_relation": no_relation,
            "kept_share": round(kept / seen, 4) if seen else 0.0,
            "relations": dict(relations.most_common()),
            "sources": dict(sources.most_common())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true",
                        help="download GenericsKB-Best and stop")
    parser.add_argument("--report", action="store_true",
                        help="count what the transform reaches and drops")
    parser.add_argument("--sample", type=int, default=0,
                        help="print this many extracted facts")
    parser.add_argument("--limit", type=int, default=0,
                        help="only read this many rows (for a quick report)")
    options = parser.parse_args(argv)

    if options.fetch:
        print(f"GenericsKB-Best at {fetch()}")
        return 0
    if options.report:
        import json

        print(json.dumps(report(options.limit), indent=2))
        return 0
    if options.sample:
        for index, fact in enumerate(facts()):
            if index >= options.sample:
                break
            term, relation, obj, _source, score = fact
            print(f"  {score:.2f}  {term:22s} {relation:14s} {obj[:60]}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Wikipedia lead paragraphs, read into facts about the kind they describe.

    python -m research.v689.articles read   [--intros data/wikipedia/intros.jsonl]
    python -m research.v689.articles report [--sample 60]

A gloss is a noun phrase. An article is sentences about the thing:

    The Beagle is a British breed of scent hound, similar in appearance to the
    much larger foxhound. It was bred primarily for hunting rabbit or hare ...

So an article is read with v689's own reading of claims about a kind -- the
`read()` and `Session._relation` that `beagles can't swim` goes through when a
person says it -- one sentence at a time, and only a sentence that is about
the kind:

- **its subject is the kind**: a lemma of the synset, the article's title, or
  `it` or `they` once a sentence has named the kind;
- **its subject is not quantified.** `Some bats ...`, `many fish ...` is a fact
  about a few, filed under the class: the over-affirmation this project spent
  weeks taking out of the store, and an article is full of it. Sentences with
  `may`, `might` or `could` are left for the same reason.

The first sentence's copula phrase -- `a British breed of scent hound` -- is
read by `GlossReader`, so the genus is checked against the taxonomy exactly as
a gloss's is. Everything else is left: sentences about the apple tree, the
word's etymology, where the breed was developed.

What it reads goes into a definitions memory of its own,
`state/v689-articles.sqlite`, so it can be loaded, measured and dropped apart
from WordNet's glosses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .definitions import Defined, DefinitionMemory, GlossReader, Reading
from .episodic import DID_NOT, name_of
from .reading import article, read

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"
INTROS = REPOSITORY_ROOT / "data" / "wikipedia" / "intros.jsonl"
MEMORY = REPOSITORY_ROOT / "state" / "v689-articles.sqlite"

#: Sentences read per article. The lead is written most-important first.
MOST = 10

#: A subject with one of these is about some of the kind, not the kind.
QUANTIFIED = frozenset({"some", "many", "most", "several", "few", "certain",
                        "other", "various", "numerous", "such"})

#: Pronouns that can stand for the kind once a sentence has named it.
STANDING = frozenset({"it", "they", "these", "this"})

#: Modal verbs that make a sentence a possibility, not a property.
POSSIBLE = frozenset({"may", "might", "could"})

#: What an encyclopedia says about a thing that is not what the thing is
#: like: how it is filed, what it is called, what it belongs to.
FILING = frozenset({"classify", "place", "include", "belong", "know", "call",
                    "name", "list", "refer", "divide", "consider", "describe",
                    "recognize", "recognise", "group", "assign", "treat"})

#: A fact that opens with one of these is history, a comparison or a
#: relative clause the reader flattened, not a property.
NOT_PROPERTIES = frozenset({"been", "being", "which", "that", "who", "where",
                            "when", "one", "first", "more", "most", "less",
                            "often", "sometimes", "generally", "also",
                            "still", "now", "then", "once"})

#: Words that say nothing on their own as a property.
EMPTY_PROPERTIES = frozenset({"capable", "similar", "generic", "able",
                              "related", "different", "various", "common",
                              "same", "such", "known", "used", "made"})


class ArticleReader:
    """Reads an article with an `Asker`'s parser and v689's claim reading."""

    def __init__(self, asker) -> None:
        from .session import Session, Taught

        self.asker = asker
        self.glosses = GlossReader(asker)
        self.session = Session(asker)
        self.lexicon = Taught(asker, self.session.memory.kinds)

    @staticmethod
    def names(concept: str, title: str) -> set:
        """Every way the article can name the kind."""
        found = {name_of(concept).lower(),
                 title.split(" (")[0].strip().lower()}
        try:
            from nltk.corpus import wordnet

            found |= {lemma.name().replace("_", " ").lower() for lemma in
                      wordnet.synset(concept.replace(" ", "_")).lemmas()}
        except Exception:                           # noqa: BLE001
            pass
        return found

    def read(self, concept: str, title: str, text: str) -> Reading:
        reading = Reading(concept, (text or "")[:2000])
        names = self.names(concept, title)
        kind = name_of(concept)
        doc = self.asker.parser.nlp(text or "")
        named = False
        for number, sentence in enumerate(doc.sents):
            if number >= MOST:
                break
            subject = next((token for token in sentence
                            if token.dep_ in ("nsubj", "nsubjpass")
                            and token.head.dep_ == "ROOT"), None)
            if subject is None:
                continue
            spread = {token.i for token in subject.subtree}
            phrase = " ".join(token.lemma_.lower() for token in subject.subtree
                              if token.dep_ not in ("det", "punct", "appos")
                              and not token.is_punct
                              and not token.text.startswith("("))
            about = (subject.lemma_.lower() in names or phrase in names
                     or subject.text.lower() in names
                     or (named and subject.text.lower() in STANDING))
            quantified = any(token.text.lower() in QUANTIFIED
                             for token in subject.subtree
                             if token.i <= subject.i)
            possible = any(token.text.lower() in POSSIBLE
                           for token in sentence)
            # History is not a property: `was invented in Europe`, `has been
            # served at banquets`. A lead reads its present tense.
            root = sentence.root
            past = root.tag_ == "VBD" or any(
                child.dep_ in ("aux", "auxpass")
                and (child.tag_ == "VBD" or child.text.lower() == "been")
                for child in root.children)
            if not about or quantified or possible or past:
                continue
            named = True
            predicate = _clean([token for token in sentence
                                if token.i not in spread])
            if not predicate:
                continue
            first = predicate.split()[0].lower()
            if first in ("is", "are") and not reading.genus:
                # The defining sentence: its copula phrase is a gloss.
                found = self.glosses.read(concept,
                                          predicate.split(" ", 1)[-1])
                reading.genus, reading.agrees = found.genus, found.agrees
                reading.facts.extend(
                    Defined(fact.relation, fact.object, "lead", sentence.text)
                    for fact in found.facts if self.properly(fact))
                continue
            reading.facts.extend(self._claims(kind, predicate,
                                              sentence.text))
        reading.facts = list({(one.relation, one.object): one
                              for one in reading.facts}.values())
        return reading

    def _claims(self, kind: str, predicate: str, sentence: str) -> list:
        """A predicate about the kind, read as a person saying it."""
        said = f"{article(kind)} {kind} {predicate}"
        parsed = read(said, self.lexicon, frozenset())
        facts = []
        for part in [parsed] + list(parsed.more):
            if part.act != "teach" or part.mention is None:
                continue
            relation, obj, _, _ = self.session._relation(
                part.aux, part.rest, part.mention.kind, part.holds)
            if not relation or relation in ("is_a", DID_NOT) or not obj:
                continue
            fact = self.glosses._normalise(Defined(relation, obj,
                                                   "sentence", sentence))
            if fact is not None and self.properly(fact):
                facts.append(fact)
        return facts

    def properly(self, fact: Defined) -> bool:
        """Is this a property of the kind, rather than prose about it?"""
        words = fact.object.split()
        if not words or words[0] in NOT_PROPERTIES or " than " in f" {fact.object} ":
            return False
        if self.glosses.asker.lemma(words[0]) in FILING:
            return False
        if fact.relation in ("has_property", "not_has_property") and len(
                words) == 1:
            if words[0] in EMPTY_PROPERTIES:
                return False
            # `mammal`, `buttercup`: a noun the parse took for a quality.
            senses = self.glosses._senses
            if senses(words[0], "n") and not (senses(words[0], "a")
                                              or senses(words[0], "s")):
                return False
        return True


def _clean(tokens) -> str:
    """Tokens back into text, without parentheticals or end punctuation."""
    depth, kept = 0, []
    for token in tokens:
        if token.text == "(":
            depth += 1
            continue
        if token.text == ")":
            depth = max(0, depth - 1)
            continue
        if not depth:
            kept.append(token.text_with_ws)
    return " ".join("".join(kept).split()).strip(" ,.;:")


def _asker():
    from research.v687.language import Parser
    from research.v687.reason import Reasoner

    from .asker import Asker

    reasoner = Reasoner(STORE)
    parser = Parser(vocabulary=reasoner.vocabulary(),
                    nouns=reasoner.noun_vocabulary())
    return Asker(reasoner, parser)


def learn(intros: Path = INTROS, memory_path: Path = MEMORY) -> dict:
    reader = ArticleReader(_asker())
    memory = DefinitionMemory(memory_path)
    started = time.time()
    counts = {"articles": 0, "skipped": 0, "facts": 0, "genus": 0,
              "agrees": 0}
    rows = [json.loads(line) for line in
            intros.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if row.get("missing") or row.get("disambiguation") or not row.get(
                "extract"):
            counts["skipped"] += 1
            continue
        found = reader.read(row["synset"], row["title"], row["extract"])
        memory.keep(found, "wikipedia")
        counts["articles"] += 1
        counts["facts"] += len(found.facts)
        counts["genus"] += bool(found.genus)
        counts["agrees"] += bool(found.agrees)
        if not number % 50:
            print(f"[articles] {number}/{len(rows)} {counts} "
                  f"{time.time() - started:.0f}s", flush=True)
    return {**counts, "seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("read", "report"))
    parser.add_argument("--intros", type=Path, default=INTROS)
    parser.add_argument("--memory", type=Path, default=MEMORY)
    parser.add_argument("--sample", type=int, default=60)
    options = parser.parse_args(argv)
    if options.stage == "read":
        result = learn(options.intros, options.memory)
    else:
        from .learn_definitions import report

        result = report(options.sample, options.memory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

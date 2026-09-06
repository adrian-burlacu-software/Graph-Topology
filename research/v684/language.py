"""Turning a typed question into a concept, a relation and a target.

spaCy does the linguistic work -- lemmatising, tagging, finding the noun the
question is about. The mapping from question shape to relation is a table
rather than a model, because it has to be inspectable: when an answer is wrong,
the first thing to check is whether the question was read correctly, and the UI
shows this parse for exactly that reason.

Without spaCy installed everything still runs on a regex fallback, and
`Parser.backend` says which one answered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Question cue -> relation. Ordered: the first phrase that matches wins, so
#: longer and more specific cues are listed before general ones.
RELATION_CUES: tuple[tuple[str, str], ...] = (
    ("what is .* made (of|from)", "made_of"),
    ("made (of|from)", "made_of"),
    ("used for", "used_for"),
    ("what is .* for", "used_for"),
    ("where .* (found|located|live|be|is|are)", "at_location"),
    ("^where", "at_location"),
    ("part of", "part_of"),
    ("(have|has|contain|include)", "has_part"),
    ("(want|desire|wish|like)", "desires"),
    ("(cause|lead to|result in)", "causes"),
    ("(need|require|prerequisite)", "has_prerequisite"),
    ("(can be|gets|is being)", "receives_action"),
    ("(can|could|able to|capable)", "capable_of"),
    ("^(what|which) .* do", "capable_of"),
    ("(is|are|was|were|be)", "has_property"),
)

#: Yes/no questions open with one of these.
POLAR = ("can", "could", "is", "are", "was", "were", "does", "do", "did",
         "has", "have", "will", "would", "should", "must", "may", "might")

#: Words that never carry the content of a question.
STOP = frozenset({
    "a", "an", "the", "some", "any", "this", "that", "these", "those",
    "do", "does", "did", "be", "is", "are", "was", "were", "been", "being",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "have", "has", "had", "of", "to", "in", "on", "at", "for", "with", "by",
    # Prepositions that only introduce an object. `into` was missing from a
    # list that already held `in`, `to`, `on` and `at`, so `fall into wrong
    # hands` answered `can a dog fall into a hole`: the verb and preposition
    # alone cleared the threshold and the object -- the only part that
    # distinguishes them -- never counted. Adverbial particles stay out of
    # this list on purpose: `fall down` and `fall over` are different claims.
    "into", "onto", "upon", "from", "within", "toward", "towards",
    "between", "among", "amongst", "beside",
    "what", "which", "who", "where", "when", "why", "how", "it", "its",
    "they", "them", "their", "there", "here", "you", "your", "i", "me",
    "and", "or", "but", "if", "then", "than", "as", "so", "such",
})


@dataclass
class Parse:
    question: str
    subject: str | None
    relation: str | None
    target: str | None
    polar: bool
    backend: str
    tokens: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "subject": self.subject,
            "relation": self.relation, "target": self.target,
            "polar": self.polar, "backend": self.backend,
            "tokens": self.tokens, "note": self.note,
        }


class Parser:
    """Reads questions. Degrades to regex when spaCy is unavailable."""

    #: How many tokens a subject may span. `fire truck` is two, `bird of
    #: prey` is three; past four it is a sentence, not a name.
    MAX_SUBJECT_TOKENS = 4

    def __init__(self, model: str = "en_core_web_sm",
                 vocabulary: set[str] | None = None):
        self.nlp = None
        self.backend = "regex"
        #: Every lemma the ontology knows, used to find where a subject ends.
        #: Optional: without it the parser still reads questions, just less
        #: well on compound subjects.
        self.vocabulary = vocabulary or set()
        try:
            import spacy
            self.nlp = spacy.load(model, disable=["ner"])
            self.backend = f"spacy:{model}"
        except Exception:                      # noqa: BLE001 - optional dependency
            pass

    def longest_known(self, doc) -> str | None:
        """The longest phrase after the auxiliary that names a known concept.

        Scans start positions left to right so the earliest subject wins, and
        lengths longest first so `fire truck` beats `fire`. The phrase has to
        end on a noun, or `can a large dog fall` would settle for `large`.

        It gives up at the first noun it cannot place rather than searching on,
        because the next noun along is usually the *target*: without that,
        `is a zzzqqq an animal` answered about animals.
        """
        rest = [t for t in doc[1:] if t.pos_ != "DET" and not t.is_punct]
        for start in range(min(3, len(rest))):
            span = rest[start:start + self.MAX_SUBJECT_TOKENS]
            for length in range(len(span), 0, -1):
                if span[length - 1].pos_ not in ("NOUN", "PROPN"):
                    continue
                for form in (" ".join(t.lemma_.lower() for t in span[:length]),
                             " ".join(t.text.lower() for t in span[:length])):
                    if form in self.vocabulary and form not in STOP:
                        return form
            if rest[start].pos_ in ("NOUN", "PROPN"):
                return None                    # an unknown subject, not a hint
        return None

    # -- lemmatisation ----------------------------------------------------
    def lemmas(self, text: str) -> list[str]:
        """Content lemmas, lowercased, stop words removed."""
        if self.nlp is not None:
            return [t.lemma_.lower() for t in self.nlp(text)
                    if not t.is_punct and not t.is_space
                    and t.lemma_.lower() not in STOP]
        return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]

    def head_noun(self, text: str, polar: bool = False) -> str | None:
        """The noun the question is about."""
        if self.nlp is None:
            words = [w for w in re.findall(r"[a-z0-9']+", text.lower())
                     if w not in STOP]
            return words[0] if words else None
        doc = self.nlp(text)

        # A polar question puts its subject right after the auxiliary, and the
        # ontology can say where that subject ends. This is needed because the
        # tagger reads `a canine fall` as one compound noun -- `canine` and
        # `fall` are both nouns -- and hands back either no subject at all or
        # the wrong end of the run:
        #
        #     can a canine fall into a hole   no subject; chunk root `fall`
        #     can a hammer break glass        nsubj `glass`
        #     does a wolf howl                nsubj `howl`
        #     can a fire truck move           no subject; chunk root `move`
        #
        # Taking the first noun fixes the first three and breaks the fourth;
        # taking the last breaks the first three. The longest phrase that names
        # something the ontology knows gets all four: `fire truck` and `police
        # dog` are lemmas, `canine fall` and `hammer break` are not.
        if polar and self.vocabulary:
            found = self.longest_known(doc)
            if found:
                return found

        # An explicit grammatical subject, when the parse found one. Reading it
        # from the noun chunk instead is the older bug: for `can a canine fall
        # into a hole` the chunk is "a canine fall" and its root is `fall`.
        for token in doc:
            if (token.dep_ in ("nsubj", "nsubjpass")
                    and token.pos_ in ("NOUN", "PROPN")
                    and token.lemma_.lower() not in STOP):
                return token.lemma_.lower()

        # No subject at all means the parse came apart entirely. In a polar
        # question the first noun after the opening auxiliary is the thing
        # being asked about, and it beats guessing from a broken tree.
        if polar:
            for token in doc[1:]:
                if (token.pos_ in ("NOUN", "PROPN")
                        and token.lemma_.lower() not in STOP):
                    return token.lemma_.lower()

        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.lemma_.lower() not in STOP:
                return head.lemma_.lower()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_.lower() not in STOP:
                return token.lemma_.lower()
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_.lower() not in STOP:
                return token.lemma_.lower()
        return None

    # -- question -> (subject, relation, target) --------------------------
    def parse(self, question: str) -> Parse:
        text = question.strip().rstrip("?").strip()
        lowered = text.lower()
        polar = lowered.split()[0] in POLAR if lowered.split() else False

        relation = None
        for pattern, mapped in RELATION_CUES:
            if re.search(pattern, lowered):
                relation = mapped
                break

        subject = self.head_noun(text, polar)

        target = None
        if polar and subject:
            # "can a dog fall down" -> everything after the subject noun
            match = re.search(rf"\b{re.escape(subject)}\w*\b(.*)$", lowered)
            tail = (match.group(1) if match else "").strip()
            # Drop the auxiliary or main verb the question already spent on the
            # relation: "does a dog have a tail" asks about a tail, not a having.
            tail = re.sub(r"^(be|is|are|was|were|to|have|has|had|get|gets|"
                          r"contain|contains|include|includes)\b", "", tail).strip()
            target = tail or None

            # "is a dog an animal" is a taxonomy question, not a property one:
            # a determiner after the copula means a kind is being named.
            if relation == "has_property" and re.match(r"^(a|an|the)\b", tail):
                relation = "is_a"
                target = re.sub(r"^(a|an|the)\b", "", tail).strip() or None
        elif not polar:
            # "what is a dog made of" -> open question, no target to match
            target = None
            if re.match(r"^(what|which) (kind|type|sort)s? of", lowered):
                relation = "is_a"

        tokens: list[dict[str, Any]] = []
        if self.nlp is not None:
            tokens = [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_,
                       "dep": t.dep_} for t in self.nlp(text)]

        note = ""
        if relation is None:
            relation = "capable_of" if polar else None
            note = "No relation cue recognised; defaulted."
        if subject is None:
            note = "Could not find a noun to reason about."

        return Parse(question=question, subject=subject, relation=relation,
                     target=target, polar=polar, backend=self.backend,
                     tokens=tokens, note=note)

    # -- matching a target phrase against a stored fact --------------------
    def matcher(self, threshold: float = 0.6):
        """Build a predicate deciding whether a fact answers the question.

        Facts from Ascent++ are free text -- "fall into hole", "fall asleep" --
        so this is lemma overlap rather than equality. The threshold is the
        share of the question's content lemmas that the fact must contain, so
        "fall down" matches "fall down the stairs" but not "fall in love".
        """
        cache: dict[str, set[str]] = {}

        def lemma_set(text: str) -> set[str]:
            if text not in cache:
                cache[text] = set(self.lemmas(text))
            return cache[text]

        def score(fact_object: str, target: str | None) -> float:
            """Share of the question's content lemmas the fact carries."""
            if not target:
                return 1.0
            wanted = lemma_set(target)
            if not wanted:
                return 1.0
            have = lemma_set(fact_object)
            if not have:
                return 0.0
            if target.lower().strip() in fact_object.lower():
                return 1.0          # the fact names the target outright
            return len(wanted & have) / len(wanted)

        def matches(fact_object: str, target: str | None) -> bool:
            return score(fact_object, target) >= threshold

        matches.score = score
        matches.threshold = threshold
        return matches

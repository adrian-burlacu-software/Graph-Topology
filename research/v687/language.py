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
    # Every cue is anchored on word boundaries. Without them a cue matched
    # inside a word: `beagle` contains `be`, so `does a beagle breathe` was
    # read as a property question and answered UNKNOWN, while `does a dog
    # breathe` fell through to the polar default of `capable_of` and answered
    # correctly. Two spellings of the same question, two different rules.
    # Anchoring means the inflections have to be spelled out, which is the
    # price of not matching `can` inside `candle`.
    (r"\bwhat is\b.*\bmade (of|from)\b", "made_of"),
    (r"\bmade (of|from)\b", "made_of"),
    (r"\bused for\b", "used_for"),
    (r"\bwhat is\b.*\bfor\b", "used_for"),
    (r"\bwhere\b.*\b(found|located|live|lives|living|be|is|are)\b",
     "at_location"),
    (r"^where\b", "at_location"),
    (r"\bpart of\b", "part_of"),
    (r"\b(have|has|had|contain|contains|containing|include|includes)\b",
     "has_part"),
    (r"\b(want|wants|desire|desires|wish|wishes|like|likes)\b", "desires"),
    (r"\b(cause|causes|caused|lead to|leads to|result in|results in)\b",
     "causes"),
    (r"\b(need|needs|needed|require|requires|required|prerequisite)\b",
     "has_prerequisite"),
    (r"\b(can be|gets|is being)\b", "receives_action"),
    (r"\b(can|could|able to|capable)\b", "capable_of"),
    (r"^(what|which)\b.*\bdo(es)?\b", "capable_of"),
    (r"\b(is|are|was|were|be)\b", "has_property"),
)

#: Words a relation cue spends on naming the relation, so they cannot also be
#: what the question is about. Only these: a cue pattern may span half the
#: question -- `what is .* made of` covers `hammer` too -- and spending the
#: whole match left `what is a hammer made of` with no subject at all.
SPENT_ON_RELATION = frozenset("""
made used found located live lives living part contain contains containing
include includes want wants desire desires wish wishes like likes cause causes
caused lead leads result results need needs needed require requires required
prerequisite able capable
""".split())

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
    #: The word the question is about that the ontology could not place. Set
    #: only when that is why there is no subject, so the answer can name it
    #: instead of quietly answering about a different word.
    unknown: str | None = None
    #: True when `is_a` was inferred from a bare noun predicate rather than
    #: read off a determiner. `is a dog an animal` names a kind outright; `is
    #: a raccoon white` and `is a chair furniture` have the same shape as each
    #: other, and only the norms can tell them apart, so a hedged reading asks
    #: them first and falls back to the taxonomy.
    hedged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "subject": self.subject,
            "relation": self.relation, "target": self.target,
            "polar": self.polar, "backend": self.backend,
            "tokens": self.tokens, "note": self.note,
            "unknown": self.unknown, "hedged": self.hedged,
        }


#: Words that make a tail more than one claim. This lived in a regex whose
#: word-boundary escapes had been mangled into control characters, so the
#: pattern never matched and the guard it belonged to was always true. A set
#: of words cannot be mangled and says the same thing.
CONNECTIVES = frozenset({"and", "or", "not", "no", "never"})


def whole_words(needle: str, haystack: str) -> bool:
    """Does `haystack` contain `needle` as whole words?

    Deliberately not a regex. Three `\b` patterns in this codebase have had
    the backslash mangled in transit into a literal backspace and silently
    stopped matching -- one of them in this very file, where it disabled a
    guard from the day it was written. Padding and a substring test cannot be
    mangled that way.

    Every character that is not a letter or a digit becomes a space, so
    "swimming-pool" and "swimming pool" read alike, and the padding makes the
    ends of the string behave like any other boundary.
    """
    def spaced(text: str) -> str:
        return " " + "".join(
            character if character.isalnum() else " "
            for character in text.lower()) + " "

    needle = spaced(needle).strip()
    return bool(needle) and f" {needle} " in spaced(haystack)


class Parser:
    """Reads questions. Degrades to regex when spaCy is unavailable."""

    #: How many tokens a subject may span. `fire truck` is two, `bird of
    #: prey` is three; past four it is a sentence, not a name.
    MAX_SUBJECT_TOKENS = 4

    def __init__(self, model: str = "en_core_web_sm",
                 vocabulary: set[str] | None = None,
                 nouns: set[str] | None = None):
        self.nlp = None
        self.backend = "regex"
        #: Every lemma the ontology knows, used to find where a subject ends.
        #: Optional: without it the parser still reads questions, just less
        #: well on compound subjects.
        self.vocabulary = vocabulary or set()
        #: Words the relation cue already spent, for this one parse.
        self._spent: set[str] = set()
        #: The subset of it that names things rather than properties.
        self.nouns = nouns or set()
        #: Set by `head_noun` when it refuses to substitute a later noun.
        self._blocked: str | None = None
        try:
            import spacy
            self.nlp = spacy.load(model, disable=["ner"])
            self.backend = f"spacy:{model}"
        except Exception:                      # noqa: BLE001 - optional dependency
            pass

    #: Tags a subject can wear. A question can be about a thing, an act or an
    #: exclamation -- `is running a sport`, `is hello a greeting` -- and
    #: requiring a noun is what sent both of those to their second word.
    SUBJECT_POS = ("NOUN", "PROPN", "VERB", "INTJ", "X")

    #: Tags that modify a subject rather than being one, so a scan may step
    #: over them: `is a large dog furry` is about the dog, not the largeness.
    MODIFIER_POS = ("ADJ", "ADV", "ADP", "PART", "AUX", "NUM", "DET", "PRON",
                    "CCONJ", "SCONJ")

    def subject_group(self, doc) -> list:
        """The tokens between the auxiliary and the predicate.

        A copular question has two halves and a determiner marks the seam:
        in `is a wemble an animal` the `an` begins the predicate, so `animal`
        was never a candidate subject and the scan should never have reached
        it. Grouping on determiners is what stops the subject scan running
        into the predicate -- without it, an unrecognised subject was quietly
        answered about the thing it was being compared *to*.
        """
        group: list = []
        started = False
        for token in doc[1:]:
            if token.is_punct:
                continue
            if token.pos_ == "DET":
                if started:
                    break              # a second determiner: the predicate
                continue
            if started and token.pos_ in ("AUX", "VERB"):
                break                  # the verb: whatever follows is asked
            started = True
            group.append(token)
        return group

    def longest_known(self, doc) -> tuple[str | None, str | None]:
        """The subject, or else the word that stopped the scan.

        Within the subject group, longest phrases first so `fire truck` beats
        `fire`, and earliest start first so the subject wins over its own
        modifiers.

        Two things it must not do, both found by the answer audit. It must not
        require the subject to be a noun: `hello` tags as an interjection and
        `running` as a verb, and both are concepts this ontology holds, so
        demanding a noun walked past them into the predicate -- `is hello a
        greeting` came back as a list of the properties of greetings. And when
        it cannot place the word the question is about it must name that word
        rather than reading on, because what follows is the predicate: that is
        how `is a wemble an animal` answered about animals.
        """
        group = self.subject_group(doc)
        for start in range(min(3, len(group))):
            span = group[start:start + self.MAX_SUBJECT_TOKENS]
            for length in range(len(span), 0, -1):
                if span[length - 1].pos_ not in ("NOUN", "PROPN"):
                    continue
                for form in (" ".join(t.lemma_.lower() for t in span[:length]),
                             " ".join(t.text.lower() for t in span[:length])):
                    if form in self.vocabulary and form not in STOP:
                        return form, None
            token = group[start]
            if token.pos_ in self.MODIFIER_POS or token.lemma_.lower() in STOP:
                continue               # a modifier of the subject, not it
            # Text before lemma: `running` and `greeting` are both concepts in
            # their own right, and lemmatising first answered about `run` and
            # `greet` instead.
            for form in (token.text.lower(), token.lemma_.lower()):
                if form in self.vocabulary and form not in STOP:
                    return form, None
            if token.pos_ in self.SUBJECT_POS:
                return None, token.text.lower()
        # Every token in the group modifies something, so the subject is a
        # modifier used as a noun: `is red a color` is about red.
        for token in group:
            for form in (token.text.lower(), token.lemma_.lower()):
                if form in self.vocabulary and form not in STOP:
                    return form, None
        return None, group[0].text.lower() if group else None

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
            found, blocked = self.longest_known(doc)
            if found:
                return found
            if blocked:
                # The question is about a word this ontology does not hold.
                # Reading on would answer about the predicate instead, which
                # is the substitution the audit named: an answer to a question
                # nobody asked, with nothing to say it happened.
                self._blocked = blocked
                return None

        # An explicit grammatical subject, when the parse found one. Reading it
        # from the noun chunk instead is the older bug: for `can a canine fall
        # into a hole` the chunk is "a canine fall" and its root is `fall`.
        for token in doc:
            if (token.dep_ in ("nsubj", "nsubjpass")
                    and token.pos_ in ("NOUN", "PROPN")
                    and self._usable(token)):
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
            if self._usable(head):
                return head.lemma_.lower()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and self._usable(token):
                return token.lemma_.lower()
        for token in doc:
            if token.pos_ == "VERB" and self._usable(token):
                return token.lemma_.lower()
        # Last resort: any word this ontology holds. `what does cold cause`
        # has no noun and no verb left once the cue is spent, and answering
        # nothing was worse than answering about cold.
        for token in doc:
            if not self._usable(token):
                continue
            for form in (token.text.lower(), token.lemma_.lower()):
                if form in self.vocabulary:
                    return form
        return None

    def _usable(self, token) -> bool:
        """Can this token be the subject, or did the question spend it?"""
        lemma = token.lemma_.lower()
        if lemma in STOP:
            return False
        return not (lemma in self._spent or token.text.lower() in self._spent)

    # -- question -> (subject, relation, target) --------------------------
    def parse(self, question: str) -> Parse:
        text = question.strip().rstrip("?").strip()
        lowered = text.lower()
        polar = lowered.split()[0] in POLAR if lowered.split() else False

        relation = None
        spent = ""
        for pattern, mapped in RELATION_CUES:
            found = re.search(pattern, lowered)
            if found:
                relation = mapped
                # The words this cue consumed. A question spends them on
                # saying *which relation* it is asking about, so they are not
                # also what it is asking about -- `what is needed to greet`
                # was answered about `need`, and `what does a greeting cause`
                # about `cause`. Both are the question's own grammar read back
                # as its subject.
                spent = found.group(0)
                break

        self._blocked = None
        hedged = False
        self._spent = {word for word in
                       (re.sub(r"[^a-z]", "", token)
                        for token in spent.lower().split())
                       if word in SPENT_ON_RELATION}
        subject = self.head_noun(text, polar)
        blocked = self._blocked

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
            elif (relation == "has_property" and tail and self.nouns
                    and not CONNECTIVES & set(tail.split())
                    and len(tail.split()) <= 3
                    and tail.strip() in self.nouns):
                # A mass noun takes no determiner and names a class all the
                # same: `is a chair furniture` was read as asking whether a
                # chair is furniture-ish, and answered by walking furniture's
                # own ancestors. The test is whether the predicate has a noun
                # sense -- `furniture` does, `telepathic` and `unmarried` do
                # not -- because the tagger calls all three nouns out of
                # context. A coordination is never one class, so `is a dog
                # furry or purple` stays the property question it is.
                relation, target, hedged = "is_a", tail, True
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
            note = (f"“{blocked}” is not a word this ontology holds, "
                    f"and it is what the question is about." if blocked else
                    "Could not find a noun to reason about.")

        return Parse(question=question, subject=subject, relation=relation,
                     target=target, polar=polar, backend=self.backend,
                     tokens=tokens, note=note, unknown=blocked, hedged=hedged)

    # -- matching a target phrase against a stored fact --------------------
    def matcher(self, threshold: float = 0.6):
        """Build a predicate deciding whether a fact answers the question.

        Facts from Ascent++ are free text -- "fall into hole", "fall asleep" --
        so this is lemma overlap rather than equality. The threshold is the
        share of the question's content lemmas that the fact must contain, so
        "fall down" matches "fall down the stairs" but not "fall in love".

        Two things the overlap alone got wrong, both found in the
        over-affirmation audit:

        * The shortcut for "the fact names the target outright" was a raw
          substring test, so `fly` was named by "attract butterfly", `walk` by
          "block the sidewalk", `run` by "get drunk" and `sing` by "go
          missing". `can a tree fly` came back VERIFIED because a plant
          attracts a butterfly. It is `whole_words` now.
        * Overlap is one-directional on purpose -- the fact may say more than
          the question -- but saying more can change the claim. "go for swim"
          carries `swim` and says nothing about a rock swimming. `plain`
          reports whether the fact states the target and nothing besides, and
          R28 is what does the refusing.
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
            if whole_words(target, fact_object):
                return 1.0          # the fact names the target outright
            return len(wanted & have) / len(wanted)

        def plain(fact_object: str, target: str | None) -> bool:
            """Does the fact state the target and add nothing to it?

            R28. "lay egg" answers `does a bird lay eggs`; "walk on land" does
            not answer `can a fish walk`, because the qualification is the
            whole of what makes it true. v687 already declines the mirror of
            this -- "denying a qualified property does not deny the property"
            -- and this is the same reading applied to yes.
            """
            if not target:
                return True
            wanted = lemma_set(target)
            return bool(wanted) and lemma_set(fact_object) <= wanted

        def matches(fact_object: str, target: str | None) -> bool:
            return score(fact_object, target) >= threshold

        matches.score = score
        matches.plain = plain
        matches.threshold = threshold
        return matches

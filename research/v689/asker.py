"""What a conversation needs from the layers below it.

Everything here is v687: its parser reads the words, its reasoner holds the
taxonomy and the store, its matcher decides whether a fact answers a question.
The one thing left abstract is `run` -- v688's loop on a question about a
kind -- because a test supplies a stand-in and the server supplies the loop.
"""
from __future__ import annotations


def _noun_lemma(word: str, read: str = "") -> str:
    """The noun a word alone is a form of, by WordNet: every form its
    morphology offers (`flies` is a noun of its own and a form of `fly`,
    `mice` only of `mouse`), and of those the one used most often, by
    WordNet's own counts -- so `flies` is a fly, and `physics` stays
    physics rather than becoming `physic`. Empty when WordNet has none."""
    try:
        from nltk.corpus import wordnet
        forms = set(wordnet._morphy(word, wordnet.NOUN))
    except Exception:                              # noqa: BLE001
        return ""
    if read and read != word:
        try:
            if wordnet.synsets(read, wordnet.NOUN):
                forms.add(read)
        except Exception:                          # noqa: BLE001
            pass
    if not forms:
        return ""

    def used(form: str) -> int:
        return sum(lemma.count() for synset in wordnet.synsets(
            form, wordnet.NOUN) for lemma in synset.lemmas()
            if lemma.name().lower() == form)

    return max(sorted(forms), key=lambda form: (used(form),
                                                form == word)).lower()


class Asker:
    """One v687 reasoner and parser, read through the questions v689 asks."""

    def __init__(self, reasoner, parser, matcher=None) -> None:
        self.reasoner = reasoner
        self.parser = parser
        self.matcher = matcher or parser.matcher()

    def subject(self, question: str):
        """What v687's parser reads a question as being about."""
        return self.parser.parse(question).subject

    def parse(self, question: str):
        """The relation and target v687 would put this question to."""
        return self.parser.parse(question)

    def lemma(self, word: str) -> str:
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None:
            return word
        read = nlp(word)
        if not len(read):
            return word
        token = read[0]
        if token.tag_.startswith("NN") and word.islower():
            # A word looked up alone, typed in lower case, read as a noun.
            # Alone it has no context, and the transformer model tags
            # `mice` as singular and `wembles` as a name, leaving both as
            # they are. WordNet's noun forms first (`_noun_lemma`), and for
            # a word WordNet does not have, spaCy's plural rules: `wembles`
            # is `wemble`.
            known = _noun_lemma(word, token.lemma_.lower())
            if known:
                return known
            lemmatizer = (nlp.get_pipe("lemmatizer")
                          if "lemmatizer" in nlp.pipe_names else None)
            rule = getattr(lemmatizer, "rule_lemmatize", None)
            if rule is not None and word.endswith("s"):
                from spacy.tokens import Doc
                alone = Doc(nlp.vocab, words=[word], pos=["NOUN"],
                            tags=["NNS"])
                found = rule(alone[0])
                if found and found[0]:
                    return found[0].lower()
        return token.lemma_.lower()

    def progressive(self, word: str):
        """The verb in `it is flying`; None for `it is boring`.

        Read in a frame rather than alone, because a bare `flying` tags as a
        noun and a bare lemma cannot tell a participle from an adjective.
        """
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not word.endswith("ing"):
            return None
        token = nlp(f"it is {word}")[-1]
        lemma = token.lemma_.lower()
        return lemma if token.pos_ == "VERB" and lemma != word else None

    def verb(self, word: str) -> str:
        """The lemma of a word used as a verb: `broke` -> break, read in
        `it ___` so a past tense is not taken for a noun."""
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not word:
            return word
        token = nlp(f"it {word}")[-1]
        return token.lemma_.lower() or word

    def participle(self, word: str):
        """The verb in `it has eaten`, `it was broken`; None for `it has
        four`. Read in a frame, as `progressive` is."""
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not word:
            return None
        token = nlp(f"it was {word}")[-1]
        # The tag decides, not the spelling: `run` and `put` are their own
        # participles.
        return token.lemma_.lower() if token.tag_ in ("VBN", "VBD") else None

    def verb_senses(self, lemma: str) -> list[str]:
        """The store's verb senses for a word, best first."""
        return [sense["id"] for sense in
                (self.reasoner.senses_of(lemma, "v") or [])]

    def verb_kinds(self, lemma: str, sense: str | None = None) -> list[str]:
        """Every verb a verb is a kind of, nearest first, as words: `chase`
        -> chase, pursue, follow, travel, go, move, locomote. Troponymy,
        walked up the store's taxonomy as R1 walks a noun. From `sense` when
        the sentence chose one (`change.senses_of`): passing a football to
        someone is giving it, not travelling."""
        senses = [sense] if sense else self.verb_senses(lemma)
        kinds = [lemma]
        if not senses:
            return kinds
        cache = self.__dict__.setdefault("_verb_kinds", {})
        if senses[0] not in cache:
            found = []
            for node, _, _ in self.reasoner.ascend(senses[0]):
                head = node.rsplit(".", 2)[0]
                rows = self.reasoner.connection.execute(
                    "SELECT lemma FROM lemmas WHERE concept = ?",
                    (node,)).fetchall()
                # The synset's own name first, then its other lemmas in a
                # fixed order: the store keeps them in none.
                found += [head] + sorted(row[0] for row in rows
                                         if row[0] != head)
            cache[senses[0]] = found
        return list(dict.fromkeys(kinds + cache[senses[0]]))

    def _parsed(self, words: list[str]):
        """spaCy over the words exactly as given, read as one sentence.

        Handed over already split, so token `i` is word `i`: `reading.py`
        expands contractions and strips punctuation first, and letting spaCy
        tokenize again would misalign the two.
        """
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not words:
            return None
        shared = getattr(nlp, "parse_words", None)
        if shared is not None:
            return shared(words)
        from spacy.tokens import Doc

        doc = Doc(nlp.vocab, words=list(words))
        for _, component in nlp.pipeline:
            doc = component(doc)
        return doc

    def tags(self, words: list[str]) -> list[str] | None:
        """Penn tags, in the context of the whole sentence."""
        doc = self._parsed(words)
        return [token.tag_ for token in doc] if doc is not None else None

    def analyse(self, words: list[str]) -> list[tuple] | None:
        """(tag, dependency, head index) per word: what `clauses.py` reads."""
        doc = self._parsed(words)
        return ([(token.tag_, token.dep_, token.head.i) for token in doc]
                if doc is not None else None)

    def words_of(self, text: str) -> list | None:
        """spaCy's own reading of a text, tokenized by spaCy: for glosses,
        where `short-legged` has to come apart at the hyphen."""
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not (text or "").strip():
            return None
        from .clauses import Word

        return [Word(token.i, token.text, token.tag_, token.dep_,
                     token.head.i, token.head.text, token.lemma_.lower())
                for token in nlp(text)]

    def antonyms(self, lemma: str) -> frozenset:
        """WordNet's antonyms of a word in any of its senses: `shrink` gives
        expand and stretch. Empty where WordNet is not installed."""
        cache = self.__dict__.setdefault("_antonyms", {})
        if lemma not in cache:
            try:
                from nltk.corpus import wordnet

                cache[lemma] = frozenset(
                    antonym.name().replace("_", " ")
                    for synset in wordnet.synsets(lemma)
                    for word in synset.lemmas()
                    for antonym in word.antonyms())
            except Exception:                   # noqa: BLE001
                cache[lemma] = frozenset()
        return cache[lemma]

    def gender(self, name: str) -> str:
        """`female` for Mary, `male` for John: NLTK's names corpus, and
        nothing where it lists a name as both or as neither -- Bill, Daniel,
        Sumit. A pronoun is matched against it (`discourse.py`), so a name
        it cannot place is never ruled out."""
        lists = self.__dict__.get("_names")
        if lists is None:
            try:
                from nltk.corpus import names

                lists = ({one.lower() for one in names.words("male.txt")},
                         {one.lower() for one in names.words("female.txt")})
            except Exception:                   # noqa: BLE001
                lists = (set(), set())
            self.__dict__["_names"] = lists
        male, female = lists
        word = (name or "").lower()
        if (word in male) == (word in female):
            return ""
        return "male" if word in male else "female"

    def known(self, phrase: str) -> bool:
        return phrase in (self.parser.nouns or self.parser.vocabulary or ())

    def sense(self, kind: str):
        """The synset a kind word is placed under: the reasoner's first noun
        reading, which is the one v687 itself would take."""
        senses = self.reasoner.senses_of(kind, "n") or []
        return senses[0]["id"] if senses else None

    def judge(self, question: str):
        """(supports, confidence) from v688's teacher, asked bare; None when
        there is no teacher. The server supplies one."""
        return None

    def run(self, question: str) -> dict:
        raise NotImplementedError("v688's loop is supplied by the caller")

"""Why someone does something, and where it takes them: the store's common
sense about motivation, read for the individuals of a conversation.

    Sumit is tired              sleeping has_prerequisite tired (ConceptNet);
                                tired has_subevent go to sleep
    Sumit went to the bedroom   bedroom used_for sleeping (ConceptNet)
    why did Sumit go there      because Sumit is tired: being tired moves one
                                to sleep, and a bedroom is for sleeping
    where will Sumit go         probably the bedroom, of the places here

A state moves one to what the store says it motivates (`motivated_by_goal`)
or is needed for (`has_prerequisite`), and to what it leads to itself
(`has_subevent`, `causes`) -- read for the adjective and for its noun, tired
and tiredness, which WordNet derives one from the other. A place or a thing is
for what its kind is `used_for`. The two meet on a content word they share,
by lemma: `go to sleep` and `sleeping` both come to `sleep`.

This is what the store holds, not anything told, so the session says it as a
reason with the rows it rests on: `probably` for where someone will go, since
nothing said they would, and `perhaps` where all there is to go on is the last
state told of them.
"""
from __future__ import annotations

from .episodic import name_of

#: Words too general to be what two phrases have in common.
GENERAL = frozenset({"a", "an", "the", "to", "in", "into", "of", "for", "and",
                     "at", "on", "with", "from", "by", "go", "get", "be",
                     "do", "have", "make", "find", "one", "someone", "thing",
                     "something", "place", "people", "person", "time", "way",
                     "use", "area"})

#: Rows whose object is the state: what it moves one to do.
MOVES = ("motivated_by_goal", "has_prerequisite")

#: Rows whose subject is the state: what it leads to.
LEADS = ("has_subevent", "causes")


def nouns(state: str) -> list[str]:
    """The nouns WordNet derives from an adjective and that keep its stem:
    tired -> tiredness, hungry -> hunger."""
    try:
        from nltk.corpus import wordnet
    except Exception:                               # noqa: BLE001
        return []
    found: list[str] = []
    for synset in wordnet.synsets(state):
        if synset.pos() not in ("a", "s"):
            continue
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                name = related.name().replace("_", " ")
                if (related.synset().pos() == "n" and name[:4] == state[:4]
                        and name not in found):
                    found.append(name)
    return found


class Motives:
    """The store's motivations and purposes, cached for one conversation."""

    def __init__(self, asker) -> None:
        self.asker = asker
        self._lemma: dict[str, str] = {}
        self._goals: dict[str, list] = {}
        self._purposes: dict[str, list] = {}

    def lemmas(self, text: str) -> set[str]:
        """The content words of a phrase, as lemmas."""
        out = set()
        for word in (text or "").lower().replace("_", " ").split():
            if word in GENERAL:
                continue
            if word not in self._lemma:
                self._lemma[word] = self.asker.lemma(word)
            if self._lemma[word] not in GENERAL:
                out.add(self._lemma[word])
        return out

    def goals(self, state: str) -> list[tuple[str, str]]:
        """(what a state moves one to, the row that says so)."""
        state = (state or "").lower()
        if state in self._goals:
            return self._goals[state]
        read = self.asker.reasoner.connection.execute
        words = [state] + nouns(state)
        marks = ",".join("?" * len(words))
        found = []
        for concept, relation, obj in read(
                f"SELECT concept, relation, object FROM facts "
                f"WHERE object IN ({marks}) AND relation IN (?, ?)",
                (*words, *MOVES)):
            found.append((name_of(concept),
                          f"{name_of(concept)} {relation} {obj}"))
        for concept, relation, obj in read(
                f"SELECT f.concept, f.relation, f.object FROM facts f "
                f"JOIN lemmas l ON l.concept = f.concept "
                f"WHERE l.lemma IN ({marks}) AND f.relation IN (?, ?)",
                (*words, *LEADS)):
            found.append((obj, f"{name_of(concept)} {relation} {obj}"))
        self._goals[state] = found
        return found

    def purposes(self, kind: str) -> list[str]:
        """What a kind is for, at the sense it is placed under."""
        if kind not in self._purposes:
            node = self.asker.sense(kind)
            rows = (self.asker.reasoner.connection.execute(
                "SELECT object FROM facts WHERE concept = ? "
                "AND relation = 'used_for'", (node,)).fetchall()
                if node else [])
            self._purposes[kind] = [row[0] for row in rows]
        return self._purposes[kind]

    def meeting(self, state: str, kind: str) -> tuple[str, str, str] | None:
        """(goal, row, purpose) where what a state moves one to and what a
        kind is for share a word: tired and a bedroom meet on sleep."""
        purposes = [(one, self.lemmas(one)) for one in self.purposes(kind)]
        for goal, row in self.goals(state):
            wanted = self.lemmas(goal)
            for purpose, words in purposes:
                if wanted & words:
                    return goal, row, purpose
        return None

    def support(self, state: str, kind: str) -> list[str]:
        """Every purpose of a kind that something the state moves one to
        shares a word with."""
        wanted = set()
        for goal, _ in self.goals(state):
            wanted |= self.lemmas(goal)
        return [one for one in self.purposes(kind)
                if wanted & self.lemmas(one)]

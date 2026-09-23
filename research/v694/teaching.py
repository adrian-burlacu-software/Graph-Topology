"""Ways and recipes, taught in English or seen done.

    you can cut the rope with a saw          a way: cut rope, tool, saw
    use a saw to cut the rope                the same way, said as advice
    i cut the rope with a saw                the same, seen done
    you can make a torch from a stick and a cloth     a recipe
    a torch is made from a stick and a cloth          the same recipe
    i made a raft from the logs and the rope          a recipe, seen done

Read off v691's parse (`hearing.parse`), by the shape of the sentence and
not by its words: a verb with an object and a `with` phrase is a way to do
it; `make` or `build` with an object and a `from` phrase is a recipe. The
sentence has to be one that teaches -- said of someone in general (*you
can*, *one can*), as advice (*use X to*), as what was done (*i cut*), or
as what a thing is made from (*is made from*) -- so an order (*cut the
rope with a saw*) stays an order.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v691 import hearing

#: Who a general statement of ability is about.
ANYONE = frozenset({"you", "one", "we", "people", "someone", "i"})
#: The auxiliaries that say it can be done, not that it is to be done now.
ABLE = frozenset({"can", "could", "may", "might"})
#: Verbs whose object is something made, and whose `from` phrase is what
#: it is made from.
MAKING = frozenset({"make", "build", "craft", "construct", "assemble"})
#: What a part list is introduced by.
FROM = frozenset({"from", "out"})


@dataclass(frozen=True)
class Taught:
    #: way | recipe
    kind: str
    #: for a way: the predicate and the thing it is done to
    predicate: str = ""
    patient: str = ""
    means: str = ""
    #: for a recipe
    product: str = ""
    parts: tuple = ()
    #: taught (said in general) or seen (said as done)
    seen: bool = False

    def said(self) -> str:
        if self.kind == "recipe":
            return (f"{_a(self.product)} can be made from "
                    + " and ".join(_a(one) for one in self.parts))
        return (f"{_a(self.means)} can {self.predicate} "
                f"{_a(self.patient)}")


def _a(name: str) -> str:
    said = name.replace("-", " ")
    return ("an " if said[:1] in "aeiou" else "a ") + said


def _noun(words, index: int) -> str:
    word = words[index]
    return word.lemma if word.tag in ("NN", "NNS") else ""


def _objects(words, index: int) -> list:
    """The noun at `index` and every noun joined to it by `and`."""
    out = [_noun(words, index)]
    out += [_noun(words, one.index) for one in hearing.children(
        words, index, {"conj"})]
    return [one for one in out if one]


def _phrase(words, verb: int, prepositions) -> list:
    for prep in hearing.children(words, verb, {"prep"}):
        if prep.text not in prepositions:
            continue
        for obj in hearing.children(words, prep.index, {"pobj"}):
            return _objects(words, obj.index)
        # `out of the logs`: `of` hangs off `out`.
        for inner in hearing.children(words, prep.index, {"prep"}):
            for obj in hearing.children(words, inner.index, {"pobj"}):
                return _objects(words, obj.index)
    return []


def read(text: str) -> Taught | None:
    """What an utterance teaches, or None."""
    words = hearing.parse(text)
    if not words or text.rstrip().endswith("?") or words[0].tag == "MD":
        # `can you cut the rope with a saw` asks for it done.
        return None
    root = next((one for one in words if one.dep == "ROOT"), None)
    if root is None or not root.tag.startswith("VB"):
        return None
    subject = next(iter(hearing.children(words, root.index,
                                         {"nsubj", "nsubjpass"})), None)
    aux = {one.text for one in hearing.children(words, root.index,
                                                {"aux", "auxpass"})}
    obj = next(iter(hearing.children(words, root.index, {"dobj"})), None)
    general = subject is not None and subject.text in ANYONE and (
        aux & ABLE or root.tag == "VBD")
    seen = root.tag == "VBD" and subject is not None
    # `use a saw to cut the rope`: advice, the tool first.
    if root.lemma == "use" and subject is None and obj is not None:
        for then in hearing.children(words, root.index, {"xcomp",
                                                          "advcl"}):
            thing = next(iter(hearing.children(words, then.index,
                                               {"dobj"})), None)
            means, patient = _noun(words, obj.index), \
                _noun(words, thing.index) if thing is not None else ""
            if means and patient:
                return Taught("way", then.lemma, patient, means)
        return None
    # `a torch is made from a stick and a cloth`
    if root.lemma in MAKING and subject is not None and \
            subject.dep == "nsubjpass":
        parts = _phrase(words, root.index, FROM)
        product = _noun(words, subject.index)
        if product and parts:
            return Taught("recipe", product=product, parts=tuple(parts))
        return None
    if not general or obj is None:
        return None
    if root.lemma in MAKING:
        parts = _phrase(words, root.index, FROM)
        product = _noun(words, obj.index)
        if product and parts:
            return Taught("recipe", product=product, parts=tuple(parts),
                          seen=seen)
        return None
    means = _phrase(words, root.index, {"with", "using"})
    patient = _noun(words, obj.index)
    if means and patient:
        return Taught("way", root.lemma, patient, means[0], seen=seen)
    return None


def learn(taught: Taught, memory) -> str:
    """Keep what was taught in `memory` (`carrying.Remembered`), and say
    what was kept."""
    from research.v694 import knowing as K
    how = "you showed me" if taught.seen else "you told me"
    if taught.kind == "recipe":
        memory.learn_recipe(K.name_of(taught.product),
                            [K.name_of(one) for one in taught.parts],
                            said=f"{how} how")
    else:
        memory.remember(taught.predicate, K.name_of(taught.patient), "tool",
                        K.name_of(taught.means), said=how)
    return f"I will remember that {taught.said()}."


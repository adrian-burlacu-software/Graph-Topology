
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import re

class CleanTokenizer:
    TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+")

    def tokenize(self,text):
        return [
            t.lower()
            for t in self.TOKEN_RE.findall(text)
            if self.is_lexical(t)
        ]

    def is_lexical(self,token):
        token=token.strip().lower()
        if not token or token.isdigit():
            return False
        letters=re.sub(r"[^a-z]","",token)
        return (
            len(letters)>=2
            and re.match(
                r"^[A-Za-z]+(?:['’-][A-Za-z]+)*$",
                token,
            ) is not None
            and token[0].isalpha()
            and token[-1].isalpha()
        )


@dataclass(frozen=True)
class GrammarHypothesis:
    hypothesis_id: str
    construction: Tuple[str, ...]
    semantic_roles: Tuple[str, ...]
    prior: float
    support: int = 0
    contradictions: int = 0

    @property
    def log_score(self):
        return (
            math.log(max(self.prior, 1e-12))
            + 1.25 * self.support
            - 2.75 * self.contradictions
            - 0.05 * len(self.construction)
        )


@dataclass(frozen=True)
class GrammarBelief:
    surface: str
    candidates: Tuple[GrammarHypothesis, ...]
    committed: Optional[str]
    confidence: float
    entropy: float
    revision: int


class GrammarHypothesisSpace:
    """
    Small explicit hypothesis space over recurring surface constructions.

    The important architectural contract is that a grammar is represented as
    hypotheses with evidence, not as opaque parser weights.
    """

    def __init__(self):
        self.tokenizer=CleanTokenizer()
        self.hypotheses: Dict[str, GrammarHypothesis]={}

    def generate(self, sentence: str):
        tokens=self.tokenizer.tokenize(sentence)
        tags=[]

        for token in tokens:
            if token in {"the","a","an"}:
                tags.append("DET")
            elif token.endswith("ing") or token.endswith("ed") or token in {
                "chases","chase","eats","eat","sees","see","likes","like"
            }:
                tags.append("VERB")
            elif token in {
                "is","are","was","were","am","be",
                "do","does","did","have","has","had",
            }:
                tags.append("AUX")
            elif token.endswith("ly"):
                tags.append("ADV")
            else:
                tags.append("NOUN")

        candidates=[]

        if len(tags)>=3:
            for i in range(len(tags)-2):
                tri=tuple(tags[i:i+3])
                if tri == ("DET","NOUN","VERB"):
                    candidates.append(
                        self._register(
                            ("DET","NOUN","VERB"),
                            ("ARG0","ENTITY","PRED"),
                        )
                    )
                elif tri == ("NOUN","VERB","DET"):
                    candidates.append(
                        self._register(
                            ("NOUN","VERB","DET"),
                            ("ARG0","PRED","ARG1"),
                        )
                    )

        if ("DET","NOUN") in zip(tags,tags[1:]):
            candidates.append(
                self._register(
                    ("DET","NOUN"),
                    ("DETERMINER","ENTITY"),
                )
            )

        return tuple(candidates)

    def _register(self, construction, roles):
        key="|".join(construction)+"::"+"|".join(roles)

        hypothesis=self.hypotheses.get(key)
        if hypothesis is None:
            hypothesis=GrammarHypothesis(
                hypothesis_id=key,
                construction=construction,
                semantic_roles=roles,
                prior=1.0/(1.0+len(construction)),
            )
            self.hypotheses[key]=hypothesis

        return hypothesis


class GrammarGroundingInterface:
    """
    Connects grammar hypotheses to the existing semantic grounding
    architecture.

    This intentionally uses semantic context to test whether a grammatical
    interpretation is compatible with a semantic graph state.
    """

    def __init__(self, semantic_architecture):
        self.semantic_architecture=semantic_architecture
        self._tokenizer=CleanTokenizer()
        self.evidence=[]

    def interpret_sentence(
        self,
        sentence: str,
        hypotheses: Iterable[GrammarHypothesis],
        semantic_queries: Tuple[
            Tuple[str, Tuple[Tuple[str,str],...]],
            ...
        ],
    ):
        hypotheses=tuple(hypotheses)
        scored=[]

        tokens=self._tokenizer.tokenize(sentence)
        observed_tags=[]
        for token in tokens:
            if token in {"the","a","an"}:
                observed_tags.append("DET")
            elif token.endswith("ing") or token.endswith("ed") or token in {
                "chases","chase","eats","eat","sees","see","likes","like"
            }:
                observed_tags.append("VERB")
            elif token.endswith("ly"):
                observed_tags.append("ADV")
            else:
                observed_tags.append("NOUN")

        for hypothesis in hypotheses:
            support=0
            contradiction=0

            semantic_commits=0
            for query, context in semantic_queries:
                state=self.semantic_architecture.perceive(
                    query,
                    context=context,
                )
                if state.committed is not None:
                    semantic_commits+=1

            # Structural compatibility is part of the evidence: the number of
            # semantic slots a hypothesis exposes should agree with the number
            # of grounded semantic items supplied by the sentence. This keeps
            # a shorter NP hypothesis from receiving identical credit to a
            # complete clause hypothesis.
            slot_gap=abs(
                len(hypothesis.semantic_roles)
                - len(semantic_queries)
            )
            structural_match=int(
                tuple(hypothesis.construction)
                ==tuple(observed_tags[:len(hypothesis.construction)])
            )
            support=max(
                0,
                semantic_commits - slot_gap
            ) + 4*structural_match

            exact_structure = (
                tuple(hypothesis.construction)
                == tuple(observed_tags[:len(hypothesis.construction)])
            )
            contradiction = (
                1 if slot_gap > 1 else 0
            ) + (
                2 if not exact_structure else 0
            )

            scored.append(
                GrammarHypothesis(
                    hypothesis_id=hypothesis.hypothesis_id,
                    construction=hypothesis.construction,
                    semantic_roles=hypothesis.semantic_roles,
                    prior=hypothesis.prior,
                    support=support,
                    contradictions=contradiction,
                )
            )

        if not scored:
            return None

        mx=max(h.log_score for h in scored)
        weights=[math.exp(h.log_score-mx) for h in scored]
        z=sum(weights) or 1.0

        posterior=sorted(
            zip(scored,weights),
            key=lambda x:x[1],
            reverse=True,
        )

        probs=[
            (h,w/z)
            for h,w in posterior
        ]
        entropy=-sum(
            p*math.log(max(p,1e-12))
            for _,p in probs
        )

        best=probs[0]
        second=probs[1] if len(probs)>1 else None
        confidence=best[1]
        margin=confidence-(second[1] if second else 0.0)

        committed=(
            best[0].hypothesis_id
            if confidence>=0.70 and margin>=0.20
            else None
        )

        belief=GrammarBelief(
            surface=sentence,
            candidates=tuple(
                h for h,_ in probs
            ),
            committed=committed,
            confidence=confidence,
            entropy=entropy,
            revision=1,
        )

        self.evidence.append(
            {
                "sentence":sentence,
                "committed":committed,
                "confidence":confidence,
            }
        )

        return belief


@dataclass(frozen=True)
class SemanticSentenceExample:
    sentence: str
    queries: Tuple[
        Tuple[str, Tuple[Tuple[str,str],...]],
        ...
    ]
    expected_construction: str


def smoke_examples():
    return (
        SemanticSentenceExample(
            "the dog chases the cat",
            (
                ("dog",(("IsA","animal"),)),
                ("chases",(("RelatedTo","pursuit"),)),
                ("cat",(("IsA","animal"),)),
            ),
            "DET|NOUN|VERB",
        ),
        SemanticSentenceExample(
            "the cat eats",
            (
                ("cat",(("IsA","animal"),)),
                ("eats",(("RelatedTo","food"),)),
            ),
            "DET|NOUN",
        ),
    )

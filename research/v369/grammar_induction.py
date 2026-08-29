
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import math
import re
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class TokenObservation:
    token: str
    position: int
    sentence_id: int


@dataclass(frozen=True)
class Construction:
    lhs: str
    rhs: Tuple[str, ...]
    count: int
    probability: float
    evidence_sentences: int
    semantic_anchors: Tuple[str, ...] = ()


@dataclass
class InducedGrammar:
    lexicon: Dict[str, str]
    constructions: List[Construction]
    vocabulary_size: int
    sentences: int
    tokens: int
    heldout_sentences: int
    source: str
    semantic_groundings: Dict[str, List[dict]] = field(default_factory=dict)

    def rule_count(self):
        return len(self.constructions)

    def top_rules(self, n=20):
        return sorted(
            self.constructions,
            key=lambda x:(x.count,x.probability),
            reverse=True,
        )[:n]


class CorpusReader:
    SUPPORTED={".txt",".text",".jsonl",".json",".tsv",".csv"}

    def files(self, path: Path) -> List[Path]:
        path=Path(path)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in self.SUPPORTED
            )
        raise FileNotFoundError(path)

    def lines(self, path: Path, limit=None):
        count=0
        for file in self.files(path):
            suffix=file.suffix.lower()

            if suffix in {".txt",".text",".tsv",".csv"}:
                with file.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    for line in f:
                        line=line.strip()
                        if line:
                            yield line
                            count+=1
                            if limit and count>=limit:
                                return
                continue

            if suffix==".jsonl":
                import json
                with file.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    for raw in f:
                        try:
                            obj=json.loads(raw)
                        except Exception:
                            continue
                        text=None
                        if isinstance(obj,str):
                            text=obj
                        elif isinstance(obj,dict):
                            for k in ("text","sentence","content"):
                                if isinstance(obj.get(k),str):
                                    text=obj[k]
                                    break
                        if text and text.strip():
                            yield text.strip()
                            count+=1
                            if limit and count>=limit:
                                return
                continue

            if suffix==".json":
                import json
                try:
                    data=json.loads(
                        file.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                    )
                except Exception:
                    continue

                def emit(x):
                    if isinstance(x,str):
                        return x.strip()
                    if isinstance(x,dict):
                        for k in ("text","sentence","content"):
                            if isinstance(x.get(k),str):
                                return x[k].strip()
                    return None

                if isinstance(data,list):
                    iterable=data
                else:
                    iterable=[
                        data.get(k)
                        for k in (
                            "text","sentences","data","examples"
                        )
                        if k in data
                    ]

                for item in iterable:
                    if isinstance(item,list):
                        for sub in item:
                            text=emit(sub)
                            if text:
                                yield text
                                count+=1
                    else:
                        text=emit(item)
                        if text:
                            yield text
                            count+=1
                    if limit and count>=limit:
                        return


class GrammarInducer:
    """
    Lightweight unsupervised grammar induction baseline.

    It learns recurrent lexical/context constructions from the raw corpus. It
    deliberately does not pretend this is a full parser; the experiment is
    designed to measure whether raw language exposure produces reusable
    grammatical constructions that can then be semantically grounded.
    """

    def __init__(self, min_count=3):
        self.min_count=min_count

    def classify(self, token: str) -> str:
        token=token.lower()

        if token in {"the","a","an"}:
            return "DET"
        if token in {
            "is","are","was","were","am","be","been","being",
            "do","does","did","have","has","had",
            "can","could","will","would","should","may","might",
            "must",
        }:
            return "AUX"
        if token.endswith("ing"):
            return "VERB"
        if token.endswith("ed"):
            return "VERB"
        if token.endswith("ly"):
            return "ADV"
        if token.endswith(("ous","ful","able","ive","al","ic")):
            return "ADJ"
        if token.isdigit():
            return "NUM"
        return "NOUN"

    def induce(
        self,
        corpus: Path,
        train_limit: int | None = None,
        heldout_limit: int = 5000,
        semantic_graph=None,
        source_name: str | None = None,
    ) -> tuple[InducedGrammar, List[str]]:
        reader=CorpusReader()
        lines=list(
            reader.lines(
                corpus,
                limit=(
                    None if train_limit is None
                    else train_limit+heldout_limit
                ),
            )
        )

        if heldout_limit:
            heldout=lines[-heldout_limit:]
            train=lines[:-heldout_limit]
        else:
            heldout=[]
            train=lines

        lexicon={}
        construction_counts=Counter()
        construction_sentences=defaultdict(set)
        anchor_words=defaultdict(set)
        token_count=0

        for sid,line in enumerate(train):
            tokens=[
                x.lower()
                for x in re.findall(
                    r"[A-Za-z0-9']+",
                    line,
                )
            ]
            if not tokens:
                continue

            tags=[]
            for token in tokens:
                cat=self.classify(token)
                lexicon.setdefault(token,cat)
                tags.append(cat)
                token_count+=1

            # Local POS n-grams.
            for n in (2,3,4):
                for i in range(
                    max(0,len(tags)-n+1)
                ):
                    rhs=tuple(tags[i:i+n])
                    construction_counts[
                        rhs
                    ]+=1
                    construction_sentences[
                        rhs
                    ].add(sid)

                    # Lexical semantic anchors are preserved so later
                    # grounding can associate constructions with ConceptNet.
                    lexical_tokens=tokens[i:i+n]
                    for tok in lexical_tokens:
                        if tok in {
                            "the","a","an",
                            "is","are","was","were",
                        }:
                            continue
                        anchor_words[rhs].add(tok)

            # Determiner-noun and simple clause templates.
            for i in range(len(tags)-1):
                if tags[i:i+2]==["DET","NOUN"]:
                    rhs=("DET","NOUN")
                    construction_counts[rhs]+=1
                    construction_sentences[rhs].add(sid)

            if len(tags)>=3:
                for i in range(len(tags)-2):
                    if tags[i]=="DET" and tags[i+1]=="NOUN" and tags[i+2]=="VERB":
                        rhs=("DET","NOUN","VERB")
                        construction_counts[rhs]+=1
                        construction_sentences[rhs].add(sid)

        total=sum(construction_counts.values()) or 1

        constructions=[]
        for rhs,count in construction_counts.items():
            if count<self.min_count:
                continue
            constructions.append(
                Construction(
                    lhs=(
                        "NP" if rhs==("DET","NOUN")
                        else "VP" if rhs[0] in ("VERB","AUX")
                        else "SLOT"
                    ),
                    rhs=rhs,
                    count=count,
                    probability=count/total,
                    evidence_sentences=len(
                        construction_sentences[rhs]
                    ),
                    semantic_anchors=tuple(
                        sorted(anchor_words[rhs])[:12]
                    ),
                )
            )

        grammar=InducedGrammar(
            lexicon=lexicon,
            constructions=constructions,
            vocabulary_size=len(lexicon),
            sentences=len(train),
            tokens=token_count,
            heldout_sentences=len(heldout),
            source=source_name or str(corpus),
        )
        return grammar,heldout


def evaluate_grammar(grammar: InducedGrammar, heldout: Iterable[str]):
    heldout=list(heldout)

    tag_sequence_hits=0
    det_noun_hits=0
    clause_hits=0
    total_nonempty=0

    known_ngrams={
        c.rhs
        for c in grammar.constructions
    }

    for line in heldout:
        tokens=[
            t.lower()
            for t in re.findall(
                r"[A-Za-z0-9']+",
                line,
            )
        ]
        if not tokens:
            continue
        total_nonempty+=1

        tags=[
            grammar.lexicon.get(
                tok,
                "NOUN",
            )
            for tok in tokens
        ]

        seen=set()
        for n in (2,3,4):
            for i in range(len(tags)-n+1):
                seen.add(tuple(tags[i:i+n]))

        if seen & known_ngrams:
            tag_sequence_hits+=1

        if (
            any(
                tags[i:i+2]==["DET","NOUN"]
                for i in range(len(tags)-1)
            )
            and ("DET","NOUN") in known_ngrams
        ):
            det_noun_hits+=1

        if any(
            tags[i:i+3]==["DET","NOUN","VERB"]
            for i in range(len(tags)-2)
        ):
            clause_hits+=1

    denom=max(1,total_nonempty)

    return {
        "heldout_sentences":total_nonempty,
        "construction_recall":tag_sequence_hits/denom,
        "det_noun_recall":det_noun_hits/denom,
        "det_noun_verb_rate":clause_hits/denom,
    }


from __future__ import annotations

import gzip
import json

from ontology import BLOCKED_RELATIONS


CN_REL_MAP={
    "IsA":"is_a","HasA":"has","PartOf":"part_of","HasPart":"contains",
    "UsedFor":"used_for","CapableOf":"capable_of","Causes":"causes",
    "CausesDesire":"causes_desire","CreatedBy":"created_by",
    "DefinedAs":"defined_as","DerivedFrom":"derived_from","Desires":"desires",
    "LocatedNear":"located_near","AtLocation":"located_in","MadeOf":"made_of",
    "HasProperty":"has_property","ReceivesAction":"receives_action",
    "Synonym":"synonym","Antonym":"antonym","SimilarTo":"similar_to",
    "RelatedTo":"related_to","FormOf":"form_of","MannerOf":"manner_of",
    "InstanceOf":"instance_of","Entails":"entails",
    "DistinctFrom":"distinct_from","SymbolOf":"symbol_of",
    "MotivatedByGoal":"motivated_by_goal",
}


def _cn_term(uri):
    parts=str(uri).split("/")
    try:
        i=parts.index("c")
    except ValueError:
        return None
    if len(parts)<=i+2 or parts[i+1]!="en":
        return None
    return parts[i+2].replace("_"," ").strip() or None


def _open_text(path):
    with open(path,"rb") as f:
        magic=f.read(2)
    if magic==b"\x1f\x8b":
        return gzip.open(path,"rt",encoding="utf-8",errors="replace")
    return open(path,"rt",encoding="utf-8",errors="replace")


def iter_conceptnet(path):
    with _open_text(path) as f:
        for line_no,line in enumerate(f,1):
            parts=line.rstrip("\n").split("\t")
            if len(parts)<5:
                continue
            edge_uri,relation,start_uri,end_uri,raw_meta=parts[:5]
            start=_cn_term(start_uri)
            end=_cn_term(end_uri)
            if not start or not end:
                continue
            rel=CN_REL_MAP.get(
                relation.rsplit("/",1)[-1],
                relation.rsplit("/",1)[-1].lower(),
            )
            try:
                meta=json.loads(raw_meta)
            except Exception:
                meta={}
            try:
                weight=float(meta.get("weight",1.0) or 1.0)
            except Exception:
                weight=1.0
            yield {
                "subject":start,
                "predicate":rel,
                "object":end,
                "fact_type":"semantic",
                "domain":None,
                "confidence":min(1.0,0.25+0.15*weight),
                "frequency":max(1.0,weight),
                "answerable":rel not in BLOCKED_RELATIONS,
                "record_key":str(line_no),
                "metadata":{
                    "edge_uri":edge_uri,
                    "conceptnet_relation":relation,
                    "conceptnet_dataset":meta.get("dataset"),
                    "raw_weight":weight,
                },
            }


def iter_wordnet():
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise RuntimeError(
            "Install NLTK: python -m pip install -U nltk"
        ) from exc

    try:
        wn.synsets("dog")
    except LookupError:
        import nltk
        print("[WORDNET] corpus missing; downloading...",flush=True)
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        wn.synsets("dog")

    for syn in wn.all_synsets():
        sid=syn.name()
        words=list(dict.fromkeys(
            lemma.name().replace("_"," ")
            for lemma in syn.lemmas()
        ))
        if not words:
            continue

        for i,word in enumerate(words):
            for other in words[i+1:]:
                yield {
                    "subject":word,
                    "predicate":"synonym",
                    "object":other,
                    "fact_type":"lexical",
                    "domain":syn.lexname(),
                    "confidence":1.0,
                    "frequency":1.0,
                    "answerable":True,
                    "record_key":f"{sid}:syn:{i}:{other}",
                    "metadata":{
                        "synset":sid,
                        "definition":syn.definition(),
                        "pos":syn.pos(),
                    },
                }

        definition=syn.definition().strip()
        for word in words:
            if definition:
                yield {
                    "subject":word,
                    "predicate":"defined_as",
                    "object":definition,
                    "fact_type":"lexical",
                    "domain":syn.lexname(),
                    "confidence":0.95,
                    "frequency":1.0,
                    "answerable":True,
                    "record_key":f"{sid}:def:{word}",
                    "metadata":{"synset":sid,"definition":definition},
                }

            for hyper in syn.hypernyms():
                if hyper.lemmas():
                    yield {
                        "subject":word,
                        "predicate":"hypernym",
                        "object":hyper.lemmas()[0].name().replace("_"," "),
                        "fact_type":"lexical",
                        "domain":syn.lexname(),
                        "confidence":1.0,
                        "frequency":1.0,
                        "answerable":True,
                        "record_key":f"{sid}:hyper:{word}:{hyper.name()}",
                        "metadata":{"synset":sid},
                    }

            for hypo in syn.hyponyms():
                if hypo.lemmas():
                    yield {
                        "subject":word,
                        "predicate":"hyponym",
                        "object":hypo.lemmas()[0].name().replace("_"," "),
                        "fact_type":"lexical",
                        "domain":syn.lexname(),
                        "confidence":1.0,
                        "frequency":1.0,
                        "answerable":True,
                        "record_key":f"{sid}:hypo:{word}:{hypo.name()}",
                        "metadata":{"synset":sid},
                    }

            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    yield {
                        "subject":word,
                        "predicate":"antonym",
                        "object":ant.name().replace("_"," "),
                        "fact_type":"lexical",
                        "domain":syn.lexname(),
                        "confidence":1.0,
                        "frequency":1.0,
                        "answerable":True,
                        "record_key":f"{sid}:ant:{word}:{ant.name()}",
                        "metadata":{"synset":sid},
                    }

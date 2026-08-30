
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactType:
    name: str
    answerable: bool
    weight: float


FACT_TYPES = {
    "semantic": FactType("semantic",True,1.0),
    "lexical": FactType("lexical",True,0.95),
    "dialogue": FactType("dialogue",True,0.90),
    "procedural": FactType("procedural",True,0.90),
    "state": FactType("state",True,1.20),
    "corpus_metadata": FactType("corpus_metadata",False,0.0),
    "grammar": FactType("grammar",False,0.0),
    "provenance": FactType("provenance",False,0.0),
    "domain_metadata": FactType("domain_metadata",False,0.0),
    "unknown": FactType("unknown",False,0.0),
}


RELATION_TYPES = {
    "is","has","has_property","color","colour","size","shape","member_of_verb_class",
    "means","synonym","antonym","hypernym","hyponym",
    "causes","used_for","capable_of","part_of","contains",
    "located_in","lives_in","created_by","creates","produces",
    "likes","desires","receives","related_to","entails",
    "example_of","next_reply","response_to","intent_of",
}


BLOCKED_RELATIONS = {
    "in_domain","domain","has_domain","belongs_to","member_of",
    "source","provenance","dataset","node_type","type","label",
    "subject","object","nsubj","nsubjpass","obj","dobj","iobj",
    "ccomp","xcomp","amod","advmod","nmod","obl","oblique",
    "root","dep","aux","auxpass","cop","det","case","mark",
    "punct","conj","cc","compound","appos","acl","advcl",
    "class","class_of","category","subcategory","tag",
}


GOAL_FACT_PREFERENCES = {
    "request_information": {
        "semantic":1.0,"lexical":0.8,"dialogue":0.5,"state":1.2,
    },
    "request_explanation": {
        "semantic":1.2,"lexical":0.5,"dialogue":0.4,"state":1.2,
    },
    "request_generation": {
        "procedural":1.0,"dialogue":0.7,"semantic":0.4,"state":0.8,
    },
    "challenge_claim": {
        "semantic":1.1,"state":1.4,"dialogue":0.6,"lexical":0.4,
    },
    "explore_assistant": {
        "state":1.4,"dialogue":0.8,"semantic":0.3,
    },
    "continue_conversation": {
        "state":1.2,"dialogue":1.0,"semantic":0.4,
    },
}


from __future__ import annotations

from model_utils import token_set
from ontology import BLOCKED_RELATIONS


def rank_static_facts(
    facts,
    goal_name,
    query_terms,
    topic=None,
    max_items=12,
):
    q=token_set(" ".join(query_terms))
    ranked=[]

    for fact in facts:
        relation=str(fact.get("predicate","")).lower()
        if relation in BLOCKED_RELATIONS:
            continue

        subject=token_set(str(fact.get("subject","")))
        obj=token_set(str(fact.get("object_text","")))
        overlap=len(q & (subject|obj))

        context_bonus=0.0
        if topic and topic.lower() in subject|obj:
            context_bonus+=3.0

        score=float(fact.get("relevance",0.0))
        score+=min(3.0,float(overlap))
        score+=context_bonus

        fact["relevance_final"]=score
        ranked.append(fact)

    ranked.sort(key=lambda x:x["relevance_final"],reverse=True)
    return ranked[:max_items]


def relation_text(fact):
    return (
        f"{fact['subject']} "
        f"{str(fact['predicate']).replace('_',' ')} "
        f"{fact['object_text']}"
    )

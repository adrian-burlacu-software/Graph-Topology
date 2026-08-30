
from __future__ import annotations

from query_target import QueryTarget
from model_utils import token_set


PROPERTY_HINTS={
    "color":{
        "red","orange","yellow","green","blue","purple","violet","pink",
        "brown","black","white","gray","grey","gold","silver",
    },
    "size":{
        "big","small","large","little","tiny","huge","vast","enormous",
        "massive","long","short","wide","narrow",
    },
    "shape":{
        "round","square","flat","spherical","circular","rectangular",
    },
    "location":{
        "at","located","location","near","in","on",
    },
    "time":{
        "before","after","during","year","old","recent","ancient",
    },
}



def target_attribute_matches(attribute,target):
    requested=(target.attribute or "").lower()
    if attribute=="color":
        return requested in {"color","colour"}
    if attribute=="size":
        return requested in {
            "size","big","small","large","tiny","huge",
            "little","vast","enormous","massive",
        }
    if attribute=="shape":
        return requested in {
            "shape","round","square","flat","spherical",
        }
    return requested==attribute

def _relation_rank(kind,fact):
    pred=str(fact.get("predicate","")).lower()
    ftype=str(fact.get("fact_type","")).lower()

    if kind=="property":
        # Property facts are only relevant after matching the requested
        # attribute/value. Merely being a `has_property` edge is insufficient.
        if pred in {
            "located_in","located_near","at_location",
        }:
            return 95

        if pred in {
            "color","colour","has_color",
        }:
            return 95 if target_attribute_matches("color", target) else 0

        if pred in {
            "has_size","size",
        }:
            return 95 if target_attribute_matches("size", target) else 0

        if pred in {
            "has_shape","shape",
        }:
            return 95 if target_attribute_matches("shape", target) else 0

        if pred=="has_property":
            return 0

        return 0


    if kind=="definition":
        if pred=="defined_as":
            return 100
        if pred=="is_a":
            return 85
        if pred=="has_property":
            return 70
        if pred=="hypernym":
            return 15
        return 25

    if kind=="general":
        if pred=="defined_as":
            return 95
        if pred=="has_property":
            return 90
        if pred=="is_a":
            return 80
        if pred in {"related_to","capable_of","used_for"}:
            return 55
        if ftype=="lexical":
            return 20
        return 10

    return 0


def select_evidence(facts,target,max_items=4):
    selected=[]

    for fact in facts:
        subject=str(fact.get("subject","")).lower()
        pred=str(fact.get("predicate","")).lower()
        obj=str(fact.get("object_text","")).lower()

        if target.subject and subject!=target.subject.lower():
            continue

        base=_relation_rank(target.kind,fact)

        if target.kind=="property":
            attr=(target.attribute or "").lower()
            hints=PROPERTY_HINTS.get(
                "color" if attr=="colour" else attr,
                set(),
            )

            # Only semantic property edges can answer a property question.
            if pred in {"has_property","color","colour","has_color","has_size","size","has_shape","shape"}:
                if target.value and target.value in token_set(obj):
                    base=220
                elif hints and any(
                    h in token_set(obj)
                    for h in hints
                ):
                    # Match the actual requested attribute to the property's
                    # value vocabulary, not merely to any property.
                    base=120
                elif attr=="size" and any(
                    x in token_set(obj)
                    for x in PROPERTY_HINTS["size"]
                ):
                    base=120
                elif attr=="color" and any(
                    x in token_set(obj)
                    for x in PROPERTY_HINTS["color"]
                ):
                    base=120
                elif attr=="shape" and any(
                    x in token_set(obj)
                    for x in PROPERTY_HINTS["shape"]
                ):
                    base=120
                else:
                    base=0
            elif attr=="location" and pred in {
                "located_in","located_near","at_location",
            }:
                base=100
            elif attr=="time" and pred in {
                "has_property","time","date",
            }:
                base=100 if any(
                    x in token_set(obj)
                    for x in PROPERTY_HINTS["time"]
                ) else 0
            else:
                base=0

            # Yes/no questions about an adjective require exact evidence for
            # that adjective; unrelated properties are never substitutes.
            if target.value:
                if target.value not in token_set(obj):
                    base=0
                else:
                    base+=100

            if base<=0:
                continue


        elif target.kind=="definition":
            pass

        else:
            base=0

        if base<=0:
            continue

        copy=dict(fact)
        copy["_target_score"]=base+float(
            fact.get("relevance_final",fact.get("relevance",0.0))
        )
        selected.append(copy)

    selected.sort(
        key=lambda f:f["_target_score"],
        reverse=True,
    )
    return selected[:max_items]


def combine_relevant_facts(facts):
    texts=[]
    for f in facts:
        subject=f["subject"]
        pred=str(f["predicate"]).replace("_"," ")
        obj=f["object_text"]

        if pred=="has property":
            texts.append(f"The {subject} is {obj}.")
        elif pred=="defined as":
            texts.append(f"{subject.capitalize()} is {obj}.")
        elif pred=="is a":
            texts.append(f"{subject.capitalize()} is a {obj}.")
        elif pred=="related to":
            texts.append(f"{subject.capitalize()} is related to {obj}.")
        else:
            texts.append(
                f"{subject.capitalize()} {pred} {obj}."
            )
    return " ".join(texts)

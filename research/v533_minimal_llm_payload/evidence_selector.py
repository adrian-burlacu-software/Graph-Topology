from __future__ import annotations

from query_target import QueryTarget
from model_utils import token_set

PROPERTY_HINTS={
    "color":{"red","orange","yellow","green","blue","purple","violet","pink","brown","black","white","gray","grey","gold","silver"},
    "size":{"big","small","large","little","tiny","huge","vast","enormous","massive","long","short","wide","narrow"},
    "shape":{"round","square","flat","spherical","circular","rectangular"},
}

LEXICAL_RELATIONS={
    "synonym","antonym","hypernym","hyponym"
}

NOISY_RELATIONS={
    "related_to","similar_to","has_context","in_domain","domain","source",
    "provenance","dataset","type","label","category","subcategory",
}

def _base_fact_score(fact, target: QueryTarget, context_subject: str | None = None):
    pred=str(fact.get("predicate","")).lower()
    ftype=str(fact.get("fact_type","")).lower()
    subject=str(fact.get("subject","")).lower()
    obj=str(fact.get("object_text","")).lower()

    if pred in NOISY_RELATIONS:
        return 0

    # Never let arbitrary facts about a related concept become evidence for a
    # question about another concept.
    if target.kind == "property":
        if subject != str(target.subject or "").lower():
            return 0
        attr=(target.attribute or "").lower()
        if pred in {"color","colour","has_color"} and attr in {"color","colour"}:
            score=120
        elif pred in {"has_size","size"} and attr=="size":
            score=120
        elif pred in {"has_shape","shape"} and attr=="shape":
            score=120
        elif attr=="location" and pred in {"located_in","located_near","at_location","lives_in"}:
            score=120
        elif attr=="time" and pred in {"time","date","has_property"}:
            score=110
        elif pred=="has_property":
            # Generic has_property is only useful when the value directly
            # answers the requested property/question.
            hints=PROPERTY_HINTS.get(attr,set())
            if target.value and target.value in token_set(obj):
                score=180
            elif hints and token_set(obj) & hints:
                score=120
            else:
                return 0
        else:
            return 0
        if target.value:
            if target.value not in token_set(obj):
                return 0
            score += 80
        return score

    if target.kind == "count":
        # Evidence must concern the counted object itself, an explicit count,
        # or a direct relation between the requested population and object.
        counted=str(target.subject or "").lower()
        qualifier=str(target.qualifier or "").lower()
        if subject == counted and pred in {"count","has_count","cardinality","number","amount"}:
            return 180
        if qualifier and subject == qualifier and obj in {counted, counted + "s"} and pred in {"has","contains","includes","part_of","comprises"}:
            return 160
        if subject == counted and pred in {"part_of","is_a","has_property","color","size","shape"}:
            # Useful only as supporting context; never enough to invent a count.
            return 30
        return 0

    if target.kind in {"definition","general"}:
        if subject != str(target.subject or "").lower():
            return 0
        if pred == "defined_as":
            return 180
        if pred == "is_a":
            return 150
        if pred == "has":
            return 125
        if pred == "has_property":
            return 115
        if pred in {"capable_of","used_for","created_by","produces","part_of","contains"}:
            return 95
        if pred in LEXICAL_RELATIONS:
            # Lexical structure can support a definition, but is strictly lower
            # priority than factual/semantic relations.
            return 35
        if ftype == "lexical":
            return 15
        return 20

    return 0


def select_evidence(facts, target: QueryTarget, max_items=4, context_subject=None):
    selected=[]
    for fact in facts:
        score=_base_fact_score(fact, target, context_subject=context_subject)
        if score <= 0:
            continue

        copy=dict(fact)
        # The model does not need ranking internals, source frequency, or other
        # retrieval metadata. Keep the score internally for architecture debug.
        copy["_target_score"]=score + float(fact.get("relevance_final",fact.get("relevance",0.0)))
        selected.append(copy)

    selected.sort(key=lambda f:f["_target_score"], reverse=True)
    return selected[:max_items]


def _is_direct_for_llm(fact, target):
    """Hard gate: only facts that can directly support the requested answer."""
    pred = str(fact.get("predicate", "")).lower()
    subject = str(fact.get("subject", "")).lower()
    obj = str(fact.get("object_text", "")).lower()
    target_subject = str(target.subject or "").lower()

    if target.kind == "count":
        counted = target_subject
        qualifier = str(target.qualifier or "").lower()
        if subject == counted and pred in {"count", "has_count", "cardinality", "number", "amount"}:
            return True
        if qualifier and subject == qualifier and obj in {counted, counted + "s"} and pred in {"has", "contains", "includes", "part_of", "comprises"}:
            return True
        # A generic fact about the population is not evidence for the count.
        return False

    if target.kind == "property":
        if subject != target_subject:
            return False
        attr = str(target.attribute or "").lower()
        if attr in {"color", "colour"} and pred in {"color", "colour", "has_color", "has_property"}:
            return True
        if attr == "size" and pred in {"size", "has_size", "has_property"}:
            return True
        if attr == "shape" and pred in {"shape", "has_shape", "has_property"}:
            return True
        if attr in {"location", "time"} and pred in {"located_in", "located_near", "at_location", "lives_in", "time", "date", "has_property"}:
            return True
        return False

    if target.kind == "definition":
        return subject == target_subject and pred in {"defined_as", "is_a"}

    if target.kind == "general":
        return subject == target_subject and pred not in LEXICAL_RELATIONS and str(fact.get("fact_type", "")).lower() != "lexical"

    return False


def select_llm_evidence(facts, target, max_items=4):
    """Select only evidence that is directly usable for the current query."""
    selected = [f for f in facts if _is_direct_for_llm(f, target)]
    selected.sort(key=lambda f: float(f.get("relevance_final", f.get("relevance", 0.0))), reverse=True)
    return selected[:max_items]


def evidence_sentence(fact):
    """Human-readable semantic fact. No provenance, scoring, or database vocabulary."""
    subject = str(fact.get("subject") or "").strip()
    pred = str(fact.get("predicate") or "").lower()
    obj = str(fact.get("object_text") or "").strip()
    if not subject or not obj:
        return None
    if pred == "has_property":
        return f"{subject.capitalize()} is {obj}."
    if pred in {"defined_as", "definition"}:
        return f"{subject.capitalize()} is {obj}."
    if pred == "is_a":
        return f"{subject.capitalize()} is a {obj}."
    if pred in {"has", "contains", "includes", "comprises", "part_of"}:
        return f"{subject.capitalize()} has {obj}."
    if pred in {"count", "has_count", "cardinality", "number", "amount"}:
        return f"The number of {subject} is {obj}."
    if pred in {"color", "colour", "has_color"}:
        return f"{subject.capitalize()} is {obj}."
    if pred in {"size", "has_size"}:
        return f"{subject.capitalize()} is {obj}."
    if pred in {"shape", "has_shape"}:
        return f"{subject.capitalize()} is {obj}."
    if pred in {"located_in", "located_near", "at_location", "lives_in"}:
        return f"{subject.capitalize()} is located in {obj}."
    return f"{subject.capitalize()} {pred.replace('_', ' ')} {obj}."


def compact_evidence(facts, target=None):
    """Return a presentation-only payload safe to expose to the LLM."""
    compact=[]
    seen=set()
    for f in facts:
        item = evidence_sentence(f)
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        compact.append(item)
    return compact


def combine_relevant_facts(facts):
    texts=[]
    for f in facts:
        subject=f.get("subject")
        pred=str(f.get("predicate","")).replace("_"," ")
        obj=f.get("object_text")
        if not subject or not obj:
            continue
        if pred=="has property":
            texts.append(f"The {subject} is {obj}.")
        elif pred=="defined as":
            texts.append(f"{subject.capitalize()} is {obj}.")
        elif pred=="is a":
            texts.append(f"{subject.capitalize()} is a {obj}.")
        elif pred in {"has","contains","includes"}:
            texts.append(f"{subject.capitalize()} has {obj}.")
        elif pred=="related to":
            texts.append(f"{subject.capitalize()} is related to {obj}.")
        else:
            texts.append(f"{subject.capitalize()} {pred} {obj}.")
    return " ".join(texts)

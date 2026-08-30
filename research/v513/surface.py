
from __future__ import annotations


def fact_to_sentence(f):
    subject=str(f.get("subject","")).strip()
    predicate=str(f.get("predicate","")).strip().replace("_"," ")
    obj=str(f.get("object_text","")).strip()

    if not subject or not predicate or not obj:
        return None

    low=predicate.lower()

    if low=="has property":
        return f"The {subject} is {obj}."

    if low=="color" or low=="colour":
        return f"The {subject} is {obj}."

    if low=="synonym":
        return f"{subject} is a synonym of {obj}."

    if low=="antonym":
        return f"{subject} is an antonym of {obj}."

    if low=="hypernym":
        return f"{subject} is a type of {obj}."

    if low=="hyponym":
        return f"{obj} is a type of {subject}."

    if low=="related to":
        return f"{subject} is related to {obj}."

    if low=="is a":
        return f"{subject} is a {obj}."

    if low=="member of verb class":
        return f"{subject} belongs to VerbNet class {obj}."

    return f"{subject} {predicate} {obj}."


def join_facts(facts,limit=2):
    sentences=[]
    for f in facts[:limit]:
        text=fact_to_sentence(f)
        if text:
            sentences.append(text)
    return " ".join(sentences)

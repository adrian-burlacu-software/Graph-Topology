"""Which of some phrases a question means most (v701).

`who is the oldest in people.csv` names no field: the reader reads that it
asks for the most or least of one, not which, nor which way -- `oldest`
was read as the smallest. Said beside each way it could be meant (`who has
the highest age`, `who has the lowest age`, ...), the sentence encoder
(MiniLM-L6-v2, as it was pretrained) tells which it is nearest: oldest,
youngest, tallest, cheapest and busiest each nearest the field and the way
a person means.
"""
from __future__ import annotations

from research.encoder import BASE

_LOADED: dict = {}


def _model():
    if "model" not in _LOADED:
        from transformers import AutoModel, AutoTokenizer
        _LOADED["tokenizer"] = AutoTokenizer.from_pretrained(str(BASE))
        _LOADED["model"] = AutoModel.from_pretrained(str(BASE)).eval()
    return _LOADED["tokenizer"], _LOADED["model"]


def _vectors(texts: list):
    import torch
    tokenizer, model = _model()
    batch = tokenizer(texts, padding=True, truncation=True,
                      return_tensors="pt")
    with torch.no_grad():
        out = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def nearest(text: str, phrases: list) -> list:
    """(likeness, phrase) of each phrase to the text, nearest first."""
    if not phrases:
        return []
    found = (_vectors([text]) @ _vectors(list(phrases)).T)[0].tolist()
    return sorted(zip(found, phrases), reverse=True)


#: how much nearer a field must bring the question than the way alone
#: (`who has the highest`): `tallest` 0.28 for height, `oldest` 0.28 for
#: age, `busiest` 0.10 for load; `which`, `the best`, `the most` below 0
FIELD_GAIN = 0.08


def extreme(text: str, fields: list) -> tuple | None:
    """(field, "max" | "min") a question asking for the most or least of
    something, naming no field, means: the field it is nearer for being
    said (beyond the way alone: `kids` is near `age` whatever is asked),
    by `FIELD_GAIN` and more than any other; the way, the nearer of
    highest and lowest of it. None where no field is."""
    ways = {"max": "who has the highest", "min": "who has the lowest"}
    alone = {phrase: score for score, phrase in nearest(text,
                                                        list(ways.values()))}
    said = {f"{ways[way]} {one.replace('_', ' ')}": (one, way)
            for one in fields for way in ways}
    scores = {phrase: score for score, phrase in nearest(text, list(said))}
    gain: dict = {}
    for phrase, (one, way) in said.items():
        gain[one] = max(gain.get(one, -1.0), scores[phrase] - alone[ways[way]])
    ranked = sorted(gain.items(), key=lambda one: -one[1])
    if not ranked or ranked[0][1] < FIELD_GAIN or (
            len(ranked) > 1 and ranked[0][1] - ranked[1][1] < FIELD_GAIN / 2):
        return None
    best = ranked[0][0]
    way = max(ways, key=lambda one: scores[
        f"{ways[one]} {best.replace('_', ' ')}"])
    return best, way

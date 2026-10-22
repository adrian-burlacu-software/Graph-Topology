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
#: (`the one with the most`): `tallest` for height, `oldest` for age;
#: `which`, `the best`, `the most` not at all
FIELD_GAIN = 0.08
#: how much nearer one way of the field chosen must be than the other, and
#: one field than the next (`youngest`: the least age 0.71, the most 0.69)
WAY_LEAD, FIELD_LEAD = 0.01, 0.02

#: each way said of a field, as what it measures: an amount, or a time --
#: of a time the most is the latest (`the newest book`: its year the
#: largest, `oldest` the least -- of an age the largest). Said several ways
#: and the likeness averaged: antonyms are near in the encoder, and one
#: phrasing alone chose `cheapest` as the highest price
FRAMES = {
    "amount": {"max": ("the {g} with the most {f}", "the {g} with the "
                       "highest {f}", "the {g} with the largest {f}",
                       "the {g} whose {f} is the greatest"),
               "min": ("the {g} with the least {f}", "the {g} with the "
                       "lowest {f}", "the {g} with the smallest {f}",
                       "the {g} whose {f} is the lowest")},
    "time": {"max": ("the {g} with the latest {f}", "the {g} with the most "
                     "recent {f}", "the {g} whose {f} came last"),
             "min": ("the {g} with the earliest {f}", "the {g} with the "
                     "first {f}", "the {g} whose {f} came first")},
}


#: each of the records' field said, listed rather than ordered
LISTED = ("every {g}'s {f}", "the {f} of each {g}", "list the {g} and their "
          "{f}", "what {f} does each {g} have")
#: how much nearer an order must be than a list to be taken as asked
ORDER_LEAD = 0.05


def ordered(text: str, field: str, kind: str = "amount",
            group: str = "one") -> str | None:
    """"max" or "min" where a question naming a field of numbers or times
    asks for the record with the most or least of it rather than each
    record's (`which member joined first?`, read as a list of every
    member); None where it lists them, or neither is ahead."""
    said = {frame.format(g=group, f=field.replace("_", " ")): way
            for way, frames in FRAMES[kind].items() for frame in frames}
    said.update({frame.format(g=group, f=field.replace("_", " ")): "list"
                 for frame in LISTED})
    scores: dict = {}
    for score, phrase in nearest(text, list(said)):
        scores.setdefault(said[phrase], []).append(score)
    mean = {way: sum(found) / len(found) for way, found in scores.items()}
    way = max(("max", "min"), key=lambda one: mean[one])
    if mean[way] - mean["list"] < ORDER_LEAD or \
            abs(mean["max"] - mean["min"]) < WAY_LEAD:
        return None
    return way


def extreme(text: str, fields: list, kinds: dict | None = None,
            group: str = "one") -> tuple | None:
    """(field, "max" | "min") a question asking for the most or least of
    something, naming no field, means: the field it is nearer for being
    said (beyond the way alone: `kids` is near `age` whatever is asked),
    by `FIELD_GAIN` and ahead of any other; the way, the nearer of its two
    by `WAY_LEAD`. `kinds` says of each field whether it is an amount or a
    time (an amount where unsaid). None where no field or way is ahead."""
    kinds = kinds or {}
    said: dict = {}
    for one in fields:
        for way, frames in FRAMES[kinds.get(one, "amount")].items():
            for frame in frames:
                said[frame.format(g=group, f=one.replace("_", " "))] = (
                    one, way)
    alone = {frame.format(g=group, f="").replace("  ", " ").rstrip(): way
             for way, frames in FRAMES["amount"].items() for frame in frames
             if frame.endswith("{f}")}
    scores: dict = {}
    for score, phrase in nearest(text, list(said) + list(alone)):
        key = said.get(phrase) or ("", alone[phrase])
        scores.setdefault(key, []).append(score)
    mean = {key: sum(found) / len(found) for key, found in scores.items()}
    gain = {one: max(mean[(one, way)] - mean[("", way)]
                     for way in ("max", "min")) for one in fields}
    ranked = sorted(gain.items(), key=lambda one: -one[1])
    if not ranked or ranked[0][1] < FIELD_GAIN or (
            len(ranked) > 1 and ranked[0][1] - ranked[1][1] < FIELD_LEAD):
        return None
    best = ranked[0][0]
    if abs(mean[(best, "max")] - mean[(best, "min")]) < WAY_LEAD:
        return None
    return best, max(("max", "min"), key=lambda way: mean[(best, way)])

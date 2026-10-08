"""A question about the project's data, answered from what the data
holds: read by the encoder, its fields looked up in the schema, carried
out exactly.

    found = answer("which port does the server listen on", held)
    found.said     "server.port in service.yaml is 8697."

The shared reader reads the question (`teach_data_talk.py`): what is asked
(`data_act`: a value, a count, which ones, a total, an average, the
largest, the smallest, whether there is one), the comparison its filter
makes (`data_op`), and which words name the field asked for (TARGET), the
field it is filtered on (FIELD) and the value (VALUE). Nothing here reads
a word: the words the encoder marked are looked up among the places of the
project's data models (`datamodel.py`) -- a field by its name, as a
function is found by its name -- and the question is carried out over
what the file holds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from research import encoder

HEADS = ("data_act", "data_op", "data_role")
#: how sure the reader must be that a message asks about data
FLOOR = 0.6


@dataclass
class Asked:
    act: str
    chance: float
    op: str
    spans: dict
    words: list


@dataclass
class Found:
    said: str
    value: object = None
    file: str = ""
    place: str = ""
    asked: Asked | None = None
    looked: dict = field(default_factory=dict)

    def json(self) -> dict:
        return {"said": self.said, "value": self.value, "file": self.file,
                "place": self.place, "looked": self.looked,
                "asked": None if self.asked is None else {
                    "act": self.asked.act, "chance": round(self.asked.chance,
                                                           3),
                    "op": self.asked.op, "spans": self.asked.spans}}


def available() -> bool:
    try:
        return "data_act" in encoder.LOADED.get().heads
    except Exception:                               # noqa: BLE001
        return False


def read(text: str) -> Asked | None:
    """What the encoder reads a message as asking of data, or None where
    the reader in use was not taught data."""
    from research.v692.corpus import words
    if not available():
        return None
    said = words(text)
    if not said:
        return None
    found = encoder.read(said, heads=HEADS)
    act, chance = found["data_act"][0]
    op = found["data_op"][0][0]
    spans: dict = {}
    current = None
    for word, role in zip(said, found["data_role"]):
        if role.startswith("B-"):
            current = role[2:]
            spans.setdefault(current, []).append([word])
        elif role.startswith("I-") and current == role[2:]:
            spans[current][-1].append(word)
        else:
            current = None
    return Asked(act, chance, op, {name: [" ".join(one) for one in found]
                                   for name, found in spans.items()}, said)


# -- the words, looked up in the schema --------------------------------------------

def _parts(path: str) -> list:
    return [one.lower() for one in path.replace("[]", "").split(".") if one]


def _said(phrase: str) -> list:
    return [one for one in re.split(r"[\s_\-]+", phrase.lower()) if one]


def _fits(phrase: str, path: str, words: list) -> tuple:
    """How well a phrase names a place: its own name said whole
    (`timeout_seconds`, `timeout seconds`), and how many of the message's
    other words name the places around it (`the server port`)."""
    parts = _parts(path)
    if not parts:
        return (0, 0)
    own = parts[-1]
    whole = phrase.lower().replace(" ", "_") == own or \
        _said(phrase) == _said(own)
    if not whole:
        return (0, 0)
    around = sum(one.lower() in parts[:-1] for one in words)
    return (1, around)


def _place(held, phrase: str, words: list, collections: bool,
           inside: str | None = None, model=None):
    """The (model, path) a phrase names: of every data file -- or of one
    -- the place whose own name it is, the most of the message's words
    naming the places around it; a collection where one is asked for.
    `inside`: a field of this collection."""
    best, score = None, (0, 0)
    for one in ([model] if model is not None else held.data().values()):
        if one.error:
            continue
        paths = list(one.collections) if collections else list(one.places)
        for path in paths:
            if inside is not None:
                if inside and not path.startswith(f"{inside}."):
                    continue
                if not inside and "." in path:
                    continue
            name = path if path else one.path.rsplit("/", 1)[-1].rsplit(
                ".", 1)[0]
            fit = _fits(phrase, name, words) if path else (
                (1, 0) if _said(phrase) == _said(name) else (0, 0))
            if fit > score:
                best, score = (one, path), fit
    return best


def _value(phrase: str):
    try:
        return int(phrase)
    except ValueError:
        pass
    try:
        return float(phrase)
    except ValueError:
        return phrase


def _holds(value, op: str, wanted) -> bool:
    if op == "contains":
        if isinstance(value, list):
            return any(str(one).lower() == str(wanted).lower()
                       for one in value)
        return str(wanted).lower() in str(value).lower()
    if op in ("eq", "ne"):
        same = str(value).lower() == str(wanted).lower()
        return same if op == "eq" else not same
    try:
        left, right = float(value), float(wanted)
    except (TypeError, ValueError):
        return False
    return {"gt": left > right, "lt": left < right, "ge": left >= right,
            "le": left <= right}.get(op, False)


def _field_of(record: dict, name: str):
    return record.get(name) if isinstance(record, dict) else None


def answer(text: str, held) -> Found | None:
    """The question, answered from the project's data -- or None where it
    is not one, or what it names is not there."""
    if held is None or not held.data():
        return None
    asked = read(text)
    if asked is None or asked.act == "none" or asked.chance < FLOOR:
        return None
    return carry(asked, held)


def _named(phrase: str, name: str) -> bool:
    """Whether a phrase says a name: `timeout seconds`, `timeout_seconds`,
    `users` for `users`."""
    return bool(phrase) and (
        phrase.lower().replace(" ", "_") == name.lower()
        or _said(phrase) == _said(name))


def _groups(held):
    """Every collection of records of the project's data: (model, path,
    its fields, its name -- a CSV's is its file's)."""
    for model in held.data().values():
        if model.error:
            continue
        for path, found in model.collections.items():
            name = _shown_name(path, model) if path else \
                model.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            yield model, path, [str(one) for one in found["fields"]], name


def _resolved(held, asked: Asked) -> dict | None:
    """What the question's words name: a collection (and the field of it
    asked for, and the field its filter is on), or a value outside any --
    each looked up by its name, the one holding most of what is named
    (and of the message's other words) taken."""
    target = (asked.spans.get("TARGET") or [""])[0]
    field_ = (asked.spans.get("FIELD") or [""])[0]
    best, score, ties = None, None, []
    for model, path, fields, name in _groups(held):
        by = next((one for one in fields if _named(field_, one)), None)
        if field_ and by is None:
            continue
        asked_field = next((one for one in fields if _named(target, one)),
                           None)
        whole = _named(target, name)
        if not (asked_field or whole or by):
            continue
        base = model.path.rsplit("/", 1)[-1]
        around = sum(_named(word, name) or _named(word, base.rsplit(
            ".", 1)[0]) or word.lower() == base.lower()
            for word in asked.words)
        fit = (by is not None, bool(asked_field or whole), around)
        one = {"model": model, "collection": path, "field": asked_field,
               "by": by, "name": name}
        if score is None or fit > score:
            best, score, ties = one, fit, [one]
        elif fit == score:
            ties.append(one)
    if best is not None and (best["field"] or best["by"] or
                             _named(target, best["name"])):
        # what fits as well elsewhere is answered too: two files' users
        # are both the users asked about
        best["also"] = ties[1:]
        return best
    if field_:
        return None
    found = _place(held, target, asked.words, collections=False)
    if found is None:
        return None
    model, path = found
    if any(path.startswith(f"{one}.") for one in model.collections):
        return None
    return {"model": model, "place": path}


def carry(asked: Asked, held) -> Found | None:
    """What was read, looked up and carried out over the data -- of every
    collection that fits it as well as the best, each said."""
    target = (asked.spans.get("TARGET") or [""])[0]
    if not target:
        return None
    found = _resolved(held, asked)
    if found is None or not found.get("also"):
        return _carried(asked, found, target)
    every = [_carried(asked, one, target) for one in [found] + found["also"]]
    first = every[0]
    first.said = " ".join(one.said for one in every)
    first.value = [one.value for one in every]
    first.looked["also"] = [one.file for one in every[1:]]
    return first


def _carried(asked: Asked, found: dict | None, target: str) -> Found:
    looked = {"target": target}
    if found is None:
        return Found(f"I find nothing called {target} in the project's "
                     f"data" + (f" with a field {asked.spans['FIELD'][0]}"
                                if asked.spans.get("FIELD") else "") + ".",
                     asked=asked, looked=looked)
    model = found["model"]
    where = model.path
    if "place" in found:
        path = found["place"]
        looked["place"] = path
        values = _at(model.value, path)
        if not values:
            return Found(f"{path} in {where} holds nothing.", None, where,
                         path, asked, looked)
        value = values[0] if len(values) == 1 else values
        return Found(f"{path} in {where} is {_said_value(value)}.", value,
                     where, path, asked, looked)
    collection, group = found["collection"], found["name"]
    records = model.records()[collection]
    looked["collection"] = collection or where
    values = asked.spans.get("VALUE") or []
    if found["by"] and values and asked.op != "none":
        wanted = _value(values[0])
        records = [one for one in records
                   if _holds(_field_of(one, found["by"]), asked.op, wanted)]
        looked["filter"] = [found["by"], asked.op, wanted]
    if asked.act == "count":
        return Found(f"{len(records)} {group} in {where}" + _because(looked)
                     + ".", len(records), where, collection, asked, looked)
    if asked.act == "exists":
        there = bool(records)
        return Found((f"Yes: {len(records)} {group} in {where}" if there
                      else f"No {group} in {where}") + _because(looked) + ".",
                     there, where, collection, asked, looked)
    name = found["field"] or (model.keys(collection) or [None])[0] or \
        next((one for one in (model.collections[collection]["fields"])
              if any(isinstance(_field_of(row, one), str)
                     for row in records)), None)
    if name is None:
        return Found(f"The {group} in {where} have no field to name them "
                     f"by.", None, where, collection, asked, looked)
    looked["field"] = name
    picked = [_field_of(one, name) for one in records]
    picked = [one for one in picked if one is not None]
    if asked.act in ("sum", "mean", "max", "min"):
        numbers = [one for one in picked if isinstance(one, (int, float))
                   and not isinstance(one, bool)]
        if not numbers:
            return Found(f"{name} of the {group} in {where} holds no "
                         f"numbers.", None, where, collection, asked, looked)
        value = {"sum": sum(numbers), "mean": sum(numbers) / len(numbers),
                 "max": max(numbers), "min": min(numbers)}[asked.act]
        word = {"sum": "The total", "mean": "The average",
                "max": "The largest", "min": "The smallest"}[asked.act]
        return Found(f"{word} {name} of the {group} in {where}"
                     + _because(looked) + f" is {_said_value(value)} (of "
                     f"{len(numbers)}).", value, where, collection, asked,
                     looked)
    if asked.act == "value" and len(picked) == 1:
        return Found(f"The {name} of the {group} in {where}"
                     + _because(looked) + f" is {_said_value(picked[0])}.",
                     picked[0], where, collection, asked, looked)
    return Found(f"{_said_list(picked)}: the {name} of the {group} in "
                 f"{where}" + _because(looked) + ".", picked, where,
                 collection, asked, looked)


def _at(value, path: str) -> list:
    from research.v701.datamodel import _at as at
    return at(value, path)


def _shown_name(path: str, model) -> str:
    if path:
        return [one for one in path.replace("[]", "").split(".") if one][-1]
    return "rows"


def _because(looked: dict) -> str:
    found = looked.get("filter")
    if not found:
        return ""
    said = {"eq": "is", "ne": "is not", "gt": "is more than",
            "lt": "is less than", "ge": "is at least", "le": "is at most",
            "contains": "has"}[found[1]]
    return f" whose {found[0]} {said} {_said_value(found[2])}"


def _said_value(value) -> str:
    import json
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _said_list(values: list) -> str:
    if not values:
        return "None"
    shown = [_said_value(one) for one in values[:12]]
    return ", ".join(shown) + (f" and {len(values) - 12} more"
                               if len(values) > 12 else "")

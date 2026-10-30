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
#: how sure the reader must be that a message is no question of data for
#: no lookup to make it one
SURE_NONE = 0.9


@dataclass
class Asked:
    act: str
    chance: float
    op: str
    spans: dict
    words: list
    #: the words naming the project's data, as told to the reader
    known: list = field(default_factory=list)
    #: every act, most likely first, as the reader ranked them
    acts: list = field(default_factory=list)
    #: whether it asks of the records the last question answered with
    #: (`how many of these are there?`), and those records where it does
    of: str = "none"
    last: dict | None = None


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


def known(held) -> set:
    """The words that name something of the project's data -- a place's
    own name, a collection's, a file's (and one of each: `models`,
    `model`) -- looked up, and told to the reader beside each word as code
    talk's names are (`CODE`)."""
    out = set()
    if held is None:
        return out
    for model in held.data().values():
        base = model.path.rsplit("/", 1)[-1].lower()
        out.update({base, base.rsplit(".", 1)[0]})
        for path in list(model.places) + list(model.collections):
            for part in _parts(path):
                out.update({part, _one(part)})
    return out


def read(text: str, names=()) -> Asked | None:
    """What the encoder reads a message as asking of data, or None where
    the reader in use was not taught data."""
    from research.v692.corpus import words
    if not available():
        return None
    said = words(text)
    if not said:
        return None
    lower = {str(one).lower() for one in names}
    tags = ["CODE" if word.lower() in lower or _one(word.lower()) in lower
            else "" for word in said]
    # whether it asks of the last answer's records: a reader taught it
    # (SParC, `teach_data_sparc`)
    of_last = "data_of" in encoder.LOADED.get().heads
    found = encoder.read(said, tags=tags,
                         heads=HEADS + (("data_of",) if of_last else ()))
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
    return Asked(act, chance, op, {name: [" ".join(one) for one in found_]
                                   for name, found_ in spans.items()}, said,
                 [word for word, tag in zip(said, tags) if tag],
                 [name for name, _ in found["data_act"]],
                 found["data_of"][0][0] if of_last else "none")


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
    """Whether a value is held to what was asked -- a list is `wanted` where
    it holds it (`projects` is `stark-db`: one of them is), a word where it
    is it or one of it (`readers`: `reader`)."""
    if op == "contains" or (isinstance(value, list) and op in ("eq", "ne")):
        if isinstance(value, list):
            there = any(_same(one, wanted) for one in value)
        else:
            there = str(wanted).lower() in str(value).lower()
        return there if op != "ne" else not there
    if op in ("eq", "ne"):
        same = _same(value, wanted)
        return same if op == "eq" else not same
    try:
        left, right = float(value), float(wanted)
    except (TypeError, ValueError):
        return False
    return {"gt": left > right, "lt": left < right, "ge": left >= right,
            "le": left <= right}.get(op, False)


#: a number written as text, whole: thousands apart (`17,956`), with a
#: currency's sign or as a percent -- read as the number it is (a table of
#: viewers held "no numbers"); one with a unit (`1.96 m`) is left as said
WRITTEN_NUMBER = re.compile(
    # (a leading zero is a code's, not an amount's: `007`, `02139`)
    r"[$€£]?(-?[1-9]\d{0,2}(?:,\d{3})+|-?(?:0|[1-9]\d*))(\.\d+)?%?")


def _field_of(record: dict, name: str):
    value = record.get(name) if isinstance(record, dict) else None
    if isinstance(value, str) and WRITTEN_NUMBER.fullmatch(value.strip()):
        said = re.sub(r"[$€£%,]", "", value.strip())
        return float(said) if "." in said else int(said)
    return value


def answer(text: str, held, last: dict | None = None) -> Found | None:
    """The question, answered from the project's data -- or None where it
    is not one, or what it names is not there. `last`: the records the last
    question answered with (`looked["kept"]`), asked of where the reader
    reads it of them."""
    if held is None or not held.data():
        return None
    asked = read(text, known(held))
    if asked is None:
        return None
    if asked.of == "last" and last:
        asked.last = last
    if asked.act == "beyond" and asked.chance >= FLOOR:
        # asked what is not looked up here yet (grouped, sorted, the top
        # few, either-or): said so, not guessed at (SParC: `sort by their
        # age` answered with every age)
        return Found("", asked=asked, looked={"beyond": True})
    if asked.act == "none" and asked.chance >= SURE_NONE:
        # read surely as no question of data (`show me the first line of
        # books.csv`, none 1.0): not made one by lookup -- it was made a
        # question of the most pages. A record named by a value it holds,
        # said whole, is still one (`who wrote Kindred?`, none 0.98)
        found = _by_value(text, held) or _place_said(asked, held)
        if found is not None:
            found.looked["read as"] = "no question"
        return found
    if asked.act == "none" or asked.chance < FLOOR:
        act = asked.act
        found = _unread(text, asked, held)
        if found is not None:
            # what the reader read as no question of data: another part's,
            # where one reads it as its own (the page's data talk yields);
            # and what the reader read most surely, below its floor (`max`
            # 0.59 of `which book is the oldest?`, `none` of `what does
            # oldest return when there are no books?`)
            found.looked["read as"] = "no question"
            found.looked["read act"] = act
        return found
    return carry(asked, held)


def _place_said(asked: Asked, held) -> Found | None:
    """A place of the data the question says whole -- every part of its
    path, two or more (`which port does the web answer on?`: web.port) --
    its value; None where no place, or more than one, is so said."""
    said = {str(one).lower() for one in asked.words} | \
        {_one(str(one).lower()) for one in asked.words}
    found = []
    for model in held.data().values():
        if model.error:
            continue
        for path in model.places:
            parts = _parts(path)
            if "[]" not in path and len(parts) >= 2 and all(
                    part in said or _one(part) in said for part in parts):
                found.append((model, path))
    if len(found) != 1:
        return None
    model, path = found[0]
    values = _at(model.value, path)
    if not values:
        return None
    value = values[0] if len(values) == 1 else values
    return Found(f"{path} in {model.path} is {_said_value(value)}.", value,
                 model.path, path, asked, {"place": path})


def _unread(text: str, asked: Asked, held) -> Found | None:
    """A question the reader read as none of data, answered from it all
    the same: a lookup, a field asked among the data's words, a record
    named by a value."""
    found = _by_lookup(asked, held)
    if found is not None:
        return found
    # the act read as no question of data, the roles as one: a field
    # asked for, among words naming the project's data -- where what it
    # names is there, a value asked (`which port does the server
    # listen on`: none 0.7, `port` the field asked, `server` there too)
    if not (asked.spans.get("TARGET") and len(asked.known) >= 2):
        return _by_value(text, held)
    asked.act = "value"
    found = carry(asked, held)
    return found if found is not None and found.place else None


#: a record named by a value: how near the rest of the question must be to
#: a field, and how much nearer than the next field
LEAST_FIELD, FIELD_LEAD = 0.2, 0.05


def _by_value(text: str, held) -> Found | None:
    """A question naming a record by one of its values, read as no
    question of data (`who wrote Kindred?`, `when did Alan Turing join?`
    -- answered from the world, `I don't have that information`): the
    record whose value the question says as it is, and the field the rest
    of it is nearest in meaning (`who wrote` -> author, as v701's
    `likeness` reads `oldest`). None where no value is said, more than one
    record has it, or no field is near enough, or nearer than the next."""
    lowered = text.lower()
    found = []
    for model in held.data().values():
        if model.error:
            continue
        for path, records in model.records().items():
            for record in records:
                if not isinstance(record, dict):
                    continue
                for name, value in record.items():
                    said = str(value).strip()
                    if not isinstance(value, str) or len(said) < 3:
                        continue
                    if re.search(r"(?<!\w)" + re.escape(said.lower()) +
                                 r"(?!\w)", lowered):
                        found.append((len(said), model, path, record, name,
                                      said))
    if not found:
        return None
    longest = max(one[0] for one in found)
    found = [one for one in found if one[0] == longest]
    if len({id(one[3]) for one in found}) != 1:
        return _by_shared_value(found)
    _, model, path, record, named, said = found[0]
    rest = re.sub(re.escape(said), " ", text, flags=re.I)
    fields = [one for one in record if one != named]
    if not fields:
        return None
    # a field the rest names by its own name (`what year was Solaris
    # published?`: year -- read by meaning alone, `published` was nearer
    # the author) -- looked up
    words = set(re.findall(r"[a-z]+", rest.lower()))
    named_here = [one for one in fields
                  if set(one.lower().replace("_", " ").split()) <= words]
    likeness_ = None
    if len(named_here) == 1:
        field_ = named_here[0]
    else:
        # else the question beside the same question of each field
        # (`what is the author of Kindred?`): the nearest, by a lead
        from research.v701.likeness import nearest
        framed = {f"what is the {one.replace('_', ' ')} of {said}?": one
                  for one in fields}
        ranked = nearest(text, list(framed))
        if ranked[0][0] < LEAST_FIELD or (
                len(ranked) > 1 and ranked[0][0] - ranked[1][0] <
                FIELD_LEAD):
            return None
        field_, likeness_ = framed[ranked[0][1]], round(ranked[0][0], 3)
    value = record[field_]
    # a question of a field states no other value for it: one that does
    # tells it (`there are now 25 Climbing Rose in stock` was answered "the
    # stock ... is 12")
    if re.fullmatch(r"-?\d+(\.\d+)?", str(value).strip()) and any(
            float(one) != float(value)
            for one in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
                                  rest)):
        return None
    place = f"{path}.{field_}" if path else field_
    return Found(said=f"The {field_} of the record whose {named} is {said} "
                      f"(in {model.path}) is {value}.",
                 value=value, file=model.path, place=place,
                 looked={"named by": named, "value": said, "field": field_,
                         "likeness": likeness_, "record": record})


def _by_shared_value(found: list) -> Found | None:
    """A proper name several records hold in one field, said as they hold it
    (`who lives in Ashby?`): the records. A word that merely is a value
    (`power`, a tool's group) names none."""
    if len({(id(one[1]), one[2], one[4]) for one in found}) != 1:
        return None
    _, model, path, _, named, said = found[0]
    if not said[:1].isupper():
        return None
    records = [one[3] for one in found]
    key = (model.keys(path) or [None])[0] or next(
        (one for one in model.collections[path]["fields"] if one != named and
         all(isinstance(_field_of(row, one), str) for row in records)), None)
    if key is None:
        return None
    return Found(said=f"{_said_list(_called(model, path, key, records))}: the "
                      f"records in {model.path} whose {named} is {said}.",
                 value=[_field_of(row, key) for row in records],
                 file=model.path, place=path,
                 looked={"named by": named, "value": said, "field": key})


def of_record(text: str, held, path: str, record: dict) -> Found | None:
    """A question of the record just answered of, naming none of its own
    (`which book is the oldest?` ... `who wrote it?`): the field the
    question is nearest asking of that record -- named by its own name, or
    by meaning, by a lead -- as `_by_value` asks of a record it names."""
    if not isinstance(record, dict) or len(record) < 2:
        return None
    named = next((name for name, value in record.items()
                  if isinstance(value, str) and len(value) >= 3), None)
    if named is None:
        return None
    said = str(record[named])
    fields = [one for one in record if one != named]
    words = set(re.findall(r"[a-z]+", text.lower()))
    named_here = [one for one in fields
                  if set(one.lower().replace("_", " ").split()) <= words]
    likeness_ = None
    if len(named_here) == 1:
        field_ = named_here[0]
    else:
        from research.v701.likeness import nearest
        framed = {f"what is the {one.replace('_', ' ')} of {said}?": one
                  for one in fields}
        ranked = nearest(text, list(framed))
        if ranked[0][0] < LEAST_FIELD or (
                len(ranked) > 1 and ranked[0][0] - ranked[1][0] <
                FIELD_LEAD):
            return None
        field_, likeness_ = framed[ranked[0][1]], round(ranked[0][0], 3)
    value = record[field_]
    return Found(said=f"The {field_} of {said} (in {path}) is "
                      f"{_said_value(value)}.",
                 value=value, file=path, place=field_,
                 looked={"of the record": said, "field": field_,
                         "likeness": likeness_, "record": record})


#: what a question may ask of a collection when the act was read too
#: unsurely to take, and the lookup must say it is one
OF_RECORDS = ("count", "list", "exists", "max", "min")


def _by_lookup(asked: Asked, held) -> Found | None:
    """A question the reader was unsure is of data -- `how many people live
    in Toronto` is also the world's -- taken as one where the project says
    so: a word of it names a collection of records, and another is a value
    exactly one of their fields holds (or, asking for the most or least, a
    collection alone). Its act the reader's likeliest of those it ranked;
    the filter equality."""
    act = next((one for one in asked.acts if one in OF_RECORDS), None)
    if act is None:
        return None
    for model, path, fields, name, base in _groups(held):
        collection = next((word for word in asked.words
                           if _named(word, name) or word.lower() == base.lower()
                           or _named(word, base.rsplit(".", 1)[0])), None)
        if collection is None:
            continue
        records = model.records()[path]
        held_by = [(word, _field_by_value(records, fields, "eq", word))
                   for word in asked.words if word != collection
                   and len(word) > 2 and not any(_named(word, one)
                                                 for one in fields)]
        held_by = [(word, by) for word, by in held_by if by is not None]
        if act in ("max", "min") and not held_by:
            spans = {"TARGET": [collection]}
        elif len(held_by) == 1 and act not in ("max", "min"):
            # the value as the data writes it (`Toronto`, not `toronto`)
            word, by = held_by[0]
            spelled = next((str(_field_of(one, by)) for one in records
                            if _same(_field_of(one, by), word)), word)
            spans = {"TARGET": [collection], "VALUE": [spelled]}
        else:
            continue
        lookup = Asked(act, asked.chance, "eq" if "VALUE" in spans else
                       "none", spans, asked.words, asked.known, asked.acts,
                       asked.of, asked.last)
        found = carry(lookup, held)
        # found: the collection's file (a CSV's rows have no path)
        if found is not None and found.file:
            found.looked["by lookup"] = True
            return found
    return None


def _one(word: str) -> str:
    """A word as one of it: `models` is `model`, `readers` `reader`."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and \
        not word.endswith("ss") else word


def _named(phrase: str, name: str) -> bool:
    """Whether a phrase says a name: `timeout seconds`, `timeout_seconds`,
    `users` for `users`, `model` for `models`."""
    if not phrase or not name:
        return False
    said, own = _said(phrase), _said(name)
    return phrase.lower().replace(" ", "_") == name.lower() or said == own \
        or (len(said) == len(own) and all(
            _one(a) == _one(b) for a, b in zip(said, own)))


def _groups(held):
    """Every collection of records of the project's data: (model, path,
    its fields, its name -- a CSV's is its file's -- and its file's)."""
    for model in held.data().values():
        if model.error:
            continue
        base = model.path.rsplit("/", 1)[-1]
        for path, found in model.collections.items():
            name = _shown_name(path, model) if path else base.rsplit(
                ".", 1)[0]
            yield model, path, [str(one) for one in found["fields"]], name, \
                base


def _same(value, wanted) -> bool:
    return str(value).lower() == str(wanted).lower() or \
        _one(str(value).lower()) == _one(str(wanted).lower())


def _field_by_value(records: list, fields: list, op: str, wanted):
    """The field a filter is on, where the question does not name it --
    found in what the records hold (`which users can see stark-db`:
    `stark-db` is in their `projects`; `older than 30`: their one field of
    numbers). None where no field, or more than one, holds it."""
    found = []
    for name in fields:
        values = [_field_of(one, name) for one in records]
        if op in ("eq", "ne", "contains"):
            if any(_same(one, wanted) or (isinstance(one, list) and any(
                    _same(item, wanted) for item in one)) for one in values):
                found.append(name)
        elif op in ("gt", "lt", "ge", "le"):
            numbers = [one for one in values if one is not None]
            if numbers and all(isinstance(one, (int, float)) and not
                               isinstance(one, bool) for one in numbers):
                found.append(name)
    return found[0] if len(found) == 1 else None


def _resolved(held, asked: Asked) -> dict | None:
    """What the question's words name: a collection (and the field of it
    asked for, and the field its filter is on), or a value outside any --
    each looked up by its name (the filter's field, where not named, by
    the value it is held to), the one holding most of what is named (and
    of the message's other words) taken."""
    target = (asked.spans.get("TARGET") or [""])[0]
    field_ = (asked.spans.get("FIELD") or [""])[0]
    value = (asked.spans.get("VALUE") or [""])[0]
    best, score, ties = None, None, []
    for model, path, fields, name, base in _groups(held):
        by = next((one for one in fields if _named(field_, one)), None)
        named_by = by is not None
        records = model.records()[path]
        if by is None and value and asked.op != "none":
            by = _field_by_value(records, fields, asked.op, _value(value))
        if by is None and value and asked.op in ("gt", "lt", "ge", "le"):
            # of several fields of numbers, the one the question names
            # anywhere (`more than 300 pages`)
            said_ = [one for one in fields if any(
                _named(word, one) for word in asked.words) and all(
                _orderable(_field_of(row, one)) for row in records)]
            by = said_[0] if len(said_) == 1 else None
        asked_field = next((one for one in fields if _named(target, one)),
                           None)
        whole = _named(target, name) or target.lower() == base.lower()
        if not (asked_field or whole):
            continue
        through = None
        if value and asked.op != "none" and by is None:
            # the value held by another collection, linked to this one
            # (`which teachers teach the math course?`: the course is math,
            # its teachers through course_arrange)
            through = _held_across(held, model, path, asked, value)
            if through is None:
                continue
        around = sum(_named(word, name) or _named(word, base.rsplit(
            ".", 1)[0]) or word.lower() == base.lower()
            for word in asked.words)
        # asked of the last answer's records: theirs, where it fits at all
        mine = bool(asked.last) and asked.last["file"] == model.path and \
            asked.last["collection"] == path
        fit = (mine, named_by, by is not None, bool(asked_field or whole),
               around)
        one = {"model": model, "collection": path, "field": asked_field,
               "by": by, "name": name, "through": through}
        if score is None or fit > score:
            best, score, ties = one, fit, [one]
        elif fit == score:
            ties.append(one)
    if best is not None:
        # what fits as well elsewhere is answered too: two files' users
        # are both the users asked about
        best["also"] = ties[1:]
        return best
    # a value outside any collection: each word marked tried as its own
    # name, the rest of the message naming the places around it
    found, fit = None, (0, 0)
    for phrase in [one for one in (target, field_) if one] or asked.known:
        for model in held.data().values():
            if model.error:
                continue
            for path in model.places:
                # a CSV's columns are its rows' fields, not values of their
                # own; nor is what a collection's records hold
                if "" in model.collections or any(
                        path.startswith(f"{one}.") or path == one
                        for one in model.collections if one):
                    continue
                parts = _parts(path)
                if not parts or not _named(phrase, parts[-1]):
                    continue
                around = sum(any(_named(word, part) for part in parts[:-1])
                             for word in asked.words)
                if (1, around) > fit:
                    found, fit = (model, path), (1, around)
    if found is None:
        return None
    model, path = found
    # a place that holds more places, one of which another word of the
    # message names: that one (`what model is the editor`: `models.editor`)
    if "object" in model.places[path].types:
        inner = [one for one in model.places if one.startswith(f"{path}.")
                 and "." not in one[len(path) + 1:]]
        named = [one for one in inner if any(
            _named(word, one.rsplit(".", 1)[-1]) for word in asked.words)]
        if len(named) == 1:
            path = named[0]
    return {"model": model, "place": path}


#: how many links a question's records may be followed across
LINKS = 2


def _links(held, start: tuple, end: tuple) -> list | None:
    """The fewest links from one collection to another (file, path), each
    followed either way -- [(relation, forward)], [] where they are one, or
    None where none joins them within `LINKS`."""
    if start == end:
        return []
    found = held.relations()
    seen, layer = {start}, [(start, [])]
    for _ in range(LINKS):
        nearer = []
        for here, way in layer:
            for link in found:
                ends = (tuple(link["from"][:2]), tuple(link["to"][:2]))
                for forward in (True, False):
                    if ends[0 if forward else 1] != here:
                        continue
                    there = ends[1 if forward else 0]
                    if there in seen:
                        continue
                    if there == end:
                        return way + [(link, forward)]
                    seen.add(there)
                    nearer.append((there, way + [(link, forward)]))
        layer = nearer
    return None


def _ways(held, start: tuple, end: tuple) -> list:
    """Every way of the fewest links from one collection to another: one
    link each way where several join them directly (a task's owner and its
    watchers), else the one `_links` finds."""
    direct = [[(link, forward)] for link in held.relations()
              for forward in (True, False)
              if (tuple(link["from"][:2]), tuple(link["to"][:2]))[
                  0 if forward else 1] == start and
              (tuple(link["from"][:2]), tuple(link["to"][:2]))[
                  1 if forward else 0] == end]
    if direct:
        return direct
    way = _links(held, start, end)
    return [way] if way else []


def _across(held, records: list, way: list) -> list:
    """The records of the far end of the links that the given records name,
    or are named by, link by link."""
    models = held.data()
    for link, forward in way:
        (file, path, field_), (to_file, to_path, key) = \
            link["from"], link["to"]
        if not forward:
            file, path, field_, to_file, to_path, key = \
                to_file, to_path, key, file, path, field_
        said = set()
        for one in records:
            held_ = _field_of(one, field_)
            said.update(str(item).lower() for item in (
                held_ if isinstance(held_, list) else [held_])
                if item is not None)
        theirs = []
        for one in models[to_file].records()[to_path]:
            held_ = _field_of(one, key)
            if any(str(item).lower() in said for item in (
                    held_ if isinstance(held_, list) else [held_])
                   if item is not None):
                theirs.append(one)
        records = theirs
    return records


def _held_across(held, model, path: str, asked: Asked, value: str):
    """Another collection holding the value a filter is held to, and the
    links to this one -- {model, collection, by, way}, the nearest; None
    where none is linked."""
    best = None
    for other, there, fields, _, _ in _groups(held):
        if (other.path, there) == (model.path, path):
            continue
        field_ = (asked.spans.get("FIELD") or [""])[0]
        by = next((one for one in fields if _named(field_, one)), None) \
            if field_ else None
        if by is None:
            by = _field_by_value(other.records()[there], fields, asked.op,
                                 _value(value))
        if by is None:
            continue
        for way in _ways(held, (other.path, there), (model.path, path)):
            # the fewest links, and of those the one whose field the
            # message says (`the watchers of t1`, not its owner)
            said = sum(any(_named(word, link["from"][2]) for word in
                           asked.words) for link, _ in way)
            fit = (-len(way), said)
            if best is None or fit > best["fit"]:
                best = {"model": other, "collection": there, "by": by,
                        "way": way, "fit": fit}
    return best


def _said_way(way: list) -> str:
    """The links followed, as said: `by course_arrange.csv's Teacher_ID`."""
    return ", ".join(f"{link['from'][0] if forward else link['to'][0]}'s "
                     f"{link['from'][2] if forward else link['to'][2]}"
                     for link, forward in way)


def _sorted_out(held, spans: dict) -> dict:
    """The fields and values read, as the data has them: a "field" that is
    a file's name is the file asked of, not a field (`how many commands
    does package.json contribute`); a "value" that is a place's name and
    no value the data holds is a place word (`contribute`)."""
    files = set()
    for model in held.data().values():
        base = model.path.rsplit("/", 1)[-1].lower()
        files.update({base, base.rsplit(".", 1)[0]})
    names = known(held) - files

    def held_somewhere(phrase: str) -> bool:
        for model in held.data().values():
            for place in model.places.values():
                if any(_same(one, phrase) for one in place.distinct):
                    return True
                if any(_same(one, phrase) for one in place.examples):
                    return True
        return False
    out = dict(spans)
    if out.get("FIELD"):
        out["FIELD"] = [one for one in out["FIELD"]
                        if one.lower() not in files]
    if out.get("VALUE"):
        out["VALUE"] = [one for one in out["VALUE"]
                        if not ((one.lower() in names or
                                 _one(one.lower()) in names or any(
                                     _one(name) == _one(one.lower())
                                     for name in names))
                                and not held_somewhere(one))]
    return {name: found for name, found in out.items() if found}


def carry(asked: Asked, held) -> Found | None:
    """What was read, looked up and carried out over the data -- of every
    collection that fits it as well as the best, each said."""
    target = (asked.spans.get("TARGET") or [""])[0]
    if not target and asked.act == "value" and asked.known:
        # no field marked, but words naming the project's data: the last
        # of them is what is asked of (`is the teacher offline`), the others
        # where it is -- as code talk's one known word stands for its
        # subject when none is marked
        target = asked.known[-1]
        asked.spans = {**asked.spans, "TARGET": [target]}
    whole = _said_whole(held, asked.words)
    if whole is not None:
        value, _, _, holder = whole
        spans = dict(asked.spans)
        # a value read in pieces, or as a field (`octavia` and `e . butler`;
        # `dispossessed` the field): the one the data holds, said whole
        if not any(_held_as(held, one) for one in spans.get("VALUE", ())) \
                or len(spans.get("VALUE", ())) > 1:
            spans["VALUE"] = [value]
        spans["FIELD"] = [one for one in spans.get("FIELD", ())
                          if not _same(one, value) and
                          _said(one)[-1] not in _said(value)]
        if not spans["FIELD"]:
            spans.pop("FIELD")
        if asked.op == "none":
            asked.op = "eq"
        # no target (`who has the email katherine@example.org?`): the
        # records holding it
        if not target:
            target = holder
            spans["TARGET"] = [target]
        asked.spans = spans
    if not target and asked.spans.get("FIELD"):
        # a field read, its records not (`which member joined first?` read
        # with `joined` alone): the one collection that has that field
        field_ = asked.spans["FIELD"][0]
        having = [group for group in _groups(held)
                  if any(_named(field_, str(one)) for one in group[2])]
        if len(having) == 1:
            target = having[0][3] or having[0][1] or field_
            asked.spans = {**asked.spans, "TARGET": [target]}
    if not target and asked.last:
        # of the last answer's records, naming none (`how many are there?`)
        target = asked.last["name"]
        asked.spans = {**asked.spans, "TARGET": [target]}
    if not target:
        return None
    asked.spans = _sorted_out(held, asked.spans)
    found = _resolved(held, asked)
    if found is None:
        joined = _joined(held, asked, target)
        if joined is not None:
            return joined
    if found is None or not found.get("also"):
        return _carried(asked, found, target, held)
    every = [_carried(asked, one, target, held)
             for one in [found] + found["also"]]
    first = every[0]
    first.said = " ".join(one.said for one in every)
    first.value = [one.value for one in every]
    first.looked["also"] = [one.file for one in every[1:]]
    return first


def _joined(held, asked: Asked, target: str) -> Found | None:
    """A question no one collection answers, answered across a link
    (`Project.relations`): the records the question names, in one
    collection -- found by the value it says -- and what they name of
    another, which holds the field asked for (`what role does the owner of
    t1 have`: the task t1, its owner, that user's role)."""
    value = (asked.spans.get("VALUE") or [""])[0]
    if not value:
        return None
    models = held.data()
    # the link whose field the message says first (`the watchers of t1`)
    links = sorted(held.relations(), key=lambda one: not any(
        _named(word, one["from"][2]) for word in asked.words))
    for link in links:
        source, field_ = models[link["from"][0]], link["from"][2]
        named, key = models[link["to"][0]], link["to"][2]
        to_fields = named.collections[link["to"][1]]["fields"]
        asked_field = next((one for one in to_fields
                            if _named(target, str(one))), None)
        if asked_field is None:
            continue
        records = source.records()[link["from"][1]]
        fields = source.collections[link["from"][1]]["fields"]
        by = _field_by_value(records, fields, "eq", _value(value))
        if by is None:
            continue
        mine = [one for one in records if _holds(_field_of(one, by), "eq",
                                                 _value(value))]
        names = set()
        for one in mine:
            held_ = _field_of(one, field_)
            names.update(str(item).lower() for item in (
                held_ if isinstance(held_, list) else [held_])
                if item is not None)
        theirs = [one for one in named.records()[link["to"][1]]
                  if str(_field_of(one, key)).lower() in names]
        picked = [_field_of(one, asked_field) for one in theirs]
        looked = {"target": target, "through": link,
                  "filter": [by, "eq", _value(value)]}
        group = link["from"][1] or link["from"][0]
        said = (f"{_said_list(picked)}: the {asked_field} of the "
                f"{field_} of the {_shown_name(link['from'][1], source)} in "
                f"{source.path} whose {by} is {value} (by {named.path}'s "
                f"{key}).")
        return Found(said, picked[0] if len(picked) == 1 else picked,
                     named.path, group, asked, looked)
    return None


def _carried(asked: Asked, found: dict | None, target: str,
             held=None) -> Found:
    looked = {"target": target}
    if found is None:
        # nothing of the data is so called: an answer of no data, which
        # another part may give (the page's data talk yields to code talk)
        looked["nothing"] = True
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
    every = model.records()[collection]
    records = every
    looked["collection"] = collection or where
    last = asked.last
    if last and last["file"] == where and last["collection"] == collection \
            and last["at"] is not None:
        # of the records the last question answered with (`which of these
        # ...`, `how many are there?`): those alone
        records = [every[at] for at in last["at"] if at < len(every)]
        looked["of the last"] = len(records)
    elif last and last["at"] is not None and held is not None and \
            last["file"] in held.data():
        # of another collection's: the records of this one they name, or
        # are named by (`what are the names of the countries they belong
        # to?` after the car makers)
        way = _links(held, (last["file"], last["collection"]),
                     (where, collection))
        if way:
            theirs = held.data()[last["file"]].records()[
                last["collection"]]
            records = _across(held, [theirs[at] for at in last["at"]
                                           if at < len(theirs)], way)
            looked["of the last"] = len(records)
            looked["through"] = _said_way(way)
    through = found.get("through")
    if through is not None:
        # the filter on the collection that holds its value, its records
        # followed across the links to these
        values_ = asked.spans.get("VALUE") or []
        wanted = _value(values_[0])
        theirs = [one for one in through["model"].records()[
            through["collection"]] if _holds(_field_of(one, through["by"]),
                                             asked.op, wanted)]
        mapped = {id(one) for one in _across(held, theirs,
                                             through["way"])}
        records = [one for one in records if id(one) in mapped]
        looked["filter"] = [through["by"], asked.op, wanted]
        looked["through"] = _said_way(through["way"])
    values = asked.spans.get("VALUE") or []
    if found["by"] and values and asked.op != "none":
        wanted = _value(values[0])
        listed = any(isinstance(_field_of(one, found["by"]), list)
                     for one in records)
        records = [one for one in records
                   if _holds(_field_of(one, found["by"]), asked.op, wanted)]
        # a list holds what it is held to: said so (`whose projects has`)
        op = {"eq": "contains", "ne": "lacks"}.get(asked.op, asked.op) \
            if listed else asked.op
        looked["filter"] = [found["by"], op, wanted]
    elif not values:
        # no value read, but a word of the message is one the records hold
        # (`the average age of the readers`: role `reader`) -- that filter,
        # where exactly one field holds it
        fields = model.collections[collection]["fields"]
        spoken = {word for found_ in asked.spans.values() for phrase in
                  found_ for word in _said(phrase)}
        for word in asked.words:
            if len(word) < 3 or word.lower() in spoken or any(
                    _named(word, str(one)) for one in fields) or \
                    _named(word, group):
                continue
            by = _field_by_value(records, fields, "eq", word)
            if by is not None and not any(isinstance(_field_of(one, by), (
                    int, float)) for one in records):
                records = [one for one in records
                           if _holds(_field_of(one, by), "eq", word)]
                looked["filter"] = [by, "eq", _one(word.lower())]
                break
    if len(records) == 1:
        # the one record answered of: what a follow-up's `it` is
        looked["record"] = records[0]
    looked["kept"] = _kept(model, collection, group, every, records)
    if asked.act == "count" and found["field"] and records and all(
            isinstance(_field_of(one, found["field"]), (int, float)) and
            not isinstance(_field_of(one, found["field"]), bool)
            for one in records):
        # how many of what a field counts (`how many pages does The
        # Dispossessed have?`, `... across all the books?`): its amount --
        # one record's, else theirs together
        name = found["field"]
        looked["field"] = name
        total = sum(_field_of(one, name) for one in records)
        if len(records) == 1:
            return Found(f"The {name} of the {group} in {where}"
                         + _because(looked) + f" is {_said_value(total)}.",
                         total, where, collection, asked, looked)
        return Found(f"The {name} of the {len(records)} {group} in {where}"
                     + _because(looked) + f" come to {_said_value(total)} "
                     f"together.", total, where, collection, asked, looked)
    if asked.act == "count":
        return Found(f"{len(records)} {group} in {where}" + _because(looked)
                     + ".", len(records), where, collection, asked, looked)
    if asked.act in ("max", "min") and found["field"] and records and \
            not all(_orderable(_field_of(one, found["field"]))
                    for one in records):
        # the most of a field that names, not counts (`which author has
        # the most books?`): the value most (least) of the records hold
        return _most_held(asked, found, model, records, looked)
    if asked.act == "exists":
        there = bool(records)
        return Found((f"Yes: {len(records)} {group} in {where}" if there
                      else f"No {group} in {where}") + _because(looked) + ".",
                     there, where, collection, asked, looked)
    by = found["by"]
    if asked.act == "list" and by and not values and records and all(
            _orderable(_field_of(one, by)) for one in records):
        # a field of numbers or times named, and every record's of it read
        # as asked: where the question is nearer the one with its most or
        # least (`which member joined first?` listed every member), that
        from research.v701 import likeness
        way = likeness.ordered(" ".join(asked.words), by,
                               _kinds([by], records)[by],
                               _one(_said(group)[-1]) if _said(group.strip(
                                   "[]")) else "one")
        if way is not None:
            looked["ordered"] = way
            asked = Asked(way, asked.chance, asked.op, asked.spans,
                          asked.words, asked.known, asked.acts, asked.of,
                          asked.last)
    if asked.act in ("max", "min") and (not found["field"] or looked.get(
            "ordered")) and records:
        return _extreme(asked, found, model, records, looked)
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
    if not found["field"] and asked.act not in ("sum", "mean", "max", "min",
                                                "value"):
        # no field asked of: each said by what tells them apart for a person
        # as well as by its key
        return Found(f"{_said_list(_called(model, collection, name, records))}"
                     f": the {group} in {where}" + _because(looked) + ".",
                     picked, where, collection, asked, looked)
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


def _called(model, collection: str, key: str, records: list) -> list:
    """Records as an answer says them: their key -- beside the other text
    of theirs that tells every record of the collection apart, where there
    is one (`M4 (Cement Mixer)`: a code alone answers nobody)."""
    every = model.records()[collection]
    fields = model.collections[collection]["fields"]

    def tells(field_):
        values = [_field_of(row, field_) for row in every]
        return all(isinstance(one, str) for one in values) and \
            len(set(values)) == len(values)
    other = next((one for one in fields if one != key and tells(one)), None)
    said = [_field_of(row, key) for row in records]
    if other is None:
        return said
    return [f"{one} ({_field_of(row, other)})"
            for one, row in zip(said, records)]


def _kept(model, collection: str, group: str, every: list,
          records: list) -> dict:
    """The records an answer ended with, as a follow-up asks of them: the
    file, the collection, its name, and where each is among all of them
    (None where they are all)."""
    ids = {id(one) for one in records}
    return {"file": model.path, "collection": collection, "name": group,
            "at": None if len(records) == len(every) else
            [at for at, one in enumerate(every) if id(one) in ids]}


def _extreme(asked: Asked, found: dict, model, records: list,
             looked: dict) -> Found:
    """The records holding the most (or least) of a field, where the
    records themselves are asked for (`which of the people has the highest
    age`, `who is the oldest in people.csv`): the field the question names
    -- else the field, and the way, it is nearest asking of (`likeness`);
    where two fields are as near, which is asked, not guessed."""
    collection, group, where = found["collection"], found["name"], model.path
    fields = [str(one) for one in model.collections[collection]["fields"]]
    numeric = [one for one in fields if all(
        _orderable(_field_of(row, one)) for row in records) and any(
        _field_of(row, one) is not None for row in records)]
    named = [one for one in fields for phrase in asked.spans.get("FIELD", ())
             if _named(phrase, one)]
    if found["by"] and found["by"] in numeric and not asked.spans.get(
            "VALUE"):
        named = [found["by"]]
    field_ = named[0] if named else None
    act = asked.act
    if field_ is None and numeric:
        # no field said (`who is the oldest`): the question beside each
        # field asked both ways (`likeness`) -- the nearest is the field
        # and the way; two fields as near, it asks which
        from research.v701 import likeness
        # the question whole, beside each field said of what the data
        # calls its records (`which book is the longest` -- `the book with
        # the most pages`), and of each what it measures: of a time the
        # most is the latest (`the newest book`, its year the largest)
        named_as = group if group.strip("[]") else \
            model.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        meant = likeness.extreme(" ".join(asked.words), numeric,
                                 _kinds(numeric, records),
                                 _one(_said(named_as)[-1]) if _said(named_as)
                                 else "one")
        if meant is not None:
            field_, act = meant
            looked["nearest"] = f"{act} {field_}"
    if act != asked.act:
        asked = Asked(act, asked.chance, asked.op, asked.spans, asked.words,
                      asked.known, asked.acts, asked.of, asked.last)
    if field_ is None:
        # which field is asked, asked back: what another part reads as its
        # own is that part's (the page's data talk yields)
        looked["ask which"] = True
        return Found(f"The {group} in {where} have "
                     f"{'no field' if not numeric else 'several fields'} of "
                     f"numbers{': ' + ', '.join(numeric) if numeric else ''}"
                     f" -- which do you mean?", None, where, collection, asked,
                     looked)
    numbers = [_field_of(row, field_) for row in records]
    held = [(number, row) for number, row in zip(numbers, records)
            if _orderable(number) and number is not None]
    if not held:
        return Found(f"{field_} of the {group} in {where} holds no numbers.",
                     None, where, collection, asked, looked)
    best = (max if asked.act == "max" else min)(one for one, _ in held)
    rows = [row for number, row in held if number == best]
    key = (model.keys(collection) or [None])[0] or next(
        (one for one in fields if one not in numeric), None)
    names = [_field_of(row, key) for row in rows] if key else rows
    looked.update({"field": field_, "extreme": asked.act, "key": key})
    if len(rows) == 1:
        looked["record"] = rows[0]
    # what a follow-up asks of (`what is its id?`): the records ordered out
    looked["kept"] = _kept(model, collection, group,
                           model.records()[collection], rows)
    word = ({"max": "latest", "min": "earliest"} if _kinds(
        [field_], records)[field_] == "time" else {
        "max": "largest", "min": "smallest"})[asked.act]
    shown = _called(model, collection, key, rows) if key else names
    return Found(f"{_said_list(shown)}: the {group} in {where} with the "
                 f"{word} {field_} ({_said_value(best)}).",
                 names[0] if len(names) == 1 else names, where, collection,
                 asked, looked)


DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ][\d:.]+)?Z?")


def _orderable(value) -> bool:
    """A number, or a date as data writes it (`2021-03-02`): what has a
    most and a least."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) or (
        isinstance(value, str) and bool(DATE.fullmatch(value)))


#: how much nearer a field's name must be to a time than to an amount
#: for whole years to be read as one (`year`, `founded`; not `pages`)
TIME_LEAD = 0.05


def _kinds(fields: list, records: list) -> dict:
    """Whether each field of numbers is an amount or a time: dates as data
    writes them, or whole years (four digits) where the field's name reads
    as a time -- of a time the most is the latest, not the largest."""
    from research.v701 import likeness
    out = {}
    for one in fields:
        values = [_field_of(row, one) for row in records]
        values = [value for value in values if value is not None]
        if values and all(isinstance(value, str) and DATE.fullmatch(value)
                          for value in values):
            out[one] = "time"
        elif values and all(isinstance(value, int) and not isinstance(
                value, bool) and 1000 <= value <= 2999 for value in values):
            ranked = dict((phrase, score) for score, phrase in
                          likeness.nearest(one.replace("_", " "),
                                           ["a year, a date", "an amount, "
                                            "a count"]))
            out[one] = "time" if ranked["a year, a date"] - ranked[
                "an amount, a count"] >= TIME_LEAD else "amount"
        else:
            out[one] = "amount"
    return out


def _held_as(held, phrase: str) -> bool:
    for model in held.data().values():
        for place in model.places.values():
            if any(_same(one, phrase) for one in place.distinct):
                return True
    return False


def _said_whole(held, words: list):
    """The longest value of the data's records the question says whole,
    word for word (`Grace Hopper's`, `Octavia E. Butler`): (value, model,
    collection path, the collection's name) -- None where none is, or two
    different ones are as long."""
    from research.v701.teach_data_talk import words as tokens
    said = [str(one).lower() for one in words]
    said = [one[:-2] if one.endswith("'s") else one for one in said]
    found = {}
    for model in held.data().values():
        if model.error:
            continue
        for path, records in model.records().items():
            for record in records:
                if not isinstance(record, dict):
                    continue
                for value in record.values():
                    if not isinstance(value, str) or len(value) < 3 or \
                            DATE.fullmatch(value):
                        continue
                    own = [one.lower() for one in tokens(value)]
                    if not own:
                        continue
                    for at in range(len(said) - len(own) + 1):
                        if said[at:at + len(own)] == own:
                            found.setdefault(value, (value, model, path,
                                                     _shown_name(path,
                                                                 model)))
                            break
    if not found:
        return None
    longest = max(len(one) for one in found)
    best = [one for one in found.values() if len(one[0]) == longest]
    return best[0] if len(best) == 1 else None


def _most_held(asked: Asked, found: dict, model, records: list,
               looked: dict) -> Found:
    """The value of a field most (or fewest) records hold: `Ursula K. Le
    Guin` -- three of the books."""
    name, group, where = found["field"], found["name"], model.path
    counts: dict = {}
    for one in records:
        value = _field_of(one, name)
        if value is not None:
            counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return Found(f"{name} of the {group} in {where} holds nothing.",
                     None, where, found["collection"], asked, looked)
    best = (max if asked.act == "max" else min)(counts.values())
    names = [value for value, count in counts.items() if count == best]
    looked.update({"field": name, "most held": asked.act})
    return Found(f"{_said_list(names)}: the {name} {'most' if asked.act == 'max' else 'fewest'} "
                 f"of the {group} in {where} have ({best} of "
                 f"{len(records)}).", names[0] if len(names) == 1 else names,
                 where, found["collection"], asked, looked)


def _at(value, path: str) -> list:
    from research.v701.datamodel import _at as at
    return at(value, path)


def _shown_name(path: str, model) -> str:
    """What a collection is called: the last name of its path -- or, where
    the path names none (a list at the file's top: `[]` in members.json,
    which failed every turn of a project holding one), its file's."""
    named = [one for one in (path or "").replace("[]", "").split(".")
             if one]
    if named:
        return named[-1]
    base = (getattr(model, "path", "") or "").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or "rows"


def _because(looked: dict) -> str:
    found = looked.get("filter")
    # asked of the last answer's records: said so
    of_last = f" (of the {looked['of the last']} just answered)" \
        if "of the last" in looked else ""
    if looked.get("through"):
        # the links followed, said (`by has_pet.csv's StuID`)
        of_last += f" (by {looked['through']})"
    if not found:
        return of_last
    said = {"eq": "is", "ne": "is not", "gt": "is more than",
            "lt": "is less than", "ge": "is at least", "le": "is at most",
            "contains": "has", "lacks": "does not have"}[found[1]]
    return f" whose {found[0]} {said} {_said_value(found[2])}" + of_last


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

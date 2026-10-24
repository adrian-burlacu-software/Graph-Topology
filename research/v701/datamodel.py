"""Data as information: a JSON, YAML or CSV file read into what it holds
and the schema it has -- the architecture's model of the file, not its
text.

    model = read("config/service.yaml", text)
    model.schema       every place in it: `server.port` integer, `users[]`
                       records of 2, `users[].role` string ...
    model.value        what it holds, as Python values
    model.records()    each collection of records: `users[]` -> [{...}]

A file is read by its format (a lookup of its ending, then the parser's
own word on it: what does not parse is said, not guessed). Its schema is
read off what it holds:

- **a place** is a path to a value -- `server.port`, `users[].name`
  (`[]`: each item of a list), a CSV column -- with the types it has
  there, how often it is there, a few of its values;
- **a collection** is a list of records (objects; a CSV's rows): its
  fields are its places, and a field whose every value is there and
  different from the others' is a **key** (`name`, `id`);
- a CSV's values are read as what they are written as: `41` an integer,
  `0.048` a number, `true` a boolean, an empty cell nothing.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

FORMATS = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".csv": "csv",
           ".tsv": "tsv"}
#: the values kept of each place, to show what it holds
EXAMPLES = 3
#: records a field must tell apart to be read as their key: of two, any
#: field may differ
LEAST_KEYED = 3


def format_of(path: str) -> str | None:
    lower = path.lower()
    return next((name for ending, name in FORMATS.items()
                 if lower.endswith(ending)), None)


@dataclass
class Place:
    path: str
    types: dict = field(default_factory=dict)
    count: int = 0
    examples: list = field(default_factory=list)
    distinct: set = field(default_factory=set)
    #: in a collection: of how many records
    of: int = 0

    def json(self) -> dict:
        out = {"path": self.path, "types": dict(sorted(
            self.types.items(), key=lambda one: -one[1])),
            "count": self.count, "examples": self.examples}
        if self.of:
            out["in"] = f"{self.count} of {self.of}"
        return out


@dataclass
class Model:
    path: str
    format: str
    value: object = None
    error: str | None = None
    places: dict = field(default_factory=dict)
    collections: dict = field(default_factory=dict)

    @property
    def schema(self) -> list:
        return [one.json() for one in self.places.values()]

    def records(self) -> dict:
        """Each collection: its path -> the records."""
        return {path: (list(self.value) if path == "" else
                       list(_at(self.value, path)))
                for path in self.collections}

    def keys(self, collection: str) -> list:
        """The fields of a collection every record has, each different."""
        found = self.collections.get(collection) or {}
        size = found.get("count", 0)
        out = []
        for name in found.get("fields", ()):
            place = self.places.get(f"{collection}.{name}" if collection
                                    else name)
            if place and size >= LEAST_KEYED and place.count == size and \
                    len(place.distinct) == size and set(place.types) <= {
                        "string", "integer"}:
                out.append(name)
        # a name tells records apart before a number does: three ages that
        # differ are not the users' key -- a number is one where no words
        # are
        words = [one for one in out if "string" in self.places[
            f"{collection}.{one}" if collection else one].types]
        return words or out

    def unique(self, collection: str) -> list:
        """Every field of a collection each record has, each different --
        what another collection may name its records by (`Stadium_ID`,
        where `keys` prefers the stadium's name)."""
        found = self.collections.get(collection) or {}
        size = found.get("count", 0)
        out = []
        for name in found.get("fields", ()):
            place = self.places.get(f"{collection}.{name}" if collection
                                    else name)
            if place and size >= LEAST_KEYED and place.count == size and \
                    len(place.distinct) == size and set(place.types) <= {
                        "string", "integer"}:
                out.append(name)
        return out

    def json(self) -> dict:
        if self.error:
            return {"path": self.path, "format": self.format,
                    "error": self.error}
        return {"path": self.path, "format": self.format,
                "collections": {path: {**one, "keys": self.keys(path)}
                                for path, one in self.collections.items()},
                "schema": self.schema}

    def said(self) -> str:
        """The schema in a few words, as an answer says it."""
        if self.error:
            return f"{self.path} does not read as {self.format}: {self.error}"
        parts = []
        for path, one in self.collections.items():
            name = path or "the rows"
            keys = self.keys(path)
            parts.append(f"{name}: {one['count']} records with "
                         f"{', '.join(one['fields'])}"
                         + (f" (each told apart by {', '.join(keys)})"
                            if keys else ""))
        inside = {path for path in self.collections}
        plain = [one for one in self.places.values()
                 if not any(one.path.startswith(f"{path}.") or
                            one.path == path for path in inside)
                 and "object" not in one.types]
        if plain:
            parts.append("; ".join(
                f"{one.path} {_type_said(one)}" +
                (f" ({_shown(one.examples[0])})" if one.examples else "")
                for one in plain[:12]) + ("; ..." if len(plain) > 12 else ""))
        return f"{self.path} ({self.format}): " + " -- ".join(parts)


def _type_said(place: Place) -> str:
    return " or ".join(sorted(place.types, key=lambda one: -place.types[one]))


def _shown(value) -> str:
    said = json.dumps(value, ensure_ascii=False)
    return said if len(said) <= 40 else said[:37] + "..."


# -- links between data ----------------------------------------------------------------

#: how many of a field's values must be another collection's keys for it to
#: name them, and how many values it must have
LINKED, LEAST_LINKED = 0.8, 3


def _name_words(name: str) -> set:
    """A field's or a collection's name as its words, one of each
    (`CountryCode` -> country, code; `car_makers` -> car, maker)."""
    import re
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    out = set()
    for word in re.split(r"[\W_]+", spaced.lower()):
        if word:
            out.add(word[:-1] if word.endswith("s") and len(word) > 3
                    else word)
    return out


def _alike(one: str, other: str) -> bool:
    """A word and another the same, or one the other's start (`cont`,
    `continent`)."""
    return one == other or (min(len(one), len(other)) >= 3 and (
        one.startswith(other) or other.startswith(one)))


def _within(words: set, others: set) -> bool:
    return all(any(_alike(one, other) for other in others) for one in words)


def _named_alike(field: str, key: str, collection: str) -> int:
    """How far a field's name says what it names: how many of its words,
    past `id`, are the key's or the collection's -- where the key's own
    words are all its (`owner_id` -- owners' `owner_id`; `Country` --
    countries' `CountryId`; `lost_in_battle` -- battles' `id`; not
    `orchestra_id` -- orchestras' `conductor_id`). 0 where it does not."""
    said = _name_words(field) - {"id"}
    own = _name_words(key) - {"id"}
    if not said or not _within(own, said):
        return 0
    return sum(any(_alike(one, other) for other in own | _name_words(
        collection)) for one in said)


def _own_key(field: str, collection: str) -> bool:
    """A field named for its own collection (`pets.PetID`): what names its
    records, not another's."""
    said = _name_words(field) - {"id"}
    return bool(said) and _within(said, _name_words(collection)) and \
        _within(_name_words(collection), said)


def relations(models: list) -> list:
    """Fields whose values are, nearly all, the keys of another collection
    -- of the same file or another: what one record names of another. Any
    field every record has, each different, is a key; numbers name records
    only where the names say so too (in car_1 every small number was some
    table's id: `Cylinders` named `cars_data`, `IndepYear` the cities)."""
    keys = []
    for model in models:
        if model.error:
            continue
        for path in model.collections:
            records = model.records()[path]
            named = path or model.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            # its first key (a field that only happens to differ -- three
            # tasks, three owners -- is not what another names them by),
            # and its numbers each different (`Stadium_ID`, where the first
            # key is the stadium's name): those linked to by name alone
            first = model.keys(path)[:1]
            for key in model.unique(path):
                if key not in first and not all(
                        isinstance(one.get(key), int) for one in records
                        if isinstance(one, dict)):
                    continue
                found = {str(one.get(key)).lower() for one in records
                         if isinstance(one, dict) and one.get(key) is not None}
                keys.append((model.path, path, key, found, named))
    out = []
    for model in models:
        if model.error:
            continue
        for path, collection in model.collections.items():
            records = model.records()[path]
            named_here = path or model.path.rsplit("/", 1)[-1].rsplit(".",
                                                                      1)[0]
            unique = set(model.unique(path))
            for name in collection["fields"]:
                if name in unique and _own_key(name, named_here):
                    continue
                values, digits = [], 0
                for one in records:
                    held = one.get(name) if isinstance(one, dict) else None
                    items = held if isinstance(held, list) else [held]
                    for item in items:
                        if item is None or isinstance(item, dict):
                            continue
                        values.append(str(item).lower())
                        # a number, or one written as text (`'130'`)
                        digits += not isinstance(item, bool) and \
                            str(item).lstrip("-").isdigit()
                if len(values) < LEAST_LINKED:
                    continue
                numbers = digits >= LINKED * len(values)
                found_ = []
                for file, other, key, found, named in keys:
                    if (file, other) == (model.path, path):
                        continue
                    alike = _named_alike(name, key, named)
                    if numbers and not alike:
                        continue
                    share = sum(one in found for one in values) / len(values)
                    if share >= LINKED:
                        found_.append((alike, {"from": [model.path, path,
                                                        name],
                                               "to": [file, other, key],
                                               "share": round(share, 3)}))
                # numbers: only what the name says most of (`student_
                # enrolment_id` the enrolments', not the students')
                best = max((alike for alike, _ in found_), default=0)
                out += [one for alike, one in found_
                        if not numbers or alike == best]
    return out


# -- reading ------------------------------------------------------------------------

def _typed(cells: list):
    """A column's cells as what they are written as -- read as a column:
    `41` an integer where every cell is one, `true` a boolean where every
    cell is one (a word list's `true` stays a word); an empty cell
    nothing."""
    said = [one.strip() for one in cells]
    filled = [one for one in said if one != ""]

    def all_are(convert) -> bool:
        try:
            for one in filled:
                convert(one)
        except ValueError:
            return False
        return bool(filled)

    def boolean(one: str) -> bool:
        if one.lower() not in ("true", "false"):
            raise ValueError(one)
        return one.lower() == "true"
    for convert in (int, float, boolean):
        if all_are(convert):
            return [convert(one) if one != "" else None for one in said]
    return [one if one.strip() != "" else None for one in cells]


def _header(lines: list) -> bool:
    """Whether the first row names the columns: a column whose cells
    below are numbers, the first not one; or, of several columns, first
    cells all there, all different, none found again below in its column
    (`name,role,age`). A list of words (`a`, `about`, ...) has none -- the
    csv sniffer said a short file with a header had none."""
    first, body = lines[0], lines[1:]
    if not body:
        return False

    def number(one: str) -> bool:
        try:
            float(one)
            return True
        except ValueError:
            return False
    for at, name in enumerate(first):
        below = [one[at].strip() for one in body if at < len(one)
                 and one[at].strip()]
        if below and all(number(one) for one in below) and \
                not number(name.strip()):
            return True
    names = [one.strip() for one in first]
    if len(names) < 2 or not all(names) or len(set(names)) < len(names):
        return False
    return not any(names[at] in {one[at].strip() for one in body
                                 if at < len(one)}
                   for at in range(len(names)))


def _rows(text: str, delimiter: str) -> list:
    """The rows, each a record of its columns. A first row that is not a
    header (the sniffer's word: a list of words has none) is a row, and
    the columns are `column1`, `column2` ..."""
    lines = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    lines = [one for one in lines if one]
    if not lines:
        raise ValueError("no rows")
    if _header(lines):
        names, body = lines[0], lines[1:]
    else:
        names = [f"column{at + 1}" for at in range(len(lines[0]))]
        body = lines
    columns = {name: _typed([one[at] if at < len(one) else ""
                             for one in body])
               for at, name in enumerate(names)}
    return [{name: columns[name][at] for name in names}
            for at in range(len(body))]


def parse(path: str, text: str, kind: str | None = None):
    kind = kind or format_of(path)
    if kind == "json":
        return json.loads(text)
    if kind == "yaml":
        import yaml
        return yaml.safe_load(text)
    if kind in ("csv", "tsv"):
        return _rows(text, "\t" if kind == "tsv" else ",")
    raise ValueError(f"not a data file: {path}")


def _kind(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _at(value, path: str):
    """The values at a path (`users[].name`): every one of them."""
    found = [value]
    for part in [one for one in path.replace("[]", ".[]").split(".")
                 if one]:
        nxt = []
        for one in found:
            if part == "[]":
                if isinstance(one, list):
                    nxt.extend(one)
            elif isinstance(one, dict) and part in one:
                nxt.append(one[part])
        found = nxt
    return found


def _walk(value, path: str, places: dict, collections: dict,
          of: int = 0) -> None:
    place = places.setdefault(path, Place(path)) if path else None
    if place is not None:
        kind = _kind(value)
        place.types[kind] = place.types.get(kind, 0) + 1
        place.count += 1
        place.of = of
        if kind not in ("object", "list"):
            try:
                place.distinct.add(value)
            except TypeError:
                pass
            if len(place.examples) < EXAMPLES and value not in \
                    place.examples:
                place.examples.append(value)
    if isinstance(value, dict):
        for name, inner in value.items():
            _walk(inner, f"{path}.{name}" if path else str(name), places,
                  collections, of)
    elif isinstance(value, list):
        inside = f"{path}[]"
        if value and all(isinstance(one, dict) for one in value):
            fields = list(dict.fromkeys(name for one in value
                                        for name in one))
            collections[inside] = {"count": len(value), "fields": fields}
            for one in value:
                _walk(one, inside, places, collections, len(value))
            # the record itself is not a place of its own
            places.pop(inside, None)
            for name in fields:
                found = places.get(f"{inside}.{name}")
                if found is not None:
                    found.of = len(value)
        else:
            for one in value:
                _walk(one, inside, places, collections, len(value))


def read(path: str, text: str, kind: str | None = None) -> Model:
    """The model of a data file: what it holds, and its schema."""
    kind = kind or format_of(path) or "json"
    model = Model(path, kind)
    try:
        model.value = parse(path, text, kind)
    except Exception as bad:                        # noqa: BLE001
        model.error = f"{type(bad).__name__}: {str(bad).splitlines()[0]}"
        return model
    if kind in ("csv", "tsv"):
        # the rows are the file: a collection at its top
        rows = model.value
        model.collections[""] = {"count": len(rows),
                                 "fields": list(rows[0]) if rows else []}
        for row in rows:
            for name, cell in row.items():
                _walk(cell, name, model.places, {}, len(rows))
        return model
    _walk(model.value, "", model.places, model.collections)
    return model

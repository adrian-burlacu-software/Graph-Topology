"""Teaching the shared reader questions about data: what is asked, of
which fields, held to which values.

    python -m research.v701.teach_data_talk seeds    the questions, from real files
    python -m research.v701.teach_data_talk write    the teacher says each again
    python -m research.v701.teach_data_talk corpus   llm/data-talk-data

A question about the project's data (`which port does the server listen
on`, `how many users are older than 30`) was read as common sense -- the
ports were a tricycle and a cart. It is read here as v692 reads
mathematics: by the shared encoder, into

    data_act   none | value | count | list | sum | mean | max | min | exists
    data_op    none | eq | ne | gt | lt | ge | le | contains   (its filter's)
    data_role  each word: the field asked for (TARGET), the field a filter
               is on (FIELD), the value it is held to (VALUE)

-- and the fields' words are looked up in the data's schema afterwards
(`querying.py`), as a function's name is in the project.

**Where the questions come from**: real data files -- CommitPackFT's JSON,
YAML and CSV (`data/commitpackft`, commits taught only) -- read into their
model (`datamodel.py`); from each, queries of what it holds (its own
fields, its own values), each said once as a seed. **The teacher**
(SmolLM3, offline) says each seed again, many ways, keeping its fields and
values exactly: the labels are where they are found, by construction.
**More of them**: each labelled question again with another file's
fields and values put where its were (the labels stay where they are).
**What is not a data question**: code talk and everyday messages, taught
with act `none`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
DATA = LLM / "data-talk-data"
SOURCE = ROOT / "data" / "commitpackft"
SEEDS = DATA / "seeds.jsonl"
#: what the first reader read wrong, live (v701): value questions said with
#: a verb, yes or no of a setting, a record named by its key, a filter's
#: field left unsaid
SEEDS_MORE = DATA / "seeds-more.jsonl"
WRITTEN = DATA / "written.jsonl"
SEED = 701
#: seeds made, and how many from each file
MOST_SEEDS, EACH_FILE = 360, 3
#: each labelled question said again with other files' names and values
SWAPS = 3

ACTS = ("none", "value", "count", "list", "sum", "mean", "max", "min",
        "exists")
OPS = ("none", "eq", "ne", "gt", "lt", "ge", "le", "contains")
ROLES = ("O", "B-TARGET", "I-TARGET", "B-FIELD", "I-FIELD", "B-VALUE",
         "I-VALUE")
STYLE = ("Vary the wording a lot: short and long, casual and formal, a few "
         "with small typos or no question mark. One per line, nothing else: "
         "no numbering, no quotes, no explanations.")


def words(text: str) -> list:
    from research.v692.corpus import words as said
    return said(text)


# -- queries of real files ---------------------------------------------------

def _name(path: str) -> str:
    """A place's own name: its last part (`server.port`: `port`)."""
    return [one for one in path.replace("[]", "").split(".") if one][-1] \
        if path.replace("[]", "").strip(".") else ""


def _plain(value) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(
        value, bool) and 0 < len(str(value)) <= 30 and "\n" not in str(value)


def _wordy(name: str) -> bool:
    """A name a person says as it is written (`port`, `timeout_seconds`):
    no spaces, slashes or dots, not a number."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,30}", name)) and \
        not re.fullmatch(r"column\d+", name)     # a CSV with no header's


def queries(model, rng: random.Random) -> list:
    """What can be asked of a file's model, each with its answer known:
    {act, op, target, field, value, collection, seed, spans}."""
    out = []
    stem = model.path.rsplit("/", 1)[-1]
    inside = [path for path in model.collections]
    for place in model.places.values():
        if any(place.path.startswith(f"{one}.") or place.path == one
               for one in inside) or not place.examples:
            continue
        name = _name(place.path)
        if not _wordy(name) or not _plain(place.examples[0]):
            continue
        parent = _name(place.path.rsplit(".", 1)[0]) if "." in place.path \
            else ""
        seed = rng.choice([
            f"what is the {name} in {stem}?",
            f"what's {name} set to?",
            f"what is the {parent} {name}?" if parent and _wordy(parent)
            else f"what value does {name} have?"])
        out.append({"act": "value", "op": "none", "target": place.path,
                    "seed": seed, "spans": {"TARGET": [name]}})
    for path, found in model.collections.items():
        if found["count"] < 3:
            continue
        group = _name(path) if path else stem.rsplit(".", 1)[0]
        if not _wordy(group):
            continue
        records = model.records()[path]
        fields = {}
        for name in found["fields"]:
            values = [one.get(name) for one in records
                      if isinstance(one, dict)]
            if _wordy(str(name)) and values:
                fields[name] = values
        if not fields:
            continue
        out.append({"act": "count", "op": "none", "target": path,
                    "seed": f"how many {group} are there?",
                    "spans": {"TARGET": [group]}})
        texts = {name: [one for one in values if _plain(one)
                        and isinstance(one, str)]
                 for name, values in fields.items()}
        numbers = {name: [one for one in values if isinstance(one, (int,
                                                                    float))
                          and not isinstance(one, bool)]
                   for name, values in fields.items()}
        lists = {name: [item for one in values if isinstance(one, list)
                        for item in one if _plain(item)]
                 for name, values in fields.items()}
        keys = model.keys(path) or [one for one in fields if texts[one]]
        for name, values in texts.items():
            if not values:
                continue
            value = rng.choice(values)
            target = rng.choice([one for one in keys if one != name] or
                                [None])
            out.append({"act": "count", "op": "eq", "target": path,
                        "field": name, "value": value,
                        "seed": f"how many {group} have {name} {value}?",
                        "spans": {"TARGET": [group], "FIELD": [name],
                                  "VALUE": [str(value)]}})
            if target:
                out.append({"act": "list", "op": "eq",
                            "target": f"{path}.{target}" if path else target,
                            "field": name, "value": value,
                            "seed": (f"which {target} have {name} "
                                     f"{value}?"),
                            "spans": {"TARGET": [target], "FIELD": [name],
                                      "VALUE": [str(value)]}})
                out.append({"act": "exists", "op": "ne",
                            "target": path, "field": name, "value": value,
                            "seed": (f"is there a {group} whose {name} is "
                                     f"not {value}?"),
                            "spans": {"TARGET": [group], "FIELD": [name],
                                      "VALUE": [str(value)]}})
        for name, values in numbers.items():
            if len(values) < 2:
                continue
            where = f"{path}.{name}" if path else name
            act = rng.choice(["max", "min", "sum", "mean"])
            said = {"max": "largest", "min": "smallest", "sum": "total",
                    "mean": "average"}[act]
            out.append({"act": act, "op": "none", "target": where,
                        "seed": f"what is the {said} {name} of the {group}?",
                        "spans": {"TARGET": [name]}})
            middle = sorted(values)[len(values) // 2]
            op = rng.choice(["gt", "lt"])
            out.append({"act": "count", "op": op, "target": path,
                        "field": name, "value": middle,
                        "seed": (f"how many {group} have {name} "
                                 f"{'more' if op == 'gt' else 'less'} than "
                                 f"{middle}?"),
                        "spans": {"TARGET": [group], "FIELD": [name],
                                  "VALUE": [str(middle)]}})
        for name, items in lists.items():
            if not items:
                continue
            value = rng.choice(items)
            target = rng.choice([one for one in keys if one != name] or
                                [None])
            if target:
                out.append({"act": "list", "op": "contains",
                            "target": f"{path}.{target}" if path else target,
                            "field": name, "value": value,
                            "seed": (f"which {target} have {value} in their "
                                     f"{name}?"),
                            "spans": {"TARGET": [target], "FIELD": [name],
                                      "VALUE": [str(value)]}})
    return out


def _files(rng: random.Random):
    """Real data files of commits taught: (path, text), shuffled by file
    kind so each kind is there."""
    from research.v700.teach_editor import split_of
    found = {"json": [], "yaml": [], "csv": []}
    for kind in found:
        for line in (SOURCE / f"{kind}.jsonl").open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = row.get("old_contents") or ""
            if split_of(row["commit"]) == "train" and 0 < len(text) < 60_000:
                found[kind].append((row["old_file"], text))
        rng.shuffle(found[kind])
    return found


def seeds() -> int:
    """Queries of real files, each said once: the teacher's input."""
    from research.v701 import datamodel
    rng = random.Random(SEED)
    files = _files(rng)
    # as many of each kind of question as can be had: what is asked is
    # spread, not what files are most of
    by_act: dict = {}
    for kind, found in files.items():
        for path, text in found[:4000]:
            model = datamodel.read(path, text)
            if model.error:
                continue
            made = queries(model, rng)
            rng.shuffle(made)
            for one in made[:EACH_FILE]:
                one["file"] = path
                by_act.setdefault((one["act"], one["op"]), []).append(one)
    out = []
    share = MOST_SEEDS // max(len(by_act), 1)
    for key in sorted(by_act):
        out += by_act[key][:share]
    DATA.mkdir(parents=True, exist_ok=True)
    with SEEDS.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    print({f"{act}|{op}": len(found[:share])
           for (act, op), found in sorted(by_act.items())}, len(out))
    return len(out)


def more_queries(model, rng: random.Random) -> list:
    """What the first reader read wrong, asked of a file: a value with a
    verb and the place around it (`which port does the server use`), yes
    or no of a setting (`is the teacher offline`), a record named by its
    key (`the title of the command graphTopology.ask`), and a filter whose
    field may go unsaid (`which users can see stark-db`)."""
    out = []
    inside = list(model.collections)
    for place in model.places.values():
        if any(place.path.startswith(f"{one}.") or place.path == one
               for one in inside) or not place.examples:
            continue
        name = _name(place.path)
        parent = _name(place.path.rsplit(".", 1)[0]) if "." in place.path \
            else ""
        if not _wordy(name) or not parent or not _wordy(parent):
            continue
        if "boolean" in place.types:
            out.append({"act": "value", "op": "none", "target": place.path,
                        "seed": rng.choice([f"is the {parent} {name}?",
                                            f"is {name} on for the {parent}?",
                                            f"does the {parent} have {name} "
                                            f"set?"]),
                        "spans": {"TARGET": [name]}})
        elif _plain(place.examples[0]):
            out.append({"act": "value", "op": "none", "target": place.path,
                        "seed": rng.choice([
                            f"which {name} does the {parent} use?",
                            f"what {name} is the {parent} on?",
                            f"the {parent}'s {name}?",
                            f"what does the {parent} have as {name}?"]),
                        "spans": {"TARGET": [name]}})
    for path, found in model.collections.items():
        if found["count"] < 3:
            continue
        group = _name(path) if path else model.path.rsplit(
            "/", 1)[-1].rsplit(".", 1)[0]
        if not _wordy(group):
            continue
        records = model.records()[path]
        keys = model.keys(path)
        for key in keys[:1]:
            others = [one for one in found["fields"] if one != key and
                      _wordy(str(one))]
            record = rng.choice(records)
            value = record.get(key) if isinstance(record, dict) else None
            if not others or not _plain(value):
                continue
            target = rng.choice(others)
            out.append({"act": "value", "op": "eq",
                        "target": f"{path}.{target}" if path else target,
                        "field": key, "value": value,
                        "seed": f"what is the {target} of the {group} "
                                f"{value}?",
                        "spans": {"TARGET": [target],
                                  "VALUE": [str(value)]}})
        for name in found["fields"]:
            values = [one.get(name) for one in records
                      if isinstance(one, dict)]
            words_ = [one for one in values if _plain(one)
                      and isinstance(one, str)]
            items = [item for one in values if isinstance(one, list)
                     for item in one if _plain(item)]
            if not _wordy(str(name)):
                continue
            if words_:
                value = rng.choice(words_)
                out.append({"act": "list", "op": "eq", "target": path,
                            "field": name, "value": value,
                            "seed": f"which {group} have {name} {value}?",
                            "spans": {"TARGET": [group], "FIELD": [name],
                                      "VALUE": [str(value)]},
                            "optional": ["FIELD"]})
            if items:
                value = rng.choice(items)
                out.append({"act": "list", "op": "contains", "target": path,
                            "field": name, "value": value,
                            "seed": (f"which {group} have {value} in their "
                                     f"{name}?"),
                            "spans": {"TARGET": [group], "FIELD": [name],
                                      "VALUE": [str(value)]},
                            "optional": ["FIELD"]})
    return out


def seeds_more(most: int = 160) -> int:
    """The second seeds: `more_queries` of real files, spread over their
    kinds -- by a choosing of their own, so the first seeds stay what they
    were."""
    from research.v701 import datamodel
    rng = random.Random(SEED + 1)
    files = _files(rng)
    by_kind: dict = {}
    for found in files.values():
        for path, text in found[:4000]:
            model = datamodel.read(path, text)
            if model.error:
                continue
            made = more_queries(model, rng)
            rng.shuffle(made)
            for one in made[:2]:
                one["file"] = path
                kind = (one["act"], one["op"], bool(one.get("optional")),
                        one["seed"].split()[0])
                by_kind.setdefault(kind, []).append(one)
    out = []
    share = most // max(len(by_kind), 1)
    for kind in sorted(by_kind):
        out += by_kind[kind][:share]
    with SEEDS_MORE.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    print({"|".join(map(str, kind)): len(found[:share])
           for kind, found in sorted(by_kind.items())}, len(out))
    return len(out)


#: what the second reader read as no question, live (v701): a count whose
#: filter's field goes unsaid (`how many people live in Toronto`), and the
#: record that holds the most of a field (`who is the oldest in people.csv`)
SEEDS_MORE2 = DATA / "seeds-more2.jsonl"


def more_queries2(model, rng: random.Random) -> list:
    out = []
    for path, found in model.collections.items():
        if found["count"] < 3:
            continue
        group = _name(path) if path else model.path.rsplit(
            "/", 1)[-1].rsplit(".", 1)[0]
        if not _wordy(group):
            continue
        records = [one for one in model.records()[path]
                   if isinstance(one, dict)]
        for name in found["fields"]:
            if not _wordy(str(name)):
                continue
            values = [one.get(name) for one in records]
            words_ = [one for one in values if _plain(one)
                      and isinstance(one, str)]
            numbers = [one for one in values if isinstance(one, (int, float))
                       and not isinstance(one, bool)]
            if words_:
                value = rng.choice(words_)
                out.append({"act": "count", "op": "eq", "target": path,
                            "field": name, "value": value,
                            "seed": f"how many {group} have {name} {value}?",
                            "spans": {"TARGET": [group], "FIELD": [name],
                                      "VALUE": [str(value)]},
                            "optional": ["FIELD"]})
            if len(numbers) >= 3:
                act = rng.choice(["max", "min"])
                most = "highest" if act == "max" else "lowest"
                out.append({"act": act, "op": "none", "target": path,
                            "field": name,
                            "seed": rng.choice([
                                f"which of the {group} has the {most} "
                                f"{name}?",
                                f"who in {group} has the {most} {name}?"]),
                            "spans": {"TARGET": [group], "FIELD": [name]},
                            "optional": ["FIELD"]})
    return out


def seeds_more2(most: int = 120) -> int:
    from research.v701 import datamodel
    rng = random.Random(SEED + 2)
    files = _files(rng)
    by_kind: dict = {}
    for found in files.values():
        for path, text in found[:4000]:
            model = datamodel.read(path, text)
            if model.error:
                continue
            made = more_queries2(model, rng)
            rng.shuffle(made)
            for one in made[:2]:
                one["file"] = path
                by_kind.setdefault(one["act"], []).append(one)
    out = []
    share = most // max(len(by_kind), 1)
    for kind in sorted(by_kind):
        out += by_kind[kind][:share]
    with SEEDS_MORE2.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    print({kind: len(found[:share]) for kind, found in sorted(
        by_kind.items())}, len(out))
    return len(out)


# -- the teacher says each again -----------------------------------------------

def _required(one: dict) -> dict:
    """The spans the teacher must keep: not an optional one -- a field the
    question may leave unsaid (`which users can see stark-db`)."""
    return {role: found for role, found in one["spans"].items()
            if role not in one.get("optional", ())}


def _kept(spans: dict) -> list:
    return [one for found in spans.values() for one in found]


def write(samples: int = 1, batch: int = 6) -> None:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for path in (SEEDS, SEEDS_MORE, SEEDS_MORE2)
            if path.exists() for line in path.open(encoding="utf-8")]
    done = set()
    if WRITTEN.exists():
        done = {json.loads(line)["seed"] for line in
                WRITTEN.open(encoding="utf-8")}
    todo = [one for one in rows if one["seed"] not in done]
    print(f"{len(todo)} seeds to say again", flush=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        prompts = [(f"Say this question about a data file in 12 different "
                    f"ways, as a person would type it to an assistant that "
                    f"has read the file: \"{one['seed']}\". Keep "
                    f"{', '.join(_kept(_required(one)))} exactly so in "
                    f"every one. " + STYLE) for one in chunk]
        replies = teacher.write(prompts, longest=600, samples=samples,
                                temperature=0.9)
        with WRITTEN.open("a", encoding="utf-8") as stream:
            for one, written in zip(chunk, replies):
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”`")
                        if 4 <= len(line) <= 200:
                            lines.append(line)
                stream.write(json.dumps({**one, "lines": lines}) + "\n")
        print(f"  {at + len(chunk)}/{len(todo)}", flush=True)


# -- the corpus -------------------------------------------------------------------

def labelled(line: str, spans: dict, optional=()) -> tuple | None:
    """(words, roles) of a question: each span's words where they are
    found, in the order given; None where one is not there -- but for an
    optional role's (a field the question may leave unsaid)."""
    from research.v698.teach_code_talk import _mark, _match
    said = words(line)
    if not said:
        return None
    roles = ["O"] * len(said)
    for role, found in spans.items():
        for phrase in found:
            start = 0
            where = _match(said, phrase, start)
            while where is not None and any(roles[at] != "O" for at in
                                            range(*where)):
                where = _match(said, phrase, where[1])
            if where is None:
                if role in optional:
                    continue
                return None
            _mark(roles, where, role)
    return said, roles


def record(said: list, roles: list, act: str, op: str, source: str) -> dict:
    return {"task": "data", "words": said, "roles": roles, "act": act,
            "op": op, "tags": [""] * len(said), "deps": [""] * len(said),
            "names": [], "said": " ".join(said), "source": source,
            "how": source}


def _swapped(said: list, roles: list, pool: dict, rng) -> list | None:
    """The words with each span put another file's word of its role: a
    field for a field, a value for a value."""
    out, at = [], 0
    while at < len(said):
        role = roles[at]
        if role.startswith("B-"):
            kind = role[2:]
            end = at + 1
            while end < len(said) and roles[end] == f"I-{kind}":
                end += 1
            new = words(rng.choice(pool["VALUE" if kind == "VALUE"
                                        else "NAME"]))
            if len(new) != end - at:
                return None
            out += new
            at = end
        else:
            out.append(said[at])
            at += 1
    return out


#: how often the fields' words are marked as naming the project's data
#: (`CODE`, as `querying.known` marks them), and how often a word of what
#: is not a data question is -- so the mark alone decides nothing
KNOWN, NOISE = 0.75, 0.15


def _tagged(rec: dict, rng: random.Random) -> dict:
    """The words told as the reader is told them at run time: those naming
    the project's data marked `CODE` -- a field's (TARGET, FIELD) most of
    the time; now and then one word of what is not a data question."""
    if rec["act"] != "none":
        if rng.random() < KNOWN:
            rec["tags"] = ["CODE" if role.endswith(("TARGET", "FIELD"))
                           else "" for role in rec["roles"]]
    elif rec["words"] and rng.random() < NOISE:
        at = rng.randrange(len(rec["words"]))
        rec["tags"] = ["CODE" if one == at else ""
                       for one in range(len(rec["words"]))]
    return rec


def corpus(negatives: int = 3000) -> dict:
    rng = random.Random(SEED)
    rows = {"train": [], "valid": []}
    written = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    pool = {"NAME": [], "VALUE": []}
    for one in written:
        for role, found in one["spans"].items():
            pool["VALUE" if role == "VALUE" else "NAME"] += [
                str(phrase) for phrase in found if len(words(str(phrase))) == 1]
    for one in written:
        held = rng.random() < 0.12
        spans = {role: [str(phrase) for phrase in found]
                 for role, found in one["spans"].items()}
        for line in dict.fromkeys(one["lines"] + [one["seed"]]):
            made = labelled(line, spans, one.get("optional", ()))
            if made is None:
                continue
            said, roles = made
            part = "valid" if held else "train"
            rows[part].append(_tagged(record(said, roles, one["act"],
                                             one["op"], "teacher"), rng))
            for _ in range(SWAPS):
                swapped = _swapped(said, roles, pool, rng)
                if swapped is not None:
                    rows[part].append(_tagged(record(
                        swapped, roles, one["act"], one["op"], "swapped"),
                        rng))
    # what is not a question about data: code talk, and everyday sentences
    others = []
    for path in sorted((LLM / "code-talk-data").glob("train-code.jsonl")):
        others += [json.loads(line)["words"] for line in
                   path.open(encoding="utf-8")]
    for path in sorted((LLM / "reader-data").glob("train*.jsonl")):
        for line in path.open(encoding="utf-8"):
            found = json.loads(line).get("words")
            if found and rng.random() < 0.05:
                others.append(found)
    for said in rng.sample(others, min(negatives, len(others))):
        said = [str(one).lower() for one in said]
        part = "valid" if rng.random() < 0.12 else "train"
        rows[part].append(_tagged(record(said, ["O"] * len(said), "none",
                                         "none", "not data"), rng))
    stats = {}
    for part, made in rows.items():
        rng.shuffle(made)
        with (DATA / f"{part}-data.jsonl").open("w", encoding="utf-8") as out:
            for one in made:
                out.write(json.dumps(one) + "\n")
        count: dict = {}
        for one in made:
            key = f"{one['act']}|{one['op']}"
            count[key] = count.get(key, 0) + 1
        stats[part] = {"records": len(made), "by label": dict(sorted(
            count.items()))}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1),
                                     encoding="utf-8")
    print({part: one["records"] for part, one in stats.items()})
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("seeds", "more", "more2", "write",
                                        "corpus"))
    options = parser.parse_args(argv)
    {"seeds": seeds, "more": seeds_more, "more2": seeds_more2,
     "write": write,
     "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

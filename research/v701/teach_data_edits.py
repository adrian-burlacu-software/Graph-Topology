"""Changes of data files, made by construction in real files: what the
editor is taught to change data with.

    python -m research.v701.teach_data_edits phrase   the teacher's statements
    python -m research.v701.teach_data_edits rows     llm/editor-data/data-edits.jsonl

Asked to add a user to `service.yaml`, the editor (taught on code commits
and code's faults) wrote seven changes; one read as YAML and named what
was asked, and it gave the user a project nobody asked for. Changes of
data are made here where they are true -- in CommitPackFT's YAML, JSON and
CSV files (commits taught only) -- as the file's own text, so its layout
stays what it was:

- **set**: a value's text replaced (`port: 8697` -> `port: 9000`);
- **add**: a record written after the last, its lines copied from it with
  the new values put in (a CSV's row; a YAML or JSON list's item);
- **remove**: a record taken out, by its key (`remove the user sam`).

Each is told in a statement the teacher wrote (`phrase`), the names put
back in; shown as the runtime shows it (`fixing.part_of`: the lines of
the file around what the statement names), answered as blocks.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.v700 import teach_editor as T

PHRASES = T.OUT / "data-edit-phrases.jsonl"
EDITS = T.OUT / "data-edits.jsonl"
SEED = 701
#: the most of each kind
MOST = {"set": 2500, "add": 2000, "remove": 1500}

SEEDS = {
    "set": ("set the max_retries of the uploader in jobs.yaml to 7",
            {"max_retries": "{x}", "uploader": "{p}", "jobs.yaml": "{f}",
             "7": "{v}"}),
    "add": ("add a gateway with region eu-west and port 7443 to "
            "gateways.yaml",
            {"gateway": "{g}", "region": "{x}", "eu-west": "{v}",
             "port": "{y}", "7443": "{w}", "gateways.yaml": "{f}"}),
    "remove": ("remove the gateway eu-west from gateways.yaml",
               {"gateway": "{g}", "eu-west": "{v}",
                "gateways.yaml": "{f}"}),
}
STORY = ("gateway", "region", "uploader", "retries", "jobs")


def phrase(samples: int = 3) -> None:
    from research.v696.teach_meaning import Teacher
    T.OUT.mkdir(parents=True, exist_ok=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    jobs = [(kind, seed, names, (
        f"Say this request to an assistant that edits a project's files in "
        f"15 different ways, as a person would type it: short and long, "
        f"casual and terse. Keep {', '.join(names)} exactly so in every "
        f"one. One per line, nothing else: \"{seed}\""))
        for kind, (seed, names) in SEEDS.items()]
    replies = teacher.write([one[3] for one in jobs], longest=900,
                            samples=samples, temperature=0.9)
    with PHRASES.open("w", encoding="utf-8") as out:
        for (kind, seed, names, _), written in zip(jobs, replies):
            kept = {seed}
            for reply in written:
                for line in reply.splitlines():
                    line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                    line = line.strip().strip("\"'“”`")
                    if 8 <= len(line) <= 200 and all(
                            re.search(r"(?<![\w.-])" + re.escape(name)
                                      + r"(?![\w-])", line)
                            for name in names):
                        kept.add(line)
            for line in sorted(kept):
                said = line
                for name, place in sorted(names.items(),
                                          key=lambda one: -len(one[0])):
                    said = re.sub(r"(?<![\w.-])" + re.escape(name)
                                  + r"(?![\w-])", place, said)
                if not any(word in said.lower() for word in STORY):
                    out.write(json.dumps({"kind": kind, "phrase": said})
                              + "\n")
            print(f"{kind}: {len(kept)} phrases", flush=True)


# -- the changes, made in a file's own text --------------------------------------------

def _one(word: str) -> str:
    """A collection's name said of one of it (`tasks`: `task`)."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and         not word.endswith("ss") else word


def _scalar(value) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(
        value, bool) and 0 < len(str(value)) <= 30 and "\n" not in str(value)


def _wordy(name) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,30}", str(name)))


def _other(value, pool: list, rng):
    """A new value of the same kind: a number moved, a word from what
    other files hold."""
    if isinstance(value, int):
        return value + rng.choice([1, 2, 10, 100, -1]) if value > 2 else \
            value + rng.choice([1, 2, 5])
    if isinstance(value, float):
        return round(value * rng.choice([0.5, 2, 1.5]), 3)
    choices = [one for one in pool if one != value]
    return rng.choice(choices) if choices else f"{value}2"


def _key_line(lines: list, name: str, value, kind: str) -> int | None:
    """The one line saying `name: value` (YAML) or `"name": value` (JSON)."""
    said = json.dumps(value) if kind == "json" else None
    found = []
    for at, line in enumerate(lines):
        if kind == "json":
            if re.search(r'"' + re.escape(str(name)) + r'"\s*:\s*'
                         + re.escape(said), line):
                found.append(at)
        elif re.match(r"\s*(-\s+)?" + re.escape(str(name)) + r"\s*:\s*['\"]?"
                      + re.escape(str(value)) + r"['\"]?\s*(#.*)?$",
                      line.rstrip("\r\n")):
            found.append(at)
    return found[0] if len(found) == 1 else None


def _set(model, kind: str, lines: list, pool: list, rng):
    """A value set: its line, its text replaced."""
    plain = [one for one in model.places.values()
             if one.count == 1 and one.examples and _scalar(
                 one.examples[0]) and _wordy(one.path.rsplit(".", 1)[-1])
             and "[]" not in one.path]
    rng.shuffle(plain)
    for place in plain[:5]:
        name = place.path.rsplit(".", 1)[-1]
        value = place.examples[0]
        at = _key_line(lines, name, value, kind)
        if at is None:
            continue
        new = _other(value, pool, rng)
        old_text = json.dumps(value) if kind == "json" else str(value)
        new_text = json.dumps(new) if kind == "json" else str(new)
        line = lines[at]
        head, sep, tail = line.partition(":")
        if old_text not in tail:
            continue
        changed = head + sep + tail.replace(old_text, new_text, 1)
        parent = place.path.rsplit(".", 1)[0].rsplit(".", 1)[-1] \
            if "." in place.path else ""
        return {"kind": "set", "at": at, "old": [line], "new": [changed],
                "names": {"x": name, "v": str(new),
                          "p": parent if _wordy(parent) else ""}}
    return None


def _yaml_items(lines: list, collection: str):
    """The line spans of a YAML list of records: [(first, end)] -- each
    item from its `- ` line to the line before the next at its indent."""
    name = collection.replace("[]", "").rsplit(".", 1)[-1]
    head = next((at for at, line in enumerate(lines)
                 if re.match(r"\s*" + re.escape(name) + r"\s*:\s*$",
                             line.rstrip("\r\n"))), None)
    if head is None:
        return []
    items, indent, at = [], None, head + 1
    while at < len(lines):
        line = lines[at]
        if not line.strip():
            at += 1
            continue
        lead = len(line) - len(line.lstrip())
        if line.lstrip().startswith("- "):
            if indent is None:
                indent = lead
            if lead == indent:
                items.append([at, at + 1])
                at += 1
                continue
        if indent is None or lead <= indent - (0 if line.lstrip()
                                               .startswith("- ") else 1):
            if indent is not None and lead <= indent and not \
                    line.lstrip().startswith("- "):
                break
        if items:
            items[-1][1] = at + 1
        at += 1
    return [tuple(one) for one in items]


def _add_remove(model, kind: str, lines: list, pool: list, rng):
    """A record added after the last (its lines copied, its values new),
    or one taken out by its key."""
    if kind not in ("yaml", "csv", "tsv"):
        return None
    for path, found in model.collections.items():
        keys = model.keys(path)
        if not keys or found["count"] < 3:
            continue
        key = keys[0]
        records = model.records()[path]
        group = path.replace("[]", "").rsplit(".", 1)[-1] if path else \
            model.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if not _wordy(group) or not _wordy(key):
            continue
        if kind == "yaml":
            items = _yaml_items(lines, path)
        else:
            items = [(at, at + 1) for at in range(1, len(lines))
                     if lines[at].strip()]
        if not items or len(items) != len(records):
            continue
        if rng.random() < 0.55:
            # add: the last record's lines, its values new
            first, end = items[-1]
            last = records[-1]
            block = lines[first:end]
            new_key = _other(last.get(key), pool, rng)
            if any(str(one.get(key)) == str(new_key) for one in records):
                continue
            text = "".join(block)
            fields = [one for one in found["fields"] if _scalar(
                last.get(one)) and one != key and _wordy(one)]
            changes = {key: new_key}
            for one in rng.sample(fields, min(len(fields), 1)):
                others = [r.get(one) for r in records if _scalar(r.get(one))]
                changes[one] = rng.choice(others)
            if kind == "yaml":
                out = text
                for name, value in changes.items():
                    out, n = re.subn(
                        r"(^\s*(-\s+)?" + re.escape(str(name))
                        + r"\s*:\s*)(['\"]?)" + re.escape(str(last.get(name)))
                        + r"\3(\s*)$", lambda m: f"{m.group(1)}{m.group(3)}"
                        f"{value}{m.group(3)}{m.group(4)}", out, count=1,
                        flags=re.M)
                    if n != 1:
                        out = None
                        break
            else:
                cells = [str(last.get(one)) if last.get(one) is not None
                         else "" for one in found["fields"]]
                for name, value in changes.items():
                    cells[found["fields"].index(name)] = str(value)
                ending = "\r\n" if text.endswith("\r\n") else "\n"
                out = ("\t" if kind == "tsv" else ",").join(cells) + ending
                if any("," in one or '"' in one for one in cells):
                    out = None
            if not out:
                continue
            new_lines = out.splitlines(True)
            if not block[-1].endswith("\n"):
                continue
            said = {"g": _one(group), "x": key, "v": str(new_key)}
            extra = [one for one in changes if one != key]
            if extra:
                said.update({"y": extra[0], "w": str(changes[extra[0]])})
            else:
                said.update({"y": key, "w": str(new_key)})
            return {"kind": "add", "at": end - 1, "old": [block[-1]],
                    "new": [block[-1]] + new_lines, "names": said}
        # remove: a record by its key, not the first
        pick = rng.randrange(1, len(records))
        first, end = items[pick]
        value = records[pick].get(key)
        if not _scalar(value):
            continue
        before = lines[first - 1]
        return {"kind": "remove", "at": first - 1,
                "old": [before] + lines[first:end], "new": [before],
                "names": {"g": _one(group), "v": str(value)}}
    return None


def row_of(record: dict, phrases: dict, pool: list, rng) -> dict | None:
    from research.v701 import datamodel
    path, text = record["old_file"], record.get("old_contents") or ""
    kind = datamodel.format_of(path)
    if kind is None or not text or len(text) > 60_000:
        return None
    model = datamodel.read(path, text)
    if model.error:
        return None
    lines = text.splitlines(True)
    if not lines or not lines[-1].endswith("\n"):
        return None
    made = None
    if rng.random() < 0.45:
        made = _set(model, kind, lines, pool, rng)
    if made is None:
        made = _add_remove(model, kind, lines, pool, rng)
    if made is None:
        made = _set(model, kind, lines, pool, rng)
    if made is None:
        return None
    at, old, new = made["at"], made["old"], made["new"]
    first = max(0, at - T.AROUND)
    end = min(len(lines), at + len(old) + T.AROUND)
    part = "".join(lines[first:end])
    if end - first > T.LONGEST_PART:
        return None
    target = T.said([(old, new)])
    if T.applied(part, target) is None:
        return None
    after = "".join(lines[:at]) + "".join(new) + "".join(lines[at + len(old):])
    if datamodel.read(path, after).error:
        return None
    pool_ = phrases.get(made["kind"])
    if not pool_:
        return None
    names = dict(made["names"])
    names["f"] = path.rsplit("/", 1)[-1]
    for one in ("x", "y", "v", "w", "g", "p"):
        names.setdefault(one, "")
    usable = [one for one in pool_ if ("{p}" not in one or names["p"])]
    if not usable:
        return None
    statement = rng.choice(usable).format(**names)
    statement = re.sub(r"\s+", " ", statement).strip()
    return {"commit": record["commit"], "split": T.split_of(record["commit"]),
            "language": kind, "statement": statement, "path": path,
            "part": part, "start": first + 1, "target": target,
            "source": f"data {made['kind']}"}


def rows() -> dict:
    from research.v700.teach_faults import _clean
    phrases: dict = {}
    for line in PHRASES.open(encoding="utf-8"):
        row = json.loads(line)
        said = _clean(row["phrase"])
        # the teacher's own instruction said back is no request
        if said and not re.search(r"exactly so|every one|different ways",
                                  said, re.I):
            phrases.setdefault(row["kind"], []).append(said)
    rng = random.Random(SEED)
    records = []
    for kind in ("yaml", "json", "csv"):
        for line in (T.DATA / f"{kind}.jsonl").open(encoding="utf-8"):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    rng.shuffle(records)
    # words other files hold: the new values a change puts in
    pool = []
    from research.v701 import datamodel
    for record in records[:3000]:
        model = datamodel.read(record["old_file"],
                               record.get("old_contents") or "")
        if model.error:
            continue
        for place in model.places.values():
            pool += [one for one in place.examples
                     if isinstance(one, str) and _wordy(one)]
    pool = sorted(set(pool))
    made, counts = [], {}
    for record in records:
        row = row_of(record, phrases, pool, rng)
        if row is None or counts.get(row["source"], 0) >= MOST[
                row["source"].split()[1]]:
            continue
        counts[row["source"]] = counts.get(row["source"], 0) + 1
        made.append(row)
        if all(counts.get(f"data {one}", 0) >= most
               for one, most in MOST.items()):
            break
    with EDITS.open("w", encoding="utf-8") as out:
        for row in made:
            out.write(json.dumps(row) + "\n")
    split = {}
    for row in made:
        key = f"{row['source']}|{row['split']}"
        split[key] = split.get(key, 0) + 1
    print(json.dumps(dict(sorted(split.items()))))
    return split


COMMITS = T.OUT / "data-commits.jsonl"
MIX = T.OUT / "edits-mix-data.jsonl"


def commits(most: int = 6000) -> int:
    """Commits of real data files, as the editor's commits are read
    (`teach_editor.row_of`): what people say a change of data is."""
    rng = random.Random(SEED)
    rows_, seen = [], set()
    for kind in ("yaml", "json", "csv"):
        for line in (T.DATA / f"{kind}.jsonl").open(encoding="utf-8"):
            try:
                row = T.row_of(json.loads(line))
            except ValueError:
                continue
            if row is None or (row["statement"], row["part"]) in seen:
                continue
            seen.add((row["statement"], row["part"]))
            rows_.append(row)
    rng.shuffle(rows_)
    rows_ = rows_[:most]
    with COMMITS.open("w", encoding="utf-8") as out:
        for row in rows_:
            out.write(json.dumps(row) + "\n")
    print(len(rows_), "data commits")
    return len(rows_)


def mix(kept: int = 12000) -> dict:
    """What editor4 is taught from editor3: the changes of data (twice),
    data files' commits, and editor3's own mix, so what it was taught is
    kept."""
    rng = random.Random(SEED)
    data = [json.loads(line) for line in EDITS.open(encoding="utf-8")]
    real = [json.loads(line) for line in COMMITS.open(encoding="utf-8")]
    before = [json.loads(line) for line in T.MIX.open(encoding="utf-8")]
    before = [one for one in before if one["split"] == "train"]
    out = [one for one in data + real if one["split"] != "train"]
    out += [one for one in data if one["split"] == "train"] * 2
    out += [one for one in real if one["split"] == "train"]
    out += rng.sample(before, min(kept, len(before)))
    rng.shuffle(out)
    with MIX.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    counts = {"train rows": sum(one["split"] == "train" for one in out),
              "data edits": len(data), "data commits": len(real)}
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("phrase", "rows", "commits", "mix"))
    options = parser.parse_args(argv)
    {"phrase": phrase, "rows": rows, "commits": commits,
     "mix": mix}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Teaching the shared reader questions of data asked one after another,
from SParC's train split: what each asks, of which fields, held to which
values -- and whether it asks of the records the last one answered with
(2026-10-24, follow-ups over data).

    python -m research.v701.teach_data_sparc corpus

SParC (`research/v704/sparc.py`) gives each turn of a sequence the SQL
that answers it. Read at teach time (`parse`: one SELECT, its filter a
conjunction of `column op literal`; no nesting, grouping or sets), the SQL
says what the reader is taught:

    data_act   count/mean/sum/max/min of an aggregate; max/min of an order
               kept to one; value where it asks of the one record the last
               turn ordered out (`what is its id?`); else list
    data_op    the new condition's comparison (`!=`: ne, `LIKE`: contains)
    data_role  TARGET the selected columns (a table counted whole: its
               name), FIELD the new condition's column, VALUE its literal --
               marked where the words are found (`teach_data_talk.labelled`)
    data_of    last where the turn keeps the last turn's filter (or the one
               record it ordered out), said or not (`how many are there?`);
               else none

A turn after one not read is left out (whether it leans on it is not
known), and so is a turn with two new conditions. As many asked of the last
are taught as not (SParC's ways of labelling a fifth of the reader's own,
not more: all 4 831 cost the core six questions). A turn the reader in use
reads as asking something done is taught to code talk as none (`show the
number for each shop` was read as a change, and the data file written) --
only those: a question of a constant is code talk's.

Only train is taught from; dev is what is said (`research/v704/sparc.py`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "sparc" / "train.json"
LLM = ROOT / "llm"
SEED = 7052
#: of each kept turn: valid apart, by its sequence
HELD = 0.1

AGG = {"count": "count", "avg": "mean", "sum": "sum", "max": "max",
       "min": "min"}
OPS = {"=": "eq", "!=": "ne", "<>": "ne", ">": "gt", "<": "lt", ">=": "ge",
       "<=": "le", "like": "contains", "not like": "ne"}
SHAPE = re.compile(
    r"^select\s+(?P<distinct>distinct\s+)?(?P<select>.+?)\s+from\s+"
    r"(?P<from>.+?)(?:\s+where\s+(?P<where>.+?))?"
    r"(?:\s+order\s+by\s+(?P<order>.+?))?(?:\s+limit\s+(?P<limit>\d+))?$",
    re.I | re.S)
CONDITION = re.compile(
    r"^(?P<col>[\w.]+)\s*(?P<op>not\s+like|like|!=|<>|>=|<=|=|>|<)\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|-?\d+(?:\.\d+)?)$", re.I)


def _column(text: str) -> str:
    return text.strip().split(".")[-1]


def parse(sql: str) -> dict | None:
    """A simple query's parts, or None: one SELECT (no nesting, grouping,
    sets); its filter a conjunction of `column op literal`."""
    sql = " ".join(sql.replace(";", " ").split())
    lowered = sql.lower()
    if lowered.count("select") != 1 or re.search(
            r"\b(group by|having|union|intersect|except|in \(|not in|"
            r"between|or)\b", lowered):
        return None
    found = SHAPE.match(sql)
    if not found:
        return None
    items = []
    for item in found["select"].split(","):
        item = item.strip()
        call = re.match(r"(\w+)\s*\(\s*(distinct\s+)?([\w.*]+)\s*\)$", item,
                        re.I)
        if call:
            if call[1].lower() not in AGG:
                return None
            items.append((AGG[call[1].lower()], _column(call[3])))
        elif re.match(r"^[\w.*]+$", item):
            items.append((None, _column(item)))
        else:
            return None
    tables = re.findall(r"(?:^|\bjoin\s+)(\w+)", found["from"], re.I)
    conditions = []
    if found["where"]:
        for part in re.split(r"\s+and\s+", found["where"], flags=re.I):
            one = CONDITION.match(part.strip())
            if not one:
                return None
            value = one["value"].strip("\"'")
            op = OPS[" ".join(one["op"].lower().split())]
            if op in ("contains", "ne") and "%" in value:
                value = value.strip("%")
            conditions.append((_column(one["col"]).lower(), op, value))
    order = None
    if found["order"]:
        part = found["order"].split(",")[0].strip()
        way = "max" if part.lower().endswith(" desc") else "min"
        part = re.sub(r"\s+(asc|desc)$", "", part, flags=re.I)
        call = re.match(r"(\w+)\s*\(\s*([\w.*]+)\s*\)$", part)
        order = (way, _column(call[2] if call else part).lower(), bool(call))
    return {"items": items, "tables": tables, "conditions": conditions,
            "order": order, "limit": found["limit"]}


#: what the data part does not look up yet, as SQL says it: grouped,
#: either-or, a range or set, nested, joined in sets
BEYOND = re.compile(r"\b(group by|having|union|intersect|except|in \(|not "
                    r"in|between|or)\b", re.I)
#: what it asks of the records where the SQL asks what is not looked up yet
OUT_OF_REACH = {"act": "beyond", "op": "none", "spans": {}, "table": []}


def beyond(sql: str) -> bool:
    """Whether the SQL asks what the data part does not look up yet: grouped,
    either-or, nested, a set -- or sorted, or the top few."""
    lowered = " ".join(sql.lower().split())
    if lowered.count("select") > 1 or BEYOND.search(lowered):
        return True
    found = SHAPE.match(" ".join(sql.replace(";", " ").split()))
    return bool(found and found["order"] and found["limit"] != "1")


def label(parsed: dict, last: dict | None) -> dict | None:
    """act, op, the spans and whether it is of the last turn's records --
    or None where more than one condition is new, or acts are mixed."""
    items, conditions = parsed["items"], parsed["conditions"]
    if parsed["order"] and parsed["limit"] != "1":
        # sorted, or the top few: not looked up yet
        return dict(OUT_OF_REACH)
    ordered_one = bool(last and last["order"] and last["limit"] == "1" and
                       last["order"] == parsed["order"] and
                       parsed["limit"] == "1")
    # of the last: its filter kept, or the one record it ordered out -- or
    # all of a table it showed whole, a filter put on them now (`which of
    # those have more than 18000 students?` after `show all colleges`: of
    # them all, as true)
    whole = bool(last and not last["conditions"] and not last["order"] and
                 last["tables"][:1] == parsed["tables"][:1] and conditions)
    inherited = last is not None and (
        (last["conditions"] and set(last["conditions"]) <= set(conditions))
        or ordered_one or whole)
    new = [one for one in conditions
           if not (inherited and one in last["conditions"])]
    if len(new) > 1:
        return None
    aggs = {agg for agg, _ in items if agg}
    if len(aggs) > 1:
        return None
    if aggs:
        act = aggs.pop()
    elif ordered_one:
        act = "value"
    elif parsed["order"] and parsed["limit"] == "1":
        act = parsed["order"][0]
    else:
        act = "list"
    targets = [col for _, col in items if col != "*"]
    if not targets:
        targets = parsed["tables"][:1]
    spans = {"TARGET": targets}
    op = "none"
    if new:
        field, op, value = new[0]
        spans["FIELD"], spans["VALUE"] = [field], [value]
    return {"act": act, "op": op, "spans": spans,
            "of": "last" if inherited else "none",
            "table": parsed["tables"][:1]}


def _spelled(name: str) -> list:
    """A column's name as said: `date_of_latest_logon` -> date of latest
    logon, `pName` -> p name."""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [" ".join(name.replace("_", " ").lower().split())]


def labelled_turns() -> list:
    """(sequence number, words said, the label) of each turn read."""
    out = []
    for at, one in enumerate(json.loads(SOURCE.read_text(encoding="utf-8"))):
        last, first = None, True
        for said, sql in zip(one["interaction_utterance"],
                             one["interaction_query"]):
            parsed = parse(sql)
            got = label(parsed, last) if parsed and (first or last) \
                else None
            if parsed is None and beyond(sql):
                # what it asks is not looked up yet -- whatever it leans on
                got = dict(OUT_OF_REACH)
            first = False
            last = parsed
            if got is not None:
                out.append((at, said, got))
    return out


def corpus() -> dict:
    from research.v701 import teach_data_talk as D
    from research.v698 import teach_code_talk as K
    rng = random.Random(SEED)
    rows = {"train": [], "valid": []}
    code = {"train": [], "valid": []}
    held = {}
    from research.v698 import reading
    for at, said, got in labelled_turns():
        part = held.setdefault(at, "valid" if rng.random() < HELD
                               else "train")
        # what is asked, as the reader's own teaching marks it: the field
        # asked for, or the records' name where the question says it
        # (`list the books published before 1970`: books)
        spans = {"TARGET": [one for name in got["spans"].get("TARGET", []) +
                            got["table"] for one in _spelled(name)]}
        if "FIELD" in got["spans"]:
            spans["FIELD"] = _spelled(got["spans"]["FIELD"][0])
            spans["VALUE"] = got["spans"]["VALUE"]
        # the field asked for, and a filter's field, may go unsaid; its
        # value may not
        made = D.labelled(said, spans, optional=("TARGET", "FIELD"))
        if made is None:
            continue
        words, roles = made
        record = D._tagged(D.record(words, roles, got["act"], got["op"],
                                    "sparc"), rng)
        if "of" in got:
            record["of"] = got["of"]
        rows[part].append(record)
        # no code talk, taught where the reader in use reads it as asking
        # something done (`show the number for each shop` read as a change,
        # the data file written) -- not every question of records: half of
        # them it reads as asking of code, which the page gives the data,
        # and a constant's is code talk's (`how many days does a loan last
        # by default?`, lost when all 4 831 were taught none)
        read = reading.code_talk(said)
        if read is not None and read.act in reading.DOING + ("run",):
            lowered = [str(one) for one in K.words(said)]
            code[part].append(K.record(lowered, ["O"] * len(lowered),
                                       "none", "none", "none", "sparc"))
    counts = {}
    for part in rows:
        # as many asked of the last as not: the reader's own teaching
        # outweighs SParC's ways of labelling (17 710 records beside them)
        last = [one for one in rows[part] if one.get("of") == "last"]
        plain = [one for one in rows[part] if one.get("of") == "none"]
        out = [one for one in rows[part] if one["act"] == "beyond"]
        rows[part] = last + rng.sample(plain, min(len(plain), len(last))) + \
            rng.sample(out, min(len(out), len(last)))
        counts[f"{part} beyond"] = min(len(out), len(last))
        rng.shuffle(rows[part])
        counts[f"{part} code"] = len(code[part])
        with (LLM / "data-talk-data" / f"{part}-sparc.jsonl").open(
                "w", encoding="utf-8") as stream:
            for one in rows[part]:
                stream.write(json.dumps(one) + "\n")
        with (LLM / "code-talk-data" / f"{part}-sparc.jsonl").open(
                "w", encoding="utf-8") as stream:
            for one in code[part]:
                stream.write(json.dumps(one) + "\n")
        counts[part] = len(rows[part])
        counts[f"{part} last"] = sum(one.get("of") == "last"
                                     for one in rows[part])
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("corpus",))
    parser.parse_args(argv)
    corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())

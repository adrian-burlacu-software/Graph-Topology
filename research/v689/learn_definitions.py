"""Definitions memory, offline: every noun gloss in the store, read into facts.

    python -m research.v689.learn_definitions read   [--workers 6] [--limit N]
    python -m research.v689.learn_definitions check  [--limit N]
    python -m research.v689.learn_definitions report [--sample 60]
    python -m ingestion.load --source definitions --into data/v684_definitions.sqlite

Three stages, each resumable and each timed, so the full run can be priced
from a small one before it is paid for.

**read** -- spaCy and v687's reasoner, no GPU. Each gloss becomes a
`definitions.Reading` in `state/v689-definitions.sqlite`, the same memory the
conversation page fills when v688 retrieves a definition. A concept already
there is skipped unless `--again`.

**check** -- the teacher, asked each fact as the bare question it amounts to,
under `CAREFUL` at `SETTLING_FLOOR`. A fact comes back agreed, disputed or
below the floor. A disputed fact is kept, so the audit can count it, and is
never read by the rules. About twenty judgements a second on one GPU, and the
judgement cache makes a re-run free.

**report** -- what was read: facts per gloss, by relation and by rule, how
often the genus agreed with the store's taxonomy, how the teacher judged each
rule and each kind of genus, and a sample to grade by hand.

The evaluation is `audit.py` against a copy of the store with the defined
facts loaded as one more source, the way GenericsKB was measured
(`AUDIT.md` §13): the two stores differ only in the source under test.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .definitions import Defined, DefinitionMemory, Reading, question_for
from .episodic import name_of

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"
MEMORY = REPOSITORY_ROOT / "state" / "v689-definitions.sqlite"

#: Glosses a worker reads between reports.
BATCH = 250

_READER = None


def _reader():
    """One gloss reader per worker process, built on first use."""
    global _READER
    if _READER is None:
        from research.v687.language import Parser
        from research.v687.reason import Reasoner

        from .asker import Asker
        from .definitions import GlossReader

        reasoner = Reasoner(STORE)
        parser = Parser(vocabulary=reasoner.vocabulary(),
                        nouns=reasoner.noun_vocabulary())
        _READER = GlossReader(Asker(reasoner, parser))
    return _READER


def _read_batch(batch: list) -> list:
    reader = _reader()
    out = []
    for concept, gloss in batch:
        try:
            out.append(reader.read(concept, gloss).as_dict())
        except Exception as bad:                    # noqa: BLE001
            out.append({"concept": concept, "gloss": gloss,
                        "error": f"{type(bad).__name__}: {bad}"})
    return out


def concepts(limit: int = 0, with_facts: bool = False) -> list:
    """(concept, gloss) for every noun with a gloss, as a strided sample
    when `limit` is given, so a small run is a sub-sample of a large one."""
    connection = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    sql = ("SELECT id, definition FROM concepts WHERE pos = 'n' "
           "AND definition IS NOT NULL AND definition != ''")
    if with_facts:
        sql += " AND id IN (SELECT DISTINCT concept FROM facts)"
    rows = connection.execute(sql + " ORDER BY id").fetchall()
    connection.close()
    if limit and limit < len(rows):
        stride = len(rows) / limit
        rows = [rows[int(index * stride)] for index in range(limit)]
    return rows


def reading_of(found: dict) -> Reading:
    return Reading(found["concept"], found["gloss"], found.get("genus", ""),
                   found.get("agrees"),
                   [Defined(**fact) for fact in found.get("facts", [])],
                   list(found.get("unread", [])))


def read(workers: int = 6, limit: int = 0, with_facts: bool = False,
         again: bool = False, memory_path: Path = MEMORY) -> dict:
    memory = DefinitionMemory(memory_path)
    known = {row[0] for row in memory.connection.execute(
        "SELECT concept FROM glosses")}
    todo = [row for row in concepts(limit, with_facts)
            if again or row[0] not in known]
    batches = [todo[index:index + BATCH]
               for index in range(0, len(todo), BATCH)]
    started = time.time()
    done = errors = facts = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for results in pool.map(_read_batch, batches):
            readings = []
            for found in results:
                if "error" in found:
                    errors += 1
                    continue
                readings.append(reading_of(found))
            memory.keep_all(readings, "offline")
            done += len(readings)
            facts += sum(len(one.facts) for one in readings)
            elapsed = time.time() - started
            print(f"[read] {done + errors:,}/{len(todo):,} glosses, "
                  f"{facts:,} facts, {errors} errors, "
                  f"{(done + errors) / max(elapsed, 1e-9):.0f}/s", flush=True)
    elapsed = time.time() - started
    return {"glosses": done, "errors": errors, "facts": facts,
            "seconds": round(elapsed, 1),
            "per_second": round((done + errors) / max(elapsed, 1e-9), 1),
            "workers": workers}


def check(limit: int = 0, memory_path: Path = MEMORY,
          dry: bool = False) -> dict:
    """Ask the teacher about every fact not yet checked.

    `dry` loads no model: it counts how many questions the judgement cache
    already answers, which is what prices a real run.
    """
    from research.v688.teacher import SETTLING_FLOOR, Teacher

    memory = DefinitionMemory(memory_path)
    rows = memory.connection.execute(
        "SELECT concept, relation, object, rule FROM defined "
        "WHERE checked IS NULL ORDER BY rowid").fetchall()
    if limit:
        rows = rows[:limit]
    if dry:
        teacher = Teacher(load=False)
        cached = sum(teacher.key("", "", question_for(
            name_of(concept), relation, obj, rule)) in teacher.cache
                     for concept, relation, obj, rule in rows)
        return {"unchecked": len(rows), "cached": cached,
                "fresh": len(rows) - cached,
                "gpu_hours_at_20_per_second": round(
                    (len(rows) - cached) / 20 / 3600, 2)}
    teacher = Teacher()
    if not teacher.available:
        return {"error": teacher.error or "no teacher"}
    started = time.time()
    counts: collections.Counter = collections.Counter()
    updates: list = []
    fresh = 0
    with teacher.batch():
        for number, (concept, relation, obj, rule) in enumerate(rows, 1):
            question = question_for(name_of(concept), relation, obj, rule)
            supports, confidence, cached = teacher.judge("", "", question)
            fresh += not cached
            expected = not relation.startswith("not_")
            verdict = ("below" if confidence < SETTLING_FLOOR else
                       "agreed" if supports == expected else "disputed")
            counts[verdict] += 1
            updates.append((verdict, confidence, concept, relation, obj))
            if len(updates) >= 500 or number == len(rows):
                memory.check_all(updates)
                updates.clear()
                elapsed = time.time() - started
                print(f"[check] {number:,}/{len(rows):,} "
                      f"{dict(counts)} {fresh / max(elapsed, 1e-9):.1f} "
                      f"fresh/s", flush=True)
    elapsed = time.time() - started
    return {"checked": len(rows), "fresh": fresh, **dict(counts),
            "seconds": round(elapsed, 1),
            "fresh_per_second": round(fresh / max(elapsed, 1e-9), 1)}


def report(sample: int = 60, memory_path: Path = MEMORY) -> dict:
    memory = DefinitionMemory(memory_path)
    read_sql = memory.connection.execute
    glosses = read_sql("SELECT count(*), sum(genus != ''), sum(agrees = 1), "
                       "sum(unread != '') FROM glosses").fetchone()
    total = glosses[0] or 1
    facts = read_sql("SELECT count(*) FROM defined").fetchone()[0]
    by_relation = dict(read_sql(
        "SELECT relation, count(*) FROM defined GROUP BY 1 ORDER BY 2 DESC"))
    by_rule = {}
    for rule, verdict, count in read_sql(
            "SELECT rule, coalesce(checked, 'unchecked'), count(*) "
            "FROM defined GROUP BY 1, 2"):
        by_rule.setdefault(rule, {})[verdict] = count
    by_genus = {}
    for agrees, verdict, count in read_sql(
            "SELECT g.agrees, coalesce(d.checked, 'unchecked'), count(*) "
            "FROM defined d JOIN glosses g USING (concept) GROUP BY 1, 2"):
        key = {1: "genus agrees", 0: "genus disagrees"}.get(agrees,
                                                           "no genus")
        by_genus.setdefault(key, {})[verdict] = count
    rows = read_sql(
        "SELECT d.concept, g.gloss, d.relation, d.object, d.rule, d.checked "
        "FROM defined d JOIN glosses g USING (concept)").fetchall()
    random.seed(0)
    picked = random.sample(rows, min(sample, len(rows))) if rows else []
    return {"glosses": glosses[0], "facts": facts,
            "facts_per_gloss": round(facts / total, 2),
            "genus_found": round((glosses[1] or 0) / total, 4),
            "genus_agrees": round((glosses[2] or 0) / total, 4),
            "some_piece_unread": round((glosses[3] or 0) / total, 4),
            "by_relation": by_relation, "by_rule": by_rule,
            "by_genus": by_genus,
            "sample": [{"concept": concept, "gloss": gloss,
                        "fact": f"{relation} {obj}", "rule": rule,
                        "checked": checked}
                       for concept, gloss, relation, obj, rule, checked
                       in picked]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("read", "check", "report"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-facts", action="store_true",
                        help="only nouns the store already has facts for")
    parser.add_argument("--again", action="store_true")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--memory", type=Path, default=MEMORY)
    parser.add_argument("--dry", action="store_true",
                        help="check: count what the cache answers, no GPU")
    options = parser.parse_args(argv)
    if options.stage == "read":
        result = read(options.workers, options.limit, options.with_facts,
                      options.again, options.memory)
    elif options.stage == "check":
        result = check(options.limit, options.memory, options.dry)
    else:
        result = report(options.sample, options.memory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

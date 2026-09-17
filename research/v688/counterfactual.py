"""Counterfactual credit: what an operator's answer was worth against what
the others would have said, on the same question.

    python -m research.v688.counterfactual --gold screened --limit 250 \
        --corrupt 300 --workers 4 --out <dir>
    python -m research.v688.counterfactual --report <dir>

The ledger's credit (`executive.Ledger`, E3) is an operator's mean reward
over the questions it did something on. Its reversals compare two of those
means, and the audit after E4b showed what that is worth: a layer declines
every question that is not its kind, so its mean is the base rate of its
kind of question, and `refused by name` (mean 0 over two questions) looked
worse than `the norms` (+0.81 over 990) though no question was ever one both
would answer. Learning utilities from that would reorder specialists by the
questions they happen to get.

So each question is asked again with operators suppressed
(`executive.suppressed`): every operator that did something, one at a time,
and further while someone still answers, up to a budget. Then, per question:

    marginal     the reward with it, less the reward without it -- what
                 firing it was worth on that question. `needed` when
                 nothing could answer without it (bookkeeping, a path's
                 only step).
    competition  without the operator that decided, another decides: the
                 pair, and both rewards. A choice learning should reverse is
                 a pair where the one passed over does better on the same
                 questions -- not on its own.
    oracle       the best any of the explored suppressions reached: the
                 most any ordering of these operators could gain here.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from . import audit

ENGINE = None

#: runs per question at most, the ordinary one included
BUDGET = 12
#: how many operators at most are suppressed together
DEPTH = 3


def _load(gold: str) -> None:
    global ENGINE
    audit.GOLD = gold
    ENGINE = audit.engine_class("shipped")(audit.USING)


def did(payload: dict) -> list[tuple[str, str]]:
    """(executive, operator) for every operator that did something toward
    the payload, in the order fired, subgoals included."""
    out: list = []

    def walk(executive: str, run: dict) -> None:
        for step in run.get("fired", ()):
            if step.get("outcome") != "declined":
                out.append((executive, step["operator"]))
        for inner in run.get("subgoals", ()):
            walk(executive, inner)

    for run in payload.get("executed") or ():
        walk(run.get("executive", ""), run)
    return out


def explore(engine, question: str, budget: int = BUDGET,
            depth: int = DEPTH) -> list[dict]:
    """The question asked as it is, then with what did something
    suppressed, breadth first: one operator at a time, then further only
    from a variant that still came to an answer."""
    from research.v687.executive import suppressed

    nodes: list[dict] = []
    seen: set = set()
    queue: list[frozenset] = [frozenset()]
    while queue and len(nodes) < budget:
        without = queue.pop(0)
        if without in seen:
            continue
        seen.add(without)
        node = {"suppressed": sorted(without)}
        try:
            with suppressed(without):
                payload = engine.ask(question)
        except Exception as bad:                    # noqa: BLE001
            node["error"] = f"{type(bad).__name__}: {bad}"
            nodes.append(node)
            continue
        read = audit.read_answer(payload, payload.get("verdict") or "")
        read.pop("executed", None)
        node.update(read, did=did(payload))
        nodes.append(node)
        if len(without) < depth:
            for pair in dict.fromkeys(node["did"]):
                if pair not in without:
                    queue.append(without | {pair})
    return nodes


def _one(item):
    key, question = item
    started = time.time()
    return {"key": key, "question": question,
            "nodes": explore(ENGINE, question),
            "seconds": round(time.time() - started, 2)}


def collect(limit: int, corrupt: int, gold: str, workers: int,
            out: Path) -> Path:
    audit.GOLD = gold
    asked = audit.all_questions(limit, corrupt, gold)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "counterfactual.jsonl"
    started = time.time()
    with Pool(workers, initializer=_load, initargs=(gold,)) as pool, \
            target.open("w", encoding="utf-8") as sink:
        for index, row in enumerate(pool.imap_unordered(
                _one, sorted(asked.items()), chunksize=4)):
            sink.write(json.dumps(row) + "\n")
            if index % 250 == 0:
                print(f"[counterfactual] {index}/{len(asked)} "
                      f"{time.time() - started:.0f}s", flush=True)
    (out / "settings.json").write_text(json.dumps(
        {"limit": limit, "corrupt": corrupt, "gold": gold,
         "budget": BUDGET, "depth": DEPTH}), encoding="utf-8")
    return target


# -- scoring --------------------------------------------------------------

def reward(side: str, node: dict) -> float | None:
    if "error" in node:
        return None
    return audit.REWARD[side].get(node.get("outcome"))


def score(rows: list[dict], sides: dict) -> dict:
    """Marginals, competitions and the oracle, over the questions whose
    truth the gold says."""
    marginal: dict = defaultdict(lambda: {"count": 0, "total": 0.0,
                                          "helped": 0, "hurt": 0,
                                          "needed": 0})
    competition: dict = defaultdict(lambda: {"count": 0, "chosen": 0.0,
                                             "instead": 0.0})
    base_total = oracle_total = 0.0
    scored = 0
    for row in rows:
        side = sides.get(row["key"])
        nodes = row["nodes"]
        if side is None or not nodes:
            continue
        base = nodes[0]
        mine = reward(side, base)
        if mine is None:
            continue
        scored += 1
        base_total += mine
        oracle_total += max(value for value in
                            (reward(side, node) for node in nodes)
                            if value is not None)
        before = set(map(tuple, base["did"]))
        for node in nodes[1:]:
            if len(node["suppressed"]) != 1:
                continue
            pair = tuple(node["suppressed"][0])
            row_of = marginal[pair]
            row_of["count"] += 1
            theirs = reward(side, node)
            if theirs is None:
                row_of["needed"] += 1
                continue
            row_of["total"] += mine - theirs
            if mine > theirs:
                row_of["helped"] += 1
            elif mine < theirs:
                row_of["hurt"] += 1
            # Who decided instead: what did something without it that did
            # nothing with it.
            instead = [tuple(one) for one in node["did"]
                       if tuple(one) not in before]
            if instead and node.get("outcome") != base.get("outcome"):
                for other in dict.fromkeys(instead):
                    if other[0] != pair[0]:
                        continue
                    entry = competition[(pair, other)]
                    entry["count"] += 1
                    entry["chosen"] += mine
                    entry["instead"] += theirs
    return {
        "questions": scored,
        "base_mean": round(base_total / scored, 4) if scored else None,
        "oracle_mean": round(oracle_total / scored, 4) if scored else None,
        "marginal": sorted((
            {"executive": pair[0], "operator": pair[1],
             "suppressed_on": entry["count"], "needed": entry["needed"],
             "helped": entry["helped"], "hurt": entry["hurt"],
             "mean_marginal": round(entry["total"] / max(
                 1, entry["count"] - entry["needed"]), 4)}
            for pair, entry in marginal.items()),
            key=lambda one: -one["suppressed_on"]),
        "competition": sorted((
            {"executive": chosen[0], "chosen": chosen[1],
             "instead": other[1], "questions": entry["count"],
             "chosen_mean": round(entry["chosen"] / entry["count"], 4),
             "instead_mean": round(entry["instead"] / entry["count"], 4)}
            for (chosen, other), entry in competition.items()),
            key=lambda one: -one["questions"]),
    }


def report(where: Path) -> dict:
    settings = json.loads((where / "settings.json").read_text(
        encoding="utf-8"))
    audit.GOLD = settings["gold"]
    chosen = audit.pairs(settings["limit"], settings["gold"])
    bad = audit.corrupted(settings["corrupt"])
    sides = audit.sides(chosen, bad)
    rows = [json.loads(line) for line in (where / "counterfactual.jsonl")
            .read_text(encoding="utf-8").splitlines() if line]
    found = score(rows, sides)
    found["runs"] = sum(len(row["nodes"]) for row in rows)
    (where / "counterfactual.json").write_text(json.dumps(found, indent=2),
                                               encoding="utf-8")
    return found


def as_text(found: dict) -> str:
    lines = [f"{found['questions']} questions scored, {found['runs']} runs; "
             f"mean reward as shipped {found['base_mean']:+.4f}, the best "
             f"any explored suppression reached {found['oracle_mean']:+.4f}",
             "", "MARGINAL -- reward with the operator less reward without "
             "it, on the same question",
             f"{'executive':8} {'operator':30} {'n':>5} {'needed':>6} "
             f"{'helped':>6} {'hurt':>5} {'mean':>8}"]
    for one in found["marginal"]:
        lines.append(f"{one['executive']:8} {one['operator']:30} "
                     f"{one['suppressed_on']:5} {one['needed']:6} "
                     f"{one['helped']:6} {one['hurt']:5} "
                     f"{one['mean_marginal']:+8.4f}")
    lines += ["", "COMPETITION -- without the one chosen, another decided "
              "differently", f"{'executive':8} {'chosen':30} {'instead':30} "
              f"{'n':>5} {'chosen':>7} {'instead':>7}"]
    for one in found["competition"]:
        flag = "  <- reverse" if one["instead_mean"] > one["chosen_mean"] \
            else ""
        lines.append(f"{one['executive']:8} {one['chosen']:30} "
                     f"{one['instead']:30} {one['questions']:5} "
                     f"{one['chosen_mean']:+7.3f} {one['instead_mean']:+7.3f}"
                     f"{flag}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="screened",
                        choices=("base", "screened"))
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--corrupt", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--report", type=Path,
                        help="score a directory already collected")
    arguments = parser.parse_args()
    where = arguments.report or arguments.out
    if where is None:
        parser.error("--out or --report")
    if not arguments.report:
        collect(arguments.limit, arguments.corrupt, arguments.gold,
                arguments.workers, where)
    print(as_text(report(where)))


if __name__ == "__main__":
    main()

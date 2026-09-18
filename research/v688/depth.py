"""How deep the loop goes, and what stops each source of questions.

    python -m research.v688.depth                 # the probe's questions
    python -m research.v688.depth --dense         # the store's densest kinds
    python -m research.v688.depth --file FILE     # one utterance a line
    python -m research.v688.depth --proving       # scored against gold

The probe asks one-hop questions about concepts the store says little
about, so a loop that stays shallow on it may only be showing that. This
asks the same loop about the kinds the store is densest on too -- the most
facts, crossed with the actions it can say what they need -- so a limit of
the material can be told from a limit of the loop.

For each set: cycles and questions per utterance, how each run ended (out
of questions, or cut off at the cycle limit), and every call of every
question source traced to the line it returned from -- the gate that
stopped it -- and whether it gave anything. A source that never fires is
stopped at one of its early returns, and which one says whether that is
the code's condition or the store having no row.

**The proving ground** (`--proving`) scores what the loop comes to, not
only how far it goes, because a depth count cannot tell a right follow-up
from a wrong one. `can a <concept> <verb>` over the abilities the rated
norms rate, with gold from what we already hold: **positive** where XCSLB's
recovered matrix (`corpora.xcslb_holders`) gives the concept the ability;
**negative** where a calibrated judge screened the COMPS foil as false
(`proving` says why XCSLB's zeros cannot be used). Two answers are scored
side by side: the
engine's first answer to the question as asked, and the loop's, after its
cycles. What separates them is what the loop -- the executive's deepest
search -- adds or costs.
"""
from __future__ import annotations

import argparse
import collections
import linecache
import sys

SOURCES = ("from_gap", "from_doubt", "from_sense", "other_sense_answers",
           "from_requirement", "from_split", "from_content",
           "from_curiosity")

#: actions the store can say what they need (`graph.Requirements`), and
#: correctly: measured over its 300 commonest abilities, 38 derive at all
#: and most of those are wrong (`benefit -> website`, `hear -> tooth`)
ACTIONS = ("run", "fly", "swim", "climb")


def probe(limit: int) -> list[str]:
    from research.v689 import probe as questions
    return list(questions.BASELINE)[:limit]


def dense(reasoner, kinds: int = 10) -> list[str]:
    """`can a dog run`, ... for the animals with the most facts."""
    found: list[str] = []
    rows = reasoner.connection.execute(
        "SELECT concept, COUNT(*) FROM facts GROUP BY concept "
        "ORDER BY COUNT(*) DESC LIMIT 400").fetchall()
    for concept, _ in rows:
        word = concept.rsplit(".", 2)[0]
        if " " in word or not concept.endswith(".n.01"):
            continue
        if not any(node == "animal.n.01"
                   for node, _, _ in reasoner.ascend(concept)):
            continue
        found.append(word)
        if len(found) == kinds:
            break
    article = {True: "an", False: "a"}
    return [f"can {article[word[0] in 'aeiou']} {word} {action}"
            for word in found for action in ACTIONS]


def proving(per_action: int = 2, actions: int = 25) -> list[tuple]:
    """(utterance, gold) pairs: `per_action` positives and negatives for
    each of the `actions` abilities with the most screened foils.

    Positives are what XCSLB's recovered matrix gives the concept -- listed,
    so sound. Negatives are **not** XCSLB's zeros: those are silences, and a
    category-exclusive reading of them said an alligator cannot hurt and a
    budgie cannot climb. They are COMPS foils a calibrated judge screened as
    false (`comps_screened.jsonl`), the nearest misses first -- taxonomic,
    then overlap -- because those are where a loop has to earn its keep.
    """
    import json

    from research.v687 import corpora
    holders = corpora.xcslb_holders()
    nearness = {"taxonomic": 0, "overlap": 1, "co-occurrence": 2, "random": 3}
    foils: dict[str, list[dict]] = {}
    with (corpora.XCSLB_DIR / "comps_screened.jsonl").open(
            encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prop = row["property"]
            if prop.startswith("can ") and len(prop.split()) == 2:
                foils.setdefault(prop, []).append(row)
    chosen = sorted(foils, key=lambda prop: (-len(foils[prop]), prop))
    out: list[tuple] = []
    for prop in chosen[:actions]:
        verb = prop.split()[1]
        rows = sorted(foils[prop], key=lambda row: (
            nearness.get(row["negative_sample_type"], 9), row["id"]))
        no, seen = [], set()
        for row in rows:
            if row["unacceptable_concept"] not in seen:
                seen.add(row["unacceptable_concept"])
                no.append(row["prefix_unacceptable"])
            if len(no) == per_action:
                break
        yes = [f"{'an' if one[0] in 'aeiou' else 'a'} {one}"
               for one in sorted(holders.get(prop, ()))[:per_action]]
        out += [(f"can {one} {verb}", "positive") for one in yes]
        out += [(f"can {one} {verb}", "negative") for one in no]
    return out


RIGHT = {"positive": "verified", "negative": "denied"}
WRONG = {"positive": "denied", "negative": "verified"}


def scored(loop, asked: list[tuple]) -> dict:
    """Seed answer and loop answer against the gold, and what moved."""
    count = {"seed": {"right": 0, "wrong": 0, "unknown": 0},
             "loop": {"right": 0, "wrong": 0, "unknown": 0}}
    moved: list = []
    cycles, questions = [], []
    for utterance, gold in asked:
        run = loop.run(utterance)
        cycles.append(len(run.cycles))
        questions.append(sum(len(one.answers) for one in run.cycles))
        first = (run.cycles[0].answers[0].as_dict(with_payload=False)
                 .get("outcome") if run.cycles and run.cycles[0].answers
                 else "unknown")
        last = run.summary.get("outcome") or "unknown"
        for label, outcome in (("seed", first), ("loop", last)):
            count[label]["right" if outcome == RIGHT[gold] else
                         "wrong" if outcome == WRONG[gold] else
                         "unknown"] += 1
        if first != last:
            moved.append((utterance, gold, first, last))
    return {"questions": len(asked), "count": count, "moved": moved,
            "cycles": sum(cycles) / max(1, len(cycles)),
            "asked": sum(questions) / max(1, len(questions))}


def proving_text(found: dict) -> str:
    lines = [f"proving ground: {found['questions']} questions, "
             f"{found['cycles']:.2f} cycles and {found['asked']:.1f} "
             f"questions asked each"]
    for label in ("seed", "loop"):
        row = found["count"][label]
        lines.append(f"  {label:5} right {row['right']:4}  wrong "
                     f"{row['wrong']:4}  unknown {row['unknown']:4}")
    lines.append(f"  the loop changed {len(found['moved'])} answers:")
    for utterance, gold, first, last in found["moved"]:
        better = ((last == RIGHT[gold]) - (first == RIGHT[gold])
                  - (last == WRONG[gold]) + (first == WRONG[gold]))
        lines.append(f"    {'+' if better > 0 else '-' if better < 0 else '='}"
                     f" {utterance} [{gold}]: {first} -> {last}")
    return "\n".join(lines)


class Gates:
    """Every return of every question source, by line."""

    def __init__(self) -> None:
        self.returns: collections.Counter = collections.Counter()
        self.calls: collections.Counter = collections.Counter()

    def tracer(self, frame, event, arg):
        name = frame.f_code.co_name
        if name not in SOURCES or "question.py" not in frame.f_code.co_filename:
            return None

        def local(frame, event, arg):
            if event == "return":
                self.calls[name] += 1
                self.returns[(name, frame.f_lineno, bool(arg))] += 1
            return local
        return local


def measure(loop, utterances: list[str]) -> dict:
    gates = Gates()
    cycles, asked, ended = [], [], collections.Counter()
    for utterance in utterances:
        sys.settrace(gates.tracer)
        try:
            run = loop.run(utterance)
        finally:
            sys.settrace(None)
        cycles.append(len(run.cycles))
        asked.append(sum(len(one.answers) for one in run.cycles))
        steps = [step for step in run.executed[0]["fired"]
                 if step["operator"] == "what to ask next"]
        ended["out of questions" if steps and steps[-1]["outcome"]
              == "declined" else "cut off"] += 1
    return {"utterances": len(utterances),
            "cycles": sum(cycles) / max(1, len(cycles)),
            "asked": sum(asked) / max(1, len(asked)),
            "ended": dict(ended), "gates": gates}


def as_text(label: str, found: dict) -> str:
    source = sys.modules["research.v688.question"].__file__
    lines = [f"{label}: {found['utterances']} utterances, "
             f"{found['cycles']:.2f} cycles and {found['asked']:.1f} "
             f"questions each; ended {found['ended']}"]
    gates = found["gates"]
    for name in SOURCES:
        rows = sorted(((line, gave, count) for (who, line, gave), count
                       in gates.returns.items() if who == name),
                      key=lambda one: -one[2])
        gave = sum(count for _, given, count in rows if given)
        lines.append(f"  {name}: {gates.calls[name]} calls, gave {gave}")
        for line, given, count in rows:
            text = linecache.getline(source, line).strip()
            lines.append(f"    {count:5}  line {line:5} "
                         f"{'GAVE' if given else '  - '}  {text[:60]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--proving", action="store_true")
    parser.add_argument("--file", default="")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=8)
    options = parser.parse_args()

    from research.v688 import audit
    from research.v688.attention import Curiosity
    from research.v688.loop import Loop
    from research.v688.pool import EnginePool

    pool = EnginePool(audit.USING, workers=options.workers)
    loop = Loop(pool, Curiosity(pool.engines[0].profiles.plan),
                max_cycles=options.cycles)
    if options.proving:
        print(proving_text(scored(loop, proving())))
        return
    if options.file:
        label = options.file
        utterances = [line.strip() for line in open(options.file,
                                                    encoding="utf-8")
                      if line.strip()][:options.limit]
    elif options.dense:
        label = "densest kinds"
        utterances = dense(pool.engines[0].reasoner)[:options.limit]
    else:
        label = "probe"
        utterances = probe(options.limit)
    print(as_text(label, measure(loop, utterances)))


if __name__ == "__main__":
    main()

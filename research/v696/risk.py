"""Risk first: six qualities of a request, and the work each combination
needs (`PLAN.md`, "Risk first"; Adrian's sprint deck).

Every request is scored 0-3 on six factors -- **high is 2 or more** -- by
six estimators that read only the request (English, signature, examples),
before anything is written for it:

    D  depth          layers of operations between the request and the
                      answer: a verified program's tree depth (`parse.py`)
    P  pairs          named values a program keeps consistent, N, counted as
                      C(N, 2) pairs
    S  semantics      rules to state: its decision points
    U  uncertainty    how open the request is: distinct behaviours, beyond
                      the examples, among programs written for it that
                      meet them
    X  external       what the search's library lacks: operations made on
                      the spot, text it cannot build, nothing to verify by
    B  blast radius   what a wrong program costs whoever runs it: throws,
                      mutates its arguments, may never end

The labels are read off verified programs (`label`): the compiler's syntax
(`tscheck.js` `qualities`), the exact reader's tree, and -- for U -- the
untaught base model's programs for the request, which it was taught none
of. The estimators (`train`) are heads over the reader of meaning's encoder,
read only, one per factor, each trained on the records that have its label
and chosen on MBPP dev.

The resolution matrix (`MATRIX`) names a move per high factor and per pair
of high factors; `moves` turns a request's scores into how it is searched
(`search.py`, `sketcher.proposals`). A request in no shaded corner is
searched with half the budget, and that budget goes to those in many.

    python -m research.v696.risk label        data/code-meaning/risk.jsonl
    python -m research.v696.risk train        llm/risk-estimators
    python -m research.v696.risk evaluate [--held]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from research.v696 import program as P

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "code-meaning"
LLM = ROOT / "llm"
LABELS = DATA / "risk.jsonl"
MODEL = LLM / "risk-estimators"
#: the reader of meaning whose encoder the estimators read with
READER = "meaning-unixcoder"
#: who writes the programs U is read from: taught none of these requests
BASE = "SmolLM2-360M-Instruct"
READINGS = 16
SEED = 696

FACTORS = ("D", "P", "S", "U", "X", "B")
NAMES = {"D": "depth", "P": "pairs", "S": "semantics", "U": "uncertainty",
         "X": "external", "B": "blast radius"}
#: a factor is high from here: fixed, as the deck fixes it
HIGH = 2

SIGNATURE = re.compile(r"function\s+(\w+)\s*\((.*)\)\s*:\s*(.+)$", re.S)
#: what the exact reader builds itself: the search's own language, not
#: something a library has to have
STRUCTURAL = frozenset({"index", "form", "force", "tuple", "let", "while",
                        "setitem", "effect", "helper", "range", "ternary",
                        "append"})


# -- the labels: read off verified programs ----------------------------------

def _bin(value: int, tops) -> int:
    """0-3: the first score whose top `value` does not pass."""
    for score, top in enumerate(tops):
        if value <= top:
            return score
    return 3


def external(tree, examples) -> int:
    """X: nothing outside the library (0), operations made on the spot from
    the compiler's types (1), text the search cannot build (2), nothing to
    verify a program by (3)."""
    if not examples:
        return 3
    if tree is None:
        return 2
    library = {op.key for op in P.library().ops}
    score = 0
    for op in tree.ops():
        if op.kind == "opaque":
            return 2
        if op.kind not in STRUCTURAL and op.key not in library:
            score = 1
    return score


def static(body: str, entry: str, params, examples) -> dict | None:
    """D, P, S, X, B of a verified program, or None if its function is not
    in it."""
    from research.v696.checker import checker
    from research.v696.parse import parse
    said = checker().qualities(body, entry)
    if not said["found"]:
        return None
    tree = parse(body, entry, params)
    names = said["names"]
    return {
        "D": 3 if tree is None else _bin(tree.depth - 1, (1, 2, 4)),
        "P": _bin(names * (names - 1) // 2, (1, 6, 15)),
        "S": _bin(said["decisions"], (0, 1, 3)),
        "X": external(tree, examples),
        "B": (3 if said["unbounded"] else 2 if said["mutates"]
              else 1 if said["partial"] else 0),
    }


def _signed(signature: str) -> tuple:
    found = SIGNATURE.match(signature.strip())
    from research.v696.tasks import _params
    return found.group(1), _params(found.group(2)), found.group(3).strip()


def records() -> list:
    """Every request with a verified program: the reader's corpus (MBPP,
    the docs, generated programs described; HumanEval held) and the
    teacher's requests (`teach_requests.py`), all train."""
    from research.v696.reader import CORPUS
    out = []
    for line in CORPUS.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("body") and row["examples"]:
            out.append({key: row[key] for key in
                        ("name", "source", "split", "english", "signature",
                         "examples", "body")})
    path = DATA / "requests.jsonl"
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if not row.get("code"):
                continue
            said = ", ".join(f"{name}: {kind}" for name, kind in
                             row["params"])
            from research.v696.teach_meaning import _english
            out.append({"name": row["name"], "source": "requests",
                        "split": "train",
                        "english": _english(row["prompt"]),
                        "signature": f"function {row['entry']}({said}): "
                                     f"{row['returns']}",
                        "examples": row["examples"], "body": row["code"]})
    return out


def spec_of(row: dict):
    from research.v696.spec import Spec
    entry, params, returns = _signed(row["signature"])
    return Spec(row["name"], params, returns,
                [(list(args), out) for args, out in row["examples"]],
                entry=entry, english=row["english"])


#: U is read where a request is a person's: not the docs' one-liners, not
#: the generated compositions
READ_U = ("mbpp-ts", "humaneval-ts", "requests")


def readings(rows: list, batch: int = 4) -> None:
    """U for each row: how many distinct behaviours -- on inputs varied
    from the examples (`meaning.probes`) -- the programs written for the
    request that meet its examples have, the verified one among them."""
    from research.v696 import meaning as M
    from research.v696 import sketcher as K
    from research.v696.checker import CheckerError, checker
    from research.v696.parse import parse
    from research.v696.reader import request
    from research.v696.teach_sketch import prompt
    writer = K.Sketcher(LLM / BASE)
    writer.torch.manual_seed(SEED)
    started = time.time()
    for at in range(0, len(rows), batch):
        chunk = rows[at:at + batch]
        specs = [spec_of(row) for row in chunk]
        texts = [prompt(*request(spec), None) for spec in specs]
        written = writer.write(texts, samples=READINGS)
        for row, spec, said in zip(chunk, specs, written):
            spec.proposals, spec.sources = [], []
            K._read(spec, said, parse)
            programs = [tree for tree in spec.proposals
                        if K._meets(spec, tree)]
            mine = parse(row["body"], spec.entry, spec.params)
            if mine is not None:
                programs.append(mine)
            cases = M.probes(spec.examples)
            distinct = set()
            if programs and cases:
                try:
                    values = checker().values(
                        spec.names, cases, [one.source() for one in programs],
                        prelude=P.prelude(programs))
                except CheckerError:
                    values = []
                for one in values:
                    distinct.add(json.dumps([cell.get("value", "!")
                                             for cell in one],
                                            sort_keys=True))
            row["labels"]["U"] = _bin(max(len(distinct), 1), (1, 2, 4))
            row["readings"] = {"met": len(programs) - (mine is not None),
                               "behaviours": len(distinct)}
            _append(row)
        print(f"  {min(at + batch, len(rows))}/{len(rows)} read "
              f"({time.time() - started:.0f}s)", flush=True)
    del writer


def _append(row: dict) -> None:
    with LABELS.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


def label() -> dict:
    """Every record's labels into `LABELS`, as each is read: a run stopped
    part-way goes on from where it was."""
    done = set()
    if LABELS.exists():
        done = {json.loads(line)["name"] for line in
                LABELS.open(encoding="utf-8")}
    pending_u = []
    counts = Counter()
    for row in records():
        if row["name"] in done:
            continue
        entry, params, _ = _signed(row["signature"])
        labels = static(row["body"], entry, params, row["examples"])
        if labels is None:
            counts["not found"] += 1
            continue
        row["labels"] = labels
        if row["source"] in READ_U:
            pending_u.append(row)
        else:
            _append(row)
            counts["static"] += 1
    print(f"static labels: {dict(counts)}; U to read: {len(pending_u)}",
          flush=True)
    if pending_u:
        readings(pending_u)
    return dict(counts)


def labelled() -> list:
    rows, seen = [], set()
    for line in LABELS.open(encoding="utf-8"):
        row = json.loads(line)
        if row["name"] not in seen:
            seen.add(row["name"])
            rows.append(row)
    return rows


# -- the six estimators --------------------------------------------------------

def _surface(english: str, params, returns: str, examples) -> list:
    """What is plain to see in a request, beside the encoder's reading."""
    kinds = [kind for _, kind in params]
    return [len(params), len(examples), len((english or "").split()) / 20.0,
            float(any(kind.endswith("]") for kind in kinds)),
            float(returns == "number"), float(returns == "string"),
            float(returns == "boolean"), float(returns.endswith("]")),
            float(returns not in ("number", "string", "boolean")
                  and not returns.endswith("]"))]


class Features:
    """A request as the estimators see it: the reader of meaning's encoder
    (pooled), its heads' probabilities, and what is plain on the surface."""

    def __init__(self, model: Path = LLM / READER) -> None:
        from research.v696.reader import Reader
        self.reader = Reader.load(model)
        self.torch = self.reader.torch

    def __call__(self, requests: list, plain: list, batch: int = 32):
        """{"pooled", "probs", "surface"}: a tensor each, one row a
        request."""
        from research.v696.reader import LONGEST
        torch, reader = self.torch, self.reader
        pooled, probs = [], []
        with torch.no_grad():
            for at in range(0, len(requests), batch):
                pairs = requests[at:at + batch]
                said = reader.tokenizer(
                    [one for one, _ in pairs], [two for _, two in pairs],
                    truncation="longest_first", max_length=LONGEST,
                    padding=True, return_tensors="pt").to(reader.device)
                said.pop("token_type_ids", None)
                hidden = reader.encoder(**said).last_hidden_state
                mask = said["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                one = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled.append(one.float().cpu())
                heads = []
                for head, layer in reader.heads.items():
                    logits = layer(one)
                    heads.append(torch.softmax(logits, -1)
                                 if head in ("returns", "root")
                                 else torch.sigmoid(logits))
                probs.append(torch.cat(heads, -1).float().cpu())
        return {"pooled": torch.cat(pooled), "probs": torch.cat(probs),
                "surface": torch.tensor(plain, dtype=torch.float32)}


#: which of the features each estimator may be given: chosen on dev
VIEWS = {"pooled": ("pooled", "surface"), "probs": ("probs", "surface"),
         "all": ("pooled", "probs", "surface")}


def _matrix(features: dict, view: str):
    return features["torch"].cat([features[one] for one in VIEWS[view]], -1)


def _head(torch, width: int, hidden: int):
    if not hidden:
        return torch.nn.Linear(width, 4)
    return torch.nn.Sequential(torch.nn.Linear(width, hidden),
                               torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(hidden, 4))


def _fit(torch, x, y, weights, hidden: int, decay: float,
         epochs: int = 300):
    torch.manual_seed(SEED)
    head = _head(torch, x.shape[1], hidden)
    optimiser = torch.optim.AdamW(head.parameters(), lr=3e-3,
                                  weight_decay=decay)
    counts = torch.bincount(y, minlength=4).float().clamp(min=1)
    balance = (counts.sum() / (4 * counts))
    loss = torch.nn.CrossEntropyLoss(weight=balance, reduction="none")
    for _ in range(epochs):
        head.train()
        optimiser.zero_grad()
        value = (loss(head(x), y) * weights).sum() / weights.sum()
        value.backward()
        optimiser.step()
    head.eval()
    return head


def scores(said: list, truth: list) -> dict:
    """Exact, within one, and high-vs-not (balanced: the mean of how many
    high and how many not-high are said so)."""
    n = len(truth)
    if not n:
        return {"n": 0}
    exact = sum(a == b for a, b in zip(said, truth)) / n
    near = sum(abs(a - b) <= 1 for a, b in zip(said, truth)) / n
    high = [b >= HIGH for b in truth]
    hit = [(a >= HIGH) == (b >= HIGH) for a, b in zip(said, truth)]
    sides = [sum(h for h, t in zip(hit, high) if t == side)
             / max(1, sum(t == side for t in high)) for side in (True, False)
             if any(t == side for t in high)]
    return {"n": n, "exact": round(exact, 3), "within one": round(near, 3),
            "high balanced": round(sum(sides) / len(sides), 3),
            "high share": round(sum(high) / n, 3)}


def baseline(train: list, truth: list) -> dict:
    """The most common training class, said of everything."""
    common = Counter(train).most_common(1)[0][0]
    return scores([common] * len(truth), truth)


class Estimators:
    """Six heads, one per factor, over `Features`."""

    def __init__(self, path: Path = MODEL) -> None:
        import torch
        self.torch = torch
        saved = torch.load(str(path / "heads.pt"), map_location="cpu")
        self.settings = json.loads((path / "risk.json").read_text(
            encoding="utf-8"))
        self.norm = saved["norm"]
        self.heads = {}
        for factor in FACTORS:
            chosen = self.settings["chosen"][factor]
            head = _head(torch, saved["width"][chosen["view"]],
                         chosen["hidden"])
            head.load_state_dict(saved["heads"][factor])
            head.eval()
            self.heads[factor] = head
        self.features = Features(LLM / self.settings["reader"])

    def __call__(self, requests: list, plain: list) -> list:
        torch = self.torch
        found = self.features(requests, plain)
        found = {key: (value - self.norm[key][0]) / self.norm[key][1]
                 for key, value in found.items()}
        found["torch"] = torch
        out = [{} for _ in requests]
        with torch.no_grad():
            for factor, head in self.heads.items():
                view = self.settings["chosen"][factor]["view"]
                said = head(_matrix(found, view)).argmax(-1).tolist()
                for row, score in zip(out, said):
                    row[factor] = int(score)
        return out


def _requests(rows: list) -> tuple:
    from research.v696.reader import said
    requests, plain = [], []
    for row in rows:
        _, params, returns = _signed(row["signature"])
        requests.append(said(row, ("english", "signature", "examples")))
        plain.append(_surface(row["english"], params, returns,
                              row["examples"]))
    return requests, plain


def train(out: Path = MODEL) -> dict:
    """Each factor's head, chosen on MBPP dev among what it is given (the
    encoder pooled, the reader's probabilities, both), its width, its
    decay, and whether the generated compositions are taught -- into a new
    `llm/` directory."""
    import torch
    rows = labelled()
    features = Features()
    requests, plain = _requests(rows)
    found = features(requests, plain)
    trained = [at for at, row in enumerate(rows) if row["split"] == "train"]
    norm = {}
    for key, value in found.items():
        part = value[trained]
        norm[key] = (part.mean(0), part.std(0).clamp(min=1e-6))
        found[key] = (value - norm[key][0]) / norm[key][1]
    found["torch"] = torch
    chosen, heads, report = {}, {}, {}
    for factor in FACTORS:
        best = None
        for view, hidden, decay, generated in itertools.product(
                VIEWS, (0, 64), (1e-4, 1e-2), (True, False)):
            train_at = [at for at in trained if factor in rows[at]["labels"]
                        and (generated or rows[at]["source"] != "generated")]
            dev_at = [at for at, row in enumerate(rows)
                      if row["split"] == "dev" and factor in row["labels"]]
            x = _matrix(found, view)
            y = torch.tensor([rows[at]["labels"][factor]
                              for at in train_at])
            # the generated compositions weigh a quarter of a person's
            weights = torch.tensor([0.25 if rows[at]["source"] == "generated"
                                    else 1.0 for at in train_at])
            head = _fit(torch, x[train_at], y, weights, hidden, decay)
            with torch.no_grad():
                said = head(x[dev_at]).argmax(-1).tolist()
            truth = [rows[at]["labels"][factor] for at in dev_at]
            got = scores(said, truth)
            rank = (got["high balanced"], got["exact"])
            if best is None or rank > best[0]:
                best = (rank, {"view": view, "hidden": hidden,
                               "decay": decay, "generated": generated},
                        head, got, baseline(y.tolist(), truth))
        _, chosen[factor], head, got, base = best
        heads[factor] = head.state_dict()
        report[factor] = {"chosen": chosen[factor], "dev": got,
                          "dev baseline": base}
        print(f"{factor} {NAMES[factor]:13} {chosen[factor]}  dev {got}  "
              f"baseline {base}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    width = {view: _matrix(found, view).shape[1] for view in VIEWS}
    torch.save({"heads": heads, "norm": norm, "width": width},
               str(out / "heads.pt"))
    (out / "risk.json").write_text(json.dumps(
        {"reader": READER, "chosen": chosen, "high": HIGH,
         "report": report, "labels": LABELS.name,
         "records": len(rows)}, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return report


def evaluate(held: bool = False, path: Path = MODEL) -> dict:
    """Each estimator against its labels -- MBPP dev, or HumanEval held
    (once) -- and against the most common training class."""
    rows = labelled()
    part = "held" if held else "dev"
    measured = [row for row in rows if row["split"] == part]
    trained = [row for row in rows if row["split"] == "train"]
    estimators = Estimators(path)
    said = estimators(*_requests(measured))
    out = {}
    for factor in FACTORS:
        pairs = [(one[factor], row["labels"][factor])
                 for one, row in zip(said, measured)
                 if factor in row["labels"]]
        truth = [b for _, b in pairs]
        out[factor] = {"estimator": scores([a for a, _ in pairs], truth),
                       "baseline": baseline(
                           [row["labels"][factor] for row in trained
                            if factor in row["labels"]], truth)}
        print(f"{part} {factor} {NAMES[factor]:13} {out[factor]}",
              flush=True)
    return out


def assess(specs: list, path: Path = MODEL) -> None:
    """Each spec's six scores (`Spec.risk`), read off its request."""
    from research.v696.reader import request
    estimators = Estimators(path)
    requests = [request(spec) for spec in specs]
    plain = [_surface(spec.english, spec.params, spec.returns,
                      spec.examples) for spec in specs]
    for spec, risk in zip(specs, estimators(requests, plain)):
        spec.risk = risk
    del estimators


def oracle(specs: list) -> int:
    """Each spec's scores from its verified program's labels, where there
    is one (`LABELS`): the matrix measured apart from the estimators'
    errors. How many had labels."""
    by = {row["name"]: row["labels"] for row in labelled()}
    found = 0
    for spec in specs:
        labels = by.get(spec.name)
        if labels is not None and all(f in labels for f in FACTORS):
            spec.risk = dict(labels)
            found += 1
    return found


# -- the resolution matrix -----------------------------------------------------

#: One move per cell: a factor alone (the diagonal) or two high together.
#: What each move does is in `search.py` and `sketcher.proposals`:
#:
#:   deeper     one more level grown; induction's allowance doubled
#:   ask        the writers asked once more where nothing met the examples
#:   pairs      probes with two arguments varied at once
#:   edges      probes at the arguments' edges (empty, zero, one)
#:   cross      the pairs at their edges: a decision table
#:   total      the judge prefers a program that throws on no probe
#:   readings   the writers asked once more even where something met, and
#:              every writer asked, not only where those before it failed
#:   agree      the judge takes the behaviour most programs written agree on
#:   four_eyes  no stopping at the first program that meets the examples: a
#:              second, independent one must agree beyond them
#:   golden     the reader of meaning's sure behaviours enforced at a lower
#:              bar (`search.GOLDEN`)
#:   park       answered, but marked unconfirmed unless independent
#:              programs agree
#:   strict     a program that throws on any probe is refused
DIAGONAL = {
    "D": {"deeper", "ask"},           # trace every layer
    "P": {"pairs"},                   # model the pairs
    "S": {"edges", "total"},          # examples first
    "U": {"readings", "agree"},       # clarify or spike
    "X": {"ask"},                     # stub the boundary
    "B": {"four_eyes", "park"},       # make it reversible
}
#: Beyond the union of their diagonals, what a pair of high factors adds.
PAIRED = {
    ("P", "S"): {"cross"},            # decision table
    ("S", "X"): {"golden"},           # golden files
    ("U", "X"): {"park"},             # park it
    ("U", "B"): {"park"},             # decide on record
    ("X", "B"): {"strict"},           # fail loudly
}
MOVES = frozenset(set().union(*DIAGONAL.values(), *PAIRED.values(),
                              {"budget"}))
#: A request in no shaded corner: ordinary review, half the budget. Each
#: corner it is in adds a quarter, to at most this.
RELIEF, CORNER, MOST = 0.5, 0.25, 3.0


@dataclass(frozen=True)
class Moves:
    on: frozenset = frozenset()
    cells: tuple = ()
    #: the budget as a share of the usual
    budget: float = 1.0

    def __contains__(self, name: str) -> bool:
        return name in self.on


def cells(risk: dict) -> tuple:
    """The shaded corners a request is in: each high factor, each pair."""
    high = [factor for factor in FACTORS if risk.get(factor, 0) >= HIGH]
    return tuple(high) + tuple(f"{a}×{b}" for a, b in
                               itertools.combinations(high, 2))


def moves(risk: dict | None, allowed=MOVES) -> Moves:
    """What the matrix says this request's search does, of the moves
    `allowed` (all of them, but for measuring one without the rest)."""
    if not risk:
        return Moves()
    corners = cells(risk)
    on = set()
    for cell in corners:
        pair = tuple(cell.split("×"))
        for factor in pair:
            on |= DIAGONAL[factor]
        on |= PAIRED.get(pair, set())
    on &= set(allowed)
    budget = 1.0
    if "budget" in allowed:
        budget = RELIEF if not corners else min(
            MOST, 1.0 + CORNER * len(corners))
    return Moves(frozenset(on), corners, budget)


def ranked(specs: list, results: list) -> dict:
    """Risk ranks failure: of the requests with n high factors, how many
    passed. {n: (passed, of)}."""
    out: dict = {}
    for spec, passed in zip(specs, results):
        if spec.risk is None:
            continue
        n = sum(spec.risk[f] >= HIGH for f in FACTORS)
        got = out.setdefault(n, [0, 0])
        got[0] += bool(passed)
        got[1] += 1
    return {n: tuple(out[n]) for n in sorted(out)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("label", "train", "evaluate"))
    parser.add_argument("--held", action="store_true")
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)
    path = LLM / args.model if args.model else MODEL
    if args.job == "label":
        label()
    elif args.job == "train":
        train(path)
    else:
        evaluate(args.held, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The reader of meaning: English and code, one sequence, one structure.

One encoder reads whatever a request holds -- the English, the signature,
examples, a body -- as **one** sequence (the English, then the code), and
its heads say the `Meaning` (`meaning.py`) that sequence has: what the
function returns, its behaviour, what it is made of, what gives its
result. There is no English reader and no code reader: either part may be
missing, and the same heads read what is there.

Taught (`teach_meaning.py`) with every record shown in several views of its
one sequence, all with the same target -- that is what makes the reading
joint, and `evaluate` measures it: each view on held records.

    python -m research.v696.reader train --base unixcoder-base
    python -m research.v696.reader evaluate --model meaning-unixcoder

Models go into their own new `llm/` directories and are never touched after.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from research.v696.meaning import Meaning

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
CORPUS = ROOT / "data" / "code-meaning" / "corpus.jsonl"

#: Where a base model is, by the name it is given here.
BASES = {"MiniLM-L6-v2": LLM / "MiniLM-L6-v2",
         "unixcoder-base": LLM / "unixcoder-base",
         "ModernBERT-base": "answerdotai/ModernBERT-base"}

#: A label is learned when it is seen this often in training.
SEEN = 3
LONGEST = 384

#: The views a record can be shown in: which of its parts are in the
#: sequence. `english` alone and `code` alone are views like any other.
VIEWS = {
    "english+signature+examples": ("english", "signature", "examples"),
    "english+signature": ("english", "signature"),
    "english": ("english",),
    "signature+examples": ("signature", "examples"),
    "body": ("signature", "body"),
    "english+body": ("english", "body"),
}


def said(record: dict, parts) -> tuple:
    """The sequence a view shows: (English, code). Either may be empty."""
    english = record["english"] if "english" in parts else ""
    code = []
    if "body" in parts and record.get("body"):
        code.append(record["body"].strip())
    elif "signature" in parts:
        code.append(record["signature"])
    if "examples" in parts:
        for args, out in record["examples"]:
            code.append(f"// f({json.dumps(args)[1:-1]}) === "
                        f"{json.dumps(out)}")
    return english or "", "\n".join(code)


def views(record: dict) -> list:
    """The views this record has everything for."""
    out = []
    for name, parts in VIEWS.items():
        if "english" in parts and not record["english"]:
            continue
        if "body" in parts and not record.get("body"):
            continue
        if "examples" in parts and not record["examples"]:
            continue
        out.append(name)
    return out


def load(path: Path = CORPUS) -> list:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def vocabulary(records: list) -> dict:
    """The labels each head says, from the training records."""
    count = {"returns": Counter(), "behaviour": Counter(),
             "uses": Counter(), "root": Counter()}
    for row in records:
        meaning = row["meaning"]
        count["returns"][meaning["returns"]] += 1
        for one in meaning["behaviour"] or ():
            count["behaviour"][one] += 1
        for one in meaning["uses"] or ():
            count["uses"][one] += 1
        if meaning["root"]:
            count["root"][meaning["root"]] += 1
    out = {}
    for head, seen in count.items():
        labels = sorted(one for one, n in seen.items() if n >= SEEN)
        if head in ("returns", "root"):
            labels.append("other")
        out[head] = labels
    return out


class Reader:
    """The encoder and its heads."""

    def __init__(self, base, labels: dict, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(str(base))
        self.encoder = AutoModel.from_pretrained(str(base)).to(device)
        width = self.encoder.config.hidden_size
        self.heads = torch.nn.ModuleDict({
            head: torch.nn.Linear(width, len(names))
            for head, names in labels.items()}).to(device)
        self.device = device

    def parameters(self):
        return list(self.encoder.parameters()) + list(
            self.heads.parameters())

    def forward(self, pairs: list) -> dict:
        torch = self.torch
        english = [one for one, _ in pairs]
        code = [two for _, two in pairs]
        batch = self.tokenizer(english, code, truncation="longest_first",
                               max_length=LONGEST, padding=True,
                               return_tensors="pt").to(self.device)
        batch.pop("token_type_ids", None)
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return {head: layer(pooled) for head, layer in self.heads.items()}

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=False)
        self.encoder.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        self.torch.save(self.heads.state_dict(), str(out / "heads.bin"))
        (out / "labels.json").write_text(json.dumps(self.labels, indent=1),
                                         encoding="utf-8")

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "Reader":
        import torch
        labels = json.loads((path / "labels.json").read_text(
            encoding="utf-8"))
        reader = cls(path, labels, device)
        reader.heads.load_state_dict(torch.load(str(path / "heads.bin"),
                                                map_location=device))
        reader.encoder.eval()
        return reader

    def read(self, pairs: list, batch: int = 32) -> list:
        """For each (English, code), the probabilities of every label."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for at in range(0, len(pairs), batch):
                logits = self.forward(pairs[at:at + batch])
                probs = {
                    head: (torch.softmax(value, -1)
                           if head in ("returns", "root")
                           else torch.sigmoid(value)).float().cpu()
                    for head, value in logits.items()}
                for row in range(len(pairs[at:at + batch])):
                    out.append({head: dict(zip(self.labels[head],
                                               probs[head][row].tolist()))
                                for head in probs})
        return out


def reading(probs: dict, takes=(), floor: float = 0.5) -> Meaning:
    """A reader's probabilities as a Meaning."""
    def best(head):
        return max(probs[head], key=probs[head].get)
    return Meaning(tuple(takes), best("returns"),
                   frozenset(one for one, p in probs["behaviour"].items()
                             if p >= floor),
                   frozenset(one for one, p in probs["uses"].items()
                             if p >= floor), best("root"))


def _targets(reader: Reader, records: list):
    torch = reader.torch
    labels = reader.labels
    index = {head: {one: at for at, one in enumerate(names)}
             for head, names in labels.items()}
    out = {"returns": [], "root": [], "behaviour": [], "uses": [],
           "has_behaviour": [], "has_uses": [], "has_root": []}
    for row in records:
        meaning = row["meaning"]
        out["returns"].append(index["returns"].get(
            meaning["returns"], index["returns"]["other"]))
        out["has_root"].append(meaning["root"] is not None)
        out["root"].append(index["root"].get(meaning["root"],
                                             index["root"]["other"]))
        for head in ("behaviour", "uses"):
            row_ = torch.zeros(len(labels[head]))
            for one in meaning[head] or ():
                if one in index[head]:
                    row_[index[head][one]] = 1
            out[head].append(row_)
            out[f"has_{head}"].append(meaning[head] is not None)
    device = reader.device
    return {key: (torch.stack(value) if key in ("behaviour", "uses")
                  else torch.tensor(value)).to(device)
            for key, value in out.items()}


def loss(reader: Reader, logits: dict, target: dict):
    torch = reader.torch
    F = torch.nn.functional
    total = F.cross_entropy(logits["returns"], target["returns"])
    if target["has_root"].any():
        total = total + F.cross_entropy(logits["root"][target["has_root"]],
                                        target["root"][target["has_root"]])
    for head in ("behaviour", "uses"):
        mask = target[f"has_{head}"]
        if mask.any():
            total = total + 4 * F.binary_cross_entropy_with_logits(
                logits[head][mask], target[head][mask])
    return total


def train(base: str, out: Path, epochs: int = 12, batch: int = 16,
          rate: float = 3e-5, seed: int = 696) -> None:
    import torch
    rng = random.Random(seed)
    torch.manual_seed(seed)
    records = [row for row in load() if row["split"] == "train"]
    labels = vocabulary(records)
    print({head: len(names) for head, names in labels.items()},
          f"{len(records)} training records", flush=True)
    reader = Reader(BASES[base], labels)
    optimiser = torch.optim.AdamW(reader.parameters(), lr=rate,
                                  weight_decay=0.01)
    steps = epochs * ((len(records) + batch - 1) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=rate, total_steps=steps, pct_start=0.1)
    started = time.time()
    for epoch in range(epochs):
        reader.encoder.train()
        rng.shuffle(records)
        total = 0.0
        for at in range(0, len(records), batch):
            chunk = records[at:at + batch]
            # each record in one of its views, chosen afresh every time
            pairs = [said(row, VIEWS[rng.choice(views(row))])
                     for row in chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = reader.forward(pairs)
            value = loss(reader, {k: v.float() for k, v in logits.items()},
                         _targets(reader, chunk))
            optimiser.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += value.item() * len(chunk)
        print(f"epoch {epoch + 1}: loss {total / len(records):.3f} "
              f"({time.time() - started:.0f}s)", flush=True)
    reader.encoder.eval()
    reader.save(out)
    print(f"-> {out}")


def request(spec) -> tuple:
    """A spec as the reader's sequence: its English, its signature, the
    examples it shows -- whatever of them it has."""
    record = {"english": spec.english, "signature": spec.signature(),
              "examples": [[list(args), out] for args, out in spec.examples]}
    parts = [one for one in ("english", "signature", "examples")
             if record[one]]
    return said(record, parts)


def expect(specs: list, model: Path) -> None:
    """Read every spec's request, and say what is expected of its program
    (`Spec.expected`: what the search grows first, and what a program must
    show beyond the examples to be taken)."""
    reader = Reader.load(model)
    for spec, probs in zip(specs, reader.read([request(one)
                                               for one in specs])):
        spec.expected = {"uses": probs["uses"],
                         "behaviour": probs["behaviour"]}
    del reader


# -- measured ----------------------------------------------------------------

def _f1(said: set, truth: set) -> tuple:
    hit = len(said & truth)
    return hit, len(said), len(truth)


def evaluate(model: Path, splits=("dev", "held")) -> dict:
    """Each view on dev and held records: returns and root right,
    behaviour F1 (micro), uses recall at 10 where a verified body says
    what the program is made of."""
    reader = Reader.load(model)
    records = load()
    out = {}
    for part in splits:
        rows = [row for row in records if row["split"] == part]
        for view in ("english", "signature+examples",
                     "english+signature", "english+signature+examples"):
            chosen = [row for row in rows if view in views(row)]
            if not chosen:
                continue
            probs = reader.read([said(row, VIEWS[view]) for row in chosen])
            hits = [0, 0, 0]
            returns = root = roots = recall = recalled = 0
            for row, prob in zip(chosen, probs):
                truth = row["meaning"]
                meaning = reading(prob)
                returns += meaning.returns == truth["returns"]
                known = set(truth["behaviour"] or ()) & set(
                    reader.labels["behaviour"])
                for at, value in enumerate(_f1(set(meaning.behaviour),
                                               known)):
                    hits[at] += value
                if truth["uses"]:
                    top = sorted(prob["uses"], key=prob["uses"].get,
                                 reverse=True)[:10]
                    wanted = set(truth["uses"]) & set(reader.labels["uses"])
                    recall += len(wanted & set(top))
                    recalled += len(wanted)
                    roots += 1
                    root += meaning.root == truth["root"]
            precision = hits[0] / max(hits[1], 1)
            recall_b = hits[0] / max(hits[2], 1)
            f1 = 2 * precision * recall_b / max(precision + recall_b, 1e-9)
            row = {"records": len(chosen),
                   "returns": round(returns / len(chosen), 3),
                   "behaviour F1": round(f1, 3),
                   "uses R@10": round(recall / recalled, 3)
                   if recalled else None,
                   "root": round(root / roots, 3) if roots else None}
            out[f"{part} {view}"] = row
            print(f"{part:5} {view:28} {row}", flush=True)
    return out


def refusals(model: Path, strong: float = 0.95) -> dict:
    """How often the round trip would refuse a program known to be right:
    each verified body on dev and held, run on its examples and on inputs
    varied from them, read back into behaviour, against what the reader
    is sure of from the request."""
    import re
    from research.v696 import meaning as M
    from research.v696.checker import checker
    reader = Reader.load(model)
    out = {}
    for part in ("dev", "held"):
        rows = [row for row in load() if row["split"] == part
                and row.get("body") and row["english"]]
        probs = reader.read([said(row, VIEWS["english+signature+examples"])
                             for row in rows])
        refused = sure_count = 0
        for row, prob in zip(rows, probs):
            sure = {one for one, p in prob["behaviour"].items()
                    if p >= strong and M.checkable(one)}
            sure_count += bool(sure)
            if not sure:
                continue
            entry = re.search(r"function\s+(\w+)", row["signature"]).group(1)
            examples = [(list(args), out) for args, out in row["examples"]]
            cases = M.probes(examples)
            got = checker().run(row["body"], entry, cases) if cases else []
            pairs = examples + [(case, one["value"]) for case, one in
                                zip(cases, got) if "value" in one]
            refused += not sure <= M.behaviour(pairs)
        out[part] = {"right programs": len(rows), "with a sure reading":
                     sure_count, "wrongly refused": refused}
        print(f"{part:5} {out[part]}", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("train", "evaluate"))
    parser.add_argument("--base", default="unixcoder-base",
                        choices=sorted(BASES))
    parser.add_argument("--model", default="")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args(argv)
    if args.job == "train":
        name = args.model or f"meaning-{args.base.split('-')[0].lower()}"
        train(args.base, LLM / name, epochs=args.epochs)
    else:
        evaluate(LLM / args.model)
        refusals(LLM / args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

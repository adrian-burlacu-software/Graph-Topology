"""Whether a part of a long message says how the solution's code is to be
written -- or describes the problem, a concept, what the code must do.

An exercise's instructions say what is required of the code where they say
it (`This exercise is designed to be completed using recursion, rather than
loops.`), not where they ask the change; and they name what a way names
all through the problem (`Creating a zipper for a binary tree.`,
list-ops' `filter`). Read of every part, the ways estimator took the
problem for what was asked. This head reads what a part does: the reader
of meaning's encoder, frozen -- the one Python requests are read with
(`risk.BY_LANGUAGE`), shared at run time -- and a small head over it,
taught the sentences the teacher wrote and checked
(`teach_code_talk.required_check`). Its floor is chosen on one half of
the held-out sentences and measured on the other; and on long messages
made of held-out sentences, that the ways read of them are the ones a
required part says.

    python -m research.v698.required train
    python -m research.v698.required read "Use recursion, not loops."
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
MODEL = LLM / "required-head"
READER = "meaning-bilingual"
SEED = 698
#: the floor the chance must be over, chosen from these
FLOORS = (0.5, 0.6, 0.7, 0.8, 0.9)
#: the precision a part read as required must surely reach (Wilson bound,
#: `asked_ways._bound`): a part taken wrongly holds the code to a way
#: nobody asked
BAR = 0.80


def _rows(held: bool) -> list:
    from research.v698.teach_code_talk import REQUIRED
    return [row for row in map(json.loads, REQUIRED.open(encoding="utf-8"))
            if row["held"] == held]


def _half(text: str) -> int:
    from research.v698.asked_ways import _half
    return _half(text)


def _features(features, texts: list):
    from research.v698.asked_ways import _features
    torch = features.torch
    found = _features(features, texts)
    return torch.cat([found["pooled"], found["probs"]], -1).float().cpu()


def _head(torch, width: int, hidden: int):
    if not hidden:
        return torch.nn.Linear(width, 1)
    return torch.nn.Sequential(torch.nn.Linear(width, hidden),
                               torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(hidden, 1))


def _fit(torch, x, y, hidden: int, decay: float, epochs: int = 60,
         batch: int = 256):
    torch.manual_seed(SEED)
    head = _head(torch, x.shape[1], hidden)
    optimiser = torch.optim.AdamW(head.parameters(), lr=2e-3,
                                  weight_decay=decay)
    seen = y.sum().clamp(min=1)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=(len(y) - seen) / seen)
    order = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        head.train()
        for at in torch.randperm(len(x), generator=order).split(batch):
            optimiser.zero_grad()
            loss(head(x[at]).squeeze(-1), y[at]).backward()
            optimiser.step()
    head.eval()
    return head


def _scores(chances: list, truth: list, floor: float) -> dict:
    from research.v698.asked_ways import _bound
    read = [chance > floor for chance in chances]
    right = sum(r and t for r, t in zip(read, truth))
    taken = sum(read)
    return {"precision": round(right / max(1, taken), 3),
            "precision bound": round(_bound(right, taken), 3),
            "recall": round(right / max(1, sum(truth)), 3),
            "read": taken, "required": sum(truth), "of": len(truth)}


def _floor(chances: list, truth: list) -> float:
    """The lowest floor at which what is read as required surely reaches
    the bar; the highest where none does."""
    for floor in FLOORS:
        if _scores(chances, truth, floor)["precision bound"] >= BAR:
            return floor
    return FLOORS[-1]


def train(out: Path = MODEL) -> dict:
    """The head, its width and decay chosen on half the held-out
    sentences, its floor on the same half, measured on the other."""
    import torch

    from research.v696.risk import Features
    features = Features(LLM / READER)
    taught, held = _rows(False), _rows(True)
    choose = [row for row in held if _half(row["text"]) == 0]
    measure = [row for row in held if _half(row["text"]) == 1]
    x = {name: _features(features, [row["text"] for row in rows])
         for name, rows in (("train", taught), ("choose", choose),
                            ("measure", measure))}
    mean, std = x["train"].mean(0), x["train"].std(0).clamp(min=1e-6)
    x = {name: (value - mean) / std for name, value in x.items()}
    y = torch.tensor([float(row["required"]) for row in taught])
    truth = {name: [row["required"] for row in rows]
             for name, rows in (("choose", choose), ("measure", measure))}
    best, report = None, {}
    for hidden, decay in itertools.product((0, 256), (1e-4, 1e-2)):
        head = _fit(torch, x["train"], y, hidden, decay)
        with torch.no_grad():
            chances = torch.sigmoid(head(x["choose"])).squeeze(-1).tolist()
        floor = _floor(chances, truth["choose"])
        got = _scores(chances, truth["choose"], floor)
        report[f"{hidden}/{decay}"] = {"floor": floor, **got}
        print(f"  {hidden}/{decay}: floor {floor} {got}", flush=True)
        # the bar held, the most of what is required read
        rank = (got["precision bound"] >= BAR, got["recall"])
        if best is None or rank > best[0]:
            best = (rank, hidden, decay, floor, head)
    _, hidden, decay, floor, head = best
    with torch.no_grad():
        chances = torch.sigmoid(head(x["measure"])).squeeze(-1).tolist()
    measured = _scores(chances, truth["measure"], floor)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "mean": mean, "std": std,
                "width": x["train"].shape[1]}, out / "head.pt")
    settings = {"reader": READER, "chosen": {"hidden": hidden,
                                             "decay": decay},
                "floor": floor, "measured": measured, "choosing": report,
                "train": len(taught), "choose": len(choose),
                "measure": len(measure)}
    (out / "required.json").write_text(json.dumps(settings, indent=1),
                                       encoding="utf-8")
    print(json.dumps({"chosen": settings["chosen"], "floor": floor,
                      "measured": measured}, indent=1))
    return settings


class Head:
    """The trained head over the reader's encoder: `chances(texts)`."""

    def __init__(self, path: Path = MODEL, features=None) -> None:
        import torch
        self.torch = torch
        self.settings = json.loads((path / "required.json").read_text(
            encoding="utf-8"))
        saved = torch.load(str(path / "head.pt"), map_location="cpu")
        self.head = _head(torch, saved["width"],
                          self.settings["chosen"]["hidden"])
        self.head.load_state_dict(saved["head"])
        self.head.eval()
        self.mean, self.std = saved["mean"], saved["std"]
        if features is None:
            from research.v696.risk import Features
            features = Features(LLM / self.settings["reader"])
        self.features = features
        self.floor = self.settings["floor"]

    def chances(self, texts: list) -> list:
        if not texts:
            return []
        x = (_features(self.features, texts) - self.mean) / self.std
        with self.torch.no_grad():
            return self.torch.sigmoid(self.head(x)).squeeze(-1).tolist()


_ONE: list = []


def available() -> bool:
    return (MODEL / "required.json").exists()


def required(texts: list) -> list:
    """Of each text, whether it says how the solution's code is to be
    written; all False where no head is trained. The reader of meaning is
    the one the coding loads (`v697.coding.Tools`)."""
    if not available():
        return [False] * len(texts)
    if not _ONE:
        from research.v696.risk import Features
        from research.v697.coding import Tools
        _ONE.append(Head(features=Features(
            reader=Tools.get().reader_for("python"))))
    head = _ONE[0]
    return [chance > head.floor for chance in head.chances(texts)]


def messages(count: int = 200, seed: int = SEED) -> list:
    """Long messages made of held-out sentences, as instructions are: eight
    that say nothing of how the code is written, and in half of them one
    that does, somewhere among them -- (text, the ways it says)."""
    rows = _rows(True)
    plain = [row["text"] for row in rows if not row["required"]]
    said = [row for row in rows if row["required"] and row["ways"]]
    rng = random.Random(seed)
    out = []
    for at in range(count):
        parts = rng.sample(plain, 8)
        ways: list = []
        if at % 2:
            one = rng.choice(said)
            parts.insert(rng.randrange(len(parts) + 1), one["text"])
            ways = list(one["ways"])
        out.append(("\n\n".join(parts), ways))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("train", "read"))
    parser.add_argument("text", nargs="?")
    args = parser.parse_args(argv)
    if args.job == "train":
        train()
    else:
        head = Head()
        chance = head.chances([args.text])[0]
        print(f"required {chance:.3f} (floor {head.floor})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

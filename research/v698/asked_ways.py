"""Which ways of writing code a message asks for, read by an encoder.

*use a switch statement*, *can you do it without recursion*, *map and filter
instead of the loop*: a request or a change can ask for any number of the
ways `ways.py` checks code against. They are read as the risk estimators
read a request (`v696/risk.py`): the reader of meaning's encoder
(`meaning-unixcoder`, taught code and English) reads the message, and a
small head over what it reads says, of each way, whether it is asked for --
a set, not one label. The shared reader (`reader-code9`) is not touched:
nothing it reads moves.

Taught from the messages the teacher wrote asking for each way
(`teach_code_talk.py`: its `ways` field, a set by construction), questions
and everyday messages asking for none; each setting chosen on one half of
the held-out messages and measured on the other.

    python -m research.v698.asked_ways tune     (its own encoder: the one used)
    python -m research.v698.asked_ways train    (frozen features, as risk's)
    python -m research.v698.asked_ways read "use a switch and recurse"
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

from research.v698.ways import NAMES

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
DATA = LLM / "code-talk-data"
MODEL = LLM / "ways-estimator"
READER = "meaning-unixcoder"
SEED = 698
#: the ways a message can ask for: every one `ways.py` checks
LABELS = [one for one in NAMES if one != "none"]
#: how likely a way must be to be taken as asked for
FLOOR = 0.5
VIEWS = {"pooled": ("pooled",), "probs": ("probs",),
         "all": ("pooled", "probs")}


def _rows(part: str) -> list:
    """(text, ways) of every message of a split the ways are known of."""
    out = []
    for line in (DATA / f"{part}-code.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if "ways" in row:
            out.append((" ".join(row["words"]), list(row["ways"])))
    return out


def _features(features, texts: list) -> dict:
    found = features([(text, "") for text in texts], [[0.0]] * len(texts))
    found.pop("surface")
    return found


def _matrix(torch, found: dict, view: str):
    return torch.cat([found[one] for one in VIEWS[view]], -1)


def _head(torch, width: int, hidden: int):
    if not hidden:
        return torch.nn.Linear(width, len(LABELS))
    return torch.nn.Sequential(torch.nn.Linear(width, hidden),
                               torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(hidden, len(LABELS)))


def _targets(torch, ways: list):
    y = torch.zeros(len(ways), len(LABELS))
    for row, asked in enumerate(ways):
        for one in asked:
            y[row, LABELS.index(one)] = 1.0
    return y


def _fit(torch, x, y, hidden: int, decay: float, epochs: int = 60,
         batch: int = 512):
    torch.manual_seed(SEED)
    head = _head(torch, x.shape[1], hidden)
    optimiser = torch.optim.AdamW(head.parameters(), lr=2e-3,
                                  weight_decay=decay)
    # a way is asked for in few messages: each said as weighty as its
    # absence (capped), so that a rare one is still learned
    seen = y.sum(0).clamp(min=1)
    weight = ((len(y) - seen) / seen).clamp(max=20.0)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    order = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        head.train()
        for at in torch.randperm(len(x), generator=order).split(batch):
            optimiser.zero_grad()
            loss(head(x[at]), y[at]).backward()
            optimiser.step()
    head.eval()
    return head


def decided(row: list, floor: float = FLOOR, floors=None) -> list:
    """[(way, chance)] of one message's chances: of each family the likeliest
    way, where it is over the floor -- the ways of a family exclude each
    other (`with a regex` and `without one` are not both asked; nor are a
    for loop and no loop), so a message reading as two of one is read as
    the likelier."""
    from research.v698.ways import WAYS
    best: dict = {}
    for at, chance in enumerate(row):
        if chance <= (floors or {}).get(LABELS[at], floor):
            continue
        family = WAYS[LABELS[at]][0]
        if family not in best or chance > best[family][1]:
            best[family] = (LABELS[at], round(chance, 3))
    return sorted(best.values(), key=lambda one: -one[1])


def _said(torch, head, x) -> list:
    with torch.no_grad():
        chances = torch.sigmoid(head(x))
    return [[way for way, _ in decided(row)] for row in chances.tolist()]


def scores(said: list, truth: list) -> dict:
    """Of the messages asking for some way: the set read exactly; of those
    asking for several, the same; of those asking for none, that none was
    read; and over every way, how many asked for were read (recall) and how
    many read were asked for (precision)."""
    asked = [(s, t) for s, t in zip(said, truth) if t]
    several = [(s, t) for s, t in asked if len(t) > 1]
    none = [(s, t) for s, t in zip(said, truth) if not t]
    hit = sum(len(set(s) & set(t)) for s, t in zip(said, truth))
    told = sum(len(t) for t in truth)
    read = sum(len(s) for s in said)

    def share(pairs):
        return round(sum(set(s) == set(t) for s, t in pairs)
                     / max(1, len(pairs)), 3)

    several_hit = sum(len(set(s) & set(t)) for s, t in several)
    several_read = sum(len(s) for s, _ in several)
    by_way = {}
    for way in LABELS:
        read_ = sum(way in s for s in said)
        told_ = sum(way in t for t in truth)
        both = sum(way in s and way in t for s, t in zip(said, truth))
        by_way[way] = {"precision": round(both / read_, 3) if read_
                       else None, "recall": round(both / told_, 3)
                       if told_ else None, "asked": told_, "read": read_}
    return {"asked": len(asked), "exactly": share(asked),
            "several": len(several), "several exactly": share(several),
            "several precision": round(several_hit / max(1, several_read),
                                       3),
            "none": len(none), "none read as none": share(none),
            "recall": round(hit / max(1, told), 3),
            "precision": round(hit / max(1, read), 3),
            "by way": by_way}


def _half(text: str) -> int:
    return hashlib.sha1(text.encode()).digest()[0] % 2


def train(out: Path = MODEL) -> dict:
    """The head, its view (the encoder pooled, the reader's probabilities,
    both), width and decay chosen on half the held-out messages, measured
    on the other half -- into a new `llm/` directory."""
    import torch

    from research.v696.risk import Features
    features = Features(LLM / READER)
    train_rows, held = _rows("train"), _rows("valid")
    choose = [one for one in held if _half(one[0]) == 0]
    measure = [one for one in held if _half(one[0]) == 1]
    found = {part: _features(features, [text for text, _ in rows])
             for part, rows in (("train", train_rows), ("choose", choose),
                                ("measure", measure))}
    norm = {key: (value.mean(0), value.std(0).clamp(min=1e-6))
            for key, value in found["train"].items()}
    for part in found.values():
        for key in list(part):
            part[key] = (part[key] - norm[key][0]) / norm[key][1]
    y = _targets(torch, [ways for _, ways in train_rows])
    best, report = None, {}
    for view, hidden, decay in itertools.product(VIEWS, (0, 256),
                                                 (1e-4, 1e-2)):
        head = _fit(torch, _matrix(torch, found["train"], view), y, hidden,
                    decay)
        said = _said(torch, head, _matrix(torch, found["choose"], view))
        got = scores(said, [ways for _, ways in choose])
        key = f"{view}/{hidden}/{decay}"
        report[key] = got
        rank = (got["exactly"] + got["none read as none"]) / 2
        print(f"  {key}: {got}", flush=True)
        if best is None or rank > best[0]:
            best = (rank, view, hidden, decay, head)
    _, view, hidden, decay, head = best
    measured = scores(_said(torch, head, _matrix(torch, found["measure"],
                                                 view)),
                      [ways for _, ways in measure])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "norm": norm,
                "width": _matrix(torch, found["train"], view).shape[1]},
               out / "head.pt")
    settings = {"reader": READER, "labels": LABELS, "floor": FLOOR,
                "chosen": {"view": view, "hidden": hidden, "decay": decay},
                "measured": measured, "choosing": report,
                "train": len(train_rows), "choose": len(choose),
                "measure": len(measure)}
    (out / "ways.json").write_text(json.dumps(settings, indent=1),
                                   encoding="utf-8")
    print(json.dumps({"chosen": settings["chosen"], "measured": measured},
                     indent=1))
    return settings


#: how precise each way must be read, on the half floors are chosen on: a
#: way read is held to, and the code made to be so -- a way read that was
#: not asked costs more than one asked and not read
PRECISE = 0.85


def _floors(chances: list, truth: list) -> dict:
    """Each way's floor: the lowest (from `FLOOR` up) at which, decided as
    it will be, what is read of it is `PRECISE` -- where none is, the
    highest tried."""
    floors = {}
    steps = [round(FLOOR + 0.05 * at, 2) for at in range(10)]
    for way in LABELS:
        chosen = steps[-1]
        for floor in steps:
            trial = dict(floors, **{way: floor})
            said = [[w for w, _ in decided(row, FLOOR, trial)]
                    for row in chances]
            read = sum(way in s for s in said)
            right = sum(way in s and way in t for s, t in zip(said, truth))
            if not read or right / read >= PRECISE:
                chosen = floor
                break
        floors[way] = chosen
    return floors


def tune(out: Path = MODEL, epochs: int = 8, batch: int = 32,
         rate: float = 3e-5) -> dict:
    """The reader of meaning's encoder, a copy of it, taught with the head:
    whether `without a regex` or `with a regex` is asked turns on a word,
    and the encoder pooled as it is (`train`) reads the two alike (40% of
    messages read exactly). The copy is this estimator's own -- the reader
    of meaning and the shared reader are not touched."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from research.v696.reader import LONGEST
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(LLM / READER))
    encoder = AutoModel.from_pretrained(str(LLM / READER)).to(device)
    head = _head(torch, encoder.config.hidden_size, 256).to(device)
    train_rows, held = _rows("train"), _rows("valid")
    choose = [one for one in held if _half(one[0]) == 0]
    measure = [one for one in held if _half(one[0]) == 1]
    y = _targets(torch, [ways for _, ways in train_rows])
    seen = y.sum(0).clamp(min=1)
    loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=((len(y) - seen) / seen).clamp(max=20.0).to(device))
    params = list(encoder.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=rate, weight_decay=0.01)
    steps = epochs * ((len(train_rows) + batch - 1) // batch)
    from transformers import get_linear_schedule_with_warmup
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * 0.06),
                                               steps)

    def pooled(texts):
        said = tokenizer(texts, truncation=True, max_length=LONGEST,
                         padding=True, return_tensors="pt").to(device)
        said.pop("token_type_ids", None)
        hidden = encoder(**said).last_hidden_state
        mask = said["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def chances_of(rows):
        encoder.eval()
        head.eval()
        out = []
        with torch.no_grad():
            for at in range(0, len(rows), 128):
                out += torch.sigmoid(head(pooled(
                    [text for text, _ in rows[at:at + 128]]).float())).tolist()
        return out

    def said_of(rows, floors=None):
        return [[way for way, _ in decided(row, FLOOR, floors)]
                for row in chances_of(rows)]

    best, report = None, {}
    order = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        encoder.train()
        head.train()
        total = 0.0
        for at in torch.randperm(len(train_rows), generator=order).split(
                batch):
            texts = [train_rows[i][0] for i in at.tolist()]
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = head(pooled(texts))
            value = loss(logits.float(), y[at].to(device))
            optimiser.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            schedule.step()
            total += float(value.detach())
        got = scores(said_of(choose), [ways for _, ways in choose])
        report[f"epoch {epoch + 1}"] = got
        print(f"  epoch {epoch + 1}: loss {total:.1f} {got}", flush=True)
        rank = (got["exactly"] + got["none read as none"]) / 2
        if best is None or rank > best[0]:
            best = (rank, epoch + 1)
            out.mkdir(parents=True, exist_ok=True)
            encoder.save_pretrained(str(out))
            tokenizer.save_pretrained(str(out))
            torch.save({"head": head.state_dict(),
                        "width": encoder.config.hidden_size},
                       out / "head.pt")
    # the best epoch, measured on the half nothing was chosen by
    encoder = AutoModel.from_pretrained(str(out)).to(device)
    head.load_state_dict(torch.load(str(out / "head.pt"),
                                    map_location=device)["head"])
    floors = _floors(chances_of(choose), [ways for _, ways in choose])
    measured = scores(said_of(measure, floors),
                      [ways for _, ways in measure])
    settings = {"reader": READER, "tuned": True, "labels": LABELS,
                "floor": FLOOR, "floors": floors,
                "chosen": {"view": "pooled", "hidden": 256,
                           "epoch": best[1]},
                "measured": measured, "choosing": report,
                "train": len(train_rows), "choose": len(choose),
                "measure": len(measure)}
    (out / "ways.json").write_text(json.dumps(settings, indent=1),
                                   encoding="utf-8")
    print(json.dumps({"chosen": settings["chosen"], "measured": measured},
                     indent=1))
    return settings


class Estimator:
    """The trained head over the reader of meaning: `read(texts)` -> for
    each, [(way, chance)], likelier first."""

    def __init__(self, path: Path = MODEL, features=None) -> None:
        import torch
        self.torch = torch
        self.settings = json.loads((path / "ways.json").read_text(
            encoding="utf-8"))
        saved = torch.load(str(path / "head.pt"), map_location="cpu")
        chosen = self.settings["chosen"]
        self.head = _head(torch, saved["width"], chosen["hidden"])
        self.head.load_state_dict(saved["head"])
        self.head.eval()
        self.tuned = None
        if self.settings.get("tuned"):
            # its own encoder, taught with the head (`tune`)
            from transformers import AutoModel, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.tuned = AutoModel.from_pretrained(str(path)).to(
                self.device).eval()
            self.head.to(self.device)
            return
        self.norm = saved["norm"]
        if features is None:
            from research.v696.risk import Features
            features = Features(LLM / self.settings["reader"])
        self.features = features

    def read(self, texts: list) -> list:
        torch = self.torch
        if self.tuned is not None:
            from research.v696.reader import LONGEST
            with torch.no_grad():
                said = self.tokenizer(texts, truncation=True,
                                      max_length=LONGEST, padding=True,
                                      return_tensors="pt").to(self.device)
                said.pop("token_type_ids", None)
                hidden = self.tuned(**said).last_hidden_state
                mask = said["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chances = torch.sigmoid(self.head(pooled)).tolist()
        else:
            found = _features(self.features, texts)
            found = {key: (value - self.norm[key][0]) / self.norm[key][1]
                     for key, value in found.items()}
            x = _matrix(torch, found, self.settings["chosen"]["view"])
            with torch.no_grad():
                chances = torch.sigmoid(self.head(x)).tolist()
        floor = self.settings.get("floor", FLOOR)
        floors = self.settings.get("floors")
        return [decided(row, floor, floors) for row in chances]


_ONE: list = []


def available() -> bool:
    return (MODEL / "ways.json").exists()


def read(text: str) -> list:
    """[(way, chance)] a message asks for; [] where no estimator is
    trained. The reader of meaning is shared with the coding that loads it
    (`v697.coding.Tools`)."""
    if not available():
        return []
    if not _ONE:
        settings = json.loads((MODEL / "ways.json").read_text(
            encoding="utf-8"))
        if settings.get("tuned"):
            _ONE.append(Estimator())
        else:
            from research.v697.coding import Tools
            from research.v696.risk import Features
            _ONE.append(Estimator(features=Features(
                reader=Tools.get().reader)))
    return _ONE[0].read([text])[0]


def measure(path: Path = MODEL) -> dict:
    """The estimator as saved, on the held-out half nothing was chosen by:
    every way's precision and recall, both directions of each family."""
    estimator = Estimator(path)
    rows = [one for one in _rows("valid") if _half(one[0]) == 1]
    said = []
    for at in range(0, len(rows), 64):
        said += [[way for way, _ in one] for one in estimator.read(
            [text for text, _ in rows[at:at + 64]])]
    got = scores(said, [ways for _, ways in rows])
    print(json.dumps({key: value for key, value in got.items()
                      if key != "by way"}, indent=1))
    from research.v698.ways import WAYS
    for way, row in sorted(got["by way"].items(),
                           key=lambda one: (WAYS[one[0]][0], one[0])):
        print(f"  {WAYS[way][0]:>14} {way:>14}: precision "
              f"{row['precision']}  recall {row['recall']}  "
              f"({row['asked']} asked, {row['read']} read)")
    return got


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("train", "tune", "measure", "read"))
    parser.add_argument("text", nargs="*")
    args = parser.parse_args(argv)
    if args.job == "train":
        train()
    elif args.job == "tune":
        tune()
    elif args.job == "measure":
        measure()
    else:
        print(read(" ".join(args.text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

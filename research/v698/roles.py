"""What each part of a long message is: what it asks, how the code is to be
written, a tip, what the code must do, the problem, an explanation, an
example, or layout.

An exercise's instructions mix all of these, and only one says how the
solution is to be written (`This exercise is designed to be completed
using recursion, rather than loops.`). Read for ways of writing, every
part was taken as asking: `Creating a zipper for a binary tree.` as zip,
list-ops' `filter (given a predicate and a list ...)` as filter, a
concept page's `the set method equivalent of & is ...` as set. Each part
is read here by its role; the ways are read only of the parts that say
how (`reading._ways_of`), never of a tip (`You can use enumerate()`).

Taught from parts of Exercism's Python docs that no benchmark measures
(its concept exercises and concepts), each labelled by hand
(`roles/labelled.jsonl`): the teacher, asked of the roles outright or by
small yes-or-no questions, agreed with the hand labels on 42% and 63% of
them, and the code model on 15% and 26%. Binding requirements are rare
in real text (9 of 2125), so sentences the teacher writes requiring each
way -- and suggesting it, beside them -- are added, each kept where the
teacher reads it as being about that way (`check`). Measured on the parts
of docs held apart by concept (`roles/gold.jsonl`, never taught or chosen
on), and on held-out requirements set among held-out real parts.

    python -m research.v698.roles write | check | tune | measure
    python -m research.v698.roles read "This exercise must use recursion."
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
HERE = Path(__file__).resolve().parent / "roles"
#: parts of never-measured docs, labelled by hand: taught (bar `DEV`)
LABELLED = HERE / "labelled.jsonl"
#: parts of docs held apart by concept: measured, never taught
GOLD = HERE / "gold.jsonl"
DATA = LLM / "code-talk-data"
WRITTEN = DATA / "roles-written.jsonl"
TAUGHT = DATA / "roles-taught.jsonl"
MODEL = LLM / "roles-reader"
READER = "meaning-bilingual"
SEED = 698
ROLES = ("asks", "how", "suggests", "does", "describes", "explains",
         "example", "layout")
#: how sure a part must be read as saying how the code is written, chosen
#: from these on the dev parts: the lowest whose precision is surely
#: (Wilson, one standard error) at the bar -- a part taken wrongly holds
#: the code to a way nobody asked
FLOORS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
BAR = 0.8


def group(doc: str) -> str:
    """A doc's concept or concept exercise: its pages copy each other."""
    parts = doc.split("/")
    return "/".join(parts[:2] if doc.startswith("concepts/") else parts[:3])


def _dev(row: dict) -> bool:
    """The parts the epoch and floor are chosen on: one concept in ten of
    the labelled; half the held-out of the teacher's."""
    if "doc" in row:
        return zlib.crc32(group(row["doc"]).encode()) % 10 == 0
    return row["held"] and zlib.crc32(row["text"].encode()) % 2 == 0


def _rows(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# -- the teacher's requirements and tips --------------------------------------

def jobs() -> list:
    """For each way, sentences requiring it and sentences suggesting it:
    the role turns on whether the code must be so, not on the way."""
    from research.v698.teach_code_talk import STYLE
    from research.v698.ways import WAYS
    out = []
    for way, (family, said, _) in WAYS.items():
        if family in (None, "language", "length"):
            continue
        out.append((f"roles|how|{way}", f"Write 12 different sentences as "
                    f"found in the instructions or notes of a programming "
                    f"exercise, each one saying that the solution must be "
                    f"written {said}: as a requirement or a constraint, or "
                    f"as what the exercise is designed to practise. " + STYLE,
                    {"role": "how", "way": way}))
        out.append((f"roles|suggests|{way}", f"Write 12 different hints as "
                    f"found under a programming exercise, each one "
                    f"suggesting -- not requiring -- that the solution "
                    f"could be written {said}: a tip, an option, something "
                    f"that might help. " + STYLE,
                    {"role": "suggests", "way": way}))
    return out


def write(samples: int = 2) -> None:
    from research.v698 import teach_code_talk as T
    DATA.mkdir(parents=True, exist_ok=True)
    todo = [one for one in jobs() if one[0] not in T._done(WRITTEN)]
    print(f"{len(todo)} jobs to write", flush=True)
    T._write(todo, WRITTEN, samples, batch=6)


def check(batch: int = 24) -> dict:
    """Each sentence kept where the teacher, asked of its way's family
    outright, reads it as about that way (`teach_code_talk._family_choice`:
    reliable where asked so, 13 of 13): {text, role, way, held}."""
    from research.v696.teach_meaning import Teacher
    from research.v698 import teach_code_talk as T
    from research.v698.ways import WAYS
    current = {key: meta for key, _, meta in jobs()}
    todo, seen = [], set()
    for row in _rows(WRITTEN):
        meta = current.get(row["key"])
        if meta is None:
            continue
        for line in row["lines"]:
            said = " ".join(T.words(line))
            if not said or said in seen or T._code_not_words(line):
                continue
            seen.add(said)
            todo.append((said, meta))
    print(f"{len(todo)} sentences to check", flush=True)
    teacher = Teacher()
    kept = 0
    with TAUGHT.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            asked = [T._family_choice(said, WAYS[meta["way"]][0])
                     for said, meta in chunk]
            replies = teacher.write([prompt for prompt, _ in asked],
                                    longest=3)
            for (said, meta), (_, options), reply in zip(chunk, asked,
                                                         replies):
                if T._picked(reply[0], options) != meta["way"]:
                    continue
                kept += 1
                out.write(json.dumps({"text": said, **meta,
                                      "held": T._held(said)}) + "\n")
    print(f"kept {kept} of {len(todo)}", flush=True)
    return {"kept": kept, "checked": len(todo)}


# -- the reader ---------------------------------------------------------------

def _texts(rows: list) -> list:
    from research.v692.corpus import words
    return [" ".join(words(row["text"])) for row in rows]


def scores(said: list, truth: list) -> dict:
    """Of each role, how many read as it are it (precision) and how many
    of it are read so (recall)."""
    out = {"agreement": round(sum(s == t for s, t in zip(said, truth))
                              / max(1, len(truth)), 3)}
    for role in ROLES:
        read = sum(s == role for s in said)
        told = sum(t == role for t in truth)
        right = sum(s == t == role for s, t in zip(said, truth))
        out[role] = {"precision": f"{right}/{read}",
                     "recall": f"{right}/{told}"}
    return out


def _decided(row: list, floor: float) -> str:
    """The likeliest role -- `how` only where sure past the floor; else the
    likeliest of the rest."""
    ranked = sorted(range(len(ROLES)), key=lambda at: -row[at])
    for at in ranked:
        if ROLES[at] != "how" or row[at] >= floor:
            return ROLES[at]
    return ROLES[ranked[0]]


def _floor(chances: list, truth: list) -> float:
    from research.v698.asked_ways import _bound
    for floor in FLOORS:
        said = [_decided(row, floor) for row in chances]
        read = sum(s == "how" for s in said)
        right = sum(s == t == "how" for s, t in zip(said, truth))
        if not read or _bound(right, read) >= BAR:
            return floor
    return FLOORS[-1]


def tune(out: Path = MODEL, epochs: int = 6, batch: int = 32,
         rate: float = 3e-5) -> dict:
    """The reader of meaning's encoder, a copy of it, taught with a head
    over the roles; the epoch (agreement) and the floor of `how` chosen on
    the dev parts, never on the gold."""
    import torch
    from transformers import (AutoModel, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from research.v696.reader import LONGEST
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(LLM / READER))
    encoder = AutoModel.from_pretrained(str(LLM / READER)).to(device)
    head = torch.nn.Linear(encoder.config.hidden_size, len(ROLES)).to(device)
    labelled = _rows(LABELLED)
    taught = [row for row in _rows(TAUGHT)
              if not row["held"] or _dev(row)]
    rows = labelled + taught
    train = [row for row in rows if not _dev(row)]
    dev = [row for row in rows if _dev(row)]
    texts = {"train": _texts(train), "dev": _texts(dev)}
    y = torch.tensor([ROLES.index(row["role"]) for row in train])
    seen = torch.bincount(y, minlength=len(ROLES)).float().clamp(min=1)
    # each role weighed as its rarity's square root: `how` is 1 in 200
    loss = torch.nn.CrossEntropyLoss(
        weight=(seen.sum() / seen).sqrt().to(device))
    params = list(encoder.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=rate, weight_decay=0.01)
    steps = epochs * ((len(train) + batch - 1) // batch)
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * 0.06),
                                               steps)

    def pooled(chunk):
        said = tokenizer(chunk, truncation=True, max_length=LONGEST,
                         padding=True, return_tensors="pt").to(device)
        said.pop("token_type_ids", None)
        hidden = encoder(**said).last_hidden_state
        mask = said["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def chances_of(chunks):
        encoder.eval()
        head.eval()
        found = []
        with torch.no_grad():
            for at in range(0, len(chunks), 128):
                found += torch.softmax(head(pooled(
                    chunks[at:at + 128]).float()), -1).tolist()
        return found

    truth = [row["role"] for row in dev]
    best, report = None, {}
    order = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        encoder.train()
        head.train()
        total = 0.0
        for at in torch.randperm(len(train), generator=order).split(batch):
            chunk = [texts["train"][i] for i in at.tolist()]
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = head(pooled(chunk))
            value = loss(logits.float(), y[at].to(device))
            optimiser.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            schedule.step()
            total += float(value.detach())
        chances = chances_of(texts["dev"])
        got = scores([_decided(row, 0.5) for row in chances], truth)
        report[f"epoch {epoch + 1}"] = got
        print(f"  epoch {epoch + 1}: loss {total:.1f} {got}", flush=True)
        if best is None or got["agreement"] > best[0]:
            best = (got["agreement"], epoch + 1)
            out.mkdir(parents=True, exist_ok=True)
            encoder.save_pretrained(str(out))
            tokenizer.save_pretrained(str(out))
            torch.save({"head": head.state_dict(),
                        "width": encoder.config.hidden_size},
                       out / "head.pt")
    encoder = AutoModel.from_pretrained(str(out)).to(device)
    head.load_state_dict(torch.load(str(out / "head.pt"),
                                    map_location=device)["head"])
    floor = _floor(chances_of(texts["dev"]), truth)
    settings = {"reader": READER, "roles": list(ROLES), "floor": floor,
                "epoch": best[1], "choosing": report, "train": len(train),
                "dev": len(dev)}
    (out / "roles.json").write_text(json.dumps(settings, indent=1),
                                    encoding="utf-8")
    print(json.dumps({"epoch": best[1], "floor": floor}), flush=True)
    return settings


class Reader:
    """The trained reader: `read(texts)` -> each one's role. On the CPU: a
    message's parts are a few dozen, and the card is the code model's."""

    def __init__(self, path: Path = MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.settings = json.loads((path / "roles.json").read_text(
            encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.encoder = AutoModel.from_pretrained(str(path)).to(device).eval()
        saved = torch.load(str(path / "head.pt"), map_location=device)
        self.head = torch.nn.Linear(saved["width"], len(ROLES)).to(device)
        self.head.load_state_dict(saved["head"])
        self.head.eval()
        self.floor = self.settings["floor"]

    def chances(self, texts: list) -> list:
        from research.v696.reader import LONGEST
        torch, out = self.torch, []
        with torch.no_grad():
            for at in range(0, len(texts), 64):
                said = self.tokenizer(texts[at:at + 64], truncation=True,
                                      max_length=LONGEST, padding=True,
                                      return_tensors="pt").to(self.device)
                said.pop("token_type_ids", None)
                hidden = self.encoder(**said).last_hidden_state
                mask = said["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                out += torch.softmax(self.head(pooled), -1).tolist()
        return out

    def read(self, texts: list) -> list:
        return [_decided(row, self.floor) for row in self.chances(texts)]


_ONE: list = []


def available() -> bool:
    return (MODEL / "roles.json").exists()


def roles(texts: list) -> list:
    """Each text's role (`ROLES`), as the words are read (lower case);
    None each where no reader is trained."""
    if not texts:
        return []
    if not available():
        return [None] * len(texts)
    if not _ONE:
        _ONE.append(Reader())
    return _ONE[0].read(texts)


# -- measured -----------------------------------------------------------------

def messages(count: int = 200, seed: int = SEED) -> list:
    """Long messages of held-apart real parts (no `how` among them), and in
    half of them one held-out requirement among them: (parts, its ways)."""
    import random
    plain = [row["text"] for row in _rows(GOLD) if row["role"] != "how"]
    said = [row for row in _rows(TAUGHT) if row["held"] and not _dev(row)
            and row["role"] == "how"]
    rng = random.Random(seed)
    out = []
    for at in range(count):
        parts = rng.sample(plain, 10)
        ways: list = []
        if at % 2 and said:
            one = rng.choice(said)
            parts.insert(rng.randrange(len(parts) + 1), one["text"])
            ways = [one["way"]]
        out.append((parts, ways))
    return out


def measure() -> dict:
    """The gold parts by role; held-out requirements and tips; and the
    ways read of long messages made of them (`reading._ways_of`)."""
    from research.v698 import reading
    from research.v698.asked_ways import scores as way_scores
    reader = Reader(device="cuda")
    gold = _rows(GOLD)
    found = {"gold": scores(reader.read(_texts(gold)),
                            [row["role"] for row in gold])}
    held = [row for row in _rows(TAUGHT) if row["held"] and not _dev(row)]
    found["held requirements and tips"] = scores(
        reader.read(_texts(held)), [row["role"] for row in held])
    _ONE[:] = [reader]
    made = messages()
    said = [[way for way in reading._ways_of(parts, []) if way != "python"]
            for parts, _ in made]
    got = way_scores(said, [ways for _, ways in made])
    got.pop("by way")
    found["messages"] = got
    print(json.dumps(found, indent=1))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("write", "check", "tune", "measure",
                                        "read"))
    parser.add_argument("text", nargs="?")
    args = parser.parse_args(argv)
    if args.job == "write":
        write()
    elif args.job == "check":
        print(check())
    elif args.job == "tune":
        tune()
    elif args.job == "measure":
        measure()
    else:
        reader = Reader()
        from research.v692.corpus import words
        row = reader.chances([" ".join(words(args.text))])[0]
        ranked = sorted(zip(ROLES, row), key=lambda one: -one[1])[:3]
        print(", ".join(f"{role} {chance:.3f}" for role, chance in ranked),
              f"(how floor {reader.floor})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Teaching an encoder to find a function by what it does (2026-10-25, the
core measured on RepoQA: a subject said by what it does).

    python -m research.v698.teach_search fetch    data/codesearchnet/*.parquet
    python -m research.v698.teach_search train    llm/search-minilm

A question that says a function by what it does (`which function merges
adjacent strings in a line?`) was found by the meaning of each symbol's
name and docstring (`asking._described`, MiniLM as it came): RepoQA's
descriptions say what a function does, which its code says and its
docstring mostly does not -- the nearest was the needle 69 times of 199.
Here MiniLM is taught to put a description near the code that does it:
CodeSearchNet's train split (Husain et al. 2019; Python and JavaScript, a
function and its docstring), each docstring beside its function's code with
the docstring taken out, the other functions of the batch the ones it is
not (in-batch negatives). Functions of RepoQA's repositories are never
taught (`HELD`): they are what is measured.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "codesearchnet"
BASE = ROOT / "llm" / "MiniLM-L6-v2"
OUT = ROOT / "llm" / "search-minilm"
SEED = 698
#: pairs taught of each language, the longest description and code read
EACH = {"python": 100000, "javascript": 60000}
ASKED, CODE = 64, 256
BATCH, RATE, TEMPERATURE = 64, 3e-5, 0.05


#: CodeSearchNet as Hugging Face keeps it in parquet: train taught,
#: validation what the search's thresholds are set on (`asking.SEARCHED`)
PARQUET = ("https://huggingface.co/api/datasets/code-search-net/"
           "code_search_net/parquet/{language}/{split}/0.parquet")


def fetch() -> None:
    import urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    for language in EACH:
        for split in ("train", "validation"):
            path = DATA / f"{language}-{split}.parquet"
            if path.exists():
                continue
            request = urllib.request.Request(
                PARQUET.format(language=language, split=split),
                headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=1800) as reply:
                path.write_bytes(reply.read())
            print(f"{path.stat().st_size} bytes -> {path}", flush=True)


def _held() -> set:
    """RepoQA's repositories: never taught."""
    source = ROOT / "data" / "repoqa" / "repoqa-2024-06-23.json"
    if not source.exists():
        return set()
    data = json.loads(source.read_text(encoding="utf-8"))
    return {repo["repo"].lower() for language in data.values()
            for repo in language}


#: a Python function's docstring, as its first statement
DOCSTRING = re.compile(r'^(\s*(?:async\s+)?def\s[^\n]*:\s*\n)\s*[rbuRBU]?'
                       r'("""|\'\'\')(?:.|\n)*?\2[^\n]*\n', re.M)


def code_of(text: str) -> str:
    """A function's code with its docstring taken out: what it does, said by
    the code alone."""
    return DOCSTRING.sub(r"\1", text, count=1)


def pairs() -> list:
    """(description, code) taught: CodeSearchNet's train split, of
    repositories not RepoQA's, a description of a few words or more."""
    import pyarrow.parquet as pq
    held = _held()
    rng = random.Random(SEED)
    out = []
    for language, most in EACH.items():
        table = pq.read_table(DATA / f"{language}-train.parquet", columns=[
            "repository_name", "func_code_string",
            "func_documentation_string"]).to_pylist()
        rows = [one for one in table
                if one["repository_name"].lower() not in held and
                len(one["func_documentation_string"].split()) >= 4]
        for one in rng.sample(rows, min(most, len(rows))):
            out.append((" ".join(one["func_documentation_string"].split()),
                        code_of(one["func_code_string"])))
    rng.shuffle(out)
    return out


def _embed(net, batch):
    import torch
    hidden = net(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    return torch.nn.functional.normalize(
        (hidden * mask).sum(1) / mask.sum(1), dim=-1)


def train(epochs: int = 1) -> None:
    import torch
    from transformers import (AutoModel, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(SEED)
    taught = pairs()
    print(f"{len(taught)} pairs", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    net = AutoModel.from_pretrained(str(BASE)).to("cuda")
    optimiser = torch.optim.AdamW(net.parameters(), lr=RATE,
                                  weight_decay=0.01)
    steps = epochs * (len(taught) // BATCH)
    schedule = get_linear_schedule_with_warmup(optimiser, steps // 20, steps)
    net.train()
    started = time.time()
    for epoch in range(epochs):
        total = 0.0
        for at in range(0, len(taught) - BATCH + 1, BATCH):
            chunk = taught[at:at + BATCH]
            asked = tokenizer([one for one, _ in chunk], padding=True,
                              truncation=True, max_length=ASKED,
                              return_tensors="pt").to("cuda")
            code = tokenizer([one for _, one in chunk], padding=True,
                             truncation=True, max_length=CODE,
                             return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = _embed(net, asked) @ _embed(net, code).T
            scores = scores.float() / TEMPERATURE
            wanted = torch.arange(len(chunk), device="cuda")
            # each description its own code among the batch's, and each
            # code its own description
            loss = (torch.nn.functional.cross_entropy(scores, wanted) +
                    torch.nn.functional.cross_entropy(scores.T, wanted)) / 2
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            if (at // BATCH) % 200 == 0:
                print(f"  {at // BATCH}/{steps}: loss "
                      f"{float(loss.detach()):.3f} "
                      f"({time.time() - started:.0f}s)", flush=True)
        print(f"epoch {epoch + 1}: mean loss "
              f"{total / max(1, len(taught) // BATCH):.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    (OUT / "taught.json").write_text(json.dumps({
        "base": str(BASE), "pairs": len(taught), "each": EACH,
        "asked": ASKED, "code": CODE, "held": sorted(_held())}, indent=1),
        encoding="utf-8")
    print(f"saved to {OUT}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "train"))
    parser.add_argument("--epochs", type=int, default=1)
    options = parser.parse_args(argv)
    if options.job == "fetch":
        fetch()
    else:
        train(options.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

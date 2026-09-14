"""Fine-tune MiniLM-L6-v2 so questions asking the same cell are near.

Off the shelf the encoder measures who and what a question is about more than
what it asks (`exemplars.py`). This teaches it the cells, from data nothing
larger wrote: the grammar's documented questions, read by the grammar into
their cells, each also said with the other verbs VerbNet gives the same
meaning (`change.meaning`: `where did it go` -> `where did it wander`, `who is
carrying it` -> `who is holding it`), in the same inflection. Keys, as
`exemplars.key` says them: slots are `it`.

    python -m research.v689.train_exemplars [--epochs 3]

Supervised contrastive loss over each batch (same cell near, every other cell
far), on the GPU, into `llm/MiniLM-L6-v2-cells`, which `exemplars.py` uses
when it is there. A few minutes.
"""
from __future__ import annotations

import argparse
import random
import sys

from research.v689 import change, exemplars

OUT = exemplars.MODEL.parent / "MiniLM-L6-v2-cells"


def _inflect(verb: str, tag: str) -> str | None:
    """A regular verb in the tag's form, checked by WordNet's morphy."""
    from nltk.corpus import wordnet
    if not verb.isalpha():
        return None
    if tag == "VBG":
        form = (verb[:-1] if verb.endswith("e") and not verb.endswith("ee")
                else verb) + "ing"
    elif tag in ("VBD", "VBN"):
        form = verb + ("d" if verb.endswith("e") else "ed")
    elif tag == "VBZ":
        form = verb + ("es" if verb.endswith(("s", "sh", "ch", "x")) else "s")
    else:
        return verb
    return form if wordnet.morphy(form, "v") == verb else None


def data(lexicon, analyse) -> list[tuple[str, tuple]]:
    """(key, cell) for every documented question and its verb variants."""
    from research.v689.clauses import parse
    from research.v689.grammar import propose
    from research.v689.reading import tokens_of

    by_meaning: dict[str, list[str]] = {}
    for verb in change.frames():
        if " " in verb:
            continue
        for sense in change.meaning(verb):
            by_meaning.setdefault(sense, []).append(verb)
    found: list[tuple[str, tuple]] = []
    for text in exemplars.documented():
        lower, typed = tokens_of(text)
        names = frozenset(word.lower() for word in typed if word[:1].isupper())
        goals = propose(lower, lexicon, names, text)
        analysis = analyse(typed)
        if not goals or not analysis:
            continue
        cell = goals[0].cell
        words = parse(typed, analysis)
        spans = exemplars.phrases(words)
        found.append((exemplars.key(lower, spans), cell))
        verb = next((word for word in words if word.dep == "ROOT"
                     and word.verbal), None)
        if verb is None:
            continue
        stem = lexicon.lemma(lower[verb.index])
        for sense in change.meaning(stem):
            for other in by_meaning.get(sense, ()):
                form = _inflect(other, verb.tag)
                if form and other != stem:
                    varied = list(lower)
                    varied[verb.index] = form
                    found.append((exemplars.key(varied, spans), cell))
    return list(dict.fromkeys(found))


def train(pairs: list[tuple[str, tuple]], epochs: int, batch: int = 96,
          rate: float = 3e-5, temperature: float = 0.05) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(exemplars.MODEL))
    model = AutoModel.from_pretrained(str(exemplars.MODEL)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=rate)
    cells = sorted({cell for _, cell in pairs})
    label = {cell: at for at, cell in enumerate(cells)}
    rng = random.Random(689)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total = 0.0
        for start in range(0, len(pairs), batch):
            chunk = pairs[start:start + batch]
            encoded = tokenizer([text for text, _ in chunk], padding=True,
                                truncation=True, max_length=64,
                                return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            vectors = torch.nn.functional.normalize(
                (hidden * mask).sum(1) / mask.sum(1), dim=1)
            labels = torch.tensor([label[cell] for _, cell in chunk],
                                  device=device)
            logits = vectors @ vectors.T / temperature
            eye = torch.eye(len(chunk), device=device, dtype=torch.bool)
            logits = logits.masked_fill(eye, -1e9)
            same = (labels[:, None] == labels[None, :]) & ~eye
            log_probability = logits - torch.logsumexp(logits, 1, keepdim=True)
            counts = same.sum(1)
            keep = counts > 0
            loss = -((log_probability * same).sum(1)[keep]
                     / counts[keep]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss)
        print(f"epoch {epoch + 1}: loss {total:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=3)
    options = parser.parse_args(argv)
    from research.v687 import build
    from research.v689 import babi
    from research.v689.session import Session

    babi._start(str(build.DEFAULT_STORE), 1)
    lexicon = Session(babi._ASKER, conversation="train", example=True).lexicon()
    pairs = data(lexicon, lexicon.analyse)
    print(f"{len(pairs)} keys over {len({c for _, c in pairs})} cells",
          flush=True)
    train(pairs, options.epochs)
    print(f"saved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

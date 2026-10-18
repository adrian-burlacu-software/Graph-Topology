"""How natural a piece of code is (v705): the code model's likelihood of
it, token by token, many pieces at once on the GPU -- to try the edits a
programmer would more likely have meant first, of the hundreds one-edit
repair makes (`repair.candidates`), before the tests judge any.

    scores(sources)   [log-likelihood per source], higher more natural

Each source is scored whole and against the code it was made from: what
counts is how much less likely the edit made it (`gained`), so a long
function is not unlike a short one for its length.

The vocabulary is 152k words: the whole of a batch's next-word scores at
once (16 sources of 1024 tokens) is 10 GB, past the card, and spilled into
the machine's memory a run went from 17 s a bug to 70 min. So the model
gives its last hidden layer only, and the words' scores are made from it a
few positions at a time, kept only for the word that came; and a batch is
as many sources as `TOKENS` tokens hold.
"""
from __future__ import annotations

#: tokens a batch holds in all, and of a source at most
TOKENS, LONGEST = 4096, 1024
#: positions whose next-word scores are made at once
POSITIONS = 128


def scores(sources: list, held=None, tokens: int = TOKENS) -> list:
    """The summed log-likelihood of each source under a language model
    held (`.tokenizer`, `.model`): the code model where none is given -- the
    server's own editor where the code model has no room beside it."""
    import torch
    from research.v705 import coder
    model = held or coder.get()
    tokenizer, network = model.tokenizer, model.model
    device = next(network.parameters()).device
    encoded = [tokenizer(one, truncation=True, max_length=LONGEST)[
        "input_ids"] for one in sources]
    order = sorted(range(len(sources)), key=lambda at: len(encoded[at]))
    out = [0.0] * len(sources)
    at = 0
    while at < len(order):
        longest = len(encoded[order[min(len(order) - 1, at)]])
        size = max(1, tokens // max(1, longest))
        # sorted by length: the last of a batch is its longest
        while size > 1 and at + size - 1 < len(order) and len(
                encoded[order[at + size - 1]]) * size > tokens:
            size -= 1
        batch = order[at:at + size]
        at += size
        width = max(len(encoded[one]) for one in batch)
        pad = tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(encoded[one])) +
                            encoded[one] for one in batch], device=device)
        mask = torch.tensor([[0] * (width - len(encoded[one])) +
                             [1] * len(encoded[one]) for one in batch],
                            device=device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            hidden = network.model(input_ids=ids, attention_mask=mask,
                                   position_ids=positions).last_hidden_state
            total = torch.zeros(len(batch), device=device)
            for start in range(0, width - 1, POSITIONS):
                end = min(width - 1, start + POSITIONS)
                # the word that came, less all words' (log-sum-exp): no
                # softmax over the vocabulary made and read again
                logits = network.lm_head(hidden[:, start:end])
                target = ids[:, start + 1:end + 1]
                picked = logits.gather(-1, target.unsqueeze(-1)).squeeze(
                    -1).float() - torch.logsumexp(logits, dim=-1).float()
                total += (picked * mask[:, start + 1:end + 1]).sum(dim=1)
                del logits
        for one, value in zip(batch, total.tolist()):
            out[one] = value
    return out


#: tokens of the code kept before an edit, and after it, when scoring it
BEFORE, AFTER = 48, 24


def gained(original: str, sources: list, said: str = "",
           held=None, given: str = "", tokens: int = TOKENS) -> list:
    """How much more likely each source is than the code it was made from
    (negative: less likely) -- scored where they differ only: the tokens
    from `BEFORE` ahead of the first that differs to `AFTER` past the last,
    of each and of the original alike (the same text either side of an
    edit scores the same; whole, a 55-line function's 1500 edits took
    three minutes). `given`: code written ahead of each window alike."""
    from research.v705 import coder
    tokenizer = (held or coder.get()).tokenizer
    # what the change is for, said before each window alike: the edits that
    # do it more likely after it than those that do not
    ahead = "".join(f"# {line}\n" for line in said.splitlines())
    # and code the window does not reach, as it is written ahead of it:
    # what the function is given (`class Bug: ... language: str`) -- out of
    # the window, `bug.language == 'python'` was the least likely field
    ahead = (given.rstrip() + "\n\n" if given.strip() else "") + ahead
    base = tokenizer(original)["input_ids"]
    windows, pairs = [], []
    for source in sources:
        ids = tokenizer(source)["input_ids"]
        first = next((at for at, (a, b) in enumerate(zip(base, ids))
                      if a != b), min(len(base), len(ids)))
        tail = next((at for at, (a, b) in enumerate(zip(base[::-1],
                                                         ids[::-1]))
                     if a != b or at >= min(len(base), len(ids)) - first),
                    0)
        start = max(0, first - BEFORE)
        mine = tokenizer.decode(ids[start:len(ids) - tail + AFTER])
        theirs = tokenizer.decode(base[start:len(base) - tail + AFTER])
        pairs.append((len(windows), len(windows) + 1))
        windows += [ahead + mine, ahead + theirs]
    # the original's windows repeat (one per edit of a place): scored once
    unique = list(dict.fromkeys(windows))
    scored = dict(zip(unique, scores(unique, held, tokens)))
    return [scored[windows[a]] - scored[windows[b]] for a, b in pairs]

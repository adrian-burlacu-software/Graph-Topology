"""How natural a piece of code is (v705): the code model's likelihood of
it, token by token, many pieces at once on the GPU -- to try the edits a
programmer would more likely have meant first, of the hundreds one-edit
repair makes (`repair.candidates`), before the tests judge any.

    scores(sources)   [log-likelihood per source], higher more natural

Each source is scored whole and against the code it was made from: what
counts is how much less likely the edit made it (`gained`), so a long
function is not unlike a short one for its length.
"""
from __future__ import annotations

#: sources scored at once
BATCH = 16
#: tokens of a source scored at most
LONGEST = 1024


def scores(sources: list) -> list:
    """The summed log-likelihood of each source under the code model."""
    import torch
    from research.v705 import coder
    model = coder.get()
    tokenizer, network = model.tokenizer, model.model
    out = []
    for at in range(0, len(sources), BATCH):
        batch = sources[at:at + BATCH]
        held = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=LONGEST).to("cuda")
        # padded on the left: each counted from its own first token
        positions = (held["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = network(**held, position_ids=positions).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = held["input_ids"][:, 1:]
        mask = held["attention_mask"][:, 1:].float()
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        out += (picked * mask).sum(dim=1).tolist()
    return out


def gained(original: str, sources: list) -> list:
    """How much more likely each source is than the code it was made from
    (negative: less likely)."""
    base = scores([original])[0]
    return [one - base for one in scores(sources)]

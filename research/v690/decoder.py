"""The decoder: a message said in English.

SmolLM2-360M-Instruct, fine-tuned by `teach_decoder.py` on the replies
SmolLM3 wrote that read back to their messages. At run time it is the only
thing that writes a reply, and nothing it writes is said before the encoder
has read it back (`roundtrip.py`, `speaking.py`).

    llm/decoder              the model (another with V690_DECODER_MODEL)
    llm/decoder/decoder.json what it was taught to be told, and the words
                             replies say around content
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from research.encoder import LLM

#: The decoder taught to say mathematics as well (`research/v692`,
#: `regenerate.py`'s `decoder-math`) where it exists, else the shipped one.
MODEL = Path(os.environ.get("V690_DECODER_MODEL") or (
    LLM / "decoder-maths" if (LLM / "decoder-maths" / "decoder.json").exists()
    else LLM / "decoder"))

#: The longest reply written, in tokens. At 96 what the page can do was cut
#: off mid-sentence in every one of its four replies.
LONGEST = 160


def enabled() -> bool:
    """Is there a decoder to say anything with?"""
    return (MODEL / "decoder.json").exists()


class Decoder:
    """The fine-tuned decoder, writing replies to messages in batches."""

    def __init__(self, path: Path = MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.config = json.loads((path / "decoder.json").read_text(
            encoding="utf-8"))
        self.device = device or os.environ.get("V690_DECODER_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.float16 if self.device == "cuda"
            else torch.float32)
        self.model.to(self.device)
        self.model.eval()
        self.lock = threading.Lock()

    @property
    def framing(self) -> dict:
        """The words replies say around content, for each stance
        (`teach_decoder.framing`)."""
        found = self.config.get("framing") or {}
        if isinstance(found, list):
            return {"": frozenset(found)}
        return {stance: frozenset(words) for stance, words in found.items()}

    def _encoded(self, prompts: list[str]):
        from .teach_decoder import encoded

        rows = [encoded(self.tokenizer, one)[0] for one in prompts]
        width = max(map(len, rows))
        pad = self.tokenizer.pad_token_id
        torch = self.torch
        ids = torch.tensor([[pad] * (width - len(one)) + one for one in rows],
                           device=self.device)
        mask = torch.tensor([[0] * (width - len(one)) + [1] * len(one)
                             for one in rows], device=self.device)
        return ids, mask

    def say(self, prompts: list[str], samples: int = 4,
            temperature: float = 0.7, greedy: bool = True) -> list[list[str]]:
        """For each message prompt, its replies: the likeliest first (when
        `greedy`), then sampled ones, each written once."""
        torch = self.torch
        if not prompts:
            return []
        ids, mask = self._encoded(prompts)
        out: list[list[str]] = [[] for _ in prompts]
        runs = []
        if greedy:
            runs.append((dict(do_sample=False), 1))
        if samples - int(greedy) > 0:
            runs.append((dict(do_sample=True, temperature=temperature,
                              top_p=0.95), samples - int(greedy)))
        pad = self.tokenizer.pad_token_id
        with self.lock, torch.no_grad():
            for settings, count in runs:
                made = self.model.generate(
                    input_ids=ids.repeat_interleave(count, 0),
                    attention_mask=mask.repeat_interleave(count, 0),
                    max_new_tokens=LONGEST, pad_token_id=pad, **settings)
                for index, tail in enumerate(made[:, ids.shape[1]:]):
                    text = self.tokenizer.decode(tail,
                                                 skip_special_tokens=True)
                    text = text.strip().split("\n")[0].strip()
                    if text and text not in out[index // count]:
                        out[index // count].append(text)
        return out


class _Loaded:
    def __init__(self) -> None:
        self.model: Decoder | None = None
        self.lock = threading.Lock()

    def get(self) -> Decoder:
        with self.lock:
            if self.model is None:
                if not enabled():
                    raise RuntimeError(
                        f"no decoder at {MODEL}: nothing can be said without "
                        f"it (python -m research.v690.teach_decoder train)")
                self.model = Decoder()
            return self.model


LOADED = _Loaded()

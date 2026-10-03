"""The decoder: from a request and its meaning, programs for the search.

SmolLM2-360M, taught (`teach_sketch.py`) to write, from the request and the
meaning the reader of meaning read in it, a program in the search's own
language. Nothing it writes is trusted: each proposal is read back into a
tree (`parse.py`) -- what does not parse is dropped -- and then it is a
candidate like any other. One that meets the examples is checked by the
round trip; one that nearly does is a near miss the search repairs
(backward error-correction swaps its subtrees), and its subtrees go into
the forward trie for the search to compose with. So a proposal is a
sketch: the parts the decoder got right are kept, the rest is searched.

    python -m research.v696.sketcher train [--no-meaning]
    python -m research.v696.sketcher evaluate --model sketcher

Models go into their own new `llm/` directories.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

from research.v696 import program as P
from research.v696.teach_sketch import (FUNCTIONS, SKETCHES, prompt,
                                        said_meaning)

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
BASE = LLM / "SmolLM2-360M-Instruct"
SAYING = ("You write TypeScript: one expression that the function "
          "returns, in the library's own words.")
#: Rung 3: the decoder writes the whole function, steps and loops as
#: people write them; what it writes is read into the tree all the same.
SAYING_FUNCTIONS = "You write TypeScript: the whole function asked for."

#: How many programs a request is given: sampled, plus one greedy.
SAMPLES = 8
#: What the sampling starts from.
SEED = 696
#: The proposer: taught again from what the widened reader reads (chosen on
#: dev; the untaught base model, level on dev and better on held, may have
#: seen HumanEval -- `DESIGN.md`, "After rung 5").
PROPOSER = "sketcher-functions2"
#: The proposers together, each asked where those before it wrote nothing
#: that meets the examples (`--sketcher proposers --rounds 2`).
PROPOSERS = "sketcher-functions2,sketcher-people,SmolLM2-360M-Instruct"
#: How often a source is repeated in training: MBPP's programs are the
#: only ones asked for by people, and are few beside the generated.
REPEAT = {"mbpp-ts": 8}


def _encoded(tokenizer, text: str, target: str | None = None,
             saying: str = SAYING):
    turns = [{"role": "system", "content": saying},
             {"role": "user", "content": text}]
    head = tokenizer.apply_chat_template(turns, tokenize=True,
                                         add_generation_prompt=True,
                                         return_dict=True)["input_ids"]
    if target is None:
        return head, len(head)
    tail = tokenizer(target + tokenizer.eos_token,
                     add_special_tokens=False)["input_ids"]
    return head + tail, len(head)


def _text(row: dict, meaning: bool) -> str:
    return prompt(row["english"], row["code"],
                  row["meaning"] if meaning else None)


def train(out: Path, meaning: bool = True, epochs: int = 4,
          batch: int = 16, rate: float = 1e-4, longest: int = 448,
          seed: int = 696, functions: bool = False,
          corpus: Path | None = None) -> None:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    saying = SAYING_FUNCTIONS if functions else SAYING
    if functions:
        longest = max(longest, 768)
    corpus = corpus or (FUNCTIONS if functions else SKETCHES)
    rows = [json.loads(line) for line in corpus.open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # A request too long to read whole (a long example) is left out, not
    # cut: a cut request asks for something else.
    rows = [one for one in rows if len(_encoded(
        tokenizer, _text(one, meaning), one["target"], saying)[0])
        <= longest]
    train_rows = [one for one in rows if one["split"] == "train"
                  for _ in range(REPEAT.get(one["source"], 1))]
    held = [one for one in rows if one["split"] == "dev"]
    model = AutoModelForCausalLM.from_pretrained(str(BASE),
                                                 dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.get_input_embeddings().weight.requires_grad_(False)
    model.to("cuda")

    def batches(chunk):
        ids, labels = [], []
        for one in chunk:
            tokens, start = _encoded(tokenizer, _text(one, meaning),
                                     one["target"], saying)
            ids.append(tokens)
            labels.append([-100] * min(start, len(tokens)) + tokens[start:])
        width = max(map(len, ids))
        pad = tokenizer.pad_token_id
        return (torch.tensor([one + [pad] * (width - len(one))
                              for one in ids], device="cuda"),
                torch.tensor([[1] * len(one) + [0] * (width - len(one))
                              for one in ids], device="cuda"),
                torch.tensor([one + [-100] * (width - len(one))
                              for one in labels], device="cuda"))

    def evaluate() -> float:
        model.eval()
        total = count = 0
        with torch.no_grad():
            for at in range(0, len(held), batch):
                ids, mask, labels = batches(held[at:at + batch])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    total += float(model(input_ids=ids, attention_mask=mask,
                                         labels=labels).loss) * len(ids)
                count += len(ids)
        model.train()
        return total / max(count, 1)

    trained = [one for one in model.parameters() if one.requires_grad]
    import bitsandbytes
    optimiser = bitsandbytes.optim.AdamW8bit(trained, lr=rate,
                                             weight_decay=0.01)
    steps = epochs * math.ceil(len(train_rows) / batch)
    schedule = get_cosine_schedule_with_warmup(optimiser, int(0.05 * steps),
                                               steps)
    print(f"{len(train_rows)} train, {len(held)} dev, {steps} steps, "
          f"meaning {'on' if meaning else 'off'}; dev loss before "
          f"{evaluate():.3f}", flush=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train_rows)
        total = 0.0
        for at in range(0, len(train_rows), batch):
            ids, mask, labels = batches(train_rows[at:at + batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=labels).loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}: train "
              f"{total / math.ceil(len(train_rows) / batch):.3f} dev "
              f"{evaluate():.3f} ({time.time() - started:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    (out / "sketcher.json").write_text(json.dumps({
        "base": BASE.name, "saying": saying, "meaning": meaning,
        "functions": functions, "corpus": corpus.name,
        "train": len(train_rows), "epochs": epochs}, indent=1),
        encoding="utf-8")
    print(f"-> {out}")


class Sketcher:
    def __init__(self, path: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = path.name
        # A model with no `sketcher.json` is one taught nothing here (the
        # base model as it came): asked for the whole function, given no
        # meaning line -- it was never taught to read one.
        said = path / "sketcher.json"
        self.settings = json.loads(said.read_text(encoding="utf-8")) \
            if said.exists() else {"saying": SAYING_FUNCTIONS,
                                   "meaning": False, "functions": True}
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    def write(self, texts: list, samples: int = SAMPLES,
              longest: int = 160, greedy: bool = True) -> list:
        """For each prompt, one greedy program (unless not `greedy`) and
        `samples` sampled."""
        torch = self.torch
        if self.settings.get("functions"):
            longest = max(longest, 360)
        heads = [_encoded(self.tokenizer, one,
                          saying=self.settings["saying"])[0]
                 for one in texts]
        width = max(map(len, heads))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(one)) + one
                            for one in heads], device="cuda")
        mask = torch.tensor([[0] * (width - len(one)) + [1] * len(one)
                             for one in heads], device="cuda")
        out = [[] for _ in texts]
        with torch.no_grad():
            first = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=longest,
                do_sample=False, pad_token_id=pad) if greedy else None
            sampled = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=longest,
                do_sample=True, temperature=0.8, top_p=0.95,
                num_return_sequences=samples, pad_token_id=pad)
        for index, row in enumerate(first[:, width:] if greedy else []):
            out[index].append(self.tokenizer.decode(
                row, skip_special_tokens=True).strip())
        for index, row in enumerate(sampled[:, width:]):
            out[index // samples].append(self.tokenizer.decode(
                row, skip_special_tokens=True).strip())
        return out


def _meets(spec, tree) -> bool:
    from research.v696.checker import CheckerError, checker
    from research.v696.search import _matches
    try:
        row = checker().values(spec.names, spec.cases, [tree.source()],
                               prelude=P.prelude([tree]))[0]
    except CheckerError:
        return False
    return _matches(row, spec.outputs)


def proposals(sketcher: Sketcher, specs: list, batch: int = 8,
              rounds: int = 1, keep: bool = False) -> None:
    """Each spec's proposals, read back into trees: `Spec.proposals`,
    distinct, in the order written (greedy first). The meaning, where the
    sketcher was taught with one, is what the reader read
    (`Spec.expected`, set by `reader.expect` first).

    With `rounds` > 1 the decoder is asked again, `SAMPLES` more each
    time, for the requests none of whose programs yet meets its examples:
    what is easy is written once, what is hard is tried more. With `keep`,
    what another decoder proposed is kept, and this one asked only where
    none of it meets the examples."""
    from research.v696.parse import parse
    from research.v696.reader import request
    texts = []
    for spec in specs:
        english, code = request(spec)
        meaning = None
        if sketcher.settings["meaning"]:
            meaning = said_meaning(spec.expected["probs"])
        texts.append(prompt(english, code, meaning))
        if not keep:
            spec.proposals, spec.sources = [], []
    # The risk matrix (`Spec.moves`, `risk.py`): where a request is deep, or
    # needs what the search's library lacks, or is open, it is asked once
    # more where nothing met (`ask`, `readings`); where it is open or a
    # wrong answer costly, every writer is asked, whatever met before it,
    # so that programs written apart can be compared (`readings`,
    # `four_eyes`) -- once each: what met is not asked again.
    def allowed(spec) -> int:
        more = spec.moves and ("ask" in spec.moves
                               or "readings" in spec.moves)
        return rounds + (1 if more else 0)

    def always(spec) -> bool:
        return bool(spec.moves) and ("readings" in spec.moves
                                     or "four_eyes" in spec.moves)

    def met(one) -> bool:
        return any(_meets(specs[one], tree) for tree in specs[one].proposals)

    todo = [one for one in range(len(specs))
            if always(specs[one]) or not (keep and met(one))]
    for round_ in range(max([rounds] + [allowed(one) for one in specs])):
        todo = [one for one in todo if round_ < allowed(specs[one])]
        for one in todo:
            # Each request sampled by a seed of its own and the round's:
            # what it is written does not hang on which others are asked
            # beside it (in a shared batch, asking one request again
            # changed every other's samples).
            # So it is the same every time, and kept (`CACHE`): a run
            # measured again asks the decoder nothing it was asked before.
            key = (f"{sketcher.name}|{specs[one].name}|{round_}|"
                   f"{zlib.crc32(texts[one].encode())}")
            written = _cached().get(key)
            if written is None:
                sketcher.torch.manual_seed(
                    SEED + zlib.crc32(specs[one].name.encode()) + round_)
                written = sketcher.write([texts[one]],
                                         greedy=round_ == 0)[0]
                _keep(key, written)
            _read(specs[one], written, parse)
        todo = [one for one in todo if not met(one)]
        if not todo:
            break


#: Where what the decoders wrote is kept, by decoder, request, round and
#: prompt (`proposals`); None keeps nothing. A file of the run's, never
#: taught from.
CACHE: Path | None = None
_CACHED: dict = {}


def _cached() -> dict:
    if CACHE is not None and not _CACHED and CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            row = json.loads(line)
            _CACHED[row["key"]] = row["written"]
    return _CACHED


def _keep(key: str, written: list) -> None:
    if CACHE is None:
        return
    _CACHED[key] = written
    with CACHE.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"key": key, "written": written}) + "\n")


def _runs(spec, tree) -> bool:
    """Whether the helpers a tree calls, printed, run at all."""
    from research.v696.checker import CheckerError, checker
    prelude = P.prelude([tree])
    if not prelude:
        return True
    try:
        row = checker().values(spec.names, spec.cases[:1], ["0"],
                               prelude=prelude)[0]
    except CheckerError:
        return False
    return not any(str(one.get("error", "")).startswith("helpers")
                   for one in row)


def _read(spec, written: list, parse) -> None:
    """What the decoder wrote, read into trees, added to the spec's."""
    seen = {one.source() for one in spec.proposals}
    for text in written:
        if "```" in text:
            # a reply in a code block: what is inside it
            inside = text.split("```")[1]
            text = inside.partition("\n")[2] or inside
        if "function " not in text:
            text = text.strip().rstrip(";")
            if text.startswith("return "):
                text = text[len("return "):]
            text = f"{spec.signature()} {{\n  return {text};\n}}\n"
        # a whole function, steps and loops as written
        tree = parse(text, spec.entry, spec.params)
        if tree is not None and not _runs(spec, tree):
            # its helpers, printed, do not run: not a program to offer
            tree = None
        if tree is not None and tree.source() not in seen:
            seen.add(tree.source())
            spec.proposals.append(tree)
            spec.sources.append(text)


def evaluate(model: str, meaning_model: str = "meaning-unixcoder") -> dict:
    """On MBPP dev and HumanEval held: how many requests get a proposal
    that parses, one that meets the examples, one that passes the tests --
    the decoder alone, before any search."""
    from research.v696 import experiment as E
    from research.v696 import reader as R
    from research.v696.checker import checker
    from research.v696.search import _matches
    from research.v696.teach_meaning import split
    out = {}
    suites = {"dev": [one for one in E.multipl_e("mbpp-ts")
                      if split(one.name) == "dev"],
              "held": E.multipl_e("humaneval-ts")}
    for part, specs in suites.items():
        R.expect(specs, R.LLM / meaning_model)
    sketcher = Sketcher(LLM / model)
    for part, specs in suites.items():
        started = time.time()
        proposals(sketcher, specs)
        parsed = met = passed = 0
        for spec in specs:
            parsed += bool(spec.proposals)
            rows = checker().values(spec.names, spec.cases,
                                    [one.source() for one in spec.proposals],
                                    prelude=P.prelude(spec.proposals)
                                    ) if spec.proposals else []
            meeting = [tree for tree, row in zip(spec.proposals, rows)
                       if _matches(row, spec.outputs)]
            met += bool(meeting)
            passed += any(checker().tests(spec.function(tree) + "\n"
                                          + spec.tests) is None
                          for tree in meeting[:3])
        out[part] = {"requests": len(specs), "a proposal parses": parsed,
                     "meets the examples": met, "passes the tests": passed,
                     "seconds": round(time.time() - started)}
        print(f"{part:5} {out[part]}", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("train", "evaluate"))
    parser.add_argument("--model", default="")
    parser.add_argument("--no-meaning", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--functions", action="store_true",
                        help="teach whole functions (rung 3)")
    parser.add_argument("--corpus", default="",
                        help="the corpus file's name in data/code-meaning, "
                             "where it is not the first made")
    args = parser.parse_args(argv)
    if args.job == "train":
        name = args.model or ("sketcher" if not args.no_meaning
                              else "sketcher-no-meaning")
        train(LLM / name, meaning=not args.no_meaning, epochs=args.epochs,
              functions=args.functions,
              corpus=FUNCTIONS.parent / args.corpus if args.corpus else None)
    else:
        evaluate(args.model or PROPOSER)
    return 0


if __name__ == "__main__":
    sys.exit(main())

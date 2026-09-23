"""The decoder's teacher: what a reply to each message says, round-tripped.

    python -m research.v690.teach_decoder replies --limit 200
    python -m research.v690.teach_decoder label
    python -m research.v690.teach_decoder train

**Replies.** SmolLM3-3B, offline and never at run time, is shown a message
(`message.prompt`) and writes the reply to it, a few times over: once greedy
and the rest sampled. It is told the stance, the subject and what is claimed
of it, the values, what the answer rests on and v689's own account, and to
say those in plain English and nothing else. The model is good at the saying
and not trusted with the content, so

**labels** keeps a reply only when it reads back to its message
(`roundtrip.py`): its stance, every phrase the message requires, and no
content word the message does not have -- save the words replies say
around any content (`you told me`, `as far as I know`), counted over the
corpus rather than listed. What is kept teaches both directions: the decoder
to write the reply from the message, and the encoder to read the message's
parts back out of the reply.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from research.encoder import LLM

from .conversations import DATA, TURNS
from .message import Message, of_turn, prompt

TEACHER = LLM / "SmolLM3-3B"
REPLIES = DATA / "replies.jsonl"

SYSTEM = """Write a reasoning system's reply to the user, in plain English.
You get what it heard, its stance, what the answer is about, the claim or \
values, and its internal account ("found"). Keep the stance. Name the \
subject and claim, or the values, in the message's words. Say briefly what \
it rests on. Never say internals (R3, T4, v688, VERIFIED, capable_of, \
dog.n.01). Never add a fact, reason or example the message lacks. Say "I" \
and "you". One or two short sentences."""

#: What the model is shown before each message: one of most stances, each
#: account cut short -- every row of a batch carries these, and at 1,200
#: tokens of them a batch spilled out of the card.
SHOWN = [
    ("heard: can it swim\nstance: yes\nabout: the beagle\nclaim: the beagle "
     "can swim\nsource: kind (corroborated)\nfound: yes — nothing was told "
     "of the beagle; v688 answers “can a beagle swim” verified",
     "Yes, the beagle can swim. You haven't told me anything about it, and "
     "beagles can swim."),
    ("heard: can the first one swim\nstance: no\nabout: the first beagle\n"
     "claim: the first beagle can swim\nquotes: it can't swim\nsource: told\n"
     "found: no — you told me so: “it can't swim”. R3",
     "No, the first beagle can't swim. You told me so."),
    ("heard: is the dog hungry\nstance: unknown\nabout: the dog\nclaim: the "
     "dog is hungry\nquotes: it was hungry\nsource: told\nfound: not told of "
     "today — yesterday, you told me “it was hungry”",
     "I don't know if the dog is hungry today. You only told me it was "
     "hungry yesterday."),
    ("heard: it can't swim\nstance: noted\nabout: the beagle\nclaim: the "
     "beagle can't swim\nsource: told\nfound: noted — stored not_capable_of "
     "“swim”",
     "Got it: the beagle can't swim."),
    ("heard: where is Mary\nstance: value\nabout: Mary\nvalues: the kitchen\n"
     "quotes: Mary went to the kitchen\nsource: told\nfound: Mary: the "
     "kitchen — “Mary went to the kitchen” changed it (T4)",
     "Mary is in the kitchen. You told me she went there."),
    ("heard: does the beagle bark\nstance: which\nabout: the beagle\nfound: "
     "which one — the second beagle or the first beagle?",
     "Which beagle do you mean, the first one or the second one?"),
    ("heard: hello\nstance: social\nact: greet\nfound: greeting",
     "Hello! Tell me about something, or ask me a question."),
]


def chat(message_prompt: str) -> list[dict]:
    turns = [{"role": "system", "content": SYSTEM}]
    for shown, reply in SHOWN:
        turns += [{"role": "user", "content": shown},
                  {"role": "assistant", "content": reply}]
    return turns + [{"role": "user", "content": message_prompt}]


#: At most this many messages of a stance are replied to; the rest all are.
#: Half of what conversations say is a statement noted.
CAPS = {"noted": 1500, "unknown": 1500, "value": 1800}

#: How many of the social phrases are replied to.
SOCIAL = 320


def social_messages(seed: int = 690) -> list[tuple[str, Message]]:
    """What the social acts are answered with (`v689/social.py`), for every
    way of saying one the reader is taught (`teach_reader.social_texts`)."""
    from research.v689 import social
    from research.v689.teach_reader import social_texts

    # The relations the operators answer are the ones `social.RELATIONS`
    # says, in its order (`Session._social` passes `Answering.cells()`).
    relations = list(social.RELATIONS)
    rng = random.Random(seed)
    texts = social_texts()
    rng.shuffle(texts)
    found = []
    for index, text in enumerate(texts[:SOCIAL]):
        act = social.act_of(text.replace(",", " ").split())
        if not act:
            continue
        turn = {"said": text, "act": act, "reading": {"act": act},
                "answer": {"outcome": "social", "source": "conversation",
                           "act": act,
                           "text": social.answer(act, relations)}}
        found.append((f"social-{index}:0", of_turn(turn)))
    return found


def messages_from(path: Path = TURNS, limit: int = 0,
                  seed: int = 690, caps: dict | None = None,
                  social: bool = True) -> list[tuple[str, Message]]:
    """(key, message) for every turn kept, one per distinct prompt, each
    stance up to its cap, and the social acts beside them."""
    caps = CAPS if caps is None else caps
    found = _messages(path, seed)
    counted: dict = {}
    kept = []
    for key, one in found:
        counted[one.stance] = counted.get(one.stance, 0) + 1
        if counted[one.stance] <= caps.get(one.stance, len(found)):
            kept.append((key, one))
    if social:
        kept += social_messages(seed)
    random.Random(seed + 1).shuffle(kept)
    return kept[:limit] if limit else kept


def _messages(path: Path, seed: int) -> list[tuple[str, Message]]:
    seen, found = set(), []
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle):
            if not raw.strip():
                continue
            turn = json.loads(raw)
            if turn.get("error") or not turn.get("answer"):
                continue
            message = of_turn(turn)
            shown = prompt(message)
            if shown in seen:
                continue
            seen.add(shown)
            conversation = turn.get("conversation",
                                    turn.get("_conversation", "probe"))
            found.append((f"{conversation}:{turn.get('index', number)}",
                          message))
    random.Random(seed).shuffle(found)
    return found


class Writer:
    """SmolLM3, writing replies in batches.

    Every prompt opens with the same system prompt and examples, about 1,500
    tokens of 1,700. They are read once (`prefix`), and each batch starts
    from a copy of what reading them left, with each message padded on the
    left of its own words, after the shared ones."""

    def __init__(self, path: Path = TEACHER, quantized: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        extra = {}
        if quantized:
            # 4-bit weights leave the rest of an 8 GB card to the batch:
            # at 16 bits the model is 6.2 GB and every batch spilled into
            # shared memory.
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.float16, device_map="cuda", **extra)
        self.model.eval()
        self.prefix, self.cache = self._prefix()
        print(f"teacher on {path.name}{' in 4 bits' if quantized else ''}: "
              f"{len(self.prefix)} tokens shared by every prompt", flush=True)

    def _ids(self, message_prompt: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            chat(message_prompt), tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True)["input_ids"]

    def _prefix(self):
        """The tokens every prompt opens with, and the cache they leave."""
        torch = self.torch
        one, other = self._ids("heard: a"), self._ids("stance: b\nfound: c")
        size = next((at for at, (x, y) in enumerate(zip(one, other))
                     if x != y), min(len(one), len(other)))
        prefix = one[:size]
        with torch.no_grad():
            out = self.model(torch.tensor([prefix], device="cuda"),
                             use_cache=True)
        return prefix, out.past_key_values

    def write(self, prompts: list[str], samples: int = 2,
              temperature: float = 0.8, longest: int = 64,
              greedy: bool = False) -> list[list[str]]:
        """For each prompt, `samples` replies: sampled, or one greedy."""
        import copy

        torch = self.torch
        size = len(self.prefix)
        tails = []
        for one in prompts:
            ids = self._ids(one)
            assert ids[:size] == self.prefix, "a prompt without the prefix"
            tails.append(ids[size:])
        width = max(len(one) for one in tails)
        pad = self.tokenizer.pad_token_id
        rows = [self.prefix + [pad] * (width - len(one)) + one
                for one in tails]
        mask = [[1] * size + [0] * (width - len(one)) + [1] * len(one)
                for one in tails]
        count = 1 if greedy else samples
        cache = copy.deepcopy(self.cache)
        cache.batch_repeat_interleave(len(prompts) * count)
        ids = torch.tensor(rows, device="cuda").repeat_interleave(count, 0)
        attention = torch.tensor(mask, device="cuda").repeat_interleave(
            count, 0)
        settings = (dict(do_sample=False) if greedy else
                    dict(do_sample=True, temperature=temperature, top_p=0.95))
        with torch.no_grad():
            made = self.model.generate(
                input_ids=ids, attention_mask=attention,
                past_key_values=cache, max_new_tokens=longest,
                pad_token_id=pad, **settings)
        out: list[list[str]] = [[] for _ in prompts]
        for index, tail in enumerate(made[:, ids.shape[1]:]):
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
            out[index // count].append(text.strip().split("\n")[0].strip())
        return out


def _kept(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(raw)["prompt"] for raw in
            path.read_text(encoding="utf-8").splitlines() if raw.strip()}


def replies(limit: int, batch: int, samples: int, path: Path = REPLIES,
            turns: Path = TURNS) -> None:
    found = messages_from(turns, limit)
    kept = _kept(path)
    todo = [(key, one) for key, one in found if prompt(one) not in kept]
    print(f"{len(found)} messages, {len(kept)} replied to, {len(todo)} to "
          f"go", flush=True)
    import os

    writer = Writer(quantized=os.environ.get("V690_TEACHER_4BIT") == "1")
    started = time.time()
    with path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            shown = [prompt(one) for _, one in chunk]
            # One reply a message is the likeliest; more are sampled.
            written = writer.write(shown, samples, greedy=samples == 1)
            for (key, one), said, text in zip(chunk, written, shown):
                sink.write(json.dumps({"key": key, "prompt": text,
                                       "message": one.as_dict(),
                                       "replies": said}) + "\n")
            sink.flush()
            done = start + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            print(f"  {done}/{len(todo)} ({rate:.2f} messages/s, "
                  f"{(len(todo) - done) / max(rate, 1e-6) / 60:.0f} min "
                  f"left)", flush=True)


# -- round trip ---------------------------------------------------------------------
#: A word replies of a stance say around content is one said without a
#: message having it in replies to at least this share of that stance's
#: messages, and at least this many of them (a noun, twice as many): `got`
#: in `got it`, `welcome` -- not `water`, said of swimming.
FRAMING_SHARE = 0.02
FRAMING_LEAST = 5

READER_DATA = LLM / "reader-data"
DECODER = LLM / "decoder"
#: Other subjects' message and reply pairs, each in a folder of its own that
#: nothing here writes to -- taught only when asked for (`--subject math`),
#: so that a rebuild of the shipped decoder is the decoder that shipped.
SUBJECT_DATA = {"math": LLM / "math-decoder-data"}
SUBJECTS: list = []
BASE = LLM / "SmolLM2-360M-Instruct"


def _split(key: str) -> str:
    """A conversation's messages go to one split: every tenth held out."""
    import zlib

    conversation = key.rsplit(":", 1)[0]
    return "valid" if zlib.crc32(conversation.encode()) % 10 == 0 else "train"


#: Processes that read replies at once, each with its own spaCy.
PROCESSES = 12

_WORDS = None


def _words():
    """This process's spaCy over reply words (`roundtrip.Words`)."""
    global _WORDS
    if _WORDS is None:
        from .roundtrip import Words
        _WORDS = Words()
    return _WORDS


def _said_around(row: dict) -> tuple:
    """(stance, the content lemmas a message's replies say around what they
    say that the message lacks, {lemma: (times as a noun, times said)})."""
    from .roundtrip import content_at, label as parts_of, vocabulary

    words = _words()
    message = Message.from_dict(row["message"])
    known = vocabulary(message, words)
    said: set = set()
    counts: dict = {}
    for reply in row["replies"]:
        if not reply:
            continue
        read = parts_of(message, reply, words)
        for word, part, tag, lemma in zip(read.words, read.parts, read.tags,
                                          read.lemmas):
            if part == "O" and content_at(word, tag, lemma):
                said.add(lemma)
                noun, seen = counts.get(lemma, (0, 0))
                counts[lemma] = (noun + tag.startswith("NN"), seen + 1)
    return message.stance, sorted(said - known), counts


def framing(rows: list[dict], words=None, pool=None) -> tuple[dict, dict]:
    """The words replies say around content, for each stance: a content word
    replies of that stance say outside what they were told to say -- `got`
    in `got it` of a statement noted, `welcome` of thanks, `include` of a
    listing -- in at least `FRAMING_SHARE` of that stance's messages, and a
    noun in twice that, since a noun is likelier to be content made up.
    Nothing the teacher was shown in its examples is one: that is content it
    can copy, `beagle` said of a pig. ({stance: words}, the counts)."""
    import collections

    global _WORDS
    if words is not None:
        _WORDS = words
    messages: dict = collections.defaultdict(collections.Counter)
    nouns: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    by_stance: collections.Counter = collections.Counter()
    read = (pool.imap(_said_around, rows, chunksize=32) if pool is not None
            else map(_said_around, rows))
    for stance, lemmas, counts in read:
        by_stance[stance] += 1
        for lemma in lemmas:
            messages[stance][lemma] += 1
        for lemma, (noun, times) in counts.items():
            nouns[lemma] += noun
            seen[lemma] += times
    shown: set = set()
    for example, _ in SHOWN:
        for line in example.splitlines():
            field, _, value = line.partition(": ")
            if field in ("heard", "about", "claim", "values", "quotes"):
                shown.update(_words().content(value))
    kept: dict = {}
    for stance, counts in messages.items():
        least = max(FRAMING_LEAST, FRAMING_SHARE * by_stance[stance])
        kept[stance] = sorted(
            lemma for lemma, count in counts.items()
            if lemma not in shown
            and count >= least * (2 if nouns[lemma] * 2 > seen[lemma] else 1))
    return kept, {stance: counts.most_common(60)
                  for stance, counts in messages.items()}


def _rows(paths) -> list[dict]:
    """Every message replied to, in any of `paths`, with every reply any of
    them wrote to it."""
    merged: dict = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            kept = merged.setdefault(row["prompt"], {**row, "replies": []})
            kept["replies"] += [one for one in row["replies"]
                                if one not in kept["replies"]]
    return list(merged.values())


def _checked(job: tuple) -> dict:
    """One message's replies, each labelled by the message and traced
    against it: what a traced one teaches the encoder and the decoder, and
    why each other one was not kept."""
    from .roundtrip import label as parts_of, trace

    row, kept_framing = job
    words = _words()
    message = Message.from_dict(row["message"])
    out = {"split": _split(row["key"]), "stance": message.stance,
           "reasons": [], "reader": [], "pairs": []}
    for reply in dict.fromkeys(row["replies"]):
        if not reply:
            out["reasons"].append("empty")
            continue
        read = parts_of(message, reply, words)
        found = trace(message, read, words, kept_framing, reply)
        if not found.traced:
            out["reasons"] += [f"{message.stance} {why}" for why in
                               ("missing", "added", "denial", "internal",
                                "source", "unfinished")
                               if getattr(found, why)]
            continue
        _, _, deps, _ = words.read(reply)
        out["reader"].append({
            "task": "reply", "words": read.words, "tags": read.tags,
            "deps": deps, "stance": message.stance, "parts": read.parts,
            "source": row["key"], "how": "reply"})
        out["pairs"].append({"key": row["key"], "stance": message.stance,
                             "prompt": prompt(message), "reply": reply})
        # And the same reply with its denial turned, labelled as its words
        # are, for the encoder only: kept replies almost never deny in a
        # sentence of their own (`You told me he did not.`), so the encoder
        # never read one as denying.
        for variant in _turned(message, reply, row["key"]):
            turned = parts_of(message, variant, words)
            _, _, turned_deps, _ = words.read(variant)
            out["reader"].append({
                "task": "reply", "words": turned.words, "tags": turned.tags,
                "deps": turned_deps, "stance": message.stance,
                "parts": turned.parts, "source": row["key"],
                "how": "turned"})
    return out


#: The sentences a denial is added in, said after a reply's claim. The
#: evaluation adds its denials another way (`evaluate.corruptions`), so what
#: it measures is not what the encoder was shown.
DENYING = ("That is not so.", "It did not.", "They did not.",
           "She did not.", "He did not.", "It isn't.")

#: One reply in this many, of a yes, a no or a noted statement, is turned.
TURNED = 3


def _turned(message: Message, reply: str, key: str) -> list[str]:
    """A reply with its denial taken out, and with one added in a sentence
    of its own: some of the replies of a claim's stance."""
    import zlib

    from .evaluate import undenied

    if message.stance not in ("yes", "no", "noted") or not message.claim:
        return []
    seed = zlib.crc32((key + reply).encode())
    if seed % TURNED:
        return []
    import re

    found = []
    dropped = undenied(reply)
    if dropped is not None:
        found.append(dropped)
    # Said right after the sentence that says the claim, which is the
    # sentence a denial of it is read in (`roundtrip.denying`); at the end of
    # the reply, after what was quoted, it would deny the quote.
    claimed = {word for word in re.findall(r"[a-z]+", message.claim.lower())
               if len(word) > 3}
    pieces = re.split(r"(?<=[.!?;:])\s+", reply.strip())
    at = next((index for index, piece in enumerate(pieces)
               if claimed & set(re.findall(r"[a-z]+", piece.lower()))), None)
    if at is not None:
        pieces.insert(at + 1, DENYING[seed % len(DENYING)])
        found.append(" ".join(pieces))
    return found


def label(paths=None, processes: int = PROCESSES) -> None:
    """Keep each reply that reads back to its message, and write what it
    teaches: `reader-data/{train,valid}-reply.jsonl` for the encoder, and
    `decoder-data/{train,valid}.jsonl` for the decoder. The replies are the
    teacher's (`replies.jsonl`) and the decoder's own (`replies-decoder.jsonl`,
    `bootstrap`), read by `processes` at once."""
    import collections
    import multiprocessing

    rows = _rows(paths or [REPLIES, BOOTSTRAPPED])
    # Each message taken apart again from its turn, as `message.py` takes
    # turns apart now: the replies were written to it as it was then, and a
    # field that changed since changes the prompt, not what was said.
    turns: dict = {}
    if TURNS.exists():
        with TURNS.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle):
                if raw.strip():
                    turn = json.loads(raw)
                    conversation = turn.get("conversation", "probe")
                    turns[f"{conversation}:{turn.get('index', number)}"] = turn
    for row in rows:
        turn = turns.get(row["key"])
        if turn is not None and turn.get("answer"):
            message = of_turn(turn)
            row["message"], row["prompt"] = message.as_dict(), prompt(message)
    turns.clear()
    reasons: collections.Counter = collections.Counter()
    by_stance: collections.Counter = collections.Counter()
    READER_DATA.mkdir(parents=True, exist_ok=True)
    sinks = {name: open(READER_DATA / f"{name}-reply.jsonl", "w",
                        encoding="utf-8") for name in ("train", "valid")}
    pairs = {name: open(DATA / f"{name}.jsonl", "w", encoding="utf-8")
             for name in ("train", "valid")}
    started = time.time()
    try:
        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            kept_framing, counted = framing(rows, pool=pool)
            print(f"{len(rows)} messages ({time.time() - started:.0f}s); "
                  f"framing words by stance: {json.dumps(kept_framing)}",
                  flush=True)
            jobs = ((row, kept_framing) for row in rows)
            for done in pool.imap(_checked, jobs, chunksize=32):
                reasons.update(done["reasons"])
                for record in done["reader"]:
                    sinks[done["split"]].write(json.dumps(record) + "\n")
                for record in done["pairs"]:
                    pairs[done["split"]].write(json.dumps(record) + "\n")
                traced = bool(done["pairs"])
                reasons["traced" if traced else "none traced"] += 1
                by_stance[(done["stance"], traced)] += 1
    finally:
        for one in list(sinks.values()) + list(pairs.values()):
            one.close()
    print(f"read back in {time.time() - started:.0f}s", flush=True)
    stats = {"messages": len(rows), "reasons": dict(reasons),
             "by stance": {f"{stance} {'traced' if ok else 'none'}": count
                           for (stance, ok), count in sorted(
                               by_stance.items())},
             "framing": kept_framing, "said around": counted}
    (DATA / "label-stats.json").write_text(json.dumps(stats, indent=1),
                                           encoding="utf-8")
    (DATA / "framing.json").write_text(json.dumps(kept_framing, indent=1),
                                       encoding="utf-8")
    print(json.dumps({key: stats[key] for key in
                      ("messages", "reasons", "by stance")}, indent=1))


# -- the decoder --------------------------------------------------------------------
#: What the decoder is told before every message.
SAYING = ("Say the answer in the message to the user, in plain English: its "
          "stance, its subject and claim or its values, and what it rests "
          "on. Nothing internal, nothing added.")


#: At most this many replies a message are taught; noted statements are drawn
#: down to this many pairs; a stance with fewer pairs than `RARE_LEAST` is
#: said again until it has about that many, never more than `RARE_TIMES`
#: times over.
MOST_REPLIES = 2
MOST_NOTED = 3000
RARE_LEAST = 400
RARE_TIMES = 4


def balanced(rows: list[dict], rng: random.Random) -> list[dict]:
    """The decoder's pairs, balanced by stance. Most of what conversations
    say is a statement noted, and the decoder's own replies to them came four
    at a time: 59% of the kept pairs were `Got it`, and ten were `which
    one?`. Ten replies said over forty times would be learned by heart, so a
    rare stance is repeated at most `RARE_TIMES` times."""
    import collections
    import math

    by_key: dict = {}
    for one in rows:
        by_key.setdefault(one["key"], []).append(one)
    kept = [one for group in by_key.values() for one in group[:MOST_REPLIES]]
    noted = [at for at, one in enumerate(kept) if one["stance"] == "noted"]
    if len(noted) > MOST_NOTED:
        dropped = set(noted) - set(rng.sample(noted, MOST_NOTED))
        kept = [one for at, one in enumerate(kept) if at not in dropped]
    counts = collections.Counter(one["stance"] for one in kept)
    return [one for one in kept for _ in range(
        min(RARE_TIMES, max(1, math.ceil(RARE_LEAST / counts[one["stance"]]))))]


def encoded(tokenizer, message_prompt: str, reply: str | None = None):
    """(ids, where the reply starts): the chat as the decoder reads it, with
    the reply after it when there is one."""
    turns = [{"role": "system", "content": SAYING},
             {"role": "user", "content": message_prompt}]
    head = tokenizer.apply_chat_template(turns, tokenize=True,
                                         add_generation_prompt=True,
                                         return_dict=True)["input_ids"]
    if reply is None:
        return head, len(head)
    tail = tokenizer(reply + tokenizer.eos_token,
                     add_special_tokens=False)["input_ids"]
    return head + tail, len(head)


def train(base: Path = BASE, out: Path = DECODER, epochs: int = 3,
          batch: int = 16, rate: float = 1e-4, longest: int = 384,
          seed: int = 690) -> None:
    """The decoder, from `base`, taught every kept reply: the loss on the
    reply's tokens only. Its embeddings, which it shares with its output,
    stay as they were."""
    import math

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    def load(name: str) -> list[dict]:
        # And every other subject's pairs, each in its own folder beside
        # this one (`research/v692/speaking.py`: mathematics).
        paths = [DATA / f"{name}.jsonl"] + sorted(
            one for folder in _subject_folders()
            for one in folder.glob(f"{name}-*.jsonl"))
        return [json.loads(raw) for path in paths
                for raw in path.read_text(encoding="utf-8").splitlines()
                if raw.strip()]
    rows, held = load("train"), load("valid")
    # Balanced by stance (`balanced`); what is held out stays as it was, so
    # losses compare with the decoder before.
    rows = balanced(rows, rng)
    stances: dict = {}
    for one in rows:
        stances[one["stance"]] = stances.get(one["stance"], 0) + 1
    print(f"balanced: {stances}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(base),
                                                 dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.get_input_embeddings().weight.requires_grad_(False)
    model.to("cuda")

    def batches(chunk: list[dict]):
        ids, labels = [], []
        for one in chunk:
            tokens, start = encoded(tokenizer, one["prompt"], one["reply"])
            tokens = tokens[:longest]
            ids.append(tokens)
            labels.append([-100] * min(start, len(tokens))
                          + tokens[start:])
        width = max(map(len, ids))
        pad = tokenizer.pad_token_id
        return (torch.tensor([one + [pad] * (width - len(one)) for one in ids],
                             device="cuda"),
                torch.tensor([[1] * len(one) + [0] * (width - len(one))
                              for one in ids], device="cuda"),
                torch.tensor([one + [-100] * (width - len(one))
                              for one in labels], device="cuda"))

    def evaluate() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(held), batch):
                ids, mask, labels = batches(held[start:start + batch])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask,
                                 labels=labels).loss
                total += float(loss) * len(ids)
                count += len(ids)
        model.train()
        return total / max(count, 1)

    trained = [one for one in model.parameters() if one.requires_grad]
    # Adam's two states at 32 bits are 2.9 GB of a 360M model: with the
    # weights and their gradients the card spilled into shared memory. At 8
    # bits they are a quarter of that.
    try:
        import bitsandbytes

        optimiser = bitsandbytes.optim.AdamW8bit(trained, lr=rate,
                                                 weight_decay=0.01)
    except Exception:                               # noqa: BLE001
        optimiser = torch.optim.AdamW(trained, lr=rate, weight_decay=0.01)
    print(f"optimiser {type(optimiser).__name__}", flush=True)
    steps = epochs * math.ceil(len(rows) / batch)
    schedule = get_cosine_schedule_with_warmup(optimiser, int(0.05 * steps),
                                               steps)
    print(f"{len(rows)} train, {len(held)} valid, {steps} steps; valid "
          f"loss before {evaluate():.3f}", flush=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(rows)
        total = 0.0
        for start in range(0, len(rows), batch):
            ids, mask, labels = batches(rows[start:start + batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=labels).loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            if (start // batch) % 200 == 0:
                print(f"  epoch {epoch + 1} step {start // batch}: loss "
                      f"{float(loss):.3f} ({time.time() - started:.0f}s)",
                      flush=True)
        print(f"epoch {epoch + 1}: train {total / math.ceil(len(rows) / batch):.3f}"
              f" valid {evaluate():.3f} ({time.time() - started:.0f}s)",
              flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    framing_path = DATA / "framing.json"
    (out / "decoder.json").write_text(json.dumps({
        "base": str(base), "saying": SAYING, "train": len(rows),
        "valid": len(held), "epochs": epochs,
        # What else it was taught to say (`SUBJECTS`): a layer that speaks
        # for itself until a decoder knows its subject asks this.
        "subjects": [folder.name.replace("-decoder-data", "")
                     for folder in _subject_folders()],
        "framing": json.loads(framing_path.read_text(encoding="utf-8"))
        if framing_path.exists() else []}, indent=1), encoding="utf-8")
    print(f"saved to {out}")


def _subject_folders() -> list:
    return [SUBJECT_DATA[one] for one in SUBJECTS
            if SUBJECT_DATA[one].exists()]


# -- the decoder teaching itself ------------------------------------------------
#: The decoder's own replies to every message.
BOOTSTRAPPED = DATA / "replies-decoder.jsonl"


def bootstrap(samples: int = 4, batch: int = 24, temperature: float = 0.9,
              path: Path = BOOTSTRAPPED, turns: Path = TURNS,
              limit: int = 0) -> None:
    """The decoder's own replies to every message, uncapped -- the teacher
    replied to a few thousand, and was slow -- written as the teacher's are,
    so `label` keeps those that read back beside the teacher's: a decoder
    taught on them has seen every shape of message, and no reply it was
    taught says what its message does not."""
    from .decoder import Decoder

    found = messages_from(turns, limit, caps={}, social=True)
    kept = _kept(path) | _kept(REPLIES)
    todo = [(key, one) for key, one in found if prompt(one) not in kept]
    print(f"{len(found)} messages, {len(todo)} to reply to", flush=True)
    decoder = Decoder()
    started = time.time()
    with path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            shown = [prompt(one) for _, one in chunk]
            written = decoder.say(shown, samples, temperature=temperature)
            for (key, one), said, text in zip(chunk, written, shown):
                sink.write(json.dumps({"key": key, "prompt": text,
                                       "message": one.as_dict(),
                                       "replies": said, "by": "decoder"})
                           + "\n")
            sink.flush()
            done = start + len(chunk)
            if (start // batch) % 20 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-6)
                print(f"  {done}/{len(todo)} ({rate:.1f} messages/s)",
                      flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("replies", "label", "train",
                                         "bootstrap"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--turns", type=Path, default=TURNS)
    parser.add_argument("--out", type=Path, default=REPLIES)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--save", type=Path, default=DECODER,
                        help="where the trained decoder goes")
    parser.add_argument("--subject", action="append", default=[],
                        choices=sorted(SUBJECT_DATA),
                        help="also teach a subject's pairs (its folder)")
    options = parser.parse_args(argv)
    SUBJECTS[:] = options.subject
    if options.what == "replies":
        replies(options.limit, options.batch, options.samples, options.out,
                options.turns)
    elif options.what == "label":
        label()
    elif options.what == "bootstrap":
        bootstrap(options.samples, options.batch, limit=options.limit)
    else:
        train(out=options.save, epochs=options.epochs, rate=options.rate)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

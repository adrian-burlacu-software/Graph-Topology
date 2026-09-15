"""How well v690 says what it means, on messages it was not taught.

    python -m research.v690.evaluate --limit 600

Every message here is from the held-out conversations (`teach_decoder._split`:
one conversation in ten), so neither the decoder nor the encoder's reply
heads saw it or a reply to it.

    said          the decoder's likeliest reply reads back to its message, and
                  any of its first four does (`speaking.FIRST`), or one of
                  them with a sentence left out (`speaking.shortened`)
    stance        the encoder reads a reply's stance as the message's
    agrees        the encoder's reading of a reply traces exactly when the
                  teacher's labels do: the round trip at run time is the one
                  the corpus was kept by
    caught        a reply that reads back, corrupted one way at a time, no
                  longer does: a value swapped for another message's, a
                  denial dropped or added, a word nothing said, a rule named

By stance, since a yes and a listing are different things to say.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

from .conversations import TURNS
from .message import Message, prompt
from .teach_decoder import DATA, _split, messages_from

OUT = DATA / "evaluation.json"


#: A denial as written, and what saying it without the denial is.
DENIED = re.compile(r"\b(\w+n't|cannot|not|never)\b\s*", re.IGNORECASE)

#: An auxiliary a denial can be added after.
AUXILIARY = re.compile(r"\b(can|is|does|was|did|has|are)\b", re.IGNORECASE)


def undenied(text: str) -> str | None:
    """The text with its first denial taken out, punctuation and all as it
    was: `can't` as `can`, `not` and `never` left out. None without one."""
    from research.v689.reading import CONTRACTIONS

    found = DENIED.search(text)
    if found is None:
        return None
    word = found.group(1).lower()
    kept = (CONTRACTIONS[word][0] if word in CONTRACTIONS
            and CONTRACTIONS[word][-1] == "not" else "")
    trailing = found.group(0)[len(found.group(1)):]
    return (text[:found.start()] + (kept + trailing if kept else "")
            + text[found.end():])


def corruptions(message: Message, text: str, others: list[Message],
                rng: random.Random) -> dict[str, str]:
    """The same reply, wrong one way at a time, written as the reply was
    written -- its sentences kept, which the denial check reads."""
    found: dict[str, str] = {}
    values = [one for one in message.values if one and one in text]
    donors = [value for other in others for value in other.values
              if value and value not in message.values]
    if values and donors:
        found["value swapped"] = text.replace(values[0], rng.choice(donors), 1)
    dropped = undenied(text)
    if dropped is not None:
        found["denial dropped"] = dropped
    elif message.claim and message.stance in ("yes", "noted"):
        auxiliary = AUXILIARY.search(text)
        if auxiliary is not None:
            found["denial added"] = (text[:auxiliary.end()] + " not"
                                     + text[auxiliary.end():])
    found["word added"] = (text.rstrip(".!") + ", because "
                           + rng.choice(("they love water", "it is very old",
                                         "the garden is large",
                                         "a doctor said so")) + ".")
    found["rule named"] = text.rstrip(".!") + " (R3)."
    return found


def evaluate(limit: int, seed: int = 690) -> dict:
    from . import decoder as decoders
    from .roundtrip import Words, label, reading_of, trace
    from .speaking import FIRST, shortened

    rng = random.Random(seed)
    found = [(key, one) for key, one in messages_from(TURNS, caps={},
                                                      social=True)
             if _split(key) == "valid"]
    rng.shuffle(found)
    found = found[:limit]
    decoder = decoders.LOADED.get()
    framing = decoder.framing
    words = Words()
    counts: dict = collections.defaultdict(collections.Counter)
    examples: dict = collections.defaultdict(list)
    batch = 16
    for start in range(0, len(found), batch):
        chunk = found[start:start + batch]
        written = decoder.say([prompt(one) for _, one in chunk], FIRST)
        others = [one for _, one in chunk]
        for (key, message), replies in zip(chunk, written):
            tally = counts[message.stance]
            tally["messages"] += 1
            traced = []
            for rank, text in enumerate(replies):
                read = reading_of(text, words)
                checked = trace(message, read, words, framing, text)
                gold = trace(message, label(message, text, words), words,
                             framing, text)
                tally["replies"] += 1
                tally["stance read"] += read.stance == message.stance
                tally["agrees"] += checked.traced == gold.traced
                if checked.traced:
                    traced.append(text)
                if rank == 0:
                    tally["first traced"] += checked.traced
                    if len(examples[message.stance]) < 6:
                        examples[message.stance].append({
                            "heard": message.heard, "reply": text,
                            "why": checked.why()})
            tally["any traced"] += bool(traced)
            shortened_traced = bool(traced) or any(
                trace(message, reading_of(short, words), words, framing,
                      short).traced
                for text in replies for short in shortened(text))
            tally["shortened traced"] += shortened_traced
            if traced:
                for how, wrong in corruptions(message, traced[0], others,
                                              rng).items():
                    read = reading_of(wrong, words)
                    caught = not trace(message, read, words, framing,
                                       wrong).traced
                    tally[f"{how} tried"] += 1
                    tally[f"{how} caught"] += caught
        print(f"  {min(start + batch, len(found))}/{len(found)}", flush=True)
    total = collections.Counter()
    for tally in counts.values():
        total.update(tally)
    counts["all"] = total

    def rates(tally: collections.Counter) -> dict:
        out = {"messages": tally["messages"],
               "first traced": _rate(tally["first traced"],
                                     tally["messages"]),
               "any of four traced": _rate(tally["any traced"],
                                           tally["messages"]),
               "or shortened": _rate(tally["shortened traced"],
                                     tally["messages"]),
               "stance read": _rate(tally["stance read"], tally["replies"]),
               "encoder agrees with labels": _rate(tally["agrees"],
                                                   tally["replies"])}
        for how in ("value swapped", "denial dropped", "denial added",
                    "word added", "rule named"):
            if tally[f"{how} tried"]:
                out[f"{how} caught"] = _rate(tally[f"{how} caught"],
                                             tally[f"{how} tried"])
        return out

    report = {"by stance": {stance: rates(tally)
                            for stance, tally in sorted(counts.items())},
              "examples": examples}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _rate(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.1%})" if whole else "-"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=600)
    options = parser.parse_args(argv)
    report = evaluate(options.limit)
    print(json.dumps(report["by stance"], indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

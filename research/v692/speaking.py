"""Mathematics said back: what the decoder is taught, and how it is checked.

    python -m research.v692.speaking             # llm/math-decoder-data,
                                                 # and the answers the
                                                 # encoder reads back

The decoder (v690, SmolLM2) says what a turn came to. For mathematics it is
taught from pairs made here: the **message** is what v690 would build from
the answer the page gives (`answer_of`, then v690's own `message.of_turn`),
and the **reply** says the result in words, the way `saying.py` says an
object -- *the derivative is three x squared*, *no, 91 is not prime: it is 7
times 13*. So the prompt it learns from is exactly the prompt it will be
given, and what it learns to say is mathematics in English.

The same saying is also an **answer record** for the encoder: its words,
which of them are the value, and the symbols each stands for. That is what
lets a reply be read back (`traced`): the encoder reads the decoder's reply
into an object and sympy says whether it is the object the message was
about. v690's round trip compares words; for mathematics it compares the
mathematics.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as S

from research.v692 import curriculum as C, doing
from research.v692.corpus import LLM, record as corpus_record, words as split
from research.v692.saying import DROP, Said, Speaker

OUT = LLM / "math-decoder-data"
ANSWERS = LLM / "math-data"

#: How each act's value is said, `{V}` for it.
SAYS = {
    "value": ("It's {V}.", "That's {V}.", "The answer is {V}.",
              "It comes to {V}.", "{V}."),
    "simplify": ("It simplifies to {V}.", "That's {V}.",
                 "Simplified, it's {V}."),
    "expand": ("Expanded, it's {V}.", "It expands to {V}.",
               "That's {V}."),
    "factor": ("It factors as {V}.", "That's {V}.", "Factorised: {V}."),
    "derivative": ("The derivative is {V}.", "It's {V}."),
    "integral": ("The integral is {V}.", "It's {V}."),
    "limit": ("The limit is {V}.", "It's {V}."),
    "at": ("It's {V}.", "That gives {V}.", "The value is {V}."),
    "gcd": ("The greatest common divisor is {V}.", "It's {V}."),
    "lcm": ("The least common multiple is {V}.", "It's {V}."),
    "mean": ("The mean is {V}.", "The average is {V}."),
    "median": ("The median is {V}.",),
    "mode": ("The mode is {V}.",),
    "range": ("The range is {V}.",),
    "sum": ("The total is {V}.", "They add up to {V}."),
    "choose": ("There are {V} ways.", "{V} ways."),
    "arrange": ("There are {V} ways.", "They can be arranged {V} ways."),
    "determinant": ("The determinant is {V}.", "It's {V}."),
    "inverse": ("The inverse is {V}.",),
    "transpose": ("The transpose is {V}.",),
    "union": ("The union is {V}.",),
    "intersection": ("The intersection is {V}.",),
    "part of": ("That's {V}.", "It's {V}.", "It comes to {V}."),
    "convert": ("It's {V}.", "That's {V}.", "That comes to {V}."),
    "next term": ("The next term is {V}.", "Next comes {V}.", "It's {V}."),
    "nth term": ("The nth term is {V}.", "The rule is {V}."),
    "term": ("That term is {V}.", "It's {V}."),
    "series": ("The sum is {V}.", "It adds up to {V}."),
    "dice": ("The probability is {V}.", "The chance is {V}."),
    "coins": ("The probability is {V}.", "The chance is {V}."),
    "round places": ("It rounds to {V}.", "That's {V}."),
    "round nearest": ("It rounds to {V}.", "That's {V}."),
    "percent of": ("That's {V}.", "It's {V}."),
    "as percent": ("It's {V} percent.", "That's {V} percent."),
    "percent change": ("It's a change of {V} percent.",
                       "That's {V} percent."),
    "dot": ("The dot product is {V}.", "It's {V}."),
    "cross": ("The cross product is {V}.",),
    "magnitude": ("The magnitude is {V}.", "Its length is {V}."),
    "modulus": ("The modulus is {V}.", "It's {V}."),
    "conjugate": ("The conjugate is {V}.",),
}


def _about(parts: dict):
    """What a kind is asked of: a number, an expression, or a sequence."""
    for role in ("A", "EXPR", "LIST"):
        if parts.get(role) is not None:
            value = parts[role]
            return list(value) if role == "LIST" else value
    return None


def claim_of(act: str, parts: dict, result: doing.Result) -> str:
    """What a yes or a no is about, said as the claim asked: `91 is prime`,
    whichever way it came out."""
    if act == "is kind":
        kind = parts.get("KIND")
        obj = _about(parts)
        written = doing.written(obj)
        if any(one in written for one in "&|~"):
            # A proposition written in symbols is not a claim in words, and
            # a decoder handed `p | ~p is a tautology` says it back as
            # words in the wrong order. A plain yes settles it.
            return ""
        return f"{written} is {kind.said[0]}" if kind else ""
    if act == "divides":
        return (f"{doing.written(parts.get('A'))} is divisible by "
                f"{doing.written(parts.get('B'))}")
    if act == "subset":
        return (f"{doing.written(parts.get('EXPR'))} is a subset of "
                f"{doing.written(parts.get('OTHER'))}")
    if act == "check":
        return result.about
    if act == "equivalent":
        return (f"{doing.written(parts.get('EXPR'))} is equivalent to "
                f"{doing.written(parts.get('OTHER'))}")
    return ""


def answer_of(said: str, act: str, parts: dict, result: doing.Result,
              symbols: dict | None = None) -> dict:
    """The answer the page gives for a result, and the turn v690 builds a
    message from: the one shape both the page and this corpus use."""
    outcome = {"value": "retrieved", "yes": "verified", "no": "denied",
               "noted": "noted"}.get(result.stance, "unknown")
    text = result.text
    if outcome == "retrieved":
        # v690 takes a value as what is said before the first dash.
        text = f"{result.text} — worked out"
    elif result.because and result.because not in text:
        text = f"{text}: {result.because}"
    answer = {"outcome": outcome, "source": "mathematics", "text": text,
              "act": f"mathematics: {act}",
              "mathematics": {"act": act, "stance": result.stance,
                              "symbols": symbols or {}}}
    reading = {"act": act}
    if result.about and act == "measure":
        # What the answer is about, so a reply says `the area is 24` and
        # not `the width is 24` after the width was the thing given.
        reading["mention"] = {"text": result.about}
    claim = claim_of(act, parts, result) if outcome in (
        "verified", "denied") else (result.text if outcome == "noted"
                                    else "")
    if claim:
        reading["confirms"] = claim
    return {"said": said, "answer": answer, "reading": reading, "run": {}}


def _said_value(value, speaker: Speaker) -> Said:
    if isinstance(value, list):
        out = Said()
        for index, one in enumerate(value):
            if index:
                out.add("and" if index == len(value) - 1 else ",", ",")
            out.then(speaker.say(one))
        return out
    return speaker.say(value)


def reply_of(act: str, parts: dict, result: doing.Result, rng,
             variable=None) -> tuple:
    """(the reply, its words with their parts and symbols): a result said
    in English, the value as `saying.py` says an object."""
    speaker = Speaker(rng, "spoken" if rng.random() < 0.75 else "written")
    words, roles, labels = [], [], []

    def plain(text: str) -> None:
        for word in split(text):
            words.append(word)
            roles.append("O")
            labels.append(DROP)

    def said(value, role: str = "EXPR") -> None:
        found = _said_value(value, speaker)
        for word, label in zip(found.words, found.labels):
            for one in split(word):
                words.append(one)
                roles.append(role)
                labels.append(label if one == split(word)[-1] else DROP)

    stance = result.stance
    if stance == "value":
        value = result.value
        if act in ("as percent", "percent change", "percent of") and \
                not value.is_Integer:
            # Said as the decimal it is written as: `37.5 percent`.
            value = S.Float(f"{float(value):.2f}".rstrip("0"))
        if act == "solve system":
            for index, one in enumerate(value):
                if index:
                    words.append("and")
                    roles.append("LIST")
                    labels.append(",")
                plain(f"{one.lhs} equals" if rng.random() < 0.7
                      else f"{one.lhs} is")
                said(one.rhs, "LIST")
            plain(".")
            return _sentence(words), (words, roles, labels)
        if act == "measure":
            from research.v692 import measuring as M
            from research.v692.corpus import _quantity_said
            plain("the")
            for word, role, label in _quantity_said(parts["WANTED"], rng,
                                                    "WANTED"):
                words.append(word)
                roles.append(role)
                labels.append(label)
            plain("is")
            said(value)
            if M.unit_of(parts["WANTED"]):
                plain(M.unit_of(parts["WANTED"]))
            plain(".")
            return _sentence(words), (words, roles, labels)
        if act == "solve" and isinstance(value, S.Rel):
            plain(rng.choice(("it holds when", "that is true when",
                              "the answer is")))
            said(value)
            plain(".")
            return _sentence(words), (words, roles, labels)
        if act == "solve":
            var = variable or S.Symbol("x")
            roots = value if isinstance(value, list) else [value]
            plain(f"{var} equals" if rng.random() < 0.7 else f"{var} is")
            said(roots, "LIST" if len(roots) > 1 else "EXPR")
            plain(".")
            return _sentence(words), (words, roles, labels)
        if isinstance(value, dict):
            # Prime factors, however asked for: `5 times 17`, each prime as
            # often as it divides.
            primes = [prime for prime, power in sorted(value.items())
                      for _ in range(power)]
            plain(rng.choice(("it's", "that's", "it is")))
            for index, prime in enumerate(primes):
                if index:
                    words.append("times")
                    roles.append("EXPR")
                    labels.append("*")
                said(S.Integer(prime))
            plain(".")
            return _sentence(words), (words, roles, labels)
        if isinstance(value, list) and act == "divisors":
            pattern = "Its factors are {V}."
        else:
            pattern = rng.choice(SAYS.get(act, SAYS["value"]))
        head, _, tail = pattern.partition("{V}")
        plain(head)
        said(value, "LIST" if isinstance(value, list) else "EXPR")
        if act == "integral" and result.text.endswith("+ C"):
            plain("plus a constant")
        plain(tail)
    elif stance in ("yes", "no"):
        claim = claim_of(act, parts, result)
        plain("yes ," if stance == "yes" else "no ,")
        if act == "is kind" and parts.get("KIND") is not None:
            obj = _about(parts)
            said(obj, "A")
            plain("is" if stance == "yes" else "is not")
            for word in parts["KIND"].said[0].split():
                words.append(word)
                roles.append("KIND")
                labels.append(DROP)
        elif act == "check" and isinstance(parts.get("EXPR"), S.Rel):
            relation = parts["EXPR"]
            if stance == "no":
                plain("it is not true that")
            said(relation)
        elif stance == "no" and claim:
            plain(claim.replace(" is ", " is not ", 1))
        else:
            plain(claim or ("it is" if stance == "yes" else "it is not"))
        if result.because:
            plain(": " + result.because.replace(" = ", " is "))
        plain(".")
    elif stance == "noted":
        plain(rng.choice(("got it ,", "all right ,", "noted :")))
        plain(result.text)
        plain(".")
    else:
        plain(result.text or "I could not work that out")
        plain(".")
    return _sentence(words), (words, roles, labels)


def _sentence(words: list) -> str:
    """Words as a sentence: stops and commas against the word before, the
    first word opened -- unless it is a variable, which keeps its case."""
    text = " ".join(words)
    for mark in (" .", " ,", " :"):
        text = text.replace(mark, mark.strip())
    first = words[0] if words else ""
    if len(first) == 1 and first.isalpha() and first not in ("a", "i"):
        return text
    return text[:1].upper() + text[1:]


def build(count: int = 8000, seed: int = 693) -> dict:
    from research.v690.message import of_turn, prompt

    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, answers, stances = [], [], {}
    acts = [one for one in C.ACTS]
    attempts = 0
    while len(pairs) < count and attempts < count * 4:
        attempts += 1
        act = rng.choice(acts)
        template = rng.choice(act.templates)
        try:
            record = corpus_record(act, template, rng)
        except (ValueError, TypeError, KeyError, ZeroDivisionError,
                AttributeError, RecursionError):
            record = None
        if record is None:
            continue
        said_question = record["said"]
        parts = dict(record["_parts"])
        result = doing.do(act.name, dict(parts))
        if result.stance == "unknown" and rng.random() < 0.8:
            continue
        turn = answer_of(said_question, act.name, parts, result)
        message = of_turn(turn)
        reply, (words, roles, labels) = reply_of(
            act.name, parts, result, rng, parts.get("VAR"))
        key = f"math-{len(pairs)}"
        pairs.append({"key": key, "stance": message.stance,
                      "prompt": prompt(message), "reply": reply})
        stances[message.stance] = stances.get(message.stance, 0) + 1
        if result.stance in ("value", "yes", "no") and len(words) <= 60:
            answers.append({"task": "math", "words": words, "act": "answer",
                            "roles": roles, "symbols": labels,
                            "tags": [""] * len(words),
                            "deps": [""] * len(words), "names": [],
                            "said": " ".join(words), "source": key,
                            "how": "answer"})
    with open(OUT / "train-math.jsonl", "w", encoding="utf-8") as train, \
            open(OUT / "valid-math.jsonl", "w", encoding="utf-8") as valid:
        for index, one in enumerate(pairs):
            (valid if index % 10 == 0 else train).write(json.dumps(one)
                                                       + "\n")
    with open(ANSWERS / "train-answer.jsonl", "w", encoding="utf-8") as \
            train, open(ANSWERS / "valid-answer.jsonl", "w",
                        encoding="utf-8") as valid:
        for index, one in enumerate(answers):
            (valid if index % 10 == 0 else train).write(json.dumps(one)
                                                       + "\n")
    stats = {"pairs": len(pairs), "answers": len(answers),
             "stances": stances}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1),
                                    encoding="utf-8")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=8000)
    options = parser.parse_args(argv)
    print(json.dumps(build(options.count), indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())


# -- reading a reply back -------------------------------------------------------

def expected_of(message) -> object | None:
    """The object a mathematics message is about, from what it says: its
    value (`x^2 - 5x + 6`, `x = 2 or x = 3`, `5 times 17`, `{0, 6}`)."""
    from research.v692.symbols import Unreadable, parsed
    if not message.values:
        return None
    text = message.values[0].split(", about ")[0]
    for tail in ("%", " degrees", "+ C", "+ C."):
        # `+ C` is said as `plus a constant`: what is compared is the
        # integral, and the constant is said in words or not at all.
        text = text[:-len(tail)].strip() if text.endswith(tail) else text
    try:
        for joiner in (" or ", " and "):
            if joiner in text and "=" in text:
                return [parsed(part.split("=", 1)[1])
                        for part in text.split(joiner)]
        # `x = 5` is said back as `x equals 5`, whose value is the 5: a
        # reply says what the variable is, not the equation.
        if text.count("=") == 1 and not any(
                one in text for one in ("<", ">", "!")):
            return parsed(text.split("=", 1)[1])
        return parsed(text.replace(" times ", " * "))
    except Unreadable:
        return None


def traced(message, text: str, read=None):
    """v690's round trip for a mathematics message (`v690.speaking.CHECKS`):
    the stance as the reply heads read it, and the value as the encoder's
    mathematics heads read it back -- the same object, or the reply is not
    what the message says. None for a message that is not mathematics."""
    from research.v690.message import INTERNAL
    from research.v690.roundtrip import Trace
    from research.v692 import reading
    from research.v692.symbols import same

    if not (message.act or "").startswith("mathematics"):
        return None
    stance = getattr(read, "stance", "") or message.stance
    found = Trace(stance, message.stance)
    found.internal = INTERNAL.findall(text)
    if not text.rstrip().endswith((".", "!", "?")):
        found.unfinished = "the reply stops before its sentence ends"
    if message.stance != "value":
        return found
    wanted = expected_of(message)
    if wanted is None:
        return found
    back = reading.read(text)
    value = None
    if back is not None:
        value = back.parts.get("EXPR", back.parts.get("LIST"))
    if value is None:
        found.missing = [message.values[0]]
        return found
    if isinstance(wanted, list):
        items = list(value) if isinstance(value, S.Tuple) else [value]
        good = len(items) == len(wanted) and all(
            any(same(one, other) for other in items) for one in wanted)
    elif isinstance(value, S.Tuple):
        good = same(S.Tuple(*wanted) if isinstance(wanted, (list, tuple))
                    else wanted, value)
    else:
        good = same(value, wanted)
    if not good:
        found.missing = [message.values[0]]
        found.added = [" ".join(back.words)]
    return found

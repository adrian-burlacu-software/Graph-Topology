"""The common-sense probe: 114 ordinary questions, asked of the page's own
reading and answered from the real store.

Run: python -m research.v689.probe [--out probe.log]

This is the acceptance criterion the executive phases are held to
(`v690/DESIGN.md`, *Where it stands*): a phase lands its structure without
moving an answer, and the way that is shown is that this file's output does
not change. The suite is not enough on its own -- 1,027 tests passed either
side of a change to R28 that `v688/audit.py` then measured as tripling
over-affirmation.

Each line is `[n/114] family | question -> outcome source`, where `outcome`
and `source` are the turn's own (`session.Turn.answer`). What is compared
between runs is the outcome and the source, never the wording: the wording
is the decoder's and changes when it is retrained.

The questions are the ones the common-sense audit was written from
(`COMMON_SENSE.md`): what the page could not say about ordinary things --
is a chair alive, is an elephant bigger than a mouse, can you cut bread with
a knife -- grouped by the family each belongs to. `BASELINE` is what they
answered on 2026-09-14, kept so a run can be scored without a previous log
to hand.

Time and events are deliberately not here. They had a probe of their own,
whose scenarios were lost with the scratch script that ran them; `test_time`
covers the same ground and is committed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from research.v688.pool import DEFAULT_WORKERS

#: (family, question), in the order they are asked.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("size absolute", "is an elephant big"),
    ("size absolute", "is a mouse small"),
    ("size absolute", "is an ant tiny"),
    ("weight absolute", "is a feather light"),
    ("weight absolute", "is a rock heavy"),
    ("weight absolute", "is a car heavy"),
    ("size compared", "is an elephant bigger than a mouse"),
    ("size compared", "is a cat bigger than a horse"),
    ("size compared", "which is bigger, a dog or an ant"),
    ("weight compared", "which is heavier, a feather or a brick"),
    ("weight compared", "is a car heavier than a bicycle"),
    ("speed compared", "is a cheetah faster than a turtle"),
    ("fit", "can an elephant fit in a car"),
    ("fit", "can a cup hold water"),
    ("fit", "can a mouse fit in a shoe"),
    ("material property", "is glass fragile"),
    ("material property", "is a rock hard"),
    ("material property", "is a pillow soft"),
    ("material property", "is a knife sharp"),
    ("material property", "is rubber flexible"),
    ("material property", "is metal shiny"),
    ("temperature", "is ice cold"),
    ("temperature", "is fire hot"),
    ("temperature", "is the sun hot"),
    ("buoyancy", "does wood float"),
    ("buoyancy", "does a rock float"),
    ("buoyancy", "can a stone float on water"),
    ("state", "is water a liquid"),
    ("state", "is ice solid"),
    ("colour", "what color is grass"),
    ("colour", "is snow white"),
    ("colour", "is the sky blue"),
    ("colour", "are strawberries red"),
    ("shape", "is a ball round"),
    ("shape", "what shape is a coin"),
    ("texture", "is sandpaper rough"),
    ("texture", "is a cat furry"),
    ("made of", "is a window made of glass"),
    ("made of", "what is a table made of"),
    ("made of", "is a shoe made of leather"),
    ("made of", "is bread made from flour"),
    ("affordance", "can you cut bread with a knife"),
    ("affordance", "can you drink from a cup"),
    ("affordance", "can you sit on a chair"),
    ("affordance", "can you eat soup with a fork"),
    ("affordance", "can you write with a pencil"),
    ("instrument", "what do you use to cut paper"),
    ("instrument", "what can you use to open a door"),
    ("purpose", "what is a chair for"),
    ("purpose", "what do you use a key for"),
    ("purpose", "is a hammer used for hitting nails"),
    ("purpose", "is a spoon used for eating"),
    ("edible", "can you eat an apple"),
    ("edible", "can people eat rocks"),
    ("edible", "is a mushroom edible"),
    ("safety", "is it safe to eat a mushroom"),
    ("safety", "is bleach poisonous"),
    ("safety", "is fire dangerous"),
    ("safety", "is a knife dangerous"),
    ("safety", "can dogs eat chocolate"),
    ("safety", "is chocolate bad for dogs"),
    ("safety", "is a snake dangerous"),
    ("need", "does a plant need water"),
    ("need", "do people need sleep"),
    ("need", "does a car need fuel"),
    ("need", "does a fish need water"),
    ("need", "what does a fish need to live"),
    ("need", "can a person live without water"),
    ("need", "what do you need to bake a cake"),
    ("need", "what does a baby need"),
    ("desire", "does a cat like milk"),
    ("desire", "do dogs like bones"),
    ("desire", "do cats like water"),
    ("desire", "what do cats like"),
    ("desire", "do children like candy"),
    ("desire", "does a person want money"),
    ("desire", "what do dogs hate"),
    ("location", "where do you keep milk"),
    ("location", "where is a pillow found"),
    ("location", "is a fridge in a kitchen"),
    ("location", "where do people sleep"),
    ("location", "is a bed in a bedroom"),
    ("location", "where would you find a book"),
    ("location", "do fish live in trees"),
    ("animacy", "is a chair alive"),
    ("animacy", "is a tree alive"),
    ("animacy", "does a rock breathe"),
    ("animacy", "can a table think"),
    ("artefact", "is a car man-made"),
    ("artefact", "is a mountain natural"),
    ("artefact", "did people make rivers"),
    ("perception", "can a fish see"),
    ("perception", "can a worm hear"),
    ("perception", "does a stone feel pain"),
    ("perception", "can a dog smell"),
    ("feeling", "can a dog be happy"),
    ("feeling", "can a robot feel sad"),
    ("ability", "can a baby walk"),
    ("ability", "can a child drive a car"),
    ("ability", "can a person breathe underwater"),
    ("ability", "can a person lift a car"),
    ("ability", "can a person see in the dark"),
    ("parts", "does a dog have cells"),
    ("parts", "does a bicycle have an engine"),
    ("parts", "does a table have legs"),
    ("parts", "does a snake have legs"),
    ("parts", "does a fish have lungs"),
    ("folk category", "is a tomato a fruit"),
    ("folk category", "is a spider an insect"),
    ("folk category", "is a whale a fish"),
    ("folk category", "is a bat a bird"),
    ("folk category", "is a dog a good pet"),
    ("folk category", "is a knife a tool"),
    ("folk category", "is milk a drink"),
)

#: question -> `outcome source`, as they answered on 2026-09-14 (the run
#: `probe-common-rulebook.log` recorded, after P5's first step). A run that
#: differs from this has moved an answer, which is the thing to explain.
BASELINE: dict[str, str] = {
    "is an elephant big": "verified kind",
    "is a mouse small": "verified kind",
    "is an ant tiny": "verified kind",
    "is a feather light": "verified kind",
    "is a rock heavy": "verified kind",
    "is a car heavy": "verified kind",
    "is an elephant bigger than a mouse": "verified kind",
    "is a cat bigger than a horse": "denied kind",
    "which is bigger, a dog or an ant": "retrieved kind",
    "which is heavier, a feather or a brick": "retrieved kind",
    "is a car heavier than a bicycle": "verified kind",
    "is a cheetah faster than a turtle": "unknown kind",
    "can an elephant fit in a car": "unknown kind",
    "can a cup hold water": "unknown kind",
    "can a mouse fit in a shoe": "unknown kind",
    "is glass fragile": "verified kind",
    "is a rock hard": "verified kind",
    "is a pillow soft": "verified kind",
    "is a knife sharp": "verified kind",
    "is rubber flexible": "verified kind",
    "is metal shiny": "unknown kind",
    "is ice cold": "verified kind",
    "is fire hot": "denied kind",
    "is the sun hot": "unknown tendency",
    "does wood float": "unknown kind",
    "does a rock float": "denied kind",
    "can a stone float on water": "denied kind",
    "is water a liquid": "verified kind",
    "is ice solid": "verified kind",
    "what color is grass": "retrieved kind",
    "is snow white": "unknown kind",
    "is the sky blue": "unknown tendency",
    "are strawberries red": "retrieved kind",
    "is a ball round": "verified definition",
    "what shape is a coin": "retrieved kind",
    "is sandpaper rough": "unknown kind",
    "is a cat furry": "verified kind",
    "is a window made of glass": "unknown kind",
    "what is a table made of": "retrieved kind",
    "is a shoe made of leather": "unknown kind",
    "is bread made from flour": "verified kind",
    "can you cut bread with a knife": "unknown kind",
    "can you drink from a cup": "verified kind",
    "can you sit on a chair": "verified kind",
    "can you eat soup with a fork": "unknown kind",
    "can you write with a pencil": "verified kind",
    "what do you use to cut paper": "retrieved kind",
    "what can you use to open a door": "retrieved kind",
    "what is a chair for": "retrieved kind",
    "what do you use a key for": "retrieved kind",
    "is a hammer used for hitting nails": "verified kind",
    "is a spoon used for eating": "verified kind",
    "can you eat an apple": "verified kind",
    "can people eat rocks": "denied kind",
    "is a mushroom edible": "verified kind",
    "is it safe to eat a mushroom": "unknown kind",
    "is bleach poisonous": "verified kind",
    "is fire dangerous": "verified kind",
    "is a knife dangerous": "verified kind",
    "can dogs eat chocolate": "unknown kind",
    "is chocolate bad for dogs": "unknown kind",
    "is a snake dangerous": "verified kind",
    "does a plant need water": "verified kind",
    "do people need sleep": "unknown kind",
    "does a car need fuel": "verified kind",
    "does a fish need water": "unknown kind",
    "what does a fish need to live": "retrieved kind",
    "can a person live without water": "unknown kind",
    "what do you need to bake a cake": "retrieved kind",
    "what does a baby need": "retrieved kind",
    "does a cat like milk": "unknown kind",
    "do dogs like bones": "verified kind",
    "do cats like water": "verified kind",
    "what do cats like": "retrieved kind",
    "do children like candy": "unknown kind",
    "does a person want money": "unknown kind",
    "what do dogs hate": "retrieved kind",
    "where do you keep milk": "retrieved kind",
    "where is a pillow found": "retrieved kind",
    "is a fridge in a kitchen": "verified kind",
    "where do people sleep": "retrieved kind",
    "is a bed in a bedroom": "verified kind",
    "where would you find a book": "retrieved kind",
    "do fish live in trees": "denied kind",
    "is a chair alive": "denied kind",
    "is a tree alive": "verified kind",
    "does a rock breathe": "denied kind",
    "can a table think": "denied kind",
    "is a car man-made": "verified definition",
    "is a mountain natural": "verified kind",
    "did people make rivers": "unknown kind",
    "can a fish see": "unknown kind",
    "can a worm hear": "unknown kind",
    "does a stone feel pain": "denied kind",
    "can a dog smell": "verified kind",
    "can a dog be happy": "unknown kind",
    "can a robot feel sad": "unknown kind",
    "can a baby walk": "verified kind",
    "can a child drive a car": "unknown kind",
    "can a person breathe underwater": "unknown kind",
    "can a person lift a car": "unknown kind",
    "can a person see in the dark": "unknown kind",
    "does a dog have cells": "verified kind",
    "does a bicycle have an engine": "denied kind",
    "does a table have legs": "verified kind",
    "does a snake have legs": "denied kind",
    "does a fish have lungs": "unknown kind",
    "is a tomato a fruit": "verified kind",
    "is a spider an insect": "denied kind",
    "is a whale a fish": "denied kind",
    "is a bat a bird": "denied kind",
    "is a dog a good pet": "unknown kind",
    "is a knife a tool": "verified kind",
    "is milk a drink": "verified kind",
}


def answered(turn) -> str:
    """`outcome source` of a turn, which is what a run is compared on."""
    answer = getattr(turn, "answer", None) or {}
    return f"{answer.get('outcome') or '-'} {answer.get('source') or '-'}"


def run(cycles: int = 8, out=None, workers: int = DEFAULT_WORKERS
        ) -> list[tuple[str, str, str]]:
    """Every question, each to a conversation of its own so nothing one
    question teaches is in memory for the next. (family, question, answer).

    **The pool width is part of the measurement**, not a speed knob. A cycle
    puts out one question per worker, so a narrow pool truncates the
    corroboration fan-out and the family conflicts never surface:
    `test_v688` says the same of its own pool, and at one worker twenty of
    these 114 answers lose a confident verdict for an `unknown`. The page
    serves 19, so that is what the probe asks at.
    """
    from research.v687 import build
    from research.v688 import server as v688
    from research.v689.server import StoreAsker
    from research.v689.session import Session

    write = (out or sys.stdout).write
    service = v688.Service(build.DEFAULT_STORE, workers, cycles,
                           teacher=False)
    found: list[tuple[str, str, str]] = []
    started = time.time()
    try:
        asker = StoreAsker(service)
        for index, (family, question) in enumerate(QUESTIONS, start=1):
            session = Session(asker, conversation=f"probe-{index}",
                              example=True)
            try:
                answer = answered(session.say(question))
            except Exception as bad:                # noqa: BLE001
                answer = f"error {type(bad).__name__}"
            found.append((family, question, answer))
            write(f"[{index}/{len(QUESTIONS)}] {time.time() - started:.0f}s "
                  f"{family} | {question} -> {answer}\n")
            (out or sys.stdout).flush()
        write(f"done in {time.time() - started:.0f}s\n")
    finally:
        service.pool.close()
    return found


def compare(found: list[tuple[str, str, str]], out=None) -> int:
    """What moved against `BASELINE`. Returns how many answers differ."""
    write = (out or sys.stdout).write
    moved = [(question, BASELINE[question], answer)
             for _, question, answer in found
             if question in BASELINE and BASELINE[question] != answer]
    if not moved:
        write(f"all {len(found)} answers as the baseline has them\n")
        return 0
    write(f"{len(moved)} of {len(found)} moved:\n")
    for question, was, now in moved:
        write(f"  {question}: {was} -> {now}\n")
    return len(moved)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="write the run to a file as well as stdout")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="one question per worker per cycle; the page's "
                             "own default, so the probe asks as the page does")
    options = parser.parse_args(argv)
    handle = options.out.open("w", encoding="utf-8") if options.out else None
    try:
        found = run(options.cycles, handle, options.workers)
        if handle is not None:
            handle.flush()
        compare(found)
    finally:
        if handle is not None:
            handle.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

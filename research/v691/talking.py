"""Talking to the agent at a terminal, in whatever world you pick.

    python -m research.v691
    python -m research.v691 --domain errands
    python -m research.v691 --gremlin

The same acts the page runs, chosen the same way, over the same `Scene`
(`page.say_to`), so what happens here and what happens in a conversation on
the v690 page cannot differ. This file is the prompt and nothing else.

On the page the agent is one act among v689's, and it stays silent until a
world is opened -- `what worlds do you have`, then `use the blocks world`.
Here `--domain` opens one at the start, because a terminal that greets you
with nothing to do is a worse demonstration than one that does.
"""
from __future__ import annotations

import argparse

from research.v691 import world as W
from research.v691.domains import DOMAINS
from research.v691.page import say_to
from research.v691.scene import Scene


class Gremlin(W.World):
    """A world with something else in it.

    Between turns a person can move something and the agent simply plans
    again from what it finds -- it is never *surprised*, because it was not
    expecting anything at the time. A surprise needs the world to move
    **while an action is being taken**, so this is a world that lets
    something do that: after each action, whatever `meddling` is gets a
    turn. It is how `--gremlin` makes the impasse in `acting.agent`
    reachable from a conversation rather than only from a test.
    """

    def __init__(self, facts, meddling=None) -> None:
        super().__init__(facts)
        self.meddling = meddling
        self.meddled = False

    def do(self, action) -> bool:
        done = super().do(action)
        if done and self.meddling is not None:
            self.meddling(self)
        return done


def undo_one(domain):
    """Once, put back the first thing that was put where the goal wants it.

    Domain-general: it takes a fact of a `goalish` predicate and returns
    whatever it was about to however the domain says a thing starts. In
    blocks that is a block falling off a stack; in errands it is something
    being taken back out of the bag.
    """
    def meddle(world: Gremlin) -> None:
        if world.meddled:
            return
        for fact in sorted(world.facts):
            parts = fact.split()
            if parts[0] not in domain.goalish or len(parts) < 3:
                continue
            world.meddled = True
            back = set(world.facts) - {fact}
            kind = domain.typing().get(parts[0], ())
            start = domain.starts.get(kind[0] if kind else "", ())
            back |= {one.replace("?x", parts[1]) for one in start}
            for predicate, where in domain.taken.items():
                if where < len(parts):
                    back.add(f"{predicate} {parts[where]}")
            world.facts = frozenset(back)
            return
    return meddle


BANNER = """Tell me what is there, then what you want done with it.
`worlds` lists them, `use the <name> world` picks one, `quit` leaves.

  there is a red block on a green block, and a blue block on the table
  put the green block on the blue block and the red block on the green block
  what is on the blue block
  why did you move the red block
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="blocks",
                        choices=sorted(DOMAINS) + [""],
                        help="the world to open at the start; empty to open "
                             "none, as the page does")
    parser.add_argument("--gremlin", action="store_true",
                        help="let something move, once, while the agent is "
                             "working: a real surprise rather than a change "
                             "between turns")
    options = parser.parse_args(argv)

    scene = Scene(DOMAINS[options.domain] if options.domain else None)
    if options.gremlin and scene.open:
        scene.world = Gremlin(scene.world.facts, undo_one(scene.domain))
    print(BANNER)
    if scene.open:
        print(f"  (the {scene.domain.name} world)\n")
    while True:
        try:
            said = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if said.lower() in ("quit", "exit", "bye"):
            print("  goodbye\n")
            return 0
        if said.lower() in ("worlds", "what worlds", "?"):
            said = "what worlds do you have"
        print(f"  {say_to(scene, said)}\n")


if __name__ == "__main__":
    raise SystemExit(main())

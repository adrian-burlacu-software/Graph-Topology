"""A bank of open-world goals, and what a sensible way to each would be.

Written before the designer was run on them, from common sense: *cut a
rope* wants something that cuts, *make the milk cold* somewhere cold. The
expectations are sets, because there are many good ways -- a knife, a
saw, scissors -- and a designer that picks any of them has designed. What
is scored is whether the way is **one of the sensible ones**, and whether
it is **the simplest the scene allows**: the scissors on the table, not a
knife from somewhere.

Each row is (what is said, wants, scene, names, ways, means):

- `ways`: the forms a sensible way could take
- `means`: what it could use -- any thing whose name, or a kind of it, is
  here; empty where a way uses nothing (open the door yourself)

    python -m research.v694.bank          # every goal, with what it came to
"""
from __future__ import annotations

import sys
import time

from research.v694 import designing, knowing as K
from research.v694.goals import Goal

CUTTERS = {"knife", "scissors", "saw", "blade", "ax", "axe", "razor",
           "machete", "shears", "cutter", "die"}

BANK = [
    # -- a doing that takes a tool --------------------------------------
    ("cut the rope", ("cut rope",), (), (), {"tool"}, CUTTERS),
    ("cut the rope; the scissors are on the table", ("cut rope",),
     ("at scissors table",), (), {"tool"}, {"scissors"}),
    ("cut the rope; there is a carving knife in the drawer", ("cut rope",),
     ("at carving-knife drawer",), (), {"tool"}, {"carving-knife"}),
    ("dig a hole", ("dig hole",), (), (), {"tool"},
     {"shovel", "spade", "trowel", "hoe", "excavator", "backhoe"}),
    ("light the room", ("light room",), (), (), {"tool"},
     {"lamp", "candle", "lantern", "torch", "flashlight", "light-bulb",
      "bulb"}),
    ("unlock the door", ("unlocked door",), (), (), {"tool"}, {"key"}),
    ("wake john", ("wake john",), (), ("john",), {"tool", "unaided"},
     {"alarm-clock", "clock", "alarm", "bugle", "gong", "bell"}),
    ("wake john; there is an alarm clock on the shelf", ("wake john",),
     ("at alarm-clock shelf",), ("john",), {"tool"}, {"alarm-clock"}),
    ("boil the water", ("boil water",), (), (), {"tool"},
     {"kettle", "teakettle", "pot", "saucepan", "stove", "microwave"}),
    ("dry the shirt", ("dry shirt",), (), (), {"tool", "place"},
     {"towel", "dryer", "clothes-dryer", "clothesline", "iron",
      "hair-dryer"}),
    ("open the can", ("open can",), (), (), {"tool"},
     {"can-opener", "opener", "knife"}),
    ("clean the floor", ("clean floor",), (), (), {"tool", "helper"},
     {"mop", "broom", "vacuum", "vacuum-cleaner", "sponge", "cloth",
      "washcloth", "cleaner", "janitor"}),
    ("sweep the floor", ("sweep floor",), (), (), {"tool"},
     {"broom", "brush", "sweeper"}),
    ("tie the package", ("tie package",), (), (), {"tool", "unaided"},
     {"string", "cord", "rope", "ribbon", "twine", "tape"}),
    ("paint the wall", ("paint wall",), (), (), {"tool", "unaided",
                                                 "helper"},
     {"brush", "paintbrush", "roller", "paint", "painter", "spray"}),
    ("water the plant", ("water plant",), (), (), {"tool", "unaided"},
     {"watering-can", "hose", "sprinkler", "can", "pitcher"}),
    ("mow the lawn", ("mow lawn",), (), (), {"tool", "helper"},
     {"mower", "lawn-mower", "lawnmower", "scythe", "gardener"}),
    ("iron the shirt", ("iron shirt",), (), (), {"tool"}, {"iron"}),
    ("sharpen the knife", ("sharpen knife",), (), (), {"tool"},
     {"sharpener", "whetstone", "file", "grindstone", "steel", "stone",
      "hone"}),
    ("cook the egg", ("cook egg",), (), (), {"tool"},
     {"pan", "frying-pan", "skillet", "stove", "pot", "microwave", "oven",
      "saucepan", "frypan"}),
    ("kill the fly", ("kill fly",), (), (), {"tool", "unaided"},
     {"flyswatter", "swatter", "fly-swatter", "insecticide", "spray",
      "newspaper"}),
    ("break the window", ("break window",), (), (), {"tool", "unaided"},
     {"hammer", "rock", "stone", "brick", "bat"}),
    ("cut john's hair", ("cut hair",), (), (), {"tool", "helper"},
     {"scissors", "razor", "shaver", "clippers", "barber",
      "hairdresser", "shears"}),
    # -- a state a thing is left in --------------------------------------
    ("make the milk cold", ("cold milk",), (), (), {"place"},
     {"refrigerator", "fridge", "freezer", "icebox", "cooler"}),
    ("make the milk cold; the fridge is in the kitchen", ("cold milk",),
     ("at fridge kitchen", "at milk table"), (), {"place"}, {"fridge"}),
    ("make the beer cold", ("cold beer",), (), (), {"place"},
     {"refrigerator", "fridge", "freezer", "icebox", "cooler"}),
    ("keep the food cold", ("cold food",), (), (), {"place"},
     {"refrigerator", "fridge", "freezer", "icebox", "cooler"}),
    ("make john warm", ("warm john",), (), ("john",), {"wear", "place",
                                                       "tool"},
     {"blanket", "coat", "jacket", "sweater", "heater", "fire", "house",
      "fireplace", "stove", "quilt", "scarf"}),
    ("warm the soup", ("warm soup",), (), (), {"tool", "place"},
     {"stove", "microwave", "oven", "pot", "saucepan", "pan", "heater"}),
    ("heat the soup", ("heat soup",), (), (), {"tool", "place"},
     {"stove", "microwave", "oven", "pot", "saucepan", "pan", "heater"}),
    ("dry john's hair", ("dry hair",), (), (), {"tool"},
     {"hair-dryer", "hairdryer", "blow-dryer", "towel", "dryer",
      "hand-blower"}),
    ("make the room bright", ("bright room",), (), (), {"tool"},
     {"lamp", "light", "candle", "lantern", "light-bulb", "bulb"}),
    ("make the door open", ("open door",), (), (), {"unaided"}, set()),
    # -- a doing one does oneself ---------------------------------------
    ("open the door", ("open door",), (), (), {"unaided"}, set()),
    ("hang the picture", ("hang picture",), (), (), {"unaided", "tool"},
     {"nail", "hook", "hammer", "picture-hook"}),
    ("write a letter", ("write letter",), (), (), {"unaided", "tool"},
     {"pen", "pencil", "typewriter", "computer", "ballpoint"}),
    # -- someone else's trade -------------------------------------------
    ("fix the car", ("fix car",), (), (), {"helper"},
     {"mechanic", "technician", "auto-mechanic", "machinist"}),
    ("fix the toilet", ("fix toilet",), (), (), {"helper"},
     {"plumber", "technician"}),
    ("pull the tooth", ("pull tooth",), (), (), {"helper", "tool"},
     {"dentist", "pliers", "forceps", "doctor"}),
    # -- places and havings, as v691 did ---------------------------------
    ("get the book to the kitchen", ("at book kitchen",),
     ("at book shop",), (), {"move"}, set()),
    ("give the key to john", ("with key john",), ("at key table",),
     ("john",), {"hand"}, set()),
    # -- more than one clause: what one way leaves, the next may use ----
    ("cut the rope and the string", ("cut rope", "cut string"), (), (),
     {"tool"}, CUTTERS),
    ("make the milk and the beer cold", ("cold milk", "cold beer"),
     ("at fridge kitchen",), (), {"place"}, {"fridge"}),
    ("dig a hole and cut the rope", ("dig hole", "cut rope"),
     ("at shovel shed",), (), {"tool"}, {"shovel"} | CUTTERS),
]


#: Written after the rules were settled on `BANK`, and run once: none of
#: what the designer does was changed after seeing these. They are the
#: generalisation number (DESIGN.md).
HELD = [
    ("cut the bread", ("cut bread",), (), (), {"tool"},
     {"knife", "bread-knife", "slicer", "cutter"}),
    ("cut the paper", ("cut paper",), (), (), {"tool"},
     {"scissors", "knife", "cutter", "paper-cutter", "blade", "razor"}),
    ("chop the wood", ("chop wood",), (), (), {"tool"},
     {"ax", "axe", "hatchet", "chopper", "saw", "cleaver"}),
    ("slice the cheese", ("slice cheese",), (), (), {"tool"},
     {"knife", "slicer", "cheese-slicer", "cutter"}),
    ("dig the garden", ("dig garden",), (), (), {"tool"},
     {"shovel", "spade", "hoe", "fork", "trowel", "pitchfork"}),
    ("light the candle", ("light candle",), (), (), {"tool"},
     {"match", "lighter", "matchstick", "candle", "taper"}),
    ("open the bottle", ("open bottle",), (), (), {"tool", "unaided"},
     {"bottle-opener", "opener", "corkscrew"}),
    ("unlock the car", ("unlocked car",), (), (), {"tool"}, {"key"}),
    ("clean the window", ("clean window",), (), (), {"tool", "helper"},
     {"cloth", "sponge", "squeegee", "rag", "towel", "washcloth",
      "window-cleaner", "cleaner", "paper-towel"}),
    ("wash the dishes", ("wash dishes",), (), (), {"tool", "unaided"},
     {"sponge", "dishwasher", "soap", "brush", "cloth", "dishcloth",
      "sink"}),
    ("dry the dishes", ("dry dishes",), (), (), {"tool", "place"},
     {"towel", "dish-towel", "dishtowel", "dryer", "dishwasher", "rack",
      "cloth", "dish-rack"}),
    ("dry the clothes", ("dry clothes",), (), (), {"tool", "place"},
     {"dryer", "clothes-dryer", "clothesline", "towel", "line"}),
    ("make the juice cold", ("cold juice",), (), (), {"place"},
     {"refrigerator", "fridge", "freezer", "icebox", "cooler"}),
    ("make mary warm", ("warm mary",), (), ("mary",),
     {"wear", "place", "tool"},
     {"blanket", "coat", "jacket", "sweater", "heater", "fire",
      "fireplace", "stove", "quilt", "scarf", "house"}),
    ("warm the milk", ("warm milk",), (), (), {"tool", "place"},
     {"stove", "microwave", "pot", "saucepan", "kettle", "pan", "oven",
      "heater"}),
    ("heat the room", ("heat room",), (), (), {"tool", "bring"},
     {"heater", "fireplace", "stove", "radiator", "furnace", "fire"}),
    ("cook the rice", ("cook rice",), (), (), {"tool"},
     {"pot", "rice-cooker", "cooker", "stove", "pan", "saucepan",
      "steamer", "microwave"}),
    ("boil the egg", ("boil egg",), (), (), {"tool"},
     {"pot", "saucepan", "kettle", "stove", "boiler", "pan"}),
    ("fry the egg", ("fry egg",), (), (), {"tool"},
     {"pan", "frying-pan", "skillet", "frypan", "stove", "fryer"}),
    ("bake the bread", ("bake bread",), (), (), {"tool", "place"},
     {"oven", "baker", "bakery", "toaster"}),
    ("sweep the kitchen", ("sweep kitchen",), (), (), {"tool"},
     {"broom", "brush", "sweeper"}),
    ("mop the floor", ("mop floor",), (), (), {"tool"}, {"mop", "swab"}),
    ("fix the sink", ("fix sink",), (), (), {"helper"}, {"plumber"}),
    ("fix the roof", ("fix roof",), (), (), {"helper"},
     {"roofer", "carpenter", "builder", "contractor"}),
    ("fix the computer", ("fix computer",), (), (), {"helper"},
     {"technician", "programmer", "engineer", "repairman"}),
    ("write the essay", ("write essay",), (), (), {"unaided", "tool"},
     {"pen", "pencil", "computer", "typewriter", "word-processor"}),
    ("tie the shoe", ("tie shoe",), (), (), {"unaided", "tool"},
     {"lace", "shoelace", "string"}),
    ("carry the water", ("carry water",), (), (), {"tool"},
     {"bucket", "pail", "jug", "container", "bottle", "can", "pitcher"}),
    ("wake mary; there is an alarm clock on the table", ("wake mary",),
     ("at alarm-clock table",), ("mary",), {"tool"}, {"alarm-clock"}),
    ("cut the bread; the knife is in the drawer", ("cut bread",),
     ("at knife drawer",), (), {"tool"}, {"knife"}),
    ("make the juice cold; the freezer is in the garage", ("cold juice",),
     ("at freezer garage",), (), {"place"}, {"freezer"}),
    ("cut the bread and the cheese", ("cut bread", "cut cheese"), (), (),
     {"tool"}, {"knife", "bread-knife", "slicer", "cutter"}),
    ("mow the grass", ("mow grass",), (), (), {"tool", "helper"},
     {"mower", "lawn-mower", "lawnmower", "scythe", "gardener"}),
    ("water the garden", ("water garden",), (), (), {"tool", "unaided"},
     {"hose", "sprinkler", "watering-can", "can", "pitcher"}),
]


#: Written before the second round of changes (similarity, building,
#: doings done in a place, parts of a thing), and run once after them:
#: `HELD` had shown what was missing, so it could no longer say whether
#: the fixes generalise. `inside` is the way the second round was to add
#: (do it with the thing inside ?P), named here before it existed.
HELD2 = [
    ("slice the bread", ("slice bread",), (), (), {"tool"},
     {"knife", "slicer", "bread-knife", "cutter"}),
    ("chop the onion", ("chop onion",), (), (), {"tool"},
     {"knife", "cleaver", "chopper", "food-processor"}),
    ("peel the potato", ("peel potato",), (), (), {"tool"},
     {"peeler", "knife", "paring-knife", "potato-peeler"}),
    ("grate the cheese", ("grate cheese",), (), (), {"tool"},
     {"grater", "cheese-grater", "food-processor"}),
    ("stir the soup", ("stir soup",), (), (), {"tool"},
     {"spoon", "ladle", "stirrer", "wooden-spoon", "whisk"}),
    ("whisk the eggs", ("whisk eggs",), (), (), {"tool"},
     {"whisk", "beater", "fork", "mixer", "eggbeater"}),
    ("cut the grass", ("cut grass",), (), (), {"tool", "helper"},
     {"mower", "lawn-mower", "lawnmower", "scythe", "shears",
      "scissors", "gardener", "sickle", "trimmer"}),
    ("trim the hedge", ("trim hedge",), (), (), {"tool", "helper"},
     {"shears", "clippers", "trimmer", "hedge-trimmer", "gardener",
      "pruner"}),
    ("saw the log", ("saw log",), (), (), {"tool"},
     {"saw", "chainsaw", "handsaw"}),
    ("drill the wall", ("drill wall",), (), (), {"tool", "helper"},
     {"drill", "power-drill", "electric-drill"}),
    ("tighten the screw", ("tighten screw",), (), (), {"tool"},
     {"screwdriver", "wrench", "spanner"}),
    ("hammer the nail", ("hammer nail",), (), (), {"tool"},
     {"hammer"}),
    ("wash the car", ("wash car",), (), (), {"tool", "helper", "inside"},
     {"hose", "sponge", "soap", "bucket", "car-wash", "brush", "cloth"}),
    ("wash the clothes", ("wash clothes",), (), (),
     {"tool", "inside", "place"},
     {"washer", "washing-machine", "soap", "detergent", "laundry",
      "washboard", "sink", "tub"}),
    ("toast the bread", ("toast bread",), (), (), {"tool", "inside",
                                                   "place"},
     {"toaster", "oven", "grill", "toaster-oven", "fire"}),
    ("freeze the meat", ("freeze meat",), (), (), {"tool", "inside",
                                                   "place"},
     {"freezer", "refrigerator", "fridge", "deep-freeze", "icebox"}),
    ("make the water cold", ("cold water",), (), (), {"place", "tool"},
     {"refrigerator", "fridge", "freezer", "cooler", "ice", "icebox"}),
    ("make the baby warm", ("warm baby",), (), (), {"wear", "tool",
                                                   "place"},
     {"blanket", "coat", "jacket", "sweater", "heater", "quilt",
      "incubator", "fire", "fireplace"}),
    ("light the fire", ("light fire",), (), (), {"tool"},
     {"match", "lighter", "matchstick", "flint", "torch"}),
    ("clean the carpet", ("clean carpet",), (), (), {"tool", "helper"},
     {"vacuum", "vacuum-cleaner", "brush", "cleaner", "shampoo",
      "carpet-sweeper", "sweeper"}),
    ("polish the shoes", ("polish shoes",), (), (), {"tool"},
     {"polish", "shoe-polish", "brush", "cloth", "rag"}),
    ("dry the hands", ("dry hands",), (), (), {"tool"},
     {"towel", "dryer", "hand-blower", "paper-towel", "hand-towel"}),
    ("fix the bike", ("fix bike",), (), (), {"helper", "tool"},
     {"mechanic", "technician", "wrench", "repairman"}),
    ("fix the pipe", ("fix pipe",), (), (), {"helper"},
     {"plumber", "technician"}),
    ("cut the wire", ("cut wire",), (), (), {"tool"},
     {"pliers", "wire-cutter", "cutter", "scissors", "knife", "cutters",
      "clippers"}),
    ("sharpen the pencil", ("sharpen pencil",), (), (), {"tool"},
     {"sharpener", "pencil-sharpener", "knife"}),
    ("heat the water", ("heat water",), (), (), {"tool", "place",
                                                 "inside"},
     {"kettle", "stove", "heater", "pot", "boiler", "microwave",
      "teakettle", "saucepan"}),
    ("cook the pasta", ("cook pasta",), (), (), {"tool", "inside"},
     {"pot", "stove", "saucepan", "pan", "cooker"}),
    ("bake the cake", ("bake cake",), (), (), {"tool", "inside",
                                               "place"},
     {"oven", "baker", "stove"}),
    ("cut the wire; the pliers are in the drawer", ("cut wire",),
     ("at pliers drawer",), (), {"tool"}, {"pliers"}),
    ("light the fire; there is a match on the shelf", ("light fire",),
     ("at match shelf",), (), {"tool"}, {"match"}),
]


def _one_of(name: str, allowed: set) -> bool:
    if name in allowed:
        return True
    kinds = K.kinds_of(name)
    return any(one.replace("-", " ") in kinds for one in allowed)


def score(design: designing.Design, ways: set, means: set) -> tuple:
    """(sensible, simplest): every clause met in one of the ways allowed,
    with one of the things allowed; and nothing supposed that the scene
    already had."""
    if not design.done:
        return False, False
    sensible = True
    for part in design.parts:
        best = part.best
        if best.form not in ways:
            sensible = False
        elif means and best.means and not _one_of(best.means, means):
            sensible = False
        elif means and not best.means and best.form not in ("unaided",
                                                            "move",
                                                            "hand"):
            sensible = False
    scene = design.goal.things()
    simplest = not any(thing in scene for thing, _ in design.supposed())
    fetched = [step.name for step in design.steps
               if step.name.split()[0] in ("get", "take", "find")]
    simplest = simplest and len(fetched) == len(set(
        " ".join(name.split()[:2]) for name in fetched))
    return sensible, simplest


def run(bank=BANK, learner=None, order=None, verbose=True) -> dict:
    right = simple = 0
    rows = []
    started = time.time()
    for said, wants, scene, names, ways, means in bank:
        goal = Goal(tuple(wants), frozenset(scene), frozenset(names),
                    said=said)
        found = designing.design(goal, learner=learner, order=order)
        sensible, simplest = score(found, ways, means)
        right += sensible
        simple += sensible and simplest
        rows.append((said, found, sensible, simplest))
        if verbose:
            mark = "ok " if sensible else "NO "
            print(f"{mark}{'   ' if simplest else '(+)'} {said}\n"
                  f"       {found.said()}")
    if verbose:
        print(f"\n{right}/{len(bank)} sensible, {simple} also simplest, "
              f"{time.time() - started:.1f}s")
    return {"right": right, "simple": simple, "total": len(bank),
            "rows": rows}


def main(argv=None) -> int:
    """`--held` for the held-out bank; `--proposer` to try ways in the
    learned proposer's order (`proposer.py`), which should change how
    many ways are fitted and never what is chosen."""
    argv = list(sys.argv[1:] if argv is None else argv)
    K.index()
    order = None
    if "--proposer" in argv:
        from research.v694.proposer import Proposer
        learned = Proposer.load()
        order = learned.order if learned is not None else None
    bank = HELD2 if "--held2" in argv else HELD if "--held" in argv         else BANK
    found = run(bank, order=order)
    fitted = sum(len(part.fitted) for _, design, _, _ in found["rows"]
                 for part in design.parts)
    print(f"ways fitted: {fitted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

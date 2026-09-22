"""How far the open world gets with arithmetic in a story.

Word problems, hand-written, each told a sentence at a time to a fresh scene
the way a person would say them on the page (`page.say_to`), and the last
sentence a question. Nothing here was written for the reader or the planner:
the counts are read off the parse, what each verb does to one is VerbNet's
(`hearing._moved`), and the sums are `numbers.py`'s.

    python -m research.v691.sums

A problem is **right** when the answer states the expected number, and
**honest** when it is right or says it does not know -- the one outcome that
must not happen is a number that is wrong. The families are the shapes
children's arithmetic problems come in, not a benchmark: transfer, using up,
finding, arriving and leaving, totals, comparisons, orders carried out, and
bare sums.
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from fractions import Fraction

from research.v691 import numbers, page


@dataclass
class Problem:
    family: str
    said: tuple
    #: the number the last answer should state
    expected: Fraction
    answer: str = ""
    right: bool = False
    honest: bool = False
    seconds: float = 0.0
    replies: list = field(default_factory=list)


def P(family: str, *said: str, expected) -> Problem:
    """A problem; `expected` is a number, or who the answer names."""
    return Problem(family, said, expected if isinstance(expected, str)
                   else Fraction(expected))


PROBLEMS = [
    # -- transfer: something moves from one holder to another
    P("transfer", "mary has 5 apples", "she gave 2 apples to john",
      "how many apples does mary have", expected=3),
    P("transfer", "mary has 5 apples", "john has 1 apple",
      "mary gave john 3 apples", "how many apples does john have",
      expected=4),
    P("transfer", "i have 12 stickers", "i gave 5 stickers to ben",
      "how many stickers do i have", expected=7),
    P("transfer", "there are 20 books on the shelf",
      "anna took 6 books from the shelf",
      "how many books are on the shelf", expected=14),
    P("transfer", "tom has 9 cards", "tom gave 4 cards to sam",
      "sam gave 1 card to tom", "how many cards does tom have",
      expected=6),
    # -- using up
    P("using up", "tom has 8 cookies", "he ate 3 cookies",
      "how many cookies does tom have", expected=5),
    P("using up", "jack had 15 marbles", "he lost 6 marbles",
      "how many marbles does jack have now", expected=9),
    P("using up", "there are 30 sweets in the jar",
      "lucy took 4 sweets from the jar", "she ate 2 sweets",
      "how many sweets does lucy have", expected=2),
    # -- finding and getting
    P("getting", "sara has 4 shells", "she found 6 shells",
      "how many shells does she have", expected=10),
    P("getting", "ben has 2 pencils", "he bought 5 pencils",
      "how many pencils does ben have", expected=7),
    P("getting", "lily had 3 books", "her mom gave her 4 books",
      "how many books does lily have", expected=7),
    # -- arriving and leaving
    P("moving", "there are 9 birds in the tree", "4 birds flew away",
      "how many birds are in the tree", expected=5),
    P("moving", "there are 3 ducks in the pond",
      "5 ducks landed in the pond", "how many ducks are in the pond",
      expected=8),
    P("moving", "there are 7 fish in the tank", "2 fish swam away",
      "how many fish are left", expected=5),
    # -- totals
    P("total", "mary has 3 apples", "john has 4 apples",
      "how many apples do mary and john have together", expected=7),
    P("total", "there are 6 red balls in the box",
      "i have 4 balls", "how many balls do i have", expected=4),
    # -- comparing
    P("compare", "tom has 12 marbles", "sam has 7 marbles",
      "how many more marbles does tom have than sam", expected=5),
    P("compare", "anna has 3 dolls", "beth has 8 dolls",
      "how many fewer dolls does anna have than beth", expected=5),
    # -- orders carried out, and then asked about
    P("order", "i have 5 apples", "give mary 2 apples",
      "how many apples do i have", expected=3),
    P("order", "i have 2 apples", "there are 10 apples in the basket",
      "give john 6 apples", "how many apples are in the basket",
      expected=6),
    P("order", "i have 6 pears", "eat 2 pears",
      "how many pears do i have", expected=4),
    # -- bare sums
    P("sum", "what is 23 plus 19", expected=42),
    P("sum", "what is 7 times 8 minus 6", expected=50),
    P("sum", "what is half of 18", expected=9),
]

#: Written after `PROBLEMS` passed and **not tuned on**: other phrasings,
#: other verbs, and shapes nothing here reads yet (`each`, `twice as many`),
#: where the honest answer is not knowing. `--held` runs these.
HELD = [
    P("transfer", "peter has 10 coins", "peter handed 3 coins to kate",
      "how many coins does peter have", expected=7),
    P("transfer", "kate has 2 coins", "peter handed kate 3 coins",
      "how many coins does kate have", expected=5),
    P("transfer", "the box holds 14 crayons",
      "emma removed 5 crayons from the box",
      "how many crayons are in the box", expected=9),
    P("transfer", "i have 8 stamps", "i sent 3 stamps to my aunt",
      "how many stamps do i have now", expected=5),
    P("using up", "the farmer had 25 eggs", "he sold 10 eggs",
      "how many eggs does the farmer have", expected=15),
    P("using up", "zoe has 6 cupcakes", "zoe ate two cupcakes",
      "how many cupcakes does zoe have left", expected=4),
    P("using up", "max had 11 balloons", "3 balloons popped",
      "how many balloons does max have", expected=8),
    P("getting", "nina has 5 stickers", "nina won 4 stickers",
      "how many stickers does nina have", expected=9),
    P("getting", "leo had 7 cards", "leo received 6 cards",
      "how many cards does leo have now", expected=13),
    P("moving", "there are 12 sheep in the field",
      "5 sheep ran away", "how many sheep are in the field", expected=7),
    P("moving", "there are 4 cars in the car park",
      "6 cars arrived at the car park",
      "how many cars are in the car park", expected=10),
    P("total", "ann has 4 pens", "bob has 5 pens", "cal has 1 pen",
      "how many pens do ann , bob and cal have altogether", expected=10),
    P("compare", "ivy has 9 beads", "jo has 13 beads",
      "how many more beads does jo have than ivy", expected=4),
    P("compare", "ivy has 9 beads", "jo has 13 beads",
      "who has more beads , ivy or jo", expected="jo"),
    P("order", "i have 7 oranges", "give sam 3 oranges",
      "how many oranges does sam have", expected=3),
    P("sum", "how much is 144 divided by 12", expected=12),
    P("sum", "what is 15 percent of 200", expected=30),
    # shapes nothing here reads yet: not knowing is the honest answer
    P("beyond", "each box has 6 eggs", "there are 4 boxes",
      "how many eggs are there", expected=24),
    P("beyond", "tim has 3 apples", "sue has twice as many apples as tim",
      "how many apples does sue have", expected=6),
    P("beyond", "a pencil costs 2 dollars", "how much do 5 pencils cost",
      expected=10),
]

#: Written after the fixes `HELD` prompted, and run once as it stands: the
#: number that says whether those fixes were general. `--fresh` runs these.
FRESH = [
    P("transfer", "grace has 6 ribbons", "grace passed 2 ribbons to hannah",
      "how many ribbons does grace have", expected=4),
    P("transfer", "the bowl has 9 plums", "i took 4 plums out of the bowl",
      "how many plums are in the bowl", expected=5),
    P("transfer", "omar had 20 dollars", "he paid 8 dollars to the shop",
      "how many dollars does omar have", expected=12),
    P("transfer", "there are 5 cups on the table",
      "dad put 3 cups on the table", "how many cups are on the table",
      expected=8),
    P("transfer", "rosa has 3 keys", "she lent 1 key to ali",
      "how many keys does rosa have", expected=2),
    P("using up", "the baker had 40 rolls", "he sold 25 rolls",
      "how many rolls does the baker have left", expected=15),
    P("using up", "i had 10 candles", "i burned 4 candles",
      "how many candles do i have", expected=6),
    P("using up", "there were 8 cookies in the jar",
      "the children ate 5 cookies", "how many cookies are in the jar",
      expected=3),
    P("getting", "ella had 12 stamps", "her friend gave her 5 stamps",
      "how many stamps does ella have", expected=17),
    P("getting", "sam has 3 fish", "sam caught 4 fish",
      "how many fish does sam have now", expected=7),
    P("moving", "there were 15 people on the bus",
      "6 people got off the bus", "how many people are on the bus",
      expected=9),
    P("moving", "there are 2 cats in the garden",
      "3 cats came into the garden", "how many cats are in the garden",
      expected=5),
    P("total", "the red box has 7 toys", "the blue box has 5 toys",
      "how many toys are in the red box", expected=7),
    P("compare", "mia read 12 pages", "leo read 8 pages",
      "how many more pages did mia read than leo", expected=4),
    P("order", "i have 9 marbles", "give tom 4 marbles",
      "how many marbles do i have", expected=5),
    P("sum", "what is 9 squared", expected=81),
    P("sum", "what is 100 minus 37", expected=63),
]

#: A number in an answer: `3`, `2.5`, `1/3`, or a word.
NUMBER = re.compile(r"-?\d+(?:/\d+|\.\d+)?")
DONT_KNOW = re.compile(r"\b(?:do not know|don't know|cannot|could not|"
                       r"did not take that|lost count)\b")


def stated(answer: str) -> list:
    """The numbers an answer states, in order."""
    out = []
    for found in NUMBER.findall(answer):
        value = numbers.value(found)
        if value is not None:
            out.append(value)
    return out


def run(problem: Problem) -> Problem:
    from research.v691.learned import Learned
    from research.v691.openworld import Open, resolver
    from research.v691.scene import Scene
    scene = Scene(Open(resolver(), Learned(None)))
    started = time.time()
    for line in problem.said:
        problem.replies.append(page.say_to(scene, line))
    problem.seconds = time.time() - started
    problem.answer = problem.replies[-1]
    if isinstance(problem.expected, str):
        problem.right = problem.answer.lower().startswith(problem.expected)
    else:
        found = stated(problem.answer)
        problem.right = bool(found) and found[0] == problem.expected
    problem.honest = problem.right or bool(DONT_KNOW.search(problem.answer))
    return problem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="",
                        help="only the problems of one family")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="every reply, not only the answers")
    parser.add_argument("--held", action="store_true",
                        help="the problems written after the others passed")
    parser.add_argument("--fresh", action="store_true",
                        help="the problems written after the held ones")
    options = parser.parse_args(argv)
    chosen = [one for one in (HELD if options.held else FRESH
                              if options.fresh else PROBLEMS)
              if not options.family or one.family == options.family]
    by_family: dict = {}
    for problem in chosen:
        run(problem)
        mark = "ok " if problem.right else ("?  " if problem.honest
                                           else "NO ")
        print(f"{mark} [{problem.family}] {' / '.join(problem.said)}")
        if options.verbose:
            for line, reply in zip(problem.said, problem.replies):
                print(f"      > {line}\n        {reply}")
        elif not problem.right:
            expected = (problem.expected if isinstance(problem.expected, str)
                        else numbers.said(problem.expected))
            print(f"      said: {problem.answer}  (expected {expected})")
        got = by_family.setdefault(problem.family, [0, 0, 0])
        got[0] += 1
        got[1] += problem.right
        got[2] += problem.honest
    print()
    print(f"{'family':<12} {'right':>7} {'honest':>7}")
    for family, (total, right, honest) in by_family.items():
        print(f"{family:<12} {right:>3}/{total:<3} {honest:>3}/{total:<3}")
    total = sum(one[0] for one in by_family.values())
    right = sum(one[1] for one in by_family.values())
    honest = sum(one[2] for one in by_family.values())
    print(f"{'all':<12} {right:>3}/{total:<3} {honest:>3}/{total:<3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

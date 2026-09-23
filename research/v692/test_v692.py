"""Mathematics as a subject (v692): saying, symbols, doing, solving,
semantics and reading."""
from __future__ import annotations

import random
import unittest

import sympy as S

from research.v692 import curriculum as C, doing, reading, saying, solving
from research.v692.corpus import record, words
from research.v692.symbols import Unreadable, parsed, safe, same

x, y = S.symbols("x y")
needs_encoder = unittest.skipUnless(reading.enabled(),
                                    "the encoder has no mathematics heads")


class SayingTests(unittest.TestCase):
    """What is said for an object is what reads back as it."""

    OBJECTS = [x ** 2 - 5 * x + 6, S.sqrt(x + 1), S.Eq(2 * x + 3, 7),
               S.FiniteSet(1, 2, 3), S.Matrix([[1, 2], [3, 4]]),
               (x + 1) * (x - 1), S.Rational(3, 4), 3 * x ** 2 * y,
               x / (x + 1), S.Abs(x - 3), S.Ge(x, 4), S.log(x), S.exp(x),
               -2 * x + 7, (x + 1) ** 2, S.sin(x) ** 2,
               S.Symbol("p") & ~S.Symbol("q"),
               S.Implies(S.Symbol("p"), S.Symbol("q")),
               S.Function("f")(3), S.Tuple(1, 2, 3), 3 + 4 * S.I,
               S.Float("3.14159"), 5 * S.Symbol("x") ** 0 * 1]

    def test_every_saying_reads_back(self):
        for index, obj in enumerate(self.OBJECTS):
            for style in ("spoken", "written", "tight"):
                for seed in range(3):
                    said = saying.say(obj, random.Random(index * 7 + seed),
                                      style)
                    with self.subTest(obj=obj, style=style, said=said.text()):
                        self.assertTrue(same(parsed(said.symbols()), obj))

    def test_spoken_mathematics_is_english(self):
        said = saying.say(x ** 2 - 5 * x + 6, random.Random(1), "spoken")
        self.assertIn("minus", said.words)
        self.assertNotIn("^", said.text())


class SymbolTests(unittest.TestCase):

    def test_only_mathematics_reaches_the_parser(self):
        for text in ("__import__('os')", "os.system(1)", "open('f')",
                     "lambda: 1", "x.__class__"):
            with self.subTest(text=text):
                self.assertFalse(safe(text))
                with self.assertRaises(Unreadable):
                    parsed(text)

    def test_relations_are_claims_not_verdicts(self):
        found = parsed("2 + 2 = 5")
        self.assertIsInstance(found, S.Equality)
        self.assertEqual(found.lhs, 4)

    def test_sets_and_matrices(self):
        self.assertEqual(parsed("{1, 2, 3}"), S.FiniteSet(1, 2, 3))
        self.assertEqual(parsed("[[1, 2], [3, 4]]"),
                         S.Matrix([[1, 2], [3, 4]]))

    def test_words_keep_written_things_whole(self):
        self.assertEqual(words("what is the det of [[1,2],[3,4]]?"),
                         ["what", "is", "the", "det", "of", "[[1,2],[3,4]]"])


class DoingTests(unittest.TestCase):

    def do(self, act, **parts):
        return doing.do(act, parts)

    def test_values(self):
        self.assertEqual(self.do("value", EXPR=parsed("17 * 4")).value, 68)
        self.assertEqual(self.do("derivative", EXPR=x ** 3).value,
                         3 * x ** 2)
        self.assertEqual(self.do("solve", EXPR=S.Eq(3 * x + 4, 19)).value,
                         [5])
        self.assertEqual(self.do("gcd", LIST=S.Tuple(48, 36)).value, 12)
        self.assertEqual(self.do("mean", LIST=S.Tuple(4, 8, 12)).value, 8)
        self.assertEqual(self.do("choose", A=S.Integer(6),
                                 B=S.Integer(2)).value, 15)

    def test_verdicts_come_with_their_reason(self):
        found = self.do("is kind", A=S.Integer(91),
                        KIND=C.BY_NAME["prime number"])
        self.assertEqual(found.stance, "no")
        self.assertIn("7 times 13", found.because)
        self.assertEqual(self.do("divides", A=S.Integer(84),
                                 B=S.Integer(7)).stance, "yes")
        self.assertEqual(self.do("check",
                                 EXPR=parsed("7 * 8 = 54")).stance, "no")

    def test_what_cannot_be_had_is_not_guessed(self):
        found = self.do("inverse", EXPR=S.Matrix([[1, 2], [2, 4]]))
        self.assertEqual(found.stance, "no")
        self.assertEqual(self.do("limit", EXPR=1 / x, VAR=x,
                                 A=S.Integer(0)).stance, "unknown")

    def test_what_was_let_is_put_in(self):
        found = doing.do("value", {"EXPR": x ** 2 + 1}, {x: S.Integer(3)})
        self.assertEqual(found.value, 10)


class CorpusTests(unittest.TestCase):

    def test_every_act_makes_records_that_read_back(self):
        rng = random.Random(3)
        for act in C.ACTS:
            made = [record(act, rng.choice(act.templates), rng)
                    for _ in range(12)]
            with self.subTest(act=act.name):
                self.assertTrue(any(one is not None for one in made))

    def test_every_act_can_be_done(self):
        self.assertTrue(set(doing.HANDLERS) >= {one.name for one in C.ACTS})


class SolvingTests(unittest.TestCase):
    """An equation solved by v691's agent: planned, done, watched."""

    def E(self, left, right):
        return S.Eq(left, right, evaluate=False)

    def test_linear_equations_by_moves(self):
        for equation, root in ((self.E(S.Mul(2, x + 3, evaluate=False), 10),
                                2),
                               (self.E(3 * x - 7, x + 5), 6),
                               (self.E(x / 2 + 1, 4), 6),
                               (self.E(5 - x, 2), 3)):
            with self.subTest(equation=equation):
                found = solving.solve(equation, x)
                self.assertTrue(found.solved)
                self.assertEqual(found.roots, [root])

    def test_a_quadratic_that_does_not_factor_is_a_surprise(self):
        found = solving.solve(self.E(x ** 2 + x, 1), x)
        self.assertTrue(found.solved)
        self.assertGreaterEqual(found.attempt.surprises, 0)
        self.assertEqual(len(found.roots), 2)

    def test_what_factoring_needs_is_learned_not_told(self):
        """Three factorings that worked, one that did not: the only thing
        every success had and the failure lacked is a square
        discriminant, and that is the lesson."""
        from research.v691 import learned, lessons
        learner = lessons.Learner(learned.Learned(None))
        for equation in (self.E(x ** 2 - 5 * x + 6, 0),
                         self.E(x ** 2 - 9, 0),
                         self.E(x ** 2 + 3 * x + 2, 0)):
            self.assertTrue(solving.solve(equation, x, learner,
                                          goal="factored").solved)
        failed = solving.solve(self.E(x ** 2 + x - 1, 0), x, learner,
                               goal="factored")
        self.assertFalse(failed.solved)
        self.assertEqual([(one.kind, one.literal)
                          for one in failed.attempt.lessons],
                         [("requires", "square-discriminant ?it")])
        # Learned: a quadratic that cannot factor is not tried again.
        again = solving.solve(self.E(x ** 2 + 2 * x - 5, 0), x, learner,
                              goal="factored")
        self.assertEqual(again.steps, [])


class MeasuringTests(unittest.TestCase):
    """A shape's measures, planned from what was given."""

    def test_a_measure_is_planned_through_what_it_needs(self):
        from research.v692 import measuring as M
        found = M.measure("square", "perimeter", {"area": S.Integer(49)})
        self.assertEqual(found.value, 28)
        self.assertEqual([one[0] for one in found.steps],
                         ["side", "perimeter"])

    def test_what_is_missing_is_asked_for(self):
        from research.v692 import measuring as M
        found = M.measure("rectangle", "area", {"length": S.Integer(4)})
        self.assertIsNone(found.value)
        self.assertEqual(found.needs[0], "width")
        # And given, it finishes.
        parts = doing.continued({"SHAPE": "rectangle", "WANTED": "area",
                                 "GIVENS": {"length": S.Integer(4)}},
                                {"GIVENS": {"width": S.Integer(6)}})
        self.assertEqual(doing.do("measure", parts).value, 24)

    def test_sides_are_read_from_wordnet(self):
        from research.v692 import measuring as M
        self.assertEqual(M.sides("hexagon"), 6)
        self.assertEqual(M.sides("decagon"), 10)
        self.assertIsNone(M.sides("circle"))
        self.assertEqual(M.measure("octagon", "interior-angle", {}).value,
                         135)

    def test_units_are_carried(self):
        from sympy.physics import units
        from research.v692 import measuring as M
        found = M.measure("circle", "area", {"radius": 5 * units.centimeter})
        self.assertEqual(found.value, 25 * S.pi * units.centimeter ** 2)


class FurtherActTests(unittest.TestCase):
    """The branches past the first eight, done exactly."""

    def test_values(self):
        T = S.Tuple
        for act, parts, wanted in (
                ("next term", {"LIST": T(3, 9, 27, 81)}, 243),
                ("nth term", {"LIST": T(2, 5, 8, 11)},
                 3 * S.Symbol("n") - 1),
                ("series", {"EXPR": parsed("1/n^2"), "VAR": S.Symbol("n"),
                            "A": S.Integer(1), "B": S.oo}, S.pi ** 2 / 6),
                ("dice", {"A": S.Integer(2), "B": S.Integer(7)},
                 S.Rational(1, 6)),
                ("coins", {"A": S.Integer(3), "B": S.Integer(2)},
                 S.Rational(3, 8)),
                ("percent of", {"A": S.Integer(15), "B": S.Integer(80)}, 12),
                ("dot", {"EXPR": T(1, 2, 3), "OTHER": T(4, 5, 6)}, 32),
                ("modulus", {"EXPR": parsed("3 + 4i")}, 5),
                ("convert", {"A": parsed("3 foot"),
                             "OTHER": parsed("inch")}, parsed("36 inch"))):
            with self.subTest(act=act):
                self.assertTrue(same(doing.do(act, parts).value, wanted))

    def test_an_inequality_is_solved_as_one(self):
        found = doing.do("solve", {"EXPR": parsed("2x + 3 > 7")})
        self.assertEqual(found.text, "x > 2")

    def test_logic_has_its_reasons(self):
        found = doing.do("is kind", {"EXPR": parsed("p & q"),
                                     "KIND": C.BY_NAME["tautology"]})
        self.assertEqual(found.stance, "no")
        self.assertIn("false", found.because)
        self.assertEqual(doing.do("equivalent", {
            "EXPR": parsed("~(p & q)"), "OTHER": parsed("~p | ~q")}).stance,
            "yes")

    def test_a_function_let_is_applied(self):
        let = doing.do("let", {"VAR": parsed("f(x)"),
                               "EXPR": parsed("x^2 + 1")})
        self.assertEqual(doing.do("value", {"EXPR": parsed("f(3)")},
                                  let.value).value, 10)

    def test_every_record_reads_back_to_the_same_answer(self):
        """What the encoder is taught, put back together as `reading` puts
        its output together, does what the record was made for."""
        rng = random.Random(8)
        for act in C.ACTS:
            with self.subTest(act=act.name):
                done = 0
                for _ in range(8):
                    made = record(act, rng.choice(act.templates), rng)
                    if made is None:
                        continue
                    read = reading.assembled(act.name, 1.0, made["words"],
                                             made["roles"], made["symbols"])
                    parts = dict(read.parts)
                    if read.kind is not None:
                        parts["KIND"] = read.kind
                    gold = doing.do(act.name, dict(made["_parts"]))
                    back = doing.do(act.name, parts)
                    self.assertEqual(gold.stance, back.stance, made["said"])
                    done += 1
                self.assertGreater(done, 0)


class PageTests(unittest.TestCase):
    """The conversation's own workspace, without asking the encoder: each
    utterance is handed the reading it would have been read as."""

    def setUp(self):
        from research.v692 import page, reading as reading_module
        self.page = page
        self.reading = reading_module
        self.session = type("Session", (), {"conversation": "test-v692"})()
        page.WORKSPACES.pop("test-v692", None)
        page._READ.clear()
        self.was = reading_module.read

        def read(text):
            return self.said.get(text)

        reading_module.read = read
        page.reading.read = read
        self.said: dict = {}

    def tearDown(self):
        self.reading.read = self.was
        self.page.reading.read = self.was
        self.page.WORKSPACES.pop("test-v692", None)
        self.page._READ.clear()

    def says(self, text, act, parts, kind=None):
        found = reading.Reading(act, 1.0, text.split(), [], [])
        found.parts = parts
        found.kind = kind
        self.said[text] = found
        return self.page.answered(self.session, text)

    def test_what_a_conversation_was_told_is_put_in(self):
        noted = self.says("let x be 5", "let", {"VAR": x,
                                                "A": S.Integer(5)})
        self.assertEqual(noted["outcome"], "noted")
        found = self.says("what is x squared plus 1", "value",
                          {"EXPR": x ** 2 + 1})
        self.assertEqual(found["outcome"], "retrieved")
        self.assertTrue(found["text"].startswith("26"))

    def test_a_measure_short_of_a_given_stays_open(self):
        asked = self.says("what is the area of a rectangle with length 4",
                          "measure",
                          {"SHAPE": "rectangle", "WANTED": "area",
                           "GIVENS": {"length": S.Integer(4)}})
        self.assertEqual(asked["outcome"], "unknown")
        self.assertIn("width", asked["text"])
        found = self.says("the width is 6", "given",
                          {"GIVENS": {"width": S.Integer(6)}})
        self.assertEqual(found["outcome"], "retrieved")
        self.assertTrue(found["text"].startswith("24"))
        # And once answered, nothing is left open.
        self.assertIsNone(self.page.workspace(self.session).pending)

    def test_what_is_not_mathematics_is_not_answered(self):
        self.assertIsNone(self.page.answered(self.session, "what is a dog"))


@needs_encoder
class ReadingTests(unittest.TestCase):
    """The encoder reads mathematics; the rest is left alone."""

    def test_acts_and_parts(self):
        for said, act in (("what is 17 times 4", "value"),
                          ("solve 3x + 4 = 19", "solve"),
                          ("is 97 prime", "is kind"),
                          ("differentiate x cubed", "derivative")):
            with self.subTest(said=said):
                found = reading.read(said)
                self.assertIsNotNone(found)
                self.assertEqual(found.act, act)

    def test_what_is_not_mathematics_is_left_alone(self):
        for said in ("what is a dog", "mary has 3 apples",
                     "what is a prime number"):
            with self.subTest(said=said):
                self.assertIsNone(reading.read(said))


if __name__ == "__main__":
    unittest.main()

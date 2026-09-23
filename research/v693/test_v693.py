"""Designing towards a goal (v693): specifications, forms, the designer."""
from __future__ import annotations

import unittest

import sympy as S

from research.v693 import designing, forms
from research.v693.spec import C, Spec, x

n = S.Symbol("n")


def P(*clauses) -> Spec:
    return Spec("polynomial", clauses)


def Q(*clauses) -> Spec:
    return Spec("sequence", clauses)


class SpecTests(unittest.TestCase):
    """A specification is checked exactly, and says what it is like."""

    def test_holds_is_exact(self):
        spec = P(C("root", 2), C("root", -3), C("value", 0, 12))
        self.assertTrue(spec.holds(-2 * (x - 2) * (x + 3)))
        self.assertFalse(spec.holds((x - 2) * (x + 3)))

    def test_nothing_is_not_a_design(self):
        """The zero polynomial has every root and whole coefficients."""
        self.assertFalse(P(C("root", 1), C("integer")).holds(S.Integer(0)))

    def test_features_describe_the_data(self):
        geometric = Q(C("term", 1, 2), C("term", 2, 6), C("term", 3, 18))
        self.assertIn("even-ratios", geometric.features())
        self.assertNotIn("even-steps", geometric.features())
        self.assertIn("same-sign-terms", geometric.features())
        stepped = Q(C("term", 1, 10), C("term", 2, 7), C("term", 3, 4))
        self.assertIn("even-steps", stepped.features())


class FitTests(unittest.TestCase):
    """A form's unknowns decided by the specification."""

    def test_a_scale_is_fixed_by_a_value(self):
        spec = P(C("root", 2), C("root", -3), C("value", 0, 12))
        self.assertEqual(designing.fit(forms.BY_NAME["roots"], spec),
                         S.expand(-2 * (x - 2) * (x + 3)))

    def test_a_form_that_cannot_meet_it_fits_nothing(self):
        spec = Q(C("term", 1, 1), C("term", 2, 4), C("term", 3, 9))
        self.assertIsNone(designing.fit(forms.BY_NAME["arithmetic"], spec))
        self.assertEqual(designing.fit(forms.BY_NAME["polynomial"], spec),
                         n ** 2)

    def test_what_no_equation_decides_is_chosen(self):
        """Whole-number coefficients is only checked; a scale of 2 meets it
        for roots 1/2 and 3, and nothing solved for it."""
        spec = P(C("root", "1/2"), C("root", 3), C("integer"))
        found = designing.fit(forms.BY_NAME["roots"], spec)
        self.assertTrue(spec.holds(found))
        self.assertEqual(S.Poly(found, x).LC(), 2)


class DesignTests(unittest.TestCase):
    """v691's agent, proposing forms until one meets the specification."""

    def test_designs_are_checked(self):
        spec = P(C("root", 1), C("value", 0, 2), C("value", 2, 6))
        found = designing.design(spec)
        self.assertTrue(found.done)
        self.assertTrue(spec.holds(found.design))

    def test_what_cannot_be_met_is_said_so(self):
        spec = P(C("degree", 1), C("root", 1), C("root", 2))
        found = designing.design(spec)
        self.assertFalse(found.done)
        self.assertIn("could not", found.said())

    def test_a_failed_form_is_not_proposed_again(self):
        spec = Q(C("term", 1, 1), C("term", 2, 4), C("term", 3, 9),
                 C("term", 4, 16))
        found = designing.design(spec)
        names = [name for name, _ in found.tried]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(found.design, n ** 2)

    def test_the_proposer_order_is_tried_first(self):
        spec = Q(C("term", 1, 2), C("term", 2, 6), C("term", 3, 18))
        found = designing.design(spec, order=["geometric"])
        self.assertEqual([name for name, _ in found.tried], ["geometric"])

    def test_when_a_form_works_is_learned_not_told(self):
        """Geometric sequences that worked had terms one ratio apart; the
        one that failed had not. That is the lesson, and after it the
        geometric form is not tried where the ratios are uneven."""
        from research.v691 import learned, lessons
        learner = lessons.Learner(learned.Learned(None))
        for terms in ((2, 6, 18), (5, 15, 45), (3, 6, 12)):
            spec = Q(*(C("term", k + 1, v) for k, v in enumerate(terms)))
            self.assertTrue(designing.design(
                spec, learner, order=["geometric"]).done)
        uneven = Q(C("term", 1, 1), C("term", 2, 4), C("term", 3, 9))
        designing.design(uneven, learner, order=["geometric"])
        self.assertIn(("fit-geometric", "even-ratios ?it"),
                      [(verb, literal) for verb, literal, _ in
                       learner.learned.requirements()])
        again = designing.design(
            Q(C("term", 1, 1), C("term", 2, 8), C("term", 3, 27),
              C("term", 4, 64)), learner, order=["geometric"])
        self.assertNotIn("geometric", [name for name, _ in again.tried])
        self.assertTrue(again.done)


class ExplorationTests(unittest.TestCase):
    """A lesson that stops a form being tried is tried against anyway."""

    def learner(self, verb, literal):
        from research.v691 import learned, lessons
        out = lessons.Learner(learned.Learned(None))
        out.learned.block(verb, literal, said="compared with one success")
        return out

    def test_curiosity_refutes_a_false_lesson(self):
        learner = self.learner("fit-roots", "degree-given ?it")
        spec = P(C("degree", 2), C("root", 1), C("root", 2),
                 C("leading", 3))
        quiet = designing.design(spec, learner, curiosity=0)
        self.assertNotIn("roots", [name for name, _ in quiet.tried])
        self.assertTrue(learner.learned.blockings())
        curious = designing.design(spec, learner, curiosity=1.0)
        self.assertEqual(curious.explored, [("roots", "curious")])
        self.assertEqual(learner.learned.blockings(), [])

    def test_nothing_is_impossible_until_everything_is_tried(self):
        """Every sequence form but one held back by a lesson, and that
        one cannot fit: the held-back form that can is tried before the
        goal is called impossible."""
        learner = self.learner("fit-polynomial", "terms-given ?it")
        spec = Q(C("term", 1, 1), C("term", 2, 4), C("term", 3, 9),
                 C("term", 4, 16))
        found = designing.design(spec, learner, curiosity=0)
        self.assertTrue(found.done)
        self.assertEqual(found.explored, [("polynomial", "last resort")])
        self.assertEqual(learner.learned.blockings(), [])


class CompositionTests(unittest.TestCase):
    """Forms made from forms, and chosen when they say more with less."""

    def test_composed_forms_are_worked_out_not_listed(self):
        names = [form.name for form in forms.FORMS]
        self.assertIn("arithmetic+geometric", names)
        self.assertIn("geometric+geometric", names)
        # A line plus a line is a line; a scale times the roots plus
        # another is one scale times them: said nothing new, so not made.
        self.assertNotIn("arithmetic+arithmetic", names)
        self.assertNotIn("roots+roots", names)

    def test_fibonacci_is_two_geometric_sequences(self):
        spec = Q(*(C("term", k + 1, v) for k, v in
                   enumerate((1, 1, 2, 3, 5, 8))))
        best = designing.simplest(spec, designing.fitting(spec))
        self.assertEqual(best, "geometric+geometric")
        found = designing.fit(forms.BY_NAME[best], spec)
        self.assertEqual(S.simplify(found.subs(n, 10)), 55)

    def test_a_composite_wins_only_when_it_says_less(self):
        doubled = Q(*(C("term", k + 1, v) for k, v in
                      enumerate((3, 5, 9, 17, 33))))
        self.assertEqual(designing.simplest(doubled,
                                            designing.fitting(doubled)),
                         "arithmetic+geometric")
        plain = Q(*(C("term", k + 1, v) for k, v in
                    enumerate((2, 6, 18, 54, 162))))
        self.assertEqual(designing.simplest(plain, designing.fitting(plain)),
                         "geometric")


class FunctionTests(unittest.TestCase):
    """Functions: derivatives and the equations they satisfy."""

    def F(self, *clauses) -> Spec:
        return Spec("function", clauses)

    def best(self, spec):
        name = designing.best_fit(spec)
        return name, designing.fit(forms.BY_NAME[name], spec)

    def test_an_equation_is_met_by_the_form_it_calls_for(self):
        name, found = self.best(self.F(C("ode", 1, 0, 1), C("value", 0, 1),
                                       C("slope", 0, 0)))
        self.assertEqual((name, found), ("oscillation", S.cos(x)))
        name, found = self.best(self.F(C("ode", 0, 1, -3),
                                       C("value", 0, 2)))
        self.assertEqual((name, found), ("exponential", 2 * S.exp(3 * x)))

    def test_a_derivative_is_integrated(self):
        name, found = self.best(self.F(C("derivative", 2 * x * S.cos(x ** 2)),
                                       C("value", 0, 1)))
        self.assertEqual(name, "antiderivative")
        self.assertEqual(found, S.sin(x ** 2) + 1)

    def test_two_exponentials_for_a_second_order_equation(self):
        spec = self.F(C("ode", 1, -3, 2), C("value", 0, 2), C("slope", 0, 3))
        name, found = self.best(spec)
        self.assertEqual(name, "exponential+exponential")
        self.assertEqual(S.expand(found), S.exp(x) + S.exp(2 * x))


class StatingTests(unittest.TestCase):
    """Design goals said in English read back to what was said."""

    def test_every_goal_said_reads_back(self):
        import random
        from research.v693 import generating, stating
        from research.v693.reading import spec_of
        rng = random.Random(21)
        done = 0
        while done < 60:
            spec = generating.goal(rng)
            made = stating.stated(spec, rng) if spec is not None else None
            if made is None:
                continue
            tokens, spec = made
            back = spec_of(*zip(*tokens))
            with self.subTest(said=" ".join(one[0] for one in tokens)):
                self.assertEqual(stating.canonical(back),
                                 stating.canonical(spec))
            done += 1

    def test_the_parts_are_put_together_by_where_they_stand(self):
        """*the 5th term* gives its place to the clause after it; *value
        12 at 0* and *value at 0 is 12* are the same."""
        from research.v693.reading import spec_of
        said = [("a", "O", "DROP"), ("sequence", "DKIND", "sequence"),
                ("whose", "O", "DROP"), ("5th", "AT", "5"),
                ("term", "CLAUSE", "term"), ("is", "O", "DROP"),
                ("48", "IS", "KEEP")]
        self.assertEqual(spec_of(*zip(*said)).clauses, (C("term", 5, 48),))
        for said in ([("value", "CLAUSE", "value"), ("12", "IS", "KEEP"),
                      ("at", "O", "DROP"), ("0", "AT", "KEEP")],
                     [("value", "CLAUSE", "value"), ("at", "O", "DROP"),
                      ("0", "AT", "KEEP"), ("is", "O", "DROP"),
                      ("12", "IS", "KEEP")]):
            spec = spec_of(*zip(*([("polynomial", "DKIND", "polynomial")]
                                  + said)))
            self.assertEqual(spec.clauses, (C("value", 0, 12),))


class GrowthTests(unittest.TestCase):

    def test_slopes_need_a_degree_more_than_they_count(self):
        """Two slopes are two conditions, and a line has one slope: the
        general form grows to a quadratic, the least that fits."""
        spec = P(C("slope", 4, -120), C("slope", 3, -70))
        found = designing.fit(forms.BY_NAME["general"], spec)
        self.assertTrue(spec.holds(found))
        self.assertEqual(S.degree(found, x), 2)


class ProposerTests(unittest.TestCase):
    """Taught from goals made from objects, ranked by the simplest fit."""

    def test_every_goal_made_from_an_object_can_be_designed(self):
        from research.v693 import generating
        for spec, best in generating.labelled(40, seed=5):
            with self.subTest(spec=spec.said()):
                self.assertIsNotNone(best)
                self.assertIsNotNone(designing.fit(forms.BY_NAME[best],
                                                   spec))

    def test_the_lazy_best_is_the_simplest_of_all(self):
        """Trying forms fewest unknowns first, and stopping, finds what
        fitting every form and taking the simplest would."""
        spec = Q(*(C("term", k + 1, v) for k, v in
                   enumerate((3, 5, 9, 17, 33))))
        self.assertEqual(designing.best_fit(spec), designing.simplest(
            spec, designing.fitting(spec)))

    def test_the_simplest_fit_is_the_label(self):
        spec = Q(C("term", 1, 2), C("term", 2, 6), C("term", 3, 18),
                 C("term", 4, 54))
        self.assertEqual(designing.simplest(spec, designing.fitting(spec)),
                         "geometric")

    def test_a_trained_proposer_puts_the_simplest_first(self):
        from research.v693 import generating
        from research.v693.proposer import Proposer
        proposer = Proposer.trained(generating.labelled(400, seed=7),
                                    epochs=20)
        geometric = Q(C("term", 1, 5), C("term", 2, 15), C("term", 3, 45),
                      C("term", 4, 135))
        self.assertEqual(proposer.order(geometric)[0], "geometric")
        squares = Q(C("term", 1, 1), C("term", 2, 4), C("term", 3, 9),
                    C("term", 4, 16))
        self.assertEqual(proposer.order(squares)[0], "polynomial")


if __name__ == "__main__":
    unittest.main()

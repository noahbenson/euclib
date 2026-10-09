# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_quadrature.py
'''Tests for the ``euclib.utils._quadrature`` module.

The rule a quadrature is checked against is the monomial integral: a rule is
exact to a degree if and only if it integrates every monomial of that degree,
and the reference simplex's is known in closed form,

    the integral of ``L0**a * L1**b * ...`` over an ``n``-simplex is
    ``(a! b! ...) / (a + b + ... + n)!``

for barycentric coordinates and a unit-measure simplex. That is the only thing
that says a rule is what it claims, and it is what found that the four-point
tetrahedron rule in circulation is degree 2 rather than the degree 3 it is often
listed as --- an error of a few parts in a thousand, which is exactly the size
that survives being read and does not survive being computed.
'''

# Dependencies ###############################################################

from __future__ import annotations

from itertools import product
from math import factorial

from numpy import allclose, prod
import unittest
from unittest import TestCase

from euclib.utils._quadrature import quadrature


# Helpers ####################################################################

def _monomials(count, degree, /):
    '''Every exponent tuple of total degree up to ``degree``.'''
    for total in range(degree + 1):
        for alpha in product(range(total + 1), repeat=count):
            if sum(alpha) == total:
                yield alpha


def _integral(alpha, dim, /):
    '''The monomial's integral over the reference ``dim``-simplex.'''
    num = 1
    for a in alpha:
        num *= factorial(a)
    return num / factorial(sum(alpha) + dim)


def _summed(points, weights, alpha, /):
    '''The rule's weighted sum of a monomial.'''
    return sum(w * prod([p[i] ** alpha[i] for i in range(len(alpha))])
               for (p, w) in zip(points, weights))


#: The reference simplexes, by order: its dimension and its measure.
SIMPLEXES = ((1, 1, 1.0), (2, 2, 0.5), (3, 3, 1.0 / 6.0))


# Tests ######################################################################

class TestTheQuadratureRules(TestCase):
    '''That each rule is what it says it is.'''

    def test_every_rule_integrates_the_monomials_it_promises(self):
        for (order, dim, measure) in SIMPLEXES:
            for degree in (1, 2, 3):
                (points, weights) = quadrature(order, degree)
                with self.subTest(order=order, degree=degree):
                    for alpha in _monomials(dim + 1, degree):
                        got = _summed(points, weights, alpha)
                        self.assertAlmostEqual(
                            got, _integral(alpha, dim) / measure, places=12,
                            msg=f"monomial {alpha} of degree {degree} on a"
                                f" {dim}-simplex")

    def test_every_weight_is_positive(self):
        # The point of the rules here: a negative weight is a cancellation
        # waiting to happen, and the rules that need one are stepped over for
        # higher-degree rules that do not.
        for (order, _dim, _measure) in SIMPLEXES:
            for degree in (1, 2, 3):
                (_points, weights) = quadrature(order, degree)
                with self.subTest(order=order, degree=degree):
                    self.assertTrue((weights > 0).all(),
                                    f"a {order}-simplex rule has a weight that"
                                    f" is not positive: {weights}")

    def test_the_weights_sum_to_one(self):
        # So that an element's measure scales them to its integral, which is how
        # `integrate` uses them.
        for (order, _dim, _measure) in SIMPLEXES:
            for degree in (1, 2, 3):
                (_points, weights) = quadrature(order, degree)
                with self.subTest(order=order, degree=degree):
                    self.assertAlmostEqual(weights.sum(), 1.0, places=12)

    def test_the_points_are_barycentric(self):
        # Every point's weights sum to 1: a point is a position within an
        # element, so its barycentric coordinates are a partition of unity.
        for (order, dim, _measure) in SIMPLEXES:
            for degree in (1, 2, 3):
                (points, _weights) = quadrature(order, degree)
                with self.subTest(order=order, degree=degree):
                    self.assertEqual(points.shape[1], dim + 1)
                    self.assertTrue(allclose(points.sum(axis=1), 1.0))
                    self.assertTrue((points >= 0).all(),
                                    "a point lies outside the reference"
                                    " simplex")

    def test_a_higher_degree_never_uses_fewer_points(self):
        # Not a law, but a rule that reached a higher degree with fewer points
        # would mean one of the two was wrong.
        for (order, _dim, _measure) in SIMPLEXES:
            counts = [len(quadrature(order, d)[1]) for d in (1, 2, 3)]
            with self.subTest(order=order):
                self.assertEqual(counts, sorted(counts))

    def test_the_rules_are_reused(self):
        # A rule is fixed by its order and degree, and `integrate` asks for one
        # per call, so the same arrays come back rather than being rebuilt.
        self.assertIs(quadrature(3, 3), quadrature(3, 3))


class TestTheQuadratureInterface(TestCase):
    '''That the arguments it will not take are refused.'''

    def test_an_order_that_is_not_a_simplex_is_refused(self):
        for order in (0, 4, -1):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    quadrature(order, 1)

    def test_a_degree_below_one_is_refused(self):
        for degree in (0, -1):
            with self.subTest(degree=degree):
                with self.assertRaises(ValueError):
                    quadrature(2, degree)

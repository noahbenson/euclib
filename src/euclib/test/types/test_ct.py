# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_ct.py
'''Tests for the Clough-Tocher element in ``euclib.types._ct``.

The element is a piecewise cubic on a triangle split into three, built from
twelve numbers: the value and the gradient at each corner and the derivative
across each edge at its midpoint. Its whole purpose is to be *smooth* --- C1
across the edges the three pieces share --- and to reproduce the polynomials
its data can determine. Both are checkable, and neither is a matter of opinion,
so this file checks them rather than the numbers that come out of the
construction.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np

from euclib.types import _ct


# Fixtures ###################################################################

#: The field the element is tested against, and its exact gradient. A cubic,
#: since the element holds the cubics: anything of lower degree comes back for
#: free, and a cubic is the strongest thing its data can determine.
CUBIC = lambda q: 0.3 * q[0] ** 3 - 0.7 * q[0] ** 2 * q[1] + 0.2 * q[1] ** 3
CUBIC_GRADIENT = lambda q: np.array([0.9 * q[0] ** 2 - 1.4 * q[0] * q[1],
                                     -0.7 * q[0] ** 2 + 0.6 * q[1] ** 2])

QUADRATIC = lambda q: (0.4 * q[0] ** 2 - 0.3 * q[0] * q[1]
                       + 0.25 * q[1] ** 2 + 0.8 * q[0] - 0.2 * q[1] + 0.5)
QUADRATIC_GRADIENT = lambda q: np.array([0.8 * q[0] - 0.3 * q[1] + 0.8,
                                         -0.3 * q[0] + 0.5 * q[1] - 0.2])


def numbers_of(triangle, field, derivative, /):
    '''The twelve numbers a known field gives an element on a triangle.

    The value and the gradient at each corner, and the derivative across each
    edge at its midpoint, in the order the elements' rows are built in.
    '''
    out = []
    for vertex in range(3):
        out.append(field(triangle[:, vertex]))
        for axis in (0, 1):
            out.append(derivative(triangle[:, vertex])[axis])
    for k in range(3):
        middle = point_of(triangle, k, np.array([0.5, 0.5, 0.0]))
        out.append(float(derivative(middle) @ _ct.across_vector(triangle, k)))
    return np.array(out)


def point_of(triangle, k, w, /):
    '''The position the weights name within one of a triangle's pieces.'''
    return w @ _ct._corners(triangle, k)


def value_of(basis, numbers, k, w, /):
    '''The element's value, from its basis and a set of numbers.'''
    return _ct.evaluate(basis[k * 10:(k + 1) * 10] @ numbers, w)


def weights(rng, /):
    '''A position within a triangle, as barycentric weights.'''
    w = rng.uniform(size=3)
    return w / w.sum()


# Tests ######################################################################

class TestTheReferenceElement(TestCase):
    '''The construction, on the triangle it is derived on.'''

    def setUp(self):
        self.basis = _ct.basis()
        self.triangle = _ct.REFERENCE
        self.rng = np.random.default_rng(0)

    def test_the_basis_is_twelve_functions_over_thirty_controls(self):
        # Three pieces of ten controls each, held to twelve numbers.
        self.assertEqual(self.basis.shape, (30, 12))

    def test_a_constant_field_is_reproduced(self):
        # The twelve basis functions sum to one: data that says "the value is
        # one, the gradient zero, every edge flat" gives one everywhere. This
        # is the check that the construction is normalized at all.
        numbers = np.zeros(12)
        numbers[0:9:3] = 1.0
        worst = 0.0
        for k in range(3):
            for _ in range(200):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w) - 1.0))
        self.assertLess(worst, 1e-12)

    def test_a_quadratic_is_reproduced(self):
        numbers = numbers_of(self.triangle, QUADRATIC, QUADRATIC_GRADIENT)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w)
                                       - QUADRATIC(point_of(self.triangle, k, w))))
        self.assertLess(worst, 1e-12)

    def test_a_cubic_is_reproduced(self):
        # The element holds the cubics, so a cubic's own numbers give it back
        # with nothing left over to choose.
        numbers = numbers_of(self.triangle, CUBIC, CUBIC_GRADIENT)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w)
                                       - CUBIC(point_of(self.triangle, k, w))))
        self.assertLess(worst, 1e-12)

    def test_the_pieces_are_c1_across_the_edges_they_share(self):
        # The gradient is a function along an interior edge, and both pieces
        # holding it must give the same one.
        numbers = numbers_of(self.triangle, QUADRATIC, QUADRATIC_GRADIENT)
        worst = 0.0
        for vertex in range(3):
            (k, other) = _ct._holders(vertex)
            for s in np.linspace(0.0, 1.0, 25):
                one = _ct.gradient(self.basis[k * 10:(k + 1) * 10] @ numbers,
                                   _ct._corners(self.triangle, k),
                                   _ct._weights_on_edge(k, vertex, s))
                two = _ct.gradient(self.basis[other * 10:(other + 1) * 10] @ numbers,
                                   _ct._corners(self.triangle, other),
                                   _ct._weights_on_edge(other, vertex, s))
                worst = max(worst, np.abs(one - two).max())
        self.assertLess(worst, 1e-12)


class TestTheElementOnAnotherTriangle(TestCase):
    '''The same properties on a triangle that is not the reference one.

    A construction that only works on a symmetric triangle is not a
    construction. This one is affine in everything except the direction the
    edge datum is taken along, which belongs to the edge and so cannot be
    carried over from another triangle --- which is why the basis is solved for
    the triangle at hand.
    '''

    #: A triangle with no symmetry to hide behind.
    TRIANGLE = np.array([[0.0, 2.0, -0.5], [0.0, 0.5, 1.7]])

    def test_it_reproduces_a_quadratic_here_too(self):
        basis = _ct.basis(self.TRIANGLE)
        numbers = numbers_of(self.TRIANGLE, QUADRATIC, QUADRATIC_GRADIENT)
        rng = np.random.default_rng(5)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(rng)
                worst = max(worst, abs(value_of(basis, numbers, k, w)
                                       - QUADRATIC(point_of(self.TRIANGLE, k, w))))
        self.assertLess(worst, 1e-12)

# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_ps.py
'''Tests for the Powell-Sabin element in ``euclib.types._ps``.

The element is a piecewise quadratic on a triangle split into six, built from
nine numbers: the value at each corner and the derivative there along each of
the triangle's two edges. Two properties are what it is for, and both are
checkable rather than a matter of opinion:

- it reproduces the quadratics, which is the strongest thing its data can
  determine, and a quadratic is the whole of what a piecewise quadratic can;
- it is *barycentric* --- the construction never names a coordinate --- so a
  triangle in a plane and a triangle standing in space are the same problem.
  That is a claim worth a test of its own, because the Clough-Tocher element had
  to be rebuilt to earn it.

This file checks those, and the shape of the split, rather than the particular
numbers that come out of the solve.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np

from euclib.types import _ps


# Fixtures ###################################################################

#: A field, quadratic, written in a triangle's *own* two coordinates --- so the
#: same formula serves a triangle in a plane and one standing in space, and a
#: quadratic piecewise patch whose data determines a quadratic must be it.
QUADRATIC = lambda q: (0.4 * q[0] ** 2 - 0.3 * q[0] * q[1]
                       + 0.25 * q[1] ** 2 + 0.8 * q[0] - 0.2 * q[1] + 0.5)
QUADRATIC_GRADIENT = lambda q: np.array([0.8 * q[0] - 0.3 * q[1] + 0.8,
                                         -0.3 * q[0] + 0.5 * q[1] - 0.2])

#: A second field, for the channel test.
SECOND = lambda q: -0.2 * q[0] ** 2 + 0.5 * q[0] * q[1] + 0.1 * q[1] + 1.0
SECOND_GRADIENT = lambda q: np.array([-0.4 * q[0] + 0.5 * q[1],
                                      0.5 * q[0] + 0.1])

#: A triangle with no symmetry to hide behind, in a plane.
FLAT = np.array([[0.0, 1.7, 0.6], [0.0, 0.1, 1.9]])

#: ...and one carried off every coordinate plane, which is the case that tells a
#: construction in the geometry from one in the coordinates.
TILTED = np.array([[0.0, 1.2, 0.4], [0.3, 0.3, 1.1], [0.7, 0.6, -0.2]])


def along_of(corners, /):
    '''The triangle's two edge directions at corner 0, as a ``(D, 2)`` matrix.'''
    return np.stack([corners[:, 1] - corners[:, 0],
                     corners[:, 2] - corners[:, 0]], axis=1)


def inplane_of(corners, point, /):
    '''A position's two coordinates within the triangle's own plane.'''
    return np.linalg.pinv(along_of(corners)) @ (point - corners[:, 0])


def ambient(corners, field, gradient, /):
    '''A field written in a triangle's own coordinates, as functions of position.

    The chain rule, with the pseudo-inverse because a triangle in space is not a
    square map: the in-plane derivatives are carried out to the ambient ones by
    the edge directions' pseudo-inverse, transposed. This is how a *surface*
    field's gradient is meant, and it is the same construction the element avoids
    needing --- the element reads slopes along edges, not gradient components.
    '''
    carried = np.linalg.pinv(along_of(corners)).T

    def value(point, /):
        return field(inplane_of(corners, point))

    def slope(point, /):
        return carried @ gradient(inplane_of(corners, point))

    return (value, slope)


def sides_of(corners, /):
    '''A triangle's three side lengths, the one opposite each corner.'''
    (one, two, three) = corners.T
    return np.array([np.linalg.norm(two - three),
                     np.linalg.norm(three - one),
                     np.linalg.norm(one - two)])


def numbers_of(corners, field, derivative, /):
    '''The element's nine numbers for a known field on a triangle.

    A value at each corner and, there, the derivative along each of the
    triangle's two edges from it --- in the order the element reads them. The
    derivative along an edge is the field's gradient in the *direction* of that
    edge, which is a direction the geometry defines, so it is the same number
    for a triangle in a plane and a triangle in space.
    '''
    out = []
    for vertex in range(3):
        out.append(field(corners[:, vertex]))
        for other in _ps.neighbours_of(vertex):
            along = corners[:, other] - corners[:, vertex]
            out.append(float(derivative(corners[:, vertex]) @ along))
    return np.array(out)


def interpolate(corners, weights, numbers, /):
    '''The element's value at positions, from its numbers on a triangle.

    ``numbers`` may carry any leading channels, and the answer carries them too.
    '''
    centre = _ps.centre_weights(sides_of(corners))
    ordinates = numbers @ _ps.basis(centre).T
    (pieces, inside) = _ps.sub_weights(weights[:2], centre)
    out = np.zeros(numbers.shape[:-1] + (weights.shape[1],))
    for k in range(len(_ps.PIECES)):
        here = pieces == k
        if not here.any():
            continue
        out[..., here] = _ps.evaluate(ordinates[..., _ps.SLOTS[k]],
                                      inside[here].T)
    return out


def random_weights(rng, count, /):
    '''Barycentric weights of positions within a triangle.

    Drawn uniformly on the triangle by normalizing three exponentials, which is
    the standard way and keeps every weight non-negative --- so a test that says
    a position lands in a piece with non-negative weights is testing the element
    and not the fixture.
    '''
    drawn = rng.exponential(size=(count, 3))
    return (drawn / drawn.sum(axis=1)[:, None]).T


# The split ##################################################################

class TestTheSplit(TestCase):
    '''The shape of the split: six pieces, nineteen ordinates, nine numbers, and
    no freedom left over.'''

    def test_the_ordinates_are_one_per_vertex_and_one_per_edge(self):
        self.assertEqual(len(_ps.ORDINATES), 19)
        self.assertEqual(len(set(_ps.ORDINATES)), 19)
        self.assertEqual(len(_ps.PIECES), 6)
        self.assertEqual(len(_ps.VERTICES), 7)

    def test_every_piece_is_a_half_edge_with_the_centre(self):
        seen = set()
        for k in range(len(_ps.PIECES)):
            piece = _ps.PIECES[k]
            on_edge = [x for x in piece if x != 'Z']
            self.assertEqual(len(on_edge), 2)
            self.assertIn('Z', piece)
            seen.add(tuple(sorted(on_edge)))
        # The six half-edges: each of the triangle's three edges, cut in two at
        # its own point.
        self.assertEqual(seen, {tuple(sorted(half)) for half in
                                (('A', 'S_AB'), ('S_AB', 'B'),
                                 ('B', 'S_BC'), ('S_BC', 'C'),
                                 ('C', 'S_CA'), ('S_CA', 'A'))})

    def test_the_element_has_full_rank_in_its_nine_numbers(self):
        # Nine columns and nineteen rows: if the columns were dependent the data
        # would not determine the interpolation, which is the whole claim.
        for centre in (np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.3, 0.2]),
                       np.array([0.2, 0.5, 0.3]), np.array([0.7, 0.1, 0.2])):
            with self.subTest(centre=centre):
                self.assertEqual(np.linalg.matrix_rank(_ps.basis(centre)), 9)

    def test_the_pieces_tile_the_triangle(self):
        # Every position within the triangle lands in a piece with three
        # non-negative weights in it --- so the six pieces cover it, and the
        # value the element gives is single-valued.
        rng = np.random.default_rng(0)
        parts = random_weights(rng, 500)
        centre = _ps.centre_weights(np.array([1.3, 1.1, 1.5]))
        (pieces, inside) = _ps.sub_weights(parts[:2], centre)
        self.assertTrue((inside > -1e-12).all())
        self.assertTrue(((pieces >= 0) & (pieces < 6)).all())

    def test_the_incenter_is_equidistant_from_the_three_edges(self):
        # What makes the centre the incenter, checked rather than assumed.
        corners = TILTED
        point = corners @ _ps.centre_weights(sides_of(corners))
        gaps = []
        for k in range(3):
            (one, two) = _ps.neighbours_of(k)
            edge = corners[:, two] - corners[:, one]
            away = np.cross(edge, point - corners[:, one])
            gaps.append(np.linalg.norm(away) / np.linalg.norm(edge))
        self.assertLess(max(gaps) - min(gaps), 1e-12)


# The element ################################################################

class TestTheElementOnATriangle(TestCase):
    '''What the element computes, on triangles of both kinds.'''

    def worst_on(self, corners, field, gradient, /):
        (value, slope) = ambient(corners, field, gradient)
        numbers = numbers_of(corners, value, slope)
        rng = np.random.default_rng(4)
        weights = random_weights(rng, 300)
        points = corners @ weights
        got = interpolate(corners, weights, numbers)
        want = np.array([value(points[:, i]) for i in range(points.shape[1])])
        return np.abs(got - want).max()

    def test_a_quadratic_is_reproduced_in_a_plane(self):
        self.assertLess(self.worst_on(FLAT, QUADRATIC, QUADRATIC_GRADIENT), 1e-12)

    def test_a_quadratic_is_reproduced_on_a_triangle_in_space(self):
        self.assertLess(self.worst_on(TILTED, QUADRATIC, QUADRATIC_GRADIENT),
                        1e-12)

    def test_a_linear_field_is_reproduced(self):
        # The simplest thing the data can determine, and the one an error in the
        # construction's constants would show up in first.
        linear = lambda q: 0.7 + 0.4 * q[0] - 0.9 * q[1]
        gradient = lambda q: np.array([0.4, -0.9])
        self.assertLess(self.worst_on(TILTED, linear, gradient), 1e-12)

    def test_the_corner_values_are_interpolated(self):
        corners = TILTED
        (value, slope) = ambient(corners, QUADRATIC, QUADRATIC_GRADIENT)
        numbers = numbers_of(corners, value, slope)
        got = interpolate(corners, np.eye(3), numbers)
        for vertex in range(3):
            with self.subTest(vertex=vertex):
                self.assertAlmostEqual(got[vertex],
                                       value(corners[:, vertex]), places=12)

    def test_a_channelled_field_is_interpolated_channel_by_channel(self):
        # The element is linear in its data, so a channel is a column and the
        # two must not mix.
        corners = TILTED
        (one, one_slope) = ambient(corners, QUADRATIC, QUADRATIC_GRADIENT)
        (two, two_slope) = ambient(corners, SECOND, SECOND_GRADIENT)
        stacked = np.stack([numbers_of(corners, one, one_slope),
                            numbers_of(corners, two, two_slope)])
        rng = np.random.default_rng(6)
        weights = random_weights(rng, 40)
        points = corners @ weights
        got = interpolate(corners, weights, stacked)
        for (channel, (value, _)) in enumerate([(one, one_slope),
                                                (two, two_slope)]):
            want = np.array([value(points[:, i])
                             for i in range(points.shape[1])])
            with self.subTest(channel=channel):
                self.assertLess(np.abs(got[channel] - want).max(), 1e-12)

    def test_the_numbers_do_not_change_when_the_triangle_is_turned(self):
        '''The element's data does not depend on where the geometry is.

        Turning a triangle in space moves the field with it but changes none of
        the nine numbers, each being a value or a derivative along a direction
        the geometry defines. A construction that read gradient components in
        coordinate directions would fail this the moment the triangle turned ---
        which is exactly what happened to the Clough-Tocher element.
        '''
        turn = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0],
                         [-0.48, 0.64, 0.6]])
        rng = np.random.default_rng(11)
        (value, slope) = ambient(TILTED, QUADRATIC, QUADRATIC_GRADIENT)
        before = numbers_of(TILTED, value, slope)
        for _ in range(20):
            shift = rng.normal(size=3)
            moved = turn @ TILTED + shift[:, None]
            (there, there_slope) = ambient(moved, QUADRATIC, QUADRATIC_GRADIENT)
            after = numbers_of(moved, there, there_slope)
            self.assertTrue(np.allclose(before, after, atol=1e-12))

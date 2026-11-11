# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_fit.py
'''Tests for the line and plane fits in ``euclib.ops._fit``.

The fits are singular value decompositions, so the tests are mostly about the
two things a decomposition does not say by itself: *which* singular vector
answers the question, and what the answer means when the points do not
determine one. The first is checked against the eigenvectors of the covariance
matrix, which is the same problem written a second way; the second is checked
by refusing rather than by comparing against an arbitrary answer.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, abs as npabs, cross, eye, mean
import unittest
from unittest import TestCase

from euclib.ops import fit_line, fit_plane
from euclib.types import Grid, GridTopology, TriMesh, TriTopology


# Fixtures ###################################################################

def _line_points(origin, direction, numbers, /):
    '''Points on the line through ``origin`` along ``direction``.'''
    direction = asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return asarray(origin, dtype=float).reshape(-1, 1) \
        + direction.reshape(-1, 1) * asarray(numbers, dtype=float).reshape(1, -1)


def _plane_points(origin, first, second, numbers, /):
    '''Points in the plane spanned by two directions from ``origin``.'''
    (u, v) = (asarray(first, dtype=float), asarray(second, dtype=float))
    (u, v) = (u / np.linalg.norm(u), v / np.linalg.norm(v))
    numbers = asarray(numbers, dtype=float)
    return (asarray(origin, dtype=float).reshape(-1, 1)
            + u.reshape(-1, 1) * numbers[0].reshape(1, -1)
            + v.reshape(-1, 1) * numbers[1].reshape(1, -1))


def _covariance_directions(points, /):
    '''The eigenvectors of a point set's covariance, in ascending order.'''
    centred = asarray(points) - asarray(points).mean(axis=1, keepdims=True)
    (values, vectors) = np.linalg.eigh(centred @ centred.T)
    return (values, vectors)


def _uncorrelated(count=40, dim=3, seed=0, /):
    '''A point cloud with no particular symmetry, so its spreads are distinct.'''
    generator = np.random.default_rng(seed)
    return generator.normal(size=(dim, count)) + asarray(
        [[1.0], [-2.0], [0.5]] if dim == 3 else [[1.0], [-2.0]])


# Tests ######################################################################

class TestTheLineFit(TestCase):
    '''The line a set of points lies nearest.'''

    def test_points_on_a_line_give_that_line(self):
        # The parameter values are chosen to sum to zero, so that the mean of
        # the points is the origin the line was drawn from --- the fit's origin
        # is the mean, not an anchor the points were generated from.
        direction = array([3.0, 4.0])
        points = _line_points([1.0, -2.0], direction, [-3.0, -1.0, 1.5, 2.5])
        (origin, found) = fit_line(points)
        self.assertTrue(allclose(origin, [1.0, -2.0]))
        self.assertTrue(allclose(npabs(found), npabs(direction / 5.0)))

    def test_the_origin_is_the_mean_of_the_points(self):
        points = _uncorrelated()
        (origin, _direction) = fit_line(points)
        self.assertTrue(allclose(origin, points.mean(axis=1)))

    def test_the_direction_is_a_unit_vector(self):
        (_origin, direction) = fit_line(_uncorrelated())
        self.assertAlmostEqual(float(np.linalg.norm(asarray(direction))), 1.0,
                               places=12)
        self.assertEqual(tuple(asarray(direction).shape), (3,))

    def test_the_direction_is_the_largest_spread(self):
        # The line's direction is the covariance's largest eigenvector, which
        # is the same problem the singular value decomposition solves. The two
        # agreeing is what says the *first* singular vector is the one wanted.
        points = _uncorrelated()
        (_origin, direction) = fit_line(points)
        (_values, vectors) = _covariance_directions(points)
        self.assertAlmostEqual(abs(float(direction @ vectors[:, -1])), 1.0,
                               places=10)

    def test_a_vertical_line_is_fitted(self):
        # An ordinary least-squares regression of y on x has no answer here:
        # the line is vertical, so the slope is infinite. A fit by orthogonal
        # distance does not care which coordinate is called x.
        points = array([[2.0] * 5, [-3.0, -1.0, 0.0, 1.5, 4.0]])
        (origin, direction) = fit_line(points)
        self.assertTrue(allclose(origin, [2.0, mean(points[1])]))
        self.assertTrue(allclose(direction, [0.0, 1.0]))

    def test_the_sign_is_chosen_deterministically(self):
        # A singular vector comes back with either sign, so the fit chooses
        # one. Negating the points negates the decomposition, and the same
        # line must still be reported the same way.
        points = _line_points([0.0, 0.0], [2.0, 1.0], [-2.0, 0.0, 3.0])
        (_origin, first) = fit_line(points)
        (_origin, second) = fit_line(-points)
        self.assertTrue(allclose(first, second))
        # The largest-magnitude component is the one made positive: here the
        # y component is the larger, so it is.
        self.assertGreater(float(first[1]), 0.0)

    def test_the_order_of_the_points_does_not_matter(self):
        # A fit is a property of the set, not of the matrix's column order.
        points = _uncorrelated()
        order = np.random.default_rng(11).permutation(points.shape[1])
        (first_o, first_d) = fit_line(points)
        (second_o, second_d) = fit_line(points[:, order])
        self.assertTrue(allclose(first_o, second_o))
        self.assertTrue(allclose(first_d, second_d))

    def test_rotating_the_points_rotates_the_line(self):
        # A fit is equivariant under a rigid motion, which is what an
        # axis-by-axis regression is not. The direction's *sign* is chosen
        # after the rotation and so is not compared; the line it names is.
        points = _uncorrelated(40, 2)
        angle = 0.7
        rotation = array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        (_origin, direction) = fit_line(points)
        (_rotated_origin, rotated) = fit_line(rotation @ points)
        self.assertAlmostEqual(abs(float(rotated @ (rotation @ direction))),
                               1.0, places=10)

    def test_a_geometry_is_fitted_at_its_data_positions(self):
        # A geometry contributes what `positions_of` says its data lives at;
        # for a simplex geometry that is its coordinates.
        points = _line_points([1.0, 1.0], [1.0, 0.0], [0.0, 1.0, 2.0])
        mesh = TriMesh(points, TriTopology([[0], [1], [2]]))
        (origin, direction) = fit_line(mesh)
        self.assertTrue(allclose(origin, [2.0, 1.0]))
        self.assertTrue(allclose(direction, [1.0, 0.0]))

    def test_a_quantity_keeps_its_units_on_the_origin_only(self):
        # A direction is a unit vector and has no units of its own, whatever
        # the positions' units are.
        import pint
        ureg = pint.UnitRegistry()
        points = _line_points([0.0, 0.0], [1.0, 2.0], [0.0, 1.0, 2.0])
        (origin, direction) = fit_line(points * ureg.meter)
        self.assertEqual(str(origin.units), 'meter')
        self.assertFalse(hasattr(direction, 'units'))

    def test_one_point_is_refused(self):
        with self.assertRaises(ValueError):
            fit_line(array([[1.0], [2.0]]))

    def test_coincident_points_are_refused(self):
        with self.assertRaises(ValueError):
            fit_line(array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    def test_a_matrix_that_is_not_positions_is_refused(self):
        for shape in ((4,), (2, 3, 3), (4, 3), (0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    fit_line(np.zeros(shape))


class TestThePlaneFit(TestCase):
    '''The plane a set of points lies nearest.'''

    def test_points_in_a_plane_give_that_plane(self):
        # The parameter values sum to zero along each direction, so the mean of
        # the points is the origin the plane was drawn from.
        points = _plane_points([1.0, 2.0, 3.0], [1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [array([-1.0, 1.0, 0.0, 0.0]),
                                array([-1.0, 0.0, 1.0, 0.0])])
        (origin, normal) = fit_plane(points)
        self.assertTrue(allclose(origin, [1.0, 2.0, 3.0]))
        self.assertTrue(allclose(normal, [0.0, 1.0, 0.0]))
        # Every point is on the plane, exactly.
        self.assertTrue(allclose(points.T @ normal - origin @ normal, 0.0))

    def test_the_normal_is_the_smallest_spread(self):
        points = _uncorrelated()
        (_origin, normal) = fit_plane(points)
        (_values, vectors) = _covariance_directions(points)
        self.assertAlmostEqual(abs(float(normal @ vectors[:, 0])), 1.0,
                               places=10)

    def test_the_normal_is_a_unit_vector(self):
        (_origin, normal) = fit_plane(_uncorrelated())
        self.assertAlmostEqual(float(np.linalg.norm(asarray(normal))), 1.0,
                               places=12)

    def test_a_tilted_plane_is_fitted(self):
        # A plane through the origin whose normal is not an axis. The normal
        # is built by crossing two in-plane edges, so a wrong pairing of the
        # singular vectors is visible here and not on an axis-aligned plane.
        first = array([1.0, 2.0, -0.5])
        second = array([0.0, 1.0, 3.0])
        points = _plane_points([-1.0, 0.5, 2.0], first, second,
                               [_uncorrelated()[0], _uncorrelated()[1]])
        (_origin, normal) = fit_plane(points)
        expected = cross(first, second)
        expected = expected / np.linalg.norm(expected)
        self.assertAlmostEqual(abs(float(normal @ expected)), 1.0, places=10)

    def test_a_plane_in_two_dimensions_is_a_line(self):
        # A plane is a hyperplane, so in two dimensions it is the line the
        # points lie on; its normal is that line's direction turned.
        points = _line_points([0.0, 0.0], [1.0, 1.0], [-2.0, 0.0, 2.0])
        (origin, normal) = fit_plane(points)
        self.assertTrue(allclose(origin, [0.0, 0.0]))
        self.assertTrue(allclose(npabs(normal), [1.0, 1.0] / np.sqrt(2.0)))

    def test_a_grid_is_fitted_at_its_cells(self):
        # A grid's data lives at its cell centers, not at its corners; a grid
        # one cell deep is a plane of them.
        (_origin, normal) = fit_plane(Grid(eye(4), GridTopology((2, 2, 1))))
        self.assertTrue(allclose(normal, [0.0, 0.0, 1.0]))

    def test_collinear_points_are_refused(self):
        # A plane through a line is a pencil of planes, all of them fitting
        # equally well, so there is no answer to report.
        points = _line_points([0.0, 0.0, 0.0], [1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 4.0])
        with self.assertRaises(ValueError):
            fit_plane(points)

    def test_too_few_points_are_refused(self):
        # Two points in three dimensions do not determine a plane either.
        with self.assertRaises(ValueError):
            fit_plane(array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))

    def test_coincident_points_are_refused(self):
        with self.assertRaises(ValueError):
            fit_plane(array([[1.0] * 3, [2.0] * 3, [3.0] * 3]))

    def test_a_quantity_keeps_its_units_on_the_origin_only(self):
        import pint
        ureg = pint.UnitRegistry()
        points = _plane_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [array([0.0, 1.0, 0.0]),
                                array([0.0, 0.0, 1.0])])
        (origin, normal) = fit_plane(points * ureg.meter)
        self.assertEqual(str(origin.units), 'meter')
        self.assertFalse(hasattr(normal, 'units'))


class TestTheTensorPath(TestCase):
    '''That a tensor is fitted as a tensor, its gradient intact.'''

    def _needs_torch(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_a_tensor_stays_a_tensor(self):
        torch = self._needs_torch()
        points = torch.tensor(asarray(_uncorrelated(), dtype=np.float64))
        (origin, normal) = fit_plane(points)
        self.assertIsInstance(origin, torch.Tensor)
        self.assertIsInstance(normal, torch.Tensor)
        self.assertEqual(tuple(origin.shape), (3,))

    def test_a_tensor_and_an_array_agree(self):
        torch = self._needs_torch()
        points = asarray(_uncorrelated(), dtype=np.float64)
        (origin, normal) = fit_plane(points)
        (t_origin, t_normal) = fit_plane(torch.tensor(points))
        self.assertTrue(allclose(asarray(t_origin.detach()), origin))
        self.assertTrue(allclose(asarray(t_normal.detach()), normal))

    def test_the_gradient_reaches_the_points(self):
        # The decomposition is a differentiable torch operation, and a
        # conversion to numpy anywhere on the way would drop the gradient
        # without a word.
        torch = self._needs_torch()
        points = torch.tensor(asarray(_uncorrelated(), dtype=np.float64),
                              requires_grad=True)
        (origin, normal) = fit_plane(points)
        self.assertTrue(bool(origin.requires_grad))
        self.assertTrue(bool(normal.requires_grad))
        (origin.sum() + normal.sum()).backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertTrue(bool((points.grad != 0).any()))


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()

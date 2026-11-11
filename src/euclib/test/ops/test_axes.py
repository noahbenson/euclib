# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_axes.py
'''Tests for the principal axes and oriented bounding boxes in ``_axes``.

The axes are the singular vectors of the centred positions, so the tests are
about the two things a decomposition does not say by itself: the *order* the
axes come in and the *sign* each one carries. The order is checked against the
spread along each axis, and against the line and plane fits, which report the
first and last of the same set. The sign convention is checked by the property
it exists for --- that the frame is a rotation --- rather than by comparing
against a decomposition whose signs are the decomposition's own business.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, abs as npabs, eye, mean
import unittest
from unittest import TestCase

from euclib.ops import (
    fit_line, fit_plane, oriented_bounding_box, principal_axes)
from euclib.types import Grid, GridTopology, TriMesh, TriTopology


# Fixtures ###################################################################

def _cloud(count=50, dim=3, seed=0, /):
    '''A point cloud with no symmetry, so its spreads are distinct.'''
    generator = np.random.default_rng(seed)
    return generator.normal(size=(dim, count)) + asarray(
        [[1.0], [-2.0], [0.5]] if dim == 3 else [[1.0], [-2.0]])


def _box_corners(center, axes, extents, /):
    '''The ``2**D`` corners of a box given by its center, axes, and extents.'''
    center = asarray(center, dtype=float)
    axes = asarray(axes, dtype=float)
    extents = asarray(extents, dtype=float)
    dim = center.shape[0]
    signs = asarray(
        [[1.0 if (i >> b) & 1 else -1.0 for b in range(dim)]
         for i in range(2 ** dim)]).T
    half = (extents / 2.0).reshape(-1, 1)
    return center.reshape(-1, 1) + axes @ (signs * half)


def _orthonormal_frame(seed=0, /):
    '''A random right-handed frame a box may be turned by.'''
    generator = np.random.default_rng(seed)
    (q, r) = np.linalg.qr(generator.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, -1] = -q[:, -1]
    return q


# Tests ######################################################################

class TestThePrincipalAxes(TestCase):
    '''The directions a set of points is aligned along.'''

    def test_the_origin_is_the_mean_of_the_points(self):
        points = _cloud()
        (origin, _axes) = principal_axes(points)
        self.assertTrue(allclose(origin, points.mean(axis=1)))

    def test_the_axes_are_orthonormal(self):
        (_origin, axes) = principal_axes(_cloud())
        self.assertTrue(allclose(axes.T @ axes, eye(3)))
        self.assertEqual(tuple(asarray(axes).shape), (3, 3))

    def test_the_axes_form_a_rotation(self):
        # Each axis is sign-oriented on its own, and a set of three such axes
        # may be a reflection. The frame is made right-handed, which is what
        # lets a caller use the axes as a rotation; the inputs here are varied
        # so that the decomposition's own signs certainly vary too.
        for seed in range(6):
            with self.subTest(seed=seed):
                (_origin, axes) = principal_axes(_cloud(30, 3, seed))
                self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0,
                                       places=12)

    def test_the_axes_are_ordered_by_spread(self):
        points = _cloud()
        (origin, axes) = principal_axes(points)
        offsets = points - origin[:, None]
        spreads = ((axes.T @ offsets) ** 2).sum(axis=1)
        self.assertTrue((np.diff(spreads) <= 1e-12).all())

    def test_the_first_axis_is_the_line_fit_s_direction(self):
        points = _cloud()
        (_origin, axes) = principal_axes(points)
        (_o, direction) = fit_line(points)
        self.assertAlmostEqual(abs(float(axes[:, 0] @ direction)), 1.0,
                               places=10)

    def test_the_last_axis_is_the_plane_fit_s_normal(self):
        points = _cloud()
        (_origin, axes) = principal_axes(points)
        (_o, normal) = fit_plane(points)
        self.assertAlmostEqual(abs(float(axes[:, 2] @ normal)), 1.0,
                               places=10)

    def test_rotating_the_points_rotates_the_axes(self):
        # The decomposition is equivariant under a rotation; the sign rule is
        # applied afterwards and is not, so the comparison is up to sign.
        points = _cloud()
        rotation = _orthonormal_frame(3)
        (_origin, axes) = principal_axes(points)
        (_rotated_origin, rotated) = principal_axes(rotation @ points)
        self.assertTrue(allclose(npabs(rotated.T @ (rotation @ axes)), eye(3),
                                 atol=1e-10))

    def test_a_geometry_is_analyzed_at_its_data_positions(self):
        # A simplex geometry's data lives at its coordinates.
        coords = array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        mesh = TriMesh(coords, TriTopology([[0], [1], [2]]))
        (origin, axes) = principal_axes(mesh)
        self.assertTrue(allclose(origin, mean(coords, axis=1)))
        self.assertEqual(tuple(asarray(axes).shape), (2, 2))

    def test_collinear_points_in_two_dimensions_are_analyzed(self):
        # In two dimensions a line's direction determines the other axis: it
        # is the direction at a right angle to it.
        points = array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
        (origin, axes) = principal_axes(points)
        self.assertTrue(allclose(origin, [1.0, 2.0]))
        expected = array([1.0, 2.0]) / np.sqrt(5.0)
        self.assertTrue(allclose(axes[:, 0], expected))
        self.assertTrue(allclose(axes[:, 1], array([-2.0, 1.0]) / np.sqrt(5.0)))

    def test_a_planar_cloud_in_three_dimensions_is_analyzed(self):
        # The third axis has no spread, but its *direction* is still the
        # normal, which is determined as long as the other two are.
        (u, v) = np.meshgrid(np.linspace(-1.0, 1.0, 5),
                             np.linspace(-2.0, 2.0, 5))
        points = array([[0.0], [0.0], [1.0]]) + array([[1.0], [0.0], [0.0]]) * \
            u.reshape(1, -1) + array([[0.0], [1.0], [0.0]]) * v.reshape(1, -1)
        (_origin, axes) = principal_axes(points)
        self.assertTrue(allclose(npabs(axes[:, 2]), [0.0, 0.0, 1.0]))

    def test_a_quantity_keeps_its_units_on_the_origin_only(self):
        import pint
        ureg = pint.UnitRegistry()
        (origin, axes) = principal_axes(_cloud() * ureg.meter)
        self.assertEqual(str(origin.units), 'meter')
        self.assertFalse(hasattr(axes, 'units'))

    def test_coincident_points_are_refused(self):
        with self.assertRaises(ValueError):
            principal_axes(array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    def test_collinear_points_in_three_dimensions_are_refused(self):
        # One axis is determined and the other two are any orthonormal pair in
        # the plane orthogonal to it, so there is no answer to give.
        points = array([[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 2.0, 4.0],
                        [0.0, 1.0, 2.0, 4.0]])
        with self.assertRaises(ValueError):
            principal_axes(points)

    def test_two_points_in_three_dimensions_are_refused(self):
        with self.assertRaises(ValueError):
            principal_axes(array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))

    def test_one_point_is_refused(self):
        with self.assertRaises(ValueError):
            principal_axes(array([[1.0], [2.0]]))

    def test_no_points_are_refused(self):
        with self.assertRaises(ValueError):
            principal_axes(np.zeros((3, 0)))

    def test_a_matrix_that_is_not_positions_is_refused(self):
        for shape in ((4,), (2, 3, 3), (4, 3), (0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    principal_axes(np.zeros(shape))


class TestTheOrientedBoundingBox(TestCase):
    '''The box, aligned with the principal axes, that holds every point.'''

    def test_an_axis_aligned_box_is_recovered(self):
        points = _box_corners([1.0, 2.0, 3.0], eye(3), [4.0, 2.0, 1.0])
        (center, axes, extents) = oriented_bounding_box(points)
        self.assertTrue(allclose(center, [1.0, 2.0, 3.0]))
        self.assertTrue(allclose(npabs(axes), eye(3)))
        self.assertTrue(allclose(extents, [4.0, 2.0, 1.0]))

    def test_a_turned_box_is_recovered(self):
        frame = _orthonormal_frame(5)
        points = _box_corners([1.0, 2.0, 3.0], frame, [4.0, 2.0, 1.0])
        (center, axes, extents) = oriented_bounding_box(points)
        self.assertTrue(allclose(center, [1.0, 2.0, 3.0]))
        self.assertTrue(allclose(npabs(axes.T @ frame), eye(3), atol=1e-10))
        self.assertTrue(allclose(extents, [4.0, 2.0, 1.0]))
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0, places=12)

    def test_every_point_is_inside_the_box(self):
        points = _cloud(60)
        (center, axes, extents) = oriented_bounding_box(points)
        offsets = axes.T @ (points - center[:, None])
        self.assertTrue((npabs(offsets) <= extents[:, None] / 2.0 + 1e-12).all())

    def test_the_extents_are_the_ranges_along_the_axes(self):
        points = _cloud(60)
        (center, axes, extents) = oriented_bounding_box(points)
        offsets = axes.T @ (points - center[:, None])
        self.assertTrue(allclose(extents,
                                 offsets.max(axis=1) - offsets.min(axis=1)))

    def test_a_planar_cloud_has_a_flat_box(self):
        (u, v) = np.meshgrid(np.linspace(-1.0, 1.0, 5),
                             np.linspace(-2.0, 2.0, 5))
        points = array([[1.0], [2.0], [3.0]]) + array([[1.0], [0.0], [0.0]]) * \
            u.reshape(1, -1) + array([[0.0], [1.0], [0.0]]) * v.reshape(1, -1)
        (center, _axes, extents) = oriented_bounding_box(points)
        self.assertTrue(allclose(center, [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(extents[2]), 0.0, places=12)

    def test_a_box_in_two_dimensions(self):
        points = array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        (center, axes, extents) = oriented_bounding_box(points)
        self.assertTrue(allclose(center, [1.0, 0.5]))
        self.assertTrue(allclose(npabs(axes), eye(2)))
        self.assertTrue(allclose(extents, [2.0, 1.0]))

    def test_a_grid_is_enclosed_at_its_cells(self):
        # A grid's data lives at its cell centers, not at its corners: a grid
        # one cell deep is a plane of centers. The grid is rectangular so that
        # the two in-plane spreads differ, which fixes which axis comes first.
        (center, _axes, extents) = oriented_bounding_box(
            Grid(eye(4), GridTopology((2, 3, 1))))
        self.assertTrue(allclose(center, [0.5, 1.0, 0.0]))
        self.assertTrue(allclose(extents, [2.0, 1.0, 0.0]))

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        (center, axes, extents) = oriented_bounding_box(_cloud() * ureg.meter)
        self.assertEqual(str(center.units), 'meter')
        self.assertEqual(str(extents.units), 'meter')
        self.assertFalse(hasattr(axes, 'units'))

    def test_collinear_points_are_refused(self):
        points = array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        with self.assertRaises(ValueError):
            oriented_bounding_box(points)


class TestTheTensorPath(TestCase):
    '''That a tensor is analyzed as a tensor, its gradient intact.'''

    def _needs_torch(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_a_tensor_stays_a_tensor(self):
        torch = self._needs_torch()
        points = torch.tensor(asarray(_cloud(), dtype=np.float64))
        (origin, axes) = principal_axes(points)
        (center, box_axes, extents) = oriented_bounding_box(points)
        for value in (origin, axes, center, box_axes, extents):
            self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(tuple(axes.shape), (3, 3))

    def test_a_tensor_and_an_array_agree(self):
        torch = self._needs_torch()
        points = asarray(_cloud(), dtype=np.float64)
        (origin, axes) = principal_axes(points)
        (center, _axes0, extents) = oriented_bounding_box(points)
        (t_origin, t_axes) = principal_axes(torch.tensor(points))
        (t_center, _t_axes, t_extents) = oriented_bounding_box(
            torch.tensor(points))
        self.assertTrue(allclose(asarray(t_origin.detach()), origin))
        self.assertTrue(allclose(asarray(t_axes.detach()), axes))
        self.assertTrue(allclose(asarray(t_center.detach()), center))
        self.assertTrue(allclose(asarray(t_extents.detach()), extents))

    def test_the_gradient_reaches_the_points(self):
        # A conversion to numpy anywhere on the way would drop the gradient
        # without a word.
        torch = self._needs_torch()
        points = torch.tensor(asarray(_cloud(), dtype=np.float64),
                              requires_grad=True)
        (origin, axes) = principal_axes(points)
        (center, _box_axes, extents) = oriented_bounding_box(points)
        self.assertTrue(bool(origin.requires_grad))
        self.assertTrue(bool(axes.requires_grad))
        self.assertTrue(bool(center.requires_grad))
        self.assertTrue(bool(extents.requires_grad))
        (origin.sum() + axes.sum() + center.sum() + extents.sum()).backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertTrue(bool((points.grad != 0).any()))


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()

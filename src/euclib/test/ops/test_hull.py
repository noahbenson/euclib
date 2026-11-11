# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_hull.py
'''Tests for the convex hull in ``_hull``.

The hull is the one geometry whose answer is a *decision* rather than a
direction: which points are its corners, and which of those a face joins. So
the tests are mostly about the properties the decision has to have, rather than
about the numbers it produces: every point is inside, every face winds outward,
the surface closes, and the volume is the one the points enclose. A few exact
shapes --- a square, a cube, a tetrahedron --- pin the answer itself, because a
property can be satisfied by a hull that is not the smallest one.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, eye, sum as npsum
import unittest
from unittest import TestCase

from euclib.ops import convex_hull
from euclib.types import Grid, GridTopology, SegPath, TriMesh, points


# Fixtures ###################################################################

def _cloud(count=50, dim=3, seed=0, /):
    '''A point cloud with no symmetry, so its hull is not a familiar solid.'''
    generator = np.random.default_rng(seed)
    return generator.normal(size=(dim, count))


def _cube_corners(center=(0.0, 0.0, 0.0), side=1.0, /):
    '''The eight corners of an axis-aligned cube.'''
    center = asarray(center, dtype=float)[:, None]
    half = side / 2.0
    signs = asarray([[sx, sy, sz]
                     for sx in (-1.0, 1.0)
                     for sy in (-1.0, 1.0)
                     for sz in (-1.0, 1.0)]).T
    return center + half * signs


def _triangles(hull, /):
    '''A hull mesh's triangles as a ``(3, M)`` integer matrix.'''
    return asarray(hull.topo.indices)


def _volume(coords, tris, /):
    '''The volume a closed, outward-wound triangle mesh encloses.'''
    coords = asarray(coords)
    total = 0.0
    for k in range(tris.shape[1]):
        (a, b, c) = (coords[:, tris[0, k]], coords[:, tris[1, k]],
                     coords[:, tris[2, k]])
        total += float(a @ np.cross(b - a, c - a)) / 6.0
    return total


def _paired(tris, /):
    '''Whether every directed edge of a mesh appears once, and its reverse too.'''
    seen = {}
    for k in range(tris.shape[1]):
        (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(n == 1 and (edge[1], edge[0]) in seen
               for (edge, n) in seen.items())


# Tests ######################################################################

class TestThePolygon(TestCase):
    '''The hull of points in a plane.'''

    def test_a_square_with_a_point_inside_is_four_corners(self):
        square = array([[0.0, 2.0, 2.0, 0.0, 1.0],
                        [0.0, 0.0, 2.0, 2.0, 1.0]])
        hull = convex_hull(square)
        self.assertIsInstance(hull, SegPath)
        self.assertEqual(hull.coord_count, 4)
        self.assertEqual(hull.topo.simplex_count[1], 4)
        self.assertTrue(allclose(np.sort(asarray(hull.coords), axis=0),
                                 np.sort(square[:, :4], axis=0)))

    def test_the_corners_are_counterclockwise(self):
        (x, y) = asarray(convex_hull(_cloud(40, 2)).coords)
        area = 0.5 * float(npsum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        self.assertGreater(area, 0.0)

    def test_every_point_is_inside_or_on_the_polygon(self):
        cloud = _cloud(200, 2)
        corners = asarray(convex_hull(cloud).coords)
        for k in range(corners.shape[1]):
            a = corners[:, k]
            b = corners[:, (k + 1) % corners.shape[1]]
            edge = b - a
            # Left of every edge, wound counterclockwise.
            cross = edge[0] * (cloud[1] - a[1]) - edge[1] * (cloud[0] - a[0])
            self.assertTrue((cross >= -1e-12).all())

    def test_the_polygon_closes(self):
        hull = convex_hull(_cloud(20, 2))
        segments = asarray(hull.topo.indices)
        self.assertEqual(segments.shape[1], hull.coord_count)
        self.assertEqual(sorted(segments[0].tolist()),
                         sorted(segments[1].tolist()))

    def test_duplicate_points_are_tolerated(self):
        square = array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
        cloud = np.hstack([square, square, [[1.0], [1.0]]])
        hull = convex_hull(cloud)
        self.assertEqual(hull.coord_count, 4)
        self.assertEqual(hull.topo.simplex_count[1], 4)

    def test_a_geometry_is_analyzed_at_its_data_positions(self):
        square = array([[0.0, 2.0, 2.0, 0.0, 1.0],
                        [0.0, 0.0, 2.0, 2.0, 1.0]])
        hull = convex_hull(points(square))
        self.assertEqual(hull.coord_count, 4)

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        hull = convex_hull(_cloud(20, 2) * ureg.meter)
        self.assertEqual(str(hull.coords.units), 'meter')

    def test_collinear_points_are_refused(self):
        with self.assertRaises(ValueError):
            convex_hull(array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))

    def test_two_points_are_refused(self):
        with self.assertRaises(ValueError):
            convex_hull(array([[0.0, 1.0], [0.0, 0.0]]))

    def test_no_points_are_refused(self):
        with self.assertRaises(ValueError):
            convex_hull(np.zeros((2, 0)))

    def test_a_matrix_that_is_not_positions_is_refused(self):
        for shape in ((4,), (2, 3, 3), (4, 3), (0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    convex_hull(np.zeros(shape))


class TestThePolyhedron(TestCase):
    '''The hull of points in space.'''

    def test_a_cube_is_recovered(self):
        hull = convex_hull(_cube_corners())
        self.assertIsInstance(hull, TriMesh)
        self.assertEqual(hull.coord_count, 8)
        self.assertEqual(hull.topo.simplex_count[2], 12)
        self.assertAlmostEqual(float(hull.measures.sum()), 6.0, places=12)
        self.assertAlmostEqual(_volume(hull.coords, _triangles(hull)), 1.0,
                               places=12)

    def test_a_turned_cube_is_recovered(self):
        generator = np.random.default_rng(5)
        (rotation, _r) = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] = -rotation[:, 0]
        corners = rotation @ _cube_corners([1.0, 2.0, 3.0], 2.0)
        hull = convex_hull(corners)
        self.assertEqual(hull.coord_count, 8)
        self.assertEqual(hull.topo.simplex_count[2], 12)
        self.assertAlmostEqual(_volume(hull.coords, _triangles(hull)), 8.0,
                               places=10)

    def test_a_tetrahedron_is_recovered(self):
        tetra = array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
        hull = convex_hull(tetra)
        self.assertEqual(hull.coord_count, 4)
        self.assertEqual(hull.topo.simplex_count[2], 4)
        self.assertAlmostEqual(_volume(hull.coords, _triangles(hull)), 1.0 / 6.0,
                               places=12)

    def test_every_face_winds_outward(self):
        hull = convex_hull(_cloud(200))
        coords = asarray(hull.coords)
        tris = _triangles(hull)
        center = coords.mean(axis=1)
        for k in range(tris.shape[1]):
            (a, b, c) = (coords[:, tris[0, k]], coords[:, tris[1, k]],
                         coords[:, tris[2, k]])
            normal = np.cross(b - a, c - a)
            self.assertGreater(float(normal @ (a - center)), 0.0)

    def test_every_point_is_inside(self):
        cloud = _cloud(200)
        hull = convex_hull(cloud)
        coords = asarray(hull.coords)
        tris = _triangles(hull)
        for k in range(tris.shape[1]):
            (a, b, c) = (coords[:, tris[0, k]], coords[:, tris[1, k]],
                         coords[:, tris[2, k]])
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            self.assertTrue((normal @ (cloud - a[:, None]) <= 1e-9).all())

    def test_the_surface_closes(self):
        self.assertTrue(_paired(_triangles(convex_hull(_cloud(100)))))

    def test_euler_s_characteristic_is_the_sphere_s(self):
        hull = convex_hull(_cloud(100))
        tris = _triangles(hull)
        edges = len({tuple(sorted((int(tris[0, k]), int(tris[1, k]))))
                     for k in range(tris.shape[1])}
                    | {tuple(sorted((int(tris[1, k]), int(tris[2, k]))))
                       for k in range(tris.shape[1])}
                    | {tuple(sorted((int(tris[2, k]), int(tris[0, k]))))
                       for k in range(tris.shape[1])})
        self.assertEqual(hull.coord_count - edges + tris.shape[1], 2)

    def test_a_point_on_a_face_is_not_a_corner(self):
        # A point in the middle of a face is on the hull but is not one of its
        # corners; the hull's coordinates are the corners.
        cloud = np.hstack([_cube_corners(), [[0.5], [0.5], [0.0]],
                           [[0.5], [0.0], [0.5]], [[0.0], [0.5], [0.5]]])
        hull = convex_hull(cloud)
        self.assertEqual(hull.coord_count, 8)
        self.assertEqual(hull.topo.simplex_count[2], 12)
        self.assertAlmostEqual(_volume(hull.coords, _triangles(hull)), 1.0,
                               places=12)

    def test_duplicate_points_are_tolerated(self):
        corners = _cube_corners()
        hull = convex_hull(np.hstack([corners, corners, [[0.5], [0.5], [0.5]]]))
        self.assertEqual(hull.coord_count, 8)
        self.assertEqual(hull.topo.simplex_count[2], 12)

    def test_a_rigid_motion_does_not_change_the_hull(self):
        generator = np.random.default_rng(9)
        (rotation, _r) = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] = -rotation[:, 0]
        cloud = _cloud(100)
        first = convex_hull(cloud)
        second = convex_hull(rotation @ cloud + array([[10.0], [-4.0], [7.0]]))
        self.assertEqual(first.coord_count, second.coord_count)
        self.assertAlmostEqual(_volume(first.coords, _triangles(first)),
                               _volume(second.coords, _triangles(second)),
                               places=10)

    def test_a_grid_is_enclosed_at_its_cells(self):
        # A grid's data lives at its cell centers; a 2-by-2-by-2 grid is a cube
        # of centers.
        hull = convex_hull(Grid(eye(4), GridTopology((2, 2, 2))))
        self.assertEqual(hull.coord_count, 8)
        self.assertEqual(hull.topo.simplex_count[2], 12)
        self.assertAlmostEqual(_volume(hull.coords, _triangles(hull)), 1.0,
                               places=12)

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        hull = convex_hull(_cloud(50) * ureg.meter)
        self.assertEqual(str(hull.coords.units), 'meter')

    def test_coplanar_points_are_refused(self):
        flat = np.vstack([_cloud(20, 2), np.full((1, 20), 3.0)])
        with self.assertRaises(ValueError):
            convex_hull(flat)

    def test_collinear_points_are_refused(self):
        line = np.outer(array([1.0, 2.0, 3.0]), np.arange(6.0))
        with self.assertRaises(ValueError):
            convex_hull(line)

    def test_coincident_points_are_refused(self):
        with self.assertRaises(ValueError):
            convex_hull(np.ones((3, 10)))

    def test_three_points_are_refused(self):
        with self.assertRaises(ValueError):
            convex_hull(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0]]))


class TestTheTensorPath(TestCase):
    '''That a tensor is analyzed as a tensor, its graph intact.'''

    def _needs_torch(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_a_tensor_stays_a_tensor(self):
        torch = self._needs_torch()
        cloud = torch.tensor(_cloud(50))
        hull = convex_hull(cloud)
        self.assertIsInstance(hull.coords, torch.Tensor)
        self.assertEqual(hull.coord_count,
                         convex_hull(_cloud(50)).coord_count)
        self.assertGreater(hull.topo.simplex_count[2], 0)

    def test_a_tensor_and_an_array_agree(self):
        torch = self._needs_torch()
        cloud = _cloud(50)
        from_array = convex_hull(cloud)
        from_tensor = convex_hull(torch.tensor(cloud))
        self.assertTrue(allclose(asarray(from_tensor.coords.detach()),
                                 asarray(from_array.coords)))
        self.assertTrue(allclose(asarray(from_tensor.topo.indices),
                                 asarray(from_array.topo.indices)))

    def test_the_gradient_reaches_the_points(self):
        # The hull's corners are coordinates of the input, selected by the
        # hull. The derivative of the selection is the selection.
        torch = self._needs_torch()
        cloud = torch.tensor(_cloud(50), requires_grad=True)
        hull = convex_hull(cloud)
        self.assertTrue(bool(hull.coords.requires_grad))
        hull.coords.sum().backward()
        self.assertIsNotNone(cloud.grad)
        self.assertTrue(bool(torch.isfinite(cloud.grad).all()))
        self.assertTrue(bool((cloud.grad != 0).any()))


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()

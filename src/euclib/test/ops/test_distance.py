# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_distance.py
'''Tests for the distance operations in ``euclib.ops._distance``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, sqrt, stack, zeros

from euclib.ops import distance, nearest, positions_of, separation
from euclib.types import (
    Grid, GridTopology, PrismMesh, PrismTopology, SegPath, SegTopology, TriMesh,
    TriTopology, VertexSet, VertexTopology)


# Fixtures ###################################################################

def _square():
    '''The unit square, split along its main diagonal.

    The corners are 0=(0, 0), 1=(1, 0), 2=(0, 1), 3=(1, 1); the triangles are
    {0, 1, 3} and {0, 3, 2}, which tile the square and share the edge 0-3.
    '''
    return TriMesh(array([[0., 1., 0., 1.],
                          [0., 0., 1., 1.]]),
                   TriTopology([[0, 0], [1, 3], [3, 2]]))


def _path():
    '''A path along the x axis, from (0, 0) to (2, 0).'''
    return SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                   SegTopology([[0, 1], [1, 2]]))


def _points(*pairs):
    '''A point cloud from a sequence of ``(x, y)`` pairs.'''
    return VertexSet(array([[p[0] for p in pairs], [p[1] for p in pairs]]),
                     VertexTopology([list(range(len(pairs)))]))


# Tests ######################################################################

class TestDistance(TestCase):
    '''Tests for ``distance``.'''

    def test_a_position_inside_has_no_distance(self):
        self.assertTrue(allclose(distance(_square(), array([[0.25], [0.25]])),
                                 [0.]))

    def test_a_position_on_the_boundary_has_no_distance(self):
        self.assertTrue(allclose(distance(_square(), array([[0.5], [0.]])),
                                 [0.]))

    def test_a_position_outside_is_measured_to_the_boundary(self):
        # (2, 0) is one unit to the right of the square's right edge.
        self.assertTrue(allclose(distance(_square(), array([[2.], [0.]])),
                                 [1.]))

    def test_a_diagonal_offset_is_measured_to_the_corner(self):
        self.assertTrue(allclose(distance(_square(), array([[2.], [2.]])),
                                 [sqrt(2.0)]))

    def test_distance_to_several_positions(self):
        # Distances from the path to (0, 1), (1, 1), and (3, 1): the last is
        # nearest to the path's end at (2, 0).
        d = distance(_path(), array([[0., 1., 3.], [1., 1., 1.]]))
        self.assertTrue(allclose(d, [1., 1., sqrt(2.0)]))

    def test_distance_from_a_geometry(self):
        other = _points((0., 0.), (0., 2.))
        # The first point is the path's start; the second is two units above
        # the line the path runs along.
        self.assertTrue(allclose(distance(_path(), other), [0., 2.]))

    def test_a_position_inside_a_path_has_no_distance(self):
        self.assertTrue(allclose(distance(_path(), array([[1.], [0.]])), [0.]))

    def test_distance_requires_a_geometry(self):
        with self.assertRaises(TypeError):
            distance('not-a-geometry', array([[0.], [0.]]))

    def test_distance_checks_the_dimension(self):
        with self.assertRaises(ValueError):
            distance(_square(), array([[0.], [0.], [0.]]))


class TestNearest(TestCase):
    '''Tests for ``nearest``.'''

    def test_nearest_position_on_a_mesh(self):
        # The nearest position on the square to (2, 0.5) is on its right edge.
        n = nearest(_square(), array([[2.], [0.5]]))
        self.assertTrue(allclose(n, [[1.], [0.5]], atol=1e-12))

    def test_nearest_position_on_a_diagonal_edge(self):
        # The nearest position to (0.75, 0.75) is on the shared diagonal.
        n = nearest(_square(), array([[0.75], [0.75]]))
        self.assertTrue(allclose(n, [[0.75], [0.75]], atol=1e-12))

    def test_nearest_position_on_a_path_clamps_to_an_end(self):
        n = nearest(_path(), array([[-1.], [0.]]))
        self.assertTrue(allclose(n, [[0.], [0.]], atol=1e-12))

    def test_nearest_agrees_with_distance(self):
        query = array([[2.], [0.5], [0.]][:2])
        mesh = _square()
        gap = distance(mesh, query)
        offset = query - nearest(mesh, query)
        self.assertTrue(allclose(gap, (offset * offset).sum(axis=0) ** 0.5,
                                 atol=1e-12))


class TestGeometriesThatAreNotCoordinateMatrices(TestCase):
    '''Distances involving a geometry whose data is not at its coords.

    A grid's coords is an affine matrix, and a prism mesh's data lies on both
    of its surfaces, so a distance must be measured from what `positions_of`
    names rather than from `coords` directly.
    '''

    def _mesh(self):
        '''A square in the z = 0 plane, spanning the unit square.'''
        return TriMesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.],
                              [0., 0., 0., 0.]]),
                       TriTopology([[0, 0], [1, 3], [3, 2]]))

    def _grid(self, lift=0.):
        '''A grid of unit cells, optionally lifted off the plane.'''
        return Grid(array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., lift], [0., 0., 0., 1.]]),
                    GridTopology((3, 3, 3)))

    def test_distance_from_a_grids_cells(self):
        # One distance per cell, not one per entry of the affine matrix.
        found = distance(self._mesh(), self._grid())
        self.assertEqual(array(found).shape, (27,))
        self.assertTrue(allclose(found,
                                 distance(self._mesh(),
                                          positions_of(self._grid()))))

    def test_the_distance_from_a_lifted_grid(self):
        # The mesh is in the plane z = 0 and the grid five units above it. The
        # cells directly over the square are five, six, and seven units away;
        # a cell offset along x or y is further, because its nearest point on
        # the square is an edge rather than the point below it.
        found = array(distance(self._mesh(), self._grid(5.)))
        self.assertTrue(allclose(found[:3], [5., 6., 7.]))
        # A cell two units along x is one unit past the square's edge.
        self.assertAlmostEqual(float(found[18]), sqrt(1. + 25.))

    def test_distance_to_a_grid(self):
        # The mesh's corners all lie in the grid's first plane.
        self.assertTrue(allclose(distance(self._grid(), self._mesh()),
                                 zeros(4)))

    def test_separation_from_a_grid(self):
        self.assertAlmostEqual(float(separation(self._mesh(), self._grid())),
                               0.)
        self.assertAlmostEqual(
            float(separation(self._mesh(), self._grid(5.))), 5.)

    def test_measuring_from_a_prism_meshs_surfaces(self):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [2.]])]),
                          PrismTopology([[0], [1], [2]]))
        # A prism's data lies on both surfaces, so there are two positions per
        # prism corner.
        self.assertEqual(array(distance(self._grid(), prism)).shape, (6,))

    def test_measuring_to_a_prism(self):
        # The prism spans z from 0 to 2; the grid five units up has its cells
        # at z = 5, so the nearest position on the prism is three units away,
        # on its upper surface.
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [2.]])]),
                          PrismTopology([[0], [1], [2]]))
        found = array(distance(prism, self._grid(5.)))
        self.assertEqual(found.shape, (27,))
        # A cell directly over the prism's first corner is three units above
        # the upper surface; one past the prism's edge is further.
        self.assertAlmostEqual(float(found[0]), 3.)
        self.assertGreater(float(found[18]), 3.)

    def test_measuring_to_a_prism_from_inside_and_below(self):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [2.]])]),
                          PrismTopology([[0], [1], [2]]))
        # A position inside the prism has no distance to it.
        self.assertAlmostEqual(
            float(distance(prism, array([[0.2], [0.2], [1.]]))[0]), 0.)
        # A position below it is measured to the lower surface.
        self.assertAlmostEqual(
            float(distance(prism, array([[0.2], [0.2], [-3.]]))[0]), 3.)


class TestSeparation(TestCase):
    '''Tests for ``separation``.'''

    def test_touching_geometries_are_zero_apart(self):
        self.assertTrue(allclose(separation(_square(), _points((0.5, 0.))),
                                 [0.]))

    def test_a_geometry_inside_another_is_zero_apart(self):
        self.assertTrue(allclose(separation(_square(), _points((0.5, 0.5))),
                                 [0.]))

    def test_separated_geometries(self):
        self.assertTrue(allclose(separation(_square(), _points((0.5, 5.))),
                                 [4.]))

    def test_separation_is_the_smallest_distance(self):
        self.assertTrue(
            allclose(separation(_square(), _points((0.5, 5.), (0.5, 2.))),
                     [1.]))

    def test_separated_paths(self):
        other = SegPath(array([[0., 1.], [3., 3.]]), SegTopology([[0], [1]]))
        # The nearest point of the path at y = 0 to either end is its start.
        self.assertTrue(allclose(separation(_path(), other), [3.]))

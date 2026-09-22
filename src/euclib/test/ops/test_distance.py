# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_distance.py
'''Tests for the distance operations in ``euclib.ops._distance``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, sqrt

from euclib.ops import distance, nearest, separation
from euclib.types import (
    SegPath, TriMesh, SegTopology, TriTopology, VertexSet, VertexTopology)


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

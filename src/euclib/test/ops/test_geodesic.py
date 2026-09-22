# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_geodesic.py
'''Tests for the geodesic distances in ``euclib.ops._geodesic``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import (
    allclose, array, inf, isinf, ones, sqrt, zeros)

from euclib.ops import distance, geodesic
from euclib.types import (
    Grid, GridTopology, SegPath, SegTopology, TriMesh, TriTopology, VertexSet,
    VertexTopology)


# Fixtures ###################################################################

def _path(*points):
    '''A path through a sequence of points, one segment per consecutive pair.'''
    coords = array(points).T
    indices = [list(range(len(points) - 1)), list(range(1, len(points)))]
    return SegPath(coords, SegTopology(indices))


def _square():
    '''The unit square, split along its main diagonal.'''
    return TriMesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.]]),
                   TriTopology([[0, 0], [1, 3], [3, 2]]))


def _independent_dijkstra(coords, edges, source):
    '''A shortest-path search written out, to check the one in the library.

    Returns a length-``N`` list of distances from ``source``, with ``inf`` for
    coordinates no path reaches.
    '''
    count = coords.shape[1]
    neighbours = [[] for _ in range(count)]
    for e in range(edges.shape[1]):
        (a, b) = (int(edges[0, e]), int(edges[1, e]))
        gap = coords[:, a] - coords[:, b]
        length = float(sqrt((gap * gap).sum()))
        neighbours[a].append((b, length))
        neighbours[b].append((a, length))
    best = [inf] * count
    best[source] = 0.0
    settled = [False] * count
    while True:
        # The unsettled coordinate with the least distance so far.
        candidates = [(d, i) for (i, d) in enumerate(best) if not settled[i]
                      and not isinf(d)]
        if not candidates:
            break
        (_, here) = min(candidates)
        settled[here] = True
        for (there, length) in neighbours[here]:
            if best[here] + length < best[there]:
                best[there] = best[here] + length
    return best


# Tests ######################################################################

class TestAlongAPath(TestCase):
    '''Distances along a path are the distances along its segments.'''

    def test_from_one_end(self):
        path = _path((0., 0.), (3., 0.), (4., 0.))
        self.assertTrue(allclose(geodesic(path, array([0])), [0., 3., 4.]))

    def test_from_the_other_end(self):
        path = _path((0., 0.), (3., 0.), (4., 0.))
        self.assertTrue(allclose(geodesic(path, array([2])), [4., 1., 0.]))

    def test_a_boolean_mask_names_the_sources(self):
        path = _path((0., 0.), (3., 0.), (4., 0.))
        self.assertTrue(allclose(geodesic(path, array([True, False, False])),
                                 [0., 3., 4.]))

    def test_several_sources_take_the_nearest(self):
        path = _path((0., 0.), (3., 0.), (4., 0.))
        self.assertTrue(allclose(geodesic(path, array([0, 2])), [0., 1., 0.]))


class TestAlongASurface(TestCase):
    '''Distances along a surface follow it, not the space around it.'''

    def test_across_a_flat_square(self):
        # The far corner is the diagonal's length away, across the surface.
        self.assertTrue(allclose(geodesic(_square(), array([0])),
                                 [0., 1., 1., sqrt(2.)], atol=1e-12))

    def test_a_folded_strip(self):
        # A path that goes far out and comes back: its last coordinate is near
        # the first through space and far from it along the path.
        folded = _path((0., 0.), (10., 0.), (0.5, 0.))
        # Ten units out and nine and a half back.
        self.assertTrue(allclose(geodesic(folded, array([0])),
                                 [0., 10., 19.5], atol=1e-12))
        # A position half a unit off the path is half a unit from it through
        # space, whatever the path's own length between the two is.
        self.assertAlmostEqual(
            float(distance(folded, array([[0.5], [0.5]]))[0]), 0.5)

    def test_the_geodesic_is_at_least_the_straight_distance(self):
        mesh = TriMesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.],
                              [0., 0., 0., 0.]]),
                       TriTopology([[0, 0], [1, 3], [3, 2]]))
        field = geodesic(mesh, array([0]))
        straight = distance(mesh, mesh.coords)
        self.assertTrue((field >= array([float(d) for d in straight]) - 1e-12)
                        .all())

    def test_through_a_tetrahedral_mesh(self):
        from euclib.types import TetMesh, TetTopology
        tet = TetMesh(array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                             [0., 0., 0., 1.]]),
                      TetTopology([[0], [1], [2], [3]]))
        field = geodesic(tet, array([0]))
        # Every other corner is one tet edge away.
        self.assertTrue(allclose(sorted(field), [0., 1., 1., 1.], atol=1e-12))


class TestEdgesOfTheDefinition(TestCase):
    '''Cases the definition insists on.'''

    def test_a_point_cloud_has_no_paths(self):
        cloud = VertexSet(array([[0., 1., 2.], [0., 0., 0.]]),
                          VertexTopology([[0, 1, 2]]))
        field = geodesic(cloud, array([1]))
        self.assertEqual(field.tolist(), [inf, 0., inf])

    def test_no_source_leaves_everything_unreachable(self):
        self.assertTrue(isinf(geodesic(_square(), array([], dtype=int))).all())

    def test_a_grid_is_not_yet_supported(self):
        with self.assertRaises(NotImplementedError):
            geodesic(Grid(zeros((3, 3)) + [[1., 0., 0.], [0., 1., 0.],
                                           [0., 0., 1.]],
                          GridTopology((4, 4))), array([0]))

    def test_it_checks_its_arguments(self):
        with self.assertRaises(TypeError):
            geodesic('not-a-geometry', array([0]))
        with self.assertRaises(ValueError):
            geodesic(_square(), array([True, False]))
        with self.assertRaises(ValueError):
            geodesic(_square(), array([99]))


class TestAgainstAnIndependentSearch(TestCase):
    '''The library's shortest paths against a search written out here.'''

    def test_a_path(self):
        path = _path((0., 0.), (3., 0.), (4., 0.), (4., 5.))
        for source in range(4):
            expected = _independent_dijkstra(path.coords,
                                             path.topo.simplices[1], source)
            with self.subTest(source=source):
                self.assertTrue(allclose(geodesic(path, array([source])),
                                         expected, atol=1e-12))

    def test_a_square(self):
        mesh = _square()
        for source in range(mesh.coord_count):
            expected = _independent_dijkstra(mesh.coords,
                                             mesh.topo.simplices[1], source)
            with self.subTest(source=source):
                self.assertTrue(allclose(geodesic(mesh, array([source])),
                                         expected, atol=1e-12))

    def test_a_three_dimensional_mesh(self):
        # A folded sheet of four triangles: the shortest path may run through
        # any of several routes.
        # A square whose far corner is lifted, so the sheet is folded and the
        # shortest path across it need not be the straight one.
        mesh = TriMesh(array([[0., 1., 0., 1.],
                              [0., 0., 1., 1.],
                              [0., 0., 0., 1.]]),
                       TriTopology([[0, 0], [1, 3], [3, 2]]))
        for source in range(mesh.coord_count):
            expected = _independent_dijkstra(mesh.coords,
                                             mesh.topo.simplices[1], source)
            with self.subTest(source=source):
                self.assertTrue(allclose(geodesic(mesh, array([source])),
                                         expected, atol=1e-12))

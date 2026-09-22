# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_geom.py
'''Tests for the concrete geometries in ``euclib.types._geom``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, zeros

from euclib.abc import is_geometry, is_simplex_geometry
from euclib.types import (
    VertexSet, SegPath, TriMesh, TetMesh,
    VertexTopology, SegTopology, TriTopology, TetTopology)


# Fixtures ###################################################################

def _cloud():
    '''A three-point cloud along the x axis.'''
    return VertexSet(array([[0., 1., 2.], [0., 0., 0.]]),
                     VertexTopology([[0, 1, 2]]))


def _path():
    '''An L-shaped path of two unit segments.'''
    return SegPath(array([[0., 1., 1.], [0., 0., 1.]]),
                   SegTopology([[0, 1], [1, 2]]))


def _mesh():
    '''Two triangles sharing an edge: 4 coordinates, 5 edges, 2 triangles.'''
    return TriMesh(array([[0., 1., 0., 1.],
                          [0., 0., 1., 1.],
                          [0., 0., 0., 0.]]),
                   TriTopology([[0, 0], [1, 2], [2, 3]]))


def _tetmesh():
    '''The unit tetrahedron spanned by the origin and the three axes.'''
    return TetMesh(array([[0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]),
                   TetTopology([[0], [1], [2], [3]]))


# Tests ######################################################################

class TestVertexSet(TestCase):
    '''Tests for the point cloud.'''

    def test_basics(self):
        cloud = _cloud()
        self.assertTrue(is_geometry(cloud))
        self.assertTrue(is_simplex_geometry(cloud))
        self.assertEqual(cloud.order, 0)
        self.assertEqual(cloud.coord_count, 3)
        self.assertEqual(cloud.dim, 2)

    def test_to_local_finds_the_nearest_point(self):
        cloud = _cloud()
        loc = cloud.to_local(array([[0.1, 1.9], [0.0, 0.1]]))
        self.assertEqual(loc.index.tolist(), [0, 2])

    def test_to_global_returns_the_point(self):
        cloud = _cloud()
        loc = cloud.to_local(array([[1.9], [0.1]]))
        self.assertTrue(allclose(cloud.to_global(loc), [[2.], [0.]]))

    def test_a_point_cloud_has_no_measure(self):
        cloud = _cloud()
        self.assertTrue(allclose(cloud.measures, [0., 0., 0.]))

    def test_a_point_cloud_computes_no_properties(self):
        cloud = _cloud()
        self.assertEqual(len(cloud.simplex_properties[0]), 0)


class TestSegPath(TestCase):
    '''Tests for the path.'''

    def test_basics(self):
        path = _path()
        self.assertEqual(path.order, 1)
        self.assertEqual(path.coord_count, 3)
        self.assertEqual(list(path.simplex_count), [3, 2])

    def test_measures_are_segment_lengths(self):
        measures = _path().measures
        self.assertTrue(allclose(measures, [1., 1.]))

    def test_length_is_exposed_as_a_simplex_property(self):
        path = _path()
        self.assertIn('length', path.simplex_properties[1])
        self.assertTrue(allclose(path[1, 'length'], [1., 1.]))

    def test_a_user_property_overrides_the_computed_one(self):
        path = _path().withprop((1, 'length'), array([9., 9.]))
        self.assertTrue(allclose(path[1, 'length'], [9., 9.]))

    def test_to_local_finds_the_position_along_a_segment(self):
        path = _path()
        loc = path.to_local(array([[0.5, 1.0], [0.5, 0.5]]))
        self.assertEqual(loc.index.tolist(), [0, 1])
        self.assertTrue(allclose(loc.weight, [[0.5, 0.5]]))

    def test_positions_outside_clamp_to_the_ends(self):
        path = _path()
        # Before the start of the first segment, and past the end of the second.
        # A weight is the first barycentric coordinate: 1 at a segment's first
        # corner and 0 at its second. (-1, 0) is nearest the first segment's
        # start; (5, 1) is nearest the second segment's end.
        loc = path.to_local(array([[-1.0, 5.0], [0.0, 1.0]]))
        self.assertTrue(allclose(loc.weight, [[1.0, 0.0]]))

    def test_round_trip(self):
        # Only positions that lie on the path round-trip: a path is 1-dimensional,
        # so a position off it is answered with the nearest position on it.
        path = _path()
        query = array([[0.5, 1.0, 0.0], [0.0, 0.5, 0.0]])
        loc = path.to_local(query)
        self.assertTrue(allclose(path.to_global(loc), query, atol=1e-12))

    def test_the_inverse_of_to_local_is_to_global(self):
        path = _path()
        loc = path.to_local(path.coords)
        self.assertTrue(allclose(path.to_global(loc), path.coords, atol=1e-12))


class TestTriMesh(TestCase):
    '''Tests for the triangle mesh.'''

    def test_basics(self):
        mesh = _mesh()
        self.assertEqual(mesh.order, 2)
        self.assertEqual(mesh.coord_count, 4)
        self.assertEqual(list(mesh.simplex_count), [4, 5, 2])

    def test_measures_are_triangle_areas(self):
        self.assertTrue(allclose(_mesh().measures, [0.5, 0.5]))

    def test_surface_area_is_exposed_as_a_simplex_property(self):
        mesh = _mesh()
        self.assertIn('surface_area', mesh.simplex_properties[2])
        self.assertTrue(allclose(mesh[2, 'surface_area'], [0.5, 0.5]))

    def test_a_2d_triangle_has_area(self):
        mesh = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                       TriTopology([[0], [1], [2]]))
        self.assertTrue(allclose(mesh.measures, [0.5]))

    def test_to_local_finds_the_triangle_and_weights(self):
        mesh = _mesh()
        # The triangle {0, 1, 2} has corners (0, 0), (1, 0), and (0, 1), so a
        # position at (0.25, 0.25) has barycentric coordinates (0.5, 0.25, 0.25),
        # of which the first two are kept.
        loc = mesh.to_local(array([[0.25], [0.25], [0.0]]))
        self.assertEqual(loc.index.tolist(), [0])
        self.assertTrue(allclose(loc.weight, [[0.5], [0.25]], atol=1e-12))

    def test_round_trip(self):
        mesh = _mesh()
        query = array([[0.25, 0.75, 0.5], [0.25, 0.25, 0.5], [0.0, 0.0, 0.0]])
        loc = mesh.to_local(query)
        self.assertTrue(allclose(mesh.to_global(loc), query, atol=1e-12))

    def test_positions_outside_are_answered_with_the_boundary(self):
        mesh = _mesh()
        loc = mesh.to_local(array([[2.0], [2.0], [0.0]]))
        # The nearest point of the unit square to (2, 2) is its far corner.
        self.assertTrue(allclose(mesh.to_global(loc), [[1.], [1.], [0.]],
                                 atol=1e-12))


class TestTetMesh(TestCase):
    '''Tests for the tetrahedral mesh.'''

    def test_basics(self):
        tet = _tetmesh()
        self.assertEqual(tet.order, 3)
        self.assertEqual(tet.coord_count, 4)
        self.assertEqual(tet.dim, 3)

    def test_measures_are_volumes(self):
        self.assertTrue(allclose(_tetmesh().measures, [1. / 6.]))

    def test_volume_is_exposed_as_a_simplex_property(self):
        tet = _tetmesh()
        self.assertIn('volume', tet.simplex_properties[3])
        self.assertTrue(allclose(tet[3, 'volume'], [1. / 6.]))

    def test_to_local_and_round_trip(self):
        tet = _tetmesh()
        query = array([[0.25, 0.1], [0.25, 0.2], [0.25, 0.3]])
        loc = tet.to_local(query)
        self.assertEqual(loc.index.tolist(), [0, 0])
        self.assertTrue(allclose(tet.to_global(loc), query, atol=1e-12))

    def test_a_tetrahedral_mesh_requires_three_dimensions(self):
        with self.assertRaises(Exception):
            TetMesh(zeros((2, 4)), TetTopology([[0], [1], [2], [3]]))


class TestMeasures(TestCase):
    '''Measures in general: correctness and units.'''

    def test_a_scaled_path_has_scaled_lengths(self):
        path = SegPath(array([[0., 3.], [0., 0.]]), SegTopology([[0], [1]]))
        self.assertTrue(allclose(path.measures, [3.]))

    def test_a_unit_triangle_in_3d_has_area(self):
        # A right triangle in the z = 0 plane.
        mesh = TriMesh(array([[0., 1., 0.],
                              [0., 0., 1.],
                              [0., 0., 0.]]),
                       TriTopology([[0], [1], [2]]))
        self.assertTrue(allclose(mesh.measures, [0.5]))

    def test_measures_carry_the_units_of_the_coordinates(self):
        import pint
        ureg = pint.UnitRegistry()
        coords = array([[0., 3.], [0., 0.]]) * ureg.meter
        path = SegPath(coords, SegTopology([[0], [1]]))
        self.assertEqual(str(path.measures.units), 'meter')
        # An area has units of length squared.
        mesh = TriMesh(array([[0., 1., 0.],
                              [0., 0., 1.],
                              [0., 0., 0.]]) * ureg.meter,
                       TriTopology([[0], [1], [2]]))
        self.assertEqual(str(mesh.measures.units), 'meter ** 2')


class TestLocalRoundTrip(TestCase):
    '''``to_global`` inverts ``to_local`` for every concrete geometry.'''

    def test_every_geometry_round_trips_its_own_coordinates(self):
        for geom in (_cloud(), _path(), _mesh(), _tetmesh()):
            with self.subTest(geom=type(geom).__name__):
                loc = geom.to_local(geom.coords)
                self.assertTrue(allclose(geom.to_global(loc), geom.coords,
                                         atol=1e-12))


class TestSpatialIndex(TestCase):
    '''The spatial index that makes the searches pay on large geometries.'''

    def setUp(self):
        import euclib._init as init
        self.init = init
        self.was = init.spatial_index_min_items
        self.addCleanup(setattr, init, 'spatial_index_min_items', self.was)

    def _mesh(self, n):
        '''A triangulated n by n grid: small, well-shaped triangles.'''
        from numpy import linspace, meshgrid, stack
        (xs, ys) = meshgrid(linspace(0, 1, n), linspace(0, 1, n))
        coords = stack([xs.reshape(-1), ys.reshape(-1)])
        triangles = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = a + 1
                c = a + n
                d = c + 1
                triangles.append([a, b, d])
                triangles.append([a, d, c])
        return TriMesh(coords, TriTopology(array(triangles).T))

    def test_a_small_geometry_builds_no_index(self):
        self.init.spatial_index_min_items = 10 ** 9
        self.assertIsNone(self._mesh(4).spatial_index)

    def test_a_large_geometry_builds_one(self):
        self.init.spatial_index_min_items = 10
        self.assertIsNotNone(self._mesh(6).spatial_index)

    def test_the_index_is_a_tree_of_the_right_dimension(self):
        from euclib.utils import SpatialTree
        self.init.spatial_index_min_items = 10
        mesh = self._mesh(6)
        self.assertIsInstance(mesh.spatial_index, SpatialTree)
        self.assertEqual(mesh.spatial_index.dim, 2)

    def test_the_index_gives_the_same_answers_as_no_index(self):
        # The point of the index is to skip simplices without skipping the
        # answer, so the two paths must agree exactly.
        from numpy import allclose
        query = array([[0.13, 0.62, 0.87, -0.2, 1.1],
                       [0.29, 0.07, 0.41, 0.5, 1.4]])
        self.init.spatial_index_min_items = 10 ** 9
        plain = self._mesh(12).to_local(query)
        self.init.spatial_index_min_items = 10
        indexed = self._mesh(12).to_local(query)
        self.assertTrue(allclose(indexed.index, plain.index))
        self.assertTrue(allclose(indexed.weight, plain.weight, atol=1e-12))

    def test_the_index_answers_a_position_outside_the_geometry(self):
        # A position outside every element has no home cell; the search grows
        # its radius until it reaches the nearest one.
        from numpy import allclose
        self.init.spatial_index_min_items = 10
        mesh = self._mesh(12)
        loc = mesh.to_local(array([[5.], [5.]]))
        self.assertTrue(allclose(mesh.to_global(loc), [[1.], [1.]], atol=1e-12))

    def test_a_point_cloud_is_indexed_too(self):
        from numpy import linspace, stack
        from euclib.utils import SpatialTree
        from euclib.types import VertexSet, VertexTopology
        self.init.spatial_index_min_items = 10
        cloud = VertexSet(stack([linspace(0, 1, 30), linspace(0, 1, 30)]),
                          VertexTopology([list(range(30))]))
        self.assertIsInstance(cloud.spatial_index, SpatialTree)
        # A position near the tenth point is answered with it.
        loc = cloud.to_local(array([[0.3], [0.3]]))
        self.assertEqual(int(loc.index[0]), 9)


class TestTorchBackend(TestCase):
    '''The Torch backend, including the gradients ``euclib`` must preserve.'''

    def setUp(self):
        from euclib._init import checktorch
        if checktorch() is None:
            self.skipTest('torch is not installed')
        self.torch = checktorch()

    def test_coordinates_are_converted_to_the_backend(self):
        t = self.torch
        path = SegPath(t.tensor([[0., 1., 1.], [0., 0., 1.]]),
                       SegTopology([[0, 1], [1, 2]]), backend='torch')
        self.assertIsInstance(path.coords, t.Tensor)
        self.assertEqual(path.dim, 2)
        self.assertEqual(path.coord_count, 3)

    def test_measures_keep_their_gradient(self):
        t = self.torch
        coords = t.tensor([[0., 3.], [0., 0.]], requires_grad=True)
        path = SegPath(coords, SegTopology([[0], [1]]), backend='torch')
        measures = path.measures
        self.assertIsNotNone(measures.grad_fn)
        measures.sum().backward()
        # The length is |x1 - x0|, so its gradient is -1 at the first
        # coordinate and +1 at the second.
        self.assertTrue(allclose(coords.grad.tolist(), [[-1., 1.], [0., 0.]]))

    def test_to_global_is_differentiable(self):
        t = self.torch
        coords = t.tensor([[0., 1., 1.], [0., 0., 1.]], requires_grad=True)
        path = SegPath(coords, SegTopology([[0, 1], [1, 2]]), backend='torch')
        loc = path.topo.Loc(t.tensor([0]), t.tensor([[0.5]]))
        g = path.to_global(loc)
        self.assertIsNotNone(g.grad_fn)
        # The position is halfway between the first two coordinates, so each
        # row of the result depends on both of that row's first two
        # coordinates, and on neither row's third.
        self.assertTrue(allclose(g.tolist(), [[0.5], [0.0]]))
        g.sum().backward()
        self.assertTrue(allclose(coords.grad.tolist(),
                                 [[0.5, 0.5, 0.], [0.5, 0.5, 0.]]))

    def test_point_location_works_on_a_tensor_that_needs_a_gradient(self):
        # Locating a position is a selection, so it cannot be differentiated;
        # it must nonetheless accept a tensor that carries a gradient.
        t = self.torch
        coords = t.tensor([[0., 1., 1.], [0., 0., 1.]], requires_grad=True)
        path = SegPath(coords, SegTopology([[0, 1], [1, 2]]), backend='torch')
        loc = path.to_local(t.tensor([[0.5], [0.5]]))
        self.assertEqual(loc.index.tolist(), [0])
        self.assertTrue(allclose(loc.weight, [[0.5]], atol=1e-6))

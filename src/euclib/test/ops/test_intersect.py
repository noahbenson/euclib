# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_intersect.py
'''Tests for the intersection operations in ``euclib.ops._intersect``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, eye, sort

from euclib.ops import (
    contains, mesh_intersections, path_crossings, path_intersections,
    tolerance_of, voxel_intersections)
from euclib.types import (
    Grid, GridTopology, SegPath, SegTopology, TetMesh, TetTopology, TriMesh,
    TriTopology, VertexSet, VertexTopology, affine_translation)


# Fixtures ###################################################################

def _path(*points):
    '''A path through a sequence of points, one segment per consecutive pair.'''
    coords = array(points).T
    indices = [list(range(len(points) - 1)), list(range(1, len(points)))]
    return SegPath(coords, SegTopology(indices))


def _square():
    '''The unit square in the z = 0 plane, split along its main diagonal.'''
    return TriMesh(array([[0., 1., 0., 1.],
                          [0., 0., 1., 1.],
                          [0., 0., 0., 0.]]),
                   TriTopology([[0, 0], [1, 3], [3, 2]]))


def _tet():
    '''The unit tetrahedron spanned by the origin and the three axes.'''
    return TetMesh(array([[0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]),
                   TetTopology([[0], [1], [2], [3]]))


# Tests ######################################################################

class TestTolerance(TestCase):
    '''The tolerance that all of these operations work to.'''

    def test_it_scales_with_the_geometry(self):
        small = tolerance_of(_square())
        big = tolerance_of(_square().transformed(
            __import__('euclib.types', fromlist=['Affine']).affine_scaling(
                [1000., 1000., 1000.])))
        self.assertLess(small, big)
        self.assertAlmostEqual(big / small, 1000.)

    def test_it_does_not_warn_about_units(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            tolerance_of(_square())


class TestPathCrossings(TestCase):
    '''Where a path crosses a triangle mesh.'''

    def test_a_segment_that_stabs_the_square(self):
        # The square is split along the diagonal y = x, so a segment that
        # crosses away from that diagonal meets exactly one triangle.
        path = _path((0.25, 0.5, -1.0), (0.25, 0.5, 1.0))
        (points, segments, triangles) = path_crossings(path, _square())
        self.assertEqual(points.shape[1], 1)
        self.assertTrue(allclose(points.ravel(), [0.25, 0.5, 0.0], atol=1e-9))

    def test_a_segment_along_the_shared_edge_meets_both_triangles(self):
        # A crossing that lands on the diagonal belongs to both triangles, and
        # both are reported.
        path = _path((0.5, 0.5, -1.0), (0.5, 0.5, 1.0))
        (points, _, triangles) = path_crossings(path, _square())
        self.assertEqual(points.shape[1], 2)
        self.assertEqual(sorted(triangles.tolist()), [0, 1])
        self.assertTrue(allclose(points[0], [0.5, 0.5]))
        self.assertTrue(allclose(points[1], [0.5, 0.5]))

    def test_a_segment_that_meets_the_surface_at_its_end(self):
        # Starting on the surface and going up: the crossing is at the start.
        path = _path((0.25, 0.5, 0.0), (0.25, 0.5, 2.0))
        (points, _, _) = path_crossings(path, _square())
        self.assertEqual(points.shape[1], 1)
        self.assertTrue(allclose(points.ravel(), [0.25, 0.5, 0.0], atol=1e-9))

    def test_a_segment_that_misses(self):
        path = _path((5., 5., -1.), (5., 5., 1.))
        self.assertEqual(path_crossings(path, _square())[0].shape[1], 0)

    def test_the_crossings_name_their_segment_and_triangle(self):
        path = _path((0.25, 0.5, -1.), (0.25, 0.5, 1.))
        (_, segments, triangles) = path_crossings(path, _square())
        self.assertEqual(segments.tolist(), [0])
        self.assertEqual(triangles.shape, (1,))
        self.assertIn(int(triangles[0]), (0, 1))

    def test_it_checks_its_arguments(self):
        with self.assertRaises(TypeError):
            path_crossings('not-a-path', _square())
        with self.assertRaises(TypeError):
            path_crossings(_path((0., 0., 0.), (1., 1., 1.)), 'not-a-mesh')

    def test_it_requires_three_dimensions(self):
        flat = SegPath(array([[0., 1.], [0., 1.]]), SegTopology([[0], [1]]))
        with self.assertRaises(ValueError):
            path_crossings(flat, _square())


class TestPathIntersections(TestCase):
    '''Where two paths cross.'''

    def test_two_segments_that_cross(self):
        first = _path((0., 0.), (2., 2.))
        second = _path((0., 2.), (2., 0.))
        (points, mine, theirs) = path_intersections(first, second)
        self.assertEqual(points.shape[1], 1)
        self.assertTrue(allclose(points.ravel(), [1., 1.], atol=1e-9))
        self.assertEqual(mine.tolist(), [0])
        self.assertEqual(theirs.tolist(), [0])

    def test_segments_that_do_not_cross(self):
        first = _path((0., 0.), (2., 2.))
        second = _path((0., 5.), (2., 5.))
        self.assertEqual(path_intersections(first, second)[0].shape[1], 0)

    def test_parallel_segments_do_not_cross(self):
        first = _path((0., 0.), (2., 0.))
        second = _path((0., 1.), (2., 1.))
        self.assertEqual(path_intersections(first, second)[0].shape[1], 0)

    def test_touching_segments_cross(self):
        first = _path((0., 0.), (2., 0.))
        second = _path((1., 0.), (1., 2.))
        self.assertEqual(path_intersections(first, second)[0].shape[1], 1)

    def test_it_checks_its_arguments(self):
        with self.assertRaises(TypeError):
            path_intersections('not-a-path', _path((0., 0.), (1., 1.)))
        flat = SegPath(array([[0., 1.], [0., 1.]]), SegTopology([[0], [1]]))
        with self.assertRaises(ValueError):
            path_intersections(flat, _path((0., 0., 0.), (1., 1., 1.)))


class TestContains(TestCase):
    '''Whether a position lies on or within a geometry.'''

    def test_within_a_tetrahedron(self):
        points = array([[0.2, 0.2, 5.0, 0.1],
                        [0.2, 0.2, 5.0, 0.1],
                        [0.2, 0.2, 5.0, 0.9]])
        self.assertEqual(contains(_tet(), points).tolist(),
                         [True, True, False, False])

    def test_on_a_triangle_mesh(self):
        # A mesh is a surface, so only positions on it belong to it.
        points = array([[0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.0, 1.0, 0.0]])
        self.assertEqual(contains(_square(), points).tolist(),
                         [True, False, True])

    def test_within_a_grid(self):
        g = Grid(array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]]),
                 GridTopology((4, 4)))
        points = array([[1.5, 9.0], [1.5, 9.0]])
        self.assertEqual(contains(g, points).tolist(), [True, False])

    def test_a_point_cloud_contains_its_own_points(self):
        cloud = VertexSet(array([[0., 1., 2.], [0., 0., 0.]]),
                          VertexTopology([[0, 1, 2]]))
        points = array([[0.0, 1.0, 1.5], [0.0, 0.0, 0.0]])
        self.assertEqual(contains(cloud, points).tolist(),
                         [True, True, False])

    def test_it_checks_its_argument(self):
        with self.assertRaises(TypeError):
            contains('not-a-geometry', array([[0.], [0.]]))


class TestVoxelIntersections(TestCase):
    '''The overlap of a tetrahedral mesh and a grid.'''

    def _grid(self):
        '''A grid of unit voxels covering the positive octant.

        A cell's index names its centre, so the affine is the identity shifted
        by half a step: cell ``(i, j, k)`` is then centred at ``(i+.5, ...)``
        and covers ``[i, i+1] x [j, j+1] x [k, k+1]``, which is what makes the
        unit tetrahedron at the origin lie inside cell ``(0, 0, 0)``.
        '''
        return Grid(affine_translation([0.5, 0.5, 0.5]).matrix,
                    GridTopology((3, 3, 3)))

    def test_an_identity_grid_centres_its_first_cell_on_the_origin(self):
        # The same statement the fixture relies on, made where it cannot be
        # missed: an unshifted identity grid's first cell is centred at the
        # origin and reaches half a unit in every direction.
        unit = Grid(eye(4), GridTopology((3, 3, 3)))
        self.assertTrue(allclose(unit.origin, [-0.5, -0.5, -0.5]))
        self.assertTrue(allclose(unit.bbox,
                                 [[-0.5, 2.5], [-0.5, 2.5], [-0.5, 2.5]]))
        self.assertTrue(all(contains(unit, array([[0.], [0.], [0.]]))))

    def test_a_tetrahedron_inside_one_voxel(self):
        (pieces, tetrahedra, voxels) = voxel_intersections(_tet(), self._grid())
        self.assertEqual(pieces.topo.simplex_count[3], 1)
        self.assertAlmostEqual(float(sum(pieces.measures)), 1. / 6.)
        self.assertEqual(tetrahedra.tolist(), [0])
        self.assertEqual(voxels.ravel().tolist(), [0, 0, 0])

    def test_a_tetrahedron_spanning_several_voxels(self):
        big = TetMesh(array([[0., 2., 0., 0.], [0., 0., 2., 0.],
                             [0., 0., 0., 2.]]),
                      TetTopology([[0], [1], [2], [3]]))
        (pieces, _, voxels) = voxel_intersections(big, self._grid())
        # Its eightfold volume is cut among the four voxels it reaches.
        self.assertAlmostEqual(float(sum(pieces.measures)), 8. / 6.)
        self.assertEqual(sorted({tuple(v) for v in voxels.T.tolist()}),
                         [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])

    def test_a_mesh_that_misses_the_grid(self):
        far = TetMesh(array([[5., 6., 5., 5.], [5., 5., 6., 5.],
                             [5., 5., 5., 6.]]),
                      TetTopology([[0], [1], [2], [3]]))
        (pieces, tetrahedra, voxels) = voxel_intersections(far, self._grid())
        # An empty result is an empty mesh, not a mesh holding a placeholder.
        self.assertEqual(pieces.topo.simplex_count[3], 0)
        self.assertEqual(pieces.coord_count, 0)
        self.assertEqual(tetrahedra.shape, (0,))
        self.assertEqual(voxels.shape, (3, 0))

    def test_the_cutting_happens_in_index_space(self):
        # A finer grid cuts the tetrahedron into more pieces, and their total
        # volume is its own however many there are.
        grid = Grid(array([[0.5, 0., 0., 0.], [0., 0.5, 0., 0.],
                           [0., 0., 0.5, 0.], [0., 0., 0., 1.]]),
                    GridTopology((6, 6, 6)))
        (pieces, _, _) = voxel_intersections(_tet(), grid)
        self.assertAlmostEqual(float(sum(pieces.measures)), 1. / 6.)

    def test_it_checks_its_arguments(self):
        with self.assertRaises(TypeError):
            voxel_intersections('not-a-mesh', self._grid())
        with self.assertRaises(TypeError):
            voxel_intersections(_tet(), 'not-a-grid')

    def test_the_grid_must_be_three_dimensional(self):
        with self.assertRaises(ValueError):
            voxel_intersections(_tet(), Grid(eye(3), GridTopology((3, 3))))


class TestMeshIntersections(TestCase):
    '''The curve along which two triangle meshes meet.'''

    def _crossing_squares(self):
        # One square in the plane y = 0, one in the plane z = 0; they meet
        # along the line y = z = 0 from x = 0 to x = 1.
        first = TriMesh(array([[0., 1., 0., 1.], [0., 0., 0., 0.],
                               [-1., -1., 1., 1.]]),
                        TriTopology([[0, 0], [1, 3], [3, 2]]))
        second = TriMesh(array([[0., 0., 1., 1.], [0., 1., 0., 1.],
                                [0., 0., 0., 0.]]),
                         TriTopology([[0, 0], [1, 3], [3, 2]]))
        return (first, second)

    def test_two_crossing_squares(self):
        (first, second) = self._crossing_squares()
        cut = mesh_intersections(first, second)
        self.assertGreater(cut.topo.simplex_count[1], 0)
        # Whatever the triangulation does, the curve is one unit long.
        self.assertAlmostEqual(float(sum(cut.measures)), 1.)

    def test_every_segment_lies_on_the_meeting_line(self):
        (first, second) = self._crossing_squares()
        cut = mesh_intersections(first, second)
        coords = cut.coords
        # The line is y = z = 0, so every end of every segment is on it and
        # within the squares' extent along x.
        self.assertTrue(allclose(coords[1], 0., atol=1e-9))
        self.assertTrue(allclose(coords[2], 0., atol=1e-9))
        self.assertTrue((coords[0] >= -1e-9).all())
        self.assertTrue((coords[0] <= 1. + 1e-9).all())

    def test_a_pair_of_meshes_that_miss(self):
        (first, _) = self._crossing_squares()
        far = TriMesh(array([[0., 1., 0., 1.], [0., 0., 0., 0.],
                             [5., 5., 6., 6.]]),
                      TriTopology([[0, 0], [1, 3], [3, 2]]))
        cut = mesh_intersections(first, far)
        self.assertEqual(cut.topo.simplex_count[1], 0)
        self.assertEqual(cut.coord_count, 0)

    def test_it_checks_its_arguments(self):
        (first, second) = self._crossing_squares()
        with self.assertRaises(TypeError):
            mesh_intersections('not-a-mesh', second)
        with self.assertRaises(TypeError):
            mesh_intersections(first, 'not-a-mesh')

    def test_it_requires_three_dimensions(self):
        (first, _) = self._crossing_squares()
        flat = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                       TriTopology([[0], [1], [2]]))
        with self.assertRaises(ValueError):
            mesh_intersections(flat, first)

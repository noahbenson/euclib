# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_parity.py
'''Tests that the C kernels agree with the pure-Python ones.

The pure-Python kernels in ``euclib.utils._pycore`` are the definition of
correct behavior; the C kernels in ``euclib._c._core`` exist to be faster. These
tests hold the two to each other, in the two ways that matter:

*Exact agreement* is required on every case this file constructs: the box, the
lattice, the prism, the touching pair, the disjoint pair. Floating point does
not enter into it, because each corner comes out of a solve and no two of them
are near enough to one another for rounding to matter.

*Agreement within the merge radius* is what a random pair of shapes can be held
to, and this is a statement about the kernels rather than a weakening of the
test. Every corner of the region is found by solving three of ten planes at a
time, so the same corner is found several times over, each copy off from the
others by whatever the solve's rounding gave it. Collecting those copies into
one corner is a threshold test: copies closer than the merge radius are the same
corner. A copy that lands exactly at that radius is decided one way by one
implementation and the other way by the other, and no choice of radius removes
that --- the random tests below therefore require the two results to be the same
set of corners *up to the radius*, which is the strongest claim that is true.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

import numpy as np

import euclib

from euclib.utils import SpatialTree, split_cells, tetrahedron_box_vertices
from euclib.utils import _spatial
from euclib.utils._core import using_c_extension
from euclib.utils._pycore import (
    split_cells as split_cells_python,
    tetrahedron_box_vertices as vertices_python)


# Fixtures ###################################################################

#: The tetrahedron filling the unit cube: its corners and three of the cube's.
UNIT_TET = np.array([[0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])

#: A second tetrahedron that together with the first fills the unit cube.
OTHER_TET = np.array([[1., 1., 0., 1.],
                      [0., 1., 1., 1.],
                      [1., 1., 1., 0.]])

#: The unit cube.
UNIT_BOX = np.array([[0., 1.], [0., 1.], [0., 1.]])

#: The relative distance within which two corners count as one.
MERGE = 1e-9


def _merge_radius(vertices, /):
    '''The radius within which two corners count as one, as the kernels take it.

    The kernels are not told a tolerance, so they take one proportional to the
    extent of what they found.
    '''
    scale = max(float(np.abs(vertices).max()) if vertices.size else 0.0, 1.0)
    return MERGE * scale


def _same_corners(a, b, /):
    '''Determines whether two corner matrices hold the same set of corners.

    A corner of either result must have a corner of the other within the merge
    radius, and the two must hold the same number of them.
    '''
    if a.shape[1] != b.shape[1]:
        return False
    for column in range(a.shape[1]):
        distance = np.sqrt(((b - a[:, column][:, None]) ** 2).sum(axis=0))
        if distance.min() > max(_merge_radius(a), _merge_radius(b)):
            return False
    return True


# Tests ######################################################################

@skipUnless(using_c_extension, "the C extension is not built")
class TestSplitCellsParity(TestCase):
    '''The C bisection against the pure-Python one.'''

    def setUp(self):
        from euclib._c import _core
        self.core = _core

    def _check(self, centers, bounds):
        centers = np.ascontiguousarray(centers, dtype='float64')
        bounds = np.ascontiguousarray(bounds, dtype='float64')
        native = split_cells_python(centers, bounds)
        accelerated = self.core.split_cells(centers, bounds)
        self.assertEqual(native.tolist(), accelerated.tolist())
        self.assertEqual(native.dtype, accelerated.dtype)

    def test_a_known_pair(self):
        self._check(np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]), UNIT_BOX)

    def test_points_on_the_midplanes(self):
        # A point on a midplane is in the lower half: the comparison is strict.
        centers = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        self._check(centers, UNIT_BOX)

    def test_a_point_outside_the_box(self):
        centers = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
        self._check(centers, UNIT_BOX)

    def test_an_empty_set_of_points(self):
        self._check(np.zeros((3, 0)), UNIT_BOX)

    def test_two_dimensions(self):
        self._check(np.array([[0.2, 0.8], [0.2, 0.8]]),
                    np.array([[0., 1.], [0., 1.]]))

    def test_a_box_away_from_the_origin(self):
        bounds = np.array([[-4., -2.], [10., 12.], [0., 1.]])
        self._check(np.array([[-3., -1., 11., 0.5],
                              [10.5, 11.5, 10.5, 11.5],
                              [0.25, 0.75, 0.25, 0.75]]), bounds)

    def test_random_boxes(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            dimension = int(rng.integers(1, 5))
            count = int(rng.integers(0, 30))
            low = rng.normal(size=dimension)
            high = low + rng.uniform(0.1, 3.0, size=dimension)
            bounds = np.stack([low, high], axis=1)
            centers = rng.normal(size=(dimension, count)) * 2.0
            self._check(centers, bounds)

    def test_a_large_set_of_points(self):
        rng = np.random.default_rng(1)
        centers = rng.normal(size=(3, 20000))
        bounds = np.array([[-2., 2.], [-2., 2.], [-2., 2.]])
        self._check(centers, bounds)


@skipUnless(using_c_extension, "the C extension is not built")
class TestVerticesParityOnConstructedCases(TestCase):
    '''The C corner search against the pure-Python one, exactly.

    Every case here is built rather than drawn, and each is one that the library
    is expected to handle exactly: a mesh corner on a voxel face is what
    ``voxel_intersections`` meets whenever a mesh and a grid line up, which is
    the ordinary case rather than the exceptional one.
    '''

    def setUp(self):
        from euclib._c import _core
        self.core = _core

    def _check(self, tet, bounds, tolerance=0.0):
        tet = np.ascontiguousarray(tet, dtype='float64')
        bounds = np.ascontiguousarray(bounds, dtype='float64')
        native = vertices_python(tet, bounds, tolerance)
        accelerated = self.core.tetrahedron_box_vertices(tet, bounds,
                                                         tolerance)
        self.assertEqual(native.shape, accelerated.shape)
        self.assertTrue(np.array_equal(native, accelerated),
                        f"\nnative:\n{native}\naccelerated:\n{accelerated}")

    def test_a_tetrahedron_inside_the_unit_cube(self):
        self._check(UNIT_TET, UNIT_BOX)

    def test_the_two_tetrahedra_that_fill_the_cube(self):
        self._check(OTHER_TET, UNIT_BOX)

    def test_a_box_that_contains_the_tetrahedron(self):
        self._check(UNIT_TET, np.array([[-1., 2.], [-1., 2.], [-1., 2.]]))

    def test_a_box_that_misses_it(self):
        self._check(UNIT_TET, np.array([[4., 5.], [4., 5.], [4., 5.]]))

    def test_a_box_that_touches_one_face(self):
        # The corner of the tetrahedron at the origin is on the box's low
        # corner: the two meet at a single point.
        self._check(UNIT_TET, np.array([[-1., 0.], [-1., 0.], [-1., 0.]]))

    def test_a_box_that_shares_a_face(self):
        self._check(UNIT_TET, np.array([[0., 1.], [0., 1.], [0., 0.]]))

    def test_a_half_scale_tetrahedron(self):
        self._check(UNIT_TET / 2.0, UNIT_BOX)

    def test_a_translated_tetrahedron(self):
        self._check(UNIT_TET + np.array([[3.], [5.], [-2.]]),
                    np.array([[3., 4.], [5., 6.], [-2., -1.]]))

    def test_a_mirrored_tetrahedron(self):
        # A negative volume is a valid tetrahedron; the planes are oriented
        # from the corners, not from the sign of the determinant.
        self._check(UNIT_TET[:, [1, 0, 2, 3]], UNIT_BOX)

    def test_every_corner_subset_of_the_cube(self):
        # The four corners of a lattice tetrahedron can be any four of the
        # cube's eight; a box that lines up with the cube meets all of them the
        # same way.
        cube = np.array(np.meshgrid(*[np.arange(2)] * 3)).reshape(3, 8) * 1.0
        from itertools import combinations
        for corners in combinations(range(8), 4):
            tet = cube[:, corners]
            edges = np.stack([tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0],
                              tet[:, 3] - tet[:, 0]], axis=1)
            if abs(np.linalg.det(edges)) < 1e-12:
                continue
            with self.subTest(corners=corners):
                self._check(tet, UNIT_BOX)
                self._check(tet, np.array([[-0.5, 1.5], [-0.5, 1.5],
                                           [-0.5, 1.5]]))

    def test_lattice_tetrahedra_against_lattice_boxes(self):
        rng = np.random.default_rng(7)
        cube = np.array(np.meshgrid(*[np.arange(2)] * 3)).reshape(3, 8) * 1.0
        for _ in range(200):
            corners = rng.permutation(8)[:4]
            tet = cube[:, corners]
            edges = np.stack([tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0],
                              tet[:, 3] - tet[:, 0]], axis=1)
            if abs(np.linalg.det(edges)) < 1e-12:
                continue
            low = rng.integers(-1, 2, size=3).astype('float64')
            high = low + rng.integers(1, 3, size=3)
            self._check(tet, np.stack([low, high], axis=1))

    def test_a_tolerance_is_passed_through(self):
        bounds = np.array([[0., 1.], [0., 1.], [2., 3.]])
        self._check(UNIT_TET, bounds, 0.5)


@skipUnless(using_c_extension, "the C extension is not built")
class TestVerticesParityOnRandomCases(TestCase):
    '''The C corner search against the pure-Python one, on drawn shapes.'''

    def setUp(self):
        from euclib._c import _core
        self.core = _core

    def test_random_pairs_agree_on_the_corners(self):
        rng = np.random.default_rng(4)
        overlapping = 0
        corners = 0
        for _ in range(500):
            tet = rng.normal(size=(3, 4)) * rng.uniform(0.3, 3.0)
            center = rng.normal(size=3) * rng.uniform(0.2, 3.0)
            half = rng.uniform(0.1, 1.5, size=3)
            bounds = np.stack([center - half, center + half], axis=1)
            native = vertices_python(tet, bounds)
            accelerated = self.core.tetrahedron_box_vertices(tet, bounds)
            if native.size:
                overlapping += 1
                corners += native.shape[1]
            self.assertTrue(_same_corners(native, accelerated),
                            f"\nnative:\n{native}\naccelerated:\n{accelerated}")
        # The draw has to actually produce overlaps, or the test says nothing.
        self.assertGreater(overlapping, 100)
        self.assertGreater(corners, 500)

    def test_an_operation_built_on_the_kernel_agrees(self):
        # The end of the chain: a voxelized mesh. The operation cuts a
        # tetrahedron against a voxel through the corner search, so pointing the
        # dispatcher at each kernel in turn is what compares the two routes
        # through the whole operation rather than through the kernel alone.
        from euclib import grid, ops, tetmesh
        from euclib.utils import _pycore
        # A three-cube block, each cube cut into six tetrahedra.
        corners = []
        tets = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    base = len(corners)
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                corners.append((i + x, j + y, k + z))
                    high = base + 7
                    for (a, b) in ((0, 1), (0, 2), (1, 0),
                                   (1, 2), (2, 0), (2, 1)):
                        tets.append([base, base + (1 << a),
                                     base + (1 << a) + (1 << b), high])
        mesh = tetmesh(np.array(corners, dtype=float).T,
                       np.array(tets, dtype=int).T)
        cells = grid((6, 6, 6), affine=np.array([[0.5, 0., 0., 0.],
                                                 [0., 0.5, 0., 0.],
                                                 [0., 0., 0.5, 0.],
                                                 [0., 0., 0., 1.]]))
        dispatching = _pycore._vertices_kernel
        answers = {}
        try:
            for (label, kernel) in (('python', dispatching.native),
                                    ('C', dispatching.accelerated)):
                _pycore._vertices_kernel = kernel
                (pieces, from_tet, from_voxel) = ops.voxel_intersections(
                    mesh, cells)
                answers[label] = (pieces.topo.simplex_count[3],
                                  from_tet.tolist(), from_voxel.tolist())
        finally:
            _pycore._vertices_kernel = dispatching
        self.assertGreater(answers['python'][0], 100)
        self.assertEqual(answers['python'], answers['C'])

    def test_the_decomposition_agrees_too(self):
        # The intersection the operations call goes through the same corner
        # search, so it is held to the same standard. The decomposition adds a
        # corner *inside* the region when the region has to be cut up --- it is
        # what the faces of the filling are joined to --- so the corners of the
        # region are the ones that lie on it, and the extra one is not.
        from euclib.utils import _pycore
        rng = np.random.default_rng(12)
        checked = 0
        for _ in range(100):
            tet = rng.normal(size=(3, 4)) * rng.uniform(0.5, 2.0)
            center = rng.normal(size=3) * 1.5
            half = rng.uniform(0.3, 1.5, size=3)
            bounds = np.stack([center - half, center + half], axis=1)
            (native, native_tets) = _pycore.tetrahedron_box_intersection(
                tet, bounds)
            accelerated = self.core.tetrahedron_box_vertices(tet, bounds)
            if not native.size:
                continue
            checked += 1
            from euclib.utils._pycore import _half_spaces, _dot
            (normals, offsets) = _half_spaces(np.asarray(tet), np.asarray(bounds))
            slack = 1e-9 * max(1.0, float(np.abs(native).max()))
            on_boundary = []
            inside_only = []
            for column in range(native.shape[1]):
                point = native[:, column].tolist()
                if any(abs(_dot(tuple(float(v) for v in normals[:, c]),
                                point) - float(offsets[c])) <= slack
                       for c in range(normals.shape[1])):
                    on_boundary.append(native[:, column])
                else:
                    inside_only.append(native[:, column])
            boundaries = (np.stack(on_boundary, axis=1) if on_boundary
                          else np.zeros((3, 0)))
            self.assertTrue(_same_corners(boundaries, accelerated))
            # Whatever is not a corner of the region is inside it: it is what
            # the faces were joined to, and it lies on none of the planes.
            self.assertEqual(len(inside_only), 0 if native.shape[1] < 5 else 1)
        self.assertGreater(checked, 20)


@skipUnless(using_c_extension, "the C extension is not built")
class TestSpatialQueryParity(TestCase):
    '''The compiled index queries against the tree's own methods.

    The tree in ``euclib.utils._spatial`` is the definition: it answers one
    position at a time, in Python, and the compiled queries answer an array of
    them. The two must agree exactly --- not merely closely --- because the
    search that uses them resolves the exact answer from the candidates they
    return, and a different candidate set would give a different simplex.

    The gate matters as much as the kernels: every query goes to the compiled
    queries when they are loaded, at every size, so these tests check a single
    position as well as an array of them --- and check that the tree's flattened
    arrays are contiguous, which is what makes a single position fast.
    '''

    def _both(self, centers, radii, query, k=1, radius=0.5):
        '''Returns the answers of the compiled path and of the tree's own.'''
        tree = SpatialTree(centers, radii)
        try:
            compiled = (tree.nearest(query, k=k),
                        tree.candidates(query, radius))
            _spatial.c_spatial = None
            native = (tree.nearest(query, k=k),
                      tree.candidates(query, radius))
        finally:
            # Restoring the module's own setting matters: a test that leaves it
            # turned off would send every later test down the native path, and
            # they would pass without testing the compiled queries at all.
            _spatial.c_spatial = self._compiled
        return (compiled, native)

    def setUp(self):
        self._compiled = _spatial.c_spatial
        self.assertTrue(self._compiled is not None)

    def test_random_trees_agree_exactly(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            dim = int(rng.integers(1, 4))
            count = int(rng.integers(0, 300))
            centers = rng.normal(size=(dim, count)) * 2.0
            radii = rng.uniform(0.0, 0.5, size=count)
            query = rng.normal(size=(dim, 12)) * 3.0
            with self.subTest(dim=dim, count=count):
                ((ci, cd), cans), ((pi, pd), pans) = self._both(
                    centers, radii, query, k=int(rng.integers(1, 4)))
                self.assertEqual(ci.shape, pi.shape)
                self.assertTrue(np.array_equal(ci, pi))
                self.assertTrue(np.allclose(cd, pd, rtol=0, atol=1e-12))
                self.assertEqual(len(cans), len(pans))
                for (mine, theirs) in zip(cans, pans):
                    self.assertTrue(np.array_equal(mine, theirs))

    def test_an_empty_tree_agrees(self):
        # Nothing to find: the answer has no rows at all, rather than k rows of
        # a placeholder, which is what the tree itself returns.
        ((ci, cd), cans), ((pi, pd), pans) = self._both(
            np.zeros((2, 0)), np.zeros(0), np.array([[0., 1.], [0., 1.]]))
        self.assertEqual(ci.shape, pi.shape)
        self.assertEqual(ci.shape, (0, 2))
        self.assertEqual(cans[0].tolist(), pans[0].tolist())
        self.assertEqual(cans[1].tolist(), pans[1].tolist())

    def test_a_tree_with_fewer_items_than_k_agrees(self):
        centers = np.array([[0., 1.], [0., 0.]])
        radii = np.array([0.1, 0.1])
        ((ci, _), _), ((pi, _), _) = self._both(
            centers, radii, np.array([[0.4, 5.0], [0.0, 5.0]]), k=5)
        self.assertEqual(ci.shape, (2, 2))
        self.assertTrue(np.array_equal(ci, pi))

    def test_a_real_index_agrees(self):
        # The index the library builds for a mesh, rather than one of points.
        # It is built only for a mesh with enough simplices to pay for it, so
        # this mesh is large enough for that threshold.
        side = 40
        axis = np.linspace(0.0, 1.0, side)
        (gridx, gridy) = np.meshgrid(axis, axis)
        coords = np.stack([gridx.ravel(), gridy.ravel()])
        faces = []
        for (i, j) in np.ndindex(side - 1, side - 1):
            (a, b) = (i * side + j, i * side + j + 1)
            (c, d) = ((i + 1) * side + j, (i + 1) * side + j + 1)
            faces += [[a, b, d], [a, d, c]]
        mesh = euclib.trimesh(coords, np.array(faces).T)
        tree = mesh.spatial_index
        self.assertTrue(tree is not None)
        query = np.array([[0.1, 0.4, 0.9], [0.1, 0.6, 0.2]])
        ((ci, cd), cans), ((pi, pd), pans) = self._both(
            tree.centers, tree.radii, query, k=2, radius=0.3)
        self.assertTrue(np.array_equal(ci, pi))
        for (mine, theirs) in zip(cans, pans):
            self.assertTrue(np.array_equal(mine, theirs))

    def test_a_single_position_goes_to_the_kernels(self):
        # One position is not a special case: the compiled queries take it, and
        # they are faster at that size too. What made them look slower was a
        # strided array being copied on every call, so this checks both the
        # answer and the arrays it is read from.
        tree = SpatialTree(np.array([[0., 1.], [0., 0.]]), np.array([0.1, 0.1]))
        (index, distance) = tree.nearest(np.array([[0.4], [0.]]), k=1)
        self.assertEqual(index.tolist(), [[0]])

    def test_the_flattened_arrays_are_contiguous(self):
        # A mesh's centers are usually a strided view, and the compiled queries
        # copy anything that is not contiguous --- six megabytes of centers, on
        # every call, if the flattening does not settle it once. This is the
        # regression test for that: a strided (D, M) center matrix, one query,
        # and the timing such a mistake would destroy.
        block = np.zeros((3, 40000, 2))
        block[:, :, 0] = np.arange(40000) % 200
        block[:, :, 1] = np.arange(40000) // 200
        centers = block[:, :, 0]
        radii = np.full(40000, 0.5)
        self.assertFalse(centers.flags.c_contiguous)
        tree = SpatialTree(centers, radii)
        for array in tree._flattened():
            self.assertTrue(array.flags.c_contiguous)

# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_remesh.py
'''Tests for subdivision and path simplification in ``_remesh``.

Subdivision is checked for the properties a refinement must have: four
triangles replace every one, the area and volume do not change, the surface
stays closed, and the maximum edge length is what was asked for. Simplification
is checked for the properties a simplification must have: a straight run
becomes one segment, a bend stays, a junction is not touched, and a tolerance
decides how straight is straight.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, sqrt
import unittest
from unittest import TestCase

from immlib import math as imath

from euclib.ops import merge_colinear, subdivide
from euclib.types import (SegPath, SegTopology, TriMesh, TriTopology, segpath,
                          trimesh)


# Fixtures ###################################################################

_CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
_CUBE_TRIS = array([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                    [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                    [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                   dtype='int64').T


def _cube(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0), /):
    '''An axis-aligned cube wound outward.'''
    corners = array([[low[k] + bit[k] * (high[k] - low[k])
                      for k in range(3)]
                     for bit in _CUBE_CORNERS], dtype=float).T
    return trimesh(corners, _CUBE_TRIS)


def _volume(mesh, /):
    '''The signed volume a closed, outward-wound triangle mesh encloses.'''
    coords = asarray(imath.mag(mesh.coords))
    tris = asarray(mesh.topo.indices)
    (a, b, c) = (coords[:, tris[0]].T, coords[:, tris[1]].T,
                 coords[:, tris[2]].T)
    return float((a * np.cross(b, c)).sum()) / 6.0


def _closed(mesh, /):
    '''Whether every directed edge of the surface is matched by its reverse.'''
    tris = asarray(mesh.topo.indices)
    seen = {}
    for k in range(tris.shape[1]):
        (a, b, c) = (int(v) for v in tris[:, k])
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(count == 1 and (edge[1], edge[0]) in seen
               for (edge, count) in seen.items())


def _longest(mesh, /):
    '''The longest edge of a triangle mesh.'''
    coords = asarray(imath.mag(mesh.coords))
    tris = asarray(mesh.topo.indices)
    longest = 0.0
    for k in range(tris.shape[1]):
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            edge = coords[:, tris[i, k]] - coords[:, tris[j, k]]
            longest = max(longest, float(sqrt(edge @ edge)))
    return longest


# Tests ######################################################################

class TestSubdivide(TestCase):
    '''Refining a triangle mesh without changing its surface.'''

    def test_every_triangle_becomes_four(self):
        fine = subdivide(_cube())
        self.assertEqual(fine.topo.simplex_count[2], 48)
        self.assertEqual(fine.coord_count, 26)
        self.assertTrue(_closed(fine))

    def test_the_area_and_volume_do_not_change(self):
        cube = _cube()
        fine = subdivide(cube, iterations=3)
        self.assertAlmostEqual(float(np.sum(fine.measures)),
                               float(np.sum(cube.measures)), places=12)
        self.assertAlmostEqual(_volume(fine), 1.0, places=12)

    def test_the_triangle_count_grows_by_four_each_time(self):
        cube = _cube()
        for iterations in (0, 1, 2, 3):
            with self.subTest(iterations=iterations):
                fine = subdivide(cube, iterations=iterations)
                self.assertEqual(fine.topo.simplex_count[2],
                                 12 * 4 ** iterations)

    def test_zero_iterations_returns_the_mesh(self):
        cube = _cube()
        self.assertIs(subdivide(cube, iterations=0), cube)

    def test_a_maximum_edge_is_respected(self):
        fine = subdivide(_cube(), max_edge=0.4)
        self.assertLessEqual(_longest(fine), 0.4 + 1e-12)
        self.assertAlmostEqual(float(np.sum(fine.measures)), 6.0, places=12)
        self.assertAlmostEqual(_volume(fine), 1.0, places=12)
        self.assertTrue(_closed(fine))

    def test_a_maximum_edge_that_is_already_met_changes_nothing(self):
        cube = _cube()
        self.assertIs(subdivide(cube, iterations=0, max_edge=2.0), cube)

    def test_iterations_and_a_maximum_edge_combine(self):
        fine = subdivide(_cube(), iterations=1, max_edge=0.4)
        self.assertEqual(fine.topo.simplex_count[2], 48 * 4)
        self.assertLessEqual(_longest(fine), 0.4 + 1e-12)

    def test_the_bounds_do_not_change(self):
        cube = _cube()
        fine = subdivide(cube)
        self.assertTrue(allclose(asarray(imath.mag(fine.bbox)),
                                 asarray(imath.mag(cube.bbox))))

    def test_a_two_dimensional_triangle_works(self):
        triangle = trimesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                           array([[0], [1], [2]]))
        fine = subdivide(triangle)
        self.assertEqual(fine.topo.simplex_count[2], 4)
        self.assertAlmostEqual(float(np.sum(fine.measures)), 0.5, places=12)

    def test_an_empty_mesh_stays_empty(self):
        empty = TriMesh(np.zeros((3, 0)),
                        TriTopology(np.zeros((3, 0), dtype=int),
                                    coord_count=0))
        self.assertIs(subdivide(empty), empty)

    def test_the_input_is_not_changed(self):
        cube = _cube()
        before = asarray(cube.coords).copy()
        subdivide(cube)
        self.assertTrue(np.array_equal(asarray(cube.coords), before))

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        cube = _cube()
        metric = TriMesh(asarray(cube.coords) * ureg.meter, cube.topo)
        fine = subdivide(metric)
        self.assertEqual(str(fine.coords.units), 'meter')
        self.assertAlmostEqual(_volume(fine), 1.0, places=12)

    def test_a_tensor_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        cube = _cube()
        tensor = TriMesh(torch.tensor(asarray(cube.coords)), cube.topo)
        fine = subdivide(tensor)
        self.assertIsInstance(fine.coords, np.ndarray)
        self.assertAlmostEqual(_volume(fine), 1.0, places=12)

    def test_a_non_mesh_is_refused(self):
        with self.assertRaises(TypeError):
            subdivide(42)

    def test_a_bad_iteration_count_is_refused(self):
        for iterations in (-1, 1.5):
            with self.subTest(iterations=iterations):
                with self.assertRaises(ValueError):
                    subdivide(_cube(), iterations=iterations)

    def test_a_bad_maximum_edge_is_refused(self):
        for max_edge in (0.0, -1.0):
            with self.subTest(max_edge=max_edge):
                with self.assertRaises(ValueError):
                    subdivide(_cube(), max_edge=max_edge)


class TestMergeColinear(TestCase):
    '''Removing the corners that lie on a straight run of a path.'''

    def test_a_straight_line_becomes_one_segment(self):
        line = segpath(array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]))
        simple = merge_colinear(line)
        self.assertEqual(simple.coord_count, 2)
        self.assertEqual(simple.topo.simplex_count[1], 1)
        self.assertAlmostEqual(float(np.sum(simple.measures)), 3.0,
                               places=12)

    def test_a_bend_is_kept(self):
        bend = segpath(array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]))
        self.assertIs(merge_colinear(bend), bend)

    def test_a_reversal_is_kept(self):
        spike = segpath(array([[0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]))
        self.assertIs(merge_colinear(spike), spike)

    def test_a_tolerance_decides_how_straight_is_straight(self):
        almost = segpath(array([[0.0, 1.0, 2.0], [0.0, 0.001, 0.0]]))
        self.assertEqual(
            merge_colinear(almost, tolerance=0.01).topo.simplex_count[1], 1)
        self.assertIs(merge_colinear(almost, tolerance=1e-6), almost)

    def test_a_closed_loop_is_simplified(self):
        square = array([[0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5]])
        loop = SegPath(square, SegTopology(
            array([[0, 1, 2, 3, 4, 5, 6, 7],
                   [1, 2, 3, 4, 5, 6, 7, 0]]),
            coord_count=8))
        simple = merge_colinear(loop)
        self.assertEqual(simple.topo.simplex_count[1], 4)
        self.assertEqual(simple.coord_count, 4)
        self.assertAlmostEqual(float(np.sum(simple.measures)), 4.0,
                               places=12)

    def test_a_junction_is_not_touched(self):
        star = SegPath(array([[0.0, 1.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]]),
                       SegTopology(array([[0, 0, 0], [1, 2, 3]]),
                                   coord_count=4))
        self.assertIs(merge_colinear(star), star)

    def test_separate_components_are_each_simplified(self):
        both = SegPath(array([[0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                       SegTopology(array([[0, 1, 3, 4], [1, 2, 4, 5]]),
                                   coord_count=6))
        simple = merge_colinear(both)
        self.assertEqual(simple.topo.simplex_count[1], 2)
        self.assertEqual(simple.coord_count, 4)

    def test_a_three_dimensional_line_is_simplified(self):
        line = segpath(array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0]]))
        self.assertEqual(merge_colinear(line).topo.simplex_count[1], 1)

    def test_the_path_length_does_not_change(self):
        line = segpath(array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.sum(merge_colinear(line).measures)),
                               float(np.sum(line.measures)), places=12)

    def test_an_empty_path_returns_itself(self):
        empty = SegPath(np.zeros((2, 0)),
                        SegTopology(np.zeros((2, 0), dtype=int),
                                    coord_count=0))
        self.assertIs(merge_colinear(empty), empty)

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        line = segpath(array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
                       * ureg.meter)
        simple = merge_colinear(line)
        self.assertEqual(str(simple.coords.units), 'meter')
        self.assertAlmostEqual(float(imath.mag(np.sum(simple.measures))), 2.0,
                               places=12)

    def test_a_tensor_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        line = segpath(array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))
        tensor = SegPath(torch.tensor(asarray(line.coords)), line.topo)
        simple = merge_colinear(tensor)
        self.assertIsInstance(simple.coords, np.ndarray)
        self.assertEqual(simple.topo.simplex_count[1], 1)

    def test_a_non_path_is_refused(self):
        with self.assertRaises(TypeError):
            merge_colinear(42)

    def test_the_input_is_not_changed(self):
        line = segpath(array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))
        before = asarray(line.coords).copy()
        merge_colinear(line)
        self.assertTrue(np.array_equal(asarray(line.coords), before))


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()

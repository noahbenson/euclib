# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_clip.py
'''Tests for clipping and sectioning in ``_clip``.

A clip is checked for the properties a solid must have: the kept positions are
on the kept side of the plane, the surface closes, its winding is outward, and
its volume is the part of the input's volume that the plane left. Cutting a
solid in two with a plane and clipping each half is the sharpest check of the
last one: the two volumes must add back to the whole.

A section is checked against curves whose lengths are known --- a plane through
a cube's middle is a square, an equator is a polygon whose perimeter follows
from the longitude count --- and for the property that every segment's ends lie
in the plane. Sections of open surfaces are checked too, because a section,
unlike a clip, does not need the surface to enclose anything.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, cos, linspace, pi, sin, sqrt
import unittest
from unittest import TestCase

from immlib import math as imath

from euclib.ops import clip, section
from euclib.types import TriMesh, TriTopology, trimesh


# Fixtures ###################################################################

_CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
_CUBE_TRIS = array([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                    [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                    [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                   dtype='int64').T

#: The cube page's triangles, four of which wind the other way from the rest.
_DOCS_TRIS = array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                    [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                    [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                   dtype='int64').T


def _cube(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0), /, tris=_CUBE_TRIS):
    '''An axis-aligned cube with a given corner winding.'''
    corners = array([[low[k] + bit[k] * (high[k] - low[k])
                      for k in range(3)]
                     for bit in _CUBE_CORNERS], dtype=float).T
    return trimesh(corners, tris)


def _sphere(n_lat=8, n_lon=12, /):
    '''A latitude-longitude sphere, with each pole's copies listed.'''
    lat = linspace(0.0, pi, n_lat + 1)
    lon = linspace(0.0, 2.0 * pi, n_lon, endpoint=False)
    coords = array([[sin(t) * cos(p), sin(t) * sin(p), cos(t)]
                    for t in lat for p in lon], dtype=float).T
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            (a, b) = (i * n_lon + j, i * n_lon + (j + 1) % n_lon)
            (c, d) = ((i + 1) * n_lon + j,
                      (i + 1) * n_lon + (j + 1) % n_lon)
            faces += [[a, b, d], [a, d, c]]
    return trimesh(coords, array(faces, dtype='int64').T)


def _volume(mesh, /):
    '''The signed volume a closed, outward-wound triangle mesh encloses.'''
    coords = asarray(imath.mag(mesh.coords))
    tris = asarray(mesh.topo.indices)
    if tris.shape[1] == 0:
        return 0.0
    (a, b, c) = (coords[:, tris[0]].T, coords[:, tris[1]].T,
                 coords[:, tris[2]].T)
    return float((a * np.cross(b, c)).sum()) / 6.0


def _area(mesh, /):
    '''The surface area of a triangle mesh.'''
    coords = asarray(imath.mag(mesh.coords))
    tris = asarray(mesh.topo.indices)
    if tris.shape[1] == 0:
        return 0.0
    (a, b, c) = (coords[:, tris[0]].T, coords[:, tris[1]].T,
                 coords[:, tris[2]].T)
    return float(sqrt((np.cross(b - a, c - a) ** 2).sum(axis=1)).sum()) / 2.0


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


def _euler(mesh, /):
    '''The Euler characteristic ``V - E + F`` of a triangle surface.'''
    tris = asarray(mesh.topo.indices)
    edges = set()
    for k in range(tris.shape[1]):
        (a, b, c) = (int(v) for v in tris[:, k])
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return mesh.coord_count - len(edges) + tris.shape[1]


def _bounds(mesh, /):
    '''A mesh's bounding box as a plain array.'''
    box = mesh.bbox
    return asarray(getattr(box, 'magnitude', box))


# Tests ######################################################################

class TestTheClip(TestCase):
    '''The part of a solid on one side of a plane, capped.'''

    def test_a_cube_cut_in_the_middle_keeps_half_its_volume(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(_volume(top), 0.5, places=12)
        self.assertTrue(allclose(_bounds(top), [[0.0, 1.0], [0.0, 1.0],
                                                [0.5, 1.0]]))

    def test_the_kept_side_is_the_one_the_normal_points_to(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        bottom = clip(_cube(), array([0.0, 0.0, 0.5]),
                      array([0.0, 0.0, -1.0]))
        self.assertTrue((asarray(top.coords)[2] >= 0.5 - 1e-12).all())
        self.assertTrue((asarray(bottom.coords)[2] <= 0.5 + 1e-12).all())

    def test_the_cap_closes_the_surface(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertTrue(_closed(top))
        self.assertEqual(_euler(top), 2)

    def test_cutting_a_solid_in_two_conserves_its_volume(self):
        plane = (array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        top = clip(_cube(), *plane)
        bottom = clip(_cube(), plane[0], -plane[1])
        self.assertAlmostEqual(_volume(top) + _volume(bottom), 1.0, places=12)

    def test_a_tilted_plane_cuts_a_cube_in_half(self):
        top = clip(_cube(), array([0.5, 0.5, 0.5]), array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(_volume(top), 0.5, places=12)
        self.assertTrue(_closed(top))
        self.assertEqual(_euler(top), 2)

    def test_a_plane_through_vertices_still_makes_a_solid(self):
        # x + y = 1 runs through the two vertical edges of the unit cube, so
        # the cut meets the surface at corners rather than in the middle of
        # triangles.
        half = clip(_cube(), array([0.5, 0.5, 0.0]), array([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(_volume(half), 0.5, places=12)
        self.assertTrue(_closed(half))
        self.assertEqual(_euler(half), 2)

    def test_a_sphere_cut_at_its_equator_keeps_half_its_volume(self):
        ball = _sphere()
        half = clip(ball, array([0.0, 0.0, 0.0]), array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(_volume(half), abs(_volume(ball)) / 2.0,
                               places=12)
        self.assertTrue(_closed(half))
        # Half the sphere's area, plus the twelve-sided cap.
        cap = 12.0 / 2.0 * sin(2.0 * pi / 12.0)
        self.assertAlmostEqual(_area(half), _area(ball) / 2.0 + cap, places=10)

    def test_a_plane_that_misses_returns_the_mesh_itself(self):
        cube = _cube()
        self.assertIs(clip(cube, array([0.0, 0.0, -1.0]),
                           array([0.0, 0.0, 1.0])), cube)

    def test_a_plane_that_leaves_nothing_returns_an_empty_mesh(self):
        empty = clip(_cube(), array([0.0, 0.0, 2.0]), array([0.0, 0.0, 1.0]))
        self.assertEqual(empty.coord_count, 0)
        self.assertEqual(empty.topo.simplex_count[2], 0)

    def test_an_empty_mesh_stays_empty(self):
        empty = TriMesh(np.zeros((3, 0)),
                        TriTopology(np.zeros((3, 0), dtype=int),
                                    coord_count=0))
        result = clip(empty, np.zeros(3), array([0.0, 0.0, 1.0]))
        self.assertEqual(result.coord_count, 0)

    def test_a_mesh_wound_inconsistently_is_repaired(self):
        # The cube page's triangles are not all wound the same way, so the
        # signed volume of the input is not the cube's. The clip repairs them,
        # as the boolean operations do.
        mixed = _cube(tris=_DOCS_TRIS)
        top = clip(mixed, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(_volume(top), 0.5, places=12)

    def test_a_mesh_with_repeated_positions_is_accepted(self):
        # The sphere lists each pole once per longitude.
        half = clip(_sphere(), array([0.0, 0.0, 0.0]), array([0.0, 0.0, 1.0]))
        self.assertTrue(_closed(half))

    def test_the_tolerance_can_be_given(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]),
                   tolerance=1e-12)
        self.assertAlmostEqual(_volume(top), 0.5, places=12)

    def test_the_input_is_not_changed(self):
        cube = _cube()
        before = asarray(cube.coords).copy()
        clip(cube, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertTrue(np.array_equal(asarray(cube.coords), before))

    def test_a_result_is_a_triangle_mesh(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertIsInstance(top, TriMesh)
        self.assertEqual(top.dim, 3)

    def test_a_result_can_be_measured(self):
        top = clip(_cube(), array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(imath.mag(np.sum(top.measures))), 4.0,
                               places=12)


class TestTheClipInputs(TestCase):
    '''What a clip refuses, and what it accepts.'''

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        cube = _cube()
        with_units = TriMesh(asarray(cube.coords) * ureg.meter, cube.topo)
        top = clip(with_units, array([0.0, 0.0, 0.5]) * ureg.meter,
                   array([0.0, 0.0, 1.0]))
        self.assertEqual(str(top.coords.units), 'meter')
        self.assertAlmostEqual(_volume(top), 0.5, places=12)

    def test_a_tensor_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        cube = _cube()
        tensor = TriMesh(torch.tensor(asarray(cube.coords)), cube.topo)
        top = clip(tensor, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertIsInstance(top.coords, np.ndarray)
        self.assertAlmostEqual(_volume(top), 0.5, places=12)

    def test_an_open_surface_is_refused(self):
        triangle = trimesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                  [0.0, 0.0, 1.0]]),
                           array([[0], [1], [2]]))
        with self.assertRaises(ValueError):
            clip(triangle, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))

    def test_a_non_mesh_is_refused(self):
        with self.assertRaises(TypeError):
            clip(42, np.zeros(3), array([0.0, 0.0, 1.0]))

    def test_a_plane_in_a_plane_is_refused(self):
        flat = TriMesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                       TriTopology(array([[0], [1], [2]])))
        with self.assertRaises(ValueError):
            clip(flat, np.zeros(3), array([0.0, 0.0, 1.0]))

    def test_a_bad_normal_is_refused(self):
        for normal in (np.zeros(3), np.zeros(2)):
            with self.subTest(normal=normal):
                with self.assertRaises(ValueError):
                    clip(_cube(), np.zeros(3), normal)

    def test_a_bad_origin_is_refused(self):
        with self.assertRaises(ValueError):
            clip(_cube(), np.zeros(2), array([0.0, 0.0, 1.0]))


class TestTheSection(TestCase):
    '''The segments where a plane cuts a surface.'''

    def test_a_cube_cut_in_the_middle_is_a_square(self):
        path = section(_cube(), array([0.0, 0.0, 0.5]),
                       array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(np.sum(path.measures)), 4.0, places=12)
        self.assertTrue(allclose(asarray(path.coords)[2], 0.5))

    def test_the_segment_ends_are_welded(self):
        path = section(_cube(), array([0.0, 0.0, 0.5]),
                       array([0.0, 0.0, 1.0]))
        indices = asarray(path.topo.indices)
        self.assertEqual(path.coord_count, 8)
        self.assertEqual(path.topo.simplex_count[1], 8)
        self.assertEqual(set(indices.reshape(-1)), set(range(8)))

    def test_a_tilted_plane_through_a_cube_is_a_hexagon(self):
        path = section(_cube(), array([0.5, 0.5, 0.5]),
                       array([1.0, 1.0, 1.0]))
        # The section is a regular hexagon with side sqrt(2)/2.
        self.assertAlmostEqual(float(np.sum(path.measures)), 3.0 * sqrt(2.0),
                               places=12)

    def test_a_diagonal_plane_through_a_cube_is_a_rectangle(self):
        path = section(_cube(), array([0.5, 0.5, 0.0]), array([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(float(np.sum(path.measures)),
                               2.0 * (1.0 + sqrt(2.0)), places=12)

    def test_a_sphere_cut_at_its_equator_is_a_polygon(self):
        path = section(_sphere(), array([0.0, 0.0, 0.0]),
                       array([0.0, 0.0, 1.0]))
        # Twelve longitudes, so twelve chords of a unit circle.
        expected = 12.0 * 2.0 * sin(pi / 12.0)
        self.assertAlmostEqual(float(np.sum(path.measures)), expected,
                               places=12)
        self.assertEqual(path.coord_count, 12)
        self.assertTrue(allclose(asarray(path.coords)[2], 0.0))

    def test_a_plane_that_misses_returns_an_empty_path(self):
        path = section(_sphere(), array([0.0, 0.0, 5.0]),
                       array([0.0, 0.0, 1.0]))
        self.assertEqual(path.coord_count, 0)
        self.assertEqual(path.topo.simplex_count[1], 0)

    def test_a_plane_along_a_face_finds_the_face_s_boundary(self):
        # The bottom face's triangles lie in the plane and contribute nothing,
        # but the four sides meet the plane along the bottom square's edges.
        path = section(_cube(), array([0.0, 0.0, 0.0]), array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(np.sum(path.measures)), 4.0, places=12)
        self.assertEqual(path.topo.simplex_count[1], 4)

    def test_a_surface_in_the_plane_has_no_section(self):
        sheet = trimesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0]]),
                        array([[0], [1], [2]]))
        path = section(sheet, np.zeros(3), array([0.0, 0.0, 1.0]))
        self.assertEqual(path.coord_count, 0)

    def test_an_open_surface_can_be_sectioned(self):
        triangle = trimesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                  [0.0, 0.0, 1.0]]),
                           array([[0], [1], [2]]))
        path = section(triangle, array([0.0, 0.0, 0.5]),
                       array([0.0, 0.0, 1.0]))
        self.assertEqual(path.topo.simplex_count[1], 1)
        self.assertAlmostEqual(float(np.sum(path.measures)), 0.5, places=12)

    def test_every_segment_lies_in_the_plane(self):
        origin = array([0.25, -0.5, 0.75])
        normal = array([1.0, 2.0, -3.0])
        normal = normal / sqrt(normal @ normal)
        path = section(_sphere(), origin, normal)
        depths = normal @ (asarray(path.coords) - origin[:, None])
        self.assertTrue(np.abs(depths).max() < 1e-12)

    def test_the_tolerance_can_be_given(self):
        path = section(_cube(), array([0.0, 0.0, 0.5]),
                       array([0.0, 0.0, 1.0]), tolerance=1e-12)
        self.assertAlmostEqual(float(np.sum(path.measures)), 4.0, places=12)

    def test_the_input_is_not_changed(self):
        cube = _cube()
        before = asarray(cube.coords).copy()
        section(cube, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertTrue(np.array_equal(asarray(cube.coords), before))

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        cube = _cube()
        with_units = TriMesh(asarray(cube.coords) * ureg.meter, cube.topo)
        path = section(with_units, array([0.0, 0.0, 0.5]) * ureg.meter,
                       array([0.0, 0.0, 1.0]))
        self.assertEqual(str(path.coords.units), 'meter')
        self.assertAlmostEqual(float(imath.mag(np.sum(path.measures))), 4.0,
                               places=12)

    def test_a_tensor_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        cube = _cube()
        tensor = TriMesh(torch.tensor(asarray(cube.coords)), cube.topo)
        path = section(tensor, array([0.0, 0.0, 0.5]), array([0.0, 0.0, 1.0]))
        self.assertIsInstance(path.coords, np.ndarray)
        self.assertAlmostEqual(float(np.sum(path.measures)), 4.0, places=12)

    def test_a_non_mesh_is_refused(self):
        with self.assertRaises(TypeError):
            section(42, np.zeros(3), array([0.0, 0.0, 1.0]))

    def test_a_plane_in_a_plane_is_refused(self):
        flat = TriMesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                       TriTopology(array([[0], [1], [2]])))
        with self.assertRaises(ValueError):
            section(flat, np.zeros(3), array([0.0, 0.0, 1.0]))

    def test_a_bad_normal_is_refused(self):
        for normal in (np.zeros(3), np.zeros(2)):
            with self.subTest(normal=normal):
                with self.assertRaises(ValueError):
                    section(_cube(), np.zeros(3), normal)

    def test_a_bad_origin_is_refused(self):
        with self.assertRaises(ValueError):
            section(_cube(), np.zeros(2), array([0.0, 0.0, 1.0]))


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()

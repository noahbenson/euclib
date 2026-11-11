# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_boolean.py
'''Tests for the boolean operations in ``_boolean``.

A boolean operation is checked on two levels. The *numbers* are checked
against solids whose volumes are known: two cubes that overlap by half, one
cube inside another, two cubes apart. The *surface* is checked for the
properties a solid must have: it closes, it winds outward, its corners are
distinct, and it has the Euler characteristic a surface of its shape has. The
identities that tie the three operations together --- union plus intersection
equals the two inputs, difference equals the first minus the intersection ---
are checked on shapes with no symmetry at all, where a wrong cut has nowhere
to hide.

The inputs are checked for the repairs they are allowed to need. A mesh may be
wound inside out, or written with inconsistent windings, or list one position
several times; all three are common in files and neither is a reason to refuse
a solid. A mesh that is genuinely open is a different matter, and is refused.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, cos, eye, linspace, pi, sin, zeros
import unittest
from unittest import TestCase

from euclib.ops import (
    boolean_difference, boolean_intersection, boolean_union)
from euclib.types import Affine, TriMesh, trimesh


# Fixtures ###################################################################

#: The corner order the example pages use: a cube's corners, then the twelve
#: triangles, wound consistently and outward.
_CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
_CUBE_TRIS = array([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                    [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                    [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                   dtype='int64').T

#: The triangles of the cube page, which are not all wound the same way. It is
#: what a user who copies the page writes, and the operations must repair it.
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


def _sphere(n_lat=8, n_lon=12, /, shift=(0.0, 0.0, 0.0)):
    '''A latitude-longitude sphere, with each pole's copies listed.'''
    lat = linspace(0.0, pi, n_lat + 1)
    lon = linspace(0.0, 2.0 * pi, n_lon, endpoint=False)
    coords = array([[sin(t) * cos(p), sin(t) * sin(p), cos(t)]
                    for t in lat for p in lon], dtype=float)
    coords = coords.T + array(shift, dtype=float)[:, None]
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            (a, b) = (i * n_lon + j, i * n_lon + (j + 1) % n_lon)
            (c, d) = ((i + 1) * n_lon + j,
                      (i + 1) * n_lon + (j + 1) % n_lon)
            faces += [[a, b, d], [a, d, c]]
    return trimesh(coords, array(faces, dtype='int64').T)


def _rotation(angle, axis, shift, /):
    '''The affine transform that turns about an axis and then translates.'''
    axis = asarray(axis, dtype=float)
    axis = axis / np.sqrt(axis @ axis)
    cross_matrix = array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
    matrix = eye(4)
    matrix[:3, :3] = (np.identity(3) + sin(angle) * cross_matrix
                      + (1.0 - cos(angle)) * (cross_matrix @ cross_matrix))
    matrix[:3, 3] = shift
    return Affine(matrix)


def _triangles(mesh, /):
    '''A mesh result's triangles as a ``(3, M)`` integer matrix.'''
    return asarray(mesh.topo.indices)


def _volume(mesh, /):
    '''The signed volume a closed, outward-wound triangle mesh encloses.'''
    coords = asarray(mesh.coords)
    tris = _triangles(mesh)
    if tris.shape[1] == 0:
        return 0.0
    (a, b, c) = (coords[:, tris[0]].T, coords[:, tris[1]].T,
                 coords[:, tris[2]].T)
    return float((a * np.cross(b, c)).sum()) / 6.0


def _area(mesh, /):
    '''The surface area of a triangle mesh.'''
    coords = asarray(mesh.coords)
    tris = _triangles(mesh)
    if tris.shape[1] == 0:
        return 0.0
    (a, b, c) = (coords[:, tris[0]].T, coords[:, tris[1]].T,
                 coords[:, tris[2]].T)
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()) / 2.0


def _paired(tris, /):
    '''Whether every directed edge appears once and its reverse appears too.'''
    seen = {}
    for k in range(tris.shape[1]):
        (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(count == 1 and (edge[1], edge[0]) in seen
               for (edge, count) in seen.items())


def _euler(mesh, /):
    '''The Euler characteristic ``V - E + F`` of a welded mesh.'''
    tris = _triangles(mesh)
    edges = set()
    for k in range(tris.shape[1]):
        (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges.add((u, v) if u < v else (v, u))
    return mesh.coord_count - len(edges) + tris.shape[1]


def _distinct(coords, /):
    '''Whether no two columns of a coordinate matrix coincide.'''
    coords = asarray(coords)
    for i in range(coords.shape[1]):
        for j in range(i + 1, coords.shape[1]):
            if allclose(coords[:, i], coords[:, j], atol=1e-12, rtol=0.0):
                return False
    return True


# Tests ######################################################################

class TestTheUnion(TestCase):
    '''The solid that holds what either input holds.'''

    def test_two_cubes_that_overlap_by_half(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        self.assertAlmostEqual(_volume(union), 1.5, places=12)

    def test_two_identical_cubes_are_one_cube(self):
        union = boolean_union(_cube(), _cube())
        self.assertAlmostEqual(_volume(union), 1.0, places=12)
        self.assertEqual(union.coord_count, 8)
        self.assertEqual(_triangles(union).shape[1], 12)

    def test_two_cubes_apart_are_both(self):
        union = boolean_union(_cube(), _cube((3.0, 0.0, 0.0),
                                            (4.0, 1.0, 1.0)))
        self.assertAlmostEqual(_volume(union), 2.0, places=12)
        self.assertAlmostEqual(_area(union), 12.0, places=12)

    def test_a_cube_inside_another_is_the_outer_one(self):
        union = boolean_union(_cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                              _cube((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))
        self.assertAlmostEqual(_volume(union), 8.0, places=12)

    def test_an_empty_mesh_leaves_the_other_alone(self):
        empty = trimesh(zeros((3, 0)), zeros((3, 0), dtype='int64'))
        self.assertIs(boolean_union(empty, _cube()).coords.__class__,
                      np.ndarray)
        self.assertAlmostEqual(_volume(boolean_union(_cube(), empty)), 1.0,
                               places=12)
        self.assertAlmostEqual(_volume(boolean_union(empty, _cube())), 1.0,
                               places=12)
        self.assertEqual(boolean_union(empty, empty).coord_count, 0)

    def test_the_surface_closes_and_faces_outward(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        self.assertTrue(_paired(_triangles(union)))
        self.assertGreater(_volume(union), 0.0)


class TestTheIntersection(TestCase):
    '''The solid that holds what both inputs hold.'''

    def test_two_cubes_that_overlap_by_half(self):
        middle = boolean_intersection(_cube(), _cube((0.5, 0.0, 0.0),
                                                     (1.5, 1.0, 1.0)))
        self.assertAlmostEqual(_volume(middle), 0.5, places=12)

    def test_two_identical_cubes_are_one_cube(self):
        middle = boolean_intersection(_cube(), _cube())
        self.assertAlmostEqual(_volume(middle), 1.0, places=12)
        self.assertEqual(middle.coord_count, 8)

    def test_two_cubes_apart_have_no_intersection(self):
        middle = boolean_intersection(_cube(), _cube((3.0, 0.0, 0.0),
                                                     (4.0, 1.0, 1.0)))
        self.assertEqual(middle.coord_count, 0)
        self.assertEqual(_triangles(middle).shape[1], 0)
        self.assertEqual(middle.topo.simplex_count[2], 0)

    def test_a_cube_inside_another_leaves_the_inner_one(self):
        middle = boolean_intersection(_cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                                      _cube((0.5, 0.5, 0.5),
                                            (1.5, 1.5, 1.5)))
        self.assertAlmostEqual(_volume(middle), 1.0, places=12)

    def test_an_empty_mesh_makes_the_intersection_empty(self):
        empty = trimesh(zeros((3, 0)), zeros((3, 0), dtype='int64'))
        for (first, second) in ((empty, _cube()), (_cube(), empty),
                                (empty, empty)):
            with self.subTest(first=first.coord_count,
                              second=second.coord_count):
                middle = boolean_intersection(first, second)
                self.assertEqual(middle.coord_count, 0)

    def test_it_is_symmetric(self):
        one = _cube()
        other = _cube((0.25, 0.25, 0.25), (1.25, 1.25, 1.25))
        self.assertAlmostEqual(
            _volume(boolean_intersection(one, other)),
            _volume(boolean_intersection(other, one)), places=12)


class TestTheDifference(TestCase):
    '''The solid that holds what the first input holds and the second not.'''

    def test_two_cubes_that_overlap_by_half(self):
        rest = boolean_difference(_cube(), _cube((0.5, 0.0, 0.0),
                                                 (1.5, 1.0, 1.0)))
        self.assertAlmostEqual(_volume(rest), 0.5, places=12)

    def test_a_cube_minus_itself_is_empty(self):
        rest = boolean_difference(_cube(), _cube())
        self.assertEqual(rest.coord_count, 0)
        self.assertEqual(_triangles(rest).shape[1], 0)

    def test_a_cube_minus_a_cube_apart_is_the_first(self):
        rest = boolean_difference(_cube(), _cube((3.0, 0.0, 0.0),
                                                 (4.0, 1.0, 1.0)))
        self.assertAlmostEqual(_volume(rest), 1.0, places=12)
        self.assertAlmostEqual(_area(rest), 6.0, places=12)

    def test_a_cube_minus_one_inside_it_is_hollow(self):
        rest = boolean_difference(_cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                                  _cube((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))
        self.assertAlmostEqual(_volume(rest), 7.0, places=12)
        # The outer surface and the cavity are two closed surfaces, so the
        # Euler characteristic is two spheres' worth rather than one's.
        self.assertEqual(_euler(rest), 4)
        self.assertTrue(_paired(_triangles(rest)))

    def test_an_empty_first_input_leaves_nothing(self):
        empty = trimesh(zeros((3, 0)), zeros((3, 0), dtype='int64'))
        self.assertEqual(boolean_difference(empty, _cube()).coord_count, 0)
        self.assertAlmostEqual(_volume(boolean_difference(_cube(), empty)),
                               1.0, places=12)

    def test_it_is_not_symmetric(self):
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        self.assertAlmostEqual(_volume(boolean_difference(one, other)), 0.5,
                               places=12)
        self.assertAlmostEqual(_volume(boolean_difference(other, one)), 0.5,
                               places=12)
        # The numbers coincide here; the surfaces do not. Subtracting one cube
        # from the other removes the *other* half.
        rest = boolean_difference(one, other)
        self.assertLessEqual(float(asarray(rest.coords)[0].max()), 0.5 + 1e-9)


class TestTheIdentities(TestCase):
    '''The relations the three operations must satisfy on any solid.'''

    def _check(self, first, second, /):
        '''Checks the union and difference identities for two solids.'''
        # An input may be wound inward, in which case its signed volume is the
        # negative of the volume it encloses; the operations repair that.
        first_volume = abs(_volume(first))
        second_volume = abs(_volume(second))
        union = boolean_union(first, second)
        middle = boolean_intersection(first, second)
        rest = boolean_difference(first, second)
        self.assertAlmostEqual(
            _volume(union) + _volume(middle),
            first_volume + second_volume, places=9)
        self.assertAlmostEqual(
            _volume(rest), first_volume - _volume(middle), places=9)
        return (union, middle, rest)

    def test_two_cubes_that_overlap(self):
        self._check(_cube(), _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0)))

    def test_two_cubes_that_do_not_meet(self):
        self._check(_cube(), _cube((3.0, 0.0, 0.0), (4.0, 1.0, 1.0)))

    def test_a_cube_and_its_inside(self):
        self._check(_cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                    _cube((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))

    def test_two_spheres(self):
        self._check(_sphere(), _sphere(shift=(0.5, 0.0, 0.0)))

    def test_a_rotated_cube(self):
        turned = _cube().transformed(_rotation(0.3, (1.0, 1.0, 1.0),
                                               (0.5, 0.5, 0.5)))
        (union, middle, rest) = self._check(_cube(), turned)
        for result in (union, middle, rest):
            self.assertTrue(_paired(_triangles(result)))

    def test_an_irregular_intersection(self):
        # A cube corner pushed through another cube's face: the cut is a
        # hexagon, and none of the surfaces are coplanar across it.
        turned = _cube().transformed(_rotation(0.7, (1.0, 2.0, 3.0),
                                               (0.4, 0.3, 0.2)))
        (union, middle, rest) = self._check(_cube(), turned)
        self.assertGreater(_volume(middle), 0.0)
        self.assertLess(_volume(middle), 1.0)


class TestTheSurface(TestCase):
    '''What a result's surface has to be, whatever its shape.'''

    def test_every_result_closes(self):
        results = [
            boolean_union(_cube(), _sphere(shift=(0.5, 0.0, 0.0))),
            boolean_intersection(_sphere(), _sphere(shift=(0.5, 0.0, 0.0))),
            boolean_difference(_sphere(), _sphere(shift=(0.5, 0.0, 0.0))),
            boolean_difference(_cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                               _cube((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))]
        for result in results:
            self.assertTrue(_paired(_triangles(result)))

    def test_every_result_faces_outward(self):
        for result in (boolean_union(_cube(), _sphere(shift=(0.5, 0.0, 0.0))),
                       boolean_intersection(_sphere(),
                                            _sphere(shift=(0.5, 0.0, 0.0))),
                       boolean_difference(_sphere(),
                                          _sphere(shift=(0.5, 0.0, 0.0)))):
            self.assertGreater(_volume(result), 0.0)

    def test_a_simple_result_has_the_euler_characteristic_of_a_sphere(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        self.assertEqual(_euler(union), 2)
        middle = boolean_intersection(_sphere(),
                                      _sphere(shift=(0.5, 0.0, 0.0)))
        self.assertEqual(_euler(middle), 2)

    def test_no_two_corners_of_a_result_coincide(self):
        for result in (boolean_union(_cube(), _sphere(shift=(0.5, 0.0, 0.0))),
                       boolean_intersection(_sphere(),
                                            _sphere(shift=(0.5, 0.0, 0.0)))):
            self.assertTrue(_distinct(result.coords))

    def test_no_triangle_of_a_result_has_zero_area(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        coords = asarray(union.coords)
        tris = _triangles(union)
        for k in range(tris.shape[1]):
            (a, b, c) = (coords[:, tris[0, k]], coords[:, tris[1, k]],
                         coords[:, tris[2, k]])
            self.assertGreater(np.linalg.norm(np.cross(b - a, c - a)), 1e-8)

    def test_the_area_is_at_most_the_two_inputs_together(self):
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        self.assertLessEqual(_area(boolean_union(one, other)),
                             _area(one) + _area(other) + 1e-9)
        self.assertGreater(_area(boolean_union(one, other)), _area(one))


class TestTheInputs(TestCase):
    '''What the operations accept, repair, and refuse.'''

    def test_a_non_mesh_is_refused(self):
        for value in (array([[0.0], [0.0], [0.0]]), object(), None):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    boolean_union(value, _cube())
                with self.assertRaises(TypeError):
                    boolean_union(_cube(), value)

    def test_a_flat_mesh_is_refused(self):
        flat = trimesh(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                       array([[0, 1, 2]], dtype='int64').T)
        with self.assertRaisesRegex(ValueError, 'three-dimensional'):
            boolean_union(flat, _cube())

    def test_an_open_surface_is_refused(self):
        corner = array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0]])
        open_mesh = trimesh(corner, array([[0, 1, 2]], dtype='int64').T)
        with self.assertRaisesRegex(ValueError, 'not closed'):
            boolean_union(open_mesh, _cube())
        with self.assertRaisesRegex(ValueError, 'not closed'):
            boolean_union(_cube(), open_mesh)

    def test_a_docs_cube_with_mixed_windings_is_repaired(self):
        # The cube page's triangle list is not consistently wound, and the
        # operations must see the solid rather than the list.
        cube = _cube(tris=_DOCS_TRIS)
        self.assertAlmostEqual(_volume(boolean_union(cube, cube)), 1.0,
                               places=12)

    def test_a_mesh_wound_inside_out_is_repaired(self):
        inside_out = _cube(tris=_CUBE_TRIS[::-1])
        union = boolean_union(inside_out, inside_out)
        self.assertAlmostEqual(_volume(union), 1.0, places=12)
        self.assertGreater(_volume(union), 0.0)

    def test_a_position_listed_twice_is_one_corner(self):
        # A sphere lists its two poles once per longitude. The mesh is a solid
        # even though its topology, taken literally, is not.
        sphere = _sphere()
        self.assertEqual(sphere.coord_count, 9 * 12)
        union = boolean_union(sphere, sphere)
        self.assertAlmostEqual(_volume(union), abs(_volume(sphere)),
                               places=12)

    def test_the_inputs_are_not_modified(self):
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        before = (asarray(one.coords).copy(), asarray(one.topo.indices).copy(),
                  asarray(other.coords).copy(),
                  asarray(other.topo.indices).copy())
        boolean_union(one, other)
        after = (asarray(one.coords), asarray(one.topo.indices),
                 asarray(other.coords), asarray(other.topo.indices))
        for (was, now) in zip(before, after):
            self.assertTrue(allclose(was, now))

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        with_units = TriMesh(asarray(one.coords) * ureg.meter,
                             one.topo)
        other_units = TriMesh(asarray(other.coords) * ureg.meter,
                              other.topo)
        union = boolean_union(with_units, other_units)
        self.assertEqual(str(union.coords.units), 'meter')

    def test_a_tensor_input_gives_an_array_result(self):
        import torch
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        tensor_one = TriMesh(torch.tensor(asarray(one.coords)), one.topo)
        tensor_other = TriMesh(torch.tensor(asarray(other.coords)),
                               other.topo)
        union = boolean_union(tensor_one, tensor_other)
        self.assertIsInstance(union.coords, np.ndarray)
        self.assertAlmostEqual(_volume(union), 1.5, places=12)

    def test_the_tolerance_can_be_given(self):
        one = _cube()
        other = _cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        union = boolean_union(one, other, tolerance=1e-12)
        self.assertAlmostEqual(_volume(union), 1.5, places=12)


class TestTheGeometry(TestCase):
    '''The object a boolean operation returns.'''

    def test_a_result_is_a_triangle_mesh(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        self.assertIsInstance(union, TriMesh)
        self.assertEqual(union.dim, 3)
        self.assertEqual(union.coord_count, asarray(union.coords).shape[1])
        self.assertEqual(union.topo.simplex_count[2],
                         _triangles(union).shape[1])

    def test_a_result_can_be_measured_and_used(self):
        union = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        self.assertAlmostEqual(float(union.measures.sum()), _area(union),
                               places=12)
        self.assertEqual(union.bbox.shape, (3, 2))

    def test_a_boolean_of_a_result_is_allowed(self):
        first = boolean_union(_cube(), _cube((0.5, 0.0, 0.0),
                                            (1.5, 1.0, 1.0)))
        second = _cube((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))
        middle = boolean_intersection(first, second)
        self.assertTrue(_paired(_triangles(middle)))
        self.assertGreater(_volume(middle), 0.0)
        self.assertLess(_volume(middle), _volume(second) + 1e-12)


# Exports ####################################################################

if __name__ == '__main__':
    unittest.main()

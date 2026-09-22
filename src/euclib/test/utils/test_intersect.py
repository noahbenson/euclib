# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_intersect.py
'''Tests for the intersection kernels in ``euclib.utils._pycore``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, isfinite, zeros
from scipy.spatial import ConvexHull

from euclib.utils import (
    barycentric_in_triangle, segments_intersect, segments_triangles_intersect,
    tetrahedron_box_intersection, triangles_segments_intersect)


# Fixtures ###################################################################

def _unit_tetrahedron():
    '''The tetrahedron spanned by the origin and the three axes.'''
    return array([[0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]])


def _unit_triangle():
    '''The triangle (0, 0), (1, 0), (0, 1) in the z = 0 plane.'''
    return (array([[0.], [0.], [0.]]),
            array([[1.], [0.], [0.]]),
            array([[0.], [1.], [0.]]))


# Tests ######################################################################

class TestSegments(TestCase):
    '''The pairwise segment test.'''

    def test_crossing_segments(self):
        (hit, point, s, t) = segments_intersect(
            array([[0.], [0.]]), array([[1.], [1.]]),
            array([[0.], [1.]]), array([[1.], [0.]]), tolerance=1e-9)
        self.assertTrue(bool(hit[0]))
        self.assertTrue(allclose(point.ravel(), [0.5, 0.5]))
        self.assertAlmostEqual(float(s[0]), 0.5)
        self.assertAlmostEqual(float(t[0]), 0.5)

    def test_parallel_segments_do_not_cross(self):
        (hit, _, _, _) = segments_intersect(
            array([[0.], [0.]]), array([[1.], [0.]]),
            array([[0.], [1.]]), array([[1.], [1.]]), tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_skew_segments_do_not_cross(self):
        # Two segments in different planes that would cross if they were.
        (hit, _, _, _) = segments_intersect(
            array([[0.], [0.], [0.]]), array([[1.], [0.], [0.]]),
            array([[0.5], [1.], [0.]]), array([[0.5], [2.], [0.]]),
            tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_touching_segments_cross(self):
        (hit, point, _, _) = segments_intersect(
            array([[0.], [0.], [0.]]), array([[1.], [0.], [0.]]),
            array([[0.5], [0.], [0.]]), array([[0.5], [1.], [0.]]),
            tolerance=1e-9)
        self.assertTrue(bool(hit[0]))
        self.assertTrue(allclose(point.ravel(), [0.5, 0., 0.]))

    def test_two_segments_that_pass_by_each_other(self):
        # A hundredth apart: a tolerance that reaches it counts as meeting, and
        # one that does not, does not. The tolerance is the whole of the
        # difference between the two answers.
        gap = 0.01
        near = (array([[0.], [0.], [0.]]), array([[1.], [0.], [0.]]),
                array([[0.5], [gap], [0.]]), array([[0.5], [1.], [0.]]))
        (hit, _, _, _) = segments_intersect(*near, tolerance=0.02)
        self.assertTrue(bool(hit[0]))
        (hit, _, _, _) = segments_intersect(*near, tolerance=1e-3)
        self.assertFalse(bool(hit[0]))


class TestSegmentsAndTriangles(TestCase):
    '''The segment-against-triangle test.'''

    def test_a_segment_through_a_triangle(self):
        (a, b, c) = _unit_triangle()
        (hit, point, weight) = segments_triangles_intersect(
            array([[0.25], [0.25], [-1.]]), array([[0.25], [0.25], [1.]]),
            a, b, c, tolerance=1e-9)
        self.assertTrue(bool(hit[0]))
        self.assertTrue(allclose(point.ravel(), [0.25, 0.25, 0.]))

    def test_a_segment_that_misses_the_triangle(self):
        (a, b, c) = _unit_triangle()
        (hit, _, _) = segments_triangles_intersect(
            array([[2.], [2.], [-1.]]), array([[2.], [2.], [1.]]),
            a, b, c, tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_a_segment_that_stops_before_the_triangle(self):
        (a, b, c) = _unit_triangle()
        (hit, _, _) = segments_triangles_intersect(
            array([[0.25], [0.25], [1.]]), array([[0.25], [0.25], [2.]]),
            a, b, c, tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_a_segment_parallel_to_the_triangle(self):
        (a, b, c) = _unit_triangle()
        (hit, _, _) = segments_triangles_intersect(
            array([[0.1], [0.1], [1.]]), array([[0.4], [0.4], [1.]]),
            a, b, c, tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_the_weights_of_a_crossing(self):
        (a, b, c) = _unit_triangle()
        (_, _, weight) = segments_triangles_intersect(
            array([[0.25], [0.25], [-1.]]), array([[0.25], [0.25], [1.]]),
            a, b, c, tolerance=1e-9)
        self.assertTrue(allclose(weight.ravel(), [0.5, 0.25, 0.25], atol=1e-9))


class TestBarycentric(TestCase):
    '''The barycentric coordinates of a triangle.'''

    def test_the_corners_themselves(self):
        (a, b, c) = _unit_triangle()
        for (corner, expected) in ((a, [1., 0., 0.]), (b, [0., 1., 0.]),
                                   (c, [0., 0., 1.])):
            self.assertTrue(allclose(
                barycentric_in_triangle(corner, a, b, c).ravel(), expected,
                atol=1e-12))

    def test_a_point_inside(self):
        (a, b, c) = _unit_triangle()
        inside = array([[0.25], [0.25], [0.]])
        self.assertTrue(allclose(
            barycentric_in_triangle(inside, a, b, c).ravel(),
            [0.5, 0.25, 0.25], atol=1e-12))

    def test_a_point_outside_has_a_negative_weight(self):
        (a, b, c) = _unit_triangle()
        outside = array([[1.], [1.], [0.]])
        weighted = barycentric_in_triangle(outside, a, b, c).ravel()
        self.assertTrue((weighted < 0).any())


class TestTriangles(TestCase):
    '''The pairwise triangle test.'''

    def test_two_triangles_crossing_along_a_line(self):
        # One in the plane y = 0, one in the plane z = 0.
        (v0, v1, v2) = (array([[0.], [0.], [-1.]]), array([[1.], [0.], [-1.]]),
                        array([[0.5], [0.], [1.]]))
        (w0, w1, w2) = (array([[0.], [1.], [0.]]), array([[1.], [1.], [0.]]),
                        array([[0.5], [-1.], [0.]]))
        (hit, start, stop) = triangles_segments_intersect(
            v0, v1, v2, w0, w1, w2, tolerance=1e-9)
        self.assertTrue(bool(hit[0]))
        ends = sorted([start.ravel().tolist(), stop.ravel().tolist()])
        self.assertTrue(allclose(ends[0], [0.25, 0., 0.], atol=1e-9))
        self.assertTrue(allclose(ends[1], [0.75, 0., 0.], atol=1e-9))

    def test_two_triangles_that_miss(self):
        (v0, v1, v2) = (array([[0.], [0.], [-1.]]), array([[1.], [0.], [-1.]]),
                        array([[0.5], [0.], [1.]]))
        (w0, w1, w2) = (array([[0.], [5.], [0.]]), array([[1.], [5.], [0.]]),
                        array([[0.5], [5.], [0.]]))
        (hit, _, _) = triangles_segments_intersect(
            v0, v1, v2, w0, w1, w2, tolerance=1e-9)
        self.assertFalse(bool(hit[0]))

    def test_coplanar_triangles_are_not_reported(self):
        # Two triangles in the plane y = 0 that share part of an edge: they
        # meet over an area, which this test does not describe.
        (v0, v1, v2) = (array([[0.], [0.], [-1.]]), array([[1.], [0.], [-1.]]),
                        array([[0.5], [0.], [1.]]))
        (w0, w1, w2) = (array([[0.], [0.], [-1.]]), array([[1.], [0.], [-1.]]),
                        array([[0.5], [0.], [-2.]]))
        (hit, _, _) = triangles_segments_intersect(
            v0, v1, v2, w0, w1, w2, tolerance=1e-9)
        self.assertFalse(bool(hit[0]))


class TestTetrahedronAndBox(TestCase):
    '''The region a tetrahedron and a box share.'''

    def test_the_whole_tetrahedron_inside_one_box(self):
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 1.], [0., 1.], [0., 1.]]))
        self.assertEqual(vertices.shape[1], 4)
        self.assertEqual(tets.shape[1], 1)
        self.assertAlmostEqual(ConvexHull(vertices.T).volume, 1. / 6.)

    def test_a_box_that_cuts_the_tetrahedron_in_half(self):
        # The part with x <= 1/2: the integral of (1 - x)^2 / 2 from 0 to 1/2.
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 0.5], [0., 1.], [0., 1.]]))
        self.assertGreater(tets.shape[1], 1)
        self.assertAlmostEqual(ConvexHull(vertices.T).volume, 7. / 48.)

    def test_a_box_entirely_inside_the_tetrahedron(self):
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 0.25], [0., 0.25], [0., 0.25]]))
        self.assertAlmostEqual(ConvexHull(vertices.T).volume, 0.25 ** 3)

    def test_a_slab_through_the_tetrahedron(self):
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 0.5], [0., 0.5], [0., 1.]]))
        self.assertAlmostEqual(ConvexHull(vertices.T).volume, 0.125)

    def test_a_box_that_misses(self):
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[5., 6.], [5., 6.], [5., 6.]]))
        self.assertEqual(vertices.shape[1], 0)
        self.assertEqual(tets.shape[1], 0)

    def test_a_box_that_touches_only_a_corner(self):
        # The region is a single point, which has no volume to fill.
        (vertices, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 0.], [0., 0.], [0., 0.]]))
        self.assertEqual(tets.shape[1], 0)

    def test_the_tetrahedra_have_the_shape_of_tetrahedra(self):
        (_, tets) = tetrahedron_box_intersection(
            _unit_tetrahedron(), array([[0., 0.5], [0., 1.], [0., 1.]]))
        self.assertEqual(tets.shape[0], 4)
        self.assertTrue(isfinite(tets).all())

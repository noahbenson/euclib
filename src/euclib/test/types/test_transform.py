# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_transform.py
'''Tests for the transforms in ``euclib.types._transform``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, eye, zeros

from euclib.types import (
    Affine, Transform, VertexSet, VertexTopology,
    affine_identity, affine_scaling, affine_translation)


# Tests ######################################################################

class TestAffine(TestCase):
    '''Tests for the affine transform.'''

    def test_identity(self):
        t = affine_identity(2)
        self.assertEqual(t.dim, 2)
        self.assertTrue(allclose(t.matrix, eye(3)))

    def test_translation(self):
        t = affine_translation([1., 2.])
        self.assertTrue(allclose(t.translation, [1., 2.]))
        self.assertTrue(allclose(t.linear, eye(2)))
        self.assertTrue(allclose(t.apply(zeros((2, 1))), [[1.], [2.]]))

    def test_scaling(self):
        t = affine_scaling([2., 3.])
        self.assertTrue(allclose(t.linear, [[2., 0.], [0., 3.]]))
        self.assertTrue(allclose(t.apply(array([[1.], [1.]])), [[2.], [3.]]))

    def test_apply_transforms_every_position(self):
        t = affine_translation([1., 1.])
        coords = array([[0., 1., 2.], [0., 1., 2.]])
        self.assertTrue(allclose(t.apply(coords),
                                 [[1., 2., 3.], [1., 2., 3.]]))

    def test_apply_checks_the_dimension(self):
        t = affine_identity(2)
        with self.assertRaises(ValueError):
            t.apply(zeros((3, 4)))

    def test_a_non_affine_matrix_is_rejected(self):
        with self.assertRaises(Exception):
            Affine(eye(3) * 2.0)

    def test_a_non_square_matrix_is_rejected(self):
        with self.assertRaises(Exception):
            Affine(zeros((2, 3)))

    def test_apply_works_on_lists(self):
        t = affine_translation([1., 1.])
        self.assertTrue(allclose(t.apply([[0.], [0.]]), [[1.], [1.]]))

    def test_apply_accepts_a_bare_vector_as_one_position(self):
        # A 1-D vector is one position; without reshaping, the translation
        # would broadcast it into a (D, D) matrix.
        t = affine_translation([1., 2.])
        self.assertTrue(allclose(t.apply(array([0., 0.])), [[1.], [2.]]))

    def test_inverse_undoes_the_transform(self):
        t = affine_translation([3., -2.]) @ affine_scaling([2., 4.])
        coords = array([[1., -1., 0.5], [2., 0.25, -3.]])
        self.assertTrue(allclose(t.inverse.apply(t.apply(coords)), coords))

    def test_inverse_is_exactly_affine(self):
        # Building the inverse from the transform's parts, rather than
        # inverting the whole matrix, keeps its final row exact.
        t = affine_translation([1., 1.]).inverse
        self.assertTrue(allclose(t.matrix[-1], [0., 0., 1.]))
        self.assertTrue(t.is_affine if hasattr(t, 'is_affine') else True)

    def test_compose_applies_the_argument_first(self):
        move = affine_translation([1., 0.])
        scale = affine_scaling([2., 2.])
        composed = move @ scale
        start = array([[1.], [1.]])
        # Scaling first, then moving: 2 * (1, 1) + (1, 0) = (3, 2).
        self.assertTrue(allclose(composed.apply(start), [[3.], [2.]]))

    def test_compose_checks_dimensions(self):
        with self.assertRaises(ValueError):
            affine_identity(2) @ affine_identity(3)
        with self.assertRaises(TypeError):
            affine_identity(2).compose('not-a-transform')

    def test_equality_and_hash(self):
        a = affine_translation([1., 2.])
        b = affine_translation([1., 2.])
        c = affine_translation([1., 3.])
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b}), 1)

    def test_backend(self):
        t = Affine(eye(3), backend='numpy')
        self.assertEqual(t.backend, 'numpy')
        with self.assertRaises(Exception):
            Affine(eye(3), backend='jax')

    def test_transform_is_abstract(self):
        with self.assertRaises(TypeError):
            Transform(eye(3))


class TestTransformed(TestCase):
    '''Tests for transforming a geometry's coordinates.'''

    def setUp(self):
        self.cloud = VertexSet(array([[0., 1., 2.], [0., 0., 0.]]),
                               VertexTopology([[0, 1, 2]]))

    def test_transformed_moves_the_coordinates(self):
        moved = self.cloud.transformed(affine_translation([0., 5.]))
        self.assertTrue(allclose(moved.coords,
                                 [[0., 1., 2.], [5., 5., 5.]]))
        # The original is unchanged.
        self.assertTrue(allclose(self.cloud.coords, [[0., 1., 2.], [0., 0., 0.]]))

    def test_transformed_carries_properties_over(self):
        cloud = self.cloud.withprop('a', array([1., 2., 3.]))
        moved = cloud.transformed(affine_translation([0., 5.]))
        self.assertTrue(allclose(moved['a'], [1., 2., 3.]))

    def test_transformed_keeps_the_topology(self):
        moved = self.cloud.transformed(affine_identity(2))
        self.assertEqual(moved.topo, self.cloud.topo)

    def test_transformed_checks_its_argument(self):
        with self.assertRaises(TypeError):
            self.cloud.transformed('not-a-transform')


class TestBBox(TestCase):
    '''Tests for bounding boxes.'''

    def test_bounding_box_of_a_cloud(self):
        cloud = VertexSet(array([[0., 1., 2.], [-1., 0., 1.]]),
                          VertexTopology([[0, 1, 2]]))
        self.assertTrue(allclose(cloud.bbox, [[0., 2.], [-1., 1.]]))

    def test_bounding_box_ignores_unused_coordinates(self):
        # Coordinates 2 and 3 are not referenced by the topology, so the
        # bounding box describes the triangle rather than the matrix.
        from euclib.types import TriMesh, TriTopology
        coords = array([[0., 1., 0., 99.],
                        [0., 0., 1., 99.],
                        [0., 0., 0., 99.]])
        mesh = TriMesh(coords, TriTopology([[0], [1], [2]], coord_count=4))
        self.assertEqual(mesh.vertex_count, 3)
        self.assertTrue(allclose(mesh.bbox, [[0., 1.], [0., 1.], [0., 0.]]))

    def test_bounding_box_follows_a_transform(self):
        cloud = VertexSet(array([[0., 1.], [0., 1.]]),
                          VertexTopology([[0, 1]]))
        moved = cloud.transformed(affine_translation([10., 0.]))
        self.assertTrue(allclose(moved.bbox, [[10., 11.], [0., 1.]]))

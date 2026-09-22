# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_pycore.py
'''Tests for the ``euclib.utils._pycore`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

from numpy import array, ones, zeros, float64, int64

from euclib.utils import is_pointdata, unique_columns, unique_coords
from euclib._init import checktorch


# Tests ######################################################################

class TestIsPointdata(TestCase):
    '''Tests for the ``is_pointdata`` predicate.'''

    def test_accepts_coordinate_matrices(self):
        self.assertTrue(is_pointdata(zeros((2, 5))))
        self.assertTrue(is_pointdata(zeros((3, 5))))
        # A single-coordinate 1xN payload is permitted for point clouds.
        self.assertTrue(is_pointdata(zeros((1, 5))))

    def test_rejects_non_arrays(self):
        self.assertFalse(is_pointdata([1, 2, 3]))
        self.assertFalse(is_pointdata('xyz'))
        self.assertFalse(is_pointdata(None))
        self.assertFalse(is_pointdata({}))

    def test_honors_dims(self):
        self.assertTrue(is_pointdata(zeros((2, 5)), dims=(2,)))
        self.assertFalse(is_pointdata(zeros((2, 5)), dims=(3,)))

    def test_honors_ndim(self):
        self.assertFalse(is_pointdata(zeros((2, 5, 4)), ndim=(1, 2)))
        self.assertTrue(is_pointdata(zeros((2, 5, 4)), ndim=(2, 3)))
        # A bare vector has no spatial dimension to speak of.
        self.assertFalse(is_pointdata(zeros(5)))

    def test_honors_shape(self):
        self.assertTrue(is_pointdata(zeros((2, 5)), shape=(5,)))
        self.assertFalse(is_pointdata(zeros((2, 5)), shape=(6,)))
        self.assertTrue(is_pointdata(zeros((3, 4, 5)), shape=(4, 5),
                                     ndim=(1, 2, 3)))

    def test_honors_dtype(self):
        self.assertTrue(is_pointdata(zeros((2, 5)), dtype=float64))
        self.assertFalse(is_pointdata(ones((2, 5), dtype=int64),
                                      dtype=float64))
        self.assertTrue(is_pointdata(ones((2, 5), dtype=int64)))

    @skipUnless(checktorch() is not None, 'torch is not installed')
    def test_accepts_tensors(self):
        torch = checktorch()
        self.assertTrue(is_pointdata(torch.zeros((2, 5))))
        self.assertFalse(is_pointdata(torch.zeros((2, 5)), dims=(3,)))


class TestUniqueColumns(TestCase):
    '''Tests for the ``unique_columns`` kernel.'''

    def test_deduplicates_and_sorts(self):
        mat = array([[0, 1, 0],
                     [1, 2, 2]])
        res = unique_columns(mat)
        self.assertEqual(res.tolist(), [[0, 0, 1],
                                        [1, 2, 2]])

    def test_is_canonical_under_column_order(self):
        # The same set of columns, presented in a different order.
        a = array([[0, 1, 0], [1, 2, 2]])
        b = array([[1, 0, 0], [2, 1, 2]])
        self.assertEqual(unique_columns(a).tolist(), [[0, 0, 1],
                                                      [1, 2, 2]])
        self.assertEqual(unique_columns(a).tolist(),
                         unique_columns(b).tolist())

    def test_is_canonical_under_row_order(self):
        # A column whose entries are permuted is the same simplex.
        a = array([[0, 1], [1, 2], [2, 0]])
        b = array([[2, 1], [0, 2], [1, 0]])
        self.assertEqual(unique_columns(a).tolist(),
                         unique_columns(b).tolist())

    def test_deduplicates_simplices(self):
        # A point cloud is its own simplex collection.
        points = array([[0], [1], [2]])
        self.assertEqual(unique_columns(points).tolist(), [[0], [1], [2]])
        # Two distinct triangles are both kept.
        tris = array([[0, 0], [1, 2], [2, 3]])
        self.assertEqual(unique_columns(tris).tolist(), [[0, 0],
                                                         [1, 2],
                                                         [2, 3]])
        # The same triangle expressed in two vertex orders deduplicates to one.
        tris = array([[0, 0], [1, 2], [2, 1]])
        self.assertEqual(unique_columns(tris).tolist(), [[0],
                                                         [1],
                                                         [2]])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            unique_columns(array([1, 2, 3]))


class TestUniqueCoords(TestCase):
    '''Tests for the ``unique_coords`` kernel.'''

    def test_numpy_deduplicates_columns(self):
        coords = array([[0., 1., 0.],
                        [0., 0., 0.]])
        res = unique_coords(coords)
        self.assertEqual(res.tolist(), [[0., 1.],
                                        [0., 0.]])

    def test_numpy_return_index(self):
        coords = array([[0., 1., 0.],
                        [0., 0., 0.]])
        (uniq, index) = unique_coords(coords, return_index=True)
        self.assertEqual(uniq.tolist(), [[0., 1.], [0., 0.]])
        self.assertEqual(index.tolist(), [0, 1])

    def test_numpy_return_inverse(self):
        coords = array([[0., 1., 0.],
                        [0., 0., 0.]])
        (uniq, inverse) = unique_coords(coords, return_inverse=True)
        self.assertEqual(uniq[:, inverse].tolist(), coords.tolist())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            unique_coords(array([1., 2., 3.]))

    @skipUnless(checktorch() is not None, 'torch is not installed')
    def test_tensor_round_trip(self):
        torch = checktorch()
        coords = torch.tensor([[0., 1., 0.],
                               [0., 0., 0.]])
        res = unique_coords(coords)
        self.assertIsInstance(res, torch.Tensor)
        self.assertEqual(res.tolist(), [[0., 1.], [0., 0.]])

# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_pycore.py
'''Tests for the ``euclib.utils._pycore`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

from numpy import allclose, array, asarray, ones, zeros, float64, int64

from euclib.utils import (is_pointdata, simplex_measures, unique_columns,
                          unique_coords)
from euclib._init import checktorch


def mag(one, /):
    '''The magnitude, whether the argument is a quantity or an array.'''
    return asarray(one.m) if hasattr(one, 'm') else asarray(one)


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


class TestSimplexMeasures(TestCase):
    '''Tests for the ``simplex_measures`` kernel.

    A simplex's measure is its extent: a length, an area, or a volume. Unlike
    the searches that share this module --- ``nearest_vertices`` and
    ``closest_simplex``, which choose and therefore have nothing to
    differentiate --- a measure *is* a continuous function of the coordinates,
    and its docstring promises the coordinates' backend. So a tensor's gradient
    has to reach the coordinates through it, which is what these check.

    Nothing did, and the function could not: it clamped a small negative gram
    determinant with ``torch.clamp``, which has no dispatch for the quantity
    that immlib's arithmetic had produced, so a tensor *raised* rather than
    being mishandled. The array tests passed throughout, which is why only a
    test that hands it a tensor finds this.
    '''

    def _torch(self):
        if checktorch() is None:                             # pragma: no cover
            self.skipTest("torch is not installed")
        return checktorch()

    #: A unit segment, a unit right triangle, and the standard tetrahedron.
    SEGMENT = array([[0., 3.], [0., 0.]])
    TRIANGLE = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    TETRAHEDRON = array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])

    def test_it_measures_each_order(self):
        self.assertTrue(allclose(
            simplex_measures(self.SEGMENT, array([[0], [1]])), [3.0]))
        self.assertTrue(allclose(
            simplex_measures(self.TRIANGLE, array([[0], [1], [2]])), [0.5]))
        self.assertTrue(allclose(
            simplex_measures(self.TETRAHEDRON, array([[0], [1], [2], [3]])),
            [1.0 / 6.0]))

    def test_a_tensor_geometry_carries_its_gradient(self):
        '''Against a central difference of the array path, entry by entry.

        Checking that a gradient merely *exists* would pass for one that was
        wrong; the difference is what says it is the right one.
        '''
        torch = self._torch()
        coords = torch.tensor(self.TRIANGLE, dtype=torch.float64,
                              requires_grad=True)
        indices = array([[0], [1], [2]])
        got = simplex_measures(coords, indices)
        self.assertTrue(getattr(got, 'requires_grad', False),
                        "the measures of a tensor must carry a gradient")
        got.sum().backward()

        want = zeros(self.TRIANGLE.shape)
        step = 1e-6
        for i in range(self.TRIANGLE.shape[0]):
            for j in range(self.TRIANGLE.shape[1]):
                up = zeros(self.TRIANGLE.shape)
                down = zeros(self.TRIANGLE.shape)
                up[i, j], down[i, j] = step, -step
                want[i, j] = (
                    sum(simplex_measures(self.TRIANGLE + up, indices))
                    - sum(simplex_measures(self.TRIANGLE + down, indices))
                ) / (2 * step)
        self.assertLess(abs(coords.grad.numpy() - want).max(), 1e-6)

    def test_the_two_backends_agree(self):
        torch = self._torch()
        for (label, coords, indices) in (
                ('a segment', self.SEGMENT, array([[0], [1]])),
                ('a triangle', self.TRIANGLE, array([[0], [1], [2]])),
                ('a tetrahedron', self.TETRAHEDRON,
                 array([[0], [1], [2], [3]]))):
            with self.subTest(shape=label):
                want = simplex_measures(coords, indices)
                got = simplex_measures(
                    torch.tensor(coords, dtype=torch.float64), indices)
                # Asserted as *equal*, not as less than a tolerance: two
                # implementations doing the same arithmetic should agree to the
                # last bit, and `assertLess(x, 0.0)` cannot say so.
                self.assertEqual(
                    float(abs(asarray(mag(got)) - asarray(mag(want))).max()), 0.0)

    def test_the_units_of_the_coordinates_survive(self):
        '''A bare zero does not align with a determinant's units.

        The clamp writes its zero as ``0.0 * x`` for this reason, and a first
        version that wrote ``0.0`` broke exactly this.
        '''
        import pint
        ureg = pint.UnitRegistry()
        coords = self.TRIANGLE * ureg.meter
        got = simplex_measures(coords, array([[0], [1], [2]]))
        self.assertEqual(str(got.units), 'meter ** 2')

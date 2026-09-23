# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/test_property.py
'''Tests for the ``euclib.abc._property`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

from numpy import array, dtype as npdtype, isnan, ones, zeros
from immlib.workflow import PlanError

from euclib.abc import (
    Property, is_property,
    QUANTITATIVE, QUALITATIVE,
    normalize_backend, normalize_vartype, normalize_interp, normalize_extrap,
    normalize_null, normalize_mask)
from euclib._init import checktorch


# Tests ######################################################################

class TestNormalizers(TestCase):
    '''Tests for the individual metadata normalizers.'''

    def test_normalize_backend(self):
        self.assertIsNone(normalize_backend(None))
        self.assertEqual(normalize_backend('numpy'), 'numpy')
        self.assertEqual(normalize_backend('torch'), 'torch')
        for bad in ('jax', 'prefer-numpy', 'prefer-torch', 'numpy '):
            with self.assertRaises(ValueError):
                normalize_backend(bad)

    @skipUnless(checktorch() is None, 'torch is installed here')
    def test_normalize_backend_requires_torch(self):
        with self.assertRaises(ValueError):
            normalize_backend('torch')

    def test_normalize_vartype_infers_from_dtype(self):
        self.assertEqual(normalize_vartype(None, zeros(3, dtype='f8')),
                         QUANTITATIVE)
        self.assertEqual(normalize_vartype(None, zeros(3, dtype='i4')),
                         QUALITATIVE)
        self.assertEqual(normalize_vartype(None, zeros(3, dtype='bool')),
                         QUALITATIVE)
        self.assertEqual(normalize_vartype('quantitative', zeros(3, dtype='i4')),
                         QUANTITATIVE)
        with self.assertRaises(ValueError):
            normalize_vartype('categorical', zeros(3))

    def test_normalize_interp(self):
        from euclib._init import default_quantitative_interp
        # Ellipsis selects the type-appropriate default.
        self.assertEqual(normalize_interp(Ellipsis, QUALITATIVE),
                         ('nearest', 0))
        self.assertEqual(normalize_interp(Ellipsis, QUANTITATIVE),
                         default_quantitative_interp)
        # A method name takes the order from the default...
        self.assertEqual(normalize_interp('nearest', QUALITATIVE),
                         ('nearest', 0))
        # ...an order takes the method from the default...
        self.assertEqual(normalize_interp(1, QUANTITATIVE),
                         ('polynomial', 1))
        # ...and a pair is taken as given.
        self.assertEqual(normalize_interp(('nearest', 0), QUANTITATIVE),
                         ('nearest', 0))

    def test_normalize_interp_rejects_bad_values(self):
        # A method euclib does not define.
        with self.assertRaises(ValueError):
            normalize_interp('spline', QUANTITATIVE)
        # An order outside 0 through 3.
        with self.assertRaises(ValueError):
            normalize_interp(4, QUANTITATIVE)
        # A value that is neither.
        with self.assertRaises(ValueError):
            normalize_interp([1], QUANTITATIVE)

    def test_normalize_interp_rejects_unsupported_combinations(self):
        # A method that euclib defines but has not built yet.
        with self.assertRaises(NotImplementedError):
            normalize_interp(('bezier', 2), QUANTITATIVE)
        with self.assertRaises(NotImplementedError):
            normalize_interp(('clough-tocher', 3), QUANTITATIVE)
        # 'nearest' is only meaningful at order 0.
        with self.assertRaises(ValueError):
            normalize_interp(('nearest', 1), QUANTITATIVE)

    def test_normalize_interp_canonicalizes_order_zero(self):
        # A fit of degree zero is the value itself, so order 0 is nearest
        # whichever method it was asked for under.
        self.assertEqual(normalize_interp(0, QUANTITATIVE), ('nearest', 0))
        self.assertEqual(normalize_interp(('polynomial', 0), QUANTITATIVE),
                         ('nearest', 0))
        self.assertEqual(normalize_interp(('bezier', 0), QUANTITATIVE),
                         ('nearest', 0))
        # ...and a bare 'nearest' takes order 0 rather than the default order.
        self.assertEqual(normalize_interp('nearest', QUANTITATIVE),
                         ('nearest', 0))

    def test_normalize_interp_rejects_non_nearest_qualitative(self):
        # A category has no meaning between its values, so anything that would
        # blend them is refused.
        for bad in (1, 2, ('polynomial', 1), ('bezier', 2), ('nearest', 1)):
            with self.assertRaises((ValueError, NotImplementedError)):
                normalize_interp(bad, QUALITATIVE)
        # ...though order 0 under any name is nearest, and so is allowed.
        self.assertEqual(normalize_interp('polynomial', QUALITATIVE),
                         ('nearest', 0))

    def test_normalize_extrap(self):
        self.assertIsNone(normalize_extrap(None))
        self.assertEqual(normalize_extrap(0), 0)
        for bad in (1, 'nearest', 'linear'):
            with self.assertRaises(ValueError):
                normalize_extrap(bad)

    def test_normalize_null_defaults_by_dtype(self):
        self.assertTrue(isnan(normalize_null(Ellipsis, npdtype('f8'),
                                             zeros(3, dtype='f8'))))
        self.assertEqual(
            normalize_null(Ellipsis, npdtype('i4'), zeros(3, dtype='i4')), 0)
        self.assertEqual(
            normalize_null(Ellipsis, npdtype('bool'), zeros(3, dtype='bool')),
            False)
        self.assertIsNone(
            normalize_null(Ellipsis, npdtype('O'), zeros(3, dtype='O')))
        # The null value falls back to the value's own dtype when none is set.
        self.assertEqual(normalize_null(Ellipsis, None, zeros(3, dtype='i4')),
                         0)
        self.assertEqual(normalize_null(-1, npdtype('i4'), zeros(3)), -1)

    def test_normalize_mask(self):
        self.assertIsNone(normalize_mask(None, (3,)))
        m = normalize_mask([True, False, True], (3,))
        self.assertEqual(m.dtype.kind, 'b')
        self.assertEqual(m.tolist(), [True, False, True])
        # A mask broadcastable to the spatial shape is permitted.
        self.assertEqual(normalize_mask(True, (3,)).shape, ())
        with self.assertRaises(ValueError):
            normalize_mask([True, False], (3,))


class TestProperty(TestCase):
    '''Tests for the ``Property`` type.'''

    def test_basic_construction(self):
        p = Property(zeros((3, 5)), (5,))
        self.assertTrue(is_property(p))
        self.assertEqual(p.spatial_shape, (5,))
        self.assertEqual(p.channel_shape, (3,))
        self.assertEqual(p.shape, (3, 5))
        self.assertEqual(p.vartype, QUANTITATIVE)
        self.assertTrue(p.is_quantitative)
        self.assertFalse(p.is_masked)
        self.assertIsNone(p.backend)

    def test_qualitative_inference(self):
        p = Property(ones(5, dtype='bool'), (5,))
        self.assertEqual(p.vartype, QUALITATIVE)
        self.assertFalse(p.is_quantitative)
        self.assertEqual(p.interp, ('nearest', 0))

    def test_shape_is_validated_eagerly(self):
        # The shape check is an eager calculation, so immlib reports its
        # failure while the property is being constructed, wrapping the error
        # in a PlanError.
        with self.assertRaises(PlanError):
            Property(zeros((3, 5)), (4,))
        with self.assertRaises(PlanError):
            Property(zeros((3,)), (5,))

    def test_metadata_is_normalized_on_construction(self):
        p = Property(zeros(5), (5,), interp='polynomial', extrap=0)
        self.assertEqual(p.interp, ('polynomial', 1))
        self.assertEqual(p.extrap, 0)
        # Qualitative data with a non-zero order is an error.
        with self.assertRaises(PlanError):
            Property(ones(5, dtype='i4'), (5,), interp=('polynomial', 1))

    def test_dtype_is_applied(self):
        p = Property(zeros(5), (5,), dtype='f4')
        self.assertEqual(p.value.dtype, npdtype('f4'))
        # Without an explicit dtype the value keeps its own.
        self.assertEqual(Property(zeros(5, dtype='f8'), (5,)).value.dtype,
                         npdtype('f8'))

    def test_backend_none_does_not_convert(self):
        v = zeros(5)
        p = Property(v, (5,))
        self.assertIs(p.value, v)

    @skipUnless(checktorch() is not None, 'torch is not installed')
    def test_backend_converts(self):
        torch = checktorch()
        p = Property(zeros(5), (5,), backend='torch')
        self.assertIsInstance(p.value, torch.Tensor)
        q = Property(torch.zeros(5), (5,), backend='numpy')
        self.assertEqual(type(q.value).__module__.split('.')[0], 'numpy')

    def test_masked(self):
        p = Property(zeros(5), (5,), mask=[True, True, False, False, False])
        self.assertTrue(p.is_masked)
        self.assertEqual(p.mask.tolist(),
                         [True, True, False, False, False])

    def test_withmeta(self):
        p = Property(zeros(5), (5,), interp=1)
        q = p.withmeta(interp=0)
        self.assertEqual(p.interp, ('polynomial', 1))
        self.assertEqual(q.interp, ('nearest', 0))
        # Changing nothing returns the same object.
        self.assertIs(p.withmeta(), p)
        # The value is never changed by withmeta.
        self.assertIs(q.value, p.value)
        with self.assertRaises(TypeError):
            p.withmeta(nonsense=1)

    def test_copy_revalidates_metadata(self):
        '''Filters run whenever an input changes, so copy() validates too.'''
        p = Property(zeros(5), (5,), interp=1)
        self.assertEqual(p.copy(interp=0).interp, ('nearest', 0))
        with self.assertRaises(PlanError):
            p.copy(interp=7)
        with self.assertRaises(PlanError):
            p.copy(extrap=2)

    def test_subprop(self):
        p = Property(zeros((2, 5)), (5,), interp=1)
        q = p.subprop((5,))
        self.assertEqual(q.spatial_shape, (5,))
        self.assertEqual(q.interp, ('polynomial', 1))

    def test_getitem_returns_raw_values(self):
        v = array([[1., 2., 3.], [4., 5., 6.]])
        p = Property(v, (3,))
        self.assertEqual(p[1].tolist(), [2., 5.])
        self.assertEqual(p[0, 2].tolist(), 3.)

    def test_equality_and_hash(self):
        a = Property(array([1., 2., 3.]), (3,), interp=1)
        b = Property(array([1., 2., 3.]), (3,), interp=1)
        c = Property(array([1., 2., 4.]), (3,), interp=1)
        d = Property(array([1., 2., 3.]), (3,), interp=0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertEqual(len({a, b, c, d}), 3)

    def test_immutability(self):
        p = Property(zeros(5), (5,))
        with self.assertRaises(TypeError):
            p.interp = 3
        with self.assertRaises(TypeError):
            p['new'] = 1

    def test_quantities_are_accepted(self):
        '''A property value may be a pint or immlib quantity.'''
        import pint
        from immlib import quant
        ureg = pint.UnitRegistry()
        # A plain pint quantity.
        p = Property(array([1., 2., 3.]) * ureg.meter, (3,))
        self.assertEqual(str(p.value.units), 'meter')
        self.assertEqual(p.vartype, QUANTITATIVE)
        self.assertEqual(p.shape, (3,))
        # An immlib quantity, which is what euclib encourages.
        q = Property(quant(array([1., 2., 3.]), 'meter'), (3,))
        self.assertEqual(str(q.value.units), 'meter')
        self.assertEqual(q.shape, (3,))

    def test_quantity_null_preserves_units(self):
        '''A quantity's default null value keeps the caller's units.'''
        import pint
        ureg = pint.UnitRegistry()
        p = Property(array([1., 2., 3.]) * ureg.meter, (3,))
        self.assertEqual(str(p.null.units), 'meter')
        self.assertTrue(isnan(float(p.null.magnitude)))
        # A dimensionless property still gets a plain null value.
        self.assertTrue(isnan(Property(array([1., 2., 3.]), (3,)).null))

    def test_quantity_equality_and_hash(self):
        import pint
        ureg = pint.UnitRegistry()
        a = Property(array([1., 2.]) * ureg.meter, (2,))
        b = Property(array([1., 2.]) * ureg.meter, (2,))
        c = Property(array([1., 3.]) * ureg.meter, (2,))
        d = Property(array([1., 2.]) * ureg.second, (2,))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_plan_inputs_are_the_constructor_arguments(self):
        self.assertEqual(
            set(Property.plan.inputs),
            {'value', 'spatial_shape', 'backend', 'vartype', 'interp', 'extrap',
             'dtype', 'mask', 'null', 'unit', 'detach'})

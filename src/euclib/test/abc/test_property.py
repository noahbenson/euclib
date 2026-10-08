# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/test_property.py
'''Tests for the ``euclib.abc._property`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

from numpy import allclose, array, dtype as npdtype, isnan, ones, zeros
from immlib.workflow import PlanError

def _innermost(call, /):
    '''Returns the innermost exception a call raises, or ``None``.

    A property's fields are validated by ``calc``s, and ``immlib`` reports a
    failure in one of those as a ``PlanError`` that wraps the error the caller
    caused. The error the caller caused is the one that says what is wrong.
    '''
    try:
        call()
    except Exception as exc:
        while exc.__cause__ is not None:
            exc = exc.__cause__
        return exc
    return None


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
            normalize_interp('bicubic', QUANTITATIVE)
        # An order outside 0 through 3.
        with self.assertRaises(ValueError):
            normalize_interp(4, QUANTITATIVE)
        # A value that is neither.
        with self.assertRaises(ValueError):
            normalize_interp([1], QUANTITATIVE)

    def test_normalize_interp_recognizes_what_is_not_built_yet(self):
        # A method euclib defines but has not built is *recognized* here: a
        # property does not know what it will be attached to, and the same
        # combination may be built for one element and not another. The geometry
        # is what refuses it, through `supported_interp`.
        self.assertEqual(normalize_interp(('bezier', 2), QUANTITATIVE),
                         ('bezier', 2))
        self.assertEqual(normalize_interp(('clough-tocher', 3), QUANTITATIVE),
                         ('clough-tocher', 3))
        # A method euclib does not define, or an order outside the range, is a
        # mistake rather than a gap.
        with self.assertRaises(ValueError):
            normalize_interp(('cubic-spline', 2), QUANTITATIVE)
        with self.assertRaises(ValueError):
            normalize_interp(('polynomial', 4), QUANTITATIVE)
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

    def test_the_shape_is_checked_when_the_field_is_read(self):
        # A `Property` never forces itself: building one runs nothing, which is
        # what lets a caller defer a value it may never read. Reading a field
        # runs that field's calc, and the shape check is one --- so a value that
        # does not fit is reported then.
        #
        # A caller who wants it reported at the point of the mistake is a
        # *geometry*, which reads `valid` when it attaches a property. That is
        # tested where it lives: see `test_geom.TestProperties`.
        p = Property(zeros((3, 5)), (4,))
        with self.assertRaises(PlanError):
            p.shape
        q = Property(zeros((3,)), (5,))
        with self.assertRaises(PlanError):
            q.shape

    def test_metadata_is_normalized_when_the_field_is_read(self):
        p = Property(zeros(5), (5,), interp='polynomial', extrap=0)
        self.assertEqual(p.interp, ('polynomial', 1))
        self.assertEqual(p.extrap, 0)
        # Qualitative data with a non-zero order is an error, reported when the
        # interpolation is read rather than when the property is built --- a
        # property never forces itself, which is what lets a value be deferred.
        q = Property(ones(5, dtype='i4'), (5,), interp=('polynomial', 1))
        with self.assertRaises(PlanError):
            q.interp

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
        '''A metadata update is checked, so copy() validates too.

        The two checks arrive differently, and that is the point of where each
        lives. The *form* of an interpolation needs only the argument, so it is
        checked as the copy is built and a bad order is a plain `ValueError`.
        The extrapolation is checked by its filter, inside the plan, so a bad
        one arrives wrapped.
        '''
        p = Property(zeros(5), (5,), interp=1)
        self.assertEqual(p.copy(interp=0).interp, ('nearest', 0))
        with self.assertRaises(ValueError):
            p.copy(interp=7)
        # Every field is lazy, so a bad extrapolation is reported by the read
        # that asks for it rather than by the copy that carried it.
        with self.assertRaises(PlanError):
            p.copy(extrap=2).extrap

    def test_the_form_of_an_interpolation_is_checked_when_it_is_given(self):
        '''A bad method or order needs no value, so it is raised on the spot.

        The value may be deferred and the vartype inferred from it, but the
        *form* of the argument --- a method name, an order, a pair of them, or
        Ellipsis --- is known the moment it is supplied. So this is the one
        interpolation check that does not wait for a read.

        It is in the constructor rather than in a filter because a filter is
        lazy: a filter runs when its field is read, and by then the caller has
        moved on from the mistake.
        '''
        for bad in (5, 'cubic-spline', 'nonsense', ('nearest', 1), ('bogus', 1),
                    ('polynomial',), object()):
            with self.subTest(interp=bad):
                with self.assertRaises(ValueError):
                    Property(zeros(4), (4,), interp=bad)
        # ...and the forms that are fine are fine, including the ones that leave
        # a part to the default.
        for good in (Ellipsis, None, 0, 2, 'nearest', 'polynomial',
                     ('polynomial', 1), ('bezier', 2)):
            with self.subTest(interp=good):
                Property(zeros(4), (4,), interp=good)

    def test_what_the_value_supports_waits_for_the_value(self):
        '''That an interpolation suits the data needs the data, so it waits.

        A property may be built before its value is available --- the value may
        be a lazy that is never read --- so this cannot be the constructor's
        check. It runs when a field is read, which is what `valid` asks for.
        '''
        from pcollections import lazy
        p = Property(lazy(lambda: zeros(4)), (4,), vartype='qualitative',
                     interp=('polynomial', 1))
        # Built, and the value untouched: a category has no meaning between the
        # values it takes, so this pair is refused, but not yet.
        self.assertIsInstance(_innermost(lambda: p.valid), ValueError)
        self.assertIsInstance(_innermost(lambda: p.interp), ValueError)

    def test_metadata_comes_from_a_property_a_lazy_yields(self):
        '''A lazy yielding a property carries that property's metadata.

        `pcollections.lazy` hands its arguments through as they were given: one
        handed a lazy gives that lazy to its function rather than resolving it
        first. `_inner_field` therefore has to resolve what it is handed. When
        it did not, `isinstance(lazy, Property)` was false and every metadata
        field quietly took its default --- a deferred qualitative property came
        back continuous, which is the kind of wrong that does not raise.
        '''
        from pcollections import lazy
        from numpy import stack
        from euclib.types import PrismMesh, PrismTopology
        source = Property(zeros(4), (4,), vartype='qualitative', interp=0)
        copied = Property(source, (4,))
        self.assertEqual(copied.vartype, 'qualitative')
        self.assertEqual(copied.interp, ('nearest', 0))
        # A derivative too, which `_build` used to override with its own `None`
        # default on the way through.
        derived = Property(zeros(4), (4,), gradient=zeros((2, 4)))
        self.assertEqual(tuple(Property._build(derived, (4,), None).gradient.shape),
                         (2, 4))
        # A lazy in the constructor cannot inherit: several of a property's
        # calcs are required, so resolving it there would read the value at
        # construction. A geometry's properties are deferred whole and built
        # from what the lazy yielded, which is where the inheritance happens.
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                          PrismTopology([[0], [1], [2]]))
        carried = prism.withprop('f', lazy(lambda: zeros(3)),
                                 vartype='qualitative', interp=0)
        self.assertEqual(carried.properties['f'].vartype, 'qualitative')
        self.assertEqual(carried.properties['f'].interp, ('nearest', 0))

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

    def test_derivatives_are_optional(self):
        # A property need not carry derivative data: the fits that need it
        # estimate it from the values when it is absent.
        p = Property(zeros(4), (4,))
        self.assertIsNone(p.gradient)
        self.assertIsNone(p.hessian)

    def test_derivative_shapes_are_checked_against_the_value(self):
        # A gradient is (C..., D, N) and a hessian (C..., D, D, N): the value's
        # channel dimensions, one axis per order of derivative, then the value's
        # spatial dimensions.
        #
        # The checks run when the field is read rather than when the property is
        # built: a property never forces itself, which is what lets a caller
        # defer a value it may never read.
        p = Property(zeros(4), (4,), gradient=zeros((2, 4)),
                     hessian=zeros((2, 2, 4)))
        self.assertEqual(tuple(p.gradient.shape), (2, 4))
        self.assertEqual(tuple(p.hessian.shape), (2, 2, 4))
        # Channel dimensions have to agree with the value's.
        channelled = Property(zeros((3, 4)), (4,), gradient=zeros((3, 2, 4)))
        self.assertEqual(tuple(channelled.gradient.shape), (3, 2, 4))
        for bad in (zeros((4, 4)),         # a dimension of 4 is neither 2 nor 3
                    zeros((2, 3)),         # the spatial shape is (4,), not (3,)
                    zeros((2,)),           # a gradient has an axis per order
                    zeros((2, 2, 3, 4))):  # one too many
            with self.subTest(shape=bad.shape):
                self.assertIsInstance(
                    _innermost(lambda: Property(zeros(4), (4,),
                                                gradient=bad).gradient),
                    ValueError)
        # A hessian's two derivative axes must be the same size.
        self.assertIsInstance(
            _innermost(lambda: Property(zeros(4), (4,),
                                        hessian=zeros((2, 3, 4))).hessian),
                        ValueError)
        # ...and a gradient for a channelled value must have its channels.
        self.assertIsInstance(
            _innermost(lambda: Property(zeros((3, 4)), (4,),
                                        gradient=zeros((2, 2, 4))).gradient),
                        ValueError)

    def test_withprop_attaches_a_gradient_and_replaces_it(self):
        # A path, because a metadata-only update to ``interp=1`` below is one
        # that a path can honour: a point cloud has no interior to interpolate
        # in, and refuses even linear interpolation when it is asked for.
        from euclib.types import SegPath, SegTopology
        cloud = SegPath(zeros((2, 4)), SegTopology([[0, 1, 2], [1, 2, 3]]))
        c = cloud.withprop('a', zeros(4), gradient=zeros((2, 4)))
        self.assertEqual(tuple(c.propinfo('a').gradient.shape), (2, 4))
        # A gradient may be given on its own, which leaves the values alone.
        c2 = c.withprop('a', gradient=ones((2, 4)))
        self.assertEqual(c2['a'].tolist(), c['a'].tolist())
        self.assertTrue(allclose(c2.propinfo('a').gradient, ones((2, 4))))
        # Giving values without a gradient drops it: derivative data describes
        # the values, so replacing them replaces it, and a fit that needs one
        # estimates it from the new values.
        c3 = c.withprop('a', ones(4))
        self.assertIsNone(c3.propinfo('a').gradient)
        # A metadata-only update, though, leaves the gradient alone.
        c4 = c.withprop('a', interp=1)
        self.assertTrue(allclose(c4.propinfo('a').gradient, zeros((2, 4))))
        # The original is untouched throughout.
        self.assertTrue(allclose(c.propinfo('a').gradient, zeros((2, 4))))

    def test_plan_inputs_are_the_constructor_arguments(self):
        self.assertEqual(
            set(Property.plan.inputs),
            {'value', 'spatial_shape', 'backend', 'vartype', 'interp', 'extrap',
             'border', 'dtype', 'mask', 'null', 'unit', 'detach', 'gradient',
             'hessian'})


class TestDeferredValues(TestCase):
    '''That a value may be deferred, and is read only when something asks.

    The library is lazy so that a value nobody reads is never computed --- a
    property may be attached to a large mesh and never looked at. A value
    supplied as a `pcollections.lazy` therefore has to survive construction, the
    geometry that takes it, and the fields that do not need it.

    What does read it is anything whose answer depends on it: `valid`, the shape,
    and the type a default interpolation is inferred from.
    '''

    @staticmethod
    def _counting(value, /):
        '''A lazy yielding ``value``, and the list that records a read.'''
        from pcollections import lazy
        read = []
        return (lazy(lambda: (read.append(True), value)[1]), read)

    def test_a_deferred_value_is_not_read_when_the_property_is_built(self):
        (value, read) = self._counting(zeros(4))
        Property(value, (4,), interp=0)
        self.assertEqual(read, [], "the value was read at construction")
        # Reading a field reads what that field needs, and for a property with a
        # deferred value that is the value itself: the metadata a property has
        # is the metadata of the value it was handed, and it is not known until
        # the value is. So there is no field whose read leaves the value unread
        # --- which is what makes `withprop(validate=True)` mean something.
        (value, read) = self._counting(zeros(4))
        p = Property(value, (4,), interp=0)
        p.backend
        self.assertEqual(read, [True], "a field read did not read the value")

    def test_a_deferred_value_is_read_when_a_field_needs_it(self):
        (value, read) = self._counting(zeros(4))
        p = Property(value, (4,), interp=0)
        self.assertEqual(read, [])
        p.valid
        self.assertEqual(read, [True], "valid did not read the value")

    def test_a_deferred_value_survives_withprop(self):
        from numpy import stack
        from euclib.types import PrismMesh, PrismTopology
        (value, read) = self._counting(zeros(3))
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                          PrismTopology([[0], [1], [2]]))
        carried = prism.withprop('f', value, interp=0)
        self.assertEqual(read, [], "withprop read the value")
        # ...and the property is there, with the metadata it was given, which
        # survives the deferral.
        self.assertEqual(carried.properties['f'].interp, ('nearest', 0))
        carried.prop('f')
        self.assertEqual(read, [True], "the value was never read")

    def test_validate_asks_for_the_checks_before_the_property_is_attached(self):
        '''`withprop(validate=True)` readies the value and checks it now.

        The default is not to. A property is validated lazily, so attaching a
        deferred value does not ready it; asking for validation is what readies
        it, and it is what makes an error stop the attachment rather than appear
        at the first read, by which point the property has long been in use.
        '''
        from numpy import stack
        from euclib.types import PrismMesh, PrismTopology
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        prism = PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                          PrismTopology([[0], [1], [2]]))
        (value, read) = self._counting(zeros(3))
        prism.withprop('f', value, interp=0, validate=True)
        self.assertEqual(read, [True], "validate=True did not read the value")
        # A value that does not fit is refused by the attachment itself...
        (value, read) = self._counting(zeros(2))
        with self.assertRaises(Exception):
            prism.withprop('f', value, interp=0, validate=True)
        # ...and without it the same value attaches, and the error waits for the
        # read that asks for the value.
        (value, read) = self._counting(zeros(2))
        carried = prism.withprop('f', value, interp=0)
        self.assertEqual(read, [], "the default readied the value")
        with self.assertRaises(Exception):
            carried.prop('f')

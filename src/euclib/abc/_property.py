# -*- coding: utf-8 -*-
###############################################################################
# euclib/abc/_property.py
'''The ``Property`` type and the metadata rules that govern it.

A ``Property`` pairs a value array with the metadata that governs how that
value behaves when it is interpolated, extrapolated, or masked. Properties are
attached to geometric objects under unique hashable names; a geometric object's
``properties`` field is a lazy dictionary (``pcollections.ldict``) whose values
are ``Property`` objects.

Every field of a property --- the value and each piece of metadata --- is
handled by a *filter*: a calculation whose output name is also one of its
inputs. A filter receives the value supplied by the caller and returns the
value that the property actually stores, normalizing and validating it in the
process. Filters are eager, so they run as soon as the property is built; and
because they belong to the calculation plan, they run again whenever the input
changes --- so ``property.copy(interp='cubic')`` is validated exactly as
``Property(..., interp='cubic')`` is.

The metadata and its normalization rules are:

``backend``
    ``None``, ``'numpy'``, or ``'torch'``. ``None`` (the default) leaves the
    value in whatever backend it already uses; an explicit backend converts the
    value to that backend.
``vartype``
    ``'quantitative'`` (interpolable) or ``'qualitative'`` (categorical),
    inferred from the value's dtype when unspecified.
``interp``
    How to fit a field through the values of the components around a position:
    a ``(method, order)`` pair such as ``('polynomial', 1)``. Qualitative data
    may only use ``('nearest', 0)``.
``extrap``
    ``None`` for "missing outside the object", or 0 for "the value at the
    nearest point on the object".
``dtype``
    The value's dtype, or ``None`` to keep the value's natural dtype.
``mask``
    An optional boolean array, broadcastable to the spatial shape, marking
    missing values.
``null``
    The value substituted for missing results; ``NaN`` for floating-point data
    and an appropriate zero-like value otherwise.
``unit``
    The value's unit. Unit algebra is not yet implemented.
``detach``
    Whether to detach a PyTorch tensor's gradient when converting it to NumPy.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    asarray, broadcast_shapes, dtype as npdtype, nan)

from pcollections import lazy

from immlib import to_array, to_tensor, is_quant, quant

from .._init import (
    checktorch, default_quantitative_interp)
from ._core import (
    planobject, calc, normalize_backend, planobject_eq, planobject_hash)


# Constants ##################################################################

#: The metadata value indicating that a field is unspecified.
UNSET = Ellipsis

#: The two kinds of property value.
QUANTITATIVE = 'quantitative'
QUALITATIVE = 'qualitative'
VARTYPES = (QUANTITATIVE, QUALITATIVE)

#: The interpolation methods that ``euclib`` defines. A method names a scheme
#: for fitting a field through a simplex's values; not all of them are
#: implemented yet, which is what ``INTERP_SUPPORTED`` records.
#:
#: Two of them are the schemes for a polynomial. ``'polynomial'`` fits the
#: monomial basis to an element's data by least squares, taking the solution of
#: least norm where the data leaves the fit under-determined; where the data
#: over-determines it, the fit is whichever polynomial the data came from, so a
#: quadratic is recovered exactly at order 2 on every element. ``'bezier'``
#: builds the Bernstein control values instead --- the construction that keeps
#: two elements agreeing on the face they share, and that spends a cubic's spare
#: freedom on the degree-elevation rule that recovers a quadratic exactly.
#:
#: At order 1 there is nothing for either to choose: one value per corner
#: determines a linear field and nothing more, so both methods are linear
#: interpolation. ``('bezier', 1)`` is the degree-1 Bernstein patch, which is
#: the barycentric blend itself, and is therefore **an alias for**
#: ``('polynomial', 1)``: the two may be written interchangeably at order 1, and
#: at no other order do they agree.
INTERP_METHODS = (
    'nearest', 'polynomial', 'clough-tocher', 'powell-sabin', 'catmull-rom',
    'bezier', 'spline', 'lanczos')

#: The interpolation orders that ``euclib`` defines, from 0 (nearest) to 3
#: (cubic).
INTERP_ORDERS = (0, 1, 2, 3)

#: The interpolation that a qualitative property uses, and the only one it may
#: use.
INTERP_QUALITATIVE = ('nearest', 0)

#: The interpolations that the element-wise engine implements for every element
#: it can be asked about. A property's ``interp`` is *recognized* whatever it
#: names, and it is a geometry that decides what it can honour: see
#: ``supported_interp``. This is the set that applies to the elements whose
#: higher orders have not been built, so a triangle and a tetrahedron support
#: it as it stands, and a segment supports more.
INTERP_SUPPORTED = (('nearest', 0), ('polynomial', 1), ('polynomial', 2),
                    ('polynomial', 3), ('bezier', 1), ('bezier', 2),
                    ('bezier', 3))

#: The interpolations a *grid* supports. A grid has no simplices: its cells are
#: the unit boxes of an index space, so its interpolation is a separable kernel
#: --- one one-dimensional kernel applied along each axis --- rather than a fit
#: on an element. What that buys is that the methods are the image-processing
#: ones, and what it costs is that they are value-only: none of them can use a
#: gradient, because none of them is built from one. See
#: ``examples/properties/grid-linear.md`` for the index space they work in and
#: the two that are built so far.
INTERP_SUPPORTED_GRID = (('nearest', 0), ('polynomial', 1), ('bezier', 1),
                         ('catmull-rom', 3), ('spline', 2), ('spline', 3),
                         ('polynomial', 2), ('polynomial', 3),
                         ('lanczos', 2), ('lanczos', 3))

#: The interpolations a *segment* supports. The element-wise schemes are built
#: one element at a time, and a segment is the first: a cubic is exactly
#: determined by the values and slopes at its two ends, and a quadratic's one
#: free coefficient is settled by the two slopes by least squares. A segment
#: takes the Bezier method and not the polynomial one above linear, because its
#: fit is the control-value construction.
INTERP_SUPPORTED_SEGMENT = (('nearest', 0), ('polynomial', 1),
                            ('polynomial', 2), ('polynomial', 3),
                            ('bezier', 1), ('bezier', 2), ('bezier', 3),
                            ('catmull-rom', 3))

#: The interpolations a *triangle* supports. A cubic's ten control values come
#: from the nine conditions a triangle's three values and three gradients
#: supply, so the fit is built edge by edge and its one interior value follows a
#: rule that reproduces quadratics; a quadratic's six are more than determined by
#: the same nine and are settled the same way. See
#: ``examples/properties/bezier-triangle.md`` for the construction and the
#: derivation. It also takes ``'clough-tocher'`` at order 3, which splits the
#: triangle into three and fits a cubic on each so that the field's *slope* is
#: continuous across an edge and not only its value; the scheme is cubic, so
#: that is the one order it answers at. It also takes ``'powell-sabin'`` at
#: order 2, which splits the triangle into six quadratics and needs nothing the
#: vertices do not carry --- see ``examples/properties/powell-sabin-triangle.md``.
#: A prism mesh reports a triangle's order
#: with a third local dimension and its own interpolation is deferred, so it
#: does not get these.
INTERP_SUPPORTED_TRIANGLE = (('nearest', 0), ('polynomial', 1),
                             ('polynomial', 2), ('polynomial', 3),
                             ('bezier', 1), ('bezier', 2), ('bezier', 3),
                             ('clough-tocher', 3), ('powell-sabin', 2))

#: The interpolations a *tetrahedron* supports. A cubic's twenty control values
#: come from the sixteen conditions a tetrahedron's four values and four
#: gradients supply, and the four that are left over sit one on each face, where
#: the triangle's own rule settles them; a quadratic's ten are settled the same
#: way, edge by edge. See ``tetrahedron_fit`` for the construction and
#: ``examples/properties/bezier-triangle.md`` for the derivation it follows.
INTERP_SUPPORTED_TETRAHEDRON = (('nearest', 0), ('polynomial', 1),
                                ('polynomial', 2), ('polynomial', 3),
                                ('bezier', 1), ('bezier', 2), ('bezier', 3))

#: The interpolations a *prism* supports. A prism is a triangle extruded along
#: an elevation rather than a simplex, and a property of one may carry more
#: elevations than the geometry has surfaces --- so the two things the higher
#: orders are built from do not apply: there is no simplex to fit, and the
#: values' own elevation axis is not something a tetrahedral fit of the
#: geometry's two surfaces can see. What the first order can do is the
#: linear blend within the triangle. The rest is deferred rather than refused;
#: see the roadmap.
INTERP_SUPPORTED_PRISM = (('nearest', 0), ('polynomial', 1), ('bezier', 1))

#: The valid extrapolation orders. Only 0 (nearest point on the object) is
#: supported; ``None`` means "no extrapolation".
EXTRAP_ORDERS = (None, 0)

#: The ways a grid's interpolation may continue its data past its own edges.
#:
#: A kernel reaches outside the grid whenever a position is near an edge --- a
#: position half a step inside the last cell's centre still wants a cell that
#: does not exist --- so an interpolation wider than one cell has to be told what
#: to find there. These are the three standard answers, all of them *symmetric*
#: in the sense that folding the data is the same as folding the kernel:
#:
#: * ``'constant'`` repeats each edge's value outward, ``. . . a a b c d e e . . .``
#: * ``'half-symmetric'`` repeats the edge value itself, ``. . . b a a b c d e e . . .``
#: * ``'whole-symmetric'`` reflects about the edge value, ``. . . b a b c d e d . . .``
#:
#: The names are Getreuer's (*Linear Methods for Image Interpolation*, IPOL 2011,
#: section 14.1), and the default is the second, which is the usual choice for
#: resampling an image because it is the one that commutes with flipping the data.
BORDER_EXTENSIONS = ('constant', 'half-symmetric', 'whole-symmetric')

#: The boundary extension a grid's interpolation uses when none is asked for.
BORDER_DEFAULT = 'half-symmetric'






def _derivative(deriv, value, spatial_shape, backend, dtype, detach, order,
                name, /):
    '''A derivative normalized against its value, on first read.

    The other half of `Property.proc_gradient` and `proc_hessian`: either the
    derivative or the value it belongs to may be deferred, and the derivative is
    normalized against the value's shape, so the work waits until both have
    arrived. Module-level, because a `lazy` closes over it.
    '''
    if isinstance(deriv, lazy):
        deriv = deriv()
    if isinstance(value, lazy):
        value = value()
    return normalize_derivative(
        convert_value(deriv, backend, dtype, detach),
        value, spatial_shape, order, name)


def _converted_value(value, backend, dtype, detach, /):
    '''A value converted, unwrapping a property if that is what arrived.

    The other half of `Property.proc_value`: a deferred value is converted when
    it is read, and what it yields may be a `Property` --- one geometry's
    property being carried onto another, say --- in which case the value is what
    it holds. Module-level, because a `lazy` closes over it.
    '''
    if isinstance(value, Property):
        value = value.value
    return convert_value(value, backend, dtype, detach)


def _value_shape(value, spatial_shape, /):
    '''A value's channel shape and full shape, checked against the spatial one.

    The other half of `Property.proc_shape`: a value given as a
    `pcollections.lazy` has no shape until it is read, so both the shape and the
    check that it ends with `spatial_shape` happen then. Module-level, because a
    `lazy` closes over it and nothing of the property is in scope.
    '''
    sh = tuple(value.shape)
    n = len(spatial_shape)
    if n > len(sh) or sh[len(sh) - n:] != tuple(spatial_shape):
        raise ValueError(
            f"value shape {sh} does not end with spatial shape"
            f" {tuple(spatial_shape)}")
    return (sh[:len(sh) - n], sh)


def _deferred_property(prop, spatial_shape, topo, /):
    '''Builds a property from a lazy, resolving what it yields first.

    `pcollections.lazy` passes its arguments through as they were given, so a
    lazy handed a lazy hands that lazy to its function rather than resolving it.
    `Property._build` wants the *value*, so the resolution belongs here: built
    from a property, the new one inherits its metadata; built from an array,
    which is a value and nothing more, it takes the defaults.

    Without this the metadata was silently dropped --- `isinstance(lazy,
    Property)` is false, so every field took its default and a deferred
    qualitative property came back continuous.
    '''
    return Property._build(prop(), spatial_shape, topo)


def _inner_field(prop, name, fallback, /):
    '''A field of the property that a deferred value yields, or a default.

    The other half of `Property.__init__`'s handling of a lazy value: a lazy may
    yield a `Property`, which has the field, or an array, which does not --- an
    array is a value and nothing more --- and then the default stands.

    What it is handed is left as it arrived, and is *not* resolved here, because
    several of a property's calcs are required: a `lazy=False` calc is realized
    when the plan is built, which resolves its inputs, so resolving here would
    read the value at construction and defeat the deferral. The consequence is
    that metadata cannot be inherited through a lazy that yields a property ---
    it is not known without reading the value. Where that inheritance is wanted,
    the property is deferred whole and built from what the lazy yielded, which
    is what `_deferred_property` does for a geometry's properties.

    Module-level, because a `pcollections.lazy` closes over it and nothing of the
    property is in scope.
    '''
    if not isinstance(prop, Property):
        return fallback
    return getattr(prop, name)


def normalize_border(border, /):
    '''Normalizes a ``border`` metadata value.

    Parameters
    ----------
    border : str or None
        The boundary extension. ``None`` means the default.

    Returns
    -------
    str
        One of ``BORDER_EXTENSIONS``.
    '''
    if border is None:
        return BORDER_DEFAULT
    if border not in BORDER_EXTENSIONS:
        raise ValueError(
            f"invalid border: {border!r}; expected one of {BORDER_EXTENSIONS}")
    return border


# Metadata Normalization #####################################################

def normalize_dtype(dtype, /):
    '''Normalizes a ``dtype`` metadata value.

    Parameters
    ----------
    dtype : dtype-like or None
        The required dtype, or ``None`` to keep the value's natural dtype.

    Returns
    -------
    numpy.dtype or None
        The resolved dtype, or ``None`` when none was requested.
    '''
    return None if dtype is None else npdtype(dtype)


def normalize_vartype(vartype, value, /):
    '''Normalizes a ``vartype`` metadata value, inferring it when unspecified.

    A value is *quantitative* if it can be meaningfully interpolated between
    samples --- real or complex data --- and *qualitative* if it cannot, such as
    booleans, integers, or object arrays.

    Parameters
    ----------
    vartype : str or None
        ``'quantitative'``, ``'qualitative'``, or ``None`` to infer the value
        from ``value``'s dtype.
    value : array-like
        The property's value.

    Returns
    -------
    str
        ``'quantitative'`` or ``'qualitative'``.
    '''
    if vartype is not None:
        if vartype not in VARTYPES:
            raise ValueError(
                f"invalid vartype: {vartype!r}; expected one of {VARTYPES} or None")
        return vartype
    # Floating-point and complex data interpolate; everything else does not.
    dt = _numpy_dtype(value)
    return QUANTITATIVE if dt is not None and dt.kind in 'fc' else QUALITATIVE


def default_interp(vartype, /):
    '''Returns the interpolation that a property uses when none is specified.

    Parameters
    ----------
    vartype : str
        The property's ``vartype``.

    Returns
    -------
    tuple of (str, int)
        The default interpolation method and order.
    '''
    if vartype == QUALITATIVE:
        return INTERP_QUALITATIVE
    return default_quantitative_interp


def check_interp_form(interp, /):
    '''Checks the form of an ``interp`` argument and returns what it names.

    The half of the interpolation check that needs nothing but the argument:
    that it is a method name, an order, a ``(method, order)`` pair, or
    ``Ellipsis``, and that the method and order it names are ones ``euclib``
    defines. Neither the value nor the ``vartype`` is needed, which is why a
    `Property` runs this when it is built --- a mistake in the argument is the
    caller's to see at once rather than at some later read.

    The other two questions are not this one. What the *value* supports needs
    the value, and is asked when a field is read; what a *geometry* honours is
    not a property's to ask at all, and is asked when it is attached.

    Parameters
    ----------
    interp : str, int, tuple of (str, int), Ellipsis, or None
        The interpolation as the caller supplied it.

    Returns
    -------
    tuple of (str or None, int or None)
        The method and order the argument names, with ``None`` standing for
        whichever of the two the argument leaves to the default --- both are
        ``None`` for ``Ellipsis``. `normalize_interp` completes it.

    Raises
    ------
    ValueError
        If the argument is not one of those four forms, or if the method or
        order it names is not one that ``euclib`` defines.
    '''
    if interp is UNSET or interp is None or interp is Ellipsis:
        return (None, None)
    if isinstance(interp, str):
        method = interp
        # A method name takes its order from the default, except that the only
        # order 'nearest' has is 0.
        order = 0 if method == 'nearest' else None
    elif isinstance(interp, int) and not isinstance(interp, bool):
        (method, order) = (None, interp)
    elif isinstance(interp, tuple) and len(interp) == 2:
        (method, order) = interp
    else:
        raise ValueError(
            f"invalid interp: {interp!r}; expected a method name, an order, a"
            f" (method, order) pair, or Ellipsis")
    if method is not None and method not in INTERP_METHODS:
        raise ValueError(
            f"unknown interpolation method: {method!r}; expected one of"
            f" {INTERP_METHODS}")
    if order is not None:
        if not isinstance(order, int) or isinstance(order, bool) \
                or order not in INTERP_ORDERS:
            raise ValueError(
                f"invalid interpolation order: {order!r}; expected one of"
                f" {INTERP_ORDERS}")
        if method == 'nearest' and order != 0:
            raise ValueError(
                f"'nearest' is only meaningful at order 0; found ('nearest',"
                f" {order})")
    return (method, order)


def normalize_interp(interp, vartype, /):
    '''Normalizes an ``interp`` metadata value to a method and an order.

    An interpolation is a pair: the *method* names the scheme that fits a field
    through a simplex's values, and the *order* says how much of that field to
    use. The two are not independent --- ``'nearest'`` is only meaningful at
    order 0 --- so they are stored and validated together.

    A value may be given in any of four forms: the pair itself; a method name,
    which takes the order from the default for the data's type; an order, which
    takes the method from the default; or ``Ellipsis`` (or ``None``) for the
    whole default.

    Parameters
    ----------
    interp : str, int, tuple, None, or Ellipsis
        The interpolation.
    vartype : str
        The property's ``vartype``. A qualitative property may only use
        ``('nearest', 0)``, because a category has no meaning between the
        values it takes.

    Returns
    -------
    tuple of (str, int)
        The interpolation method and order.

    Raises
    ------
    ValueError
        If the method is not one that ``euclib`` defines, if the order is not
        0 through 3, or if a qualitative property asks for anything but
        ``('nearest', 0)``.
    NotImplementedError
        If the method and order are recognized but not yet implemented.
    '''
    default = default_interp(vartype)
    (method, order) = check_interp_form(interp)
    if method is None and order is None:
        res = default
    elif method is None:
        res = (default[0], order)
    elif order is None:
        res = (method, 0 if method == 'nearest' else default[1])
    else:
        res = (method, order)
    (method, order) = res
    if order == 0:
        # A fit of degree zero is the value itself, so every method's order 0
        # is nearest-neighbour. Canonicalizing keeps 'nearest' the one spelling
        # of "no interpolation", so that `interp=0` means what it says however
        # the property's other metadata reads.
        method = 'nearest'
    res = (method, order)
    if vartype == QUALITATIVE and res != INTERP_QUALITATIVE:
        raise ValueError(
            f"qualitative properties can only use {INTERP_QUALITATIVE}; found"
            f" {res}")
    # What a geometry can honour is not decided here: a property does not know
    # what it will be attached to, and the same combination may be built for one
    # element and not another. `supported_interp` answers for a geometry, and
    # `check_property_interp` asks it when the property is attached.
    return res


def normalize_extrap(extrap, /):
    '''Normalizes an ``extrap`` metadata value.

    Parameters
    ----------
    extrap : None or int
        ``None`` means that points outside the object yield the property's null
        value; ``0`` means that they yield the value at the nearest point on
        the object. No higher-order extrapolation is supported.

    Returns
    -------
    None or int
        The validated extrapolation order.
    '''
    if extrap is not None and extrap != 0:
        raise ValueError(
            f"invalid extrap: {extrap!r}; expected one of {EXTRAP_ORDERS}")
    return extrap


def normalize_mask(mask, spatial_shape, /):
    '''Normalizes a ``mask`` metadata value.

    Parameters
    ----------
    mask : array-like or None
        A boolean mask marking missing values. It must be broadcastable to the
        property's spatial shape.
    spatial_shape : tuple of int
        The property's spatial shape.

    Returns
    -------
    numpy.ndarray or None
        The mask as a boolean array, or ``None`` when no mask was given.
    '''
    if mask is None:
        return None
    marr = asarray(mask).astype(bool)
    try:
        broadcast_shapes(marr.shape, tuple(spatial_shape))
    except ValueError as exc:
        raise ValueError(
            f"mask shape {marr.shape} is not broadcastable to spatial shape"
            f" {tuple(spatial_shape)}") from exc
    return marr


#: The numbers of dimensions a gradient or a hessian may be taken in. A
#: derivative is taken with respect to a position in the space the geometry
#: occupies, so it has as many axes as that space has dimensions. A property
#: cannot know which of the two its geometry lives in --- the same property may
#: be attached to a 2-D and a 3-D geometry --- so both are accepted here, and the
#: geometry checks that the derivative is the size of its own dimension.
DERIVATIVE_DIMS = (2, 3)


def normalize_derivative(deriv, value, spatial_shape, order, name, /):
    '''Normalizes a property's gradient or hessian.

    A derivative is data rather than metadata: it says how a value changes away
    from the component it belongs to, which is what a higher-order
    interpolation fits through when the values alone do not determine the
    polynomial.

    Its shape is the value's channel dimensions, then one axis per order of
    derivative, then the value's spatial shape --- ``(C..., D, N)`` for a
    gradient and ``(C..., D, D, N)`` for a hessian. The derivative axes go at
    the end of the channel block so that the spatial dimensions stay last, as
    the ``(C..., X...)`` ordering requires, and the same value may be described
    at every order because the channel dimensions do not move.

    Parameters
    ----------
    deriv : array-like or None
        The derivative to normalize. ``None`` means the property carries none.
    value : array-like
        The value the derivative belongs to; its shape says what channel
        dimensions the derivative must have and where its spatial dimensions
        begin.
    spatial_shape : tuple of int
        The property's spatial shape.
    order : int
        ``1`` for a gradient and ``2`` for a hessian.
    name : str
        The name of the field, for the error message.

    Returns
    -------
    array-like or None
        The derivative, or ``None`` when none was given.

    Raises
    ------
    ValueError
        If the derivative is not the shape of a derivative of that order for
        this value.
    '''
    if deriv is None:
        return None
    n = len(spatial_shape)
    channel = tuple(value.shape)[:len(value.shape) - n]
    sh = tuple(deriv.shape)
    axes = 'axis' if order == 1 else 'axes'

    def wrong():
        return ValueError(
            f"{name} must be the shape of a derivative of order {order} for a"
            f" value of shape {tuple(value.shape)}: the value's channel"
            f" dimensions {channel}, then {order} {axes} of size 2 or 3, then"
            f" its spatial shape {tuple(spatial_shape)}; found {sh}")

    if len(sh) != len(channel) + order + n:
        raise wrong()
    dims = sh[len(channel):len(channel) + order]
    if (sh[:len(channel)] != channel
            or sh[len(channel) + order:] != tuple(spatial_shape)
            or len(set(dims)) != 1
            or dims[0] not in DERIVATIVE_DIMS):
        raise wrong()
    return deriv


def normalize_null(null, dtype, value, /):
    '''Normalizes a ``null`` metadata value.

    Parameters
    ----------
    null : object or Ellipsis
        The value to substitute for missing results. ``Ellipsis`` selects the
        default for the dtype: ``NaN`` for floating-point data, ``0`` for
        integers, ``False`` for booleans, and ``None`` otherwise.
    dtype : numpy.dtype or None
        The property's requested dtype.
    value : array-like
        The property's value, used to determine the dtype when ``dtype`` is
        unspecified.

    Returns
    -------
    object
        The null value. When the value is a ``pint`` or ``immlib`` quantity,
        the null value is likewise a quantity with the same units, so that a
        null result remains unit-consistent; ``euclib`` never invents units of
        its own.
    '''
    if null is not UNSET:
        return null
    dt = dtype if dtype is not None else _numpy_dtype(value)
    if dt is None:
        return None
    if dt.kind == 'f':
        base = nan
    elif dt.kind == 'c':
        base = complex(nan, nan)
    elif dt.kind in 'iu':
        base = 0
    elif dt.kind == 'b':
        base = False
    else:
        return None
    # Preserve the units that the caller supplied, if any.
    if is_quant(value):
        return quant(base, value.units)
    return base


def normalize_unit(unit, /):
    '''Normalizes a ``unit`` metadata value.

    Unit algebra is not yet implemented; this records the unit so that it can
    be reported and round-tripped.

    Parameters
    ----------
    unit : object or None
        The value's unit.

    Returns
    -------
    object or None
        The unit.
    '''
    return None if unit is UNSET else unit


def convert_value(value, backend, dtype, detach, /):
    '''Converts a property value to the requested backend and dtype.

    Parameters
    ----------
    value : array-like
        The value to convert.
    backend : str or None
        The resolved backend. ``None`` leaves the value in its own backend and
        converts only if ``dtype`` is specified.
    dtype : numpy.dtype or None
        The dtype to convert to, or ``None`` to keep the natural dtype.
    detach : bool
        Whether to detach a tensor's gradient when converting it to NumPy.

    Returns
    -------
    array-like
        The converted value.
    '''
    if backend == 'numpy':
        return to_array(value, dtype=dtype, detach=detach)
    elif backend == 'torch':
        return to_tensor(value, dtype=dtype)
    elif dtype is None:
        # No conversion is requested; leave the value exactly as it was given.
        return value
    elif _is_tensor(value):
        return to_tensor(value, dtype=dtype)
    return to_array(value, dtype=dtype, detach=detach)


def _is_tensor(value, /):
    '''Determines whether ``value`` is a PyTorch tensor.'''
    t = checktorch()
    return t is not None and isinstance(value, t.Tensor)


def _numpy_dtype(value, /):
    '''Returns a NumPy dtype describing ``value``, or ``None`` if impossible.'''
    dt = getattr(value, 'dtype', None)
    if dt is None:
        try:
            return asarray(value).dtype
        except Exception:
            return None
    if isinstance(dt, npdtype):
        return dt
    # Handle PyTorch dtypes, whose string form is like 'torch.float32'.
    try:
        return npdtype(str(dt).split('.')[-1])
    except TypeError:
        return None


# Property ###################################################################

class Property(planobject):
    '''A value attached to a geometric object, with its interpolation metadata.

    ``Property`` pairs the value array ``value`` with the metadata that governs
    how it is interpolated, extrapolated, and masked. The value's spatial
    dimensions must match ``spatial_shape``; any leading dimensions are "channel"
    dimensions, following the ``(C..., X...)`` ordering convention that
    ``euclib`` uses throughout.

    Fields are normalized and validated by eager filters when the property is
    built, so invalid metadata raises immediately. To obtain a property with
    altered metadata, use ``withmeta``; ``copy`` works as well, and its
    arguments are validated in the same way.

    A ``value`` that is itself a ``Property`` --- or a lazy that will yield one
    --- supplies the *defaults* for every field not given here, and an argument
    that is given always wins. So ``Property(p, shape)`` clones ``p`` with the
    same fields and the same value array, and ``Property(p, shape, interp=0)``
    clones it with a different interpolation. When ``value`` is neither a
    property nor a lazy, the defaults documented for each parameter apply.

    The value, gradient and hessian may be lazy and stay so: they are the fields
    that may be large, on disk, or over a network, and nothing reads them until
    something asks. The metadata is small and is resolved when the property is
    built.

    Parameters
    ----------
    value : array-like
        The property's value. Its trailing dimensions must equal
        ``spatial_shape``.
    spatial_shape : tuple of int
        The shape of the spatial dimensions the property is defined over ---
        the *domain*, not the value. A coordinate property of a mesh with 3
        coordinates has ``(3,)``, and its value may be anything ending in those
        dimensions: a length-3 vector has shape ``(3,)``, and a 2-by-3 value
        has shape ``(2, 3)``, whose leading ``2`` is a channel dimension and
        whose spatial shape is still ``(3,)``.
    backend : str or None, optional
        ``'numpy'``, ``'torch'``, or ``None``. The default, ``None``, leaves
        the value in its own backend; an explicit backend converts it.
    vartype : str or None, optional
        ``'quantitative'`` or ``'qualitative'``. The default, ``None``, infers
        it from the value's dtype: floating-point and complex data are
        quantitative and everything else is qualitative.
    interp : str, int, tuple, None, or Ellipsis, optional
        The interpolation, as a ``(method, order)`` pair. A bare method name
        takes the order from the default for the data's type, a bare order
        takes the method from the default, and ``Ellipsis`` (the default, and
        the same as ``None``) takes both from the default: ``('nearest', 0)``
        for qualitative data and ``('polynomial', 1)`` for quantitative data.
    extrap : None or 0, optional
        How to handle points outside the object. ``None`` (the default) yields
        the null value; ``0`` yields the value at the nearest point on the
        object.
    border : str or None, optional
        How a *grid*'s interpolation continues the data past its own edges,
        where a kernel wider than one cell needs a cell that does not exist:
        ``'constant'``, ``'half-symmetric'`` (the default) or
        ``'whole-symmetric'``. It is ignored by every geometry that is not a
        grid, whose elements supply their own neighbours and have no edge to be
        continued past.
    dtype : dtype-like or None, optional
        The value's dtype. The default, ``None``, keeps the natural dtype.
    mask : array-like or None, optional
        A boolean mask, broadcastable to ``spatial_shape``, marking values that
        are missing. The default, ``None``, marks nothing as missing.
    null : object or Ellipsis, optional
        The value substituted for missing results. The default, ``Ellipsis``,
        selects ``NaN`` for floating-point data and an appropriate zero-like
        value otherwise.
    unit : object or None, optional
        The unit of the value. Unit algebra is not yet implemented.
    detach : bool, optional
        Whether to detach a PyTorch tensor's gradient when converting it to
        NumPy. The default is ``True``.
    gradient : array-like or None, optional
        How the value changes with position, for the interpolations that fit a
        polynomial through more than the values alone. Its shape is
        ``(C..., D, N)``: the value's channel dimensions, one axis of size 2 or
        3, and the value's spatial dimensions. The default, ``None``, means the
        property does not carry one, and a geometry that needs one estimates it
        from the values.
    hessian : array-like or None, optional
        The second derivative, shaped ``(C..., D, D, N)`` in the same way. The
        default, ``None``, means the property does not carry one.

    Attributes
    ----------
    channel_shape : tuple of int
        The shape of the value's leading (non-spatial) dimensions.
    shape : tuple of int
        The full shape of the value, ``channel_shape + spatial_shape``.
    is_quantitative : bool
        Whether the property is quantitative.
    is_masked : bool
        Whether the property has a mask.
    gradient : array-like or None
        The gradient, if the property carries one.
    hessian : array-like or None
        The hessian, if the property carries one.
    '''

    def __init__(self, value, spatial_shape, **kw):
        # A field not given here is taken from `value`, when `value` is a
        # `Property` or a lazy that will yield one; an argument that *is* given
        # always wins. When `value` is neither, the defaults below apply.
        #
        # A lazy that yields a `Property` cannot supply its fields yet --- they
        # are inside a value nobody has read --- so each is deferred to a lazy
        # of its own, which reads the property and then the field.
        source = value if isinstance(value, Property) else None
        deferred = source is None and isinstance(value, lazy)

        def field(name, fallback, /):
            if name in kw:
                return kw[name]
            if source is not None:
                return getattr(source, name)
            if deferred:
                return lazy(_inner_field, value, name, fallback)
            return fallback

        # Every field is assigned exactly as given; the filters below normalize
        # and validate it, and re-run whenever an input changes.
        #
        # The value is stored *as it came*, even when it is a `Property` or a
        # lazy: dereferencing here would compute a value that may be deferred and
        # may never be read. `proc_value` unwraps it, and a filter runs when the
        # value is asked for rather than when the property is built.
        self.value = value
        self.spatial_shape = tuple(spatial_shape)
        self.backend = field('backend', None)
        self.vartype = field('vartype', None)
        self.interp = field('interp', UNSET)
        self.extrap = field('extrap', None)
        self.border = field('border', None)
        self.dtype = field('dtype', None)
        self.mask = field('mask', None)
        self.null = field('null', UNSET)
        self.unit = field('unit', None)
        self.detach = field('detach', True)
        self.gradient = field('gradient', None)
        self.hessian = field('hessian', None)

        # The *form* of the interpolation is checked here, eagerly, because it
        # is a question about the argument alone: that it is a method name, an
        # order, a pair of them, or Ellipsis, and that the method and order are
        # ones euclib defines. It needs neither the value nor the vartype, so
        # deferring it would only postpone an error the caller can see now ---
        # and the caller is the one who made the mistake.
        #
        # A deferred value is skipped: its interp is a lazy of its own, and
        # reading it to check it would force the value this exists to defer.
        # The check runs when the property is built from what it yielded, which
        # is `_build`'s job.
        #
        # The other two questions are elsewhere. What the *value* supports needs
        # the value, so `proc_valid` asks it when a field is read; what a
        # *geometry* honours is not a property's to ask, and
        # `check_property_interp` asks it when the property is attached.
        if not isinstance(self.interp, lazy):
            check_interp_form(self.interp)

    @staticmethod
    def _build(values, spatial_shape, topo, /, gradient=None, hessian=None,
               **meta):
        '''A property built from what a caller supplied, with its checks run.

        This is the deferral a geometry\'s properties filter needs. A `lazy` value
        cannot be a property\'s *field* --- the plan resolves every field on its
        way into the calcs that read it, so no constructor can keep one --- and a
        *container* is not resolved. So the property itself is deferred, stored
        as a `pcollections.lazy` in the mapping, and this is what builds it when
        something reads it.

        A static method rather than a module function, so that the one place a
        property is built from deferred arguments is named by the type it builds.

        Parameters
        ----------
        values, spatial_shape : array-like, tuple of int
            As for the constructor.
        topo : Topology
            The topology of the geometry the property is being attached to,
            which is what its interpolation is checked against.
        gradient, hessian : array-like or None, optional
            As for the constructor.
        **meta
            The rest of the constructor\'s arguments.

        Returns
        -------
        Property
            The property, ready: every check has run, and any failure is raised
            here rather than at the point the value was supplied.
        '''
        # Imported here rather than at module scope: the geometries are built on
        # this module, so it cannot import them back.
        from ._geom import check_property_interp
        # Only what was given: a `None` here means the caller named no
        # derivative, and passing it on would override one the value brought
        # with it --- which is what a property built from another property has.
        derived = {}
        if gradient is not None:
            derived['gradient'] = gradient
        if hessian is not None:
            derived['hessian'] = hessian
        built = Property(values, spatial_shape, **derived, **meta)
        built.valid
        check_property_interp(built, topo)
        return built

    @calc('backend', lazy=False)
    def proc_backend(backend):
        '''Validates the property's backend.

        Returns
        -------
        backend : str or None
            ``'numpy'``, ``'torch'``, or ``None``.
        '''
        return normalize_backend(backend)

    @calc('dtype', lazy=False)
    def proc_dtype(dtype):
        '''Validates the property's dtype.

        Returns
        -------
        dtype : numpy.dtype or None
            The requested dtype, or ``None`` to keep the natural dtype.
        '''
        return normalize_dtype(dtype)

    @calc('value')
    def proc_value(value, backend, dtype, detach):
        '''Converts the property's value to the requested backend and dtype.

        Returns
        -------
        value : array-like
            The converted value.
        '''
        if isinstance(value, Property):
            # A property may be handed in where a value is wanted --- a caller
            # carrying one geometry's property onto another, say. What it holds
            # is the value, so that is what this is.
            value = value.value
        if isinstance(value, lazy):
            # Converted when it is read, not here: a value that is expensive to
            # produce should not be produced merely to be converted. The lazy may
            # yield a `Property` as readily as an array, so the unwrapping is
            # inside what it resolves to rather than before it.
            return lazy(_converted_value, value, backend, dtype, detach)
        return convert_value(value, backend, dtype, detach)

    @calc('gradient')
    def proc_gradient(gradient, value, spatial_shape, backend, dtype, detach):
        '''Validates the property's gradient, if it has one.

        Returns
        -------
        gradient : array-like or None
            The gradient, converted to the property's backend and dtype, or
            ``None`` when the property carries none.
        '''
        if gradient is None:
            return None
        if isinstance(gradient, lazy) or isinstance(value, lazy):
            # Deferred with whatever is deferred: the derivative is normalized
            # against the value's shape, so neither can be read until both are.
            return lazy(_derivative, gradient, value, spatial_shape, backend, dtype,
                        detach, 1, 'gradient')
        return normalize_derivative(
            convert_value(gradient, backend, dtype, detach),
            value, spatial_shape, 1, 'gradient')

    @calc('hessian')
    def proc_hessian(hessian, value, spatial_shape, backend, dtype, detach):
        '''Validates the property's hessian, if it has one.

        Returns
        -------
        hessian : array-like or None
            The hessian, converted to the property's backend and dtype, or
            ``None`` when the property carries none.
        '''
        if hessian is None:
            return None
        if isinstance(hessian, lazy) or isinstance(value, lazy):
            # Deferred with whatever is deferred: the derivative is normalized
            # against the value's shape, so neither can be read until both are.
            return lazy(_derivative, hessian, value, spatial_shape, backend, dtype,
                        detach, 2, 'hessian')
        return normalize_derivative(
            convert_value(hessian, backend, dtype, detach),
            value, spatial_shape, 2, 'hessian')

    @calc('vartype')
    def proc_vartype(vartype, value):
        '''Infers the property's value type when it is unspecified.

        A value given as a `pcollections.lazy` has no dtype yet, so the type is
        inferred when the value is first read --- the lazy's own job --- rather
        than here. Reading it now would compute it merely to attach it.

        Returns
        -------
        vartype : str
            ``'quantitative'`` or ``'qualitative'``, or a lazy that infers it.
        '''
        if isinstance(value, lazy):
            return lazy(normalize_vartype, vartype, value)
        return normalize_vartype(vartype, value)

    @calc('interp', 'interp_specified')
    def proc_interp(interp, vartype):
        '''Normalizes the property's interpolation to a method and an order.

        ``vartype`` arrives already inferred, because the plan orders a filter
        ahead of every calc that consumes the value it filters. A filter's own
        parameter, by contrast, is the raw value the caller supplied, which is
        how ``interp_specified`` can tell an interpolation that was asked for
        from one that was merely defaulted to.

        Returns
        -------
        interp : tuple of (str, int)
            The interpolation method and order. A calculation with a single
            output may return either its value or a one-tuple holding it, so a
            tuple *value* must be wrapped to keep it from being read as a
            sequence of outputs; the dictionary form below says the same thing
            more plainly.
        interp_specified : bool
            Whether the caller supplied an interpolation.
        '''
        if isinstance(vartype, lazy):
            # The type is deferred with the value it comes from, so the
            # interpolation is normalized when both arrive.
            return {'interp': lazy(normalize_interp, interp, vartype),
                    'interp_specified': interp is not UNSET}
        return {'interp': normalize_interp(interp, vartype),
                'interp_specified': interp is not UNSET}

    @calc('interp_method')
    def proc_interp_method(interp):
        '''The property's interpolation method.

        Returns
        -------
        interp_method : str
            The method.
        '''
        return interp[0]

    @calc('interp_order')
    def proc_interp_order(interp):
        '''The property's interpolation order.

        Returns
        -------
        interp_order : int
            The order, 0 through 3.
        '''
        return interp[1]

    @calc('extrap', lazy=False)
    def proc_extrap(extrap):
        '''Validates the property's extrapolation order.

        Returns
        -------
        extrap : None or int
            ``None`` or 0.
        '''
        return normalize_extrap(extrap)

    @calc('prefiltered')
    def proc_prefiltered(value, interp, border, spatial_shape):
        '''The B-spline coefficients of the property's own values.

        A B-spline basis is not interpolating, so a spline's coefficients are
        not its values: they are the values filtered by the inverse of the
        basis's samples, as the method's documentation page derives. That makes
        this a function of the *values*, which is why it belongs on the property
        rather than among a geometry's data --- and why it is lazy, so that only
        a property actually being read with a spline ever pays for it.

        Returns
        -------
        coefficients : array-like
            The coefficients, one margin longer at each end of every grid axis.
        first : tuple of int
            The sample index the first entry along each grid axis belongs to.
        '''
        # Imported here because the types package depends on this one.
        from ..types import _grid
        (method, order) = interp
        if method != 'spline' or order not in _grid.BASES:
            raise ValueError(
                f"a prefilter is built for the spline method at orders"
                f" {sorted(_grid.BASES)}; this property asks for"
                f" {(method, order)!r}")
        return _grid.prefilter(value, spatial_shape, border, order)

    @calc('border', lazy=False)
    def proc_border(border):
        '''Validates the property's boundary extension.

        Returns
        -------
        border : str
            One of ``BORDER_EXTENSIONS``.
        '''
        return normalize_border(border)

    @calc('mask', lazy=False)
    def proc_mask(mask, spatial_shape):
        '''Normalizes the property's mask to a boolean array.

        Returns
        -------
        mask : numpy.ndarray or None
            The mask, or ``None`` when none was given.
        '''
        return normalize_mask(mask, spatial_shape)

    @calc('null')
    def proc_null(null, dtype, value):
        '''Selects the property's null value.

        Returns
        -------
        null : object
            The value substituted for missing results.
        '''
        if isinstance(value, lazy):
            # The null is chosen from the value's dtype, so it waits with it.
            return lazy(normalize_null, null, dtype, value)
        return normalize_null(null, dtype, value)

    @calc('unit', lazy=False)
    def proc_unit(unit):
        '''Records the property's unit.

        Returns
        -------
        unit : object or None
            The unit.
        '''
        return normalize_unit(unit)

    @calc('channel_shape', 'shape')
    def proc_shape(value, spatial_shape):
        '''The channel shape and the full shape of the value.

        Returns
        -------
        channel_shape : tuple of int
            The shape of the value's leading dimensions.
        shape : tuple of int
            The full shape of the value.
        '''
        return _value_shape(value, spatial_shape)
        if n > len(sh) or sh[len(sh) - n:] != tuple(spatial_shape):
            raise ValueError(
                f"value shape {sh} does not end with spatial shape"
                f" {tuple(spatial_shape)}")
        if isinstance(value, lazy):
            # A value that has not been produced has no shape to check, so both
            # the shape and the check wait until it does.
            return lazy(_value_shape, value, spatial_shape)
        return _value_shape(value, spatial_shape)

    @calc('valid')
    def proc_valid(backend, dtype, value, gradient, hessian, vartype, interp,
                   extrap, border, mask, null, unit, shape):
        '''Whether every check this property makes has passed.

        Its parameters *are* the checks: each of those fields is produced by a
        calc that validates it, so calling this runs them all, and whatever any
        of them raises is raised here. Otherwise it is always ``True``.

        It exists so that a caller who needs a property *ready* --- a geometry
        whose filter must know whether a property is usable, say --- can say so
        in one place rather than reading a list of fields and hoping the list is
        complete. Deferred, like the fields it reads, so that a property nobody
        has asked about costs nothing.

        Returns
        -------
        valid : bool
            ``True``, or an exception from whichever check failed.
        '''
        return True

    @calc('is_quantitative')
    def proc_is_quantitative(vartype):
        '''Whether the property's values may be interpolated.

        Returns
        -------
        bool
            ``True`` if the property is quantitative.
        '''
        return vartype == QUANTITATIVE

    @calc('is_masked')
    def proc_is_masked(mask):
        '''Whether the property carries a mask.

        Returns
        -------
        bool
            ``True`` if the property has a mask.
        '''
        return mask is not None

    def copy(self, **kwargs):
        '''Returns a copy of the property with the named fields changed.

        The interpolation is checked here as well as in the constructor, because
        ``copy`` never passes through the constructor: `immlib` builds the copy
        with ``object.__new__`` and swaps the plan underneath it, so a check the
        constructor ran would be skipped by every metadata update. It is the same
        check --- the form of the argument, and whether the method and order are
        ones ``euclib`` defines --- and needs nothing but the argument, which is
        why it can run here rather than at the read the update would otherwise
        wait for.

        A lazy interpolation is skipped: reading it to check it would force the
        value this exists to defer.

        Parameters
        ----------
        **kwargs
            The fields to change, named as in the constructor.

        Returns
        -------
        Property
            The copy.
        '''
        if 'interp' in kwargs and not isinstance(kwargs['interp'], lazy):
            check_interp_form(kwargs['interp'])
        return super().copy(**kwargs)

    def withmeta(self, **kwargs):
        '''Returns a copy of the property with altered metadata.

        Only metadata may be changed; the value and the spatial shape are fixed.

        Parameters
        ----------
        **kwargs
            The metadata fields to change, named as in the constructor.

        Returns
        -------
        Property
            A property with the requested metadata; ``self`` when nothing
            changes.
        '''
        if not kwargs:
            return self
        fields = ('backend', 'vartype', 'interp', 'extrap', 'border', 'dtype',
                  'mask', 'null', 'unit', 'detach')
        updates = {}
        for (k, v) in kwargs.items():
            if k not in fields:
                raise TypeError(f"unknown property metadata field: {k!r}")
            updates[k] = v
        res = self.copy(**updates)
        return self if res == self else res

    def subprop(self, spatial_shape, /):
        '''Returns a copy of the property with a different spatial shape.

        This is used when a property is re-expressed over a different set of
        spatial positions, for example when a coordinate property is restricted
        to a mesh's active vertices.

        Parameters
        ----------
        spatial_shape : tuple of int
            The new spatial shape. The value's trailing dimensions are
            re-interpreted accordingly.

        Returns
        -------
        Property
            The re-shaped property.
        '''
        return Property(
            value=self.value, spatial_shape=spatial_shape, backend=self.backend,
            vartype=self.vartype, interp=self.interp, extrap=self.extrap,
            border=self.border, dtype=self.dtype, mask=self.mask, null=self.null,
            unit=self.unit, detach=self.detach)

    def __getitem__(self, index):
        '''Returns the value indexed along its spatial dimensions.

        The leading channel dimensions are preserved. Consistent with
        ``euclib``'s convention that property lookups return raw values, this
        returns an array-like rather than a ``Property``.
        '''
        ind = index if isinstance(index, tuple) else (index,)
        return self.value[(Ellipsis,) + ind]

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return planobject_eq(self, other)

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __hash__(self):
        return planobject_hash(self)


# Utilities ##################################################################

def is_property(x, /):
    '''Determines whether ``x`` is a ``Property`` object.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is a ``Property``.
    '''
    return isinstance(x, Property)

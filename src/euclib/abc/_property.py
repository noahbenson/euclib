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
INTERP_METHODS = (
    'nearest', 'polynomial', 'clough-tocher', 'powell-sabin', 'catmull-rom',
    'bezier')

#: The interpolation orders that ``euclib`` defines, from 0 (nearest) to 3
#: (cubic).
INTERP_ORDERS = (0, 1, 2, 3)

#: The interpolation that a qualitative property uses, and the only one it may
#: use.
INTERP_QUALITATIVE = ('nearest', 0)

#: The interpolation combinations that are implemented today. A combination
#: outside this set is recognized --- so the caller is told what is missing
#: rather than that they misspelled something --- but raises
#: ``NotImplementedError``. As each remaining method is built, its combinations
#: are added here and begin to work.
INTERP_SUPPORTED = (('nearest', 0), ('polynomial', 1))

#: The valid extrapolation orders. Only 0 (nearest point on the object) is
#: supported; ``None`` means "no extrapolation".
EXTRAP_ORDERS = (None, 0)


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
    if interp is UNSET or interp is None or interp is Ellipsis:
        res = default
    elif isinstance(interp, str):
        # A method name takes the order from the default, except that the only
        # order 'nearest' has is 0.
        res = (interp, 0 if interp == 'nearest' else default[1])
    elif isinstance(interp, int) and not isinstance(interp, bool):
        res = (default[0], interp)
    elif isinstance(interp, tuple) and len(interp) == 2:
        res = (interp[0], interp[1])
    else:
        raise ValueError(
            f"invalid interp: {interp!r}; expected a method name, an order, a"
            f" (method, order) pair, or Ellipsis")
    (method, order) = res
    if method == 'nearest' and order != 0:
        raise ValueError(
            f"'nearest' is only meaningful at order 0; found ('nearest',"
            f" {order})")
    if order == 0:
        # A fit of degree zero is the value itself, so every method's order 0
        # is nearest-neighbour. Canonicalizing keeps 'nearest' the one spelling
        # of "no interpolation", so that `interp=0` means what it says however
        # the property's other metadata reads.
        method = 'nearest'
    res = (method, order)
    if method not in INTERP_METHODS:
        raise ValueError(
            f"unknown interpolation method: {method!r}; expected one of"
            f" {INTERP_METHODS}")
    if not isinstance(order, int) or isinstance(order, bool) \
            or order not in INTERP_ORDERS:
        raise ValueError(
            f"invalid interpolation order: {order!r}; expected one of"
            f" {INTERP_ORDERS}")
    if vartype == QUALITATIVE and res != INTERP_QUALITATIVE:
        raise ValueError(
            f"qualitative properties can only use {INTERP_QUALITATIVE}; found"
            f" {res}")
    if res not in INTERP_SUPPORTED:
        raise NotImplementedError(
            f"the interpolation {res} is not implemented yet; supported today:"
            f" {INTERP_SUPPORTED}")
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
    '''

    def __init__(self, value, spatial_shape, backend=None, vartype=None,
                 interp=UNSET, extrap=None, dtype=None, mask=None, null=UNSET,
                 unit=None, detach=True):
        # Every field is assigned exactly as given; the filters below normalize
        # and validate it, and re-run whenever an input changes.
        self.value = value
        self.spatial_shape = tuple(spatial_shape)
        self.backend = backend
        self.vartype = vartype
        self.interp = interp
        self.extrap = extrap
        self.dtype = dtype
        self.mask = mask
        self.null = null
        self.unit = unit
        self.detach = detach

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

    @calc('value', lazy=False)
    def proc_value(value, backend, dtype, detach):
        '''Converts the property's value to the requested backend and dtype.

        Returns
        -------
        value : array-like
            The converted value.
        '''
        return convert_value(value, backend, dtype, detach)

    @calc('vartype', lazy=False)
    def proc_vartype(vartype, value):
        '''Infers the property's value type when it is unspecified.

        Returns
        -------
        vartype : str
            ``'quantitative'`` or ``'qualitative'``.
        '''
        return normalize_vartype(vartype, value)

    @calc('interp', 'interp_specified', lazy=False)
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

    @calc('mask', lazy=False)
    def proc_mask(mask, spatial_shape):
        '''Normalizes the property's mask to a boolean array.

        Returns
        -------
        mask : numpy.ndarray or None
            The mask, or ``None`` when none was given.
        '''
        return normalize_mask(mask, spatial_shape)

    @calc('null', lazy=False)
    def proc_null(null, dtype, value):
        '''Selects the property's null value.

        Returns
        -------
        null : object
            The value substituted for missing results.
        '''
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

    @calc('channel_shape', 'shape', lazy=False)
    def proc_shape(value, spatial_shape):
        '''The channel shape and the full shape of the value.

        Returns
        -------
        channel_shape : tuple of int
            The shape of the value's leading dimensions.
        shape : tuple of int
            The full shape of the value.
        '''
        sh = tuple(value.shape)
        n = len(spatial_shape)
        if n > len(sh) or sh[len(sh) - n:] != tuple(spatial_shape):
            raise ValueError(
                f"value shape {sh} does not end with spatial shape"
                f" {tuple(spatial_shape)}")
        return (sh[:len(sh) - n], sh)

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
        fields = ('backend', 'vartype', 'interp', 'extrap', 'dtype', 'mask',
                  'null', 'unit', 'detach')
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
            dtype=self.dtype, mask=self.mask, null=self.null, unit=self.unit,
            detach=self.detach)

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

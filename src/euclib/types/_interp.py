# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_interp.py
'''The interpolation engine: reading a property at arbitrary positions.

Every geometric object can already say where a position lies in its own terms
--- that is what ``to_local`` is for --- so interpolation is mostly bookkeeping:
convert the position to local coordinates, find the components whose values bear
on it, combine those values, and decide what to do about the positions for which
no answer exists.

Three pieces of metadata govern the last step, and they are the reason a
property carries metadata at all:

``interp``
    How to combine the values of the components around a position. Order 0
    takes the value of the nearest; order 1 blends them linearly by distance.
    Orders 2 and 3 --- quadratic and cubic --- need more than one value per
    component to be determined at all, and are not yet implemented; see the
    note at the end of this module.
``mask``
    Which components' values are missing. A masked component poisons any blend
    that would have drawn on it, so that a value is never invented from data
    the user marked as absent.
``null``
    What to report when there is no answer: for a position outside the object
    when ``extrap`` is ``None``, and for any result that a mask poisons.

``extrap`` decides the case of a position outside the object. ``None`` reports
the null value; 0 reports the value at the nearest position *on* the object,
which is exactly what the local coordinate already names, so it needs no further
work.
'''

# Dependencies ###############################################################

from __future__ import annotations

from collections.abc import Mapping

from numpy import (
    arange, asarray, concatenate, floor, ones, ravel_multi_index, where,
    zeros)

from ..abc import SimplexGeometry, as_coords, is_loc
from ..abc._property import (
    INTERP_SUPPORTED, UNSET, normalize_interp)
from ._geom import Grid


#: Relative tolerance for deciding that a position lies on a geometry.
TOLERANCE = 1e-9

#: Tolerance, in cells, for a grid position that lies within the grid.
GRID_TOLERANCE = 1e-9


# Positions ##################################################################


def is_local_at(at, topo, /):
    '''Determines whether an ``at`` argument names local coordinates.

    Local coordinates are always supplied unambiguously, and what makes them
    unambiguous is the *names* they are given under. A ``Loc`` is local by
    construction. A mapping is local when its keys are the local coordinate
    type's field names --- ``{'sx': 0, 'sy': 0}`` --- and global when they are
    the coordinate names ``x``, ``y``, and ``z``: a grid's local coordinates
    and its global ones are both ``(D, Q)`` matrices, so only the names can
    tell them apart. A bare array is taken to be global.

    Parameters
    ----------
    at : object
        The ``at`` argument.
    topo : Topology
        The geometry's topology.

    Returns
    -------
    bool
        ``True`` if ``at`` names local coordinates.
    '''
    if is_loc(at):
        return True
    if isinstance(at, Mapping):
        return set(at) == set(topo.Loc._fields)
    return False


def to_query(at, /):
    '''Returns an ``at`` argument as a matrix of global positions.

    A mapping is read as a coordinate tuple, so ``{'x': 0, 'y': 0}`` is the
    position ``(0, 0)``. The keys must be coordinate names, and the conversion
    is ``as_coords``'s, which is also what a geometry's own coordinates are
    given through.

    Parameters
    ----------
    at : array-like or mapping
        The positions.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix of positions.
    '''
    if isinstance(at, Mapping):
        return as_coords(at)
    if not hasattr(at, 'shape'):
        return asarray(at)
    return at.reshape(-1, 1) if at.ndim == 1 else at


def to_loc(geom, at, /):
    '''Expresses an ``at`` argument in a geometry's local coordinates.

    Parameters
    ----------
    geom : Geometry
        The geometry to locate positions within.
    at : object
        Local coordinates, or a ``(D, Q)`` matrix of global positions.

    Returns
    -------
    loc : LocMixin
        The local coordinates.
    outside : numpy.ndarray or None
        A boolean vector marking the positions that fall outside the geometry,
        or ``None`` when ``at`` was given as local coordinates --- in which case
        every position is on the object by construction.
    '''
    if is_local_at(at, geom.topo):
        return (geom.topo.check_loc(at), None)
    query = to_query(at)
    loc = geom.to_local(query)
    return (loc, _outside(geom, query, loc))


def _outside(geom, query, loc, /):
    '''Marks the positions of a query that fall outside a geometry.

    For a simplex geometry the test is whether reconstructing the position from
    its local coordinate returns the position itself: ``to_local`` answers every
    query with the nearest position *on* the object, so a query that was already
    on it is the one that comes back. For a grid the test is whether the index
    coordinate left the grid's extent, since a grid's affine is defined
    everywhere.

    A point cloud is the extreme case of the same rule: its points have no
    interior, so the only positions *on* it are the points themselves, and every
    other position is answered by the null value unless extrapolation was asked
    for.
    '''
    if isinstance(geom, Grid):
        parts = [_flat(getattr(loc, f)) for f in loc._fields]
        bad = zeros(parts[0].shape, dtype=bool)
        for (p, s) in zip(parts, geom.shape):
            # An index names a cell's center, so the grid's region runs from
            # half a step before the first center to half a step past the last.
            bad |= (p < -0.5 - GRID_TOLERANCE) | (
                p > (s - 0.5) + GRID_TOLERANCE)
        return bad
    back = asarray(geom.to_global(loc))
    diff = asarray(query) - back
    d2 = (diff * diff).sum(axis=0)
    scale = float(abs(back).max()) if back.size else 1.0
    tol = TOLERANCE * max(scale, 1.0)
    return d2 > tol * tol


def _flat(x, /):
    '''Returns a value as a 1-dimensional NumPy array.'''
    return asarray(x).reshape(-1)


# Interpolation ##############################################################

def interpolate(geom, prop, at, /, interp=UNSET, extrap=UNSET, null=UNSET,
                mask=UNSET):
    '''Reads a property at a set of positions.

    Parameters
    ----------
    geom : Geometry
        The geometry that the property belongs to.
    prop : Property
        The property to read.
    at : object
        Local coordinates, or a ``(D, Q)`` matrix of global positions.
    interp : str, int, tuple, None, or Ellipsis, optional
        The interpolation, overriding the property's own.
    extrap : None or int, optional
        How to treat positions outside the geometry, overriding the property's
        own.
    null : object, optional
        The value to report when there is no answer, overriding the property's
        own.
    mask : array-like or None, optional
        A mask, overriding the property's own.

    Returns
    -------
    array-like
        The property's values, with the property's channel dimensions and one
        value per position.

    Raises
    ------
    NotImplementedError
        If the interpolation is recognized but not yet implemented; see
        ``euclib.abc.INTERP_SUPPORTED``.
    '''
    (method, order) = prop.interp if interp is UNSET else normalize_interp(
        interp, prop.vartype)
    if getattr(geom, 'order', None) == 0:
        # A point cloud has no interior, so its interpolation carries no
        # meaning: whatever the property asked for, a position can only be
        # answered with the value of the nearest point.
        (method, order) = ('nearest', 0)
    if (method, order) not in INTERP_SUPPORTED:
        raise NotImplementedError(
            f"the interpolation ({method!r}, {order}) is not implemented yet;"
            f" supported today: {INTERP_SUPPORTED}")
    extrap = prop.extrap if extrap is UNSET else extrap
    null = prop.null if null is UNSET else null
    mask = prop.mask if mask is UNSET else mask
    (loc, outside) = to_loc(geom, at)
    if isinstance(geom, Grid):
        (res, drawn) = _interp_grid(geom, prop, loc, order)
        missed = _masked_grid(mask, drawn)
    else:
        (res, corners, drawn) = _interp_simplex(geom, prop, loc, order)
        missed = _masked_corners(mask, corners, drawn)
    # A position outside the object has no answer unless extrapolation was
    # asked for. Extrapolation of order 0 is the value at the nearest position
    # on the object, which is the value already computed, so it needs nothing.
    if outside is not None and extrap is None:
        missed = outside if missed is None else (missed | outside)
    if missed is not None and missed.any():
        res = _substitute_null(res, missed, null)
    return res


def _substitute_null(res, missed, null, /):
    '''Replaces the marked positions of a result with a null value.

    The result's channel dimensions are leading, so the mask is broadcast
    across them.
    '''
    res = asarray(res).copy()
    res[(Ellipsis, missed)] = 0 if null is None else null
    return res


def _interp_simplex(geom, prop, loc, order, /):
    '''Interpolates a simplex geometry's property at local coordinates.

    Returns
    -------
    res : numpy.ndarray
        The values, with the property's channel dimensions and one value per
        position.
    corners : numpy.ndarray
        The ``(K+1, Q)`` matrix of the corners each position draws on.
    drawn : numpy.ndarray
        A ``(K+1, Q)`` boolean array marking the corners whose values the
        result draws on.
    '''
    index = asarray(loc.index)
    corners = geom.topo.indices[:, index]              # (K+1, Q)
    values = asarray(prop.value)[(Ellipsis, corners)]  # (C..., K+1, Q)
    q = index.shape[0]
    cols = arange(q)
    if geom.order == 0:
        # A point has neither interior nor weights, so a local coordinate is
        # just a point index and that point's value is the whole answer.
        return (values[..., 0, :], corners, ones(corners.shape, dtype=bool))
    weight = asarray(loc.weight)
    # A local coordinate stores the first K barycentric weights; the last
    # corner's weight is what they leave of the unit sum.
    full = concatenate([weight, (1.0 - weight.sum(axis=0))[None, :]], axis=0)
    if order == 0:
        best = full.argmax(axis=0)
        res = values[(Ellipsis, best, cols)]
        drawn = zeros(full.shape, dtype=bool)
        drawn[best, cols] = True
    else:
        # full is (K+1, Q), so it broadcasts against the values' trailing two
        # dimensions however many channel dimensions precede them. A corner
        # whose weight is zero must not contribute, or a missing value there
        # would poison the result through 0 * nan.
        drawn = full > 0
        res = where(drawn, values * full, zeros(1)).sum(axis=-2)
    return (res, corners, drawn)


def _interp_grid(geom, prop, loc, order, /):
    '''Interpolates a grid's property at index-space coordinates.

    Returns
    -------
    res : numpy.ndarray
        The values, with the property's channel dimensions and one value per
        position.
    drawn : list of tuple of numpy.ndarray
        The cell indices whose values the result draws on, as one index array
        per axis, one entry per contributing cell.
    '''
    shape = tuple(geom.shape)
    parts = [_flat(getattr(loc, f)) for f in loc._fields]
    values = asarray(prop.value)
    if order == 0:
        idx = tuple(_nearest_cell(p, s) for (p, s) in zip(parts, shape))
        return (values[(Ellipsis,) + idx], [idx])
    return _blend_cells(values, parts, shape)


def _nearest_cell(p, s, /):
    '''The index of the cell nearest an index-space position.'''
    return asarray(floor(p + 0.5)).clip(0, s - 1).astype(int)


def _blend_cells(values, parts, shape, /):
    '''Blends a grid property linearly across the cells around each position.

    Each axis contributes the two cells that straddle the position, weighted by
    how near each is; the result is the sum over the 2**D combinations of cells.
    '''
    d = len(shape)
    q = parts[0].shape[0]
    low = []
    frac = []
    for (p, s) in zip(parts, shape):
        if s < 2:
            # A single-cell axis has nowhere to blend to.
            low.append(zeros(q, dtype=int))
            frac.append(zeros(q))
        else:
            i0 = floor(p).astype(int).clip(0, s - 2)
            low.append(i0)
            frac.append(p - i0)
    res = zeros(values.shape[:values.ndim - d] + (q,))
    drawn = []
    for bits in range(2 ** d):
        weight = zeros(q) + 1.0
        idx = []
        for ax in range(d):
            upper = (bits >> ax) & 1
            idx.append(low[ax] + upper)
            weight = weight * (frac[ax] if upper else (1.0 - frac[ax]))
        # A cell whose weight is zero must not contribute, or a missing value
        # there would poison the result through 0 * nan.
        term = values[(Ellipsis,) + tuple(idx)] * weight
        res = res + where(weight > 0, term, zeros(term.shape))
        drawn.append(tuple(idx))
    return (res, drawn)


def _masked_corners(mask, corners, drawn, /):
    '''Marks the positions of a result that a mask poisons.

    Parameters
    ----------
    mask : array-like or None
        The property's mask, indexed by component.
    corners : numpy.ndarray
        The ``(K+1, Q)`` indices of the components the result could draw on.
    drawn : numpy.ndarray
        A ``(K+1, Q)`` boolean array marking the components it actually draws
        on.

    Returns
    -------
    numpy.ndarray or None
        A length-``Q`` boolean vector, or ``None`` when there is no mask.
    '''
    if mask is None:
        return None
    hit = asarray(mask).reshape(-1)[corners]
    return (hit & drawn).any(axis=0)


def _masked_grid(mask, drawn, /):
    '''Marks the positions of a grid result that a mask poisons.

    Parameters
    ----------
    mask : array-like or None
        The property's mask, with the grid's shape.
    drawn : list of tuple of numpy.ndarray
        The cell indices each result draws on.

    Returns
    -------
    numpy.ndarray or None
        A length-``Q`` boolean vector, or ``None`` when there is no mask.
    '''
    if mask is None:
        return None
    flat = asarray(mask).reshape(-1)
    missed = None
    for idx in drawn:
        at = ravel_multi_index([i.reshape(-1) for i in idx], asarray(mask).shape)
        hit = flat[at]
        missed = hit if missed is None else (missed | hit)
    return missed


# Exports ####################################################################

__all__ = ('interpolate', 'to_loc', 'is_local_at', 'to_query')

#
# Quadratic and cubic interpolation
# --------------------------------
#
# Orders 0 and 1 are determined by one value per component: order 0 takes the
# nearest component's value, and order 1 blends the surrounding components'
# values linearly, which has a unique solution. Orders 2 and 3 do not. A
# quadratic or cubic field within a simplex is not determined by the values at
# its corners --- there are more coefficients than corners --- so a basis has to
# be chosen, and with it whether a Property must be able to carry derivative
# data alongside its values.
#
# The candidates are Clough-Tocher elements for triangles, which subdivide each
# triangle into three and fit cubics that agree in value and gradient across the
# shared edges; tensor-product Bezier patches for tetrahedra; and Catmull-Rom
# splines for paths, where the derivative at each vertex can be estimated from
# its neighbours. This choice was deferred to this release and remains open.

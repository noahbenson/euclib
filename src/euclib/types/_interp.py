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
    takes the value of the nearest; order 1 blends them linearly by distance;
    orders 2 and 3 fit a polynomial of that degree, which needs derivative data
    because an element's values alone do not determine it. Those fits are built
    one element at a time: a segment's are done, and triangles, tetrahedra, and
    the rest follow.
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
from math import factorial

from numpy import (
    arange, asarray, concatenate, einsum, floor, linalg, moveaxis, ones,
    ravel_multi_index, sqrt, where, zeros)

from ..abc import SimplexGeometry, as_coords, is_loc, supported_interp
from ..abc._property import (
    INTERP_SUPPORTED, UNSET, normalize_interp)
from ._geom import Grid


#: Relative tolerance for deciding that a position lies on a geometry.
TOLERANCE = 1e-9

#: Tolerance, in cells, for a grid position that lies within the grid.
GRID_TOLERANCE = 1e-9

#: The relative size below which a direction a stencil spans counts as not
#: spanned at all, when the gradient estimate decides how many dimensions its
#: fit can rely on.
_RANK_TOLERANCE = 1e-9


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
    # A position of the wrong dimension is refused here rather than left to fail
    # inside the search, where a caller cannot tell what went wrong: it reaches
    # the spatial index and comes back as a broadcasting error about shapes.
    if query.ndim != 2 or query.shape[0] != geom.dim:
        raise ValueError(
            f"a position in this geometry has {geom.dim} dimensions, so `at`"
            f" must be a ({geom.dim}, Q) matrix; found shape"
            f" {tuple(query.shape)}")
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
                mask=UNSET, gradient=UNSET, hessian=UNSET):
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
    gradient : array-like or None, optional
        The gradient the fit should use, overriding any the property carries.
        An explicit ``None``, like leaving the argument out, means the property's
        own is not used and one is estimated from the values. Only the fits that
        need derivative data consult it.
    hessian : array-like or None, optional
        The hessian, on the same terms. No method uses one yet --- the cubic
        fit of a tetrahedron will --- so this is accepted and ignored.

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
    supported = supported_interp(geom.topo)
    if (method, order) not in supported:
        raise NotImplementedError(
            f"the interpolation ({method!r}, {order}) is not implemented for"
            f" this geometry; it supports {' and '.join(map(str, supported))}")
    extrap = prop.extrap if extrap is UNSET else extrap
    null = prop.null if null is UNSET else null
    mask = prop.mask if mask is UNSET else mask
    # The fits above linear need derivative data: an element's values alone do
    # not determine a quadratic or a cubic. A caller's gradient overrides the
    # property's, and a property that carries none has one estimated from the
    # values around the geometry.
    fitted = None
    if order >= 2 and geom.order in (1, 2):
        if gradient is not UNSET and gradient is not None:
            fitted = asarray(gradient)
        elif prop.gradient is not None:
            fitted = asarray(prop.gradient)
        else:
            fitted = estimate_gradient(geom, prop, order)
    (loc, outside) = to_loc(geom, at)
    if isinstance(geom, Grid):
        (res, drawn) = _interp_grid(geom, prop, loc, order)
        missed = _masked_grid(mask, drawn)
    else:
        (res, corners, drawn) = _interp_simplex(geom, prop, loc, order, fitted)
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


def _interp_simplex(geom, prop, loc, order, gradient=None, /):
    '''Interpolates a simplex geometry's property at local coordinates.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the property belongs to.
    prop : Property
        The property being read.
    loc : LocMixin
        The local coordinates to read it at.
    order : int
        The order of the interpolation being asked for.
    gradient : array-like or None, optional
        The gradient to fit through, when one is needed and one was found. The
        default, ``None``, means the fit has none to use.

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
    if order >= 2:
        # The higher orders are built one element at a time.
        fit = segment_fit if geom.order == 1 else triangle_fit
        return (fit(geom, loc, values, corners, gradient, order),
                corners, ones(corners.shape, dtype=bool))
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


def segment_fit(geom, loc, values, corners, gradient, order, /):
    '''Fits a polynomial of the given order through one segment's data.

    A segment's polynomial has more coefficients than its two endpoints have
    values, so the values alone cannot determine it: the slopes at the ends are
    what settles the rest. The values are *interpolated* --- the fit passes
    through them, which is what keeps the field continuous where two segments
    meet, as linear interpolation already is --- and the slopes fill in whatever
    freedom is left.

    At order 3 that freedom is exactly filled: a cubic has four coefficients and
    value and slope at each end are four conditions, so the fit is the classical
    cubic Hermite and there is nothing to choose. At order 2 the cubic's four
    conditions over-determine a quadratic's three coefficients, and the
    remaining one is settled by least squares over the two slopes. Writing the
    quadratic as a straight line plus a bump that vanishes at both ends,

        ``p(s) = v0 + (v1 - v0) s + c s (1 - s)``,

    makes the values exact whatever ``c`` is, and the least-squares answer is
    ``c = (a - b) / 2``: the slopes it achieves are the requested ones split
    evenly about the segment's own average slope.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the segment belongs to.
    loc : LocMixin
        The local coordinates: a segment index and the weight of its first
        corner, which is 1 at that corner and 0 at the other.
    values : numpy.ndarray
        A ``(C..., 2, Q)`` array of the two corners' values at each position.
    corners : numpy.ndarray
        The ``(2, Q)`` matrix of the corners each position draws on.
    gradient : array-like
        A ``(C..., D, N)`` gradient over the geometry's coordinates.
    order : int
        ``2`` or ``3``.

    Returns
    -------
    numpy.ndarray
        The fitted values, with the property's channel dimensions and one value
        per position.
    '''
    coords = asarray(geom.coords)
    d = coords.shape[0]
    gradient = asarray(gradient)
    if gradient.shape[-2] != d:
        raise ValueError(
            f"the gradient has {gradient.shape[-2]} dimensions, but this"
            f" geometry occupies {d} of them")
    # The slope each corner asks for is the gradient's component along the
    # segment, scaled by the segment's length, because the fit's parameter runs
    # from 0 to 1 along the segment rather than over its length.
    ends = coords[:, corners]                          # (D, 2, Q)
    step = ends[:, 1, :] - ends[:, 0, :]               # (D, Q)
    length = sqrt((step * step).sum(axis=0))           # (Q,)
    unit = step / where(length > 0, length, 1.0)
    # The gradient at each corner, projected onto the segment's direction.
    slopes = (gradient[(Ellipsis, corners)]            # (C..., D, 2, Q)
              * unit[:, None, :]).sum(axis=-3) * length[None, :]
    s = 1.0 - asarray(loc.weight)[0]                   # (Q,) from corner 0 to 1
    (v0, v1) = (values[..., 0, :], values[..., 1, :])
    (a, b) = (slopes[..., 0, :], slopes[..., 1, :])
    if order == 2:
        return v0 + (v1 - v0) * s + ((a - b) / 2.0) * (s * (1.0 - s))
    (s2, s3) = (s * s, s * s * s)
    return ((2 * s3 - 3 * s2 + 1) * v0 + (s3 - 2 * s2 + s) * a
            + (-2 * s3 + 3 * s2) * v1 + (s3 - s2) * b)


def monomial_exponents(dim, order, /):
    '''Returns the exponent of every monomial in ``dim`` variables to ``order``.

    The exponents are ordered by the size of the first variable's exponent, so
    the constant monomial comes first and the linear ones --- which are the
    gradient --- come next, before the terms of higher degree.

    Parameters
    ----------
    dim : int
        The number of variables, which for a geometry is the number of
        dimensions it occupies.
    order : int
        The greatest total degree to include.

    Returns
    -------
    list of tuple of int
        One exponent tuple per monomial, of length ``dim``, each summing to at
        most ``order``.
    '''
    if dim == 1:
        return [(power,) for power in range(order + 1)]
    res = []
    for power in range(order + 1):
        for tail in monomial_exponents(dim - 1, order - power):
            res.append((power,) + tail)
    return res


def _neighbours(edges, count, /):
    '''Returns each coordinate's neighbours in a geometry's edge matrix.'''
    res = [set() for _ in range(count)]
    for (a, b) in zip(edges[0], edges[1]):
        (a, b) = (int(a), int(b))
        if a != b:
            res[a].add(b)
            res[b].add(a)
    return res


def _stencil(neighbours, node, wanted, /):
    '''Returns the nodes within a growing graph distance of one node.

    The distance grows until the stencil holds ``wanted`` nodes or the geometry
    runs out of them, so that a node with few neighbours is estimated from a
    wider set than its immediate ring rather than from too little data.

    Parameters
    ----------
    neighbours : sequence of set
        One set of neighbours per node.
    node : int
        The node whose stencil is wanted.
    wanted : int
        How many nodes the stencil should hold if the geometry has them.

    Returns
    -------
    list of int
        The stencil's nodes, sorted, so that the fit through them is the same
        every time.
    '''
    seen = {node}
    frontier = [node]
    while len(seen) < wanted and frontier:
        following = []
        for j in frontier:
            for k in neighbours[j]:
                if k not in seen:
                    seen.add(k)
                    following.append(k)
        frontier = following
    return sorted(seen)


def simplex_exponents(order, /):
    '''Returns the multi-indices of a Bezier simplex's control values.

    A polynomial of degree ``order`` on a triangle is a combination of the
    Bernstein basis polynomials, one per multi-index whose entries sum to the
    order. Those with a single non-zero entry sit at the corners, those with two
    along the edges, and the rest --- on a triangle, only ``(1, 1, 1)`` ---
    inside it.

    Parameters
    ----------
    order : int
        The degree.

    Returns
    -------
    list of tuple of int
        One index tuple per control value.
    '''
    res = []
    for i in range(order + 1):
        for j in range(order - i + 1):
            res.append((i, j, order - i - j))
    return res


def triangle_fit(geom, loc, values, corners, gradient, order, /):
    '''Fits a polynomial of the given order through one triangle's data.

    A triangle's polynomial is determined by its values and its corners'
    gradients --- except for one degree of freedom, because a cubic has ten
    coefficients where a triangle's three values and three gradients give nine
    conditions, and a quadratic has six where they give more than enough. The
    construction that settles the rest is the Bezier one, edge by edge:

    * Each edge carries a one-dimensional fit of its own, through the values and
      the slopes at *its* two corners, exactly as a segment's fit is made. Two
      triangles sharing an edge therefore give that edge the same polynomial,
      which is what keeps the field continuous across it, and it is why the
      construction starts with the edges rather than the triangle. Its middle
      control value is the *reflection* of the edge's midpoint value about the
      endpoints' average: a quadratic's middle control is not the value at the
      midpoint, and using that value instead costs the fit its reproduction.
    * A cubic has one control value left inside, and it is the average of the
      three edges' *degree-2* control values --- the reflected midpoints below,
      not the cubic edges' midpoint values. Degree elevation of a quadratic
      gives exactly that average, so the rule is what reproduces quadratics
      exactly, which is the most the corners can determine: a general cubic's
      interior value is not knowable from them, which is why the C1 schemes
      split the triangle instead of fitting it whole.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the triangle belongs to.
    loc : LocMixin
        The local coordinates: a triangle index and the first two barycentric
        weights.
    values : numpy.ndarray
        A ``(C..., 3, Q)`` array of the corners' values at each position.
    corners : numpy.ndarray
        The ``(3, Q)`` matrix of the corners each position draws on.
    gradient : array-like
        A ``(C..., D, N)`` gradient over the geometry's coordinates.
    order : int
        ``2`` or ``3``.

    Returns
    -------
    numpy.ndarray
        The fitted values, with the property's channel dimensions and one value
        per position.
    '''
    coords = asarray(geom.coords)
    dim = coords.shape[0]
    gradient = asarray(gradient)
    if gradient.shape[-2] != dim:
        raise ValueError(
            f"the gradient has {gradient.shape[-2]} dimensions, but this"
            f" geometry occupies {dim} of them")
    powers = simplex_exponents(order)
    exponents = asarray(powers)
    q = corners.shape[1]
    ends = coords[:, corners]                                  # (D, 3, Q)
    # The gradient at each of the corners, which is also how the channels are
    # carried: a scalar property's gradient is (D, N) and a channelled one's is
    # (C..., D, N), and the corner axis is the last either way.
    at = gradient[..., :, corners]                             # (C..., D, 3, Q)
    control = zeros((len(powers),) + tuple(values.shape[:-2]) + (q,))
    # The corners hold their own values.
    for c in range(3):
        corner = tuple(order if i == c else 0 for i in range(3))
        control[powers.index(corner)] = values[..., c, :]
    # Each edge holds the one-dimensional fit of its two ends.
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        step = ends[:, j, :] - ends[:, i, :]                   # (D, Q)
        # The slope each end asks for along the edge is the gradient's component
        # in the edge's direction: the parameter runs from 0 at one corner to 1
        # at the other, so the edge's length is already accounted for.
        at_i = (at[..., :, i, :] * step).sum(axis=-2)          # (C..., Q)
        at_j = (at[..., :, j, :] * step).sum(axis=-2)
        near_i = tuple(order - 1 if c == i else (1 if c == j else 0)
                       for c in range(3))
        if order == 3:
            control[powers.index(near_i)] = values[..., i, :] + at_i / 3.0
            near_j = tuple(1 if c == i else (order - 1 if c == j else 0)
                           for c in range(3))
            control[powers.index(near_j)] = values[..., j, :] - at_j / 3.0
        else:
            # The edge's middle control value: the *reflection* of the midpoint's
            # value about the endpoints' average. A quadratic's Bezier middle
            # control is not the value at the midpoint --- that value is the
            # average of the three controls (b0 + 2 b1 + b2) / 4 --- so the
            # correction term enters at twice its size, not once.
            control[powers.index(near_i)] = (
                (values[..., i, :] + values[..., j, :]) / 2.0
                + (at_i - at_j) / 4.0)
    if order == 3:
        # The one interior control value: the average of the three edges'
        # *degree-2* control values, which is what degree elevation asks for.
        # Taking the average of the cubic edges' midpoint values instead --- the
        # same tempting mistake as above --- is what stops a naive cubic patch
        # reproducing quadratics.
        middle = []
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            near_i = tuple(order - 1 if c == i else (1 if c == j else 0)
                           for c in range(3))
            near_j = tuple(1 if c == i else (order - 1 if c == j else 0)
                           for c in range(3))
            at_i = tuple(order if c == i else 0 for c in range(3))
            at_j = tuple(order if c == j else 0 for c in range(3))
            # The cubic edge's own value at its midpoint ...
            halfway = (control[powers.index(at_i)]
                       + 3.0 * control[powers.index(near_i)]
                       + 3.0 * control[powers.index(near_j)]
                       + control[powers.index(at_j)]) / 8.0
            # ... reflected, as a control value of degree 2 must be.
            middle.append(2.0 * halfway
                          - (values[..., i, :] + values[..., j, :]) / 2.0)
        control[powers.index((1, 1, 1))] = sum(middle) / 3.0
    # Evaluate the Bernstein basis at the barycentric weights. The last weight
    # is what the first two leave of the unit sum.
    weight = asarray(loc.weight)
    full = concatenate([weight, (1.0 - weight.sum(axis=0))[None, :]], axis=0)
    # The Bernstein basis, one value per control point: the product of the
    # barycentric weights taken to the control point's exponents, over the three
    # corners, scaled by the multinomial coefficient. The weights are (Q, 3), so
    # the exponents have to be (1, 3) alongside them for the product to be taken
    # over the corners rather than over the queries.
    basis = (full.T[None, :, :] ** exponents[:, None, :]).prod(axis=-1)
    counts = asarray([factorial(p) for p in exponents.ravel()]
                     ).reshape(exponents.shape).prod(axis=1)
    basis = basis * (factorial(order) / counts)[:, None]
    return einsum('wq,w...q->...q', basis, control)


def estimate_gradient(geom, prop, order, /):
    '''Estimates each coordinate's gradient from the values around it.

    A property need not carry derivative data, and a fit above linear needs it.
    When it does not, it comes from here: a least-squares fit of a polynomial of
    the interpolation's own order through each coordinate's neighbourhood,
    whose linear coefficients are that coordinate's gradient.

    **The estimate reproduces the polynomial.** The neighbourhood is every
    coordinate within a graph distance of the one being estimated, and the
    distance grows until the stencil holds at least as many coordinates as a
    polynomial of that order has coefficients. A value that a polynomial of the
    order could have produced therefore gives back that polynomial's own
    derivative, and the fit that uses it reproduces the polynomial exactly ---
    which is the point: the two ways of getting derivative data, supplied and
    estimated, should agree on the fields that the fit is meant to represent.

    Two limits are worth knowing. A geometry with fewer coordinates within reach
    than the order needs cannot say what the polynomial was, and the estimate is
    then the shortest one that fits what there is --- the same graceful answer as
    a node whose neighbours leave a direction unmeasured, such as the nodes of a
    straight path, which say nothing about the gradient across it. And the
    stencil's size is fixed by the order rather than offered to the caller;
    a future version may expose it.

    A masked value is estimated like any other, and poisons the fit that uses it,
    which is the rule for the engine as a whole.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the property belongs to.
    prop : Property
        The property to estimate a gradient for.
    order : int
        The order of the interpolation that needs the gradient, which is also
        the degree of the polynomial the estimate reproduces.

    Returns
    -------
    numpy.ndarray
        A ``(C..., D, N)`` gradient: the value's channel dimensions, the ``D``
        dimensions of the space, and one gradient per coordinate.
    '''
    coords = asarray(geom.coords)
    values = asarray(prop.value)
    (dim, count) = (coords.shape[0], coords.shape[1])
    powers = monomial_exponents(dim, order)
    neighbours = _neighbours(asarray(geom.topo.simplices[1]), count)
    res = zeros(tuple(values.shape[:-1]) + (dim, count))
    for i in range(count):
        # The stencil grows until the polynomial of the order asked for is
        # *determined* by it. Enough nodes is not enough: a grid's nodes can
        # number more than the monomials of an order and still not span them, so
        # the test is the rank of the fit's design matrix, not its row count.
        stencil = [i]
        settled = False
        while True:
            # A polynomial of the order has at least ``order + 1`` monomials
            # whatever it spans --- one direction gives its degree plus one --- so
            # a stencil smaller than that cannot determine one however it is
            # placed, and neither the frame nor the rank has to be found to know
            # it. Most of the frame's work is skipped this way, since the first
            # stencil that passes this test is usually the one that settles.
            taken = None
            if len(stencil) >= order + 1:
                taken = _stencil_frame(coords, stencil, i)
                (step, frame, room) = taken
                (basis, linear, design) = _monomial_design(step, frame, room,
                                                           order)
                settled = (len(basis) <= len(stencil)
                           and linalg.matrix_rank(design) == len(basis))
                if settled:
                    break
            wider = _stencil(neighbours, i, len(stencil) + 1)
            if len(wider) == len(stencil):
                # The whole neighbourhood is here and the order asked for is
                # still not determined by it; fit the highest degree that is.
                if taken is None:
                    (step, frame, room) = _stencil_frame(coords, stencil, i)
                    (basis, linear, design) = _monomial_design(
                        step, frame, room, order)
                break
            stencil = wider
        degree = order
        while degree > 1 and not settled:
            degree -= 1
            (basis, linear, design) = _monomial_design(step, frame, room, degree)
            settled = (len(basis) <= len(stencil)
                       and linalg.matrix_rank(design) == len(basis))
        # The values of the stencil, one row per node and the channels after, so
        # that one solve answers for every channel at once.
        rhs = moveaxis(values[..., stencil], -1, 0)               # (M, C...)
        coeffs = linalg.lstsq(design, rhs)[0]                     # (W, C...)
        # The gradient the fit found is in the stencil's frame; the caller wants
        # it in the geometry's own axes.
        hull = moveaxis(coeffs[linear], 0, -1) @ frame[:room]      # (C..., D)
        res[..., :, i] = hull
    return res


def _stencil_frame(coords, stencil, node, /):
    '''Returns a stencil's displacements, the directions it spans, and how many.

    A path's nodes lie on a line *whatever* it is placed in --- a diagonal path
    in the plane no less than an axis-aligned one --- so a fit of the line's own
    dimension is determined where a fit of the plane's is not. The frame's rows
    are the directions the stencil spans, and the rest count for nothing, which
    is the same as treating them as flat.
    '''
    step = (coords[:, stencil] - coords[:, node:node + 1]).T      # (M, D)
    (_, sizes, frame) = linalg.svd(step, full_matrices=False)
    room = (sizes > _RANK_TOLERANCE * sizes[0]).sum() if sizes.size else 0
    return (step, frame, max(room, 1))


def _monomial_design(step, frame, room, degree, /):
    '''Returns a polynomial basis and the least-squares design matrix for it.

    The polynomial is written in the stencil's own frame, so the design is over
    the `room` directions the stencil actually spans. The returned linear
    indices pick the gradient's components out of the solved coefficients in the
    axes' own order: the monomials are ordered by their exponents rather than by
    axis, so taking them in order would transpose the components.
    '''
    basis = monomial_exponents(room, degree)
    linear = [basis.index(tuple(1 if b == a else 0 for b in range(room)))
              for a in range(room)]
    exponents = asarray(basis)
    local = step @ frame[:room].T                                 # (M, room)
    design = (local[:, None, :] ** exponents[None, :, :]).prod(axis=-1)
    return (basis, linear, design)


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

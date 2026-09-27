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
    because an element's values alone do not determine it. The *method* says
    how that fit is made --- the monomial basis by least squares, or the Bernstein
    control values that keep two elements agreeing on a shared face --- and the
    fits are built one element at a time: a segment's and a triangle's are done,
    a tetrahedron's with them, and the rest follow.
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
from itertools import combinations, product
from math import factorial

from numpy import (
    arange, argsort, asarray, clip, concatenate, einsum, eye, finfo,
    flatnonzero, floor, linalg, maximum, minimum, moveaxis, ones,
    ravel_multi_index, sqrt, unique, where, zeros)

from ..abc import SimplexGeometry, as_coords, is_loc, supported_interp
from ..abc._property import (
    normalize_border,
    INTERP_SUPPORTED, UNSET, normalize_interp)
from . import _ct, _grid, _ps
from ._geom import Grid, SegPath, TetMesh, TriMesh


#: Relative tolerance for deciding that a position lies on a geometry.
TOLERANCE = 1e-9

#: Tolerance, in cells, for a grid position that lies within the grid.
GRID_TOLERANCE = 1e-9

#: The relative size below which a direction a stencil spans counts as not
#: spanned at all, when the gradient estimate decides how many dimensions its
#: fit can rely on.
_RANK_TOLERANCE = 1e-9

#: How many coordinates the gradient estimate advances together in one block.
#: A block gives a stack of designs of shape ``(B, M, W)``, which is what the
#: estimate's memory is spent on: at three thousand coordinates, a dozen nodes
#: per stencil, and ten monomials, that is a few megabytes.
_ESTIMATE_BLOCK = 3072

#: The rounding of the floating-point type the fits are done in, which is what
#: decides a singular value small enough to be one direction a stencil does not
#: really span. It is the tolerance NumPy's own rank and solve use.
_FLOAT_EPSILON = float(finfo('float64').eps)


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
                mask=UNSET, border=UNSET, gradient=UNSET, hessian=UNSET):
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
    border : str or None, optional
        How a *grid*'s interpolation continues the data past its own edges,
        overriding the property's own: ``'constant'``, ``'half-symmetric'`` or
        ``'whole-symmetric'``. A kernel wider than one cell reaches outside the
        grid near an edge, and this is what it finds there. It is ignored by
        every geometry that is not a grid, whose elements supply their own
        neighbours and have no edge to be continued past.
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
    border = prop.border if border is UNSET else normalize_border(border)
    # The fits above linear need derivative data: an element's values alone do
    # not determine a quadratic or a cubic. A caller's gradient overrides the
    # property's, and a property that carries none has one estimated from the
    # values around the geometry.
    fitted = None
    # A grid has no `order` and needs no gradient: every method it supports is
    # a generalised *value*, so the estimate is not merely unused but
    # meaningless there.
    if order >= 2 and getattr(geom, 'order', None) in (1, 2, 3):
        if gradient is not UNSET and gradient is not None:
            fitted = asarray(gradient)
        elif prop.gradient is not None:
            fitted = asarray(prop.gradient)
        else:
            fitted = estimate_gradient(geom, prop, order)
    (loc, outside) = to_loc(geom, at)
    if isinstance(geom, Grid):
        (res, drawn) = _interp_grid(geom, prop, loc, method, order, border)
        missed = _masked_grid(mask, drawn)
    else:
        (res, corners, drawn) = _interp_simplex(geom, prop, loc, method,
                                               order, fitted)
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


def _element_fit(geom, method, /):
    '''The fit that builds an element's field above linear.

    Which fit it is depends on the method and on the element. The *Bezier*
    method settles the freedom a quadratic or a cubic leaves by a construction
    that is the element's own: a segment's data determines its cubic exactly and
    leaves its quadratic one coefficient short, a triangle's leaves one control
    value, and a tetrahedron's leaves the four on its faces. The *polynomial*
    method fits the monomial basis instead, and that fit makes no use of how the
    corners are arranged, so one fit serves every kind of element.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry whose elements are being fitted.
    method : str
        The interpolation method: ``'bezier'``, ``'polynomial'``, or
        ``'clough-tocher'``.

    Returns
    -------
    callable
        The fit, called as ``fit(geom, loc, values, corners, gradient, order,
        whole=values)``. Every fit takes the last two arguments the same way and
        most of them ignore ``whole``: it is the property's values at *every*
        coordinate, which only a fit that reads the mesh's *edges* --- the
        Clough-Tocher element does --- has any use for, since a triangle's patch
        is built from its own corners and those corners need not be among the
        positions being asked about.

    Raises
    ------
    NotImplementedError
        If no quadratic or cubic fit is built for this method and element.
    '''
    if method == 'polynomial':
        return polynomial_fit
    if method == 'bezier':
        if isinstance(geom, SegPath):
            return segment_fit
        if isinstance(geom, TriMesh):
            return triangle_fit
        if isinstance(geom, TetMesh):
            return tetrahedron_fit
    if method == 'clough-tocher' and isinstance(geom, TriMesh):
        # A triangle is the element this scheme splits; there is no segmented
        # or tetrahedral form of it.
        return clough_tocher_fit
    if method == 'powell-sabin' and isinstance(geom, TriMesh):
        return powell_sabin_fit
    raise NotImplementedError(
        f"no quadratic or cubic fit is built for {method!r} on"
        f" {type(geom).__name__}")


def _interp_simplex(geom, prop, loc, method, order, gradient=None, /):
    '''Interpolates a simplex geometry's property at local coordinates.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the property belongs to.
    prop : Property
        The property being read.
    loc : LocMixin
        The local coordinates to read it at.
    method : str
        The interpolation method, which chooses the fit above linear.
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
        fit = _element_fit(geom, method)
        return (fit(geom, loc, values, corners, gradient, order,
                    whole=prop.value),
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


def segment_fit(geom, loc, values, corners, gradient, order, /, *,
                whole=None):
    '''Fits a Bezier polynomial of the given order through one segment's data.

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


def simplex_exponents(order, parts=3, /):
    '''Returns the multi-indices of a Bezier simplex's control values.

    A polynomial of degree ``order`` on a simplex is a combination of the
    Bernstein basis polynomials, one per multi-index whose entries sum to the
    order. Those with a single non-zero entry sit at the corners, those with two
    along the edges, and the rest further in: on a triangle only ``(1, 1, 1)``,
    and on a tetrahedron the four that have two non-zero entries, one on each
    face. The order in which they come back is by the first entry, then the
    second, and so on.

    Parameters
    ----------
    order : int
        The degree.
    parts : int, optional
        How many corners the simplex has, which is one more than its dimension:
        3 for a triangle, the default, and 4 for a tetrahedron.

    Returns
    -------
    list of tuple of int
        One index tuple per control value.
    '''
    if parts == 1:
        return [(order,)]
    res = []
    for i in range(order + 1):
        for tail in simplex_exponents(order - i, parts - 1):
            res.append((i,) + tail)
    return res


def triangle_fit(geom, loc, values, corners, gradient, order, /, *,
                 whole=None):
    '''Fits a Bezier polynomial of the given order through one triangle's data.

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


def tetrahedron_fit(geom, loc, values, corners, gradient, order, /, *,
                    whole=None):
    '''Fits a Bezier polynomial of the given order through one tetrahedron's
    data.

    A tetrahedron's polynomial is determined by its values and its corners'
    gradients except for four degrees of freedom, because a cubic has twenty
    control values where four values and four gradients give sixteen conditions,
    and a quadratic has ten where they give more than enough. The construction
    that settles the rest is the triangle's, applied to each of the four faces:

    * Each edge carries the one-dimensional fit of its own two ends, exactly as a
      segment's fit is made and as a triangle's edges are. Two tetrahedra
      sharing a face give each of that face's edges the same polynomial, which is
      what keeps the field continuous across it, and it is why the construction
      starts with the edges. Its middle control value is the *reflection* of the
      edge's midpoint value about the endpoints' average: a quadratic's middle
      control is not the value at the midpoint, and using that value instead
      costs the fit its reproduction.
    * A cubic has one control value left on each face, and each is the average of
      that face's three edges' *degree-2* control values --- the reflected
      midpoints, not the cubic edges' midpoint values. Degree elevation of a
      quadratic gives exactly those averages, so the rule is what reproduces
      quadratics exactly, which is the most the corners can determine: a general
      cubic's face values are not knowable from them.

    Nothing settles a value *inside* the tetrahedron, because a cubic has none
    there; the first that does is a quartic's, at ``(1, 1, 1, 1)``.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the tetrahedron belongs to.
    loc : LocMixin
        The local coordinates: a tetrahedron index and the first three
        barycentric weights.
    values : numpy.ndarray
        A ``(C..., 4, Q)`` array of the corners' values at each position.
    corners : numpy.ndarray
        The ``(4, Q)`` matrix of the corners each position draws on.
    gradient : array-like
        A ``(C..., D, N)`` gradient over the geometry's coordinates.
    order : int
        The order of the fit, ``2`` or ``3``.

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
    powers = simplex_exponents(order, 4)
    exponents = asarray(powers)
    q = corners.shape[1]
    ends = coords[:, corners]                                  # (D, 4, Q)
    # The gradient at each of the corners, which is also how the channels are
    # carried: a scalar property's gradient is (D, N) and a channelled one's is
    # (C..., D, N), and the corner axis is the last either way.
    at = gradient[..., :, corners]                             # (C..., D, 4, Q)
    control = zeros((len(powers),) + tuple(values.shape[:-2]) + (q,))
    # The corners hold their own values.
    for c in range(4):
        corner = tuple(order if i == c else 0 for i in range(4))
        control[powers.index(corner)] = values[..., c, :]
    # Each edge holds the one-dimensional fit of its two ends.
    for (i, j) in combinations(range(4), 2):
        step = ends[:, j, :] - ends[:, i, :]                   # (D, Q)
        # The slope each end asks for along the edge is the gradient's component
        # in the edge's direction: the parameter runs from 0 at one corner to 1
        # at the other, so the edge's length is already accounted for.
        at_i = (at[..., :, i, :] * step).sum(axis=-2)          # (C..., Q)
        at_j = (at[..., :, j, :] * step).sum(axis=-2)
        near_i = tuple(order - 1 if c == i else (1 if c == j else 0)
                       for c in range(4))
        if order == 3:
            control[powers.index(near_i)] = values[..., i, :] + at_i / 3.0
            near_j = tuple(1 if c == i else (order - 1 if c == j else 0)
                           for c in range(4))
            control[powers.index(near_j)] = values[..., j, :] - at_j / 3.0
        else:
            control[powers.index(near_i)] = (
                (values[..., i, :] + values[..., j, :]) / 2.0
                + (at_i - at_j) / 4.0)
    if order == 3:
        # One control value per face, each the average of that face's own three
        # edges' *degree-2* control values, which is what degree elevation asks
        # for. It is the triangle's rule, and it uses only the three corners of
        # the face, so the two tetrahedra that share the face agree on it.
        for face in combinations(range(4), 3):
            middle = []
            for (i, j) in ((face[0], face[1]), (face[1], face[2]),
                           (face[2], face[0])):
                near_i = tuple(order - 1 if c == i else (1 if c == j else 0)
                               for c in range(4))
                near_j = tuple(1 if c == i else (order - 1 if c == j else 0)
                               for c in range(4))
                at_i = tuple(order if c == i else 0 for c in range(4))
                at_j = tuple(order if c == j else 0 for c in range(4))
                # The cubic edge's own value at its midpoint ...
                halfway = (control[powers.index(at_i)]
                           + 3.0 * control[powers.index(near_i)]
                           + 3.0 * control[powers.index(near_j)]
                           + control[powers.index(at_j)]) / 8.0
                # ... reflected, as a control value of degree 2 must be.
                middle.append(
                    2.0 * halfway
                    - (values[..., i, :] + values[..., j, :]) / 2.0)
            inside = tuple(1 if c in face else 0 for c in range(4))
            control[powers.index(inside)] = sum(middle) / 3.0
    # Evaluate the Bernstein basis at the barycentric weights. The last weight
    # is what the first three leave of the unit sum.
    weight = asarray(loc.weight)
    full = concatenate([weight, (1.0 - weight.sum(axis=0))[None, :]], axis=0)
    # The Bernstein basis, one value per control point: the product of the
    # barycentric weights taken to the control point's exponents, over the four
    # corners, scaled by the multinomial coefficient. The weights are (Q, 4), so
    # the exponents have to be (1, 4) alongside them for the product to be taken
    # over the corners rather than over the queries.
    basis = (full.T[None, :, :] ** exponents[:, None, :]).prod(axis=-1)
    counts = asarray([factorial(p) for p in exponents.ravel()]
                     ).reshape(exponents.shape).prod(axis=1)
    basis = basis * (factorial(order) / counts)[:, None]
    return einsum('wq,w...q->...q', basis, control)


def _monomials(powers, u, /):
    '''The monomials of a list of exponent tuples at one position.'''
    res = []
    for power in powers:
        value = 1.0
        for (entry, weight) in zip(power, u):
            value *= weight ** entry
        res.append(value)
    return res


def _monomial_gradients(powers, u, axis, /):
    '''The derivatives of the monomials at one position, along one axis.'''
    res = []
    for power in powers:
        if power[axis] == 0:
            res.append(0.0)
            continue
        value = float(power[axis])
        for (i, (entry, weight)) in enumerate(zip(power, u)):
            value *= weight ** (entry - (1 if i == axis else 0))
        res.append(value)
    return res


def _local_design(powers, count, /):
    '''The design matrix of the monomial fit on one element of a kind.

    Each corner contributes one **value** condition and one **gradient**
    condition per local axis, and the columns are the monomials. The corners'
    local coordinates are the same for every element of a kind --- a corner is
    at a unit vector of the local coordinates, and the last corner is at their
    origin, because its weight is the one the others leave --- so the matrix is
    built once and used for every element.

    Parameters
    ----------
    powers : list of tuple of int
        The exponents of the monomials, as ``monomial_exponents`` gives them.
    count : int
        The corners the elements have.

    Returns
    -------
    design : numpy.ndarray
        A ``(R, W)`` matrix of the monomials at each condition.
    conditions : list of tuple
        One entry per row: ``('value', corner)`` or ``('gradient', corner,
        axis)``, which says where the row's right-hand side comes from.
    '''
    k = count - 1
    rows = []
    conditions = []
    for c in range(count):
        # A corner's local coordinates: the unit vector naming it, and the
        # origin for the last, whose weight is what the others leave.
        u = [1.0 if i == c else 0.0 for i in range(k)]
        rows.append(_monomials(powers, u))
        conditions.append(('value', c))
        for axis in range(k):
            rows.append(_monomial_gradients(powers, u, axis))
            conditions.append(('gradient', c, axis))
    return (asarray(rows), conditions)


def polynomial_fit(geom, loc, values, corners, gradient, order, /, *,
                   whole=None):
    '''Fits a polynomial of the given order through elements' data by least
    squares, in the monomial basis.

    Unlike the Bezier fits, this one is the same for every kind of element,
    because it makes no use of how the corners are arranged: it writes a
    polynomial of the given degree in the element's own local coordinates ---
    the barycentric weights a local coordinate stores, which name a position
    within an element uniquely --- and finds its coefficients by least squares
    from the conditions the data supplies.

    Those conditions are the corners' values and the corners' gradients: one
    value and one gradient per corner, in the element's own coordinates, which
    is ``(K+1)(K+1)`` conditions against ``C(order+K, K)`` coefficients. Which
    of the two is the greater decides what the fit can promise, and the two
    cases alternate with the order and the element:

    ===================  ==========  ============  ==========================
    element              conditions  coefficients  what the fit gives back
    ===================  ==========  ============  ==========================
    segment, order 2     4           3             the data's own polynomial
    segment, order 3     4           4             the data's own polynomial
    triangle, order 2    9           6             the data's own polynomial
    triangle, order 3    9           10            the data, and least-norm
    tetrahedron, order 2 16          10            the data's own polynomial
    tetrahedron, order 3 16          20            the data, and least-norm
    ===================  ==========  ============  ==========================

    Where the conditions outnumber the coefficients, the data *is* the
    polynomial: a quadratic's values and gradients over-determine the fit, so
    the least-squares solution is that quadratic, and a field of that degree
    comes back exactly. Where the coefficients outnumber the conditions, the fit
    matches the data --- exactly, since there is room to --- and the freedom left
    over is settled by taking the solution of least norm, which is what a
    least-squares solver returns. That last part is a choice, and it is not the
    choice that reproduces quadratics: a cubic that agrees with a quadratic at
    the corners may differ from it between them, and the least-norm one does.
    The Bezier fits make the other choice, spending those degrees of freedom on
    the degree-elevation rule that recovers a quadratic exactly.

    **And what it does not promise.** The fit is over the data as it stands, so
    where the conditions outnumber the coefficients it does not interpolate the
    corners unless the data is consistent --- a value and a gradient at the same
    corner that a quadratic could not have produced are settled against one
    another rather than both honoured, and the field need not be continuous
    across a shared face. The Bezier fits hold the corners exactly whatever the
    data, which is what makes them continuous, at the cost of using the gradients
    in a fixed way rather than a least-squares one.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry the elements belong to.
    loc : LocMixin
        The local coordinates: an element index and the barycentric weights.
    values : numpy.ndarray
        A ``(C..., K+1, Q)`` array of the corners' values at each position.
    corners : numpy.ndarray
        The ``(K+1, Q)`` matrix of the corners each position draws on.
    gradient : array-like
        A ``(C..., D, N)`` gradient over the geometry's coordinates.
    order : int
        The order of the fit, ``2`` or ``3``.

    Returns
    -------
    numpy.ndarray
        The fitted values, with the property's channel dimensions and one value
        per position.
    '''
    coords = asarray(geom.coords)
    (dim, count) = (coords.shape[0], corners.shape[0])
    k = count - 1
    gradient = asarray(gradient)
    if gradient.shape[-2] != dim:
        raise ValueError(
            f"the gradient has {gradient.shape[-2]} dimensions, but this"
            f" geometry occupies {dim} of them")
    index = asarray(loc.index)
    weight = asarray(loc.weight)
    channels = tuple(values.shape[:-2])
    width = 1
    for c in channels:
        width *= c
    powers = monomial_exponents(k, order)
    exponents = asarray(powers)
    (design, conditions) = _local_design(powers, count)
    inverse = linalg.pinv(design)                          # (W, R)
    # The element each position falls in, and where its data is read from: the
    # corners of an element do not depend on the position, so one position per
    # element is enough to read them all.
    (elements, back) = unique(index, return_inverse=True)
    if elements.size == 0:
        return zeros(channels + (0,))
    (_, first) = unique(back, return_index=True)
    here = geom.topo.indices[:, elements]                  # (K+1, E)
    ends = coords[:, here]                                 # (D, K+1, E)
    data_values = values[(Ellipsis, slice(None), first)]   # (C..., K+1, E)
    data_gradient = gradient[(Ellipsis, slice(None), here)]  # (C..., D, K+1, E)
    # The gradient the data gives, written in the element's own coordinates. A
    # position is ``X_K + J u`` in those coordinates, where ``J`` has the
    # vectors from the last corner to the others as its columns, so the chain
    # rule says ``grad_x = J^T grad_u`` and ``grad_u = J^T grad_x`` the other
    # way. The transpose is the whole of it, and it is also a projection: the
    # element spans K directions however many dimensions the geometry has, and
    # a gradient with a component across that space has no component left after
    # the multiply, which is the only part of it the field could have meant.
    origin = ends[:, k, :]                                 # (D, E)
    jacobian = ends[:, :k, :] - origin[:, None, :]         # (D, K, E)
    local = einsum('dke,...dce->...kce', jacobian, data_gradient)
    # One row of the right-hand side per condition: the corner's value, or its
    # gradient along one of the element's own axes.
    target = zeros((len(conditions),) + (width,) + (elements.size,))
    for (row, condition) in enumerate(conditions):
        if condition[0] == 'value':
            target[row] = data_values[..., condition[1], :].reshape(
                (width, elements.size))
        else:
            target[row] = local[..., condition[2], condition[1], :].reshape(
                (width, elements.size))
    coefficients = einsum('wr,rce->wce', inverse, target)  # (W, C, E)
    # Evaluate at the positions, each one with its own element's polynomial.
    chosen = coefficients[(Ellipsis, back)]                # (W, C, Q)
    basis = ones((len(powers), weight.shape[1]))
    for axis in range(k):
        basis = basis * (weight[axis][None, :]
                         ** exponents[:, axis][:, None])
    res = einsum('wq,wcq->cq', basis, chosen)
    return res.reshape(channels + (weight.shape[1],))


def clough_tocher_fit(geom, loc, values, corners, slopes, order, /, *,
                      whole=None):
    '''Fits a C1 piecewise cubic through one triangle's data.

    The element is the Clough-Tocher one: the triangle is split into three by
    joining its centroid to its corners, and each third carries a cubic Bezier
    patch. Its twelve numbers are the value and the gradient at each corner and
    the derivative across each edge at its midpoint, and they determine a
    piecewise cubic that is C1 across the edges the pieces share. The
    construction is in ``euclib.types._ct``, with the derivation on the method's
    documentation page.

    Three of the twelve numbers are per *edge* and a `Property` carries its data
    per coordinate, so they are estimated from the mesh --- see
    ``_ct.edge_data``, which averages what the triangles sharing an edge say the
    derivative is, and which makes them agree because a cross-derivative along
    an edge is a quadratic whose ends are pinned by the corner gradients.

    This is the *cubic* scheme, so it answers at order 3 and no other: a
    quadratic piecewise patch that is C1 across the same split is Powell-Sabin's
    method, which is a different construction.

    '''
    coords = asarray(geom.coords)
    dim = coords.shape[0]
    # A triangle in space is no different to construct on than one in a plane:
    # its three pieces are coplanar however it sits, its control net is the
    # four-point net the plane's is, and its interior conditions are the plane's.
    # What makes that true is where the twelve numbers *come from*. Two of the
    # three kinds are the corners' values and the slopes at each corner along
    # each of its edges --- and a slope is a derivative in a direction defined by
    # the geometry, so the number is the same one for the triangle in the plane
    # and the triangle in space. The third kind is the derivative across an edge,
    # whose direction is the edge turned a quarter turn *within the triangle's
    # plane*; that direction is a vector in space, the derivative along it is a
    # derivative of the same field, and the turn keeps two triangles sharing an
    # edge agreeing about which way it points. Nothing here ever needs a
    # coordinate component of a gradient, which is the one thing a triangle in
    # space would not have to offer.
    channels = tuple(values.shape[:-2])
    width = 1
    for entry in channels:
        width *= entry
    # An edge's derivative is what the two triangles holding it say, so the
    # estimate is made over the whole mesh and not only where the positions are:
    # a triangle's patch is built from its own three corners, and those corners
    # need not be among the positions asked about. Scattering the gathered
    # values back would leave every other coordinate zero and the estimate would
    # be taken from a field that is mostly nothing, which is what it did before
    # `whole` was passed in.
    if whole is None:
        raise ValueError(
            "the Clough-Tocher fit needs the property's values at every"
            " coordinate, as `whole`; it reads the mesh's edges, which a"
            " triangle's own corners do not determine")
    (operator, edges, rows_of) = geom.interp_data['edge_data']
    stacked = concatenate([whole.reshape((width, coords.shape[1])),
                           slopes.reshape((width, dim * coords.shape[1]))],
                          axis=-1)
    across = stacked @ operator.T                          # (C, E)
    indices = asarray(geom.topo.indices)
    index = asarray(loc.index)
    weight = asarray(loc.weight)
    (elements, back) = unique(index, return_inverse=True)
    (_, first) = unique(back, return_index=True)
    res = zeros(channels + (index.shape[0],))
    for (slot, element) in enumerate(elements):
        here = indices[:, element]
        triangle = coords[:, here]
        # The twelve numbers, in the order the element's rows are built: a
        # value and two gradient components for each corner in turn, and then
        # the derivative across each of its three edges.
        numbers = zeros((width, 12))
        for vertex in range(3):
            numbers[:, 3 * vertex] = values[
                (Ellipsis, vertex, first[slot])].reshape(width)
            # The derivative at this corner *along each of its edges*, which is
            # a direction in the geometry and so is the same number whether the
            # triangle is in a plane or in space. A component of the gradient in
            # a coordinate direction is only this when the edge runs along that
            # axis.
            for (place, other) in enumerate(_ct.neighbours_of(vertex)):
                along = (coords[:, here[other]]
                         - coords[:, here[vertex]]).reshape(dim)
                numbers[:, 3 * vertex + 1 + place] = einsum(
                    '...d,d->...', slopes[(Ellipsis, slice(None), here[vertex])],
                    along).reshape(width)
        for k in range(3):
            numbers[:, 9 + k] = across[:, rows_of[element, k]]
        # The twelve control vectors for this triangle's shape, and the
        # controls they give these numbers.
        controls = numbers @ _ct.basis(triangle).T
        rows = flatnonzero(back == slot)
        (pieces, inside) = _ct.sub_weights(weight[:, rows])
        for k in range(3):
            at = flatnonzero(pieces == k)
            if at.size == 0:
                continue
            got = _ct.evaluate(controls[:, k * 10:(k + 1) * 10], inside[at].T)
            res[(Ellipsis, rows[at])] = got.reshape(channels + (at.size,))
    return res


def powell_sabin_fit(geom, loc, values, corners, slopes, order, /, *,
                     whole=None):
    '''Fits a C1 piecewise quadratic through one triangle's data.

    The element is the Powell-Sabin one: the triangle is split into six by
    joining its incenter to its three corners and to the midpoint of each edge,
    and each sixth carries a quadratic Bezier patch. Its nine numbers are the
    value at each corner and the derivative there along each of the triangle's
    two edges, and they determine a piecewise quadratic that is C1 across the
    edges the pieces share. The construction is in ``euclib.types._ps``, with the
    derivation on the method's documentation page.

    **It needs nothing else.** The Clough-Tocher element's twelve numbers include
    the derivative *across* each edge, which a property does not carry and which
    therefore has to be estimated from the mesh --- which is what its ``whole``
    argument and its per-edge operator are for. This element does not read the
    mesh at all: nine numbers at three corners is exactly the freedom the split
    leaves, so ``whole`` is accepted and ignored, and a triangle's patch is built
    from its own corners alone.

    The split's point on an edge is that edge's *midpoint*. Both triangles
    sharing an edge compute it the same way without consulting each other, which
    is what conformity needs, and it is the only choice available in a mesh
    embedded in three dimensions --- where the segment joining two triangles'
    incenters, the classical recipe for it, need not meet their shared edge at
    all.

    This is the *quadratic* scheme, so it answers at order 2 and no other. The
    cubic scheme that is C1 across a split of the same kind is Clough-Tocher's.
    '''
    coords = asarray(geom.coords)
    dim = coords.shape[0]
    channels = tuple(values.shape[:-2])
    width = 1
    for entry in channels:
        width *= entry
    indices = asarray(geom.topo.indices)
    index = asarray(loc.index)
    weight = asarray(loc.weight)
    (elements, back) = unique(index, return_inverse=True)
    (_, first) = unique(back, return_index=True)
    res = zeros(channels + (index.shape[0],))
    for (slot, element) in enumerate(elements):
        here = indices[:, element]
        # The triangle's shape enters the element in exactly one place --- the
        # incenter --- and the incenter is the side lengths. Everything else is
        # barycentric arithmetic on the corners, which is to say it is the same
        # for every triangle.
        sides = asarray([sqrt(((coords[:, here[two]] - coords[:, here[one]]) ** 2)
                              .sum())
                         for (one, two) in ((1, 2), (2, 0), (0, 1))])
        centre = _ps.centre_weights(sides)
        # The nine numbers, in the order the element reads them: for each corner,
        # its value and the derivative along each of the triangle's two edges
        # from it.
        numbers = zeros((width, 9))
        for vertex in range(3):
            numbers[:, 3 * vertex] = values[
                (Ellipsis, vertex, first[slot])].reshape(width)
            for (place, other) in enumerate(_ps.neighbours_of(vertex)):
                along = (coords[:, here[other]]
                         - coords[:, here[vertex]]).reshape(dim)
                numbers[:, 3 * vertex + 1 + place] = einsum(
                    '...d,d->...', slopes[(Ellipsis, slice(None), here[vertex])],
                    along).reshape(width)
        ordinates = numbers @ _ps.basis(centre).T
        rows = flatnonzero(back == slot)
        (pieces, inside) = _ps.sub_weights(weight[:, rows], centre)
        for k in range(len(_ps.PIECES)):
            at = flatnonzero(pieces == k)
            if at.size == 0:
                continue
            got = _ps.evaluate(ordinates[:, _ps.SLOTS[k]], inside[at].T)
            res[(Ellipsis, rows[at])] = got.reshape(channels + (at.size,))
    return res


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

    **The coordinates advance together.** Each one's stencil starts as itself and
    grows by a graph step at a time, so at every turn they all hold the same
    number of coordinates, and the frames, designs, and solves of a whole turn
    are taken as arrays rather than one at a time. A turn is cut into blocks of
    ``_ESTIMATE_BLOCK`` coordinates, because the designs and the arrays built
    from them are ``(B, M, W)`` and a mesh's worth of those at once is what the
    estimate would be spending its memory on. Measured on a mesh of 133,103
    vertices at order 3, one block of 3,072 coordinates needs about 23 MB beyond
    the mesh, where taking the mesh in one turn needs about 878 MB --- and the
    second grows with the mesh where the first does not. The frames are taken for
    the block, the designs are built by the dimensions each stencil spans --- a
    stencil on a line spans one where the mesh around it spans three --- and each
    group of like ones is solved in one call.

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
    # The estimate is the operator applied to the values, and the operator is
    # the mesh's: it is built when the geometry first asks for it and kept, so
    # this is a sparse matrix product rather than the fit it used to be.
    # `_estimate_gradient` below computes the same thing from the mesh directly
    # and is what the operator is held to.
    count = geom.coords.shape[1]
    return _gradient_from_operator(
        geom.interp_data[f'gradient_{order}'], asarray(prop.value),
        geom.coords.shape[0], count)


def _gradient_from_operator(operator, values, dim, count, /):
    '''One property's estimated gradient, from an operator and its values.

    The channels are flattened because the operator is applied to each of them
    at once: a matrix product with the operator's transpose is one call where a
    product per channel would be as many as there are channels.
    '''
    channels = tuple(values.shape[:-1])
    width = 1
    for entry in channels:
        width *= entry
    applied = values.reshape((width, count)) @ operator.T        # (C, D*N)
    return applied.reshape((width, dim, count)).reshape(
        channels + (dim, count))


def _gradient_blocks(coords, edges, order, /):
    '''Yields the operator that turns a stencil's values into a gradient.

    For each coordinate this yields the coordinates its stencil holds, their
    positions, and the operator that takes the values there to the gradient at
    the coordinate --- a ``(D, M)`` block and a many-at-once ``(B, D, M)`` one,
    since the coordinates are taken in blocks and in runs of equal stencil size.

    **The stencil and the operator are both the mesh's.** Which coordinates
    make up a stencil, how many directions it spans, whether a polynomial of the
    order is determined by it and at what degree --- none of that reads the
    *values*. What is left is linear in them, so everything here is a fixed
    operator on the values, and the same one for every property and every call.
    The estimator below applies it; ``_gradient_operator`` collects it into a
    sparse matrix; both consume this one description of it, so there is no
    second place where the stencils are decided.
    '''
    (dim, count) = (coords.shape[0], coords.shape[1])
    if count == 0:
        return
    neighbours = _neighbours(edges, count)
    stencils = [[i] for i in range(count)]
    pending = arange(count)
    stuck = []
    while pending.size:
        # A stencil grows by whole rings of the neighbourhood, so a turn brings
        # one node some number of others, and the coordinates of one turn do not
        # all hold the same number. They are taken in runs of equal size, and
        # each run in blocks of `_ESTIMATE_BLOCK` of them.
        sizes = asarray([len(stencils[i]) for i in pending])
        order_of = argsort(sizes, kind='stable')
        pending = pending[order_of]
        sizes = sizes[order_of]
        growing = []
        start = 0
        while start < pending.size:
            stop = start
            while stop < pending.size and sizes[stop] == sizes[start]:
                stop += 1
            size = int(sizes[start])
            for here in range(start, stop, _ESTIMATE_BLOCK):
                block = pending[here:min(here + _ESTIMATE_BLOCK, stop)]
                points = asarray([stencils[i] for i in block])    # (B, M)
                steps = (coords[:, points].transpose(1, 2, 0)
                         - coords[:, block].T[:, None, :])        # (B, M, D)
                (_, lengths, frames) = linalg.svd(steps, full_matrices=False)
                # How many directions the stencil spans. The rest of the frame
                # counts for nothing, which is the same as treating the stencil
                # as flat in those directions, and a stencil that spans nothing
                # at all is still given one direction to be fit along.
                rooms = maximum(
                    (lengths > _RANK_TOLERANCE * lengths[:, :1]).sum(axis=1), 1)
                for room in unique(rooms):
                    rows = flatnonzero(rooms == room)
                    (basis, linear, design) = _block_design(
                        steps[rows], frames[rows], int(room), order)
                    if len(basis) > size:
                        # Fewer coordinates than the polynomial has monomials:
                        # no rank of the design can settle this, and these grow.
                        growing.extend(block[rows].tolist())
                        continue
                    # One decomposition of the design answers both questions
                    # asked of it: its rank says whether the stencil determines
                    # the polynomial, and its singular values give the solve.
                    # Asking `matrix_rank` and `pinv` separately decomposes the
                    # same matrix twice, which on a mesh of any size is the
                    # greater part of what the estimate costs.
                    (left, singular, right) = linalg.svd(
                        design, full_matrices=False)
                    cutoff = (singular[:, :1]
                              * max(design.shape[1], design.shape[2])
                              * _FLOAT_EPSILON)
                    keep = singular > cutoff
                    settled = keep.sum(axis=1) == len(basis)
                    if settled.any():
                        chosen = rows[settled]
                        # The design's pseudo-inverse, which is the least-norm
                        # solution of the fit, and then the two indices that
                        # pick the gradient out of the solved coefficients. A
                        # rank-deficient design has singular values dropped by
                        # the same cutoff the rank test used, which is what
                        # `pinv` would do --- `lstsq` takes one design and one
                        # right-hand side rather than a stack of either, and
                        # for a design of full rank the two agree to within the
                        # drivers' rounding.
                        inverse = 1.0 / where(keep[settled], singular[settled],
                                              1.0)
                        local = ((right[settled].transpose(0, 2, 1)
                                  * inverse[:, None, :])
                                 @ left[settled].transpose(0, 2, 1))
                        # The operator: what the gradient is, per unit of each
                        # stencil coordinate's value.
                        yield (block[chosen], points[chosen],
                               einsum('grm,grd->gdm', local[:, linear, :],
                                      frames[chosen][:, :room, :]))
                    growing.extend(block[rows[~settled]].tolist())
            start = stop
        # What is left grows by a ring of the neighbourhood, unless the geometry
        # has no more to give it, in which case it is fit at whatever degree its
        # neighbourhood determines.
        pending = []
        for i in growing:
            wider = _stencil(neighbours, i, len(stencils[i]) + 1)
            if len(wider) == len(stencils[i]):
                stuck.append(i)
            else:
                stencils[i] = wider
                pending.append(i)
        pending = asarray(pending, dtype=int)
    for i in stuck:
        (step, frame, room) = _stencil_frame(coords, stencils[i], i)
        degree = order
        (basis, linear, design) = _monomial_design(step, frame, room, degree)
        while degree > 1 and (len(basis) > len(stencils[i])
                              or linalg.matrix_rank(design) < len(basis)):
            degree -= 1
            (basis, linear, design) = _monomial_design(step, frame, room,
                                                       degree)
        # The operator is the solve applied to a unit right-hand side, which
        # gives the same answer as solving for the values directly --- the
        # least-squares solution is linear in them --- and is what lets this
        # coordinate be handled the same way as the settled ones. The solve is
        # `lstsq` and not the pseudo-inverse the batched path takes, because a
        # stencil that has run out of neighbours is rank-deficient by
        # construction, and this is the path where that matters.
        operator = (frame[:room].T
                    @ linalg.lstsq(design, eye(design.shape[0]),
                                   rcond=None)[0][linear])       # (D, M)
        # One coordinate, in the same shapes the blocks above come in.
        yield (asarray([i]), asarray(stencils[i])[None, :],
               operator[None, :, :])


def _estimate_gradient(coords, edges, values, order, /):
    '''The estimate, from the fields it reads rather than from an object.

    A geometry's ``interp_data`` is a calc, and a calc is given fields and not
    the object they came from, so the work is written here where both can reach
    it.
    '''
    (dim, count) = (coords.shape[0], coords.shape[1])
    channels = tuple(values.shape[:-1])
    res = zeros(channels + (dim, count))
    if count == 0:
        return res
    # The channels are flattened for the solves --- one column of the
    # right-hand side per channel --- and folded back into the answer at the
    # end. A solve with a stack of right-hand sides is one call where a solve
    # per channel would be as many as there are channels.
    width = 1
    for c in channels:
        width *= c
    flat = values.reshape((width, count))
    for (nodes, points, operator) in _gradient_blocks(coords, edges, order):
        rhs = flat[:, points].transpose(1, 2, 0)                 # (B, M, C)
        hull = einsum('bdm,bmc->bcd', operator, rhs)             # (B, C, D)
        for (b, node) in enumerate(nodes):
            res[..., :, node] = hull[b].reshape(channels + (dim,))
    return res


def _gradient_operator(coords, edges, order, /):
    '''The operator that turns a property's values into its estimated gradient.

    A sparse matrix of shape ``(D*N, N)`` whose rows are laid out as a
    gradient is: the first ``N`` of them are the first dimension's derivative at
    each coordinate, then the next ``N`` are the second's, and so on. Applied as
    ``operator @ values`` it gives a vector that reads back as ``(D, N)``; a
    property with channels applies it to each channel, as
    ``values.reshape((C, N)) @ operator.T``.

    **It is the same matrix for every property and every call**, which is the
    point of having it: everything the estimate does that is not linear in the
    values is a function of the mesh --- which coordinates make up a stencil,
    the directions it spans, whether a polynomial of the order is determined by
    it --- and what is left is linear. So an estimate is one sparse matrix
    applied to the values, and the matrix can be found once per mesh rather
    than the estimate being computed once per call. Measured: the estimate is
    linear in the values to 1e-13, and the operator here reproduces it to the
    same.

    The stencils come from ``_gradient_blocks``, which is where they are
    decided; this only collects the blocks it yields.
    '''
    from scipy.sparse import coo_matrix
    (dim, count) = (coords.shape[0], coords.shape[1])
    rows = []
    columns = []
    entries = []
    for (nodes, points, operator) in _gradient_blocks(coords, edges, order):
        width = points.shape[1]
        for (b, node) in enumerate(nodes):
            for axis in range(dim):
                rows.extend([axis * count + int(node)] * width)
                columns.extend(int(x) for x in points[b])
                entries.extend(float(x) for x in operator[b, axis])
    return coo_matrix((entries, (rows, columns)),
                      shape=(dim * count, count)).tocsr()


def _block_design(steps, frames, room, degree, /):
    '''Returns the design matrix of each of several stencils at once.

    This is ``_monomial_design`` for a block of stencils: the polynomial is
    written in each stencil's own frame, so the design is over the directions
    that stencil spans, and the whole block is built with array operations ---
    the monomials are raised over the block at once, and the block axis is kept
    beside the corner axis rather than being one of the powers.

    Parameters
    ----------
    steps : numpy.ndarray
        A ``(B, M, D)`` array of each stencil's displacements from its node.
    frames : numpy.ndarray
        A ``(B, D, D)`` array of the directions each stencil spans.
    room : int
        How many of those directions count.
    degree : int
        The degree of the polynomial.

    Returns
    -------
    basis : list of tuple of int
        The exponents of the monomials, as ``monomial_exponents`` gives them.
    linear : list of int
        Where the gradient's monomials sit among them.
    design : numpy.ndarray
        A ``(B, M, W)`` array of the monomials at each stencil's coordinates.
    '''
    basis = monomial_exponents(room, degree)
    linear = [basis.index(tuple(1 if b == a else 0 for b in range(room)))
              for a in range(room)]
    exponents = asarray(basis)
    local = steps @ frames[:, :room, :].transpose(0, 2, 1)        # (B, M, room)
    design = (local[:, :, None, :] ** exponents[None, None, :, :]).prod(axis=-1)
    return (basis, linear, design)


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


def _interp_grid(geom, prop, loc, method, order, border, /):
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
    if (method, order) not in GRID_KERNELS:
        raise NotImplementedError(
            f"no kernel is built for {method!r} on a grid at order {order}")
    (kernel, start, count) = GRID_KERNELS[(method, order)]
    if method == 'spline':
        # A spline convolves its *coefficients*, not its values, and those come
        # with a margin of the boundary extension at each end of every axis ---
        # so the stencil is shifted into the padded array's indexing, and a
        # position inside the grid never reaches its edge.
        if prop.interp == (method, order):
            # The property's own method, so the coefficients it caches are the
            # ones wanted. A read that asks for a spline the property does not
            # carry cannot use that cache -- it is a filter of the same values
            # at a different degree -- and filters them here instead.
            (coefficients, first) = prop.prefiltered
        else:
            (coefficients, first) = _grid.prefilter(values, shape, border,
                                                    order)
        coefficients = asarray(coefficients)
        padded = tuple(coefficients.shape[-len(shape):])
        shifted = [(p - first[axis]) for (axis, p) in enumerate(parts)]
        return _convolve_cells(coefficients, shifted, padded, kernel,
                               start, count, border)
    return _convolve_cells(values, parts, shape, kernel, start, count, border)


def _nearest_cell(p, s, /):
    '''The index of the cell nearest an index-space position.'''
    return asarray(floor(p + 0.5)).clip(0, s - 1).astype(int)


def _fold(index, size, border, /):
    '''The real cell an index outside the grid stands for, under an extension.

    A kernel reaches past the grid's edge whenever a position is within half a
    cell of one, and what it finds there is a choice. Each choice is a rule that
    carries an index outside ``[0, size)`` back inside, and because the rule is
    the same for every cell of the axis the extension is symmetric in the sense
    that folding the *data* and folding the *kernel* come to the same thing.

    Parameters
    ----------
    index : numpy.ndarray
        Integer cell indices, possibly negative or past the end.
    size : int
        The number of cells along the axis.
    border : str
        One of ``euclib.abc.BORDER_EXTENSIONS``.

    Returns
    -------
    numpy.ndarray
        The indices within ``[0, size)`` whose values stand in for those.
    '''
    index = asarray(index)
    if border == 'constant':
        # The edge value repeats: . . . a a b c d e e . . .
        return clip(index, 0, size - 1)
    if border == 'half-symmetric':
        # The edge value repeats once and then the data reflects about it:
        # . . . b a a b c d e e . . .
        period = 2 * size
        return minimum(index % period, (period - 1 - index) % period)
    # Whole-sample symmetric: the reflection is about the edge value itself, so
    # that value appears once per period rather than twice: . . . b a b c d e d .
    period = 2 * size - 2
    if period == 0:
        # A single-cell axis has nowhere to reflect to.
        return zeros(index.shape, dtype=int)
    return minimum(index % period, (period - index) % period)


def linear_kernel(t, /):
    '''The triangular kernel a linear interpolation convolves with.

    Its value at an integer is one at zero and zero at the others, which is what
    makes the method interpolating, and it is supported on one cell, so a
    position draws on the two samples that straddle it.
    '''
    return maximum(1.0 - abs(asarray(t, dtype='float64')), 0.0)


def cubic_kernel(t, alpha=-0.5, /):
    '''The cubic convolution kernel, two pieces supported on two cells.

    The standard one, on the standard parameter: with ``alpha`` of minus a half
    the kernel's moments up to the second vanish, so the method reproduces
    quadratics and has approximation order 3. It interpolates --- one at zero,
    zero at the other integers --- and is C1 at the knots.

    Parameters
    ----------
    t : array-like
        The distances to evaluate at.
    alpha : float, optional
        The slope of the kernel at a knot. The default is the one that
        reproduces quadratics.

    Returns
    -------
    numpy.ndarray
        The weights.
    '''
    t = abs(asarray(t, dtype='float64'))
    inside = where(t <= 1.0,
                   (alpha + 2.0) * t ** 3 - (alpha + 3.0) * t ** 2 + 1.0, 0.0)
    outside = where((t > 1.0) & (t < 2.0),
                    alpha * t ** 3 - 5.0 * alpha * t ** 2
                    + 8.0 * alpha * t - 4.0 * alpha, 0.0)
    return inside + outside


#: The one-dimensional kernel each grid method convolves with, its half-width in
#: cells, and whether it reproduces the data at the samples. A method that is not
#: here is either order 0, whose rule is not a convolution, or not built.
#: The kernel each grid method convolves with, the lowest *cell offset* its
#: stencil uses, and how many cells follow. The B-splines' stencils are of
#: different widths --- three cells for the quadratic, four for the cubic --- so
#: a symmetric "half-width either side" would not describe them.
GRID_KERNELS = {
    ('polynomial', 1): (linear_kernel, 0, 2),
    ('bezier', 1): (linear_kernel, 0, 2),
    ('catmull-rom', 3): (cubic_kernel, -1, 4),
    # The quadratic basis reaches one and a half cells, so which three of the
    # four candidates a position draws on depends on where in its cell it falls
    # -- near the cell's centre it is the lower three, near an edge the upper
    # three. Giving it four and letting the kernel zero the far one is what
    # makes a single stencil work, and is why this one is wider than the basis
    # alone would suggest.
    ('spline', 2): (_grid.bspline2, -1, 4),
    ('spline', 3): (_grid.bspline3, -1, 4),
}


def _convolve_cells(values, parts, shape, kernel, start, count, border, /):
    '''Blends a grid property with a separable kernel across the cells around
    each position.

    The kernel is one-dimensional and applied along each axis in turn, which is
    what "separable" means: a position's weight for a cell is the product of its
    one-dimensional weights along the axes, and the result is the sum over the
    ``(2*half)**D`` cells of the stencil. A position near an edge straddles cells
    that do not exist, and what stands in for them is the ``border`` extension's
    business --- two stencil offsets may even name the same real cell, which is
    why the terms are added rather than assigned.

    Parameters
    ----------
    values : numpy.ndarray
        The property's values, with its channel dimensions leading.
    parts : sequence of numpy.ndarray
        The index-space position along each axis, as a length-``Q`` vector.
    shape : tuple of int
        The grid's extent.
    kernel : callable
        The one-dimensional kernel, taking distances.
    start : int
        The offset of the stencil's lowest cell from the position's own.
    count : int
        How many cells the stencil has.
    border : str
        One of ``euclib.abc.BORDER_EXTENSIONS``.

    Returns
    -------
    res : numpy.ndarray
        The values, one per position, with the channel dimensions leading.
    drawn : list of tuple of numpy.ndarray
        The cell indices the result draws on, as one index array per axis.
    '''
    d = len(shape)
    q = parts[0].shape[0]
    base = []
    weights = []
    for (p, s) in zip(parts, shape):
        if s < 2:
            # A single-cell axis has nowhere to blend to.
            base.append(zeros(q, dtype=int))
            weights.append(ones((q, count)))
        else:
            # The lowest cell of the stencil, unclipped: a position near an edge
            # has cells past the end, and the extension is what decides which
            # real cells those are.
            low = floor(p).astype(int) + start
            base.append(low)
            offsets = arange(count)
            weights.append(kernel(p[:, None] - (low[:, None] + offsets[None, :])))
    res = zeros(values.shape[:values.ndim - d] + (q,))
    drawn = []
    for combo in product(range(count), repeat=d):
        weight = ones(q)
        idx = []
        for (ax, offset) in enumerate(combo):
            idx.append(_fold(base[ax] + offset, shape[ax], border))
            weight = weight * (weights[ax][:, offset] if shape[ax] >= 2 else 1.0)
        # A cell whose weight is zero must not contribute, or a missing value
        # there would poison the result through 0 * nan. The test is on the
        # weight being *zero* and not on its sign: a cubic kernel's weights are
        # negative on part of its stencil, and that is where its sharpness comes
        # from, so discarding them would quietly turn the method into a blurrier
        # one.
        term = values[(Ellipsis,) + tuple(idx)] * weight
        res = res + where(weight != 0, term, zeros(term.shape))
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
# What is built is the Bezier construction: the control values that make two
# elements agree on a shared face, which is a quadratic or a cubic's worth of
# the corners' data and no more. The schemes that go further --- Clough-Tocher,
# which subdivides a triangle into three so that a cubic can agree in value *and*
# gradient across a shared edge, Powell-Sabin for quadratics, and Catmull-Rom
# splines for paths, where a vertex's derivative comes from its neighbours ---
# are recognized by name and rejected with NotImplementedError, so the API is in
# place for them.

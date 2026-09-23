# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_pycore.py
'''Pure-Python implementations of ``euclib``'s performance-critical kernels.

Every function in this module has a counterpart in the optional ``euclib._c``
extension. The versions here are the definition of correct behavior: the C
implementations are tested for parity against them. When the C extension cannot
be built or loaded, these are what run.

Functions here dispatch between NumPy and PyTorch themselves; they do not rely
on ``immlib``'s ``numapi`` machinery, so that they remain usable as a fallback
in any environment.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import (
    arange, asarray, clip, concatenate, full, inf, isfinite, linalg, maximum,
    nan, ones, sort, sqrt, stack, unique, where, zeros)

from itertools import combinations

from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

from immlib import is_numeric, math as imath

from .._init import checktorch


# Predicates #################################################################

def is_pointdata(x, /, dims=(1, 2, 3), *, shape=None, ndim=(1, 2), dtype=None):
    '''Determines whether ``x`` is a valid array of point or coordinate data.

    A valid coordinate payload is a numeric array-like --- a NumPy array, a
    PyTorch tensor, a SciPy sparse array, or a ``pint`` quantity with one of
    those as its magnitude --- that has a dimensionality in ``ndim`` and whose
    leading ("channel") dimension is one of ``dims``. ``immlib.is_numeric``
    performs the type, dtype, and dimensionality checks.

    Parameters
    ----------
    x : object
        The object to test.
    dims : sequence of int, optional
        The permitted sizes of the leading channel dimension. The default is
        ``(1, 2, 3)``.
    shape : tuple or None, optional
        When provided, the trailing spatial dimensions of ``x`` must match this
        shape exactly. The default, ``None``, permits any trailing shape.
    ndim : int or sequence of int, optional
        The permitted total numbers of dimensions. The default is ``(1, 2)``.
    dtype : dtype-like or None, optional
        When provided, ``x``'s dtype must match this value (or one of its
        values, if a tuple is given). The default, ``None``, permits any
        numeric dtype.

    Returns
    -------
    bool
        ``True`` if ``x`` is valid coordinate data and ``False`` otherwise.
    '''
    if not is_numeric(x, dtype=dtype, ndim=ndim, sparse=False):
        return False
    xshape = tuple(x.shape)
    if len(xshape) < 1 or xshape[0] not in dims:
        return False
    if shape is not None and xshape[1:] != tuple(shape):
        return False
    return True


# Backend Conversion ########################################################

def _as_numpy(x, /):
    '''Returns a NumPy view of an array or tensor, detaching gradients.

    The kernels that use this search for the simplex that contains a position,
    which is a selection: the answer is a simplex index, and the weights within
    the chosen simplex are the only part that varies continuously. Such a
    selection is not differentiable, so detaching a tensor's gradient is
    correct here rather than merely convenient --- and, unlike ``numpy.asarray``
    on a tensor that requires a gradient, it does not fail.
    '''
    if type(x).__module__.split('.')[0] == 'torch':
        return asarray(x.detach().cpu())
    return asarray(x)


# Measures ###################################################################

def simplex_measures(coords, indices):
    '''Computes the measure of each simplex: its length, area, or volume.

    The measure of a simplex is its 0-dimensional extent: the length of a
    segment, the area of a triangle, and the volume of a tetrahedron. Vertices
    have no extent, so their measure is zero.

    Parameters
    ----------
    coords : numpy.ndarray or torch.Tensor
        A ``(D, N)`` matrix of coordinates.
    indices : array-like
        A ``(K+1, M)`` integer matrix of simplex corners.

    Returns
    -------
    measures : array-like
        A length-``M`` vector of measures, in the backend of ``coords``.
    '''
    indices = asarray(indices)
    order = indices.shape[0] - 1
    if order == 0:
        # A vertex is a point; it has no length, area, or volume.
        return zeros(indices.shape[1])
    # The vectors from the simplex's first corner to each of its others.
    origin = coords[:, indices[0]]
    edges = [coords[:, indices[i]] - origin
             for i in range(1, order + 1)]
    if order == 1:
        return imath.sqrt(imath.sum(edges[0] * edges[0], axis=0))
    if order == 2:
        gram = imath.sum(edges[0] * edges[0], axis=0) * imath.sum(
            edges[1] * edges[1], axis=0) - imath.sum(
                edges[0] * edges[1], axis=0) ** 2
        # The Gram determinant is the squared area of the parallelogram; it can
        # dip fractionally below zero for a degenerate triangle.
        return 0.5 * imath.sqrt(_clip_positive(gram, coords))
    if order == 3:
        det = _determinant3(edges[0], edges[1], edges[2])
        return _abs(det) / 6.0
    raise ValueError(f"no measure is defined for simplices of order {order}")


def _clip_positive(x, like, /):
    '''Clamps small negative values to zero, keeping the backend of ``like``.'''
    if type(like).__module__.split('.')[0] == 'torch':
        import torch
        return torch.clamp(x, min=0.0)
    return clip(x, 0.0, None)


def _abs(x, /):
    '''Returns the elementwise absolute value, for any supported backend.'''
    return x.__abs__()


def _determinant3(u, v, w, /):
    '''Returns the determinant of three 3-dimensional column vectors.'''
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


# Point Location #############################################################

def nearest_vertices(coords, query, tree=None):
    '''Finds the nearest coordinate to each query position.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``(D, N)`` matrix of coordinates.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.
    tree : SpatialTree or None, optional
        A spatial index over the coordinates. When one is given, the nearest
        coordinate is found through it; the answer is the same either way. The
        default, ``None``, compares every coordinate.

    Returns
    -------
    index : numpy.ndarray
        A length-``Q`` vector of the column of ``coords`` nearest each query.
    '''
    if tree is not None:
        (index, _) = tree.nearest(_as_numpy(query), k=1)
        return index[0]
    coords = _as_numpy(coords)
    query = _as_numpy(query)
    if coords.ndim != 2 or query.ndim != 2:
        raise ValueError("coords and query must be 2-dimensional matrices")
    if coords.shape[0] != query.shape[0]:
        raise ValueError(
            f"coords and query must share a dimension; found {coords.shape[0]}"
            f" and {query.shape[0]}")
    # (N, Q) squared distances, one column per query position.
    diff = coords[:, :, None] - query[:, None, :]
    d2 = (diff * diff).sum(axis=0)
    return d2.argmin(axis=0)


def project_onto_face(face, query):
    '''Projects query positions onto the affine hull of a set of corners.

    Parameters
    ----------
    face : numpy.ndarray
        A ``(D, S, M)`` array of the corners of ``M`` faces, each with ``S``
        corners.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.

    Returns
    -------
    weight : numpy.ndarray
        A ``(S, M, Q)`` array of barycentric weights within each face. The
        weights of each face and query sum to one, but need not be
        non-negative: a negative weight means the projection lies outside the
        face.
    inside : numpy.ndarray
        A ``(M, Q)`` boolean array that is ``True`` where the projection lies
        within the face, and so where the weights are all non-negative.
    d2 : numpy.ndarray
        A ``(M, Q)`` array of squared distances from each query to its
        projection.
    '''
    face = _as_numpy(face)
    query = _as_numpy(query)
    (d, s, m) = face.shape
    q = query.shape[1]
    if d != query.shape[0]:
        raise ValueError(
            f"the query positions have dimension {query.shape[0]}, but the"
            f" geometry's coordinates have dimension {d}")
    origin = face[:, 0].T[:, :, None]                  # (M, D, 1)
    if s == 1:
        weight = ones((1, m, q))
    else:
        # The edge vectors from each face's first corner to its others.
        edges = face[:, 1:].transpose(2, 0, 1) - origin  # (M, D, S-1)
        rel = query.T[None, :, :].transpose(0, 2, 1) - origin   # (M, D, Q)
        et = edges.transpose(0, 2, 1)                  # (M, S-1, D)
        # Solve the normal equations of the regression of the query onto the
        # face's edges, by pseudo-inverse so that a degenerate face yields a
        # regression rather than an error.
        rest = linalg.pinv(et @ edges) @ (et @ rel)    # (M, S-1, Q)
        # The regression solves for the weights of the face's corners 1 through
        # S-1; corner 0's weight is what remains of the unit sum. The weights
        # come back in corner order, so that a caller can drop the last one and
        # imply it, as local coordinates are stored.
        last = 1.0 - rest.sum(axis=1)                  # (M, Q)
        weight = concatenate(
            [last[None, :, :], rest.transpose(1, 0, 2)], axis=0)  # (S, M, Q)
    near = (face[:, :, :, None] * weight[None, :, :, :]).sum(axis=1)
    diff = query[:, None, :] - near
    d2 = (diff * diff).sum(axis=0)
    # A projection onto a face is only meaningful if it lies within the face.
    inside = (weight >= -_TOLERANCE).all(axis=0)
    return (weight, inside, d2)


#: How far outside a face a projection may fall and still count as inside it.
#: A projection that is fractionally outside is accepted because it is, to
#: within rounding, the same point as the true closest one.
_TOLERANCE = 1e-9


def barycentric_coords(coords, indices, query):
    '''Computes the barycentric coordinates of query positions on simplices.

    The coordinates are obtained by least squares: the query position, relative
    to the simplex's first corner, is regressed onto the simplex's edges. The
    regression is exact when the simplex spans the space the query lives in ---
    the usual case for a triangle mesh in 2-D and a tetrahedral mesh in 3-D ---
    and is the projection of the query onto the simplex's affine hull
    otherwise. A position inside a simplex has barycentric coordinates that are
    all non-negative and sum to one; a position outside has at least one
    negative coordinate.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``(D, N)`` matrix of coordinates.
    indices : array-like
        A ``(K+1, M)`` integer matrix of simplex corners.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.

    Returns
    -------
    weights : numpy.ndarray
        A ``(K+1, M, Q)`` array of barycentric coordinates.
    '''
    coords = _as_numpy(coords)
    indices = asarray(indices)
    if indices.ndim != 2 or indices.shape[0] < 2:
        raise ValueError(
            "indices must be a (K+1, M) matrix with K at least 1; found shape"
            f" {indices.shape}")
    (weight, _, _) = project_onto_face(coords[:, indices], query)
    return weight


def simplex_points(coords, indices, index, weight):
    '''Returns the position within each named simplex at the given weights.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    indices : array-like
        A ``(K+1, M)`` integer matrix of simplex corners.
    index : array-like
        A length-``Q`` vector of simplex indices.
    weight : numpy.ndarray
        A ``(K, Q)`` matrix of the first ``K`` barycentric weights; the last
        corner's weight is their complement.

    Returns
    -------
    numpy.ndarray
        A ``(D, Q)`` matrix of positions.
    '''
    coords = _as_numpy(coords)
    corners = coords[:, asarray(indices)[:, asarray(index)]]   # (D, K+1, Q)
    count = weight.shape[0]
    if count == 0:
        return corners[:, 0]
    last = 1.0 - weight.sum(axis=0)
    res = corners[:, 0] * weight[0]
    for j in range(1, count):
        res = res + corners[:, j] * weight[j]
    return res + corners[:, count] * last


def _closest_simplex_indexed(coords, indices, query, tree):
    '''The nearest simplex to each query, by way of a spatial index.

    The index answers with the simplices whose bounding spheres come within a
    radius of a position, which is conservative in the safe direction: the
    simplex that is truly nearest is among them whenever the radius reaches it.
    So each position is searched with a radius that grows until the answer
    found is nearer than the radius, at which point no unexamined simplex could
    be nearer and the answer is the true one.
    '''
    coords = _as_numpy(coords)
    indices = asarray(indices)
    query = _as_numpy(query)
    count = indices.shape[0] - 1
    total = query.shape[1]
    out_index = zeros(total, dtype=int)
    out_weight = zeros((count, total))
    for i in range(total):
        point = query[:, i:i + 1]
        # The nearest sphere is a lower bound on the nearest position, so it
        # makes a good radius to start from.
        radius = max(float(tree.nearest(point, k=1)[1][0, 0]), _TOLERANCE)
        selected = None
        for _ in range(40):
            nearby = tree.candidates(point, radius)[0]
            if nearby.size == 0:
                radius *= 2.0
                continue
            (idx, weight) = _closest_simplex_brute(coords, indices[:, nearby],
                                                   point)
            found = simplex_points(coords, indices[:, nearby], idx, weight)
            away = float(sqrt(((point[:, 0] - found[:, 0]) ** 2).sum()))
            selected = (nearby, idx, weight)
            if away <= radius:
                break
            radius = max(away, radius * 2.0)
        if selected is None:
            # The index found nothing at any radius, which can only happen for
            # a degenerate geometry; the whole search answers it.
            (idx, weight) = _closest_simplex_brute(coords, indices, point)
            out_index[i] = idx[0]
            out_weight[:, i] = weight[:, 0]
        else:
            (nearby, idx, weight) = selected
            out_index[i] = nearby[idx[0]]
            out_weight[:, i] = weight[:, 0]
    return (out_index, out_weight)


def closest_simplex(coords, indices, query, tree=None):
    '''Finds the nearest simplex to each query position, and the position
    within it.

    The closest point on a simplex is not, in general, the projection of the
    query onto the simplex's affine hull with its negative barycentric weights
    clamped: for a query beside a triangle's edge, that yields the wrong point
    on the edge. The closest point lies in one of the simplex's *faces* --- one
    of its vertices, edges, or (for a tetrahedron) triangular sides --- and the
    search is therefore over the faces, projecting onto each face in turn and
    keeping the closest projection that actually lies within its face.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``(D, N)`` matrix of coordinates.
    indices : array-like
        A ``(K+1, M)`` integer matrix of simplex corners.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.
    tree : SpatialTree or None, optional
        A spatial index over the simplices. When one is given, the search
        examines only the simplices the index reports as near, which is what
        makes it pay for a mesh of any size; the answer is the same either way.
        The default, ``None``, examines every simplex.

    Returns
    -------
    index : numpy.ndarray
        A length-``Q`` vector of the simplex containing or nearest each query.
    weight : numpy.ndarray
        A ``(K, Q)`` matrix of the first ``K`` barycentric weights within the
        chosen simplex; the final weight is their complement.
    '''
    if tree is not None:
        return _closest_simplex_indexed(coords, indices, query, tree)
    return _closest_simplex_brute(coords, indices, query)


def _closest_simplex_brute(coords, indices, query):
    '''The nearest simplex to each query, by examining every simplex.'''
    coords = _as_numpy(coords)
    query = _as_numpy(query)
    indices = asarray(indices)
    if indices.ndim != 2 or indices.shape[0] < 2:
        raise ValueError(
            "indices must be a (K+1, M) matrix with K at least 1; found shape"
            f" {indices.shape}")
    corners = coords[:, indices]                       # (D, K+1, M)
    k1 = indices.shape[0]
    m = indices.shape[1]
    q = query.shape[1]
    best_d2 = full((m, q), inf)
    best_w = zeros((k1, m, q))
    for s in range(1, k1 + 1):
        for mem in combinations(range(k1), s):
            (weight, inside, d2) = project_onto_face(
                corners[:, list(mem)], query)
            better = inside & (d2 < best_d2)
            if not better.any():
                continue
            cand = zeros((k1, m, q))
            for (j, corner) in enumerate(mem):
                cand[corner] = weight[j]
            best_d2 = where(better, d2, best_d2)
            best_w = where(better[None, :, :], cand, best_w)
    best = best_d2.argmin(axis=0)
    cols = arange(q)
    return (best, best_w[:, best, cols][:-1])


# Intersections ##############################################################

#: The relative size below which a quantity counts as zero.
_EPSILON = 1e-12

#: How far outside a plane, as a fraction of the sizes involved, a point may
#: fall and still count as lying on it. Solving for a corner and then asking
#: which side of each plane it is on does not answer the same way twice when
#: the corner lies on that plane, which it usually does; this is the width of
#: the band in which the answer is taken to be "on it".
_FEASIBILITY_EPSILON = 1e-12


def cross3(a, b, /):
    '''Returns the cross product of two sets of three-dimensional vectors.

    Parameters
    ----------
    a, b : numpy.ndarray
        ``(3, M)`` matrices of vectors.

    Returns
    -------
    numpy.ndarray
        A ``(3, M)`` matrix of cross products.
    '''
    return concatenate([
        (a[1] * b[2] - a[2] * b[1])[None, :],
        (a[2] * b[0] - a[0] * b[2])[None, :],
        (a[0] * b[1] - a[1] * b[0])[None, :]], axis=0)


def closest_segment_params(a0, a1, b0, b1):
    '''Returns the parameters of the closest points of two sets of segments.

    Each segment pair is answered with the position along each segment of the
    pair of points, one on each, that are nearest one another. Two segments
    that cross have a distance of zero, and the parameters say where they
    cross; two that do not have the parameters of their closest approach.

    Parameters
    ----------
    a0, a1 : numpy.ndarray
        The endpoints of the first segments, each ``(D, Q)``.
    b0, b1 : numpy.ndarray
        The endpoints of the second segments, each ``(D, Q)``.

    Returns
    -------
    s : numpy.ndarray
        A length-``Q`` vector of positions along the first segments, from 0 at
        ``a0`` to 1 at ``a1``.
    t : numpy.ndarray
        A length-``Q`` vector of positions along the second segments.
    '''
    a0 = _as_numpy(a0)
    a1 = _as_numpy(a1)
    b0 = _as_numpy(b0)
    b1 = _as_numpy(b1)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    aa = (u * u).sum(axis=0)
    bb = (u * v).sum(axis=0)
    cc = (v * v).sum(axis=0)
    dd = (u * w).sum(axis=0)
    ee = (v * w).sum(axis=0)
    denom = aa * cc - bb * bb
    # Parallel segments leave the system singular; any position on the first
    # segment will do, and the clamp below settles the second.
    parallel = denom <= _EPSILON * maximum(aa * cc, _EPSILON)
    safe = where(parallel, 1.0, denom)
    s = clip(where(parallel, 0.0, (bb * ee - cc * dd) / safe), 0.0, 1.0)
    # The position along the second segment that suits that point, clamped; and
    # where the clamp bit, the position along the first must be retaken.
    degenerate = cc <= _EPSILON
    t = where(degenerate, 0.0, (bb * s + ee) / where(degenerate, 1.0, cc))
    low = t <= 0.0
    high = t >= 1.0
    t = clip(t, 0.0, 1.0)
    on_a = where(aa <= _EPSILON, 1.0, aa)
    s = where(low, clip(-dd / on_a, 0.0, 1.0),
              where(high, clip((bb - dd) / on_a, 0.0, 1.0), s))
    return (s, t)


def segments_intersect(a0, a1, b0, b1, tolerance=0.0):
    '''Finds where each pair of segments meets.

    Parameters
    ----------
    a0, a1 : numpy.ndarray
        The endpoints of the first segments, each ``(D, Q)``.
    b0, b1 : numpy.ndarray
        The endpoints of the second segments, each ``(D, Q)``.
    tolerance : float, optional
        How near the two segments must come to count as meeting. The default,
        ``0``, requires an exact crossing, which is rarely true of floating
        point arithmetic; callers usually pass a fraction of the geometry's
        size.

    Returns
    -------
    hit : numpy.ndarray
        A length-``Q`` boolean vector.
    point : numpy.ndarray
        A ``(D, Q)`` matrix of meeting points, meaningful where ``hit``.
    s, t : numpy.ndarray
        The positions along each segment, as ``closest_segment_params`` gives.
    '''
    (s, t) = closest_segment_params(a0, a1, b0, b1)
    near = a0 + (a1 - a0) * s
    far = b0 + (b1 - b0) * t
    gap = near - far
    apart = sqrt((gap * gap).sum(axis=0))
    return (apart <= tolerance, (near + far) / 2.0, s, t)


def barycentric_in_triangle(point, a, b, c):
    '''Returns the barycentric coordinates of positions within a triangle.

    The positions are projected onto the triangle's plane, so the coordinates
    describe where the projection lands; they are all non-negative when the
    projection is inside the triangle.

    Parameters
    ----------
    point : numpy.ndarray
        A ``(D, Q)`` matrix of positions.
    a, b, c : numpy.ndarray
        The triangle's corners, each ``(D, Q)``.

    Returns
    -------
    weights : numpy.ndarray
        A ``(3, Q)`` matrix of barycentric weights, summing to one.
    '''
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = (v0 * v0).sum(axis=0)
    d01 = (v0 * v1).sum(axis=0)
    d11 = (v1 * v1).sum(axis=0)
    d20 = (v2 * v0).sum(axis=0)
    d21 = (v2 * v1).sum(axis=0)
    denom = d00 * d11 - d01 * d01
    flat = denom <= _EPSILON * maximum(d00 * d11, _EPSILON)
    safe = where(flat, 1.0, denom)
    # A degenerate triangle has no plane to speak of; its first corner answers.
    weight_b = where(flat, 0.0, (d11 * d20 - d01 * d21) / safe)
    weight_c = where(flat, 0.0, (d00 * d21 - d01 * d20) / safe)
    return stack([1.0 - weight_b - weight_c, weight_b, weight_c], axis=0)


def segments_triangles_intersect(p0, p1, v0, v1, v2, tolerance=0.0):
    '''Finds where each segment crosses each triangle.

    The segment is followed to the plane the triangle lies in, and the crossing
    is kept only if it falls within the segment and within the triangle. A
    segment parallel to the triangle's plane does not cross it.

    Parameters
    ----------
    p0, p1 : numpy.ndarray
        The endpoints of the segments, each ``(3, Q)``.
    v0, v1, v2 : numpy.ndarray
        The triangles' corners, each ``(3, Q)``.
    tolerance : float, optional
        How far outside the triangle a crossing may fall and still count, for
        the same reason ``segments_intersect`` takes one. The default is ``0``.

    Returns
    -------
    hit : numpy.ndarray
        A length-``Q`` boolean vector.
    point : numpy.ndarray
        A ``(3, Q)`` matrix of crossing points, meaningful where ``hit``.
    weight : numpy.ndarray
        A ``(3, Q)`` matrix of the barycentric weights of each crossing within
        its triangle.
    '''
    p0 = _as_numpy(p0)
    p1 = _as_numpy(p1)
    direction = p1 - p0
    normal = cross3(v1 - v0, v2 - v0)
    denom = (normal * direction).sum(axis=0)
    # Parallel segments do not cross the plane at a point.
    parallel = abs(denom) <= _EPSILON * maximum(
        sqrt((normal * normal).sum(axis=0)) * sqrt((direction * direction).sum(axis=0)),
        _EPSILON)
    reach = (normal * (v0 - p0)).sum(axis=0) / where(parallel, 1.0, denom)
    point = p0 + direction * reach
    weight = barycentric_in_triangle(point, v0, v1, v2)
    within = ((weight >= -tolerance).all(axis=0)
              & (reach >= -tolerance) & (reach <= 1.0 + tolerance)
              & ~parallel)
    return (within, point, weight)


def triangles_segments_intersect(v0, v1, v2, w0, w1, w2, tolerance=0.0):
    '''Finds the segment where each pair of triangles meets.

    Two triangles that meet meet along a segment, not over an area: each lies
    in its own plane, and the two planes meet in a line. The segment's ends are
    therefore points where an edge of one triangle crosses the other, so the
    six edge crossings --- three from each triangle --- contain the ends, and
    the pair of them that are furthest apart *is* the segment.

    This is what makes the computation cheap: it is six applications of the
    segment-against-triangle test, which already knows how to find a crossing
    and whether it lands inside, and a comparison among the results.

    Two triangles in the same plane are not handled: they meet over an area
    rather than along a segment, and no edge of either crosses the other.

    Parameters
    ----------
    v0, v1, v2 : numpy.ndarray
        The corners of the first triangles, each ``(3, Q)``.
    w0, w1, w2 : numpy.ndarray
        The corners of the second triangles, each ``(3, Q)``.
    tolerance : float, optional
        How near an edge must come to the other triangle to count as crossing
        it. The default is ``0``.

    Returns
    -------
    hit : numpy.ndarray
        A length-``Q`` boolean vector.
    start : numpy.ndarray
        A ``(3, Q)`` matrix of one end of each segment, meaningful where
        ``hit``.
    stop : numpy.ndarray
        A ``(3, Q)`` matrix of the other end. It is equal to ``start`` when
        the triangles meet at a single point.
    '''
    corners = [_as_numpy(x) for x in (v0, v1, v2, w0, w1, w2)]
    (v0, v1, v2, w0, w1, w2) = corners
    crossings = []
    for (p, q) in ((v0, v1), (v1, v2), (v2, v0)):
        (found, point, _) = segments_triangles_intersect(
            p, q, w0, w1, w2, tolerance=tolerance)
        crossings.append(where(found[None, :], point, nan))
    for (p, q) in ((w0, w1), (w1, w2), (w2, w0)):
        (found, point, _) = segments_triangles_intersect(
            p, q, v0, v1, v2, tolerance=tolerance)
        crossings.append(where(found[None, :], point, nan))
    points = stack(crossings, axis=1)               # (3, 6, Q)
    count = points.shape[1]
    spread = ((points[:, :, None, :] - points[:, None, :, :]) ** 2).sum(axis=0)
    # A crossing that never happened is not a candidate, and neither is a pair
    # of them; marking those as ``-1`` leaves the furthest real pair to win.
    spread = where(isfinite(spread), spread, -1.0)
    best = spread.reshape(count * count, points.shape[2]).argmax(axis=0)
    columns = arange(points.shape[2])
    (first, second) = (best // count, best % count)
    start = points[:, first, columns]
    stop = points[:, second, columns]
    hit = isfinite(start).all(axis=0)
    return (hit, start, stop)


# Tetrahedron and box ########################################################

def _half_spaces(tet, bounds):
    '''The half-spaces that a tetrahedron and a box occupy.

    Each is returned as a normal and an offset, with the interior on the side
    where ``normal . x <= offset``. The tetrahedron contributes the four planes
    of its faces, each oriented so that the opposite corner is inside; the box
    contributes the six planes of its sides.

    Parameters
    ----------
    tet : numpy.ndarray
        A ``(3, 4)`` matrix of the tetrahedron's corners.
    bounds : numpy.ndarray
        A ``(3, 2)`` box.

    Returns
    -------
    normals : numpy.ndarray
        A ``(3, 10)`` matrix of plane normals.
    offsets : numpy.ndarray
        A length-10 vector of plane offsets.
    '''
    normals = []
    offsets = []
    for i in range(4):
        others = [tet[:, j] for j in range(4) if j != i]
        # cross3 takes matrices of vectors, so the corners are given columns;
        # the result is taken back to a plain vector, because an offset is the
        # inner product of two vectors and not the sum of their outer one.
        normal = cross3((others[1] - others[0])[:, None],
                        (others[2] - others[0])[:, None])[:, 0]
        # Orient the plane so that the corner it omits is on the inside.
        if float((normal * (tet[:, i] - others[0])).sum()) > 0:
            normal = -normal
        normals.append(normal)
        offsets.append(float((normal * others[0]).sum()))
    for axis in range(3):
        unit = zeros(3)
        unit[axis] = 1.0
        normals.append(unit)
        offsets.append(bounds[axis, 1])
        normals.append(-unit)
        offsets.append(-bounds[axis, 0])
    return (stack(normals, axis=1), asarray(offsets))


def tetrahedron_box_vertices(tet, bounds, tolerance=0.0):
    '''Finds the corners of the region a tetrahedron and a box share.

    The region is the part of space that lies inside both, so its corners are
    the points where three of the ten bounding planes meet and no plane
    excludes them. That makes the search a matter of trying every triple, which
    is bounded and small: at most 120 points to test, of which the real corners
    are the ones that survive every constraint.

    Parameters
    ----------
    tet : numpy.ndarray
        A ``(3, 4)`` matrix of the tetrahedron's corners.
    bounds : numpy.ndarray
        A ``(3, 2)`` box, such as one voxel of a grid.
    tolerance : float, optional
        How far outside a plane a corner may lie and still be counted, for the
        same reason every other test here takes one. The default is ``0``.

    Returns
    -------
    numpy.ndarray
        A ``(3, V)`` matrix of the region's corners, in no particular order.
        It is empty when the two do not meet.
    '''
    tet = _as_numpy(tet)
    bounds = _as_numpy(bounds)
    (normals, offsets) = _half_spaces(tet, bounds)
    found = []
    for (i, j, k) in combinations(range(normals.shape[1]), 3):
        matrix = stack([normals[:, i], normals[:, j], normals[:, k]])
        if abs(linalg.det(matrix)) <= _EPSILON:
            continue
        point = linalg.solve(matrix, offsets[[i, j, k]])
        # A corner usually lies exactly on some of the planes it is not built
        # from, and whether a solved copy of it falls just inside or just
        # outside one of them is decided by rounding. A slack proportional to
        # the sizes involved keeps those corners, which is the difference
        # between a mesh and a grid that share a face meeting there or missing
        # each other. See the C counterpart in euclib._c._core.
        slack = _FEASIBILITY_EPSILON * max(
            float(abs(offsets).max()), float(abs(point).max()), 1.0)
        if ((normals * point[:, None]).sum(axis=0)
                <= offsets + tolerance + slack).all():
            found.append(point)
    if not found:
        return zeros((3, 0))
    # Every feasible triple yields its own copy of a corner, and the copies
    # differ in the last bits of their coordinates; exact equality would keep
    # them all, leaving a set so nearly degenerate that the hull of it cannot
    # be taken. Two copies within `step` of one another are taken to be the
    # same corner, and the first of them is the one kept. Comparing them to
    # one another rather than to a rounding grid matters: a grid puts two
    # copies of one corner in different cells whenever they fall on either
    # side of a cell's edge, which is a decision rounding should not make.
    points = asarray(found)                 # (V, 3): one row per candidate
    scale = max(float(abs(points).max()), 1.0)
    step = max(float(tolerance), 1e-9 * scale)
    kept = []
    for point in points:
        if not any(float(((point - q) ** 2).sum()) <= step * step
                   for q in kept):
            kept.append(point)
    return asarray(kept).T                  # (3, V)


#: The corner search that ``tetrahedron_box_intersection`` fills its region
#: from. It is looked up here, once per intersection, rather than called by
#: name, so that the dispatcher in ``euclib.utils._core`` can replace it with
#: the C kernel when the extension provides one: the corner search is 120 small
#: linear solves and accounts for essentially all of an intersection's cost,
#: while the decomposition that follows it operates on a dozen points.
_vertices_kernel = tetrahedron_box_vertices


def tetrahedron_box_intersection(tet, bounds, tolerance=0.0):
    '''Decomposes the region a tetrahedron and a box share into tetrahedra.

    Parameters
    ----------
    tet : numpy.ndarray
        A ``(3, 4)`` matrix of the tetrahedron's corners.
    bounds : numpy.ndarray
        A ``(3, 2)`` box, such as one voxel of a grid.
    tolerance : float, optional
        How far outside a plane a corner may lie and still be counted. The
        default is ``0``.

    Returns
    -------
    vertices : numpy.ndarray
        A ``(3, V)`` matrix of the region's corners.
    tetrahedra : numpy.ndarray
        A ``(4, T)`` integer matrix of the tetrahedra that fill the region,
        indexing ``vertices``.
    '''
    vertices = _vertices_kernel(tet, bounds, tolerance)
    if vertices.shape[1] < 4:
        # Three corners make a triangle and two make a segment; neither has a
        # volume to fill.
        return (vertices, zeros((4, 0), dtype=int))
    try:
        # Delaunay rather than ConvexHull: the hull describes the region by its
        # surface, whose facets are triangles, where what is wanted here is a
        # filling of it by tetrahedra.
        filled = Delaunay(vertices.T)
    except QhullError:
        # A degenerate region --- every corner in one plane, say --- has no
        # volume to fill either.
        return (vertices, zeros((4, 0), dtype=int))
    return (vertices, asarray(filled.simplices).T)


# Prisms #####################################################################

def closest_prism(coords0, coords1, indices, tetrahedra, query,
                  per_prism=3):
    '''Locates positions within a set of prisms.

    A position within a prism is the position within its triangle on each
    surface, blended by an elevation: ``(1 - e) * X0(u, v) + e * X1(u, v)``.
    Expanding that shows it carries ``u * e`` and ``v * e`` terms, so the local
    coordinates are a *nonlinear* function of the position whenever the two
    surfaces are not parallel --- the linear machinery that inverts a simplex
    cannot invert a prism.

    .. note:: The Newton loop below is the only optimization loop in the
        library. Before the 1.0 release, its performance must be measured
        against realistic prism meshes and, if it is not acceptable, replaced
        --- by a closed-form solve on the tetrahedron rather than a global
        Newton iteration, by a smaller iteration budget, or by the C kernel
        that ``euclib._c`` will provide.

    The search therefore has two stages. The prism's tetrahedral decomposition
    says which prism contains the position and, because a tetrahedron's corners
    are prism corners, gives a first estimate of the local coordinates; that
    estimate is already exact when the surfaces are parallel. Newton's method
    then refines the estimate until it reproduces the position, which converges
    in one step for a parallel-sided prism and in a handful for a skewed one.

    Parameters
    ----------
    coords0, coords1 : numpy.ndarray
        The two surfaces, each a ``(D, N)`` matrix of coordinates.
    indices : array-like
        The ``(3, M)`` integer matrix of triangle corners.
    tetrahedra : array-like
        The ``(4, 3*M)`` integer matrix of the tetrahedra that the prisms
        decompose into, as produced by ``PrismTopology.tetrahedra``: the
        tetrahedra of each prism occupy ``per_prism`` consecutive columns, and
        index the surfaces laid end to end.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.
    per_prism : int, optional
        The number of tetrahedra each prism decomposes into. The default is 3.

    Returns
    -------
    index : numpy.ndarray
        A length-``Q`` vector of the prism containing or nearest each query.
    weight : numpy.ndarray
        A ``(2, Q)`` matrix of the first two barycentric weights within each
        prism's triangle; the third is their complement.
    height : numpy.ndarray
        A ``(1, Q)`` matrix of elevations, from 0 at the first surface to 1 at
        the second.
    '''
    coords0 = _as_numpy(coords0)
    coords1 = _as_numpy(coords1)
    indices = asarray(indices)
    tetrahedra = asarray(tetrahedra)
    query = _as_numpy(query)
    (d, n) = coords0.shape
    q = query.shape[1]
    merged = concatenate([coords0, coords1], axis=1)
    # (1) The tetrahedron containing or nearest each query position.
    (tet, w) = closest_simplex(merged, tetrahedra, query)
    prism = tet // int(per_prism)
    # The tetrahedra fill the prisms, and ``closest_simplex`` clamps to the
    # simplices it is given, so the position it answers with is the nearest
    # position *on* the prisms --- the query itself when the query is inside
    # one, and the nearest position on the boundary when it is not. Solving the
    # parameterization for that position rather than for the query is what
    # keeps a position outside a prism from being answered with its own
    # extrapolation: the parameterization can be inverted just as well for a
    # position beyond the prism as for one within it.
    wanted = simplex_points(merged, tetrahedra, tet, w)
    # (2) The first estimate: each tetrahedron corner is a prism corner, and the
    # prism corners have known local coordinates --- (0,0,at the first surface)
    # for the first, (1,0,...) for the second, and (0,1,...) for the third.
    (a, b, c) = (indices[0][prism], indices[1][prism], indices[2][prism])
    corners = tetrahedra[:, tet]                        # (4, Q)
    side = (corners >= n).astype(float)
    within = corners % n
    # Each local coordinate stores the weights of the triangle's first two
    # corners; the third is their complement, as it is for a triangle mesh. A
    # tetrahedron's corners are prism corners, whose weights are known: the
    # first is 1 where the second and third are 0, and so on.
    w0 = (within == a[None, :]) * 1.0
    w1 = (within == b[None, :]) * 1.0
    full = concatenate([w, (1.0 - w.sum(axis=0))[None, :]], axis=0)  # (4, Q)
    x = (np.stack([w0, w1, side], axis=-1) * full[:, :, None]).sum(axis=0)
    # (3) Newton's method on p(w0, w1, e) - query = 0, where the position within
    # the triangle is written relative to the *third* corner so that the two
    # weights are the ones the local coordinate stores.
    (a0, b0, c0) = (coords0[:, a], coords0[:, b], coords0[:, c])
    (a1, b1, c1) = (coords1[:, a], coords1[:, b], coords1[:, c])
    e0 = a0 - c0
    e1 = b0 - c0
    f0 = (a1 - c1) - e0
    f1 = (b1 - c1) - e1
    dc = c1 - c0
    for _ in range(40):
        (u, v, e) = (x[:, 0], x[:, 1], x[:, 2])
        drift = dc + u * f0 + v * f1
        res = (c0 + u * e0 + v * e1 + e * drift) - wanted
        jac = np.stack([e0 + e * f0, e1 + e * f1, drift], axis=1)
        # The right-hand side is given a trailing axis so that solve reads it as
        # one column per query rather than as a matrix of batches.
        step = linalg.solve(jac.transpose(2, 0, 1), (-res).T[:, :, None])[:, :, 0]
        x = x + step
        if not isfinite(step).all() or np.abs(step).max() < _TOLERANCE:
            break
    return (prism, x[:, :2].T, x[:, 2][None, :])


# Spatial Subdivision ########################################################

def bounds_of(coords, indices=None):
    '''Returns the bounding box of a set of points or of a set of simplices.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    indices : array-like or None, optional
        Simplex corners, as a ``(K+1, M)`` integer matrix. The default,
        ``None``, takes the box of every coordinate.

    Returns
    -------
    bounds : numpy.ndarray
        A ``(D, 2)`` matrix whose first column is the minimum along each axis
        and whose second column is the maximum.
    '''
    coords = _as_numpy(coords)
    pts = coords if indices is None else coords[:, asarray(indices)].reshape(
        coords.shape[0], -1)
    return concatenate([pts.min(axis=1)[:, None], pts.max(axis=1)[:, None]],
                       axis=1)


def simplex_boxes(coords, indices):
    '''Returns a center and a radius for each simplex.

    The center is the middle of the simplex's bounding box and the radius is
    the distance from that center to a corner of the box, so every point of the
    simplex lies within ``radius`` of the center. The box is a cheap thing for
    a spatial index to store, where the simplex itself would be expensive.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    indices : array-like
        A ``(K+1, M)`` integer matrix of simplex corners.

    Returns
    -------
    centers : numpy.ndarray
        A ``(D, M)`` matrix of box centers.
    radii : numpy.ndarray
        A length-``M`` vector of box radii.
    '''
    coords = _as_numpy(coords)
    corners = coords[:, asarray(indices)]          # (D, K+1, M)
    low = corners.min(axis=1)
    high = corners.max(axis=1)
    half = (high - low) / 2.0
    return ((low + high) / 2.0, sqrt((half * half).sum(axis=0)))


def split_cells(centers, bounds):
    '''Places points in the sub-cells of a box's bisection.

    The box is halved along each axis, which makes ``2**D`` sub-cells --- four
    quadrants in two dimensions and eight octants in three --- and each point is
    placed in the one that contains it. This is the single step that a
    quadtree or an octree repeats as it descends, and the reason it is a kernel
    of its own: it is the whole of the work that a C implementation would
    accelerate.

    Parameters
    ----------
    centers : numpy.ndarray
        A ``(D, M)`` matrix of points.
    bounds : numpy.ndarray
        The ``(D, 2)`` box being subdivided.

    Returns
    -------
    cells : numpy.ndarray
        A length-``M`` vector of sub-cell indices, from 0 to ``2**D - 1``. The
        cell whose index has bit *d* set is the upper half along axis *d*.
    '''
    centers = _as_numpy(centers)
    bounds = _as_numpy(bounds)
    if centers.shape[0] != bounds.shape[0]:
        raise ValueError(
            f"the points have dimension {centers.shape[0]}, but the box has"
            f" dimension {bounds.shape[0]}")
    middle = (bounds[:, 0] + bounds[:, 1]) / 2.0
    upper = centers > middle[:, None]
    powers = (1 << arange(centers.shape[0]))[:, None]
    return (upper * powers).sum(axis=0)


def octree_split(centers, bounds):
    '''Places points in the octants of a box's bisection.

    Equivalent to ``split_cells``, with a name that says which of the two
    subdivision structures it belongs to and a check that the space really is
    three-dimensional.

    Parameters
    ----------
    centers : numpy.ndarray
        A ``(3, M)`` matrix of points.
    bounds : numpy.ndarray
        The ``(3, 2)`` box being subdivided.

    Returns
    -------
    cells : numpy.ndarray
        A length-``M`` vector of octant indices, 0 through 7.
    '''
    if _as_numpy(centers).shape[0] != 3:
        raise ValueError("an octree subdivides three-dimensional space")
    return split_cells(centers, bounds)


def quadtree_split(centers, bounds):
    '''Places points in the quadrants of a box's bisection.

    Equivalent to ``split_cells``, with a name that says which of the two
    subdivision structures it belongs to and a check that the space really is
    two-dimensional.

    Parameters
    ----------
    centers : numpy.ndarray
        A ``(2, M)`` matrix of points.
    bounds : numpy.ndarray
        The ``(2, 2)`` box being subdivided.

    Returns
    -------
    cells : numpy.ndarray
        A length-``M`` vector of quadrant indices, 0 through 3.
    '''
    if _as_numpy(centers).shape[0] != 2:
        raise ValueError("a quadtree subdivides two-dimensional space")
    return split_cells(centers, bounds)


# Simplices ##################################################################

def unique_columns(mat):
    '''Returns the unique columns of an integer matrix, in sorted order.

    This is used to derive the unique *k*-simplices of a mesh from its
    higher-order simplices. The result is canonical: the entries within each
    column are sorted, and the columns are sorted lexicographically by their
    contents. Two matrices that contain the same columns in different vertex
    orders therefore produce identical results.

    Parameters
    ----------
    mat : array-like
        A 2-dimensional integer array whose columns are the items to
        deduplicate.

    Returns
    -------
    numpy.ndarray
        An array with the same number of rows as ``mat`` whose columns are the
        unique, row-sorted columns of ``mat`` in lexicographic order.
    '''
    mat = asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"expected a 2-dimensional matrix, found {mat.ndim}")
    if mat.shape[0] < 1:
        raise ValueError("expected at least one row")
    # Sort the entries within each column, then deduplicate the rows of the
    # transpose, which sorts them lexicographically.
    return unique(sort(mat, axis=0).T, axis=0).T


def unique_coords(coords, return_index=False, return_inverse=False):
    '''Returns the unique columns of a coordinate matrix.

    Parameters
    ----------
    coords : numpy.ndarray or torch.Tensor
        A coordinate matrix of shape ``(D, N)``.
    return_index : bool, optional
        Whether to also return the indices of ``coords`` that yield the unique
        columns. The default is ``False``.
    return_inverse : bool, optional
        Whether to also return the indices that reconstruct ``coords`` from the
        unique columns. The default is ``False``.

    Returns
    -------
    unique : numpy.ndarray or torch.Tensor
        The unique columns of ``coords``.
    index : numpy.ndarray or torch.Tensor, optional
        The indices of the first occurrences of the unique columns; returned
        only when ``return_index`` is ``True``.
    inverse : numpy.ndarray or torch.Tensor, optional
        The indices reconstructing ``coords``; returned only when
        ``return_inverse`` is ``True``.
    '''
    t = checktorch()
    is_tensor = t is not None and isinstance(coords, t.Tensor)
    if is_tensor:
        dev = coords.device
        x = asarray(coords.detach().cpu())
    else:
        x = asarray(coords)
    if x.ndim != 2:
        raise ValueError(
            f"expected a 2-dimensional coordinate matrix, found {x.ndim}D")
    res = unique(x, axis=1,
                 return_index=return_index, return_inverse=return_inverse)
    if not isinstance(res, tuple):
        res = (res,)
    if is_tensor:
        res = tuple(t.from_numpy(r).to(dev) for r in res)
    return res[0] if len(res) == 1 else res

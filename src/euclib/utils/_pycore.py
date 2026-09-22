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
    arange, asarray, clip, concatenate, full, inf, isfinite, linalg, ones,
    sort, unique, where, zeros)

from itertools import combinations

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

def nearest_vertices(coords, query):
    '''Finds the nearest coordinate to each query position.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``(D, N)`` matrix of coordinates.
    query : numpy.ndarray
        A ``(D, Q)`` matrix of query positions.

    Returns
    -------
    index : numpy.ndarray
        A length-``Q`` vector of the column of ``coords`` nearest each query.
    '''
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


def closest_simplex(coords, indices, query):
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

    Returns
    -------
    index : numpy.ndarray
        A length-``Q`` vector of the simplex containing or nearest each query.
    weight : numpy.ndarray
        A ``(K, Q)`` matrix of the first ``K`` barycentric weights within the
        chosen simplex; the final weight is their complement.
    '''
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
        res = (c0 + u * e0 + v * e1 + e * drift) - query
        jac = np.stack([e0 + e * f0, e1 + e * f1, drift], axis=1)
        # The right-hand side is given a trailing axis so that solve reads it as
        # one column per query rather than as a matrix of batches.
        step = linalg.solve(jac.transpose(2, 0, 1), (-res).T[:, :, None])[:, :, 0]
        x = x + step
        if not isfinite(step).all() or np.abs(step).max() < _TOLERANCE:
            break
    return (prism, x[:, :2].T, x[:, 2][None, :])


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

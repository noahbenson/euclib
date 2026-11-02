# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_smooth.py
'''Smoothing a property over a mesh, or an image over a grid.

The mesh side is the discrete Laplace operator: the *stiffness* matrix, whose
off-diagonal entries say how strongly two vertices are coupled, and the *mass*
matrix, whose diagonal says how much of the mesh each vertex owns. The pair is
what a diffusion equation is written with, and the smoothing step is one step of
it --- implicit by default, which is stable at any step size.

The stiffness is the cotangent one: an edge's weight is half the sum of the
cotangents of the angles opposite it, which is what the linear finite-element
discretization gives on a triangle and what it gives on a tetrahedron too. One
implementation serves both, the difference being only what each simplex's
measure is.

The mass is Meyer's *mixed* Voronoi area: the Voronoi area of a vertex within
each triangle, except that a triangle obtuse at a corner gives that corner half
its area and the other two a quarter each. The plain Voronoi area goes negative
on an obtuse triangle, and the mixed form is what keeps the mass positive.

The Laplacian is normalized by the mass before it is used, so that the step size
is dimensionless: `M^-1 L` has eigenvalues in [0, 2] whatever the mesh's size or
shape, and a step of one means the same thing on every mesh.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (add, arange, asarray, concatenate, maximum, minimum,
                   ones, sqrt, unique, where, zeros)
from scipy.sparse import coo_matrix


# The operator ###############################################################

def _cotangent(u, v, /):
    '''The cotangent of the angle between two vectors.

    `(u . v) / |u x v|` rather than `1 / tan`, because the cross product's
    magnitude is the sine and is never zero for two of a triangle's edges, where
    a tangent of a right angle would be infinite. The magnitude comes from the
    Gram determinant, `|u|^2 |v|^2 - (u . v)^2`, which is the squared cross
    product in any dimension --- numpy defines a cross only in three.
    '''
    dot = (u * v).sum(axis=0)
    sine = sqrt(maximum((u * u).sum(axis=0) * (v * v).sum(axis=0) - dot * dot,
                        0.0))
    return dot / where(sine > 0.0, sine, 1.0)


def cotangents(coords, indices, /):
    '''The cotangent of each corner's angle, for each simplex.

    A corner's angle is between the two edges leaving it, so corner ``k``'s
    cotangent is that of the angle between the vectors to the other two corners.
    '''
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    out = zeros((count, number))
    for k in range(count):
        # The *positions* of the other two corners, not the difference of their
        # indices: the angle at corner k is between the edges leaving it.
        u = coords[:, indices[(k + 1) % count]] - coords[:, indices[k]]
        v = coords[:, indices[(k + 2) % count]] - coords[:, indices[k]]
        out[k] = _cotangent(u, v)
    return out


def stiffness(coords, indices, /):
    '''The cotangent stiffness matrix of a simplex mesh, as a sparse COO.

    Each edge's weight is half the sum of the cotangents of the angles opposite
    it --- the angles at the corners the edge does not touch --- which is the
    linear finite-element discretization of the Laplacian. The diagonal is the
    negated row sum, so the matrix annihilates constants.
    '''
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    cot = cotangents(coords, indices)
    rows = []
    cols = []
    data = []
    for k in range(count):
        # The edge opposite corner `k` is the one between the other two, and its
        # weight is half of that corner's cotangent.
        (one, two) = ((k + 1) % count, (k + 2) % count)
        rows.append(indices[one])
        cols.append(indices[two])
        data.append(-0.5 * cot[k])
        rows.append(indices[two])
        cols.append(indices[one])
        data.append(-0.5 * cot[k])
    rows = concatenate(rows)
    cols = concatenate(cols)
    data = concatenate(data)
    size = int(indices.max()) + 1 if indices.size else 0
    at = rows * size + cols
    total = zeros(size)
    for (one, value, entry) in zip(rows, data, at):
        total[one] += value
    rows = concatenate([rows, arange(size)])
    cols = concatenate([cols, arange(size)])
    data = concatenate([data, -total])
    return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def mass(coords, indices, /):
    '''Meyer's mixed Voronoi area, one entry per vertex, as a sparse diagonal.'''
    from ..utils import simplex_measures
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    area = asarray(simplex_measures(coords, indices))
    cot = cotangents(coords, indices)
    size = int(indices.max()) + 1 if indices.size else 0
    out = zeros(size)
    # Whether the triangle is obtuse at all, and if so at which corner: a
    # cotangent is negative exactly there, the angle being greater than a right
    # one.
    blunt = (number == 3) & (cot < 0).any(axis=0)
    where_blunt = (cot < 0).argmax(axis=0)
    for k in range(count):
        # A corner's Voronoi share of a triangle is an eighth of the sum over
        # its two leaving edges of `cot(opposite) * |edge|^2`.
        (one, two) = ((k + 1) % count, (k + 2) % count)
        u = coords[:, indices[one]] - coords[:, indices[k]]
        v = coords[:, indices[two]] - coords[:, indices[k]]
        share = ((u * u).sum(axis=0) * cot[two]
                 + (v * v).sum(axis=0) * cot[one]) / 8.0
        # Meyer's mixed area: a triangle obtuse at a corner gives that corner
        # half its area and the other two a quarter each, because the Voronoi
        # cells of those two fall outside it and the Voronoi share goes negative.
        mine = share
        blunt_here = blunt & (where_blunt == k)
        blunt_else = blunt & (where_blunt != k)
        mine = (1.0 - blunt) * asarray(mine) + blunt_here * (area / 2.0) \
            + blunt_else * (area / 4.0)
        # `add.at` rather than `out[...] += `: a vertex is a corner of several
        # triangles, and the fancy-indexed `+=` keeps only the last of a
        # repeated index instead of accumulating them --- the shares then total
        # less than the mesh's area, which is how this was found.
        add.at(out, indices[k], asarray(mine))
    return coo_matrix((out, (arange(size), arange(size))),
                      shape=(size, size)).tocsr()

def boundary_vertices(indices, /):
    '''Which vertices lie on a boundary, for a mesh of simplices.

    An edge belongs to two elements in the interior and to one at a boundary, so
    an edge used once marks both its ends.
    '''
    indices = asarray(indices)
    (count, number) = indices.shape
    size = int(indices.max()) + 1 if indices.size else 0
    pairs = []
    for k in range(count):
        (one, two) = ((k + 1) % count, (k + 2) % count)
        low = minimum(indices[one], indices[two])
        high = maximum(indices[one], indices[two])
        pairs.append(low * size + high)
    flat = concatenate(pairs) if pairs else zeros(0, dtype=int)
    (keys, counts) = unique(flat, return_counts=True)
    on_edge = keys[counts == 1]
    out = zeros(size, dtype=bool)
    out[on_edge // size] = True
    out[on_edge % size] = True
    return out


def _solve(matrix, right, /):
    '''Solves a sparse system with one right-hand side per channel.

    `right` is ``(N, C)`` --- one row per unknown, one column per channel ---
    and so is the answer. `spsolve` handles both shapes it can take, but a
    single channel comes back flat, so that is made two-dimensional.
    '''
    from scipy.sparse.linalg import spsolve
    solved = spsolve(matrix.tocsc(), asarray(right))
    if solved.ndim == 1:
        solved = solved.reshape(-1, 1)
    return solved


def smooth(values, coords, indices, /, loglambda=0.0, method='implicit',
           fixboundary=True):
    '''Smooths a value array over a simplex mesh.

    One step of the diffusion equation, with the cotangent Laplacian and Meyer's
    mixed mass. The operator is `M^-1 L`, which is dimensionless --- its
    eigenvalues lie in [0, 2] whatever the mesh's size or shape --- so a step of
    one means the same thing on every mesh, and `loglambda` is exponential in it
    so that linear changes in the parameter are multiplicative in the step.

    Parameters
    ----------
    values : array-like
        ``(C..., N)``: the value at each vertex, with the channel dimensions
        leading.
    coords, indices : numpy.ndarray
        The mesh's coordinates and its ``(K+1, M)`` matrix of corners.
    loglambda : float, optional
        The logarithm of the step. The default, ``0``, is a step of one.
    method : str, optional
        ``'implicit'``, the default, which is stable at any step, or
        ``'explicit'``, which is cheaper per step and stable only below a bound
        the mesh decides.
    fixboundary : bool, optional
        Whether boundary vertices keep their values. The default, ``True``,
        does; ``False`` smooths them along their interior edges.

    Returns
    -------
    numpy.ndarray
        The smoothed values, shaped as the input was.
    '''
    from math import exp
    values = asarray(values, dtype='float64')
    # The channels lead and the vertices trail, so the arithmetic is done on
    # `(C, N)` and the shape is put back at the end.
    shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])          # (C, N)
    L = stiffness(coords, indices)
    M = mass(coords, indices)
    step = exp(float(loglambda))
    free = ones(L.shape[0], dtype=bool)
    if fixboundary:
        free = ~boundary_vertices(indices)
    if method == 'explicit':
        # f - lambda M^-1 L f, which needs no solve.
        moves = (L @ flat.T) / M.diagonal()[:, None]
        out = flat.copy()
        out[:, free] -= step * moves.T[:, free]
        return out.reshape(shape + (out.shape[-1],))
    if method != 'implicit':
        raise ValueError(
            f"unknown smoothing method {method!r}; expected 'implicit' or"
            f" 'explicit'")
    # (M + lambda L) f' = M f, restricted to the vertices whose values move.
    # The restricted system's right-hand side is not just its own rows: a free
    # vertex's equation involves its neighbours' values, and the fixed ones are
    # known --- so their contribution moves to the other side. Without that the
    # operator stops annihilating constants, and a constant field comes back
    # smoothed toward whatever the fixed vertices happen to be.
    full = (M + L * step).tocsr()
    known = ~free
    right = M @ flat.T
    if known.any():
        right = right - full[:, known] @ flat.T[known, :]
    system = full[free][:, free]
    solved = _solve(system, right[free, :])
    out = flat.copy()
    out[:, free] = solved.T
    return out.reshape(shape + (out.shape[-1],))

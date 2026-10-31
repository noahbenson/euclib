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

from numpy import (arange, asarray, concatenate, maximum, sqrt, where, zeros)
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
    for k in range(count):
        # A corner's Voronoi share of a triangle is an eighth of the sum over
        # its two leaving edges of `cot(opposite) * |edge|^2`.
        (one, two) = ((k + 1) % count, (k + 2) % count)
        u = coords[:, indices[one]] - coords[:, indices[k]]
        v = coords[:, indices[two]] - coords[:, indices[k]]
        share = ((u * u).sum(axis=0) * cot[two]
                 + (v * v).sum(axis=0) * cot[one]) / 8.0
        # An obtuse corner owns half the triangle and the others a quarter.
        obtuse = (number == 3) & (cot[k] < 0)
        share = asarray(share)
        share = (1.0 - obtuse) * share + obtuse * (area / 2.0)
        out[indices[k]] += share
    return coo_matrix((out, (arange(size), arange(size))),
                      shape=(size, size)).tocsr()

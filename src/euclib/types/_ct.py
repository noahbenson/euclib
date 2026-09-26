# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_ct.py
'''The Clough-Tocher element: a C1 piecewise cubic on a triangle.

A macro triangle is split into three by joining its centroid to its corners, and
each third carries a cubic Bezier patch. The element's twelve numbers --- the
value and the gradient at each vertex, and the derivative across each edge at
its midpoint --- determine a piecewise cubic that is C1 across the interior
edges, which is the classical Clough-Tocher element. This module holds that
construction, and the derivation of it is on the documentation page for the
method.

**Why the numbers and not the controls.** The interpolant is a *linear* function
of the twelve numbers, so the twelve basis functions can be found by solving one
system: the conditions that hold the pieces together on the left, and a unit
number at a time on the right. Everything this module does is that solve, done
once, and then an affine change of the answer for each element.

**The solve is per triangle, and it has to be.** The construction is affine
everywhere except in one place, and that place decides it: the derivative across
an edge is taken along a direction that belongs to the *edge* --- the edge's
corners put in order by where they are, turned a quarter turn --- because the two
elements sharing an edge have to mean the same thing by it. A quarter turn does
not commute with a scaling or a shear, so the direction on an element is not the
reference's direction mapped, and there is no change of the numbers that turns
one triangle's solve into another's. An earlier plan was to derive the element
once on the reference triangle and transform each element's numbers through its
affine map; it is wrong, and the test on a lopsided triangle is what says so.

What *is* affine, and saves the rest, is the evaluation: barycentric coordinates
survive an affine map unchanged, so a position within a sub-triangle is named by
the same three numbers on any triangle, and no per-position change is needed
once the basis is in hand.
'''

# Dependencies ###############################################################

from __future__ import annotations

from math import factorial

from numpy import (
    argmin, array, asarray, einsum, eye, linalg, stack, zeros)

#: The reference triangle: the corners the element is derived on.
REFERENCE = array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

#: Which macro edge each sub-triangle holds, as a pair of the triangle's
#: corners. Sub-triangle ``k`` is that edge together with the centroid.
EDGES = ((0, 1), (1, 2), (2, 0))

#: The multi-indices of a cubic on a triangle, and of its derivative's.
POWERS = tuple((i, j, 3 - i - j) for i in range(4) for j in range(4 - i))
LOWER = tuple((i, j, 2 - i - j) for i in range(3) for j in range(3 - i))


def _coefficient(power, /):
    '''The multinomial coefficient of one multi-index.'''
    top = factorial(sum(power))
    for entry in power:
        top //= factorial(entry)
    return float(top)


COEF = {p: _coefficient(p) for p in POWERS}
COEF2 = {p: _coefficient(p) for p in LOWER}
SPOT = {p: n for (n, p) in enumerate(POWERS)}


def bernstein(w, /):
    """The Bernstein polynomials of degree three at a set of weights.

    One set of weights gives a vector of ten; a ``(3, Q)`` matrix of them gives
    a ``(10, Q)`` matrix, one column per position, which is what evaluating at
    many positions at once wants.
    """
    w = asarray(w)
    powers = asarray(POWERS)
    coef = asarray([COEF[p] for p in POWERS])
    if w.ndim == 1:
        return coef * (w[None, :] ** powers).prod(axis=-1)
    return coef[:, None] * (w[None, :, :] ** powers[:, :, None]).prod(axis=1)


def evaluate(controls, w, /):
    """A Bezier cubic's value at the weights ``w``.

    A property's channel dimensions lead, as they do everywhere in this
    library, so the controls are ``(C..., 10)`` and the sum is over the last
    axis alone. A ``(3, Q)`` matrix of weights answers ``(C..., Q)``.
    """
    w = asarray(w)
    if w.ndim == 1:
        return einsum('w,...w->...', bernstein(w), controls)
    return einsum('wq,...w->...q', bernstein(w), controls)


def directional(controls, w, axis, /):
    '''The derivative along the direction from corner 0 to corner ``axis``.

    This is the standard form: a degree-``n`` Bernstein sum's derivative is
    ``n`` times the sum of the differences of the controls one step apart, over
    the degree-``n-1`` basis.
    '''
    total = 0.0
    for p in LOWER:
        high = list(p)
        high[axis] += 1
        base = list(p)
        base[0] += 1
        total += COEF2[p] * (w[0] ** p[0]) * (w[1] ** p[1]) * (w[2] ** p[2]) \
            * (controls[..., SPOT[tuple(high)]]
               - controls[..., SPOT[tuple(base)]])
    return 3.0 * total


def gradient(controls, corners, w, /):
    '''A patch's spatial gradient at the weights ``w``.

    The derivative along ``(corner_axis - corner_0)`` is the gradient's
    component in that direction, so the two directional derivatives fix it as
    ``M^T g = d`` with the directions as the *columns* of ``M``. Solving
    ``M g = d`` gives a different answer unless the two directions are at right
    angles --- which is why a triangle whose corners are axis-aligned, or
    symmetric in the right way, cannot tell the two apart and a lopsided one
    can.
    '''
    # The two directions from the first corner, as the columns of a matrix.
    along = array([corners[:, 1] - corners[:, 0],
                   corners[:, 2] - corners[:, 0]]).T
    wanted = stack([directional(controls, w, 1),
                    directional(controls, w, 2)], axis=-1)
    return einsum('jk,...k->...j', linalg.inv(along.T), wanted)


# The pieces #################################################################

def _corners(triangle, k, /):
    """The three corners of sub-triangle k: a macro edge and the centre.

    A ``(2, 3)`` matrix, as coordinates are everywhere in this library: the
    dimensions leading and the corners following.
    """
    (a, b) = EDGES[k]
    centre = triangle.mean(axis=1)
    return array([triangle[:, a], triangle[:, b], centre]).T


def _holders(vertex, /):
    """The two sub-triangles that hold a macro corner."""
    return [k for k in range(3) if vertex in EDGES[k]]


def _slot(k, vertex, /):
    """Which corner of sub-triangle k a macro corner is."""
    (a, b) = EDGES[k]
    return 0 if a == vertex else 1


def _internal(k, vertex, /):
    """The two interior controls of piece k's edge from a corner to the centre."""
    slot = _slot(k, vertex)
    opposite = 1 - slot
    out = []
    for (near, far) in ((2, 1), (1, 2)):
        spot = [0, 0, 0]
        spot[slot] = near
        spot[2] = far
        spot[opposite] = 0
        out.append(tuple(spot))
    return out


def _row_from(functional, k, /):
    """The row of a condition that touches only piece k's ten controls.

    Each column is that control set to one and the rest to zero, so the row is
    the condition evaluated on unit controls: nothing is written down by hand,
    and nothing can be misremembered.
    """
    row = zeros(30)
    for n in range(10):
        controls = [zeros(10) for _ in range(3)]
        controls[k][n] = 1.0
        row[k * 10 + n] = functional(controls)
    return row


def _sharing(triangle, /):
    """The controls two pieces hold in common, where they meet."""
    rows = []
    for vertex in range(3):
        row = zeros(30)
        # One control is subtracted from the other: the row says the two are
        # equal, which wants +1 and -1 rather than +1 twice.
        for (n, k) in enumerate(_holders(vertex)):
            corner = _slot(k, vertex)
            spot = tuple(3 if x == corner else 0 for x in range(3))
            row[k * 10 + SPOT[spot]] = 1.0 if n == 0 else -1.0
        rows.append(row)
    for k in (1, 2):
        row = zeros(30)
        row[SPOT[(0, 0, 3)]] = 1.0
        row[k * 10 + SPOT[(0, 0, 3)]] = -1.0
        rows.append(row)
    for vertex in range(3):
        (k, other) = _holders(vertex)
        for (mine, theirs) in zip(_internal(k, vertex),
                                  _internal(other, vertex)):
            row = zeros(30)
            row[k * 10 + SPOT[mine]] = 1.0
            row[other * 10 + SPOT[theirs]] = -1.0
            rows.append(row)
    return rows


def _weights_on_edge(k, vertex, s, /):
    """The weights of the point a fraction s from a macro corner to the centre."""
    w = zeros(3)
    w[_slot(k, vertex)] = s
    w[2] = 1.0 - s
    return w


def _c1(triangle, samples=5, /):
    """The pieces' gradients agreeing along the edges they share.

    The gradient is a function along an edge, so sampling it at a few points
    holds the pieces to each other; each row is built from the evaluation at
    one of those points, so no condition is stated in a form that could mean
    something else.
    """
    rows = []
    for vertex in range(3):
        (k, other) = _holders(vertex)
        for s in [i / (samples - 1) for i in range(samples)]:
            for component in (0, 1):
                row = zeros(30)
                for (piece, sign) in ((k, 1.0), (other, -1.0)):
                    w = _weights_on_edge(piece, vertex, s)
                    corners = _corners(triangle, piece)
                    for n in range(10):
                        controls = [zeros(10) for _ in range(3)]
                        controls[piece][n] = 1.0
                        row[piece * 10 + n] += sign * gradient(
                            controls[piece], corners, w)[component]
                rows.append(row)
    return rows


def across_vector(triangle, k, /):
    """The vector the derivative across piece k's outer edge is taken along.

    Two elements that share an edge have to take the derivative across it in
    the same direction, or their numbers mean different things and they cannot
    agree; so the direction belongs to the *edge* and not to the element. It is
    the edge's two corners put in order by where they are --- not by their
    index, which the two elements number differently --- turned a quarter turn
    one way.
    """
    (a, b) = EDGES[k]
    (i, j) = sorted((a, b), key=lambda x: tuple(triangle[:, x]))
    edge = triangle[:, j] - triangle[:, i]
    return array([-edge[1], edge[0]])


def element_rows(triangle, /):
    """The rows that read off the twelve numbers, and what each one is."""
    rows = []
    for vertex in range(3):
        k = _holders(vertex)[0]
        corner = _slot(k, vertex)
        w = zeros(3)
        w[corner] = 1.0
        rows.append((_row_from(lambda c, w=w, k=k: evaluate(c[k], w), k),
                     ('value', vertex)))
        for component in (0, 1):
            rows.append((_row_from(
                lambda c, w=w, k=k, m=component:
                gradient(c[k], _corners(triangle, k), w)[m], k),
                ('gradient', vertex, component)))
    for k in range(3):
        w = array([0.5, 0.5, 0.0])
        across = across_vector(triangle, k)
        rows.append((_row_from(
            lambda c, w=w, k=k, d=across:
            float(gradient(c[k], _corners(triangle, k), w) @ d), k),
            ('across', k)))
    return rows


def basis(triangle=None, /):
    """The twelve control vectors of the element on one triangle.

    This is the whole of the construction: the conditions that hold the pieces
    together, and then a unit number at a time. It is solved for the triangle it
    is given, and there is no way to avoid that --- see the note at the top of
    this module for why the reference triangle's answer does not carry over.
    """
    if triangle is None:
        triangle = REFERENCE
    held = _sharing(triangle) + _c1(triangle)
    data = [row for (row, _) in element_rows(triangle)]
    rows = asarray(held + data)
    targets = zeros((rows.shape[0], 12))
    targets[len(held):, :] = eye(12)
    return linalg.lstsq(rows, targets, rcond=None)[0]


# The number a Property cannot carry #########################################

def _bezier_controls(values, gradients, corners, /):
    """The ten Bezier controls of a cubic through one triangle's own data.

    This is the construction the triangle method uses --- each edge fitted
    through its two ends' values and slopes, and a cubic's one interior value
    the mean of the three edges' degree-2 controls --- written here for a single
    triangle rather than for a whole query, because the edge estimate below
    asks it one triangle at a time.
    """
    (ends, at) = (corners, gradients)
    # A property's channel dimensions lead and the corners follow the
    # dimensions, so the corner is the *last* axis of the values and the first
    # of the controls, which are indexed as ``control[..., spot]``.
    control = zeros(gradients.shape[:-2] + (len(POWERS),))

    def spot(power, /):
        return SPOT[tuple(power)]

    def along(i, j, /):
        step = ends[:, j] - ends[:, i]
        return (at[..., :, i] @ step, at[..., :, j] @ step)


    for c in range(3):
        control[..., spot([3 if x == c else 0 for x in range(3)])] = values[..., c]
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        (da, db) = along(i, j)
        # The control a third of the way in from each end, which carries the
        # slope there: on a cubic its exponent is two at that corner and one
        # at the other.
        near_i = [2 if x == i else (1 if x == j else 0) for x in range(3)]
        near_j = [1 if x == i else (2 if x == j else 0) for x in range(3)]
        control[..., spot(near_i)] = values[..., i] + da / 3.0
        control[..., spot(near_j)] = values[..., j] - db / 3.0
    # The one interior control: the average of the three edges' *degree-2*
    # control values, which is what degree elevation of a quadratic gives.
    middle = []
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        (da, db) = along(i, j)
        (bi, bj) = (values[..., i] + da / 3.0, values[..., j] - db / 3.0)
        halfway = (values[..., i] + 3.0 * bi + 3.0 * bj + values[..., j]) / 8.0
        middle.append(2.0 * halfway
                      - (values[..., i] + values[..., j]) / 2.0)
    control[..., spot([1, 1, 1])] = sum(middle) / 3.0
    return control


def _across_of(coords, one, two, /):
    """The vector the derivative across an edge is taken along.

    The same rule the element's own edges use: the edge's two corners in the
    order of where they are, turned a quarter turn. Both triangles sharing the
    edge compute this and get the same vector, which is the point of it.
    """
    (i, j) = (one, two) if tuple(coords[:, one]) < tuple(coords[:, two]) \
        else (two, one)
    edge = coords[:, j] - coords[:, i]
    return array([-edge[1], edge[0]])


def edge_data(geom, values, slopes, /):
    """The derivative across each edge of a triangle mesh at its midpoint.

    The element's twelfth kind of number --- the derivative across an edge at
    its midpoint --- is per *edge*, and a `Property` carries its data per
    coordinate, so there is nowhere to put one. It is estimated instead, from
    the mesh: each triangle carries a cubic Bezier patch from its own corners'
    values and gradients, and the patch's gradient at the midpoint of an edge,
    in the direction the two triangles sharing it agree on, is what that
    triangle says the derivative there is. The estimate is the average of the
    triangles sharing the edge.

    Why the average is the right one: the derivative across an edge, as a
    function along it, is a quadratic whose values at the two ends are already
    fixed by the corner gradients --- which both triangles share. Two quadratics
    that agree at both ends and at the middle are the same quadratic, so taking
    the average makes the two triangles' across-derivatives identical, and that
    is exactly C1.

    Returns
    -------
    dict
        One entry per edge, keyed by its two corner indices in order, holding
        the estimate.
    """
    coords = asarray(geom.coords)
    indices = asarray(geom.topo.indices)
    total = {}
    count = {}
    for t in range(indices.shape[1]):
        here = indices[:, t]
        (values_here, slopes_here) = (values[..., here], slopes[..., here])
        control = _bezier_controls(values_here, slopes_here, coords[:, here])
        for (one, two) in ((0, 1), (1, 2), (2, 0)):
            (i, j) = (int(here[one]), int(here[two]))
            middle = (coords[:, i] + coords[:, j]) / 2.0
            across = _across_of(coords, i, j)
            w = array([0.5, 0.5, 0.0]) if (one, two) == (0, 1) else (
                array([0.0, 0.5, 0.5]) if (one, two) == (1, 2)
                else array([0.5, 0.0, 0.5]))
            # The gradient of the patch at the midpoint of that edge.
            direction = gradient(control, coords[:, here], w)
            key = (min(i, j), max(i, j))
            total[key] = total.get(key, 0) + asarray(direction @ across)
            count[key] = count.get(key, 0) + 1
    return {key: total[key] / count[key] for key in total}


def sub_weights(weights, /):
    """The weights within the piece that holds a position, from the whole
    triangle's.

    The three pieces are the triangle's three edges with the centre. A position
    with barycentric weights ``u`` has, within the piece holding the two corners
    of the edge it leaves out, the weights

        ``(u_a - u_c, u_b - u_c, 3 u_c)``,

    where the piece's own corners are ``a`` and ``b`` and ``c`` is the corner it
    leaves out. This comes of writing the centre as the average of the three
    corners and reading the weights off. The piece that holds the position is
    the one leaving out the corner whose weight is *smallest* --- which is the
    same as its three weights being non-negative --- and that corner is the one
    *before* the piece in the list, since piece 0 holds the edge (0, 1) and so
    leaves out corner 2.

    Parameters
    ----------
    weights : numpy.ndarray
        A ``(2, Q)`` matrix of the first two barycentric weights of positions
        within one triangle; the third is what they leave of the unit sum.

    Returns
    -------
    pieces : numpy.ndarray
        A length-``Q`` vector naming the piece each position falls in.
    inside : numpy.ndarray
        A ``(Q, 3)`` matrix of the weights within that piece.
    """
    whole = stack([weights[0], weights[1],
                   1.0 - weights[0] - weights[1]], axis=-1)     # (Q, 3)
    pieces = (argmin(whole, axis=-1) + 1) % 3                    # (Q,)
    out = zeros((whole.shape[0], 3))
    for k in range(3):
        here = pieces == k
        if not here.any():
            continue
        (a, b) = EDGES[k]
        left_out = [x for x in range(3) if x not in (a, b)][0]
        out[here, 0] = whole[here, a] - whole[here, left_out]
        out[here, 1] = whole[here, b] - whole[here, left_out]
        out[here, 2] = 3.0 * whole[here, left_out]
    return (pieces, out)

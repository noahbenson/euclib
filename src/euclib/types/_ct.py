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
    array, asarray, einsum, eye, linalg, zeros)

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


def bernstein(w, powers=POWERS, coef=None, /):
    '''The Bernstein polynomials of a degree at one set of weights.'''
    return array([(coef or COEF)[p] * (w[0] ** p[0]) * (w[1] ** p[1])
                  * (w[2] ** p[2]) for p in powers])


def evaluate(controls, w, /):
    '''A Bezier cubic's value at the weights ``w``.'''
    return float(bernstein(w) @ controls)


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
            * (controls[SPOT[tuple(high)]] - controls[SPOT[tuple(base)]])
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
    along = array([corners[1] - corners[0], corners[2] - corners[0]]).T
    wanted = array([directional(controls, w, 1),
                    directional(controls, w, 2)])
    return linalg.solve(along.T, wanted)


# The pieces #################################################################

def _corners(triangle, k, /):
    """The three corners of sub-triangle k: a macro edge and the centre."""
    (a, b) = EDGES[k]
    centre = triangle.mean(axis=1)
    return array([triangle[:, a], triangle[:, b], centre])


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

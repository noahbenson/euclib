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

import immlib.math as im
from immlib import quant, to_array
from numpy import (
    argmin, array, arange, asarray, concatenate, einsum, empty, eye,
    linalg, ones, stack, where, zeros)

import numpy as np

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
    # The weights are handed back rather than converted, and the arithmetic is
    # immlib's: a weight is a *position's* coordinate within a piece, so a tensor
    # position has a derivative through the basis. It is the basis' magnitude
    # that comes back, so that a caller reading this as an array still can.
    w = im.mag(w) if hasattr(w, 'requires_grad') else asarray(w)
    powers = asarray(POWERS)
    coef = asarray([COEF[p] for p in POWERS])
    if w.ndim == 1:
        return im.mag(im.multiply(
            coef, im.prod(im.pow(w[None, :], powers), axis=-1)))
    return im.mag(im.multiply(coef[:, None], im.prod(
        im.pow(w[None, :, :], powers[:, :, None]), axis=1)))


def evaluate(controls, w, /):
    """A Bezier cubic's value at the weights ``w``.

    A property's channel dimensions lead, as they do everywhere in this
    library, so the controls are ``(C..., 10)`` and the sum is over the last
    axis alone. A ``(3, Q)`` matrix of weights answers ``(C..., Q)``.
    """
    # Through immlib, because the control values may be a tensor: the
    # interpolation's data reaches this contraction, and a numpy ``einsum`` on a
    # tensor reaches its dispatch instead of computing.
    # The magnitude and not a conversion to an array: since
    # ``bernstein`` works in immlib's arithmetic now, a tensor weight
    # keeps its derivative through the contraction.
    w = im.mag(w) if hasattr(w, 'requires_grad') else asarray(w)
    if w.ndim == 1:
        return im.einsum('w,...w->...', bernstein(w), controls)
    return im.einsum('wq,...w->...q', bernstein(w), controls)


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
    # The two directions from the first corner, as the columns of a matrix,
    # and the same choice of arithmetic as `_corners`: this is the inner loop of
    # the element's conditions, and dispatch through immlib costs more than the
    # two-by-two it is dispatching.
    two = directional(controls, w, 1)
    if not (_is_tensor(corners) or _is_tensor(two)):
        along = array([corners[:, 1] - corners[:, 0],
                       corners[:, 2] - corners[:, 0]]).T
        wanted = stack([two, directional(controls, w, 2)], axis=-1)
        # The pseudo-inverse and not the inverse: a triangle is usually a
        # surface in three dimensions, where the two edge directions do not span
        # the space and no inverse exists. What it gives is the gradient's part
        # *in* the triangle's plane, which is the only part a field on the
        # triangle can mean.
        return einsum('jk,...k->...j', linalg.pinv(along.T), wanted)
    along = im.mag(im.stack([im.subtract(corners[:, 1], corners[:, 0]),
                             im.subtract(corners[:, 2], corners[:, 0])],
                            axis=-1))
    wanted = im.mag(im.stack([two, directional(controls, w, 2)], axis=-1))
    return im.mag(im.einsum('jk,...k->...j', im.pinv(along.T), wanted))




def _is_tensor(one, /):
    """Whether an argument is a torch tensor rather than an array."""
    return type(one).__module__.split('.')[0] == 'torch'



def _stacked(values, /):
    """The values as one array, in whichever backend they are.

    Immlib's stack for a tensor and numpy's for arrays: the rows are built some
    fifty times per basis, and a quantity wrapped and unwrapped on each of them
    is most of what the array path would otherwise pay for this.
    """
    if any(_is_tensor(one) for one in values):
        return im.mag(im.stack(values))
    return asarray(values)



def _difference(one, two, /):
    """``one - two``, in whichever backend the arguments are."""
    if _is_tensor(one) or _is_tensor(two):
        return im.mag(im.subtract(one, two))
    return one - two


def _dot(one, two, /):
    """The last-axis dot product, in whichever backend the arguments are.

    Through immlib for a tensor and numpy's own for arrays, for the same reason
    `_stacked` is: this is the innermost expression of the element's conditions
    and is evaluated once per control per row.
    """
    if _is_tensor(one) or _is_tensor(two):
        return im.mag(im.sum(im.multiply(one, two), axis=-1))
    return asarray(one) @ asarray(two)


def _zeros_of(like, count, /):
    """``count`` zeros in the backend of ``like``."""
    return im.mag(quant(like).new_zeros((count,)))


def _unit_controls(piece, n, /):
    """Three pieces' controls, with one of ``piece``'s set to one."""
    controls = [zeros(10) for _ in range(3)]
    controls[piece][n] = 1.0
    return controls


def _placed(values, piece, /):
    """A length-30 row, with a piece's ten values at their own offset.

    Stacked rather than assigned into a row made beforehand: the values are a
    function of the triangle --- through the element's own gradient, evaluated
    there --- and a tensor among them will not be assigned into an array made as
    numpy.
    """
    raw = im.mag(values)
    if not _is_tensor(raw):
        row = zeros(30)
        row[piece * 10:(piece + 1) * 10] = raw
        return row
    return im.mag(im.concatenate(
        [_zeros_of(values, piece * 10), raw,
         _zeros_of(values, 30 - (piece + 1) * 10)]))


# The pieces #################################################################

def _corners(triangle, k, /):
    """The three corners of sub-triangle k: a macro edge and the centre.

    A ``(2, 3)`` matrix, as coordinates are everywhere in this library: the
    dimensions leading and the corners following.
    """
    (a, b) = EDGES[k]
    # immlib for a tensor, because `array` of a list reaches numpy and a list
    # holding one that requires a gradient has no numpy array to reach --- and
    # numpy for an array, because this is called six hundred times per basis and
    # immlib wraps a quantity on each of them.
    if not _is_tensor(triangle):
        return array([triangle[:, a], triangle[:, b],
                      triangle.mean(axis=1)]).T
    return im.mag(im.stack([triangle[:, a], triangle[:, b],
                            im.mean(triangle, axis=1)], axis=-1))


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
    return _placed(_stacked([functional(_unit_controls(k, n))
                             for n in range(10)]), k)


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
                # Two pieces reach this row, so it is their two placements
                # added rather than written one after the other into one array.
                tensor = _is_tensor(triangle)
                parts = []
                for (piece, sign) in ((k, 1.0), (other, -1.0)):
                    w = _weights_on_edge(piece, vertex, s)
                    corners = _corners(triangle, piece)
                    values = [gradient(_unit_controls(piece, n)[piece],
                                       corners, w)[component]
                              for n in range(10)]
                    if tensor:
                        values = [im.mag(im.multiply(sign, im.mag(one)))
                                  for one in values]
                    else:
                        values = [sign * one for one in values]
                    parts.append(_placed(_stacked(values), piece))
                rows.append(im.mag(im.add(parts[0], parts[1])) if tensor
                            else parts[0] + parts[1])
    return rows


def plane_normal(corners, /):
    """The normal of the plane a triangle lies in.

    A triangle in two dimensions is lifted to the plane it is drawn in, which is
    the one its points are named in; a triangle in three has a plane of its own.
    The *sign* does not matter to anything below, because a normal turned round
    turns the edge round with it and the turn below settles its own sign.
    """
    if corners.shape[0] == 2:
        # A constant, so numpy will do; the arithmetic below promotes it along
        # with whatever it meets.
        return array([0.0, 0.0, 1.0])
    (u, v) = (im.subtract(corners[:, 1], corners[:, 0]),
              im.subtract(corners[:, 2], corners[:, 0]))
    return im.stack([im.subtract(im.multiply(u[1], v[2]),
                                 im.multiply(u[2], v[1])),
                     im.subtract(im.multiply(u[2], v[0]),
                                 im.multiply(u[0], v[2])),
                     im.subtract(im.multiply(u[0], v[1]),
                                 im.multiply(u[1], v[0]))])


def turn(edge, normal, /):
    """An edge turned a quarter turn *within the plane it lies in*.

    For an edge of a triangle in a plane this is the quarter turn the element
    has always used; for an edge of a triangle in space it is the turn within
    that triangle's own plane, which the flat formula is not --- the flat formula
    turns the edge in the coordinate plane, which for a triangle in space is not
    the triangle's plane at all, and can point off the surface entirely.

    The sign is settled from the result rather than from the normal: the first
    component that is not zero decides it. Two triangles in the *same* plane
    give parallel turns differing only in sign, and this makes them agree on one
    vector, which is what lets two elements in a plane be C1 across the edge
    they share. Two triangles in *different* planes give genuinely different
    turns whatever it says, and no rule can reconcile them --- the surface is
    creased there and the two derivatives mean different things.
    """
    if len(edge) == 2:
        candidate = im.stack([edge[1], im.negative(edge[0])])
    else:
        (a, b, c) = edge
        (u, v, w) = normal
        candidate = im.stack([im.subtract(im.multiply(b, w), im.multiply(c, v)),
                              im.subtract(im.multiply(c, u), im.multiply(a, w)),
                              im.subtract(im.multiply(a, v), im.multiply(b, u))])
    # The sign is a *selection*, and stays one: which way the turn goes is
    # decided by the first component that is not zero, a comparison with no
    # derivative. What keeps its graph is the vector, which is a continuous
    # function of the corners --- so the decision is taken on the magnitude and
    # the magnitude is what is returned, negated or not.
    candidate = im.mag(candidate)
    for entry in candidate:
        if entry > 0.0:
            return candidate
        if entry < 0.0:
            return im.mag(im.negative(candidate))
    return candidate


def across_vector(triangle, k, /):
    """The vector the derivative across piece k's outer edge is taken along.

    Two elements that share an edge have to take the derivative across it in
    the same direction, or their numbers mean different things and they cannot
    agree; so the direction belongs to the *edge* and not to the element. It is
    the edge's two corners put in order by where they are --- not by their
    index, which the two elements number differently --- turned a quarter turn
    within the triangle's plane.
    """
    (a, b) = EDGES[k]
    # The ordering is a selection too --- two corners put in order by where they
    # are, not by their index, which the two elements sharing the edge number
    # differently --- and is left as one.
    (i, j) = sorted((a, b), key=lambda x: tuple(im.mag(triangle)[:, x]))
    return turn(im.subtract(triangle[:, j], triangle[:, i]),
                plane_normal(triangle))


def neighbours_of(vertex, /):
    """The other two corners of a triangle, in increasing order. A triangle's
    corners are 0, 1 and 2, so this is the two that are not this one."""
    return tuple(x for x in range(3) if x != vertex)


def element_rows(triangle, /):
    """The rows that read off the twelve numbers, and what each one is.

    The twelve are the three corners' values, the derivative at each corner
    *along its two edges towards the other corners* --- six numbers --- and the
    derivative across each edge at its midpoint. They are the twelve the
    classical construction is stated in, and the derivation on the method's
    page works entirely in them.

    **Why the slopes and not the gradient's components.** The element has to
    serve a triangle in a plane and a triangle in space alike, and what makes
    that possible is taking its second-order data as derivatives in *geometric*
    directions: the vector from a corner to its neighbour is a direction in the
    geometry, however the geometry is named, and the derivative along it is the
    same number for the triangle in a plane and for the triangle in space. A
    component of the gradient in a *coordinate* direction is only the slope
    along an edge when the edge happens to run along that axis, and nothing else
    in the construction repairs it --- which is exactly how the element came to
    answer wrongly for a mesh in three dimensions.
    """
    rows = []
    for vertex in range(3):
        k = _holders(vertex)[0]
        corner = _slot(k, vertex)
        w = zeros(3)
        w[corner] = 1.0
        rows.append((_row_from(lambda c, w=w, k=k: evaluate(c[k], w), k),
                     ('value', vertex)))
        for other in neighbours_of(vertex):
            # Through immlib and not `float(...)`: a float takes the number and
            # drops whatever graph was behind it, without failing --- so the row
            # would come back looking right with the coordinates' derivative
            # quietly gone.
            along = _difference(triangle[:, other], triangle[:, vertex])
            rows.append((_row_from(
                lambda c, w=w, k=k, d=along:
                _dot(gradient(c[k], _corners(triangle, k), w), d), k),
                ('slope', vertex, other)))
    for k in range(3):
        w = array([0.5, 0.5, 0.0])
        across = across_vector(triangle, k)
        rows.append((_row_from(
            lambda c, w=w, k=k, d=across:
            _dot(gradient(c[k], _corners(triangle, k), w), d), k),
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
    rows = im.mag(im.stack(held + data))
    # The targets are constants --- a unit number at a time --- so they stay
    # arrays; the solve promotes them along with the rows, and that is what
    # carries the triangle's derivative through the basis.
    targets = zeros((len(held) + len(data), 12))
    targets[len(held):, :] = eye(12)
    return im.mag(im.lstsq(rows, targets)[0])


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


def _across_of(coords, one, two, normal, /):
    """The vector the derivative across an edge is taken along.

    The same rule the element's own edges use: the edge's two corners in the
    order of where they are, turned a quarter turn within the triangle's plane.
    Two triangles *in that plane* compute this and get the same vector, which is
    the point of it.
    """
    (i, j) = (one, two) if tuple(coords[:, one]) < tuple(coords[:, two]) \
        else (two, one)
    return turn(coords[:, j] - coords[:, i], normal)


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
            across = _across_of(coords, i, j,
                                 plane_normal(coords[:, here]))
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
        A length-``Q`` vector naming the piece each position falls in. It is
        always an array: which piece holds a position is a comparison, and the
        callers read it as an index.
    inside : numpy.ndarray or torch.Tensor
        A ``(Q, 3)`` matrix of the weights within that piece, in the backend of
        ``weights`` so that a tensor's derivative survives the reading-off.
    """
    whole = im.mag(im.stack([im.mag(weights[0]), im.mag(weights[1]),
                             im.mag(im.subtract(
                                 im.subtract(1.0, weights[0]), weights[1]))],
                            axis=-1))                            # (Q, 3)
    pieces = asarray((im.mag(im.argmin(whole, axis=-1)) + 1) % 3)  # (Q,)
    # A position lies in exactly one piece, so each of the three weights is the
    # sum of its three candidates masked to the piece that holds the position.
    # Masked and summed rather than assigned, because a tensor's weights carry a
    # derivative and will not be assigned into an array.
    columns = []
    for j in range(3):
        picked = None
        for k in range(3):
            (a, b) = EDGES[k]
            left_out = [x for x in range(3) if x not in (a, b)][0]
            if j == 0:
                value = im.subtract(whole[:, a], whole[:, left_out])
            elif j == 1:
                value = im.subtract(whole[:, b], whole[:, left_out])
            else:
                value = im.multiply(3.0, whole[:, left_out])
            masked = im.where(pieces == k, value, im.multiply(value, 0.0))
            picked = masked if picked is None else im.add(picked, masked)
        columns.append(picked)
    return (pieces, im.mag(im.stack(columns, axis=-1)))


# The operator the element's edge numbers come from ##########################

def edge_key(coords, i, j, /):
    """An edge, named by its two corners in order by where they are.

    The order has to be one both triangles holding the edge can see, and their
    own numbering of it is not: a derivative across an edge means nothing
    without a direction.
    """
    return (i, j) if tuple(coords[:, i]) <= tuple(coords[:, j]) else (j, i)


def across_of(coords, i, j, normal, /):
    """The direction the derivative across an edge is taken along."""
    (a, b) = edge_key(coords, i, j)
    return turn(coords[:, b] - coords[:, a], normal)


def _slope_column(corner, towards, /):
    """Which of a triangle's twelve numbers is the slope at one corner along
    the edge towards another: a value and then the two slopes per corner, the
    slopes in the order the neighbours come in."""
    return 3 * corner + 1 + neighbours_of(corner).index(towards)


def triangle_blocks(corners, /):
    """What one triangle says its three edges' derivatives are.

    A ``(3, 3*(D+1))`` matrix: for each of its edges, the coefficients of the
    derivative across it at its midpoint, in terms of its corners' values and
    gradients --- a value and then one gradient per corner, in the order the
    element's own rows are built.

    **The controls it goes through**, which is where the element's content is:
    the corners hold their own values; the two controls a third of the way along
    an edge carry the slope there, being the gradient's component along the
    edge; and the one interior control is the mean of the three edges'
    *reflected* midpoints, each ``(v_i + v_j)/2 + (s_i - s_j)/4``. That
    reflection is the same form the quadratic edge's middle control has, and it
    is not the midpoint value, which is ``(v_i + v_j)/2 + (s_i - s_j)/8``. The
    gradient at an edge's midpoint is then the pair of directional derivatives
    there combined through the triangle's own axes and dotted with the edge's
    across direction.
    """
    # A triangle's twelve numbers: three values, the slope at each corner along
    # each of its two edges, and the derivative across each edge at its
    # midpoint. The same twelve whatever the dimension --- which is what lets
    # this serve a triangle in a plane and a triangle in space alike --- and the
    # slopes are in *geometric* directions, from a corner towards a neighbour,
    # so they are the same numbers for either.
    width = 12
    block = zeros((3, width))
    controls = zeros((10, width))
    for c in range(3):
        controls[SPOT[tuple(3 if x == c else 0 for x in range(3))], 3 * c] = 1.0
    for (u, v) in ((0, 1), (1, 2), (2, 0)):
        near_u = SPOT[tuple(2 if x == u else (1 if x == v else 0)
                            for x in range(3))]
        near_v = SPOT[tuple(1 if x == u else (2 if x == v else 0)
                            for x in range(3))]
        # A third of the way from u towards v the field is u's value plus a
        # third of the slope at u along that edge --- and the same at the other
        # end, the slope there being the one along the edge towards u.
        controls[near_u, 3 * u] = 1.0
        controls[near_u, _slope_column(u, v)] = 1.0 / 3.0
        controls[near_v, 3 * v] = 1.0
        controls[near_v, _slope_column(v, u)] = 1.0 / 3.0
        # The interior control is the mean of the three edges' *reflected*
        # midpoints, each of which is
        # ``(v_u + v_v)/2 + (s(u->v) + s(v->u))/4`` --- the same form the
        # quadratic edge's middle control has, and not the midpoint value, which
        # has the slopes over eight instead of over four.
        inside = SPOT[(1, 1, 1)]
        controls[inside, 3 * u] += 0.5 / 3.0
        controls[inside, 3 * v] += 0.5 / 3.0
        controls[inside, _slope_column(u, v)] += 0.25 / 3.0
        controls[inside, _slope_column(v, u)] += 0.25 / 3.0
    axes = array([corners[:, 1] - corners[:, 0],
                  corners[:, 2] - corners[:, 0]]).T
    for (e, (a, b)) in enumerate(((0, 1), (1, 2), (2, 0))):
        w = zeros(3)
        w[a] = 0.5
        w[b] = 0.5
        # The coefficients the two directional derivatives take, from the
        # across direction --- the pseudo-inverse again, and for the same
        # reason.
        share = turn(corners[:, b] - corners[:, a], plane_normal(corners)) \
            @ linalg.pinv(axes.T)
        for (m, coefficient) in enumerate(share, start=1):
            row = zeros(10)
            for p in LOWER:
                high = list(p)
                high[m] += 1
                base = list(p)
                base[0] += 1
                weight = (3.0 * COEF2[p] * (w[0] ** p[0]) * (w[1] ** p[1])
                          * (w[2] ** p[2]) * coefficient)
                row[SPOT[tuple(high)]] += weight
                row[SPOT[tuple(base)]] -= weight
            block[e] += row @ controls
    return block


def _block_constant():
    """The fixed ``(3, 2, 12)`` matrix `triangle_blocks` is a matmul against.

    The construction in `triangle_blocks` looks irreducible --- it has a
    pseudo-inverse, a quarter-turn and a loop over the degree-two monomials ---
    but only the *triangle* varies. The midpoint weights are the edge's own two
    corners halved, the controls are built from `SPOT` indices and constants
    alone, and the monomial rows are linear in the two coefficients the across
    direction gives. So everything but those two coefficients is a constant, and
    one triangle's block is ``share @ CONST[edge]``.

    Computed once here rather than once per triangle: it is the same array for
    every mesh.
    """
    controls = zeros((10, 12))
    for c in range(3):
        controls[SPOT[tuple(3 if x == c else 0 for x in range(3))], 3 * c] = 1.0
    for (u, v) in EDGES:
        near_u = SPOT[tuple(2 if x == u else (1 if x == v else 0)
                            for x in range(3))]
        near_v = SPOT[tuple(1 if x == u else (2 if x == v else 0)
                            for x in range(3))]
        controls[near_u, 3 * u] = 1.0
        controls[near_u, _slope_column(u, v)] = 1.0 / 3.0
        controls[near_v, 3 * v] = 1.0
        controls[near_v, _slope_column(v, u)] = 1.0 / 3.0
        inside = SPOT[(1, 1, 1)]
        controls[inside, 3 * u] += 0.5 / 3.0
        controls[inside, 3 * v] += 0.5 / 3.0
        controls[inside, _slope_column(u, v)] += 0.25 / 3.0
        controls[inside, _slope_column(v, u)] += 0.25 / 3.0
    out = zeros((3, 2, 12))
    for (e, (a, b)) in enumerate(EDGES):
        w = zeros(3)
        w[a] = 0.5
        w[b] = 0.5
        R = zeros((2, 10))
        for m in range(1, 3):
            row = zeros(10)
            for p in LOWER:
                high = list(p)
                high[m] += 1
                base = list(p)
                base[0] += 1
                weight = (3.0 * COEF2[p] * (w[0] ** p[0]) * (w[1] ** p[1])
                          * (w[2] ** p[2]))
                row[SPOT[tuple(high)]] += weight
                row[SPOT[tuple(base)]] -= weight
            R[m - 1] = row
        out[e] = R @ controls
    return out


#: What `triangle_blocks` multiplies the across direction's coefficients by.
BLOCK_CONSTANT = _block_constant()


def _turned(candidate, /):
    """A quarter-turn's sign, for a stack of candidate vectors.

    The rule is `turn`'s, applied to every triangle at once: the first component
    that is not zero decides. It is a *selection* --- the sign is a comparison
    and has no derivative --- and this leaves the vector's magnitude alone.
    """
    out = ones(len(candidate))
    undecided = np.ones(len(candidate), dtype=bool)
    for m in range(candidate.shape[1]):
        here = undecided & (candidate[:, m] != 0.0)
        out[here] = np.sign(candidate[here, m])
        undecided &= ~here
    return out


def triangle_blocks_many(corners, /):
    """`triangle_blocks` for a stack: ``(M, D, 3)`` corners -> ``(M, 3, 12)``.

    The same numbers as `triangle_blocks`, which is what the test holds it to,
    with the per-triangle loop replaced by batched arithmetic: the
    pseudo-inverses in one call, the quarter-turns without a Python loop, and
    the blocks in one contraction against `BLOCK_CONSTANT`.
    """
    (m, dim) = (corners.shape[0], corners.shape[1])
    axes = stack([corners[:, :, 1] - corners[:, :, 0],
                  corners[:, :, 2] - corners[:, :, 0]], axis=-1)   # (M, D, 2)
    inverse = linalg.pinv(np.swapaxes(axes, 1, 2))                 # (M, 2, D)
    share = zeros((m, 3, 2))
    for (e, (a, b)) in enumerate(EDGES):
        edge = corners[:, :, b] - corners[:, :, a]                 # (M, D)
        if dim == 2:
            candidate = stack([edge[:, 1], -edge[:, 0]], axis=-1)
        else:
            u = corners[:, :, 1] - corners[:, :, 0]
            v = corners[:, :, 2] - corners[:, :, 0]
            normal = stack([u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                            u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                            u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]], axis=-1)
            (p, q, r) = (edge[:, 0], edge[:, 1], edge[:, 2])
            (x, y, z) = (normal[:, 0], normal[:, 1], normal[:, 2])
            candidate = stack([q * z - r * y, r * x - p * z, p * y - q * x],
                              axis=-1)
        candidate = candidate * _turned(candidate)[:, None]
        share[:, e, :] = einsum('md,mdc->mc', candidate, inverse)
    return einsum('mec,ecw->mew', share, BLOCK_CONSTANT)


def _edge_rows(coords, indices, /):
    """Each triangle's three edges, numbered in the order they are first met.

    Returns the ``(M, 3)`` matrix of the row each triangle's own edge is in, and
    the edge list those rows name. Both have to match `edge_key`'s ordering
    exactly: the direction a derivative is taken across an edge belongs to the
    *edge*, so two triangles sharing one must mean the same thing by it, and
    their own numbering of the corners is not the same.

    `edge_key` orders a pair by where its corners are; this does the same
    comparison over the whole mesh at once, which is the difference between a
    tuple per edge per triangle and one pass.
    """
    (a, b) = array(EDGES).T
    pairs = stack([indices[a].T, indices[b].T], axis=-1)        # (M, 3, 2)
    first = coords[:, pairs[:, :, 0]].transpose(1, 2, 0)        # (M, 3, D)
    second = coords[:, pairs[:, :, 1]].transpose(1, 2, 0)
    before = zeros(first.shape[:2], dtype=bool)
    decided = zeros(first.shape[:2], dtype=bool)
    for d in range(first.shape[2]):
        before |= ~decided & (first[:, :, d] < second[:, :, d])
        decided |= first[:, :, d] != second[:, :, d]
    ordered = stack([where(before, pairs[:, :, 0], pairs[:, :, 1]),
                     where(before, pairs[:, :, 1], pairs[:, :, 0])], axis=-1)
    (unique, inverse) = np.unique(ordered.reshape(-1, 2), axis=0,
                                  return_inverse=True)
    # `np.unique` sorts; the row an edge is given is the order it is first met,
    # so the unique rows are renumbered by where they first appear.
    # `return_index` is exactly that, and scanning per edge for it instead is
    # quadratic in the mesh.
    (_, appears) = np.unique(inverse, return_index=True)
    rank = empty(len(unique), dtype=int)
    rank[appears.argsort()] = np.arange(len(unique))
    return (rank[inverse].reshape(-1, 3),
            [tuple(int(x) for x in unique[k]) for k in appears.argsort()])


def _numbers(coords, indices, /):
    """The nine numbers each triangle reads from its corners' data.

    One row per number --- a corner's value, or its slope towards a neighbour ---
    and the columns they are read from: the coordinate itself, or one of its
    gradient's components. The sparsity pattern is fixed by the topology and only
    the ``along`` values depend on the mesh, so this is index arithmetic and not
    a loop over triangles.
    """
    (dim, count) = (coords.shape[0], coords.shape[1])
    triangles = indices.shape[1]
    here = indices.T                                            # (M, 3)
    base = 9 * arange(triangles)
    all_corners = coords[:, here]                               # (D, M, 3)
    (rows, columns, data) = ([], [], [])
    for vertex in range(3):
        rows.append(base + 3 * vertex)
        columns.append(here[:, vertex])
        data.append(ones(triangles))
        for (place, other) in enumerate(neighbours_of(vertex)):
            along = coords[:, here[:, other]][:, :, None] - all_corners
            for m in range(dim):
                rows.append(base + 3 * vertex + 1 + place)
                columns.append((m + 1) * count + here[:, vertex])
                data.append(along[m, :, vertex])
    return (concatenate(rows), concatenate(columns), concatenate(data),
            (9 * triangles, count + dim * count))


def edge_operator(coords, indices, /):
    """The operator giving the derivative across each edge at its midpoint.

    The element's twelfth kind of number is per *edge* and a `Property` carries
    its data per coordinate, so it is estimated from the mesh: each triangle says
    what the derivative across an edge is, from its own Bezier patch, and the
    estimate is the average of the triangles sharing it. That estimate is
    **linear** --- a triangle's control values are linear in its corners' values
    and gradients, and the gradient at an edge's midpoint is linear in the
    controls --- so it is one fixed operator, depending on the mesh alone.

    Its input is a coordinate's value and *its gradient*, stacked: the values
    first, then the gradient components as ``(D, N)`` flattened. It is built that
    way rather than as an operator on the values alone because a property may
    carry its own gradient, and one that is given is not a function of the
    values --- so the caller supplies the gradients it is fitting with, whether
    they were given or estimated, and this does not care which.

    **Built without a loop over triangles.** It used to be a Python loop that
    built one small sparse matrix per triangle and block-diagonalised them, plus
    two more loops appending to lists --- some thirty thousand iterations and
    three thousand matrix constructions, which for a mesh of nine thousand
    triangles took **2.25 seconds**. Nothing in it needed a loop: see
    `triangle_blocks_many`, `_edge_rows` and `_numbers`. The same operator now
    takes 41 ms, and a test holds it to `triangle_blocks`, which is still the
    readable statement of what it computes.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``(D, N)`` matrix of coordinates.
    indices : numpy.ndarray
        A ``(3, M)`` integer matrix of triangle corners.

    Returns
    -------
    scipy.sparse.csr_matrix
        An ``(E, N + D*N)`` operator: one row per distinct edge, reading the
        values and then the gradients.
    edges : list of tuple of int
        The two corners of each edge, in order.
    rows : numpy.ndarray
        A ``(M, 3)`` matrix of the row each triangle's own three edges are in,
        so that a caller can read off what a triangle's edges come to without
        looking the edges up again.
    """
    from scipy.sparse import csr_matrix
    # Detached, and it must be: this is a *constant*, the same for every
    # property and every mesh of this shape, which is why it is built once into
    # ``interp_data`` and applied as one sparse product. A SciPy matrix cannot
    # hold a tensor's graph --- and the entries are cast to floats below, which
    # a tensor requiring a gradient refuses --- so the coordinates arrive here
    # as coordinates alone. What that costs is the *edge estimate's* own
    # derivative with respect to where the mesh's corners are; the estimate's
    # derivative with respect to the values is untouched, flowing through the
    # values rather than through the operator.
    coords = to_array(coords, detach=True)
    (dim, count) = (coords.shape[0], coords.shape[1])
    triangles = indices.shape[1]
    (rows_of, edges) = _edge_rows(coords, indices)
    blocks = triangle_blocks_many(coords[:, indices].transpose(2, 0, 1))

    # What each triangle says, as one block-diagonal stack of its three rows.
    # The element's map is over all twelve of its numbers; the three that are
    # derivatives *across* an edge are not among a property's data --- they are
    # what this operator produces --- so only the nine a property can supply are
    # read here, and the whole point of the operator is to say what the tenth,
    # eleventh and twelfth would be.
    says = csr_matrix(
        (blocks[:, :, :9].ravel(),
         ((3 * arange(triangles)[:, None] + np.arange(3)[None, :])
          .repeat(9, axis=1).ravel(),
          (9 * arange(triangles)[:, None] + np.arange(9)[None, :])[:, None, :]
          .repeat(3, axis=1).ravel())),
        shape=(3 * triangles, 9 * triangles))

    (rows, columns, data, shape) = _numbers(coords, indices)
    numbers = csr_matrix((data, (rows, columns)), shape=shape)

    # Which triangles say anything about which edge, averaged.
    (rows, columns) = (rows_of.ravel(),
                       (3 * arange(triangles)[:, None]
                        + np.arange(3)[None, :]).ravel())
    sharing = csr_matrix((ones(rows.size), (rows, columns)),
                         shape=(len(edges), 3 * triangles))
    mean = sharing.multiply(
        1.0 / asarray(sharing.sum(axis=1)).ravel()[:, None])
    return ((mean @ says @ numbers).tocsr(), edges, rows_of)

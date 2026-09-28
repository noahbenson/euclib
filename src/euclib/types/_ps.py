# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_ps.py
'''The Powell-Sabin element: a C1 piecewise quadratic on a triangle.

A macro triangle is split into six by joining its incenter to its three corners
and to the midpoint of each edge, and each sixth carries a quadratic Bezier
patch. The element's nine numbers --- the value at each corner and the derivative
there along each of the triangle's two edges --- determine a piecewise quadratic
that is C1 across the interior edges, which is the classical Powell-Sabin
element. This module holds that construction, and the derivation of it is on the
documentation page for the method.

**Nine numbers, and that is the point of the element.** The Clough-Tocher element
is built from twelve, and three of its twelve are per *edge* rather than per
vertex --- the derivative across an edge at its midpoint --- which a property
does not carry and which therefore has to be estimated from the surrounding mesh.
Powell-Sabin needs nothing of the sort: the value and the gradient at each vertex
are enough, because the freedom that Clough-Tocher spends on the cross-boundary
datum is spent here on the *split*, which is geometry rather than data. The count
is exact --- nineteen ordinates, a nine-dimensional space of C1 piecewise
quadratics, and nine numbers of data --- so nothing is left over to choose.

**No coordinates anywhere.** Every number the element is built from is either a
value or a derivative along a direction the triangle's own edges define, so the
whole construction is barycentric arithmetic: the split's vertices are named by
their barycentric weights, the derivative along an edge is the chord rule applied
to a barycentric difference, and the triangle's shape enters only through the
incenter. A triangle in a plane and a triangle in space are the same problem
here, nothing has to be cached per mesh, and there is no analogue of the
Clough-Tocher element's per-edge operator.

**Why the solve is per triangle.** The barycentric weights of the incenter depend
on the triangle's shape, so the element does too, and there is no change of the
numbers that turns one triangle's solve into another's --- the same situation the
Clough-Tocher element is in. What *is* affine is the evaluation: barycentric
weights survive an affine map unchanged, so a position within a mini-triangle is
named by the same three numbers on any triangle.
'''

# Dependencies ###############################################################

from __future__ import annotations

import immlib.math as im
from immlib import quant
from numpy import (
    arange, argmin, array, asarray, einsum, eye, linalg, stack, zeros)

# The split ##################################################################

#: The seven vertices of the split, in the order the ordinates are numbered: the
#: three corners, the centre, and the point on each edge.
VERTICES = ('A', 'B', 'C', 'Z', 'S_AB', 'S_BC', 'S_CA')

#: The six half-edges, each of which is a mini-triangle's edge with the centre as
#: that mini-triangle's third corner.
HALVES = (('A', 'S_AB'), ('S_AB', 'B'), ('B', 'S_BC'),
          ('S_BC', 'C'), ('C', 'S_CA'), ('S_CA', 'A'))


def _name(one, two, /):
    '''The name of the edge between two vertices of the split.'''
    return '-'.join(sorted((one, two)))


#: The six mini-triangles, as the two ends of a half-edge and the centre.
PIECES = tuple((one, two, 'Z') for (one, two) in HALVES)

#: Every ordinate: one per vertex of the split and one per edge of it.
ORDINATES = (VERTICES
             + tuple(_name(one, two) for (one, two) in HALVES)
             + tuple(_name(point, 'Z') for point in VERTICES if point != 'Z'))

#: Where each ordinate sits in that list.
WHERE = {name: i for (i, name) in enumerate(ORDINATES)}

#: The multi-indices of a quadratic on a triangle, in the order `bernstein` and
#: therefore `SLOTS` put them: the three vertices, then the midpoints of the edges
#: between the first two, the first and third, and the second and third.
_MULTI = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))

#: Where each multi-index sits in that list, for the derivative rule to index by.
_AT = {multi: i for (i, multi) in enumerate(_MULTI)}

#: The six ordinates of each mini-triangle's patch, in the basis's order --- its
#: three corners and then the midpoint of the edge opposite each of them.
SLOTS = tuple(
    [WHERE[x] for x in (one, two, three,
                        _name(one, two), _name(one, three),
                        _name(two, three))]
    for (one, two, three) in PIECES)


# The quadratic simplex patch ###############################################

def bernstein(w, /):
    '''The six Bernstein polynomials of degree two, at a set of weights.

    The degree-two counterpart of the cubic element's, in the same order: the
    three vertices, then the midpoint of each edge with the vertex that follows
    it around --- which is the order `SLOTS` puts a mini-triangle's ordinates in.
    One set of weights gives a vector of six; a ``(3, Q)`` matrix of them gives a
    ``(6, Q)`` matrix, one column per position.
    '''
    # The weights are handed back rather than converted, and the arithmetic is
    # immlib's: a weight is a *position's* coordinate within a piece, so a tensor
    # position has a derivative through the basis. It is the basis' magnitude
    # that comes back, so that a caller reading this as an array still can. One
    # body serves both shapes: a vector of weights unpacks to three numbers and
    # a ``(3, Q)`` matrix to three vectors, and the stack is the same either way.
    w = im.mag(w) if hasattr(w, 'requires_grad') else asarray(w)
    (one, two, three) = w
    return im.mag(im.stack([
        im.multiply(one, one), im.multiply(two, two), im.multiply(three, three),
        im.multiply(2.0, im.multiply(one, two)),
        im.multiply(2.0, im.multiply(one, three)),
        im.multiply(2.0, im.multiply(two, three))]))


def evaluate(ordinates, w, /):
    '''A quadratic patch's value at the weights ``w``.

    A property's channel dimensions lead, as they do everywhere in this library,
    so the ordinates are ``(C..., 6)`` and the sum is over the last axis alone. A
    ``(3, Q)`` matrix of weights answers ``(C..., Q)``.
    '''
    # Through immlib, because the control values may be a tensor: the
    # interpolation's data reaches this contraction, and a numpy ``einsum`` on a
    # tensor reaches its dispatch instead of computing.
    # The magnitude and not a conversion to an array: since
    # ``bernstein`` works in immlib's arithmetic now, a tensor weight
    # keeps its derivative through the contraction.
    w = im.mag(w) if hasattr(w, 'requires_grad') else asarray(w)
    if w.ndim == 1:
        return im.einsum('w,...w->...', bernstein(w), ordinates)
    return im.einsum('wq,...w->...q', bernstein(w), ordinates)


# The geometry, in barycentric weights #######################################

def neighbours_of(vertex, /):
    '''The other two corners of a triangle, in increasing order.'''
    return tuple(x for x in range(3) if x != vertex)


def centre_weights(sides, /):
    '''The incenter's barycentric weights, from a triangle's side lengths.

    The weights are the lengths of the sides *opposite* the corners, which is
    what makes the point equidistant from the three edges. The sides are given
    in the order of the corners, so the side opposite corner ``i`` is the one
    between the other two.

    Parameters
    ----------
    sides : array-like
        The three side lengths, the one opposite each corner.

    Returns
    -------
    numpy.ndarray
        A length-three vector summing to one.
    '''
    # Through immlib, so the *side lengths* keep whatever graph the triangle's
    # corners gave them: the incenter is a function of where the corners are,
    # and so is every ordinate the element is built from.
    lengths = im.mag(sides)
    return im.mag(im.divide(lengths, im.sum(lengths)))


def _places(centre, /):
    '''Every vertex of the split, in the macro triangle's barycentric weights.

    The corners are the unit vectors, the centre is what it is, and the point on
    an edge is the midpoint of that edge --- the one choice of it that both
    triangles sharing an edge make the same way without consulting each other,
    and the only one available in a mesh embedded in three dimensions, where the
    segment between two triangles' incenters need not meet their shared edge at
    all.
    '''
    out = {'A': array([1.0, 0.0, 0.0]), 'B': array([0.0, 1.0, 0.0]),
           'C': array([0.0, 0.0, 1.0]), 'Z': im.mag(centre)}
    out['S_AB'] = array([0.5, 0.5, 0.0])
    out['S_BC'] = array([0.0, 0.5, 0.5])
    out['S_CA'] = array([0.5, 0.0, 0.5])
    return out


def _within(places, piece, /):
    '''The map from the macro triangle's weights to one mini-triangle's.

    The two sets of weights are related affinely, and the map is the inverse of
    the matrix whose columns are the mini-triangle's three corners' weights ---
    so a position's weights within the piece are that matrix's inverse applied to
    its weights within the macro triangle.
    '''
    # The pseudo-inverse, which is the inverse here: the three columns are a
    # mini-triangle's corners' weights and are independent.
    return im.mag(im.pinv(im.stack([places[x] for x in piece], axis=1)))


def _value_row(piece, point, places, within, /):
    '''The row reading one mini-triangle's value at a vertex of the split.'''
    return _place(bernstein(im.matmul(im.mag(within[piece]),
                                       im.mag(places[point]))), piece)


def _slope_row(piece, point, direction, places, within, /):
    '''The row reading one mini-triangle's derivative at a vertex, along a
    direction given as a barycentric difference.

    A degree-``n`` Bernstein sum's derivative in a direction ``d`` is ``n`` times
    the sum of the degree-``n-1`` basis against the controls one step apart along
    ``d`` --- which for a quadratic is

        ``2 sum_j w_j sum_k d_k b_{e_j + e_k}``,

    the controls indexed by their multi-indices. The cubic element's directional
    derivative is the same rule for ``n = 3`` with ``d`` one of the reference
    triangle's own axes; writing it for a general difference is what lets the
    directions here be the triangle's *edges*, and so what lets the element serve
    a triangle in space as readily as one in a plane.
    '''
    step = im.mag(im.matmul(im.mag(within[piece]), im.mag(direction)))
    centre = im.mag(im.matmul(im.mag(within[piece]), im.mag(places[point])))
    # Accumulated, because the multi-indices collide: ``e_here + e_along`` is
    # the same index for (0, 1) as for (1, 0), so two of the nine pairs reach
    # each of the six slots and the terms add. Setting rather than adding is how
    # this first read --- and it answered wrongly, which is what the tests on
    # the array path caught.
    zero = im.multiply(im.mag(centre)[0], 0.0)
    entries = [zero for _ in range(6)]
    for (here, weight) in enumerate(centre):
        one = [0, 0, 0]
        one[here] += 1
        for (along, entry) in enumerate(step):
            two = list(one)
            two[along] += 1
            index = _AT[tuple(two)]
            entries[index] = im.add(entries[index], im.multiply(
                im.multiply(2.0, weight), entry))
    return _place(im.stack(entries), piece)


def _sharing():
    '''The interior edges of the split, and what each one needs.

    One per vertex of the split other than the centre: the spoke from that vertex
    to the centre, which is an edge of exactly two mini-triangles. Each is
    reported with the two pieces sharing it and with their *third* corners ---
    the vertex of each piece that is not on the spoke --- because the difference
    of those two is a direction across the spoke.
    '''
    out = []
    for point in VERTICES:
        if point == 'Z':
            continue
        (one, two) = _name(point, 'Z').split('-')
        here = [k for (k, piece) in enumerate(PIECES)
                if one in piece and two in piece]
        thirds = [next(x for x in PIECES[k] if x not in (one, two))
                  for k in here]
        out.append((point, here, thirds))
    return out



def _zeros_of(like, count, /):
    """``count`` zeros in the backend of ``like``."""
    return im.mag(quant(like).new_zeros((count,)))


def _place(values, piece, /):
    """A length-nineteen row, with a mini-triangle's six values at its ordinates.

    Summed from one-hot placings rather than assigned into a row made
    beforehand: the values are a function of the triangle --- through the
    incenter, and so through the side lengths --- and a tensor among them will
    not be assigned into an array made as numpy.
    """
    out = _zeros_of(values, len(ORDINATES))
    for (n, slot) in enumerate(SLOTS[piece]):
        onehot = zeros(len(ORDINATES))
        onehot[slot] = 1.0
        out = im.add(out, im.multiply(onehot, im.mag(values)[n]))
    return im.mag(out)


def basis(centre, /):
    '''The element, as the nineteen ordinates of each of its nine numbers.

    The nine numbers are, for each corner in turn, the value there and the
    derivative there along each of the triangle's two edges from it --- six
    numbers of slope and three of value, in that order within each corner. They
    are the same *kind* of number the Clough-Tocher element is built from, and
    they are read from a property's values and gradients the same way, by dotting
    the gradient with the edge's direction; a triangle in space needs nothing
    more.

    The conditions are the nine the data supplies and C1 across the six interior
    edges, and the answer is one column per number. The rank is full in exactly
    the sense that matters: nineteen ordinates, nine columns, no freedom left.

    Parameters
    ----------
    centre : numpy.ndarray
        The incenter's barycentric weights, from `centre_weights`.

    Returns
    -------
    numpy.ndarray
        A ``(19, 9)`` matrix: the ordinates of a unit datum.
    '''
    places = _places(centre)
    within = [_within(places, piece) for piece in PIECES]
    rows = []
    columns = []

    def add(row, column=None, /):
        '''One condition, and the datum it asks for --- a unit number for a data
        row, and nothing at all for one of the C1 conditions, which ask for
        zero.'''
        rows.append(row)
        entry = zeros(9)
        if column is not None:
            entry[column] = 1.0
        columns.append(entry)

    def piece_of(vertex, /):
        '''The pieces touching a vertex of the split.'''
        name = VERTICES[vertex]
        return [k for (k, piece) in enumerate(PIECES) if name in piece]

    for vertex in range(3):
        for piece in piece_of(vertex):
            add(_value_row(piece, VERTICES[vertex], places, within), 3 * vertex)
        for (place, other) in enumerate(neighbours_of(vertex)):
            direction = im.subtract(places[VERTICES[other]],
                                    places[VERTICES[vertex]])
            for piece in piece_of(vertex):
                add(_slope_row(piece, VERTICES[vertex], direction, places, within),
                    3 * vertex + 1 + place)
    for (point, (one, two), (first, second)) in _sharing():
        # C1 across the spoke: the two pieces' derivatives in one direction
        # across it must agree. Their derivatives *along* it already do, since
        # they share the ordinates on it, so agreement in one transverse
        # direction is agreement in gradient. The direction is the difference of
        # the two pieces' third corners, which is transverse to the spoke and is
        # a barycentric difference --- so this is a condition the geometry states
        # without a metric, and one a triangle in space can state just as well.
        direction = im.subtract(places[first], places[second])
        for name in (point, 'Z'):
            add(im.mag(im.subtract(
                _slope_row(one, name, direction, places, within),
                _slope_row(two, name, direction, places, within))))
    # The columns are constants --- a unit number at a time --- so they stay
    # arrays; the solve promotes them along with the rows, and that is what
    # carries the triangle's derivative through the basis.
    return im.mag(im.lstsq(im.mag(im.stack(rows)), asarray(columns))[0])


def sub_weights(weights, centre, /):
    '''The weights within the mini-triangle that holds a position, from the whole
    triangle's.

    The pieces tile the triangle, so a position within it lies in exactly one,
    and the six sets of weights are read off and the piece whose smallest weight
    is largest is the one that holds it. Taking the largest smallest weight
    rather than the first non-negative one is what makes a position *outside* the
    triangle --- which the interpolation engine's extrapolation asks for --- land
    somewhere sensible and continuously, rather than wherever the search
    happened to end.

    Parameters
    ----------
    weights : numpy.ndarray
        A ``(2, Q)`` matrix of the first two barycentric weights of positions
        within one triangle; the third is what they leave of the unit sum.
    centre : numpy.ndarray
        The incenter's barycentric weights.

    Returns
    -------
    pieces : numpy.ndarray
        A length-``Q`` vector naming the mini-triangle each position falls in. It
        is always an array: which piece holds a position is a comparison, and the
        callers read it as an index.
    inside : numpy.ndarray or torch.Tensor
        A ``(Q, 3)`` matrix of the weights within that mini-triangle, in the
        backend of ``weights`` so that a tensor's derivative survives the
        reading-off.
    '''
    places = _places(centre)
    whole = im.mag(im.stack([im.mag(weights[0]), im.mag(weights[1]),
                             im.mag(im.subtract(
                                 im.subtract(1.0, weights[0]), weights[1]))],
                            axis=-1))                            # (Q, 3)
    inside = im.mag(im.stack([im.matmul(whole, _within(places, piece).T)
                              for piece in PIECES], axis=1))     # (Q, 6, 3)
    pieces = asarray(im.mag(im.argmin(
        im.negative(im.amin(inside, axis=-1)), axis=-1)))        # (Q,)
    # The weights within the piece that holds a position are its row of
    # ``inside``, read by masking the six rows and summing them rather than by
    # indexing with the piece --- indexing would break a tensor's derivative.
    picked = None
    for k in range(len(PIECES)):
        row = inside[:, k, :]
        masked = im.where((pieces == k)[:, None], row, im.multiply(row, 0.0))
        picked = masked if picked is None else im.add(picked, masked)
    return (pieces, im.mag(picked))


# Exports ####################################################################

__all__ = ('VERTICES', 'PIECES', 'ORDINATES', 'WHERE', 'SLOTS', 'neighbours_of',
           'centre_weights', 'basis', 'sub_weights', 'evaluate', 'bernstein')

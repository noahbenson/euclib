# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_fit.py
'''Fitting a line or a plane to a set of positions.

Both fits answer the same question of a cloud of points --- which direction does
it spread along, and which does it not --- and both are answered by the same
decomposition. The points are centred on their mean, and the centred matrix is
factored. The direction the points spread along most is the line's; the
direction they spread along least is the plane's normal.

The factorization is the singular value decomposition, and the sense in which
the fit is best is the *orthogonal* one: the sum of the squared distances from
the points to the line, or to the plane, is as small as it can be. That is not
the same as the ``y = mx + b`` fit that a regression on one coordinate gives. An
ordinary regression minimizes the distance measured along one axis, so it gives
a different answer if the axes are relabelled or the points are rotated, and it
has nothing to say about a vertical line. Fitting by distance has no preferred
direction, and it is defined whatever the points do.

A plane here is a *hyperplane*. In three dimensions that is the two-dimensional
plane one expects; in two it is the line orthogonal to the direction the points
spread along least, so `fit_plane` and `fit_line` in two dimensions describe the
same line --- one from its normal side and one from its length side.

A singular vector is defined only up to a sign, so both fits choose one: the
vector's largest-magnitude component is made positive. A fit that is *not*
determined --- a line through a single point, a plane through collinear points
--- is refused rather than answered with an arbitrary direction.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import abs as npabs, finfo, inexact, issubdtype

from immlib import math as imath

from ..abc import Geometry, as_query
from ._cross import positions_of


# Helpers ####################################################################

def _points_of(points, /):
    '''Returns the positions a fit is made to.

    A geometry contributes the positions its own data lives at, as it does for
    ``distance``: a simplex geometry's coordinates, a grid's cell centers, or a
    prism mesh's two surfaces. Anything else is taken to be a matrix of
    positions already.

    Parameters
    ----------
    points : Geometry or array-like
        A geometry, or a ``(D, Q)`` matrix of positions.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix of positions.
    '''
    if isinstance(points, Geometry):
        return positions_of(points)
    return as_query(points)


def _checked(points, /):
    '''Checks that a set of positions is a fit's to make.

    Parameters
    ----------
    points : array-like
        The candidate positions.

    Returns
    -------
    dim : int
        The number of rows, which is the dimension of the space.
    count : int
        The number of columns, which is the number of points.

    Raises
    ------
    ValueError
        If the positions are not a matrix, or are not two- or
        three-dimensional.
    '''
    shape = getattr(points, 'shape', None)
    if shape is None or len(tuple(shape)) != 2:
        raise ValueError(
            f"a fit needs a (D, Q) matrix of positions --- D rows, one column"
            f" per point; found shape {tuple(shape) if shape is not None else None}")
    (dim, count) = (int(shape[0]), int(shape[1]))
    if dim not in (2, 3):
        raise ValueError(
            f"a fit needs positions in two or three dimensions; found {dim}")
    return (dim, count)


def _decomposed(bare, /):
    '''The centred singular value decomposition of a set of magnitudes.

    The decomposition is of the *magnitudes*, because a singular value
    decomposition has no meaning for a quantity: its parts would carry
    different units, and ``immlib`` refuses one.

    Parameters
    ----------
    bare : array-like
        A ``(D, Q)`` matrix of positions, without units.

    Returns
    -------
    u : array-like
        A ``(D, min(D, Q))`` matrix whose columns are the directions, ordered
        by how far the points spread along each.
    s : array-like
        A length-``min(D, Q)`` vector of those spreads: the singular values,
        in descending order.
    '''
    centred = bare - bare.sum(axis=1, keepdims=True) / bare.shape[1]
    (u, s, _vh) = imath.svd(centred, full_matrices=False)
    return (u, s)


def _is_tensor(x, /):
    '''Whether a value belongs to a PyTorch tensor's backend.'''
    return type(x).__module__.split('.')[0] == 'torch'


def _eps(like, /):
    '''The machine epsilon of a value's floating-point type.'''
    if _is_tensor(like):
        from torch import finfo as torch_finfo
        return float(torch_finfo(like.dtype).eps)
    dtype = getattr(like, 'dtype', None)
    if dtype is None or not issubdtype(dtype, inexact):
        dtype = float
    return float(finfo(dtype).eps)


def _scalar(x, /):
    '''A scalar of either backend, as a Python float.

    A tensor is detached first. These numbers decide whether a fit is
    determined and which way a direction points, and neither is a value a
    caller can differentiate: reading them must not warn about cutting the
    graph, and the directions the decision is applied to keep theirs.
    '''
    return float(x.detach()) if _is_tensor(x) else float(x)


def _absmax(x, /):
    '''The largest absolute value, as a Python float.'''
    if _is_tensor(x):
        return float(x.detach().abs().max())
    return float(npabs(x).max())


def _orient(vector, /):
    '''Flips a direction so its largest-magnitude component is positive.

    A singular vector is defined up to a sign, and a fit that answered with one
    of the two at random would not be reproducible. The largest-magnitude
    component is the one least changed by a rounding error, which is why the
    sign is read from it rather than from, say, the first. ``argmax`` breaks a
    tie by the first index, so two runs of the same fit agree.

    Parameters
    ----------
    vector : array-like
        A direction from a singular value decomposition.

    Returns
    -------
    array-like
        The same direction, or its negation, of the same backend.
    '''
    magnitudes = vector.abs() if _is_tensor(vector) else npabs(vector)
    index = int(magnitudes.argmax())
    return vector if _scalar(vector[index]) >= 0.0 else -vector


# Operations #################################################################

def fit_line(points, /):
    '''Fits a line to a set of positions by orthogonal distance.

    The line is the one the points have the least sum of squared distances to.
    Its origin is the points' mean, which is the point on the line nearest the
    cloud as a whole, and its direction is the direction the points spread
    along most.

    Parameters
    ----------
    points : Geometry or array-like
        The points to fit, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    origin : array-like
        A length-``D`` position on the line: the mean of the points. It has
        the units and the backend of ``points``.
    direction : array-like
        A length-``D`` unit vector along the line, without units and in the
        backend of ``points``. Its largest-magnitude component is positive.

    Raises
    ------
    ValueError
        If ``points`` is not a ``(D, Q)`` matrix with ``D`` equal to 2 or 3, if
        there are fewer than two points, or if every point is the same point,
        which leaves the direction undetermined.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import fit_line
    >>> # Four points on the line y = 2 x + 1.
    >>> points = np.array([[0., 1., 2., 3.], [1., 3., 5., 7.]])
    >>> (origin, direction) = fit_line(points)
    >>> origin
    array([1.5, 4. ])
    >>> direction.round(6)
    array([0.447214, 0.894427])
    '''
    points = _points_of(points)
    (dim, count) = _checked(points)
    if count < 2:
        raise ValueError(
            f"a line needs at least two points; found {count}")
    bare = imath.mag(points)
    (u, s) = _decomposed(bare)
    # The first singular value is how far the points reach along the line. If
    # they do not reach at all --- every point is the same point --- then any
    # direction is as good as any other, so none is reported.
    if _scalar(s[0]) <= _eps(s) * max(dim, count) * _absmax(bare):
        raise ValueError(
            "a line is not determined when every point is the same point")
    origin = points.sum(axis=1) / count
    return (origin, _orient(u[:, 0]))


def fit_plane(points, /):
    '''Fits a plane (a hyperplane) to a set of positions by orthogonal distance.

    The plane is the one the points have the least sum of squared distances to.
    Its origin is the points' mean and its normal is the direction the points
    spread along least. In two dimensions the plane is a line, so the normal is
    that line's direction turned a quarter turn.

    Parameters
    ----------
    points : Geometry or array-like
        The points to fit, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    origin : array-like
        A length-``D`` position on the plane: the mean of the points. It has
        the units and the backend of ``points``.
    normal : array-like
        A length-``D`` unit vector orthogonal to the plane, without units and
        in the backend of ``points``. Its largest-magnitude component is
        positive.

    Raises
    ------
    ValueError
        If ``points`` is not a ``(D, Q)`` matrix with ``D`` equal to 2 or 3, if
        there are fewer than ``D`` points, or if the points all lie on a line,
        which leaves the plane undetermined.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import fit_plane
    >>> # A square in the plane z = 2.
    >>> points = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.],
    ...                    [2., 2., 2., 2.]])
    >>> (origin, normal) = fit_plane(points)
    >>> origin
    array([0.5, 0.5, 2. ])
    >>> normal.round(6)
    array([0., 0., 1.])
    '''
    points = _points_of(points)
    (dim, count) = _checked(points)
    if count < dim:
        raise ValueError(
            f"a plane in {dim} dimensions needs at least {dim} points;"
            f" found {count}")
    bare = imath.mag(points)
    (u, s) = _decomposed(bare)
    # The smallest spread is how far the points reach out of the plane. It is
    # unique only when the points span one dimension less than the space: a
    # plane through collinear points is a whole pencil of planes, all of them
    # fitting equally well.
    if _scalar(s[dim - 2]) <= _eps(s) * max(dim, count) * _scalar(s[0]):
        raise ValueError(
            f"a plane in {dim} dimensions is not determined by points that all"
            f" lie on a line")
    origin = points.sum(axis=1) / count
    return (origin, _orient(u[:, -1]))


# Exports ####################################################################

__all__ = ('fit_line', 'fit_plane')

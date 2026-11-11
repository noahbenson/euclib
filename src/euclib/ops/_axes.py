# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_axes.py
'''The directions a set of positions is aligned along, and the box it fills.

A cloud of points has no orientation of its own, but which way it is stretched
does: the directions it spreads along most, and least, are its *principal axes*.
They are the singular vectors of the centred positions --- the same
decomposition ``fit_line`` takes one vector from and ``fit_plane`` another ---
ordered here by how far the points spread along each.

An oriented bounding box is the smallest box, whose faces follow those axes,
that holds every point. It is not the smallest box of *any* orientation ---
finding that one is a different problem --- so the name is exact: the box is
oriented along the points' own axes, and its extents are the points' ranges
along them.

Both operations are the same decomposition, so they refuse the same way. A
single point spreads nowhere and fixes no axis; points on a line in three
dimensions fix one axis and leave the two orthogonal to it arbitrary. Neither
is answered with an invented orientation.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy.linalg import det

from immlib import math as imath, to_array

from ._fit import (
    _absmax, _checked, _decomposed, _eps, _is_tensor, _points_of, _scalar,
    _sign_of)


# Helpers ####################################################################

def _with_units(value, points, /):
    '''Gives a bare value the units the positions it came from carried.

    The arithmetic is done on the magnitudes, because the decomposition has no
    meaning for a quantity, and the units are put back at the end. They are
    taken from the positions themselves rather than from a registry of this
    module's own, so that a caller's unit registry is the one the answer
    belongs to.
    '''
    return value * points.units if hasattr(points, 'units') else value


def _stacked(columns, signs, /):
    '''Stacks signed columns into a matrix of either backend.'''
    if _is_tensor(columns[0]):
        from torch import stack as stack_tensors
        return stack_tensors([c * s for (c, s) in zip(columns, signs)], dim=1)
    from numpy import stack as stack_arrays
    return stack_arrays([c * s for (c, s) in zip(columns, signs)], axis=1)


def _framed(u, dim, /):
    '''The right-handed principal frame from a decomposition's left vectors.

    Each axis is oriented so its largest-magnitude component is positive, as
    ``fit_line`` and ``fit_plane`` orient theirs. Three axes have eight sign
    combinations and only four of them form a right-handed frame, so the last
    axis is flipped when the frame that the rule chose is left-handed. That
    keeps a full set of axes a *rotation*, which is what a caller who uses them
    as one needs.

    Parameters
    ----------
    u : array-like
        The ``(D, D)`` matrix of left singular vectors.
    dim : int
        The number of axes to take: the leading ``dim`` columns of ``u``.

    Returns
    -------
    array-like
        A ``(D, D)`` matrix whose columns are the axes, of the same backend.
    '''
    columns = [u[:, i] for i in range(dim)]
    signs = [_sign_of(column) for column in columns]
    frame = _stacked(columns, signs)
    if float(det(to_array(frame, detach=True))) < 0.0:
        signs = signs[:-1] + [-signs[-1]]
        frame = _stacked(columns, signs)
    return frame


def _ranges(values, /):
    '''The smallest and largest value in each row, for either backend.'''
    if _is_tensor(values):
        return (values.amin(axis=1), values.amax(axis=1))
    return (values.min(axis=1), values.max(axis=1))


def _principal(points, subject, /):
    '''The left singular vectors and centred magnitudes of a point set.

    The singular vectors are returned whole rather than as an assembled frame,
    because the frame depends on the dimension and on a sign convention that a
    caller may not want. The centred magnitudes are returned as well, since the
    box needs the projections it would otherwise recompute.

    Parameters
    ----------
    points : array-like
        A ``(D, Q)`` matrix of positions, without units.
    subject : str
        What the caller asked for, as a phrase with its article, so that a
        refusal names the operation.

    Returns
    -------
    u : array-like
        The ``(D, min(D, Q))`` matrix of left singular vectors.
    centred : array-like
        The positions with their mean subtracted.

    Raises
    ------
    ValueError
        If there are no points, if every point is the same point, or if points
        in three dimensions all lie on a line.
    '''
    (dim, count) = _checked(points, subject)
    if count < 1:
        raise ValueError(f"{subject} needs at least one point; found {count}")
    bare = imath.mag(points)
    (centred, u, s) = _decomposed(bare)
    tolerance = _eps(s) * max(dim, count)
    # With no spread at all, every direction is as good as every other one.
    if _scalar(s[0]) <= tolerance * _absmax(bare):
        raise ValueError(
            "the principal axes are not determined when every point is the"
            " same point")
    # One spread determines one axis. In two dimensions the other axis is what
    # is left over, and it is determined; in three, the two that are left over
    # span a plane, and any pair in it would do.
    if dim == 3 and _scalar(s[1]) <= tolerance * _scalar(s[0]):
        raise ValueError(
            "points on a line in three dimensions determine one principal axis,"
            " but not the two orthogonal to it")
    return (u, centred)


# Operations #################################################################

def principal_axes(points, /):
    '''Finds the directions a set of positions is aligned along.

    The principal axes are the directions the points spread along, ordered by
    how far they spread: the first is the direction of greatest spread and the
    last of least. They are mutually orthogonal, and they are the singular
    vectors of the centred positions --- the first is the direction
    ``fit_line`` reports and the last is the normal ``fit_plane`` reports.

    Each axis is oriented so its largest-magnitude component is positive, and
    the last is then flipped if that is what makes the frame right-handed. The
    axes therefore form a rotation, not a reflection.

    Parameters
    ----------
    points : Geometry or array-like
        The points to analyze, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    origin : array-like
        A length-``D`` position: the mean of the points, which is the point the
        spreads are measured about. It has the units and the backend of
        ``points``.
    axes : array-like
        A ``(D, D)`` matrix whose columns are the axes, ordered by descending
        spread and forming a right-handed frame. It has no units and the
        backend of ``points``.

    Raises
    ------
    ValueError
        If ``points`` is not a ``(D, Q)`` matrix with ``D`` equal to 2 or 3, if
        it holds no points, if every point is the same point, or if points in
        three dimensions all lie on a line, which leaves the two axes
        orthogonal to it undetermined.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import principal_axes
    >>> # The corners of a rectangle twice as wide as it is tall.
    >>> points = np.array([[0., 2., 2., 0.], [0., 0., 1., 1.]])
    >>> (origin, axes) = principal_axes(points)
    >>> origin
    array([1. , 0.5])
    >>> bool(np.allclose(axes, np.eye(2)))
    True
    '''
    points = _points_of(points)
    (u, _centred) = _principal(points, 'a set of principal axes')
    axes = _framed(u, int(points.shape[0]))
    origin = _with_units(
        imath.mag(points).sum(axis=1) / int(points.shape[1]), points)
    return (origin, axes)


def oriented_bounding_box(points, /):
    '''Finds the box, aligned with the principal axes, that holds every point.

    The box's faces are perpendicular to the points' principal axes. Its extent
    along each axis is the distance between the points' extreme projections
    onto that axis, and its center is the midpoint of those extremes, so every
    point is inside it. It is the smallest box whose faces follow the
    directions the points themselves suggest; a smaller box of some other
    orientation may exist, and finding that one is a different problem.

    Parameters
    ----------
    points : Geometry or array-like
        The points to enclose, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    center : array-like
        The length-``D`` center of the box, with the units and the backend of
        ``points``.
    axes : array-like
        A ``(D, D)`` matrix whose columns are the box's axes, as
        ``principal_axes`` returns them: ordered by descending spread, right-
        handed, and without units.
    extents : array-like
        The length-``D`` full width of the box along each axis, with the units
        and the backend of ``points``. A position ``p`` is inside the box when
        ``axes[:, i] . (p - center)`` lies between ``-extents[i] / 2`` and
        ``extents[i] / 2`` for every ``i``.

    Raises
    ------
    ValueError
        Under the same conditions as ``principal_axes``.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import oriented_bounding_box
    >>> # A rectangle twice as wide as it is tall, turned a quarter turn.
    >>> rectangle = np.array([[0., 2., 2., 0.], [0., 0., 1., 1.]])
    >>> turn = np.array([[0., -1.], [1., 0.]])
    >>> (center, axes, extents) = oriented_bounding_box(turn @ rectangle)
    >>> center
    array([-0.5,  1. ])
    >>> extents
    array([2., 1.])
    >>> round(float(np.linalg.det(axes)), 6)
    1.0
    '''
    points = _points_of(points)
    (u, centred) = _principal(points, 'an oriented bounding box')
    dim = int(points.shape[0])
    axes = _framed(u, dim)
    # The projection of each point onto each axis: element (i, q) is how far
    # point q reaches along axis i. The extremes of a row are the box's two
    # faces perpendicular to that axis.
    projections = (axes.T[:, :, None] * centred[None, :, :]).sum(axis=1)
    (low, high) = _ranges(projections)
    extents = high - low
    displacement = (axes * ((high + low) * 0.5)[None, :]).sum(axis=1)
    center = imath.mag(points).sum(axis=1) / int(points.shape[1]) + displacement
    return (_with_units(center, points), axes, _with_units(extents, points))


# Exports ####################################################################

__all__ = ('oriented_bounding_box', 'principal_axes')

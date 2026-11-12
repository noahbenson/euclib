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

A *minimum-volume* box is the smallest box of any orientation, and it is found
differently. A smallest box always has one face flush with a face of the
points' convex hull, so the hull's faces supply the candidate orientations.
For each candidate the points are projected onto the face's plane, where the
best rectangle is the smallest one around a planar point set, and the box is
the rectangle extended across the hull's depth. Trying every face and keeping
the smallest box is what makes the answer the minimum, rather than one
orientation's idea of small.

Most of these operations refuse the same way. A single point spreads nowhere
and fixes no axis; points on a line in three dimensions fix one axis and leave
the two orthogonal to it arbitrary. Neither is answered with an invented
orientation. The minimum-volume box needs the points to span their space for
the same reason: a flat set's box is flat, and a flat box is not a box.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    abs as npabs, argsort, asarray, cross, eye, roll, sqrt)
from numpy.linalg import det

from immlib import math as imath, to_array

from ._fit import (
    _absmax, _checked, _decomposed, _eps, _is_tensor, _points_of, _scalar,
    _sign_of)
from ._hull import _Flat, _bare, _closed, _flat, _quickhull, _tolerance


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


# Minimum-volume boxes #######################################################

def _rectangle_of(points, /):
    '''The directions of the smallest rectangle around a planar point set.

    One side of the smallest rectangle is flush with an edge of the set's
    convex hull, so the candidate directions are the hull's edges. For each,
    the points' ranges along that edge and across it give a rectangle, and the
    smallest of those rectangles is the smallest rectangle of any direction.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(2, Q)`` matrix of positions.

    Returns
    -------
    tuple of numpy.ndarray or None
        Two perpendicular unit directions --- along the rectangle and across
        it --- or ``None`` if the points do not span a plane, which leaves no
        rectangle to find.
    '''
    corners = _flat(points)
    if len(corners) < 3:
        return None
    hull = points[:, corners]
    edges = roll(hull, -1, axis=1) - hull
    lengths = sqrt((edges * edges).sum(axis=0))
    keep = lengths > 0.0
    if not keep.any():
        return None
    edges = edges[:, keep] / lengths[keep]
    across = asarray([-edges[1], edges[0]])
    along_span = edges.T @ hull
    across_span = across.T @ hull
    areas = ((along_span.max(axis=1) - along_span.min(axis=1))
             * (across_span.max(axis=1) - across_span.min(axis=1)))
    best = int(areas.argmin())
    return (edges[:, best], across[:, best])


def _ordered(axes, extents, /):
    '''A box's axes ordered by extent and wound right-handed.

    The minimum-volume search has no preferred order for the box's three
    directions: they are the directions that made the box smallest, and any
    permutation of them describes it. Ordering by descending extent is the
    convention ``oriented_bounding_box`` uses, and it makes the answer
    reproducible. A permutation may turn a right-handed frame into a
    left-handed one, so the last axis is flipped when it does.

    Parameters
    ----------
    axes : numpy.ndarray
        A ``(D, D)`` matrix whose columns are the box's axes.
    extents : numpy.ndarray
        A length-``D`` vector of the box's widths along those axes.

    Returns
    -------
    axes : numpy.ndarray
        The axes, signed consistently and ordered by descending extent.
    extents : numpy.ndarray
        The extents, in the same order.
    '''
    signs = asarray([_sign_of(axes[:, i]) for i in range(axes.shape[1])])
    axes = axes * signs[None, :]
    order = argsort(-extents, kind='stable')
    axes = axes[:, order]
    extents = extents[order]
    if float(det(axes)) < 0.0:
        axes[:, -1] = -axes[:, -1]
    return (axes, extents)


def _minimum_box(points, dim, /):
    '''The smallest box of any orientation that holds a set of positions.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(D, Q)`` matrix of positions, without units.
    dim : int
        The number of dimensions, two or three.

    Returns
    -------
    axes : numpy.ndarray
        The box's axes, ordered and right-handed.
    extents : numpy.ndarray
        The box's widths along those axes.

    Raises
    ------
    ValueError
        If the points do not span their space, so that no box of the space's
        own dimension exists.
    '''
    if dim == 2:
        frame = _rectangle_of(points)
        if frame is None:
            raise ValueError(
                "a minimum-volume bounding box in two dimensions needs three"
                " points that are not all on one line")
        axes = asarray(frame).T
        projections = axes.T @ points
        extents = (projections.max(axis=1) - projections.min(axis=1))
        return _ordered(axes, extents)
    try:
        (used, tris) = _quickhull(points, _tolerance(points, 3,
                                                     points.shape[1]))
    except _Flat:
        raise ValueError(
            "a minimum-volume bounding box in three dimensions needs four"
            " points that are not all in one plane") from None
    if not _closed(tris):
        raise ValueError(
            "the points are too nearly coplanar for a minimum-volume bounding"
            " box to be built reliably")
    corners = points[:, used]
    best = None
    seen = set()
    for k in range(tris.shape[1]):
        normal = cross(corners[:, tris[1, k]] - corners[:, tris[0, k]],
                       corners[:, tris[2, k]] - corners[:, tris[0, k]])
        length = float(sqrt(normal @ normal))
        if length == 0.0:
            continue
        normal = normal / length
        key = tuple(float(v) for v in (normal * 1e9).round())
        if key in seen:
            continue
        seen.add(key)
        # A frame whose last axis is the face's normal. The coordinate axis
        # the normal leans on least is the safe one to cross it with.
        helper = eye(3)[int(npabs(normal).argmin())]
        first = cross(normal, helper)
        first = first / sqrt(float(first @ first))
        second = cross(normal, first)
        frame = _rectangle_of(
            asarray([first @ corners, second @ corners]))
        if frame is None:
            continue
        (along, across) = frame
        axes = asarray([first * along[0] + second * along[1],
                        first * across[0] + second * across[1],
                        normal]).T
        projections = axes.T @ points
        extents = projections.max(axis=1) - projections.min(axis=1)
        volume = float(extents.prod())
        if best is None or volume < best[0]:
            best = (volume, axes, extents)
    if best is None:
        raise ValueError(
            "the points do not determine a minimum-volume bounding box")
    return _ordered(best[1], best[2])


def minimum_volume_bounding_box(points, /):
    '''Finds the smallest box of any orientation that holds every point.

    Unlike ``oriented_bounding_box``, whose faces follow the points' principal
    axes, this box is the smallest one in *any* orientation. It is found by
    taking the convex hull's faces as candidate orientations --- a smallest box
    always has a face flush with one of them --- and, for each, finding the
    smallest rectangle around the hull's projection onto that face's plane.
    Trying every face and keeping the smallest box is what makes this the
    minimum rather than one orientation's approximation of it.

    In two dimensions the box is the smallest rectangle, and "volume" is its
    area. The same search applies: a side of the smallest rectangle is flush
    with an edge of the hull, so the hull's edges are the candidates.

    The axes are ordered by descending extent, as
    ``oriented_bounding_box`` orders its own, and they form a right-handed
    frame. The choice of box is combinatorial --- which hull face wins is a
    decision --- so the answer is computed on NumPy arrays; a tensor input is
    read, but the result does not carry a gradient.

    Parameters
    ----------
    points : Geometry or array-like
        The points to enclose, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    center : numpy.ndarray
        The length-``D`` center of the box, with the units of ``points`` if it
        has any.
    axes : numpy.ndarray
        A ``(D, D)`` matrix whose columns are the box's axes, without units,
        ordered by descending extent and forming a right-handed frame.
    extents : numpy.ndarray
        The length-``D`` full width of the box along each axis, with the units
        of ``points`` if it has any. A position ``p`` is inside the box when
        ``axes[:, i] . (p - center)`` lies between ``-extents[i] / 2`` and
        ``extents[i] / 2`` for every ``i``.

    Raises
    ------
    ValueError
        If ``points`` is not a ``(D, Q)`` matrix with ``D`` equal to 2 or 3, if
        there are fewer points than the space needs to be spanned --- three in
        a plane, four in space --- or if the points do not span it. A flat set
        has a flat box, and a flat box is not a box of the space's dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import minimum_volume_bounding_box
    >>> # A box twice as wide as it is tall, and three times as deep.
    >>> cube = np.array([[x, y, z] for x in (0., 2.) for y in (0., 1.)
    ...                  for z in (0., 3.)]).T
    >>> (center, axes, extents) = minimum_volume_bounding_box(cube)
    >>> center
    array([1. , 0.5, 1.5])
    >>> extents
    array([3., 2., 1.])
    >>> round(float(np.linalg.det(axes)), 6)
    1.0
    '''
    points = _points_of(points)
    (dim, count) = _checked(points, 'a minimum-volume bounding box')
    least = dim + 1
    if count < least:
        raise ValueError(
            f"a minimum-volume bounding box needs at least {least} points;"
            f" found {count}")
    bare = _bare(points)
    (axes, _extents) = _minimum_box(bare, dim)
    projections = axes.T @ bare
    low = projections.min(axis=1)
    high = projections.max(axis=1)
    center = (axes * ((low + high) * 0.5)[None, :]).sum(axis=1)
    return (_with_units(center, points), axes,
            _with_units(high - low, points))


# Exports ####################################################################

__all__ = ('minimum_volume_bounding_box', 'oriented_bounding_box',
           'principal_axes')

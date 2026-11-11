# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_hull.py
'''The smallest convex geometry that contains a set of positions.

The convex hull of a set of points is the smallest convex set that holds all of
them, and its boundary is what a caller usually wants: the polygon or
polyhedron that wraps the points. In a plane the boundary is a polygon, found
here by Andrew's monotone chain; in space it is a polyhedron, found here by
quickhull.

The two are returned as the geometry that says what they are: a closed path
through the polygon's vertices, and a triangle mesh whose triangles wind
outward. A caller can then measure the hull, ask what it contains, or use it as
any other geometry --- none of which a matrix of indices would allow.

A hull is only a solid when the points span the space they are in. Points on a
line in two dimensions, or in a plane in three, have a hull of their own, but it
is one dimension flatter than the space, and it is refused rather than returned
as a degenerate solid. Projecting the points into that flat space, with the
principal axes, is what a caller who wants it can do. Points that are coplanar
to within the arithmetic's own rounding error are refused for the same reason:
the answer would be a surface whose faces the comparison joined inconsistently,
and a refusal is better than that.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    abs as npabs, arange, argmax, array, asarray, concatenate, cross, empty,
    lexsort, ptp, roll, sqrt)

from immlib import math as imath, to_array

from ..types import SegPath, SegTopology, TriMesh, TriTopology
from ._fit import _checked, _eps, _points_of


# Helpers ####################################################################

def _bare(points, /):
    '''The positions as plain numbers, which is what the hull is built from.'''
    return to_array(imath.mag(points), detach=True)


def _tolerance(points, dim, count, /):
    '''How far from a face a point may sit and still count as on it.

    A hull is decided by comparisons --- is a point outside this face, is this
    triple of points on one line --- and each of them is a floating-point
    number that should have been zero. The tolerance scales with the extent of
    the points, because a set of points a micrometer across has a micrometer's
    worth of rounding error, and with the number of points, because more of
    them make more chances to round.
    '''
    extent = float(ptp(points, axis=1).max())
    return _eps(points) * max(dim, count) * extent


def _assign(points, candidates, faces, tris, normals, offsets, tol, /):
    '''Sorts candidate points onto the faces they are outside of.

    Each point is given to the face it is farthest outside of, and a point that
    is outside no face --- one that is inside the hull, or on it --- is given to
    none. Assigning each point to its farthest face is what keeps the number of
    point-face tests down when the hull is built: a face only has to be tested
    against the points that are near it.
    '''
    if candidates.size == 0:
        return {}
    normal = asarray([normals[f] for f in faces])
    offset = asarray([offsets[f] for f in faces])
    distances = normal @ points[:, candidates] - offset[:, None]
    nearest = distances.argmax(axis=0)
    outside = distances.max(axis=0) > tol
    assigned = {}
    for (i, face) in enumerate(faces):
        chosen = candidates[(nearest == i) & outside]
        if chosen.size:
            assigned[face] = chosen
    return assigned


def _flat(points, /):
    '''The counterclockwise corners of a planar point set's hull.

    Andrew's monotone chain: sort the points by one coordinate and then the
    other, and walk them twice, once building the lower boundary and once the
    upper, popping a corner whenever the walk turns the wrong way. The points
    that survive both walks are the hull's corners, in counterclockwise order
    and with points that lie on an edge between two others left out --- they
    are on the hull but they are not corners of it.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(2, Q)`` matrix of positions.

    Returns
    -------
    list of int
        The columns of ``points`` that are the hull's corners, in
        counterclockwise order.
    '''
    order = [int(i) for i in lexsort((points[1], points[0]))]

    def turn(origin, first, second, /):
        '''Twice the signed area of the triangle, positive when it turns left.'''
        return ((points[0, first] - points[0, origin])
                * (points[1, second] - points[1, origin])
                - (points[1, first] - points[1, origin])
                * (points[0, second] - points[0, origin]))

    lower = []
    for i in order:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], i) <= 0.0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(order):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], i) <= 0.0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


class _Flat(Exception):
    '''Raised internally when the points do not span three dimensions.'''


def _initial_tetrahedron(points, tol, /):
    '''Four points that span space, ordered so the tetrahedron is positive.

    The four are the point farthest along each new direction from the ones
    already chosen: a point, the point farthest from it, the point farthest
    from their line, and the point farthest from their plane. Each step is what
    tests whether the points span the next dimension, so a set that is
    coincident, collinear, or coplanar fails at the step it fails to span, and
    the caller learns that no hull solid exists.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of positions.
    tol : float
        The distance below which a spread counts as none.

    Returns
    -------
    list of int
        The four column indices, ordered so that the tetrahedron they form has
        positive volume.

    Raises
    ------
    _Flat
        If the points are coincident, collinear, or coplanar, so that no four
        of them span space.
    '''
    first = int(lexsort((points[2], points[1], points[0]))[0])
    from_first = points - points[:, [first]]
    second = int(argmax((from_first * from_first).sum(axis=0)))
    axis = points[:, second] - points[:, first]
    axis_squared = float(axis @ axis)
    # Every point is the same point: there is no line to measure from, let
    # alone a plane, and the division below would have nothing to divide by.
    if axis_squared <= tol * tol:
        raise _Flat()
    along = (axis[:, None] * from_first).sum(axis=0)
    # The squared distance from the line, with the division by |axis|² left as
    # a scale factor rather than applied to every point.
    squared = ((from_first * from_first).sum(axis=0) * axis_squared
               - along * along)
    third = int(argmax(squared))
    if sqrt(max(float(squared[third]), 0.0) / axis_squared) <= tol:
        raise _Flat()
    plane = cross(points[:, second] - points[:, first],
                  points[:, third] - points[:, first])
    height = ((plane[:, None] * from_first).sum(axis=0)
              / sqrt(float(plane @ plane)))
    fourth = int(argmax(npabs(height)))
    if abs(float(height[fourth])) <= tol:
        raise _Flat()
    corners = [first, second, third, fourth]
    if float(height[fourth]) < 0.0:
        # The fourth point is below the plane, which is the negative side. A
        # tetrahedron is positive when it is above it, so the two that define
        # the plane swap.
        corners[1], corners[2] = corners[2], corners[1]
    return corners


def _closed(tris, /):
    '''Whether a triangle set's edges are each matched by a neighbour.

    A surface that closes on itself traverses every edge once in each
    direction, because the two triangles that share it wind opposite ways along
    it. Checking that is how a hull that the arithmetic has spoilt is caught
    rather than returned: the faces are joined by comparing distances against
    zero, and a set of points that is flat to within the rounding error of the
    comparison can leave two faces joined along an edge they wind the same way.

    Parameters
    ----------
    tris : numpy.ndarray
        A ``(3, M)`` matrix of triangles.

    Returns
    -------
    bool
        ``True`` if every directed edge appears once and its reverse does too.
    '''
    seen = {}
    for k in range(tris.shape[1]):
        (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(count == 1 and (edge[1], edge[0]) in seen
               for (edge, count) in seen.items())


def _quickhull(points, tol, /):
    '''The triangles of a spatial point set's hull, wound outward.

    Quickhull starts from a tetrahedron known to be inside the hull and then,
    for as long as some point is outside it, takes the point farthest from any
    face, removes every face that point can see, and joins the point to the
    ring of edges that borders what was removed. Only points that are outside a
    removed face are reconsidered; every other point was either inside the hull
    already or is inside the part that survives, so the work shrinks as the
    hull grows.

    The ring is what makes the result watertight. The faces that a point can
    see form a patch of the surface; the edges of the patch that are shared by
    a face outside it are the boundary of the patch, and each of them, joined
    to the new point, closes the surface again. The faces are wound so that
    their normals point away from the hull's interior, which makes the new
    triangles wind that way too when the ring's edges are taken in the order
    the removed faces gave them.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of positions.
    tol : float
        The distance below which a point counts as on a face rather than
        outside it.

    Returns
    -------
    used : numpy.ndarray
        The columns of ``points`` that are the hull's corners, in ascending
        order.
    tris : numpy.ndarray
        A ``(3, M)`` matrix of triangles, each a triple of indices into
        ``used``.
    '''
    corners = _initial_tetrahedron(points, tol)
    (p0, p1, p2, p3) = corners
    tris = [(p0, p2, p1), (p0, p1, p3), (p0, p3, p2), (p1, p2, p3)]
    normals = []
    offsets = []
    outside = []
    alive = []
    for (a, b, c) in tris:
        normal = cross(points[:, b] - points[:, a],
                       points[:, c] - points[:, a])
        length = float(sqrt(normal @ normal))
        if length > 0.0:
            normal = normal / length
            offset = float(normal @ points[:, a])
        else:
            normal = normal * 0.0
            offset = 0.0
        normals.append(normal)
        offsets.append(offset)
        outside.append(empty(0, dtype=int))
        alive.append(True)
    rest = arange(points.shape[1])
    rest = rest[(rest != p0) & (rest != p1) & (rest != p2) & (rest != p3)]
    initial = _assign(points, rest, [0, 1, 2, 3], tris, normals, offsets,
                      tol)
    for (face, chosen) in initial.items():
        outside[face] = chosen
    while True:
        pending = [f for f in range(len(tris)) if alive[f]
                   and outside[f].size > 0]
        if not pending:
            break
        face = pending[0]
        distances = (normals[face] @ points[:, outside[face]]
                     - offsets[face])
        point = int(outside[face][int(argmax(distances))])
        live = [f for f in range(len(tris)) if alive[f]]
        normal = asarray([normals[f] for f in live])
        offset = asarray([offsets[f] for f in live])
        visible = [live[i] for i in
                   asarray(normal @ points[:, point] - offset > tol)
                   .nonzero()[0]]
        # The edges of the visible patch, each counted as the winding of the
        # face that gave it. An edge inside the patch appears once each way;
        # the ring is the edges that appear only once.
        edges = {}
        for f in visible:
            (a, b, c) = tris[f]
            for (u, v) in ((a, b), (b, c), (c, a)):
                edges[(u, v)] = edges.get((u, v), 0) + 1
        ring = [(u, v) for (u, v) in edges if (v, u) not in edges]
        orphans = concatenate([outside[f] for f in visible])
        for f in visible:
            alive[f] = False
            outside[f] = empty(0, dtype=int)
        fresh = []
        for (u, v) in ring:
            normal = cross(points[:, v] - points[:, u],
                           points[:, point] - points[:, u])
            length = float(sqrt(normal @ normal))
            if length > 0.0:
                normal = normal / length
                offset = float(normal @ points[:, u])
            else:
                # The new point is collinear with this edge. The triangle it
                # would make has no area, but the surface still needs it to
                # close, so it is kept with a zero normal, which no point is
                # ever outside of.
                normal = normal * 0.0
                offset = 0.0
            tris.append((u, v, point))
            normals.append(normal)
            offsets.append(offset)
            outside.append(empty(0, dtype=int))
            alive.append(True)
            fresh.append(len(tris) - 1)
        orphans = orphans[orphans != point]
        assigned = _assign(points, orphans, fresh, tris, normals, offsets, tol)
        for (f, chosen) in assigned.items():
            outside[f] = chosen
    faces = [tris[i] for i in range(len(tris)) if alive[i]]
    used = sorted({v for face in faces for v in face})
    index = {v: i for (i, v) in enumerate(used)}
    return (asarray(used), array([[index[v] for v in face]
                                  for face in faces], dtype=int).T)


# Operations #################################################################

def convex_hull(points, /):
    '''Finds the smallest convex geometry that contains a set of positions.

    The convex hull is the smallest convex set that holds every one of the
    points, and what is returned is its boundary: a polygon in a plane, a
    polyhedron in space. A point that lies on the boundary without being a
    corner of it --- one in the middle of a face, say --- is inside the hull
    rather than a part of its boundary, so the returned geometry has the
    hull's corners as its coordinates and none of the points that the corners
    already enclose.

    The hull is combinatorial: which points are its corners is a decision, and
    it does not move smoothly as the points move. It is computed from the
    positions' magnitudes, and the geometry that comes back is built on the
    positions themselves, so it has their units and their backend.

    Parameters
    ----------
    points : Geometry or array-like
        The points to enclose, as a ``(D, Q)`` matrix or as a geometry whose
        data positions are used.

    Returns
    -------
    hull : SegPath or TriMesh
        The hull's boundary. In two dimensions it is a ``SegPath`` through the
        hull's corners in counterclockwise order, with a segment closing the
        polygon. In three dimensions it is a ``TriMesh`` on the hull's corners,
        whose triangles wind counterclockwise seen from outside the hull.

    Raises
    ------
    ValueError
        If ``points`` is not a ``(D, Q)`` matrix with ``D`` equal to 2 or 3, if
        there are fewer points than the dimension needs --- three in a plane,
        four in space --- or if the points do not span the dimension: all on a
        line in two dimensions, or all in a plane in three. A hull that is
        flatter than the space it is asked about is not a solid, so it is
        refused rather than returned degenerate. The same goes for points that
        are coplanar to within the rounding error of the comparison that
        decides which faces a point is outside of: rather than return a surface
        whose faces are joined inconsistently, the hull is refused.

    Examples
    --------
    In a plane the hull is a polygon, and the point inside this square is not
    one of its corners:

    >>> import numpy as np
    >>> from euclib.ops import convex_hull
    >>> square = np.array([[0., 2., 2., 0., 1.], [0., 0., 2., 2., 1.]])
    >>> hull = convex_hull(square)
    >>> hull.coord_count
    4
    >>> hull.topo.simplex_count[1]
    4

    In space it is a triangle mesh; a cube's hull has eight corners and twelve
    triangles, and the point at its center is inside it:

    >>> cube = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.)
    ...                  for z in (0., 1.)]).T
    >>> cloud = np.concatenate([cube, [[0.5], [0.5], [0.5]]], axis=1)
    >>> hull = convex_hull(cloud)
    >>> hull.coord_count
    8
    >>> hull.topo.simplex_count[2]
    12
    '''
    points = _points_of(points)
    (dim, count) = _checked(points, 'a convex hull')
    least = dim + 1
    if count < least:
        raise ValueError(f"a convex hull needs at least {least} points;"
                         f" found {count}")
    bare = _bare(points)
    if dim == 2:
        corners = _flat(bare)
        if len(corners) < least:
            raise ValueError(
                "a convex hull in two dimensions needs three points that are"
                " not all on one line")
        coords = points[:, corners]
        loop = arange(len(corners))
        return SegPath(coords,
                       SegTopology(array([loop, roll(loop, -1)]),
                                   coord_count=len(corners)))
    try:
        (used, tris) = _quickhull(bare, _tolerance(bare, dim, count))
    except _Flat:
        raise ValueError(
            "a convex hull in three dimensions needs four points that are not"
            " all in one plane") from None
    if not _closed(tris):
        raise ValueError(
            "the points are too nearly coplanar for a convex hull to be built"
            " reliably")
    return TriMesh(points[:, used], TriTopology(tris, coord_count=len(used)))


# Exports ####################################################################

__all__ = ('convex_hull',)

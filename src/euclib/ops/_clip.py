# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_clip.py
'''Cutting a volume with a plane: the solid on one side, and the cut's curve.

A plane divides space in two, and it divides a solid into two solids. Asking
for one of them is *clipping*: the part of the solid on the plane's positive
side, with the cut closed by a flat cap so that the answer is itself a solid.
Asking for the curve where the plane meets the surface is *sectioning*: the
outline a caller would draw on the cutting plane.

The two questions have different answers and different machinery here.

A section is a set of segments, one for each triangle the plane crosses. The
triangle is cut where its edges meet the plane, and the piece of the plane
inside it is the segment between those two points. Triangles that lie wholly
in the plane contribute no segment, because they meet the plane over an area
rather than along a curve; that is the same convention ``mesh_intersections``
makes. The segments' ends are welded together, so a caller can walk the curve
from one segment to the next even though the segments themselves come back in
no particular order.

A clip is an intersection with a half-space, and a half-space is not a mesh.
What is done instead is to take a box that encloses everything on the kept side
of the plane and has one face *on* the plane, and to intersect the mesh with
that box. The box's other faces lie outside the mesh, so the only face that
takes part in the answer is the one on the cutting plane, and the part of it
inside the mesh is exactly the cap the operation needs. The box's extent along
the plane is chosen from the mesh's own, so the result does not depend on any
number the caller has to supply.

A clip needs a closed surface, because the solid on one side of a plane is only
a solid if the input was one. A section does not: the curve is defined for any
surface, open or closed, and a caller who wants the outline of an open sheet
gets it.
'''

# Dependencies ###############################################################

from __future__ import annotations

from itertools import combinations

from numpy import abs as npabs, asarray, cross, eye, float64, sqrt, zeros

from immlib import math as imath, to_array

from ..types import SegPath, SegTopology, TriMesh, TriTopology
from ._boolean import (
    _empty_like, _units_of, _vertices_of, _welded, boolean_intersection)
from ._intersect import DEFAULT_TOLERANCE


# The cutting plane ##########################################################

def _position(value, units, /):
    '''A point on the plane, as bare numbers in the mesh's own units.'''
    if units is not None and hasattr(value, 'units'):
        value = imath.mag(value, units)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size != 3:
        raise ValueError(
            "a cutting plane's origin must be a three-dimensional position;"
            f" found {bare.size} components")
    return bare


def _direction(value, /):
    '''The plane's unit normal, as bare numbers.'''
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size != 3:
        raise ValueError(
            "a cutting plane's normal must be a three-dimensional direction;"
            f" found {bare.size} components")
    length = float(sqrt(bare @ bare))
    if length == 0.0:
        raise ValueError(
            "a cutting plane's normal must not be the zero vector")
    return bare / length


def _checked(mesh, /):
    '''Checks that a mesh is one a plane can cut.'''
    if not isinstance(mesh, TriMesh):
        raise TypeError(f"expected a TriMesh; found {type(mesh)}")
    if mesh.dim != 3:
        raise ValueError(
            "clipping and sectioning are defined for triangle meshes in"
            f" three-dimensional space; found dimension {mesh.dim}")


def _tolerance(points, tolerance, /):
    '''The tolerance the operation is to use, scaled to the mesh's extent.

    The positions are passed in already reduced to their magnitudes, so this
    is ``tolerance_of`` without the geometry's bounding box, which may be a
    tensor whose gradient is intact and cannot be read as an array.
    '''
    if tolerance is not None:
        return float(tolerance)
    if points.size == 0:
        return DEFAULT_TOLERANCE
    span = float((points.max(axis=1) - points.min(axis=1)).max())
    return DEFAULT_TOLERANCE * max(span, 1.0)


# Clipping ###################################################################

#: The corners of a box, in the order the example pages use: the origin, then
#: the three single-axis corners, then the three two-axis corners, then the
#: far corner. Each entry says which of the box's three coordinates is high.
_BOX_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))

#: The twelve triangles of a box, wound outward for the corner order above.
#: The frame they are built in is right-handed, so this winding stays outward
#: under the mapping from the box's own coordinates to the mesh's.
_BOX_TRIS = asarray([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                     [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                     [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                    dtype='int64').T


def _half_space_box(points, origin, normal, tol, /):
    '''A box that holds the kept half-space out to the mesh and no further.

    One face of the box lies in the cutting plane and the box extends along
    the normal from there, far enough that every retained corner of the mesh is
    inside it. Its extent along the plane is the mesh's own extent, inflated by
    a margin, so that no part of the mesh reaches the box's other faces: those
    faces must not cut anything, or the answer would be bounded by a box the
    caller never asked for.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of the mesh's positions, without units.
    origin : numpy.ndarray
        A length-3 position on the cutting plane.
    normal : numpy.ndarray
        The plane's unit normal. The kept side is the one it points to.
    tol : float
        The tolerance the operation is using, which the margin exceeds.

    Returns
    -------
    TriMesh
        The box, wound outward, whose face at the cutting plane is the cap.
    '''
    # An orthonormal frame whose last axis is the normal. Any direction
    # orthogonal to the normal will do for the first, and taking the coordinate
    # axis the normal leans on least keeps the cross product away from zero.
    helper = eye(3)[int(npabs(normal).argmin())]
    first = cross(normal, helper)
    first = first / sqrt(float(first @ first))
    second = cross(normal, first)
    offsets = points - origin[:, None]
    along = first @ offsets
    across = second @ offsets
    depth = normal @ offsets
    extent = points.max(axis=1) - points.min(axis=1)
    margin = max(float(sqrt(extent @ extent)), 1.0) + tol
    low = (along.min() - margin, across.min() - margin, 0.0)
    high = (along.max() + margin, across.max() + margin,
            depth.max() + margin)
    local = asarray([[high[b] if high_corner[b] else low[b]
                      for b in range(3)] for high_corner in _BOX_CORNERS],
                    dtype=float64).T
    corners = (origin[:, None] + first[:, None] * local[0]
               + second[:, None] * local[1] + normal[:, None] * local[2])
    return TriMesh(corners, TriTopology(_BOX_TRIS, coord_count=8))


def clip(mesh, origin, normal, /, *, tolerance=None):
    '''Keeps the part of a solid on one side of a plane, and caps the cut.

    The plane is the set of positions ``p`` with
    ``normal . (p - origin) == 0``, and the kept half is the one the normal
    points into: a position is kept when ``normal . (p - origin) >= 0``. The
    cut leaves an opening in the surface, which is closed with a flat cap on
    the plane, so the result is a solid with the same kind of surface the input
    had.

    A plane that does not reach the mesh leaves it whole: if every position is
    on the kept side the mesh itself is returned, and if every position is on
    the other side the result is empty. A plane that does reach it cuts the
    mesh, and the result is a new mesh whose coordinates are NumPy arrays.

    Parameters
    ----------
    mesh : TriMesh
        The solid to clip: a closed triangle mesh in three-dimensional space.
        Its triangles need not wind outward, but they must close.
    origin : array-like
        A length-3 position on the cutting plane.
    normal : array-like
        A length-3 direction the plane is perpendicular to. The side it points
        to is the side that is kept. It need not be a unit vector.
    tolerance : float or None, optional
        How near a corner may be to the plane and still count as on it. The
        default, ``None``, is a small fraction of the mesh's extent.

    Returns
    -------
    TriMesh
        The clipped solid, capped on the cutting plane, with the units of
        ``mesh`` if it has any. It is empty when the plane leaves nothing on
        the kept side, and it is ``mesh`` itself when the plane cuts nothing
        away.

    Raises
    ------
    TypeError
        If ``mesh`` is not a triangle mesh.
    ValueError
        If the mesh occupies fewer than three dimensions, if it is not a
        closed surface --- the part of an open surface on one side of a plane
        is not a solid --- or if ``origin`` or ``normal`` is not a
        three-dimensional position or direction.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import clip
    >>> corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                     [0., 0., 1.], [1., 1., 0.], [1., 0., 1.],
    ...                     [0., 1., 1.], [1., 1., 1.]]).T
    >>> faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
    ...                   [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
    ...                   [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    >>> cube = trimesh(corners, faces)
    >>> top = clip(cube, np.array([0., 0., 0.5]), np.array([0., 0., 1.]))
    >>> top.coord_count
    12
    '''
    _checked(mesh)
    units = _units_of(mesh)
    if mesh.coord_count == 0:
        return _empty_like(mesh)
    points = _vertices_of(mesh)
    tol = _tolerance(points, tolerance)
    origin = _position(origin, units)
    normal = _direction(normal)
    depth = normal @ (points - origin[:, None])
    # The whole mesh on one side: there is nothing to cut, and no cap to make.
    if float(depth.max()) <= tol:
        return _empty_like(mesh)
    if float(depth.min()) >= -tol:
        return mesh
    box = _half_space_box(points, origin, normal, tol)
    return boolean_intersection(mesh, box, tolerance=tol)


# Sectioning #################################################################

def _empty_path(units, /):
    '''A path with no segments, carrying the units it is given.'''
    coords = zeros((3, 0))
    if units is not None:
        coords = coords * units
    return SegPath(coords, SegTopology(zeros((2, 0), dtype=int),
                                       coord_count=0))


def _segment_of(points, corners, depth, tol, /):
    '''The segment in which one triangle meets the plane, or ``None``.

    The plane meets a triangle in a point, a segment, or the whole triangle.
    The first and last contribute no curve, so only a segment is returned. A
    corner on the plane is a meeting point whether or not the triangle crosses,
    and the crossing points are computed on the edges whose ends lie on
    opposite sides.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of the mesh's positions.
    corners : numpy.ndarray
        The three column indices of the triangle.
    depth : numpy.ndarray
        A length-``Q`` vector of signed distances from the plane.
    tol : float
        How near the plane a corner may lie and count as on it.

    Returns
    -------
    tuple of numpy.ndarray or None
        The segment's two ends, or ``None`` when the meeting is not a segment.
    '''
    distances = depth[corners]
    on = npabs(distances) <= tol
    if bool(on.all()):
        return None
    found = []
    for i in range(3):
        j = (i + 1) % 3
        if on[i]:
            found.append(points[:, corners[i]])
        if ((distances[i] > tol and distances[j] < -tol)
                or (distances[i] < -tol and distances[j] > tol)):
            share = float(distances[i] / (distances[i] - distances[j]))
            found.append(
                points[:, corners[i]]
                + share * (points[:, corners[j]] - points[:, corners[i]]))
    # A corner on the plane is met by both of its edges, so it is listed
    # twice. Collapsing the copies is what turns a touching triangle into one
    # point and a crossing one into two.
    picked = []
    for point in found:
        if not any(float(npabs(point - other).max()) <= tol
                   for other in picked):
            picked.append(point)
    if len(picked) < 2:
        return None
    if len(picked) == 2:
        return (picked[0], picked[1])
    # Three distinct points cannot be the plane's intersection with a
    # triangle, but a tolerance can make a crossing point look distinct from a
    # corner it is essentially on. The farthest pair is the segment.
    def apart(pair, /):
        difference = pair[0] - pair[1]
        return float(difference @ difference)

    return max(combinations(picked, 2), key=apart)


def section(mesh, origin, normal, /, *, tolerance=None):
    '''Finds the segments where a plane cuts a triangle mesh.

    The plane is the set of positions ``p`` with
    ``normal . (p - origin) == 0``. Where it crosses a triangle it does so
    along a straight segment, and the collection of those segments is the
    outline the plane would trace on the surface. It is returned as a
    ``SegPath`` whose segment ends are welded together, so the segments that
    meet at a corner share that corner's coordinate.

    A triangle that lies wholly in the plane contributes no segment: it meets
    the plane over an area rather than along a curve. The same holds for a
    triangle that only touches the plane at one corner, which has no length to
    contribute.

    A section is defined for any triangle mesh, open or closed, because the
    curve where a plane crosses a surface does not depend on the surface
    enclosing anything. An open sheet's outline is as much a section as a
    solid's.

    Parameters
    ----------
    mesh : TriMesh
        The surface to cut, in three-dimensional space.
    origin : array-like
        A length-3 position on the cutting plane.
    normal : array-like
        A length-3 direction the plane is perpendicular to. It need not be a
        unit vector.
    tolerance : float or None, optional
        How near the plane a corner may lie and count as on it. The default,
        ``None``, is a small fraction of the mesh's extent.

    Returns
    -------
    SegPath
        The segments where the plane meets the surface, with the units of
        ``mesh`` if it has any. It has no segments when the plane misses the
        surface, when the surface lies in the plane, or when it meets the
        surface only at isolated corners.

    Raises
    ------
    TypeError
        If ``mesh`` is not a triangle mesh.
    ValueError
        If the mesh occupies fewer than three dimensions, or if ``origin`` or
        ``normal`` is not a three-dimensional position or direction.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import section
    >>> corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                     [0., 0., 1.], [1., 1., 0.], [1., 0., 1.],
    ...                     [0., 1., 1.], [1., 1., 1.]]).T
    >>> faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
    ...                   [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
    ...                   [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    >>> cube = trimesh(corners, faces)
    >>> path = section(cube, np.array([0., 0., 0.5]),
    ...                np.array([0., 0., 1.]))
    >>> round(float(np.sum(path.measures)), 6)
    4.0
    '''
    _checked(mesh)
    units = _units_of(mesh)
    if mesh.coord_count == 0:
        return _empty_path(units)
    points = _vertices_of(mesh)
    tol = _tolerance(points, tolerance)
    origin = _position(origin, units)
    normal = _direction(normal)
    depth = normal @ (points - origin[:, None])
    tris = asarray(mesh.topo.indices)
    found = []
    for k in range(tris.shape[1]):
        segment = _segment_of(points, tris[:, k], depth, tol)
        if segment is not None:
            found.append(segment)
    if not found:
        return _empty_path(units)
    ends = asarray(found).reshape(-1, 3)
    (unique, inverse, _origin) = _welded(ends, tol)
    # A segment shared by two triangles is found twice, once from each of
    # them. Keeping one copy is what makes the path a curve rather than a
    # double-drawn one; the welding above is what makes the two copies the
    # same segment even when their ends were computed separately.
    pairs = []
    seen = set()
    for k in range(len(found)):
        (first, second) = (int(inverse[2 * k]), int(inverse[2 * k + 1]))
        if first == second:
            continue
        key = (first, second) if first < second else (second, first)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if not pairs:
        return _empty_path(units)
    coords = unique.T
    if units is not None:
        coords = coords * units
    return SegPath(coords, SegTopology(asarray(pairs, dtype=int).T,
                                       coord_count=coords.shape[1]))


# Exports ####################################################################

__all__ = ('clip', 'section')

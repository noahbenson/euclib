# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_remesh.py
'''Adding detail to a triangle mesh, and removing needless detail from a path.

Subdivision replaces one triangle with four by joining its edge midpoints. The
midpoints are shared between the triangles that meet at an edge, so the surface
stays closed, and each new triangle is a quarter of its parent's area, so the
total area and the enclosed volume are unchanged. Repeating the operation
multiplies the triangle count by four each time; a caller who asks for a
maximum edge length gets as many repetitions as that length needs.

Simplification is the other direction, and here it is the path's: a chain of
segments whose interior corners lie on a straight line is the same curve as the
two segments that join its ends, and `merge_colinear` returns the path with
those corners removed. A branching path is left alone, since a corner where
three segments meet is not an interior point of one curve, and a closed loop is
simplified from a fixed corner around.
'''

# Dependencies ###############################################################

from __future__ import annotations

from math import ceil, log2

from numpy import (
    asarray, concatenate, dot, int64, sqrt)

from immlib import math as imath, to_array

from ..types import SegPath, SegTopology, TriMesh, TriTopology
from ._boolean import _units_of, _vertices_of
from ._intersect import DEFAULT_TOLERANCE


# Subdivision ################################################################

def _checked(mesh, /):
    '''Checks that a mesh is a triangle mesh subdivision can split.'''
    if not isinstance(mesh, TriMesh):
        raise TypeError(f"expected a TriMesh; found {type(mesh)}")


def _split_once(points, tris, /):
    '''Splits every triangle at its edge midpoints, welding shared midpoints.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(D, Q)`` matrix of the mesh's positions.
    tris : numpy.ndarray
        A ``(3, M)`` matrix of corner indices.

    Returns
    -------
    points, tris : numpy.ndarray
        The new positions and the four triangles each old one became.
    '''
    count = points.shape[1]
    midpoint = {}
    added = []
    faces = []
    for k in range(tris.shape[1]):
        (a, b, c) = (int(v) for v in tris[:, k])
        middle = []
        for (u, v) in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            found = midpoint.get(key)
            if found is None:
                found = count + len(added)
                midpoint[key] = found
                added.append((points[:, u] + points[:, v]) * 0.5)
            middle.append(found)
        (ab, bc, ca) = middle
        # Each piece keeps the parent's winding: its boundary runs the same
        # way around, so a surface that faced outward still does.
        faces += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    if added:
        points = concatenate([points, asarray(added).T], axis=1)
    return (points, asarray(faces, dtype=int64).T)


def subdivide(mesh, /, *, iterations=1, max_edge=None):
    '''Splits each triangle of a mesh into four, preserving its surface.

    Each triangle is joined to the midpoints of its three edges. The midpoint
    of an edge is shared by the two triangles that meet at it, so the result is
    closed wherever the input was, and each of the four children is a quarter
    of its parent's area, so the total area and the enclosed volume do not
    change. Repeating the operation refines the mesh further: ``iterations``
    repetitions multiply the triangle count by ``4 ** iterations``.

    A maximum edge length may be asked for instead of a number of repetitions,
    or as well as one. The mesh is subdivided uniformly until every edge is no
    longer than ``max_edge``; this can refine a face more than its own size
    needs, but it needs no conformity pass, because every face is refined the
    same number of times.

    Parameters
    ----------
    mesh : TriMesh
        The triangle mesh to subdivide, in two or three dimensions.
    iterations : int, optional
        How many times to split every triangle. The default is 1. Zero returns
        the mesh itself when no ``max_edge`` is given.
    max_edge : float or None, optional
        The longest edge the result may have. The default, ``None``, sets no
        limit. It may be a quantity with units, which are converted to the
        mesh's own.

    Returns
    -------
    TriMesh
        The subdivided mesh, with the units of ``mesh`` if it has any. Its
        coordinates are NumPy arrays. The input itself is returned when
        ``iterations`` is zero and its longest edge is already short enough.

    Raises
    ------
    TypeError
        If ``mesh`` is not a triangle mesh.
    ValueError
        If ``iterations`` is negative or not an integer, or if ``max_edge`` is
        not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import subdivide
    >>> triangle = trimesh(np.array([[0., 1., 0.], [0., 0., 1.]]),
    ...                    np.array([[0], [1], [2]]))
    >>> fine = subdivide(triangle)
    >>> fine.topo.simplex_count[2]
    4
    >>> fine.coord_count
    6
    '''
    _checked(mesh)
    if int(iterations) != iterations or int(iterations) < 0:
        raise ValueError(
            f"subdivision's iteration count must be a non-negative integer;"
            f" found {iterations!r}")
    iterations = int(iterations)
    units = _units_of(mesh)
    if mesh.coord_count == 0 or mesh.topo.simplex_count[2] == 0:
        return mesh
    points = _vertices_of(mesh)
    tris = asarray(mesh.topo.indices)
    rounds = iterations
    if max_edge is not None:
        if units is not None and hasattr(max_edge, 'units'):
            max_edge = imath.mag(max_edge, units)
        longest = _longest_edge(points, tris) / (2.0 ** iterations)
        limit = float(to_array(imath.mag(max_edge), detach=True))
        if not limit > 0.0:
            raise ValueError(
                f"subdivision's maximum edge must be positive; found {limit}")
        if longest > limit:
            rounds += int(ceil(log2(longest / limit)))
    if rounds == 0:
        return mesh
    for _ in range(rounds):
        (points, tris) = _split_once(points, tris)
    coords = points
    if units is not None:
        coords = coords * units
    return TriMesh(coords, TriTopology(tris, coord_count=coords.shape[1]))


def _longest_edge(points, tris, /):
    '''The longest edge of a triangle mesh.'''
    corners = points[:, tris]
    first = corners[:, 0]
    longest = 0.0
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        difference = corners[:, i] - corners[:, j]
        length = sqrt((difference * difference).sum(axis=0))
        longest = max(longest, float(length.max()))
    return longest


# Simplification #############################################################

def _checked_path(path, /):
    '''Checks that a geometry is a path simplification can edit.'''
    if not isinstance(path, SegPath):
        raise TypeError(f"expected a SegPath; found {type(path)}")


def _chains(indices, count, /):
    '''The vertex order of each component that is a chain or a cycle.

    Returns a list of ``(order, closed)`` pairs. A component with a vertex of
    degree greater than two is not a chain, and is not listed: its corners are
    junctions rather than interior points of a curve, so removing one would
    change which segments meet.
    '''
    neighbours = [[] for _ in range(count)]
    for k in range(indices.shape[1]):
        (a, b) = (int(indices[0, k]), int(indices[1, k]))
        if a == b:
            continue
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = set()
    chains = []
    for start in range(count):
        if start in seen or not neighbours[start]:
            continue
        # The component's vertices, breadth-first, so that a branch is found
        # before a walk is attempted.
        component = []
        pending = [start]
        seen.add(start)
        while pending:
            v = pending.pop()
            component.append(v)
            for w in neighbours[v]:
                if w not in seen:
                    seen.add(w)
                    pending.append(w)
        degrees = {v: len(neighbours[v]) for v in component}
        if any(degree > 2 for degree in degrees.values()):
            continue
        ends = [v for v in component if degrees[v] == 1]
        closed = len(ends) == 0
        if not closed and len(ends) != 2:
            continue
        current = min(ends) if ends else min(component)
        previous = None
        order = []
        while True:
            order.append(current)
            following = [w for w in neighbours[current] if w != previous]
            if not following:
                break
            # A simple chain has at most one way forward; on a cycle the first
            # step picks a direction and every later step has exactly one.
            following = following[0]
            if following == order[0]:
                break
            previous = current
            current = following
        if len(order) < 2:
            continue
        chains.append((order, closed))
    return chains


def _collinear(points, previous, here, following, tolerance, /):
    '''Whether three corners lie on one line, in one direction.'''
    (a, b, c) = (points[:, previous], points[:, here], points[:, following])
    forward = c - a
    length = float(sqrt(forward @ forward))
    if length <= tolerance:
        return True
    # A reversal is not a straight continuation: the path doubles back, and
    # removing the corner would erase the excursion rather than simplify it.
    if float(dot(b - a, c - b)) < 0.0:
        return False
    offset = b - a
    along = float(offset @ forward) / (length * length)
    closest = a + along * forward
    distance = b - closest
    return float(sqrt(distance @ distance)) <= tolerance


def _simplify_order(points, order, closed, tolerance, /):
    '''The indices an ordered chain keeps.'''
    if len(order) < 3:
        return list(order)
    extended = order + [order[0]] if closed else order
    kept = [extended[0]]
    for k in range(1, len(extended) - 1):
        if not _collinear(points, kept[-1], extended[k], extended[k + 1],
                          tolerance):
            kept.append(extended[k])
    if not closed:
        kept.append(extended[-1])
    else:
        # The closing vertex is the same as the first; it is dropped from the
        # list because the rebuilt loop closes itself.
        kept = kept[:-1] if kept[-1] == kept[0] else kept
    return kept


def merge_colinear(path, /, *, tolerance=None):
    '''Removes the corners of a path that lie on a straight run.

    A path is walked one connected component at a time. Where three consecutive
    corners lie on one line and continue in the same direction, the middle one
    adds nothing to the curve and is dropped, and the two segments that met
    there become one. The result is the same curve with fewer segments: its
    length and its shape are unchanged to within the tolerance.

    A corner where more than two segments meet is a junction, not an interior
    point of a curve, so its component is left alone. A closed loop is
    simplified from its lowest-numbered corner around, and keeps that corner.

    Parameters
    ----------
    path : SegPath
        The path to simplify.
    tolerance : float or None, optional
        How far a corner may lie from the line through its neighbours and
        still count as on it. The default, ``None``, is a small fraction of the
        path's extent.

    Returns
    -------
    SegPath
        The simplified path, with the units of ``path`` if it has any. A path
        with nothing to merge, or with no segments, is returned itself.

    Raises
    ------
    TypeError
        If ``path`` is not a segment path.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import segpath
    >>> from euclib.ops import merge_colinear
    >>> line = segpath(np.array([[0., 1., 2.], [0., 0., 0.]]))
    >>> merge_colinear(line).topo.simplex_count[1]
    1
    '''
    _checked_path(path)
    if path.coord_count == 0 or path.topo.simplex_count[1] == 0:
        return path
    points = _vertices_of(path)
    span = float((points.max(axis=1) - points.min(axis=1)).max())
    tol = (DEFAULT_TOLERANCE * max(span, 1.0)
           if tolerance is None else float(tolerance))
    indices = asarray(path.topo.indices)
    chains = _chains(indices, path.coord_count)
    if not chains:
        return path
    kept_pairs = []
    chain_vertices = set()
    for (order, closed) in chains:
        kept = _simplify_order(points, order, closed, tol)
        kept_pairs.extend(_segments_of(kept, closed))
        chain_vertices.update(order)
    # The components that are not chains keep their segments as written. Their
    # vertices are disjoint from every chain's, so this adds each such segment
    # exactly once.
    for k in range(indices.shape[1]):
        (a, b) = (int(indices[0, k]), int(indices[1, k]))
        if a not in chain_vertices and b not in chain_vertices:
            kept_pairs.append((a, b))
    if len(kept_pairs) == indices.shape[1]:
        return path
    (coords, pairs) = _compact(points, kept_pairs)
    units = _units_of(path)
    if units is not None:
        coords = coords * units
    return SegPath(coords, SegTopology(asarray(pairs, dtype=int64).T,
                                       coord_count=coords.shape[1]))


def _segments_of(order, closed, /):
    '''The segments that join consecutive vertices of an ordered chain.'''
    pairs = [(order[k], order[k + 1]) for k in range(len(order) - 1)]
    if closed and len(order) > 2:
        pairs.append((order[-1], order[0]))
    return pairs


def _compact(points, pairs, /):
    '''The used positions, and the segment pairs renumbered to them.'''
    used = sorted({v for pair in pairs for v in pair})
    where = {v: k for (k, v) in enumerate(used)}
    coords = points[:, used]
    return (coords, [(where[a], where[b]) for (a, b) in pairs])


# Exports ####################################################################

__all__ = ('merge_colinear', 'subdivide')

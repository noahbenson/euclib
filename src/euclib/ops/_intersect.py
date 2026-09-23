# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_intersect.py
'''Intersections: where geometric objects meet.

A crossing is a position that belongs to two objects at once, so every
operation here reduces to a pairwise test on the *elements* of the two objects
--- a segment against a segment, a segment against a triangle --- followed by
the question of which pairs need testing at all. That second question is what
the spatial index is for: a segment can only meet the elements near it, so the
index narrows a quadratic search to a handful of candidates each.

The tests themselves are tolerant by construction. Floating-point arithmetic
rarely produces an exact crossing, so every operation takes a tolerance
defaulting to a small fraction of the geometry's size, and a pair that comes
within it counts as meeting.
'''

# Dependencies ###############################################################

from __future__ import annotations

from itertools import product

from numpy import (
    arange, asarray, ceil, concatenate, floor, full, ones, repeat, sqrt,
    stack, zeros)

from ..abc import Geometry, as_query
from ..types import Grid, SegPath, TriMesh
from ..utils import segments_intersect, segments_triangles_intersect


# Helpers ####################################################################

#: The fraction of an object's size within which two elements count as meeting.
DEFAULT_TOLERANCE = 1e-9


def tolerance_of(geom, /):
    '''Returns a tolerance suited to a geometry's size.

    Parameters
    ----------
    geom : Geometry
        The geometry.

    Returns
    -------
    float
        A small fraction of the geometry's largest extent, so that a tolerance
        means the same thing for a mesh measured in millimetres and one
        measured in kilometres.
    '''
    # The bounding box may be a quantity, whose magnitude is what the extent is
    # measured in; asking for the array directly would warn about the units.
    box = asarray(getattr(geom.bbox, 'magnitude', geom.bbox))
    span = float((box[:, 1] - box[:, 0]).max()) if box.size else 1.0
    return DEFAULT_TOLERANCE * max(span, 1.0)


def _nearby(geom, center, radius, /):
    '''The elements of a geometry that a sphere of a radius reaches.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry.
    center : numpy.ndarray
        A ``(D, 1)`` position.
    radius : float
        The radius around it.

    Returns
    -------
    numpy.ndarray
        The indices of the elements to test.
    '''
    index = geom.spatial_index
    if index is None:
        return arange(geom.topo.simplex_count[geom.order])
    return index.candidates(center, radius)[0]


# Intersections ##############################################################

def path_crossings(path, mesh, /, tolerance=None):
    '''Finds where a path crosses a triangle mesh.

    One crossing is reported per (segment, triangle) pair that meets, so a
    crossing that lands where two triangles share an edge is reported twice ---
    once naming each triangle. That is the pair that met, and it is the answer
    rather than a duplicate of it: the two rows say which triangles the path
    passes between. A caller who wants one row per location can compare the
    points, which are equal in the two rows.

    Parameters
    ----------
    path : SegPath
        The path, which must occupy three-dimensional space.
    mesh : TriMesh
        The mesh, which must occupy three-dimensional space.
    tolerance : float or None, optional
        How near a segment and a triangle must come to count as meeting. The
        default, ``None``, uses a small fraction of the mesh's size.

    Returns
    -------
    points : numpy.ndarray
        A ``(3, K)`` matrix of crossing points, in no particular order.
    segments : numpy.ndarray
        A length-``K`` vector of the path segment each crossing lies on.
    triangles : numpy.ndarray
        A length-``K`` vector of the mesh triangle each crossing lies on.

    Raises
    ------
    TypeError
        If either argument is not of the expected kind.
    ValueError
        If either occupies fewer than three dimensions.
    '''
    if not isinstance(path, SegPath):
        raise TypeError(f"expected a SegPath; found {type(path)}")
    if not isinstance(mesh, TriMesh):
        raise TypeError(f"expected a TriMesh; found {type(mesh)}")
    if path.dim != 3 or mesh.dim != 3:
        raise ValueError(
            "a path crosses a triangle mesh in three-dimensional space;"
            f" found dimensions {path.dim} and {mesh.dim}")
    tol = tolerance_of(mesh) if tolerance is None else float(tolerance)
    found_points = []
    found_segments = []
    found_triangles = []
    indices = path.topo.indices
    for i in range(indices.shape[1]):
        start = path.coords[:, indices[0, i]][:, None]
        stop = path.coords[:, indices[1, i]][:, None]
        middle = (start + stop) / 2.0
        # A crossing lies on the segment, so a triangle it crosses is within
        # half the segment's length of the segment's middle.
        reach = float(sqrt(((stop - start) ** 2).sum())) / 2.0
        near = _nearby(mesh, middle, reach)
        if near.size == 0:
            continue
        corners = mesh.coords[:, mesh.topo.indices[:, near]]
        (hit, point, _) = segments_triangles_intersect(
            repeat(start, near.size, axis=1), repeat(stop, near.size, axis=1),
            corners[:, 0], corners[:, 1], corners[:, 2], tolerance=tol)
        if hit.any():
            found_points.append(point[:, hit])
            found_segments.append(full(int(hit.sum()), i))
            found_triangles.append(near[hit])
    if not found_points:
        return (zeros((3, 0)), zeros(0, dtype=int), zeros(0, dtype=int))
    return (concatenate(found_points, axis=1),
            concatenate(found_segments),
            concatenate(found_triangles))


def path_intersections(first, second, /, tolerance=None):
    '''Finds where two paths cross.

    Parameters
    ----------
    first, second : SegPath
        The paths, which must occupy the same space.
    tolerance : float or None, optional
        How near two segments must come to count as meeting. The default,
        ``None``, uses a small fraction of the first path's size.

    Returns
    -------
    points : numpy.ndarray
        A ``(D, K)`` matrix of crossing points.
    first_segments : numpy.ndarray
        A length-``K`` vector of the segment of the first path each crossing
        lies on.
    second_segments : numpy.ndarray
        The same for the second path.

    Raises
    ------
    TypeError
        If either argument is not a path.
    ValueError
        If the paths occupy different dimensions.
    '''
    if not isinstance(first, SegPath) or not isinstance(second, SegPath):
        raise TypeError("expected two paths")
    if first.dim != second.dim:
        raise ValueError(
            f"the paths occupy different dimensions: {first.dim} and"
            f" {second.dim}")
    tol = tolerance_of(first) if tolerance is None else float(tolerance)
    found_points = []
    found_first = []
    found_second = []
    (a, b) = (first.topo.indices, second.topo.indices)
    for i in range(a.shape[1]):
        start = first.coords[:, a[0, i]][:, None]
        stop = first.coords[:, a[1, i]][:, None]
        reach = float(sqrt(((stop - start) ** 2).sum())) / 2.0
        near = _nearby(second, (start + stop) / 2.0, reach)
        if near.size == 0:
            continue
        (hit, point, _, _) = segments_intersect(
            repeat(start, near.size, axis=1), repeat(stop, near.size, axis=1),
            second.coords[:, b[0, near]], second.coords[:, b[1, near]],
            tolerance=tol)
        if hit.any():
            found_points.append(point[:, hit])
            found_first.append(full(int(hit.sum()), i))
            found_second.append(near[hit])
    if not found_points:
        return (zeros((first.dim, 0)), zeros(0, dtype=int), zeros(0, dtype=int))
    return (concatenate(found_points, axis=1),
            concatenate(found_first),
            concatenate(found_second))


def contains(geom, points, /, tolerance=None):
    '''Determines whether positions lie on or within a geometry.

    A position belongs to a geometry when the nearest position of the geometry
    is the position itself, which is the same test the interpolation engine
    makes when it decides that a position has left the object. A triangle mesh
    is a surface, so a position belongs to it when it lies on it; a tetrahedral
    mesh encloses a volume, so a position within it belongs to it too.

    Parameters
    ----------
    geom : Geometry
        The geometry.
    points : array-like
        A ``(D, Q)`` matrix of positions.
    tolerance : float or None, optional
        How near counts as belonging. The default, ``None``, uses a small
        fraction of the geometry's size.

    Returns
    -------
    numpy.ndarray
        A length-``Q`` boolean vector.

    Raises
    ------
    TypeError
        If the first argument is not a geometry.
    '''
    if not isinstance(geom, Geometry):
        raise TypeError(f"expected a Geometry; found {type(geom)}")
    query = as_query(points)
    tol = tolerance_of(geom) if tolerance is None else float(tolerance)
    if isinstance(geom, Grid):
        # A grid has an affine rather than coordinates, so belonging to it
        # means the index coordinate lying within the grid's extent. An index
        # names a cell's center, so the extent runs from half a step before the
        # first center to half a step past the last.
        loc = geom.to_local(query)
        parts = [asarray(getattr(loc, f)) for f in loc._fields]
        inside = ones(parts[0].reshape(-1).shape, dtype=bool)
        for (p, size) in zip(parts, geom.shape):
            inside = inside & (p.reshape(-1) >= -0.5 - tol) & (
                p.reshape(-1) <= (size - 0.5) + tol)
        return inside
    back = geom.to_global(geom.to_local(query))
    gap = asarray(query) - asarray(back)
    return sqrt((gap * gap).sum(axis=0)) <= tol


def voxel_intersections(mesh, grid, /, tolerance=None):
    '''Decomposes the overlap of a tetrahedral mesh and a grid into tetrahedra.

    Each tetrahedron is cut against each voxel it reaches, and the piece they
    share is filled with tetrahedra, so that a volume mesh can be resampled onto
    a grid --- or a grid's cells expressed as part of the mesh.

    The cutting happens in the grid's *index space*, where a voxel is the unit
    box and every voxel looks alike. That is what lets an affine grid whose axes
    are not aligned with the coordinate axes be handled by a test that knows
    only about boxes: the tetrahedron is carried into index space, cut there,
    and the pieces carried back.

    Parameters
    ----------
    mesh : TetMesh
        The tetrahedral mesh.
    grid : Grid
        A three-dimensional grid.
    tolerance : float or None, optional
        How far outside a face a corner may lie and still count. The default,
        ``None``, uses a small fraction of the mesh's size.

    Returns
    -------
    pieces : TetMesh
        A tetrahedral mesh of the pieces, in no particular order.
    tetrahedra : numpy.ndarray
        A length-``T`` vector naming the mesh tetrahedron each piece came from.
    voxels : numpy.ndarray
        A ``(3, T)`` matrix of the grid cell indices each piece came from.

    Raises
    ------
    TypeError
        If either argument is not of the expected kind.
    ValueError
        If the grid is not three-dimensional.
    '''
    from ..types import TetMesh as _TetMesh
    from ..types import TetTopology as _TetTopology
    from ..utils import tetrahedron_box_intersection
    if not isinstance(mesh, _TetMesh):
        raise TypeError(f"expected a TetMesh; found {type(mesh)}")
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    if len(grid.shape) != 3:
        raise ValueError(
            f"a voxel grid has three dimensions; this one has"
            f" {len(grid.shape)}")
    tol = tolerance_of(mesh) if tolerance is None else float(tolerance)
    shape = tuple(grid.shape)
    to_index = grid.affine_inverse
    to_global = grid.affine
    corners = mesh.coords[:, mesh.topo.indices]      # (3, 4, M)
    pieces = []
    from_tet = []
    from_voxel = []
    for i in range(corners.shape[2]):
        inside = to_index.apply(corners[:, :, i])    # (3, 4) in index space
        # The voxels the tetrahedron can reach, from its own extent in index
        # space: a tetrahedron reaches only the cells its box meets. An index
        # names a cell's center, so the cell numbered `v` covers the half step
        # on either side of it, and the first cell whose region reaches a
        # coordinate `x` is the one whose center is within half a step of it.
        low = floor(inside.min(axis=1) + 0.5).astype(int).clip(0, None)
        high = ceil(inside.max(axis=1) - 0.5).astype(int).clip(
            None, [s - 1 for s in shape])
        for voxel in product(*(range(low[a], high[a] + 1)
                               for a in range(3))):
            bounds = stack([asarray(voxel, dtype=float) - 0.5,
                            asarray(voxel, dtype=float) + 0.5], axis=1)
            (vertices, tets) = tetrahedron_box_intersection(
                inside, bounds, tol)
            if tets.shape[1] == 0:
                continue
            pieces.append((to_global.apply(vertices), tets))
            from_tet.append(full(tets.shape[1], i))
            from_voxel.append(stack([full(tets.shape[1], voxel[a])
                                     for a in range(3)]))
    if not pieces:
        # An empty result is a tetrahedral mesh with no coordinates and no
        # tetrahedra, not a mesh with a placeholder in it.
        empty = _TetMesh(zeros((3, 0)),
                         _TetTopology(zeros((4, 0), dtype=int), coord_count=0))
        return (empty, zeros(0, dtype=int), zeros((3, 0), dtype=int))
    # Lay every piece's vertices end to end, and shift each piece's own
    # tetrahedra to point at the place its vertices landed.
    coords = []
    tets = []
    offset = 0
    for (vertices, local) in pieces:
        coords.append(vertices)
        tets.append(local + offset)
        offset += vertices.shape[1]
    all_coords = concatenate(coords, axis=1)
    all_tets = concatenate(tets, axis=1)
    built = _TetMesh(all_coords,
                     _TetTopology(all_tets, coord_count=all_coords.shape[1]))
    return (built, concatenate(from_tet), concatenate(from_voxel, axis=1))


def mesh_intersections(first, second, /, tolerance=None):
    '''Finds the segments along which two triangle meshes meet.

    Two surfaces that cross do so along a curve, and that curve is a set of
    straight segments, one for each pair of triangles that meet. Each triangle
    of the first mesh is tested against the triangles of the second that lie
    near it, which the spatial index supplies.

    Two triangles *in the same plane* are not handled: they meet over an area
    rather than along a segment, and neither has an edge that crosses the
    other. Two meshes that share a face therefore report nothing there.

    Parameters
    ----------
    first, second : TriMesh
        The meshes, which must occupy three-dimensional space.
    tolerance : float or None, optional
        How near an edge and a triangle must come to count as meeting. The
        default, ``None``, uses a small fraction of the first mesh's size.

    Returns
    -------
    SegPath
        The segments along which the meshes meet. It has no segments when they
        do not meet.

    Raises
    ------
    TypeError
        If either argument is not a triangle mesh.
    ValueError
        If either mesh occupies fewer than three dimensions.
    '''
    from ..types import SegPath as _SegPath
    from ..types import SegTopology as _SegTopology
    from ..utils import triangles_segments_intersect
    if not isinstance(first, TriMesh) or not isinstance(second, TriMesh):
        raise TypeError("expected two triangle meshes")
    if first.dim != 3 or second.dim != 3:
        raise ValueError(
            "triangle meshes meet along segments in three-dimensional space;"
            f" found dimensions {first.dim} and {second.dim}")
    tol = tolerance_of(first) if tolerance is None else float(tolerance)
    found = []
    for i in range(first.topo.simplex_count[2]):
        corners = first.coords[:, first.topo.indices[:, i]]
        # The triangle reaches no further than its own radius from its middle.
        middle = corners.mean(axis=1)[:, None]
        reach = float(sqrt(((corners - middle) ** 2).sum(axis=0)).max())
        near = _nearby(second, middle, reach)
        if near.size == 0:
            continue
        others = second.coords[:, second.topo.indices[:, near]]
        (hit, start, stop) = triangles_segments_intersect(
            repeat(corners[:, 0][:, None], near.size, axis=1),
            repeat(corners[:, 1][:, None], near.size, axis=1),
            repeat(corners[:, 2][:, None], near.size, axis=1),
            others[:, 0], others[:, 1], others[:, 2], tolerance=tol)
        if not hit.any():
            continue
        (start, stop) = (start[:, hit], stop[:, hit])
        # A pair that meets at a single point contributes no length to the
        # intersection *curve*: two surfaces that touch are not crossing there.
        apart = sqrt(((stop - start) ** 2).sum(axis=0))
        if not (apart > tol).any():
            continue
        found.append((start[:, apart > tol], stop[:, apart > tol]))
    if not found:
        empty = _SegPath(zeros((3, 0)),
                         _SegTopology(zeros((2, 0), dtype=int), coord_count=0))
        return empty
    # Lay the ends of every segment out, and pair them up as segments.
    coords = concatenate([concatenate([s for (s, _) in found], axis=1),
                          concatenate([t for (_, t) in found], axis=1)],
                         axis=1)
    count = coords.shape[1] // 2
    return _SegPath(coords, _SegTopology([list(range(count)),
                                          list(range(count, 2 * count))],
                                         coord_count=2 * count))


# Exports ####################################################################

__all__ = ('path_crossings', 'path_intersections', 'contains', 'tolerance_of',
           'voxel_intersections', 'mesh_intersections')

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

from immlib import to_array

from numpy import (
    arange, asarray, ceil, concatenate, cross, cumsum, einsum, floor, full,
    intp, ones, repeat, sqrt, stack, tile, zeros)

from ..abc import Geometry, as_query
from ..types import Grid, SegPath, TriMesh
from ..utils import (segments_intersect, segments_triangles_intersect,
                     triangle_box_polygon)
from ..utils._pycore import polygon_fan


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


def _nearby_runs(geom, centers, radii, /):
    '''The elements a sphere around each of several positions reaches.

    This is ``_nearby`` for a whole array of positions at once, which is what an
    operation that walks its own elements wants: the query is one call rather
    than one per element, and the answers come back flattened into runs with a
    count for each position.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry.
    centers : numpy.ndarray
        A ``(D, Q)`` matrix of positions.
    radii : numpy.ndarray
        A length-``Q`` vector of the radius around each.

    Returns
    -------
    counts : numpy.ndarray
        A length-``Q`` vector of the number of elements per position.
    found : numpy.ndarray
        The elements, one run after another.
    '''
    index = geom.spatial_index
    if index is None:
        total = geom.topo.simplex_count[geom.order]
        counts = [total] * centers.shape[1]
        return (asarray(counts, dtype=intp),
                tile(arange(total), centers.shape[1]))
    return index.candidate_runs(centers, radii)


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
    indices = path.topo.indices
    starts = path.coords[:, indices[0]]                    # (3, M)
    stops = path.coords[:, indices[1]]
    middles = (starts + stops) / 2.0
    # A crossing lies on the segment, so a triangle it crosses is within half
    # the segment's length of the segment's middle. The segments are of
    # different lengths, so each asks with its own reach.
    reaches = sqrt(((stops - starts) ** 2).sum(axis=0)) / 2.0
    (counts, near) = _nearby_runs(mesh, middles, reaches)
    if counts.size == 0 or near.size == 0:
        return (zeros((3, 0)), zeros(0, dtype=int), zeros(0, dtype=int))
    # One flat list of (segment, triangle) pairs: the run counts say how many
    # triangles belong to each segment, so a segment's index repeated as many
    # times as it has candidates names the segment each pair comes from.
    here = repeat(arange(indices.shape[1]), counts)
    corners = mesh.coords[:, mesh.topo.indices[:, near]]
    (hit, point, _) = segments_triangles_intersect(
        starts[:, here], stops[:, here], corners[:, 0], corners[:, 1],
        corners[:, 2], tolerance=tol)
    if not hit.any():
        return (zeros((3, 0)), zeros(0, dtype=int), zeros(0, dtype=int))
    return (point[:, hit], here[hit], near[hit])


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
        # `to_array(detach=True)` and not `asarray`: whether a position lies
        # within a grid is a *selection* --- the index coordinate against the
        # grid's extent is a comparison --- so detaching is the right thing and
        # not merely a convenience. And `asarray` would raise on a tensor that
        # requires a gradient rather than reading past it, which would make a
        # tensor's query fail here instead of answering.
        parts = [to_array(getattr(loc, f), detach=True) for f in loc._fields]
        inside = ones(parts[0].reshape(-1).shape, dtype=bool)
        for (p, size) in zip(parts, geom.shape):
            inside = inside & (p.reshape(-1) >= -0.5 - tol) & (
                p.reshape(-1) <= (size - 0.5) + tol)
        return inside
    back = geom.to_global(geom.to_local(query))
    # Detached, and for the same reason the grid branch above is: whether a
    # position belongs to a geometry is a *selection* --- a comparison against a
    # tolerance --- and a selection has no derivative. `to_array(detach=True)`
    # and not `asarray`, which raises on a tensor that requires a gradient
    # rather than reading past it, and so would refuse a query it can answer.
    gap = to_array(query, detach=True) - to_array(back, detach=True)
    return sqrt((gap * gap).sum(axis=0)) <= tol


#: How flat a tetrahedron may be and still count as enclosing volume. Against
#: the cube of a third of its longest edge, so it means the same at any scale.
#: The flat tetrahedra of a mesh built from real surfaces are *exactly* flat
#: --- the smallest nonzero one measured is 8.5e-07 against this --- so the
#: value only has to be small.
_FLAT_RATIO = 1e-9


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
        A tetrahedral mesh of the pieces, in no particular order. Every
        tetrahedron of it lies entirely within one voxel, and no coordinate
        appears twice: pieces that meet share their vertices, so the mesh has
        the connectivity the surface had rather than being a heap of separate
        pieces that happen to touch.
    voxels : numpy.ndarray
        A ``(3, T)`` matrix of the grid cell indices each piece came from, one
        column per tetrahedron of ``pieces``.

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
    # Every tetrahedron carried into index space in *one* affine call rather
    # than one call each. An affine costs about 28 microseconds whatever it is
    # given --- it builds a quantity and a unit context per call --- and the same
    # 1,536 points cost 86 microseconds together, so the difference is not
    # arithmetic but the per-call overhead. The corners are put end to end for
    # the call and the shape is restored after.
    (dim, per_tet, tets) = corners.shape
    flat = to_index.apply(corners.reshape(dim, per_tet * tets))
    # Laid out as one *C-contiguous* block per tetrahedron, rather than as three
    # planes that a tetrahedron is a column of, and copied rather than
    # transposed, because the kernel below is sensitive to the layout: on a
    # tetrahedron held as a transposed view it takes 1,335 microseconds where a
    # contiguous one takes 2.1 --- a factor of 635, and the whole operation was
    # thirty times slower before this copy. `_half_spaces` is indifferent to it,
    # so it is the kernel alone. One copy of the batch costs a few microseconds;
    # the alternative, a copy per tetrahedron, costs more than it saves.
    index_space = flat.reshape(dim, per_tet, tets).transpose(2, 0, 1).copy()
    # Vertices are *welded* as the pieces are built: a location already seen
    # keeps the index it was given, so no coordinate appears twice and pieces
    # that meet share their corners. This is done here rather than by
    # deduplicating afterwards, which would mean generating every duplicate
    # first and then paying to throw them away --- measured at 6.3 copies of
    # each location on a 30 mm patch.
    #
    # A plain position is the key, and it is exact rather than merely close: the
    # cut computes a shared point identically from either side of it. Measured
    # over 300 randomly oriented tetrahedra, 1,038 shared locations and every
    # one of them bit-identical.
    coords = []
    seen = {}
    faces = []
    voxel_of_piece = []
    # The voxels each tetrahedron can reach, from its own extent in index space:
    # a tetrahedron reaches only the cells its box meets. An index names a cell's
    # center, so the cell numbered `v` covers the half step on either side of it,
    # and the first cell whose region reaches a coordinate `x` is the one whose
    # center is within half a step of it.
    #
    # Every tetrahedron's reach at once, rather than one at a time: a `min` over
    # the axes of all of them is one pass, where a `min` per tetrahedron is 384.
    reach_lo = floor(index_space.min(axis=2) + 0.5).astype(int).clip(0, None)
    reach_hi = ceil(index_space.max(axis=2) - 0.5).astype(int).clip(
        None, [s - 1 for s in shape])
    spans = (reach_hi - reach_lo + 1).clip(0)          # (M, 3)
    # A tetrahedron of no volume meets no voxel, and the cut cannot be relied on
    # to say so: the C kernel returns a region for a flat one where the
    # pure-Python kernel returns none. On the left hemisphere an exactly flat
    # tetrahedron (two coincident corners) came back as 8 tetrahedra, and 4.69%
    # of that mesh's tetrahedra are flat.
    #
    # What that costs is large, and it took three attempts to measure. The
    # pieces of a 60 mm patch of the left hemisphere sum to 66177.443800124
    # against the mesh's own 66177.443805136 --- and to *110067.042370502* with
    # this guard disabled, 66% more than the mesh contains. A 30 mm patch showed
    # no difference at all, which is what an earlier note here mistook for the
    # guard being cosmetic: that patch has no flat tetrahedra the walk reaches,
    # and one mesh is not another.
    #
    # Giving a flat tetrahedron no voxels to reach settles it for either
    # implementation and costs one vectorized determinant.
    #
    # The test is on the simplex's own volume against the cube of its longest
    # edge, so that it means the same for a mesh at any size.
    tet_edges = [index_space[:, :, i] - index_space[:, :, 0] for i in (1, 2, 3)]
    tet_volume = abs(einsum('mi,mi->m', cross(tet_edges[1], tet_edges[2],
                                              axis=1), tet_edges[0])) / 6.0
    tet_span = sqrt(sum((e * e).sum(axis=1) for e in tet_edges))
    flat = tet_volume <= _FLAT_RATIO * (tet_span / 3.0) ** 3
    spans[flat] = 0
    counts = spans.prod(axis=1)                        # (M,)

    total = int(counts.sum())
    # Every pair of a tetrahedron and a voxel, as arrays. A tetrahedron's reach
    # is its own, so the pairs are ragged, and `repeat` is what turns a count per
    # tetrahedron into one entry per pair. `within` then counts up through each
    # box; it is broken into the three axes with the last varying fastest, which
    # is the order a nested walk over them would have taken.
    pair_of_tet = repeat(arange(tets), counts)
    starts = concatenate([[0], cumsum(counts)[:-1]])
    within = arange(total) - repeat(starts, counts)
    offsets = zeros((total, 3), dtype=int)
    for axis in (2, 1, 0):
        offsets[:, axis] = within % spans[pair_of_tet, axis]
        within = within // spans[pair_of_tet, axis]
    voxel_of = reach_lo[pair_of_tet] + offsets          # (P, 3)
    bounds_of = stack([voxel_of - 0.5, voxel_of + 0.5], axis=2)  # (P, 3, 2)
    # Indexed rather than sliced, so that each pair's tetrahedron is a
    # *contiguous* array: the kernel is 635 times slower on one that is not, and
    # a fancy index copies. The cut is the one part of this that stays a loop,
    # because each pair's region is a different shape; it is 1.7 microseconds a
    # pair, against the walk it used to sit inside.
    inside_of = index_space[pair_of_tet]                # (P, 3, 4)
    for j in range(total):
        (vertices, local) = tetrahedron_box_intersection(
            inside_of[j], bounds_of[j], tol)
        count = local.shape[1]
        if count == 0:
            continue
        # `zeros` rather than `empty`: the name `empty` is taken below by the
        # empty-result mesh, and every element here is assigned anyway.
        remap = zeros(vertices.shape[1], dtype='intp')
        for i in range(vertices.shape[1]):
            key = (float(vertices[0, i]), float(vertices[1, i]),
                   float(vertices[2, i]))
            at = seen.get(key)
            if at is None:
                at = len(coords)
                seen[key] = at
                coords.append(vertices[:, i])
            remap[i] = at
        faces.append(remap[local])
        voxel_of_piece.append(tile(voxel_of[j][:, None], (1, count)))
    if not faces:
        # An empty result is a tetrahedral mesh with no coordinates and no
        # tetrahedra, not a mesh with a placeholder in it.
        empty = _TetMesh(zeros((3, 0)),
                         _TetTopology(zeros((4, 0), dtype=int), coord_count=0))
        return (empty, zeros((3, 0), dtype=int))
    # Every piece is carried out of index space by *one* affine call at the end
    # rather than one each, for the same reason the corners went in that way.
    all_coords = to_global.apply(stack(coords, axis=1))
    all_faces = concatenate(faces, axis=1)
    built = _TetMesh(all_coords,
                     _TetTopology(all_faces, coord_count=all_coords.shape[1]))
    return (built, concatenate(voxel_of_piece, axis=1))



def voxel_surface_intersections(mesh, grid, /, tolerance=None):
    '''Decomposes the overlap of a triangle mesh and a grid into triangles.

    Each triangle is cut against each voxel it reaches, and the piece they share
    is a polygon --- a triangle clipped by a box --- which is then covered with
    triangles. So a *surface* is resampled onto a grid: what a triangle
    contributes to a voxel is an area, not a volume, because a surface has no
    interior.

    The cutting happens in the grid's *index space*, where a voxel is the unit
    box and every voxel looks alike, as it does for the tetrahedral version. That
    is what lets an affine grid whose axes are not aligned with the coordinate
    axes be handled by a test that knows only about boxes.

    Parameters
    ----------
    mesh : TriMesh
        The triangle mesh.
    grid : Grid
        A three-dimensional grid.
    tolerance : float or None, optional
        How far outside a face a corner may lie and still count. The default,
        ``None``, uses a small fraction of the mesh's size.

    Returns
    -------
    pieces : TriMesh
        A triangle mesh of the pieces, in no particular order. Its surface area
        is the mesh's own, since the pieces tile it.
    triangles : numpy.ndarray
        A length-``T`` vector naming the mesh triangle each piece came from.
    voxels : numpy.ndarray
        A ``(3, T)`` matrix of the grid cell indices each piece came from.

    Raises
    ------
    TypeError
        If either argument is not of the expected kind.
    ValueError
        If the grid is not three-dimensional.
    '''
    from ..types import TriMesh as _TriMesh
    from ..types import TriTopology as _TriTopology
    from ..utils import triangle_box_polygon
    if not isinstance(mesh, TriMesh):
        raise TypeError(f"expected a TriMesh; found {type(mesh)}")
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    if len(grid.shape) != 3:
        raise ValueError(
            f"a voxel grid has three dimensions; this one has"
            f" {len(grid.shape)}")
    tol = tolerance_of(mesh) if tolerance is None else float(tolerance)
    shape = tuple(grid.shape)
    to_index = grid.affine_inverse
    corners = mesh.coords[:, mesh.topo.indices]           # (3, 3, M)
    (dim, per_tri, triangles) = corners.shape
    flat = to_index.apply(corners.reshape(dim, per_tri * triangles))
    index_space = flat.reshape(dim, per_tri, triangles).transpose(2, 0, 1).copy()
    # A triangle of no area shares no area with anything, and clipping one gives
    # a polygon that is a segment at best. Skipping them here is the same
    # treatment the tetrahedral version gives a tetrahedron of no volume.
    edges = [index_space[:, :, i] - index_space[:, :, 0] for i in (1, 2)]
    area = sqrt((cross(edges[0], edges[1], axis=1) ** 2).sum(axis=1)) / 2.0
    span = sqrt(sum((e * e).sum(axis=1) for e in edges))
    live = area > _FLAT_RATIO * (span / 2.0) ** 2

    # A voxel `v` owns `[v - 0.5, v + 0.5)` --- a *half-open* extent, so that a
    # face belongs to the voxel above it and not to both. A surface can lie
    # exactly *in* a shared face, and with a closed extent such a triangle is
    # enumerated in both voxels and its area counted twice; the pieces then
    # overweight the mesh. Half-open enumeration puts it in one voxel, and every
    # piece has its own area with nothing for a caller to scale.
    #
    # So `v` is reached when `[v - 0.5, v + 0.5)` meets `[lo, hi]`, which is
    # `v - 0.5 <= hi` and `v + 0.5 > lo`. For an integer `v` that is
    # `floor(lo - 0.5) + 1` through `floor(hi + 0.5)`.
    #
    # The tetrahedral version writes this the other way round, as a closed
    # extent, and is right to: a tetrahedron with volume has `lo < hi` in every
    # axis, where the two forms agree. A triangle is *flat* in one axis, and
    # there they do not --- the closed form collapses to `lo > hi` and
    # enumerates nothing at all.
    reach_lo = (floor(index_space.min(axis=2) - 0.5) + 1).astype(int).clip(0, None)
    reach_hi = floor(index_space.max(axis=2) + 0.5).astype(int).clip(
        None, [s - 1 for s in shape])
    spans = (reach_hi - reach_lo + 1).clip(0)
    spans[~live] = 0
    counts = spans.prod(axis=1)
    total = int(counts.sum())
    if total == 0:
        empty = _TriMesh(zeros((3, 0)),
                         _TriTopology(zeros((3, 0), dtype=int), coord_count=0))
        return (empty, zeros(0, dtype=int), zeros((3, 0), dtype=int))
    pair_of_tri = repeat(arange(triangles), counts)
    starts = concatenate([[0], cumsum(counts)[:-1]])
    within = arange(total) - repeat(starts, counts)
    offsets = zeros((total, 3), dtype=int)
    spans_of = spans[pair_of_tri]
    for axis in (2, 1, 0):
        offsets[:, axis] = within % spans_of[:, axis]
        within = within // spans_of[:, axis]
    voxel_of = reach_lo[pair_of_tri] + offsets
    bounds_of = stack([voxel_of - 0.5, voxel_of + 0.5], axis=2)
    inside_of = index_space[pair_of_tri]
    pieces = []
    from_tri = []
    from_voxel = []
    for i in range(total):
        polygon = triangle_box_polygon(inside_of[i], bounds_of[i], tol)
        count = polygon.shape[1]
        if count < 3:
            continue
        local = polygon_fan(count)
        pieces.append((polygon, local))
        from_tri.append(full(local.shape[1], pair_of_tri[i]))
        from_voxel.append(tile(voxel_of[i][:, None], (1, local.shape[1])))
    if not pieces:
        empty = _TriMesh(zeros((3, 0)),
                         _TriTopology(zeros((3, 0), dtype=int), coord_count=0))
        return (empty, zeros(0, dtype=int), zeros((3, 0), dtype=int))
    coords = []
    faces = []
    offset = 0
    for (vertices, local) in pieces:
        coords.append(vertices)
        faces.append(local + offset)
        offset += vertices.shape[1]
    all_coords = concatenate(coords, axis=1)
    all_faces = concatenate(faces, axis=1)
    built = _TriMesh(all_coords,
                     _TriTopology(all_faces, coord_count=all_coords.shape[1]))
    return (built, concatenate(from_tri), concatenate(from_voxel, axis=1))


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
    count = first.topo.simplex_count[2]
    corners = first.coords[:, first.topo.indices]        # (3, 3, M)
    middles = corners.mean(axis=1)                       # (3, M)
    # A triangle reaches no further than its own radius from its middle. Every
    # triangle's reach is its own --- a mesh on a sphere has small ones and
    # large ones side by side --- so the index is asked with one radius per
    # triangle rather than one for all of them.
    reaches = sqrt(((corners - middles[:, None, :]) ** 2).sum(axis=0)).max(
        axis=0)
    (counts, found) = _nearby_runs(second, middles, reaches)
    # Every candidate of every triangle, as one flat list of pairs: the run
    # counts say how many belong to each triangle, so an index repeated as many
    # times as it has candidates names the triangle each pair starts from.
    if counts.size == 0 or found.size == 0:
        return _SegPath(zeros((3, 0)),
                        _SegTopology(zeros((2, 0), dtype=int), coord_count=0))
    here = repeat(arange(count), counts)
    others = second.coords[:, second.topo.indices[:, found]]
    (hit, start, stop) = triangles_segments_intersect(
        corners[:, 0][:, here], corners[:, 1][:, here], corners[:, 2][:, here],
        others[:, 0], others[:, 1], others[:, 2], tolerance=tol)
    # A pair that meets at a single point contributes no length to the
    # intersection *curve*: two surfaces that touch are not crossing there.
    keep = hit & (sqrt(((stop - start) ** 2).sum(axis=0)) > tol)
    (start, stop) = (start[:, keep], stop[:, keep])
    if start.shape[1] == 0:
        return _SegPath(zeros((3, 0)),
                        _SegTopology(zeros((2, 0), dtype=int), coord_count=0))
    # Lay the ends of every segment out, and pair them up as segments.
    coords = concatenate([start, stop], axis=1)
    each = coords.shape[1] // 2
    return _SegPath(coords, _SegTopology([list(range(each)),
                                          list(range(each, 2 * each))],
                                         coord_count=2 * each))


# Exports ####################################################################

__all__ = ('path_crossings', 'path_intersections', 'contains', 'tolerance_of',
           'voxel_intersections', 'mesh_intersections')

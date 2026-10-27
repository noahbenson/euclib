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

from functools import lru_cache
from itertools import product

from immlib import to_array

from numpy import (
    arange, asarray, bincount, ceil, concatenate, cross, cumsum, einsum, floor,
    full, intp, maximum, ones, prod, repeat, sqrt, stack, tile, zeros)
from numpy.linalg import det

from pcollections import holdlazy, lazy

from ..abc import Geometry, UNSET, as_query
from ..abc._geom import _carried_property
from ..types._geom import _prop_interp
from ..types import Grid, SegPath, TriMesh
from ..utils import (segments_intersect, segments_triangles_intersect,
                     triangle_box_polygon)
from ..utils._quadrature import quadrature
from ..utils._pycore import polygon_fan


# Helpers ####################################################################

#: The fraction of an object's size within which two elements count as meeting.
DEFAULT_TOLERANCE = 1e-9


def _carried_gradient(mesh, name, coords, order, /):
    '''A carried property's gradient at the pieces' coordinates.

    A piece carries its corners' *values* and no derivative data, so a fit of
    order 2 or more built on the piece alone would be a different function from
    the source's: a quadratic is ten coefficients and four corners' values are
    four conditions. Carrying the gradient too makes a piece's own fit the
    source's polynomial, restricted --- measured, it agrees with the source to
    2e-14 where it would not otherwise.

    Nothing is computed for an order that has no use for it: a fit below the
    quadratic order reads no gradient, so those answer ``None`` and the property
    carries nothing, which keeps the common linear case from carrying three times
    its value data for a derivative no fit will read.

    The *hessian* is not carried, because nothing reads one. No fit consults a
    property's hessian --- the Bezier cubic settles its freedom by face rules and
    the polynomial cubic is least-norm --- so a piece would carry it for no one.

    Parameters
    ----------
    mesh : SimplexGeometry
        The mesh the property belongs to.
    name : hashable
        The property's name.
    coords : array-like
        The coordinates to read it at, which are the pieces'.
    order : int
        The interpolation order that would need it.
    Returns
    -------
    gradient : numpy.ndarray or None
        The gradient, or ``None`` if this property's interpolation reads none.
    '''
    prop = mesh.propinfo(name)
    if prop.interp[1] < order:
        return None
    return mesh.propgrad(name, at=coords)


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
    # The mesh's properties come along. Every coordinate of the pieces is inside
    # the original mesh, so each property can be read at them --- and lazily,
    # because an intersected mesh's properties are usually not read at all and
    # interpolating one over every coordinate is not cheap. The interpolation
    # the property was given is carried with it, since a caller who chose one
    # meant it.
    # `holdlazy` rather than `items()`: reading a lazy dictionary resolves its
    # values, so `items()` would build every property --- reading every value
    # the deferral exists to avoid.
    for (name, prop) in holdlazy(mesh.properties).items():
        # The interpolation is part of the property, so reading it here would
        # build the property and read the value. Carried as a lazy of its own,
        # it resolves when the new property is asked for one.
        interp = (lazy(_prop_interp, prop) if isinstance(prop, lazy)
                  else getattr(prop, 'interp', UNSET))
        built = built.withprop(
            name, lazy(_carried_property, mesh, name, built.coords),
            gradient=lazy(_carried_gradient, mesh, name, built.coords, 2),
            **({} if interp is UNSET else {'interp': interp}))
    return (built, concatenate(voxel_of_piece, axis=1))




#: How many intersected meshes to keep. One, because the case the cache exists
#: for is integrating several properties of *one* mesh over one grid, and a
#: single entry serves that completely; more only helps a caller cycling between
#: meshes, which is not what this is for.
#:
#: The measurements behind that, on blocks of tetrahedra cut against grids of
#: their own size:
#:
#: =========  ========  =========  ===========  ==========  ============
#: block      tets      first      second       intersection  entry
#: =========  ========  =========  ===========  ==========  ============
#: 8^3        2,048     0.628 s    0.548 s      13%          ~19x source
#: 16^3       16,384    5.979 s    5.051 s      16%          ~19x source
#: =========  ========  =========  ===========  ==========  ============
#:
#: So an entry saves a sixth of a repeat call and holds about nineteen times its
#: source mesh --- of which only a fifth is the pieces' own arrays, the rest
#: being the planobject machinery around them --- and it pins the source mesh and
#: grid as well, since they are the key. Four of those is a lot of memory
#: indefinitely held for a sixth of a call.
#:
#: The intersection could look larger in a measurement that cut a small mesh
#: against a large grid, where it is nearly the whole cost. It is not the whole
#: cost on a mesh of any size, because the quadrature that reads the source at
#: every rule point grows with the pieces and does not come from the cache.
CACHE_SIZE = 1


@lru_cache(maxsize=CACHE_SIZE)
def _intersections(mesh, grid, /):
    '''The pieces of an intersection, kept for a repeat of the same question.

    Integrating a mesh over a grid cuts it against the grid once and then reads
    the field at points within the pieces. Asking about a *second* property of
    the same mesh over the same grid cuts the same mesh the same way and gets
    the same pieces, so they are kept: a geometry is immutable and hashes by its
    content, so the same two arguments are the same question.

    A geometry hashes by its *plan* rather than by its computed fields, so a
    property whose value is deferred hashes without being readied --- which
    matters here, since the whole point of a deferred value is that nobody has
    read it.

    Parameters
    ----------
    mesh : SimplexGeometry
        The geometry being cut.
    grid : Grid
        The grid to cut it against.

    Returns
    -------
    pieces : SimplexGeometry
        The pieces, of the source's own kind.
    voxels : numpy.ndarray
        The grid cell each piece came from.
    measures : numpy.ndarray
        The extent of each piece: a tetrahedron's volume, a triangle's area, a
        segment's length.
    '''
    from ..types import TetMesh as _TetMesh
    from ..types import TriMesh as _TriMesh
    if isinstance(mesh, _TetMesh):
        (pieces, voxels) = voxel_intersections(mesh, grid)
        return (pieces, voxels, asarray(pieces.volume))
    if isinstance(mesh, _TriMesh):
        (pieces, voxels) = voxel_surface_intersections(mesh, grid)
        return (pieces, voxels, asarray(pieces.area))
    (pieces, voxels) = voxel_segment_intersections(mesh, grid)
    return (pieces, voxels, asarray(pieces.length))


def integrate(mesh, grid, values, /, emptyconst=0):
    '''Integrates a tetrahedral mesh's field over a grid's voxels.

    Each voxel's total is the integral of the field over the part of the mesh
    that lies within it: the mesh is cut against the grid by
    `voxel_intersections`, and each piece contributes its own integral --- its
    measure times the field's mean over it, which is exact for the linear
    interpolation the pieces are cut with, since a linear field's mean over a
    tetrahedron is its value at the tetrahedron's centroid.

    The result is a *total*, not an average. Dividing by a volume is one
    division for the whole result, so it is left to the caller. The average over
    a voxel is the total divided by the volume of the grid's cells; the average
    over the part of a voxel the mesh covers is the total divided by the volume
    the mesh occupies there, which is ``integrate(mesh, grid, 1)``.

    Parameters
    ----------
    mesh : TetMesh
        The tetrahedral mesh carrying the field.
    grid : Grid
        A three-dimensional grid.
    values : str or float
        What to integrate: the name of one of the mesh's properties, or a
        constant. A field that is a value per coordinate and nothing else must
        be attached with ``withprop`` first, so that it can be interpolated onto
        the pieces like any other --- which is also what makes it interpolated
        *correctly*, at the pieces' corners rather than at the mesh's.
    emptyconst : float, optional
        What the *empty* part of a voxel contributes. The default, ``0``, leaves
        it out, so a voxel's total is the integral over the mesh's part of it;
        any other constant adds that constant times the empty volume. A voxel
        the mesh misses entirely is empty throughout, so its total is
        ``emptyconst`` times the volume of a cell.

    Returns
    -------
    totals : numpy.ndarray
        The integral over each voxel, shaped like the grid, with the field's
        channel dimensions first if it has any.

    Raises
    ------
    TypeError
        If either geometry is not of the expected kind.
    ValueError
        If the grid is not three-dimensional.
    '''
    from ..types import PrismMesh as _PrismMesh
    from ..types import SegPath as _SegPath
    from ..types import TetMesh as _TetMesh
    from ..types import TriMesh as _TriMesh
    if isinstance(mesh, _PrismMesh):
        # A prism's properties are read through its stack of tetrahedra --- the
        # prism has no simplex to fit --- so the stack is what is integrated.
        mesh = mesh.tetmesh
    if not isinstance(mesh, (_TetMesh, _TriMesh, _SegPath)):
        raise TypeError(
            f"expected a TetMesh, a TriMesh, a SegPath, or a PrismMesh; found"
            f" {type(mesh)}")
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    shape = tuple(grid.shape)
    if len(shape) != mesh.dim:
        raise ValueError(
            f"this geometry occupies {mesh.dim} dimensions, so its grid must"
            f" have {mesh.dim}; this one has {len(shape)}")
    from ..abc import QUALITATIVE
    # The measure of a piece is the extent the source has: a tetrahedron's
    # volume, a triangle's area. Each is integrated over the voxels it falls in,
    # so a surface's total is its area and not a volume.
    # The pieces, the cell each came from, and each one's extent --- kept
    # between calls, since a second property of the same mesh over the same grid
    # is cut the same way.
    (pieces, voxels, measures) = _intersections(mesh, grid)
    if emptyconst != 0 and mesh.order != len(shape):
        # The empty part of a cell is the measure the pieces do *not* fill, and
        # that is only defined when they fill the cell's own extent: a
        # tetrahedron in a volume, or a triangle in a plane. A surface in a
        # volume has no part of it to leave empty --- its pieces measure area,
        # so filling the rest with a constant would be adding an area to a
        # volume --- and a path is further short again.
        raise ValueError(
            f"emptyconst needs a geometry whose pieces fill the space they are"
            f" in; a {type(mesh).__name__}'s elements have order {mesh.order}"
            f" and the grid has {len(shape)} dimensions, so there is no empty"
            f" part of a cell for it to fill")
    tets = measures.shape[0]
    if isinstance(values, str):
        prop = mesh.propinfo(values)
        if prop.vartype == QUALITATIVE:
            raise ValueError(
                f"cannot integrate {values!r}: it is a qualitative property,"
                f" and a category has no meaning between the values it takes")
        (method, order) = prop.interp
        if method == 'nearest':
            # A nearest field is piecewise constant, so no quadrature rule is
            # exact for it: the value jumps between the regions a vertex is
            # nearest to. Its integral over an *element* is the element's
            # measure times the mean of the values at its corners --- the
            # regions being equal in measure --- and that is what this is, read
            # at the piece's corners, which is where the field was carried to.
            held = asarray(pieces.prop(values))
            per_piece = held[..., pieces.topo.indices].mean(axis=-2)
            channels = per_piece.shape[:-1]
        else:
            # The fit is a polynomial of the method's order in the element's
            # barycentric coordinates, so a rule exact to that degree integrates
            # it exactly. The rule's points are carried through the piece's own
            # corners into global coordinates and the *piece's* field is read
            # there --- which is the source's polynomial, because
            # `voxel_intersections` carries the values and, above the linear
            # order, the derivatives a fit of that order needs. Reading the
            # source instead would give the same answer and cost a second
            # spatial query; reading a piece that carried only values would give
            # a different one, a re-fit on the sub-simplex.
            (lam, wts) = quadrature(pieces.order, order)
            corners = asarray(pieces.coords)[:, pieces.topo.indices]  # (D,K,T)
            pts = einsum('qk,dkt->dqt', lam, corners)
            (dim, points, count) = pts.shape
            # The *source* mesh's field, not the piece's. Above the linear
            # order a fit on the piece alone is a different function from the
            # source's: a piece carries its corners' values and --- since
            # `voxel_intersections` began carrying them --- their derivatives,
            # but a cubic is still under-determined by those sixteen conditions
            # against its twenty coefficients, and the hessian that would pin it
            # is not yet something a fit reads. Measured: reading the pieces
            # integrates x^3 over the unit tetrahedron to 0.00720 where the
            # answer is 0.00833.
            held = asarray(
                mesh.prop(values, at=pts.reshape(dim, points * count)))
            held = held.reshape(held.shape[:-1] + (points, count))
            channels = held.shape[:-2]
            if held.ndim == 2:
                held = held[None]
            per_piece = einsum('q,cqt->ct', wts, held)
    else:
        # A constant integrates to itself times the measure, which is what the
        # degree-1 rule gives it. There is no property to read, so it never
        # reaches the quadrature above.
        per_piece = full((1, tets), float(values))
        channels = ()
    # A property with no channels is a vector and not a matrix, so a channel
    # axis is added for the arithmetic below and taken off by the reshape at the
    # end: one path for both, rather than two that differ in an axis. `channels`
    # is what the caller gets back, so it is recorded before the axis goes on.
    if per_piece.ndim == 1:
        per_piece = per_piece[None]
    contrib = per_piece.reshape((-1, tets)) * measures
    # The voxel each piece came from, flattened, and the voxels flattened, so
    # that the sum over a voxel is a `bincount` rather than a scatter --- one
    # call per channel, and no loop over voxels.
    # The cell each piece came from, as one index into the flattened grid ---
    # raveled from the last axis up, so that the leading coordinate varies
    # slowest whatever the dimension is.
    flat = zeros(voxels.shape[1], dtype=intp)
    for axis in range(len(shape)):
        flat = flat * shape[axis] + voxels[axis]
    cells = int(prod(shape))
    totals = zeros((contrib.shape[0], cells))
    for (c, one) in enumerate(contrib):
        totals[c] = bincount(flat, weights=one, minlength=cells)
    if emptyconst != 0:
        # The empty part of a voxel is what the mesh does not fill. Clipped at
        # zero, because a piece that reaches the boundary can cover a voxel to
        # within rounding, and a negative empty volume would be a lie about it.
        covered = bincount(flat, weights=measures, minlength=cells)
        matrix = asarray(grid.affine.matrix)
        cell = abs(det(matrix[:len(shape), :len(shape)]))
        totals += float(emptyconst) * maximum(cell - covered, 0.0)
    return totals.reshape(channels + shape)


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
        A grid of the same dimension as the mesh: two for a mesh in a plane,
        three for one in space.
    tolerance : float or None, optional
        How far outside a face a corner may lie and still count. The default,
        ``None``, uses a small fraction of the mesh's size.

    Returns
    -------
    pieces : TriMesh
        A triangle mesh of the pieces, in no particular order. Its surface area
        is the mesh's own, since the pieces tile it; every triangle of it lies
        entirely within one voxel; and no coordinate appears twice, so pieces
        that meet share their corners.
    voxels : numpy.ndarray
        A ``(D, T)`` matrix of the grid cell indices each piece came from, one
        column per triangle of ``pieces``, ``D`` being the space's dimension.

    Raises
    ------
    TypeError
        If either argument is not of the expected kind.
    ValueError
        If the grid's dimension is not the mesh's.
    '''
    from ..types import TriMesh as _TriMesh
    from ..types import TriTopology as _TriTopology
    from ..utils import triangle_box_polygon
    if not isinstance(mesh, TriMesh):
        raise TypeError(f"expected a TriMesh; found {type(mesh)}")
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    # The grid must live in the same space the mesh does. A surface in a plane
    # cannot be voxelized by a volume grid --- there is no way to bring the
    # coordinates into register, since a plane in space is some other plane's
    # image and nothing here says which --- so that is refused rather than
    # guessed at.
    if len(grid.shape) != mesh.dim:
        raise ValueError(
            f"this mesh occupies {mesh.dim} dimensions, so its grid must have"
            f" {mesh.dim}; this one has {len(grid.shape)}")
    tol = tolerance_of(mesh) if tolerance is None else float(tolerance)
    shape = tuple(grid.shape)
    to_index = grid.affine_inverse
    to_global = grid.affine
    corners = mesh.coords[:, mesh.topo.indices]           # (D, 3, M)
    (dim, per_tri, triangles) = corners.shape
    flat = to_index.apply(corners.reshape(dim, per_tri * triangles))
    index_space = flat.reshape(dim, per_tri, triangles).transpose(2, 0, 1).copy()
    # A triangle of no area shares no area with anything, and clipping one gives
    # a polygon that is a segment at best. Skipping them here is the same
    # treatment the tetrahedral version gives a tetrahedron of no volume.
    edges = [index_space[:, :, i] - index_space[:, :, 0] for i in (1, 2)]
    # The area from the Gram determinant, ``|e0|^2 |e1|^2 - (e0 . e1)^2``, which
    # is half the squared area and needs no cross product --- the one numpy
    # defines only in three dimensions, and which it has dropped for two.
    (norm0, norm1) = [(e * e).sum(axis=1) for e in edges]
    dot = (edges[0] * edges[1]).sum(axis=1)
    area = sqrt(norm0 * norm1 - dot * dot) / 2.0
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
        empty = _TriMesh(zeros((dim, 0)),
                         _TriTopology(zeros((3, 0), dtype=int), coord_count=0))
        return (empty, zeros(0, dtype=int), zeros((dim, 0), dtype=int))
    pair_of_tri = repeat(arange(triangles), counts)
    starts = concatenate([[0], cumsum(counts)[:-1]])
    within = arange(total) - repeat(starts, counts)
    offsets = zeros((total, dim), dtype=int)
    spans_of = spans[pair_of_tri]
    # Unpacked from the last axis down, so that the offset's leading coordinate
    # varies slowest --- whatever the dimension is.
    for axis in range(dim - 1, -1, -1):
        offsets[:, axis] = within % spans_of[:, axis]
        within = within // spans_of[:, axis]
    voxel_of = reach_lo[pair_of_tri] + offsets
    bounds_of = stack([voxel_of - 0.5, voxel_of + 0.5], axis=2)
    inside_of = index_space[pair_of_tri]
    # The polygon's corners are *welded* as the pieces are built, as they are
    # for the tetrahedral version: a location already seen keeps the index it
    # was given, so no coordinate appears twice and pieces that meet along an
    # edge share it. A surface is where this matters most --- a triangle's
    # corners are shared with every neighbour, and a clipped edge produces the
    # same point for every voxel it passes through.
    coords = []
    seen = {}
    faces = []
    voxel_of_piece = []
    for i in range(total):
        polygon = triangle_box_polygon(inside_of[i], bounds_of[i], tol)
        count = polygon.shape[1]
        if count < 3:
            continue
        local = polygon_fan(count)
        remap = zeros(count, dtype='intp')
        for k in range(count):
            key = tuple(float(c) for c in polygon[:, k])
            at = seen.get(key)
            if at is None:
                at = len(coords)
                seen[key] = at
                coords.append(polygon[:, k])
            remap[k] = at
        faces.append(remap[local])
        voxel_of_piece.append(tile(voxel_of[i][:, None], (1, local.shape[1])))
    if not faces:
        empty = _TriMesh(zeros((dim, 0)),
                         _TriTopology(zeros((3, 0), dtype=int), coord_count=0))
        return (empty, zeros((dim, 0), dtype=int))
    all_coords = to_global.apply(stack(coords, axis=1))
    all_faces = concatenate(faces, axis=1)
    built = _TriMesh(all_coords,
                     _TriTopology(all_faces, coord_count=all_coords.shape[1]))
    # `holdlazy` rather than `items()`: reading a lazy dictionary resolves its
    # values, so `items()` would build every property --- reading every value
    # the deferral exists to avoid.
    for (name, prop) in holdlazy(mesh.properties).items():
        # The interpolation is part of the property, so reading it here would
        # build the property and read the value. Carried as a lazy of its own,
        # it resolves when the new property is asked for one.
        interp = (lazy(_prop_interp, prop) if isinstance(prop, lazy)
                  else getattr(prop, 'interp', UNSET))
        built = built.withprop(
            name, lazy(_carried_property, mesh, name, built.coords),
            gradient=lazy(_carried_gradient, mesh, name, built.coords, 2),
            **({} if interp is UNSET else {'interp': interp}))
    return (built, concatenate(voxel_of_piece, axis=1))


def _segment_box_interval(a, d, voxel, tol, /):
    '''The parameter interval over which a segment lies in a voxel's box.

    The slab method: a box is three pairs of parallel planes, each cuts the
    segment's parameter interval in two, and the three intervals meet where the
    segment is in the box. A segment parallel to a slab is either in it for all
    of the interval or for none of it, which the degenerate case says.

    Parameters
    ----------
    a, d : numpy.ndarray
        The segment's start, and the vector to its end, in the grid's index
        space.
    voxel : sequence of int
        The voxel's index.
    tol : float
        How far outside the box a point may lie and still count.

    Returns
    -------
    interval : tuple of float
        ``(t0, t1)``, empty --- with ``t1 < t0`` --- when the segment misses.
    '''
    (t0, t1) = (0.0, 1.0)
    for axis in range(3):
        # The exact box. The tolerance is applied to the *reach* --- which cells
        # a segment is considered for --- and not to the cut, because a cut made
        # `tol` past a face is a piece that overlaps its neighbour by that much
        # and a path whose pieces are longer than the path.
        lo = voxel[axis] - 0.5
        hi = voxel[axis] + 0.5
        if d[axis] == 0.0:
            if a[axis] < lo or a[axis] > hi:
                return (1.0, 0.0)
            continue
        (u0, u1) = ((lo - a[axis]) / d[axis], (hi - a[axis]) / d[axis])
        if u0 > u1:
            (u0, u1) = (u1, u0)
        t0 = max(t0, u0)
        t1 = min(t1, u1)
    return (t0, t1)


def voxel_segment_intersections(path, grid, /, tolerance=None):
    '''Decomposes a path into the pieces of it that lie within each voxel.

    A path has no area and no volume, so this is not the same question the
    tetrahedral and triangle versions ask: it clips each segment against each
    voxel the segment reaches, and the pieces are the parts of the path inside
    each. A property integrated over them is integrated along the path's length.

    A voxel is the unit box of the grid's index space, as it is for the other
    two, so a path is carried into that space, clipped there, and the pieces
    carried back.

    Parameters
    ----------
    path : SegPath
        The path.
    grid : Grid
        A three-dimensional grid.
    tolerance : float or None, optional
        How far outside a voxel a point may lie and still count. The default,
        ``None``, uses a small fraction of the path's size.

    Returns
    -------
    pieces : SegPath
        A path of the pieces, in no particular order. Every segment of it lies
        entirely within one voxel.
    voxels : numpy.ndarray
        A ``(3, T)`` matrix of the grid cell indices each piece came from, one
        column per segment of ``pieces``.

    Raises
    ------
    TypeError
        If either argument is not of the expected kind.
    ValueError
        If the grid is not three-dimensional.
    '''
    from ..types import SegPath as _SegPath
    from ..types import SegTopology as _SegTopology
    if not isinstance(path, SegPath):
        raise TypeError(f"expected a SegPath; found {type(path)}")
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    if len(grid.shape) != 3:
        raise ValueError(
            f"a voxel grid has three dimensions; this one has"
            f" {len(grid.shape)}")
    tol = tolerance_of(path) if tolerance is None else float(tolerance)
    shape = tuple(grid.shape)
    to_index = grid.affine_inverse
    to_global = grid.affine
    corners = path.coords[:, path.topo.indices]           # (3, 2, M)
    (dim, per_seg, segments) = corners.shape
    flat = to_index.apply(corners.reshape(dim, per_seg * segments))
    index_space = flat.reshape(dim, per_seg, segments).transpose(2, 0, 1).copy()
    # A voxel `v` owns `[v - 0.5, v + 0.5)`, as it does for a triangle --- a
    # segment may lie exactly in a shared face, and a closed extent would count
    # it in both voxels and so count its length twice.
    reach_lo = (floor(index_space.min(axis=2) - 0.5) + 1).astype(int).clip(
        0, None)
    reach_hi = floor(index_space.max(axis=2) + 0.5).astype(int).clip(
        None, [s - 1 for s in shape])
    (starts, ends, cells) = ([], [], [])
    for seg in range(segments):
        (lo, hi) = (reach_lo[seg, :], reach_hi[seg, :])
        if (hi < lo).any():
            continue
        a = index_space[seg, :, 0]
        d = index_space[seg, :, 1] - a
        for vx in range(lo[0], hi[0] + 1):
            for vy in range(lo[1], hi[1] + 1):
                for vz in range(lo[2], hi[2] + 1):
                    (t0, t1) = _segment_box_interval(a, d, (vx, vy, vz), tol)
                    if t1 - t0 <= 0.0:
                        continue
                    starts.append(a + t0 * d)
                    ends.append(a + t1 * d)
                    cells.append((vx, vy, vz))
    if not starts:
        empty = _SegPath(zeros((3, 0)),
                         _SegTopology(zeros((2, 0), dtype=int), coord_count=0))
        return (empty, zeros((3, 0), dtype=int))
    # Each piece carries its own two coordinates: the pieces of one segment meet
    # end to end, and sharing those coordinates would be the tidier mesh, but a
    # path's pieces are not the mesh a caller goes on to read --- they are the
    # pieces of one integral --- and the indexing to share them is the whole of
    # the difference.
    local = asarray(stack([starts, ends], axis=1))        # (T, 2, 3)
    coords = to_global.apply(local.reshape(2 * len(starts), dim).T)
    topo = _SegTopology(
        asarray([[2 * i for i in range(len(starts))],
                 [2 * i + 1 for i in range(len(starts))]]),
        coord_count=2 * len(starts))
    built = _SegPath(coords, topo)
    # `holdlazy` for the same reason the mesh versions use it.
    for (name, prop) in holdlazy(path.properties).items():
        interp = (lazy(_prop_interp, prop) if isinstance(prop, lazy)
                  else getattr(prop, 'interp', UNSET))
        built = built.withprop(
            name, lazy(_carried_property, path, name, built.coords),
            gradient=lazy(_carried_gradient, path, name, built.coords, 2),
            **({} if interp is UNSET else {'interp': interp}))
    return (built, asarray(cells, dtype=intp).T)


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
           'voxel_intersections', 'voxel_surface_intersections',
           'voxel_segment_intersections', 'integrate', 'mesh_intersections')

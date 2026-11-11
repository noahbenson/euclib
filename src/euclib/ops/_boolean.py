# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_boolean.py
'''Building one solid out of two: union, intersection, and difference.

A boolean operation answers a question about *space*: which positions are in
one solid, in both, or in one and not the other. The answer's surface is made
of pieces of the two inputs' surfaces, cut along the curve where those surfaces
cross, so the operation has two halves --- cutting the surfaces against one
another, and keeping the pieces the answer asks for.

The cutting is done by a binary space partition: each solid's triangles are
filed into a tree of planes, and a triangle of one solid is clipped by walking
the other's tree, keeping the part on the outside (or the inside) of every
plane it meets. That is the classic constructive-solid-geometry algorithm, and
it needs no separate intersection-curve extraction, no constrained
triangulation, and no stitching: a triangle that is cut becomes two polygons,
and a triangle that is not is kept whole. The price is that the result's
surface is a polygon soup whose shared corners were computed more than once, so
the corners are welded back together at the end, within a tolerance.

A boolean operation is only defined for *closed* surfaces, because the
inside of an open surface is not a solid at all. Both inputs are checked for
that first, and a surface whose triangles do not join edge to edge is refused.
The inputs' triangles are also made to wind outward, so a mesh that was built
inside out is still a solid rather than a hole.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    abs as npabs, asarray, concatenate, cross, cumsum, empty, floor, sqrt,
    zeros)

from immlib import math as imath, to_array

from ..types import TriMesh, TriTopology
from ._intersect import DEFAULT_TOLERANCE, tolerance_of


# Polygons and planes ########################################################

#: How a polygon sits relative to a plane. The two sides may be combined by a
#: bitwise or, which is how a polygon is recognized as spanning the plane.
_COPLANAR = 0
_FRONT = 1
_BACK = 2
_SPANNING = 3


class _Polygon:
    '''A convex, planar face, with the plane it lies in and its winding.

    The plane is kept alongside the vertices rather than recomputed from them,
    because cutting a polygon leaves pieces whose first three corners may be
    collinear even though the polygon is not. The winding and the normal are
    kept in agreement: the normal points to the face's outside.
    '''

    __slots__ = ('vertices', 'normal', 'offset')

    def __init__(self, vertices, normal, offset, /):
        self.vertices = vertices
        self.normal = normal
        self.offset = offset

    def flip(self, /):
        '''Turns the polygon inside out.'''
        self.vertices = self.vertices[::-1]
        self.normal = -self.normal
        self.offset = -self.offset


class _Plane:
    '''An oriented plane: the positions where ``normal . x == offset``.'''

    __slots__ = ('normal', 'offset')

    def __init__(self, normal, offset, /):
        self.normal = normal
        self.offset = offset

    def flip(self, /):
        '''Turns the plane around, so that its two sides trade places.'''
        self.normal = -self.normal
        self.offset = -self.offset

    def split(self, polygon, coplanar_front, coplanar_back, front, back, tol,
              /):
        '''Sorts a polygon into the pieces on each side of this plane.

        A polygon entirely above the plane goes to ``front`` and one entirely
        below goes to ``back``, both whole. One that lies in the plane goes to
        ``coplanar_front`` or ``coplanar_back`` according to whether it faces
        the same way this plane does --- which is what lets two solids that
        share a face keep it once. One that crosses is cut along the line where
        it meets the plane, and each half is added to its own side.

        Parameters
        ----------
        polygon : _Polygon
            The polygon to sort or cut.
        coplanar_front, coplanar_back, front, back : list
            The lists the pieces are appended to.
        tol : float
            How near the plane a corner may lie and still count as on it.
        '''
        vertices = polygon.vertices
        distances = vertices @ self.normal - self.offset
        types = [(_FRONT if d > tol else _BACK if d < -tol else _COPLANAR)
                 for d in distances]
        kind = _COPLANAR
        for t in types:
            kind |= t
        if kind == _COPLANAR:
            if float(self.normal @ polygon.normal) > 0.0:
                coplanar_front.append(polygon)
            else:
                coplanar_back.append(polygon)
            return
        if kind == _FRONT:
            front.append(polygon)
            return
        if kind == _BACK:
            back.append(polygon)
            return
        ahead = []
        behind = []
        count = len(types)
        for i in range(count):
            j = (i + 1) % count
            (vi, vj) = (vertices[i], vertices[j])
            (ti, tj) = (types[i], types[j])
            # A corner on the plane belongs to both pieces, so that each keeps
            # the cut edge and stays a polygon rather than an open strip.
            if ti != _BACK:
                ahead.append(vi)
            if ti != _FRONT:
                behind.append(vi)
            if (ti | tj) == _SPANNING:
                denominator = float(self.normal @ (vj - vi))
                if denominator == 0.0:      # pragma: no cover - rounding
                    continue
                t = (self.offset - float(self.normal @ vi)) / denominator
                middle = vi + (vj - vi) * t
                ahead.append(middle)
                behind.append(middle)
        if len(ahead) >= 3:
            front.append(_Polygon(asarray(ahead), polygon.normal,
                                  polygon.offset))
        if len(behind) >= 3:
            back.append(_Polygon(asarray(behind), polygon.normal,
                                 polygon.offset))


class _Node:
    '''A binary space partition of a solid's polygons.

    Each node holds a plane, the polygons that lie in it, and a subtree for
    each side. Clipping a polygon by the solid means sorting it down the tree,
    keeping the part on the outside of every plane it crosses; inverting the
    tree turns the solid inside out, which is how the operations compose the
    inside and the outside of their two inputs.
    '''

    __slots__ = ('plane', 'front', 'back', 'polygons', 'tol')

    def __init__(self, polygons=None, tol=0.0, /):
        self.plane = None
        self.front = None
        self.back = None
        self.polygons = []
        self.tol = tol
        if polygons:
            self.build(polygons)

    def _child(self, /):
        '''An empty child that remembers the tolerance.'''
        node = _Node()
        node.tol = self.tol
        return node

    def build(self, polygons, /):
        '''Adds polygons to the tree, splitting them by its planes.'''
        if not polygons:
            return
        if self.plane is None:
            first = polygons[0]
            self.plane = _Plane(first.normal.copy(), first.offset)
        front = []
        back = []
        for polygon in polygons:
            # A polygon in the node's own plane joins the node, whether it
            # faces this way or the other: the node keeps both kinds.
            self.plane.split(polygon, self.polygons, self.polygons, front,
                             back, self.tol)
        if front:
            if self.front is None:
                self.front = self._child()
            self.front.build(front)
        if back:
            if self.back is None:
                self.back = self._child()
            self.back.build(back)

    def clip_polygons(self, polygons, /):
        '''Keeps the parts of polygons that lie outside this solid.'''
        if self.plane is None:
            return list(polygons)
        front = []
        back = []
        for polygon in polygons:
            self.plane.split(polygon, front, back, front, back, self.tol)
        if self.front is not None:
            front = self.front.clip_polygons(front)
        # A part on the inside of the last plane is inside the solid, so it is
        # dropped; there is nowhere further to send it.
        back = self.back.clip_polygons(back) if self.back is not None else []
        return front + back

    def clip_to(self, other, /):
        '''Removes every polygon of this tree that lies inside ``other``.'''
        self.polygons = other.clip_polygons(self.polygons)
        if self.front is not None:
            self.front.clip_to(other)
        if self.back is not None:
            self.back.clip_to(other)

    def invert(self, /):
        '''Turns the solid inside out.'''
        for polygon in self.polygons:
            polygon.flip()
        if self.plane is not None:
            self.plane.flip()
        if self.front is not None:
            self.front.invert()
        if self.back is not None:
            self.back.invert()
        (self.front, self.back) = (self.back, self.front)

    def all_polygons(self, /):
        '''Every polygon in the tree.'''
        polygons = list(self.polygons)
        if self.front is not None:
            polygons += self.front.all_polygons()
        if self.back is not None:
            polygons += self.back.all_polygons()
        return polygons


# Construction and assembly ##################################################

def _vertices_of(mesh, /):
    '''A mesh's coordinates as bare numbers, which is what a plane needs.'''
    return to_array(imath.mag(mesh.coords), detach=True)


def _volume(points, tris, /):
    '''The signed volume a closed triangle surface encloses.

    The divergence theorem turns a volume into a sum over the surface, and for
    a triangle that sum is one sixth of the triple product of its corners. The
    sign is the answer to which way the triangles wind: positive when they face
    out of the solid, negative when they face into it.
    '''
    corners = points[:, tris].transpose(2, 1, 0)        # (M, 3, 3)
    return float((corners[:, 0] * cross(corners[:, 1],
                                        corners[:, 2])).sum()) / 6.0


def _oriented(points, tris, which, /):
    '''The triangles, wound consistently and outward, or a refusal.

    A mesh says which way its surface faces by the order of each triangle's
    corners, and a closed surface has to say it consistently: two triangles
    that share an edge must traverse it in opposite directions, so that the
    windings agree across the seam. Meshes are often written without that care
    --- the seam is invisible until something depends on it --- so the
    windings are repaired here rather than trusted.

    Repair means two things. First, the triangles are walked from one to the
    next across their shared edges, and a triangle whose edge is traversed the
    same way as its neighbour's is turned around; if the walk finds a surface
    that cannot be wound consistently, the mesh is not a surface at all and is
    refused. Second, the whole surface is turned inside out if its signed
    volume is negative, which is what makes "outward" mean the same thing for
    every mesh.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of positions.
    tris : numpy.ndarray
        A ``(3, M)`` matrix of triangles.
    which : str
        A name for the mesh, used in the refusal: ``'first'`` or ``'second'``.

    Returns
    -------
    numpy.ndarray
        A ``(3, M)`` matrix of triangles, wound consistently and outward. It
        is a new array even when no triangle needed turning.

    Raises
    ------
    ValueError
        If the triangles are not a closed surface: an edge that does not
        border exactly two triangles leaves the surface open (or overlapping),
        and one that cannot be wound consistently does not bound a solid.
        Either way there is no inside to speak of, so there is no boolean to
        answer with.
    '''
    tris = asarray(tris)
    count = tris.shape[1]
    edges = {}
    for k in range(count):
        (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
        for (u, v) in ((a, b), (b, c), (c, a)):
            if u == v:
                raise ValueError(
                    f"a boolean operation needs closed surfaces, and the"
                    f" {which} mesh has a triangle with a repeated corner,"
                    " which is a degenerate surface rather than a solid")
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append((k, u < v))
    for uses in edges.values():
        if len(uses) != 2:
            raise ValueError(
                f"a boolean operation needs closed surfaces, and the {which}"
                " mesh is not closed: some of its edges are not shared by"
                " exactly two triangles")
    parity = [-1] * count
    for start in range(count):
        if parity[start] >= 0:
            continue
        parity[start] = 0
        queue = [start]
        while queue:
            k = queue.pop()
            (a, b, c) = (int(tris[0, k]), int(tris[1, k]), int(tris[2, k]))
            for (u, v) in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                mine = u < v
                for (other, theirs) in edges[key]:
                    if other == k:
                        continue
                    # Neighbours that traverse the edge the same way are wound
                    # oppositely to one another, so one of them is turned.
                    want = parity[k] ^ (mine == theirs)
                    if parity[other] < 0:
                        parity[other] = want
                        queue.append(other)
                    elif parity[other] != want:
                        raise ValueError(
                            f"a boolean operation needs closed surfaces, and"
                            f" the {which} mesh cannot be wound consistently:"
                            " it is not an orientable surface, so it does not"
                            " bound a solid")
    result = tris.copy()
    for k in range(count):
        if parity[k]:
            result[[1, 2], k] = result[[2, 1], k]
    if _volume(points, result) < 0.0:
        result[[1, 2]] = result[[2, 1]]
    return result


def _polygons(points, tris, /):
    '''A mesh's triangles as polygons, skipping the ones with no area.'''
    polygons = []
    for k in range(tris.shape[1]):
        (a, b, c) = (points[:, tris[0, k]], points[:, tris[1, k]],
                     points[:, tris[2, k]])
        plane = cross(b - a, c - a)
        length = sqrt(float(plane @ plane))
        if length == 0.0:
            continue
        plane = plane / length
        polygons.append(_Polygon(asarray([a, b, c]), plane, float(plane @ a)))
    return polygons


def _welded(points, tol, /):
    '''The distinct positions in a list, and where each input point went.

    The cut corners of a boolean operation are computed more than once, once
    from each polygon that reaches them, so the same position arrives as
    several numbers that differ in the last bits. Welding them is what turns a
    polygon soup into a surface with a topology. The search is on a grid of
    cells the size of the tolerance, with the neighbouring cells also checked,
    because a position on a cell's boundary and one just across it are the same
    position even though they hash differently.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(Q, 3)`` matrix of positions.
    tol : float
        The distance below which two positions are the same position.

    Returns
    -------
    unique : numpy.ndarray
        A ``(U, 3)`` matrix of the distinct positions, in the order they were
        first met.
    inverse : numpy.ndarray
        A length-``Q`` vector giving each input point's row in ``unique``.
    origin : numpy.ndarray
        A length-``U`` vector giving the input point that each distinct
        position was first seen at.
    '''
    cell = tol if tol > 0.0 else 1.0
    grid = {}
    unique = []
    origin = []
    inverse = empty(points.shape[0], dtype=int)
    for (i, point) in enumerate(points):
        key = tuple(int(v) for v in floor(point / cell))
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in grid.get((key[0] + dx, key[1] + dy,
                                               key[2] + dz), ()):
                        apart = unique[candidate] - point
                        if float(npabs(apart).max()) <= tol:
                            found = candidate
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            found = len(unique)
            unique.append(point)
            origin.append(i)
            grid.setdefault(key, []).append(found)
        inverse[i] = found
    return (asarray(unique), inverse, asarray(origin))


def _simplified(points, tris, tol, /):
    '''A mesh's triangles, with coincident corners made one.

    A surface is often written with the same position listed several times ---
    a sphere's pole is the textbook case, where every longitude has its own
    copy of the one point --- and the triangles that join them have no area and
    leave edges that no neighbour shares. Collapsing the copies first is what
    lets the closure check see the surface rather than its spelling: the corner
    that the copies stand for is one corner, and the triangles they flatten are
    no triangles at all.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(3, Q)`` matrix of positions.
    tris : numpy.ndarray
        A ``(3, M)`` matrix of triangles.
    tol : float
        The distance below which two positions are the same position.

    Returns
    -------
    numpy.ndarray
        A ``(3, M')`` matrix of triangles, indexing the original positions
        through the first of each set of coincident ones.
    '''
    tris = asarray(tris)
    (_unique, inverse, origin) = _welded(points.T, tol)
    representative = origin[inverse]
    faces = []
    for k in range(tris.shape[1]):
        (a, b, c) = (int(representative[tris[0, k]]),
                     int(representative[tris[1, k]]),
                     int(representative[tris[2, k]]))
        if a != b and b != c and a != c:
            faces.append((a, b, c))
    if not faces:
        return zeros((3, 0), dtype=int)
    return asarray(faces, dtype=int).T


def _conform(points, faces, tol, /):
    '''Splits triangles where another triangle's corner lies on an edge.

    A boolean operation cuts each surface against the other's planes, and the
    two sides of a seam are not always cut at the same places: one side may end
    an edge at a corner that the other side runs straight past. The surface is
    still closed in the geometric sense, but its edges no longer pair, and a
    caller walking the surface --- from a triangle to the one across each of
    its edges --- meets a gap. This is the T-junction that constructive solid
    geometry is known for.

    The repair is local. Every edge that no neighbour shares is a seam, and the
    seam's split points are the corners at the far ends of the other unpaired
    edges, so only those corners are candidates. A triangle with a split point
    on one of its edges is fanned from the opposite corner through the points
    in order, which keeps every piece a triangle and makes the long edge into
    the shorter edges the other side already has.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(Q, 3)`` matrix of welded positions.
    faces : list of tuple of int
        The triangles, as triples of indices into ``points``.
    tol : float
        The distance below which a corner counts as lying on an edge, and the
        distance below which a corner counts as being at an edge's end.

    Returns
    -------
    list of tuple of int
        The triangles, split so that no corner lies in the middle of an edge.
    '''
    faces = list(faces)
    for _ in range(32):
        uses = {}
        for k in range(len(faces)):
            (a, b, c) = faces[k]
            for (u, v) in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                uses.setdefault(key, []).append(k)
        loose = [key for (key, seen) in uses.items() if len(seen) != 2]
        if not loose:
            break
        # The middle of a seam is the end of another seam, so these are the
        # only corners that can be sitting on an edge.
        corners = {v for key in loose for v in key}
        splits = {}
        for (u, v) in loose:
            (pu, pv) = (points[u], points[v])
            direction = pv - pu
            squared = float(direction @ direction)
            if squared == 0.0:
                continue
            found = []
            for q in corners:
                if q == u or q == v:
                    continue
                offset = points[q] - pu
                t = float(offset @ direction) / squared
                if t <= 0.0 or t >= 1.0:
                    continue
                if sqrt(float((offset - t * direction)
                              @ (offset - t * direction))) <= tol:
                    found.append((t, q))
            if found:
                found.sort()
                splits[(u, v)] = [q for (_t, q) in found]
        if not splits:
            break
        fresh = []
        stack = list(faces)
        while stack:
            (a, b, c) = stack.pop()
            pieces = None
            for (p, q, w) in ((a, b, c), (b, c, a), (c, a, b)):
                key = (p, q) if p < q else (q, p)
                middles = splits.get(key)
                if middles is None:
                    continue
                if p != key[0]:
                    middles = list(reversed(middles))
                ring = [p] + middles + [q]
                pieces = [(w, ring[i], ring[i + 1])
                          for i in range(len(ring) - 1)]
                break
            if pieces is None:
                fresh.append((a, b, c))
            else:
                stack.extend(pieces)
        faces = fresh
    return faces


def _assemble(polygons, tol, /):
    '''Turns a polygon soup into coordinates and triangles.

    Each polygon is fanned into triangles from its first corner, which is a
    valid triangulation because every polygon the algorithm makes is convex.
    Two kinds of triangle are dropped. One has two corners that weld together,
    and the other is a sliver whose height is below the tolerance: a polygon
    that was cut by another solid's face often has a corner sitting on one of
    its edges, and fanning it then produces triangles that are geometrically
    nothing. Keeping them would cost the caller edges that no neighbour
    shares, for a surface whose area is the same either way. The T-junctions
    that the cuts leave behind are repaired by ``_conform``.
    '''
    if not polygons:
        return (zeros((3, 0)), zeros((3, 0), dtype=int))
    pieces = [polygon.vertices for polygon in polygons]
    points = concatenate(pieces, axis=0)
    (unique, inverse, _origin) = _welded(points, tol)
    starts = cumsum([0] + [len(piece) for piece in pieces[:-1]])
    faces = []
    for (polygon, start) in zip(polygons, starts):
        local = inverse[start:start + len(polygon.vertices)]
        for i in range(1, len(local) - 1):
            (a, b, c) = (int(local[0]), int(local[i]), int(local[i + 1]))
            if a == b or b == c or a == c:
                continue
            (pa, pb, pc) = (unique[a], unique[b], unique[c])
            (u, v) = (pb - pa, pc - pa)
            area = sqrt(float(cross(u, v) @ cross(u, v)))
            longest = max(sqrt(float(u @ u)), sqrt(float(v @ v)),
                          sqrt(float((pc - pb) @ (pc - pb))))
            if area <= tol * longest:
                continue
            faces.append((a, b, c))
    faces = _conform(unique, faces, tol)
    if not faces:
        return (zeros((3, 0)), zeros((3, 0), dtype=int))
    return (unique.T, asarray(faces, dtype=int).T)


def _units_of(*meshes):
    '''The units the result carries: the first mesh's, if it has any.'''
    for mesh in meshes:
        if hasattr(mesh.coords, 'units'):
            return mesh.coords.units
    return None


def _empty_like(mesh, /):
    '''An empty mesh with a mesh's units.'''
    coords = zeros((3, 0))
    units = _units_of(mesh)
    if units is not None:
        coords = coords * units
    return TriMesh(coords, TriTopology(zeros((3, 0), dtype=int),
                                       coord_count=0))


def _checked(first, second, /):
    '''Checks that the arguments are triangle meshes a boolean can use.'''
    if not isinstance(first, TriMesh) or not isinstance(second, TriMesh):
        raise TypeError("expected two triangle meshes")
    if first.dim != 3 or second.dim != 3:
        raise ValueError(
            "boolean operations are defined for triangle meshes in"
            f" three-dimensional space; found dimensions {first.dim} and"
            f" {second.dim}")


def _boolean(first, second, kind, tolerance, /):
    '''The machinery the three operations share.'''
    _checked(first, second)
    # An empty mesh has no bounding box to scale a tolerance by, and it needs
    # none: the shortcuts below answer for it without any cutting.
    if tolerance is None:
        tolerances = [tolerance_of(mesh) for mesh in (first, second)
                      if mesh.coord_count > 0]
        tol = max(tolerances) if tolerances else DEFAULT_TOLERANCE
    else:
        tol = float(tolerance)
    a_points = _vertices_of(first)
    b_points = _vertices_of(second)
    a_faces = _simplified(a_points, first.topo.indices, tol)
    b_faces = _simplified(b_points, second.topo.indices, tol)
    a_tris = _oriented(a_points, a_faces, 'first')
    b_tris = _oriented(b_points, b_faces, 'second')
    a_empty = a_tris.shape[1] == 0
    b_empty = b_tris.shape[1] == 0
    # An empty mesh has no surface to cut against, and the general machinery
    # below would treat it as a solid that contains everything.
    if kind == 'union':
        if a_empty:
            return second
        if b_empty:
            return first
    if kind == 'intersection':
        if a_empty or b_empty:
            return _empty_like(first)
    if kind == 'difference':
        if a_empty:
            return _empty_like(first)
        if b_empty:
            return first
    a_tree = _Node(_polygons(a_points, a_tris), tol)
    b_tree = _Node(_polygons(b_points, b_tris), tol)
    if kind == 'union':
        a_tree.clip_to(b_tree)
        b_tree.clip_to(a_tree)
        b_tree.invert()
        b_tree.clip_to(a_tree)
        b_tree.invert()
        a_tree.build(b_tree.all_polygons())
    elif kind == 'intersection':
        a_tree.invert()
        b_tree.clip_to(a_tree)
        b_tree.invert()
        a_tree.clip_to(b_tree)
        b_tree.clip_to(a_tree)
        a_tree.build(b_tree.all_polygons())
        a_tree.invert()
    else:
        a_tree.invert()
        a_tree.clip_to(b_tree)
        b_tree.clip_to(a_tree)
        b_tree.invert()
        b_tree.clip_to(a_tree)
        b_tree.invert()
        a_tree.build(b_tree.all_polygons())
        a_tree.invert()
    (coords, tris) = _assemble(a_tree.all_polygons(), tol)
    units = _units_of(first, second)
    if units is not None:
        coords = coords * units
    return TriMesh(coords, TriTopology(tris, coord_count=coords.shape[1]))


# Operations #################################################################

def boolean_union(first, second, /, tolerance=None):
    '''The solid that holds the positions either mesh holds.

    The union's surface is made of the parts of each mesh that are outside the
    other, so a position is inside the result exactly when it is inside
    ``first`` or inside ``second`` (or both). If one of the meshes is empty,
    the union is the other one.

    Parameters
    ----------
    first, second : TriMesh
        The two solids, each a closed triangle mesh in three-dimensional space.
        Their triangles need not wind outward --- a mesh that is inside out is
        turned the right way round before it is used.
    tolerance : float or None, optional
        How near a corner may be to a cutting plane and still count as on it.
        The default, ``None``, is a small fraction of the larger mesh's extent,
        which is what makes the same number mean the same thing for a mesh
        measured in millimetres and one measured in kilometres.

    Returns
    -------
    TriMesh
        The union, as a closed triangle mesh whose coordinates are NumPy
        arrays. If either mesh has units, the result carries the first one
        that does.

    Raises
    ------
    TypeError
        If either argument is not a triangle mesh.
    ValueError
        If either mesh occupies fewer than three dimensions, or if either is
        not a closed surface --- its inside is then not a solid, and there is
        no answer to give.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import boolean_union
    >>> corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                     [0., 0., 1.], [1., 1., 0.], [1., 0., 1.],
    ...                     [0., 1., 1.], [1., 1., 1.]]).T
    >>> faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
    ...                   [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
    ...                   [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    >>> cube = trimesh(corners, faces)
    >>> union = boolean_union(cube, cube)
    >>> union.coord_count
    8
    '''
    return _boolean(first, second, 'union', tolerance)


def boolean_intersection(first, second, /, tolerance=None):
    '''The solid that holds the positions both meshes hold.

    The intersection's surface is made of the parts of each mesh that are
    inside the other, so a position is inside the result exactly when it is
    inside ``first`` and inside ``second``. Two solids that do not overlap have
    an empty intersection, which is returned as a mesh with no coordinates.

    Parameters
    ----------
    first, second : TriMesh
        The two solids, each a closed triangle mesh in three-dimensional space.
    tolerance : float or None, optional
        How near a corner may be to a cutting plane and still count as on it.
        The default, ``None``, is a small fraction of the larger mesh's extent.

    Returns
    -------
    TriMesh
        The intersection. It is empty --- no coordinates and no triangles ---
        when the solids do not meet.

    Raises
    ------
    TypeError
        If either argument is not a triangle mesh.
    ValueError
        If either mesh occupies fewer than three dimensions, or if either is
        not a closed surface.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import boolean_intersection
    >>> corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                     [0., 0., 1.], [1., 1., 0.], [1., 0., 1.],
    ...                     [0., 1., 1.], [1., 1., 1.]]).T
    >>> faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
    ...                   [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
    ...                   [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    >>> cube = trimesh(corners, faces)
    >>> apart = trimesh(corners + np.array([[3.], [0.], [0.]]), faces)
    >>> boolean_intersection(cube, apart).coord_count
    0
    '''
    return _boolean(first, second, 'intersection', tolerance)


def boolean_difference(first, second, /, tolerance=None):
    '''The positions the first mesh holds and the second does not.

    The difference's surface is made of the parts of ``first`` outside
    ``second`` and the parts of ``second`` inside ``first``, the latter turned
    inside out: a position is inside the result exactly when it is inside
    ``first`` and not inside ``second``. Subtracting a solid that lies wholly
    inside another therefore leaves a cavity, whose surface winds the other
    way from the outer one.

    Parameters
    ----------
    first, second : TriMesh
        The two solids, each a closed triangle mesh in three-dimensional space.
        The result is ``first`` with ``second`` taken out of it.
    tolerance : float or None, optional
        How near a corner may be to a cutting plane and still count as on it.
        The default, ``None``, is a small fraction of the larger mesh's extent.

    Returns
    -------
    TriMesh
        The difference, as a closed triangle mesh. It is empty when
        ``second`` contains all of ``first``.

    Raises
    ------
    TypeError
        If either argument is not a triangle mesh.
    ValueError
        If either mesh occupies fewer than three dimensions, or if either is
        not a closed surface.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import trimesh
    >>> from euclib.ops import boolean_difference
    >>> corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...                     [0., 0., 1.], [1., 1., 0.], [1., 0., 1.],
    ...                     [0., 1., 1.], [1., 1., 1.]]).T
    >>> faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
    ...                   [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
    ...                   [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    >>> cube = trimesh(corners, faces)
    >>> boolean_difference(cube, cube).coord_count
    0
    '''
    return _boolean(first, second, 'difference', tolerance)


# Exports ####################################################################

__all__ = ('boolean_difference', 'boolean_intersection', 'boolean_union')

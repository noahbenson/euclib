# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_raster.py
'''Turning a two-dimensional path into a grid of covered cells.

A path is a curve in continuous space; a grid is a lattice of cells. Rasterizing
the path decides, for every cell, whether the curve passes through it or
encloses it. The answer is a `Grid` whose property is a boolean image: `True`
where the cell is covered, `False` where it is not.

A closed loop encloses its interior, and the cells whose centers lie inside are
filled. Several loops are combined by the even-odd rule: a cell inside an odd
number of loops is filled and one inside an even number is not, which is what
makes a loop inside a loop a hole. An open path encloses nothing, so it
contributes only its outline; so does a closed one when a width is asked for.
The outline is the set of cells whose centers lie within half the requested
width of a segment.

Which cells a curve passes through is a decision made at the cells' centers.
A cell whose center is exactly on the curve is answered by the same crossing
rule as any other, and a caller who needs a curve to cover a cell it only
touches can ask for a wider outline.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    arange, asarray, ceil, clip, eye, float64, int64, meshgrid, zeros)

from immlib import math as imath, to_array

from ..types import Grid, GridTopology, SegPath
from ._boolean import _units_of, _vertices_of
from ._intersect import DEFAULT_TOLERANCE


# Inputs #####################################################################

def _checked(path, /):
    '''Checks that a geometry is a two-dimensional path.'''
    if not isinstance(path, SegPath):
        raise TypeError(f"expected a SegPath; found {type(path)}")
    if path.dim != 2:
        raise ValueError(
            "rasterization is defined for paths in two-dimensional space;"
            f" found dimension {path.dim}")


def _pitch(value, units, dim, /):
    '''A cell size as a length-``dim`` vector of bare numbers.'''
    if units is not None and hasattr(value, 'units'):
        value = imath.mag(value, units)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size == 1:
        bare = bare.repeat(dim)
    if bare.size != dim:
        raise ValueError(
            f"a raster's pitch must be one length or {dim} lengths; found"
            f" {bare.size}")
    if not (bare > 0.0).all():
        raise ValueError(f"a raster's pitch must be positive; found {bare}")
    return bare


def _origin(value, units, dim, /):
    '''A corner position as a length-``dim`` vector of bare numbers.'''
    if units is not None and hasattr(value, 'units'):
        value = imath.mag(value, units)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size != dim:
        raise ValueError(
            f"a raster's origin must be a length-{dim} position; found"
            f" {bare.size} components")
    return bare


def _resolution(value, dim, /):
    '''A cell count as a length-``dim`` vector of positive integers.'''
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size == 1:
        bare = bare.repeat(dim)
    if bare.size != dim:
        raise ValueError(
            f"a raster's resolution must be one count or {dim} counts; found"
            f" {bare.size}")
    counts = bare.astype(int64)
    if not (counts >= 1).all():
        raise ValueError(
            f"a raster's resolution must be positive; found {bare}")
    return counts


# Geometry ###################################################################

def _welded(points, tol, /):
    '''The distinct positions of a path, and where each position went.

    A closed path is often written with its first corner repeated as its last,
    which makes the closing segment and the first one meet at two coordinate
    indices that carry the same position. The loop test is about the *shape* of
    the path, so the positions are welded first and the segments are read
    against the welded indices.

    Parameters
    ----------
    points : numpy.ndarray
        A ``(2, Q)`` matrix of positions.
    tol : float
        The distance below which two positions are the same position.

    Returns
    -------
    unique : numpy.ndarray
        A ``(2, U)`` matrix of the distinct positions.
    inverse : numpy.ndarray
        A length-``Q`` vector giving each position's column in ``unique``.
    '''
    cell = tol if tol > 0.0 else 1.0
    found = {}
    unique = []
    inverse = zeros(points.shape[1], dtype=int64)
    for i in range(points.shape[1]):
        key = tuple(int(round(float(points[d, i]) / cell)) for d in range(2))
        column = found.get(key)
        if column is None:
            column = len(unique)
            found[key] = column
            unique.append(points[:, i])
        inverse[i] = column
    return (asarray(unique).T, inverse)


def _loops(indices, count, /):
    '''The ordered vertices of each closed component of a path.

    A component is closed when every one of its vertices has two incident
    segments, so it can be walked from a vertex back to itself. A component
    that is open, or that branches, encloses nothing and is not returned.
    '''
    neighbours = [[] for _ in range(count)]
    for k in range(indices.shape[1]):
        (a, b) = (int(indices[0, k]), int(indices[1, k]))
        if a == b:
            continue
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = set()
    loops = []
    for start in range(count):
        if start in seen or not neighbours[start]:
            continue
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
        if any(len(neighbours[v]) != 2 for v in component):
            continue
        current = min(component)
        previous = None
        order = []
        while True:
            order.append(current)
            following = [w for w in neighbours[current] if w != previous]
            if not following:
                break
            following = following[0]
            if following == order[0]:
                break
            (previous, current) = (current, following)
        if len(order) >= 3:
            loops.append(order)
    return loops


def _filled(loops, points, xs, ys, /):
    '''The cells whose centers are inside an odd number of the loops.'''
    (grid_x, grid_y) = meshgrid(xs, ys, indexing='ij')
    (flat_x, flat_y) = (grid_x.reshape(-1), grid_y.reshape(-1))
    inside = zeros(flat_x.shape, dtype=bool)
    for order in loops:
        polygon = points[:, order]
        for k in range(len(order)):
            (x1, y1) = (float(polygon[0, k]), float(polygon[1, k]))
            (x2, y2) = (float(polygon[0, (k + 1) % len(order)]),
                        float(polygon[1, (k + 1) % len(order)]))
            if y1 == y2:
                continue
            crosses = (y1 > flat_y) != (y2 > flat_y)
            crossing = x1 + (x2 - x1) * (flat_y - y1) / (y2 - y1)
            inside ^= crosses & (flat_x < crossing)
    return inside.reshape(grid_x.shape)


def _outlined(points, indices, xs, ys, half, /):
    '''The cells whose centers are within ``half`` of a segment.'''
    (grid_x, grid_y) = meshgrid(xs, ys, indexing='ij')
    (flat_x, flat_y) = (grid_x.reshape(-1), grid_y.reshape(-1))
    near = zeros(flat_x.shape, dtype=bool)
    limit = half * half
    for k in range(indices.shape[1]):
        first = points[:, int(indices[0, k])]
        second = points[:, int(indices[1, k])]
        step = second - first
        length = float(step @ step)
        if length == 0.0:
            distance = ((flat_x - first[0]) ** 2
                        + (flat_y - first[1]) ** 2)
        else:
            share = clip(((flat_x - first[0]) * step[0]
                          + (flat_y - first[1]) * step[1]) / length,
                         0.0, 1.0)
            closest_x = first[0] + share * step[0]
            closest_y = first[1] + share * step[1]
            distance = ((flat_x - closest_x) ** 2
                        + (flat_y - closest_y) ** 2)
        near |= distance <= limit
    return near.reshape(grid_x.shape)


# The operation ##############################################################

def rasterize(path, /, *, pitch=None, origin=None, resolution=None, fill=True,
              width=None, name='raster'):
    '''Marks the cells of a grid that a two-dimensional path covers.

    The grid's cells are decided at their centers. A closed loop encloses an
    interior; a cell whose center lies inside an odd number of loops is
    filled, and one inside an even number is left empty, so a loop inside a
    loop makes a hole. An open path encloses nothing, and a closed one need not
    be filled either: ``fill=False`` leaves only the outline.

    The outline is the set of cells whose centers lie within half of
    ``width`` cells of a segment. ``width=None`` draws no outline, so a
    ``fill=False`` raster of an open path is empty; a ``fill=True`` raster
    with no width is the filled interior and nothing else.

    Parameters
    ----------
    path : SegPath
        The path to rasterize, in two-dimensional space. It may have several
        connected components, open or closed.
    pitch : float, sequence of float, or None, optional
        The size of one cell. A single length is used along both axes; a pair
        gives a rectangular cell. The default, ``None``, derives the pitch from
        the path's extent: one two-hundred-and-fifty-sixth of the longer side,
        or, when ``resolution`` is given, the extent divided by the resolution.
        It may be a quantity with units.
    origin : array-like or None, optional
        The global position of the corner of the first cell. The default,
        ``None``, is the path's low corner, two cells further out along each
        axis.
    resolution : int, sequence of int, or None, optional
        The number of cells along each axis. The default, ``None``, is enough
        cells to cover the path from ``origin`` plus two.
    fill : bool, optional
        Whether to fill the interior of closed loops. The default is ``True``.
    width : float or None, optional
        The width of the outline, in cells of the smaller pitch. The default,
        ``None``, draws no outline.
    name : hashable, optional
        The name of the property that holds the raster. The default is
        ``'raster'``.

    Returns
    -------
    Grid
        A grid whose property ``name`` is a boolean image, ``True`` where the
        cell is covered. Its affine maps cell indices to cell centers, so its
        ``origin`` is the corner asked for, and it carries the path's units if
        it has any.

    Raises
    ------
    TypeError
        If ``path`` is not a segment path.
    ValueError
        If the path is not two-dimensional or has no segments, if ``pitch``,
        ``resolution``, or ``width`` is not positive, or if ``origin`` or
        ``resolution`` does not have two components.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import segpath
    >>> from euclib.ops import rasterize
    >>> square = segpath(np.array([[0., 2., 2., 0., 0.],
    ...                            [0., 0., 2., 2., 0.]]))
    >>> image = rasterize(square, pitch=1.0, origin=np.array([0., 0.]),
    ...                   resolution=(2, 2))
    >>> int(image['raster'].sum())
    4
    >>> image.shape
    (2, 2)
    '''
    _checked(path)
    units = _units_of(path)
    if path.coord_count == 0 or path.topo.simplex_count[1] == 0:
        raise ValueError("a raster needs a path with at least one segment")
    points = _vertices_of(path)
    extent = points.max(axis=1) - points.min(axis=1)
    if resolution is not None:
        resolution = _resolution(resolution, 2)
    if pitch is None:
        longest = float(extent.max())
        if not longest > 0.0:
            raise ValueError(
                "a raster's pitch cannot be derived from a path with no"
                " extent; give a pitch")
        if resolution is not None:
            pitch = extent / (resolution + 2.0)
            # A path that runs along one axis has no extent across it; the
            # other axis sets the size for both.
            fallback = longest / float((resolution + 2).max())
            pitch = asarray([entry if entry > 0.0 else fallback
                             for entry in pitch], dtype=float64)
        else:
            pitch = asarray([longest / 256.0, longest / 256.0])
    else:
        pitch = _pitch(pitch, units, 2)
    if origin is None:
        origin = points.min(axis=1) - 2.0 * pitch
    else:
        origin = _origin(origin, units, 2)
    if resolution is None:
        span = points.max(axis=1) - origin
        resolution = ceil(span / pitch).astype(int64) + 2
    if width is not None:
        if units is not None and hasattr(width, 'units'):
            width = imath.mag(width, units)
        width = float(to_array(imath.mag(width), detach=True).reshape(-1)[0])
        if not width >= 0.0:
            raise ValueError(
                f"a raster's outline width must not be negative; found"
                f" {width}")
    (nx, ny) = (int(resolution[0]), int(resolution[1]))
    xs = origin[0] + (arange(nx) + 0.5) * pitch[0]
    ys = origin[1] + (arange(ny) + 0.5) * pitch[1]
    values = zeros((nx, ny), dtype=bool)
    if fill:
        tolerance = DEFAULT_TOLERANCE * max(float(extent.max()), 1.0)
        (welded, inverse) = _welded(points, tolerance)
        segments = inverse[asarray(path.topo.indices)]
        values |= _filled(_loops(segments, welded.shape[1]), welded, xs, ys)
    if width is not None and width > 0.0:
        values |= _outlined(points, asarray(path.topo.indices), xs, ys,
                            0.5 * (width * float(pitch.min())))
    affine = eye(3)
    affine[0, 0] = pitch[0]
    affine[1, 1] = pitch[1]
    affine[0, 2] = origin[0] + 0.5 * pitch[0]
    affine[1, 2] = origin[1] + 0.5 * pitch[1]
    if units is not None:
        affine = affine * units
    return Grid(affine, GridTopology((nx, ny))).withprop(name, values)


# Exports ####################################################################

__all__ = ('rasterize',)

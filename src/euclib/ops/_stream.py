# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_stream.py
'''Integrating a velocity field into streamlines.

A vector field on a grid says, at every position, which way the flow goes and
how fast. A streamline is the curve a massless particle would follow: it starts
at a seed position and moves along the field, its direction at every instant the
field's direction there.

The integration is a Runge--Kutta step with the velocity normalized at every
stage, so the curve depends on the field's *direction* rather than its speed ---
which is what makes a streamline a curve in space rather than a schedule. The
step length is given in the grid's own length units, and the walk stops when it
leaves the grid, reaches a position where the field vanishes, or has taken the
maximum number of steps. A seed whose field is zero everywhere around it gives
no segment rather than a degenerate one.

Several seeds may be given at once. Each produces its own polyline, and the
segments of all of them are returned in one `SegPath`; a seed's polyline is a
connected run of segments, and separate seeds are separate runs.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (
    all as npall, asarray, concatenate, float64, isfinite, ones, zeros)

from immlib import math as imath, to_array

from ..types import Grid, SegPath, SegTopology
from ..types._interp import GRID_TOLERANCE
from ._boolean import _units_of
from ._intersect import DEFAULT_TOLERANCE


# Helpers ####################################################################

def _checked(grid, vector, /):
    '''Checks the grid and the property that holds its velocity field.'''
    if not isinstance(grid, Grid):
        raise TypeError(f"expected a Grid; found {type(grid)}")
    if grid.dim < 2:
        raise ValueError(
            "a streamline needs a field on a grid of at least two dimensions;"
            f" found a {grid.dim}-dimensional grid")
    prop = grid.propinfo(vector)
    channel = tuple(prop.channel_shape)
    if channel != (grid.dim,):
        raise ValueError(
            "a streamline's velocity field must have one component per"
            f" spatial dimension; property {vector!r} has channel shape"
            f" {channel}, and the grid has {grid.dim} dimensions")
    return prop


def _bare_positions(value, units, dim, /):
    '''A ``(D, Q)`` matrix of positions as bare numbers, or one position.'''
    if units is not None and hasattr(value, 'units'):
        value = imath.mag(value, units)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64)
    if bare.ndim == 1:
        bare = bare.reshape(-1, 1)
    if bare.ndim != 2 or bare.shape[0] != dim:
        raise ValueError(
            f"a start position must be a length-{dim} vector or a"
            f" ({dim}, Q) matrix; found shape {tuple(bare.shape)}")
    return bare


def _length(value, units, /):
    '''A length as a bare number in the grid's units.'''
    if units is not None and hasattr(value, 'units'):
        value = imath.mag(value, units)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size != 1:
        raise ValueError(f"a step length must be one number; found {bare.size}")
    return float(bare[0])


def _inside(grid, positions, slack, /):
    '''Which of a matrix of positions lie in the grid's region.

    A grid's cells are indexed from their centers, so its region runs from half
    a cell before the first center to half a cell past the last. The comparison
    is made in index space, where those bounds are the same along every axis,
    and ``slack`` is how far past them a position may reach and still count.
    '''
    local = to_array(imath.mag(grid.affine_inverse.apply(positions)),
                     detach=True)
    inside = asarray(local, dtype=float64)
    ok = ones(inside.shape[1], dtype=bool)
    for (axis, size) in enumerate(grid.shape):
        ok &= (inside[axis] >= -0.5 - slack)
        ok &= (inside[axis] <= size - 0.5 + slack)
    return ok


def _field(grid, vector, position, /):
    '''The velocity at one position, as a bare ``(D,)`` vector, or ``None``.

    ``None`` covers both a position where the property has no value --- a masked
    cell, or one whose value is not finite --- and a position where the field
    vanishes, which is where a streamline stops.
    '''
    value = grid.prop(vector, at=position)
    bare = to_array(imath.mag(value), detach=True)
    bare = asarray(bare, dtype=float64).reshape(-1)
    if bare.size != grid.dim or not npall(isfinite(bare)):
        return None
    return bare


def _direction_of(velocity, /):
    '''A velocity as a unit direction, or ``None`` if it is too small.'''
    speed = float((velocity @ velocity) ** 0.5)
    if not isfinite(speed) or speed <= 0.0:
        return None
    return velocity / speed


def _advance(grid, vector, position, step, order, direction, slack, /):
    '''One Runge--Kutta step from a position, or ``None`` if it cannot be made.

    The velocity is normalized at every stage, so the step is a distance rather
    than a quantity of time: ``step`` is how far the curve advances, whatever
    the field's magnitude. A stage that leaves the grid, or reaches a position
    at which the field vanishes, stops the whole step --- a partial step would
    put the curve where the field has not said it should go.
    '''
    def at(point, /):
        velocity = _field(grid, vector, point[:, None])
        if velocity is None:
            return None
        unit = _direction_of(velocity)
        if unit is None:
            return None
        return direction * unit

    first = at(position)
    if first is None:
        return None
    if order == 2:
        middle = position + 0.5 * step * first
        if not _inside(grid, middle[:, None], slack)[0]:
            return None
        second = at(middle)
        if second is None:
            return None
        return position + step * second
    second = at(position + 0.5 * step * first)
    if second is None:
        return None
    third = at(position + 0.5 * step * second)
    if third is None:
        return None
    fourth = at(position + step * third)
    if fourth is None:
        return None
    return position + (step / 6.0) * (first + 2.0 * second + 2.0 * third
                                      + fourth)


def _walk(grid, vector, seed, step, order, direction, max_steps, slack, /):
    '''The polyline one seed traces through a field, as a ``(D, N)`` matrix.'''
    points = [seed]
    position = seed
    for _ in range(int(max_steps)):
        following = _advance(grid, vector, position, step, order, direction,
                             slack)
        if following is None or not _inside(grid, following[:, None], slack)[0]:
            break
        points.append(following)
        position = following
    return asarray(points, dtype=float64).T


# The operation ##############################################################

def streamlines(grid, vector, /, *, start=None, step=None, order=4,
                direction=1, max_steps=2000, tolerance=None):
    '''Traces the curves a velocity field on a grid carries a particle along.

    The field is a property of the grid whose channel shape is the grid's
    dimension: one component per axis. Starting from each seed, the path is
    advanced by a Runge--Kutta step along the field's *direction*, normalized at
    every stage, so ``step`` is a distance along the curve rather than an
    amount of time. The walk stops when it leaves the grid, when it reaches a
    position where the field vanishes or is missing, or when ``max_steps``
    steps have been taken.

    Several seeds may be given at once. Each traces its own polyline, and all of
    them are returned in one `SegPath`: a seed's polyline is a connected run of
    segments, and separate seeds are separate runs. A seed whose field vanishes
    where it stands contributes no segment.

    Parameters
    ----------
    grid : Grid
        The grid the field is defined on, in two or three dimensions.
    vector : hashable
        The name of the property that holds the field. Its channel shape must
        be ``(D,)``, one component per spatial dimension.
    start : array-like or None, optional
        The seed positions, as a length-``D`` vector or a ``(D, Q)`` matrix.
        The default, ``None``, seeds one streamline at the center of the grid.
    step : float or None, optional
        The distance one integration step advances along the curve. The
        default, ``None``, is half the shortest cell of the grid. It may be a
        quantity with units, which are converted to the grid's own.
    order : int, optional
        The order of the Runge--Kutta step: 2 for the midpoint rule, 4 for the
        classical fourth-order rule. The default is 4.
    direction : int, optional
        ``1`` to integrate along the field, ``-1`` against it. The default is
        1.
    max_steps : int, optional
        The greatest number of steps one seed may take. The default is 2000.
    tolerance : float or None, optional
        How near the edge of the grid a position may be and still count as
        inside it. The default, ``None``, is a small fraction of the grid's
        scale.

    Returns
    -------
    SegPath
        The streamlines, with the grid's units if it has any. It is empty when
        every seed's field vanishes where it stands.

    Raises
    ------
    TypeError
        If ``grid`` is not a grid.
    ValueError
        If the grid has fewer than two dimensions, if ``vector`` is not a
        property whose channel shape is the grid's dimension, if ``start`` is
        not a position or a matrix of them, if a seed lies outside the grid, if
        ``step`` is not positive, if ``order`` is not 2 or 4, or if
        ``direction`` is not 1 or -1.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import grid
    >>> from euclib.ops import streamlines
    >>> flow = grid((3, 3)).withprop(
    ...     'v', np.broadcast_to(np.array([1.0, 0.0])[:, None, None],
    ...                          (2, 3, 3)))
    >>> path = streamlines(flow, 'v', start=np.array([0.0, 0.0]),
    ...                    step=0.5, max_steps=1)
    >>> path.coord_count
    2
    >>> path.topo.simplex_count[1]
    1
    '''
    _checked(grid, vector)
    units = _units_of(grid)
    if tolerance is None:
        extent = to_array(imath.mag(grid.spacing), detach=True)
        tolerance = DEFAULT_TOLERANCE * max(float(asarray(extent).max()), 1.0)
    else:
        tolerance = float(tolerance)
    if order not in (2, 4):
        raise ValueError(f"a streamline's step order must be 2 or 4; found"
                         f" {order!r}")
    if direction not in (1, -1):
        raise ValueError(
            f"a streamline's direction must be 1 or -1; found {direction!r}")
    if int(max_steps) < 0:
        raise ValueError(
            f"a streamline's max_steps must not be negative; found {max_steps}")
    if step is None:
        spacing = to_array(imath.mag(grid.spacing), detach=True)
        step = 0.5 * float(asarray(spacing).min())
    else:
        step = _length(step, units)
    if not step > 0.0:
        raise ValueError(f"a streamline's step must be positive; found {step}")
    if start is None:
        center = asarray([[(size - 1.0) / 2.0] for size in grid.shape],
                         dtype=float64)
        seed = to_array(imath.mag(grid.affine.apply(center)),
                        detach=True)[:, 0]
        seeds = asarray(seed, dtype=float64).reshape(-1, 1)
    else:
        seeds = _bare_positions(start, units, grid.dim)
    # The slack is in index space; the tolerance is a length, so it is divided
    # by the shortest cell to become one.
    shortest = float(asarray(
        to_array(imath.mag(grid.spacing), detach=True)).min())
    slack = tolerance / shortest + GRID_TOLERANCE
    if not _inside(grid, seeds, slack).all():
        raise ValueError(
            "every streamline seed must lie inside the grid; at least one"
            " does not")
    coordinates = []
    pairs = []
    count = 0
    for k in range(seeds.shape[1]):
        polyline = _walk(grid, vector, seeds[:, k], step, order, direction,
                         max_steps, slack)
        if polyline.shape[1] < 2:
            continue
        coordinates.append(polyline)
        for i in range(polyline.shape[1] - 1):
            pairs.append((count + i, count + i + 1))
        count += polyline.shape[1]
    coords = zeros((grid.dim, 0))
    if coordinates:
        coords = asarray(concatenate(coordinates, axis=1), dtype=float64)
    if units is not None:
        coords = coords * units
    if pairs:
        indices = asarray(pairs, dtype=int).T
    else:
        indices = zeros((2, 0), dtype=int)
    return SegPath(coords, SegTopology(indices, coord_count=coords.shape[1]))


# Exports ####################################################################

__all__ = ('streamlines',)

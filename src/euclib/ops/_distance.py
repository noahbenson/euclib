# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_distance.py
'''Distances between geometries and positions.

Locating a position within a geometry is the primitive that distances are built
from: ``Geometry.to_local`` already answers every position with the nearest
position on the geometry, so a distance is the length of the difference between
a position and its image under that lookup.

These functions examine every simplex of the geometry they are given. The
accelerations that avoid doing so, and the exact simplex-to-simplex separation
that a pair of meshes needs, belong to the releases that add octrees and the
intersection kernels.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import asarray

from immlib import math as imath

from ..abc import Geometry, SimplexGeometry, as_query
from ._cross import positions_of


# Helpers ####################################################################

def _points_of(b, /):
    '''Returns the positions to measure from ``b``.

    A geometry's data does not always live at its ``coords``: a grid's coords
    is an affine matrix and its data lives at its cells, and a prism mesh's
    lives at both of its surfaces. ``positions_of`` is the one place that
    knows where each kind of geometry's data is, so distances are measured
    from what it names rather than from ``coords`` directly.

    Parameters
    ----------
    b : Geometry or array-like
        A geometry, or a ``(D, Q)`` matrix of positions.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix of positions.
    '''
    if isinstance(b, Geometry):
        return positions_of(b)
    return as_query(b)


def _check_source(a, /):
    '''Checks that a geometry may be measured to.

    Parameters
    ----------
    a : object
        The geometry to measure to.

    Raises
    ------
    TypeError
        If it is not a geometry.
    '''
    if not hasattr(a, 'to_local'):
        raise TypeError(f"expected a Geometry; found {type(a)}")


def _check_dim(geom, points, /):
    '''Checks that a set of positions matches a geometry's dimension.'''
    dim = getattr(geom, 'dim', None)
    if dim is not None and int(points.shape[0]) != int(dim):
        raise ValueError(
            f"{type(geom).__name__} occupies {dim}-dimensional space, but the"
            f" positions given have dimension {points.shape[0]}")


def _norm(diff, /):
    '''Returns the length of each column of a difference matrix.'''
    return imath.sqrt(imath.sum(diff * diff, axis=0))


# Operations #################################################################

def distance(a, b, /):
    '''Returns the distance from each position of ``b`` to the geometry ``a``.

    A position inside the geometry has distance zero, because it is its own
    nearest point; a position outside is measured to the nearest point on the
    geometry, so nothing is ever extrapolated.

    Parameters
    ----------
    a : Geometry
        The geometry to measure to.
    b : Geometry or array-like
        The positions to measure from. A geometry's coordinates are used.

    Returns
    -------
    array-like
        A length-``Q`` vector of distances, one per position in ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import TriMesh, TriTopology
    >>> mesh = TriMesh(np.array([[0., 1., 0.], [0., 0., 1.]]),
    ...                TriTopology([[0], [1], [2]]))
    >>> distance(mesh, np.array([[0.25], [0.25]])).tolist()
    [0.0]
    '''
    _check_source(a)
    points = _points_of(b)
    _check_dim(a, points)
    near = a.to_global(a.to_local(points))
    return _norm(points - near)


def nearest(a, b, /):
    '''Returns the nearest position on the geometry ``a`` to each of ``b``.

    Parameters
    ----------
    a : Geometry
        The geometry to search.
    b : Geometry or array-like
        The positions to search for. A geometry's coordinates are used.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix of the nearest positions on ``a``.
    '''
    _check_source(a)
    points = _points_of(b)
    _check_dim(a, points)
    return a.to_global(a.to_local(points))


def separation(a, b, /):
    '''Returns the smallest distance between two geometries.

    The distance is measured from the positions of ``b`` to the geometry ``a``,
    so it is exact when the closest approach between the two is at one of
    ``b``'s coordinates. Two meshes that pass close to each other between their
    vertices are not yet handled; that needs the simplex-to-simplex
    intersection kernels.

    Parameters
    ----------
    a : Geometry
        The geometry to measure to.
    b : Geometry or array-like
        The geometry or positions to measure from.

    Returns
    -------
    array-like
        The smallest distance.
    '''
    # amin is the plain reduction; imath.min returns the value together with
    # its index, as torch's does.
    return imath.amin(distance(a, b), axis=0)


# Exports ####################################################################

__all__ = ('distance', 'nearest', 'separation')

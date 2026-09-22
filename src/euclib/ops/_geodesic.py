# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_geodesic.py
'''Geodesic distances: how far apart things are *along* a geometry.

A geodesic distance is the length of the shortest path that stays on the object.
Two positions on opposite sides of a folded sheet of paper are near one another
through the air and far apart along the paper, and it is the second that a
geodesic measures.

The object is therefore treated as a graph: its coordinates are the vertices,
its edges --- the 1-dimensional simplices its topology implies --- are the
connections, and each connection is as long as the distance between the
coordinates it joins. A geodesic distance is then the shortest path in that
graph, which is what ``scipy``'s Dijkstra routine computes.

Sources are named by *coordinate*, not by position. A source part way along an
edge is not a vertex of the graph, and starting from one would mean splitting
the edge and the paths that run through it; that is left for a later release.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import arange, asarray, full, inf, sqrt, zeros
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ..abc import Geometry, SimplexGeometry
from ..types import Grid


# Helpers ####################################################################

def _source_indices(geom, sources, /):
    '''Turns a set of source coordinates into indices.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry.
    sources : array-like
        A boolean mask over the coordinates, or a sequence of their indices.

    Returns
    -------
    numpy.ndarray
        The indices of the source coordinates.

    Raises
    ------
    ValueError
        If the mask has the wrong length, or a source names no coordinate.
    '''
    values = asarray(sources)
    if values.dtype.kind == 'b':
        if values.shape != (geom.coord_count,):
            raise ValueError(
                f"a source mask must have one entry per coordinate"
                f" ({geom.coord_count}); found {values.shape}")
        return arange(geom.coord_count)[values.reshape(-1)]
    indices = values.reshape(-1).astype(int)
    if indices.size and (indices.min() < 0 or indices.max() >= geom.coord_count):
        raise ValueError(
            f"a source index must name a coordinate, of which there are"
            f" {geom.coord_count}")
    return indices


def _edge_graph(geom, /):
    '''The geometry's coordinates and connections as a sparse graph.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry.

    Returns
    -------
    graph : scipy.sparse.coo_matrix
        A symmetric matrix of edge lengths.
    count : int
        The number of coordinates.
    '''
    # The unit registry may carry units; the graph itself is measured in them.
    coords = asarray(getattr(geom.coords, 'magnitude', geom.coords))
    count = geom.coord_count
    if geom.order < 1:
        # A point cloud has no connections for a path to run along.
        return (coo_matrix((count, count)), count)
    edges = geom.topo.simplices[1]
    apart = coords[:, edges[0]] - coords[:, edges[1]]
    lengths = sqrt((apart * apart).sum(axis=0))
    graph = coo_matrix((lengths, (edges[0], edges[1])), shape=(count, count))
    # A connection runs both ways; ``coo_matrix`` sums the two directions where
    # both were listed, which is right, since a duplicated edge is one edge.
    return ((graph + graph.T).tocoo(), count)


# Operations #################################################################

def geodesic(geom, sources, /):
    '''Measures how far each coordinate is from the nearest source, along the object.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry, whose edges are the paths a distance may follow.
    sources : array-like
        A boolean mask over the coordinates marking the sources, or a sequence
        of their indices.

    Returns
    -------
    numpy.ndarray
        A length-``coord_count`` vector of distances. A coordinate that no path
        reaches is infinite, and a source is at distance zero from itself.

    Raises
    ------
    TypeError
        If the geometry is not a simplex geometry.
    NotImplementedError
        If the geometry is a grid, whose adjacencies are between cells rather
        than along edges, and are not built yet.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import geodesic
    >>> from euclib.types import SegPath, SegTopology
    >>> path = SegPath(np.array([[0., 3., 4.], [0., 0., 0.]]),
    ...                SegTopology([[0, 1], [1, 2]]))
    >>> geodesic(path, np.array([True, False, False])).tolist()
    [0.0, 3.0, 4.0]
    '''
    if isinstance(geom, Grid):
        raise NotImplementedError(
            "a grid's distances run between cells rather than along edges;"
            " that is not built yet")
    if not isinstance(geom, SimplexGeometry):
        raise TypeError(f"expected a SimplexGeometry; found {type(geom)}")
    rows = _source_indices(geom, sources)
    (graph, count) = _edge_graph(geom)
    distances = full(count, inf)
    if rows.size == 0:
        return distances
    # Every source at once: the shortest of the paths from all of them is the
    # distance to the nearest, which is what a geodesic distance field is.
    found = dijkstra(graph, directed=False, indices=rows)
    return found.min(axis=0) if found.ndim > 1 else found


# Exports ####################################################################

__all__ = ('geodesic',)

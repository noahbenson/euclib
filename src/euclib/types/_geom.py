# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_geom.py
'''The concrete geometry types.

Each geometry pairs one of the topologies in ``euclib.types._topo`` with the
data that places it in space. For all four types here that data is a ``(D, N)``
matrix of coordinates, and the geometry's simplices are the columns of its
topology's corner matrix.

Two operations relate a geometry's coordinates to its local coordinates.
``to_local`` locates positions in space, answering each with a simplex index and
the barycentric weights of the position within that simplex; positions outside
the geometry are answered with the nearest position on it. ``to_global`` is the
inverse, turning local coordinates back into positions in space. Both are
implemented by delegating the searching to the kernels in ``euclib.utils``,
which currently examine every simplex; the accelerations that avoid doing so
--- octrees, quadtrees, and their C counterparts --- arrive in a later release.

Each geometry also computes a measure of its own simplices --- a path's segment
lengths, a mesh's surface areas, a tetrahedral mesh's volumes --- and exposes it
as a simplex property under the name ``'length'``, ``'surface_area'``, or
``'volume'``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import asarray
from pcollections import ldict, llist

from ..abc import (
    Property, SimplexGeometry, as_query, calc, check_coordinfo)
from ..utils import closest_simplex, nearest_vertices


# Helpers ####################################################################

def _corner_coords(coords, indices, index, /):
    '''Returns the corners of the simplices named by local indices.

    The local index is converted to a NumPy array first. Indexing a NumPy array
    with a length-1 PyTorch tensor yields a scalar rather than a one-element
    array, which would silently drop the dimension that holds the positions.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    indices : numpy.ndarray
        The topology's ``(K+1, M)`` corner matrix.
    index : array-like
        The simplex index of each local coordinate.

    Returns
    -------
    array-like
        A ``(D, K+1, Q)`` array of the corner coordinates of the chosen
        simplices.
    '''
    return coords[:, indices[:, asarray(index)]]


def _auto_measure_property(measures, topo, order, name, /):
    '''Builds the per-simplex property dictionary that exposes a measure.

    Parameters
    ----------
    measures : array-like
        One measure per simplex of ``order``.
    topo : SimplexTopology
        The geometry's topology.
    order : int
        The simplex order that the measures belong to.
    name : str
        The property name to expose the measures under.

    Returns
    -------
    pcollections.llist
        One dictionary of computed properties per simplex order.
    '''
    res = [ldict() for _ in range(order + 1)]
    res[order] = ldict({name: Property(measures, (topo.simplex_count[order],))})
    return llist(res)


# Geometries #################################################################

class VertexSet(SimplexGeometry):
    '''A point cloud: a set of points in 2- or 3-dimensional space.

    A point cloud's simplices are its points, so its ``coords`` is a ``(D, N)``
    matrix with one column per point, and its local coordinate is simply the
    index of a point. A point has no extent, so a point cloud computes no
    measure of its own.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of point positions.
    topo : VertexTopology
        The point cloud's topology.
    properties : mapping or None, optional
        Coordinate properties.
    backend : str or None, optional
        The numeric backend.
    simplex_properties : sequence or None, optional
        Properties of the points, as a single mapping (there is only one
        simplex order).

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.types import VertexSet, VertexTopology
    >>> cloud = VertexSet(np.array([[0., 1.], [0., 0.]]),
    ...                   VertexTopology([[0, 1]]))
    >>> cloud.coord_count
    2
    '''

    def to_local(self, coords, /):
        '''Locates positions within the point cloud.

        Because a point has no interior, a position is always answered with the
        nearest point.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        VertexLoc
            The index of the nearest point for each position.
        '''
        return self.topo.Loc(nearest_vertices(self.coords,
                                              as_query(coords)))

    def to_global(self, locs, /):
        '''Expresses local coordinates as positions in space.

        Parameters
        ----------
        locs : VertexLoc, mapping, or sequence
            The local coordinates.

        Returns
        -------
        array-like
            A ``(D, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        return _corner_coords(self.coords, self.topo.indices, loc.index)[:, 0]


class SegPath(SimplexGeometry):
    '''A path: a collection of line segments in 2- or 3-dimensional space.

    Each segment is a simplex of order 1, so the corner matrix is ``(2, M)``
    and a local coordinate is a segment index plus one barycentric weight: the
    first of the two barycentric coordinates, which is 1 at the segment's first
    corner and 0 at its second.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    topo : SegTopology
        The path's topology.
    properties : mapping or None, optional
        Coordinate properties.
    backend : str or None, optional
        The numeric backend.
    simplex_properties : sequence or None, optional
        Properties of the simplices, as a mapping for order 0 and one for
        order 1.

    Attributes
    ----------
    measures : array-like
        The length of each segment.
    '''

    @calc('_auto_simplex_properties')
    def proc_auto_simplex_properties(measures, order, topo):
        '''Exposes each segment's length as the ``'length'`` property.

        Returns
        -------
        _auto_simplex_properties : pcollections.llist
            One dictionary of computed properties per simplex order.
        '''
        return _auto_measure_property(measures, topo, order, 'length')

    def to_local(self, coords, /):
        '''Locates positions along the path.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        SegLoc
            The segment containing or nearest each position, and that
            position's first barycentric coordinate within it.
        '''
        (index, weight) = closest_simplex(self.coords, self.topo.indices,
                                          as_query(coords))
        return self.topo.Loc(index, weight)

    def to_global(self, locs, /):
        '''Expresses local coordinates as positions in space.

        Parameters
        ----------
        locs : SegLoc, mapping, or sequence
            The local coordinates.

        Returns
        -------
        array-like
            A ``(D, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        corners = _corner_coords(self.coords, self.topo.indices, loc.index)
        # The weight names the position at the segment's first corner; the
        # second corner's weight is its complement.
        w = loc.weight[0]
        return corners[:, 0] * w + corners[:, 1] * (1.0 - w)


class TriMesh(SimplexGeometry):
    '''A triangle mesh: a collection of triangles in 2- or 3-dimensional space.

    Each triangle is a simplex of order 2, so the corner matrix is ``(3, M)``
    and a local coordinate is a triangle index plus two barycentric weights; the
    third is their complement.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    topo : TriTopology
        The mesh's topology.
    properties : mapping or None, optional
        Coordinate properties.
    backend : str or None, optional
        The numeric backend.
    simplex_properties : sequence or None, optional
        Properties of the simplices, as a mapping for each order from 0 through
        2.

    Attributes
    ----------
    measures : array-like
        The area of each triangle.
    '''

    @calc('_auto_simplex_properties')
    def proc_auto_simplex_properties(measures, order, topo):
        '''Exposes each triangle's area as the ``'surface_area'`` property.

        Returns
        -------
        _auto_simplex_properties : pcollections.llist
            One dictionary of computed properties per simplex order.
        '''
        return _auto_measure_property(measures, topo, order, 'surface_area')

    def to_local(self, coords, /):
        '''Locates positions within the mesh.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        TriLoc
            The triangle containing or nearest each position, and that
            position's first two barycentric weights within it.
        '''
        (index, weight) = closest_simplex(self.coords, self.topo.indices,
                                          as_query(coords))
        return self.topo.Loc(index, weight)

    def to_global(self, locs, /):
        '''Expresses local coordinates as positions in space.

        Parameters
        ----------
        locs : TriLoc, mapping, or sequence
            The local coordinates.

        Returns
        -------
        array-like
            A ``(D, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        corners = _corner_coords(self.coords, self.topo.indices, loc.index)
        w = loc.weight
        last = 1.0 - w.sum(axis=0)
        return (corners[:, 0] * w[0]
                + corners[:, 1] * w[1]
                + corners[:, 2] * last)


class TetMesh(SimplexGeometry):
    '''A tetrahedral mesh: a collection of tetrahedra in 3-dimensional space.

    Each tetrahedron is a simplex of order 3, so the corner matrix is
    ``(4, M)`` and a local coordinate is a tetrahedron index plus three
    barycentric weights; the fourth is their complement. A tetrahedron spans
    three dimensions, so a tetrahedral mesh is only valid in 3-dimensional
    space.

    Parameters
    ----------
    coords : array-like
        A ``(3, N)`` matrix of coordinates.
    topo : TetTopology
        The mesh's topology.
    properties : mapping or None, optional
        Coordinate properties.
    backend : str or None, optional
        The numeric backend.
    simplex_properties : sequence or None, optional
        Properties of the simplices, as a mapping for each order from 0 through
        3.

    Attributes
    ----------
    measures : array-like
        The volume of each tetrahedron.
    '''

    @calc('dim', 'coord_count', lazy=False)
    def proc_coordinfo(coords, topo):
        '''Determines the dimension and coordinate count of the matrix.

        A tetrahedral mesh must occupy 3-dimensional space, which this checks
        in addition to the checks the base class makes.

        Returns
        -------
        dim : int
            The number of rows of ``coords``, which must be 3.
        coord_count : int
            The number of columns of ``coords``.
        '''
        (dim, count) = check_coordinfo(coords, topo, dims=(3,))
        if dim != 3:
            raise ValueError(
                f"a tetrahedral mesh must occupy 3-dimensional space; found"
                f" {dim} dimensions")
        return (dim, count)

    @calc('_auto_simplex_properties')
    def proc_auto_simplex_properties(measures, order, topo):
        '''Exposes each tetrahedron's volume as the ``'volume'`` property.

        Returns
        -------
        _auto_simplex_properties : pcollections.llist
            One dictionary of computed properties per simplex order.
        '''
        return _auto_measure_property(measures, topo, order, 'volume')

    def to_local(self, coords, /):
        '''Locates positions within the mesh.

        Parameters
        ----------
        coords : array-like
            A ``(3, Q)`` matrix of positions.

        Returns
        -------
        TetLoc
            The tetrahedron containing or nearest each position, and that
            position's first three barycentric weights within it.
        '''
        (index, weight) = closest_simplex(self.coords, self.topo.indices,
                                          as_query(coords))
        return self.topo.Loc(index, weight)

    def to_global(self, locs, /):
        '''Expresses local coordinates as positions in space.

        Parameters
        ----------
        locs : TetLoc, mapping, or sequence
            The local coordinates.

        Returns
        -------
        array-like
            A ``(3, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        corners = _corner_coords(self.coords, self.topo.indices, loc.index)
        w = loc.weight
        last = 1.0 - w.sum(axis=0)
        return (corners[:, 0] * w[0]
                + corners[:, 1] * w[1]
                + corners[:, 2] * w[2]
                + corners[:, 3] * last)


# Exports ####################################################################

__all__ = ('VertexSet', 'SegPath', 'TriMesh', 'TetMesh')

# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_topo.py
'''The concrete topology types.

Each of these is a ``SimplexTopology`` whose simplices are of one fixed order,
and each defines the ``Loc`` type that describes a position inside one of its
simplices. A local coordinate names a simplex and gives the barycentric weights
of the position within it; the final weight is implied, since the weights of a
simplex sum to one.

For a point cloud the local coordinate is just the index of the point. For a
path, a triangle, and a tetrahedron, the weights are the first 1, 2, and 3
barycentric coordinates respectively.

Grid topologies are not simplex topologies and belong to the release that
introduces grids; prisms likewise, with their elevation coordinate.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import asarray

from ..abc import (
    SimplexTopology, calc, check_simplex_loc, make_loc, normalize_indices)


# Local coordinate types #####################################################

#: The local coordinate of a point cloud: the index of the point.
VertexLoc = make_loc('VertexLoc', ('index',))

#: The local coordinate of a path: a segment index and one barycentric weight.
SegLoc = make_loc('SegLoc', ('index', 'weight'))

#: The local coordinate of a triangle mesh: a triangle index and two weights.
TriLoc = make_loc('TriLoc', ('index', 'weight'))

#: The local coordinate of a tetrahedral mesh: a tetrahedron index and three
#: barycentric weights.
TetLoc = make_loc('TetLoc', ('index', 'weight'))


# Topologies #################################################################

class VertexTopology(SimplexTopology):
    '''The connectivity of a point cloud: simplices of order 0.

    Each simplex is a single coordinate, so the corner matrix has one row.

    Parameters
    ----------
    indices : array-like
        A ``(1, N)`` integer matrix of coordinate indices.
    coord_count : int or None, optional
        The number of coordinates the topology is valid for. The default,
        ``None``, uses one more than the largest index.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    Loc : type
        ``VertexLoc``, whose only field is ``index``.
    '''

    Loc = VertexLoc

    @calc('indices', lazy=False)
    def proc_indices(indices):
        '''Validates a point cloud's corner matrix.

        Returns
        -------
        indices : numpy.ndarray
            The ``(1, N)`` matrix of coordinate indices.
        '''
        mat = normalize_indices(indices)
        if mat.shape[0] != 1:
            raise ValueError(
                f"a point cloud's indices must have exactly 1 row; found"
                f" {mat.shape[0]}")
        return mat

    def check_loc(self, locs, /):
        '''Coerces and validates a point cloud local coordinate.

        A point cloud's local coordinate names a point and carries no weights,
        because a point has no interior.

        Parameters
        ----------
        locs : VertexLoc, mapping, or sequence
            The local coordinate.

        Returns
        -------
        VertexLoc
            The local coordinate.
        '''
        loc = self.Loc.from_value(locs)
        index = asarray(loc.index)
        if index.ndim != 1:
            raise ValueError(
                f"a local coordinate's index must be a vector; found shape"
                f" {index.shape}")
        return loc


class SegTopology(SimplexTopology):
    '''The connectivity of a path: simplices of order 1.

    Each simplex is a line segment, so the corner matrix has two rows: the
    start and end coordinates of each segment.

    Parameters
    ----------
    indices : array-like
        A ``(2, M)`` integer matrix of segment corners.
    coord_count : int or None, optional
        The number of coordinates the topology is valid for. The default,
        ``None``, uses one more than the largest index.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    Loc : type
        ``SegLoc``, whose fields are ``index`` and ``weight``.
    '''

    Loc = SegLoc

    @calc('indices', lazy=False)
    def proc_indices(indices):
        '''Validates a path's corner matrix.

        Returns
        -------
        indices : numpy.ndarray
            The ``(2, M)`` matrix of segment corners.
        '''
        mat = normalize_indices(indices)
        if mat.shape[0] != 2:
            raise ValueError(
                f"a path's indices must have exactly 2 rows; found"
                f" {mat.shape[0]}")
        return mat

    def check_loc(self, locs, /):
        '''Coerces and validates a path local coordinate.

        Parameters
        ----------
        locs : SegLoc, mapping, or sequence
            The local coordinate: a segment index and a position along it.

        Returns
        -------
        SegLoc
            The local coordinate.
        '''
        return check_simplex_loc(self.Loc, self.local_dim, locs)


class TriTopology(SimplexTopology):
    '''The connectivity of a triangle mesh: simplices of order 2.

    Each simplex is a triangle, so the corner matrix has three rows.

    Parameters
    ----------
    indices : array-like
        A ``(3, M)`` integer matrix of triangle corners.
    coord_count : int or None, optional
        The number of coordinates the topology is valid for. The default,
        ``None``, uses one more than the largest index.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    Loc : type
        ``TriLoc``, whose fields are ``index`` and ``weight``.
    '''

    Loc = TriLoc

    @calc('indices', lazy=False)
    def proc_indices(indices):
        '''Validates a triangle mesh's corner matrix.

        Returns
        -------
        indices : numpy.ndarray
            The ``(3, M)`` matrix of triangle corners.
        '''
        mat = normalize_indices(indices)
        if mat.shape[0] != 3:
            raise ValueError(
                f"a triangle mesh's indices must have exactly 3 rows; found"
                f" {mat.shape[0]}")
        return mat

    def check_loc(self, locs, /):
        '''Coerces and validates a triangle local coordinate.

        Parameters
        ----------
        locs : TriLoc, mapping, or sequence
            The local coordinate: a triangle index and its first two
            barycentric weights.

        Returns
        -------
        TriLoc
            The local coordinate.
        '''
        return check_simplex_loc(self.Loc, self.local_dim, locs)


class TetTopology(SimplexTopology):
    '''The connectivity of a tetrahedral mesh: simplices of order 3.

    Each simplex is a tetrahedron, so the corner matrix has four rows. A
    tetrahedron spans three dimensions, so a tetrahedral mesh is only valid in
    3-dimensional space; the geometry enforces that.

    Parameters
    ----------
    indices : array-like
        A ``(4, M)`` integer matrix of tetrahedron corners.
    coord_count : int or None, optional
        The number of coordinates the topology is valid for. The default,
        ``None``, uses one more than the largest index.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    Loc : type
        ``TetLoc``, whose fields are ``index`` and ``weight``.
    '''

    Loc = TetLoc

    @calc('indices', lazy=False)
    def proc_indices(indices):
        '''Validates a tetrahedral mesh's corner matrix.

        Returns
        -------
        indices : numpy.ndarray
            The ``(4, M)`` matrix of tetrahedron corners.
        '''
        mat = normalize_indices(indices)
        if mat.shape[0] != 4:
            raise ValueError(
                f"a tetrahedral mesh's indices must have exactly 4 rows;"
                f" found {mat.shape[0]}")
        return mat

    def check_loc(self, locs, /):
        '''Coerces and validates a tetrahedron local coordinate.

        Parameters
        ----------
        locs : TetLoc, mapping, or sequence
            The local coordinate: a tetrahedron index and its first three
            barycentric weights.

        Returns
        -------
        TetLoc
            The local coordinate.
        '''
        return check_simplex_loc(self.Loc, self.local_dim, locs)


# Exports ####################################################################

__all__ = (
    'VertexLoc', 'SegLoc', 'TriLoc', 'TetLoc',
    'VertexTopology', 'SegTopology', 'TriTopology', 'TetTopology')

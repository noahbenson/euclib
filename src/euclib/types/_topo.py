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

from numpy import asarray, concatenate, stack

from ..abc import (
    SimplexTopology, Topology, calc, check_simplex_loc, make_loc,
    normalize_indices)


# Local coordinate types #####################################################

VertexLoc = make_loc(
    'VertexLoc', ('index',),
    "The local coordinate of a point cloud: the index of the point.")

SegLoc = make_loc(
    'SegLoc', ('index', 'weight'),
    "The local coordinate of a path: the index of the segment and the weight"
    " of its first corner. The weight is 1 at the segment's first corner and 0"
    " at its second, so a position part way along the segment takes a value"
    " between the two.")

TriLoc = make_loc(
    'TriLoc', ('index', 'weight'),
    "The local coordinate of a triangle mesh: the index of the triangle and"
    " the first two of its three barycentric weights. The third is whatever"
    " makes them sum to one.")

TetLoc = make_loc(
    'TetLoc', ('index', 'weight'),
    "The local coordinate of a tetrahedral mesh: the index of the tetrahedron"
    " and the first three of its four barycentric weights. The fourth is"
    " whatever makes them sum to one.")


PrismLoc = make_loc(
    'PrismLoc', ('index', 'weight', 'height'),
    "The local coordinate of a prism: the index of the triangle, the first two"
    " of its three barycentric weights within that triangle, and the elevation"
    " between the prism's two sides. A prism's two sides share one triangle"
    " topology, so naming a position in it needs the elevation as well as the"
    " position within the triangle.")

#: A grid's components are scaled --- that is, fractional --- index
#: coordinates, one per axis of the grid. The fields are named after the
#: axes, so a grid's local coordinate reads as a position in index space.
GridLoc1 = make_loc(
    'GridLoc1', ('sx',),
    "The local coordinate of a one-dimensional grid: the fractional index"
    " along its single axis.")

GridLoc2 = make_loc(
    'GridLoc2', ('sx', 'sy'),
    "The local coordinate of a two-dimensional grid: the fractional indices"
    " along its two axes, `sx` and `sy`.")

GridLoc3 = make_loc(
    'GridLoc3', ('sx', 'sy', 'sz'),
    "The local coordinate of a three-dimensional grid: the fractional indices"
    " along its three axes, `sx`, `sy`, and `sz`.")

#: The grid local coordinate type for each supported number of dimensions.
GRID_LOCS = {1: GridLoc1, 2: GridLoc2, 3: GridLoc3}


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


class PrismTopology(TriTopology):
    '''The connectivity of a prism mesh: two triangle meshes sharing one
    topology.

    A prism is a pair of triangles whose corners are connected: each triangle
    of one surface is joined to the corresponding triangle of the other,
    enclosing a volume. Both surfaces therefore share a single triangle
    topology, which is what a prism mesh stores --- which is also why a prism's
    coordinates are a pair of coordinate matrices rather than one.

    A position within a prism is named by three numbers rather than two: the
    triangle, the barycentric weights within it, and an *elevation* saying how
    near the position is to the first surface (0) or the second (1).

    Parameters
    ----------
    indices : array-like
        A ``(3, M)`` integer matrix of triangle corners, shared by both
        surfaces.
    coord_count : int or None, optional
        The number of prism positions the topology is valid for. The default,
        ``None``, uses one more than the largest index.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    Loc : type
        ``PrismLoc``, whose fields are ``index``, ``weight``, and ``height``.
    n_sides : int
        The number of surfaces, which is 2.
    tetrahedra : numpy.ndarray
        A lazy ``(4, 3M)`` integer matrix decomposing each prism into three
        tetrahedra, indexing the prism's two surfaces laid end to end: corner
        ``i`` of the first surface and ``N + i`` of the second, where ``N`` is
        the coordinate count.
    '''

    Loc = PrismLoc

    @calc('order', 'dim', 'local_dim', lazy=False)
    def proc_order(indices):
        '''The primary simplex order and the topology's dimensions.

        A prism's primary simplices are triangles, but a position within one
        is named by three numbers rather than two, because the two surfaces add
        an elevation to the position within the triangle.

        Returns
        -------
        order : int
            The number of corners per triangle, minus one.
        dim : int
            The topological dimension, equal to the order.
        local_dim : int
            The number of components in a local coordinate, which is 3.
        '''
        order = normalize_indices(indices).shape[0] - 1
        return (order, order, 3)

    @calc('n_sides', lazy=False)
    def proc_n_sides(indices):
        '''The number of surfaces a prism has.

        Returns
        -------
        n_sides : int
            Two.
        '''
        return 2

    @calc('tetrahedra')
    def proc_tetrahedra(indices, coord_count):
        '''The tetrahedra that each prism decomposes into.

        Each prism splits into three tetrahedra, which is how a prism mesh is
        turned into a tetrahedral mesh. The decomposition is the standard one
        that runs from a corner of the first surface across to the second.

        Returns
        -------
        tetrahedra : numpy.ndarray
            A ``(4, 3M)`` integer matrix whose columns index the prism's
            coordinates, with the second surface's coordinates following the
            first's.
        '''
        idx = normalize_indices(indices)
        n = int(coord_count)
        (a, b, c) = (idx[0], idx[1], idx[2])
        parts = []
        for corners in ((a, b, c, n + a),
                        (b, c, n + a, n + b),
                        (c, n + a, n + b, n + c)):
            parts.append(concatenate([asarray(v)[None, :] for v in corners],
                                     axis=0))
        # The three tetrahedra of one prism occupy three consecutive columns, so
        # that a tetrahedron's index divided by the number of tetrahedra per
        # prism is the index of the prism it belongs to.
        return stack(parts, axis=2).reshape(4, -1)

    def check_loc(self, locs, /):
        '''Coerces and validates a prism local coordinate.

        Parameters
        ----------
        locs : PrismLoc, mapping, or sequence
            The local coordinate: a triangle index, its first two barycentric
            weights, and an elevation.

        Returns
        -------
        PrismLoc
            The local coordinate.

        Raises
        ------
        ValueError
            If the components do not have matching shapes.
        '''
        loc = self.Loc.from_value(locs)
        parts = [asarray(getattr(loc, f))
                 for f in ('index', 'weight', 'height')]
        (index, weight, height) = parts
        if index.ndim != 1:
            raise ValueError(
                f"a local coordinate's index must be a vector; found shape"
                f" {index.shape}")
        if weight.ndim != 2 or weight.shape[0] != 2:
            raise ValueError(
                f"a prism local coordinate's weight must have shape (2, N);"
                f" found {weight.shape}")
        if height.ndim != 2 or height.shape[0] != 1:
            raise ValueError(
                f"a prism local coordinate's height must have shape (1, N);"
                f" found {height.shape}")
        for (f, p) in zip(('weight', 'height'), (weight, height)):
            if p.shape[1] != index.shape[0]:
                raise ValueError(
                    f"a local coordinate's index and {f} disagree:"
                    f" {index.shape[0]} indices but {p.shape[1]} {f}s")
        return loc


class GridTopology(Topology):
    '''The connectivity of a grid image.

    A grid has no simplices: its cells are the unit boxes of an index space
    whose extent is ``shape``. Its local coordinate is a position in that index
    space --- a fractional index per axis --- which is why the local coordinate
    type depends on the number of dimensions: ``GridLoc2(sx, sy)`` for a grid
    of pixels and ``GridLoc3(sx, sy, sz)`` for a grid of voxels.

    Parameters
    ----------
    shape : sequence of int
        The number of cells along each axis, from 2 to 3 axes.
    backend : str or None, optional
        The numeric backend for coordinate data.
    metadata : mapping or None, optional
        Arbitrary hashable metadata.

    Attributes
    ----------
    shape : tuple of int
        The grid's extent.
    dim : int
        The number of axes, which is also the number of local dimensions and
        the dimension of the space the grid occupies.
    coord_count : int
        The number of cells, which is the number of values a grid-shaped
        property has.
    '''

    def __init__(self, shape, backend=None, metadata=None):
        self.shape = shape
        self.backend = backend
        self.metadata = metadata

    @calc('shape', lazy=False)
    def proc_shape(shape):
        '''Validates the grid's extent.

        Returns
        -------
        shape : tuple of int
            The number of cells along each axis.
        '''
        sh = tuple(int(s) for s in shape)
        if len(sh) not in GRID_LOCS:
            raise ValueError(
                f"a grid must have between 1 and 3 axes; found {len(sh)}")
        if any(s < 1 for s in sh):
            raise ValueError(
                f"every axis of a grid must have at least one cell; found"
                f" {sh}")
        # A single-output calc may return its value or a one-tuple holding it,
        # so a tuple *value* needs either that wrapping or the dictionary form;
        # a bare tuple would be read as a sequence of outputs.
        return {'shape': sh}

    @calc('dim', 'local_dim', 'coord_count', lazy=False)
    def proc_gridinfo(shape):
        '''The grid's dimension and number of cells.

        Returns
        -------
        dim : int
            The number of axes.
        local_dim : int
            The number of components in a local coordinate, equal to the
            number of axes.
        coord_count : int
            The number of cells.
        '''
        d = len(shape)
        count = 1
        for s in shape:
            count *= s
        return (d, d, count)

    @property
    def Loc(self):
        '''The local coordinate type for this grid's number of dimensions.'''
        return GRID_LOCS[len(self.shape)]

    def check_loc(self, locs, /):
        '''Coerces and validates a grid local coordinate.

        Parameters
        ----------
        locs : LocMixin, mapping, or sequence
            The local coordinate, whose components are fractional index
            positions, one per axis.

        Returns
        -------
        LocMixin
            The local coordinate.
        '''
        loc = self.Loc.from_value(locs)
        parts = [asarray(getattr(loc, f)) for f in self.Loc._fields]
        shape = parts[0].shape
        for (f, p) in zip(self.Loc._fields[1:], parts[1:]):
            if p.shape != shape:
                raise ValueError(
                    f"the components of a local coordinate must all have the"
                    f" same shape; {self.Loc._fields[0]} has {shape} but {f}"
                    f" has {p.shape}")
        return loc


# Exports ####################################################################

__all__ = (
    'VertexLoc', 'SegLoc', 'TriLoc', 'TetLoc',
    'VertexTopology', 'SegTopology', 'TriTopology', 'TetTopology',
    'PrismLoc', 'PrismTopology',
    'GridLoc1', 'GridLoc2', 'GridLoc3', 'GRID_LOCS', 'GridTopology')

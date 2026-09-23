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

from collections.abc import Mapping

from numpy import arange, asarray, concatenate, eye, meshgrid, stack
from immlib import math as imath, to_array, to_tensor
from pcollections import ldict, llist

from ..abc import (
    Geometry, Property, SimplexGeometry, UNSET, as_query, calc,
    check_coordinfo, split_property_name)
from ..utils import (
    closest_prism, closest_simplex, nearest_vertices, simplex_measures)
from ._topo import (
    GridTopology, PrismTopology, SegTopology, TetTopology, TriTopology,
    VertexTopology)
from ._transform import Affine


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


def _to_like(like, values, /):
    '''Returns values in the backend of a reference array or tensor.'''
    if type(like).__module__.split('.')[0] == 'torch':
        import torch
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)
    return asarray(values)


def _stack_parts(parts, /):
    '''Stacks the components of a local coordinate into a single matrix.

    Parameters
    ----------
    parts : sequence
        The components, one per spatial axis.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix.
    '''
    parts = list(parts)
    if type(parts[0]).__module__.split('.')[0] == 'torch':
        import torch
        return torch.stack(parts, dim=0)
    return asarray(parts)


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
        return self.topo.Loc(nearest_vertices(self.coords, as_query(coords),
                                              self.spatial_index))

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
                                          as_query(coords),
                                          self.spatial_index)
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
                                          as_query(coords),
                                          self.spatial_index)
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
                                          as_query(coords),
                                          self.spatial_index)
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


class PrismMesh(SimplexGeometry):
    '''A prism mesh: a pair of triangle sheets joined corner to corner.

    A prism is two triangles whose corners are connected, enclosing a volume.
    Both surfaces share one triangle topology, so a prism mesh stores a *pair*
    of coordinate matrices --- its ``coords`` is a ``(2, D, N)`` array whose
    first plane holds one surface and whose second holds the other --- and a
    position within it is named by a triangle, the barycentric weights within
    that triangle, and an *elevation* between the two surfaces.

    This is the shape that makes prisms useful for sheets: the two surfaces of
    a thin object can share one tesselation while differing in position, so a
    prism mesh describes a layered structure without duplicating its
    connectivity.

    A prism property may carry an elevation dimension --- a temperature that
    varies through the thickness of a sheet is a ``(E, N)`` matrix, one value
    per elevation and per position --- and the elevations themselves may be
    given alongside the values as a ``(elevs, values)`` pair.

    Parameters
    ----------
    coords : array-like
        A ``(2, D, N)`` array of coordinates, where the first plane is one
        surface's coordinates and the second is the other's.
    topo : PrismTopology
        The prism mesh's topology, shared by both surfaces.
    properties : mapping or None, optional
        Properties, each of whose last dimension is the number of positions.
    backend : str or None, optional
        The numeric backend.
    simplex_properties : sequence or None, optional
        Properties of the simplices, one mapping per simplex order.
    elevations : mapping or None, optional
        The elevation axis of each property that has one, keyed by property
        name. A vector applies to every position; a matrix must match the
        property's values.

    Attributes
    ----------
    coords0, coords1 : array-like
        The two surfaces, each a ``(D, N)`` coordinate matrix.
    tetrahedra : numpy.ndarray
        The tetrahedra that each prism decomposes into.
    '''

    def __init__(self, coords, topo, properties=None, backend=None,
                 simplex_properties=None, elevations=None, metadata=None):
        self.coords = coords
        self.topo = topo
        self.properties = properties
        self.backend = backend
        self.simplex_properties = simplex_properties
        self.elevations = elevations
        self.metadata = metadata

    @calc('coords', lazy=False)
    def proc_coords(coords, backend, topo):
        '''Validates the pair of coordinate matrices.

        Returns
        -------
        coords : array-like
            A ``(2, D, N)`` array of coordinates.
        '''
        if not isinstance(topo, PrismTopology):
            raise ValueError(
                f"a prism mesh's topo must be a PrismTopology; found"
                f" {type(topo)}")
        if backend == 'torch':
            res = to_tensor(coords)
        elif backend == 'numpy' or not hasattr(coords, 'shape'):
            res = to_array(coords)
        else:
            res = coords
        sh = tuple(res.shape)
        if len(sh) != 3 or sh[0] != 2:
            raise ValueError(
                f"a prism mesh's coords must be a (2, D, N) array, one plane"
                f" per surface; found {sh}")
        return res

    @calc('dim', 'coord_count', lazy=False)
    def proc_coordinfo(coords, topo):
        '''Determines the dimension and coordinate count of the pair.

        Returns
        -------
        dim : int
            The number of rows of each surface, which must be 3: a prism
            encloses a volume and so occupies three-dimensional space.
        coord_count : int
            The number of columns of each surface.
        '''
        (planes, dim, count) = (int(s) for s in coords.shape)
        if dim != 3:
            raise ValueError(
                f"a prism mesh must occupy 3-dimensional space; found {dim}"
                f" dimensions")
        if count != topo.coord_count:
            raise ValueError(
                f"coords has {count} positions, but the topology declares"
                f" {topo.coord_count}")
        return (dim, count)

    @calc('coords0')
    def proc_coords0(coords):
        '''The first surface's coordinates.

        Returns
        -------
        coords0 : array-like
            A ``(D, N)`` matrix.
        '''
        return coords[0]

    @calc('coords1')
    def proc_coords1(coords):
        '''The second surface's coordinates.

        Returns
        -------
        coords1 : array-like
            A ``(D, N)`` matrix.
        '''
        return coords[1]

    @calc('tetrahedra')
    def proc_tetrahedra(topo):
        '''The tetrahedra that each prism decomposes into.

        Returns
        -------
        tetrahedra : numpy.ndarray
            A ``(4, 3M)`` integer matrix indexing this mesh's coordinates, with
            the second surface's positions following the first's.
        '''
        return topo.tetrahedra

    @calc('measures')
    def proc_measures(coords, topo):
        '''The volume of each prism.

        A prism is not a simplex, so its measure is not the measure of one: the
        volume is found by decomposing each prism into tetrahedra and adding
        theirs.

        Returns
        -------
        measures : array-like
            A length-``M`` vector of volumes, one per prism.
        '''
        merged = concatenate([coords[0], coords[1]], axis=1)
        tets = topo.tetrahedra
        volumes = simplex_measures(merged, tets)
        # The three tetrahedra of each prism are consecutive columns.
        return imath.sum(volumes.reshape(-1, 3), axis=1)

    @calc('elevations', lazy=False)
    def proc_elevations(elevations, coord_count):
        '''Normalizes the elevation axis of each property that has one.

        Returns
        -------
        elevations : pcollections.ldict
            The elevation axis of each property, keyed by name.
        '''
        if elevations is None:
            return ldict()
        if not isinstance(elevations, Mapping):
            raise ValueError(
                f"elevations must be a mapping or None; found"
                f" {type(elevations)}")
        res = {}
        for (name, elevs) in elevations.items():
            arr = asarray(elevs)
            if arr.ndim not in (1, 2):
                raise ValueError(
                    f"the elevations of {name!r} must be a vector or a matrix;"
                    f" found shape {arr.shape}")
            if arr.ndim == 2 and arr.shape[-1] != coord_count:
                raise ValueError(
                    f"the elevations of {name!r} have {arr.shape[-1]} columns,"
                    f" but there are {coord_count} positions")
            res[name] = arr
        return ldict(res)

    def to_global(self, locs, /):
        '''Expresses local coordinates as positions in space.

        A position within a prism is the position within its triangle on each
        surface, blended by the elevation: at an elevation of 0 it lies on the
        first surface and at 1 on the second.

        Parameters
        ----------
        locs : PrismLoc, mapping, or sequence
            The local coordinates.

        Returns
        -------
        array-like
            A ``(D, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        corners = _corner_coords(self.coords0, self.topo.indices, loc.index)
        weight = loc.weight
        last = 1.0 - weight.sum(axis=0)
        lower = (corners[:, 0] * weight[0] + corners[:, 1] * weight[1]
                 + corners[:, 2] * last)
        corners = _corner_coords(self.coords1, self.topo.indices, loc.index)
        upper = (corners[:, 0] * weight[0] + corners[:, 1] * weight[1]
                 + corners[:, 2] * last)
        height = loc.height[0]
        return lower * (1.0 - height) + upper * height

    def to_local(self, coords, /):
        '''Locates positions within the prism mesh.

        A position within a prism is a nonlinear function of its coordinates
        whenever the two surfaces are not parallel: expanding the blend of the
        surfaces shows ``u * e`` and ``v * e`` terms, so the linear machinery
        that inverts a simplex cannot invert a prism. The search therefore uses
        the tetrahedral decomposition to find the prism and to estimate the
        local coordinates --- which is already exact when the surfaces are
        parallel --- and then refines that estimate until it reproduces the
        position.

        A position *outside* every prism is answered with the nearest position
        on one of them, as it is for the simplex geometries. That is the
        position the tetrahedra give --- they fill the prisms and the search
        clamps to them --- and it is *that* position the refinement solves for,
        rather than the query, since a prism's parameterization can be inverted
        for a position beyond the prism just as well as for one within it.

        Parameters
        ----------
        coords : array-like
            A ``(3, Q)`` matrix of positions.

        Returns
        -------
        PrismLoc
            The prism containing or nearest each position, the position within
            its triangle, and the elevation between the two surfaces.
        '''
        (index, weight, height) = closest_prism(
            self.coords0, self.coords1, self.topo.indices, self.tetrahedra,
            as_query(coords))
        return self.topo.Loc(index, weight, height)


    def elevation(self, height, /):
        '''Returns the triangle mesh at a given elevation.

        At an elevation of 0 the mesh is the first surface and at 1 the second;
        in between it is the linear blend of the two.

        Parameters
        ----------
        height : float
            The elevation, between 0 and 1.

        Returns
        -------
        TriMesh
            The triangle mesh through the prisms at that elevation, sharing
            this mesh's triangle topology.
        '''
        e = float(height)
        topo = TriTopology(self.topo.indices, coord_count=self.coord_count,
                           backend=self.backend)
        return TriMesh(self.coords0 * (1.0 - e) + self.coords1 * e, topo,
                       backend=self.backend)

    def to_tetmesh(self):
        '''Returns the tetrahedral mesh that decomposes this prism mesh.

        Returns
        -------
        TetMesh
            A tetrahedral mesh whose tetrahedra are the three-per-prism
            decomposition of this mesh's prisms.
        '''
        coords = concatenate([self.coords0, self.coords1], axis=1)
        topo = TetTopology(self.topo.tetrahedra,
                           coord_count=2 * self.coord_count,
                           backend=self.backend)
        return TetMesh(coords, topo, backend=self.backend)

    def withprop(self, name, values=UNSET, /, **meta):
        '''Returns a copy of the mesh with a property added or altered.

        This behaves as ``Geometry.withprop`` does, except that a prism
        property may be given its elevations along with its values, as a
        ``(elevs, values)`` pair.

        Parameters
        ----------
        name : hashable or tuple
            The property's name, optionally as a ``(order, name)`` pair.
        values : array-like, tuple, or Ellipsis, optional
            The property's values, or an ``(elevs, values)`` pair.
        **meta
            Metadata for the property.

        Returns
        -------
        PrismMesh
            A copy of the mesh with the property set.
        '''
        elevs = None
        if (values is not UNSET and isinstance(values, tuple)
                and len(values) == 2):
            (elevs, values) = values
        (order, pname) = split_property_name(name)
        res = super().withprop((order, pname) if order is not None else pname,
                               values, **meta)
        if elevs is None:
            return res
        kept = dict(res.elevations)
        kept[pname] = asarray(elevs)
        return res.copy(elevations=ldict(kept))


class Grid(Geometry):
    '''A grid image: pixels or voxels laid out on a regular grid.

    A grid is not made of simplices, and it does not store its coordinates. It
    stores the *affine transformation* that maps its index space --- the
    integer positions of its cells --- into global coordinates, together with
    the extent of that index space. Its ``coords`` payload is therefore an
    affine matrix rather than a coordinate matrix, which is what makes a grid
    cheap to make and to move: a grid of a million voxels describes itself with
    four numbers per axis.

    A grid's local coordinate is a position in index space, with one fractional
    axis per dimension: ``GridLoc2(sx, sy)`` for pixels, ``GridLoc3(sx, sy,
    sz)`` for voxels.

    Parameters
    ----------
    coords : array-like
        The ``(D+1, D+1)`` affine matrix that maps index space to global
        coordinates, where ``D`` is the grid's number of dimensions. Its final
        row must be ``[0, ..., 0, 1]``.
    topo : GridTopology
        The grid's topology, which carries its extent.
    properties : mapping or None, optional
        Properties, each of whose spatial dimensions equal the grid's extent.
    backend : str or None, optional
        The numeric backend.

    Attributes
    ----------
    affine : Affine
        The transform from index space to global coordinates.
    shape : tuple of int
        The grid's extent, from its topology.
    origin : array-like
        The global position of the index-space origin.
    spacing : array-like
        The length of one index step along each axis.
    '''

    def __init__(self, coords, topo, properties=None, backend=None,
                 metadata=None):
        self.coords = coords
        self.topo = topo
        self.properties = properties
        self.backend = backend
        self.metadata = metadata

    @calc('coords', lazy=False)
    def proc_coords(coords, backend, topo):
        '''Validates the grid's affine matrix.

        Returns
        -------
        coords : array-like
            The ``(D+1, D+1)`` affine matrix.
        '''
        if backend == 'torch':
            res = to_tensor(coords)
        elif backend == 'numpy' or not hasattr(coords, 'shape'):
            res = to_array(coords)
        else:
            res = coords
        d = len(topo.shape)
        if tuple(res.shape) != (d + 1, d + 1):
            raise ValueError(
                f"a {d}-dimensional grid needs a {(d+1, d+1)} affine matrix;"
                f" found {tuple(res.shape)}")
        # Constructing an Affine validates that the final row is [0, ..., 0, 1].
        Affine(res)
        return res

    @calc('dim', 'coord_count', lazy=False)
    def proc_coordinfo(coords, topo):
        '''The grid's dimension and number of cells.

        Returns
        -------
        dim : int
            The number of axes.
        coord_count : int
            The number of cells.
        '''
        if not isinstance(topo, GridTopology):
            raise ValueError(
                f"a grid's topo must be a GridTopology; found {type(topo)}")
        count = 1
        for s in topo.shape:
            count *= s
        return (len(topo.shape), count)

    @calc('shape', lazy=False)
    def proc_shape(topo):
        '''The grid's extent.

        Returns
        -------
        shape : tuple of int
            The number of cells along each axis.
        '''
        return {'shape': tuple(topo.shape)}

    @calc('property_shape', lazy=False)
    def proc_property_shape(topo):
        '''The spatial shape a grid property must have.

        A grid property has the grid's extent: a property of a grid of voxels
        is itself an image.

        Returns
        -------
        property_shape : tuple of int
            The grid's extent.
        '''
        return {'property_shape': tuple(topo.shape)}

    @calc('affine')
    def proc_affine(coords):
        '''The transform from index space to global coordinates.

        Returns
        -------
        affine : Affine
            The grid's affine.
        '''
        return Affine(coords)

    @calc('affine_inverse')
    def proc_affine_inverse(coords):
        '''The transform from global coordinates to index space.

        Returns
        -------
        affine_inverse : Affine
            The inverse of the grid's affine.
        '''
        return Affine(coords).inverse

    @calc('origin')
    def proc_origin(coords):
        '''The global position of the index-space origin.

        Returns
        -------
        origin : array-like
            A length-``D`` vector.
        '''
        return coords[:-1, -1]

    @calc('spacing')
    def proc_spacing(coords):
        '''The length of one index step along each axis.

        Returns
        -------
        spacing : array-like
            A length-``D`` vector of the lengths of the affine matrix's
            columns.
        '''
        matrix = coords[:-1, :-1]
        return imath.sqrt(imath.sum(matrix * matrix, axis=0))

    @calc('bbox')
    def proc_bbox(coords, topo):
        '''The bounding box of the grid in global coordinates.

        Returns
        -------
        bbox : array-like
            A ``(D, 2)`` matrix of the minimum and maximum of each coordinate
            dimension.
        '''
        d = len(topo.shape)
        # The 2**D corners of the index space, each axis running from 0 to one
        # less than its extent.
        axes = [arange(2) * (s - 1) for s in topo.shape]
        mesh = meshgrid(*axes, indexing='ij')
        pts = _to_like(coords, stack([m.reshape(-1) for m in mesh], axis=0))
        world = Affine(coords).apply(pts)
        return concatenate([imath.amin(world, axis=1)[:, None],
                            imath.amax(world, axis=1)[:, None]], axis=1)

    def _transformed_coords(self, transform, /):
        '''Returns the grid's affine matrix with a transform composed onto it.

        Transforming a grid transforms the positions it describes, so the
        transform is composed with the affine rather than applied to a matrix
        of coordinates: the grid keeps its extent and gains a new placement.
        '''
        return transform.matrix @ self.coords

    def to_local(self, coords, /):
        '''Expresses global positions as positions in index space.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        LocMixin
            The index-space positions, one fractional axis each.
        '''
        idx = self.affine_inverse.apply(as_query(coords))
        return self.topo.Loc(*[idx[i] for i in range(idx.shape[0])])

    def to_global(self, locs, /):
        '''Expresses index-space positions as global coordinates.

        Parameters
        ----------
        locs : LocMixin, mapping, or sequence
            The index-space positions.

        Returns
        -------
        array-like
            A ``(D, Q)`` matrix of positions.
        '''
        loc = self.topo.check_loc(locs)
        pts = _stack_parts([getattr(loc, f) for f in loc._fields])
        return self.affine.apply(pts)


# Constructors ###############################################################

def points(coords, vertices=None, properties=None, metadata=None):
    '''Returns a point cloud over a matrix of coordinates.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of point positions.
    vertices : array-like or None, optional
        The coordinates to use, by index. The default, ``None``, uses all of
        them.
    properties : mapping or None, optional
        Coordinate properties.
    metadata : mapping or None, optional
        Metadata to attach to the point cloud.

    Returns
    -------
    VertexSet
        The point cloud.

    Examples
    --------
    >>> import numpy as np
    >>> import euclib
    >>> cloud = euclib.points(np.array([[0., 1.], [0., 0.]]))
    >>> cloud.coord_count
    2
    '''
    coords = asarray(coords)
    count = coords.shape[1]
    if vertices is None:
        vertices = arange(count)
    indices = asarray(vertices).reshape(1, -1)
    return VertexSet(coords, VertexTopology(indices, coord_count=count),
                     properties=properties, metadata=metadata)


def segpath(coords, vertices=None, properties=None, metadata=None):
    '''Returns a path through a matrix of coordinates.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    vertices : array-like or None, optional
        The coordinates to use, by index. The default, ``None``, uses all of
        them, in the order given.
    properties : mapping or None, optional
        Coordinate properties.
    metadata : mapping or None, optional
        Metadata to attach to the path.

    Returns
    -------
    SegPath
        The path, one segment per consecutive pair of coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import euclib
    >>> path = euclib.segpath(np.array([[0., 1., 2.], [0., 0., 0.]]))
    >>> path.topo.simplex_count[1]
    2
    '''
    coords = asarray(coords)
    count = coords.shape[1]
    if vertices is None:
        vertices = arange(count)
    vertices = asarray(vertices).reshape(-1)
    indices = asarray([vertices[:-1], vertices[1:]])
    return SegPath(coords, SegTopology(indices, coord_count=count),
                   properties=properties, metadata=metadata)


def trimesh(coords, corners, properties=None, metadata=None):
    '''Returns a triangle mesh from coordinates and their triangles.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates.
    corners : array-like
        A ``(3, M)`` integer matrix of the coordinates that form each triangle.
    properties : mapping or None, optional
        Coordinate properties.
    metadata : mapping or None, optional
        Metadata to attach to the mesh.

    Returns
    -------
    TriMesh
        The mesh.
    '''
    coords = asarray(coords)
    return TriMesh(coords,
                   TriTopology(corners, coord_count=coords.shape[1]),
                   properties=properties, metadata=metadata)


def tetmesh(coords, corners, properties=None, metadata=None):
    '''Returns a tetrahedral mesh from coordinates and their tetrahedra.

    Parameters
    ----------
    coords : array-like
        A ``(3, N)`` matrix of coordinates.
    corners : array-like
        A ``(4, M)`` integer matrix of the coordinates that form each
        tetrahedron.
    properties : mapping or None, optional
        Coordinate properties.
    metadata : mapping or None, optional
        Metadata to attach to the mesh.

    Returns
    -------
    TetMesh
        The mesh.
    '''
    coords = asarray(coords)
    return TetMesh(coords,
                   TetTopology(corners, coord_count=coords.shape[1]),
                   properties=properties, metadata=metadata)


def prismmesh(coords, corners, properties=None, metadata=None):
    '''Returns a prism mesh from a pair of surfaces and their triangles.

    Parameters
    ----------
    coords : array-like
        A ``(2, D, N)`` array of coordinates, one plane per surface.
    corners : array-like
        A ``(3, M)`` integer matrix of the coordinates that form each
        triangle; both surfaces share it.
    properties : mapping or None, optional
        Coordinate properties.
    metadata : mapping or None, optional
        Metadata to attach to the mesh.

    Returns
    -------
    PrismMesh
        The mesh.
    '''
    coords = asarray(coords)
    return PrismMesh(coords,
                     PrismTopology(corners, coord_count=coords.shape[2]),
                     properties=properties, metadata=metadata)


def grid(shape, affine=None, dtype=None, properties=None, metadata=None):
    '''Returns a grid image of a given extent.

    Parameters
    ----------
    shape : sequence of int
        The number of cells along each axis, from 2 to 3 axes.
    affine : array-like or None, optional
        The ``(D+1, D+1)`` matrix that maps index space to global coordinates.
        The default, ``None``, uses the identity, so that a cell's indices are
        its coordinates.
    dtype : dtype-like or None, optional
        The dtype of the affine matrix. The default, ``None``, keeps the one
        the matrix was given with.
    properties : mapping or None, optional
        Properties, each of the grid's shape.
    metadata : mapping or None, optional
        Metadata to attach to the grid.

    Returns
    -------
    Grid
        The grid.

    Examples
    --------
    >>> import euclib
    >>> grid = euclib.grid((4, 5))
    >>> grid.shape
    (4, 5)
    '''
    shape = tuple(shape)
    if affine is None:
        affine = eye(len(shape) + 1, dtype=dtype)
    elif dtype is not None:
        affine = asarray(affine, dtype=dtype)
    return Grid(affine, GridTopology(shape), properties=properties,
                metadata=metadata)


# Exports ####################################################################

__all__ = ('VertexSet', 'SegPath', 'TriMesh', 'TetMesh', 'PrismMesh', 'Grid',
           'points', 'segpath', 'trimesh', 'tetmesh', 'prismmesh', 'grid')

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

import immlib.math as im
from numpy import (arange, asarray, concatenate, eye, meshgrid, ones, stack,
                   where, zeros)
from immlib import math as imath, to_array, to_tensor
from pcollections import ldict, lazy, llist

from ..abc._geom import _carried_property, _value_of
from ..abc import (
    Geometry, Property, SimplexGeometry, UNSET, as_coords, as_query, calc,
    check_coordinfo, split_property_name)
from ..utils import (
    face_weights, refine_prism,
    closest_prism, closest_simplex, nearest_vertices, simplex_measures)
from ._topo import (
    GridTopology, PrismTopology, SegTopology, TetTopology, TriTopology,
    VertexTopology, prism_tetrahedra)
from ._transform import Affine


# Helpers ####################################################################

def _corner_coords(coords, indices, index, /):
    '''Returns the corners of the simplices named by local indices.

    The local index is converted to a NumPy array first. Indexing a NumPy array
    with a length-1 PyTorch tensor yields a scalar rather than a one-element
    array, which would silently drop the dimension that holds the positions.

    Parameters
    ----------
    coords : array-like or mapping
        A ``(D, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
    coords : array-like or mapping
        A ``(D, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
        query = as_query(coords)
        # The search selects a simplex, and a selection carries no gradient ---
        # which simplex is nearest is a comparison and an `argmin` --- so it
        # detaches. The weights *within* the chosen simplex are the continuous
        # part, so they are computed again here, on the undetached positions and
        # with the arithmetic those and the coordinates use.
        (index, _) = closest_simplex(self.coords, self.topo.indices, query,
                                     self.spatial_index)
        weight = face_weights(self.coords[:, self.topo.indices[:, index]],
                              query)
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
        # Through immlib, because the weight may be a tensor: multiplying a
        # numpy coordinate by one reaches numpy's reflected operator, which
        # converts the tensor and raises.
        return im.mag(im.add(im.multiply(corners[:, 0], w),
                             im.multiply(corners[:, 1],
                                         im.subtract(1.0, w))))


class TriMesh(SimplexGeometry):
    '''A triangle mesh: a collection of triangles in 2- or 3-dimensional space.

    Each triangle is a simplex of order 2, so the corner matrix is ``(3, M)``
    and a local coordinate is a triangle index plus two barycentric weights; the
    third is their complement.

    Parameters
    ----------
    coords : array-like or mapping
        A ``(D, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
        query = as_query(coords)
        # The search selects a simplex, and a selection carries no gradient ---
        # which simplex is nearest is a comparison and an `argmin` --- so it
        # detaches. The weights *within* the chosen simplex are the continuous
        # part, so they are computed again here, on the undetached positions and
        # with the arithmetic those and the coordinates use.
        (index, _) = closest_simplex(self.coords, self.topo.indices, query,
                                     self.spatial_index)
        weight = face_weights(self.coords[:, self.topo.indices[:, index]],
                              query)
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
        last = im.subtract(1.0, im.sum(w, axis=0))
        return im.mag(im.add(im.add(im.multiply(corners[:, 0], w[0]),
                                    im.multiply(corners[:, 1], w[1])),
                             im.multiply(corners[:, 2], last)))


class TetMesh(SimplexGeometry):
    '''A tetrahedral mesh: a collection of tetrahedra in 3-dimensional space.

    Each tetrahedron is a simplex of order 3, so the corner matrix is
    ``(4, M)`` and a local coordinate is a tetrahedron index plus three
    barycentric weights; the fourth is their complement. A tetrahedron spans
    three dimensions, so a tetrahedral mesh is only valid in 3-dimensional
    space.

    Parameters
    ----------
    coords : array-like or mapping
        A ``(3, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
        query = as_query(coords)
        # The search selects a simplex, and a selection carries no gradient ---
        # which simplex is nearest is a comparison and an `argmin` --- so it
        # detaches. The weights *within* the chosen simplex are the continuous
        # part, so they are computed again here, on the undetached positions and
        # with the arithmetic those and the coordinates use.
        (index, _) = closest_simplex(self.coords, self.topo.indices, query,
                                     self.spatial_index)
        weight = face_weights(self.coords[:, self.topo.indices[:, index]],
                              query)
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
        last = im.subtract(1.0, im.sum(w, axis=0))
        return im.mag(im.add(im.add(im.multiply(corners[:, 0], w[0]),
                                    im.multiply(corners[:, 1], w[1])),
                             im.add(im.multiply(corners[:, 2], w[2]),
                                    im.multiply(corners[:, 3], last))))


def _tetlayer(coords0, coords1, indices, coord_count, elevations, backend, /):
    """The stack of layers at a set of elevations, as a tetrahedral mesh.

    A property may carry more elevations than the geometry has surfaces, and the
    way those interpolate is to fill the *stack*: every layer is the triangle
    surface `PrismMesh.elevation` gives at that elevation, and the tetrahedra are
    `prism_tetrahedra`'s fan between each pair of adjacent layers. A position's
    height then falls between two layers, and those are the two layers the
    tetrahedron it lies in has for corners --- so the blend the property wants is
    the one the tetrahedral methods already give, at whatever order they are
    asked for.

    The layer coordinates are the geometry's own two surfaces blended, which is
    exactly what `elevation` does for one of them.

    Module-level, and called from inside a `pcollections.lazy`: the lazy closes
    over the fields, so nothing of the mesh is in scope by the time it runs.
    """
    steps = asarray(elevations, dtype='float64')
    corners = concatenate([coords0 * (1.0 - t) + coords1 * t for t in steps],
                          axis=1)
    layers = int(steps.size)
    topo = TetTopology(prism_tetrahedra(indices, layers, coord_count),
                       coord_count=layers * int(coord_count), backend=backend)
    return TetMesh(corners, topo, backend=backend)


def prism_layer_values(values, indices, coord_count, /):
    """The values a stack of layers gives its vertices.

    Parameters
    ----------
    values : array-like
        A ``(C..., K, M, 3)`` array of a prism property's values: the channel
        dimensions leading, then one row per elevation, per triangle, per
        corner.
    indices : array-like
        A ``(3, M)`` integer matrix of triangle corners, the prism mesh's own
        coordinates.
    coord_count : int
        How many coordinates the prism mesh has, so that layer ``k``\'s vertices
        are ``k * coord_count + i``.

    Returns
    -------
    array-like
        A ``(C..., K, N)`` array, one column per coordinate of one layer, in the
        backend of ``values``.
    """
    values = asarray(values)
    indices = asarray(indices)
    (layers, triangles) = (values.shape[-3], values.shape[-2])
    count = int(coord_count)
    # The flat order of the values is ``triangle * 3 + corner``, which is the
    # transpose of the indices' own, so the two are read in the same order.
    named = indices.T.ravel()
    # One column per coordinate, holding the mean of the triangles naming it.
    shares = zeros((triangles * 3, count))
    shares[arange(triangles * 3), named] = 1.0
    seen = shares.sum(axis=0)
    shares = shares * where(seen > 0, 1.0 / where(seen > 0, seen, 1.0), 0.0)
    # One contraction over the values' triangle-and-corner axis at once.
    flat = im.mag(im.reshape(values, values.shape[:-2] + (triangles * 3,)))
    joined = im.mag(im.einsum('...kq,qn->...kn', flat, shares))
    return im.mag(im.reshape(joined, values.shape[:-2] + (count,)))



def _prism_property(coords0, coords1, topo, prop, name, coords, backend, /):
    '''A prism property, interpolated at new coordinates.

    A helper rather than a method, because it is called from inside a
    `pcollections.lazy` that a *calc* built: a calc is a function in a class
    body, so it has no `self` to reach the mesh through. What it takes is the
    fields the interpolation needs rather than the mesh itself, so the mesh may
    be collected once the new one exists.

    It rebuilds the prism it interpolates from. That is a little work per read,
    but reads are rare and the alternative --- holding the mesh --- is not.

    A module-level function, so that a name it uses is resolved when it is
    called and not when it is defined; `PrismMesh` is below it.
    '''
    mesh = PrismMesh(stack([coords0, coords1]), topo, backend=backend)
    mesh = mesh.withprop(name, lazy(_value_of, prop))
    return mesh.prop(name, at=coords)


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
        # Through immlib rather than Python's own arithmetic, because the
        # coordinates are the geometry's numbers while the weights may be a
        # tensor: `numpy * tensor` raises from numpy's side with a message that
        # does not name the line. This is also what lets a position carry a
        # derivative back to the local coordinates it came from.
        weight = loc.weight
        last = im.mag(im.subtract(1.0, im.mag(im.sum(im.mag(weight), axis=0))))

        def blended(surface, /):
            '''The position within one surface, at these weights.'''
            corners = _corner_coords(surface, self.topo.indices, loc.index)
            terms = [im.multiply(corners[:, i], w)
                     for (i, w) in enumerate((weight[0], weight[1], last))]
            return im.mag(im.add(im.add(terms[0], terms[1]), terms[2]))

        lower = blended(self.coords0)
        upper = blended(self.coords1)
        height = loc.height[0]
        return im.mag(im.add(im.multiply(lower, im.subtract(1.0, height)),
                             im.multiply(upper, height)))

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

        The search for the prism a position lies in is detached from any
        derivative: which prism holds a position is a *choice*, and a choice has
        no derivative. The Newton refinement that follows is not a choice, so a
        query that carries a gradient gets the refinement re-run where the
        gradient survives, and the coordinates within the prism come back with
        it.

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
        if getattr(coords, 'requires_grad', False):
            # The search is detached, and rightly: which prism holds a position
            # is a *choice*, and a choice has no derivative. What follows it is
            # not a choice --- Newton's method on `p(u, v, e) - query = 0` --- so
            # a query that carries a derivative gets the refinement run again
            # where that derivative survives. The prism each position was found
            # in still comes from the search, so only the coordinates within the
            # prism are re-derived.
            x = concatenate([asarray(weight).T, asarray(height).T], axis=1)
            # `im.mag` unwraps the quantity without cutting the graph --- and a
            # `asarray` here would cut it --- so the coordinates come back as the
            # query's own backend, with the derivative attached.
            refined = refine_prism(self.coords0, self.coords1, asarray(index),
                                   self.topo.indices, x, as_query(coords))
            # The refinement solves for the *query*, which is what the answer is
            # for a position on or within the prism. For one *outside* it, the
            # answer is the nearest position instead --- and that is what the
            # search already found, since the search clamps to the tetrahedra.
            # Refining toward the query there would move off the prism, because a
            # prism's parameterization inverts for a position beyond it just as
            # well as for one within. So the refinement is kept only where the
            # query is *strictly* interior, and the search's own answer --- which
            # is the boundary position there --- is kept elsewhere.
            (u, v, e) = (x[:, 0], x[:, 1], x[:, 2])
            strict = (u > 0.0) & (v > 0.0) & (u + v < 1.0) & (e > 0.0) & (e < 1.0)
            kept = im.mag(im.add(im.multiply(refined, strict[:, None]),
                                 im.multiply(x, (1.0 - strict)[:, None])))
            (weight, height) = (kept[:, :2].T, kept[:, 2][None, :])
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
        built = TriMesh(self.coords0 * (1.0 - e) + self.coords1 * e, topo,
                        backend=self.backend)
        # The properties come along, as they do for `tetmesh`: the surface at an
        # elevation lies inside the prism, so each property can be read at its
        # coordinates. Lazily, since a mesh's properties are usually not read.
        for (name, prop) in self.properties.items():
            try:
                interp = prop.interp
            except AttributeError:
                interp = UNSET
            built = built.withprop(
                name, lazy(_carried_property, self, name, built.coords),
                **({} if interp is UNSET else {'interp': interp}))
        return built

    @calc('tetmesh')
    def proc_tetmesh(coords0, coords1, topo, backend, properties):
        '''The tetrahedral mesh that decomposes this prism mesh.

        The geometry's own two surfaces, laid end to end --- coordinate ``i``
        of the first is ``i`` and of the second is ``N + i`` --- and filled by
        the topology's tetrahedra. A prism mesh is a stack of two surfaces, and
        this is the case of it the geometry itself supplies; a *property* may
        carry more elevations than that, and its stack is built on demand by
        `tetlayer`.

        A calc rather than a method because it is a function of the mesh alone
        and a mesh may want it more than once: the method this replaces
        rebuilt the joined coordinates and the whole topology on every call.

        Returns
        -------
        TetMesh
            A tetrahedral mesh whose tetrahedra are the three-per-prism
            decomposition of this mesh's prisms.
        '''
        built = TetMesh(concatenate([coords0, coords1], axis=1), topo.tettopo,
                        backend=backend)
        # The mesh's properties come along, interpolated onto the tetrahedra's
        # coordinates --- which are the prism's two surfaces, so every one of
        # them lies inside the prism and each property can be read at them.
        # Lazily, since a mesh's properties are usually not read at all.
        for (name, prop) in properties.items():
            try:
                interp = prop.interp
            except AttributeError:
                interp = UNSET
            built = built.withprop(
                name,
                lazy(_prism_property, coords0, coords1, topo, prop, name,
                     built.coords, backend),
                **({} if interp is UNSET else {'interp': interp}))
        return built

    @calc('_tetlayer_cache')
    def proc_tetlayer_cache(properties, elevations, topo, coords0, coords1,
                            coord_count, backend):
        '''The stack of layers each property's elevations ask for.

        A prism's property may name a *vector* of elevations, and its values then
        live on that many layers --- more than the geometry's two surfaces.
        Interpolating one means filling the stack with tetrahedra, which is worth
        doing once: two properties naming the same elevations want the same
        stack, and a mesh may be read repeatedly.

        So this is a mapping from an elevation vector, as a tuple, to the
        property names that use it and a `pcollections.lazy` that builds the mesh
        when `tetlayer` first asks. A property with **matrix** elevations --- one
        elevation per position, which is rare --- is left out: it has no single
        stack to build, and recomputing it on use costs little.

        The mapping is returned as ``(mapping,)``: a calc's return value *is* its
        outputs --- a plain mapping would be read as outputs *named* by its keys,
        and a tuple as one output per name --- so a single output that is itself
        a structure has to be wrapped.

        A `lazy` and not the mesh, because the mesh is built when it is first
        asked for and the same one comes back after that.

        Returns
        -------
        dict
            One entry per distinct elevation vector: the property names that use
            it, and the lazy mesh.
        '''
        entries = {}
        for (pname, _) in properties.items():
            named = elevations.get(pname, None)
            if named is None:
                continue
            named = asarray(named)
            if named.ndim != 1:
                continue
            key = tuple(float(t) for t in named)
            if key not in entries:
                entries[key] = ([], lazy(_tetlayer, coords0, coords1,
                                         topo.indices, coord_count, named,
                                         backend))
            entries[key][0].append(pname)
        return (dict((key, (tuple(names), mesh))
                     for (key, (names, mesh)) in entries.items()),)

    def tetlayer(self, elevations, /):
        '''The stack of layers at a set of elevations, as a tetrahedral mesh.

        Parameters
        ----------
        elevations : array-like
            A vector of elevations, which is what a prism property's may be.

        Returns
        -------
        TetMesh
            The same stack for the same elevations however often it is asked
            for, and shared with any property that names the same elevations.
        '''
        key = tuple(float(t) for t in asarray(elevations))
        entry = self._tetlayer_cache.get(key, None)
        if entry is None:
            raise KeyError(
                f"no layer of this mesh is at the elevations {key}; a stack is"
                f" built for the elevations a *property* names, and this mesh"
                f" names {sorted(self._tetlayer_cache)}")
        return entry[1]()

    def to_tetmesh(self):
        '''Returns the tetrahedral mesh that decomposes this prism mesh.

        Returns
        -------
        TetMesh
            The same mesh `tetmesh` is; this is the name it had before there
            was a calc, kept because a method reads better at a call site.
        '''
        return self.tetmesh

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
        The ``(D+1, D+1)`` affine matrix that maps a cell's index to the global
        coordinates of that cell's *center*, where ``D`` is the grid's number
        of dimensions. Its final row must be ``[0, ..., 0, 1]``.
    topo : GridTopology
        The grid's topology, which carries its extent.
    properties : mapping or None, optional
        Properties, each of whose spatial dimensions equal the grid's extent.
    backend : str or None, optional
        The numeric backend.

    Attributes
    ----------
    affine : Affine
        The transform from index space to global coordinates, an index to a
        cell's center.
    shape : tuple of int
        The grid's extent, from its topology.
    origin : array-like
        The global position of the grid's corner: the corner of the first cell,
        half a step before its center along each axis.
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
        # The affine carries an integer index to the *center* of that cell, the
        # convention every image format in this field uses: the cell numbered
        # ``i`` has its data at ``affine(i)`` and occupies half a step on either
        # side of it. ``origin`` is therefore the corner of the first cell
        # rather than the affine's translation, and a position belongs to the
        # grid over the half-step beyond the first and last centers.
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
    def proc_origin(coords, topo):
        '''The global position of the grid's corner.

        A cell's index names its center, so the grid begins half a step before
        the first index along every axis. This is the corner the README
        describes a grid as being specified by, and it is the only position of
        a grid that is not a cell center.

        Returns
        -------
        origin : array-like
            A length-``D`` vector.
        '''
        d = len(topo.shape)
        # The corner is half a step back along every axis of index space. The
        # affine takes a matrix of positions, so it is handed one and the
        # column is taken back off, leaving the length-``D`` vector.
        corner = _to_like(coords, (-0.5 * ones((d, 1))))
        return Affine(coords).apply(corner)[:, 0]

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
        # The 2**D corners of the grid's region: half a step before the first
        # cell center along each axis to half a step beyond the last, so the
        # box encloses the cells rather than the centers.
        axes = [arange(2) * s - 0.5 for s in topo.shape]
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
    # `as_coords` and not `asarray` around it: the former leaves an
    # array-like alone and converts anything else, which is what is
    # wanted --- the outer `asarray` converted a *tensor* too, and
    # `numpy.asarray` on one that requires a gradient raises rather than
    # sharing its memory, so a geometry could not be built on one.
    coords = as_coords(coords)
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
    coords : array-like or mapping
        A ``(D, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
    # `as_coords` and not `asarray` around it: the former leaves an
    # array-like alone and converts anything else, which is what is
    # wanted --- the outer `asarray` converted a *tensor* too, and
    # `numpy.asarray` on one that requires a gradient raises rather than
    # sharing its memory, so a geometry could not be built on one.
    coords = as_coords(coords)
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
    coords : array-like or mapping
        A ``(D, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
    # `as_coords` and not `asarray` around it: the former leaves an
    # array-like alone and converts anything else, which is what is
    # wanted --- the outer `asarray` converted a *tensor* too, and
    # `numpy.asarray` on one that requires a gradient raises rather than
    # sharing its memory, so a geometry could not be built on one.
    coords = as_coords(coords)
    return TriMesh(coords,
                   TriTopology(corners, coord_count=coords.shape[1]),
                   properties=properties, metadata=metadata)


def tetmesh(coords, corners, properties=None, metadata=None):
    '''Returns a tetrahedral mesh from coordinates and their tetrahedra.

    Parameters
    ----------
    coords : array-like or mapping
        A ``(3, N)`` matrix of coordinates, or a mapping of the axis names
        ``'x'``, ``'y'``, and ``'z'`` whose values are the coordinates along
        them, as the README allows.
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
    # `as_coords` and not `asarray` around it: the former leaves an
    # array-like alone and converts anything else, which is what is
    # wanted --- the outer `asarray` converted a *tensor* too, and
    # `numpy.asarray` on one that requires a gradient raises rather than
    # sharing its memory, so a geometry could not be built on one.
    coords = as_coords(coords)
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
    coords = as_coords(coords)
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

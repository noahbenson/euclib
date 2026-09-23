# -*- coding: utf-8 -*-
###############################################################################
# euclib/abc/_geom.py
'''The ``Geometry`` types: objects that occupy space.

A geometric object pairs a topology with the data that places it in space, and
with a set of user-defined properties attached to its coordinates or its
simplices.

The data that locates an object in space is its ``coords``. ``coords`` is not
always a coordinate matrix: its form is specific to each kind of geometry,
because it holds *whatever* the object needs to report where it lies. A simplex
geometry stores a ``(D, N)`` matrix of coordinates; a prism mesh stores a
``(2, D, N)`` array holding both of its surfaces; a grid stores the affine
matrix that maps its index space to global coordinates. Each concrete type
supplies its own validation for its own ``coords`` form, by redefining the
``proc_coords`` filter; the abstract base does not guess.

Properties come in two families. *Coordinate* properties attach a value to each
coordinate, and live in ``properties``. *Simplex* properties attach a value to
each simplex of a given order, and live in ``simplex_properties``, one entry per
simplex order. Both are lazy dictionaries (``pcollections.ldict``) mapping a
name to a ``Property``. Looking a property up returns the raw values, not the
``Property`` object; ``propinfo`` returns the ``Property``, which is where the
interpolation metadata lives.
'''

# Dependencies ###############################################################

from __future__ import annotations

from collections.abc import Mapping
from abc import abstractmethod

from numpy import asarray, concatenate, integer
from immlib import math as imath, to_array, to_tensor
from pcollections import ldict, llist

from .. import _init
from ._core import MetaObject, calc, normalize_backend, plantypeABC
from ..utils import (
    SpatialTree, content_hash, simplex_boxes, values_equal, simplex_measures)
from ._property import (
    INTERP_QUALITATIVE, INTERP_SUPPORTED, Property, UNSET, is_property)
from ._topo import Topology, SimplexTopology


# Normalization ##############################################################

def normalize_properties(properties, form_shape, /):
    '''Validates a mapping of names to ``Property`` objects.

    Parameters
    ----------
    properties : mapping or None
        A mapping whose keys are property names and whose values are
        ``Property`` objects, or ``None`` for no properties.
    form_shape : tuple of int
        The spatial shape that every property must have.

    Returns
    -------
    pcollections.ldict
        The properties as a lazy dictionary.
    '''
    if properties is None:
        return ldict()
    if not isinstance(properties, Mapping):
        raise ValueError(
            f"properties must be a mapping or None; found {type(properties)}")
    res = {}
    for (name, prop) in properties.items():
        if not is_property(prop):
            raise ValueError(
                f"property {name!r} must be a Property object; found"
                f" {type(prop)}")
        if tuple(prop.form_shape) != tuple(form_shape):
            raise ValueError(
                f"property {name!r} has form shape {tuple(prop.form_shape)},"
                f" but must have {tuple(form_shape)}")
        res[name] = prop
    return ldict(res)


def supported_interp(topo, /):
    '''The interpolations that a geometry with a given topology can honour.

    A point cloud is the special case: its points have no interior, so there is
    no position *within* the cloud at which a value could be interpolated, and
    the only thing its local coordinate can say is which point is nearest.

    Parameters
    ----------
    topo : Topology
        The geometry's topology.

    Returns
    -------
    tuple of (str, int)
        The interpolation method and order pairs that are valid.
    '''
    if getattr(topo, 'order', None) == 0:
        return (INTERP_QUALITATIVE,)
    return INTERP_SUPPORTED


def check_property_interp(prop, topo, /):
    '''Checks that a geometry can honour a property's interpolation.

    The check happens when the property is attached to the geometry, so that a
    property the geometry cannot read fails where it was written rather than
    when it is first read. Only an interpolation the caller *asked* for is
    checked: a property that merely took the default is the geometry's business,
    and a geometry that cannot interpolate at all ignores it.

    Parameters
    ----------
    prop : Property
        The property being attached.
    topo : Topology
        The geometry's topology.

    Raises
    ------
    ValueError
        If the property asks for an interpolation the geometry cannot honour.
    '''
    if not prop.interp_specified:
        return
    supported = supported_interp(topo)
    if tuple(prop.interp) not in supported:
        order = getattr(topo, 'order', None)
        why = ("a point cloud has no interior, so it can only report the"
               " nearest point's value" if order == 0
               else f"its topology has order {order}")
        raise ValueError(
            f"this geometry does not support the interpolation"
            f" {tuple(prop.interp)}; it supports"
            f" {' and '.join(map(str, supported))} because {why}")


def as_query(coords, /):
    '''Returns coordinates as an array-like suitable for point location.

    Parameters
    ----------
    coords : array-like
        A ``(D, Q)`` matrix of positions, or anything convertible to one.

    Returns
    -------
    array-like
        The positions, as a NumPy array unless they were already array-like.
    '''
    return coords if hasattr(coords, 'shape') else asarray(coords)


def check_coordinfo(coords, topo, /, dims=(2, 3)):
    '''Validates a simplex geometry's coordinate matrix against its topology.

    Parameters
    ----------
    coords : array-like
        A validated ``(D, N)`` coordinate matrix.
    topo : SimplexTopology
        The geometry's topology.
    dims : sequence of int, optional
        The permitted numbers of spatial dimensions. The default is ``(2, 3)``.

    Returns
    -------
    dim : int
        The number of rows of ``coords``.
    coord_count : int
        The number of columns of ``coords``.

    Raises
    ------
    ValueError
        If the topology is not a simplex topology, if the coordinate matrix has
        a forbidden number of rows, or if its number of columns disagrees with
        the number of coordinates the topology declares.
    '''
    if not isinstance(topo, SimplexTopology):
        raise ValueError(
            f"topo must be a SimplexTopology; found {type(topo)}")
    (dim, count) = (int(coords.shape[0]), int(coords.shape[1]))
    if dim not in tuple(dims):
        raise ValueError(
            f"coords must have {len(tuple(dims))} rows chosen from {tuple(dims)}"
            f" (one per spatial dimension); found {dim}")
    if count != topo.coord_count:
        raise ValueError(
            f"coords has {count} coordinates, but the topology declares"
            f" {topo.coord_count}; pass a topo whose coord_count matches the"
            " matrix (extra coordinates require the topology to declare them)")
    return (dim, count)


def split_property_name(name, /):
    '''Splits a property name into an optional simplex order and a name.

    A property may be named either as a bare name, which refers to a coordinate
    property, or as a ``(order, name)`` pair, which refers to the property of
    the simplices of that order. ``Ellipsis`` or ``None`` in place of the order
    means the same as a bare name.

    Parameters
    ----------
    name : hashable or tuple
        The name to split.

    Returns
    -------
    order : int or None
        The simplex order, or ``None`` for a coordinate property.
    name : hashable
        The property's name.
    '''
    if isinstance(name, tuple) and len(name) == 2:
        (order, nm) = name
        if order is None or order is Ellipsis:
            return (None, nm)
        if isinstance(order, (int, integer)):
            return (int(order), nm)
    return (None, name)


# Geometry ###################################################################

class Geometry(MetaObject, metaclass=plantypeABC):
    '''The abstract base class of every geometric object.

    A geometry realizes a topology with data that places it in space, and
    carries properties attached to its coordinates. Subclasses define the form
    of ``coords`` that they accept by redefining ``proc_coords`` and
    ``proc_coordinfo``, and implement ``to_local`` and ``to_global``.

    Parameters
    ----------
    coords : array-like
        The data that places the object in space. Its form is specific to the
        kind of geometry; a simplex geometry, for example, takes a ``(D, N)``
        matrix of coordinates.
    topo : Topology
        The object's topology. The number of coordinates it declares must match
        the size of ``coords``.
    properties : mapping or None, optional
        Coordinate properties, mapping a name to a ``Property``. The default,
        ``None``, attaches none.
    backend : str or None, optional
        ``'numpy'``, ``'torch'``, or ``None`` (the default) to leave the data
        in whatever backend it already uses.

    Attributes
    ----------
    coords : array-like
        The validated coordinate payload.
    topo : Topology
        The object's topology.
    properties : pcollections.ldict
        The object's coordinate properties.
    backend : str or None
        The numeric backend of the object's data.
    dim : int
        The dimension of the space the object occupies.
    coord_count : int
        The number of coordinates the object has.
    property_shape : tuple of int
        The spatial shape that a coordinate property must have.
    '''

    def __init__(self, coords, topo, properties=None, backend=None,
                 metadata=None):
        self.coords = coords
        self.topo = topo
        self.properties = properties
        self.backend = backend
        self.metadata = metadata

    @calc('topo', lazy=False)
    def proc_topo(topo):
        '''Validates that the object's topology is a ``Topology``.

        Returns
        -------
        topo : Topology
            The topology.
        '''
        if not isinstance(topo, Topology):
            raise ValueError(
                f"topo must be a Topology; found {type(topo)}")
        return topo

    @calc('backend', lazy=False)
    def proc_backend(backend):
        '''Validates the object's backend.

        Returns
        -------
        backend : str or None
            ``'numpy'``, ``'torch'``, or ``None``.
        '''
        return normalize_backend(backend)

    @calc('coords', lazy=False)
    def proc_coords(coords, backend):
        '''Validates the object's coordinate payload.

        The form of ``coords`` is specific to each kind of geometry, so this
        filter must be redefined by every concrete geometry. The base
        implementation only reports that it has not been.

        Returns
        -------
        coords : array-like
            The validated payload.
        '''
        raise NotImplementedError(
            f"{type(coords).__name__}: this geometry does not define the form"
            " of its coords payload")

    @calc('dim', 'coord_count', lazy=False)
    def proc_coordinfo(coords, topo):
        '''Determines the object's dimension and coordinate count.

        Like ``proc_coords``, this must be redefined by every concrete
        geometry, because it depends on the form of the coordinate payload.

        Returns
        -------
        dim : int
            The dimension of the space the object occupies.
        coord_count : int
            The number of coordinates the object has.
        '''
        raise NotImplementedError(
            "this geometry does not define how to read its coords payload")

    @calc('property_shape', lazy=False)
    def proc_property_shape(coord_count):
        '''The spatial shape that a coordinate property must have.

        Returns
        -------
        property_shape : tuple of int
            ``(coord_count,)``.
        '''
        # A single-output calc may return its value or a one-tuple holding it,
        # so a tuple *value* needs either that wrapping or the dictionary form;
        # a bare tuple would be read as a sequence of outputs.
        return {'property_shape': (coord_count,)}

    @calc('_auto_properties')
    def proc_auto_properties(topo, coord_count):
        '''Properties that the geometry computes for itself.

        This is the hook through which derived quantities --- a mesh's simplex
        measures, for example --- reach ``properties``. The base geometry
        computes none.

        Returns
        -------
        _auto_properties : pcollections.ldict
            The computed properties, keyed by name.
        '''
        return ldict()

    @calc('properties', lazy=False)
    def proc_properties(properties, property_shape, topo, _auto_properties):
        '''Normalizes the object's coordinate properties.

        Computed properties are merged in first, so that a property the user
        supplied under the same name takes precedence. Each property's
        interpolation is checked against what this geometry can honour, so that
        a property this object cannot read fails here rather than when it is
        first read.

        Returns
        -------
        properties : pcollections.ldict
            The coordinate properties.
        '''
        merged = dict(_auto_properties)
        merged.update(normalize_properties(properties, property_shape))
        for prop in merged.values():
            check_property_interp(prop, topo)
        return ldict(merged)

    @abstractmethod
    def to_local(self, coords, /):
        '''Expresses global coordinates in this object's local coordinates.

        Parameters
        ----------
        coords : array-like
            Global coordinates.

        Returns
        -------
        LocMixin
            The local coordinates of the given positions.
        '''

    @abstractmethod
    def to_global(self, locs, /):
        '''Expresses local coordinates as global coordinates.

        Parameters
        ----------
        locs : LocMixin, mapping, or sequence
            Local coordinates.

        Returns
        -------
        array-like
            The global coordinates of the given positions.
        '''

    # Transforming ##########################################################

    def _transformed_coords(self, transform, /):
        '''Returns this geometry's coordinate payload transformed.

        The base implementation transforms a ``(D, N)`` coordinate matrix. A
        geometry whose payload has another form --- a grid, whose payload is an
        affine matrix --- overrides this.
        '''
        return transform.apply(self.coords)

    def transformed(self, transform, /):
        '''Returns a copy of the object with its coordinates transformed.

        Parameters
        ----------
        transform : Transform
            The transformation to apply to the object's coordinates.

        Returns
        -------
        Geometry
            A copy of the object located by the transformed coordinates. Its
            properties are carried over unchanged, because they are attached to
            the object's components rather than to positions in space.
        '''
        if not hasattr(transform, 'apply'):
            raise TypeError(
                f"expected a Transform; found {type(transform)}")
        return self.copy(coords=self._transformed_coords(transform))

    # Properties #############################################################

    def _prop_for(self, name, order, /):
        '''Returns the ``Property`` named ``name``, optionally by simplex order.

        Parameters
        ----------
        name : hashable
            The property's name.
        order : int or None
            The simplex order, or ``None`` for a coordinate property.

        Returns
        -------
        Property
            The property.

        Raises
        ------
        KeyError
            If no such property exists.
        IndexError
            If the simplex order is out of range.
        '''
        if order is not None:
            raise IndexError(
                f"{type(self).__name__} has no simplex properties, so order"
                f" {order} is not valid")
        props = self.properties
        if name not in props:
            raise KeyError(f"no such property: {name!r}")
        return props[name]

    def _prop_value(self, name, order, rest, /):
        '''Returns a property's values, restricted to ``rest`` if given.'''
        prop = self._prop_for(name, order)
        val = prop.value
        return val[(Ellipsis,) + tuple(rest)] if rest else val

    def propinfo(self, name, /):
        '''Returns the ``Property`` object for a property.

        Use this to reach a property's metadata; looking a property up with
        ``self[...]`` or ``prop`` returns only its values.

        Parameters
        ----------
        name : hashable or tuple
            The property's name, optionally as a ``(order, name)`` pair.

        Returns
        -------
        Property
            The property.
        '''
        (order, pname) = split_property_name(name)
        return self._prop_for(pname, order)

    def __getitem__(self, index):
        '''Returns the values of a property, or of several.

        The index may be a bare property name (``geom['flux']``), a
        ``(order, name)`` pair naming a simplex property (``geom[2, 'flux']``),
        a sequence of names returning a tuple of values (``geom[['a', 'b']]``),
        or a name followed by indices applied to the value's spatial dimensions
        (``geom['flux', mask]``).
        '''
        ordered = index if isinstance(index, tuple) else (index,)
        if len(ordered) == 0:
            raise IndexError("empty property index")
        first = ordered[0]
        # Several names at once, as in geom[['a', 'b']].
        if isinstance(first, list):
            return tuple(self[(n,) + tuple(ordered[1:])] for n in first)
        # A (order, name) pair, or a bare name.
        if isinstance(first, (int, integer)) or first is None:
            if len(ordered) < 2:
                raise IndexError(
                    "a simplex property index needs both an order and a name")
            order = None if first is None else int(first)
            return self._prop_value(ordered[1], order, ordered[2:])
        return self._prop_value(first, None, ordered[1:])

    def __setitem__(self, index, value):
        raise TypeError(f"type {type(self)} is immutable")

    def prop(self, property, /, at=UNSET, **kw):
        '''Extracts a property's values, interpolating them if asked to.

        With no ``at`` argument, this returns the property's values as they are
        stored. Given a boolean mask or a sequence of integer indices, it
        returns the values at those components. Given a matrix of global
        positions, or local coordinates supplied as a ``Loc`` or a mapping, it
        interpolates the property at those positions according to the
        property's metadata.

        Only coordinate properties can be interpolated; a property attached to
        simplices is read with ``self[order, name]``.

        Parameters
        ----------
        property : hashable
            The property's name.
        at : Ellipsis, array-like, or None, optional
            Where to extract the property. The default, ``Ellipsis``, returns
            the whole property.
        **kw
            Metadata that overrides the property's own: ``order``, ``extrap``,
            ``null``, and ``mask``.

        Returns
        -------
        array-like
            The property's values.
        '''
        if at is UNSET:
            at = Ellipsis
        (order, pname) = split_property_name(property)
        if order is not None:
            raise ValueError(
                f"prop reads coordinate properties; the simplex property"
                f" ({order}, {pname!r}) is read with geom[{order}, {pname!r}]")
        if at is Ellipsis or at is None:
            return self[pname]
        if isinstance(at, bool) or _is_boolean_mask(at):
            return self[pname, at]
        if isinstance(at, (list, tuple)) and all(
                isinstance(a, (int, integer)) for a in at):
            return self[pname, list(at)]
        # Imported here rather than at module scope: the interpolation engine
        # is a concrete-type concern, and euclib.abc must not import
        # euclib.types.
        from ..types._interp import interpolate
        return interpolate(self, self._prop_for(pname, None), at, **kw)

    # Updating ##############################################################

    def _prop_container(self, order, /):
        '''The mapping of properties for a given order.'''
        if order is not None:
            raise IndexError(
                f"{type(self).__name__} has no simplex properties")
        return self.properties

    def _install_props(self, order, props, /):
        '''Returns the ``copy`` keyword that installs a set of properties.

        Parameters
        ----------
        order : int or None
            The simplex order whose properties are being replaced, or ``None``
            for the coordinate properties.
        props : mapping
            The complete new set of properties for that order.

        Returns
        -------
        dict
            Keyword arguments for ``copy``.
        '''
        if order is not None:
            raise IndexError(
                f"{type(self).__name__} has no simplex properties")
        return {'properties': ldict(props)}

    def _props_updated(self, order, changes, /):
        '''Returns the ``copy`` keyword that merges properties into an order.'''
        props = dict(self._prop_container(order))
        props.update(changes)
        return self._install_props(order, props)

    def withprop(self, name, values=UNSET, /, **meta):
        '''Returns a copy of the object with a property added or altered.

        Given values, the property is created (or replaced) with the supplied
        metadata. Given only metadata --- that is, with no values --- an
        existing property's metadata is updated and its values are left alone.

        Parameters
        ----------
        name : hashable or tuple
            The property's name, optionally as a ``(order, name)`` pair.
        values : array-like or Ellipsis, optional
            The property's new values. The default, ``Ellipsis``, keeps the
            existing values and updates only the metadata.
        **meta
            Metadata for the property, named as in ``Property``.

        Returns
        -------
        Geometry
            A copy of the object with the property set.
        '''
        (order, pname) = split_property_name(name)
        container = self._prop_container(order)
        existing = container.get(pname, None)
        if values is UNSET:
            if existing is None:
                raise KeyError(
                    f"cannot update the metadata of {pname!r}: no such"
                    " property; supply values to create it")
            new = existing.withmeta(**meta)
        else:
            new = Property(values, self._prop_form_shape(order), **meta)
        return self.copy(**self._props_updated(order, {pname: new}))

    def dropprop(self, name, /):
        '''Returns a copy of the object without a property.

        Parameters
        ----------
        name : hashable or tuple
            The property's name, optionally as a ``(order, name)`` pair.

        Returns
        -------
        Geometry
            A copy of the object without the property.

        Raises
        ------
        KeyError
            If the object has no such property.
        '''
        (order, pname) = split_property_name(name)
        container = self._prop_container(order)
        if pname not in container:
            raise KeyError(f"no such property: {pname!r}")
        props = dict(container)
        del props[pname]
        return self.copy(**self._install_props(order, props))

    def _prop_form_shape(self, order, /):
        '''The spatial shape a property of a given order must have.'''
        if order is not None:
            raise IndexError(
                f"{type(self).__name__} has no simplex properties")
        return self.property_shape

    # Comparison ############################################################

    def _eq_data(self, /):
        '''The data that participates in equality.

        Properties are plan outputs and do not take part in equality; only the
        topology and the coordinate payload identify a geometry.
        '''
        return (self.topo, self.coords)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        (at, ac) = self._eq_data()
        (bt, bc) = other._eq_data()
        return at == bt and values_equal(ac, bc)

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __hash__(self):
        (topo, coords) = self._eq_data()
        return hash((type(self), topo, content_hash(coords)))


# Simplex Geometry ###########################################################

class SimplexGeometry(Geometry):
    '''The abstract base class of geometries made of simplices.

    A simplex geometry's coordinates are a ``(D, N)`` matrix, and its simplices
    are the columns of its topology's ``indices`` matrix. Its coordinate
    properties have one value per column of ``coords``; its simplex properties
    have one value per simplex of the order they belong to.

    Parameters
    ----------
    coords : array-like
        A ``(D, N)`` matrix of coordinates, where ``D`` is 2 or 3 and ``N`` is
        the number of coordinates.
    topo : SimplexTopology
        The object's topology.
    properties : mapping or None, optional
        Coordinate properties. The default, ``None``, attaches none.
    backend : str or None, optional
        The numeric backend. The default, ``None``, leaves the data as it is.
    simplex_properties : sequence or None, optional
        Properties of the simplices, as a sequence with one entry per simplex
        order, from 0 through the topology's order. Each entry is a mapping of
        names to ``Property`` objects, or ``None`` for none. The default,
        ``None``, attaches none.

    Attributes
    ----------
    order : int
        The dimension of the primary simplices.
    vertex_mask : numpy.ndarray
        A boolean mask over the coordinates marking those the topology uses.
    vertex_count : int
        The number of coordinates the topology uses.
    simplex_count : pcollections.llist
        The number of simplices at each order.
    simplex_properties : pcollections.llist
        One lazy dictionary of properties per simplex order.
    '''

    def __init__(self, coords, topo, properties=None, backend=None,
                 simplex_properties=None, metadata=None):
        self.coords = coords
        self.topo = topo
        self.properties = properties
        self.backend = backend
        self.simplex_properties = simplex_properties
        self.metadata = metadata

    @calc('coords', lazy=False)
    def proc_coords(coords, backend):
        '''Validates a coordinate matrix and converts it to the backend.

        Returns
        -------
        coords : array-like
            A ``(D, N)`` coordinate matrix, with ``D`` equal to 2 or 3.
        '''
        # Coordinate data must end up array-like, whatever was passed. When no
        # backend is requested, anything that is already array-like is left
        # alone, so that a tensor stays a tensor and a quantity keeps its
        # units; anything else is converted to a NumPy array.
        if backend == 'torch':
            res = to_tensor(coords)
        elif backend == 'numpy' or not hasattr(coords, 'shape'):
            res = to_array(coords, detach=True)
        else:
            res = coords
        sh = tuple(res.shape)
        if len(sh) != 2:
            raise ValueError(
                f"coords must be a (D, N) matrix; found {len(sh)} dimensions")
        if sh[0] not in (2, 3):
            raise ValueError(
                f"coords must have 2 or 3 rows (one per spatial dimension);"
                f" found {sh[0]}")
        return res

    @calc('dim', 'coord_count', lazy=False)
    def proc_coordinfo(coords, topo):
        '''Determines the dimension and coordinate count of the matrix.

        Returns
        -------
        dim : int
            The number of rows of ``coords``.
        coord_count : int
            The number of columns of ``coords``.
        '''
        return check_coordinfo(coords, topo)

    @calc('order', lazy=False)
    def proc_order(topo):
        '''The order of the geometry's primary simplices.

        The order belongs to the topology, not to the coordinates: a point
        cloud has simplices of order 0 while its coordinates may still be
        2- or 3-dimensional.

        Returns
        -------
        order : int
            The topology's simplex order.
        '''
        return topo.order

    @calc('vertex_mask')
    def proc_vertex_mask(topo):
        '''The coordinates that the topology uses.

        Returns
        -------
        vertex_mask : numpy.ndarray
            A boolean array of length ``coord_count``.
        '''
        return topo.vertex_mask

    @calc('vertex_count')
    def proc_vertex_count(topo):
        '''The number of coordinates that the topology uses.

        Returns
        -------
        vertex_count : int
            The number of used coordinates.
        '''
        return topo.vertex_count

    @calc('simplex_count')
    def proc_simplex_count(topo):
        '''The number of simplices at each order.

        Returns
        -------
        simplex_count : pcollections.llist
            The simplex counts, indexed by order.
        '''
        return topo.simplex_count

    @calc('spatial_index')
    def proc_spatial_index(coords, topo):
        '''A spatial index over this geometry's simplices, where one would pay.

        The searches that locate a position examine the simplices near it.
        Below a few hundred simplices there are few enough that examining them
        all, vectorized, is faster than the bookkeeping a subdivided search
        needs; above that the index divides the work by a factor that grows
        with the mesh. The threshold is
        ``euclib._init.spatial_index_min_items``.

        Returns
        -------
        spatial_index : SpatialTree or None
            An index over the simplices, or ``None`` when the geometry is too
            small for one to help.
        '''
        count = topo.simplex_count[topo.order]
        if count < _init.spatial_index_min_items:
            return None
        (centers, radii) = simplex_boxes(coords, topo.indices)
        return SpatialTree(centers, radii)

    @calc('bbox')
    def proc_bbox(coords, vertex_mask):
        '''The bounding box of the coordinates the topology uses.

        Coordinates that the topology does not reference are left out, so that
        a bounding box describes the object rather than the matrix that happens
        to hold it.

        Returns
        -------
        bbox : array-like
            A ``(D, 2)`` matrix whose first column is the minimum of each
            coordinate dimension and whose second column is the maximum.
        '''
        used = coords if vertex_mask is None else coords[(Ellipsis, vertex_mask)]
        # amin/amax are the plain reductions; imath.min/imath.max return the
        # value together with its index, as torch's do.
        return concatenate([imath.amin(used, axis=1)[:, None],
                            imath.amax(used, axis=1)[:, None]], axis=1)

    @calc('measures')
    def proc_measures(coords, topo):
        '''The extent of each primary simplex.

        A simplex's measure is its 0-dimensional extent: the length of a
        segment, the area of a triangle, or the volume of a tetrahedron.
        Vertices have no extent, so a point cloud's measures are all zero.

        Returns
        -------
        measures : array-like
            A length-``M`` vector of measures.
        '''
        return simplex_measures(coords, topo.indices)

    @calc('_auto_simplex_properties')
    def proc_auto_simplex_properties(measures, order):
        '''Properties that the geometry computes about its own simplices.

        This is the hook through which derived per-simplex quantities --- a
        mesh's surface areas, for example --- reach ``simplex_properties``.
        The base geometry computes none.

        Returns
        -------
        _auto_simplex_properties : pcollections.llist
            One lazy dictionary of computed properties per simplex order.
        '''
        return llist(ldict() for _ in range(order + 1))

    @calc('simplex_properties', lazy=False)
    def proc_simplex_properties(simplex_properties, order, topo,
                                _auto_simplex_properties):
        '''Normalizes the properties of each simplex order.

        Computed properties are merged in first, so that a property the user
        supplied under the same name takes precedence.

        Returns
        -------
        simplex_properties : pcollections.llist
            One lazy dictionary of properties per simplex order.
        '''
        n = order + 1
        seq = [None] * n if simplex_properties is None else list(
            simplex_properties)
        if len(seq) != n:
            raise ValueError(
                f"simplex_properties must have {n} entries (one per simplex"
                f" order, 0 through {order}); found {len(seq)}")
        res = []
        for k in range(n):
            merged = dict(_auto_simplex_properties[k])
            merged.update(normalize_properties(seq[k], (topo.simplex_count[k],)))
            for prop in merged.values():
                check_property_interp(prop, topo)
            res.append(ldict(merged))
        return llist(res)

    def _prop_for(self, name, order, /):
        if order is None:
            return super()._prop_for(name, None)
        if not (0 <= order <= self.order):
            raise IndexError(
                f"simplex order {order} is out of range; this geometry has"
                f" orders 0 through {self.order}")
        props = self.simplex_properties[order]
        if name in props:
            return props[name]
        # A vertex (order 0) property may be asked for when only a coordinate
        # property exists; in that case the coordinate property restricted to
        # the coordinates the topology actually uses is the answer.
        if order == 0 and name in self.properties:
            return _restrict_property(
                self.properties[name], self.vertex_mask, self.vertex_count)
        raise KeyError(f"no such simplex property: ({order}, {name!r})")

    def _prop_container(self, order, /):
        if order is None:
            return self.properties
        return self.simplex_properties[order]

    def _install_props(self, order, props, /):
        # simplex_properties is a sequence with one entry per simplex order, so
        # replacing the properties of one order means rebuilding the sequence;
        # installing the entry itself in place of the sequence is an error.
        if order is None:
            return super()._install_props(None, props)
        seq = [dict(d) for d in self.simplex_properties]
        seq[order] = props
        return {'simplex_properties': llist(seq)}

    def _prop_form_shape(self, order, /):
        if order is None:
            return self.property_shape
        return (self.simplex_count[order],)

    def _eq_data(self, /):
        # Only the coordinates the topology uses identify the geometry; extra
        # coordinates are ignored, as the README requires.
        return (self.topo, self.coords[(Ellipsis, self.vertex_mask)]
                if self.vertex_mask is not None else self.coords)

    def __getitem__(self, index):
        ordered = index if isinstance(index, tuple) else (index,)
        # `geom[prop]` and `geom[0, prop]` differ for simplex geometries: the
        # former is the coordinate property for every coordinate, while the
        # latter is the vertex property, which falls back to the coordinate
        # property restricted to the coordinates the topology uses.
        if len(ordered) == 1 and not isinstance(
                ordered[0], (list, tuple)):
            return self._prop_value(ordered[0], None, ())
        return super().__getitem__(index)


# Utilities ##################################################################

def _restrict_property(prop, mask, count, /):
    '''Returns a property restricted to a mask of its spatial positions.'''
    sel = (Ellipsis, mask)
    pmask = prop.mask
    if pmask is not None and getattr(pmask, 'ndim', 0) > 0:
        pmask = pmask[sel]
    return Property(
        prop.value[sel], (count,), backend=prop.backend,
        vartype=prop.vartype, interp=prop.interp, extrap=prop.extrap,
        dtype=prop.dtype, mask=pmask,
        null=prop.null, unit=prop.unit, detach=prop.detach)


def _is_boolean_mask(x, /):
    '''Determines whether ``x`` is an array-like boolean mask.'''
    if not hasattr(x, 'dtype'):
        return False
    return getattr(x.dtype, 'kind', None) == 'b'


def is_geometry(x, /):
    '''Determines whether ``x`` is a ``Geometry``.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is a ``Geometry``.
    '''
    return isinstance(x, Geometry)


def is_simplex_geometry(x, /):
    '''Determines whether ``x`` is a ``SimplexGeometry``.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is a ``SimplexGeometry``.
    '''
    return isinstance(x, SimplexGeometry)

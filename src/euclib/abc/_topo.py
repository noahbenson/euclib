# -*- coding: utf-8 -*-
###############################################################################
# euclib/abc/_topo.py
'''The ``Topology`` types and the machinery for local coordinates.

A *topology* describes the connectivity of a geometric object independently of
where that object's coordinates lie. The same topology can be realized by many
different sets of coordinates: a triangle mesh's connectivity is fixed while its
vertex positions warp, and a prism mesh is two triangle meshes --- two sets of
coordinates --- sharing one topology.

Because a topology is purely structural, it is the natural home for the object's
*local* coordinate system. Local coordinates name a position inside the object
in the object's own terms, and there is deliberately no single representation of
them (see the README): a triangle mesh's local coordinate is a triangle index
plus barycentric weights, while a grid's is a scaled index per axis. Each
topology therefore defines its own ``Loc`` type, built with ``make_loc`` from a
list of field names. A ``Loc`` can be constructed from a matching sequence or
from a mapping whose keys are its field names, and ``is_loc`` identifies one.

Index matrices (``indices``, ``simplices``) are always NumPy integer arrays.
Coordinate and property data, by contrast, follow the object's ``backend``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from collections import namedtuple
from itertools import combinations

from numpy import (
    arange, asarray, concatenate, full, integer, unique, zeros)
from pcollections import llist, pdict

from ..utils import unique_columns
from ._core import (
    MetaObject, calc, normalize_backend, plantypeABC, planobject,
    planobject_eq, planobject_hash)


# Local Coordinates ##########################################################

class LocMixin:
    '''Mixin that gives a local-coordinate namedtuple a uniform constructor.

    ``Loc`` types are namedtuples whose fields name the components of a local
    coordinate --- for example ``index`` and ``weight`` for a simplex topology,
    or ``sx``, ``sy``, and ``sz`` for a grid. This mixin adds the ``from_value``
    constructor, which accepts another ``Loc``, a mapping keyed by the field
    names, or a sequence of the field values.
    '''
    __slots__ = ()

    @classmethod
    def from_value(cls, value, /):
        '''Builds a local coordinate from a ``Loc``, mapping, or sequence.

        Parameters
        ----------
        value : LocMixin, mapping, or sequence
            The value to interpret. A mapping must supply every field by name;
            a sequence must supply the field values in order.

        Returns
        -------
        LocMixin
            The local coordinate.
        '''
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        if isinstance(value, (tuple, list)):
            return cls(*value)
        return cls(value)


def make_loc(name, fields, /):
    '''Creates a local-coordinate namedtuple type with the given fields.

    Parameters
    ----------
    name : str
        The name of the resulting type.
    fields : sequence of str
        The names of the coordinate's components.

    Returns
    -------
    type
        A ``namedtuple`` subclass of ``LocMixin``.
    '''
    nt = namedtuple(name, tuple(fields))
    return type(name, (nt, LocMixin), {'__slots__': (), '__module__': __name__})


def is_loc(x, /):
    '''Determines whether ``x`` is a local coordinate.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is an instance of a ``Loc`` type.
    '''
    return isinstance(x, LocMixin)


# Metadata Normalization #####################################################

def normalize_indices(indices, /):
    '''Validates a matrix of simplex corners.

    Parameters
    ----------
    indices : array-like
        The candidate corner matrix.

    Returns
    -------
    numpy.ndarray
        The corners as a 2-dimensional integer array.

    Raises
    ------
    ValueError
        If the value is not a 2-dimensional integer matrix with at least one
        row.
    '''
    mat = asarray(indices)
    if mat.ndim != 2:
        raise ValueError(
            f"indices must be a 2-dimensional matrix; found {mat.ndim}"
            f" dimensions")
    if mat.shape[0] < 1:
        raise ValueError("indices must have at least one row")
    if not issubclass(mat.dtype.type, integer):
        raise ValueError(
            f"indices must be integers; found dtype {mat.dtype}")
    return mat


def check_simplex_loc(loc_type, local_dim, locs, /):
    '''Coerces and validates a simplex local coordinate.

    A simplex local coordinate names a simplex and gives the barycentric
    weights of a position within it: ``Loc(index, weight)``, where ``index`` is
    a length-``N`` vector of simplex indices and ``weight`` is an
    ``(order, N)`` matrix holding the first ``order`` barycentric coordinates.
    The final barycentric coordinate is implied, because the weights of a
    simplex sum to one --- which is also what keeps a simplex local coordinate
    unambiguous for a path, whose ``weight`` is ``(1, N)``.

    Concrete simplex topologies implement ``check_loc`` by delegating to this
    function, so that the rule lives in one place without making
    ``SimplexTopology`` itself instantiable.

    Parameters
    ----------
    loc_type : type
        The topology's ``Loc`` type.
    local_dim : int
        The number of barycentric weights the coordinate must supply.
    locs : LocMixin, mapping, or sequence
        The local coordinate.

    Returns
    -------
    LocMixin
        The local coordinate.

    Raises
    ------
    ValueError
        If the coordinate's index and weight arrays are not correctly shaped
        or do not describe the same number of positions.
    '''
    loc = loc_type.from_value(locs)
    index = asarray(loc.index)
    weight = asarray(loc.weight)
    if index.ndim != 1:
        raise ValueError(
            f"a local coordinate's index must be a vector; found shape"
            f" {index.shape}")
    if weight.ndim != 2 or weight.shape[0] != local_dim:
        raise ValueError(
            f"a local coordinate's weight must have shape ({local_dim}, N);"
            f" found {weight.shape}")
    if weight.shape[1] != index.shape[0]:
        raise ValueError(
            f"a local coordinate's index and weight disagree:"
            f" {index.shape[0]} indices but {weight.shape[1]} weights")
    return loc


# Topology ###################################################################

class Topology(MetaObject, metaclass=plantypeABC):
    '''The abstract connectivity of a geometric object.

    A topology records how an object's components are connected, and defines
    the object's local coordinate system. It carries no coordinates: the same
    topology can be realized by any number of coordinate matrices, which is what
    allows prisms and meshes to share connectivity.

    Subclasses must implement ``to_local`` and ``from_local``, and must provide
    the ``coord_count``, ``dim``, and ``local_dim`` values that the class
    docstring describes.

    Parameters
    ----------
    backend : str or None, optional
        The numeric backend for coordinate data, ``'numpy'``, ``'torch'``, or
        ``None`` (the default) to use whatever backend the data already uses.
    metadata : mapping or None, optional
        Arbitrary metadata to associate with the topology. The default,
        ``None``, attaches none.

    Attributes
    ----------
    backend : str or None
        The numeric backend for coordinate data.
    metadata : pcollections.ldict
        The topology's metadata, empty when none was attached.
    coord_count : int
        The number of coordinates that this topology is valid for. This may
        exceed the number of coordinates the topology actually references.
    dim : int
        The topological dimension of the object's simplices or cells.
    local_dim : int
        The number of components in a local coordinate.
    '''

    def __init__(self, backend=None, metadata=None):
        self.backend = backend
        self.metadata = metadata

    @calc('backend', lazy=False)
    def proc_backend(backend):
        '''Validates the topology's backend.

        Returns
        -------
        backend : str or None
            ``'numpy'``, ``'torch'``, or ``None``.
        '''
        return normalize_backend(backend)

    #: The type of this topology's local coordinates. Every concrete topology
    #: sets this to a type built with ``make_loc``.
    Loc = None

    @abstractmethod
    def check_loc(self, locs, /):
        '''Coerces and validates a local coordinate.

        A topology has no coordinates of its own --- the same topology can be
        realized by any number of coordinate matrices --- so the work it can do
        with local coordinates is to say what a well-formed one looks like.
        Geometries convert whole coordinate matrices to and from local
        coordinates; this method handles a single local coordinate, or a
        collection of them.

        Parameters
        ----------
        locs : LocMixin, mapping, or sequence
            A local coordinate, or a value that the topology's ``Loc`` accepts
            through its ``from_value`` constructor.

        Returns
        -------
        LocMixin
            The local coordinate, canonicalized.
        '''

    def local_shape(self, count, /):
        '''The shape of a local coordinate array for ``count`` positions.

        Parameters
        ----------
        count : int
            The number of positions.

        Returns
        -------
        tuple of int
            ``(local_dim, count)``.
        '''
        return (self.local_dim, int(count))

    #: Names of plan inputs that identify a topology but do not take part in
    #: equality. Subclasses list every name they exclude, since an assignment
    #: replaces this rather than adding to it. Metadata is excluded because it
    #: is a label rather than part of the object.
    _eq_excluded = ('metadata',)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return planobject_eq(self, other, self._eq_excluded)

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __hash__(self):
        return planobject_hash(self, self._eq_excluded)


# Simplex Topology ###########################################################

class SimplexTopology(Topology):
    '''The connectivity of a collection of simplices of one dimension.

    A simplex topology stores the corners of its primary simplices in
    ``indices``, an integer matrix with one column per simplex. The lower-order
    simplices implied by those corners --- the edges of a triangle mesh, the
    vertices of a mesh, and so on --- are derived on demand and stored in
    ``simplices``.

    Parameters
    ----------
    indices : array-like
        An integer matrix of shape ``(order+1, M)``, where ``M`` is the number
        of primary simplices. Each column lists the coordinates, by index, that
        form one simplex.
    coord_count : int or None, optional
        The number of coordinates the topology is valid for. The default,
        ``None``, uses one more than the largest index in ``indices``. When
        given, it may be larger than that, which is how a topology can be paired
        with a coordinate matrix containing coordinates it does not reference.
    backend : str or None, optional
        The numeric backend for coordinate data. The default, ``None``, uses
        whatever backend the data already uses.
    metadata : mapping or None, optional
        Arbitrary hashable metadata. The default, ``None``, attaches none.

    Attributes
    ----------
    indices : numpy.ndarray
        The ``(order+1, M)`` matrix of primary simplex corners.
    order : int
        The dimensionality of the primary simplices.
    simplices : pcollections.llist
        A lazy list with one entry per simplex order, 0 through ``order``.
        Entry *k* is the integer matrix of the unique *k*-dimensional simplices,
        with shape ``(k+1, M_k)``. Entry ``order`` is ``indices`` itself, in the
        orientation the caller supplied; lower-order entries are canonical, with
        their corners sorted and the simplices ordered lexicographically.
    simplex_count : pcollections.llist
        A lazy list of the number of simplices at each order, ``M_k``.
    vertex_count : int
        The number of coordinates actually referenced by the simplices; this may
        be smaller than ``coord_count``.
    vertex_mask : numpy.ndarray
        A boolean array of length ``coord_count``, ``True`` where a coordinate
        is referenced by the topology.
    index_of : numpy.ndarray
        An integer array of length ``coord_count`` mapping each coordinate to
        its position in ``vertex_indices``, or -1 if it is not referenced.
    '''

    #: ``coord_count`` records how large a coordinate matrix the topology is
    #: valid for, which is a statement about the data the topology is paired
    #: with rather than about the connectivity itself. Two topologies that
    #: connect the same simplices are the same topology, whether or not they
    #: were declared for coordinate matrices of different sizes.
    _eq_excluded = ('coord_count', 'metadata')

    def __init__(self, indices, coord_count=None, backend=None, metadata=None):
        self.indices = indices
        self.coord_count = coord_count
        self.backend = backend
        self.metadata = metadata

    @calc('indices', lazy=False)
    def proc_indices(indices):
        '''Validates the matrix of simplex corners.

        Returns
        -------
        indices : numpy.ndarray
            The corners as a 2-dimensional integer array.
        '''
        return normalize_indices(indices)

    @calc('coord_count', lazy=False)
    def proc_coord_count(coord_count, indices):
        '''Determines the number of coordinates the topology is valid for.

        Returns
        -------
        coord_count : int
            The coordinate count, defaulting to one more than the largest index.
        '''
        # An index matrix with no columns describes no simplices at
        # all, so there is no largest index to take.
        needed = int(indices.max()) + 1 if indices.shape[1] else 0
        if coord_count is None:
            return needed
        count = int(coord_count)
        if count < needed:
            raise ValueError(
                f"coord_count {count} is smaller than the largest index in"
                f" indices, which requires {needed}")
        return count

    @calc('order', 'dim', 'local_dim', lazy=False)
    def proc_order(indices):
        '''The primary simplex order and the topology's dimensions.

        Returns
        -------
        order : int
            The number of corners per simplex, minus one.
        dim : int
            The topological dimension, equal to the order.
        local_dim : int
            The number of components in a local coordinate, equal to the order.
        '''
        order = indices.shape[0] - 1
        return (order, order, order)

    @calc('simplices')
    def proc_simplices(indices, order):
        '''Derives the unique simplices of every order.

        Entry *k* holds the unique *k*-dimensional simplices. Entry ``order`` is
        ``indices`` itself, so that the caller's orientation is preserved;
        lower orders are canonicalized, because a simplex expressed with its
        corners in a different order is the same simplex.

        Returns
        -------
        simplices : pcollections.llist
            One integer matrix per simplex order, 0 through ``order``.
        '''
        res = []
        rows = tuple(range(order + 1))
        for k in range(order + 1):
            if k == order:
                # The primary simplices, with the orientation as given.
                res.append(indices)
            else:
                blocks = [indices[list(combo)]
                          for combo in combinations(rows, k + 1)]
                res.append(unique_columns(concatenate(blocks, axis=1)))
        return llist(res)

    @calc('simplex_count')
    def proc_simplex_count(simplices):
        '''The number of simplices at each order.

        Returns
        -------
        simplex_count : pcollections.llist
            The simplex counts, indexed by order.
        '''
        return llist(s.shape[1] for s in simplices)

    @calc('vertex_count')
    def proc_vertex_count(simplex_count):
        '''The number of coordinates actually referenced by the simplices.

        Returns
        -------
        vertex_count : int
            The number of referenced coordinates.
        '''
        return simplex_count[0]

    @calc('vertex_indices')
    def proc_vertex_indices(simplices):
        '''The indices of the coordinates referenced by the simplices.

        Returns
        -------
        vertex_indices : numpy.ndarray
            A 1-dimensional array of coordinate indices, in sorted order.
        '''
        return simplices[0][0]

    @calc('vertex_mask')
    def proc_vertex_mask(indices, coord_count):
        '''Marks the coordinates that the topology references.

        Returns
        -------
        vertex_mask : numpy.ndarray
            A boolean array of length ``coord_count``.
        '''
        mask = zeros(int(coord_count), dtype=bool)
        mask[unique(indices)] = True
        return mask

    @calc('index_of')
    def proc_index_of(indices, coord_count):
        '''Maps each coordinate to its position among the referenced vertices.

        Returns
        -------
        index_of : numpy.ndarray
            An integer array of length ``coord_count``, holding -1 for
            coordinates the topology does not reference.
        '''
        verts = unique(indices)
        res = full(int(coord_count), -1, dtype=int)
        res[verts] = arange(verts.shape[0])
        return res


# Utilities ##################################################################

def is_topology(x, /):
    '''Determines whether ``x`` is a ``Topology``.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is a ``Topology``.
    '''
    return isinstance(x, Topology)


def is_simplex_topology(x, /):
    '''Determines whether ``x`` is a ``SimplexTopology``.

    Parameters
    ----------
    x : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``x`` is a ``SimplexTopology``.
    '''
    return isinstance(x, SimplexTopology)

# -*- coding: utf-8 -*-
###############################################################################
# euclib/__init__.py
'''``euclib``: a library for tracking data associated with 2D/3D Euclidean space.

``euclib`` represents spatial data such as point clouds, paths, triangle and
tetrahedral meshes, grid images, and prisms; attaches user-defined properties
to those objects; and provides operations for transforming, interpolating,
and measuring them. Its data structures are immutable and lazily computed,
built on the ``immlib`` and ``pcollections`` libraries, and its numeric
operations work with both NumPy arrays and PyTorch tensors.

The primary entry points are the abstract types in ``euclib.abc`` and the
concrete geometric types in ``euclib.types``, both of which are also imported
here: ``euclib.Grid`` and ``euclib.grid`` work as well as
``euclib.types.Grid`` does. The constructors --- ``euclib.points``,
``euclib.segpath``, ``euclib.trimesh``, ``euclib.tetmesh``,
``euclib.prismmesh``, and ``euclib.grid`` --- build an object's topology for it,
so that a mesh needs only coordinates and corners. The operations are here too:
``euclib.distance``, ``euclib.geodesic``, ``euclib.contains``, and the
intersections, with the tools they are built from in ``euclib.ops``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from importlib import reload
from sys import modules

from ._version import version as __version__


# Submodules #################################################################

from . import _init  # noqa: F401
from . import _version  # noqa: F401
from . import utils
from . import abc
from . import types
from . import ops

#: The ``euclib`` submodules, in the order in which they are loaded. This is
#: the order in which they must be reloaded.
#:
#: The kernel modules are named individually rather than left to be reached
#: through ``euclib.utils``, because reloading a package does not reload what it
#: imports. ``euclib.utils._core`` is the module that reads
#: ``EUCLIB_NO_C_EXTENSIONS`` and ``EUCLIB_REQUIRE_C``, so naming it is what
#: lets a change to either take effect without restarting the interpreter.
submodules = (
    'euclib._init', 'euclib._version',
    'euclib.utils._pycore', 'euclib.utils._dispatch', 'euclib.utils._core',
    'euclib.utils', 'euclib.abc', 'euclib.types', 'euclib.ops')


# Types and constructors #####################################################

# The library's own types and the functions that build them are imported here
# as well as into their own subpackages, so that the library can be used
# without remembering which subpackage each name lives in.
from .abc import (
    Geometry, SimplexGeometry, Topology, SimplexTopology, Property,
    is_geometry, is_simplex_geometry, is_topology, is_simplex_topology,
    is_loc, is_property)
from .types import (
    VertexSet, SegPath, TriMesh, TetMesh, PrismMesh, Grid,
    VertexTopology, SegTopology, TriTopology, TetTopology, PrismTopology,
    GridTopology,
    VertexLoc, SegLoc, TriLoc, TetLoc, PrismLoc, GridLoc1, GridLoc2, GridLoc3,
    points, segpath, trimesh, tetmesh, prismmesh, grid)
# The operations answer the questions a caller has about geometries --- how far
# apart they are, where they meet --- and are imported here for the same reason.
# The tools those operations are built from (``positions_of``,
# ``tolerance_of``) and the ones whose names would not stand on their own
# (``sample``, ``transfer``) stay in ``euclib.ops``.
from .ops import (
    contains, distance, geodesic, mesh_intersections, nearest,
    path_crossings, path_intersections, separation, voxel_intersections)

#: The public modules, shallowest first. A name that appears in more than one
#: of them belongs to the shallowest.
_PUBLIC_MODULES = (
    'euclib', 'euclib.abc', 'euclib.types', 'euclib.utils', 'euclib.ops')


def _resolve_public_modules():
    '''Names the shallowest public module as the source of each public object.

    A type or function defined in a private module --- ``euclib.types._geom``,
    say --- is reachable as ``euclib.types.TriMesh``, and it should say so:
    ``__module__`` is what a representation, a piece of documentation, and a
    newly written pickle all quote. Each public object is therefore given the
    name of the shallowest public module that holds it, so that
    ``euclib.TriMesh`` wins over ``euclib.types.TriMesh``.

    An object that this library merely re-exports --- the ``planobject`` it is
    built on, say --- keeps its own module, since ``euclib`` did not define it.
    '''
    from importlib import import_module
    from types import ModuleType
    for module_name in _PUBLIC_MODULES:
        module = import_module(module_name)
        for exported in getattr(module, '__all__', ()):
            obj = getattr(module, exported, None)
            if obj is None or isinstance(obj, ModuleType):
                continue
            source = getattr(obj, '__module__', None)
            # Only what this library defines in a private module is renamed.
            if (isinstance(source, str) and source.startswith('euclib.')
                    and source.rsplit('.', 1)[-1].startswith('_')):
                obj.__module__ = module_name


# Backend introspection ######################################################

#: ``True`` if the optional ``euclib._c`` extension is in use.
using_c_extension = utils.using_c_extension

#: The exception raised while loading ``euclib._c``, or ``None`` if it loaded
#: (or was never attempted). This is useful for diagnosing why the pure-Python
#: backend is in use.
backend_error = utils.backend_error


# Utilities ##################################################################

def reload_euclib():
    '''Reloads every ``euclib`` module and returns the ``euclib`` module.

    This is intended for interactive development, where a change to a module's
    source should take effect without restarting the interpreter. Modules are
    reloaded in dependency order (see ``submodules``).

    Returns
    -------
    module
        The reloaded ``euclib`` module.
    '''
    # Imported here rather than at the top of the module so that they need not
    # remain in euclib's public namespace once it has finished loading.
    from importlib import reload
    from sys import modules
    for name in submodules:
        if name in modules:
            reload(modules[name])
    return reload(modules[__name__])


# Exports ####################################################################

__all__ = (
    '__version__',
    'utils', 'abc', 'types', 'ops',
    'submodules', 'reload_euclib',
    'using_c_extension', 'backend_error',
    # the abstract types
    'Geometry', 'SimplexGeometry', 'Topology', 'SimplexTopology', 'Property',
    # the concrete types and their topologies
    'VertexSet', 'SegPath', 'TriMesh', 'TetMesh', 'PrismMesh', 'Grid',
    'VertexTopology', 'SegTopology', 'TriTopology', 'TetTopology',
    'PrismTopology', 'GridTopology',
    # their local coordinate types
    'VertexLoc', 'SegLoc', 'TriLoc', 'TetLoc', 'PrismLoc',
    'GridLoc1', 'GridLoc2', 'GridLoc3',
    # the constructors that build an object's topology for it
    'points', 'segpath', 'trimesh', 'tetmesh', 'prismmesh', 'grid',
    # the predicates that identify them
    'is_geometry', 'is_simplex_geometry', 'is_topology', 'is_simplex_topology',
    'is_loc', 'is_property',
    # the operations, which remain in euclib.ops as well
    'contains', 'distance', 'geodesic', 'mesh_intersections', 'nearest',
    'path_crossings', 'path_intersections', 'separation',
    'voxel_intersections')


# The resolver runs here rather than beside its definition, because it reads
# each public module's exports and euclib's own are listed just above.
_resolve_public_modules()


# A public namespace should hold the library and nothing else: the names that
# importing and initializing this module happened to bind --- including the
# future statement's own name --- are removed now that they are no longer
# needed.
del annotations, reload, modules, _PUBLIC_MODULES, _resolve_public_modules

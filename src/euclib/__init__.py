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
concrete geometric types in ``euclib.types``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from importlib import reload
from sys import modules

from ._version import version as __version__


# Submodules #################################################################

from . import _init  # noqa: F401
from . import utils
from . import abc

#: The ``euclib`` submodules, in the order in which they are loaded. This is
#: the order in which they must be reloaded.
submodules = ('euclib._init', 'euclib.utils', 'euclib.abc')


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
    for name in submodules:
        if name in modules:
            reload(modules[name])
    return reload(modules[__name__])


# Exports ####################################################################

__all__ = (
    '__version__',
    'utils', 'abc',
    'submodules', 'reload_euclib',
    'using_c_extension', 'backend_error')

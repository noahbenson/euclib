# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_core.py
'''Selects the implementation backend for ``euclib``'s kernels.

``euclib`` ships an optional C extension, ``euclib._c``, that accelerates a
small number of kernels. The library is nonetheless required to work without
it: on platforms where the extension cannot be built, the pure-Python
implementations in ``euclib.utils._pycore`` are used instead.

This module mirrors the backend-selection pattern used by the ``pcollections``
library. On import it attempts to load the C extension and records the outcome
in ``using_c_extension`` and ``backend_error``. Two environment variables
adjust the behavior:

``EUCLIB_NO_C_EXTENSIONS``
    When set to a true value, the C extension is not loaded even if it is
    available.
``EUCLIB_REQUIRE_C``
    When set to a true value, a failure to load the C extension is raised
    rather than tolerated.
'''

# Dependencies ###############################################################

from __future__ import annotations

from os import environ


# Backend selection ##########################################################

def _is_true(s, /):
    '''Returns ``True`` if an environment variable's value means "yes".'''
    return s is not None and s.strip().lower() in ('1', 'true', 'yes', 'on')


#: ``True`` if the ``euclib._c`` extension was loaded, ``False`` otherwise.
using_c_extension = False

#: The exception raised while trying to load ``euclib._c``, or ``None``.
backend_error = None

if _is_true(environ.get('EUCLIB_NO_C_EXTENSIONS')):
    backend_error = ImportError(
        "the euclib C extensions were disabled by EUCLIB_NO_C_EXTENSIONS")
else:
    try:
        from .._c import _core as _c  # noqa: F401
        using_c_extension = True
    except Exception as exc:  # pragma: no cover - depends on the build
        backend_error = exc

if _is_true(environ.get('EUCLIB_REQUIRE_C')) and not using_c_extension:
    raise ImportError(
        "EUCLIB_REQUIRE_C is set but the euclib C extension could not be"
        " loaded") from backend_error


# Kernels ####################################################################

# The pure-Python implementations are always imported. Where the C extension
# provides a faster version of a kernel, it is substituted here, so that the
# rest of the library imports its kernels from one place and never has to know
# which backend is active.
from ._pycore import (  # noqa: E402
    is_pointdata, unique_columns, unique_coords, simplex_measures,
    nearest_vertices, project_onto_face, closest_simplex,
    closest_prism, barycentric_coords)

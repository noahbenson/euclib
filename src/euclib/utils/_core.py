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
        from .._c import _core as _c
        from .._c import _spatial as _c_spatial
        using_c_extension = True
    except Exception as exc:  # pragma: no cover - depends on the build
        backend_error = exc

if _is_true(environ.get('EUCLIB_REQUIRE_C')) and not using_c_extension:
    raise ImportError(
        "EUCLIB_REQUIRE_C is set but the euclib C extension could not be"
        " loaded") from backend_error


#: The compiled spatial-index queries, or ``None`` when the extension is not
#: loaded. Unlike the kernels below, these are not registered against a
#: Pure-Python counterpart in ``utils._dispatch``: the tree is a Python class
#: whose own methods are the definition, and euclib.utils._spatial calls these
#: from inside them when they are there.
c_spatial = _c_spatial if using_c_extension else None


# Kernels ####################################################################

# Every kernel is imported from one of two places: the pure-Python module, which
# is the definition of correct behavior, or the dispatching registry, which
# chooses between that and the C extension per call. The rest of the library
# imports its kernels from here and never has to know which backend is active.
from ._pycore import (  # noqa: E402
    is_pointdata, unique_columns, unique_coords, simplex_measures,
    bounds_of, simplex_boxes, octree_split, quadtree_split,
    nearest_vertices, project_onto_face, closest_simplex,
    closest_prism, barycentric_coords, cross3, closest_segment_params,
    segments_intersect, barycentric_in_triangle,
    segments_triangles_intersect, triangles_segments_intersect,
    tetrahedron_box_intersection)

# The kernels the C extension provides. Each is registered against its
# pure-Python counterpart, which is what any call that the C version cannot take
# runs on.
from ._dispatch import kernel as _kernel  # noqa: E402
from . import _pycore as _python  # noqa: E402

if using_c_extension:
    split_cells = _kernel(_python.split_cells, _c.split_cells)
    tetrahedron_box_vertices = _kernel(
        _python.tetrahedron_box_vertices, _c.tetrahedron_box_vertices)
    tetrahedron_box_intersection = _kernel(
        _python.tetrahedron_box_intersection, _c.tetrahedron_box_region)
else:
    split_cells = _kernel(_python.split_cells)
    tetrahedron_box_vertices = _kernel(_python.tetrahedron_box_vertices)
    tetrahedron_box_intersection = _kernel(
        _python.tetrahedron_box_intersection)

# ``tetrahedron_box_intersection`` builds its region from a corner search, and
# takes that search from a name in its own module rather than calling it
# directly. Pointing that name at the dispatching kernel is what lets an
# intersection --- the form of the operation that the library's callers
# actually use --- take the C path without the pure-Python module having to know
# that a C path exists.
_python._vertices_kernel = tetrahedron_box_vertices

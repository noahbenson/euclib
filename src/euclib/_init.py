# -*- coding: utf-8 -*-
###############################################################################
# euclib/_init.py
'''Global constants and lazy imports for ``euclib``.

This module is imported before any other ``euclib`` module. It establishes the
small amount of global state that the rest of the library assumes: the names of
the supported numeric backends, a lazy handle on the optional ``torch``
dependency, and the environment-variable switches that control the optional
``euclib._c`` extension.

Nothing in this module imports another ``euclib`` module.
'''

# Dependencies ###############################################################

from __future__ import annotations

from os import environ


# Backends ###################################################################

#: The concrete numeric backends that ``euclib`` supports. A ``backend`` of
#: ``None`` means "use whatever backend the data already uses"; an explicit
#: backend means "convert the data to that backend".
backend_names = ('numpy', 'torch')

#: The default ``backend`` used when none is requested. ``None`` means that the
#: backend of each value is taken from the value itself.
default_backend = environ.get('EUCLIB_DEFAULT_BACKEND', None)
if default_backend is not None and default_backend not in backend_names:
    raise ValueError(
        f"invalid EUCLIB_DEFAULT_BACKEND: {default_backend!r}; expected one of"
        f" {backend_names} or None")

#: The interpolation that a quantitative property uses when its metadata leaves
#: it unspecified. The design calls for ``('bezier', 2)``, a quadratic scheme
#: that suits smooth real-valued data on a simplex; ``('polynomial', 1)`` is
#: what is implemented, and this returns to the design's default once the
#: higher orders are built (see ``euclib.types._interp``).
default_quantitative_interp = ('polynomial', 1)

#: The number of simplices a geometry must have before it builds a spatial
#: index for its point-location searches. Below this, examining every simplex
#: --- one vectorized call over the whole mesh --- is faster than the
#: bookkeeping a subdivided search needs, which is paid per query.
#:
#: Measured on a triangulated grid mesh with 200 queries: the two are roughly
#: level between 1,500 and 2,500 triangles, and the index's advantage grows
#: from there --- 3.6 times at 7,000 triangles, 7.4 at 12,500, and 10.5 at
#: 19,600. Below the crossover it loses by up to a factor of two. The number is
#: a rough knob rather than a constant: it depends on how many positions are
#: asked about and on the shape of the elements, so it is worth revisiting
#: alongside the rest of the performance work before the 1.0 release.
spatial_index_min_items = 2048

#: The interpolation that a qualitative property uses, and the only one it may
#: use: a category has no meaning between the values it takes.
default_qualitative_interp = ('nearest', 0)


# Optional torch #############################################################

_torch = None
_torch_tested = False

def checktorch():
    '''Returns the ``torch`` module if it can be imported and ``None`` otherwise.

    The result is cached, so the import is attempted at most once per process.
    Importing ``euclib`` never imports ``torch`` on its own; this function is
    how the rest of the library obtains it on demand.

    Returns
    -------
    module or None
        The ``torch`` module, or ``None`` if it is not installed.
    '''
    global _torch, _torch_tested
    if not _torch_tested:
        _torch_tested = True
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
    return _torch

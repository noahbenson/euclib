# -*- coding: utf-8 -*-
###############################################################################
# setup.py
'''The build configuration for the optional ``euclib._c`` extension.

The extension is declared ``optional``: a machine without a compiler gets an
installation that warns rather than one that fails, and the library then runs on
the pure-Python kernels in ``euclib.utils._pycore``, which are the definition of
correct behavior. Setting ``EUCLIB_REQUIRE_C`` makes a failed build an error
instead, for a packager who needs to know that the extension is there.

Everything else about the package --- its metadata, dependencies, and package
layout --- is declared in ``pyproject.toml``; this file exists only because an
extension's sources have to be named somewhere Python can compute them.
'''

# Dependencies ###############################################################

from os import environ

from setuptools import Extension, setup

# The extension needs NumPy's headers. A build environment that has setuptools
# but not NumPy is not a reason to fail the installation: the kernels have
# pure-Python counterparts, and a machine that cannot build the extension runs
# on those. Leaving numpy out of the include path turns a missing header into a
# compile error, which `optional` then tolerates.
try:
    import numpy
    INCLUDE_DIRS = [numpy.get_include()]
except ImportError:                                  # pragma: no cover
    INCLUDE_DIRS = []


# The extension ##############################################################

# A failed build is a warning unless the packager asks otherwise.
OPTIONAL = not environ.get('EUCLIB_REQUIRE_C')

CORE = Extension(
    'euclib._c._core',
    sources=['src/euclib/_c/_core.c'],
    include_dirs=INCLUDE_DIRS,
    optional=OPTIONAL)

# The spatial index's queries, which answer a whole array of positions in one
# call rather than one position at a time.
SPATIAL = Extension(
    'euclib._c._spatial',
    sources=['src/euclib/_c/_spatial.c'],
    include_dirs=INCLUDE_DIRS,
    optional=OPTIONAL)

setup(ext_modules=[CORE, SPATIAL])

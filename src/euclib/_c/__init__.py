# -*- coding: utf-8 -*-
###############################################################################
# euclib/_c/__init__.py
'''The optional C kernels of the ``euclib`` library.

This package exists only to hold the compiled extension ``euclib._c._core``.
The extension is optional: ``euclib`` must import and work without it, and does,
because every kernel it provides has a pure-Python counterpart in
``euclib.utils._pycore`` that is the definition of correct behavior. The C
version is used where it is available and eligible, which
``euclib.utils._dispatch`` decides, and ``euclib.using_c_extension`` reports.

A Python file is present so that the directory is an ordinary package whether or
not the extension was built: an empty directory would be a namespace package,
which ``find_packages`` does not collect, and the compiled module would then not
be found on installation.
'''

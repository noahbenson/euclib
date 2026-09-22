# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_pycore.py
'''Pure-Python implementations of ``euclib``'s performance-critical kernels.

Every function in this module has a counterpart in the optional ``euclib._c``
extension. The versions here are the definition of correct behavior: the C
implementations are tested for parity against them. When the C extension cannot
be built or loaded, these are what run.

Functions here dispatch between NumPy and PyTorch themselves; they do not rely
on ``immlib``'s ``numapi`` machinery, so that they remain usable as a fallback
in any environment.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import asarray, sort, unique

from immlib import is_numeric

from .._init import checktorch


# Predicates #################################################################

def is_pointdata(x, /, dims=(1, 2, 3), *, shape=None, ndim=(1, 2), dtype=None):
    '''Determines whether ``x`` is a valid array of point or coordinate data.

    A valid coordinate payload is a numeric array-like --- a NumPy array, a
    PyTorch tensor, a SciPy sparse array, or a ``pint`` quantity with one of
    those as its magnitude --- that has a dimensionality in ``ndim`` and whose
    leading ("channel") dimension is one of ``dims``. ``immlib.is_numeric``
    performs the type, dtype, and dimensionality checks.

    Parameters
    ----------
    x : object
        The object to test.
    dims : sequence of int, optional
        The permitted sizes of the leading channel dimension. The default is
        ``(1, 2, 3)``.
    shape : tuple or None, optional
        When provided, the trailing spatial dimensions of ``x`` must match this
        shape exactly. The default, ``None``, permits any trailing shape.
    ndim : int or sequence of int, optional
        The permitted total numbers of dimensions. The default is ``(1, 2)``.
    dtype : dtype-like or None, optional
        When provided, ``x``'s dtype must match this value (or one of its
        values, if a tuple is given). The default, ``None``, permits any
        numeric dtype.

    Returns
    -------
    bool
        ``True`` if ``x`` is valid coordinate data and ``False`` otherwise.
    '''
    if not is_numeric(x, dtype=dtype, ndim=ndim, sparse=False):
        return False
    xshape = tuple(x.shape)
    if len(xshape) < 1 or xshape[0] not in dims:
        return False
    if shape is not None and xshape[1:] != tuple(shape):
        return False
    return True


# Simplices ##################################################################

def unique_columns(mat):
    '''Returns the unique columns of an integer matrix, in sorted order.

    This is used to derive the unique *k*-simplices of a mesh from its
    higher-order simplices. The result is canonical: the entries within each
    column are sorted, and the columns are sorted lexicographically by their
    contents. Two matrices that contain the same columns in different vertex
    orders therefore produce identical results.

    Parameters
    ----------
    mat : array-like
        A 2-dimensional integer array whose columns are the items to
        deduplicate.

    Returns
    -------
    numpy.ndarray
        An array with the same number of rows as ``mat`` whose columns are the
        unique, row-sorted columns of ``mat`` in lexicographic order.
    '''
    mat = asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"expected a 2-dimensional matrix, found {mat.ndim}")
    if mat.shape[0] < 1:
        raise ValueError("expected at least one row")
    # Sort the entries within each column, then deduplicate the rows of the
    # transpose, which sorts them lexicographically.
    return unique(sort(mat, axis=0).T, axis=0).T


def unique_coords(coords, return_index=False, return_inverse=False):
    '''Returns the unique columns of a coordinate matrix.

    Parameters
    ----------
    coords : numpy.ndarray or torch.Tensor
        A coordinate matrix of shape ``(D, N)``.
    return_index : bool, optional
        Whether to also return the indices of ``coords`` that yield the unique
        columns. The default is ``False``.
    return_inverse : bool, optional
        Whether to also return the indices that reconstruct ``coords`` from the
        unique columns. The default is ``False``.

    Returns
    -------
    unique : numpy.ndarray or torch.Tensor
        The unique columns of ``coords``.
    index : numpy.ndarray or torch.Tensor, optional
        The indices of the first occurrences of the unique columns; returned
        only when ``return_index`` is ``True``.
    inverse : numpy.ndarray or torch.Tensor, optional
        The indices reconstructing ``coords``; returned only when
        ``return_inverse`` is ``True``.
    '''
    t = checktorch()
    is_tensor = t is not None and isinstance(coords, t.Tensor)
    if is_tensor:
        dev = coords.device
        x = asarray(coords.detach().cpu())
    else:
        x = asarray(coords)
    if x.ndim != 2:
        raise ValueError(
            f"expected a 2-dimensional coordinate matrix, found {x.ndim}D")
    res = unique(x, axis=1,
                 return_index=return_index, return_inverse=return_inverse)
    if not isinstance(res, tuple):
        res = (res,)
    if is_tensor:
        res = tuple(t.from_numpy(r).to(dev) for r in res)
    return res[0] if len(res) == 1 else res

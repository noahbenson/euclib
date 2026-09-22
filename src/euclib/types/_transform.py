# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_transform.py
'''Transformations of space.

A transform maps positions in one Euclidean space to positions in another. It is
represented by a square matrix acting on *homogeneous* coordinates, which is
what lets a single matrix carry both a linear part and a translation: a
``D``-dimensional position ``x`` is written ``(x, 1)`` and multiplied by a
``(D+1, D+1)`` matrix.

Only affine transforms are implemented so far. An affine matrix has a final row
of ``[0, ..., 0, 1]``, so the last homogeneous coordinate of every transformed
position stays 1 --- which is what makes the transform affine rather than merely
linear, and what makes it map straight lines to straight lines.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import allclose, asarray, eye, linalg, zeros
from immlib import to_array, to_tensor

from ..abc import abstractmethod, calc, plantypeABC, planobject
from ..abc._core import normalize_backend


# Helpers ####################################################################

def to_numpy(x, /):
    '''Returns a NumPy view of an array or tensor, detaching gradients.

    Transform matrices are small and are validated rather than optimized
    through, so the linear algebra is done with NumPy throughout.
    '''
    if type(x).__module__.split('.')[0] == 'torch':
        return asarray(x.detach().cpu())
    return asarray(x)


def check_transform_matrix(matrix, backend, /):
    '''Validates a transform matrix and converts it to the requested backend.

    Parameters
    ----------
    matrix : array-like
        The candidate matrix.
    backend : str or None
        ``'numpy'``, ``'torch'``, or ``None`` to leave the matrix as given.

    Returns
    -------
    array-like
        A square matrix of size at least 2.

    Raises
    ------
    ValueError
        If the matrix is not square, or is smaller than 2-by-2.
    '''
    if backend == 'torch':
        res = to_tensor(matrix)
    elif backend == 'numpy' or not hasattr(matrix, 'shape'):
        res = to_array(matrix)
    else:
        res = matrix
    sh = tuple(res.shape)
    if len(sh) != 2 or sh[0] != sh[1]:
        raise ValueError(f"a transform matrix must be square; found {sh}")
    if sh[0] < 2:
        raise ValueError(
            "a transform matrix must be at least 2x2, so that it acts on at"
            " least one spatial dimension")
    return res


def check_affine_matrix(matrix, backend, /):
    '''Validates an affine transform matrix.

    Parameters
    ----------
    matrix : array-like
        The candidate matrix.
    backend : str or None
        ``'numpy'``, ``'torch'``, or ``None``.

    Returns
    -------
    array-like
        A square matrix whose final row is ``[0, ..., 0, 1]``.

    Raises
    ------
    ValueError
        If the matrix is not square, too small, or not affine.
    '''
    res = check_transform_matrix(matrix, backend)
    d = int(res.shape[0]) - 1
    last = to_numpy(res[d])
    expected = zeros(d + 1)
    expected[d] = 1.0
    if not allclose(last, expected):
        raise ValueError(
            f"an affine matrix's final row must be [0, ..., 0, 1]; found"
            f" {last.tolist()}")
    return res


def _match_backend(matrix, values, /):
    '''Converts a NumPy result to the backend of a reference matrix.'''
    if type(matrix).__module__.split('.')[0] == 'torch':
        import torch
        return torch.as_tensor(values, dtype=matrix.dtype,
                               device=matrix.device)
    return values


def _inverse_affine_matrix(matrix, /):
    '''Returns the inverse of an affine matrix, computed from its parts.

    Inverting the linear part and the translation separately, rather than
    inverting the whole matrix, guarantees that the result is exactly affine:
    its final row is bit-for-bit ``[0, ..., 0, 1]`` rather than something that
    is merely close to it.
    '''
    m = to_numpy(matrix)
    d = int(m.shape[0]) - 1
    inv_linear = linalg.inv(m[:d, :d])
    res = eye(d + 1)
    res[:d, :d] = inv_linear
    res[:d, d] = -inv_linear @ m[:d, d]
    return _match_backend(matrix, res)


# Transform ##################################################################

class Transform(planobject, metaclass=plantypeABC):
    '''The abstract base class of transformations of space.

    A transform is represented by a square matrix acting on homogeneous
    coordinates. Subclasses define how that matrix is applied to a coordinate
    matrix, which is the only part of the operation that differs between one
    kind of transform and another.

    Parameters
    ----------
    matrix : array-like
        The ``(D+1, D+1)`` transform matrix.
    backend : str or None, optional
        The numeric backend for the matrix. The default, ``None``, leaves the
        matrix in whatever backend it already uses.

    Attributes
    ----------
    matrix : array-like
        The transform matrix.
    dim : int
        The number of dimensions the transform acts on.
    inverse : Transform
        The transform that undoes this one.
    '''

    def __init__(self, matrix, backend=None):
        self.matrix = matrix
        self.backend = backend

    @calc('backend', lazy=False)
    def proc_backend(backend):
        '''Validates the transform's backend.

        Returns
        -------
        backend : str or None
            ``'numpy'``, ``'torch'``, or ``None``.
        '''
        return normalize_backend(backend)

    @calc('matrix', lazy=False)
    def proc_matrix(matrix, backend):
        '''Validates the transform matrix.

        Returns
        -------
        matrix : array-like
            A square matrix of size at least 2.
        '''
        return check_transform_matrix(matrix, backend)

    @calc('dim', lazy=False)
    def proc_dim(matrix):
        '''The number of spatial dimensions the transform acts on.

        Returns
        -------
        dim : int
            One less than the size of the matrix.
        '''
        return int(matrix.shape[0]) - 1

    @calc('inverse_matrix')
    def proc_inverse_matrix(matrix):
        '''The matrix of the inverse transform.

        A calculation is not given the object it belongs to, so it cannot
        construct a transform of the right type; it returns the matrix, and the
        ``inverse`` property wraps it.

        Returns
        -------
        inverse_matrix : array-like
            The inverse matrix.
        '''
        return _match_backend(matrix, linalg.inv(to_numpy(matrix)))

    @abstractmethod
    def apply(self, coords, /):
        '''Applies the transform to a matrix of positions.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        array-like
            The transformed positions.
        '''

    @property
    def inverse(self):
        '''The transform that undoes this one.

        Returns
        -------
        Transform
            The inverse transform, of the same type as this one.
        '''
        return type(self)(self.inverse_matrix)

    def compose(self, other, /):
        '''Returns the transform that applies ``other`` and then this one.

        Parameters
        ----------
        other : Transform
            The transform to apply first. Its dimension must match this one's.

        Returns
        -------
        Transform
            The composed transform.
        '''
        if not isinstance(other, Transform):
            raise TypeError(f"expected a Transform; found {type(other)}")
        if other.dim != self.dim:
            raise ValueError(
                f"cannot compose transforms of dimensions {self.dim} and"
                f" {other.dim}")
        return type(self)(self.matrix @ other.matrix)

    def __matmul__(self, other):
        return self.compose(other)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (self.dim == other.dim
                and allclose(to_numpy(self.matrix), to_numpy(other.matrix)))

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __hash__(self):
        return hash((type(self), self.dim,
                     tuple(to_numpy(self.matrix).ravel().tolist())))


# Affine #####################################################################

class Affine(Transform):
    '''An affine transformation of space.

    An affine transform carries a linear part --- a rotation, scale, shear, or
    any combination of them --- and a translation. It maps straight lines to
    straight lines and preserves ratios of distances along them.

    Parameters
    ----------
    matrix : array-like
        A ``(D+1, D+1)`` matrix whose final row is ``[0, ..., 0, 1]``.
    backend : str or None, optional
        The numeric backend for the matrix.

    Attributes
    ----------
    linear : array-like
        The ``(D, D)`` linear part of the matrix.
    translation : array-like
        The length-``D`` translation.
    '''

    @calc('matrix', lazy=False)
    def proc_matrix(matrix, backend):
        '''Validates the affine matrix.

        Returns
        -------
        matrix : array-like
            A square matrix whose final row is ``[0, ..., 0, 1]``.
        '''
        return check_affine_matrix(matrix, backend)

    @calc('linear')
    def proc_linear(matrix):
        '''The linear part of the transform.

        Returns
        -------
        linear : array-like
            The ``(D, D)`` upper-left block of the matrix.
        '''
        return matrix[:-1, :-1]

    @calc('translation')
    def proc_translation(matrix):
        '''The translation part of the transform.

        Returns
        -------
        translation : array-like
            The length-``D`` final column of the matrix, without its last
            entry.
        '''
        return matrix[:-1, -1]

    def apply(self, coords, /):
        '''Applies the affine transform to a matrix of positions.

        Parameters
        ----------
        coords : array-like
            A ``(D, Q)`` matrix of positions.

        Returns
        -------
        array-like
            The transformed positions.
        '''
        coords = coords if hasattr(coords, 'shape') else asarray(coords)
        if len(coords.shape) == 1:
            # A bare vector is one position, not a matrix of them; without this
            # the translation would broadcast it into a (D, D) matrix.
            coords = coords.reshape(-1, 1)
        if int(coords.shape[0]) != self.dim:
            raise ValueError(
                f"this transform acts on {self.dim}-dimensional positions, but"
                f" was given {coords.shape[0]}-dimensional ones")
        return self.linear @ coords + self.translation[:, None]

    @calc('inverse_matrix')
    def proc_inverse_matrix(matrix):
        '''The matrix of the inverse affine transform.

        Returns
        -------
        inverse_matrix : array-like
            The inverse matrix, whose final row is exactly ``[0, ..., 0, 1]``.
        '''
        return _inverse_affine_matrix(matrix)


# Constructors ###############################################################

def affine_identity(dim, /):
    '''Returns the identity affine transform of a ``dim``-dimensional space.

    Parameters
    ----------
    dim : int
        The number of spatial dimensions.

    Returns
    -------
    Affine
        The transform that leaves every position unchanged.
    '''
    return Affine(eye(int(dim) + 1))


def affine_translation(offset, /):
    '''Returns the affine transform that translates by ``offset``.

    Parameters
    ----------
    offset : array-like
        A length-``D`` vector.

    Returns
    -------
    Affine
        The translation.
    '''
    offset = asarray(offset)
    if offset.ndim != 1:
        raise ValueError(
            f"offset must be a vector; found shape {offset.shape}")
    d = offset.shape[0]
    res = eye(d + 1)
    res[:d, d] = offset
    return Affine(res)


def affine_scaling(scale, /):
    '''Returns the affine transform that scales each axis.

    Parameters
    ----------
    scale : array-like
        A length-``D`` vector of per-axis scale factors.

    Returns
    -------
    Affine
        The scaling.
    '''
    scale = asarray(scale)
    if scale.ndim != 1:
        raise ValueError(
            f"scale must be a vector; found shape {scale.shape}")
    res = eye(scale.shape[0] + 1)
    for (i, s) in enumerate(scale):
        res[i, i] = s
    return Affine(res)


# Exports ####################################################################

__all__ = (
    'Transform', 'Affine',
    'affine_identity', 'affine_translation', 'affine_scaling')

# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_hash.py
'''Canonical content hashing for the array-like payloads that ``euclib`` stores.

``euclib`` types overload ``__eq__`` to compare their *content* rather than
their identity (see ``euclib.abc.Geometry``). Python then requires that they
define ``__hash__`` consistent with that equality, which means hashing arrays,
tensors, and ``pint`` quantities. The builtin ``hash`` cannot do this: NumPy
arrays are unhashable, and no array-like type's ``__eq__`` returns a plain
``bool``.

``content_hash`` fills that gap. Given two objects ``a`` and ``b`` that
``euclib`` considers equal, ``content_hash(a) == content_hash(b)``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from collections.abc import Mapping
from numpy import array_equal, asarray, generic, ndarray


# Content Hashing ############################################################

def content_hash(x, /):
    '''Returns a hash of the content of ``x``.

    The hash is computed recursively: mappings are hashed by their sorted
    key/value pairs, sequences by their elements in order, and array-likes by
    their shape, dtype, and bytes. Objects of any other type fall back to the
    builtin ``hash``.

    Parameters
    ----------
    x : object
        The object whose content is to be hashed.

    Returns
    -------
    int
        A hash consistent with equality as ``euclib`` defines it; that is, two
        objects that compare equal under ``euclib``'s semantics return the same
        value.
    '''
    # Mappings first: check before sequences, since a mapping may also be
    # iterable.
    if isinstance(x, Mapping):
        try:
            items = frozenset((content_hash(k), content_hash(v))
                              for (k, v) in x.items())
        except TypeError:
            items = tuple(sorted((repr(k), content_hash(v))
                                 for (k, v) in x.items()))
        return hash((type(x).__name__, 'mapping', items))
    # Array-likes. torch tensors are detected without importing torch.
    elif type(x).__module__.split('.')[0] == 'torch':
        return hash(('torch', content_hash(_tensor_to_numpy(x))))
    elif isinstance(x, ndarray):
        return hash(('ndarray', x.shape, x.dtype.str, x.tobytes()))
    elif isinstance(x, generic):
        return hash(('npscalar', x.dtype.str, x.tobytes()))
    # Quantities: hash the magnitude and the units together. This is checked
    # before the sequence cases because a pint quantity may be array-like.
    elif hasattr(x, 'magnitude') and hasattr(x, 'units'):
        return hash(('quantity', content_hash(x.magnitude), str(x.units)))
    # Sequences.
    elif isinstance(x, (tuple, list)):
        return hash((type(x).__name__, tuple(content_hash(y) for y in x)))
    # Scalars and everything else.
    else:
        return hash(x)


def _tensor_to_numpy(t):
    '''Converts a torch tensor to a NumPy array for hashing.'''
    return asarray(t.detach().cpu())


# Content Equality ###########################################################

def values_equal(a, b, /):
    '''Determines whether two values are equal by content.

    This is the equality relation that ``content_hash`` hashes. Unlike the
    ``==`` operator, it returns a plain ``bool`` for array-likes, and unlike
    ``bool(a == b)`` it never raises on ambiguous or unsupported comparisons.

    Parameters
    ----------
    a, b : object
        The values to compare.

    Returns
    -------
    bool
        ``True`` if the two values have the same content.
    '''
    if a is b:
        return True
    # Quantities are compared by units and magnitude. They are checked before
    # the array cases because a quantity's magnitude may itself be an array,
    # and because ``quantity == quantity`` yields an array for array-valued
    # quantities rather than a bool.
    aquant = hasattr(a, 'magnitude') and hasattr(a, 'units')
    bquant = hasattr(b, 'magnitude') and hasattr(b, 'units')
    if aquant or bquant:
        if not (aquant and bquant):
            return False
        return (str(a.units) == str(b.units)
                and values_equal(a.magnitude, b.magnitude))
    aarr = isinstance(a, ndarray) or type(a).__module__.split('.')[0] == 'torch'
    barr = isinstance(b, ndarray) or type(b).__module__.split('.')[0] == 'torch'
    if aarr or barr:
        if not (aarr and barr):
            return False
        try:
            return bool(array_equal(_to_array(a), _to_array(b)))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _to_array(x, /):
    '''Returns a NumPy view of an array or tensor.'''
    if isinstance(x, ndarray):
        return x
    return _tensor_to_numpy(x)


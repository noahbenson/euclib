# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_math.py
'''The array operations, in whichever backend the arguments use.

``immlib.math`` is a common subset of NumPy and PyTorch whose backend follows
its arguments: if any of them is a tensor, the operation is done with torch, and
otherwise with numpy. That is exactly what euclib's interpolation needs, because
the *values*, the *positions* and a *geometry's coordinates* may each be a tensor
that requires a gradient, and the weights an interpolation builds depend on all
three. If the weights are built with the inputs' arithmetic, autograd supplies
the gradients without any analytic derivative code.

**What this module adds is the unwrapping.** Every ``immlib.math`` function
returns an ``immlib.Quantity``, which is right for a library that tracks units
and wrong here: an interpolation's intermediate values are raw arrays --- an
ordinate, a weight, a design matrix --- and a unit around each one would be a
unit system nobody asked for, and one that would have to be stripped at every
step. So each name in this module is ``immlib.math``'s, called the same way, with
its result returned as its magnitude.

Nothing here decides the backend; it asks, and gives back what it was given. If
every argument is a NumPy array the answer is a NumPy array, and if any is a
tensor the answer is a tensor carrying the graph it came from.
'''

# Dependencies ###############################################################

from __future__ import annotations

from functools import wraps

from immlib import math as _math


# Helpers ####################################################################

def _magnitude(value, /):
    '''The array inside a quantity, or the value itself if it is already one.'''
    return getattr(value, 'magnitude', value)


def _unwrapping(function, /):
    '''The function, with its result reduced to its magnitude.'''
    @wraps(function)
    def wrapped(*args, **kwargs):
        return _magnitude(function(*args, **kwargs))
    return wrapped


def __getattr__(name, /):
    '''Every name ``immlib.math`` has, with the result unwrapped.

    A module-level ``__getattr__``, so that the names are not enumerated here:
    whatever ``immlib.math`` carries, this carries, and a function added there is
    available here without a change. The one exception is the parameterless
    helper above, which has no quantity to unwrap and is defined directly.
    '''
    if name.startswith('_'):
        raise AttributeError(name)
    return _unwrapping(getattr(_math, name))


# Exports ####################################################################

__all__ = ('_magnitude',)

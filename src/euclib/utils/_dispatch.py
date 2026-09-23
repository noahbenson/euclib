# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_dispatch.py
'''The registry that chooses between a kernel's two implementations.

Every kernel that has a C counterpart also has a pure-Python one, and the
pure-Python one is the definition of correct behavior. This module decides which
of the two a call gets.

The decision is not only whether the extension loaded. The C kernels read
``float64`` arrays through the buffer protocol, so they handle NumPy arrays of
that dtype whose elements are contiguous in memory, and *convert* anything else
before reading it --- a ``float32`` array, a transposed view, and even a PyTorch
tensor all work, because the conversion machinery is happy to copy. The gate
declines to take the C path for those, and it is worth being exact about why,
because it is not that the answer would be wrong: for the two kernels here
today, every one of those inputs gives the same answer either way.

What it avoids is the conversion. A hidden copy of the arguments can easily cost
more than the kernel does --- a million strided points against a bisection that
takes 1.6 µs --- which would make the extension slower than the code it exists
to replace, in the case where it is hardest to notice. A caller who has
``float64`` NumPy arrays already pays nothing to be eligible; a caller who has
something else is told by the rule what the fast path is.

There is a second reason, which does not apply yet and will. A kernel such as
``simplex_measures`` is written in terms of ``immlib``'s numeric helpers and
gives back whatever backend it was handed, so a PyTorch caller keeps a tensor
and a gradient. A C version of such a kernel could not, and the gate is written
once for the family rather than per kernel so that it is already there.

The gate is deliberately cheap, for the same reason: a handful of attribute
checks and no argument binding, because a kernel such as
``tetrahedron_box_vertices`` costs a couple of microseconds and an expensive
gate would spend what it saves.
'''

# Dependencies ###############################################################

from __future__ import annotations

from inspect import signature

import numpy as np


# Eligibility ################################################################

def is_native_float64(value, /):
    '''Determines whether a value is one the C kernels can read directly.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        ``True`` if the value is a ``float64`` NumPy array whose elements are
        contiguous in memory. Everything else is ``False``: a ``float32``
        array, a transposed view, a list of lists, and an object with a ``shape``
        and a ``dtype`` that is not a NumPy array, such as a PyTorch tensor.
        The C kernels can convert most of those, and the conversion is a copy.
    '''
    return (isinstance(value, np.ndarray)
            and value.dtype == np.float64
            and value.flags.c_contiguous)


def is_dispatchable(args, kwargs, /):
    '''Determines whether every array-like argument admits the C kernels.

    A value that is not array-like --- a tolerance, an index, a flag --- has no
    bearing: the C kernels take those as plain Python numbers. A value that is
    array-like has to be a ``float64`` NumPy array with contiguous elements, or
    the call goes to the pure-Python kernel.

    Parameters
    ----------
    args : tuple
        The positional arguments of the call.
    kwargs : dict
        The keyword arguments of the call.

    Returns
    -------
    bool
        ``True`` if the C kernels can take this call.
    '''
    for value in args:
        if _is_arraylike(value) and not is_native_float64(value):
            return False
    for value in kwargs.values():
        if _is_arraylike(value) and not is_native_float64(value):
            return False
    return True


def _is_arraylike(value, /):
    '''Determines whether a value looks like an array rather than a number.'''
    return hasattr(value, 'shape') and hasattr(value, 'dtype')


# The registry ###############################################################

class Kernel:
    '''One kernel of the library: its two implementations and its gate.

    A kernel is a callable, so the rest of the library imports it and calls it
    without knowing that two implementations exist. Its name, docstring, and
    signature are the pure-Python kernel's, because that is the interface the
    library documents.

    Parameters
    ----------
    native : callable
        The pure-Python implementation, which is the definition of correct
        behavior and is used whenever the accelerated one does not apply.
    accelerated : callable or None
        The C implementation, or ``None`` if the extension did not load.
    '''

    def __init__(self, native, accelerated=None):
        self.native = native
        self.accelerated = accelerated
        self.__name__ = native.__name__
        self.__qualname__ = native.__qualname__
        self.__doc__ = native.__doc__
        self.__module__ = native.__module__
        # A kernel is called as its pure-Python counterpart is called, and its
        # documentation describes that call, so it has to answer to
        # introspection as that counterpart does. Without this a kernel reports
        # the signature of the generic ``__call__`` --- ``(*args, **kwargs)``
        # --- to ``help``, to an editor, and to the test that holds every
        # public docstring to its signature. The C implementation is given the
        # same parameters, which is what test_dispatch checks.
        self.__wrapped__ = native
        self.__signature__ = signature(native)

    def __call__(self, *args, **kwargs):
        accelerated = self.accelerated
        if (accelerated is not None
                and is_dispatchable(args, kwargs)):
            return accelerated(*args, **kwargs)
        return self.native(*args, **kwargs)

    def __repr__(self):
        where = 'C' if self.accelerated is not None else 'Python'
        return f"<euclib kernel {self.__name__} ({where})>"

    @property
    def accelerated_here(self):
        '''Whether this kernel's C implementation is in use.'''
        return self.accelerated is not None


# The kernels ################################################################

def kernel(native, accelerated=None, /):
    '''Builds the dispatching kernel for a pair of implementations.

    Parameters
    ----------
    native : callable
        The pure-Python implementation.
    accelerated : callable or None, optional
        The C implementation. The default, ``None``, leaves the kernel on the
        pure-Python path, which is what happens when the extension is absent.

    Returns
    -------
    Kernel
        A callable that selects between the two.
    '''
    return Kernel(native, accelerated)

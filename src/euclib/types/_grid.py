# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/_grid.py
'''B-splines, and the prefilter that makes them interpolate.

A grid's other kernels -- the box, the triangle, the cubic convolution -- are
*interpolating*: their value is one at zero and zero at every other integer, so
the data can serve as their coefficients and no work is needed before the sum.
The B-splines are not. The cubic is two thirds at zero and a sixth at each
neighbour, so using the data as coefficients gives a smoothed field that misses
the data at every sample -- which is the whole reason this module exists.

**The two-step.** Interpolation with a basis ``phi`` means finding coefficients
``c`` with ``sum_n c_n phi(t - n) = f_m`` at every sample ``m``. Writing ``p_m =
phi(m)`` for the basis's values at the samples, that condition is the convolution
``p * c = f``, so ``c = (p)^-1 * f``: a **prefilter**, followed by the same
weighted sum every grid method does. For a B-spline the inverse is a pair of
first-order recursions, one running forward and one backward, so the whole
prefilter costs a few operations per cell rather than a solve.

The recursions' rate is the root of ``p``'s reciprocal that lies inside the unit
circle, and the scale is what the recursions must carry so that their filter
inverts ``p``. Both are computed here from the basis's values rather than quoted,
and the tests check them against a direct solve of ``p * c = f``.

**The boundary enters twice.** A position near an edge needs coefficients from
outside the data, and which ones is the same question the boundary extension
answers for a kernel's stencil. Rather than closing each recursion with an
endpoint condition derived per extension, this module *extends the data* over a
margin by the extension's rule and filters that; the recursions are then the same
for every extension and the answer is exact to the rate raised to the margin.
'''

# Dependencies ###############################################################

from __future__ import annotations

import immlib.math as im
from immlib import quant
from numpy import (
    abs, argsort, asarray, ones_like, pi, sin, sqrt, where, zeros)


def _zeros_for(like, shape, /):
    """An array of zeros in the backend and dtype of another array.

    ``immlib.math`` deliberately carries no allocator, because naming the backend
    per call is the same as calling torch or numpy and a caller who knows the
    backend may as well call it. A *quantity* does carry one, so this wraps the
    array, allocates, and hands back the magnitude --- which is the backend, the
    dtype and, where it matters, the device of the array it follows.
    """
    return quant(like).new_zeros(shape).m

# The bases ##################################################################

def bspline2(t, /):
    '''The quadratic B-spline, normalised so that its values sum to one.

    Even, supported on ``(-3/2, 3/2)``, so a position draws on three samples
    along each axis. Its values at the samples are ``3/4`` at the centre and
    ``1/8`` at each neighbour.
    '''
    t = abs(asarray(t, dtype='float64'))
    return (where(t <= 0.5, 0.75 - t ** 2, 0.0)
            + where((t > 0.5) & (t < 1.5), 0.5 * (1.5 - t) ** 2, 0.0))


def bspline3(t, /):
    '''The cubic B-spline, normalised so that its values sum to one.

    Even, supported on ``(-2, 2)``, so a position draws on four samples along
    each axis -- the same support the cubic convolution kernel has, spent on one
    more order of reproduction and one more degree of continuity. Its values at
    the samples are ``2/3`` at the centre and ``1/6`` at each neighbour.
    '''
    t = abs(asarray(t, dtype='float64'))
    return (where(t <= 1.0, 2.0 / 3.0 - t ** 2 + t ** 3 / 2.0, 0.0)
            + where((t > 1.0) & (t < 2.0), (2.0 - t) ** 3 / 6.0, 0.0))


def _sinc(t, /):
    '''The sinc function, defined to be one at the origin.'''
    t = asarray(t, dtype='float64')
    out = ones_like(t)
    nonzero = t != 0
    out[nonzero] = sin(pi * t[nonzero]) / (pi * t[nonzero])
    return out


def lanczos(t, n, /):
    '''The sinc windowed by a narrowed sinc, supported on ``(-n, n)``.

    The window is what makes this usable: it vanishes wherever ``t`` is a
    non-zero multiple of ``n``, so the kernel goes to zero at the edge of its
    support rather than being cut off there, and it leaves the kernel's zeros at
    the other integers alone --- which is what keeps the method interpolating.
    '''
    t = abs(asarray(t, dtype='float64'))
    return where(t < n, _sinc(t) * _sinc(t / n), 0.0)


def lanczos2(t, /):
    '''Lanczos-2: the sinc windowed by ``sinc(t/2)``, on four samples.'''
    return lanczos(t, 2)


def lanczos3(t, /):
    '''Lanczos-3: the sinc windowed by ``sinc(t/3)``, on six samples.'''
    return lanczos(t, 3)


#: The basis of each degree, and how many samples of the *coefficients* a
#: position draws on: the lowest offset from its own cell, and how many follow.
#: The quadratic needs three and the cubic four, so a fixed "half-width either
#: side" would not do.
BASES = {2: (bspline2, -1, 3), 3: (bspline3, -1, 4)}


# The prefilter ##############################################################

#: The half a B-spline patch needs in order that a position's nearest three or
#: four coefficients are the ones that matter, and that the recursion's rate has
#: decayed below the arithmetic's noise before the data is reached. At the cubic
#: rate this is under 1e-22 of the data's own scale.
MARGIN = 32


def _rate(degree, /):
    '''The prefilter's rate and scale, from the basis's values at the samples.

    The prefilter's Z-transform is ``1/p(z)``, with ``p`` the sum of the basis's
    values at the samples weighted by powers of ``z``. Being an even basis, that
    polynomial is palindromic, so its two roots are reciprocals; one lies inside
    the unit circle and one outside, and the reciprocal factors into a causal
    factor in the inside one and an anti-causal factor in the outside one ---
    which is what makes the filter two first-order recursions rather than a
    solve.

    Both numbers are computed rather than quoted. The roots come from the
    basis's own values, so a change to the normalisation would change them too;
    the scale is measured by running the recursions on an impulse, whose
    response is known to be the inverse of ``p`` at the centre.

    Parameters
    ----------
    degree : int
        Two for the quadratic basis, three for the cubic.

    Returns
    -------
    r : float
        The rate: the root inside the unit circle. Negative for both degrees.
    scale : float
        What the recursions' result must be multiplied by.
    '''
    basis = BASES[degree][0]
    # p(z) is proportional to z + (p_0 / p_1) + z^-1 for an even basis whose
    # sample values are p_0 at the centre and p_1 at the neighbours, so the
    # roots are those of z^2 + (p_0 / p_1) z + 1.
    (centre, beside) = (float(basis(0)), float(basis(1)))
    slope = centre / beside
    disc = sqrt(slope * slope - 4.0)
    roots = [(-slope + disc) / 2.0, (-slope - disc) / 2.0]
    inside = min(roots, key=abs)
    if abs(inside) >= 1.0:
        raise ValueError(f"the degree-{degree} prefilter has no rate inside"
                         f" the unit circle: its roots are {roots}")
    return (inside, _scale_from_impulse(inside, degree))


def _recursions(values, r, /):
    '''The two-step prefilter of an array of rows, along its last axis.

    The causal pass runs forward with rate ``r``, the anti-causal one backward
    with the same rate, and the second is what turns the first's partial result
    into the inverse of ``p``. Both passes are seeded at their far ends with the
    fixed point a constant input would give, which is right to
    ``|r| ** (the distance from that end)`` and so is right to the arithmetic's
    precision anywhere the data itself extends far enough.
    '''
    width = values.shape[-1]
    forward = _zeros_for(values, values.shape)
    forward[..., 0] = values[..., 0] / (1.0 - r)
    for k in range(1, width):
        forward[..., k] = values[..., k] + r * forward[..., k - 1]
    backward = _zeros_for(values, values.shape)
    backward[..., -1] = forward[..., -1] / (1.0 - r)
    for k in range(width - 2, -1, -1):
        backward[..., k] = r * (backward[..., k + 1] - forward[..., k])
    return backward


def _scale_from_impulse(r, degree, /):
    '''What the recursions must be multiplied by for their filter to invert
    ``p``.

    Inverting ``p`` means the result, summed against ``p``, is the impulse back
    again; the recursions give something proportional to that inverse, and the
    scale is what makes the proportion hold at the impulse's own position. It
    is measured rather than derived so that it cannot drift away from the rate:
    whatever the rate is, this is the scale that goes with it.
    '''
    width = 4 * MARGIN + 1
    at = 2 * MARGIN
    impulse = zeros(width)
    impulse[at] = 1.0
    response = _recursions(impulse, r)
    basis = BASES[degree][0]
    total = 0.0
    for k in range(-degree, degree + 1):
        total += float(basis(k)) * float(response[at + k])
    return 1.0 / total


def prefilter(values, spatial_shape, border, degree, /):
    '''The B-spline coefficients of a grid's values, and where they start.

    The values' leading axes are channels and its trailing ones are the grid's,
    so the filter runs along each grid axis in turn -- the two-dimensional case
    being "filter along each column, then along each row", which is what
    separability buys.

    Each axis is extended past both ends by ``border``'s rule over `MARGIN`
    cells before it is filtered, and the extension's coefficients are returned
    with the data's: a position within two cells of an edge draws on them.

    Parameters
    ----------
    values : array-like
        The grid's values, with the channel axes leading.
    spatial_shape : sequence of int
        The grid's extent, whose length says how many trailing axes are the
        grid's and which are channels.
    border : str
        One of ``euclib.abc.BORDER_EXTENSIONS``.
    degree : int
        Two for the quadratic basis, three for the cubic.

    Returns
    -------
    coefficients : numpy.ndarray
        The coefficients, one margin longer at each end of every grid axis.
    first : tuple of int
        The sample index the first entry along each grid axis belongs to, which
        is ``-MARGIN`` for each of them.
    '''
    (r, scale) = _rate(degree)
    # The values are used as they are, including their dtype: a property whose
    # values are float32 is interpolated in float32 rather than quietly widened,
    # so that the answer is what the caller's arithmetic can express.
    out = asarray(values) if not hasattr(values, 'shape') else values
    axes = tuple(range(out.ndim - len(spatial_shape), out.ndim))
    for axis in axes:
        out = _filter_axis(out, axis, r, scale, border)
    return (out, (-MARGIN,) * len(axes))


def _filter_axis(values, axis, r, scale, border, /):
    '''The two-step prefilter along one axis of an array.'''
    moved = _move_front(values, axis)
    (lead, length) = (moved.shape[:-1], moved.shape[-1])
    flat = moved.reshape(-1, length)
    width = 2 * MARGIN + length
    padded = _zeros_for(flat, (flat.shape[0], width))
    padded[:, MARGIN:MARGIN + length] = flat
    _extend_into(padded, MARGIN, border)
    backward = _recursions(padded, r)
    got = (scale * backward).reshape(lead + (width,))
    return _move_back(got, axis, values.ndim)


def _extend_into(padded, margin, border, /):
    '''Fills a padded array's margins from its middle, by the extension's rule.

    The three extensions are the ones the boundary page derives: constant
    repeats the edge value, half-sample symmetric repeats it once and then
    reflects, and whole-sample symmetric reflects at once.
    '''
    inner = padded[:, margin:-margin]
    length = inner.shape[1]
    for k in range(margin):
        if border == 'constant':
            padded[:, margin - 1 - k] = inner[:, 0]
            padded[:, -margin + k] = inner[:, -1]
        elif border == 'half-symmetric':
            period = 2 * length
            index = (margin - 1 - k) % period
            padded[:, margin - 1 - k] = inner[:, min(index, period - 1 - index)]
            index = (length + k) % period
            padded[:, -margin + k] = inner[:, min(index, period - 1 - index)]
        else:
            period = 2 * length - 2
            index = (margin - 1 - k) % period
            padded[:, margin - 1 - k] = inner[:, min(index, period - index)
                                               % length]
            index = (length + k) % period
            padded[:, -margin + k] = inner[:, min(index, period - index)
                                           % length]


def _move_front(values, axis, /):
    '''The array with one axis moved to the end, as a contiguous copy.'''
    # A tuple, because both backends' permutation takes the axes as one and
    # numpy will not read a list of them where a shape is expected.
    order = tuple([x for x in range(values.ndim) if x != axis] + [axis])
    # ``immlib.math`` wraps its result in a quantity; the axis moves here are
    # bookkeeping on the array itself, so the magnitude is what is wanted.
    return im.mag(im.permute(values, order))


def _move_back(values, axis, ndim, /):
    '''The inverse of `_move_front`.'''
    order = tuple(int(x) for x in
                  argsort([x for x in range(ndim) if x != axis] + [axis]))
    return im.mag(im.permute(values, order))


# Exports ####################################################################

__all__ = ('bspline2', 'bspline3', 'BASES', 'MARGIN', 'prefilter',
           'lanczos', 'lanczos2', 'lanczos3')

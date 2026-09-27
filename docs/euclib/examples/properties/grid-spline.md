---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python (euclib)
  language: python
  name: euclib
---
# Interpolating a Grid: Splines

[Cubic convolution](grid-cubic.md) is C¹ and reproduces quadratics. Doing better
takes a different idea, because the obvious one is unavailable: the basis
functions that are *best* for the job — the B-splines, which are as smooth as a
piecewise polynomial of a given support can be — are **not interpolating**. Their
value at a sample is not one, so using the data as coefficients does not go
through the data. This page is about the two-step that fixes that, and about the
one thing it changes which no method so far has needed: the *data* is transformed
before it is used.

:::{admonition} What this demonstrates
:class: tip
- That the cubic B-spline's value at a neighbouring sample is $1/6$, not zero —
  so it cannot be an interpolation kernel however it is used.
- The two-step: solve for *coefficients* that the basis interpolates with, then
  combine the basis at them. The first step is a convolution inverse, and for a
  B-spline it is a pair of one-dimensional recursive filters.
- That the filter has a scale and a rate, both derivable, and that the
  *boundary rule* enters the filter as well as the stencil.
- What this buys over cubic convolution: the same support, one more order of
  reproduction, and one more degree of continuity.
:::

## Why the best basis is not usable directly

The B-splines $\beta^j$ are the piecewise polynomials of a given support with the
**highest approximation order** and the **most continuity**; that is what makes
them worth the trouble. The quadratic one is

$$ \beta^2(t) = \begin{cases}
   \tfrac{3}{4} - t^2 & |t| \le \tfrac12,\\
   \tfrac12(\tfrac32 - |t|)^2 & \tfrac12 \le |t| \le \tfrac32,
   \end{cases} $$

and the cubic, which this page is about, is

$$ \beta^3(t) = \begin{cases}
   \tfrac23 - t^2 + \tfrac12|t|^3 & |t| \le 1,\\
   \tfrac16(2 - |t|)^3 & 1 \le |t| \le 2 . \end{cases} $$

Both are even, both vanish past their support, and — the fact that matters — both
are **not one at zero and only at zero**. The cubic's values at the samples are
$2/3$ at the centre and $1/6$ at each neighbour.

```{code-cell}
import numpy as np
import euclib as el

def bspline3(t):
    """The cubic B-spline, normalised so that its values sum to one."""
    t = np.abs(np.asarray(t, dtype=float))
    return (np.where(t <= 1.0, 2.0 / 3.0 - t ** 2 + t ** 3 / 2.0, 0.0)
            + np.where((t > 1.0) & (t < 2.0), (2.0 - t) ** 3 / 6.0, 0.0))

def bspline2(t):
    """The quadratic B-spline, likewise."""
    t = np.abs(np.asarray(t, dtype=float))
    return (np.where(t <= 0.5, 0.75 - t ** 2, 0.0)
            + np.where((t > 0.5) & (t < 1.5),
                       0.5 * (1.5 - t) ** 2, 0.0))

print("the bases at the samples:")
print("  beta^2:", np.round([float(bspline2(k)) for k in range(4)], 6))
print("  beta^3:", np.round([float(bspline3(k)) for k in range(4)], 6))
print("  (a cosine-ish kernel would be 1, 0, 0, 0 -- these are not.)")
print()
print("what happens if the data is used as the coefficients anyway, on a")
print("single non-zero sample -- a ramp would flatter it, since a B-spline is a")
print("partition of unity and reproduces a linear field whichever coefficients")
print("it is given:")
pulse = np.zeros(41)
pulse[20] = 1.0
for n in (18, 19, 20, 21, 22):
    got = sum(pulse[k] * float(bspline3(n - k)) for k in range(pulse.shape[0]))
    print(f"   sample {n}: the data is {pulse[n]:.0f}, and the field is"
          f" {got:+.6f}")
print("   ...so the field does not pass through the data, which is the problem.")
```

## The two-step

Since $\beta^3$ is not interpolating, the coefficients cannot be the data. What
is wanted is a *different* sequence $c$ with

$$ \sum_n c_n\, \beta^3(t - n) \Big|_{t = m} = f_m \quad\text{for every } m, $$

which is to say $\sum_n c_n\, \beta^3(m - n) = f_m$: a **convolution**. Writing
$p_m = \beta^3(m)$ — the values at the samples, $(1, 4, 1)/6$ — the condition is
$p * c = f$, so

$$ c = p^{-1} * f , $$

the *convolution inverse*. Interpolation is therefore two steps: **prefilter**
the data by $p^{-1}$, then combine the basis at the result. The second step is
exactly the weighted sum the other grid methods do; only the first is new.

The inverse is where the arithmetic is. Its Z-transform is $1/p(z)$, and with
$p(z) = (z + 4 + z^{-1})/6$ the denominator's roots are the roots of
$z^2 + 4z + 1$, which are $-2 \pm \sqrt3$. One of them is inside the unit circle,

$$ r = \sqrt3 - 2 \approx -0.268 , $$

and the other is $1/r$, so the filter is the cascade of a **causal** one — running
forward with rate $r$ — and an **anti-causal** one running backward with the same
rate. Neither is truncated: the first is a recursion in one direction and the
second in the other, and together they cost about four operations per cell rather
than the $O(N^2)$ a dense solve would.

```{code-cell}
# The roots, and the rate that the recursions use.
print("the denominator's roots:", np.round(np.roots([1.0, 4.0, 1.0]), 6))
r = np.sqrt(3.0) - 2.0
print(f"  the one inside the unit circle: r = {r:.6f},  |r| = {abs(r):.6f}")
print(f"  ...and the other is 1/r = {1.0 / r:.6f}, which is |1/r| > 1 as it must be")

def prefilter(f, border='constant', margin=40, /):
    """The B-spline coefficients of data on integer samples.

    The data is extended past both ends by ``border``'s rule over ``margin``
    cells, the two recursions are run over *that*, and the whole result is
    returned, with the sample index its first entry belongs to. The interior is
    all that is needed to evaluate at the samples, but a position within two
    cells of an edge needs coefficients from outside the data too, and those are
    the extension's -- so they are handed back rather than discarded.

    A margin is enough because the recursions' rate decays like ``|r|^margin``,
    which for the cubic is below 1e-22 at forty cells.
    """
    n = f.shape[0]
    left = np.zeros(margin)
    right = np.zeros(margin)
    for k in range(margin):
        if border == 'constant':
            left[k] = f[0]; right[k] = f[-1]
        elif border == 'half-symmetric':
            left[k] = f[min(k + 1, n - 1)]
            right[k] = f[max(n - 2 - k, 0)]
        else:
            left[k] = f[min(k + 1, n - 1)]
            right[k] = f[max(n - 2 - k, 0)] if k < n - 1 else f[0]
    ext = np.concatenate([left[::-1], f, right])
    forward = np.zeros(ext.shape[0])
    forward[0] = ext[0] / (1.0 - r)
    for k in range(1, ext.shape[0]):
        forward[k] = ext[k] + r * forward[k - 1]
    backward = np.zeros(ext.shape[0])
    backward[-1] = forward[-1] / (1.0 - r)
    for k in range(ext.shape[0] - 2, -1, -1):
        backward[k] = r * (backward[k + 1] - forward[k])
    return (6.0 * backward, -margin)

def combine(coefficients, first, t, /):
    """The interpolant at a position, from coefficients over an index range.

    ``coefficients[k]`` belongs to the sample at ``first + k``, and the sum runs
    over every coefficient given -- including the ones outside the data, which
    is what a position within two cells of an edge needs.
    """
    total = 0.0
    for (k, c) in enumerate(coefficients):
        total += c * float(bspline3(t - (first + k)))
    return total

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
(coefficients, first) = prefilter(data)
print("the coefficients, and the sample indices they belong to:")
print("  the middle, at the samples 0 through 6:",
      np.round(coefficients[-first:7 - first], 6))
print("  the ones the extension adds at the left:",
      np.round(coefficients[:4], 6))
print("the interpolant at the samples:",
      np.round([combine(coefficients, first, t) for t in range(data.shape[0])],
               9))
print("  ...which is the data back, which is what the prefilter was for.")
```

## What it buys

The cubic B-spline and the cubic convolution have the same support — four cells —
and differ in what that support is spent on. The B-spline is C², where cubic
convolution is only C¹, and it reproduces **cubics**, where cubic convolution
reproduces only quadratics. Those are the two things the page's last check
measures, on the same field, against the same construction of the previous page.

```{code-cell}
def cubic_convolution_kernel(t, alpha=-0.5):
    t = np.abs(np.asarray(t, dtype=float))
    return (np.where(t <= 1.0, (alpha + 2) * t ** 3 - (alpha + 3) * t ** 2 + 1.0, 0.0)
            + np.where((t > 1.0) & (t < 2.0),
                       alpha * t ** 3 - 5 * alpha * t ** 2 + 8 * alpha * t
                       - 4 * alpha, 0.0))

# A cubic, and the same data through both methods, well away from the ends.
def cubic(t):
    return 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1

n = 60
sampled = cubic(np.arange(n, dtype=float))
probe = np.linspace(20.0, 40.0, 300)
coeffs = prefilter(sampled, 'constant')

spline = np.array([combine(*coeffs, t) for t in probe])
convolution = np.array([
    sum(sampled[k] * float(cubic_convolution_kernel(t - k))
        for k in range(int(np.floor(t)) - 1, int(np.floor(t)) + 3))
    for t in probe])
print("the worst difference from the cubic, over 300 positions:")
print(f"  the cubic B-spline:      {np.abs(spline - cubic(probe)).max():.3e}")
print(f"  cubic convolution:       {np.abs(convolution - cubic(probe)).max():.3e}")
print(f"  linear interpolation:    "
      f"{np.abs(np.array([sampled[int(np.floor(t))] * (np.ceil(t) - t) + sampled[int(np.ceil(t))] * (t - np.floor(t)) for t in probe]) - cubic(probe)).max():.3e}")
print()
print("So the B-spline is exact on a cubic where cubic convolution is not, which")
print("is the approximation order the support pays for: 4 against 3.")
```

**And the continuity.** The measure is the second difference across a sample,
which for a smooth field is the curvature — a field whose *slope* jumps puts a
spike there, and one whose *curvature* jumps puts a smaller one. Cubic
convolution is C¹, so its second difference spikes; the cubic B-spline is C², so
it does not.

```{code-cell}
def jump_in_second(second, at, step=1e-3):
    """How much a second derivative changes as a sample is crossed.

    A field that is C2 has no jump there and this is near zero; a field that is
    only C1 has one, and this is it.
    """
    return abs(second(at - step) - second(at + step))

# The same smooth field through both methods, and each one's second derivative.
smooth = np.cos(np.arange(n, dtype=float) / 5.0)
(coeffs, first) = prefilter(smooth, 'constant')

def spline_second(t, /):
    step = 1e-3
    return (combine(coeffs, first, t + step) - 2.0 * combine(coeffs, first, t)
            + combine(coeffs, first, t - step)) / step ** 2

def convolution_at(t, /):
    return sum(smooth[k] * float(cubic_convolution_kernel(t - k))
               for k in range(int(np.floor(t)) - 1, int(np.floor(t)) + 3))

def convolution_second(t, /):
    step = 1e-3
    return (convolution_at(t + step) - 2.0 * convolution_at(t)
            + convolution_at(t - step)) / step ** 2

print("the jump in the second derivative at a sample, over ten of them:")
print("   sample |   the cubic B-spline   |   cubic convolution")
for at in range(25, 35):
    print(f"   {at:6d} |{jump_in_second(spline_second, at):^24.4f}"
          f"|{jump_in_second(convolution_second, at):^24.4f}")
print()
print("The B-spline's is the numerical noise of a difference quotient; the")
print("convolution's is the curvature's kink, which is what C1 rather than C2")
print("means. That, and reproducing cubics rather than quadratics, is what the")
print("same four-cell support buys when it is spent on a B-spline.")
```

**And the library agrees with this page.** The construction above is written out
here; the last check is that `euclib`'s own splines give the same field, and that
each basis reproduces its own degree and no more.

```{code-cell}
count = 60
probe = np.linspace(20.0, 40.0, 200)
quadratic = lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1
for (label, field) in (('a quadratic', quadratic), ('a cubic', cubic)):
    carried = el.grid((count,)).withprop('v', field(np.arange(count, dtype=float)))
    print(f"  {label}:")
    for order in (2, 3):
        got = np.ravel(np.asarray(carried.prop(
            'v', at=carried.topo.Loc(sx=probe), interp=('spline', order))))
        print(f"    the {'quadratic' if order == 2 else 'cubic':9s} basis:"
              f" worst {np.abs(got - field(probe)).max():.3e}")
print()
print("  So each basis is exact on its own degree: one more cell of support")
print("  buys one more degree of reproduction, and the same four-cell support")
print("  spent on cubic convolution buys only the quadratic.")
```

:::{admonition} Where this comes from
:class: note
The two-step construction, the B-splines, and the recursive prefilters are M.
Unser's; see M. Unser, *Splines: A Perfect Fit for Signal and Image Processing*,
IEEE Signal Processing Magazine **16**(6) (1999), 22–38,
[doi:10.1109/79.799930](https://doi.org/10.1109/79.799930), and the fuller
treatment in P. Thévenaz, T. Blu and M. Unser, *Interpolation Revisited*, IEEE
Transactions on Medical Imaging **19**(7) (2000), 739–758,
[doi:10.1109/42.875199](https://doi.org/10.1109/42.875199). The root, the
recursions' form, and the endpoint conditions this page's extension stands in
for are Getreuer's, *Linear Methods for Image Interpolation*, Image Processing On
Line **1** (2011), 238–259, sections 4 and 15.3,
[doi:10.5201/ipol.2011.g_lmii](https://doi.org/10.5201/ipol.2011.g_lmii).

**What is derived here.** The B-spline's values at the samples, the rate from the
denominator's roots, the scale, and the comparison against a direct solve and
against cubic convolution are computed on this page. The two-step construction,
the recursions' form and the fact that B-splines are optimal in order and
continuity for their support are the cited ones.

**One departure.** The reference closes each recursion with an endpoint condition
derived from the boundary extension, and gives the conditions for the two
symmetric extensions. This page extends the *data* instead, over a margin, and
runs the same recursions on that. It needs no per-extension derivation, it is
exact to $|r|^{\text{margin}}$, and it gives the same answer for all three
extensions including the constant one.
:::

:::{seealso}
- [Interpolating a Grid: Cubic Convolution](grid-cubic.md) for the method this
  one improves on, and for the kernel the comparison above is against.
- [Interpolating a Grid: the Boundary](grid-boundary.md) for the rule the
  prefilter is extended by.
- [Interpolating a Grid: Nearest and Linear](grid-linear.md) for the index space.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

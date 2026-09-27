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
# Interpolating a Grid: Lanczos

Every method so far has been a *piecewise polynomial*: a triangle, a cubic, a
B-spline, each one reproducing the polynomials its support can pay for, exactly.
Lanczos is a different kind of thing — an approximation to the **ideal**
interpolator, the sinc, cut down to a finite support — and it behaves
differently in a way worth being precise about. This page derives the kernel,
measures what it does and does not reproduce, and ends with the whole family's
properties in one table.

:::{admonition} What this demonstrates
:class: tip
- Why the sinc is the interpolator to approximate, and why it has to be cut
  short: it is exact on bandlimited data and has infinite support.
- The Lanczos window, and why *that* window: it vanishes at the multiples of $n$,
  so the kernel stays interpolating where it is truncated.
- The one thing it needs that no other method has: **normalization**, because its
  weights do not sum to one.
- The difference between the two kinds of method, measured: a polynomial kernel
  reproduces its polynomials *exactly*, an approximation to the sinc reproduces
  them only in the limit.
:::

## The ideal interpolator, and why it is unusable

The sinc kernel is the one that is exact on bandlimited data: sampling a
bandlimited function and summing $f_m\operatorname{sinc}(t - m)$ gives the
function back exactly. It is also the interpolating kernel *nearest* to the ideal
in a mean-square sense, and it is the slowest to decay of any of them — it falls
off like $1/t$ — so a sample that is wrong affects the whole reconstruction, and
an edge in the data shows up as ripples across the field.

It is also of infinite support, which is the practical obstacle: no finite sum
can compute it. Cutting it off at some distance is the obvious fix and the wrong
one, because a hard cut makes the kernel discontinuous at the cut, and that
discontinuity is itself an artefact.

The Lanczos window is the standard answer. It replaces the truncated kernel by

$$ K(t) = \operatorname{sinc}(t)\;\operatorname{sinc}(t/n) \qquad \text{for }
   |t| < n , $$

a product of the sinc with a *narrowed* sinc. The narrowing is what makes it
work: $\operatorname{sinc}(t/n)$ vanishes wherever $t$ is a non-zero multiple of
$n$, so at the edge of the support the window is going to zero of its own accord
rather than being cut, and — the point that keeps the method interpolating — it
does not disturb the kernel's zeros at the other integers, because there the
*sinc* factor is already zero.

```{code-cell}
import numpy as np
import euclib as el

def sinc(t):
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    nz = t != 0
    out[nz] = np.sin(np.pi * t[nz]) / (np.pi * t[nz])
    return out

def lanczos(t, n):
    """The sinc windowed by a narrowed sinc, supported on (-n, n)."""
    t = np.abs(np.asarray(t, dtype=float))
    return np.where(t < n, sinc(t) * sinc(t / n), 0.0)

print("the kernel at the samples, where it must be one at zero and zero")
print("at every other integer:")
for n in (2, 3):
    print(f"  n = {n}:", np.round([float(lanczos(k, n)) for k in range(6)], 12))
print()
print("...and where its support ends, which is where a cut would have shown:")
for n in (2, 3):
    print(f"  n = {n}: K({n} - 1e-6) = {float(lanczos(n - 1e-6, n)):+.3e}"
          f"   K({n}) = {float(lanczos(n, n)):+.3e}")
print("  (the window takes it to zero, so there is nothing to cut.)")
```

## It does not reproduce constants, and that has to be fixed

Here is what separates Lanczos from everything so far. A kernel reproduces a
constant field when its values sum to one at every position — the *partition of
unity* the polynomial kernels get for free from their moment conditions and the
B-splines from their normalization. The Lanczos kernel does not have it: its
weights sum to a little over one at some positions and a little under at others,
so a constant field does not come back constant.

The fix is to divide by the sum, position by position. That is not the same as
scaling the kernel, because the sum depends on where between the samples the
position falls; it is a normalization of each position's weights, and it costs
one division.

```{code-cell}
print("the weights over the stencil at some positions, in each dimension:")
for n in (2, 3):
    for off in (0.0, 0.25, 0.5):
        base = int(np.floor(off))
        ks = np.arange(base - n + 1, base + n + 1)
        w = np.array([float(lanczos(off - k, n)) for k in ks])
        print(f"  n = {n} at {off:.2f}: sum {w.sum():.12f}"
              f"   ({(w.sum() - 1.0) * 100:+.3f}% of one)")
print()
print("So the field has to divide by that sum. The check below is the point of")
print("it: a constant field, before and after.")
count = 60
probe = np.linspace(20.0, 40.0, 200)
flat = np.full(count, 3.0)
for n in (2, 3):
    (raw, normed) = ([], [])
    for t in probe:
        base = int(np.floor(t))
        ks = np.arange(base - n + 1, base + n + 1)
        w = np.array([float(lanczos(t - k, n)) for k in ks])
        raw.append(float(flat[ks] @ w))
        normed.append(float(flat[ks] @ w) / float(w.sum()))
    print(f"  n = {n}: raw worst {np.abs(np.array(raw) - 3.0).max():.3e}"
          f"   normalized worst {np.abs(np.array(normed) - 3.0).max():.3e}")
```

## What it does *not* reproduce, measured

Normalization fixes constants and nothing else. This is the difference between
the two families, and it is worth seeing plainly: the polynomial kernels
reproduce their polynomials to the arithmetic's precision, and Lanczos — even
normalized — reproduces none of them.

The reason is not a defect and not an oversight. The moment conditions the other
methods satisfy *identically* are, for a sinc approximation, satisfied only in
the limit: the error decays as the sampling is refined, and the order names how
fast, but at any fixed sampling there is an error. The ideal sinc does not
reproduce polynomials either — it reproduces *bandlimited* functions, and a
polynomial is not one.

```{code-cell}
def cubic_kernel(t, alpha=-0.5):
    t = np.abs(np.asarray(t, dtype=float))
    return (np.where(t <= 1.0, (alpha + 2) * t ** 3 - (alpha + 3) * t ** 2 + 1.0, 0.0)
            + np.where((t > 1.0) & (t < 2.0),
                       alpha * t ** 3 - 5 * alpha * t ** 2 + 8 * alpha * t
                       - 4 * alpha, 0.0))

def linear_kernel(t):
    return np.maximum(1.0 - np.abs(np.asarray(t, dtype=float)), 0.0)

def worst_on(kernel, half, field, normalize, /):
    """The worst error, over positions well away from the ends of the data."""
    n = 90
    data = field(np.arange(n, dtype=float))
    worst = 0.0
    for t in np.linspace(30.0, 60.0, 300):
        base = int(np.floor(t))
        ks = np.arange(base - half + 1, base + half + 1)
        w = np.array([float(kernel(t - k)) for k in ks])
        if normalize:
            w = w / w.sum()
        worst = max(worst, abs(float(data[ks] @ w) - float(field(t))))
    return worst

fields = (('constant', lambda t: 0.0 * t + 3.0),
          ('affine', lambda t: 0.4 * t + 1.1),
          ('quadratic', lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1),
          ('cubic', lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1))
print("the worst error on each polynomial, away from the ends:")
print(f"  {'method':22s} " + " ".join(f"{name:>11}" for (name, _) in fields))
for (label, kernel, half, norm) in (
        ('linear', linear_kernel, 1, False),
        ('cubic convolution', cubic_kernel, 2, False),
        ('lanczos 2', lambda t: lanczos(t, 2), 2, True),
        ('lanczos 3', lambda t: lanczos(t, 3), 3, True)):
    row = [worst_on(kernel, half, field, norm) for (_, field) in fields]
    print(f"  {label:22s} " + " ".join(f"{v:11.2e}" for v in row))
print()
print("The two polynomial kernels are exact where their order says they should")
print("be, to the arithmetic's precision. Lanczos is exact on the constant and")
print("on nothing else -- its conditions hold in the limit, not now.")
```

**But the error does decay.** That is what an approximation order means, and it
is measurable: refine the sampling and watch the error fall. On a smooth field
the error of every method here falls at a rate set by its order — and, for a
kernel that is *symmetric*, a rate faster than the nominal order, because a
symmetric kernel's odd-order error terms vanish identically. Cubic convolution is
of order 3 by the definition and its error falls like the fourth power; Lanczos
does the same.

```{code-cell}
smooth = lambda t: np.sin(t) * np.exp(-t / 8.0)

def decay(kernel, half, normalize, /, levels=6):
    """The error at the cell midpoints as the sampling step halves."""
    errors = []
    for level in range(levels):
        h = 0.5 ** level
        xs = np.arange(0.0, 40.0, h)
        data = smooth(xs)
        base = np.arange(half, xs.size - half)
        probe = xs[base] + h / 2
        got = np.zeros(probe.shape)
        for i in range(base.size):
            ks = np.arange(base[i] - half + 1, base[i] + half + 1)
            w = np.array([float(kernel((probe[i] - xs[k]) / h)) for k in ks])
            if normalize:
                w = w / w.sum()
            got[i] = data[ks] @ w
        errors.append(np.abs(got - smooth(probe)).max())
    return errors

print("the error at the cell midpoints, and how it falls when the step halves:")
for (label, kernel, half, norm) in (
        ('linear', linear_kernel, 1, False),
        ('cubic convolution', cubic_kernel, 2, False),
        ('lanczos 2', lambda t: lanczos(t, 2), 2, True),
        ('lanczos 3', lambda t: lanczos(t, 3), 3, True)):
    errors = decay(kernel, half, norm)
    ratios = [errors[k] / errors[k + 1] for k in range(2, len(errors) - 1)]
    print(f"  {label:20s} " + " ".join(f"{e:.1e}" for e in errors[:4])
          + f"   ratio {np.mean(ratios):.2f}  (8 is a cube, 16 a fourth power)")
print()
print("So Lanczos is not *worse* on smooth data -- it is a different kind of")
print("method, approximating the ideal rather than fitting a polynomial, and on")
print("piecewise smooth data that is exactly what is wanted: no polynomial of")
print("small support can follow an edge without either blurring it or ringing.")
```

## The family, in one table

Every method a grid has, by what it is made of and what it gives. The orders are
the definition's — the degree reproduced plus one — and "reproduces" is which
polynomials come back *exactly*, measured above for the ones a grid has so far.

| method | order | reproduces | continuity | cells per axis | made of |
|---|---|---|---|---|---|
| `('nearest', 0)` | 0 | constants | piecewise constant | 1 | a box |
| `('polynomial', 1)` | 2 | affine | C⁰ | 2 | a triangle |
| `('catmull-rom', 3)` | 3 | quadratics | C¹ | 4 | two cubics |
| `('spline', 2)` | 3 | quadratics | C¹ | 4 | a quadratic B-spline |
| `('spline', 3)` | 4 | cubics | C² | 4 | a cubic B-spline |
| `('lanczos', 2)` | 2 | constants | C¹ | 4 | a windowed sinc |
| `('lanczos', 3)` | 3 | constants | C² | 6 | a windowed sinc |

The last column is the one that explains the rest. The first five are *polynomial*
kernels, and for those the order means exact reproduction at any sampling. The
last two approximate the ideal interpolator, so their orders hold only as the
sampling is refined, and their exact reproduction stops at the constant. Neither
is better in general: on smooth data the higher orders win, and across an edge
the polynomial of small support either blurs it or rings, which is what the
reference's own comparison of a step edge shows.

:::{admonition} Where this comes from
:class: note
The kernel, the window and the normalization are from C. E. Shannon's sampling
theorem on one side and from the Lanczos-window literature on the other; the
treatment this page follows is Pascal Getreuer's, *Linear Methods for Image
Interpolation*, Image Processing On Line **1** (2011), 238–259, sections 10, 11
and 6, [doi:10.5201/ipol.2011.g_lmii](https://doi.org/10.5201/ipol.2011.g_lmii),
which also gives the RMSE comparisons the last paragraph refers to. The
observation that a *symmetric* kernel's odd-order error terms vanish, and so that
its error falls faster than its nominal order, is standard in the Strang–Fix
literature (G. Strang and G. Fix, *A Fourier analysis of the finite element
variational method*, 1973).

**What is derived here.** The window's zeros, the measured weight sums, the
normalization's effect on a constant, the exact-reproduction table and the
convergence measurement are all computed on this page. The kernel itself, and the
fact that it is the standard windowed-sinc approximation, are cited.
:::

:::{seealso}
- [Interpolating a Grid: Splines](grid-spline.md) and [Cubic
  Convolution](grid-cubic.md) for the polynomial kernels this page compares
  against, and whose orders the table reports.
- [Interpolating a Grid: the Boundary](grid-boundary.md) for the rule a
  six-cell stencil needs at the grid's edge.
- [Interpolating a Grid: Nearest and Linear](grid-linear.md) for the index space.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

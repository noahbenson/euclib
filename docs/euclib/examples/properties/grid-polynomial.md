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
# Interpolating a Grid: the Polynomial Fit

Every method on the [family page](grid-lanczos.md) is an *interpolator*: a field
that passes through the data and is at least continuous between the samples.
`('polynomial', k)` is not. It is the monomial least-squares fit the method's
name means everywhere in this library — the same fit `('polynomial', k)` makes on
a segment, a triangle or a tetrahedron — applied to a block of cells around the
position, and it has neither of those properties. This page says what it is,
measures what it does not do, and says when it is nevertheless the right choice.

:::{admonition} What this demonstrates
:class: tip
- The method the vocabulary already implies: a monomial basis of total degree
  $k$, fitted to the cells around the position by least squares.
- How wide the block has to be, from the number of monomials a degree needs.
- The two properties it does **not** have, measured: it does not pass through the
  samples, and it is not continuous across a cell boundary.
- Why a higher degree is not always more accurate here, and where a fit rather
  than an interpolation is what is actually wanted.
:::

## What the method is

A polynomial of total degree $k$ in $d$ dimensions has
$\binom{d + k}{k}$ coefficients: six at degree 2 in two dimensions, ten at degree
3, ten at degree 2 in three, twenty at degree 3. The fit takes every cell in a
cube around the position, writes each cell's coordinates as a multi-index and its
value as the right-hand side, and solves for the coefficients by least squares.
The answer at the position is that polynomial evaluated there.

The block has to hold at least as many cells as there are coefficients, so its
half-width comes from the degree rather than being a free choice — and it grows
fast, by a factor of $2d + 1$ for each step in degree.

```{code-cell}
import numpy as np
import euclib as el

def monomials(k, d):
    """The multi-indices of total degree at most k in d dimensions."""
    if d == 1:
        return [(i,) for i in range(k + 1)]
    return [(i,) + tail for i in range(k + 1)
            for tail in monomials(k - i, d - 1)]

def half_width(k, d):
    """The smallest half-width whose cube determines a degree-k polynomial."""
    need = len(monomials(k, d))
    width = 0
    while (2 * width + 1) ** d < need:
        width += 1
    return width

print("how many coefficients a degree needs, and how wide the block must be:")
for d in (1, 2, 3):
    for k in (2, 3):
        w = half_width(k, d)
        print(f"  {d} dimension{'s' if d > 1 else ' '}, degree {k}:"
              f" {len(monomials(k, d)):2d} coefficients"
              f" -> a block of half-width {w}, {((2*w+1)**d):2d} cells")
print()
print("  (degrees 0 and 1 are the nearest cell and the linear blend, which the")
print("   other pages cover; the fit only differs above them.)")
```

```{code-cell}
def fit(data, at, k, /):
    """The monomial least-squares fit of a block of cells, at a position."""
    d = data.ndim
    w = half_width(k, d)
    base = [int(np.floor(one)) for one in at]
    corners = np.array(np.meshgrid(
        *[np.arange(b - w, b + w + 1) for b in base], indexing='ij'))
    cells = corners.reshape(d, -1).T
    inside = np.array([all(0 <= c[a] < data.shape[a] for a in range(d))
                       for c in cells])
    cells = cells[inside]
    powers = monomials(k, d)
    # The local coordinates are scaled to [-1, 1] before the fit: the monomials
    # of a wide block in cell units span a much larger range than of a narrow
    # one, and an unscaled basis makes the solve needlessly ill-conditioned.
    design = np.array([[np.prod([((c[a] - base[a]) / (w + 0.5)) ** p[a]
                                 for a in range(d)]) for p in powers]
                       for c in cells])
    values = np.array([data[tuple(c)] for c in cells])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    here = [(at[a] - base[a]) / (w + 0.5) for a in range(d)]
    return sum(coefficients[col] * np.prod([here[a] ** p[a] for a in range(d)])
               for (col, p) in enumerate(powers))

count = 40
(ix, iy) = np.meshgrid(np.arange(count, dtype=float),
                       np.arange(count, dtype=float), indexing='ij')
field = lambda x, y: np.sin(x / 6.0) * np.cos(y / 7.0)
grid = field(ix, iy)

print("the fit's error on a smooth field, away from the ends:")
for k in (1, 2, 3):
    worst = 0.0
    for (x, y) in ((20.0, 20.0), (20.5, 20.5), (20.25, 21.0), (19.8, 20.4)):
        worst = max(worst, abs(fit(grid, (x, y), k) - field(x, y)))
    print(f"  degree {k} (a block of half-width {half_width(k, 2)}):"
          f" worst error {worst:.3e}")
```

## What it does not do

Two things, and both are measured rather than asserted.

**It does not pass through the samples.** Every method so far is an
interpolator: at a sample it returns that sample. A least-squares fit does not —
it returns the value its polynomial has there, which is generally not the datum,
and the difference is the point of a fit rather than a flaw in it.

**It is not continuous.** The block shifts by a whole cell when the position
crosses a cell boundary, and the fit is recomputed on a different set of cells,
so the field jumps. Every other method on the ladder is at least C⁰; this one is
piecewise smooth and discontinuous between the pieces.

```{code-cell}
print("at a sample, where an interpolating method returns the datum:")
for k in (1, 2, 3):
    got = fit(grid, (20.0, 20.0), k)
    print(f"  degree {k}: the datum {grid[20, 20]:.12f}, the fit {got:.12f}"
          f"   difference {abs(got - grid[20, 20]):.3e}")
print()
print("and across a cell boundary, which an interpolator's field does not see:")
for k in (1, 2, 3):
    (left, right) = (fit(grid, (19.999, 20.0), k), fit(grid, (20.001, 20.0), k))
    print(f"  degree {k}: {left:.9f} then {right:.9f}"
          f"   a jump of {abs(left - right):.3e}")
print()
print("  (the jump is smaller at degree 3 only because its fit is smoother over")
print("   a wider block; it is a jump all the same, and it does not vanish as")
print("   the position approaches the boundary from either side.)")
```

## Why a higher degree is not always better

A wider block sees more of the field, which is good where the field is genuinely
smooth and bad everywhere else: the fit is pulled by cells that are far away and
may have nothing to do with the position being asked about. That is the
bias-variance trade the fit is making, and in the table above it shows up as
degree 3 being *less* accurate than degree 2 on this field, because degree 3's
block is five cells across where degree 2's is three.

That is not an argument against the method; it is what says when to use it. A fit
is the right tool when the data are noisy — where passing exactly through every
sample is a *defect*, since it reproduces the noise — and the block is the
smoothing window. On clean data, every other method on these pages is better.

```{code-cell}
# The same field with noise, which is where a fit earns its place.
rng = np.random.default_rng(0)
noisy = grid + rng.normal(scale=0.05, size=grid.shape)
probe = [(20.0 + s, 20.0 + t) for s in (0.0, 0.3, 0.7)
         for t in (0.0, 0.4)]
print("against the *clean* field, on data with noise of standard deviation 0.05:")
print(f"  {'method':22s} {'worst error':>12s}")
for (label, got) in (
        ('the noisy data itself',
         [noisy[int(x), int(y)] for (x, y) in probe]),
        ('degree 1 fit', [fit(noisy, p, 1) for p in probe]),
        ('degree 2 fit', [fit(noisy, p, 2) for p in probe]),
        ('degree 3 fit', [fit(noisy, p, 3) for p in probe])):
    worst = max(abs(g - field(*p)) for (g, p) in zip(got, probe))
    print(f"  {label:22s} {worst:12.3e}")
print()
print("At a sample the data carries its own noise; the fit's value there is")
print("pulled toward its neighbours, which is what makes it nearer the truth.")
```

**And the library agrees with this page.** `('polynomial', 2)` and
`('polynomial', 3)` are the fit above, with the block width derived from the
degree rather than given, and the same two properties — no interpolation, no
continuity — hold.

```{code-cell}
carried = el.grid((count, count)).withprop('v', grid)
print("the library's fit against this page's, over a grid of positions:")
for degree in (2, 3):
    xs = np.linspace(19.0, 39.0, 60)
    ys = np.linspace(19.0, 39.0, 60)
    got = np.ravel(np.asarray(carried.prop(
        'v', at=carried.topo.Loc(sx=xs, sy=ys), interp=('polynomial', degree))))
    want = np.array([fit(grid, (x, y), degree) for (x, y) in zip(xs, ys)])
    print(f"  degree {degree}: worst difference {np.abs(got - want).max():.3e}")
print()
print("and the two properties, through the library:")
at = carried.topo.Loc(sx=np.array([20.0]), sy=np.array([20.0]))
got = float(np.ravel(np.asarray(carried.prop(
    'v', at=at, interp=('polynomial', 2))))[0])
print(f"  at a sample: the datum {grid[20, 20]:.9f}, the fit {got:.9f}")
left = carried.topo.Loc(sx=np.array([20.0 - 1e-6]), sy=np.array([20.0]))
right = carried.topo.Loc(sx=np.array([20.0 + 1e-6]), sy=np.array([20.0]))
(a, b) = (float(np.ravel(np.asarray(carried.prop('v', at=one,
                                                 interp=('polynomial', 2))))[0])
          for one in (left, right))
print(f"  across a boundary: {a:.9f} then {b:.9f}, a jump of {abs(a - b):.3e}")
```

:::{admonition} Where this comes from
:class: note
The fit is the monomial least-squares fit that `('polynomial', k)` names
throughout this library; see [Interpolating a Property within a
Geometry](interpolation.md) for the vocabulary and
[Bézier interpolation on a triangle](bezier-triangle.md) for the same fit on an
element, where the data over-determine a quadratic and the least-norm solution is
taken. On a grid the fit is the local one of the moving-least-squares literature
(W. S. Cleveland, *Robust locally weighted regression and smoothing scatterplots*,
Journal of the American Statistical Association **74** (1979), 829–836,
[doi:10.1080/01621459.1979.10481038](https://doi.org/10.1080/01621459.1979.10481038)),
without the weighting that paper's name carries.

**What is derived here.** The block width from the coefficient count, and the
three measurements — the error on a smooth field, the failure to interpolate, the
discontinuity at a cell boundary — are computed on this page.
:::

:::{seealso}
- [Interpolating a Grid: Lanczos](grid-lanczos.md) for the family's table, and
  for what the interpolating methods do differently.
- [Interpolating a Grid: Nearest and Linear](grid-linear.md) for the index space
  and for the two lowest degrees, which are separate methods here.
- [Interpolating a Property within a Geometry](interpolation.md) for what
  `'polynomial'` means on an element.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

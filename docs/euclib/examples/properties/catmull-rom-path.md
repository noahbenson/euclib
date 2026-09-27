---
jupytext:
  cell_metadata_filter: -all
  formats:
    md:myst:
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
# Catmull–Rom Interpolation along a Path

A path's cubic — `('polynomial', 3)` or `('bezier', 3)` — is a cubic Hermite
curve: it needs a *slope* at each vertex, and a property that does not carry one
has it [estimated](../properties/interpolation.md) from the mesh. Catmull–Rom is
the other way to get those slopes, and the classical one: **the slope at a vertex
is the difference of its two neighbours**, halved. It needs no solve, no
gradient, and no data beyond the values either side of the vertex — and it is the
same kernel, sample for sample, as [`('catmull-rom', 3)` on a
grid](grid-cubic.md).

:::{admonition} What this demonstrates
:class: tip
- The cardinal cubic: the same Hermite curve the other cubic methods build, with
  the slopes taken from the neighbours rather than from the data or a fit.
- That the resulting weights are **identical** to the grid's cubic convolution
  kernel — the two methods are one method in two settings.
- What the rule reproduces: quadratics exactly, cubics not, which is the
  approximation order 3 the grid page derives.
- What it does *not* use: a supplied gradient. The slopes come from the values or
  not at all.
:::

## The rule

Write $f_i$ for the value at vertex $i$ and $m_i$ for the slope wanted there. The
Hermite cubic on the segment from $i$ to $i + 1$ is determined by the two values
and the two slopes, so the whole curve is a cubic per segment sharing a slope at
every interior vertex — and that is what makes it C¹. Catmull–Rom's choice is

$$ m_i = \frac{f_{i+1} - f_{i-1}}{2} \qquad \text{for an interior vertex}, $$

a *central* difference. It is a rule rather than a fit: no neighbourhood has to be
grown, nothing is solved, and the slope at a vertex depends on its two neighbours
and nothing else. The ends of a path have only one neighbour each, and take the
one-sided difference.

```{code-cell}
import numpy as np
import euclib as el

def tangents(f, /):
    """The Catmull-Rom slopes of values at the vertices of a path."""
    n = f.shape[0]
    m = np.zeros_like(f)
    m[1:-1] = (f[2:] - f[:-2]) / 2.0
    m[0] = f[1] - f[0]
    m[-1] = f[-1] - f[-2]
    return m

def hermite(f, at, /):
    """The cubic through the values with those slopes, at a position in cell
    units along the path."""
    i = int(np.floor(at))
    s = at - i
    m = tangents(f)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * f[i] + h10 * m[i] + h01 * f[i + 1] + h11 * m[i + 1]

print("the slope the rule gives at each interior vertex:")
f = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
print("  the values   ", f.astype(int))
print("  the slopes   ", tangents(f).astype(int))
print("  ...which for this quadratic are exactly its derivative 2i, at every")
print("  interior vertex and not at the ends:")
print("  the exact     ", [2.0 * i for i in range(f.shape[0])])
```

## It is the grid's kernel

The grid page derives a cubic convolution kernel from eight conditions and says,
in passing, that it is the same one-dimensional kernel this method uses. That is
checkable rather than incidental. The weights a cubic puts on the four samples
around a position are linear in the four values; working out what the tangent
rule's Hermite curve gives them, and comparing with the kernel's own values at
the same offsets, shows the two agree exactly.

```{code-cell}
def weights_of_the_rule(s, /):
    """The weight the four samples get, at a position s past the second."""
    out = []
    for j in range(4):
        trial = [0.0] * 4
        trial[j] = 1.0
        (below, first, second, above) = trial
        m0 = (second - below) / 2.0
        m1 = (above - first) / 2.0
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out.append(h00 * first + h10 * m0 + h01 * second + h11 * m1)
    return np.array(out)

def cubic_convolution(t, alpha=-0.5):
    t = np.abs(np.asarray(t, dtype=float))
    return (np.where(t <= 1.0, (alpha + 2) * t ** 3 - (alpha + 3) * t ** 2 + 1.0, 0.0)
            + np.where((t > 1.0) & (t < 2.0),
                       alpha * t ** 3 - 5 * alpha * t ** 2 + 8 * alpha * t
                       - 4 * alpha, 0.0))

print("the weights on the four samples around a position, two ways:")
print(f"  {'position':>9s} | {'from the tangent rule':>34s} | {'from the grid kernel':>34s}")
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    by_rule = weights_of_the_rule(s)
    by_kernel = np.array([float(cubic_convolution(s - k)) for k in (-1, 0, 1, 2)])
    print(f"  {s:9.2f} | " + " ".join(f"{v:+.6f}" for v in by_rule)
          + " | " + " ".join(f"{v:+.6f}" for v in by_kernel))
print(f"  ...worst difference over a fine sweep:"
      f" {max(np.abs(weights_of_the_rule(s) - np.array([float(cubic_convolution(s - k)) for k in (-1, 0, 1, 2)])).max() for s in np.linspace(0.0, 1.0, 200)):.3e}")
```

## What it reproduces

The rule is a central difference, which is exact for a quadratic and not for a
cubic: at a vertex of a cubic it gives $3i + i^3$ where the derivative is $3i^2$.
So the curve through a quadratic's values is that quadratic, exactly, and through
a cubic's it is a cubic that misses — which is the approximation order 3 the grid
page derives, seen from the other side.

```{code-cell}
def worst_on(field, /, count=12):
    data = np.array([field(t) for t in range(count)], dtype=float)
    probe = np.linspace(2.0, count - 3.0, 200)
    got = np.array([hermite(data, t) for t in probe])
    return np.abs(got - field(probe)).max()

print("the curve against the fields it is built from:")
for (label, field) in (('a constant', lambda t: 0.0 * t + 3.0),
                       ('an affine', lambda t: 0.4 * t + 1.1),
                       ('a quadratic', lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1),
                       ('a cubic', lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1)):
    print(f"  {label:12s} worst error {worst_on(field):.3e}")
print()
print("An affine field is reproduced because the rule is a difference and a")
print("difference of a linear sequence is its slope; a quadratic because the")
print("central difference of a quadratic is exact; a cubic is where it runs out.")
```

## Against the other way of getting slopes

A property that carries no gradient can also have one [estimated
](../properties/interpolation.md), and the estimate is a polynomial fit over a
neighbourhood that grows until it can determine the order. It is exact on every
polynomial the order holds, where the Catmull–Rom rule is exact only on the
quadratic — and it costs a solve per vertex, where the rule costs a subtraction.
The two are the fast local choice and the accurate global one.

```{code-cell}
count = 12
path = el.segpath(np.stack([np.arange(count, dtype=float),
                            np.zeros(count)]),
                  np.array([np.arange(count - 1), np.arange(1, count)]))
cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
carried = path.withprop('f', np.array([[cubic(t) for t in range(count)]]))
probe = np.linspace(2.0, count - 3.0, 60)

def along(method, /):
    out = []
    for t in probe:
        i = int(np.floor(t))
        at = path.topo.Loc(index=np.array([i]),
                           weight=np.array([[1.0 - (t - i)]]))
        out.append(float(np.ravel(np.asarray(
            carried.prop('f', at=at, interp=method)))[0]))
    return np.array(out)

print("the same cubic values down a path, by each method that builds a cubic:")
for method in (('bezier', 3), ('polynomial', 3)):
    print(f"  {str(method):18s} worst {np.abs(along(method) - cubic(probe)).max():.3e}"
          f"   (the slopes come from the estimator)")
print(f"  {'(catmull-rom rule)':18s} worst "
      f"{np.abs(np.array([hermite(np.array([cubic(t) for t in range(count)]), t) for t in probe]) - cubic(probe)).max():.3e}"
      f"   (the slopes come from the neighbours)")
```

**And the library agrees with this page.** `('catmull-rom', 3)` is the rule
above, reading each vertex's neighbours from the path's own connectivity, and the
curve it gives is this page's to the arithmetic's precision.

```{code-cell}
count = 12
path = el.segpath(np.stack([np.arange(count, dtype=float), np.zeros(count)]),
                  np.array([np.arange(count - 1), np.arange(1, count)]))
probe = np.linspace(2.0, count - 3.0, 60)

for (label, field) in (('an affine', lambda t: 0.4 * t + 1.1),
                       ('a quadratic', lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1),
                       ('a cubic', lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1)):
    values = np.array([field(t) for t in range(count)], dtype=float)
    carried = path.withprop('f', values[None])
    mine = np.array([hermite(values, t) for t in probe])
    library = []
    for t in probe:
        i = int(np.floor(t))
        at = path.topo.Loc(index=np.array([i]),
                           weight=np.array([[1.0 - (t - i)]]))
        library.append(float(np.ravel(np.asarray(
            carried.prop('f', at=at, interp=('catmull-rom', 3))))[0]))
    library = np.array(library)
    print(f"  {label:12s} against the field {np.abs(library - field(probe)).max():.3e}"
          f"   against this page {np.abs(library - mine).max():.3e}")
```

:::{admonition} Where this comes from
:class: note
The rule and the curve are E. Catmull and R. Rom, *A class of local interpolating
splines*, in *Computer Aided Geometric Design* (1974), 317–326. The identity with
the cubic convolution kernel is R. Keys', *Cubic Convolution Interpolation for
Digital Image Processing*, IEEE Transactions on Acoustics, Speech, and Signal
Processing **29**(6) (1981), 1153–1160,
[doi:10.1109/TASSP.1981.1163711](https://doi.org/10.1109/TASSP.1981.1163711) —
Keys derives the kernel as a cubic convolution and, at the parameter this library
uses, it is Catmull and Rom's curve; the grid page
[derives that parameter](grid-cubic.md) from the moment conditions. The
parametrizations of the curve for *non-uniform* vertices, and the argument for the
centripetal one, are C. Yuksel, S. Schaefer and J. Keyser, *Parameterization and
applications of Catmull-Rom curves*, Computer-Aided Design **43**(8) (2011),
847–855,
[doi:10.1016/j.cad.2011.04.008](https://doi.org/10.1016/j.cad.2011.04.008).
`euclib` interpolates within a segment's own coordinates, so its vertices are
evenly spaced by construction and the uniform rule is the one that applies.

**What is derived here.** The weights table, the agreement with the grid kernel,
and the reproduction table are computed on this page.
:::

:::{seealso}
- [Interpolating a Grid: Cubic Convolution](grid-cubic.md) for the same kernel on
  a grid, and the derivation of its parameter.
- [Interpolating a Property within a Geometry](interpolation.md) for the
  estimated gradient this method is the alternative to.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

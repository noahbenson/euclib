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
# Interpolating a Grid: Cubic Convolution

[Linear interpolation](grid-linear.md) is C⁰: the field it produces has a kink at
every sample. What an image resampler wants instead is a field that is *smooth*
where the data is, and the cheapest such method is a **cubic convolution** — four
cubic pieces, supported on four cells, whose weights are the same wherever they
are used. This page derives the kernel from the properties it should have, and
then settles the one free parameter in it by asking what the kernel should
*reproduce*.

:::{admonition} What this demonstrates
:class: tip
- The eight conditions that pin a two-piece cubic kernel down to **one** free
  parameter, and what that parameter means: the slope at a knot.
- That the choice of that parameter is not a matter of taste — only one value —
  makes the kernel reproduce quadratics, and that the *degrees of the pieces* and
  the *order of the method* are different things, one apart.
- That the four weights are the same at every position, which is what makes the
  method a *convolution* and its implementation a change of basis rather than a
  fit.
- Where the [boundary rule](grid-boundary.md) enters, now that the stencil is
  four cells wide instead of two.
:::

## The shape of a kernel, and what pins it down

Write $K(t)$ for the weight a sample $t$ away contributes. Three things are
wanted of it, and they are what give it a shape:

1. **It interpolates.** The value at a sample must be that sample:
   $K(0) = 1$ and $K(n) = 0$ for every other integer $n$.
2. **It does not reach far.** $K$ vanishes outside $(-2, 2)$ — so a position draws
   on at most four samples along each axis.
3. **It is smooth.** $K$ and $K'$ are continuous, including at the knots where
   the pieces meet.

Being even, $K$ is determined by its values on $t \ge 0$, where the conditions
leave two cubic pieces — one on $[0, 1]$ and one on $[1, 2]$ — with eight
coefficients between them. Eight conditions fix them:

$$ K(0) = 1, \quad K'(0) = 0, \quad K(1) = 0, \quad
   K(2) = K'(2) = 0, \quad\text{and } K \text{ and } K' \text{ continuous at } 1 . $$

That is eight conditions and eight unknowns, so there ought to be no freedom at
all — except that the continuity at the knot names *the same* slope on both
sides, and nothing has yet said what that slope is. Calling it $\alpha$, the
system has exactly one degree of freedom, and $\alpha$ *is* that degree: the
slope of the kernel at a sample.

```{code-cell}
import numpy as np
import euclib as el

def pieces(alpha, /):
    """The two cubic pieces of the kernel on [0,1] and [1,2], from the slope at
    the knot.  Eight conditions on eight coefficients, one of them a parameter."""
    # a0 + a1 t + a2 t^2 + a3 t^3 on [0,1] and b0 + ... on [1,2]
    rows = []
    rhs = []
    def add(row, value):
        rows.append(row); rhs.append(value)
    def coeffs(offset, t, order=0):
        # the row for the `order`-th derivative at t of that piece
        row = np.zeros(8)
        for (power, c) in enumerate((1.0, t, t ** 2, t ** 3)):
            if power >= order:
                factor = 1.0
                for k in range(order):
                    factor *= (power - k)
                row[offset + power] = factor * t ** (power - order)
        return row
    add(coeffs(0, 0.0), 1.0)            # K(0) = 1
    add(coeffs(0, 0.0, 1), 0.0)         # K'(0) = 0, by symmetry
    add(coeffs(0, 1.0), 0.0)            # K(1) = 0
    add(coeffs(0, 1.0, 1), alpha)       # K'(1) = alpha, the free one
    add(coeffs(4, 1.0), 0.0)            # the second piece likewise
    add(coeffs(4, 1.0, 1), alpha)
    add(coeffs(4, 2.0), 0.0)            # K(2) = K'(2) = 0
    add(coeffs(4, 2.0, 1), 0.0)
    return np.linalg.solve(np.array(rows), np.array(rhs))

def kernel(t, alpha=-0.5, /):
    """The even kernel, from its two pieces."""
    t = np.abs(np.asarray(t, dtype=float))
    flat = t.reshape(-1)
    got = np.zeros(flat.shape)
    (a, b) = (pieces(alpha)[:4], pieces(alpha)[4:])
    low = flat <= 1.0
    for (mask, (offset, c)) in ((low, (0, a)), (~low & (flat < 2.0), (4, b))):
        here = flat[mask]
        if here.size:
            got[mask] = sum(c[k] * here ** k for k in range(4))
    return got.reshape(t.shape)

alpha = -0.5
solved = pieces(alpha)
print("solving for the pieces with the knot slope set to alpha:")
print(f"  on [0,1]: {np.round(solved[:4], 6)}")
print(f"  on [1,2]: {np.round(solved[4:], 6)}")
print()
print("and the closed form that solve is equal to, from the reference:")
print(f"  on [0,1]: (a+2)t^3 - (a+3)t^2 + 1 with a = {alpha}"
      f"  ->  {np.round([1.0, 0.0, -(alpha + 3), alpha + 2], 6)}")
print(f"  on [1,2]: at^3 - 5at^2 + 8at - 4a      with a = {alpha}"
      f"  ->  {np.round([-4 * alpha, 8 * alpha, -5 * alpha, alpha], 6)}")
```

## Which slope, and why there is only one answer

The parameter is not a taste. What an interpolator of approximation order $J$
must reproduce is every polynomial up to degree $J - 1$ --- one less than the
order, since the order counts the derivative that still fails to converge --- and
a convolution reproduces them exactly when its **moments** cancel:

$$ \sum_{n} K(t - n) = 1, \qquad
   \sum_{n} (t - n)^{j}\,K(t - n) = 0, \quad j = 1, \dots, J-1 . $$

The first is the kernel summing to one, which is what makes a constant field come
back; the rest are the kernel's moments vanishing, which is what makes the
*slope* and then the *curvature* of the data come back. Two moments vanish for
this kernel when $\alpha = -\tfrac12$, which is what makes its approximation order
3 and reproduces the quadratics. Curvature is where the kernel runs out: a
two-piece cubic cannot be asked for the third moment as well, however $\alpha$ is
chosen, which is as it should be --- a piecewise cubic whose order equalled its
degree would be the whole cubic and not a piecewise one.

Any $\alpha$ satisfies the first condition and the affine one, which is why every
$\alpha$ reproduces affine data, and the check below finds the value that does
more by measurement rather than by assertion: for a spread of values of
$\alpha$, how well does the method reproduce a quadratic?

```{code-cell}
def convolve(samples, t, alpha, /):
    """The cubic convolution of samples on integer positions, at positions t.

    The stencil is the four samples from -1 to +2 around each position, which is
    the kernel's support; samples are taken to be zero outside the array, so this
    is only meaningful for positions a clear two cells from either end.
    """
    count = samples.shape[0]
    out = np.zeros(np.asarray(t).shape)
    flat = np.asarray(t, dtype=float).reshape(-1)
    for (i, one) in enumerate(flat):
        base = int(np.floor(one))
        total = 0.0
        for (offset, n) in enumerate(range(base - 1, base + 3)):
            if 0 <= n < count:
                total += samples[n] * kernel(one - n, alpha)
        out.reshape(-1)[i] = total
    return out

# A quadratic and a cubic, sampled at the integers over a wide enough window,
# which is what tells the achievable order apart from the degree of the pieces.
quadratic = lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1
cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
wide = np.arange(30.0)
probe = np.linspace(8.0, 20.0, 200)          # well away from both ends

print("how well each choice of the knot slope reproduces the polynomials")
print("the approximation order says it should:")
print("   alpha       quadratic      cubic")
for alpha in (-1.0, -0.75, -0.5, -0.25, 0.0):
    got = convolve(quadratic(wide), probe, alpha)
    worst = np.abs(got - quadratic(probe)).max()
    also = np.abs(convolve(cubic(wide), probe, alpha) - cubic(probe)).max()
    mark = "" if worst < 1e-12 else "   <- not exact"
    print(f"  {alpha:6.2f}     {worst:.3e}    {also:.3e}{mark}")
print()
print("So alpha = -1/2 is the one, and it is the same value in every dimension,")
print("since a separable kernel's moments are the products of the one-dimensional")
print("ones. Note what the last column says: even at that value the cubic does")
print("not come back exactly, because a piecewise cubic reproduces polynomials")
print("one degree below its own pieces. Its approximation order is 3, and not 4.")
```

**It is the Catmull–Rom kernel.** With that slope fixed, the kernel is the one
Catmull and Rom's cardinal cubic gives, and the one Keys' cubic convolution
gives; in `euclib` the method is named `'catmull-rom'`, because the path method
is the same one-dimensional kernel and only the tensor product differs.

**And it interpolates, smoothly.** The conditions the kernel was built from are
worth checking on the result rather than the construction: at a sample the weight
is one and its neighbours' weights are all zero, and the derivative of the field
is continuous at the samples, which is what makes the field C¹ rather than C⁰.

```{code-cell}
print("the kernel at the four stencil offsets, and its derivative:")
for t in (0.0, 1.0, 2.0, 3.0):
    print(f"  K({t:.0f}) = {float(kernel(t)):+.12f}"
          f"    K'({t:.0f}) = {float((kernel(t + 1e-6) - kernel(t - 1e-6)) / 2e-6):+.6f}")
print()
print("the weights over the stencil at a position, which sum to one:")
for off in (0.0, 0.25, 0.5, 0.75):
    weights = [float(kernel(off - n)) for n in (-1, 0, 1, 2)]
    print(f"  at {off:.2f}: " + "  ".join(f"{w:+.4f}" for w in weights)
          + f"   sum {sum(weights):.12f}")
```

## Where the boundary comes in

A stencil four cells wide reaches two cells past a position, so a position
anywhere in the first or last **one and a half cells** of an axis has a stencil
that leaves the grid — six times the room the linear method's two-cell stencil
gave the boundary rule. That is what
[the boundary page](grid-boundary.md) is for, and it is worth seeing here that
the choice is not academic: the same position and the same data give different
answers under the three extensions.

```{code-cell}
def convolve_bordered(samples, t, border, /):
    """Cubic convolution with the grid's edge continued by a rule."""
    count = samples.shape[0]

    def fold(n):
        if border == 'constant':
            return int(np.clip(n, 0, count - 1))
        if border == 'half-symmetric':
            period = 2 * count
            return int(min(n % period, (period - 1 - n) % period))
        period = 2 * count - 2
        return int(min(n % period, (period - n) % period))

    total = 0.0
    for n in range(int(np.floor(t)) - 1, int(np.floor(t)) + 3):
        total += samples[fold(n)] * kernel(t - n)
    return total

ramp = np.array([1., 2., 3., 4., 5., 6.])
print("a ramp on six samples, at positions just inside the last sample:")
print("  position |" + "|".join(f"{b:^17}"
                               for b in ('constant', 'half-symmetric',
                                         'whole-symmetric')))
for t in (5.0, 5.25, 5.5):
    print(f"  {t:8.2f} |" + "|".join(
        f"{convolve_bordered(ramp, t, b):^17.4f}"
        for b in ('constant', 'half-symmetric', 'whole-symmetric')))
print()
print("Inside the samples the three agree and all give the ramp; past the last")
print("one they part company, and by more than the linear method's rules did.")
```

**And the library agrees with this page.** The construction above is written out
here; the last check is that `euclib`'s own cubic convolution gives the same
field, and that it is the same method the page's weights produce.

```{code-cell}
wide = np.arange(40.0)
probe = np.linspace(10.0, 30.0, 300)
carried = el.grid((wide.shape[0],)).withprop('v', quadratic(wide))
got = np.ravel(np.asarray(carried.prop(
    'v', at=carried.topo.Loc(sx=probe), interp=('catmull-rom', 3))))
print(f"  the library against this page's convolution, over {probe.size}"
      f" positions: {np.abs(got - convolve(quadratic(wide), probe, -0.5)).max():.3e}")
print(f"  the library against the quadratic itself:"
      f" {np.abs(got - quadratic(probe)).max():.3e}")
plain = np.ravel(np.asarray(carried.prop(
    'v', at=carried.topo.Loc(sx=probe), interp=('polynomial', 1))))
print(f"  ...and linear interpolation of the same data, for scale:"
      f" {np.abs(plain - quadratic(probe)).max():.3e}")
```

:::{admonition} Where this comes from
:class: note
The kernel is R. Keys', *Cubic Convolution Interpolation for Digital Image
Processing*, IEEE Transactions on Acoustics, Speech, and Signal Processing
**29**(6) (1981), 1153–1160,
[doi:10.1109/TASSP.1981.1163711](https://doi.org/10.1109/TASSP.1981.1163711),
whose free parameter and whose closed forms for the two pieces this page uses.
The same kernel is the cardinal cubic spline of E. Catmull and R. Rom, *A
class of local interpolating splines*, in *Computer Aided Geometric Design*
(1974), 317–326 — which is where `euclib`'s method name comes from, and which is
also the path method's kernel. The moment conditions for polynomial reproduction,
and the observation that the parameter is only determined by them at the cubic
order, are the Strang–Fix conditions in the form G. Strang and G. Fix gave them,
*A Fourier analysis of the finite element variational method* (1973); the
treatment this page follows is Getreuer's, *Linear Methods for Image
Interpolation*, Image Processing On Line **1** (2011), 238–259, section 9,
[doi:10.5201/ipol.2011.g_lmii](https://doi.org/10.5201/ipol.2011.g_lmii).

**What is derived here.** The eight-condition solve for the pieces, and the
measurement of which slope reproduces cubics, are done on this page. The closed
forms the solve is compared against, and the fact that they are the classical
kernel, are the cited ones.
:::

:::{seealso}
- [Interpolating a Grid: the Boundary](grid-boundary.md) for the rule a stencil
  that leaves the grid is extended by.
- [Interpolating a Grid: Nearest and Linear](grid-linear.md) for the methods this
  one improves on, and for the index space it works in.
- [Interpolating a Property within a Geometry](interpolation.md) for the metadata
  that chooses an order.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

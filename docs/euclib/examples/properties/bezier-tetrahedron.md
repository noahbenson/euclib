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
# Bézier Interpolation on a Tetrahedron

[Bézier interpolation on a triangle](bezier-triangle.md) derives the rules that
turn a triangle's vertex values and derivatives into a smooth field over it. The
same construction works on a tetrahedron, one dimension up, and this page is
about the one thing that is genuinely new there: a cubic's field has values on
the *faces*, not only along the edges, and those four values are what the
construction has to settle. The expectation is that a reader has the triangle
page in hand; where a rule is the same one dimension up, this page says so and
does not derive it again.

:::{admonition} What this demonstrates
:class: tip
- The Bézier patch of degree 2 and degree 3 on a tetrahedron, and where its
  control values sit: **10** of them at degree 2 and **20** at degree 3.
- Why a tetrahedron's cubic needs a rule for its faces, and what that rule is.
- The two checks again — reproduction of quadratics, and agreement across a
  shared face — this time for two tetrahedra that meet along a triangle.
- What the construction does *not* do: a general cubic is not reproduced, and
  cannot be, from the corners alone.
:::

## One more coordinate

A point inside a tetrahedron is named by four numbers that add up to one, its
**barycentric coordinates** $w = (w_0, w_1, w_2, w_3)$, each of which is 1 at one
corner and 0 at the others. The Bernstein polynomials of the triangle page
generalize by carrying one more exponent:

$$ B^n_{\,ijkl}(w) = \frac{n!}{i!\,j!\,k!\,l!}\, w_0^i w_1^j w_2^k w_3^l ,
   \qquad i + j + k + l = n , $$

and a **tetrahedral Bézier patch** of degree $n$ is $b(w) = \sum B^n_{\,ijkl}(w)\,
c_{ijkl}$. There are $\binom{n+3}{3}$ control values — **10** at degree 2 and
**20** at degree 3 — and, as on a triangle, the multinomial factor is what makes
them sum to one, so the patch is a weighted average of the control values and
interpolates whichever of them sit at the corners.

Where they sit is the whole story of what has to be computed. At degree 2 the
ten are the four corners and the six edge midpoints. At degree 3 the twenty are
the four corners, **two on each of the six edges**, and **one on each of the
four faces**:

| control | degree 2 | degree 3 | whose data |
|---|---|---|---|
| corners | $(2,0,0,0)$ etc. | $(3,0,0,0)$ etc. | the corner's own value |
| edge midpoints | $(1,1,0,0)$ etc. | — | the two ends |
| edge third-points | — | $(2,1,0,0)$ and $(1,2,0,0)$, etc. | the two ends' slopes |
| faces | — | $(1,1,1,0)$ etc. | the face's three edges |
| interior | — | — | nobody yet |

The last row is worth a sentence. A cubic has no control value *inside* a
tetrahedron; the first that does is a quartic's, at $(1,1,1,1)$. So a cubic
tetrahedral patch is settled by its faces and leaves the volume between them
alone — which is exactly where the naive hope that the corners determine
everything runs out.

## The rules, one dimension up

**The corners hold their own values**, $c_{n000} = v_0$ and so on, as on a
triangle.

**An edge's control values come from that edge's two ends**, by the same rules
the triangle page derives for its edges — the edge is a one-dimensional object
either way, and nothing about the tetrahedron changes it:

$$ b_1 = \frac{v_a + v_b}{2} + \frac{d_a - d_b}{4}
   \quad\text{(degree 2)}, \qquad
   b = v_a + \frac{d_a}{3},\; v_b - \frac{d_b}{3}
   \quad\text{(degree 3)}, $$

where $d_a$ and $d_b$ are the gradients' components along the edge at its two
ends. The degree-2 value is the *reflection* of the edge's midpoint value about
the endpoints' average, not the midpoint value; the triangle page shows why, and
the reason does not change here.

**A degree-3 face's value is the average of that face's three edges' degree-2
control values.** This is the new rule, and it is the triangle's interior rule
applied to a face: the face is a triangle, its three edges are edges of the
tetrahedron, and the value that goes at $(1,1,1,0)$ is the mean of the three
reflected midpoint values that degree elevation of a quadratic asks for. There
are four faces and four such values, and with them the twenty are complete.

The construction's point, in both cases, is *where each value comes from*. Every
control value on a face of the tetrahedron — its three corner values, its six
edge values, and its face value — is computed from that face's three corners'
values and gradients and from nothing else. Two tetrahedra that share a face
therefore compute the same numbers on it, and their patches agree there. That is
the property the section on continuity below checks, and it is why the
construction goes face by face rather than fitting the solid as a whole.

## A worked example: two tetrahedra sharing a face

The mesh below is two tetrahedra that meet along the triangle $(1, 2, 3)$: the
first is the corner tetrahedron on the origin and the three axes, and the second
is the one beyond the shared face, reaching to $(1, 1, 1)$. Between them they
fill the cube. Each corner is given a position, a value, and a gradient, and the
values are those of a quadratic with cross terms,

$$ f(x, y, z) = 0.4x^2 + 0.3y^2 + 0.2z^2 + 0.25xy - 0.15yz + 0.5x , $$

so that the interpolation can be compared against a formula rather than against
intuition.

```{code-cell}
from itertools import combinations
from math import factorial

import numpy as np
import euclib as el

# The five corners, and the two tetrahedra as corner indices.
coords = np.array([[0., 1., 0., 0., 1.],
                   [0., 0., 1., 0., 1.],
                   [0., 0., 0., 1., 1.]])
tetrahedra = [(0, 1, 2, 3), (1, 2, 3, 4)]

def f(x, y, z):
    '''The field the corners carry, and the answer to compare against.'''
    return (0.4 * x ** 2 + 0.3 * y ** 2 + 0.2 * z ** 2
            + 0.25 * x * y - 0.15 * y * z + 0.5 * x)

def df(x, y, z):
    '''Its exact gradient, which is the data the corners are given.'''
    return np.array([0.8 * x + 0.25 * y + 0.5,
                     0.6 * y + 0.25 * x - 0.15 * z,
                     0.4 * z - 0.15 * y])
```

The construction itself, exactly as described above. It is written out here
rather than called from `euclib`, so that the page is a derivation and not a
demonstration; the library's own answer is compared with it at the end.

```{code-cell}
def powers(order, parts=4):
    '''The multi-indices of a degree-order simplex on ``parts`` corners.'''
    if parts == 1:
        return [(order,)]
    return [(i,) + tail for i in range(order + 1)
            for tail in powers(order - i, parts - 1)]

def multinomial(power):
    '''The number of ways one multi-index's exponents can be arranged.'''
    count = factorial(sum(power))
    for entry in power:
        count //= factorial(entry)
    return count

def reflected_middle(ends, slopes):
    '''A degree-2 edge's middle control value, from its ends and slopes.'''
    (a, b) = (ends[0], ends[1])
    (da, db) = (slopes[0], slopes[1])
    return (a + b) / 2.0 + (da - db) / 4.0

def control_values(order, tet):
    '''The Bezier control values of one tetrahedron, from its corners' data.'''
    ends = {c: coords[:, c] for c in tet}
    values = {c: f(*coords[:, c]) for c in tet}
    slopes = {c: df(*coords[:, c]) for c in tet}
    index = {c: i for (i, c) in enumerate(tet)}
    net = {}

    def along(one, two):
        '''The slopes at two corners, along the edge between them.'''
        step = ends[two] - ends[one]
        return (slopes[one] @ step, slopes[two] @ step)

    # The corners hold their own values.
    for (c, slot) in index.items():
        net[tuple(order if i == slot else 0 for i in range(4))] = values[c]
    # Each edge holds the one-dimensional fit of its two ends.
    for (a, b) in combinations(tet, 2):
        (i, j) = (index[a], index[b])
        (da, db) = along(a, b)
        near_a = tuple(order - 1 if k == i else (1 if k == j else 0)
                       for k in range(4))
        if order == 2:
            net[near_a] = reflected_middle((values[a], values[b]), (da, db))
        else:
            near_b = tuple(1 if k == i else (order - 1 if k == j else 0)
                           for k in range(4))
            net[near_a] = values[a] + da / 3.0
            net[near_b] = values[b] - db / 3.0
    # A cubic's four remaining values sit one on each face, at the average of
    # that face's three edges' *degree-2* control values.
    if order == 3:
        for face in combinations(range(4), 3):
            middle = []
            for (a, b) in ((face[0], face[1]), (face[1], face[2]),
                           (face[2], face[0])):
                middle.append(reflected_middle((values[tet[a]], values[tet[b]]),
                                               along(tet[a], tet[b])))
            net[tuple(1 if c in face else 0 for c in range(4))] = (
                sum(middle) / 3.0)
    return net

def evaluate(order, net, w):
    '''The Bernstein sum: each control value weighted by its basis polynomial.'''
    total = 0.0
    for power in powers(order):
        basis = multinomial(power)
        for (entry, weight) in zip(power, w):
            basis *= weight ** entry
        total += basis * net[power]
    return total
```

Printing the values of one tetrahedron at degree 3 shows the shape of the
answer: four corners, two per edge, one per face, and nothing in the middle.

```{code-cell}
def describe(power):
    '''Which part of the tetrahedron a multi-index names.'''
    nonzero = [i for (i, entry) in enumerate(power) if entry]
    if len(nonzero) == 1:
        return f"corner {nonzero[0]}"
    if len(nonzero) == 2:
        return f"edge {nonzero[0]}-{nonzero[1]}"
    return f"face {nonzero[0]}-{nonzero[1]}-{nonzero[2]}"

net = control_values(3, tetrahedra[0])
print(f"the first tetrahedron at degree 3: {len(net)} control values")
for (power, value) in net.items():
    print(f"  {power}  {describe(power):<12} {value:+.6f}")
```

## How we know the answer is right

The two checks from the triangle page apply unchanged, and both are arithmetic
rather than opinion.

**It reproduces quadratics.** The corners give four values and four gradients —
sixteen conditions — which is more than a quadratic's ten control values need,
and fewer than a cubic's twenty. So the patch must equal $f$ exactly when $f$ is
a quadratic, and it cannot in general when $f$ is a cubic. That asymmetry is not
a defect of the construction; it is what the data can and cannot say.

```{code-cell}
def barycentric(tet, point):
    '''The barycentric coordinates of a point within a tetrahedron.'''
    matrix = np.stack([coords[:, c] for c in tet], axis=1)
    return np.linalg.solve(np.vstack([matrix, np.ones((1, 4))]),
                           np.append(point, 1.0))

rng = np.random.default_rng(0)
print("difference between the patch and the quadratic it was built from:")
for order in (2, 3):
    worst = 0.0
    for _ in range(4000):
        w = rng.uniform(size=4)
        w /= w.sum()
        tet = tetrahedra[rng.integers(2)]
        point = coords[:, list(tet)] @ w
        here = barycentric(tet, point)
        if (here < -1e-12).any():
            continue
        got = evaluate(order, control_values(order, tet), here)
        worst = max(worst, abs(got - f(*point)))
    print(f"  degree {order}: worst difference over 4000 random points"
          f" = {worst:.3e}")

# ...and the same test for a cubic, which the corners cannot determine.
cubic = lambda x, y, z: x ** 3 + 0.5 * y * z
dcubic = lambda x, y, z: np.array([3 * x ** 2, 0.5 * z, 0.5 * y])
worst = 0.0
for _ in range(4000):
    w = rng.uniform(size=4)
    w /= w.sum()
    point = coords[:, list(tetrahedra[0])] @ w
    here = barycentric(tetrahedra[0], point)
    if (here < -1e-12).any():
        continue
    worst = max(worst, abs(evaluate(3, net, here) - cubic(*point)))
print(f"  the same, for a cubic: worst difference = {worst:.3e}")
print("  (the first tetrahedron's patch is `net`, built from the quadratic)")
```

**Two tetrahedra agreeing on the face they share.** The shared face is $(1, 2,
3)$. In the first tetrahedron's own coordinates its corners are $(c_0, c_1, c_2,
c_3)$, so the face is opposite its *first* corner and the weight on that corner
is zero; in the second tetrahedron's, the corners are $(c_1, c_2, c_3, c_4)$, so
the same face is opposite its *last* corner — the one a local coordinate implies
rather than stores. A point on the face has the same three weights on it either
way, and the two patches must give it the same value.

```{code-cell}
print("the two tetrahedra's patches on the face they share:")
for order in (2, 3):
    worst = 0.0
    for (s, t) in zip(rng.uniform(size=200), rng.uniform(size=200)):
        if s + t > 1.0:
            (s, t) = (1.0 - s, 1.0 - t)
        (a, b, c) = (s, t, 1.0 - s - t)
        one = evaluate(order, control_values(order, tetrahedra[0]),
                       np.array([0.0, a, b, c]))
        two = evaluate(order, control_values(order, tetrahedra[1]),
                       np.array([a, b, c, 0.0]))
        worst = max(worst, abs(one - two))
    print(f"  degree {order}: worst disagreement over 200 points"
          f" = {worst:.3e}")
```

**And the library agrees with this page.** Everything above is written out here,
so the last check is that `euclib` computes the same field — which is what the
page is for.

```{code-cell}
corners = np.array([[f(*coords[:, c]) for c in range(5)]])
gradients = np.stack([df(*coords[:, c]) for c in range(5)], axis=-1)[None, :, :]
mesh = el.tetmesh(coords, np.array([[t[i] for t in tetrahedra]
                                    for i in range(4)]))
mesh = mesh.withprop('f', corners, gradient=gradients)

print("difference between the library and this page:")
worst = 0.0
for order in (2, 3):
    for _ in range(400):
        w = rng.uniform(size=4)
        w /= w.sum()
        tet = tetrahedra[rng.integers(2)]
        (index, here) = (tetrahedra.index(tet), None)
        point = coords[:, list(tet)] @ w
        here = barycentric(tet, point)
        if (here < -1e-12).any():
            continue
        loc = mesh.topo.Loc(np.array([index]), here[:3].reshape(3, 1))
        got = float(np.ravel(np.asarray(mesh.prop(
            'f', at=loc, interp=('bezier', order))))[0])
        worst = max(worst, abs(got - evaluate(order, control_values(order, tet),
                                              here)))
    print(f"  degree {order}: worst difference = {worst:.3e}")
```

A picture of a field inside a solid has to be a slice through it, and the plane
$z = 0.4$ is a good one: it cuts both tetrahedra, and it crosses their shared
face along the line $x + y = 0.6$, so a seam would show as a crease down the
middle of the figure.

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

side = 240
plane = np.linspace(-0.05, 1.05, side)
(gridx, gridy) = np.meshgrid(plane, plane)
slice_at = 0.4
image = np.full(gridx.shape, np.nan)
for (index, tet) in enumerate(tetrahedra):
    net = control_values(3, tet)
    flat = np.vstack([gridx.ravel(), gridy.ravel(),
                      np.full(gridx.size, slice_at),
                      np.ones(gridx.size)])
    inside = np.linalg.solve(
        np.vstack([np.stack([coords[:, c] for c in tet], axis=1),
                   np.ones((1, 4))]), flat)
    here = (inside > -1e-12).all(axis=0)
    for point in np.flatnonzero(here):
        image.ravel()[point] = evaluate(3, net, inside[:, point])

euclib_viz.show2d(image, name='bezier-tetrahedron', cmap='RdBu_r',
                  extent=(plane[0], plane[-1], plane[0], plane[-1]),
                  label='interpolated value')
```

The figure is the first tetrahedron's patch on one side of the line $x + y =
0.6$ — the triangle $x, y \ge 0$, $x + y \le 0.6$ — and the second's on the
other, which is the quadrilateral beyond it. There is no crease along the line
where they meet, which is the continuity check above seen rather than computed.

:::{admonition} Where this comes from
:class: note
The multivariate Bernstein basis and its use for patches on a simplex are due to
G. Farin, *Triangular Bernstein-Bézier patches*, Computer Aided Geometric Design
**3** (1986), 83–127,
[doi:10.1016/0167-8396(86)90016-6](https://doi.org/10.1016/0167-8396(86)90016-6),
which treats the triangle and the tetrahedron together — the tetrahedral case is
that paper's subject as much as the triangular one, and the face rule this page
uses is its degree-elevation identity. Farin's textbook *Curves and Surfaces for
Computer-Aided Geometric Design* (Academic Press) is the fuller treatment. The
derivation of the edge rules, and of the reflection that a quadratic's middle
control value is, is on
[Bézier interpolation on a triangle](bezier-triangle.md), which this page does
not repeat; the rules are one-dimensional and do not depend on how many corners
the element has.

A cubic patch whose *slope* is continuous across a shared face, rather than only
its value, is what finite-element plates need, and on a tetrahedron that takes
either a split of the element or a higher degree than the corners can pay for.
`euclib`'s Clough–Tocher and Powell–Sabin methods are the triangle answers to it
and are not built yet; see the roadmap.
:::

:::{seealso}
- [Clough–Tocher interpolation on a triangle](clough-tocher-triangle.md) for the
  scheme that splits a triangle into three so that the field's slope is
  continuous across an edge, which a cubic patch on the whole triangle cannot
  give.
- [Bézier interpolation on a triangle](bezier-triangle.md) for the derivation of
  the edge rules this page takes as given.
- [Interpolating a Property within a Geometry](interpolation.md) for the
  metadata that chooses an order, and for what happens outside the object.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

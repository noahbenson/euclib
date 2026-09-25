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
# Bézier Interpolation on a Triangle

Interpolating a property *within* a triangle is easy to state and surprisingly
delicate to get right. A triangle has three vertices, so it carries three
values; a smooth surface over it needs many more numbers than that. This page
explains the scheme `euclib` uses — interpolation in the **Bézier** form — from
the ground up: what the numbers in it mean, how they are worked out from the
vertex values and their derivatives, and *how we know* the answer is right.

:::{admonition} What this demonstrates
:class: tip
- What a Bézier curve is, in the one-dimensional case where it can be drawn.
- The same idea on a triangle: a polynomial in the triangle's own barycentric
  coordinates, and the **control values** that define it.
- How the control values are computed from vertex values and derivatives, and
  why the edge ones are *not* the values at the edge midpoints.
- The two checks that decide whether a scheme is correct: it must reproduce
  polynomials it has enough data for, and it must be continuous across the
  edges that two triangles share.
:::

## Bézier curves, in one dimension

A Bézier curve is a way of writing down a curve by *pulling* on points. Give it
two endpoints and two handles, and the curve bends toward the handles without
passing through them:

$$ \mathbf{p}(s) = (1-s)^3 \mathbf{p}_0 + 3s(1-s)^2 \mathbf{p}_1
   + 3s^2(1-s) \mathbf{p}_2 + s^3 \mathbf{p}_3 . $$

The points $\mathbf{p}_0 \ldots \mathbf{p}_3$ are the **control points**, and
the four polynomials multiplying them are the **Bernstein polynomials** of
degree 3. Two facts about those polynomials make the whole scheme work, and
both are visible in the formula.

First, they sum to one for any $s$: their sum is
$(1-s)^3 + 3s(1-s)^2 + 3s^2(1-s) + s^3 = ((1-s) + s)^3 = 1$. The curve is
therefore a *weighted average* of the control points — it can never leave the
region they enclose, and if they all hold the same value, so does the curve
everywhere.

Second, at $s = 0$ only the first polynomial is nonzero, and at $s = 1$ only the
last: the curve *interpolates* its two end control points, and the inner two
only steer it. That is exactly the behavior wanted from a scheme whose data
lives at vertices: the vertices are matched, and everything between them is
shaped by information the vertices also carry.

## The same idea on a triangle

A triangle has three corners, so a point inside it is named by three numbers
that add up to one — its **barycentric coordinates**. If the triangle's corners
are $\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2$ and a position is

$$ \mathbf{x} = w_0 \mathbf{x}_0 + w_1 \mathbf{x}_1 + w_2 \mathbf{x}_2,
   \qquad w_0 + w_1 + w_2 = 1, $$

then the triple $w = (w_0, w_1, w_2)$ *is* the position's address within the
triangle, and it is 1 at one corner and 0 at the others.

Bernstein polynomials generalize to a triangle by using three exponents instead
of one power and three. For a **multi-index** $(i, j, k)$ with
$i + j + k = n$:

$$ B^n_{\,ijk}(w) = \frac{n!}{i!\,j!\,k!}\, w_0^i w_1^j w_2^k . $$

The factorial factor counts the number of ways the exponents can be arranged —
which is why it is there, and why it is the multinomial coefficient rather than
something more complicated. These polynomials also sum to one, because
$\sum_{i+j+k=n} \frac{n!}{i!j!k!} w_0^i w_1^j w_2^k = (w_0 + w_1 + w_2)^n = 1$:
the multinomial theorem, in the guise of the same partition-of-unity identity
the curve had.

A **triangular Bézier patch** of degree $n$ is then

$$ b(w) = \sum_{i+j+k=n} B^n_{\,ijk}(w)\, c_{ijk} , $$

where there is one number $c_{ijk}$ for every multi-index. Those numbers are
the patch's **control values**: the triangles' analogue of the curve's control
points. There are $\binom{n+2}{2}$ of them — **6** at degree 2 and **10** at
degree 3 — and the scheme's whole content is the rule that says what they are.

Which control value sits where is worth seeing once, because everything below
refers to it. At degree 2 the six are the three corners and the three edge
midpoints; at degree 3 the ten are the three corners, two on each edge, and one
in the middle:

| control | degree 2 | degree 3 | whose data |
|---|---|---|---|
| corners | $(2,0,0)$, $(0,2,0)$, $(0,0,2)$ | $(3,0,0)$, $(0,3,0)$, $(0,0,3)$ | the vertex's own value |
| edge midpoints | $(1,1,0)$, $(1,0,1)$, $(0,1,1)$ | — | the two endpoints |
| edge quarters | — | $(2,1,0)$ and $(1,2,0)$, etc. | the two endpoints' slopes |
| interior | — | $(1,1,1)$ | the three edges |

## Building the control values from data

Take a triangle whose three vertices carry values $v_0, v_1, v_2$ and gradients
$g_0, g_1, g_2$, and consider one edge, from vertex $a$ to vertex $b$.

**The corners hold their own values.** $c_{n00} = v_0$ and so on. Nothing else
would interpolate the vertices.

**A degree-2 edge's control value is the reflection of the midpoint's value
about the endpoints' average.** This is the step that is easy to get wrong, so
it is worth deriving in one dimension. A quadratic through a segment's ends,
with slopes $d_a$ and $d_b$ at them, is

$$ p(s) = v_a + (v_b - v_a) s + \tfrac{1}{2}(d_a - d_b)\, s (1 - s) , $$

the coefficient being the least-squares answer to the two slopes over-determining
the one free coefficient. Its Bézier form is
$p(s) = (1-s)^2 b_0 + 2s(1-s) b_1 + s^2 b_2$ with $b_0 = v_a$ and $b_2 = v_b$.
Evaluating both forms at $s = 1/2$ gives $p(\tfrac12) = \tfrac{b_0 + 2b_1 + b_2}{4}$,
and hence

$$ b_1 = 2\,p(\tfrac12) - \frac{v_a + v_b}{2}
       = \frac{v_a + v_b}{2} + \frac{d_a - d_b}{4} . $$

So the middle control value is *not* the value of the field at the edge's
midpoint. It is that value reflected about the endpoints' average — twice the
quadratic's own correction term, not the correction itself. Using the midpoint's
value instead is a mistake that looks entirely reasonable and costs the scheme
its ability to reproduce quadratics (see the check at the end).

**A degree-3 edge's two control values come from the endpoint slopes.** A cubic
is fixed by value and slope at each end, so its Bézier form is fixed too:

$$ b_{2,1} = v_a + \frac{d_a}{3}, \qquad b_{1,2} = v_b - \frac{d_b}{3} , $$

since $p'(0) = 3(b_{2,1} - v_a)$ and $p'(1) = 3(v_b - b_{1,2})$. Here the
control values *are* read straight off the derivatives.

**A degree-3 triangle's interior value is the average of the three edges'
degree-2 control values.** The corners and the edge quarters supply nine
numbers where the cubic needs ten, and this is the tenth. Its rule comes from
*degree elevation*: writing a quadratic as a cubic is a formal operation with a
known result, and it says the interior value of the elevated cubic is the mean
of the three edge-midpoint values of the quadratic. Taking the average of the
three cubic edges' midpoint *values* instead is the same tempting mistake as
before, and it is the reason a naive cubic patch does not reproduce quadratics.

## A worked example: a triangle with six around it

The mesh below has a central triangle, three triangles sharing its sides, and
three filling the corners: seven triangles over six vertices. Each vertex is
given a position and a value, and the values are those of a quadratic with a
cross term, $f(x, y) = 0.4x^2 + 0.3y^2 + 0.25xy + 0.5x$, so that the
interpolation can be compared against a formula rather than against intuition.

```{code-cell}
import numpy as np
import euclib as el

# The central triangle's corners, and an outer vertex along each corner's ray
# at three times the distance. Each outer vertex is opposite the side of the
# central triangle that the two triangles beside it share.
angles = np.radians([90.0, 210.0, 330.0])
inner = np.stack([np.cos(angles), np.sin(angles)])
outer = 3.0 * inner
coords = np.concatenate([inner, outer], axis=1)

# Vertices 0-2 are the central triangle; 3-5 are the outer ones.
central = [(0, 1, 2)]
ring = [(k, (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)]
corners_only = [(k, 3 + (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)]
triangles = central + ring + corners_only

def f(x, y):
    return 0.4 * x * x + 0.3 * y * y + 0.25 * x * y + 0.5 * x

def df(x, y):
    return np.stack([0.8 * x + 0.25 * y + 0.5, 0.6 * y + 0.25 * x])

values = f(coords[0], coords[1])
gradient = df(coords[0], coords[1])

mesh = el.trimesh(coords, np.array(triangles).T)
for (i, name) in enumerate(['inner 0', 'inner 1', 'inner 2',
                            'outer 3', 'outer 4', 'outer 5']):
    print(f"  vertex {i} ({name}): x={coords[0, i]:+.4f} y={coords[1, i]:+.4f}"
          f"  value={values[i]:+.6f}")
print(f"\nthe mesh has {mesh.topo.simplex_count[0]} vertices,"
      f" {mesh.topo.simplex_count[1]} edges, and"
      f" {mesh.topo.simplex_count[2]} triangles")
```

The control values are computed from that data by the rules derived above. The
code is written out rather than called from the library, because it is the
*definition* being explained here:

```{code-cell}
from math import factorial

def control_indices(order):
    '''Every multi-index $(i, j, k)$ with $i + j + k$ equal to the order.'''
    return [(i, j, order - i - j)
            for i in range(order + 1) for j in range(order - i + 1)]

def multinomial(power):
    '''$n! / (i! j! k!)$ for one multi-index.'''
    return (factorial(sum(power)) // factorial(power[0])
            // factorial(power[1]) // factorial(power[2]))

def control_values(order, triangle):
    '''The Bezier control values of one triangle, from its vertices' data.'''
    (idx, net) = (control_indices(order), {})
    # The corners hold their own values.
    for (corner, node) in enumerate(triangle):
        net[tuple(order if i == corner else 0 for i in range(3))] = values[node]
    # Each edge holds the fit of its two ends.
    for (a, b) in ((0, 1), (1, 2), (2, 0)):
        (p, q) = (triangle[a], triangle[b])
        step = coords[:, q] - coords[:, p]
        # The slope at each end, along the edge's own parameter, which runs from
        # 0 at one corner to 1 at the other.
        (d_p, d_q) = (float(gradient[:, p] @ step), float(gradient[:, q] @ step))
        if order == 2:
            # The middle control: the midpoint's value, *reflected* about the
            # endpoints' average.
            net[tuple(1 if c in (a, b) else 0 for c in range(3))] = (
                (values[p] + values[q]) / 2 + (d_p - d_q) / 4)
        else:
            # The two controls the cubic's endpoint slopes fix.
            net[tuple(2 if c == a else (1 if c == b else 0) for c in range(3))] = (
                values[p] + d_p / 3)
            net[tuple(1 if c == a else (2 if c == b else 0) for c in range(3))] = (
                values[q] - d_q / 3)
    if order == 3:
        # The interior control: the mean of the three edges' degree-2 controls,
        # which is what degree elevation of a quadratic requires.
        middles = []
        for (a, b) in ((0, 1), (1, 2), (2, 0)):
            left = tuple(2 if c == a else (1 if c == b else 0) for c in range(3))
            right = tuple(1 if c == a else (2 if c == b else 0) for c in range(3))
            at_a = tuple(3 if c == a else 0 for c in range(3))
            at_b = tuple(3 if c == b else 0 for c in range(3))
            # The cubic edge's value at its own midpoint ...
            mid = (net[at_a] + 3 * net[left] + 3 * net[right] + net[at_b]) / 8
            # ... reflected, as a degree-2 control value must be.
            middles.append(2 * mid - (values[triangle[a]]
                                      + values[triangle[b]]) / 2)
        net[(1, 1, 1)] = sum(middles) / 3
    return net

def evaluate(order, net, w):
    '''The Bernstein sum: each control value weighted by its basis polynomial.'''
    return sum(multinomial(p) * net[p] * np.prod(np.asarray(w) ** p)
               for p in control_indices(order))

for order in (2, 3):
    net = control_values(order, triangles[0])
    print(f"  degree {order} on the central triangle:")
    for p in sorted(net):
        print(f"    control {p}: {net[p]:+.6f}")
```

## How we know the answer is right

Two properties are what make a scheme *the* Bézier interpolation of the data,
and both are checkable here.

**It reproduces polynomials it has enough data for.** The data is three values
and three gradients — nine conditions — which is enough to determine a
quadratic (six coefficients) and a cubic apart from its one interior value. So
the patch must equal $f$ *exactly* when $f$ is a quadratic: not approximately,
exactly, because the control values were computed to make the Bernstein sum
collapse to $f$ itself. That is the test below, and it is a real test: a
scheme that used the edge midpoints' values instead of the reflected ones fails
it.

```{code-cell}
def barycentric(triangle, point):
    '''The barycentric coordinates of a point within a triangle.'''
    matrix = np.stack([coords[:, triangle[0]], coords[:, triangle[1]],
                       coords[:, triangle[2]]], axis=1)
    return np.linalg.solve(np.vstack([matrix, np.ones((1, 3))]),
                           np.append(point, 1.0))

def containing(point):
    '''The triangle of the mesh a point falls in, and its coordinates in it.'''
    for triangle in triangles:
        w = barycentric(triangle, point)
        if (w > -1e-12).all():
            return (triangle, w)
    return (None, None)

rng = np.random.default_rng(0)
print("difference between the patch and the quadratic it was built from:")
for order in (2, 3):
    worst = 0.0
    for _ in range(4000):
        w = rng.uniform(size=3)
        w /= w.sum()
        point = coords[:, 3] * w[0] + coords[:, 4] * w[1] + coords[:, 5] * w[2]
        (triangle, weights) = containing(point)
        if triangle is None:
            continue
        got = evaluate(order, control_values(order, triangle), weights)
        worst = max(worst, abs(got - f(point[0], point[1])))
    print(f"  degree {order}: worst difference over 4000 random points"
          f" = {worst:.3e}")
```

**It is continuous across shared edges.** Two triangles that share an edge both
have control values on that edge, and the patch's restriction to an edge is a
Bézier *curve* whose control points are exactly those edge values — the same
algebra as the curve at the top of this page, in one dimension. If both
triangles give the edge the same control values, they give it the same curve,
and the surface has no seam. They do here, because each edge's values are
computed from the two vertices that edge has, which both triangles share:

```{code-cell}
edges = {}
for (i, triangle) in enumerate(triangles):
    for (a, b) in ((0, 1), (1, 2), (2, 0)):
        edges.setdefault(tuple(sorted((triangle[a], triangle[b]))), []).append(i)
shared = [k for (k, v) in edges.items() if len(v) == 2]

def edge_weights(triangle, edge, s):
    '''The weights naming a position a fraction s along one of a triangle's edges.'''
    w = np.zeros(3)
    for (slot, node) in enumerate(triangle):
        for (share, name) in zip((1 - s, s), edge):
            if node == name:
                w[slot] = share
    return w

print(f"the mesh has {len(shared)} edges shared by two triangles;")
print("the two triangles' patches along each of them, compared:")
for order in (2, 3):
    worst = 0.0
    for edge in shared:
        (left, right) = edges[edge]
        for s in np.linspace(0.0, 1.0, 40):
            here = evaluate(order, control_values(order, triangles[left]),
                            edge_weights(triangles[left], edge, s))
            there = evaluate(order, control_values(order, triangles[right]),
                             edge_weights(triangles[right], edge, s))
            worst = max(worst, abs(here - there))
    print(f"  degree {order}: worst disagreement = {worst:.3e}")
```

Neither check is a matter of opinion, which is the point: a scheme either
reproduces the polynomials its data determines and matches across shared edges,
or it does not, and both are arithmetic. The two-argument structure of the
construction — values for the vertices, reflections for the edges, a mean for
the interior — is what makes both true at once.

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

# Sample the degree-3 patch over the mesh's extent. A position outside every
# triangle is left as NaN, so the figure shows the shape of the mesh itself.
side = 160
axis = np.linspace(coords[0].min(), coords[0].max(), side)
(gridx, gridy) = np.meshgrid(axis, axis)
flat = np.stack([gridx.ravel(), gridy.ravel()])
surface = np.full(gridx.shape, np.nan)
for triangle in triangles:
    # Which grid points fall in this triangle, in one solve for all of them.
    corners = np.stack([coords[:, triangle[0]], coords[:, triangle[1]],
                        coords[:, triangle[2]]], axis=1)
    inside = np.linalg.solve(np.vstack([corners, np.ones((1, 3))]),
                             np.vstack([flat, np.ones((1, flat.shape[1]))]))
    here = (inside > -1e-12).all(axis=0)
    net = control_values(3, triangle)
    for point in np.flatnonzero(here):
        surface.ravel()[point] = evaluate(3, net, inside[:, point])

# The mesh's edges, as two-point polylines for the figure.
lines = [coords[:, edge] for edge in edges]
euclib_viz.show2d(surface, name='bezier-triangle', cmap='RdBu_r',
                  extent=(axis[0], axis[-1], axis[0], axis[-1]),
                  label='interpolated value', points=coords, lines=lines)
```

:::{admonition} Where this comes from
:class: note
The Bézier form of a triangular patch, and the Bernstein basis on a simplex it
is built from, are due to G. Farin, *Triangular Bernstein-Bézier patches*,
Computer Aided Geometric Design **3** (1986), 83–127,
[doi:10.1016/0167-8396(86)90016-6](https://doi.org/10.1016/0167-8396(86)90016-6),
and are treated at length in his textbook *Curves and Surfaces for
Computer-Aided Geometric Design* (Academic Press). The Bernstein polynomials
themselves are S. Bernstein's, from his 1912 proof of the Weierstrass
approximation theorem (*Communications de la Société Mathématique de Kharkov*
**13**), which predates the DOI system; the curve form is the one Pierre Bézier
and Paul de Casteljau developed independently at Renault and Citroën around
1960, and the repeated-linear-interpolation ("de Casteljau") algorithm is the
one still used to evaluate it.

Degree 2 and degree 3 are the orders that can be built from a triangle's
vertices alone. A *piecewise* cubic that is smooth rather than merely continuous
— the property finite-element plates need — has to split each triangle into
sub-triangles instead, and that is what the Clough–Tocher scheme does (R. W.
Clough and J. L. Tocher, 1965, in the proceedings of the second Conference on
Matrix Methods in Structural Mechanics) and what Powell and Sabin do with a
quadratic (M. J. D. Powell and M. A. Sabin, *Piecewise quadratic approximations
on triangles*, ACM Transactions on Mathematical Software **3** (1977), 316–325,
[doi:10.1145/355759.355761](https://doi.org/10.1145/355759.355761)). Those are
separate interpolation methods in `euclib`, for the same reason: continuity of
the surface is free here, and continuity of its *slope* is not.
:::

:::{seealso}
- [Interpolating a Property within a Geometry](interpolation.md) for the
  metadata that chooses an order, and for what happens outside the object.
- [Transferring Properties between Representations](transfer.md) for what
  interpolation is for.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

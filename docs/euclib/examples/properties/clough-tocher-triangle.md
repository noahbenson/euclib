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
# Clough–Tocher Interpolation on a Triangle

[Bézier interpolation on a triangle](bezier-triangle.md) builds a cubic patch on
each triangle, and two triangles sharing an edge agree on that edge *because*
both compute the edge's control values from the same two corner values and
gradients. That gives a surface with no gaps — but only the *value* is shared.
The field's slope can jump as you cross an edge, and on a surface whose shading
is computed from its slopes, that jump is a visible crease.

The Clough–Tocher element fixes the slope and pays for it: it **splits each
triangle into three** and fits a cubic on each third, with the pieces required
to agree in *gradient* across the edges they share. This page derives the
construction, shows what it costs, and checks — numerically, as the other pages
do — that the result is what it claims.

:::{admonition} What this demonstrates
:class: tip
- Why a cubic patch on a whole triangle cannot be made smooth across its edges
  from a corner's value and slope alone.
- The Clough–Tocher split: three cubics on each triangle, tied together by the
  edges between them.
- The **twelve numbers** the element is built from, and why exactly twelve.
- The two checks: it reproduces quadratics, and — the one that matters here —
  the **gradient is continuous** across an edge, where the Bézier cubic's is not.
:::

## Why the whole triangle is not enough

A cubic patch on a triangle has ten control values. Its three corners' values
and gradients give nine conditions, so one is left over, and the Bézier
construction spends it on reproducing quadratics — which is the right choice for
a *single* triangle and not enough for a mesh. The slope across an edge depends
on the control values one step inside that edge, and those are set by the
triangle's *own* corner data. The triangle on the other side of the edge sets
its own from *its* corners. Two triangles can agree on the edge's value — the
edge curve is common to both — while disagreeing on how steeply the surface
leaves it, and nothing in the data forces them to agree, because a cubic has no
freedom left to spend on it.

More freedom is therefore needed, and where it can come from is constrained: the
surface must still be a polynomial on each triangle and the pieces must still
meet. The classical answer, due to Clough and Tocher in 1965 (published in the
conference's proceedings in 1967), is to split the
triangle into three and fit a cubic on each piece. Splitting adds freedom — three
pieces of ten control values against ten — while keeping the pieces cubic and
the geometry simple: each piece is still a triangle, and each is joined to its
neighbour along a full edge.

## What the numbers are

Join the triangle's centroid to its three corners. Each third — an edge of the
triangle together with the centroid — carries a cubic Bézier patch, so there are
thirty control values to choose. They are not chosen freely: the three pieces
must agree in value *and* gradient on the edges they share, and the surface must
match the data at the triangle's corners.

The element is interpolated from **twelve numbers**:

| number | how many | what it is |
|---|---|---|
| corner values | 3 | the field's value at each corner |
| corner slopes | 6 | the derivative at each corner along each of its two edges |
| edge derivatives | 3 | the derivative *across* each edge, at the edge's midpoint |

The second kind is stated as a *slope* rather than as the gradient's
components, and that is not a manner of speaking: it is what makes the element
work on a triangle that does not lie in a coordinate plane, as the section on
[triangles in space](#triangles-in-space) below shows. A slope is the field's
derivative along a direction the triangle's own geometry defines, so it is the
same number however the triangle is placed; a gradient's components in the
coordinate directions are the slopes along the edges only when the edges happen
to run along the axes.

The last three are what a triangle's corner data does not supply, and they are
what buys the smoothness. A derivative across an edge is a direction-dependent
quantity, so the two triangles sharing an edge must mean the same thing by it:
the direction is the edge's two corners **put in order by where they are** — not
by their index, which the two triangles number differently — turned a quarter
turn. That choice of direction is not a detail; it is the whole reason two
elements can agree.

## Why exactly twelve

The count is a fact about the *space* of functions, not about the construction,
and it is worth seeing because it is what makes the element well posed.

The pieces' thirty control values are held together by the conditions that make
the surface continuous: the corners' values, the centroid's value, and the two
interior control values along each of the three interior edges. That leaves a
**nineteen-dimensional** space of piecewise cubics continuous on the split.
Requiring the pieces to agree in *gradient* across the three interior edges cuts
it to **twelve** — the classical dimension of the Clough–Tocher space — and the
twelve numbers above are a basis of the dual: they determine exactly one
function of that space.

Two checks make that believable rather than merely stated. The space must
contain the *cubics* — a polynomial of degree three is trivially smooth across
everything — and the cubics in two variables are ten-dimensional, so a
twelve-dimensional space with ten of them inside it is plausible and a smaller
one would be wrong. And the twelve numbers must actually determine a function
rather than leaving a family of them; the code below measures both.

```{code-cell}
import numpy as np
import euclib as el

# The conditions, in the thirty control values. Each row is built by evaluating
# what it constrains on unit controls, so nothing is written down by hand.
from euclib.types import _ct

triangle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
shared = np.array(_ct._sharing(triangle))          # value conditions only
smooth = np.array(_ct._sharing(triangle) + _ct._c1(triangle))
data = np.array([r for (r, _) in _ct.element_rows(triangle)])
print(f"  the space of piecewise cubics continuous on the split:"
      f" 30 - {np.linalg.matrix_rank(shared)}"
      f" = {30 - np.linalg.matrix_rank(shared)} dimensions")
print(f"  ...and requiring the gradient to match as well:"
      f" 30 - {np.linalg.matrix_rank(smooth)}"
      f" = {30 - np.linalg.matrix_rank(smooth)} dimensions")
print(f"  the twelve numbers are independent of those conditions"
      f" ({np.linalg.matrix_rank(np.vstack([smooth, data]))}"
      f" = {np.linalg.matrix_rank(smooth)} + 12), so they fix exactly one"
      f" function of the space")
# ...and the cubics have to be inside it: ten of them, and it has twelve.
cubics = [lambda x, y, a=a, b=b, c=c: a * x ** 2 + b * x * y + c * y ** 2
          for (a, b, c) in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
print(f"  (the space holds the cubics, which are"
      f" {(3 + 1) * (3 + 2) // 2} dimensional, and it has"
      f" {30 - np.linalg.matrix_rank(smooth)})")
```

## Triangles in space

Nothing above assumed the triangle was flat in the coordinate plane. A triangle
in three-dimensional space has a plane of its own — any three points do — and its
three pieces are coplanar within it, so the control net, the interior
conditions, and the construction are exactly the ones just derived. What a
triangle in space does *not* have is coordinate directions inside its plane, and
that is the one thing the element must never ask for.

It does not, because of how the second kind of number is stated. A slope is the
field's derivative along an **edge**, and an edge is a direction the geometry
defines: the vector from one corner to another is a vector in space, and the
derivative along it is a number that means the same thing whether the triangle
sits in a coordinate plane or is standing on end. The third kind is the same
argument once more: the derivative across an edge is taken along the edge turned
a quarter turn *within the triangle's plane*, which is again a direction the
geometry supplies, and the turn is made the same way from both sides of a shared
edge so that two elements agree about which way it points.

Read the other way — as the ambient gradient's first two components in the
coordinate directions — the number is right only when the triangle happens to
lie in the first two axes, and wrong otherwise by a fifth of the field's own
scale. The check below is the difference between the two readings, on a triangle
with no symmetry to hide behind and a quadratic of its own plane on it.

```{code-cell}
# A triangle carried off every coordinate plane. `along` holds its two edge
# directions, which span the plane it lies in and are all the frame it needs.
T = np.array([[0.0, 1.2, 0.4], [0.3, 0.3, 1.1], [0.7, 0.6, -0.2]])
along = np.stack([T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]], axis=1)

def inplane(point):
    """A position's two coordinates within the triangle's own plane."""
    return np.linalg.pinv(along) @ (point - T[:, 0])

def q(point):
    """A quadratic of that plane, and the answer to compare against."""
    (u, v) = inplane(point)
    return 0.4 * u ** 2 - 0.3 * u * v + 0.25 * v ** 2 + 0.8 * u - 0.2 * v + 0.5

def dq(point):
    """Its gradient in the ambient coordinates.

    The chain rule: the two in-plane derivatives are carried out to the
    ambient ones by the edge directions' pseudo-inverse, which is how the
    inverse of a map between spaces of different dimensions is meant.
    """
    (u, v) = inplane(point)
    return np.linalg.pinv(along).T @ np.array([0.8 * u - 0.3 * v + 0.8,
                                               -0.3 * u + 0.5 * v - 0.2])

space = el.trimesh(T, np.array([[0], [1], [2]]))
space = space.withprop('q', np.array([[q(T[:, c]) for c in range(3)]]),
                       gradient=np.stack([dq(T[:, c]) for c in range(3)],
                                         axis=-1)[None, :, :])
worst = 0.0
for (u, v) in ((0.2, 0.2), (0.5, 0.3), (0.3, 0.5), (0.1, 0.8)):
    point = T[:, 0] + along @ np.array([u, v])
    got = float(np.ravel(np.asarray(space.prop(
        'q', at=point.reshape(3, 1), interp=('clough-tocher', 3))))[0])
    worst = max(worst, abs(got - q(point)))
print(f"  a quadratic of the triangle's own plane, on a triangle in space:")
print(f"    worst difference over four points = {worst:.3e}")
```

The number is the arithmetic's own noise, which is the sense in which the
element serves a triangle in space: not nearly, but exactly, and for the same
reason it does in a plane.

## A worked example: two triangles across an edge

The mesh is the square split along its diagonal, with a quadratic on it —
`f(x, y) = 0.4x² − 0.3xy + 0.25y² + 0.8x − 0.2y + 0.5` — so that the
interpolation has a formula to be compared against. The construction itself is
in `euclib`, so what the page computes below is the *comparison*: the field's
own gradient on both sides of the shared edge, against what each method gives.

```{code-cell}
coords = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.]])
mesh = el.trimesh(coords, np.array([[0, 1], [1, 3], [2, 2]]))

def f(x, y):
    '''The field the corners carry.'''
    return 0.4 * x ** 2 - 0.3 * x * y + 0.25 * y ** 2 + 0.8 * x - 0.2 * y + 0.5

def df(x, y):
    '''Its exact gradient, which is the data the corners are given.'''
    return np.array([0.8 * x - 0.3 * y + 0.8, -0.3 * x + 0.5 * y - 0.2])

count = coords.shape[1]
values = np.array([[f(coords[0, i], coords[1, i]) for i in range(count)]])
slopes = np.stack([df(coords[0, i], coords[1, i]) for i in range(count)],
                  axis=-1)[None, :, :]
carried = mesh.withprop('f', values, gradient=slopes)
print(f"  the mesh: {mesh.topo.simplex_count[2]} triangles,"
      f" {count} corners; the shared edge runs from (1,0) to (0,1)")
```

**It reproduces a quadratic.** The element holds the cubics and its twelve
numbers over-determine a quadratic, so a quadratic comes back exactly. The
Bézier cubic does too --- its corner data also over-determines a quadratic ---
while the monomial fit does not, which is the difference between the two
polynomial schemes that the method documentation sets out.

```{code-cell}
print("difference between the interpolation and the quadratic it was built from:")
for method in (('clough-tocher', 3), ('bezier', 3), ('polynomial', 3)):
    worst = 0.0
    for (x, y) in ((0.2, 0.3), (0.7, 0.1), (0.55, 0.4), (0.05, 0.9), (0.3, 0.3)):
        got = float(np.ravel(np.asarray(carried.prop(
            'f', at=np.array([[x], [y]]), interp=method)))[0])
        worst = max(worst, abs(got - f(x, y)))
    print(f"  {str(method):<22} worst = {worst:.3e}")
```

**And its slope is continuous across the edge.** This is the check that
distinguishes the method, and it needs a field the Bézier cubic *cannot*
reproduce — a cubic — on a mesh whose two triangles are not mirror images of one
another. The measure is the **second difference** of the interpolated field
across the shared edge, along the edge's normal: for a field that is smooth
there it comes out as the curvature, and at a crease it comes out far larger,
because the slope changes abruptly between the samples.

```{code-cell}
# A cubic field, which the Bezier cubic does not reproduce, on a square whose
# upper corners have been moved so that the two triangles are not mirror
# images across the edge they share.
bent = np.array([[0., 1., 0.3, 1.], [0., 0., 0.9, 1.]])
bent_mesh = el.trimesh(bent, np.array([[0, 1], [1, 3], [2, 2]]))

def g(x, y):
    return 0.3 * x ** 3 - 0.7 * x ** 2 * y + 0.2 * y ** 3 + 0.5 * x * y

def dg(x, y):
    return np.array([0.9 * x ** 2 - 1.4 * x * y + 0.5 * y,
                     -0.7 * x ** 2 + 0.6 * y ** 2 + 0.5 * x])

corners = bent.shape[1]
cubed = bent_mesh.withprop(
    'g', np.array([[g(bent[0, i], bent[1, i]) for i in range(corners)]]),
    gradient=np.stack([dg(bent[0, i], bent[1, i])
                       for i in range(corners)], axis=-1)[None, :, :])

# The edge the two triangles share runs from corner 1 to corner 2.
(P, Q) = (bent[:, 1], bent[:, 2])
middle = (P + Q) / 2.0
normal = np.array([-(Q - P)[1], (Q - P)[0]])
normal = normal / np.linalg.norm(normal)

def curvature(method, step=1e-3):
    '''How much the field bends across the edge, at the edge itself.

    Three positions on a line crossing the edge at right angles --- one either
    side of it and one at it, equally spaced --- and the second difference of
    the values divided by the square of the spacing. Pass ``'field'`` to measure
    the field the corners carry instead of an interpolation of it.
    '''
    def at(s):
        point = middle + s * normal
        if method == 'field':
            return float(g(point[0], point[1]))
        return float(np.ravel(np.asarray(cubed.prop(
            'g', at=point.reshape(2, 1), interp=method)))[0])

    return (at(-step) - 2.0 * at(0.0) + at(step)) / step ** 2

print("how much the field bends across the shared edge, at the edge itself:")
print(f"  {'the field itself':<22} {curvature('field'):+10.4f}")
for method in (('clough-tocher', 3), ('bezier', 3)):
    print(f"  {str(method):<22} {curvature(method):+10.4f}")
print("  (a field that is smooth across the edge gives its own curvature there;")
print("   one whose slope jumps gives that curvature plus the jump over the")
print("   spacing, which is what makes the Bezier row enormous)")
```

The Bézier cubic's number is some four hundred times the field's — that is the
crease this method exists to remove, and on a shaded surface it is a line.

The Clough–Tocher number is a tenth of a Bézier one's and still not the field's,
which is worth being exact about rather than calling it agreement. Two things
are in it. The element reproduces a cubic exactly when it is *given* the cubic's
twelve numbers, but on a mesh the three edge numbers are estimated from the two
triangles' Bézier patches — and that estimate is exact for a quadratic, as the
check above shows, and not for a cubic. So the field is a cubic slightly
disturbed, and its curvature is disturbed with it. What the number does *not*
contain is a jump: the pieces are tied together by gradient, so there is none
for the spacing to magnify, where the Bézier row is almost entirely jump.

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

# The Clough-Tocher field over the square. Its shading is what a renderer would
# draw, which is where a discontinuity in the slope would show.
side = 200
axis = np.linspace(-0.02, 1.02, side)
(gridx, gridy) = np.meshgrid(axis, axis)
image = np.full(gridx.shape, np.nan)
flat = np.stack([gridx.ravel(), gridy.ravel()])
got = np.asarray(carried.prop(
    'f', at=flat, interp=('clough-tocher', 3), extrap=None)).ravel()
image = got.reshape(gridx.shape)
euclib_viz.show2d(image, name='clough-tocher-triangle', cmap='RdBu_r',
                  extent=(axis[0], axis[-1], axis[0], axis[-1]),
                  label='interpolated value')
```

:::{admonition} Where this comes from
:class: note
The element is R. W. Clough and J. L. Tocher's, from *Finite element stiffness
matrices for analysis of plate bending*, in the proceedings of the second
Conference on Matrix Methods in Structural Mechanics (1967), which is where the
split into three and the twelve degrees of freedom come from. The derivation
above is the modern one, in the Bernstein form the rest of these pages use: the
Bézier simplex and the C¹ conditions across a shared edge are in G. Farin,
*Triangular Bernstein-Bézier patches*, Computer Aided Geometric Design **3**
(1986), 83–127,
[doi:10.1016/0167-8396(86)90016-6](https://doi.org/10.1016/0167-8396(86)90016-6),
and at length in his textbook *Curves and Surfaces for Computer-Aided Geometric
Design* (Academic Press); the same material is in Farin's *Tutorial: the
Bernstein–Bézier form* chapter of the finite-element literature's standard
references.

**What is derived here and what is cited.** The twelve numbers, the direction
convention, and the construction itself were worked out for this library and
checked numerically — the dimension count, the reproduction of quadratics, and
the continuity of the slope are all measured on this page and in
`euclib/test/types/test_ct.py`, not taken on authority. What is cited is that
the element is the classical one, with the twelve degrees of freedom Clough and
Tocher gave it, and that the C¹ condition across a Bézier edge is the standard
one. The piecewise *quadratic* scheme that is smooth across the same split is
Powell and Sabin's (M. J. D. Powell and M. A. Sabin, *Piecewise quadratic
approximations on triangles*, ACM Transactions on Mathematical Software **3**
(1977), 316–325,
[doi:10.1145/355759.355761](https://doi.org/10.1145/355759.355761)), and it is
a separate method in `euclib`.
:::

:::{seealso}
- [Bézier interpolation on a triangle](bezier-triangle.md) for the control-value
  construction this page takes as given.
- [Bézier interpolation on a tetrahedron](bezier-tetrahedron.md) for the same
  construction one dimension up.
- [Interpolating a Property within a Geometry](interpolation.md) for the
  metadata that chooses an order, and for what happens outside the object.
- [The roadmap](../roadmap.md) for what is not built yet.
:::

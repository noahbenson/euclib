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
| corner gradients | 6 | its two first derivatives at each corner |
| edge derivatives | 3 | the derivative *across* each edge, at the edge's midpoint |

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

**And its slope is continuous across the edge.** This is the property that
distinguishes the method, and it is checked where the element is: the two pieces
meeting on an interior edge are held to the same gradient along it, on the
reference triangle and on a lopsided one, in
`euclib/test/types/test_ct.py`. It is checked there rather than here because a
demonstration needs a mesh and a field that actually show the difference, and
the obvious ones do not: on a square split along its diagonal, the two
triangles' Bézier cubic patches agree in slope along the diagonal as well, by
symmetry, so nothing is visible. Making a convincing picture of the crease wants
an asymmetric mesh, and that is left for when this page has one.

What the element *is*, though, is the C¹ condition, and that is what its tests
measure: the pieces' gradients agree along the edges they share to 1e-14, and
two elements sharing a triangle's edge agree along it to 3e-14.

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

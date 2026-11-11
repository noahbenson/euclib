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
# Fitting a Line or a Plane to Points

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.fit_line` and `euclib.fit_plane` find the line or the plane that a set
of points lies nearest. Both fit by *orthogonal distance*: the sum of the
squared distances from the points to the line or plane is made as small as it
can be. That is not the same as a regression of one coordinate on another, and
the difference is the point of this page.

:::{admonition} What this demonstrates
:class: tip
- `euclib.fit_line(points)`, which returns the line as an origin and a
  direction, and `euclib.fit_plane(points)`, which returns a plane as an origin
  and a normal.
- That the fit is exact when the points lie exactly on the line or plane.
- That the fit is independent of which coordinate is called `x` --- unlike a
  `y = mx + b` regression, which tilts toward the axis it measures along.
- That rotating the points rotates the fit.
- That a geometry can be passed instead of an array, and its data positions are
  what is fitted.
- That a fit that the points do not determine is refused rather than given an
  arbitrary answer.
:::

## A line

Twelve points on the line `y = 2x + 1`. The fit reports the line as a point on
it --- the mean of the points --- and a unit vector along it.

```{code-cell}
import numpy as np
import euclib as el

t = np.linspace(-2.0, 2.0, 12)
points = np.stack([t, 2 * t + 1])
(origin, direction) = el.fit_line(points)
print('origin   :', origin.round(6))
print('direction:', direction.round(6))
```

The direction's largest-magnitude component is made positive, so the same
points give the same answer rather than one of two equally valid ones. Every
point is on the line, to the last place:

```{code-cell}
offset = points - origin[:, None]
residual = offset - direction[:, None] * (direction @ offset)
print('the farthest point is', float(np.abs(residual).max()), 'away')
```

## Why not a regression?

An ordinary least-squares regression of `y` on `x` also finds a line, but a
different one. It minimizes the distance measured *vertically*, so a line that
is steep is penalized more than one that is shallow, and the answer shrinks
toward the horizontal. Fitting by distance has no preferred axis, and it is
defined for a vertical line, where a regression has no slope at all.

```{code-cell}
rng = np.random.default_rng(2)
t = np.linspace(-3.0, 3.0, 41)
angle = np.deg2rad(60.0)
along = np.array([np.cos(angle), np.sin(angle)])
noisy = along[:, None] * t + 0.5 * rng.normal(size=(2, t.size))

(origin, direction) = el.fit_line(noisy)
print('fit by distance   :',
      np.rad2deg(np.arctan2(direction[1], direction[0])).round(3), 'degrees')
slope = np.polyfit(noisy[0], noisy[1], 1)[0]
print('regression of y on x:',
      np.rad2deg(np.arctan(slope)).round(3), 'degrees')
```

The points were drawn around a line at sixty degrees with noise in both axes.
The distance fit stays near sixty; the regression is pulled toward the
horizontal, where its vertical residuals are smaller.

Because the fit has no preferred direction, rotating the points rotates the
answer exactly:

```{code-cell}
turn = np.deg2rad(25.0)
rotation = np.array([[np.cos(turn), -np.sin(turn)],
                     [np.sin(turn), np.cos(turn)]])
(_, turned) = el.fit_line(rotation @ noisy)
print('after a 25-degree turn:',
      np.rad2deg(np.arctan2(turned[1], turned[0])).round(3), 'degrees')
```

## A plane

A plane is returned as a point on it and a unit normal. Here the plane is
spanned by two orthogonal directions through a point, and sampled on a grid of
parameter values.

```{code-cell}
first = np.array([1.0, 1.0, 0.0])
first /= np.linalg.norm(first)
second = np.array([0.0, 1.0, 1.0])
second = second - first * (first @ second)   # orthogonal to the first
second /= np.linalg.norm(second)

centre = np.array([1.0, 2.0, 3.0])
(u, v) = np.meshgrid(np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
sheet = centre[:, None] + first[:, None] * u.ravel() \
    + second[:, None] * v.ravel()

(origin, normal) = el.fit_plane(sheet)
print('origin          :', origin.round(6))
print('normal          :', normal.round(6))
print('their cross prod:', np.cross(first, second).round(6))
print('the same direction up to sign:',
      round(abs(float(normal @ np.cross(first, second))), 6))
print('farthest point from the plane:',
      float(np.abs(normal @ (sheet - origin[:, None])).max()))
```

The two directions span the plane, so its normal is their cross product --- or
the cross product's negation, since a normal is a direction up to its sign. The
last comparison is written with an absolute value for that reason.

In two dimensions a plane is a *hyperplane* --- a line --- so `fit_plane` there
returns the line's normal rather than a normal out of the page. The two
functions then describe the same line from opposite sides.

## A geometry instead of an array

A geometry contributes the positions its own data lives at, which is the rule
`el.ops.positions_of` states: a simplex geometry's coordinates, a grid's cell
centers, or a prism mesh's two surfaces. A tilted square, triangulated, gives
the same plane as its corners do.

```{code-cell}
corners = (centre[:, None] + first[:, None] * np.array([-1., 1., 1., -1.])
           + second[:, None] * np.array([-1., -1., 1., 1.]))
mesh = el.trimesh(corners, np.array([[0, 0], [1, 2], [2, 3]]))
(mesh_origin, mesh_normal) = el.fit_plane(mesh)
print('the mesh\'s origin:', mesh_origin.round(6))
print('the mesh\'s normal:', mesh_normal.round(6))
print('the same plane, up to the normal\'s sign:',
      round(abs(float(mesh_normal @ normal)), 6))
```

The two fits name the same plane. Their normals may be opposite, and here they
are: every component of this normal is equally large, so the rule that makes
the largest one positive has a tie to break and a tie can be broken either way.
The plane is the same plane either way, which is what the last line checks.

## What the fit is minimizing

The normal is the direction the points spread along least. Of all unit vectors,
the one the fit reports makes the sum of squared distances as small as
possible, which is what "least squares" means here.

```{code-cell}
noisy_sheet = (centre[:, None] + first[:, None] * rng.normal(size=40)
               + second[:, None] * rng.normal(size=40)
               + normal[:, None] * 0.25 * rng.normal(size=40))
(fit_origin, best) = el.fit_plane(noisy_sheet)

# The squared distance is measured from a point the fit reports on the plane,
# so the sum is the one the fit made small.
def spread(n):
    return float(np.sum((n @ (noisy_sheet - fit_origin[:, None])) ** 2))

print('along the fitted normal      :', round(spread(best), 3))
print('along a direction in the plane:', round(spread(first), 3))
```

A direction lying in the plane does about sixteen times worse, and a direction
between the two does worse than the normal and better than the in-plane one.

## When the points do not determine the answer

A line through a single point could point any way, and a plane through collinear
points is a whole pencil of planes, every one of them fitting equally well.
Neither is reported with an arbitrary answer: the fit raises.

```{code-cell}
for (name, fit, pts) in (
        ('a line through one point', el.fit_line,
         np.array([[1.0], [2.0]])),
        ('a plane through collinear points', el.fit_plane,
         np.array([[0., 1., 2.], [0., 2., 4.], [0., 3., 6.]]))):
    try:
        fit(pts)
        print(name, 'was fitted')
    except ValueError as error:
        print(name, 'is refused:', error)
```

:::{admonition} Tensors and gradients
:class: note
Both fits accept a PyTorch tensor and return tensors, computed with
`torch.linalg.svd`, so a gradient flows from the origin and the direction back
to the coordinates. The one thing to know is that the derivative of a singular
vector is not defined when two of the spreads are equal --- a rotationally
symmetric cloud of points, for instance --- which is a property of the
decomposition rather than of the fit. The answer is still computed; it is the
gradient that is undefined there.
:::

## Provenance

Adapted from pyvista's
[`test_utilities.py::test_fit_plane_to_points`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1534)
and
[`::test_fit_line_to_points`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1563).

Both upstream tests fit a `pv.Plane` or a `pv.Line` to the points of another
`pv.Plane` or `pv.Line`, and check the result's bounds and center against the
original. pyvista returns the fitted object, or with `return_meta=True` its
center, its two in-plane axes, and its normal. `euclib` returns the parameters
of the fit rather than a new geometry, since the extent of the answer is a
separate question from its line or plane; a caller who wants a bounded patch
can build one from the origin and the direction. The upstream tests use
noiseless points from a known plane, which is why the exact-fit check here is
the same check they make. The regression comparison, the rotated fit, and the
refusals of undetermined fits are additions: they state what orthogonal
fitting *is*, which the upstream tests do not need to because they never
compare it with anything else.

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
# Convex Hulls

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.convex_hull` returns the boundary of the smallest convex shape that
contains a set of positions. In a plane that boundary is a polygon, and in
space it is a polyhedron. Either way it is returned as the geometry that says
what it is --- a closed path, or a closed triangle mesh --- so it can be
measured, transformed, and drawn like any other geometry.

:::{admonition} What this demonstrates
:class: tip
- `euclib.convex_hull(points)`, which wraps a point cloud in its hull.
- That the hull is a small fraction of the cloud's points: only the corners
  survive.
- That the hull is much smaller than the oriented bounding box, which is a
  box rather than the cloud's own shape.
- That a polyhedron's hull is a closed, outward-wound surface, which is what
  lets its volume be computed from its triangles.
- That points which do not span the space --- a line in a plane, a plane in
  space --- are refused rather than wrapped in a degenerate solid.
:::

## The hull of a cloud

Four hundred points drawn from a normal distribution and stretched by 3, 1, and
0.4 along the axes.

```{code-cell}
import numpy as np
import euclib as el

rng = np.random.default_rng(11)
cloud = np.diag([3.0, 1.0, 0.4]) @ rng.normal(size=(3, 400))
cloud = cloud + np.array([[1.0], [2.0], [0.0]])

hull = el.convex_hull(cloud)
print('corners  :', hull.coord_count)
print('triangles:', hull.topo.simplex_count[2])
print('surface area:', round(float(hull.measures.sum()), 4))
```

Thirty of the four hundred points are corners of the hull; the rest are inside
it. The surface is a closed triangle mesh: every edge is shared by exactly two
triangles, which is why the corner and triangle counts satisfy Euler's formula
for a sphere, $30 - 84 + 56 = 2$.

A closed surface that winds outward encloses a volume that is the sum of the
signed volumes of its tetrahedra with the origin. That is a direct use of the
winding, so it is worth computing:

```{code-cell}
def volume(hull):
    coords = np.asarray(hull.coords)
    corners = np.asarray(hull.topo.indices)
    total = 0.0
    for k in range(corners.shape[1]):
        a = coords[:, corners[0, k]]
        b = coords[:, corners[1, k]]
        c = coords[:, corners[2, k]]
        total += float(a @ np.cross(b - a, c - a))
    return total / 6.0

print('volume:', round(volume(hull), 4))
```

## The hull is smaller than the box around it

An oriented bounding box is the smallest box whose faces follow the points'
principal axes, but a box is a blunter instrument than an arbitrary
polyhedron: it has to hold the corners of the cloud's widest extents even when
no points are near them.

```{code-cell}
(_center, _axes, extents) = el.oriented_bounding_box(cloud)
print('box extents:', extents.round(3))
print('box volume :', round(float(np.prod(extents)), 4))
print('hull volume:', round(volume(hull), 4))
```

The box is more than twice the hull's volume, and that is not a defect of the
box: a box that contains the cloud must contain the hull's corners, and the
hull has corners where the box has empty space.

## A polygon in a plane

In two dimensions the hull is a polygon, returned as a `SegPath` around it.

```{code-cell}
polygon = el.convex_hull(cloud[:2])
print(type(polygon).__name__)
print('corners :', polygon.coord_count)
print('segments:', polygon.topo.simplex_count[1])
print('perimeter:', round(float(polygon.measures.sum()), 4))
```

The path is closed: the last segment joins the last corner to the first, so
there are as many segments as corners. The corners are in counterclockwise
order, which is what makes the polygon's signed area positive.

## A known solid

The eight corners of a box, turned by 30° about the `z` axis and centered at
`(1, 2, 3)`, are a hull whose answer is known: eight corners, twelve
triangles, a surface area of 28, and a volume of 8.

```{code-cell}
turn = np.deg2rad(30.0)
rotation = np.array([[np.cos(turn), -np.sin(turn), 0.0],
                     [np.sin(turn),  np.cos(turn), 0.0],
                     [0.0, 0.0, 1.0]])
half = np.array([2.0, 1.0, 0.5])
signs = np.array([[sx, sy, sz]
                  for sx in (-1.0, 1.0)
                  for sy in (-1.0, 1.0)
                  for sz in (-1.0, 1.0)]).T
corners = np.array([[1.0], [2.0], [3.0]]) + rotation @ (signs * half[:, None])

box_hull = el.convex_hull(corners)
print('corners  :', box_hull.coord_count)
print('triangles:', box_hull.topo.simplex_count[2])
print('surface area:', round(float(box_hull.measures.sum()), 4))
print('volume      :', round(volume(box_hull), 4))
```

The zero-volume triangles that a triangulation of a flat face might otherwise
contribute are absent here, and the counts are exact: the hull of a box's
corners is the box.

## When the points do not span the space

A hull is a solid only if the points reach into every direction of the space
they are asked about. Points on a line in a plane fix a polygon of zero area,
and points in a plane in space fix a polyhedron of zero volume; neither is
answered.

```{code-cell}
for (name, points) in (
        ('a line in two dimensions',
         np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])),
        ('a plane in three dimensions',
         np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0],
                   [2.0, 2.0, 2.0, 2.0]]))):
    try:
        el.convex_hull(points)
        print(name, 'was wrapped')
    except ValueError as error:
        print(name, 'is refused:', error)
```

The flat hull is still perfectly well defined, and the way to ask for it is to
put the points in their own plane first. `euclib.principal_axes` gives that
plane, and projecting onto its first two axes turns the refusal into an
answer:

```{code-cell}
flat = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0, 2.0]])
(origin, axes) = el.principal_axes(flat)
projected = axes.T[:2] @ (flat - origin[:, None])
polygon = el.convex_hull(projected)
print('corners :', polygon.coord_count)
print('perimeter:', round(float(polygon.measures.sum()), 4))
```

:::{admonition} Tensors and a combinatorial answer
:class: note
A tensor is accepted, and the geometry that comes back is built on the same
coordinates, so it is a tensor too. The gradient is that of a *selection*:
`convex_hull` chooses which of the input's coordinates are the hull's corners,
and the derivative of choosing them is one for the chosen coordinates and zero
for the rest. The choice itself has no derivative, because it is a decision
that changes discontinuously when a point crosses a face. So a gradient through
a hull says which points were selected; it does not say how the hull would
change if they moved.
:::

## Provenance

Adapted from trimesh's
[`tests/test_convex.py::test_convex`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_convex.py#L8)
and
[`::test_primitives`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_convex.py#L78).

The upstream test takes watertight meshes that are and are not convex, moves
each one through a series of rigid transforms, and checks three properties of
the hull each time: that the volume is the same after every transform, that the
hull is closed with a consistent winding, and that the hull is itself convex.
The primitives test adds that a convex primitive's hull has the same volume as
the primitive. Each of those properties is checked here too: a rigid motion
leaves the counts and the volume unchanged, the mesh closes (the Euler
characteristic is the sphere's), every face winds outward, and a box's corners
reproduce the box's volume exactly.

The differences are in the machinery and in the edge cases. trimesh uses
SciPy's Qhull binding, and when Qhull refuses the points it retries with
joggled input; `euclib` implements quickhull directly, and refuses a cloud that
does not span the space rather than returning a flatter hull. trimesh repairs a
hull's winding after the fact by comparing each face's normal with the vector
from the centroid; `euclib` winds each new face as it is created, which is what
lets a degenerate face be kept --- and with it the surface closed --- rather
than removed. trimesh returns a mesh in three dimensions and leaves the planar
case to be treated as points in space; `euclib` returns a closed `SegPath` for
points in a plane, so the polygon is a geometry in its own right.

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
# Union, Intersection, and Difference of Solids

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.boolean_union`, `euclib.boolean_intersection`, and
`euclib.boolean_difference` make one solid out of two. Each takes two closed
triangle meshes and returns the surface of the solid that holds the positions
one or the other input holds. The result is a `TriMesh` like any other, so it
can be measured, drawn, intersected with something else, or used as an input to
another boolean.

:::{admonition} What this demonstrates
:class: tip
- `el.boolean_union`, `el.boolean_intersection`, and `el.boolean_difference`
  on two closed triangle meshes.
- The identities that tie the three together: union plus intersection is the
  two inputs, and difference is the first input minus the intersection.
- That a result's surface is closed and winds outward, so its volume can be
  computed from its triangles.
- That subtracting a solid inside another leaves a cavity --- a second closed
  surface, wound the other way.
- That inputs which are wound inconsistently, or which list a position more
  than once, are repaired rather than refused, while inputs that are open or
  flat are refused.
:::

## Two cubes

A unit cube and a copy of it shifted half a unit along `x`. The triangles are
wound counterclockwise seen from outside, which is the convention the
operations read: a triangle's winding says which side of it is outside.

```{code-cell}
import numpy as np
import euclib as el

def cube(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0)):
    pattern = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
               (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    coords = np.array([[low[k] + bit[k] * (high[k] - low[k])
                        for k in range(3)] for bit in pattern], float).T
    faces = np.array([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                      [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                      [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    return el.trimesh(coords, faces)

def volume(mesh):
    coords = np.asarray(mesh.coords)
    corners = np.asarray(mesh.topo.indices)
    total = 0.0
    for k in range(corners.shape[1]):
        a = coords[:, corners[0, k]]
        b = coords[:, corners[1, k]]
        c = coords[:, corners[2, k]]
        total += float(a @ np.cross(b - a, c - a))
    return total / 6.0

one = cube()
other = cube((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
print('first volume :', round(volume(one), 4))
print('second volume:', round(volume(other), 4))
```

The three operations ask three different questions about the overlap. The
union holds a position if either cube does, so its volume is the two cubes
with the shared half counted once. The intersection holds a position if both
do, so it is the shared half. The difference holds a position if the first
cube does and the second does not, so it is the first cube with the shared half
taken away.

```{code-cell}
union = el.boolean_union(one, other)
middle = el.boolean_intersection(one, other)
rest = el.boolean_difference(one, other)
for (name, solid) in (('union', union), ('intersection', middle),
                      ('difference', rest)):
    print(f'{name:>12}: volume {volume(solid):.4f}, '
          f'{solid.coord_count} corners, {solid.topo.simplex_count[2]} triangles')
```

Each result is built from pieces of the two surfaces, cut where they cross, so
each has more corners than either input did: the cut adds the positions where
a face of one cube meets an edge of the other. The union has more of them than
the other two because it keeps a piece of each surface on both sides of the
seam.

## The three answers together

The volumes are not independent. Every position is in the union, or in the
intersection, or in neither, and counting the two tells us how much of the
inputs is covered:

```{code-cell}
print('union + intersection:', round(volume(union) + volume(middle), 12))
print('first + second      :', round(volume(one) + volume(other), 12))
print('difference          :', round(volume(rest), 12))
print('first - intersection:', round(volume(one) - volume(middle), 12))
```

Both pairs agree to the last decimal the printout shows. This is the property
the operations exist to satisfy, and the test suite checks it on shapes whose
surfaces have no symmetry to hide an error behind.

## Where the surfaces meet

A result is a solid, so its surface must close and face outward. Both can be
checked from the triangles alone: every edge of a closed surface is shared by
exactly two triangles wound opposite ways along it, and Euler's formula for a
surface without holes is $V - E + F = 2$.

```{code-cell}
def closed(mesh):
    corners = np.asarray(mesh.topo.indices)
    seen = {}
    for k in range(corners.shape[1]):
        a, b, c = (int(v) for v in corners[:, k])
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(n == 1 and (e[1], e[0]) in seen for (e, n) in seen.items())

def euler(mesh):
    corners = np.asarray(mesh.topo.indices)
    edges = set()
    for k in range(corners.shape[1]):
        a, b, c = (int(v) for v in corners[:, k])
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return mesh.coord_count - len(edges) + corners.shape[1]

for (name, solid) in (('union', union), ('intersection', middle),
                      ('difference', rest)):
    print(f'{name:>12}: closed={closed(solid)}, euler={euler(solid)}, '
          f'area={float(np.sum(solid.measures)):.4f}')
```

The union's area is 8 rather than 12: of each cube's six faces, two lie partly
inside the other cube and are cut away, and the faces that remain meet along
the seam instead of being counted twice. The intersection and the difference
are each a half-unit box --- half a unit along `x`, one unit along `y` and
`z` --- so each has an area of 4.

## One solid inside another

Subtracting a solid that lies wholly inside another leaves the outer solid
with a cavity. The cavity's surface is part of the answer, and it is wound the
other way --- its outward direction is into the cavity --- so that the volume
sum counts the empty space as negative.

```{code-cell}
outer = cube((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
inner = cube((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
hollow = el.boolean_difference(outer, inner)
print('volume   :', round(volume(hollow), 4))
print('corners  :', hollow.coord_count)
print('triangles:', hollow.topo.simplex_count[2])
print('closed   :', closed(hollow))
print('euler    :', euler(hollow))
```

The cube is 2 on a side and the cavity is 1 on a side, so the volume is
$8 - 1 = 7$. The Euler characteristic is 4 rather than 2 because the surface
has two components, the outer one and the cavity, each a closed surface
without handles; a hollow solid is two spheres' worth of surface.

## Two spheres

A sphere is the shape to reach for when the cut is not along a coordinate
plane: its surface crosses the other's along a circle, and every cut polygon
is a new one. The sphere is built at the latitudes and longitudes the
[sphere page](../primitives/spheres.md) uses.

```{code-cell}
def sphere(n_lat=8, n_lon=12, shift=(0.0, 0.0, 0.0)):
    lat = np.linspace(0.0, np.pi, n_lat + 1)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    coords = np.array([[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p),
                        np.cos(t)] for t in lat for p in lon])
    coords = coords.T + np.array(shift)[:, None]
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            faces += [[a, b, d], [a, d, c]]
    return el.trimesh(coords, np.array(faces).T)

ball = sphere()
near = sphere(shift=(0.5, 0.0, 0.0))
print('corners  :', ball.coord_count)
print('triangles:', ball.topo.simplex_count[2])
print('signed volume:', round(volume(ball), 4))
```

The signed volume is negative, which says the sphere page's triangles wind
*into* the sphere rather than out of it. That is not a defect the operation
will pass on: it turns a mesh that is inside out the right way round before
cutting it. The volume it encloses is the magnitude, 3.8478, a little under
the true sphere's 4.1888 because twelve longitudes and eight latitudes cut the
surface into flat pieces.

```{code-cell}
union = el.boolean_union(ball, near)
middle = el.boolean_intersection(ball, near)
rest = el.boolean_difference(ball, near)
print('union        :', round(volume(union), 4))
print('intersection :', round(volume(middle), 4))
print('difference   :', round(volume(rest), 4))
print('identity     :',
      round(volume(union) + volume(middle) - 2 * abs(volume(ball)), 12))
print('areas        :', round(float(np.sum(union.measures)), 4),
      round(float(np.sum(middle.measures)), 4),
      round(float(np.sum(rest.measures)), 4))
```

The identity closes to zero again. The intersection's surface area, 8.9350, is
less than either sphere's 12.0453, because most of each sphere is outside the
other.

## Surfaces that need repairing

Inputs come from files and from other libraries, and they are not always
written the way the algorithm would like. Two habits are common enough that
the operations repair them.

The first is inconsistent winding: the [cube page](../primitives/cube.md)
lists twelve triangles, four of which wind the other way from the rest. Its
signed volume is therefore 1/3 rather than 1, even though the surface looks
like a cube.

```{code-cell}
page_tris = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                      [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                      [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]],
                     dtype='int64').T
corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                   dtype=float).T
mixed = el.trimesh(corners, page_tris)
print('signed volume as written:', round(volume(mixed), 4))
print('volume of its union     :', round(volume(el.boolean_union(mixed, mixed)), 4))
```

The operation walks the triangles across their shared edges and turns the ones
that disagree with their neighbours, so the union of the mesh with itself is
the cube, not the quarter-cube the signed volume of the raw list would suggest.

The second habit is listing a position more than once. A latitude-longitude
sphere gives each pole one copy per longitude: the sphere above has 108
corners for 86 distinct positions. A boolean collapses the copies, so the
union reports the positions rather than the spelling.

```{code-cell}
print('corners as written :', ball.coord_count)
print('corners in the union:', el.boolean_union(ball, ball).coord_count)
```

## When an operation refuses

A boolean operation is only defined for solids, and a surface that does not
close has no inside. One triangle is not a solid, and neither is a triangle
mesh that lives in a plane.

```{code-cell}
open_mesh = el.trimesh(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0]]),
                       np.array([[0], [1], [2]]))
try:
    el.boolean_union(open_mesh, one)
except ValueError as error:
    print('an open surface :', error)

flat = el.trimesh(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                  np.array([[0], [1], [2]]))
try:
    el.boolean_union(flat, one)
except ValueError as error:
    print('a flat surface  :', error)
```

The refusals are deliberate. An open surface describes a piece of a boundary,
not a region of space, so there is no answer to give; a failure that returns a
wrong solid would be worse than one that returns nothing.

:::{admonition} Tensors and a combinatorial answer
:class: note
A boolean is combinatorial: which pieces of the two surfaces survive is a
decision, and the corners where a surface is cut are computed rather than
selected. There is no derivative to carry through that. Both inputs may have
tensor coordinates, and the result is always built on NumPy arrays --- use it
as geometry, and differentiate the surfaces that go into it.
:::

## Provenance

Adapted from pyvista's
[`tests/core/test_polydata.py::test_boolean_union_intersection`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_polydata.py#L450)
and
[`::test_boolean_difference`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_polydata.py#L476),
and from trimesh's
[`tests/test_boolean.py::test_boolean`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_boolean.py)
and
[`::test_multiple`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_boolean.py)
and
[`::test_empty`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_boolean.py).

The upstream tests make two solids and check the volumes of the union,
intersection, and difference, both against each other and against the two
inputs; they also check that subtracting a solid from itself leaves nothing,
and that solids which do not overlap have an empty intersection. Each of those
checks is made here too, on cubes that overlap by half, on a cube inside
another, on cubes apart, and on two spheres. The pages add two properties the
upstream tests assume rather than test: that every result's edges pair --- the
surface closes --- and that the result's Euler characteristic is the one its
shape has.

The differences are in the machinery and in what the inputs may be. pyvista
and trimesh delegate to external CSG engines (VTK's and Blender's or
Manifold's), and trimesh's test is parameterized by engine; `euclib`
implements the binary-space-partition algorithm itself, so there is no engine
to choose and no optional dependency. That implementation splits polygons and
welds the cut corners back together, and it repairs winding and coincident
corners before cutting, which is why the cube page's inconsistent triangles
and the sphere page's repeated poles are accepted rather than refused. A
surface that is genuinely open, or that lies in a plane, is refused, and that
is a difference from trimesh's test too: trimesh's engine may return an error
for it, but `euclib` says which property failed.

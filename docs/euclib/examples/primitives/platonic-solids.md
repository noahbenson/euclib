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
# The Platonic Solids

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

A Platonic solid is a convex polyhedron whose faces are all the same regular
polygon and whose corners all meet the same number of faces. There are five,
and each is determined by its vertices: once the corners are placed, the
surface is their convex hull. `euclib.convex_hull` builds that surface as a
`TriMesh`, which lets the page check the counts, the areas and volumes, and the
symmetry that make the solids regular.

:::{admonition} What this demonstrates
:class: tip
- `el.convex_hull(points)` on the vertices of each of the five solids.
- The vertex, edge, and triangle counts, and Euler's characteristic
  $V - E + F = 2$ for every one of them.
- That every vertex lies on the unit sphere, and that the shortest distance
  between two vertices is the same for every such pair --- the solid's edge.
- That a symmetry rotation maps each solid's vertices onto themselves.
- That a name that is not one of the five is a lookup that fails rather than a
  solid that is invented.
:::

## The five sets of vertices

The tetrahedron, cube, and octahedron are placed on the axes and the cube's
corners. The icosahedron uses the golden ratio $\varphi = (1 + \sqrt{5})/2$,
whose powers place the vertices of both it and its dual, the dodecahedron.

```{code-cell}
import itertools
import numpy as np
import euclib as el

phi = (1.0 + np.sqrt(5.0)) / 2.0
inverse = 1.0 / phi
signs = (-1.0, 1.0)

tetrahedron = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                        [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]).T
cube = np.array(list(itertools.product(signs, repeat=3))).T
octahedron = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]])

icosahedron = []
for (s, t) in itertools.product(signs, repeat=2):
    icosahedron += [[0.0, s, t * phi],
                    [s, t * phi, 0.0],
                    [t * phi, 0.0, s]]
icosahedron = np.unique(np.array(icosahedron), axis=0).T

dodecahedron = [list(corner) for corner in itertools.product(signs, repeat=3)]
for (s, t) in itertools.product(signs, repeat=2):
    dodecahedron += [[0.0, inverse * s, phi * t],
                     [inverse * s, phi * t, 0.0],
                     [phi * t, 0.0, inverse * s]]
dodecahedron = np.unique(np.array(dodecahedron), axis=0).T

solids = {'tetrahedron': tetrahedron, 'cube': cube,
          'octahedron': octahedron, 'icosahedron': icosahedron,
          'dodecahedron': dodecahedron}
for (name, points) in solids.items():
    print(f'{name:>13}: {points.shape[1]} vertices')
```

## The hull of each set

`convex_hull` returns the smallest convex surface holding the points. Because
the points are a Platonic solid's vertices, that surface is the solid itself.
The coordinates are scaled to the unit circumsphere first, so all five have the
same distance from center to corner.

```{code-cell}
def unit(points):
    return points / np.sqrt((points * points).sum(axis=0))

def euler(mesh):
    return (mesh.coord_count - mesh.topo.simplex_count[1]
            + mesh.topo.simplex_count[2])

meshes = {name: el.convex_hull(unit(points))
          for (name, points) in solids.items()}
for (name, mesh) in meshes.items():
    print(f'{name:>13}: V={mesh.coord_count:>2} '
          f'E={mesh.topo.simplex_count[1]:>2} '
          f'T={mesh.topo.simplex_count[2]:>2} '
          f'V-E+T={euler(mesh):>2} '
          f'area={float(np.sum(mesh.measures)):.6f}')
```

The triangle count is not the solid's face count when a face is a pentagon:
the dodecahedron's twelve pentagons are triangulated into thirty-six
triangles, and its thirty polyhedron edges plus twenty-four face diagonals
make the fifty-four edges of the triangle mesh. The Euler characteristic is 2
in every row, and the area is the surface area of the unit-circumradius solid.

## Equal edges

A solid is regular when its edge lengths are equal. The shortest distance
between two vertices is an edge; every other pair is a face or body diagonal
and is longer. Counting the pairs at the shortest distance gives the solid's
edge count.

```{code-cell}
def edges(points):
    (dim, count) = points.shape
    lengths = []
    for (i, j) in itertools.combinations(range(count), 2):
        step = points[:, i] - points[:, j]
        lengths.append(float(np.sqrt(step @ step)))
    lengths = np.array(lengths)
    shortest = lengths.min()
    return (int(np.isclose(lengths, shortest, atol=1e-12).sum()), shortest)

for (name, points) in solids.items():
    (count, length) = edges(unit(points))
    print(f'{name:>13}: {count:>2} edges of length {length:.6f}')
```

The counts are the Platonic solids' own: 6, 12, 12, 30, and 30 edges. The
edge lengths differ from solid to solid because all five are scaled to the same
circumradius rather than to the same edge length.

## Symmetry

A Platonic solid is unchanged by a rotation that carries a corner to another
corner and a face to another face. Rotating the cube by 120° about a body
diagonal, and the icosahedron by 72° about the axis through one of its
vertices, permutes the vertex sets and leaves the solids looking the same.

```{code-cell}
def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * (cross @ cross))

def permutes(points, turn):
    moved = turn @ points
    return all(any(np.allclose(moved[:, i], points[:, j], atol=1e-12)
                   for j in range(points.shape[1]))
               for i in range(points.shape[1]))

for (name, axis, angle) in (
        ('tetrahedron', unit(tetrahedron)[:, 0], 2.0 * np.pi / 3.0),
        ('cube', np.ones(3), 2.0 * np.pi / 3.0),
        ('octahedron', np.array([1.0, 1.0, 1.0]), 2.0 * np.pi / 3.0),
        ('icosahedron', unit(icosahedron)[:, 0], 2.0 * np.pi / 5.0),
        ('dodecahedron', unit(dodecahedron)[:, 0], 2.0 * np.pi / 3.0)):
    turn = rotation(axis, angle)
    print(f'{name:>13}: {permutes(unit(solids[name]), turn)}')
```

Each `True` says the rotated vertices are the original vertices in a different
order: the rotation is a symmetry of the set, and therefore of its hull.

:::{admonition} Why there are five
:class: note
At every corner of a convex polyhedron, the faces that meet there must leave
an angle less than a full turn; otherwise the surface would be flat or would
fold through itself. The faces are regular $p$-gons, so each contributes an
interior angle of $(p - 2)\pi/p$, and the number $q$ of faces at a corner must
satisfy $q(p - 2)/p < 2$. The integer solutions are the five solids. Euler's
formula $V - E + F = 2$ then fixes their counts: the values in the table above
are not a coincidence of these particular coordinates but the only counts a
closed convex surface of regular polygons can have.
:::

## A name that is not one of the five

There is no constructor here that takes a kind and returns a solid; the
coordinates are data, and the five names are a dictionary. A caller who asks
for a sixth finds that the dictionary has no such key, which is the same
refusal pyvista gives for an invalid kind --- made by the lookup rather than by
a validator.

```{code-cell}
try:
    solids['sphere']
except KeyError as error:
    print('no such Platonic solid:', error)
```

## Provenance

Adapted from pyvista's
[`tests/core/test_geometric_objects.py::test_platonic_solids`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L1092)
and
[`::test_platonic_invalids`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L1103).

The upstream test is parameterized over the five solids and their integer
aliases: it builds each by name and by number, checks that the two agree, and
checks the vertex and face counts. The companion test asks for an invalid name,
an invalid number, and an invalid list, and expects a `ValueError` from each.
The page makes the same count checks on surfaces built from the solids'
vertices, and adds the properties the upstream test does not state: every
vertex on one sphere, equal edge lengths, Euler's characteristic, the area of
each, and a symmetry rotation that permutes each solid's vertices.

The difference is in the interface. pyvista has a `PlatonicSolid` constructor
that looks a solid up by name or number and returns a `PolyData`; `euclib` has
no primitive generator yet --- it is on the [roadmap](../roadmap.md) --- so the
page supplies the vertices and takes their convex hull. That makes the
dependence on the coordinates explicit, and it uses the same `convex_hull`
operation a caller would use on measured points. pyvista's faces are polygons
with the solid's own face count; `euclib`'s are triangles, so a pentagonal face
is triangulated in the mesh and the polygon count is reported in prose rather
than as a field.

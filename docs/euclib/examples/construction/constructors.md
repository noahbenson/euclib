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
# One-Line Constructors

Every geometry in `euclib` can be built from its coordinates and a topology, but
writing out a topology object each time is tedious. The library therefore
exposes one constructor per geometry type that builds the topology for you:
`euclib.points`, `euclib.segpath`, `euclib.trimesh`, `euclib.tetmesh`,
`euclib.prismmesh`, and `euclib.grid`. This page is the tour of all six.

:::{admonition} What this demonstrates
:class: tip
- The one-line constructor for each of the six geometry types, and the argument
  each one expects.
- How the constructor's implicit choices (inferring the topology from the corner
  matrix, connecting a path's points in order, defaulting a grid's affine to the
  identity) keep the common case short.
- That every constructor returns a fully immutable geometry, identical to the
  one the explicit `Topology`-plus-geometry route builds.
:::

## Points, segments, and triangles

The three simplex constructors that need only a coordinate matrix take one
argument: the coordinates. A path is connected in the order the coordinates are
given, so a closed loop is written by repeating the first point at the end.

```{code-cell}
import numpy as np
import euclib as el

cloud = el.points(np.array([[0.0, 1.0, 0.5, 1.5],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]]))

# A path joins consecutive coordinates; 4 points give 3 segments.
path = el.segpath(np.array([[0.0, 1.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0]]))

print('cloud :', type(cloud).__name__, cloud.topo.vertex_count, 'points')
print('path  :', type(path).__name__, path.topo.simplex_count[1], 'segments,',
      'length', round(float(np.sum(path.measures)), 4))
```

A triangle mesh and a tetrahedral mesh take the coordinates and the corner
matrix, whose row count is what selects the simplex: three rows for triangles,
four for tetrahedra.

```{code-cell}
corners = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])
tri = el.trimesh(corners, np.array([[0], [1], [2]], dtype='int64'))
print('trimesh:', tri.topo.simplex_count[2], 'triangle,',
      'area', round(float(np.sum(tri.measures)), 4))

corners4 = np.array([[0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])
tet = el.tetmesh(corners4, np.array([[0], [1], [2], [3]], dtype='int64'))
print('tetmesh:', tet.topo.simplex_count[3], 'tetrahedron,',
      'volume', round(float(np.sum(tet.measures)), 6))
```

## Prisms and grids

The remaining two constructors differ because their inputs do. A prism mesh
takes a pair of surfaces whose coordinates are stacked along a leading axis; a
grid takes a shape and an optional affine matrix, defaulting to the identity so
that a cell's index is its position.

```{code-cell}
bottom = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
top = bottom + np.array([[0.5], [0.0], [1.5]])
sheet = el.prismmesh(np.array([bottom, top]), np.array([[0], [1], [2]], dtype='int64'))
print('prismmesh: coords', np.shape(sheet.coords),
      'volume', round(float(np.sum(sheet.measures)), 4))

image = el.grid((4, 5))
print('grid (identity affine): shape', image.shape,
      'origin', image.origin, 'spacing', image.spacing)

scaled = el.grid((4, 5), np.array([[2.0, 0.0, 1.0],
                                   [0.0, 3.0, 1.0],
                                   [0.0, 0.0, 1.0]]))
print('grid (given affine)   : origin', scaled.origin, 'spacing', scaled.spacing)
```

## Properties at construction time

Every constructor accepts a `properties` mapping, so a geometry can be born
carrying its values rather than being given them afterward. The values are
`Property` objects, whose second argument is the shape of the positions they
belong to — here the three coordinates of the point cloud:

```{code-cell}
from euclib.abc import Property

points_xy = np.array([[0.0, 1.0, 1.5], [0.0, 0.0, 1.0]])
labelled = el.points(points_xy,
                     properties={'height': Property(np.array([0.0, 0.0, 1.0]),
                                                    (points_xy.shape[1],))})
print('properties:', sorted(labelled.properties.keys()),
      '->', np.asarray(labelled['height']))
```

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(tri, color_by='z', name='constructors')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
functions `test_line` (line 665) and `test_cube` (line 751); and
`trimesh/trimesh` — `tests/test_creation.py`, function `test_box` (line 9).

**Pinned revisions:** pyvista
[`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L665);
trimesh
[`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_creation.py#L9).

**How this was adapted.** The upstream tests exercise each library's primitive
constructors — `pv.Line`, `pv.Cube`, and `g.trimesh.creation.box` — checking
that the object they return has the expected size, bounds, and cell counts. No
code is copied. The adaptation makes the same point for `euclib`: that each
geometry type has a single call that builds it, and that the call infers the
connectivity the caller would otherwise have to write. The upstream
constructors take *shape* parameters (a length, a radius, a number of
subdivisions); `euclib`'s take *coordinates*, because it does not generate
geometry for you. That difference is why the page builds each object from
explicit arrays rather than from a size, and why the checks are on the resulting
simplex counts and measures.
:::

:::{seealso}
- [A cube as a triangle mesh](../primitives/cube.md), which builds the same
  object through the explicit topology route.
- [Image grids from an array and an affine](../grids/grids-from-arrays.md) for
  the grid affine in more detail.
:::

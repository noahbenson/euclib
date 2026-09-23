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
# A Cube as a Triangle Mesh

A cube is the simplest object that shows how `euclib` separates *where* a
geometry's corners are from *how* those corners are connected. This page builds
a unit cube as a `euclib.types.TriMesh`, attaches a property to its corners, and
renders it.

:::{admonition} What this demonstrates
:class: tip
- Building a geometry from a coordinate matrix (`(D, N)`) and a topology
  object that stores the triangle corners (`(3, M)`).
- `TriTopology`'s implied simplices: a triangle mesh also knows its edges and
  its vertices, not just its triangles.
- Attaching a corner property and coloring the figure by it.
- Immutability: `withprop` returns a *new* mesh rather than modifying the old
  one.
:::

## Constructing the cube

`euclib` geometries take two separate objects: the coordinates (a `(3, N)`
matrix for a 3D geometry) and a topologies that says which coordinates form
each simplex. For a cube we write the eight corners down explicitly and list
the twelve triangles.

```{code-cell}
import numpy as np
import euclib as el

# The eight corners of the unit cube, as a (3, 8) coordinate matrix.
corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float).T

# Twelve triangles: two per face, as a (3, 12) matrix of corner indices.
tris = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                 [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                 [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]], dtype='int64').T
```

The coordinate matrix and the triangle matrix are separate because many
geometries share a topology but differ in their coordinates — the
[prism mesh](../index.md) is built entirely around that idea. Combining them
gives a `TriMesh`:

```{code-cell}
cube = el.trimesh(corners, tris)
cube
```

Because only the *triangles* are stored, the edges and vertices are implied.
`euclib` computes them lazily and caches them, so asking for the simplex counts
is cheap after the first call:

```{code-cell}
import numpy as np
counts = cube.topo.simplex_count
print(f'vertices: {counts[0]}, edges: {counts[1]}, triangles: {counts[2]}')
```

A cube has 8 corners and 12 edges; each of its 6 square faces is split by a
diagonal, giving 6 more edges and 12 triangles in total.

## Properties

A property is a named array attached to a geometry. A corner (coordinate)
property has one value per coordinate, so its final dimension must equal the
number of coordinates. Here we record each corner's height above the base:

```{code-cell}
cube = cube.withprop('height', corners[2])
cube['height']
```

Note that `withprop` returned a new mesh: `cube` is now the property-carrying
copy, while the original object was left untouched. This is `euclib`'s
immutability in practice — an object is never modified in place, and an updated
copy shares all of the untouched, lazily-computed data with the old one.

## Figure

The figure helper uses `k3d` when it is running in a live notebook and a static
matplotlib render otherwise, so the published page always shows something:

```{code-cell}
import pathlib
import sys

# The figure helper lives at the site root (the directory holding myst.yml).
_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(cube, color_by='height', name='cube-height')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
function `test_cube` (line 751).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L751).

**How this was adapted.** The upstream test calls `pv.Cube()`, checks that
`cube.bounds` equals the bounds it was given, and checks that the face normals
are unit diagonals. No upstream code is copied here. `euclib` has no
center-and-length box constructor, so the eight corners and twelve triangles
are written out explicitly instead, and the upstream bounds assertion is
re-expressed as a printout of the simplex counts. The upstream `pv.Plotter`
screenshot is replaced by `euclib_viz.show3d`, and the color-by-property
demonstration is new: it shows the property API, which `pv.Cube()` has no
analogue for.
:::

:::{seealso}
- [The example gallery](../index.md) for the other pages, including the
  [grids and images](../grids/grids-from-arrays.md) page.
- [Operations that euclib does not implement yet](../roadmap.md), such as
  boolean operations and convex hulls, which many mesh libraries demonstrate
  around a cube.
:::

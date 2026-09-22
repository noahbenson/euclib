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
# Measures: Lengths, Areas, and Volumes

The same question — "how big is this?" — has a different answer for each
dimensionality of simplex. A segment has a length, a triangle an area, and a
tetrahedron a volume. `euclib` answers all three with one mechanism: every
geometry reports one measure per simplex, and the total is their sum.

:::{admonition} What this demonstrates
:class: tip
- `SegPath.measures`, `TriMesh.measures`, and `TetMesh.measures`, which always
  return one value *per simplex*, not a total.
- Summing measures to get a length, surface area, or volume.
- How measures are computed from coordinates and topology together, so the same
  topology with different coordinates has different measures.
- Checking a measure against a hand-computed value.
:::

## Lengths

A segment path measures its segments:

```{code-cell}
import numpy as np
import euclib as el
from euclib import types as et, utils

# A path through three points: two segments of length 3 and 4.
pts = np.array([[0.0, 3.0, 3.0], [0.0, 0.0, 4.0]])
path = et.SegPath(pts, et.SegTopology(np.array([[0, 1], [1, 2]], dtype='int64')))

print('segment lengths:', path.measures)
print('total length   :', float(np.sum(path.measures)))
```

## Areas

A triangle mesh measures its triangles. Here a unit right triangle has area
`1/2`, and joining two of them into a square doubles it:

```{code-cell}
# One triangle, then the same triangle mirrored into a unit square.
right = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
tri_top = np.array([[0], [1], [2]], dtype='int64')
one = et.TriMesh(np.vstack([right, [0.0, 0.0, 0.0]]), et.TriTopology(tri_top))
print('one triangle :', one.measures, '-> area', float(np.sum(one.measures)))

square_coords = np.array([[0.0, 1.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 0.0]])
square_top = et.TriTopology(np.array([[0, 0], [1, 2], [2, 3]], dtype='int64'))
square = et.TriMesh(square_coords, square_top)
print('two triangles:', square.measures, '-> area', float(np.sum(square.measures)))
```

## Volumes

Tetrahedra measure volume, and a tetrahedron's volume is one sixth of the
triple product of the three edges from one corner:

```{code-cell}
corners = np.array([[0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
tet = et.TetMesh(corners, et.TetTopology(np.array([[0], [1], [2], [3]], dtype='int64')))
print('corner tetrahedron volume:', float(np.sum(tet.measures)))
print('expected (1/6)           :', 1 / 6)
```

## The same topology, different coordinates

Measures are a function of *both* the coordinates and the topology. Stretching
the triangle's coordinates in one direction changes the area while leaving the
topology identical:

```{code-cell}
stretched = et.TriMesh(np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
                       et.TriTopology(tri_top))
print('original area :', float(np.sum(one.measures)))
print('stretched area:', float(np.sum(stretched.measures)))
print('topologies equal:',
      np.array_equal(one.topo.indices, stretched.topo.indices))
```

:::{admonition} The kernel underneath
:class: tip
The geometry methods are thin wrappers over `euclib.utils.simplex_measures`,
which takes a coordinate matrix and a simplex-index matrix directly. That
separation is what lets the same code measure a geometry's coordinates without
constructing the geometry:

```python
>>> utils.simplex_measures(square_coords, square_top.indices)
```
:::

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_cell_lengths.py`, function
`test_cell_edge_lengths` (line 100).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_cell_lengths.py#L100).

**How this was adapted.** The upstream test computes per-cell edge lengths for
about seventeen mesh types — spheres, image and rectilinear grids, pointsets,
and empty meshes — and compares them against a hand-rolled reference
implementation, largely to exercise the many cell types VTK supports. No code
is copied. `euclib` has no VTK-style zoo of cell types, so the adaptation keeps
the *principle* (a measure reported per simplex, comparable to a reference
value) and reduces it to the three simplex dimensions `euclib` actually has:
segments, triangles, and tetrahedra. The upstream parametrization over mesh
types is replaced by the unit-triangle and unit-tetrahedron checks, whose exact
values are known by hand, and by the stretched-coordinate case that shows
measures depending on coordinates as well as topology.
:::

:::{seealso}
- [Distance, separation, and containment](distance-and-separation.md) for
  *between*-geometry queries rather than within-geometry measures.
- [Tetrahedral meshes](../primitives/tetrahedral-meshes.md) for volumes at
  mesh scale.
:::

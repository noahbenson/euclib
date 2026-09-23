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
# Prism Meshes: Sweeping a Sheet between Two Surfaces

A prism mesh is `euclib`'s type for a thin sheet — two triangle meshes that
share one topology but have different coordinates. It is the natural
representation of a layered structure such as a sheet of metal or the cerebral
cortex, where the two faces of the sheet are the same surface seen from either
side. This page builds a curved sheet as a `euclib.types.PrismMesh`, reads
intermediate surfaces out of it, and converts it to tetrahedra.

:::{admonition} What this demonstrates
:class: tip
- `PrismMesh`'s two-sided coordinate matrix of shape `(2, D, N)`.
- Why the two faces share a topology rather than each owning one.
- `PrismMesh.elevation(h)`: the surface at a height between the two sides,
  computed as `coords0 * (1 - h) + coords1 * h`.
- `PrismMesh.to_tetmesh()`: decomposing each prism into three tetrahedra so a
  volume-based algorithm can consume it.
- Prism measures, which are per-prism volumes.
:::

## A curved sheet

The sheet is a ribbon two points wide and `n` points long. Each face is the
same grid of `2 * n` points; only their heights differ, with the second face
bowed upward. The topology records only the triangle corners, exactly as for a
flat mesh, and it is shared by both faces:

```{code-cell}
import numpy as np
import euclib as el

n, width = 6, 0.8
x = np.linspace(0, 3, n)
grid_x, grid_y = np.meshgrid(x, [0.0, width])       # (2, 2n) parameter grid
N = grid_x.size                                      # 2 * n points per face

bottom = np.array([grid_x.ravel(), grid_y.ravel(), np.zeros(N)])
top = np.array([grid_x.ravel(), grid_y.ravel(),
                1.0 + 0.3 * np.sin(grid_x.ravel())])

# Triangle corners over the 2n points of a face; the two faces share this.
faces = []
for i in range(n - 1):
    a, b, c, d = i, i + 1, n + i, n + i + 1
    faces += [[a, b, d], [a, d, c]]
corners = np.array(faces, dtype='int64').T

sheet = el.prismmesh(np.array([bottom, top]), corners)
sheet
```

The important structural fact is that `sheet.coords` has shape `(2, 3, N)`: the
two faces, not the two sides of each prism:

```{code-cell}
print('coords shape:', np.shape(sheet.coords))
print('triangles   :', sheet.topo.simplex_count[2])
print('prism volumes:', np.round(sheet.measures, 4))
print('total volume :', round(float(np.sum(sheet.measures)), 4))
```

## Intermediate surfaces

`elevation(h)` returns a `TriMesh` at height `h` between the two faces, with
`h = 0` giving the first face and `h = 1` the second. Because the two faces
share a topology, the intermediate surface needs only new coordinates:

```{code-cell}
for h in (0.0, 0.5, 1.0):
    surf = sheet.elevation(h)
    print(f'elevation {h}: type={type(surf).__name__}, '
          f'mean height={np.mean(surf.coords[2]):.3f}')
```

The halfway surface is the average of the two faces, which is worth checking
directly:

```{code-cell}
mid = sheet.elevation(0.5)
print('mid surface is the mean of the faces:',
      np.allclose(mid.coords, (bottom + top) / 2))
```

## From prisms to tetrahedra

Each prism decomposes into three tetrahedra, so a `PrismMesh` converts to a
`TetMesh` whose volume algorithms can be used. The total volume is unchanged:

```{code-cell}
tet = sheet.to_tetmesh()
print(f'{tet.topo.simplex_count[3]} tetrahedra')
print('tet volume :', round(float(np.sum(tet.measures)), 4))
print('prism volume:', round(float(np.sum(sheet.measures)), 4))
```

This conversion is how a layered structure written as prisms becomes usable by
algorithms that expect a volumetric mesh.

## Figure

The figure draws *both* faces of the sheet — the prism's two coordinate
matrices — as one two-sheet triangle mesh, so that both sides are visible:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

tri = np.asarray(corners)
both_faces = el.trimesh(
    np.concatenate([bottom, top], axis=1),
    np.concatenate([tri, tri + N], axis=1))
euclib_viz.show3d(both_faces, color_by='z', name='prism-sheet')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
function `test_tube` (line 705); and `trimesh/trimesh` —
`tests/test_sweep.py`, function `test_simple_extrude` (line 39).

**Pinned revisions:** pyvista
[`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L705);
trimesh
[`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_sweep.py#L39).

**How this was adapted.** The pyvista test sweeps a polygonal cross-section
along a line into a capped tube and inspects the resulting point and cell
counts; the trimesh test sweeps a circle along a straight path and checks that
the volume equals the circle's area times the height. No code is copied. Both
upstream tests produce a *closed* swept solid, whereas `euclib`'s `PrismMesh`
is deliberately built for *open* two-sided sheets, so the adaptation keeps the
sweep and drops the caps: it sweeps an open strip between two curves rather
than a closed polygon, and dispenses with the watertightness that the upstream
tests rely on. The upstream volume check is retained as the prism-total and
tetrahedron-total comparison, and the `elevation`/`to_tetmesh` demonstrations
have no upstream counterpart — they show the two features that make
`PrismMesh` more than a thin `TriMesh`.
:::

:::{seealso}
- [Surfaces of revolution](revolution.md) for the special case of sweeping
  around a circle.
- [Tetrahedral meshes](tetrahedral-meshes.md) for the volumetric type the
  prism conversion produces.
:::

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
# Resampling a Volume onto a Grid

A tetrahedral mesh and an image grid can describe the same region in different
ways, and it is often useful to express one in terms of the other: which voxels
does a piece of this volume fall in, and how much of it? `euclib.voxel_intersections`
answers by cutting every tetrahedron against every voxel it reaches and returning
the shared pieces as tetrahedra.

:::{admonition} What this demonstrates
:class: tip
- `euclib.voxel_intersections(mesh, grid)`, which returns three things: the
  pieces as a `TetMesh`, the source tetrahedron of each piece, and the grid cell
  each piece came from.
- That a volume measured from the pieces agrees with the original even as the
  grid cuts it into more and smaller pieces.
- Using grid cell indices to relate a piece back to a voxel.
- Why the cut is done in the grid's index space: an affine grid may be rotated
  or sheared, and index space turns every voxel into the same unit box.
:::

## A volume and a grid

The volume is the unit cube, split into six tetrahedra. The grid is the identity
grid at the cube's own scale, so its first cell is exactly the cube.

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1]], dtype=float)
tets = np.array([[0, 1, 2, 5], [0, 2, 5, 6], [0, 1, 5, 4],
                 [0, 3, 6, 7], [0, 4, 5, 7], [0, 5, 6, 7]], dtype='int64').T
solid = el.tetmesh(corners, tets)
print('the solid is', solid.topo.simplex_count[3], 'tetrahedra,',
      'volume', round(float(np.sum(solid.measures)), 6))

grid = el.grid((2, 2, 2))
print('the grid is', grid.shape, 'with origin', grid.origin,
      'and spacing', grid.spacing)
```

## The pieces of the overlap

```{code-cell}
pieces, tetrahedra, voxels = el.voxel_intersections(solid, grid)
print('pieces   :', pieces.topo.simplex_count[3])
print('volume   :', round(float(np.sum(pieces.measures)), 6))
print('source tetrahedron of each piece:', tetrahedra)
print('grid cell of each piece (columns):', voxels.T.tolist())
```

Every piece comes from cell `(0, 0, 0)`, because that cell is exactly the cube:
the overlap is the whole solid, so nothing is cut away.

## Refining the grid

A finer grid cuts the same volume into more, smaller pieces. The count explodes,
but the total volume does not change — the pieces still tile the solid exactly.

```{code-cell}
for size in (2, 4, 8):
    step = 1.0 / size
    affine = np.array([[step, 0, 0, 0],
                       [0, step, 0, 0],
                       [0, 0, step, 0],
                       [0, 0, 0, 1.0]])
    fine = el.grid((size, size, size), affine)
    cut, _, cells = el.voxel_intersections(solid, fine)
    print(f'{size}x{size}x{size} grid: {cut.topo.simplex_count[3]:4d} pieces, '
          f'volume {float(np.sum(cut.measures)):.6f}, '
          f'{np.unique(cells, axis=1).shape[1]} voxels touched')
```

## Which voxel a piece came from

The third return value names the grid cell of each piece, so a piece can be
attributed to a voxel — the operation that makes it possible to carry a value
from a volume mesh onto an image, or the reverse:

```{code-cell}
affine = np.array([[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1.0]])
coarse = el.grid((2, 2, 2), affine)
cut, _, cells = el.voxel_intersections(solid, coarse)

import collections
tally = collections.Counter(map(tuple, cells.T.tolist()))
for cell, count in sorted(tally.items()):
    print(f'voxel {cell}: {count} pieces')
```

## Figure

The tetrahedral mesh is shown by its bounding surface — the cube's twelve
triangles — since a figure draws surfaces rather than volumes:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                  [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                  [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]], dtype='int64').T
euclib_viz.show3d(el.trimesh(corners, faces), color_by='z', name='mesh-grid')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_dataset_filters.py`, function
`test_delaunay_3d_grid` (line 1995); and `trimesh/trimesh` —
`tests/test_voxel.py`, method `test_voxel` (line 11).

**Pinned revisions:** pyvista
[`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L1995);
trimesh
[`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_voxel.py#L11).

**How this was adapted.** The pyvista test fills an image grid with tetrahedra
and asserts only that the result has cells; the trimesh test voxelizes a mesh
and checks that the filled volume is positive, that the voxel count grows when
the surface is filled, and that points inside report as filled while a point
outside does not. No code is copied. The adaptation takes the two tests'
meeting point — a volume and a grid over the same space — and keeps their
central invariants, volume and cell membership, while dropping the parts
`euclib` does not offer: there is no voxel *fill* method, so the direction shown
is the mesh-driven one (`voxel_intersections`), and no marching cubes, so the
pieces stay tetrahedra rather than becoming a surface. The volume-versus-
resolution table and the per-voxel piece tally are the additions; they make
visible the two things the upstream tests assert separately — that the pieces
are a decomposition, and that they belong to specific cells.
:::

:::{seealso}
- [Tetrahedral meshes and volumes](../primitives/tetrahedral-meshes.md) for the
  volume geometry this page cuts up.
- [Image grids from an array and an affine](../grids/grids-from-arrays.md) for
  the grid type, including its affine.
:::

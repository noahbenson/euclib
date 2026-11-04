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
# Integrating a Volume's Field over a Grid

[The previous page](mesh-grid.md) cuts a tetrahedral mesh against a grid and
returns the pieces. That answers *which voxels a volume falls in*; this answers
the question that usually follows — *how much of a field is in each voxel*.
`euclib.integrate` reduces the pieces to one total per voxel, using the
property's own interpolation rather than its value at a point.

:::{admonition} What this demonstrates
:class: tip
- `euclib.integrate(mesh, grid, values)`, which returns a grid-shaped array of
  per-voxel totals.
- That the total over the grid is the mesh's own volume, whatever the grid —
  the grid decides how the volume is divided up, not how much of it there is.
- That a *linear* field's integral is exact: the integral of `x` over the unit
  cube is its volume times the mean of `x`, and the answer is `0.5` to the last
  place.
- `emptyconst`, which says what the part of a voxel the mesh does not reach
  contributes.
- That `integrate` takes a `TriMesh` and a `SegPath` too, where the totals are
  areas and lengths rather than volumes.
:::

## A volume, a grid, and a field

The volume is the unit cube, split into six tetrahedra — the same solid the
previous page uses. The grid is the identity grid at the cube's own scale, so
each of its cells is a unit cube.

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1]], dtype=float)
tets = np.array([[0, 1, 2, 5], [0, 2, 5, 6], [0, 1, 5, 4],
                 [0, 3, 6, 7], [0, 4, 5, 7], [0, 5, 6, 7]], dtype='int64').T
solid = el.tetmesh(corners, tets)
grid = el.grid((3, 3, 3), np.eye(4))

print('the solid is', solid.topo.simplex_count[3], 'tetrahedra')
print('its volume is', float(np.sum(solid.volume)))
```

## The simplest integral

Integrating the constant `1` over each voxel gives the volume of the mesh that
lies in it. Summed over the grid, that is the mesh's volume — the cells far from
the cube contribute nothing.

```{code-cell}
per_voxel = el.ops.integrate(solid, grid, 1)
print('the totals are shaped like the grid:', per_voxel.shape)
print('the whole grid holds  :', float(np.sum(per_voxel)))
print('...which is the volume:', float(np.sum(solid.volume)))
print('voxels the cube reaches:', int(np.sum(per_voxel > 0)), 'of', per_voxel.size)
```

## Integrating a field

A property is integrated using its *interpolation*, not its value at the
voxel's centre. For a linear field the two coincide, and the answer is exactly
the mesh's volume times the field's mean.

```{code-cell}
x = solid.withprop('x', np.asarray(solid.coords)[0],
                   interp=('polynomial', 1))
total = float(np.sum(el.ops.integrate(x, grid, 'x')))
print('integral of x        :', total)
print('volume times mean(x) :', float(np.sum(solid.volume)) * 0.5)
```

Higher orders are integrated just as exactly. A quadratic is determined by a
tetrahedron's values and its corners' gradients, so supplying those makes the
fit the field itself — and its integral is the closed form.

```{code-cell}
coords = np.asarray(solid.coords)
gradient = np.zeros((3, solid.coord_count))
gradient[0] = 2 * coords[0]
square = solid.withprop('sq', coords[0] ** 2, gradient=gradient,
                        interp=('polynomial', 2))
print('integral of x^2      :', float(np.sum(el.ops.integrate(square, grid, 'sq'))))
print('the cube is one unit :', 'the exact value is 1/3')
```

## What the empty part of a voxel contributes

By default the part of a cell the mesh does not reach contributes nothing. A
value for `emptyconst` fills it, which is also how a caller asks a different
question — *how much of each voxel is *not* filled* is `emptyconst=1` minus the
totals above.

```{code-cell}
filled = el.ops.integrate(solid, grid, 1, emptyconst=1)
print('every cell accounted for:', float(np.sum(filled)))
print('...which is the cells    :', int(np.prod(grid.shape)))
```

:::{admonition} Areas and lengths
:class: note
`integrate` takes a `TriMesh` and a `SegPath` as well, and then a cell's total is
an *area* or a *length* rather than a volume. `emptyconst` is refused for those
— the empty part of a voxel is a volume, and a surface has none to leave empty —
except for a triangle mesh in a plane over a plane grid, where both are areas
and it means what it says.
:::

## Provenance

Adapted from pyvista's
[`test_dataset_filters.py::test_delaunay_3d_grid`](https://github.com/pyvista/pyvista/blob/main/tests/test_dataset_filters.py)
and trimesh's
[`test_voxel.py::test_voxel`](https://github.com/mikedh/trimesh/blob/main/tests/test_voxel.py).

Both libraries voxelize a volume: pyvista by tetrahedralizing a grid and
trimesh by filling voxels the mesh passes through. Neither integrates a *field*
over the cells — the quantity `euclib.integrate` reports. The example keeps the
same unit cube and grid so that the totals can be checked against the volume
directly, which is the check both upstream tests make in their own way.

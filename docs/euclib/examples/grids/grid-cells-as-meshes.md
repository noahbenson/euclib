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
# Grid Cells as Triangles and Tetrahedra

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

A grid stores its cells by their centers and an affine matrix, not by their
corners. `euclib.Grid.to_trimesh` and `euclib.Grid.to_tetmesh` turn those cells
into a mesh of simplices: each four-sided cell becomes two triangles in a
two-dimensional grid, and each six-sided cell becomes six tetrahedra in a
three-dimensional one. The mesh's coordinates are the grid's *corners*, so
adjacent cells share them exactly, and the cells meet without cracks.

:::{admonition} What this demonstrates
:class: tip
- `grid.to_trimesh()` on a two-dimensional grid, and the cell count, corner
  count, and area of the result.
- `grid.to_tetmesh()` on a three-dimensional grid, and the same three numbers
  for a volume.
- That the corners follow the grid's affine, including a grid that is scaled,
  moved, or reflected.
- That the simplexes meet at shared corners and shared face diagonals, which is
  why no conformity pass is needed.
- That an input's units are carried through, and that a grid of the wrong
  dimension is refused.
:::

## A two-dimensional grid

A grid of 2 by 3 cells has 3 by 4 corners, and each cell is split into two
triangles, so the mesh has twelve triangles and an area of six --- the grid's
own region, measured from the corners rather than from the centers.

```{code-cell}
import numpy as np
import euclib as el

grid = el.grid((2, 3))
mesh = grid.to_trimesh()
print('corners  :', mesh.coord_count)
print('triangles:', mesh.topo.simplex_count[2])
print('area     :', round(float(np.sum(mesh.measures)), 12))
print('first corners, which are cell corners and not centers:')
print(np.asarray(mesh.coords)[:, :4])
```

The corners run from `-0.5` to `1.5` in the first axis and from `-0.5` to
`2.5` in the second: half a cell before the first center and half a cell past
the last, which is the region the grid describes. Each cell's diagonal runs
from its low corner to its high one, so the split is the same however the cell
is reached.

## A three-dimensional grid

A grid of 2 by 2 by 2 cells has 3 by 3 by 3 corners. Each of its eight cells is
a cube, and a cube is six tetrahedra --- one for each order in which a path can
walk from the cell's low corner to its high one, taking one axis step at a
time. The volume is the grid's, eight cubic units.

```{code-cell}
volume = el.grid((2, 2, 2)).to_tetmesh()
print('corners    :', volume.coord_count)
print('tetrahedra :', volume.topo.simplex_count[3])
print('volume     :', round(float(np.sum(volume.measures)), 12))
print('smallest   :', round(float(np.min(volume.measures)), 12))
print('one sixth  :', round(1.0 / 6.0, 12))
```

Every tetrahedron has the same volume, one sixth of a cell. That is a property
of this cell: a cube's six Kuhn tetrahedra all have the same volume, because
each is a corner and three axis steps. It is a useful check that the split did
not lose or duplicate any part of the cell.

## A grid that is not axis-aligned

The corners are placed by the grid's affine, so a scaled, moved, or turned
grid gives a mesh whose cells are parallelograms or parallelepipeds. The areas
and volumes scale with the affine's determinant.

```{code-cell}
affine = np.array([[2.0, 0.0, 0.0, 1.0],
                   [0.0, 3.0, 0.0, -2.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
stretched = el.grid((2, 2), affine=affine).to_trimesh()
print('area  :', round(float(np.sum(stretched.measures)), 12))
print('bounds:')
print(np.round(np.asarray(stretched.coords).min(axis=1), 6))
print(np.round(np.asarray(stretched.coords).max(axis=1), 6))
```

The grid's region is four units along `x` and six along `y`, so the area is
24; it is the identity grid's area of four with each cell stretched by 2 and
3. A reflection behaves the same way for triangles; for tetrahedra, a
reflection would turn every cell inside out, so the method swaps two corners
of each tetrahedron when its signed volume is negative. The result is always a
mesh whose simplex volumes are positive.

## Why the cells meet

The corners are not stored per cell. A grid of `nx` by `ny` cells has
`(nx + 1)(ny + 1)` corners, and each cell refers to four of them. A corner on
the boundary between two cells is one corner with two cells referring to it,
and an interior face of a three-dimensional cell is split along the same
diagonal its neighbour uses, because both follow the axes in the same order.
That is what makes the mesh conforming, and it is why the conversion needs no
pass over the seams.

```{code-cell}
shared = el.grid((4, 4)).to_trimesh()
print('cells     :', shared.topo.simplex_count[2] // 2)
print('corners   :', shared.coord_count)
print('four by four would be', 4 * 4, 'cells and',
      (4 + 1) * (4 + 1), 'corners')
```

The count is the corner lattice's, not four corners per cell: the 16 cells
share the 25 corners, and the triangles meet along edges rather than merely
along coincident positions.

:::{admonition} Cell data does not move to the corners by itself
:class: note
A grid property has the grid's shape: one value per cell. A mesh of the grid's
corners has one position per corner, and there are more of them, so the
conversion returns the geometry alone. To carry a cell property onto the mesh,
interpolate it at the mesh's coordinates with `euclib.ops.transfer`, which
reads the grid's field at the corner positions. The conversion accepts a grid
with units and returns a mesh with those units, and it returns NumPy
coordinates even when the grid's affine is a tensor.
:::

## Refusals

Each conversion is defined for one dimension of grid, and asking for the other
is a mistake rather than something to guess at.

```{code-cell}
for (name, grid, method) in (
        ('a 3D grid as triangles', el.grid((2, 2, 2)), 'to_trimesh'),
        ('a 2D grid as tetrahedra', el.grid((2, 2)), 'to_tetmesh')):
    try:
        getattr(grid, method)()
        print(name, 'was converted')
    except ValueError as error:
        print(name, 'is refused:', error)
```

## Provenance

Adapted from pyvista's
[`tests/core/test_grid.py::test_to_quads`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_grid.py#L1229)
and
[`::test_to_hexahedra`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_grid.py#L1204).

The upstream tests convert a two-dimensional image to an `UnstructuredGrid`
whose cells are all `QUAD`, and a three-dimensional one to an
`UnstructuredGrid` whose cells are all `HEXAHEDRON`, and check that the cell
and point counts equal the grid's, that the active scalars survive, and that a
grid of the wrong dimension is refused with a message naming the dimension it
has and the filter to use instead. The page checks the counts and the refusals
on a grid of each dimension, and checks the area and volume against the
region the grid describes.

The difference is in the cell type. VTK has four-sided and six-sided cells,
and pyvista's conversion changes the *kind* of cell without splitting it;
`euclib` has triangle, tetrahedron, and prism meshes and no quadrilateral or
hexahedral element type, so its conversion splits each cell into simplexes:
two triangles for a quadrilateral and six tetrahedra for a hexahedron. That is
the same surface or volume, and it is a mesh the rest of `euclib` ---
interpolation, measures, clipping, and the boolean operations --- already
knows how to read. pyvista's tests also check that the active scalars survive
the conversion; `euclib`'s cell properties cannot move to the corner positions
without interpolation, and the page says so rather than carrying them
silently.

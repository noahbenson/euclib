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
# Image Grids from an Array and an Affine

`euclib.types.Grid` represents image-like data: a rectangular array of cells
laid out in space. It stores the *shape* of the array and an *affine*
transformation from index space to global coordinates, and nothing else — the
image itself is just a property whose shape matches the grid. This page builds
a grid from an affine matrix, attaches an image-valued property, and draws it.

:::{admonition} What this demonstrates
:class: tip
- Building a `GridTopology` from a shape and a `Grid` from that topology plus
  an affine matrix.
- Reading a grid's geometry back through `origin` and `spacing`.
- Attaching an image-valued property (`(R, C)` for a 2D grid) and looking it up
  by name.
- Converting a grid's cells to global coordinates with `euclib.ops.positions_of`.
:::

## The topology and the affine

A grid is defined by two things that are deliberately kept apart: how many
cells there are, and where those cells are in space. The shape lives in a
`GridTopology`; the mapping from index space to global coordinates lives in an
affine matrix.

```{code-cell}
import numpy as np
import euclib as el

shape = (4, 5)

# The affine matrix is (D+1, D+1); its final row must be [0, ..., 0, 1].
# Here each column step is 2 units in x and each row step is 3 units in y.
# An index names the *center* of the cell it numbers, so this affine puts the
# first cell's center at the origin and the grid's corner at (-1, -1.5).
affine = np.array([[2, 0, 0],
                   [0, 3, 0],
                   [0, 0, 1]], dtype=float)

grid = el.grid(shape, affine)
grid
```

The point of storing an affine rather than separate origin and spacing fields is
that it can also encode rotation and shear. The common case where it does not
is exposed as convenience properties:

```{code-cell}
print('shape  :', grid.shape)
print('origin :', grid.origin)
print('spacing:', grid.spacing)
```

## An image-valued property

A property attached to a grid is an image whose shape matches the grid's shape.
There is no separate "image" type: the data lives in the geometry's property
dictionary, keyed by a name.

```{code-cell}
rows = np.arange(shape[0], dtype=float)[:, None]
cols = np.arange(shape[1], dtype=float)[None, :]

# A smooth radial function of the index coordinates.
radial = np.hypot(rows - 1.5, cols - 2.0)
grid = grid.withprop('radial', radial)
grid['radial'].shape
```

## Figure

To draw the image in global coordinates rather than index coordinates, we need
the global position of each cell. `euclib.ops.positions_of` reports where a
geometry's data lives: for a grid, each cell's center, obtained by applying the
affine to its index. A cell spans half a `spacing` step on either side of that
center, so the whole image covers `shape * spacing` starting from `origin` —
the corner of the first cell, half a step before its center.

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

centers = el.ops.positions_of(grid)
extent = (grid.origin[0], grid.origin[0] + shape[0] * grid.spacing[0],
          grid.origin[1], grid.origin[1] + shape[1] * grid.spacing[1])
euclib_viz.show2d(grid['radial'], name='grid-radial', extent=extent,
                  points=centers, label='radial value')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_grid.py`, functions
`test_init_from_numpy_arrays` (line 65) and `test_create_image_data_from_specs`
(line 1259).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_grid.py#L65).

**How this was adapted.** The first upstream test builds an `UnstructuredGrid`
by hand from a raw connectivity array, a cell-type array, and a point array;
the second builds `ImageData` from dimensions, spacing, and an origin and
checks that negative spacing is rejected. Neither test is copied. The
adaptation keeps their subject — a grid constructed from raw numeric arrays
rather than loaded from a file — but expresses it in `euclib`'s terms, where
the connectivity that `UnstructuredGrid` carries explicitly is instead implied
by a `GridTopology` shape and an affine matrix. The upstream spacing/origin
parameters appear as the affine matrix's diagonal and translation, and the
upstream validation test is dropped because `Grid` performs its own affine
validation at construction (an ill-formed matrix raises immediately). The
`positions_of` figure is new, and shows the affine being applied to index
coordinates.
:::

:::{seealso}
- [The example gallery](../index.md) for the other pages.
- [A cube as a triangle mesh](../primitives/cube.md) for the simplex-based
  counterpart to grid-based geometry.
:::

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
# Transferring a Property between Geometries

Two geometries can occupy the same space while representing it differently — a
triangle mesh and an image grid, say. `euclib` can move a property from one to
the other, sampling it at the target's own positions. This is the operation
that makes it practical to measure something once and use it in a different
representation.

:::{admonition} What this demonstrates
:class: tip
- `euclib.ops.transfer(source, target, property)`: a copy of `target` carrying
  the source's property sampled at the target's positions.
- `euclib.ops.sample`, the lower-level call that returns the sampled values
  without building a new geometry.
- Sampling an image-valued grid property onto a mesh's vertices, and the
  reverse.
- What happens when a target position lies outside the source: the value is not
  available, because properties do not extrapolate by default.
:::

## A mesh and a grid over the same space

Both geometries cover the unit square, one as triangles and one as an 8x8
image. The grid carries an image-valued property:

```{code-cell}
import numpy as np
import euclib as el

n = 6
gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
coords = np.array([gx.ravel(), gy.ravel()])
tris = []
for i in range(n - 1):
    for j in range(n - 1):
        a, b, c, d = i * n + j, i * n + j + 1, i * n + j + n, i * n + j + n + 1
        tris += [[a, b, d], [a, d, c]]
mesh = el.trimesh(coords, np.array(tris, dtype='int64').T)

affine = np.array([[1 / 8, 0, 0.5 / 8],
                   [0, 1 / 8, 0.5 / 8],
                   [0, 0, 1]], dtype=float)


rows, cols = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing='ij')
image = np.cos(rows / 2) * np.cos(cols / 2)
grid = el.grid((8, 8), affine).withprop('im', image)

print('mesh coordinates :', mesh.coords.shape)
print('grid image shape :', grid['im'].shape)
```

## From the grid to the mesh

Transferring the grid's image onto the mesh samples the image at each mesh
vertex. Vertices lying outside the grid's extent have no value, and because
properties do not extrapolate by default, they come back as `NaN`:

```{code-cell}
on_mesh = el.ops.transfer(grid, mesh, 'im')
values = np.asarray(on_mesh['im'])

print('property shape    :', values.shape)     # one value per mesh vertex
print('available values  :', int(np.sum(~np.isnan(values))), 'of', values.size)
print('missing (outside) :', int(np.sum(np.isnan(values))))
```

The values themselves match the image sampled at the vertex positions, which is
worth checking directly for a vertex near the grid's center:

```{code-cell}
center_vertex = int(np.argmin(np.linalg.norm(mesh.coords - 0.5, axis=0)))
print('vertex nearest the center:', np.round(mesh.coords[:, center_vertex], 3))
print('transferred value        :', round(float(values[center_vertex]), 4))
print('sample() gives the same  :',
      round(float(np.asarray(el.ops.sample(grid, mesh, 'im'))[center_vertex]), 4))
```

`el.ops.sample` returns just the values; `el.ops.transfer` returns a new copy of the
target geometry with the property attached. The latter is what lets the result
feed straight into further geometry operations.

## From the mesh to the grid

The reverse direction works the same way, resampling a mesh property onto the
grid's cells. Here the mesh carries a smooth function of position:

```{code-cell}
wave = np.sin(3 * coords[0]) * np.cos(3 * coords[1])
mesh = mesh.withprop('wave', wave)

on_grid = el.ops.transfer(mesh, grid, 'wave')
resampled = np.asarray(on_grid['wave'])
print('resampled image shape:', resampled.shape)
```

The resampled image approximates the original function at the grid's cell
centers --- the position of each cell, which is where its data lives; it is not
exact, because the mesh stores the function only at its vertices and `euclib`
interpolates linearly within each triangle.

:::{admonition} Transfers must share a dimension
:class: tip
A transfer samples the source *at the target's positions*, so the two
geometries must live in the same dimensional space: a 2D grid takes values from
a 2D mesh, a 3D grid from a 3D mesh. Mismatched dimensions raise rather than
silently projecting.
:::

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show2d(resampled, name='transfer', label='wave (mesh -> grid)')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_dataset_filters.py`, function
`test_implicit_distance` (line 456).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L456).

**How this was adapted.** The upstream test computes the implicit (signed)
distance from a cone surface to a `RectilinearGrid` and checks that the
resulting scalar field appears in the grid's point data — a surface informing
a regular grid, which is exactly the cross-representation idea `euclib` calls
transfer. No code is copied. The adaptation keeps the "surface and grid over
the same space exchange a value" structure but replaces the distance field
with a transfer of an ordinary property, because `euclib` deliberately has no
signed distance (`el.distance` is non-negative and the inside/outside question
is asked with `el.contains`); the distance-field example is instead the subject
of [the distance page](../queries/distance-and-separation.md). Sampling a grid
onto a mesh's vertices is added because it is the direction that exercises the
"no extrapolation by default" rule, which the upstream test avoids by
constructing the grid to contain the surface.
:::

:::{seealso}
- [Interpolating a property within a geometry](interpolation.md) for reads that
  stay within one geometry.
- [Image grids from an array and an affine](../grids/grids-from-arrays.md) for
  the grid type used here.
:::
